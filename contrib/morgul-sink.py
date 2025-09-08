#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "h5py",
#     "hdf5plugin",
#     "numpy",
#     "zmq",
#     "pydantic>2",
#     "rich",
# ]
# ///
"""
Morgul Sink - Connect to Morgul and receive data stream.

This allows an actual load to be attached to the output of Morgul, so
that it isn't processing then discarding data.
"""

import datetime
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, managers
from pathlib import Path
from typing import Tuple

import hdf5plugin  # noqa: F401
import zmq
from pydantic import BaseModel
from rich import print

PV_PATH = os.getenv("MORGUL_PV_PATH") or "BL24I-EA-EIGER-01:OD:FilePath_RBV"
PV_FILENAME = os.getenv("MORGUL_PV_FILENAME") or "BL24I-EA-EIGER-01:OD:FP:FileName_RBV"
PV_COUNT = os.getenv("MORGUL_PV_COUNT") or "BL24I-EA-EIGER-01:OD:NumCapture"

CAGET_EXE = shutil.which("caget")
if not CAGET_EXE:
    sys.exit("Error: caget must be present on the path")

parser = ArgumentParser()
parser.add_argument("host", help="IP to connect to")
parser.add_argument("port", help="TCP start port to connect to", type=int)
parser.add_argument("num_listeners", help="Number of listeners to run", type=int)
parser.add_argument("--write", help="Do image file writing", action="store_true")
args = parser.parse_args()


def caget(pv, as_string: bool = True) -> str:
    proc = subprocess.run(
        [CAGET_EXE, "-tS", pv], capture_output=True, check=True, text=True
    )
    return proc.stdout.strip()


class Header(BaseModel):
    frameIndex: int
    row: int
    column: int
    shape: Tuple[int, int]
    detshape: Tuple[int, int]
    bitmode: int
    expLength: int
    acquisition: int


def get_filename_template() -> str | None:
    if not args.write:
        return None
    return str(
        Path(caget(PV_PATH, as_string=True))
        / (caget(PV_FILENAME, as_string=True) + "_{}_{}.h5")
    )


class Writer:
    def __init__(
        self,
        port: int,
        stop: threading.Event,
        first: bool,
        barrier: threading.Barrier,
        started: threading.Event,
        shared_counts=managers.DictProxy,
    ):
        self.port = port
        self.stop = stop
        self.first = first
        self.barrier = barrier
        self.started = started
        self.shared_counts = shared_counts

        try:
            self.listen()
        except Exception:
            # Mark the barrier as broken
            self.barrier.abort()
            raise

    def listen(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 50000)
        self.socket.setsockopt(zmq.RCVTIMEO, 200)

        connect_addr = f"tcp://{args.host}:{self.port}"
        self.socket.connect(connect_addr)

        first_out = self.barrier.wait() == 0
        if first_out:
            print(
                f"{args.num_listeners} listeners waiting for images on ports {port}-{port + args.num_listeners - 1}",
                flush=True,
            )

        while not self.stop.is_set():
            self.socket.setsockopt(zmq.RCVTIMEO, 200)

            expected_images, num_images = self.do_acquisition()
            self.shared_counts[self.port] = (expected_images, num_images)
            first_out = self.barrier.wait() == 0
            if first_out and not self.stop.is_set():
                self.started.clear()
                print(self.shared_counts)
                print(
                    f"Acquisition completed at {datetime.datetime.now().isoformat().replace('T', ' ')} with {num_images} images",
                    flush=True,
                )

    def do_acquisition(self) -> Tuple[int, int] | Tuple[None, None]:
        """
        Run a single acquisition.

        Returns: Tuple of (expected, observed) image counts
        """
        # What was the started flag the last time we went round?
        last_started = False
        while not stop.is_set():
            try:
                messages = self.socket.recv_multipart()
                print(f"{self.port}: Initial message")
                break
            except zmq.Again:
                if last_started:
                    print(
                        f"{self.port}: Error - waited extra 200ms to start with group but never got messages"
                    )
                    return (None, None)
                else:
                    # If any of the threads have started, then we should wait only one more round
                    last_started = started.is_set()
        # Don't do anything else if stop requested
        if stop.is_set():
            return (None, None)

        # Wait longer for late frames, once we have started
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.started.set()
        template_path = get_filename_template()
        expected_images = int(caget(PV_COUNT))
        header = Header.model_validate_json(messages[0])
        if self.first:
            print(
                f"Started acquisition {header.acquisition}, expect {expected_images} images",
                flush=True,
            )

        module_hmi = header.detshape[1] * header.column + header.row
        if template_path:
            print(
                f"{self.port}: Writing to {template_path.format(module_hmi, 0)}",
                flush=True,
            )
        start = time.monotonic()
        last_seen = time.monotonic()
        if len(messages) != 2:
            print(f"{self.port}: Error: Got unexpected start message: {messages}")
            return (None, None)

        num_images = 1
        # Wait for the rest of the data;
        while not stop.is_set() and num_images < expected_images:
            try:
                messages = self.socket.recv_multipart()
                last_seen = time.monotonic()
                if len(messages) > 1:
                    num_images += 1
                if len(messages) == 1:
                    print(
                        f"{self.port}: Got image end packet. Saw {num_images} images.",
                        flush=True,
                    )
            except zmq.Again:
                print(
                    f"{self.port}: Got timeout waiting for more images. Saw {num_images} images in {1000 * (last_seen - start):.0f} ms",
                    flush=True,
                )
                break
        return (expected_images, num_images)


print(r""" ███▄ ▄███▓ ▒█████   ██▀███    ▄████  █    ██  ██▓
▓██▒▀█▀ ██▒▒██▒  ██▒▓██ ▒ ██▒ ██▒ ▀█▒ ██  ▓██▒▓██▒
▓██    ▓██░▒██░  ██▒▓██ ░▄█ ▒▒██░▄▄▄░▓██  ▒██░▒██░
▒██    ▒██ ▒██   ██░▒██▀▀█▄  ░▓█  ██▓▓▓█  ░██░▒██░
▒██▒   ░██▒░ ████▓▒░░██▓ ▒██▒░▒▓███▀▒▒▒█████▓ ░██████▒
░ ▒░   ░  ░░ ▒░▒░▒░ ░ ▒▓ ░▒▓░ ░▒   ▒ ░▒▓▒ ▒ ▒ ░ ▒░▓  ░
░  ░      ░  ░ ▒ ▒░   ░▒ ░ ▒░  ░   ░ ░░▒░ ░ ░ ░ ░ ▒  ░
░      ░   ░ ░ ░ ▒    ░░   ░ ░ ░   ░  ░░░ ░ ░   ░ ░
       ░       ░ ░     ░           ░    ░         ░  ░
             _       __     _ __
            | |     / /____(_) /____  _____
            | | /| / / ___/ / __/ _ \/ ___/
            | |/ |/ / /  / / /_/  __/ /
            |__/|__/_/  /_/\__/\___/_/""")

print(f"Start template file: {get_filename_template() or 'None (not in write mode)'}")
with Manager() as manager:
    stop = manager.Event()
    barrier = manager.Barrier(args.num_listeners)
    started = manager.Event()
    states = manager.dict()

    # Set the stop signal if we hit ctrl-c
    def handler(_signal, _frame):
        stop.set()

    signal.signal(signal.SIGINT, handler)
    # Now run the workers
    with ProcessPoolExecutor(max_workers=args.num_listeners) as pool:
        jobs = []
        for port in range(args.port, args.port + args.num_listeners):
            jobs.append(
                pool.submit(
                    Writer,
                    port,
                    stop,
                    first=(port == args.port),
                    barrier=barrier,
                    started=started,
                    shared_counts=states,
                )
            )
        for job in jobs:
            if not stop:
                job.result()

print("done")
