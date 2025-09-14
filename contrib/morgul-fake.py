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
#     "tqdm",
# ]
# ///

import json
import signal
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Tuple

import h5py
import zmq
from pydantic import BaseModel
from rich import print
from tqdm import tqdm


class Header(BaseModel):
    frameIndex: int
    row: int
    column: int
    shape: Tuple[int, int]
    detshape: Tuple[int, int]
    bitmode: int
    expLength: int
    acquisition: int


parser = ArgumentParser()
parser.add_argument(
    "input", help="Input files to stream raw data from", nargs="+", type=Path
)
parser.add_argument(
    "--port", "-p", help="Port to serve first stream from", type=int, default=30000
)

args = parser.parse_args()
# if not all(x.is_file() for x in args.input):
#     sys.exit("Error: Not all files exist")

print(r"""    ███▄ ▄███▓ ▒█████   ██▀███    ▄████  █    ██  ██▓
   ▓██▒▀█▀ ██▒▒██▒  ██▒▓██ ▒ ██▒ ██▒ ▀█▒ ██  ▓██▒▓██▒
   ▓██    ▓██░▒██░  ██▒▓██ ░▄█ ▒▒██░▄▄▄░▓██  ▒██░▒██░
   ▒██    ▒██ ▒██   ██░▒██▀▀█▄  ░▓█  ██▓▓▓█  ░██░▒██░
   ▒██▒   ░██▒░ ████▓▒░░██▓ ▒██▒░▒▓███▀▒▒▒█████▓ ░██████▒
   ░ ▒░   ░  ░░ ▒░▒░▒░ ░ ▒▓ ░▒▓░ ░▒   ▒ ░▒▓▒ ▒ ▒ ░ ▒░▓  ░
   ░  ░      ░  ░ ▒ ▒░   ░▒ ░ ▒░  ░   ░ ░░▒░ ░ ░ ░ ░ ▒  ░
   ░      ░   ░ ░ ░ ▒    ░░   ░ ░ ░   ░  ░░░ ░ ░   ░ ░
          ░       ░ ░     ░           ░    ░         ░  ░
    ______      __           __  ___                       __
   / ____/___ _/ /_____     /  |/  /___  _________ ___  __/ /
  / /_  / __ `/ //_/ _ \   / /|_/ / __ \/ ___/ __ `/ / / / /
 / __/ / /_/ / ,< /  __/  / /  / / /_/ / /  / /_/ / /_/ / /
/_/    \__,_/_/|_|\___/  /_/  /_/\____/_/   \__, /\__,_/_/
                                           /____/""")

print(
    f"\nServing {len(args.input)} streams from ports tcp://0.0.0.0:{args.port}-{args.port + len(args.input)}"
)


class Shared:
    def __init__(self, manager: SyncManager, num_listeners: int):
        self.cancelled = manager.Event()
        self.wait = manager.Condition()
        self.barrier = manager.Barrier(num_listeners + 1)


def find_dataset(file: h5py.File) -> h5py.Dataset:
    return file["data"]


def swallow_keyboardexcept(fun, *args, **kwargs):
    try:
        fun(*args, **kwargs)
    except KeyboardInterrupt:
        pass
    except threading.BrokenBarrierError:
        pass


def stream_data(shared: Shared, port: int, input: Path):
    file = h5py.File(input, "r")
    data = find_dataset(file)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 50000)
    socket.setsockopt(zmq.SNDTIMEO, 2000)
    socket.bind(f"tcp://0.0.0.0:{args.port + port}")

    header = {
        "frameIndex": 0,
        "row": 0,
        "column": 0,
        "shape": [256, 1024],
        "detShape": [4, 1],
        "bitmode": 16,
        "expLength": 1000000,
        "acquisition": 0,
    }
    if "row" in file:
        header["row"] = int(file["row"][()])
    if "column" in file:
        header["column"] = int(file["column"][()])

    while not shared.cancelled.is_set():
        # Wait for trigger
        with shared.wait:
            shared.wait.wait()
        last_send = 0.0

        if port == 0:
            progress = tqdm(total=data.shape[0])
        for index in range(data.shape[0]):
            header["frameIndex"] = index
            header_data = json.dumps(header).encode()
            _, raw_data = data.id.read_direct_chunk((index, 0, 0))
            if port == 0:
                progress.update()
            # Wait until it is time to send this
            remaining = last_send + 0.001 - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

            try:
                socket.send_multipart([header_data, raw_data])
            except zmq.Again:
                if port == 0:
                    print("Warning: Timeout trying to send; did you connect your sink?")
                break
            last_send = time.monotonic()
        shared.barrier.wait()
        header["acquisition"] += 1  # type: ignore

    # print(f"{port}: Done")


with Manager() as manager:
    shared = Shared(manager, len(args.input))

    # Set the stop signal if we hit ctrl-c
    def handler(_signal, _frame):
        shared.cancelled.set()
        shared.barrier.abort()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)

    # Now run the workers
    with ProcessPoolExecutor(max_workers=len(args.input)) as pool:
        jobs = []
        for i, inputfile in enumerate(args.input):
            jobs.append(
                pool.submit(swallow_keyboardexcept, stream_data, shared, i, inputfile)
            )

        try:
            while not shared.cancelled.is_set():
                input("\n\nPress return to send")
                # Start the threads
                with shared.wait:
                    shared.wait.notify_all()
                # Let them all restart at the same time
                shared.barrier.wait()
        except KeyboardInterrupt:
            pass
        finally:
            shared.cancelled.set()
            shared.barrier.abort()
            # Resume anything waiting
            with shared.wait:
                shared.wait.notify_all()
        for job in jobs:
            try:
                job.result()
            except threading.BrokenBarrierError:
                pass
        # print("\nAll cleaned up, terminating.")
