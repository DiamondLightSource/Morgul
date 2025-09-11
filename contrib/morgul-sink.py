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
"""
Morgul Sink - Connect to Morgul and receive data stream.

This allows an actual load to be attached to the output of Morgul, so
that it isn't processing then discarding data.
"""

import datetime
import logging
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

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import zmq
from pydantic import BaseModel
from rich import print
from tqdm import tqdm

logger = logging.getLogger(__name__)

PV_PATH = os.getenv("MORGUL_PV_PATH") or "BL24I-JUNGFRAU-META:FD:FilePath_RBV"
PV_FILENAME = os.getenv("MORGUL_PV_FILENAME") or "BL24I-JUNGFRAU-META:FD:FileName_RBV"
PV_COUNT = (
    os.getenv("MORGUL_PV_COUNT") or "BL24I-JUNGFRAU-META:FD:NumCapture_RBV"
)  # BL24I-EA-JFRAU-01:FramesPerAcq_RBV
PV_CAPTURED = os.getenv("MORGUL_PV_CAPTURED") or "BL24I-JUNGFRAU-META:FD:NumCaptured"
PV_SUBFOLDER = os.getenv("MORGUL_PV_SUBFOLDER") or "BL24I-JUNGFRAU-META:FD:Subfolder"
PV_READY = os.getenv("MORGUL_PV_READY") or "BL24I-JUNGFRAU-META:FD:Ready"


CAGET_EXE = shutil.which("caget")
if not CAGET_EXE:
    sys.exit("Error: caget must be present on the path")
CAPUT_EXE = shutil.which("caput")
if not CAPUT_EXE:
    sys.exit("Error: caput must be present on the path")

parser = ArgumentParser()
parser.add_argument("host", help="IP to connect to")
parser.add_argument("port", help="TCP start port to connect to", type=int)
parser.add_argument("num_listeners", help="Number of listeners to run", type=int)
parser.add_argument("--write", help="Do image file writing", action="store_true")
parser.add_argument(
    "--start-index", help="The stream index for the first listener", type=int, default=0
)
parser.add_argument("-v", "--verbose", help="Show debug output", action="store_true")
args = parser.parse_args()
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
)

MORGUL_EXE = shutil.which("morgul")
if not MORGUL_EXE:
    logger.warning("Could not find morgul: Will not autogenerate VDS")


def caget(pv, as_string: bool = True) -> str:
    proc = subprocess.run(
        [CAGET_EXE, "-tS", pv], capture_output=True, check=True, text=True
    )
    return proc.stdout.strip()


def caput(pv, value: str | int | float) -> None:
    subprocess.Popen(
        [CAPUT_EXE, pv, str(value)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class Header(BaseModel):
    frameIndex: int
    row: int
    column: int
    shape: Tuple[int, int]
    detshape: Tuple[int, int]
    bitmode: int
    expLength: int
    acquisition: int

    @property
    def hmi(self):
        return self.detshape[1] * self.column + self.row


def get_filename_template() -> str | None:
    if not args.write:
        return None
    subfolder = bool(int(caget(PV_SUBFOLDER, as_string=True)))
    path = Path(caget(PV_PATH, as_string=True))
    name = caget(PV_FILENAME, as_string=True)
    if subfolder:
        path = path / f"{{acquisition:04}}_{name}"
    return str(path / (name + "{acquisition}_{stream:02}_{file_index:06}.h5"))


class HDF5Writer:
    """
    Take care of writing to rolling HDF5 image files
    """

    def __init__(
        self,
        filename_template: str,
        *,
        header: Header,
        stream_index: int,
        expected_frames: int,
        images_per_file: int = 0,
    ):
        self.current_filename: Path | None = None
        self.template = filename_template
        self.last_image_written = None
        self.broken = False
        self.current_file: h5py.File | None = None
        self.expected_frames = expected_frames
        self.images_per_file = images_per_file
        self.header = header
        self.stream_index = stream_index
        self.filenames: list[Path] = []

    def _get_filename(self, image_index: int) -> Path:
        return Path(
            self.template.format(
                acquisition=self.header.acquisition,
                stream=self.stream_index,
                file_index=image_index,
            )
        )

    def _ensure_file_for(self, image_index: int) -> h5py.File:
        # Check if we need to open a new file
        file_index = self._get_file_index(image_index)
        filename = self._get_filename(file_index)

        if filename != self.current_filename:
            # Make sure the directory exists
            if not filename.parent.is_dir():
                try:
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Creating target folder: {filename.parent}")
                except FileExistsError:
                    # Most likely a race condition with other threads
                    pass
            self.dataset = None
            if self.current_file:
                self.current_file.close()
            self.current_filename = filename
            self.current_file = self._create_new_datafile(
                filename, file_index, self.header
            )
            self.filenames.append(filename)
            self.dataset = self.current_file["/data"]
        return self.current_file

    def _create_new_datafile(
        self, filename: Path, file_index: int, header: Header
    ) -> h5py.File:
        logger.debug(f"Creating new data file {filename}")
        if filename.exists():
            raise RuntimeError(f"Path {filename} already exists, not overwriting")
        file = h5py.File(filename, "w")
        file.create_dataset("timestamp", data=datetime.datetime.now().timestamp())
        file.create_dataset("exptime", data=header.expLength * 1e-9)
        file.create_dataset("row", data=header.row)
        file.create_dataset("column", data=header.column)
        dataset_size = self.expected_frames
        # If we are splitting output, work out how many images this file has
        if self.images_per_file:
            last_file_index = (
                self.expected_frames // self.images_per_file
                if self.images_per_file
                else 0
            )
            if file_index == last_file_index:
                dataset_size = self.expected_frames % self.images_per_file

        shape = list(reversed(header.shape))
        file.create_dataset(
            "data",
            shape=(dataset_size, *shape),
            chunks=(1, *shape),
            dtype=np.uint16,
            compression=32008,
            compression_opts=(0, 2),
        )
        return file

    def _get_file_index(self, image_index: int) -> int:
        if self.images_per_file:
            return image_index // self.images_per_file
        return 0

    def write_image(self, index: int, data: bytes):
        if self.broken:
            return
        if self.last_image_written is not None and index <= self.last_image_written:
            self.broken = True
            self.close()
            logger.error(
                f"Attempting to write out-of-order/overwrite already written images. Skipping remaining {self.template} images."
            )
        self._ensure_file_for(index)
        self.dataset.id.write_direct_chunk((index, 0, 0), data)

    def close(self):
        if self.current_file:
            self.current_file.close()
        self.current_file = None
        self.current_filename = None


class Writer:
    def __init__(
        self,
        port: int,
        stop: threading.Event,
        first: bool,
        barrier: threading.Barrier,
        started: threading.Event,
        stream_index: int,
        shared_counts=managers.DictProxy,
        shared_filenames=managers.ListProxy,
    ):
        self.port = port
        self.stop = stop
        self.first = first
        self.barrier = barrier
        self.started = started
        self.shared_counts = shared_counts
        # Once we have read an HMI it's an error if we change
        self.known_hmi = None
        self.stream_index = stream_index
        self.shared_filenames = shared_filenames

        try:
            self.listen()
        except threading.BrokenBarrierError:
            pass
        except Exception:
            logger.exception("Got exception")
            # print(f"Got non-barrier exception: {e}")
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
            if self.first:
                caput(PV_READY, 1)

            self.socket.setsockopt(zmq.RCVTIMEO, 200)

            expected_images, num_images, filenames = self.do_acquisition()
            self.shared_filenames.extend(filenames)
            self.shared_counts[self.port] = (expected_images, num_images)
            first_out = self.barrier.wait() == 0
            if first_out and not self.stop.is_set():
                self.started.clear()
                print(self.shared_counts)
                print(
                    f"Acquisition completed at {datetime.datetime.now().isoformat().replace('T', ' ')} with {num_images} images",
                    flush=True,
                )

            if first_out and MORGUL_EXE and len(self.shared_filenames):
                print("Generating VDS")
                cmd = [MORGUL_EXE, "vds", *shared_filenames]
                print(f"+ {' '.join(str(x) for x in cmd)}")
                subprocess.run(cmd)
                self.shared_filenames[:] = []

    def write_frame(self, hmi: int, data: bytes):
        pass

    def do_acquisition(
        self,
    ) -> Tuple[int, int, list[Path]] | Tuple[None, None, list[Path]]:
        """
        Run a single acquisition.

        Returns: Tuple of (expected, observed) image counts
        """
        # What was the started flag the last time we went round?
        last_started = False
        while not stop.is_set():
            try:
                messages = self.socket.recv_multipart()
                break
            except zmq.Again:
                if last_started:
                    print(
                        f"{self.port}: Error - waited extra 200ms to start with group but never got messages"
                    )
                    return (None, None, [])
                else:
                    # If any of the threads have started, then we should wait only one more round
                    last_started = started.is_set()
        # Don't do anything else if stop requested
        if stop.is_set():
            return (None, None, [])

        # Wait longer for late frames, once we have started
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.started.set()
        template_path = get_filename_template()
        expected_images = int(caget(PV_COUNT))
        header = Header.model_validate_json(messages[0])
        if self.first:
            caput(PV_READY, 0)
            if int(caget(PV_CAPTURED)) != 0:
                print(
                    "Warning: Started capture but captured image count != 0. Ophyd should reset this!"
                )

            print(
                f"Started acquisition {header.acquisition}, expect {expected_images} images",
                flush=True,
            )
            progress = tqdm(total=expected_images, desc="Stream 1")

        module_hmi = header.detshape[1] * header.column + header.row
        assert module_hmi == header.hmi
        if self.known_hmi is None:
            self.known_hmi = module_hmi
        # Check this hasn't changed
        if module_hmi != self.known_hmi:
            raise RuntimeError(
                f"{self.port}: HMI index mismatch, got {module_hmi} instead of expected {self.known_hmi}"
            )

        start = time.monotonic()
        last_seen = time.monotonic()
        if len(messages) != 2:
            print(f"{self.port}: Error: Got unexpected start message: {messages}")
            return (None, None, [])

        writer = None
        if template_path:
            print(
                f"{self.port}: Writing new acquisition to {template_path.format(acquisition=header.acquisition, stream=module_hmi, file_index=0)}",
                flush=True,
            )
            writer = HDF5Writer(
                template_path,
                header=header,
                stream_index=self.stream_index,
                expected_frames=expected_images,
            )
            writer.write_image(header.frameIndex, messages[1])

        num_images = 1
        # Wait for the rest of the data;
        while num_images < expected_images:
            # Update captured images, but not every frame because uses subprocess
            if num_images % 100 == 0 and self.first:
                caput(PV_CAPTURED, num_images)
            if self.first:
                progress.update()
            try:
                messages = self.socket.recv_multipart()
                last_seen = time.monotonic()
                if len(messages) > 1:
                    num_images += 1
                    header = Header.model_validate_json(messages[0])
                    if writer is not None:
                        writer.write_image(header.frameIndex, messages[1])
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
        written_filenames = []
        if writer:
            writer.close()
            written_filenames = writer.filenames
        if self.first:
            caput(PV_CAPTURED, num_images)
            progress.close()
        return (expected_images, num_images, written_filenames)


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
    shared_filenames = manager.list()

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
                    stream_index=(port - args.port) + args.start_index,
                    shared_filenames=shared_filenames,
                )
            )
        for job in jobs:
            try:
                job.result()
            except threading.BrokenBarrierError:
                pass


print("done")
