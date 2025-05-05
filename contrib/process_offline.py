#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "h5py",
#     "hdf5plugin",
#     "numpy",
#     "rich",
#     "zmq",
# ]
# ///
import contextlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import zmq
from rich import print

# Basic ports for internal send/recv
SEND_PORT = 30001
RECV_PORT = 31001
h5_compression = {"compression": 32008, "compression_opts": (0, 2)}

MORGUL = shutil.which("morgul-cuda") or "/scratch/nickd/morgul/_build/morgul-cuda"


class FileInfo(NamedTuple):
    pos: tuple[int, int]
    frames: int
    filename: Path
    exptime: float
    gainmode: str
    file: h5py.File
    data: h5py.Dataset


def start_sender(port: int) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://localhost:{port}")
    return socket


class DataChunker:
    def __init__(self, data, chunksize=100):
        self.data = data
        self.cached = None
        self.chunk = None
        self.chunksize = chunksize

    def __getitem__(self, i):
        key = i // self.chunksize
        if self.cached != key:
            # print(f"Hit: {i} == {key}")
            # else:
            print(f"Miss: {i} == {key}")
            self.chunk = np.copy(
                self.data[key * self.chunksize : (key + 1) * self.chunksize][:]
            )
            self.cached = key
        return self.chunk[i % self.chunksize]


def read_h5_file(file: Path) -> FileInfo:
    f = h5py.File(file)
    assert f["data"].shape[1:] == (256, 1024), f"Unexpected data shape in {file}"
    return FileInfo(
        pos=(int(f["row"][()]), int(f["column"][()])),
        frames=f["data"].shape[0],
        filename=file,
        exptime=float(f["exptime"][()]),
        gainmode=f["gainmode"][()],
        file=f,
        data=f["data"],
    )


def only(items):
    """Return the only item in an iterable, or error"""
    items = list(items)
    if len(items) > 1:
        raise RuntimeError(f"Got more than one result in collection: {items}")
    return items[0]


class Common(NamedTuple):
    exptime: float
    gain_mode: str
    timestamp: float
    num_images: int


def run_sink(port: int, output: Path, common: Common, *, ignore: int = 0):
    """Receive frames from morgul and write them to file, like the IOC."""

    file: h5py.File | None = None

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 50000)
    socket.connect(f"tcp://localhost:{port}")

    print(f"Starting frame sink {port}")
    if ignore:
        for i in range(ignore):
            socket.recv_multipart()
    print(f"Sink {port} ignored first {ignore} frames")

    with contextlib.ExitStack() as stack:
        for i in range(common.num_images):
            # Receive this message
            [header, frame] = socket.recv_multipart()
            # After the initial message, have a short timeout
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            header = json.loads(header)
            if file is None:
                filename = (
                    output.parent
                    / f"{output.stem}_{header['row']}_{header['column']}{output.suffix}"
                )
                file = stack.enter_context(h5py.File(filename, "w"))
                file["column"] = header["column"]
                file["row"] = header["row"]
                file["exptime"] = common.exptime
                file["gainmode"] = common.gain_mode
                file["timestamp"] = common.timestamp
                data = file.create_dataset(
                    "data",
                    shape=(common.num_images, 256, 1024),
                    chunks=(1, 256, 1024),
                    dtype=np.uint16,
                    **h5_compression,
                )
            # Write the actual chunk data
            data.id.write_direct_chunk((i, 0, 0), frame)

        print(f"Received all images on sink port {port}")


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "pedestal", help="Folder for pedestal data", type=Path, metavar="PEDESTAL_DIR"
    )
    parser.add_argument(
        "data", help="Folder containing data files", type=Path, metavar="DATA_DIR"
    )
    parser.add_argument(
        "--loops",
        help="Number of loops per gain mode of pedestal data",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--frames",
        help="Number of frames per pedestal loop. The last frame will be in the expected mode.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--detector",
        choices=["JF1M"],
        default="JF1M",
    )
    parser.add_argument(
        "-o",
        help="Output filename",
        default=Path("output.h5"),
        dest="output",
        type=Path,
    )
    parser.add_argument(
        "--morgul",
        type=Path,
        help="Location of the morgul-cuda executable",
        default=MORGUL,
    )

    args = parser.parse_args()

    if not args.morgul.is_file():
        sys.exit(
            "Error: Could not find morgul-cuda. Please add to PATH or pass as --morgul="
        )

    pedestal_files = list(args.pedestal.glob("*.h5"))
    data_files = list(args.data.glob("*.h5"))

    collection_info_file = args.data / "collection_info.json"
    if not collection_info_file.is_file():
        sys.exit("Error: Could not get energy from data collection_info.json")
    collection_info = json.loads(collection_info_file.read_bytes())
    print(f"Copying {collection_info_file}")
    shutil.copy(collection_info_file, args.output.parent / "collection_info.json")

    if len(pedestal_files) != len(data_files):
        print("Error: Number of pedestal files does not match number of data files")
        sys.exit(1)

    pedestal_info = {f.pos: f for f in [read_h5_file(x) for x in pedestal_files]}
    data_info = {f.pos: f for f in [read_h5_file(x) for x in data_files]}
    send_order = sorted(pedestal_info.keys())
    assert set(pedestal_info) == set(data_info), "Module position mismatch"

    # Get bulk values to apply across the whole data set
    pedestal_frames = only({x.frames for x in pedestal_info.values()})
    data_frames = only({x.frames for x in data_info.values()})
    common = Common(
        exptime=only({x.exptime for x in pedestal_info.values()}),
        timestamp=time.time(),
        gain_mode=only({x.gainmode for x in data_info.values()}),
        num_images=data_frames,
    )
    assert common.exptime == only(
        {x.exptime for x in data_info.values()}
    ), "Data/Pedestal exposure time mismatch"
    print(f"Read exposure time: {common.exptime}")

    # print(matched_data)

    # Make the sending sockets
    senders = [
        start_sender(x) for x in range(SEND_PORT, SEND_PORT + len(pedestal_info))
    ]

    sink_pool = ProcessPoolExecutor(len(pedestal_info))
    for port in range(RECV_PORT, RECV_PORT + len(pedestal_info)):
        sink_pool.submit(run_sink, port, args.output, common, ignore=pedestal_frames)

    # Run the morgul processor
    cmd = [
        MORGUL,
        "--detector",
        args.detector,
        "live",
        "--no-progress",
        "--zmq-host",
        "localhost",
        "--zmq-port",
        str(SEND_PORT),
        str(len(senders)),
    ]
    print("+ " + shlex.join(cmd))
    morgul_live = subprocess.Popen(args=cmd)
    # Wait for this to be ready.. Just a guess as ZMQ makes this a bit opaque
    time.sleep(2)

    base_header = {
        "acqIndex": 0,
        "bitmode": 16,
        "column": 0,
        "completeImage": 0,
        "data": 0,
        "detshape": [1, 4],
        "detSpec1": 0,
        "detSpec2": 0,
        "detSpec3": 0,
        "detSpec4": 0,
        "detType": 0,
        "expLength": 0.0,
        "fileIndex": 0,
        "flipRows": 0,
        "fname": "",
        "frameIndex": 0,
        "frameNumber": 0,
        "jsonversion": 0,
        "modId": 0,
        "packetNumber": 0,
        "progress": 0.0,
        "quad": 0,
        "row": 0,
        "shape": [1024, 256],
        "size": 0,
        "timestamp": 0,
        "version": 0,
    }
    try:
        # Send all the pedestal data
        for i in range(pedestal_frames):
            for sock, ped in zip(senders, [pedestal_info[x] for x in send_order]):
                frame_data = ped.data[i].tobytes()
                header = base_header | {
                    "acqIndex": 1,
                    "shape": [1024, 256],
                    "detshape": [1, 4],
                    "row": ped.pos[0],
                    "column": ped.pos[1],
                    "expLength": common.exptime,
                    "frameNumber": i,
                    "frameIndex": i,
                    "bitmode": 16,
                    "progress": 100 * i / pedestal_frames,
                    "addJsonHeader": {
                        "pedestal": True,
                        "pedestal_frames": args.frames,
                        "pedestal_loops": args.loops,
                        "wavelength": str(collection_info["wavelength"]),
                    },
                    "timestamp": int(time.time()),
                }
                sock.send_multipart([json.dumps(header).encode(), frame_data])

        # Send the end packets
        for sock in senders:
            sock.send(json.dumps(base_header | {"bitmode": 0}).encode())

        print("Send all pedestal data, now sending measurement data")
        time.sleep(2)

        # Now send the data frames
        for i in range(data_frames):
            for sock, data in zip(senders, [data_info[x] for x in send_order]):
                # print(i, data)
                frame_data = data.data[i].tobytes()
                header = base_header | {
                    "acqIndex": 2,
                    "shape": [1024, 256],
                    "detshape": [1, 4],
                    "row": data.pos[0],
                    "column": data.pos[1],
                    "expLength": common.exptime,
                    "frameNumber": i + pedestal_frames,
                    "frameIndex": i,
                    "bitmode": 16,
                    "progress": 100 * i / data_frames,
                    "addJsonHeader": {
                        "wavelength": str(collection_info["wavelength"]),
                    },
                    "timestamp": int(time.time()),
                }
                sock.send_multipart([json.dumps(header).encode(), frame_data])

        # Send the end packets
        for sock in senders:
            sock.send(json.dumps(base_header | {"bitmode": 0}).encode())

        # Wait for all sinks to be done
        print("Waiting to finish receiving all data")
        sink_pool.shutdown(wait=True)

        print("Received all data; Shutting down live processor...")
    finally:
        morgul_live.terminate()
        morgul_live.wait()


if __name__ == "__main__":
    run()
