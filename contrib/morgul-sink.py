#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "h5py",
#     "hdf5plugin",
#     "numpy",
#     "zmq",
# ]
# ///
"""
Morgul Sink - Connect to Morgul and receive data stream.

This allows an actual load to be attached to the output of Morgul, so
that it isn't processing then discarding data.
"""

import datetime
import signal
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import hdf5plugin  # noqa: F401
import zmq

parser = ArgumentParser()
parser.add_argument("host", help="IP to connect to")
parser.add_argument("port", help="TCP start port to connect to", type=int)
parser.add_argument("num_listeners", help="Number of listeners to run", type=int)
args = parser.parse_args()

# host = args.host
# port = args.port


def run_listener(
    port: int, stop: threading.Event, first: bool, barrier: threading.Barrier
):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 50000)
    socket.setsockopt(zmq.RCVTIMEO, 200)
    socket.connect(f"tcp://{args.host}:{port}")
    barrier.wait()
    if first:
        print(
            f"All threads waiting on ports {port}-{port + args.num_listeners - 1}",
            flush=True,
        )

    while not stop.is_set():
        num_images = 0

        while not stop.is_set():
            socket.setsockopt(zmq.RCVTIMEO, 200)
            try:
                messages = socket.recv_multipart()
                socket.setsockopt(zmq.RCVTIMEO, 2000)
                print(f"{port}: Got initial frame", flush=True)
                start = time.monotonic()
                last_seen = time.monotonic()
                if len(messages) > 1:
                    num_images += 1
                else:
                    print(f"{port}: Got single start message")
                break
            except zmq.Again:
                pass

        while not stop.is_set():
            try:
                messages = socket.recv_multipart()
                last_seen = time.monotonic()
                if len(messages) > 1:
                    num_images += 1
                if len(messages) == 1:
                    print(
                        f"{port}: Got image end packet. Saw {num_images} images.",
                        flush=True,
                    )
            except zmq.Again:
                print(
                    f"{port}: Got timeout waiting for more images. Saw {num_images} images in {1000 * (last_seen - start):.0f} ms",
                    flush=True,
                )
                break
        barrier.wait()
        if first and not stop.is_set():
            print(
                f"All acquisition threads completed at {datetime.datetime.now().isoformat()}",
                flush=True,
            )


with Manager() as manager:
    stop = manager.Event()
    barrier = manager.Barrier(args.num_listeners)

    # Set the stop signal if we hit ctrl-c
    def handler(_signal, _frame):
        stop.set()

    signal.signal(signal.SIGINT, handler)
    # Now run the workers
    with ProcessPoolExecutor(max_workers=args.num_listeners) as pool:
        for port in range(args.port, args.port + args.num_listeners):
            pool.submit(
                run_listener, port, stop, first=(port == args.port), barrier=barrier
            )

print("done")
