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
from argparse import ArgumentParser

import hdf5plugin  # noqa: F401
import zmq

parser = ArgumentParser()
parser.add_argument("host", help="IP to connect to")
parser.add_argument("port", help="TCP port to connect to", type=int)
# parser.add_argument("num_images", help="How many images to expect", type=int)
args = parser.parse_args()

host = args.host
port = args.port

compression = {"compression": 32008, "compression_opts": (0, 2)}

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.setsockopt(zmq.RCVHWM, 50000)
print("Connecting...")
socket.connect(f"tcp://{host}:{port}")
print("Connected.")
# total = args.num_images

# fout = h5py.File(f"data_{port}.h5", "w")

# data = fout.create_dataset(
#     "data",
#     shape=(total, 256, 1024),
#     chunks=(1, 256, 1024),
#     dtype=numpy.uint16,
#     **compression,
# )

# timestamp = numpy.zeros(shape=(total,), dtype=numpy.float64)

# try:
print("Starting wait loop")
while True:
    num_images = 0
    try:
        while True:
            messages = socket.recv_multipart()
            num_images += 1
            socket.setsockopt(zmq.RCVTIMEO, 2000)
    except zmq.Again:
        if num_images > 0:
            print(f"{port}: Got timeout waiting for more images; got {num_images}")

#     for count in range(total):
#         messages = socket.recv_multipart()
#         socket.setsockopt(zmq.RCVTIMEO, 2000)

#         # header = json.loads(messages[0])
#         # frame = header["frameIndex"]
#         # offset = (frame, 0, 0)
#         # data.id.write_direct_chunk(offset, messages[1])
#         # timestamp[frame] = time.time()
#         # if count == 0:
#         #     with open(f"raw_{port}.dat", "wb") as f:
#         #         f.write(messages[1])
# except zmq.Again:
#     print("Got timeout waiting for more images")
# finally:
#     fout.create_dataset(
#         "timestamp", shape=(total,), data=timestamp, dtype=numpy.float64
#     )

#     fout.close()
#     print("Closed data file")


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
# import json
# import sys
# import time
# from argparse import ArgumentParser
# import zmq

# parser = ArgumentParser()
# parser.add_argument("host", help="IP to connect to")
# parser.add_argument("port", help="TCP port to connect to", type=int)
# parser.add_argument("num_images", help="How many images to expect", type=int)
# args = parser.parse_args()

# host = args.host
# port = args.port

# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# socket.setsockopt(zmq.SUBSCRIBE, b"")
# socket.setsockopt(zmq.RCVHWM, 50000)
# socket.connect(f"tcp://{host}:{port}")

# frames = []

# t0 = None

# while True:
#     messages = socket.recv_multipart()
#     header = json.loads(messages[0])
#     if header["bitmode"] == 0:
#         break
#     n = header["frameIndex"]

#     if n == 0:
#         t0 = time.time()

#     if not n % 1000:
#         print(n)
#     frames.append(n)

# t1 = time.time()

# print(f"Read {n + 1} frames in {t1 - t0:.2f}s")

# assert len(frames) == len(set(frames))
