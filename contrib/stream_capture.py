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
parser.add_argument("num_images", help="How many images to expect", type=int)
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
total = args.num_images

print("Starting wait loop")

num = 0

while True:
    messages = socket.recv_multipart()
    if num < 100:
        print(f"Got {num}")
        num += 1
    continue


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
