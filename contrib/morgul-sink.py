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
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.RCVHWM, 50000)
socket.connect(f"tcp://{host}:{port}")

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
print(f"Waiting on port {port}", flush=True)

while True:
    num_images = 0
    while True:
        try:
            messages = socket.recv_multipart()
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            print("Got initial frame", messages[0], flush=True)
            if len(messages) > 1:
                num_images += 1
            break
        except zmq.Again:
            pass

    while True:
        try:
            messages = socket.recv_multipart()
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            if len(messages) > 1:
                num_images += 1
            if len(messages) == 1:
                print(f"Got image end packet. Saw {num_images} images.", flush=True)
        except zmq.Again:
            print(
                f"Got timeout waiting for more images. Saw {num_images} images",
                flush=True,
            )
            break

    # print(f"seen frame {num}", flush=True)
    # if not seen_frames:
    #     print("Seen Frame)

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
