#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5py",
#     "hdf5plugin",
#     "numpy",
#     "pillow",
# ]
# ///

from argparse import ArgumentParser

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image

parser = ArgumentParser()
parser.add_argument("files", nargs=2)
args = parser.parse_args()

files = [h5py.File(x) for x in args.files]
print([x["entry/data/data_000001"].shape for x in files])
height = max(
    files[0]["entry/data/data_000001"].shape[1],
    files[1]["entry/data/data_000001"].shape[1],
)
width = max(
    files[0]["entry/data/data_000001"].shape[2],
    files[1]["entry/data/data_000001"].shape[2],
)

col_data = np.zeros(shape=(height, width, 3), dtype=np.uint8)

for i, file in enumerate(files):
    print(file.filename)
    data = file["entry/data/data_000001"][0]
    # breakpoint()
    for y in range(data.shape[0]):
        # if y % 50 == 0:
        #     print(file.filename, y)
        for x in range(data.shape[1]):
            rx = x
            ry = y
            if 0 <= data[y, x] < 1000:
                if i > 0:
                    rx += 1
                    ry += 1
                col_data[ry, rx, i] = 255
            # if y == 1 and x < 20:
            #     print(y, x, data[y,x], col_data[y,x])

# breakpoint()
im = Image.fromarray(col_data)
im.save("out.png")
# with Image.open("overlap.png") as im:
#     im.
# breakpoint()


# col_data[:,:,0] =
