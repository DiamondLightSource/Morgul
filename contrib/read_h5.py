#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5py",
#     "hdf5plugin",
#     "numpy",
# ]
# ///

import time
from argparse import ArgumentParser

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

f = h5py.File(args.filename)

for i in range(f["data"].shape[0]):
    print(i, np.sum(f["data"][i]), repr(f["data"][i]))
    time.sleep(0.1)
