#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5py",
#     "hdf5plugin",
# ]
# ///

from argparse import ArgumentParser

import h5py
import hdf5plugin  # noqa: F401

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

f = h5py.File(args.filename)

for i in range(f["data"].shape[0]):
    print(repr(f["data"][i]))
