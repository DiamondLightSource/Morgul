#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5py",
#     "tqdm",
# ]
# ///
"""
Take a Jungfrau data h5, and dump images to separate files for simulator streaming.
"""

import argparse
import os

import h5py
import tqdm


def extract_raw_module(prefix, number, link):
    for j in range(4):
        assert os.path.exists(f"{prefix}_{j}_0.h5")
    fins = [h5py.File(f"{prefix}_{j}_0.h5", "r") for j in range(4)]

    dins = [fin["data"] for fin in fins]

    for j in range(3):
        assert dins[j].shape == dins[3].shape

    if not number or number > dins[3].shape[0]:
        number = dins[3].shape[0]

    fouts = [open(f"module_{j}_0.raw", "wb") for j in (0, 2)]

    for k in tqdm.tqdm(range(number)):
        ms = [dins[j][k] for j in range(4)]
        fouts[0].write(ms[0].tobytes())
        fouts[0].write(ms[1].tobytes())
        fouts[1].write(ms[2].tobytes())
        fouts[1].write(ms[3].tobytes())

    for fin in fins:
        fin.close()
    for fout in fouts:
        fout.close()

    if link:
        for j in range(1, 9):
            os.link("module_0_0.raw", f"module_{4 * j}_0.raw")
            os.link("module_2_0.raw", f"module_{4 * j + 2}_0.raw")


def main():
    parser = argparse.ArgumentParser(
        prog="extractor",
        description="Extract raw data from JUNGFRAU HDF5 files",
    )
    parser.add_argument("prefix")
    parser.add_argument("-n", "--number", type=int, default=0)
    parser.add_argument("-l", "--link", action="store_true")

    args = parser.parse_args()
    extract_raw_module(args.prefix, args.number, args.link)


if __name__ == "__main__":
    main()
