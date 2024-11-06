import itertools
import logging
import sys
from os.path import commonprefix
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer

logger = logging.getLogger(__name__)


def merge(
    data_files: Annotated[
        list[Path], typer.Argument(help="Data files, for corrections.", metavar="DATA")
    ],
):
    """Merge split modules into single modules"""
    rows: dict[int, h5py.File] = {}
    filenames: dict[int, Path] = {}
    columns: set[int] = set()

    # Read and validate all the input files
    for data_file in data_files:
        logger.info(f"Reading {data_file}")
        file = h5py.File(data_file, "r")
        columns.add(int(file["column"][()]))
        if len(columns) > 1:
            sys.exit("Unexpected: Got more than one column from input files")
        row = int(file["row"][()])
        if row in rows:
            sys.exit(f"Error: Got duplicate entry for row {row}")
        rows[row] = file
        filenames[row] = Path(data_file)

        data = file["data"]
        if not data.shape[-2:] == (256, 1024):
            sys.exit(f"Error: {data_file} has unexpected shape {data.shape}")

    if len(rows) % 2 != 0:
        sys.exit("Error: Got uneven number of rows")

    for top, bottom in itertools.batched(sorted(rows.keys()), n=2):
        if top % 2 != 0:
            sys.exit(f"Error: Got odd top row {top} ({filenames[top]}")
        common = commonprefix([filenames[top], filenames[bottom]]).rstrip("_")
        output_filename = f"{common}_{top}-{bottom}_merged.h5"
        logger.info(f"Merging rows {top} and {bottom} into {output_filename}")

        if rows[top]["data"].shape != rows[bottom]["data"].shape:
            sys.exit(
                f"Error: Data in {filenames[top]} and {filenames[bottom]} look like they should merge but have different shapes."
            )
        shape = rows[top]["data"].shape
        with h5py.File(output_filename, "w") as out:
            layout = h5py.VirtualLayout(
                shape=(shape[0], shape[1] * 2, shape[2]), dtype=rows[top]["data"].dtype
            )
            source_top = h5py.VirtualSource(
                filenames[top].resolve(), "data", shape=shape
            )
            source_btm = h5py.VirtualSource(
                filenames[bottom].resolve(), "data", shape=shape
            )
            layout[:, : shape[1], :] = source_top[:, :, :]
            layout[:, shape[1] :, :] = source_btm[:, :, :]
            out.create_virtual_dataset("data", layout)
            out["row"] = top // 2

            # Copy everything else
            for key in rows[top].keys() - {"data", "row"}:
                out[key] = np.copy(rows[top][key])

        print(f"Written output to {output_filename}")
