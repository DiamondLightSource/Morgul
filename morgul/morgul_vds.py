import itertools
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from pprint import pprint
from typing import Annotated, NamedTuple, Tuple

import h5py
import numpy as np
import typer

from .config import Detector, get_detector

logger = logging.getLogger(__name__)


class Geometry(NamedTuple):
    gap_fast: int
    gap_slow: int
    mod_size_fast: int
    mod_size_slow: int


GEOMETRIES = {
    Detector.JF9MB: Geometry(
        gap_fast=4, gap_slow=38, mod_size_fast=1024, mod_size_slow=512
    )
}


def lazy_cumsum(series: Iterable[int]) -> Iterator[int]:
    sum = 0
    for value in series:
        yield sum
        sum += value


def vds(
    data_files: Annotated[
        list[Path], typer.Argument(help="Data files, for corrections.", metavar="DATA")
    ],
    output_filename: Annotated[Path, typer.Option("-o", "--output")] = Path(
        "merged.h5"
    ),
):
    """Merge split modules into single modules"""
    # tiles = dict[Tuple[int, int], h5py.File] = {}
    tiles: dict[Tuple[int, int], Path] = {}

    assert data_files, "Cannot merge no data files"

    num_images: int
    dtype: np.dtype
    image_shape: tuple[int, int]
    # Read and validate all the input files
    for data_file in data_files:
        logger.info(f"Reading {data_file}")
        with h5py.File(data_file, "r") as file:
            position = (int(file["column"][()]), int(file["row"][()]))
            if position in tiles:
                sys.exit(
                    f"Unexpected: Got more than one entry for tile position {position}; cannot handle rolled files yet"
                )
            tiles[position] = Path(data_file)

            data = file["data"]
            if not data.shape[-2:] == (256, 1024):
                sys.exit(f"Error: {data_file} has unexpected shape {data.shape}")
            num_images = data.shape[0]
            image_shape = tuple(data.shape[1:])
            dtype = data.dtype

    # Work out the dataset size
    rows = max(x[1] for x in tiles.keys()) + 1
    cols = max(x[0] for x in tiles.keys()) + 1
    print(f"Constructing VDS for {rows}x{cols} detector, have {len(tiles)} data sets")
    assert rows % 2 == 0, "Uneven row count"

    detector = get_detector()
    if detector not in GEOMETRIES:
        sys.exit(f"Error: Do not know geometry for detector {detector}")
    # geo = GEOMETRIES[detector]
    # TODO: Make this generic, just hardcode for now
    # total_size_s = rows//2 * geo.mod_size_slow + geo.gap_slow*(rows//2-1)
    # total_size_f = cols * geo.mod_size_fast + geo.gap_fast*(cols-1)

    # Calculations:
    # - Raw data is 256 x 1024 in halfmodules
    # - Edge pixels are 2x in each dimension, so the actual space this
    #   takes on the front of the detector is 258x1026. This is what we
    #   tile.
    # - We trim the outer layer of pixels for a 254 x 1022 sized inner module
    # - This module is placed offset 2,2 into a 258x1026 block
    # - Two half-modules are continous, so we have a 4 pixel gap between them, but
    #   this is handled intrinsically by placing offset panels into 258x1026 blocks
    # - Horizontal gap between modules is 8 pixels
    # - Vertical gap between modules is 36 pixels
    # - To avoid a bare strip round all sides, shift -2,-2 and reduce dimensions by 4

    # Calculate an incrementing offset for each row
    slow_offsets = list(itertools.islice(lazy_cumsum(itertools.cycle([0, 36])), rows))
    # Fast doesn't have any offset except pixel boundaries (is this true?)
    fast_offsets = [0, 0, 0]
    # ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    # │             │  │             │  │             │
    # ├─────────────┤  ├─────────────┤  ├─────────────┤
    # │             │  │             │  │             │
    # └─────────────┘  └─────────────┘  └─────────────┘
    # ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    # │             │  │             │  │             │
    # ├─────────────┤  ├─────────────┤  ├─────────────┤
    # │             │  │             │  │             │
    # └─────────────┘  └─────────────┘  └─────────────┘
    # ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    # │             │  │             │  │             │
    # ├─────────────┤  ├─────────────┤  ├─────────────┤
    # │             │  │             │  │             │
    # └─────────────┘  └─────────────┘  └─────────────┘
    # ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    # │             │  │             │  │             │
    # ├─────────────┤  ├─────────────┤  ├─────────────┤
    # │             │  │             │  │             │
    # └─────────────┘  └─────────────┘  └─────────────┘
    # ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    # │             │  │             │  │             │
    # ├─────────────┤  ├─────────────┤  ├─────────────┤
    # │             │  │             │  │             │
    # └─────────────┘  └─────────────┘  └─────────────┘
    # ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    # │             │  │             │  │             │
    # ├─────────────┤  ├─────────────┤  ├─────────────┤
    # │             │  │             │  │             │
    # └─────────────┘  └─────────────┘  └─────────────┘
    total_size_slow = 258 * rows + slow_offsets[-1] - 4
    total_size_fast = 1026 * cols - 4

    with h5py.File(output_filename, "w") as out:
        layout = h5py.VirtualLayout(
            shape=(num_images, total_size_slow, total_size_fast), dtype=dtype
        )
        pprint(tiles)
        # For every module, make a source
        for row, slow_offset in zip(range(rows), slow_offsets):
            for col, fast_offset in zip(range(cols), fast_offsets):
                if (col, row) not in tiles:
                    print(f"No entry for {(row, col)=}")
                    break
                filename = tiles[col, row]

                source = h5py.VirtualSource(
                    filename.resolve(), "data", shape=(num_images, *image_shape)
                )
                # Calculate the full-sized module corner position
                x = col * 1026 + fast_offset - 2
                y = row * 258 + slow_offset - 2

                layout[:, y + 2 : y + 258 - 2, x + 2 : x + 1026 - 2] = source[
                    :, 1:-1, 1:-1
                ]
                print(f"Tile (r={row:2}, c={col:2}) placed at s={y:4}, f={x:4}")

        #         out.create_virtual_dataset("data", layout)
        out.create_virtual_dataset("data", layout)

    print(f"Written output to {output_filename}")
    # for top, bottom in itertools.batched(sorted(rows.keys()), n=2):
    #     if top % 2 != 0:
    #         sys.exit(f"Error: Got odd top row {top} ({filenames[top]}")
    #     common = commonprefix([filenames[top], filenames[bottom]]).rstrip("_")
    #     output_filename = f"{common}_{top}-{bottom}_merged.h5"
    #     logger.info(f"Merging rows {top} and {bottom} into {output_filename}")

    #     if rows[top]["data"].shape != rows[bottom]["data"].shape:
    #         sys.exit(
    #             f"Error: Data in {filenames[top]} and {filenames[bottom]} look like they should merge but have different shapes."
    #         )
    #     shape = rows[top]["data"].shape
    #     with h5py.File(output_filename, "w") as out:
    #         layout = h5py.VirtualLayout(
    #             shape=(shape[0], shape[1] * 2, shape[2]), dtype=rows[top]["data"].dtype
    #         )
    #         source_top = h5py.VirtualSource(
    #             filenames[top].resolve(), "data", shape=shape
    #         )
    #         source_btm = h5py.VirtualSource(
    #             filenames[bottom].resolve(), "data", shape=shape
    #         )
    #         layout[:, : shape[1], :] = source_top[:, :, :]
    #         layout[:, shape[1] :, :] = source_btm[:, :, :]
    #         out.create_virtual_dataset("data", layout)
    #         out["row"] = top // 2

    #         # Copy everything else
    #         for key in rows[top].keys() - {"data", "row"}:
    #             out[key] = np.copy(rows[top][key])
