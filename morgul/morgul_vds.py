import itertools
import logging
import os.path
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, NamedTuple, Tuple

import h5py
import numpy as np
import typer
from rich import print
from tqdm import tqdm

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
    output_filename: Annotated[Path | None, typer.Option("-o", "--output")] = None,
):
    """Merge split modules into single modules"""
    # tiles = dict[Tuple[int, int], h5py.File] = {}
    tiles: dict[Tuple[int, int], Path] = {}

    assert data_files, "Cannot merge no data files"
    # If we don't have an output filename, work out the common prefix
    common = os.path.commonprefix([str(x.name) for x in data_files])
    if not common:
        output_filename = Path("virtual.h5")
    else:
        output_filename = Path(f"{common.rstrip('_')}_virtual.h5")

    if len(set(x.parent for x in data_files)) == 1:
        parent = data_files[0].parent
        output_filename = parent / output_filename

    # Work out a nice way to print this
    output_print = str(output_filename)
    if str(output_filename.parent) == ".":
        output_print = f"./{output_print}"

    print(f"Reading {len(data_files)} data files")
    # print(f"Writing VDS to {output_print}")

    num_images: int
    dtype: np.dtype
    image_shape: tuple[int, int]
    # Read and validate all the input files
    for data_file in tqdm(data_files, leave=False):
        # Ignore previously generated virtual files
        if "virtual" in data_file.name:
            continue
        # logger.info(f"Reading {data_file}")
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
    print(f"Constructing VDS for {rows} x {cols} detector, have {len(tiles)} data sets")
    assert rows % 2 == 0, "Uneven row count"

    detector = get_detector()
    if detector not in GEOMETRIES:
        sys.exit(f"Error: Do not know geometry for detector {detector}")
    # geo = GEOMETRIES[detector]
    # TODO: Make this generic, just hardcode for now

    # Calculations:
    # - Raw data is 256 x 1024 in halfmodules
    # - This is formed of 4xASIC (256x256) with gaps between each ASIC
    # - Edge pixels (including on the ASIC) are 2x in each dimension, so
    #   the actual space this takes on the front of the detector is:
    #       256+2 border=258   x   1024 + 2*4 asic = 1032
    # - We take each 256x256 ASIC and trim the outer layer of pixels for
    #   a 254x254 block
    # - Each ASIC is placed offset 2,2+asic_offset spaced in the 258x1032 block
    # - Two half-modules are continous, so we have a 4 pixel gap between them, but
    #   this is handled intrinsically by placing offset panels into 258x1026 blocks
    # - Horizontal gap between modules is 8 pixels
    # - Vertical gap between modules is 36 pixels
    # - To avoid a bare strip round all sides, shift -2,-2 and reduce dimensions by 4

    # Calculate an incrementing offset for each row
    slow_offsets = list(itertools.islice(lazy_cumsum(itertools.cycle([0, 36])), rows))
    # Fast doesn't have any offset except pixel boundaries (is this true?)
    fast_offsets = [0, 8, 16]
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
    total_size_fast = 1032 * cols + fast_offsets[-1] - 4
    print(f"Target image size (s, f) = {total_size_slow}, {total_size_fast}")
    with h5py.File(output_filename, "w") as out:
        layout = h5py.VirtualLayout(
            shape=(num_images, total_size_slow, total_size_fast), dtype=dtype
        )
        # For every module, make a source
        for row, slow_offset in zip(range(rows), slow_offsets):
            for col, fast_offset in zip(range(cols), fast_offsets):
                if (col, row) not in tiles:
                    print(f"No entry for {(row, col)=}")
                    break
                filename = tiles[col, rows - row - 1]

                source = h5py.VirtualSource(
                    filename.resolve(), "data", shape=(num_images, *image_shape)
                )
                # Calculate the full-sized module corner destination position
                x = col * 1032 + fast_offset - 2
                y = row * 258 + slow_offset - 2
                # print(f"Tile (r={row:2}, c={col:2}) placed at s={y:4}, f={x:4}")

                for asic in range(4):
                    asic_src_x = 256 * asic
                    asic_dst_x = x + asic * 258
                    # print(f"    Tile ASIC {asic} placed at fast={asic_dst_x:4}")

                    layout[
                        :, y + 2 : y + 258 - 2, asic_dst_x + 2 : asic_dst_x + 258 - 2
                    ] = source[
                        :,
                        1:-1,
                        asic_src_x + 1 : asic_src_x + 256 - 1,
                    ]

        #         out.create_virtual_dataset("data", layout)
        out.create_virtual_dataset("data", layout)

    print(f"Written output to {output_print}")
