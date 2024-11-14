from __future__ import annotations

import contextlib
import datetime
import glob
import os
import re
import shutil
import time
from enum import Enum, auto
from logging import getLogger
from pathlib import Path
from typing import Annotated, List, NamedTuple, Optional

import dateutil.tz as tz
import h5py
import numpy
import numpy.typing
import tqdm
import typer

from .config import (
    Detector,
    get_detector,
    get_known_module_layout_for_detector,
    get_module_info,
)
from .util import NC, B, G, elapsed_time_string

logger = getLogger(__name__)


class ModuleMode(Enum):
    FULL = auto()
    HALF = auto()

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> ModuleMode:
        if shape[-2:] == (256, 1024):
            return cls.HALF
        elif shape[-2:] == (512, 1024):
            return cls.FULL
        raise ValueError(f"Unrecognised module shape {shape[-2]},{shape[-1]}")


def average_pedestal(
    gain_mode: int,
    dataset: h5py.Dataset,
    *,
    parent_progress: tqdm.tqdm | None = None,
    progress_title: str | None = None,
) -> tuple[
    numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray[numpy.bool_]
]:
    s = dataset.shape
    image = numpy.zeros(shape=(s[1], s[2]), dtype=numpy.float64)
    n_obs = numpy.zeros(shape=(s[1], s[2]), dtype=numpy.uint32)
    image_sq = numpy.zeros(shape=(s[1], s[2]), dtype=numpy.float64)

    # Handle gain mode 2 being ==3
    real_gain_mode = GAIN_MODE_REAL[gain_mode]

    for j in tqdm.tqdm(
        range(s[0]), desc=progress_title or f"Gain Mode {gain_mode}", leave=False
    ):
        i = dataset[j]
        gain = numpy.right_shift(i, 14)
        i = numpy.bitwise_and(i, 0x3FFF)
        valid = gain == real_gain_mode
        i = i.astype(numpy.float64) * valid
        n_obs += valid
        image += i
        image_sq += numpy.square(i)

        if parent_progress:
            parent_progress.update(1)

    # cope with zero valid observations
    assert (
        numpy.sum(n_obs) > 0
    ), f"Error: Got completely blank pedestal in {progress_title}"

    mask = n_obs == 0
    n_obs[n_obs == 0] = 1

    mean = image / n_obs
    variance = (image_sq / n_obs) - numpy.square(mean)
    return mean, variance, mask


class PedestalData(NamedTuple):
    """Pull together data about a particular file and what it represents"""

    filename: Path
    row: int
    col: int
    data: numpy.typing.NDArray
    exptime: float
    gainmode: str
    halfmodule_index: int
    module_mode: ModuleMode
    module_position: str | None
    module_serial_number: str
    num_images: int
    timestamp: datetime.datetime

    @classmethod
    def from_h5(
        cls, filename: Path, h5: h5py.Dataset, detector: Detector
    ) -> "PedestalData":
        # Work out the data shape
        mode = ModuleMode.from_shape(h5["data"].shape)
        w, h = get_known_module_layout_for_detector(detector)
        if mode == ModuleMode.FULL:
            column, row = h5["column"][()], h5["row"][()]
            hmi = column * h * 2 + row * 2
        else:
            column, row = h5["column"][()], h5["row"][()] // 2
            # Calculate the halfmodule-index; see https://github.com/graeme-winter/jungfrau/blob/d7aa198915bdddcef4dfed7ee5b94507bb5fea81/doc/FORMAT.md#module-arrangement
            hmi = column * h * 2 + h5["row"][()]

        # Work out what the module serial number and "position" is
        module = get_module_info(detector, col=column, row=row)

        return PedestalData(
            filename,
            row=h5["row"][()],
            col=h5["column"][()],
            data=h5["data"],
            exptime=h5["exptime"][()],
            gainmode=h5["gainmode"][()].decode(),
            halfmodule_index=hmi,
            module_mode=mode,
            module_position=module.get("position"),
            module_serial_number=module["module"],
            num_images=h5["data"].shape[0],
            timestamp=datetime.datetime.fromtimestamp(h5["timestamp"][()]),
        )


# Mapping from gain mode string to numeric gain mode
GAIN_MODES = {
    "dynamic": 0,
    "forceswitchg1": 1,
    "forceswitchg2": 2,
}
# The "Real" gain mode, as stored in the data
GAIN_MODE_REAL = {0: 0, 1: 1, 2: 3}


def write_pedestal_output(
    root: h5py.Group, pedestal_data: dict[int, dict[int, PedestalData]]
) -> None:
    """Calculate pedestals from source files and write into the output file"""

    module_mode: ModuleMode | None = None
    # Calculate how many images total, for progress purposes
    num_images_total = 0
    for modes in pedestal_data.values():
        for data in modes.values():
            num_images_total += data.num_images
            # Validate we don't have mixed module modes
            module_mode = module_mode or data.module_mode
            if data.module_mode != module_mode:
                raise RuntimeError("Error: Got mixed module half/full mode data")

    root.create_dataset("module_mode", data=module_mode.name.lower())

    # Analyse the pedestal data and write the output
    with tqdm.tqdm(total=num_images_total, leave=False) as progress:
        for halfmodule_index, modes in pedestal_data.items():
            for gain_mode, data in sorted(modes.items(), key=lambda x: x[0]):
                if data.module_mode == ModuleMode.FULL:
                    progress_title = f" {data.module_serial_number} Gain {gain_mode}"
                else:
                    progress_title = f" HMI{data.halfmodule_index} ({data.module_serial_number}/{data.halfmodule_index % 2}) Gain {gain_mode}"

                pedestal_mean, pedestal_variance, pedestal_mask = average_pedestal(
                    gain_mode,
                    data.data,
                    parent_progress=progress,
                    progress_title=progress_title,
                )
                if data.module_mode == ModuleMode.FULL:
                    group_name = data.module_serial_number
                else:
                    group_name = f"hmi_{data.halfmodule_index:02d}"
                if group_name not in root:
                    group = root.create_group(group_name)
                    if data.module_position is not None:
                        group.attrs["position"] = data.module_position.strip("\"'")
                    group.attrs["row"] = data.row
                    group.attrs["col"] = data.col
                    group.attrs["module"] = data.module_serial_number
                    group.attrs["halfmodule_index"] = data.halfmodule_index

                group = root[group_name]
                dataset = group.create_dataset(
                    f"pedestal_{gain_mode}", data=pedestal_mean
                )
                dataset.attrs["timestamp"] = int(data.timestamp.timestamp())
                dataset.attrs["filename"] = str(data.filename)
                dataset = group.create_dataset(
                    f"pedestal_{gain_mode}_variance", data=pedestal_variance
                )
                dataset = group.create_dataset(
                    f"pedestal_{gain_mode}_mask", data=pedestal_mask
                )


def pedestal(
    pedestal_runs: Annotated[
        List[Path],
        typer.Argument(
            help="Data files containing pedestal runs, or a folder containing h5 files for the pedestal runs. There should be a pedestal run for every gain mode."
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            help="Name for the output HDF5 file. [default: (detector)_(exptime)ms_(timestamp)_pedestal.h5]",
            show_default=False,
        ),
    ] = None,
    register_calibration: Annotated[
        bool,
        typer.Option(
            "--register",
            help="Copy the pedestal file and register in the central calibration log (pointed to by the JUNGFRAU_CALIBRATION_LOG environment variable)",
        ),
    ] = False,
):
    """
    Given dark images at different fixed gain modes, calculate the pedestal corrections tables.
    """
    start_time = time.monotonic()
    detector = get_detector()
    print(f"Using detector: {G}{detector.value}{NC}")

    # Cache all the data
    pedestal_data: dict[int, dict[int, PedestalData]] = {}

    exposure_time: float | None = None

    file_timestamps: set[float] = set()

    # Expand any arguments passed through with wildcards
    expanded_runs: list[Path] = []
    for filename in pedestal_runs:
        if filename.is_dir():
            expanded_runs.extend(filename.glob("*.h5"))
        elif "*" in str(filename):
            print(filename)
            expanded_runs.extend(Path(x) for x in glob.glob(str(filename)))
        else:
            expanded_runs.append(filename)

    if not expanded_runs:
        assert False, str(pedestal_runs)

    with contextlib.ExitStack() as stack:
        # Open all pedestal files and validate we don't have any duplicate data
        for filename in expanded_runs:
            logger.debug(f"Reading {filename}")
            h5 = stack.enter_context(h5py.File(filename, "r"))
            data = PedestalData.from_h5(filename, h5, detector)

            file_timestamps.add(h5["timestamp"][()])
            gain_mode = GAIN_MODES[data.gainmode]
            if exposure_time is None:
                exposure_time = data.exptime
                logger.info(
                    f"Generating pedestals for exposure time: {G}{exposure_time*1000:g}{NC} ms"
                )
            else:
                # Validate that this file matches the previously determined exposure time
                if data.exptime != exposure_time:
                    logger.error(
                        f"Error: pedestal file {filename} exposure time ({data.exptime}) does not match others ({exposure_time})"
                    )
                    raise typer.Abort()

            halfmodule = pedestal_data.setdefault(data.halfmodule_index, dict())

            # Validate we didn't get passed two from the same mode
            if gain_mode in halfmodule:
                logger.error(
                    f"Error: Duplicate gain mode {gain_mode} (both {halfmodule[gain_mode].filename} and {filename})"
                )
                raise typer.Abort()
            halfmodule[gain_mode] = data
            logger.info(
                f"Got file {B}{filename}{NC} with gain mode {G}{gain_mode}{NC} for module (HMI= {G}{data.halfmodule_index}{NC} [{G}{data.row}{NC}, {G}{data.col}{NC}]) ({G}{data.module_serial_number}{NC})"
            )

        # Validate that every module had a complete set of gain modes
        for module_addr, gains in pedestal_data.items():
            if not len(gains) == len(GAIN_MODES):
                logger.error(
                    f"Error: Incomplete data set. Module HMI:{module_addr} only has {len(gains)} gain modes, expected {len(GAIN_MODES)}"
                )
                raise typer.Abort()

        # Work out a timestamp name
        timestamp_name = datetime.datetime.fromtimestamp(
            sorted(file_timestamps)[0]
        ).strftime("%Y-%m-%d_%H-%M-%S")
        output = output or Path(
            f"{detector.value}_{exposure_time*1000:g}ms_{timestamp_name}_pedestal.h5"
        )
        with h5py.File(output, "w") as f_output:
            write_pedestal_output(f_output, pedestal_data)
            # Write extra metadata into the file
            f_output.create_dataset("exptime", data=exposure_time)

        print()
        logger.info(
            f"Written output file {B}{output}{NC} in {elapsed_time_string(start_time)}."
        )

    if "JUNGFRAU_CALIBRATION_LOG" in os.environ and register_calibration:
        pedestal_log = Path(os.environ["JUNGFRAU_CALIBRATION_LOG"])
        logged_pedestal = pedestal_log.parent / output.name
        logger.info(f"Copying {B}{output}{NC} to {B}{logged_pedestal}{NC}")
        shutil.move(output, logged_pedestal)
        utc_ts = datetime.datetime.fromtimestamp(sorted(file_timestamps)[0]).replace(
            tzinfo=tz.UTC
        )
        log_entry = (
            f"PEDESTAL {utc_ts.isoformat()} {exposure_time} {logged_pedestal.resolve()}"
        )
        # Ensure that we don't duplicate log lines

        logger.info(f"Writing calibration log entry:\n    {log_entry}")
        with pedestal_log.open("a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        # logger.info("")
    elif register_calibration:
        logger.error(
            f"Error: Have generated calibration {output.name} but cannot register as JUNGFRAU_CALIBRATION_LOG is not set."
        )
        raise typer.Abort()


def pedestal_fudge(
    input: Annotated[
        Path,
        typer.Argument(
            help="Input pedestal file, to use for generation", show_default=False
        ),
    ],
    exposure: Annotated[
        float,
        typer.Argument(
            help="New exposure to generate the fake-pedestal", show_default=False
        ),
    ],
    force: Annotated[
        bool, typer.Option("-f", "--force", help="Overwrite files that already exist")
    ] = False,
    register: Annotated[
        bool,
        typer.Option(
            "--register",
            help="Automatically copy and register this pedestal to the calibration log",
        ),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            help="Name for the output HDF5 file. Default: <detector>_<exptime>ms_pedestal.h5",
            show_default=False,
        ),
    ] = None,
):
    """A terrible idea. Generate new pedestal files by extrapolating the exposure times."""

    detector = get_detector()
    timestamps = set()

    # Get the timestamp out of the input file
    with h5py.File(input, "r") as fin:
        for group in fin:
            if not isinstance(fin[group], h5py.Group):
                continue
            for dset in fin[group]:
                if not isinstance(fin[group][dset], h5py.Dataset):
                    continue
                if "timestamp" in fin[group][dset].attrs:
                    timestamps.add(
                        datetime.datetime.fromtimestamp(
                            fin[group][dset].attrs["timestamp"]
                        ).replace(tzinfo=tz.UTC)
                    )
    timestamp_name = sorted(timestamps)[0].strftime("%Y-%m-%d_%H-%M-%S")
    # Work out what the output filename is now
    output = output or Path(
        f"{detector.value}_{exposure*1000:g}ms_{timestamp_name}_pedestal_fudged.h5"
    )
    if output.exists() and not force:
        logger.error(
            f"Error: output file {output} already exists. Please pass --force to overwrite."
        )
        raise typer.Abort()

    with h5py.File(input, "r") as fin, h5py.File(output, "w") as fout:
        exp_ratio = exposure / fin["exptime"][()]
        print(f"Adjusting pedestals with ratio {G}{exp_ratio:.2f}{NC}")
        # Read every module
        for entry in fin:
            # Copy groups
            if isinstance(fin[entry], h5py.Group):
                print(f"Creating output group {G}{entry}{NC}")
                # Create the group and copy any attributes
                g = fout.create_group(entry)
                if fin[entry].attrs:
                    for k, v in fin[entry].attrs.items():
                        g.attrs[k] = v
                # And now copy all pedestal data from inside this group
                for dset in fin[entry]:
                    # if dset == "pedestal_0":
                    #     breakpoint()
                    if not isinstance(fin[entry][dset], h5py.Dataset):
                        continue
                    if not re.match(r"^pedestal_\d+$", dset):
                        continue

                    logger.info(f"Copying and adjusting {G}{entry}/{dset}{NC}")
                    data = fin[entry][dset]
                    data = data * exp_ratio
                    new_dset = g.create_dataset(dset, data=data)
                    for k, v in fin[entry][dset].attrs.items():
                        new_dset.attrs[k] = v

        # Finally, write the new exposure time
        fout.create_dataset("exptime", data=exposure)
        fout.create_dataset("exptime_original", data=fin["exptime"][()])

        logger.info(
            f"Pedestal converted from {G}{fin['exptime'][()]*1000:g}{NC} ms to {G}{exposure*1000:g}{NC} ms and written to {B}{output}{NC}."
        )

    if register:
        if "JUNGFRAU_CALIBRATION_LOG" not in os.environ:
            logger.error(
                "Error: No JUNGFRAU_CALIBRATION_LOG environment variable, cannot update."
            )
            raise typer.Abort()
        pedestal_log = Path(os.environ["JUNGFRAU_CALIBRATION_LOG"])
        logged_pedestal = pedestal_log.parent / output.name
        if logged_pedestal.exists() and not force:
            logger.error(
                f"Error: Output file {logged_pedestal} already exists. Pass --force to overwrite."
            )
            raise typer.Abort()

        logger.info(f"Copying {B}{output}{NC} to {B}{logged_pedestal}{NC}")
        shutil.move(output, logged_pedestal)
        utc_ts = sorted(timestamps)[0]
        log_entry = (
            f"PEDESTAL {utc_ts.isoformat()} {exposure} {logged_pedestal.resolve()}"
        )
        # Check this line does not exist already
        if log_entry in pedestal_log.read_text():
            logger.info(
                "Not updating calibration log as identical entry already exists."
            )
        else:
            logger.info(f"Writing calibration log entry:\n    {log_entry}")
            with pedestal_log.open("a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
