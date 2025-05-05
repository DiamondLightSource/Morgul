import contextlib
import datetime
import functools
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Annotated, Optional

import dateutil.tz as tz
import h5py
import numpy
import numpy.typing
import tqdm
import typer

from .config import (
    ModuleMode,
    get_detector,
    get_known_module_layout_for_detector,
    get_module_info,
    psi_gain_maps,
)
from .morgul_correct import PedestalCorrections, correct_frame
from .util import NC, B, G, elapsed_time_string

logger = logging.getLogger(__name__)


def _calculate(
    h5: h5py.Group,
    pedestals: dict[int, numpy.typing.NDArray],
    gain_maps: dict[str, numpy.typing.NDArray],
    energy: float,
    *,
    progress_desc: str | None = None,
    parent_progress: tqdm.tqdm | None = None,
) -> numpy.typing.NDArray[numpy.uint32]:
    """Use the data given in filename to derive a trusted pixel mask"""

    data = h5["data"]
    s = data.shape

    # fetch the correct gain maps for this module
    image = numpy.zeros(shape=(s[1], s[2]), dtype=numpy.float64)
    square = numpy.zeros(shape=(s[1], s[2]), dtype=numpy.float64)

    gain_mode = h5["gainmode"][()].decode()
    assert gain_mode == "dynamic", (
        f"Data with gain mode 'dynamic' (this is {gain_mode}) required for mask calculation"
    )

    # compute sum, sum of squares down stack
    for j in tqdm.tqdm(range(data.shape[0]), desc=progress_desc or "Mask", leave=False):
        frame = correct_frame(data[j], pedestals, gain_maps, energy)
        image += frame
        square += numpy.square(frame)
        if parent_progress is not None:
            parent_progress.update(1)

    mean = image / data.shape[0]
    var = square / data.shape[0] - numpy.square(mean)
    mean[mean == 0] = 1
    disp = var / mean
    writer = print if parent_progress is None else parent_progress.write
    writer(f"{progress_desc}: Masking {numpy.count_nonzero(disp > 3)} pixels")

    return (disp > 3).astype(numpy.uint32)


def mask(
    pedestal: Annotated[
        Path,
        typer.Argument(
            help="Pedestal data file for the module(s), from 'morgul pedestal'. Used when correcting in order to calculate the mask.",
            show_default=False,
        ),
    ],
    flat: Annotated[
        list[Path],
        typer.Argument(
            help="Flat-field data to use for mask generation. Multiple modules for a single exposure time can be passed, but must be present in the pedestal file.",
            show_default=False,
        ),
    ],
    energy: Annotated[
        float,
        typer.Option(
            "-e",
            "--energy",
            help="photon energy (keV)",
            show_default=False,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            help="Name for the output HDF5 file. Default: <detector>_<exptime>ms_mask.h5",
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
    """Calculate a pixel mask from flatfield data."""
    start_time = time.monotonic()
    detector = get_detector()
    logger.info(f"Using detector: {G}{detector.value}{NC}")

    gain_maps = psi_gain_maps(detector)

    exposure_time: float | None = None

    pedestals = PedestalCorrections(detector, pedestal)
    logger.info(f"Reading pedestals from: {B}{pedestal}{NC}")

    timestamps: set[float] = set()

    # Open the flat-field, and validate that they are matches
    with contextlib.ExitStack() as stack:
        # Read the data from the pedestal file
        # with h5py.File(pedestal, "r") as f:
        #     # Fix the exposure time for this pedestal file
        #     exposure_time = f["exptime"][()]

        total_images = 0
        calls = []
        for filename in flat:
            h5 = stack.enter_context(h5py.File(filename, "r"))
            exptime = h5["exptime"][()]
            timestamps.add(h5["timestamp"][()])

            # Validate that this exposure time is identical and present in the pedestal data
            if exposure_time is None:
                exposure_time = exptime
                if exptime not in pedestals.exposure_times:
                    logger.error(
                        f"Error: Flatfield has exposure {exptime} but pedestal data only contains ({', '.join(pedestals.exposure_times)})"
                    )
                    raise typer.Abort()
            elif exptime != exposure_time:
                logger.error(
                    f"Error: flatfield file {filename} exposure time ({exptime} does not match the pedestal data ({exposure_time})"
                )
                raise typer.Abort()
            # Validate that this module is present in the pedestal data
            module_mode = ModuleMode.from_shape(h5["data"].shape)
            module: str | int
            if module_mode == ModuleMode.FULL:
                module = get_module_info(
                    detector, int(h5["column"][()]), int(h5["row"][()])
                )["module"]
            else:
                _, rows = get_known_module_layout_for_detector(detector)
                module = h5["column"][()] * rows * 2 + h5["row"][()]

            if not pedestals.has_pedestal(exposure_time, module):
                logger.error(
                    f"Error: No data in pedestal file for module {module} for exposure {exposure_time * 1000:g} ms"
                )
                raise typer.Abort()

            if h5["gainmode"][()].decode() != "dynamic":
                logger.error(
                    f"Error: Data in file {filename} is not gainmode=dynamic (instead is '{h5['gainmode'][()]}') and is not suitable for masking"
                )
                raise typer.Abort()

            logger.info(
                f"Generating mask for {B}{filename}{NC}, module {G}{module}{NC} at {G}{exposure_time * 1000:g}{NC} ms"
            )
            # We know that we have data for this flatfield. Do the masking calculation.
            # _calculate(h5, pedestals[exposure_time, module], gain_maps[module], energy)
            total_images += h5["data"].shape[0]
            calls.append(
                (
                    module,
                    filename,
                    functools.partial(
                        _calculate,
                        h5,
                        pedestals[exposure_time, module],
                        gain_maps[module],
                        energy,
                    ),
                )
            )

        # Run these
        with tqdm.tqdm(total=total_images, leave=False) as progress:
            timestamp_name = datetime.datetime.fromtimestamp(
                sorted(timestamps)[0]
            ).strftime("%Y-%m-%d_%H-%M-%S")
            output = output or Path(
                f"{detector.value}_{exposure_time * 1000:g}ms_{timestamp_name}_mask.h5"
            )
            with h5py.File(output, "w") as h5_out:
                h5_out.create_dataset("exptime", data=exposure_time)
                for module, filename, call in calls:
                    mask_data = call(
                        parent_progress=progress, progress_desc=f" {module.strip()}"
                    )
                    if module not in h5_out:
                        h5_out.create_group(module)
                    h5_out[module].create_dataset("mask", data=mask_data)
                    h5_out[module]["mask"].attrs["from_flatfield"] = str(
                        filename.resolve()
                    )

        print()
        logger.info(
            f"Written output file {B}{output}{NC} in {elapsed_time_string(start_time)}."
        )

    if "JUNGFRAU_CALIBRATION_LOG" in os.environ and register_calibration:
        calib_log = Path(os.environ["JUNGFRAU_CALIBRATION_LOG"])
        logged_calib = calib_log.parent / output.name
        logger.info(f"Copying {B}{output}{NC} to {B}{logged_calib}{NC}")
        shutil.move(output, logged_calib)
        utc_ts = datetime.datetime.fromtimestamp(sorted(timestamps)[0]).replace(
            tzinfo=tz.UTC
        )
        log_entry = (
            f"MASK {utc_ts.isoformat()} {exposure_time} {logged_calib.resolve()}"
        )
        logger.info(f"Writing calibration log entry:\n    {log_entry}")
        with calib_log.open("a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
