import contextlib
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Optional, cast, overload

import h5py
import hdf5plugin
import numpy
import numpy.typing
import tqdm
import typer

from .config import (
    Detector,
    ModuleMode,
    get_detector,
    get_known_module_layout_for_detector,
    get_known_modules_for_detector,
    get_module_info,
    psi_gain_maps,
)
from .util import (
    NC,
    B,
    G,
    elapsed_time_string,
    find_mask,
    find_pedestal,
    strip_escapes,
)

logger = logging.getLogger(__name__)


class PedestalCorrections:
    """
    Read in pedestal correction files, and make it easy to retrieve the data
    """

    detector: Detector
    filename: Path
    module_mode: ModuleMode
    _tables: dict[tuple[float, str | int, int], numpy.typing.NDArray]

    def __init__(self, detector: Detector, filename: Path):
        self.detector = detector
        self.filename = filename

        self._tables = {}
        with h5py.File(filename, "r") as f:
            if "module_mode" in f:
                self.module_mode = ModuleMode(f["module_mode"][()])
            else:
                self.module_mode = ModuleMode.FULL

            module_group_names: dict[str, str | int]
            if self.module_mode == ModuleMode.FULL:
                module_group_names = {
                    x: x for x in get_known_modules_for_detector(detector)
                }
                assert module_group_names
            else:
                (det_cols, det_rows) = get_known_module_layout_for_detector(detector)
                module_group_names = {
                    f"hmi_{x:02d}": x for x in range(0, det_cols * det_rows * 2 + 1)
                }

            # For now, just read all data into memory
            exposure_time = f["exptime"][()]
            for group_name, mod in module_group_names.items():
                if group_name not in f:
                    logger.warning(
                        f"Could not find module group named '{group_name}'. Pedestal data may be invalid."
                    )
                    continue
                # Read any gain-mode pedestals out of this file
                for gainmode in range(3):
                    if f"pedestal_{gainmode}" in f[group_name]:
                        self._tables[exposure_time, mod, gainmode] = numpy.copy(
                            f[group_name][f"pedestal_{gainmode}"]
                        )

    @property
    def exposure_times(self):
        return set(x for x, _, _ in self._tables)

    def has_exposure(self, exposure: float) -> bool:
        return any(abs(x - exposure) < 1e-9 for x, _, _ in self._tables)

    def get_pedestal(
        self, exposure_time: float, module: str, gain_mode: int
    ) -> numpy.typing.NDArray:
        try:
            return self._tables[exposure_time, module, gain_mode]
        except KeyError:
            raise KeyError(
                f"No pedestal data for (exposure: {exposure_time}, module: {module}, mode: {gain_mode})"
            )

    def has_pedestal(
        self, exposure_time: float, module: str, gain_mode: int | None = None
    ) -> bool:
        """Do we contain the pedestal correction tables for a specific module?"""
        if gain_mode is not None:
            return (exposure_time, module, gain_mode) in self._tables
        else:
            # Only say yes if we have all pedestal data (which should always be
            # true - we should never generate a file without all three)
            matches = [
                g for t, m, g in self._tables if t == exposure_time and m == module
            ]
            assert len(matches) == 3, f"Expected 3 pedestal maps, got {len(matches)}"
            return True

    def get_pedestals_dict(
        self, exposure_time: float, module: str
    ) -> dict[str, numpy.typing.NDArray]:
        pass

    @overload
    def __getitem__(
        self, key: float | tuple[float]
    ) -> dict[str, dict[int, numpy.typing.NDArray]]:
        pass

    @overload
    def __getitem__(self, key: tuple[float, str]) -> dict[int, numpy.typing.NDArray]:
        pass

    @overload
    def __getitem__(self, key: tuple[float, str, int]) -> numpy.typing.NDArray:
        pass

    def __getitem__(
        self, key: float | tuple[float] | tuple[float, str] | tuple[float, str, int]
    ):
        output: Any = {}
        try:
            if isinstance(key, float):
                e_in = key
            else:
                e_in = key[0]
            exact_exptime = [x for x in self.exposure_times if abs(e_in - x) < 1e-9][0]
        except IndexError:
            raise KeyError(f"No exposure time entry in pedestal matching {key}")

        if isinstance(key, float):
            key = (key,)
        if len(key) == 1:
            # Get all entries for one exposure time
            for module, gainmode in [
                (m, g) for e, m, g in self._tables if e == exact_exptime
            ]:
                output.setdefault(module, dict())[gainmode] = self._tables[
                    key[0], module, gainmode
                ]
        elif len(key) == 2:
            # Get all entries for one exposure time
            key = cast(tuple[float, str], key)
            for gainmode in [
                g for e, m, g in self._tables if e == exact_exptime and m == key[1]
            ]:
                # output.setdefault(module, dict())[gainmode] = self._tables(key, module, gainmode)
                output[gainmode] = self._tables[exact_exptime, key[1], gainmode]
        elif len(key) == 3:
            return self._tables[exact_exptime, key[1], key[2]]  # type: ignore
        return output


class Masker:
    filename: Path
    exposure_times: set[float]

    def __init__(self, detector: Detector, filename: Path):
        self.filename = filename
        modules = get_known_modules_for_detector(detector)

        self._table = {}

        # Read the masks out of this data file
        with h5py.File(filename, "r") as f:
            exptime = f["exptime"][()]
            self.exposure_times = {exptime}
            for module in modules:
                if module in f:
                    self._table[exptime, module] = numpy.copy(f[module]["mask"])

    def __getitem__(self, key: tuple[float, str]) -> numpy.typing.NDArray:
        if key not in self._table:
            time_keys = {x[0] for x in self._table}
            fudge_time = list(time_keys)[0]
            if len(time_keys) == 1:
                return self._table[fudge_time, key[1]]
        return self._table[key]

    def __contains__(self, key: tuple[float, str]) -> bool:
        return key in self._table


def get_pedestals(pedestal_file):
    pedestals = {}

    with h5py.File(pedestal_file, "r") as f:
        for k in "p0", "p1", "p2":
            if k in f:
                pedestals[k] = f[k][()]

    return pedestals


def embiggen(packed):
    """Unpack the data from ASICS to the pixel-doubled form, masking the affected
    pixels so this has the result of just slightly embiggening the images. Since
    this is a copy have it also do the inversion (pay attention people.)"""

    assert packed.shape == (512, 1024)

    bigger = numpy.full((514, 1030), -1, dtype=numpy.int32)

    for i in range(2):
        for j in range(4):
            for k in range(1, 255):
                I = i * 256
                _I = 513 - i * 258 - k
                J = j * 256
                _J = j * 258 - 1 if j else 0
                bigger[_I, (_J + 1) : (_J + 255)] = packed[I + k, (J + 1) : (J + 255)]

    return bigger


def correct_frame(
    raw: numpy.typing.NDArray[numpy.uint16],
    pedestals: dict[int, numpy.typing.NDArray],
    g012,
    energy: float,
    mask: numpy.typing.NDArray | None = None,
):
    """Correct pixel values to photons in frame"""

    if mask is None:
        mask = numpy.full(raw.shape, False, dtype=bool)

    assert 1 in pedestals and 2 in pedestals and 0 in pedestals

    gain = numpy.right_shift(raw, 14)

    mask = mask == False
    m0 = (pedestals[0] != 0) * mask
    frame = ((gain == 0) * (numpy.bitwise_and(raw, 0x3FFF)) * m0 - pedestals[0]) / (
        g012[0] * energy
    )

    if 1 in pedestals:
        m1 = (pedestals[1] != 0) * mask
        frame += (
            (gain == 1) * ((numpy.bitwise_and(raw, 0x3FFF)) * m1 - pedestals[1])
        ) / (g012[1] * energy)

    if 2 in pedestals:
        m2 = (pedestals[2] != 0) * mask
        frame += (
            (gain == 3) * ((numpy.bitwise_and(raw, 0x3FFF)) * m2 - pedestals[2])
        ) / (g012[2] * energy)
    return frame


def output_filename(filename: Path, output_dir: Path | None) -> Path:
    # Work out where the output file will go
    return (
        output_dir or filename.parent
    ) / f"{filename.stem}_corrected{filename.suffix}"


def datafile_prechecks(
    data_files: list[Path], force: bool, output_dir: Path, stack: contextlib.ExitStack
) -> dict[Path, h5py.File]:
    """Open data files, and do basic pre-correction sanity checks"""
    # Do a pre-pass so that we can count the total number of images
    h5s = {}
    total_images = 0
    existing_output_filenames = []
    # Go through every data file input on a first pass
    for filename in data_files:
        h5 = stack.enter_context(h5py.File(filename, "r"))
        # If this file is already corrected, ignore it
        if "data" not in h5:
            logger.error(f"Error: File {filename} does not have a 'data' dataset")
            raise typer.Abort()
        # If this was previously corrected, ignore it
        if "corrected" in h5["data"].attrs and h5["data"].attrs["corrected"]:
            logger.warning(f"File {filename} contains corrected data, ignoring.")
            h5.close()
            continue
        total_images += h5["data"].shape[0]
        h5s[filename] = h5
        # Work out what the output filename would be
        out = output_filename(filename, output_dir)
        if out.is_file():
            existing_output_filenames.append(out)

    # Handle output filename existence. Do this so that we print everything
    # that could be overwritten, instead of the first - in which case it
    # might unexpectedly overwrite a file the user didn't expect
    if existing_output_filenames and not force:
        outputs = "\n".join(["  - " + str(x) for x in existing_output_filenames])
        logger.error(
            f"""
Error: The following files already exist and would be overwritten:

{outputs}

please pass --force/-f if you want to overwrite these files.
"""
        )
        raise typer.Abort()

    if not h5s:
        logger.error("Error: No data files present after filtering out corrected")
        raise typer.Abort()

    return h5s


def correct(
    data_files: Annotated[
        list[Path], typer.Argument(help="Data files, for corrections.", metavar="DATA")
    ],
    energy: Annotated[
        float, typer.Option("-e", "--energy", help="photon energy (keV)")
    ],
    pedestal_file: Annotated[
        Optional[Path],
        typer.Option(
            "-p",
            "--pedestal",
            help="Pedestal data file for the module(s), from 'morgul pedestal'. If not specified, JUNGFRAU_CALIBRATION_LOG search will be used.",
        ),
    ] = None,
    mask_file: Annotated[
        Optional[Path],
        typer.Option(
            "-m",
            "--mask",
            help="Pixel mask, from 'morgul mask'. If not specified, JUNGFRAU_CALIBRATION_LOG will be used to find a mask.",
        ),
    ] = None,
    no_mask: Annotated[
        bool,
        typer.Option("--no-mask", help="If set, a mask will not be used"),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            help="Output folder for the corrected files. Files will be written here with the suffix '_corrected.<ext>'. Defaults to same as input file. ",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("-f", "--force", help="Overwrite files that already exist")
    ] = False,
    lookup_tolerance: Annotated[
        Optional[int],
        typer.Option(
            help="If set, only pedestal data collected within this many minutes will be selected when using the calibration log."
        ),
    ] = None,
):
    """
    Given data, pedestal and mask files, correct the data into photon counts.
    """

    start_time = time.monotonic()
    detector = get_detector()
    logger.info(f"Using detector: {G}{detector.value}{NC}")

    gain_maps = psi_gain_maps(detector)

    pedestal_readers: dict[Path, PedestalCorrections] = {}
    mask_readers: dict[Path, Masker] = {}
    cached_pedestals = {}
    cached_maskers = {}

    with contextlib.ExitStack() as stack:
        # Do basic cross-checks and filter out corrected files
        h5s = datafile_prechecks(data_files, force, output, stack)

        # If given explicit pedestal/mask file, open and assign them now
        if pedestal_file:
            pedestal_reader = PedestalCorrections(detector, pedestal_file)
            logger.info(f"Reading pedestals from: {B}{pedestal_file}{NC}")

            for filename in data_files:
                pedestal_readers[filename] = pedestal_reader
        if mask_file and not no_mask:
            logger.info(f"Using mask from:        {B}{mask_file}{NC}")
            masker = Masker(detector, mask_file)
            for filename in data_files:
                mask_readers[filename] = masker

        total_images = sum(x["data"].shape[0] for x in h5s.values())
        print(f"Correcting total of: {G}{total_images}{NC} images")

        # Do validations for everything before we start correcting
        for filename, h5 in h5s.items():
            exposure_time = h5["exptime"][()]
            timestamp = h5["timestamp"][()]
            if filename not in pedestal_readers:
                # Try to find one
                reader = find_pedestal(
                    timestamp,
                    exposure_time,
                    within_minutes=lookup_tolerance,
                )
                if reader not in cached_pedestals:
                    logger.info(
                        f"Reading {G}{exposure_time*1000:g}ms{NC} pedestals from: {B}{reader}{NC}"
                    )
                    cached_pedestals[reader] = PedestalCorrections(detector, reader)

                pedestal_readers[filename] = cached_pedestals[reader]
            if filename not in mask_readers and not no_mask:
                # Try to find one
                reader = find_mask(
                    timestamp,
                    exposure_time,
                )
                if reader not in cached_maskers:
                    logger.info(
                        f"Reading {G}{exposure_time*1000:g}ms{NC} mask from:        {B}{reader}{NC}"
                    )
                    cached_maskers[reader] = Masker(detector, reader)
                mask_readers[filename] = cached_maskers[reader]

            # Validate that the pedestal reader has this timestamp. This
            # could happen if the user requested a specific pedestal file
            if not pedestal_readers[filename].has_exposure(exposure_time):
                availables = ", ".join(
                    f"{x*1000:g}" for x in pedestal_readers[filename].exposure_times
                )
                logger.error(
                    f"Error: {filename} is exposure {exposure_time*1000:g} ms, only: {availables} ms available."
                )
                raise typer.Abort()

            # Validate that the mask reader has this exposure time. This
            # is not an error, but we do want to print the user a warning
            if not no_mask and exposure_time not in (
                exps := mask_readers[filename].exposure_times
            ):
                availables = ", ".join(f"{x*1000:g}" for x in exps)
                if availables:
                    logger.warning(
                        f"Warning: Using masker time point {availables}ms instead of {exposure_time*1000:g}ms"
                    )

            # Validate that the file is dynamic
            if not (gainmode := h5["gainmode"][()].decode()) == "dynamic":
                logger.error(f"Error: {filename} is '{gainmode}', not 'dynamic'")
                raise typer.Abort()

        # Start the correction/output process
        progress = stack.enter_context(tqdm.tqdm(total=total_images, leave=False))
        for filename, h5 in h5s.items():
            # Get the module this data was taken with
            module = get_module_info(detector, h5["column"][()], h5["row"][()])[
                "module"
            ]

            data = h5["data"]
            exposure_time = h5["exptime"][()]

            # Check we have a mask for this module
            # if (exposure_time, module) not in masker:
            #     logger.warning(
            #         f"Error: Do not have mask for (exptime: {exposure_time*1000:g} ms, module: {module})"
            #     )

            out_filename = output_filename(filename, output)
            pre_msg = f"Processing {G}{data.shape[0]}{NC} images from module {G}{module}{NC} in "
            progress.write(
                f"{pre_msg}{B}{filename}{NC}\n{' '*(len(strip_escapes(pre_msg))-5)}into {B}{out_filename}{NC}"
            )

            # Safety check - don't overwrite if no --force
            if out_filename.is_file() and not force:
                logger.error(
                    f"Error: {out_filename} exists but will not overwrite. Pass --force to overwrite."
                )
                raise typer.Abort()

            with h5py.File(out_filename, "w") as f:
                out_dataset = f.create_dataset(
                    "data",
                    shape=(data.shape[0], 514, 1030),
                    dtype=numpy.int32,
                    chunks=(1, 514, 1030),
                    **hdf5plugin.Bitshuffle(cname="lz4"),
                )
                out_dataset.attrs["corrected"] = True
                for n in tqdm.tqdm(
                    range(data.shape[0]), leave=False, desc=f"{filename.name}"
                ):
                    frame = correct_frame(
                        data[n],
                        pedestal_readers[filename][exposure_time, module],
                        gain_maps[module],
                        energy,
                        (
                            mask_readers[filename][exposure_time, module]
                            if not no_mask
                            else None
                        ),
                    )
                    progress.update(1)
                    out_dataset[n] = embiggen(numpy.around(frame))
                # Copy over all other metadata
                for k, v in h5.items():
                    if isinstance(v, h5py.Dataset) and v.shape == ():
                        f.create_dataset(k, data=v)

    print()
    logger.info(
        f"Written {G}{total_images}{NC} images in {G}{len(data_files)}{NC}s corrected data files in {elapsed_time_string(start_time)}."
    )
