from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import shutil
import subprocess
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, overload

import requests
import typer

logger = logging.getLogger(__name__)


# Collection start/end script to kick off analysis
COLLECTION_START_SCRIPT = "/dls_sw/i24/scripts/RunAtStartOfCollect-i24-ssx.sh"
COLLECTION_END_SCRIPT = "/dls_sw/i24/scripts/RunAtEndOfCollect-i24-ssx.sh"

DEFAULT_ISPYB_SERVER = "https://ssx-dcserver-test.diamond.ac.uk"

CREDENTIALS_LOCATION = "/scratch/ssx_dcserver_test.key"

# Real distance = readback + this
DISTANCE_OFFSET = 7.5


@lru_cache
def _get_caget_location() -> Path:
    if path := shutil.which("caget"):
        return Path(path)
    ppath = Path("/dls_sw/epics/R3.14.12.7/base/bin/linux-x86_64/caget")
    if ppath.is_file():
        return ppath

    raise RuntimeError("Could not find caget executable")


class EPICSTimeout(RuntimeError):
    pass


@overload
def caget(pv: str, string: Literal[True]) -> str:
    pass


@overload
def caget(pv: str, string: Literal[False] = False) -> int | float | str:
    pass


def caget(pv: str, string: bool = False) -> int | float | str:
    try:
        extra_flags = ["-S"] if string else []
        value = subprocess.run(
            [_get_caget_location(), "-t", *extra_flags, pv],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass

        return value

    except subprocess.CalledProcessError as e:
        if "timed out" in e.stderr:
            raise EPICSTimeout(f"Could not fetch PV: {e.stderr}")
        else:
            raise RuntimeError(f"Unknown error fetching PV: {e.stderr}")


def cagetstring(pv: str) -> str:
    return caget(pv, string=True)


class Jungfrau9M:
    id = (
        124
        if DEFAULT_ISPYB_SERVER != "https://ssx-dcserver-test.diamond.ac.uk"
        else 114
    )
    # fast, slow / width, height
    image_size_pixels = (3262, 3108)
    pixel_size_mm = (0.075, 0.075)
    image_size_mm = tuple(
        round(a * b, 3) for a, b in zip(image_size_pixels, pixel_size_mm)
    )

    class pv:
        detector_distance = "BL24I-EA-DET-01:Z.VAL"
        wavelength = "BL24I-EA-EIGER-01:CAM:Wavelength"
        transmission = "BL24I-EA-PILAT-01:cam1:FilterTransm"
        flux = "BL24I-EA-FLUX-01:XBPM-02"
        beamx = None
        beamy = None


type Detector = Jungfrau9M


@lru_cache(maxsize=1)
def get_auth_header() -> dict:
    """Read the credentials file and build the Authorisation header"""
    if not os.path.isfile(CREDENTIALS_LOCATION):
        logger.warning(
            "Could not read %s; attempting to proceed without credentials",
            CREDENTIALS_LOCATION,
        )
        return {}
    with open(CREDENTIALS_LOCATION) as f:
        token = f.read().strip()
    return {"Authorization": "Bearer " + token}


class ExperimentKind(Enum):
    OSC = "OSC"
    FIXED = "Serial Fixed"
    EXTRUDER = "Serial Jet"


class DCID:
    """
    Interfaces with ISPyB to allow ssx DCID/synchweb interaction.

    Args:
        server: The URL for the bridge server, if not the default.
        emit_errors:
            If False, errors while interacting with the DCID server will
            not be propagated to the caller. This decides if you want to
            stop collection if you can't get a DCID
        timeout: Length of time to wait for the DB server before giving up
        experiment_kind: The type of experiment this is for
        visit: The name of the visit e.g. "mx12345-4"
        image_dir: The location the images will be written



    Attributes:
        error:
            If an error has occured. This will be set, even if emit_errors = True
    """

    def __init__(
        self,
        kind: ExperimentKind,
        nexus_filename: Path,
        *,
        server: str = None,
        emit_errors: bool = True,
        timeout: float = 10,
        start_time: datetime.datetime | None = None,
        num_images: int,
        exposure_time: float,
        axisEnd: float,
        rotation_increment: float,
    ):
        detector = Jungfrau9M()
        nexus_filename = nexus_filename.resolve()
        # coll = nexus_filename.parent / "collection_info.json"
        # if coll.is_file():
        #     collection_info = json.loads(coll.read_bytes())
        #     rotation_increment = collection_info.get("angular_increment_deg", rotation_increment)

        self.server = server or DEFAULT_ISPYB_SERVER
        self.emit_errors = emit_errors
        self.error = False
        self.timeout = timeout
        self.experiment_kind = ExperimentKind(kind)
        self.dcid = None
        try:
            if not start_time:
                start_time = datetime.datetime.now().astimezone()
            elif not start_time.timetz:
                start_time = start_time.astimezone()

            # Gather data from the beamline
            detector_distance = (
                float(caget(detector.pv.detector_distance)) + DISTANCE_OFFSET
            )
            wavelength = float(caget(detector.pv.wavelength))
            resolution = get_resolution(detector, detector_distance, wavelength)
            beamsize_x, beamsize_y = get_beamsize()
            transmission = float(caget(detector.pv.transmission)) * 100
            # xbeam, ybeam = get_beam_center(detector)
            ybeam, xbeam = (1771.97, 1689.09)
            flux = float(caget(detector.pv.flux))
            if isinstance(detector, Jungfrau9M):
                # Mirror the construction that the PPU does
                fileTemplate = str(nexus_filename)
                startImageNumber = 1
            else:
                raise ValueError("Unknown detector:", detector)

            (_, _, _, _, _, visit, *_) = Path.cwd().parts

            data = {
                "detectorDistance": float(detector_distance),
                "detectorId": detector.id,
                "exposureTime": float(exposure_time),
                "fileTemplate": fileTemplate,
                "imageDirectory": str(nexus_filename.parent),
                "numberOfImages": int(num_images),
                "resolution": float(resolution),
                "startImageNumber": startImageNumber,
                "startTime": start_time.isoformat(),
                "transmission": float(transmission),
                "visit": visit,
                "wavelength": float(wavelength),
                "group": {"experimentType": self.experiment_kind.value},
                "xBeam": xbeam,
                "yBeam": ybeam,
                "rotationAxis": "Omega",
                "axisStart": 0.0,
                "axisEnd": axisEnd or 360.0,
                "axisRange": rotation_increment,
                "flux": flux,
            }
            if beamsize_x and beamsize_y:
                data["beamSizeAtSampleX"] = beamsize_x / 1000
                data["beamSizeAtSampleY"] = beamsize_y / 1000

            # Log what we are doing here
            try:
                logger.info(
                    "BRIDGE: POST /dc --data %s",
                    repr(json.dumps(data)),
                )
            except Exception:
                logger.info(
                    "Caught exception converting data to JSON. Data:\n%s\nVERBOSE:\n%s",
                    str({k: type(v) for k, v in data.items()}),
                )
                raise

            resp = requests.post(
                f"{self.server}/dc",
                json=data,
                timeout=self.timeout,
                headers=get_auth_header(),
            )
            resp.raise_for_status()
            self.dcid = resp.json()["dataCollectionId"]
            logger.info("Generated DCID %s", self.dcid)
        except requests.HTTPError as e:
            self.error = True
            logger.error(
                "DCID generation Failed; Reason from server: %s", e.response.text
            )
            if self.emit_errors:
                raise
            logger.exception("Error generating DCID: %s", e)
        except Exception as e:
            self.error = True
            if self.emit_errors:
                raise
            logger.exception("Error generating DCID: %s", e)

    def __int__(self):
        return self.dcid

    def notify_start(self):
        """Send notifications that the collection is now starting"""
        if self.dcid is None:
            return None
        try:
            command = [COLLECTION_START_SCRIPT, str(self.dcid)]
            logger.info("Running %s", " ".join(command))
            subprocess.Popen(command)
        except Exception as e:
            self.error = True
            if self.emit_errors:
                raise
            logger.warning("Error starting start of collect script: %s", e)

    def notify_end(self):
        """Send notifications that the collection has now ended"""
        if self.dcid is None:
            return
        try:
            command = [COLLECTION_END_SCRIPT, str(self.dcid)]
            logger.info("Running %s", " ".join(command))
            subprocess.Popen(command)
        except Exception as e:
            self.error = True
            if self.emit_errors:
                raise
            logger.warning("Error running end of collect notification: %s", e)

    def collection_complete(
        self, end_time: str | datetime.datetime = None, aborted: bool = False
    ) -> None:
        """
        Mark an ispyb DCID as completed.

        Args:
            dcid: The Collection ID to mark as finished
            end_time: The predetermined end time
            aborted: Was this collection aborted?
        """
        try:
            # end_time might be a string from time.ctime
            if isinstance(end_time, str):
                end_time = datetime.datetime.strptime(end_time, "%a %b %d %H:%M:%S %Y")
                logger.info("Parsed end time: %s", end_time)

            if not end_time:
                end_time = datetime.datetime.now().astimezone()
            if not end_time.tzinfo:
                end_time = end_time.astimezone()

            status = (
                "DataCollection Cancelled" if aborted else "DataCollection Successful"
            )
            data = {
                "endTime": end_time.isoformat(),
                "runStatus": status,
            }
            if self.dcid is None:
                # Print what we would have sent. This means that if something is failing,
                # we still have the data to upload in the log files.
                logger.info(
                    'BRIDGE: No DCID but Would PATCH "/dc/XXXX" --data=%s',
                    repr(json.dumps(data)),
                )
                return

            logger.info(
                'BRIDGE: PATCH "/dc/%s" --data=%s', self.dcid, repr(json.dumps(data))
            )
            response = requests.patch(
                f"{self.server}/dc/{self.dcid}",
                json=data,
                timeout=self.timeout,
                headers=get_auth_header(),
            )
            response.raise_for_status()
            logger.info("Successfully updated end time for DCID %d", self.dcid)
        except Exception as e:
            try:
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None:
                    resp_str = resp_obj.text
                # resp_str = repr(getattr(e, "Iresponse", "<no attribute>"))
                else:
                    resp_str = "Resp object is None"
            except Exception:
                resp_str = f"<failed to determine {resp_obj!r}>"

            self.error = True
            if self.emit_errors:
                raise
            logger.warning("Error completing DCID: %s (%s)", e, resp_str)


def get_pilatus_filename_template_from_pvs() -> str:
    """
    Get the template file path by querying the detector PVs.

    Returns: A template string, with the image numbers replaced with '#'
    """

    filename = cagetstring(pv.pilat_filename)
    filename_template = cagetstring(pv.pilat_filetemplate)
    file_number = int(caget(pv.pilat_filenumber))
    # Exploit fact that passing negative numbers will put the - before the 0's
    expected_filename = filename_template % (filename, f"{file_number:05d}_", -9)
    # Now, find the -09 part of this
    numberpart = re.search(r"(-0+9)", expected_filename)
    # Make sure this was the only one
    if re.search(r"(-0+9)", expected_filename[numberpart.end() :]) is not None:
        logger.error(
            f"Got unexpected extra numerical part in filename {expected_filename} : {expected_filename[numberpart.end() :]}",
            stack_info=True,
        )
    template_fill = "#" * len(numberpart.group(0))
    return (
        expected_filename[: numberpart.start()]
        + template_fill
        + expected_filename[numberpart.end() :]
    )


def get_beamsize() -> tuple[float | None, float | None]:
    """
    Read the PVs to get the current beamsize.

    Returns:
        A tuple (x, y) of beam size (in µm). These values can be 'None'
        if the focus mode was unrecognised.
    """
    # These I24 modes are from GDA
    focus_modes = {
        "focus10": ("7x7", 7, 7),
        "focus20d": ("20x20", 20, 20),
        "focus30d": ("30x30", 30, 30),
        "focus50d": ("50x50", 50, 50),
        "focus1050d": ("10x50", 10, 50),
        "focus5010d": ("50x10", 50, 10),
        "focus3010d": ("30x10", 30, 10),
    }
    v_mode = cagetstring("BL24I-OP-MFM-01:G0:TARGETAPPLY")
    h_mode = cagetstring("BL24I-OP-MFM-01:G1:TARGETAPPLY")
    # Validate these and note an error otherwise
    if not v_mode.startswith("VMFM") or v_mode[4:] not in focus_modes:
        logger.error("Unrecognised vertical beam mode %s", v_mode)
    if not h_mode.startswith("HMFM") or h_mode[4:] not in focus_modes:
        logger.error("Unrecognised horizontal beam mode %s", h_mode)
    _, h, _ = focus_modes.get(h_mode[4:], (None, None, None))
    _, _, v = focus_modes.get(v_mode[4:], (None, None, None))

    return (h, v)


def get_resolution(detector: Detector, distance: float, wavelength: float) -> float:
    """
    Calculate the inscribed resolution for detector.

    This assumes perfectly centered beam as I don't know where to
    extract the beam position parameters yet.

    Args:
        distance: Distance to detector (mm)
        wavelength: Beam wavelength (Å)

    Returns:
        Maximum resolution (Å)
    """
    width = detector.image_size_mm[0]
    return round(wavelength / (2 * math.sin(math.atan(width / (2 * distance)) / 2)), 2)


def get_beam_center(detector: Detector) -> tuple[float, float]:
    """Get the detector beam center, in mm"""
    beamX = float(caget(detector.pv.beamx)) * detector.pixel_size_mm[0]
    beamY = float(caget(detector.pv.beamy)) * detector.pixel_size_mm[1]
    return (beamX, beamY)


def test_beamsize():
    print("Beam size:", get_beamsize())


def dcid(
    nexus_filename: Annotated[Path, typer.Argument()],
    energy: Annotated[float | None, typer.Option(help="Beam energy, in keV")] = None,
    rotation_angle: Annotated[
        float | None, typer.Option(help="Rotation increment per frame")
    ] = None,
):
    """Create a DCID for a Nexus'd jungfrau collection"""
    # Look for an associated collection info
    collection_info = nexus_filename.parent / "collection_info.json"
    if collection_info.is_file():
        print(f"Reading data out of {collection_info}")
        cp = json.loads(collection_info.read_bytes())
        energy = energy or cp["energy_kev"]
        rotation_angle = rotation_angle or cp["angular_increment_deg"]

    dcid = DCID(
        "OSC",
        nexus_filename,
        num_images=3600,
        exposure_time=0.001,
        axisEnd=360,
        rotation_increment=rotation_angle,
    )
    # def __init__(
    #     self,
    #     kind: ExperimentKind,
    #     nexus_filename: Path,
    #     *,
    #     server: str = None,
    #     emit_errors: bool = True,
    #     timeout: float = 10,
    #     start_time: datetime.datetime,
    #     num_images: int,
    #     exposure_time: float,
    #     axisEnd: float,
    # ):

    # {"wavelength_in_a": 0.9999, "energy_kev": 12.4, "angular_increment_deg": 0.1, "beam_xy_mm": [133.5, 150.8], "detector_distance_mm": null}
    # assert data_files, "Cannot merge no data files"

    # assert len({x.resolve().parent for x in data_files}) == 1, (
    #     "Input files are spread over multiple folders"
    # )

    # # If we don't have an output filename, work out the common prefix
    # if not output_filename:
    #     output_filename = common_output_filename(data_files)

    # # Even if the user supplied their own filename, write the output
    # # file to the same place as the images (if all in same place)
    # if len(set(x.parent for x in data_files)) == 1:
    #     parent = data_files[0].parent
    #     output_filename = parent / output_filename.name

    # # Work out a nice way to print this
    # output_print = str(output_filename)
    # if str(output_filename.parent) == ".":
    #     output_print = f"./{output_print}"

    # print(f"Reading {len(data_files)} data files")

    # with h5py.File(output_filename, "w") as out:
    #     create_vds_dataset(out, "data", data_files)

    # print(f"Written output to {output_print}")
