from __future__ import annotations

import configparser
import enum
import importlib.resources
import logging
import os
import socket
import sys
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy
import numpy.typing

from .util import BOLD, NC, B

logger = logging.getLogger(__name__)

# def get_known_detectors() -> set[str]:
#     """Get a list of known detectors from the configuration"""
#     # Since we don't have a literal detector list, learn by inspection.
#     # We expect every detector listed:
#     # - To have modules sections named "<detname>-<module>"
#     # - For each module to have at least a "module" key
#     config = get_config()
#     modules = [x for x in config if "module" in config[x]]
#     # Now, make a set of everything before the last -
#     return {module.rpartition("-")[0] for module in modules}

# TODO: We used to dynamically generate this. This causes mypy issues,
# so for now just hardcode the detectors (e.g. the one that we know we
# are testing)
# Dynamically create detector
# enum Detector = enum.StrEnum("Detector", [x.upper() for x in get_known_detectors()])


class Detector(str, enum.Enum):
    JF1MD = "jf1md"
    JF4MPSI = "jf4mpsi"

    def __str__(self):
        return self.value


_DETECTOR: Detector | None = None

NUM_GAIN_MODES = 3


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

    @classmethod
    def _missing_(cls, name):
        for member in cls:
            if member.name.lower() == name.lower():
                return member


def get_detector() -> Detector:
    """Get the currently selected detector"""
    if _DETECTOR is None:
        raise RuntimeError("Default detector not yet initialised")
    return _DETECTOR


@lru_cache
def get_config():
    """Get the local configuration from the installation directory"""
    configuration = configparser.ConfigParser()
    with importlib.resources.as_file(
        importlib.resources.files("morgul") / "morgul.ini"
    ) as fo:
        configuration.read(fo)
    return configuration


def get_known_modules_for_detector(detector: Detector) -> list[str]:
    """Get a list of known module IDs for a given detector"""
    config = get_config()
    return [
        x["module"]
        for k, x in config.items()
        if k.lower().startswith(str(detector.value).lower() + "-")
    ]


def get_known_module_layout_for_detector(detector: Detector) -> tuple[int, int]:
    """Returns [column_count, row_count] for a known detector"""
    if detector == Detector.JF1MD:
        return (1, 2)
    raise ValueError(f"Unsupported detector: {detector}")


class ModulePosition(NamedTuple):
    column: int
    row: int


def get_module_position(module_id: str) -> ModulePosition:
    module = get_module_from_id(module_id)
    return ModulePosition(column=module["column"], row=module["row"])


def get_module_index(detector: Detector, module_id: str) -> int:
    cols, rows = get_known_module_layout_for_detector(detector)
    pos = get_module_position(module_id)
    return pos.column * rows + pos.row


def get_module_info(detector: Detector, col: int, row: int) -> dict[str, Any]:
    """Get metainformation about a module from the detector and position"""
    return get_config()[f"{detector.value}-{col}{row}"]


def get_module_from_id(module_id: str) -> dict[str, Any]:
    # Find a module with this ID
    config = get_config()
    cands = [
        k
        for k, v in config.items()
        if "module" in v and config[k]["module"] == module_id
    ]
    logger.debug(f"Got candidates: {cands}\nfor: {config.keys()}")
    if not cands:
        raise KeyError(f"Could not find module entry for module {module_id}")
    elif len(cands) > 1:
        raise KeyError(f"More than one module entry for module {module_id}")

    return config[cands[0]]


@lru_cache
def get_calibration_path(hostname: str | None = None) -> Path:
    """Determine the calibration folder location"""

    if gain_maps_path := os.getenv("JUNGFRAU_GAIN_MAPS"):
        return Path(gain_maps_path)

    # Try the "Original" way, by resolving the hostname mapped to hardcoded paths
    hostname = hostname or socket.getfqdn()
    config = get_config()
    candidates = sorted([x for x in config.keys() if hostname.endswith(x)], key=len)
    if not candidates:
        logger.error(
            f"""\
Error:  Could not find hardcoded gain maps path for hostname {hostname}.
        Please either update the hardcoded map file, or - more likely -
        set the environment variable {BOLD}JUNGFRAU_GAIN_MAPS{NC} to point
        to their location."""
        )
        sys.exit(1)
    longest_match = candidates[-1]
    logging.debug(f"Matched configuration section {BOLD}{longest_match}{NC}")
    try:
        return Path(get_config()[longest_match]["calibration"])
    except KeyError as e:
        raise RuntimeError(f"Could not find calibration section: {e}")


@lru_cache
def psi_gain_maps(detector: Detector) -> dict[str, numpy.typing.NDArray[numpy.float64]]:
    """Read gain maps from installed location, return as 3 x numpy array g0, g1, g2"""
    config = get_config()
    calib = get_calibration_path()
    logger.info(f"Reading gain maps from: {B}{calib}{NC}")
    result = {}
    modules = [x for x in config.keys() if x.startswith(detector)]
    for k in modules:
        module = config[k]["module"]
        gain_file = list(calib.joinpath(f"{module}_fullspeed").glob("*.bin"))
        assert len(gain_file) == 1
        shape = 3, 512, 1024
        count = shape[0] * shape[1] * shape[2]
        gains = numpy.fromfile(
            open(gain_file[0], "r"), dtype=numpy.float64, count=count
        ).reshape(*shape)
        result[module] = gains

    if not result:
        raise RuntimeError(f"Got no gain map results for detector {detector.value}")

    return result
