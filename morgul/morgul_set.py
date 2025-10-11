import os
import shutil
import subprocess
import sys
from enum import Enum

from rich import print

CAPUT = shutil.which("caput")
CAGET = shutil.which("caget")


class PV(str, Enum):
    Path = "path"
    Name = "name"
    Frames = "frames"
    Captured = "captured"
    Subfolder = "subfolder"
    Ready = "ready"


PREFIX = os.getenv("FAUXDIN_PREFIX", "BL24I-JUNGFRAU-META:FD:")

PV_NAME = {
    PV.Path: "FilePath",
    PV.Name: "FileName",
    PV.Frames: "NumCapture",
    PV.Captured: "NumCaptured",
    PV.Subfolder: "Subfolder",
    PV.Ready: "Ready",
}

CA_FLAGS = {PV.Path: ["-S"], PV.Name: ["-S"]}


def get(pv: PV | None = None):
    if not CAGET:
        sys.exit("Error: Could not find caget on PATH")
    pvs = [pv] if pv else list(PV)
    cmd = [CAGET, "-tS", *[PREFIX + PV_NAME[x] for x in pvs]]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for pv, ret in zip(pvs, out.stdout.splitlines()):
        print(f"{pv.title():10} {ret}")


def set(pv: PV, value: str):
    """Set a Morgul-Jungfrau parameter"""
    if not CAPUT:
        sys.exit("Error: Could not find caput on PATH")
    cmd = [CAPUT, *CA_FLAGS.get(pv, []), PREFIX + PV_NAME[pv], value]
    subprocess.run(cmd, check=True, capture_output=True)
    get(pv)
