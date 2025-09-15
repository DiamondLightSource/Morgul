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


PV_NAME = {
    PV.Path: "BL24I-JUNGFRAU-META:FD:FilePath",
    PV.Name: "BL24I-JUNGFRAU-META:FD:FileName",
    PV.Frames: "BL24I-JUNGFRAU-META:FD:NumCapture",
    PV.Captured: "BL24I-JUNGFRAU-META:FD:NumCaptured",
    PV.Subfolder: "BL24I-JUNGFRAU-META:FD:Subfolder",
    PV.Ready: "BL24I-JUNGFRAU-META:FD:Ready",
}

CA_FLAGS = {PV.Path: ["-S"]}


def get(pv: PV | None = None):
    if not CAGET:
        sys.exit("Error: Could not find caget on PATH")
    pvs = [pv] if pv else list(PV)
    cmd = [CAGET, "-tS", *[PV_NAME[x] for x in pvs]]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for pv, ret in zip(pvs, out.stdout.splitlines()):
        print(f"{pv.title():10} {ret}")


def set(pv: PV, value: str):
    """Set a Morgul-Jungfrau parameter"""
    if not CAPUT:
        sys.exit("Error: Could not find caput on PATH")
    cmd = [CAPUT, *CA_FLAGS.get(pv, []), PV_NAME[pv], value]
    subprocess.run(cmd, check=True, capture_output=True)
    get(pv)
