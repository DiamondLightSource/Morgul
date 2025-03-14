#!/usr/bin/env python3
"""
Merge a set of /dev/shm pedestal 1024x256 uint16 dumps of the form:
   /dev/shm/half-module-ped{gain}-{index}
"""

from pathlib import Path

import h5py
import numpy as np

f = h5py.File("pedestal.h5", "w")

string_dt = h5py.string_dtype()
module_mode = f.create_dataset("module_mode", (), dtype=string_dt, data="half")
exptime = f.create_dataset("exptime", (), dtype=np.double, data=0.001)
exptime.attrs["units"] = "s"

# Load each module
for hmi in range(36):
    group = f.create_group(f"hmi_{hmi:02d}")
    for gain in range(3):
        raw_data = Path(f"/dev/shm/half-module-ped{gain}-{hmi:02d}.dat").read_bytes()
        data = (
            np.frombuffer(raw_data, dtype=np.uint16)
            .reshape((256, 1024))
            .astype(np.float32)
        )
        group[f"pedestal_{gain}"] = data

f.close()
