"""
morgul_nxmx - Create still-shot nexus files with VDS pointers to data.

Very roughly created by copying rayonix2nxs.py:
    https://gist.github.com/ndevenish/3a5e577b7f7e6b654853fe5ff87bd6d6

and hacking to make relevant to the Jungfrau 1M commissioning
collections on I24 in June 2022. This may need lots extra work to make
general.
"""

import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, Type, TypeVar

import dateutil.tz
import dateutil.tz as tz
import h5py
import numpy as np
import pint
import typer
from pydantic import BaseModel, Field

T = TypeVar("T")

BOLD = "\033[1m"
ERR = "\033[1;31m"
NC = "\033[0m"


class AttrValue(BaseModel, Generic[T]):
    def __init__(self, value: T, **kwargs):
        super().__init__(value=value, **kwargs)

    value: T

    model_config = {
        "arbitrary_types_allowed": True,
    }


class AttrStringShortName(AttrValue[str]):
    short_name: str = None


class AttrTransformation(AttrValue[pint.Quantity]):
    transformation_type: Literal["translation", "rotation"] | None = None
    vector: tuple[float, float, float]
    offset: tuple[float, float, float] | None = None
    depends_on: str = "."


def _handle_grouped_default(
    group: str,
    kwargs: dict[str, Any],
    *,
    solo_name: str = None,
    rename: str | None = None,
) -> None:
    """For a class that has groups, allow a singular init property"""
    if (item := kwargs.pop(solo_name, None)) is not None:
        rename = rename or solo_name
        solo_name = solo_name or group
        if group in kwargs:
            raise ValueError(
                f"Cannot specify singular {solo_name} and {group} dictionary"
            )
        kwargs.setdefault(group, {})[rename] = item


def _read_all_nexus_attrs(
    name: str, target: Type[BaseModel] | BaseModel
) -> set[str] | dict[str, Any] | None:
    if not isinstance(target, type):
        target = type(target)

    attrs = None

    for parent in target.__mro__:
        if not hasattr(parent, "model_config") or name not in parent.model_config:
            continue
        if attrs is None:
            copied_attrs = copy.copy(parent.model_config[name])
            if not isinstance(copied_attrs, (set, dict)):
                copied_attrs = set(copied_attrs)
            attrs = copied_attrs
        else:
            attrs.update(parent.model_config[name])

    return attrs or None


def _external_link_via_vds(
    target_obj: h5py.Dataset, target_file_path: Path = None
) -> h5py.VirtualLayout:
    """
    Create an ExternalLink equivalent (1:1) but as a VDS layout.

    This is required to put attributes on ExternalLinks. Technique copied
    from Aaron's jf4m_geom2nexus.py.
    """
    layout = h5py.VirtualLayout(shape=target_obj.shape, dtype=target_obj.dtype)
    source = h5py.VirtualSource(
        str(target_file_path), target_obj.name, target_obj.shape
    )
    layout[:] = source
    return layout


def _apply_single_to_node(
    target: h5py.Group,
    name: str,
    value: Any,
    *,
    is_attr: bool = False,
    path: list[str] = [],
):
    if isinstance(value, datetime.datetime):
        timestamp = (
            value.replace(microsecond=0)
            .astimezone(dateutil.tz.UTC)
            .replace(tzinfo=None)
            .isoformat()
            + "Z"
        )
        if is_attr:
            target.attrs[name] = timestamp
        else:
            target[name] = timestamp
    elif isinstance(value, NXobject):
        node = target.create_group(name)
        value.apply_to_node(node, path=[*path, name])
    elif isinstance(value, h5py.VirtualLayout):
        target.create_virtual_dataset(name, value)
    elif isinstance(value, (h5py.Dataset, h5py.ExternalLink)):
        target[name] = value
    elif isinstance(value, (int, float)):
        target[name] = value
    elif isinstance(value, AttrValue):
        if is_attr:
            raise ValueError(
                f"Cannot have an AttrValue ({name}) classed as an attribute"
            )
        # target[name] = value.value
        _apply_single_to_node(target, name, value.value, path=path)
        attrs = value.dict()
        attrs.pop("value")
        for attrname, attrval in attrs.items():
            if attrval is None:
                continue
            if isinstance(attrval, tuple):
                attrval = np.array(attrval)
            target[name].attrs[attrname] = attrval
    elif isinstance(value, str):
        if is_attr:
            target.attrs[name] = value
        else:
            target[name] = value
    elif isinstance(value, pint.Quantity):
        _apply_single_to_node(target, name, value.magnitude, path=path)
        target[name].attrs["units"] = str(value.units)
    elif isinstance(value, np.generic):
        if is_attr:
            target.attrs[name] = value
        else:
            target[name] = value
    elif isinstance(value, tuple):
        nval = np.array(value)
        if is_attr:
            target.attrs[name] = nval
        else:
            target[name] = nval
    elif isinstance(value, np.ndarray):
        target[name] = value
    elif value is None:
        pass
    else:
        raise RuntimeError(
            f"Do not know how to handle field type: {type(value)} on path {'/'.join(path+[name])}"
        )


class NXobject(BaseModel):
    def apply_to_node(self, target: h5py.Group, path: list[str] | None = None):
        path = path or target.name.split("/")[1:]
        target.attrs["NX_class"] = type(self).__name__
        attrs = _read_all_nexus_attrs("nexus_attrs", self) or set()

        base_fields = set(self.__fields__.keys())
        # Grouped attrs can have any name but are are dicts for typing
        grouped_attrs = _read_all_nexus_attrs("nexus_groups", self) or set()
        assert isinstance(grouped_attrs, set)
        base_fields -= grouped_attrs
        # Make a map of all the grouped subobjects
        # TODO: Validate for name collisions
        subobjs = {k: v for d in grouped_attrs for k, v in getattr(self, d).items()}

        # Base fields
        for name in base_fields:
            _apply_single_to_node(
                target, name, getattr(self, name), is_attr=name in attrs, path=path
            )

        # Handle grouped attributes
        for k, v in subobjs.items():
            _apply_single_to_node(target, k, v, path=path)

    extra_fields: dict[str, Any] = {}

    model_config = {
        "nexus_groups": ["extra_fields"],
        "arbitrary_types_allowed": True,
    }


class NXsource(NXobject):
    name: AttrStringShortName | None = None
    type: str | None = None
    frequency: pint.Quantity | None = None


class NXbeam(NXobject):
    incident_wavelength: Any


class NXdetector_module(NXobject):
    data_origin: tuple[float, float] | tuple[float, float, float]
    data_size: tuple[float, float] | tuple[float, float, float]
    fast_pixel_direction: AttrTransformation
    slow_pixel_direction: AttrTransformation
    module_offset: AttrTransformation | None = None


class NXdetector(NXobject):
    description: str = None
    local_name: str = None
    depends_on: str = None
    sensor_material: str = None
    sensor_thickness: pint.Quantity | None = None
    bit_depth_readout: int = None
    beam_center_x: pint.Quantity | None = None
    beam_center_y: pint.Quantity | None = None
    x_pixel_size: pint.Quantity | None = None
    y_pixel_size: pint.Quantity | None = None
    distance: pint.Quantity | None = None
    saturation_value: float | None = None

    detector_module: dict[str, NXdetector_module] = {}

    type: str | None = None

    def __init__(self, **kwargs):
        _handle_grouped_default("detector_module", solo_name="module", kwargs=kwargs)
        super().__init__(**kwargs)

    model_config = {
        "nexus_groups": ["detector_module"],
    }


class NXinstrument(NXobject):
    def __init__(self, **kwargs):
        _handle_grouped_default("detectors", kwargs, solo_name="detector")
        super().__init__(**kwargs)

    name: AttrStringShortName | None = None
    beam: NXbeam | None = None

    detectors: dict[str, NXdetector] = {}

    model_config = {
        "nexus_groups": ["detectors"],
    }


class NXdata(NXobject):
    def __init__(self, **kwargs):
        _handle_grouped_default("datas", solo_name="data", kwargs=kwargs, rename="data")
        super().__init__(**kwargs)

    datas: dict[str, h5py.Dataset | h5py.ExternalLink] = {}

    model_config = {
        "nexus_groups": ["datas"],
    }


class NXtransformations(NXobject):
    axes: dict[str, AttrTransformation] = {}

    model_config = {
        "nexus_groups": ["axes"],
    }


class NXsample(NXobject):
    name: str | None
    depends_on: str | None = "."
    transformations: dict[str, NXtransformations] = {}

    model_config = {
        "nexus_groups": ["transformations"],
    }


class NXentry(NXobject):
    definition: Literal["NXmx"] = "NXmx"
    start_time: datetime.datetime | None
    end_time: datetime.datetime | None = None
    end_time_estimated: datetime.datetime | None
    data: NXdata | None = None
    sample: NXsample | None = None
    source: NXsource | None = None
    instrument: NXinstrument | None = None


class NXroot(NXobject):
    file_name: str | None
    file_time: datetime.datetime | None = Field(
        default_factory=datetime.datetime.utcnow
    )
    HDF5_Version: str = h5py.version.hdf5_version
    entry: NXentry | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_config = {
        "nexus_attrs": [
            "file_name",
            "file_time",
            "HDF5_Version",
        ],
    }


class JF1MD:
    def __init__(self, filenames: list[Path]):
        self.filenames = filenames
        self._handles = [h5py.File(x, "r") for x in filenames]

        self.modules = {}
        for h in self._handles:
            self.modules[(h["row"][()], h["column"][()])] = h

        assert len(self.modules) == 4, "Not enough modules"

    def __getitem__(self, path):
        # Make sure all files have the same value
        values = {x[path][()] for x in self._handles}
        assert len(values) == 1
        return values.pop()

    def get_all(self, path):
        return [x[path] for x in self._handles]

    def make_vfs(self, group: h5py.Group):
        frames = max(*[x.shape[0] for x in self.get_all("data")])

        MOD_FAST = 1028
        MOD_SLOW = 512
        GAP_SLOW = 36

        slow = (2 * MOD_SLOW) + GAP_SLOW
        fast = MOD_FAST

        layout = h5py.VirtualLayout(shape=(frames, slow, fast), dtype="i4")

        y_offsets = np.cumsum([0, 4, GAP_SLOW, 4])
        for row, offset in enumerate(y_offsets):
            source_filename = Path(self.modules[(3 - row, 0)].filename).resolve()
            source = h5py.VirtualSource(
                source_filename, "data", shape=(frames, 256, 1024)
            )
            y = row * 254 + offset
            for fast in range(4):
                x = fast * (254 + 4)
                real_x = 256 * fast
                layout[:, y : y + 254, x : x + 254] = source[
                    :, 1:255, real_x + 1 : real_x + 255
                ]

        group.create_virtual_dataset("data_000001", layout, fillvalue=0x8000)


def nxmx(
    input: Annotated[list[Path], typer.Argument()],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("output.h5"),
    energy: Annotated[
        float | None, typer.Option("-e", "--energy", help="Energy of the beam, in keV")
    ] = None,
    rotation_angle: Annotated[float | None, typer.Option("--rotation")] = None,
):
    """Create an NXmx Nexus file pointing to corrected Jungfrau data."""

    print(f"Reading {BOLD}{input}{NC}")

    ep_file = Path(input[0]).parent / "experiment_params.json"
    ep = None
    if ep_file.exists():
        ep = json.loads(ep_file.read_bytes())
        rotation_angle = ep["image_width_deg"]

    cp_file = Path(input[0]).parent / "collection_info.json"
    if cp_file.exists():
        cp = json.loads(cp_file.read_bytes())
        energy = 12.39841984055037 / cp["wavelength"]
    elif energy is None:
        sys.exit(
            f"{ERR}Error: No collection_info.json file alongside h5, so you must pass energy{NC}"
        )

    source = JF1MD(input)
    # if args.input.resolve().parent == args.output.resolve().parent:
    #     print("Output to same folder as input; doing relative links")
    #     source.filename = Path(args.input.name)

    # NXmx requires an estimated end time
    start_time = datetime.datetime.fromtimestamp(source["timestamp"][()]).replace(
        tzinfo=tz.UTC
    )

    num_images = max(*[x.shape[0] for x in source.get_all("data")])
    end_time_estimated = start_time + datetime.timedelta(
        seconds=num_images * source["exptime"]
    )
    # pal_frequency = pint.Quantity(rate, "Hz")

    # Generate the tz_offset NXmx wants. The spec asks for: "ISO 8601
    # time_zone offset from UTC". Interpreting as e.g. +08:00.
    tz_offset = start_time.tzinfo.utcoffset(start_time).total_seconds()
    tz_offset_str = f"{tz_offset // 3600:+03.0f}:{tz_offset % 3600:02.0f}"

    # 1066x1030
    # Work out the dimension values
    # detector_size = pint.Quantity(0.07995, "m")
    # max_pixels = 5760
    # size_s, size_f = source.run["header/detector_0_number_of_pixel"]
    size_s, size_f = 1060, 1028
    detector_distance = pint.Quantity(66, "mm")
    pixel_size = pint.Quantity(75, "microns")

    beam_center_sf_px = (552, 520)
    beam_center_sf_mm = tuple((x * pixel_size).to("m") for x in beam_center_sf_px)

    detector = NXdetector(
        description="Jungfrau 1M ",
        local_name="JF1MD",
        depends_on="/entry/instrument/transformations/det_z",
        type="CCD",
        x_pixel_size=pixel_size,
        y_pixel_size=pixel_size,
        sensor_material="Si",
        sensor_thickness=pint.Quantity(320, "microns"),
        distance=detector_distance,
        saturation_value=12000,
        detector_module={
            "module": NXdetector_module(
                data_origin=(0, 0),
                data_size=(size_s, size_f),
                fast_pixel_direction=AttrTransformation(
                    pixel_size,
                    transformation_type="translation",
                    offset=(0, 0, 0),
                    vector=(-1, 0, 0),
                    depends_on="/entry/instrument/detector/module/module_offset",
                ),
                slow_pixel_direction=AttrTransformation(
                    pixel_size,
                    transformation_type="translation",
                    offset=(0, 0, 0),
                    vector=(0, -1, 0),
                    depends_on="/entry/instrument/detector/module/module_offset",
                ),
                module_offset=AttrTransformation(
                    pint.Quantity(0, "m"),
                    transformation_type="translation",
                    offset=(
                        beam_center_sf_mm[1].magnitude,
                        beam_center_sf_mm[0].magnitude,
                        0,
                    ),
                    vector=(1, 0, 0),
                    depends_on="/entry/instrument/transformations/det_z",
                ),
            )
        },
    )

    sample = NXsample(
        name="sample",
        depends_on="/entry/sample/transformations/omega",
        transformations={
            "transformations": NXtransformations(
                axes={
                    "omega": AttrTransformation(
                        pint.Quantity(
                            np.cumsum(np.ones((num_images,)) * rotation_angle),
                            "deg",
                        ),
                        transformation_type="rotation",
                        vector=(0, -1, 0),
                        depends_on=".",
                    )
                }
            )
        },
    )
    root = NXroot(
        file_name=output.name,
        entry=NXentry(
            start_time=start_time,
            end_time_estimated=end_time_estimated,
            source=NXsource(
                name=AttrStringShortName("DLS", short_name="DLS"),
                type="Synchrotron",
            ),
            data=NXdata(),  # data=h5py.ExternalLink(source.filename, source.run.data_path)),
            sample=sample,
            instrument=NXinstrument(
                name=AttrStringShortName("I24", short_name="I24"),
                detector=detector,
                extra_fields={
                    "time_zone": tz_offset_str,
                    "transformations": NXtransformations(
                        axes={
                            "det_z": AttrTransformation(
                                detector_distance,
                                transformation_type="translation",
                                vector=(0, 0, 1),
                            )
                        }
                    ),
                },
            ),
        ),
    )
    ureg = pint.UnitRegistry()
    wavelength = (
        (ureg.speed_of_light * ureg.planck_constant) / ureg.Quantity(energy, "keV")
    ).to("angstrom")
    root.entry.instrument.beam = NXbeam(incident_wavelength=wavelength.to("angstrom"))

    print(f"Writing to {BOLD}{output}{NC}")
    nxs = h5py.File(output, "w")
    root.apply_to_node(nxs)
    # data = nxs["entry"].create_group("data")
    # data.attrs["NX_class"] = "NXData"
    source.make_vfs(nxs["entry"]["data"])
    # h5py.VirtualLayout()

    # TODO:
    # Verify detector sensor material and thickness
    # Beam incident_polarization_stokes


if __name__ == "__main__":
    typer.run(nxmx)
