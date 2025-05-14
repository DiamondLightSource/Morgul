import contextlib
import enum
import logging
import operator
from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import Annotated, TypeAlias

import h5py
import napari
import typer

from . import config
from .util import NC, B, G

logger = logging.getLogger(__name__)


class FileKind(enum.Enum):
    # UNKNOWN = enum.auto()
    GAIN_MAP = enum.auto()
    MASK = enum.auto()
    PEDESTAL = enum.auto()
    RAW = enum.auto()
    CORRECTED = enum.auto()
    NXMX = enum.auto()


ViewCallable: TypeAlias = Callable[[dict[Path, h5py.Group]], None]
view_functions: dict[FileKind, ViewCallable] = {}


def viewer(kind: FileKind) -> Callable[[ViewCallable], ViewCallable]:
    def _wrapped(view_func: ViewCallable) -> ViewCallable:
        if kind in view_functions:
            raise ValueError(f"Viewer for {kind} is already registered")
        view_functions[kind] = view_func
        return view_func

    return _wrapped


def determine_kinds(root: h5py.Group) -> set[FileKind]:
    """Given an HSD5 data group, work out what kind of data it contains"""
    detector = config.get_detector()
    modules = config.get_known_modules_for_detector(detector)
    kinds = set()
    # Work out what sort of file we have
    module_subgroups = [
        x for x in modules if x in root and isinstance(root[x], h5py.Group)
    ]
    for module in module_subgroups:
        if "g0" in root[module] or "g1" in root[module] or "g2" in root[module]:
            kinds.add(FileKind.GAIN_MAP)
        if "mask" in root[module]:
            kinds.add(FileKind.MASK)
        if any(
            x.startswith("pedestal_")
            for x in root[module]
            if isinstance(root[module][x], h5py.Dataset)
        ):
            kinds.add(FileKind.PEDESTAL)
    if "data" in root and isinstance(root["data"], h5py.Dataset):
        if root["data"].attrs.get("corrected", False):
            kinds.add(FileKind.CORRECTED)
        else:
            kinds.add(FileKind.RAW)

    return kinds


def determine_kind(root: h5py.Group) -> FileKind | None:
    """Return a single file kind"""
    kinds = determine_kinds(root)
    if kinds:
        return sorted(kinds, key=lambda x: x.value)[-1]
    return None


def _module_transforms(
    module: str,
    shape: tuple[int, int],
    offset: tuple[float, float] = (0, 0),
    corrected: bool = False,
) -> dict[str, tuple[float, float]]:
    module_info = config.get_module_from_id(module)

    translate = offset
    scale = (1, 1) if corrected else (-1, 1)

    h, _ = shape

    if module_info["position"] == "bottom":
        translate = (translate[0] + h + 36, translate[1])

    return {"scale": scale, "translate": translate}


def _label_for_module(
    module: str,
    shape: tuple[int, int],
    offset: tuple[float, float] = (0, 0),
    *,
    corrected: bool = False,
) -> tuple[float, float]:
    module_info = config.get_module_from_id(module)
    h, w = shape

    point_vertical = -h - 20
    if module_info["position"] == "bottom":
        point_vertical = h + 36 + 20

    if corrected:
        point_vertical += h
    return (offset[0] + point_vertical, offset[1] + (w / 2))


@viewer(FileKind.PEDESTAL)
def view_pedestal(files: dict[Path, h5py.Group]) -> None:
    assert len(files) == 1, "Cannot view multiple pedestal files at once"
    filename, root = next(iter(files.items()))

    viewer = napari.Viewer()
    detector = config.get_detector()
    modules = config.get_known_modules_for_detector(detector)

    points: dict[str, tuple[float, float]] = {}
    # point_texts = []
    for module in modules:
        for mode in 0, 1, 2:
            name = f"pedestal_{mode}"
            if name in root[module]:
                h, w = root[module][name].shape
                # Offset this module so we show all gain modes
                x_offset = mode * (w + 20)
                transform = _module_transforms(
                    module, root[module][name].shape, (0, x_offset)
                )

                viewer.add_image(
                    root[module][name][()],
                    name=f"{module}/{mode}",
                    **transform,
                )
                points[f"{module}/{mode}"] = _label_for_module(
                    module, (h, w), (0, mode * (w + 20))
                )

    pt_text, pt_data = zip(*points.items())
    viewer.add_points(pt_data, text=pt_text, size=0)

    viewer.reset_view()


def view_image(files: dict[Path, h5py.Group], corrected: bool):
    # assert len(files) == 1
    # filename, root = next(iter(files.items()))
    # assert len(files) <= 2
    detector = config.get_detector()

    viewer = napari.Viewer()

    points: dict[str, tuple[float, float]] = {}

    for h5 in files.values():
        # Get the module for this file
        h, w = h5["data"].shape[1:]
        module = config.get_module_info(detector, h5["column"][()], h5["row"][()])[
            "module"
        ]
        # Work out the transform
        transform = _module_transforms(module, (h, w), corrected=corrected)
        limits = (-1, 10) if corrected else None
        viewer.add_image(
            h5["data"],
            name=module,
            **transform,
            contrast_limits=limits,
            gamma=0.8,
        )

        points[f"{module}"] = _label_for_module(module, (h, w), corrected=corrected)

        pt_text, pt_data = zip(*points.items())

    viewer.add_points(pt_data, text=pt_text, size=0)

    viewer.reset_view()


@viewer(FileKind.RAW)
def view_raw(files: dict[Path, h5py.Group]):
    view_image(files, corrected=False)


@viewer(FileKind.CORRECTED)
def view_corrected(files: dict[Path, h5py.Group]):
    view_image(files, corrected=True)


@viewer(FileKind.MASK)
def view_mask(files: dict[Path, h5py.Group]):
    assert len(files) == 1, "Cannot view multiple mask files at once"
    filename, root = next(iter(files.items()))

    viewer = napari.Viewer()
    detector = config.get_detector()
    modules = config.get_known_modules_for_detector(detector)

    points: dict[str, tuple[float, float]] = {}
    # point_texts = []
    for module in modules:
        if "mask" in root[module]:
            h, w = root[module]["mask"].shape
            # Offset this module so we show all gain modes
            transform = _module_transforms(module, root[module]["mask"].shape)

            viewer.add_image(
                root[module]["mask"][()],
                name=f"{module}",
                **transform,
            )
            points[f"{module}"] = _label_for_module(module, (h, w))

    pt_text, pt_data = zip(*points.items())
    viewer.add_points(pt_data, text=pt_text, size=0)

    viewer.reset_view()


def view(filenames: Annotated[list[Path], typer.Argument(help="Data files to view")]):
    """Launch a napari-based viewer"""

    with contextlib.ExitStack() as stack:
        open_files = {
            path: stack.enter_context(h5py.File(path, "r")) for path in filenames
        }

        # Determine a common kind for all these files
        common_kind = reduce(
            operator.and_, [determine_kinds(x) for x in open_files.values()]
        )
        if not common_kind and len(filenames) == 1 and filenames[0].suffix == ".nxs":
            pass
        elif not common_kind:
            logger.error("Error: Could not determine common filekind for input files.")
            raise typer.Abort()
        kind = sorted(common_kind, key=lambda x: x.value)[-1]

        list_of_files = "\n".join("  - " + str(x) for x in filenames)
        if kind is None:
            logger.error(
                f"Error: Could not determine common file kind for\n{list_of_files}"
            )
            raise typer.Abort()

        logger.info(f"Opening:\n{B}{list_of_files}\n{NC}as {G}{kind.name.title()}{NC}")

        if kind in view_functions:
            view_functions[kind](open_files)
            napari.run()
        else:
            logger.error(f"Error: File kind {kind.name} is not currently supported")
            raise typer.Abort()
