import datetime
import logging
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import IO, Annotated, Any

import dateutil.tz as tz
import h5py
import typer
from watchdir import Watcher

from morgul.util import NC, B, G, R

logger = logging.getLogger(__name__)

# Only consider files matching
# FILTER_REGEX = r".+\.h5"
FILTER_REGEX = r".+\d\d\d\d\d.+\/.+\.h5|.+2024-09-\d\d.+\/.+\.h5|.+\d\d09\d\d\/.+\.h5|.+ssx/.+/.+\.h5"

# A cache file so that we can quickly re-present results
CACHE_FILE = ".watcher_history"
# Time to sleep between scans
SLEEP_TIME = 5


class Settings:
    root_path: Path

    @classmethod
    @lru_cache
    def get(cls) -> "Settings":
        return cls()


def read_h5_info(filename: Path) -> dict[str, Any] | None:
    with h5py.File(filename, "r") as f:
        if "timestamp" not in f:
            logger.info(f"Skipping file {f} as no timestamp")
            return None
        timestamp = datetime.datetime.fromtimestamp(f["timestamp"][()]).replace(
            tzinfo=tz.UTC
        )
        exptime = f["exptime"][()]

        # # Assume timestamp is unique per collection
        # if timestamp in entries:
        #     if filename.name > entries[timestamp]["path"].name:
        #         continue

        # if root:
        #     filename = filename.relative_to(root)
        # else:
        #     root = Path(*filename.parts[:6])
        #     filename = filename.relative_to(root)

        return {
            "timestamp": timestamp,
            "filename": filename,
            "exptime": exptime,
            "gainmode": f["gainmode"][()].decode(),
            "nimage": f"{f['data'].shape[0]:5}",
            "bad": False,
        }


class EmitHandler:
    _entries: dict[Path, dict[str, Any]]
    _last_folder: Path | None = None
    _output_stream: IO
    _longest_path: int

    def __init__(self, output_stream: IO = sys.stdout, *, fzf: bool = False):
        self._entries = {}
        self._entry_order: list[Path] = []
        self._fzf = fzf
        self._output_stream = output_stream
        self._longest_path = 0
        # self._lines_to_update = set()

    def set_output_stream(self, stream: IO) -> None:
        self._output_stream = stream

    def print(self, *args, sep: str = " ", end: str = "\n") -> None:
        self._output_stream.write(sep.join(str(x) for x in args) + end)

    def _generate_entry_line(self, entry: dict[str, Any]) -> str:
        MAX_WIDTH, _ = os.get_terminal_size()

        # pre-filename=26
        # gain_mode=13+ =14
        # ms=4+ = 5
        # n_image=6+ =7
        #    post = 26
        max_path_length = MAX_WIDTH - 26 - 26 - 2
        path_length = min(self._longest_path, 87, max_path_length)
        # Let's not be crazy
        if path_length < 20:
            path_length = 20

        filename = entry["filename"].resolve()
        # Handle unicode prettiness
        prefix = "┃"
        # Handle dividers if our new entry is in a different folder
        if filename.parent != self._last_folder:
            if self._last_folder is not None:
                # Tie it off
                self.print(
                    ("- " if self._fzf else "  ")
                    + "┗"
                    + "━" * 21
                    + "┷"
                    + "━" * (path_length + 27)
                    + "┛"
                )
            prefix = "┏"
            self._last_folder = filename.parent

        dec = [" " if not self._fzf else str(entry["filename"].resolve()), prefix]
        # if filename in self._entry_order:

        # Work out how to show the filename with maximum length
        filename_trunc = str(filename.relative_to(Settings.get().root_path))
        try:
            filename_trunc = (
                filename_trunc[: filename_trunc.rfind("/") + 1]
                + " "
                + filename_trunc[filename_trunc.rfind("/") + 1 :]
            )
        except ValueError:
            pass
        filename_trunc = filename_trunc.ljust(path_length)
        if len(filename_trunc) > path_length:
            # Try subtracting from the name, not the folder name
            # root_path = entry["filename"].relative_to(Settings.get().root_path).parent()
            if len(filename_trunc) - len(filename.name) + 3 > path_length:
                # Too long, even when truncated. Just cut down the whole hting
                filename_trunc = filename_trunc[: path_length - 3] + "..."
            else:
                shorten_by = len(filename_trunc) - path_length + 3
                new_filename = " ..." + filename.name[shorten_by:]
                logger.info(
                    f"Truncating '{filename_trunc}' to shorten by {shorten_by=}, {path_length=}, {len(filename_trunc)=} into {new_filename=} ({len(new_filename)=}), {MAX_WIDTH=}"
                )

                filename_trunc = str(
                    filename.parent.relative_to(Settings.get().root_path) / new_filename
                ).ljust(path_length)

        # Manage the colour
        if entry["bad"]:
            # We want to only show partial information
            return (
                " ".join(
                    [
                        *dec,
                        R
                        + entry["timestamp"].astimezone().strftime("%Y-%m-%d %H:%M:%S")
                        + NC,
                        "│",
                        R + filename_trunc,
                        # entry["reason"] + NC,
                    ]
                )
                + NC
            )

        return " ".join(
            str(x)
            for x in [
                *dec,
                entry["timestamp"].astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "│",
                filename_trunc,
                entry["gainmode"].ljust(13),
                f"{entry['exptime']*1000:4g}ms",
                str(entry["nimage"]).ljust(6),
            ]
        )

    def emit_new_entries(self, entries: list[dict[str, Any]]) -> None:
        # Check to see if we need to extend the longest path for these
        self._longest_path = max(
            self._longest_path,
            *[
                len(str(x["filename"].relative_to(Settings.get().root_path)))
                for x in entries
            ],
        )

        for entry in sorted(entries, key=lambda x: x["timestamp"]):
            # For now, just emit the lines
            self.print(self._generate_entry_line(entry))

    def reemit(self) -> None:
        """Print everything again"""
        raise NotImplementedError()


def watch(
    verbose: Annotated[
        int,
        typer.Option(
            "-v", help="Verbose logging. Pass twice to log to stdout.", count=True
        ),
    ] = 0,
    logfile: Annotated[Path, typer.Option(help="Output log file")] = Path(
        "watching.log"
    ),
    plain: Annotated[bool, typer.Option(help="Plain (no fzf) output")] = False,
    root_path: Annotated[
        Path,
        typer.Argument(
            metavar="PATH",
            help="Path to watch. Defaults to environment VISIT_DATA_ROOT",
        ),
    ] = None,
    use_fzf: Annotated[bool, typer.Option("--fzf")] = False,
    wait: Annotated[
        bool,
        typer.Option("--wait", help="If the root path does not exist, wait for it to."),
    ] = False,
):
    """Watch a data folder for new files appearing."""

    # This command wants to control logging completely but the root
    # morgul object sets up logging itself, so we want to undo what it
    # did.
    for loghandler in logging.root.handlers[:]:
        logging.root.removeHandler(loghandler)

    # Set up the logging. Nothing to stdout unless we asked for verbose.
    logger_dest = {"filename": logfile} if verbose < 2 else {}
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        **logger_dest,  # type: ignore
    )
    logging.getLogger("morgul.watcher.watcher").setLevel(logging.INFO)

    if root_path is None:
        if "VISIT_DATA_ROOT" in os.environ:
            root_path = Path(os.environ["VISIT_DATA_ROOT"])
        else:
            print(
                "Error: Have not specified root_path and VISIT_DATA_ROOT not set. Please do one of these."
            )
            raise typer.Abort()

    # Update the singleton root object
    Settings.get().root_path = root_path

    if not root_path.is_dir():
        if not wait:
            print(f"{R}Error: Root path {root_path} does not exist{NC}")
            raise typer.Abort()
        else:
            print(f"Root path {B}{root_path}{NC} does not exist, waiting,")
            while not root_path.exists():
                time.sleep(5)

    logger.debug(f"Starting watch on {root_path}")
    watcher = Watcher(root_path)

    # Keep track of files we couldn't open yet, with a timestamp so we don't get stuck
    unscanned_files: set[Path] = set()

    first_scan = True
    if not use_fzf:
        print("Doing initial scan, this could take a minute...")
        start_time = time.monotonic()

    if use_fzf:
        fzf = subprocess.Popen(
            [
                "fzf",
                "--multi",
                "--tac",
                # "--ansi",
                "--bind",
                "tab:toggle",
                "--with-nth=2..",
                "--phony",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        handler = EmitHandler(fzf.stdin, fzf=True)
    else:
        handler = EmitHandler(sys.stdout)

    re_filter = re.compile(FILTER_REGEX)
    while True:
        new_files, dropped_paths = watcher.scan()

        if first_scan:
            print(f"    Found {G}{len(new_files)}{NC} files, reading...")

        # Drop any unscanned files in any dropped paths
        for filename in list(unscanned_files):
            if any(x in filename.parents for x in dropped_paths):
                logger.debug(
                    f"Dropping unscanned file {filename} because path is dropped"
                )
                unscanned_files.remove(filename)

        # Store a list of processed - we may want to group/reorder once we have metadata
        processed = []
        # For each new file, open it and get some details
        for filename in sorted(new_files + list(unscanned_files)):
            # Filter the filenames
            if "merged" in str(filename) or "corrected" in str(filename):
                continue

            if not re_filter.match(str(filename)):
                logger.debug(f"Ignoring file {filename} as does not match filter")
                continue
            try:
                data = read_h5_info(filename)
                if not data:
                    logger.info(f"File {filename} is not a data file")
                    continue
                processed.append(data)
                if filename in unscanned_files:
                    unscanned_files.remove(filename)

            except (IOError, KeyError):
                unscanned_files.add(filename)
                # processed.append(
                #     {
                #         "filename": filename,
                #         "bad": True,
                #         # Some fake "In the future" timestamp
                #         "timestamp": datetime.datetime.fromtimestamp(
                #             filename.stat().st_mtime
                #         ).astimezone(),
                #         "reason": str(e),
                #     }
                # )
                continue

        if first_scan:
            first_scan = False
            print(f"    kept {G}{len(processed)}{NC} files.")
            print(f"    done in {G}{1000*(time.monotonic()-start_time):g}{NC} ms")
            if not new_files:
                print("No files found on first scan.")
            print()

        if processed:
            handler.emit_new_entries(processed)

        if use_fzf:
            try:
                fzf.wait(timeout=SLEEP_TIME)
                fzf_output = fzf.stdout.read()
                logger.info("Output: " + fzf_output)
                return
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(SLEEP_TIME)

    # If fzf, then we might have gotten something to use


if __name__ == "__main__":
    typer.run(watch)
