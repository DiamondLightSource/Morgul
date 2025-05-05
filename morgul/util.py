from __future__ import annotations

import os
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import dateutil.tz as tz

BOLD = "\033[1m"
R = "\033[31m"
G = "\033[32m"
B = "\033[34m"
NC = "\033[0m"
Y = "\033[33m"
GRAY = "\033[37m"


def elapsed_time_string(start_time: float) -> str:
    elapsed_time = time.monotonic() - start_time
    if elapsed_time > 60:
        return f"{G}{elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s{NC}"
    else:
        return f"{G}{elapsed_time:.1f}s{NC}"


def strip_escapes(input: str) -> str:
    return re.sub("\033" + r"\[[\d;]+m", "", input)


@lru_cache
def read_calibration_file(
    filter: Literal["PEDESTAL"] | Literal["MASK"],
) -> dict[tuple[datetime, float], Path]:
    """Read the calibration log (if it exists) and get all entries"""
    if "JUNGFRAU_CALIBRATION_LOG" not in os.environ:
        raise RuntimeError(
            "Could not find calibration log; please set JUNGFRAU_CALIBRATION_LOG"
        )

    entries: dict[tuple[datetime, float], Path] = {}
    for line in Path(os.environ["JUNGFRAU_CALIBRATION_LOG"]).read_text().splitlines():
        if not line.startswith(filter):
            continue
        _, ts, exposure, filename = line.split()
        timestamp = datetime.fromisoformat(ts)
        entries[timestamp, float(exposure)] = Path(filename)
    return entries


def _convert_ts_to_utc_datetime(ts: float | datetime) -> datetime:
    # Make sure we have a UTC timestamp
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=tz.UTC)
        else:
            return ts
    else:
        return datetime.fromtimestamp(ts).replace(tzinfo=tz.UTC)


def _find_entry(
    kind: Literal["PEDESTAL"] | Literal["MASK"],
    ts: float | datetime,
    exposure: float | None = None,
    *,
    within_minutes: int | None = None,
) -> Path:
    timestamp = _convert_ts_to_utc_datetime(ts)
    lookup = read_calibration_file(kind)
    candidates = [
        (t, e)
        for t, e in sorted(
            lookup, key=lambda x: abs((x[0] - timestamp).total_seconds())
        )
    ]
    # from pprint import pprint

    # pprint(candidates)
    if not candidates:
        raise RuntimeError(f"Could not find {kind.title()} entry")

    if within_minutes is not None:
        candidates = [
            (t, e)
            for t, e in candidates
            if abs((t - timestamp).total_seconds()) < within_minutes * 60
        ]
        if not candidates:
            raise RuntimeError(
                f"Could not find {kind.title()} entry taken within {within_minutes} minutes"
            )

    if exposure is not None:
        candidates = [(t, e) for t, e in candidates if abs(e - exposure) < 1e-9]
        if not candidates:
            raise RuntimeError(
                f"Could not find {kind.title()} entry taken with exposure {exposure}"
            )

    # Return the closest candidate in time
    return lookup[candidates[0]]


def find_mask(
    timestamp_utc: float | datetime,
    exposure_time: float | None,
    *,
    within_minutes: int | None = None,
) -> Path | None:
    return _find_entry("MASK", timestamp_utc, None, within_minutes=within_minutes)


def find_pedestal(
    timestamp_utc: float | datetime,
    exposure_time: float | None,
    *,
    within_minutes: int | None = None,
) -> Path:
    return _find_entry(
        "PEDESTAL", timestamp_utc, exposure_time, within_minutes=within_minutes
    )
