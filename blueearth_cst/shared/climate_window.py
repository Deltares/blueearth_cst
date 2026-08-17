"""What a climate source ACTUALLY delivers against what the config requested.

``shared.historical_window`` is ONE window for every source. That is right for
the project's own ``shared.clim_historical`` — the forcing the model runs on —
but wrong as a hard demand on the extra ``workflows.analyze_climate.
candidate_sources``, which exist to be COMPARED: CHIRPS begins in 1981, E-OBS
covers Europe only, and a locally staged subset of any of them can start
wherever the staging did. A window one candidate cannot satisfy is not a
misconfigured project; it is the ordinary case of two datasets with different
records.

So the requested window is a CEILING, not a demand. Each source is extracted
over the widest span it actually holds inside that window, and the narrowing is
reported rather than raised (owner ruling 2026-08-16).

Two facts this module keeps apart, because they fail for different reasons:

* **Narrowed** — the delivered span is shorter than the requested one. Always
  reported; never an error.
* **Below the floor** — the delivered span is shorter than
  ``MIN_HISTORICAL_YEARS``. An error for the source that FEEDS THE PIPELINE
  (weathergenr's wavelet decomposition needs that many annual observations),
  reported-only for a source that merely appears in a comparison figure. The
  caller decides which it is via ``enforce_min_years``; nothing here infers it.

Why a module of its own rather than a helper in ``snake_utils``: this needs
pandas, and ``snake_utils`` is imported at Snakefile PARSE time and is
deliberately stdlib+yaml. The floor itself still comes from there, so there is
one definition of it.
"""

# NO `from __future__ import annotations`: this module is imported by
# `script:` modules, whose Snakemake preamble displaces the first statement.
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from blueearth_cst.shared.snake_utils import (
    MIN_HISTORICAL_YEARS,
    log_row,
    meets_min_historical_years,
)

#: How far a source may fall short of the requested window before the report
#: calls it NARROWED rather than merely late.
#:
#: A daily product whose first file starts three weeks into January is not a
#: coverage problem, and flagging it would train the reader to ignore the line
#: that matters. The tolerance belongs to the WORDING only: the effective window
#: is always the delivered one, and the ``MIN_HISTORICAL_YEARS`` floor never
#: gets a tolerance — a floor with a tolerance is not a floor.
NARROWING_TOLERANCE = pd.Timedelta(days=31)


@dataclass(frozen=True)
class WindowCoverage:
    """What one source delivered inside one requested window."""

    source: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    effective_start: pd.Timestamp
    effective_end: pd.Timestamp

    @property
    def years(self) -> float:
        """Delivered span in years, for reporting only.

        ``days / 365.25`` is fine HERE and wrong for the floor: this number is
        printed, never compared. ``meets_floor`` does calendar arithmetic.
        """
        return (self.effective_end - self.effective_start).days / 365.25

    @property
    def is_narrowed(self) -> bool:
        """Did the source fall short of the request by more than the tolerance?"""
        return (
            self.effective_start > self.requested_start + NARROWING_TOLERANCE
            or self.effective_end < self.requested_end - NARROWING_TOLERANCE
        )

    @property
    def meets_floor(self) -> bool:
        """Does the DELIVERED span reach ``MIN_HISTORICAL_YEARS``?"""
        return meets_min_historical_years(self.effective_start, self.effective_end)

    def describe(self) -> str:
        """The one line every caller logs: requested, delivered, and how long.

        When nothing was narrowed the two spans are the SAME dates, and stating
        both spent 40 characters saying it twice -- enough to wrap the row. The
        requested span is then dropped and only the delivered one is named,
        which is the span every later stage actually uses. A narrowed record
        still prints both, because there the difference is the whole point.
        """
        delivered = f"{self.effective_start.date()}..{self.effective_end.date()}"
        requested = f"{self.requested_start.date()}..{self.requested_end.date()}"
        if delivered == requested:
            return f"{self.source}: {delivered} (~{self.years:.1f} years)"
        return (
            f"{self.source}: requested {requested}, "
            f"delivered {delivered} (~{self.years:.1f} years)"
        )


def time_axis_bounds(ds) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """``(first, last)`` timestamp of a dataset's time axis, or ``None``.

    ``None`` for anything this cannot honestly read — no ``time`` coordinate, an
    empty axis, a non-datetime calendar. A caller that cannot introspect the
    axis must skip its checks rather than guess, which is the stance the
    extraction took before this module existed.
    """
    try:
        values = ds.time.values
        start = pd.Timestamp(pd.to_datetime(values.min()))
        end = pd.Timestamp(pd.to_datetime(values.max()))
    except (AttributeError, KeyError, ValueError, TypeError):
        return None
    if pd.isna(start) or pd.isna(end):
        return None
    return start, end


def intersect_bounds(first, second):
    """The overlap of two ``(start, end)`` pairs, or ``None`` when they miss.

    Used where a store is assembled from TWO sources — the chirps branch takes
    precipitation from CHIRPS and everything else from era5 — so the store's
    window is what BOTH cover, never what the wider one does.
    """
    if first is None or second is None:
        return None
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    if start > end:
        return None
    return start, end


def resolve_coverage(
    bounds, starttime, endtime, source: str
) -> Optional[WindowCoverage]:
    """Pair delivered ``bounds`` with the requested window, or ``None``.

    ``None`` when either side is unreadable — same skip-rather-than-guess stance
    as :func:`time_axis_bounds`.
    """
    if bounds is None:
        return None
    try:
        requested_start = pd.Timestamp(pd.to_datetime(starttime))
        requested_end = pd.Timestamp(pd.to_datetime(endtime))
    except (ValueError, TypeError):
        return None
    return WindowCoverage(
        source=source,
        requested_start=requested_start,
        requested_end=requested_end,
        effective_start=bounds[0],
        effective_end=bounds[1],
    )


def floor_shortfall_message(coverage: WindowCoverage, *, where: str) -> str:
    """The below-the-floor text, shared by the raising and reporting paths.

    One string for both so the enforced and the relaxed extraction cannot
    describe the same record differently; only the closing ``where`` clause
    changes, because only the consequence does.
    """
    return (
        f"Extracted {coverage.source} record covers "
        f"{coverage.effective_start.date()}..{coverage.effective_end.date()} "
        f"(~{coverage.years:.1f} years) for the requested "
        f"{coverage.requested_start.date()}..{coverage.requested_end.date()}, "
        f"below the {MIN_HISTORICAL_YEARS}-year minimum this toolbox requires "
        f"(weathergenr's wavelet decomposition needs at least "
        f"{MIN_HISTORICAL_YEARS} annual observations). {where}"
    )


def report_coverage(
    coverage: Optional[WindowCoverage],
    *,
    enforce_min_years: bool,
    where: str,
    module: str = "extract",
) -> Optional[WindowCoverage]:
    """Log what a source delivered; raise only when the floor is BEING enforced.

    Always emits the ``describe()`` line, narrowed or not — "this source covers
    exactly what you asked for" is worth reading in a comparison run, and a line
    that appears only on trouble cannot be checked for absence.

    ``where`` closes the message: for an enforcing caller it says what to do
    about it, for a reporting one it says what the consequence is. Both are
    caller knowledge — this module does not know which config key named the
    source.
    """
    if coverage is None:
        return None
    log_row(coverage.describe(), module=module)
    if coverage.is_narrowed:
        log_row(
            f"{coverage.source}: the staged source does not cover the full "
            f"shared.historical_window; extracted the widest range it holds "
            f"inside it",
            module=module,
            level="WARNING",
        )
    if not coverage.meets_floor:
        message = floor_shortfall_message(coverage, where=where)
        if enforce_min_years:
            raise ValueError(message)
        log_row(message, module=module, level="WARNING")
    return coverage


# ---------------------------------------------------------------------------
# The CONSUMER side: what a store on disk holds
# ---------------------------------------------------------------------------
#
# Relaxing the floor for wf0's candidates opens exactly one hole, and it is the
# reuse the design deliberately built: candidate stores land in the SAME
# ``data/climate/historical/<source>_<window>/`` family WF1 and WF3 read from, so
# a candidate that wins the comparison costs no re-extraction.
#
# Switching ``shared.clim_historical`` onto such a candidate changes the params
# WF1/WF3 declare for the store rule (the relaxed store carries
# ``enforce_min_years``, theirs does not), and params is a Snakemake rerun
# trigger — so the ordinary case re-extracts under the floor and fails in the
# producer. That trigger reads ``.snakemake/`` metadata under the WORKING
# DIRECTORY, though, and AGENTS.md documents at length that one ``project_dir``
# driven from two checkouts gets two independent stores of it. A checkout with no
# record of the wf0 run decides by mtime alone, finds the store present and
# newer than the catalog, and never re-extracts.
#
# So the consumers check the store they actually read. Cheap — one time axis off
# an already-declared input — and it fails where the store is named.


def store_time_bounds(climate_nc):
    """``(first, last)`` timestamp of a store on disk, or ``None``.

    Opens read-only and closes immediately: the caller wants the axis, not the
    data, and a lingering handle on a store several rules share is its own
    hazard (``HDF5_USE_FILE_LOCKING=FALSE`` in the pixi env makes a concurrent
    read abort below Python with no traceback).
    """
    import xarray as xr

    with xr.open_dataset(climate_nc) as ds:
        return time_axis_bounds(ds)


def require_min_years(bounds, source: str, climate_nc, *, where: str) -> None:
    """Raise unless a store on disk reaches ``MIN_HISTORICAL_YEARS``.

    Unreachable for a store the CALLING workflow extracted — the producer
    applies the same arithmetic to the same bounds and would have failed first.
    It fires on exactly one thing: a store written elsewhere with the floor
    relaxed, being read by a workflow the floor does bind.
    """
    if bounds is None:
        return
    if meets_min_historical_years(*bounds):
        return
    raise ValueError(
        f"The {source} store at {climate_nc} covers "
        f"{bounds[0].date()}..{bounds[1].date()} "
        f"(~{(bounds[1] - bounds[0]).days / 365.25:.1f} years), below the "
        f"{MIN_HISTORICAL_YEARS}-year minimum this toolbox requires "
        f"(weathergenr's wavelet decomposition needs at least "
        f"{MIN_HISTORICAL_YEARS} annual observations). If wf0 wrote it as a "
        f"comparison candidate, where the floor is relaxed, re-extract it under "
        f"shared.clim_historical -- or move shared.historical_window onto years "
        f"the source covers. {where}"
    )
