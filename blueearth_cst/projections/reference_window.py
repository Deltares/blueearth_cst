"""Reference-window clipping and its warnings (design §5.4 D1, step 5e).

The change-factor reference is the GCM **historical** experiment, which ends
2014-12-31. A configured reference window reaching past that cannot be satisfied,
and the design's ruling (R1) is to **clip, never splice**: 2015+ exists only under
the per-scenario ScenarioMIP entries, and stitching them onto the historical tail
would silently mix two experiments inside one reference (N8).

The interesting part is not the clip — it is *which conditions warn where*. D1
separates three, on the principle that **a signal that fires on every run is a
signal nobody reads**:

===================  ==================  =========================================
condition            stderr at DAG build  durable record
===================  ==================  =========================================
clip                 yes                  requested + effective + ``clipped``
alignment differs    **no** by default    both windows + ``reference_alignment``
short window         yes                  effective length in years
===================  ==================  =========================================

Alignment is silent by default for a concrete reason: the shipped seed config has
``historical_year_range: [1990, 2010]`` against ``shared.historical_window``
2000–2020, so the two differ on **100 %** of runs. Warning there would train the
reader to filter the channel. It is promoted to stderr only when the windows
*would have been equal but for the clip* — the case where the user plausibly
intended alignment and did not get it.

Falsifiers: ``dev/milestones/r08/2026-07-30_wf2-5e-falsifier.md`` K3–K6.
"""
from __future__ import annotations

from typing import NamedTuple

#: Last year the historical experiment covers. From
#: ``series_identity.ACQUISITION_WINDOWS["historical"]`` (…1950-01-01 → 2014-12-31),
#: restated here as a year because the reference window is configured in years.
HISTORICAL_END_YEAR = 2014

#: First year the historical experiment covers.
HISTORICAL_START_YEAR = 1950

#: Below this many years the effective window earns a stderr warning (D1).
#: A boundary the seed sits exactly on — `[1990, 2010]` is 20 — so an off-by-one
#: here is invisible on the fixture and wrong everywhere else.
SHORT_WINDOW_YEARS = 20


class ReferenceWindow(NamedTuple):
    """A requested reference window and what the source can actually supply."""

    requested: tuple[int, int]
    effective: tuple[int, int]
    clipped: bool
    n_years: int

    @property
    def requested_label(self) -> str:
        return f"{self.requested[0]}-{self.requested[1]}"

    @property
    def effective_label(self) -> str:
        return f"{self.effective[0]}-{self.effective[1]}"


def _years(window) -> tuple[int, int]:
    """Normalise ``[1990, 2010]`` / ``"1990, 2010"`` / datetimes to a year pair."""
    if isinstance(window, str):
        parts = [p.strip() for p in window.split(",")]
    else:
        parts = list(window)
    if len(parts) != 2:
        raise ValueError(f"expected a start,end pair; got {window!r}")
    return tuple(int(str(p)[:4]) for p in parts)


def clip_reference_window(requested) -> ReferenceWindow:
    """Clip a requested reference window to what the historical experiment holds.

    ``effective = requested ∩ [HISTORICAL_START_YEAR, HISTORICAL_END_YEAR]``.

    Raises when the requested window lies **entirely** after the historical end —
    the one case D1 makes an error rather than a clip, because there is nothing to
    clip *to* and a zero-length reference would otherwise propagate as an empty
    denominator into every relative change factor.
    """
    start, end = _years(requested)
    if start > HISTORICAL_END_YEAR:
        raise ValueError(
            f"reference window {start}-{end} lies entirely after the historical "
            f"experiment, which ends {HISTORICAL_END_YEAR}. There is nothing to "
            "clip to. The design does not splice scenario data into the reference "
            "(N8), so this is a configuration error rather than a warning."
        )
    effective = (max(start, HISTORICAL_START_YEAR), min(end, HISTORICAL_END_YEAR))
    return ReferenceWindow(
        requested=(start, end),
        effective=effective,
        clipped=effective != (start, end),
        # Matches `hydrological_year_bounds`: complete hydrological years between
        # the two starts, i.e. last - first.
        n_years=effective[1] - effective[0],
    )


def window_warnings(window: ReferenceWindow, shared_window=None) -> list[str]:
    """stderr warnings for one reference window — per condition, never lumped.

    Returns the lines to print at DAG build. An empty list is the normal case for
    a well-configured run, which is what makes a non-empty one worth reading.
    """
    lines: list[str] = []

    if window.clipped:
        lines.append(
            f"reference window clipped: requested {window.requested_label}, "
            f"effective {window.effective_label} (the historical experiment ends "
            f"{HISTORICAL_END_YEAR}; reference data is clipped, never spliced)."
        )

    if window.n_years < SHORT_WINDOW_YEARS:
        lines.append(
            f"short reference window: {window.n_years} years "
            f"({window.effective_label}). Statistics derived from it are "
            f"correspondingly uncertain; {SHORT_WINDOW_YEARS} is the design's "
            "reference length."
        )

    # Alignment: silent unless the clip is what broke it. See the module docstring.
    if shared_window is not None:
        shared = _years(shared_window)
        if window.effective != shared and window.requested == shared:
            lines.append(
                f"reference window no longer aligns with shared.historical_window "
                f"{shared[0]}-{shared[1]}: they were equal as configured, and the "
                f"clip to {window.effective_label} broke the alignment."
            )
    return lines


def dropped_months(data_start, data_end, effective_start, effective_end) -> tuple[int, int]:
    """Months discarded at each end by the complete-hydrological-year policy.

    A1 requires artifacts stating an analysis window to state the **per-end
    dropped-month counts** alongside nominal and effective bounds — because
    "29 years from a 30-year window" is not self-explaining, and "9 dropped at the
    front, 3 at the back" is.

    Returns ``(leading, trailing)`` in whole months.
    """
    import pandas as pd

    def months_between(a, b):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        return max(0, (b.year - a.year) * 12 + (b.month - a.month))

    return (
        months_between(data_start, effective_start),
        months_between(effective_end, data_end),
    )


def alignment_record(window: ReferenceWindow, shared_window=None) -> dict:
    """The durable facts, for the composition record (and `provenance.json` at 6a).

    Separate from :func:`window_warnings` because silence and absence are
    different: a difference that does not warrant a stderr line must still be
    recoverable afterwards, which is the whole of D1's "the disclaimer is what
    surfaces that".
    """
    record = {
        "reference_window_requested": window.requested_label,
        "reference_window_effective": window.effective_label,
        "reference_window_clipped": window.clipped,
        "reference_window_years": window.n_years,
    }
    if shared_window is not None:
        shared = _years(shared_window)
        record["shared_historical_window"] = f"{shared[0]}-{shared[1]}"
        record["reference_alignment"] = (
            "matches" if window.effective == shared else "differs"
        )
    return record
