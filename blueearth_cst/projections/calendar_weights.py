"""Month-length weights in the model's own calendar (design §5.6, step 5b).

The annual aggregate treated all twelve months alike: `.sum()` for precip,
`.mean()` for temp. That makes a 28-day February count as much as a 31-day
January, and — worse for the design's actual purpose — it makes annual means
differ between a `360_day` model and a `noleap` one **for procedural reasons**,
not physical ones. This module supplies the weights that remove that.

**Why the lengths are derived from `(calendar, year, month)` and never from the
time axis.** The series' axis is `datetime64`: our catalog requests
`preprocess: harmonise_dims`, whose time branch converts a `CFTimeIndex` away
(`hydromt/data_catalog/drivers/preprocessing.py:66`). `time.dt.days_in_month` on
that axis returns **Gregorian** lengths — 29 for February in a leap year — for a
model that has no such day. The calendar name is carried separately on the series
(`cst_calendar`, put there by the fetch reading the store; see
`dev/milestones/r08/2026-07-30_wf2-5b-calendar-blocker.md`), and the converted axis still
carries year and month exactly, which is all that is needed alongside it.

Falsifiers: `dev/milestones/r08/2026-07-30_wf2-5b-falsifier.md`.
"""
from __future__ import annotations

import numpy as np

from blueearth_cst.projections.series_identity import CALENDAR_UNKNOWN

#: CF calendars this module can weight. Anything else refuses rather than
#: guessing — the design requires stage B to "raise on a calendar it cannot
#: weight", and A3 made that reachable by recording an honest sentinel instead of
#: an empty string.
SUPPORTED_CALENDARS = frozenset(
    {
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "julian",
        "noleap",
        "365_day",
        "all_leap",
        "366_day",
        "360_day",
    }
)


class CalendarError(ValueError):
    """A series whose annual aggregate cannot be honestly weighted."""


def assert_weightable(calendar, source=""):
    """Raise unless ``calendar`` is one this module can weight, naming the source."""
    where = f" ({source})" if source else ""
    name = str(calendar or "").strip()
    if not name or name == CALENDAR_UNKNOWN:
        raise CalendarError(
            f"calendar is unknown{where}: the annual aggregate cannot be weighted "
            "by month length without it. A slice written before schema 3 records "
            "no calendar; re-fetch it. Refusing rather than silently applying "
            "Gregorian month lengths, which is the defect this guard exists for."
        )
    if name not in SUPPORTED_CALENDARS:
        raise CalendarError(
            f"calendar {name!r} is not weightable{where}. Known: "
            f"{sorted(SUPPORTED_CALENDARS)}. Refused rather than approximated."
        )
    return name


def days_in_month(year, month, calendar):
    """Length of one month in ``calendar``, from the calendar alone.

    Uses ``cftime`` so every CF calendar is handled by the same code path rather
    than by a hand-maintained table of leap rules — including ``360_day``, where
    the answer is 30 for every month and no Gregorian rule applies.
    """
    import cftime

    year, month = int(year), int(month)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    start = cftime.datetime(year, month, 1, calendar=calendar)
    end = cftime.datetime(next_year, next_month, 1, calendar=calendar)
    return (end - start).days


def month_length_weights(times, calendar, source=""):
    """Weight per monthly timestamp: its length in ``calendar``.

    ``times`` supplies only year and month — deliberately. Its own dtype is
    irrelevant and, for a converted axis, actively misleading.
    """
    name = assert_weightable(calendar, source)
    index = getattr(times, "indexes", {}).get("time", None)
    if index is None:
        index = np.asarray(times)
        years = np.array([t.year for t in index])
        months = np.array([t.month for t in index])
    else:
        years, months = index.year.values, index.month.values
    return np.array(
        [days_in_month(y, m, name) for y, m in zip(years, months)], dtype="float64"
    )
