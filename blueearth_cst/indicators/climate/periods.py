"""Explicit Gregorian reporting periods with start-year labels."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Period:
    """A half-open reporting interval; the label does not define membership."""

    label: str
    start: pd.Timestamp
    stop: pd.Timestamp

    def __post_init__(self) -> None:
        for name in ("start", "stop"):
            date = pd.Timestamp(getattr(self, name))
            if pd.isna(date) or date.tz is not None or date != date.normalize():
                raise ValueError("Period boundaries require timezone-free daily dates")
            object.__setattr__(self, name, date)
        if self.start >= self.stop:
            raise ValueError("Period requires start < stop")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Period requires a nonempty label")


def annual_periods(start: int, end: int, start_month: int = 1) -> tuple[Period, ...]:
    """Construct complete annual intervals labelled by their start year."""
    if any(type(value) is not int for value in (start, end, start_month)):
        raise ValueError("Years and start month require integers")
    if start > end or not 1 <= start_month <= 12:
        raise ValueError("Require ordered years and a start month in 1..12")
    return tuple(
        Period(
            str(year),
            pd.Timestamp(year, start_month, 1),
            pd.Timestamp(year + 1, start_month, 1),
        )
        for year in range(start, end + 1)
    )


def seasonal_periods(
    start: int, end: int, *, name: str, months: tuple[int, ...]
) -> tuple[Period, ...]:
    """Construct an explicitly ordered contiguous season, labelled by start year.

    For example, ``months=(12, 1, 2)`` gives December through February, with
    December's year in the label. No geographical or rainfall-season inference
    is made. Disconnected month selections are not continuous event periods.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A season requires a nonempty name")
    if (
        not months
        or len(months) > 12
        or any(type(month) is not int or not 1 <= month <= 12 for month in months)
        or any(right != left % 12 + 1 for left, right in zip(months, months[1:]))
    ):
        raise ValueError("Season months must be unique, contiguous and ordered")
    years = annual_periods(start, end, months[0])
    return tuple(
        Period(
            f"{name}:{period.label}",
            period.start,
            period.start + pd.DateOffset(months=len(months)),
        )
        for period in years
    )
