"""Validated neutral response series consumed by indicator implementations."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseSeries:
    """One run, canonical variable and location with explicit temporal metadata."""

    run_id: str
    variable: str
    location_id: str
    location_ordinal: int
    time: pd.DatetimeIndex
    calendar: str
    timestep: timedelta
    time_label: str
    units: str
    values: np.ndarray
    missing: np.ndarray

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the unique response key."""
        return self.run_id, self.variable, self.location_id


def validate_responses(
    series: Sequence[ResponseSeries],
    *,
    evaluated_runs: Collection[str],
    required_keys: Collection[tuple[str, str, str]] | None = None,
) -> None:
    """Refuse ambiguous, incomplete or falsely labelled response series.

    Extras remain explicit in the supplied sequence. A consumer with a fixed
    request supplies its exact required keys instead of treating extras as requests.
    """
    if not series:
        raise ValueError("response set is empty")
    keys: set[tuple[str, str, str]] = set()
    ordinals: dict[tuple[str, str], list[int]] = {}
    for item in series:
        if item.key in keys:
            raise ValueError(f"duplicate response {item.key}")
        keys.add(item.key)
        if item.run_id not in evaluated_runs:
            raise ValueError(f"response {item.key}: run is not evaluated")
        if (
            not item.variable
            or not item.location_id
            or not item.units
            or not item.calendar
        ):
            raise ValueError(f"response {item.key}: metadata must be explicit")
        if item.calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError(
                f"response {item.key}: calendar is incompatible with the datetime64 binding"
            )
        if item.time_label not in {"interval_end", "instant"}:
            raise ValueError(f"response {item.key}: invalid time label")
        if (
            not isinstance(item.time, pd.DatetimeIndex)
            or item.time.hasnans
            or len(item.time) < 2
        ):
            raise ValueError(
                f"response {item.key}: time must be a nonempty typed daily coordinate"
            )
        if not item.time.is_unique or not item.time.is_monotonic_increasing:
            raise ValueError(f"response {item.key}: time must increase uniquely")
        if (
            item.timestep <= timedelta(0)
            or not (item.time[1:] - item.time[:-1] == item.timestep).all()
        ):
            raise ValueError(
                f"response {item.key}: timestep does not match the coordinate"
            )
        if (
            item.values.ndim != 1
            or item.missing.ndim != 1
            or len(item.values) != len(item.time)
            or len(item.missing) != len(item.time)
        ):
            raise ValueError(f"response {item.key}: value/time/missing lengths differ")
        if item.missing.dtype != np.dtype(bool) or not np.array_equal(
            item.missing, np.isnan(item.values)
        ):
            raise ValueError(
                f"response {item.key}: missing mask must identify native NaNs exactly"
            )
        if np.isinf(item.values).any():
            raise ValueError(
                f"response {item.key}: infinite values are not missing data"
            )
        if type(item.location_ordinal) is not int or item.location_ordinal < 0:
            raise ValueError(f"response {item.key}: invalid location ordinal")
        ordinals.setdefault((item.run_id, item.variable), []).append(
            item.location_ordinal
        )
    for key, values in ordinals.items():
        if sorted(values) != list(range(len(values))):
            raise ValueError(
                f"response {key}: native location ordinals must be unique and contiguous"
            )
    if required_keys is not None:
        missing = set(required_keys) - keys
        if missing:
            raise ValueError(f"missing requested responses: {sorted(missing)}")


def compatible_bundle(series: Sequence[ResponseSeries]) -> None:
    """Require identical temporal and physical interpretation before pooling."""
    if not series:
        raise ValueError("response bundle is empty")
    first = series[0]
    for item in series[1:]:
        if (
            item.variable,
            item.location_id,
            item.units,
            item.calendar,
            item.timestep,
            item.time_label,
        ) != (
            first.variable,
            first.location_id,
            first.units,
            first.calendar,
            first.timestep,
            first.time_label,
        ) or not item.time.equals(first.time):
            raise ValueError(
                f"incompatible response bundle: {first.key} and {item.key}"
            )


def response_frame(series: Sequence[ResponseSeries]) -> pd.DataFrame:
    """Expose one variable/run as location-labelled columns in native order."""
    if not series:
        raise ValueError("cannot frame an empty response set")
    first = series[0]
    validate_responses(series, evaluated_runs={first.run_id})
    if any(
        (item.run_id, item.variable) != (first.run_id, first.variable)
        or not item.time.equals(first.time)
        for item in series
    ):
        raise ValueError("a response frame requires one run/variable and time axis")
    return pd.DataFrame(
        {
            item.location_id: item.values
            for item in sorted(series, key=lambda item: item.location_ordinal)
        },
        index=first.time,
    )
