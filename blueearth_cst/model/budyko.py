"""Observed water-balance reductions for explicitly aligned daily intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def budyko_settings(value: object, observations_path: object) -> dict[str, str] | None:
    """Validate the optional explicit observation-support declaration."""
    if value is None or value is False:
        return None
    example = (
        "evaluation.budyko requires {discharge_statistic: daily_mean, "
        "time_label: interval_end}; time_label also accepts interval_start"
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"discharge_statistic", "time_label"}
        or value.get("discharge_statistic") != "daily_mean"
        or value.get("time_label") not in ("interval_start", "interval_end")
    ):
        raise ValueError(example)
    if observations_path is None or observations_path == "":
        raise ValueError("evaluation.budyko requires observations.river discharge")
    return dict(value)


def read_daily_discharge(path: str, time_label: str) -> pd.DataFrame:
    """Read user-declared daily means, returning derived UTC interval-end labels.

    This checks structural consistency only. It cannot verify the declared
    daily-mean statistic or UTC support from the numerical values.
    """
    if time_label not in ("interval_start", "interval_end"):
        raise ValueError("time_label must be interval_start or interval_end")
    data = pd.read_csv(path, sep=";", dtype={"time": str})
    if "time" not in data or len(data.columns) < 2:
        raise ValueError("discharge CSV requires time;<wflow_id>;... columns")
    index = pd.DatetimeIndex(pd.to_datetime(data.pop("time"), utc=True, format="mixed"))
    if index.hasnans or index.has_duplicates or not index.equals(index.normalize()):
        raise ValueError("discharge timestamps must be unique midnight UTC labels")
    data = data.apply(pd.to_numeric, errors="raise")
    if np.isinf(data.to_numpy()).any() or (data < 0).any().any():
        raise ValueError("daily mean discharge must be nonnegative or missing")
    data.index = index.tz_localize(None)
    if time_label == "interval_start":
        data.index += pd.Timedelta(days=1)
    return data.sort_index()


def discharge_depth(discharge: pd.Series, area_m2: float) -> pd.Series:
    """Convert daily mean discharge (m3/s) to depth over its upstream support."""
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("contributing area_m2 must be finite and positive")
    values = discharge.to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("daily mean discharge must be nonnegative or missing")
    return discharge * (86400 * 1000 / area_m2)


def matched_water_balance(
    daily: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    water_year_start: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reduce complete matched water years to ratios of totals.

    Args:
        daily: Columns precip, pet, runoff_mm, in mm per daily interval. The
            unique UTC or naive-UTC index labels midnight interval ends;
            callers must establish this provenance before invoking this function.
        start: Inclusive first interval start of the requested window.
        end: Exclusive final interval end of the requested window.
        water_year_start: First month of the reporting year (1 through 12).

    Returns:
        Annual coverage/totals and a summary over the same eligible years.
        Years are labelled by their starting calendar year. Partial edge years
        and years with any missing paired value are retained as exclusions.
    """
    if type(water_year_start) is not int or not 1 <= water_year_start <= 12:
        raise ValueError("water_year_start must be an integer month from 1 to 12")
    columns = ["precip", "pet", "runoff_mm"]
    if set(daily.columns) != set(columns):
        raise ValueError(f"daily water balance requires exactly {columns}")
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("daily index must contain midnight UTC interval ends")
    index = daily.index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if index.hasnans or index.has_duplicates or not index.equals(index.normalize()):
        raise ValueError("daily index must contain unique midnight UTC intervals")
    values = daily[columns].astype(float).copy()
    values.index = index - pd.Timedelta(days=1)
    if np.isinf(values.to_numpy()).any() or (values < 0).any().any():
        raise ValueError("water-balance depths must be nonnegative or missing")
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    if first.tz is not None:
        first = first.tz_convert("UTC").tz_localize(None)
    if stop.tz is not None:
        stop = stop.tz_convert("UTC").tz_localize(None)
    if first != first.normalize() or stop != stop.normalize() or first >= stop:
        raise ValueError("window must have increasing midnight UTC boundaries")
    calendar = pd.date_range(first, stop, freq="D", inclusive="left")
    aligned = values.reindex(calendar)
    labels = calendar.year - (calendar.month < water_year_start).astype(int)
    rows = []
    for year in sorted(set(labels)):
        year_start = pd.Timestamp(int(year), water_year_start, 1)
        year_stop = pd.Timestamp(int(year) + 1, water_year_start, 1)
        expected = (year_stop - year_start).days
        subset = aligned.loc[labels == year]
        matched = subset.notna().all(axis=1)
        complete = len(subset) == expected and bool(matched.all())
        reason = (
            ""
            if complete
            else "partial_window"
            if len(subset) != expected
            else "missing_days"
        )
        row = {
            "water_year": int(year),
            "expected_days": expected,
            "window_days": len(subset),
            "matched_days": int(matched.sum()),
            "eligible": complete,
            "reason": reason,
        }
        for column in columns:
            row[f"{column}_mm" if column != "runoff_mm" else column] = (
                float(subset[column].sum()) if complete else float("nan")
            )
        rows.append(row)
    annual = pd.DataFrame(rows)
    retained = annual.loc[annual["eligible"]]
    years = retained["water_year"].astype(int).tolist()
    precip = float(retained["precip_mm"].sum())
    pet = float(retained["pet_mm"].sum())
    runoff = float(retained["runoff_mm"].sum())
    available = bool(years) and precip > 0
    summary = {
        "available": available,
        "reason": ""
        if available
        else "no_complete_years"
        if not years
        else "zero_precipitation",
        "retained_years": years,
        "year_count": len(years),
        "discontinuous": any(b != a + 1 for a, b in zip(years, years[1:])),
        "precip_mm": precip,
        "pet_mm": pet,
        "runoff_mm": runoff,
        "aridity_index": pet / precip if available else float("nan"),
        "runoff_coefficient": runoff / precip if available else float("nan"),
    }
    return annual, summary
