"""Strict daily rain-gauge inputs and matched observed-only calculations."""

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.climate_analysis.diagnostics import reporting_year


def parse_rain_gauges(block: object) -> dict | None:
    """Validate the optional, closed observation settings at the WF0 boundary."""
    if block is None:
        return None
    keys = {"stations_path", "precip_path", "time_label"}
    if (
        not isinstance(block, dict)
        or not keys.issubset(block)
        or set(block) - keys - {"temp_path"}
    ):
        raise ValueError(
            f"rain_gauges requires {sorted(keys)}; only temp_path is optional"
        )
    if block["time_label"] not in ("interval_start", "interval_end"):
        raise ValueError(
            "rain_gauges.time_label requires interval_start or interval_end"
        )
    for key in (
        "stations_path",
        "precip_path",
        *(["temp_path"] if "temp_path" in block else []),
    ):
        if not isinstance(block[key], str) or not block[key].strip():
            raise ValueError(f"rain_gauges.{key} requires a nonempty path string")
        if not Path(block[key]).is_file():
            raise FileNotFoundError(f"rain_gauges.{key}: {block[key]}")
    return dict(block)


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read literal strings without pandas' duplicate-header or NA coercion."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or not rows[0] or len(set(rows[0])) != len(rows[0]):
        raise ValueError(f"{path}: missing or duplicate CSV columns")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError(f"{path}: inconsistent CSV row width")
    return pd.DataFrame(rows[1:], columns=rows[0])


def read_stations(path: str | Path) -> pd.DataFrame:
    """Read stable station IDs and finite WGS84 coordinates without rewriting IDs."""
    frame = _read_csv(path)
    if not {"id", "x", "y"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Station CSV requires id, x, y and at least one station")
    if frame.id.duplicated().any() or frame.id.str.strip().eq("").any():
        raise ValueError("Station IDs must be nonempty unique strings")
    for key, lower, upper in (("x", -180, 180), ("y", -90, 90)):
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if not (np.isfinite(frame[key]) & frame[key].between(lower, upper)).all():
            raise ValueError(f"Station {key} must be finite WGS84 degrees")
    if "elevtn" in frame:
        frame["elevtn"] = pd.to_numeric(
            frame.elevtn.replace("", np.nan), errors="raise"
        )
        if np.isinf(frame.elevtn).any():
            raise ValueError("Station elevtn must be finite or blank")
    # Keep ordinary IDs readable; digest every unsafe ID to prevent normalization collisions.
    frame["token"] = [
        value
        if re.fullmatch(r"[a-z0-9]+", value)
        else "id_" + hashlib.sha256(value.encode()).hexdigest()
        for value in frame.id
    ]
    if frame.token.duplicated().any():
        raise ValueError("Station filename token collision")
    return frame.set_index("id", drop=False)


def _read_daily(path: str | Path, time_label: str, variable: str) -> pd.DataFrame:
    """Read daily UTC measurements, preserving missing dates and station IDs."""
    frame = _read_csv(path)
    if "time" not in frame or len(frame.columns) < 2:
        raise ValueError(f"{variable.title()} CSV requires time and station columns")
    times = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("time"), errors="raise", utc=True)
    )
    if times.hasnans or times.has_duplicates or not times.equals(times.normalize()):
        raise ValueError("Observation dates must be unique midnight UTC timestamps")
    values = (
        frame.replace("", np.nan).apply(pd.to_numeric, errors="raise").astype(float)
    )
    if np.isinf(values.to_numpy()).any() or (
        variable == "precipitation" and (values < 0).any().any()
    ):
        raise ValueError(
            f"Observed {variable} must be finite {'nonnegative mm' if variable == 'precipitation' else 'degrees Celsius'} or blank"
        )
    if time_label not in ("interval_start", "interval_end"):
        raise ValueError("Unknown observation time_label")
    values.index = times.tz_localize(None) + pd.Timedelta(
        days=int(time_label == "interval_start")
    )
    return values.sort_index()


def read_precip(path: str | Path, time_label: str) -> pd.DataFrame:
    """Read mm per midnight UTC interval, normalizing start labels to ends."""
    return _read_daily(path, time_label, "precipitation")


def read_temperature(path: str | Path, time_label: str) -> pd.DataFrame:
    """Read daily mean degrees Celsius, normalizing start labels to ends."""
    return _read_daily(path, time_label, "temperature")


def matched_tables(
    daily: pd.DataFrame, m0: int, *, observed: bool = True
) -> dict[str, pd.DataFrame]:
    """Calculate metrics and strictly complete totals on one common valid-day mask.

    Input labels are interval ends. Calendar attribution uses the preceding day.
    The caller supplies the full expected daily axis, including absent rows.
    """
    if observed and "observed" not in daily:
        raise ValueError("Daily values require an observed column")
    valid = np.isfinite(daily) & (daily >= 0)
    joint = valid.all(axis=1)
    paired = daily.where(joint)
    days = daily.index - pd.Timedelta(days=1)
    n = int(joint.sum())
    bounds = {
        "expected_days": len(daily),
        "paired_days": n,
        "coverage": n / len(daily) if len(daily) else np.nan,
        "first_interval_end": str(daily.index[joint][0]) if n else "",
        "last_interval_end": str(daily.index[joint][-1]) if n else "",
    }
    coverage = [
        {"source": key, "valid_days": int(valid[key].sum()), **bounds} for key in daily
    ]
    metrics = []
    reference = paired.observed.dropna() if observed else pd.Series(dtype=float)
    for key in daily.columns.drop("observed") if observed else []:
        source = paired[key].dropna()
        errors = source - reference
        corr_ok = n >= 2 and reference.nunique() > 1 and source.nunique() > 1
        metrics.append(
            {
                "source": key,
                **bounds,
                "bias_mm": errors.mean(),
                "mae_mm": errors.abs().mean(),
                "rmse_mm": np.sqrt((errors**2).mean()),
                "percentage_bias": 100 * errors.sum() / reference.sum()
                if n and reference.sum()
                else np.nan,
                "pearson_r": source.corr(reference) if corr_ok else np.nan,
                "reason": "" if n else "no_common_valid_days",
                "correlation_reason": "" if corr_ok else "too_few_pairs_or_constant",
                "percentage_bias_reason": ""
                if n and reference.sum()
                else "zero_or_missing_observed_total",
            }
        )
    monthly, annual = [], []
    for month in days.to_period("M").unique():
        mask = days.to_period("M") == month
        complete = int(mask.sum()) == month.days_in_month and bool(joint[mask].all())
        for key in daily:
            monthly.append(
                {
                    "month": str(month),
                    "calendar_month": month.month,
                    "source": key,
                    "expected_days": month.days_in_month,
                    "paired_days": int(joint[mask].sum()),
                    "total_mm": paired.loc[mask, key].sum() if complete else np.nan,
                    "reason": "" if complete else "incomplete_month",
                }
            )
    years = reporting_year(days, m0)
    for year in np.unique(years):
        mask = years == year
        expected = (
            pd.Timestamp(year=year + 1, month=m0, day=1)
            - pd.Timestamp(year=year, month=m0, day=1)
        ).days
        complete = int(mask.sum()) == expected and bool(joint[mask].all())
        for key in daily:
            annual.append(
                {
                    "reporting_year": int(year),
                    "source": key,
                    "expected_days": expected,
                    "paired_days": int(joint[mask].sum()),
                    "total_mm": paired.loc[mask, key].sum() if complete else np.nan,
                    "reason": "" if complete else "incomplete_reporting_year",
                }
            )
    monthly_df = pd.DataFrame(monthly)
    monthly_df["climatology_n"] = monthly_df.groupby(["source", "calendar_month"])[
        "total_mm"
    ].transform("count")
    return {
        "coverage": pd.DataFrame(coverage),
        "agreement": pd.DataFrame(
            metrics,
            columns=[
                "source",
                *bounds,
                "bias_mm",
                "mae_mm",
                "rmse_mm",
                "percentage_bias",
                "pearson_r",
                "reason",
                "correlation_reason",
                "percentage_bias_reason",
            ],
        ),
        "monthly": monthly_df,
        "annual": pd.DataFrame(annual),
    }
