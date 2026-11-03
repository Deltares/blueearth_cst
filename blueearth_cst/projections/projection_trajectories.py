"""Unsmoothed annual member anomalies and equal-model trajectory envelopes."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from blueearth_cst.projections.get_change_climate_proj import (
    AnalysisWindowUnavailable,
    analysis_slice,
)
from blueearth_cst.projections.projection_analysis import (
    MODEL_KEY,
    _identity,
    period_values,
)
from blueearth_cst.projections.series_checks import assert_calendar_agreement

REFERENCE_KEY = ["reference_start", "reference_end"]
TRAJECTORY_CELL = ["scenario", *REFERENCE_KEY, "variable", "year"]
MEMBER_COLUMNS = [
    *MODEL_KEY,
    "member",
    *TRAJECTORY_CELL,
    "value",
    "annual_rate",
    "reference_value",
    "reference_rate",
    "anomaly",
    "units",
    "anomaly_units",
    "status",
    "reason",
]


def member_anomalies(
    historical,
    future,
    *,
    identity,
    reference_window,
    water_year_start="Jan",
    min_reference=None,
):
    """Keep annual values even when this member's own reference is unavailable.

    Temperature anomalies are rate differences. Precipitation anomalies use
    annual integrals relative to this member's integral reference mean; annual
    rates are retained separately for the annual-mean figure. No smoothing is
    applied. Corrupt identities, calendars, or month coordinates raise.
    """
    if set(historical.data_vars) != set(future.data_vars):
        raise ValueError("trajectory has asymmetric historical/scenario variables")
    assert_calendar_agreement(historical, future, str(identity["model"]))
    month0 = list(calendar.month_abbr).index(str(water_year_start)[:3].title())
    # Month keys here are zero based, matching the complete-WY boundary formula.
    keys = future.time.dt.year.values * 12 + future.time.dt.month.values - 1
    if not len(keys):
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    hist_keys = (
        historical.time.dt.year.values * 12 + historical.time.dt.month.values - 1
    )
    if not len(hist_keys):
        raise ValueError("trajectory historical series is empty")
    first = int(np.ceil((hist_keys[0] - (month0 - 1)) / 12))
    last = int((keys[-1] + 1 - (month0 - 1)) // 12 - 1)
    if first > last:
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    owned = analysis_slice(
        future, (first, last), water_year_start, historical=historical
    )
    reason = ""
    try:
        reference = analysis_slice(historical, reference_window, water_year_start)
    except AnalysisWindowUnavailable as error:
        reference, reason = None, str(error)
    base = {
        **_identity(identity),
        "reference_start": reference_window[0],
        "reference_end": reference_window[1],
    }
    rows = []
    for variable in sorted(set(owned.data_vars) & {"temp", "precip"}):
        annual = period_values(owned, variable, water_year_start)["annual"]
        ref = (
            period_values(reference, variable, water_year_start)["annual"]
            if reference is not None
            else None
        )
        ref_rate = (
            float(np.mean(ref["level"]))
            if ref is not None and len(ref["years"])
            else np.nan
        )
        ref_value = (
            float(np.mean(ref["total"] if variable == "precip" else ref["level"]))
            if ref is not None and len(ref["years"])
            else np.nan
        )
        status, why = "ok", ""
        if not np.isfinite(ref_value):
            status, why = (
                "unavailable_reference",
                reason or "no complete finite reference years",
            )
        elif variable == "precip" and ref_rate < (min_reference or {}).get(
            "precip", 0.1
        ):
            status, why = "dry", "reference precipitation rate below threshold"
        for year, rate, total in zip(
            annual["years"], annual["level"], annual["total"], strict=True
        ):
            value = total if variable == "precip" else rate
            anomaly = (
                (
                    (value - ref_value) / ref_value * 100
                    if variable == "precip"
                    else value - ref_value
                )
                if status == "ok"
                else np.nan
            )
            rows.append(
                {
                    **base,
                    "variable": variable,
                    "year": int(year),
                    "value": value,
                    "annual_rate": rate,
                    "reference_value": ref_value,
                    "reference_rate": ref_rate,
                    "anomaly": anomaly,
                    "units": "mm/year" if variable == "precip" else "degC",
                    "anomaly_units": "%" if variable == "precip" else "degC",
                    "status": status,
                    "reason": why,
                }
            )
    return pd.DataFrame(rows, columns=MEMBER_COLUMNS)


def model_trajectories(members):
    """Average available member anomalies within each model and actual year."""
    columns = [
        *MODEL_KEY,
        *TRAJECTORY_CELL,
        "anomaly",
        "annual_rate",
        "n_members",
        "n_members_excluded",
    ]
    rows = []
    for key, group in members.groupby(
        [*MODEL_KEY, *TRAJECTORY_CELL], dropna=False, sort=True
    ):
        eligible = group.loc[np.isfinite(group.anomaly)]
        rows.append(
            {
                **dict(zip([*MODEL_KEY, *TRAJECTORY_CELL], key, strict=True)),
                "anomaly": eligible.anomaly.mean(),
                "annual_rate": group.annual_rate.mean(),
                "n_members": len(eligible),
                "n_members_excluded": len(group) - len(eligible),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def trajectory_range(models):
    """Equal-model mean, median and range on unsmoothed annual anomalies."""
    columns = [
        *TRAJECTORY_CELL,
        "mean",
        "median",
        "min",
        "max",
        "n_models",
        "n_models_excluded",
    ]
    rows = []
    for key, group in models.groupby(TRAJECTORY_CELL, dropna=False, sort=True):
        values = group.anomaly[np.isfinite(group.anomaly)]
        rows.append(
            {
                **dict(zip(TRAJECTORY_CELL, key, strict=True)),
                "mean": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "n_models": len(values),
                "n_models_excluded": len(group) - len(values),
            }
        )
    return pd.DataFrame(rows, columns=columns)
