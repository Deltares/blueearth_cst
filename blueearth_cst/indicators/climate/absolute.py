"""Absolute daily precipitation and temperature indices on explicit periods."""

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.indicators.climate.daily import daily_array
from blueearth_cst.indicators.climate.periods import Period


def absolute_indices(
    data: xr.DataArray,
    periods: tuple[Period, ...],
    *,
    variable: str,
    wet_day_threshold_mm: float = 1.0,
    max_missing_days_month: int = 3,
    max_missing_days_period: int = 15,
) -> xr.Dataset:
    """Calculate observed-value indices, masking inadequately covered periods.

    Precipitation returns R10mm, R20mm, Rx1day, Rx5day, wet-day count, wet-day
    precipitation total and SDII. Temperature returns TXx or TNx. Rolling
    precipitation windows stay inside each period and require all five days.
    Missing dates and period-edge days absent from the input count as missing.
    No amount is infilled. Valid/expected days and an eligibility mask accompany
    the values. Arbitrary non-time dimensions, including a location axis, survive.
    """
    if variable not in {"precip", "temp_max", "temp_min"}:
        raise ValueError("variable must be precip, temp_max or temp_min")
    if not periods or len({p.label for p in periods}) != len(periods):
        raise ValueError("Require nonempty periods with unique labels")
    for tolerance in (max_missing_days_month, max_missing_days_period):
        if type(tolerance) is not int or tolerance < 0:
            raise ValueError("Missing-day tolerances require nonnegative integers")
    if (
        isinstance(wet_day_threshold_mm, bool)
        or not np.isfinite(wet_day_threshold_mm)
        or wet_day_threshold_mm <= 0
    ):
        raise ValueError("Wet-day threshold must be finite and positive")
    data = daily_array(data, units="mm/day" if variable == "precip" else "degC")
    if variable == "precip" and bool((data < 0).any()):
        raise ValueError("Precipitation must be nonnegative")
    results = []
    for period in periods:
        dates = pd.date_range(period.start, period.stop, freq="D", inclusive="left")
        values = data.reindex(time=dates)
        valid = values.notnull()
        count = valid.sum("time")
        missing_month = (~valid).resample(time="MS").sum().max("time")
        eligible = (
            (count > 0)
            & (len(dates) - count <= max_missing_days_period)
            & (missing_month <= max_missing_days_month)
        )
        units = {}
        if variable == "precip":
            wet = valid & (values >= wet_day_threshold_mm)
            wet_count = wet.sum("time")
            wet_total = values.where(wet, 0).sum("time")
            metrics = {
                "r10mm": (valid & (values >= 10)).sum("time"),
                "r20mm": (valid & (values >= 20)).sum("time"),
                "rx1day": values.max("time"),
                "rx5day": values.rolling(time=5, min_periods=5).sum().max("time"),
                "wet_days": wet_count,
                "prcptot": wet_total,
                "sdii": wet_total / wet_count.where(wet_count > 0),
            }
            units = {
                "r10mm": "days",
                "r20mm": "days",
                "wet_days": "days",
                "rx1day": "mm",
                "rx5day": "mm",
                "prcptot": "mm",
                "sdii": "mm/day",
            }
        else:
            name = "txx" if variable == "temp_max" else "tnx"
            metrics = {name: values.max("time")}
            units[name] = "degC"
        result = xr.Dataset(
            {name: value.where(eligible) for name, value in metrics.items()}
        )
        for name, unit in units.items():
            result[name].attrs["units"] = unit
        result["valid_days"] = count
        result["expected_days"] = len(dates)
        result["eligible"] = eligible
        result = result.expand_dims(period=[period.label])
        result = result.assign_coords(
            period_start=("period", [period.start.to_datetime64()]),
            period_stop=("period", [period.stop.to_datetime64()]),
        )
        results.append(result)
    output = xr.concat(results, dim="period")
    output.attrs.update(
        variable=variable,
        calendar="proleptic_gregorian",
        missing_policy="observed values only; no infilling",
        wet_day_threshold_mm=float(wet_day_threshold_mm),
        max_missing_days_month=max_missing_days_month,
        max_missing_days_period=max_missing_days_period,
    )
    return output
