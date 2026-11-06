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
    mean_temperature_threshold_degc: float = 25.0,
    max_missing_days_month: int = 3,
    max_missing_days_period: int = 15,
) -> xr.Dataset:
    """Calculate observed-value indices, masking inadequately covered periods.

    Precipitation returns R10mm, R20mm, Rx1day, Rx5day, wet-day count, wet-day
    precipitation total and SDII. Temperature returns TXx or TNx; daily mean
    temperature returns counts strictly below zero and above the declared
    threshold (25°C by default). These are not TN frost or TX summer-day indices. Rolling
    precipitation windows stay inside each period and require all five days.
    Missing dates and period-edge days absent from the input count as missing.
    No amount is infilled. Valid/expected days and an eligibility mask accompany
    the values. Arbitrary non-time dimensions, including a location axis, survive.
    """
    if variable not in {"precip", "temp", "temp_max", "temp_min"}:
        raise ValueError("variable must be precip, temp, temp_max or temp_min")
    if (
        isinstance(mean_temperature_threshold_degc, bool)
        or not isinstance(mean_temperature_threshold_degc, (int, float))
        or not np.isfinite(mean_temperature_threshold_degc)
    ):
        raise ValueError("mean_temperature_threshold_degc must be a finite number")
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
        elif variable == "temp":
            metrics = {
                "mean_temp_below_zero_days": (valid & (values < 0)).sum("time"),
                "mean_temp_above_threshold_days": (
                    valid & (values > mean_temperature_threshold_degc)
                ).sum("time"),
            }
            units = {name: "days" for name in metrics}
        else:
            name = "txx" if variable == "temp_max" else "tnx"
            metrics = {name: values.max("time")}
            units[name] = "degC"
        result = xr.Dataset(
            {name: value.where(eligible) for name, value in metrics.items()}
        )
        for name, unit in units.items():
            result[name].attrs["units"] = unit
        if variable == "temp":
            for name, threshold, operator in (
                ("mean_temp_below_zero_days", 0.0, "<"),
                (
                    "mean_temp_above_threshold_days",
                    mean_temperature_threshold_degc,
                    ">",
                ),
            ):
                result[name].attrs.update(
                    input_variable="daily mean temperature",
                    threshold_degc=float(threshold),
                    operator=operator,
                    missing_policy="observed days; no scaling or infilling",
                )
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
