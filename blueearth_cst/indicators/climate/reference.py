"""Retained type-8 climate thresholds and replacement-year bootstrap samples.

Temperature sampling follows the PCIC calendar convention: exclude February 29
from fitting, use a five-day window wrapping within each 365-day reference year,
and use February 28's threshold on February 29 during evaluation. No upstream
package is modified. NumPy's median_unbiased quantile is Hyndman–Fan type 8.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.indicators.climate.daily import daily_array


def calendar_day(dates: pd.DatetimeIndex) -> np.ndarray:
    """Zero-based 365-day index, mapping February 29 to February 28."""
    return np.asarray(
        dates.dayofyear
        - 1
        - (dates.is_leap_year & (dates.month > 2))
        - ((dates.month == 2) & (dates.day == 29)),
        dtype=int,
    )


def temperature_quantile(
    years: np.ndarray, minimum_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit five-day circular-window q90 and retain sample/year support.

    Input shape is year × 365 × optional spatial dimensions. All reference
    years participate in the expected sample count, including missing years.
    """
    shape = years.shape[1:]
    flat = years.reshape((len(years), 365, -1))
    quantile = np.empty(flat.shape[1:])
    counts = np.empty(flat.shape[1:], dtype=int)
    contributing_years = np.empty(flat.shape[1:], dtype=int)
    # Bound the window expansion without truncating cells or breaking time.
    for start in range(0, flat.shape[-1], 32):
        block = slice(start, start + 32)
        samples = np.stack(
            [np.roll(flat[:, :, block], offset, axis=1) for offset in range(-2, 3)],
            axis=1,
        )
        finite = np.isfinite(samples)
        counts[:, block] = finite.sum(axis=(0, 1))
        contributing_years[:, block] = finite.any(axis=1).sum(axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered", category=RuntimeWarning
            )
            quantile[:, block] = np.nanquantile(
                samples, 0.9, axis=(0, 1), method="median_unbiased"
            )
    supported = (counts >= np.ceil(5 * len(years) * minimum_fraction)) & (
        contributing_years >= 2
    )
    return (
        np.where(supported, quantile, np.nan).reshape(shape),
        counts.reshape(shape),
        contributing_years.reshape(shape),
    )


def fit_reference(
    data: xr.Dataset,
    start_year: int,
    end_year: int,
    *,
    wet_day_threshold_mm: float = 1.0,
    minimum_fraction: float = 0.9,
) -> xr.Dataset:
    """Fit source-specific thresholds, retaining temperature data for bootstrap.

    The reference spans complete calendar years; absent dates count as missing.
    At least two declared years are required for replacement-year bootstrap.
    The default 90% sample-support policy is a project quality choice, not a
    claim of a universal ETCCDI eligibility cutoff. Short records remain
    exploratory regardless of coverage. The returned Dataset is serializable.
    """
    if (
        type(start_year) is not int
        or type(end_year) is not int
        or end_year <= start_year
    ):
        raise ValueError("Reference requires at least two ordered integer years")
    if isinstance(minimum_fraction, bool) or not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be in (0, 1]")
    if (
        isinstance(wet_day_threshold_mm, bool)
        or not np.isfinite(wet_day_threshold_mm)
        or wet_day_threshold_mm <= 0
    ):
        raise ValueError("Wet-day threshold must be finite and positive")
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    years = np.arange(start_year, end_year + 1)
    result = xr.Dataset()
    for variable in ("precip", "temp_max", "temp_min"):
        if variable not in data:
            continue
        units = "mm/day" if variable == "precip" else "degC"
        field = daily_array(data[variable], units=units).reindex(time=dates)
        spatial = tuple(dim for dim in field.dims if dim != "time")
        field = field.transpose("time", *spatial)
        coords = {dim: field[dim] for dim in spatial}
        values = field.values
        if variable == "precip":
            if np.any(values < 0):
                raise ValueError("Precipitation must be nonnegative")
            wet = np.where(values >= wet_day_threshold_mm, values, np.nan)
            count = np.isfinite(wet).sum(axis=0)
            observed = np.isfinite(values).sum(axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                q95 = np.nanquantile(wet, 0.95, axis=0, method="median_unbiased")
            q95 = np.where(
                (count >= 2) & (observed >= np.ceil(len(dates) * minimum_fraction)),
                q95,
                np.nan,
            )
            result["precip_p95"] = xr.DataArray(
                q95, dims=spatial, coords=coords, attrs={"units": "mm/day"}
            )
            result["precip_wet_samples"] = xr.DataArray(
                count, dims=spatial, coords=coords
            )
            result["precip_valid_days"] = xr.DataArray(
                observed, dims=spatial, coords=coords
            )
        else:
            keep = ~((dates.month == 2) & (dates.day == 29))
            cube = values[keep].reshape((len(years), 365, *values.shape[1:]))
            threshold, count, support_years = temperature_quantile(
                cube, minimum_fraction
            )
            threshold_coords = {"calendar_day": np.arange(1, 366), **coords}
            dims = ("calendar_day", *spatial)
            result[f"{variable}_p90"] = xr.DataArray(
                threshold, dims=dims, coords=threshold_coords, attrs={"units": "degC"}
            )
            result[f"{variable}_samples"] = xr.DataArray(
                count, dims=dims, coords=threshold_coords
            )
            result[f"{variable}_years"] = xr.DataArray(
                support_years, dims=dims, coords=threshold_coords
            )
            result[f"{variable}_reference"] = xr.DataArray(
                cube,
                dims=("reference_year", *dims),
                coords={"reference_year": years, **threshold_coords},
                attrs={"units": "degC"},
            )
    result.attrs.update(
        reference_start=start_year,
        reference_end=end_year,
        minimum_fraction=float(minimum_fraction),
        wet_day_threshold_mm=float(wet_day_threshold_mm),
        quantile_method="Hyndman-Fan type 8",
        temperature_window_days=5,
        calendar_policy="365-day fitting; Feb29 uses Feb28; circular within-year window",
        qualification="exploratory; coverage does not establish scientific adequacy",
        method_version="climate-reference/1",
    )
    if not result.data_vars:
        raise ValueError("No supported climate variables supplied")
    return result


def exceedance_probability(
    values: np.ndarray, dates: pd.DatetimeIndex, reference: xr.Dataset, variable: str
) -> np.ndarray:
    """Daily bootstrap exceedance probabilities, or fixed-reference out of base.

    Replace each held-out reference year with each other year in turn, evaluate
    the original observations, and average exceedance indicators. All replicates
    must be supported. Extending the evaluation cannot change existing results.
    This probability is for TX90p/TN90p aggregation, never event detection.
    """
    day = calendar_day(dates)
    threshold = reference[f"{variable}_p90"].values
    fixed = threshold[day]
    probability = np.where(
        np.isfinite(values) & np.isfinite(fixed), values > fixed, np.nan
    ).astype(float)
    years = reference.reference_year.values
    cube = reference[f"{variable}_reference"].values.reshape((len(years), 365, -1))
    flat_values = values.reshape((len(dates), -1))
    flat_probability = probability.reshape(flat_values.shape)
    for start in range(0, flat_values.shape[1], 32):
        block = slice(start, start + 32)
        for index, year in enumerate(years):
            selected = np.asarray(dates.year == year)
            if not selected.any():
                continue
            observations = flat_values[selected, block]
            total = np.zeros_like(observations, dtype=float)
            supported = np.isfinite(observations)
            for replacement in range(len(years)):
                if replacement == index:
                    continue
                sample = cube[:, :, block].copy()
                sample[index] = cube[replacement, :, block]
                quantile, _, _ = temperature_quantile(
                    sample, reference.attrs["minimum_fraction"]
                )
                q = quantile[day[selected]]
                supported &= np.isfinite(q)
                total += observations > q
            flat_probability[selected, block] = np.where(
                supported, total / (len(years) - 1), np.nan
            )
    return probability
