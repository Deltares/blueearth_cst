"""Hand-calculated absolute indices and spatial/period invariants."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.indicators.climate.absolute import absolute_indices
from blueearth_cst.indicators.climate.periods import (
    Period,
    annual_periods,
    seasonal_periods,
)


def test_precipitation_boundaries_and_wet_day_denominator():
    data = xr.DataArray(
        [0.9, 1.0, 10.0, 20.0, 0.0],
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=5)},
        attrs={"units": "mm/day"},
    )
    period = (Period("test", "2001-01-01", "2001-01-06"),)
    result = absolute_indices(data, period, variable="precip")
    expected = {
        "r10mm": 2,
        "r20mm": 1,
        "rx1day": 20,
        "rx5day": 31.9,
        "wet_days": 3,
        "prcptot": 31,
        "sdii": 31 / 3,
    }
    for name, value in expected.items():
        assert result[name].item() == pytest.approx(value)
    assert result.eligible.item()
    assert result.valid_days.item() == 5


def test_missing_period_edges_count_against_completeness():
    data = xr.DataArray(
        [10.0],
        dims="time",
        coords={"time": pd.date_range("2001-06-01", periods=1)},
        attrs={"units": "degC"},
    )
    result = absolute_indices(data, annual_periods(2001, 2001), variable="temp_max")
    assert not result.eligible.item()
    assert np.isnan(result.txx.item())
    assert result.expected_days.item() == 365


def test_one_cell_and_location_calculations_agree():
    data = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims="time",
        coords={"time": pd.date_range("2000-01-01", periods=5)},
        attrs={"units": "mm/day"},
    )
    periods = (Period("test", "2000-01-01", "2000-01-06"),)
    cell = absolute_indices(
        data.expand_dims(latitude=[1], longitude=[2]), periods, variable="precip"
    )
    location = absolute_indices(
        data.expand_dims(location=["gauge"]), periods, variable="precip"
    )
    for name in cell.data_vars:
        np.testing.assert_equal(
            cell[name].values.squeeze(), location[name].values.squeeze()
        )


def test_all_dry_sdii_is_undefined_but_counts_are_zero():
    data = xr.DataArray(
        np.zeros(5),
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=5)},
        attrs={"units": "mm/day"},
    )
    result = absolute_indices(
        data, (Period("test", "2001-01-01", "2001-01-06"),), variable="precip"
    )
    assert result.wet_days.item() == 0
    assert result.prcptot.item() == 0
    assert np.isnan(result.sdii.item())


def test_mean_temperature_counts_use_strict_boundaries_and_observed_days():
    data = xr.DataArray(
        [-1.0, 0.0, 25.0, 26.0, np.nan],
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=5)},
        attrs={"units": "degC"},
    )
    periods = (Period("test", "2001-01-01", "2001-01-06"),)
    result = absolute_indices(data, periods, variable="temp")
    assert result.mean_temp_below_zero_days.item() == 1
    assert result.mean_temp_above_threshold_days.item() == 1
    assert result.valid_days.item() == 4
    assert result.mean_temp_above_threshold_days.attrs["threshold_degc"] == 25
    changed = absolute_indices(
        data, periods, variable="temp", mean_temperature_threshold_degc=26
    )
    assert changed.mean_temp_above_threshold_days.item() == 0
    refused = absolute_indices(
        data, periods, variable="temp", max_missing_days_period=0
    )
    assert np.isnan(refused.mean_temp_below_zero_days.item())
    assert np.isnan(refused.mean_temp_above_threshold_days.item())


def test_mean_temperature_counts_need_no_percentile_reference_or_extrema():
    from blueearth_cst.indicators.climate.evaluate import evaluate
    from blueearth_cst.indicators.climate.reference import fit_reference

    data = xr.Dataset(
        {
            "temp": xr.DataArray(
                [-2.0, 0.0, 20.0, 25.0, 27.0],
                dims="time",
                coords={"time": pd.date_range("2003-01-01", periods=5)},
                attrs={"units": "degC"},
            )
        }
    )
    reference = fit_reference(data, 2000, 2001)  # Entire reference is missing.
    result, events = evaluate(
        data, reference, (Period("test", "2003-01-01", "2003-01-06"),)
    )
    assert result.mean_temp_below_zero_days.item() == 1
    assert result.mean_temp_above_threshold_days.item() == 1
    assert "txx" not in result
    assert events.empty


def test_water_year_periods_are_start_labelled():
    (period,) = annual_periods(2000, 2000, 10)
    assert period.label == "2000"
    assert period.start == pd.Timestamp("2000-10-01")
    assert period.stop == pd.Timestamp("2001-10-01")


def test_season_preserves_cross_year_membership_and_leap_day():
    (period,) = seasonal_periods(1999, 1999, name="DJF", months=(12, 1, 2))
    assert period.label == "DJF:1999"
    assert period.start == pd.Timestamp("1999-12-01")
    assert period.stop == pd.Timestamp("2000-03-01")
    assert (period.stop - period.start).days == 91


@pytest.mark.parametrize("months", [(), (1, 3), (12, 12), (True,), (13,)])
def test_disconnected_or_invalid_seasons_are_rejected(months):
    with pytest.raises(ValueError, match="months"):
        seasonal_periods(2000, 2001, name="season", months=months)
