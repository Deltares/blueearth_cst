"""Independent reference/bootstrap arithmetic and calendar boundary checks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.indicators.climate.evaluate import evaluate
from blueearth_cst.indicators.climate.periods import annual_periods
from blueearth_cst.indicators.climate.reference import (
    calendar_day,
    exceedance_probability,
    fit_reference,
    temperature_quantile,
)


def temperature_data():
    dates = pd.date_range("2000-01-01", "2002-12-31")
    values = np.asarray(dates.year - 2000, dtype=float) * 10
    return xr.Dataset(
        {
            "temp_max": xr.DataArray(
                values, dims="time", coords={"time": dates}, attrs={"units": "degC"}
            )
        }
    )


def test_full_overlap_bootstrap_averages_exceedances_not_thresholds():
    data = temperature_data()
    reference = fit_reference(data, 2000, 2002)
    dates = pd.DatetimeIndex(data.time.values)
    probability = exceedance_probability(
        data.temp_max.values, dates, reference, "temp_max"
    )
    assert (probability[dates.year == 2000] == 0).all()
    assert (probability[dates.year == 2001] == 0).all()
    assert (probability[dates.year == 2002] == 1).all()
    # Full-reference q90 is 20: disabling bootstrap would instead give zero.
    assert (reference.temp_max_p90 == 20).all()
    subset = dates.year == 2002
    np.testing.assert_equal(
        exceedance_probability(
            data.temp_max.values[subset], dates[subset], reference, "temp_max"
        ),
        probability[subset],
    )


def test_out_of_reference_keeps_frozen_threshold():
    data = temperature_data()
    reference = fit_reference(data, 2000, 2002)
    dates = pd.date_range("2003-01-01", periods=3)
    probability = exceedance_probability(
        np.array([19.0, 20.0, 21.0]), dates, reference, "temp_max"
    )
    np.testing.assert_equal(probability, [0, 0, 1])


def test_fractional_bootstrap_and_failed_replicate():
    data = temperature_data()
    data.temp_max[:] = 0
    data.temp_max.loc[{"time": "2002-07-01"}] = 30
    reference = fit_reference(data, 2000, 2002)
    dates = pd.date_range("2001-07-01", periods=1)
    # Fifteen samples: type-8 q90 is 4 with one hot sample, 30 with two.
    # A value of 15 exceeds one replicate, while it does not exceed their mean (17).
    assert (
        exceedance_probability(np.array([15.0]), dates, reference, "temp_max")[0] == 0.5
    )
    data.temp_max.loc[{"time": slice("2000-01-01", "2000-12-31")}] = np.nan
    reference = fit_reference(data, 2000, 2002, minimum_fraction=0.6)
    assert np.isfinite(reference.temp_max_p90).all()
    assert np.isnan(
        exceedance_probability(np.array([15.0]), dates, reference, "temp_max")[0]
    )


def test_calendar_dates_preserve_february_and_circular_windows():
    dates = pd.to_datetime(
        ["2000-02-28", "2000-02-29", "2000-03-01", "2001-03-01", "2000-12-31"]
    )
    np.testing.assert_equal(calendar_day(dates), [58, 58, 59, 59, 364])
    values = np.zeros((2, 365))
    values[:, -2:] = 10
    quantile, counts, years = temperature_quantile(values, 1.0)
    assert quantile[0] == 10  # January 1's window includes Dec 30/31 of each year.
    assert counts[0] == 10
    assert years[0] == 2
    assert quantile[4] == 0


def test_reference_support_failure_is_not_averaged_away():
    data = temperature_data()
    data.temp_max.loc[{"time": slice("2001-01-01", "2001-12-31")}] = np.nan
    reference = fit_reference(data, 2000, 2002)
    assert reference.temp_max_p90.isnull().all()
    assert (reference.temp_max_years == 2).all()


def test_wet_day_type8_threshold_and_strict_exceedance():
    dates = pd.date_range("2000-01-01", "2001-12-31")
    values = np.zeros(len(dates))
    values[:20] = np.arange(1, 21)
    data = xr.Dataset(
        {
            "precip": xr.DataArray(
                values, dims="time", coords={"time": dates}, attrs={"units": "mm/day"}
            )
        }
    )
    reference = fit_reference(data, 2000, 2001)
    # Type 8: h=(20+1/3)*.95+1/3 = 19.65, interpolating values 19 and 20.
    assert reference.precip_p95.item() == pytest.approx(19.65)
    output, events = evaluate(data, reference, annual_periods(2000, 2001))
    assert output.r95p_amount.sel(period="2000").item() == 20
    assert output.r95p_fraction.sel(period="2000").item() == pytest.approx(
        100 * 20 / 210
    )
    assert np.isnan(output.r95p_fraction.sel(period="2001").item())
    assert not events.empty


def test_bootstrap_percentages_and_fixed_event_products_are_distinct():
    data = temperature_data()
    reference = fit_reference(data, 2000, 2002)
    output, events = evaluate(data, reference, annual_periods(2000, 2002))
    np.testing.assert_equal(output.tx90p.values, [0, 0, 100])
    np.testing.assert_equal(output.wsdi.values, [0, 0, 0])
    assert events.empty


def test_continuous_warm_spell_reconciles_across_calendar_year():
    reference = fit_reference(temperature_data(), 2000, 2002)
    dates = pd.date_range("2003-01-01", "2004-12-31")
    values = np.zeros(len(dates))
    values[(dates >= "2003-12-29") & (dates <= "2004-01-03")] = 25
    data = xr.Dataset(
        {
            "temp_max": xr.DataArray(
                values, dims="time", coords={"time": dates}, attrs={"units": "degC"}
            )
        }
    )
    result, events = evaluate(data, reference, annual_periods(2003, 2004))
    np.testing.assert_equal(result.wsdi.values, [3, 3])
    np.testing.assert_equal(result.heatwave_days.values, [3, 3])
    np.testing.assert_equal(result.heatwave_count.values, [1, 0])
    np.testing.assert_equal(result.longest_heatwave.values, [3, 3])
    assert events.loc[events.event_type == "heatwave", "duration"].tolist() == [6]


def test_short_censored_warm_fragment_cannot_establish_zero_event_count():
    reference = fit_reference(temperature_data(), 2000, 2002)
    dates = pd.date_range("2003-01-01", "2003-12-31")
    values = np.zeros(len(dates))
    values[:2] = 25
    data = xr.Dataset(
        {
            "temp_max": xr.DataArray(
                values, dims="time", coords={"time": dates}, attrs={"units": "degC"}
            )
        }
    )
    result, events = evaluate(data, reference, annual_periods(2003, 2003))
    assert result.heatwave_days.item() == 0
    assert np.isnan(result.heatwave_count.item())
    assert events.empty
    assert "event_id" in events.columns


def test_bootstrap_spatial_blocks_preserve_every_location():
    data = (
        temperature_data()
        .expand_dims(location=np.arange(33))
        .transpose("time", "location")
    )
    reference = fit_reference(data, 2000, 2002)
    dates = pd.DatetimeIndex(data.time.values)
    probability = exceedance_probability(
        data.temp_max.values, dates, reference, "temp_max"
    )
    expected = np.broadcast_to((dates.year == 2002)[:, None], (len(dates), 33))
    np.testing.assert_equal(probability, expected)


def test_grid_mean_of_indices_differs_from_index_of_daily_mean():
    dates = pd.date_range("2000-01-01", "2001-12-31")
    values = np.zeros((len(dates), 2))
    values[0] = [20, 0]
    values[1] = [0, 20]
    data = xr.Dataset(
        {
            "precip": xr.DataArray(
                values,
                dims=("time", "location"),
                coords={"time": dates, "location": [0, 1]},
                attrs={"units": "mm/day"},
            )
        }
    )
    reference = fit_reference(data, 2000, 2001)
    grid, _ = evaluate(data, reference, annual_periods(2000, 2001))
    basin_data = data.mean("location", keep_attrs=True)
    basin, _ = evaluate(
        basin_data, fit_reference(basin_data, 2000, 2001), annual_periods(2000, 2001)
    )
    assert grid.rx1day.sel(period="2000").mean().item() == 20
    assert basin.rx1day.sel(period="2000").item() == 10
