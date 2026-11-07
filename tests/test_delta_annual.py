"""Annual diagnostics must aggregate climate amounts before member summaries."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.experiment.delta_annual import annual_climatology, summarize_members


def field(values, units="mm/day", year=2000):
    return xr.DataArray(
        values,
        dims="time",
        coords={"time": pd.date_range(f"{year}-01-01", periods=12, freq="MS")},
        attrs={"units": units},
    )


def test_temperature_uses_days_and_flux_uses_annual_amounts():
    temperature = field(np.arange(12.0), "degC")
    days = temperature.time.dt.days_in_month.values
    period = {"start": 2000, "end": 2000}
    assert float(
        annual_climatology(temperature, period, temperature=True)
    ) == pytest.approx(np.dot(np.arange(12), days) / 366)
    ref = field([100.0] + [1.0] * 11)
    fut = field([100.0] + [2.0] * 11)
    annual_ratio = annual_climatology(
        fut, period, temperature=False
    ) / annual_climatology(ref, period, temperature=False)
    assert float(annual_ratio) == pytest.approx(
        np.dot(fut.values, days) / np.dot(ref.values, days)
    )
    assert float(annual_ratio) != pytest.approx(float((fut / ref).mean()))
    assert (
        float(
            annual_climatology(
                field(np.ones(12), "mm/month"), period, temperature=False
            )
        )
        == 12
    )


def test_annual_coverage_missingness_and_calendar():
    period = {"start": 2000, "end": 2000}
    assert (
        float(annual_climatology(field(np.ones(12)), period, temperature=False)) == 366
    )
    assert (
        float(
            annual_climatology(
                field(np.ones(12), year=2001),
                {"start": 2001, "end": 2001},
                temperature=False,
            )
        )
        == 365
    )
    incomplete = field(np.ones(12)).isel(time=slice(1, None))
    with pytest.raises(ValueError, match="ordered sample"):
        annual_climatology(incomplete, period, temperature=False)
    missing = field(np.ones(12))
    missing[1] = np.nan
    assert np.isnan(annual_climatology(missing, period, temperature=False))


def test_member_summary_is_per_cell_and_does_not_skip_missing_members():
    values = [np.array([v, np.nan]) for v in (0, 2, 100)]
    assert summarize_members(values)[0] == 2
    assert summarize_members(values, "mean")[0] == 34
    assert np.isnan(summarize_members(values)[1])
    with pytest.raises(ValueError, match="median or mean"):
        summarize_members(values, "maximum")


def test_legacy_scalar_scenario_labels_do_not_block_paired_changes(tmp_path):
    from blueearth_cst.experiment.delta_annual import annual_member_maps

    for scenario, value in (("historical", 1.0), ("ssp245", 2.0)):
        dataset = xr.Dataset(
            {
                "temp": field(np.full(12, value), "degC"),
                "precip": field(np.full(12, value)),
            }
        ).expand_dims(scenario=[scenario])
        dataset.to_netcdf(tmp_path / f"{scenario}.nc")
    historical = xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), np.ones((1, 2, 2)))
            for name in ("temp", "precip")
        },
        coords={
            "time": [pd.Timestamp("2000-01-01")],
            "latitude": [1, 0],
            "longitude": [0, 1],
        },
    )
    settings = {
        "reference_period": {"start": 2000, "end": 2000},
        "future_period": {"start": 2000, "end": 2000},
        "members": [
            {
                "model": "M",
                "scenario": "ssp245",
                "member": "r1",
                "reference_path": "historical.nc",
                "future_path": "ssp245.nc",
            }
        ],
    }
    results = annual_member_maps(tmp_path, settings, historical)
    np.testing.assert_array_equal(
        results["temp_offset_degc"][("M", "ssp245")][0], np.ones((2, 2))
    )
    np.testing.assert_array_equal(
        results["precip_change_percent"][("M", "ssp245")][0], np.full((2, 2), 100.0)
    )
