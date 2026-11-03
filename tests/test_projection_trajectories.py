"""Own-member reference and unsmoothed historical/scenario continuity."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import projection_analysis as analysis
from blueearth_cst.projections import projection_trajectories as trajectories


def _series(start, end, temp, precip):
    time = pd.date_range(start, end, freq="MS")
    return xr.Dataset(
        {
            "temp": ("time", np.full(len(time), temp), {"units": "degC"}),
            "precip": ("time", np.full(len(time), precip), {"units": "mm/day"}),
        },
        coords={"time": time},
        attrs={"cst_calendar": "360_day"},
    )


def test_full_record_own_reference_and_unsmoothed_model_range():
    hist = _series("1950-01-01", "2014-12-01", 10.0, 2.0)
    future = _series("2015-01-01", "2100-12-01", 12.0, 3.0)
    identity = {"model": "TEST/A", "scenario": "ssp245", "member": "r1"}
    first = trajectories.member_anomalies(
        hist, future, identity=identity, reference_window=(1985, 2014)
    )
    assert (first.year.min(), first.year.max()) == (1950, 2100)
    temp = first.loc[first.variable == "temp"].set_index("year")
    assert temp.loc[2014, "anomaly"] == 0
    assert temp.loc[2015, "anomaly"] == 2
    second = trajectories.member_anomalies(
        hist.assign(temp=hist.temp + 10),
        future,
        identity=dict(identity, member="r2"),
        reference_window=(1985, 2014),
    )
    assert (
        second.loc[
            (second.variable == "temp") & (second.year == 2015), "anomaly"
        ].item()
        == -8
    )
    models = trajectories.model_trajectories(pd.concat([first, second]))
    assert (
        models.loc[
            (models.variable == "temp") & (models.year == 2015), "anomaly"
        ].item()
        == -3
    )
    envelope = trajectories.trajectory_range(models)
    assert (
        envelope.loc[
            (envelope.variable == "temp") & (envelope.year == 2015), "mean"
        ].item()
        == -3
    )
    october = trajectories.member_anomalies(
        hist,
        future,
        identity=identity,
        reference_window=(1985, 2014),
        water_year_start="Oct",
    )
    assert (october.year.min(), october.year.max()) == (1950, 2099)
    missing = trajectories.member_anomalies(
        hist, future, identity=identity, reference_window=(1940, 1969)
    )
    assert missing.anomaly.isna().all()
    assert missing.annual_rate.notna().all()
    assert missing.status.eq("unavailable_reference").all()


def test_product_predicates_preserve_partial_ensemble_and_window_identity():
    hist = _series("1985-01-01", "2014-12-01", 10.0, 2.0)
    future = _series("2036-01-01", "2065-12-01", 12.0, 3.0)
    # Positive nonlinear year variability makes both ratio statistics eligible.
    noise = np.repeat(np.sin(np.arange(30)), 12)
    hist = hist + xr.DataArray(noise, dims="time", coords={"time": hist.time})
    future = future + xr.DataArray(2 * noise, dims="time", coords={"time": future.time})
    hist = hist.expand_dims(
        model=["TEST/A"], scenario=["historical"], member=["r1"], clim_project=["cmip6"]
    )
    future = future.expand_dims(
        model=["TEST/A"], scenario=["ssp245"], member=["r1"], clim_project=["cmip6"]
    )
    changes = analysis.analysis_change_table(
        hist,
        future,
        identity={"model": "TEST/A", "scenario": "ssp245", "member": "r1"},
        window_id="2036-2065",
        reference_window=(1985, 2014),
        future_window=(2036, 2065),
    )
    ratios = analysis.member_ratios(changes)
    bands = analysis.member_null_bands(ratios)
    assert len(bands) == len(ratios)
    eligible = bands.loc[bands.variability_eligible & bands.ratio.gt(0)]
    assert eligible.null_status.eq("ok").all()
    assert eligible.null_n_valid.eq(20000).all()
    tables = {
        "coverage": pd.DataFrame({"available": [True]}),
        "models": analysis.model_changes(changes),
        "climatology": analysis.climatology(changes),
        "member_ratios": ratios,
        "model_ratios": analysis.model_ratios(ratios, include_null=False),
    }
    records = [{"window_id": "2036-2065"}, {"window_id": "2036-2065"}]
    available = analysis.product_availability(tables, windows=records)
    assert len(available) == 20  # fourteen window products and six full_period products
    window = available.loc[available.folder == "period_2036_2065"].set_index("filename")
    assert window.loc[
        "cmip6_precip_temp_annual_change_cloud_basin_avg.png", "available"
    ]
    assert not window.loc[
        "cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png", "available"
    ]
    assert not window.loc[
        "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png", "available"
    ]
    assert window.loc["cmip6_temp_monthly_clim_basin_avg.png", "available"]
    bad = dict(tables, models=pd.DataFrame({"model": ["bad"]}))
    with pytest.raises(ValueError, match="missing availability columns"):
        analysis.product_availability(bad)
