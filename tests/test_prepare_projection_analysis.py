"""Declared-input and distinct-window contracts for the analysis producer."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import prepare_projection_analysis as producer
from blueearth_cst.projections import projection_analysis as analysis
from blueearth_cst.projections import resolution, series_identity
from blueearth_cst.projections.get_change_climate_proj import analysis_windows


def test_skipped_product_messages_explain_and_group_windows():
    availability = pd.DataFrame(
        [
            {
                "filename": "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png",
                "window_id": window,
                "available": False,
                "reason": "no model with two eligible member ratios",
            }
            for window in ("2041-2060", "2081-2100")
        ]
        + [
            {
                "filename": "cmip6_ensemble_coverage_basin_avg.png",
                "window_id": "",
                "available": False,
                "reason": "no resolved members",
            },
            {
                "filename": "cmip6_temp_annual_overview_basin_avg.png",
                "window_id": "",
                "available": True,
                "reason": "",
            },
        ]
    )
    assert list(producer.skipped_product_messages(availability)) == [
        "Annual variability member-ratio boxplots skipped: "
        "not enough eligible ensemble members",
        "Ensemble coverage skipped: no resolved members",
    ]


def _series(path, start, count, scenario, components, region):
    years = np.arange(count) // 12
    dataset = xr.Dataset(
        {
            "temp": ("time", 10 + np.sin(years), {"units": "degC"}),
            "precip": ("time", 2 + np.cos(years) / 10, {"units": "mm/day"}),
        },
        coords={"time": pd.date_range(start, periods=count, freq="MS")},
        attrs={
            "cst_calendar": "360_day",
            "cst_schema_version": series_identity.SCHEMA_VERSION,
            "cst_series_digest": series_identity.series_digest(components, region),
        },
    ).expand_dims(
        model=["TEST/M1"], member=["r1"], scenario=[scenario], clim_project=["cmip6"]
    )
    dataset.to_netcdf(path)


def test_shared_windows_compute_once_with_identity_checks(tmp_path):
    region = "analysis-producer-fixture"
    hist, fut = tmp_path / "hist.nc", tmp_path / "fut.nc"
    hist_components, fut_components = {"series": "hist"}, {"series": "fut"}
    _series(hist, "1950-01-01", 65 * 12, "historical", hist_components, region)
    _series(fut, "2015-01-01", 86 * 12, "ssp245", fut_components, region)
    point = {
        "model": "TEST/M1",
        "scenario": "ssp245",
        "member": "r1",
        "series_path_hist": str(hist),
        "series_path": str(fut),
        "digest_components_hist": hist_components,
        "digest_components_fut": fut_components,
        "reference_series_key": "hist",
        "series_key": "fut",
    }
    windows = analysis_windows(
        {"far1": (2081, 2100), "far2": (2081, 2100)}, reference_window=(2000, 2014)
    )
    tables, record = producer.prepare_tables([point], windows, region_fp=region)
    # Exercise the same resolver -> explicit input set -> producer path as WF2.
    # A leftover CIESM cache must not rejoin either variable through a file glob.
    stale = tmp_path / "cmip6_THU_CIESM_ssp245_r1.nc"
    stale.write_bytes(b"excluded source cache: must never be opened")
    catalog = {
        resolution.entry_key("cmip6", "TEST/M1", experiment): {
            "placeholders": {"member": ["r1"]}
        }
        for experiment in ["historical", "ssp245"]
    }
    combinations = resolution.resolve(
        catalog,
        clim_project="cmip6",
        models=["TEST/M1", "THU/CIESM"],
        scenarios=["ssp245"],
        members=["r1"],
    )
    requested_points = {
        "TEST/M1": point,
        "THU/CIESM": {**point, "model": "THU/CIESM", "series_path": str(stale)},
    }
    filtered, _ = producer.prepare_tables(
        [requested_points[c.dataset] for c in combinations if c.resolved],
        windows,
        region_fp=region,
        combinations=[c._asdict() for c in combinations],
    )
    for name in tables.keys() - {"coverage"}:
        pd.testing.assert_frame_equal(tables[name], filtered[name])
    coverage = filtered["coverage"]
    assert coverage.loc[coverage.model == "THU/CIESM", "status"].item() == (
        "source_quality_excluded"
    )
    assert not coverage.loc[coverage.model == "THU/CIESM", "available"].item()
    assert set(filtered["member_anomalies"].variable) == {"precip", "temp"}
    assert set(filtered["member_anomalies"].model) == {"TEST/M1"}
    assert stale.read_bytes() == b"excluded source cache: must never be opened"
    assert len(record["windows"]) == 1
    assert record["windows"][0]["horizons"] == ["far1", "far2"]
    assert (
        record["windows"][0]["horizon_windows"]["far1"]["requested"]
        == record["windows"][0]["horizon_windows"]["far2"]["requested"]
    )
    keys = [
        "model",
        "scenario",
        "member",
        "window_id",
        "period",
        "variable",
        "statistic",
        "quantity",
    ]
    assert not tables["changes"].duplicated(keys).any()
    assert tables["changes"].window_id.unique().tolist() == ["2081-2100"]
    assert record["reference"] == (2000, 2014)
    annual = tables["changes"].loc[tables["changes"].period == "annual"]
    assert set(annual.n_reference) == {15}
    assert set(annual.n_future) == {20}
    assert set(annual.status) == {"ok"}
    seasonal = tables["changes"].loc[
        tables["changes"].period.isin(["DJF", "MAM", "JJA", "SON"])
    ]
    assert set(seasonal.status) == {"ok"}
    assert record["methods"]["season_min_years"] == {
        "2081-2100": {"reference": 14, "future": 18}
    }
    assert tables["member_anomalies"].year.min() == 1950
    assert tables["member_anomalies"].year.max() == 2100
    assert len(tables["availability"]) == 20
    assert (
        not tables["availability"]
        .loc[
            tables["availability"].filename
            == "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png",
            "available",
        ]
        .any()
    )
    assert record["methods"]["null_band"]["simulations"] == 20_000
    with pytest.raises(RuntimeError, match="digest"):
        producer.prepare_tables([point], windows, region_fp="different-region")


@pytest.mark.parametrize("reference_available", [True, False])
def test_missing_gcm_window_coverage_is_recorded_without_stopping_analysis(
    tmp_path, reference_available
):
    region = "coverage-fixture"
    hist, fut = tmp_path / "hist.nc", tmp_path / "fut.nc"
    historical_identity, future_identity = {"series": "hist"}, {"series": "fut"}
    _series(
        hist,
        "1950-01-01" if reference_available else "1900-01-01",
        65 * 12 if reference_available else 20 * 12,
        "historical",
        historical_identity,
        region,
    )
    _series(fut, "2020-01-01", 5 * 12, "ssp245", future_identity, region)
    point = {
        "model": "TEST/M1",
        "scenario": "ssp245",
        "member": "r1",
        "series_path_hist": str(hist),
        "series_path": str(fut),
        "digest_components_hist": historical_identity,
        "digest_components_fut": future_identity,
        "reference_series_key": "hist",
        "series_key": "fut",
    }
    windows = analysis_windows({"near": (2040, 2060)}, reference_window=(1985, 2004))
    tables, record = producer.prepare_tables([point], windows, region_fp=region)
    assert set(tables["changes"].status) == {"unavailable"}
    assert tables["changes"].reason.str.len().gt(0).all()
    assert record["unavailable_members"]
    expected = 20 if reference_available else 0
    assert tables["coverage"].n_reference_years.tolist() == [expected]


def test_undeclared_series_refused_before_output_mutation():
    sm = SimpleNamespace(
        params=SimpleNamespace(
            points=[{"series_path_hist": "hist.nc", "series_path": "fut.nc"}]
        ),
        input=SimpleNamespace(series_nc=["hist.nc", "fut.nc", "leftover.nc"]),
    )
    with pytest.raises(RuntimeError, match="series set differs"):
        producer.run(sm)


def test_display_bands_preserve_masks_and_equal_model_reference_cv():
    rows = []
    for model, cv, first in (("A", 0.1, 1985), ("A", 0.3, 1985), ("B", 0.8, 1988)):
        rows.append(
            {
                "model": model,
                "scenario": "ssp245",
                "window_id": "2036-2065",
                "variable": "precip",
                "statistic": "cv_detrended",
                "period": "annual",
                "reference_value": cv,
                "variability_eligible": True,
                "reference_years": json.dumps(list(range(first, 2015))),
                "future_years": json.dumps(list(range(2036, 2066))),
            }
        )
    bands = producer.display_null_bands(pd.DataFrame(rows))
    assert len(bands) == 2
    assert {len(band["reference_years"]) for band in bands} == {27, 30}
    assert all(band["reference_cv"] == pytest.approx(0.5) for band in bands)
    assert all(band["n_mask_strata"] == 2 for band in bands)
    assert all(band["n_valid"] + band["n_invalid"] == 20_000 for band in bands)


def test_model_ratio_summary_uses_log_middle_pair():
    common = dict.fromkeys(analysis.CELL, "test")
    models = pd.DataFrame(
        [
            {**common, "model": "A", "value": 0.5, "n_members": 1},
            {**common, "model": "B", "value": 2.0, "n_members": 1},
        ]
    )
    summary = producer._model_ratio_summary(models).iloc[0]
    assert summary["median"] == pytest.approx(1.0)
    assert summary["median"] != pytest.approx(1.25)
    assert (summary["min"], summary["max"]) == pytest.approx((0.5, 2.0))
    assert not summary.iqr_shown
