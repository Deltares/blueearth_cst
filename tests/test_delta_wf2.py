"""Delta collections sourced from WF2 gridded changes (delta-change/3)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from blueearth_cst.experiment.delta_scenarios import (
    generate_collection,
    gridded_factors,
    read_delta_collection,
    validate_settings,
)
from blueearth_cst.projections import series_identity, spatial_diagnostics
from blueearth_cst.projections.get_change_climate_proj import analysis_windows
from blueearth_cst.projections.variable_spec import VariableSpec
from tests.test_delta_scenarios import historical, settings

SPEC = {
    "temp": VariableSpec("temp", "temp", "level", "degC", "absolute"),
    "precip": VariableSpec("precip", "precip", "rate", "mm/day", "relative", 0.1),
}
REFERENCE, FUTURE = (1985, 2004), (2040, 2059)


def _raw(path, start, years, *, warming, wetting, dry_cell=False):
    """Monthly raw grid on the historical fixture's cells, varying by year/month."""
    import hydromt  # noqa: F401

    time = pd.date_range(start, periods=12 * years, freq="MS")
    rng = np.random.default_rng(len(start) + years)
    month = time.month.values[:, None, None]
    temp = 20 + month / 2 + rng.normal(0, 0.5, (time.size, 2, 2)) + warming
    precip = (1 + month / 4 + rng.gamma(2, 0.2, (time.size, 2, 2))) * wetting
    if dry_cell:
        precip[:, 1, 1] = 0.05
    ds = xr.Dataset(
        {
            "temp": (("time", "lat", "lon"), temp, {"units": "degC"}),
            "precip": (("time", "lat", "lon"), precip, {"units": "mm/day"}),
        },
        coords={"time": time, "lat": [1.0, 0.0], "lon": [0.0, 1.0]},
        attrs={"cst_calendar": "standard"},
    )
    ds.raster.set_crs(4326)
    ds.to_netcdf(path)
    return ds


def wf2_project(tmp_path, monkeypatch, *, dry_cell=False):
    """A project root holding WF2 products for one member and one window."""
    monkeypatch.setattr(series_identity, "assert_raw_identity", lambda *a: None)
    monkeypatch.setattr(series_identity, "assert_raw_coverage", lambda *a: None)
    project = tmp_path / "project"
    ensemble = project / "data/climate/projections/cmip6"
    raw = tmp_path / "raw"
    raw.mkdir()
    hist = _raw(
        raw / "hist.nc", "1980-01-01", 35, warming=0, wetting=1, dry_cell=dry_cell
    )
    fut = _raw(
        raw / "fut.nc", "2015-01-01", 86, warming=2, wetting=1.2, dry_cell=dry_cell
    )
    point = {
        "model": "A",
        "scenario": "ssp245",
        "member": "r1",
        "raw_path_hist": str(raw / "hist.nc"),
        "raw_path_fut": str(raw / "fut.nc"),
        "raw_digest_hist": "h",
        "raw_digest_fut": "f",
    }
    windows = list(
        analysis_windows({"near": FUTURE}, reference_window=REFERENCE).values()
    )
    spatial_diagnostics.prepare_spatial_changes(
        [point],
        windows,
        ensemble / "analysis/gridded_changes",
        variable_spec=SPEC,
        min_reference={"precip": 0.1},
    )
    record = ensemble / "_engine/analysis_record.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"water_year_start": "Jan", "variable_spec": {}}))
    return project, hist, fut


def wf2_settings():
    base = settings()
    return {
        "schema_version": "delta-change/3",
        "historical_path": "historical.nc",
        "source_like": "era5",
        "reader": base["reader"],
        "simulation_window": base["simulation_window"],
        "unit_id_capacity": 100,
        "projections": {"ensemble": "cmip6", "window_id": "2040-2059"},
    }


def _write_inputs(tmp_path, config=None):
    import hydromt  # noqa: F401

    baseline = historical().rename({"y": "latitude", "x": "longitude"})
    baseline.raster.set_crs(4326)
    baseline.to_netcdf(tmp_path / "historical.nc")
    path = tmp_path / "delta.yml"
    path.write_text(yaml.safe_dump(config or wf2_settings()), encoding="utf-8")
    return path, baseline


def test_wf2_factors_equal_the_file_based_grid_path(tmp_path, monkeypatch):
    project, hist, fut = wf2_project(tmp_path, monkeypatch)
    config, baseline = _write_inputs(tmp_path)
    manifest = generate_collection(config, project)
    marker = read_delta_collection(manifest)
    intent = json.loads(manifest.with_name("collection_intent.json").read_text())
    resolved = intent["settings"]
    assert resolved["reference_period"] == {"start": 1985, "end": 2004}
    assert resolved["future_period"] == {"start": 2040, "end": 2059}
    assert [m["changes_path"] for m in resolved["members"]] == ["inputs/changes_1.nc"]
    assert resolved["wf2"]["unavailable"] == []
    assert intent["factor_policy"]["precip"]["min_denominator"] == 0.1
    assert len(marker["series"]) == 3
    # The same raw member through the existing delta-change/2 grid arithmetic.
    v2 = {
        **settings(),
        "schema_version": "delta-change/2",
        "factor_layout": "grid",
        "reference_period": {"start": REFERENCE[0], "end": REFERENCE[1]},
        "future_period": {"start": FUTURE[0], "end": FUTURE[1]},
    }
    rename = {"lat": "latitude", "lon": "longitude"}
    offsets, ratios = gridded_factors(
        hist.rename(rename),
        fut.rename(rename),
        baseline,
        v2,
        {"min_denominator": 0.1, "units": "mm/day"},
    )
    factors = intent["factors"][0]
    np.testing.assert_allclose(factors["temp_offset_degC"], offsets, rtol=1e-12)
    np.testing.assert_allclose(factors["precip_ratio"], ratios, rtol=1e-12)
    assert generate_collection(config, project) == manifest


def test_wf2_dry_reference_cell_is_refused_with_months(tmp_path, monkeypatch):
    project, *_ = wf2_project(tmp_path, monkeypatch, dry_cell=True)
    config, _ = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match=r"below 0.1 mm/day.*months 01, 02"):
        generate_collection(config, project)


def test_wf2_window_and_product_errors(tmp_path, monkeypatch):
    config, _ = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="WF2 product missing"):
        generate_collection(config, tmp_path / "empty")
    project, *_ = wf2_project(tmp_path, monkeypatch)
    other = {
        **wf2_settings(),
        "projections": {"ensemble": "cmip6", "window_id": "2070-2099"},
    }
    config, _ = _write_inputs(tmp_path, other)
    with pytest.raises(ValueError, match="available: \\['2040-2059'\\]"):
        generate_collection(config, project)


@pytest.mark.parametrize(
    "change",
    [
        {"members": []},
        {"pet": {"mode": "projection_ratio", "method": "makkink"}},
        {"projections": {"ensemble": "cmip6"}},
        {"projections": {"ensemble": "a/b", "window_id": "2040-2059"}},
    ],
)
def test_wf2_settings_are_closed(change):
    with pytest.raises(ValueError):
        validate_settings({**wf2_settings(), **change})
