"""Delta collections sourced from WF2 gridded changes (delta-change/3)."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from blueearth_cst.experiment.delta_scenarios import (
    delta_groups,
    generate_collection,
    gridded_factors,
    read_delta_collection,
    validate_settings,
)
from blueearth_cst.projections import series_identity, spatial_diagnostics
from blueearth_cst.projections.get_change_climate_proj import analysis_windows
from blueearth_cst.projections.prepare_projection_analysis import distinct_windows
from blueearth_cst.projections.variable_spec import VariableSpec
from tests.test_delta_scenarios import settings

SPEC = {
    "temp": VariableSpec("temp", "temp", "level", "degC", "absolute"),
    "precip": VariableSpec("precip", "precip", "rate", "mm/day", "relative", 0.1),
}
REFERENCE, FUTURE, FAR = (1985, 2004), (2040, 2059), (2070, 2089)


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
    by_name = analysis_windows({"near": FUTURE, "far": FAR}, reference_window=REFERENCE)
    windows = list(by_name.values())
    spatial_diagnostics.prepare_spatial_changes(
        [point],
        windows,
        ensemble / "analysis/gridded_changes",
        variable_spec=SPEC,
        min_reference={"precip": 0.1},
    )
    record = ensemble / "_engine/analysis_record.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "water_year_start": "Jan",
                "variable_spec": {},
                "windows": distinct_windows(by_name),
            }
        )
    )
    return project, hist, fut


def wf2_settings(**changes):
    return {
        "schema_version": "delta-change/3",
        "projections": {"ensemble": "cmip6"},
        **changes,
    }


STORE = "data/climate/historical/era5_1980-2010/extract_historical.nc"
ERA5 = ("temp", "temp_min", "temp_max", "precip", "press_msl", "kin", "kout")


def era5_historical(start="1984-01-01", end="2005-12-31"):
    """Daily 2x2 forcing with every supported ERA5 variable (design 3.3)."""
    import hydromt  # noqa: F401

    times = pd.date_range(start, end)
    data = np.full((len(times), 2, 2), 10.0, dtype="float32")
    data[:, 0, 0] = np.nan
    ds = xr.Dataset(
        {name: (("time", "latitude", "longitude"), data.copy()) for name in ERA5},
        coords={"time": times, "latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    ds.raster.set_crs(4326)
    return ds


def _write_inputs(tmp_path, project, config=None):
    baseline = era5_historical()
    path = project / STORE
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_netcdf(path)
    config_path = tmp_path / "delta.yml"
    config_path.write_text(yaml.safe_dump(config or wf2_settings()), encoding="utf-8")
    return config_path, baseline, {"path": path, "source": "era5"}


def _intent(manifest):
    return json.loads(manifest.with_name("collection_intent.json").read_text())


def test_wf2_factors_equal_the_file_based_grid_path(tmp_path, monkeypatch):
    project, hist, fut = wf2_project(tmp_path, monkeypatch)
    near = wf2_settings(projections={"ensemble": "cmip6", "future_windows": ["near"]})
    config, baseline, binding = _write_inputs(tmp_path, project, near)
    with pytest.raises(ValueError, match="derives its historical record"):
        generate_collection(config, project)
    manifest = generate_collection(config, project, historical=binding)
    marker = read_delta_collection(manifest)
    intent = _intent(manifest)
    resolved = intent["settings"]
    folder = manifest.parent.parent
    assert folder == project / "scenarios" / "delta-cmip6-near"
    assert resolved["historical_path"] == STORE
    assert resolved["unit_id_capacity"] == 3
    assert resolved["historical_period"]["effective"] == ["1985-01-01", "2004-12-31"]
    (window,) = resolved["windows"]
    assert window["names"] == ["near"]
    assert window["future"]["labels"] == [2040, 2059]
    assert [m["changes_path"] for m in resolved["members"]] == [
        "data/climate/projections/cmip6/analysis/gridded_changes/member_0001.nc"
    ]
    assert intent["factor_policy"]["precip"]["min_denominator"] == 0.1
    assert len(marker["series"]) == 3
    # Series cover exactly the WF2 reference dates.
    with xr.open_dataset(folder / "series/run_1.nc") as series:
        assert str(series.time.values[0])[:10] == "1985-01-01"
        assert str(series.time.values[-1])[:10] == "2004-12-31"
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
    with xr.open_dataset(folder / "delta/factors/run_3.nc") as retained:
        np.testing.assert_allclose(retained["temp_offset"], offsets, rtol=1e-12)
        np.testing.assert_allclose(retained["precip_ratio"], ratios, rtol=1e-12)
    assert generate_collection(config, project, historical=binding) == manifest
    # A rebuilt store (changed in-place input) leaves the set readable.
    member = project / resolved["members"][0]["changes_path"]
    member.write_bytes(member.read_bytes() + b"extra")
    read_delta_collection(manifest)


def test_all_windows_by_default_run_per_member_and_window(tmp_path, monkeypatch):
    project, *_ = wf2_project(tmp_path, monkeypatch)
    config, _, binding = _write_inputs(tmp_path, project)
    manifest = generate_collection(config, project, historical=binding)
    folder = manifest.parent.parent
    assert folder.name == "delta-cmip6-near-far"
    lines = (folder / "scenario_run_lookup.csv").read_text().splitlines()
    header = lines[0].split(",")
    assert header[-1] == "window_id"
    rows = [dict(zip(header, line.split(","), strict=True)) for line in lines[1:]]
    assert [row["window_id"] for row in rows] == ["", "", "2040-2059", "2070-2089"]
    groups, references = delta_groups(rows)
    assert len(groups) == 4 and references == ("run_1",)
    # A renamed rerun of identical content reuses the set.
    renamed = tmp_path / "renamed.yml"
    renamed.write_text(yaml.safe_dump(wf2_settings(name="mine")), encoding="utf-8")
    assert generate_collection(renamed, project, historical=binding) == manifest


def test_historical_period_override_and_coverage(tmp_path, monkeypatch):
    project, *_ = wf2_project(tmp_path, monkeypatch)
    period = wf2_settings(historical_override={"period": {"start": 1990, "end": 1999}})
    config, _, binding = _write_inputs(tmp_path, project, period)
    manifest = generate_collection(config, project, historical=binding)
    resolved = _intent(manifest)["settings"]
    assert resolved["historical_period"]["effective"] == ["1990-01-01", "1999-12-31"]
    assert resolved["historical_period"]["override"] is True
    era5_historical("1990-01-01", "2000-12-31").to_netcdf(binding["path"])
    plain = tmp_path / "plain.yml"
    plain.write_text(yaml.safe_dump(wf2_settings()), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover the required period 1985"):
        generate_collection(plain, project, historical=binding)


def test_wf2_dry_reference_cell_is_refused_with_months(tmp_path, monkeypatch):
    project, *_ = wf2_project(tmp_path, monkeypatch, dry_cell=True)
    config, _, binding = _write_inputs(tmp_path, project)
    with pytest.raises(ValueError, match="below 0.1 mm/day.*months 01, 02"):
        generate_collection(config, project, historical=binding)


def test_wf2_window_and_product_errors(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    config, _, binding = _write_inputs(tmp_path, empty)
    with pytest.raises(ValueError, match="WF2 product missing"):
        generate_collection(config, empty, historical=binding)
    project, *_ = wf2_project(tmp_path, monkeypatch)
    unknown = wf2_settings(projections={"ensemble": "cmip6", "future_windows": ["mid"]})
    config, _, binding = _write_inputs(tmp_path, project, unknown)
    with pytest.raises(ValueError, match="unknown WF2 future window"):
        generate_collection(config, project, historical=binding)


OVERRIDE_FILE = {"path": "x.nc", "source": "era5", "units_note": "note"}


@pytest.mark.parametrize(
    "change",
    [
        {"members": []},
        {"pet": {"mode": "projection_ratio", "method": "makkink"}},
        {"historical_path": "x.nc"},
        {"projections": {"ensemble": "cmip6", "window_id": "2040-2059"}},
        {"projections": {"ensemble": "a/b"}},
        {"projections": {"ensemble": "cmip6", "future_windows": []}},
        {"projections": {"ensemble": "cmip6", "future_windows": ["a", "a"]}},
        {"historical_override": {"path": "x.nc"}},
        {"historical_override": {**OVERRIDE_FILE, "units": {"temp": "K"}}},
        {"name": "_engine"},
        {"unit_id_capacity": 2},
    ],
)
def test_wf2_settings_are_closed(change):
    with pytest.raises(ValueError):
        validate_settings({**wf2_settings(), **change})


def test_override_file_requires_the_full_source_units():
    units = dict(zip(ERA5, ("degC",) * 3 + ("mm/day", "hPa", "W/m2", "W/m2")))
    validate_settings(
        wf2_settings(historical_override={**OVERRIDE_FILE, "units": units})
    )


def test_derived_historical_uses_the_stochastic_store_rule(tmp_path):
    from blueearth_cst.experiment.delta_generation import historical_binding

    config = {
        "project": {"project_dir": str(tmp_path), "catalog": "catalog.yml"},
        "climate": {
            "selected": "era5",
            "window": {"start": 1980, "end": 2010},
        },
        "basin": {"region": "{'subbasin': [10.0, 0.5]}"},
    }
    binding = historical_binding(config, wf2_settings())
    assert binding["source"] == "era5"
    assert binding["path"].name == "extract_historical.nc"
    assert binding["path"].parent.parent == tmp_path / "data/climate/historical"
    override = {**OVERRIDE_FILE, "path": str(tmp_path / "own.nc")}
    assert historical_binding(config, wf2_settings(historical_override=override)) == {
        "path": tmp_path / "own.nc",
        "source": "era5",
    }


def test_default_name_sanitises_horizon_names():
    from blueearth_cst.experiment.delta_scenarios import default_delta_name

    windows = [
        {"names": ["mid century"], "window_id": "2040-2069"},
        {"names": ["—"], "window_id": "2070-2099"},
    ]
    resolved = {"projections": {"ensemble": "cmip6"}, "windows": windows}
    assert default_delta_name(resolved) == "delta-cmip6-mid-century-2070-2099"
