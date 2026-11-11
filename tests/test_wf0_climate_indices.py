"""Source qualification and compatible retained climate-index readers."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis.climate_indices import (
    load_publication,
    qualified_source_data,
)
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings
from tests.test_climate_index_table import exported as exported


def inputs(tmp_path):
    dates = pd.date_range("2000-01-01", "2002-12-31")
    rng = np.random.default_rng(30)
    shape = (len(dates), 2, 2)
    ds = xr.Dataset(
        coords={
            "time": dates,
            "latitude": [1.0, 2.0],
            "longitude": [3.0, 4.0],
            "spatial_ref": 0,
        }
    )
    for name, values, units in (
        ("precip", rng.gamma(1, 5, shape), "mm/day"),
        ("temp_max", rng.normal(25, 3, shape), "degC"),
        ("temp_min", rng.normal(15, 3, shape), "degC"),
        ("temp", rng.normal(20, 7, shape), "degC"),
    ):
        ds[name] = xr.DataArray(
            values, dims=("time", "latitude", "longitude"), attrs={"units": units}
        )
    climate = tmp_path / "climate.nc"
    ds.to_netcdf(climate)
    cells = tmp_path / "cells.csv"
    pd.DataFrame({"latitude": [1.0, 2.0], "longitude": [3.0, 4.0]}).to_csv(
        cells, index=False
    )
    settings = parse_settings(
        {"climate_indices": {"enabled": True, "seasons": {"DJF": [12, 1, 2]}}},
        {"start": 2000, "end": 2002},
        ["era5", "chirps"],
    )
    return climate, cells, settings


@pytest.mark.parametrize("source", ["era5", "chirps"])
def test_qualification_preserves_source_identity(tmp_path, source):
    climate, cells, settings = inputs(tmp_path)
    data, unavailable = qualified_source_data(climate, source)
    assert "precip" in data
    assert ("temp" in data) == (source == "era5")
    assert ("temp_max" in unavailable) == (source == "chirps")
    assert data.precip.attrs["source"] == source


def test_disabled_default_and_closed_settings():
    settings = parse_settings(None, {"start": 2000, "end": 2002}, ["era5"])
    assert settings["climate_indices"]["enabled"] is False
    with pytest.raises(ValueError, match="accepts"):
        parse_settings(
            {"climate_indices": {"typo": True}}, {"start": 2000, "end": 2002}, ["era5"]
        )


@pytest.mark.parametrize("value", [True, "25", float("nan"), float("inf")])
def test_mean_temperature_threshold_requires_finite_number(value):
    with pytest.raises(ValueError, match="finite number"):
        parse_settings(
            {"climate_indices": {"mean_temperature_threshold_degc": value}},
            {"start": 2000, "end": 2002},
            ["era5"],
        )


def test_manifest_cannot_escape_source_store(tmp_path):
    pointer = tmp_path / "_engine" / "climate_indices.json"
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({"manifest": "../foreign.json", "sha256": "unused"}))
    with pytest.raises(ValueError, match="escapes"):
        load_publication(pointer)


def test_new_inventory_is_opt_in_and_keeps_legacy_schema():
    from blueearth_cst.climate_analysis.diagnostic_outputs import (
        diagnostic_source_outputs,
    )
    from blueearth_cst.climate_analysis.diagnostic_tables import SCHEMA_VERSION

    settings = parse_settings(None, {"start": 2000, "end": 2002}, ["era5"])
    ordinary = diagnostic_source_outputs("source", "era5", settings)
    assert "climate_indices" not in ordinary["compute"]
    settings["climate_indices"]["enabled"] = True
    enabled = diagnostic_source_outputs("source", "era5", settings)
    assert "climate_indices" in enabled["compute"]
    assert "index_products" in enabled["render"]
    assert SCHEMA_VERSION == 2


def test_retained_rerender_routes_enabled_bundle_to_separate_output(
    tmp_path, monkeypatch
):
    from blueearth_cst.climate_analysis import climate_index_figures as figures
    from blueearth_cst.climate_analysis import diagnostic_rerender as retained

    store = tmp_path / "project" / "chirps"
    metadata = store / "_engine" / "diagnostics.json"
    metadata.parent.mkdir(parents=True)
    settings = parse_settings(
        {"climate_indices": {"enabled": True}}, {"start": 2000, "end": 2002}, ["chirps"]
    )
    metadata.write_text(json.dumps({"settings": settings, "water_year_start": 1}))
    reference = {"retained": "fixture"}
    monkeypatch.setattr(retained, "load_retained_reference", lambda path: reference)
    monkeypatch.setattr(retained, "source_stores", lambda *args: {"chirps": store})
    monkeypatch.setattr(retained, "render", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(
        figures, "render_indices", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    destination = tmp_path / "preview"
    retained.rerender_sources(tmp_path / "project", destination)
    (pointer, paths), arguments = calls[0]
    assert str(pointer).startswith(str(store))
    assert all(str(path).startswith(str(destination)) for path in paths.values())
    assert arguments["reference_record"] is reference


def test_indices_and_anomalies_share_inventory_and_retained_rerender(
    tmp_path, monkeypatch
):
    from blueearth_cst.climate_analysis import (
        climate_index_figures,
        diagnostic_rerender,
    )
    from blueearth_cst.climate_analysis.annual_spatial_anomalies import parse_settings

    settings = parse_settings(
        {"climate_indices": {"enabled": True}, "annual_spatial_anomalies": True},
        {"start": 2000, "end": 2002},
        ["era5"],
    )
    store = tmp_path / "era5"
    (store / "_engine").mkdir(parents=True)
    (store / "_engine/diagnostics.json").write_text(
        json.dumps({"settings": settings, "water_year_start": 10})
    )
    (store / "plots/_engine").mkdir(parents=True)
    (store / "plots/_engine/figure_captions.json").write_text(
        json.dumps(
            {"figures": {"annual.png": {"encoding": "annual_spatial_anomalies"}}}
        )
    )
    monkeypatch.setattr(diagnostic_rerender, "load_retained_reference", lambda _: {})
    monkeypatch.setattr(
        diagnostic_rerender, "source_stores", lambda *args: {"era5": store}
    )
    ordinary, indices = [], []
    monkeypatch.setattr(
        diagnostic_rerender, "render", lambda *args, **kwargs: ordinary.append(args)
    )
    monkeypatch.setattr(
        climate_index_figures,
        "render_indices",
        lambda *args, **kwargs: indices.append(args),
    )
    destination = tmp_path / "preview"
    diagnostic_rerender.rerender_sources(tmp_path, destination)
    paths = ordinary[0][0]
    assert paths["annual_spatial"]["fields"] == str(
        store / "tables/annual_spatial_anomalies.nc"
    )
    assert any(key.startswith("annual-map-") for key in paths["figures"])
    assert "index_products" in paths["render"]
    assert indices[0][0] == paths["compute"]["climate_indices"]
    assert all(
        str(path).startswith(str(destination)) for path in paths["render"].values()
    )


def test_chirps_native_daily_units_are_verified(tmp_path):
    climate, cells, settings = inputs(tmp_path)
    with xr.open_dataset(climate) as source:
        data = source.load()
    data.precip.attrs["units"] = "mm d**-1"
    data.to_netcdf(climate)
    data, _ = qualified_source_data(climate, "chirps")
    assert data.precip.attrs["units"] == "mm/day"


def test_source_qualification_preserves_native_grid(tmp_path):
    climate, cells, settings = inputs(tmp_path)
    with xr.open_dataset(climate) as source:
        data = source.load()
    data.precip[:] = 2.0
    data.precip.loc[dict(latitude=1.0, longitude=4.0)] = 50.0
    data.to_netcdf(climate)
    qualified, _ = qualified_source_data(climate, "chirps")
    xr.testing.assert_equal(qualified.precip.drop_attrs(), data.precip.drop_attrs())


def test_stable_results_render_series_without_spatial_maps(exported, tmp_path):
    from blueearth_cst.climate_analysis.climate_index_figures import render_indices
    from blueearth_cst.climate_analysis.climate_indices import load_index_results
    from blueearth_cst.climate_analysis.figure_naming import figure_filename

    _, options, expected, _ = exported
    table, spells, record = load_index_results(options["record_path"])
    assert len(table) == len(expected)
    assert record["source"] == options["source"]
    output = tmp_path / "plots/indices"
    output.mkdir(parents=True)
    old_name = figure_filename(
        options["source"], "r20mm", "annual_clim_map", "basin_ext"
    )
    old_map = output / old_name
    old_map.write_bytes(b"old managed index map")
    unrelated = output / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    (output / "figure_availability.json").write_text(
        json.dumps({"figures": [old_name]}), encoding="utf-8"
    )
    metadata = tmp_path / "_engine/plots/indices"
    render_indices(
        options["record_path"],
        {
            "index_products": str(output),
            "index_metadata": str(metadata),
        },
    )
    inventory = json.loads((metadata / "figure_availability.json").read_text())
    assert (metadata / "figure_titles.json").is_file()
    assert (metadata / "figure_captions.json").is_file()
    assert not (output / "_engine").exists()
    assert inventory["figures"]
    assert not any("_map_" in name for name in inventory["figures"])
    assert not old_map.exists()
    assert unrelated.read_text() == "keep me"
    assert any("annual" in name for name in inventory["figures"])
    assert any("seasonal" in name for name in inventory["figures"])
