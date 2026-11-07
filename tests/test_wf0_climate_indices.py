"""Source attribution, immutable publication, and retained climate products."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis.climate_indices import load_publication, publish
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings


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
def test_publication_preserves_source_identity_reuse_and_integrity(tmp_path, source):
    climate, cells, settings = inputs(tmp_path)
    root = tmp_path / source
    pointer = publish(climate, cells, source, root, settings, 1)
    folder, manifest = load_publication(pointer)
    hashes = manifest["products"].copy()
    with xr.open_dataset(folder / "annual_grid.nc") as ds:
        assert "r95p_amount" in ds
        assert ("tx90p" in ds) == (source == "era5")
        assert ("mean_temp_above_threshold_days" in ds) == (source == "era5")
    if source == "era5":
        with (
            xr.open_dataset(climate) as raw,
            xr.open_dataset(folder / "annual_grid.nc") as result,
        ):
            series = raw.temp.sel(latitude=1, longitude=3, time=slice("2000", "2000"))
            assert result.mean_temp_below_zero_days.sel(
                period="2000", latitude=1, longitude=3
            ).item() == int((series < 0).sum())
            assert result.mean_temp_above_threshold_days.sel(
                period="2000", latitude=1, longitude=3
            ).item() == int((series > 25).sum())
        settings["climate_indices"]["mean_temperature_threshold_degc"] = 30.0
        publish(climate, cells, source, root, settings, 1)
        warmer, _ = load_publication(pointer)
        assert warmer != folder
        with xr.open_dataset(warmer / "annual_basin.nc") as result:
            assert result.mean_temp_above_threshold_days.attrs["threshold_degc"] == 30
        settings["climate_indices"]["mean_temperature_threshold_degc"] = 25.0
    if source == "chirps":
        assert "temp_max" in manifest["unavailable_variables"]
        assert "temp" in manifest["unavailable_variables"]
    publish(climate, cells, source, root, settings, 1)
    assert load_publication(pointer)[1]["products"] == hashes
    settings["wet_day_threshold_mm"] = 2.0
    publish(climate, cells, source, root, settings, 1)
    second, _ = load_publication(pointer)
    assert second != folder
    assert (folder / "calculation.json").is_file()
    (second / "events_basin.csv").write_text("damaged", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_publication(pointer)


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
    from blueearth_cst.indicators.climate.definitions import INDEX_METADATA

    settings = parse_settings(None, {"start": 2000, "end": 2002}, ["era5"])
    ordinary = diagnostic_source_outputs("source", "era5", settings)
    assert "climate_indices" not in ordinary["compute"]
    settings["climate_indices"]["enabled"] = True
    enabled = diagnostic_source_outputs("source", "era5", settings)
    assert "climate_indices" in enabled["compute"]
    assert sum(key.startswith("index_map_") for key in enabled["render"]) == len(
        INDEX_METADATA
    )
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
    assert any(key.startswith("index_map_") for key in paths["render"])
    assert indices[0][0] == paths["compute"]["climate_indices"]
    assert all(
        str(path).startswith(str(destination)) for path in paths["render"].values()
    )
