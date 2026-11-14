"""Reuse of historical climate stores and CMIP6 raw slices across projects."""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401 -- registers the raster accessor
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from blueearth_cst.experiment.content_identity import netcdf_content_sha256
from blueearth_cst.shared import data_reuse
from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe

REGION = gpd.GeoDataFrame(geometry=[box(9.66, 0.35, 9.86, 0.48)], crs=4326)


def _catalog(path: Path, root: str, *, version: str = "v1") -> Path:
    path.write_text(
        f"meta:\n  root: {root}\n"
        "era5:\n  data_type: RasterDataset\n"
        f"  uri: meteo/era5_daily.zarr\n  metadata:\n    version: {version}\n"
        "era5_orography:\n  data_type: RasterDataset\n  uri: meteo/era5_orography.nc\n"
        "merit_hydro_ihu:\n  data_type: RasterDataset\n  uri: topography/merit/{variable}.tif\n"
        "chirps:\n  alias: chirps_v2\n"
        "chirps_v2:\n  data_type: RasterDataset\n  uri: meteo/chirps/{year}.nc\n",
        encoding="utf-8",
    )
    return path


def _record(catalog: Path, **overrides):
    record = data_reuse.store_record(
        catalogs=str(catalog),
        clim_source="era5",
        forcing_required=True,
        recipe=normalize_forcing_recipe("era5"),
        hydrography="merit_hydro_ihu",
        region=REGION,
        starttime="1995-01-01T00:00:00",
        endtime="2014-12-31T00:00:00",
    )
    record.update(overrides)
    return record


def _dataset(start: str, end: str) -> xr.Dataset:
    time = pd.date_range(start, end, freq="D")
    lat = np.array([0.475, 0.425, 0.375], dtype="float32")
    lon = np.array([9.675, 9.725, 9.775, 9.825], dtype="float32")
    rng = np.random.default_rng(7)
    full = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    values = rng.random((full.size, lat.size, lon.size), dtype="float32")
    precip = xr.DataArray(
        values,
        coords={"time": full, "latitude": lat, "longitude": lon},
        dims=("time", "latitude", "longitude"),
        attrs={"units": "mm d**-1"},
    ).sel(time=time)
    ds = precip.to_dataset(name="precip")
    ds["temp"] = (precip * 30).assign_attrs(units="degC")
    ds.attrs.update({"category": "meteo", "source_version": "v2.0"})
    ds.raster.set_crs(4326)
    return ds


def _write_fresh(ds, path: Path, region_file: Path, *, time_units=None) -> Path:
    """The fresh-extraction write tail of ``prep_historical_climate``."""
    from blueearth_cst.shared.provenance import file_sha256

    ds = ds.copy()
    if time_units is not None:
        ds["time"].encoding["units"] = time_units
    ds.attrs["region_bbox"] = [float(v) for v in REGION.total_bounds]
    ds.attrs["region_geojson_sha256"] = file_sha256(region_file)
    ds.attrs["region_source"] = Path(
        os.path.relpath(region_file, path.parent)
    ).as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding={k: {"zlib": True} for k in ds.raster.vars}, mode="w")
    return path


def _store(project: Path, name: str, ds, record, region_file) -> Path:
    store = project / "data" / "climate" / "historical" / name
    _write_fresh(ds, store / "extract_historical.nc", region_file)
    if record is not None:
        data_reuse.write_store_record(store, record)
    return store


@pytest.fixture
def region_file(tmp_path):
    path = tmp_path / "region.geojson"
    REGION.to_file(path, driver="GeoJSON")
    return path


def test_sources_come_from_config_then_environment(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv(data_reuse.REUSE_ENV, os.pathsep.join([str(b), str(a)]))
    config = {"project": {"reuse_data_from": [str(a)]}}
    assert data_reuse.reuse_sources(config) == [a.resolve(), b.resolve()]
    monkeypatch.delenv(data_reuse.REUSE_ENV)
    assert data_reuse.reuse_sources({"project": {"reuse_data_from": str(a)}}) == [
        a.resolve()
    ]
    assert data_reuse.reuse_sources({"project": {}}) == []


def test_dataset_identity_ignores_the_catalog_root(tmp_path):
    pdrive = _catalog(tmp_path / "pdrive.yml", "p:/wflow_global/hydromt")
    local = _catalog(tmp_path / "local.yml", "d:/mirror/hydromt")
    changed = _catalog(tmp_path / "changed.yml", "d:/mirror/hydromt", version="v2")
    names = ["era5", "chirps"]
    identity = data_reuse.catalog_dataset_identity(str(pdrive), names)
    assert identity == data_reuse.catalog_dataset_identity(str(local), names)
    assert (
        identity["era5"]
        != data_reuse.catalog_dataset_identity(str(changed), names)["era5"]
    )
    assert data_reuse.catalog_dataset_identity("deltares_data", names) is None


def test_alias_target_change_changes_identity(tmp_path):
    base = _catalog(tmp_path / "base.yml", "x")
    overlay = tmp_path / "overlay.yml"
    overlay.write_text(
        "chirps_v2:\n  uri: meteo/chirps_v3/{year}.nc\n", encoding="utf-8"
    )
    before = data_reuse.catalog_dataset_identity(str(base), ["chirps"])
    after = data_reuse.catalog_dataset_identity([str(base), str(overlay)], ["chirps"])
    assert before["chirps"] != after["chirps"]


def test_covering_store_is_found_and_others_are_reported(tmp_path, region_file):
    catalog = _catalog(tmp_path / "catalog.yml", "x")
    record = _record(catalog)
    source = tmp_path / "v2" / "outputs"
    _store(
        source,
        "era5_20000101_20101231",
        _dataset("2000-01-02", "2010-12-31"),
        record,
        region_file,
    )
    _store(
        source,
        "era5_19910101_20201231_old",
        _dataset("1991-01-02", "2020-12-31"),
        None,
        region_file,
    )
    other = dict(record, hydrography="merit_hydro")
    _store(
        source,
        "era5_19900101_20201231_dem",
        _dataset("1990-01-02", "2020-12-31"),
        other,
        region_file,
    )
    wanted = _store(
        source,
        "era5_wide",
        _dataset("1991-01-02", "2020-12-31"),
        record,
        region_file,
    )

    found, reasons = data_reuse.find_covering_store(
        [source], record, starttime="1995-01-01T00:00:00", endtime="2014-12-31T00:00:00"
    )

    assert found == wanted
    assert any("differs in hydrography" in reason for reason in reasons)
    assert any("no extraction record" in reason for reason in reasons)
    assert any("holds 2000-01-02..2010-12-31" in reason for reason in reasons)


def test_no_store_matches_another_region_or_its_own_folder(tmp_path, region_file):
    catalog = _catalog(tmp_path / "catalog.yml", "x")
    record = _record(catalog)
    source = tmp_path / "v2" / "outputs"
    shifted = gpd.GeoDataFrame(geometry=[box(9.6, 0.35, 9.86, 0.48)], crs=4326)
    moved = dict(
        record, region_geometry_sha256=data_reuse.region_geometry_sha256(shifted)
    )
    _store(
        source,
        "era5_19910101_20201231",
        _dataset("1991-01-02", "2020-12-31"),
        moved,
        region_file,
    )
    own = _store(
        source,
        "era5_19950101_20141231",
        _dataset("1995-01-02", "2014-12-31"),
        record,
        region_file,
    )

    found, reasons = data_reuse.find_covering_store(
        [source],
        record,
        starttime="1995-01-01T00:00:00",
        endtime="2014-12-31T00:00:00",
        exclude=own,
    )

    assert found is None
    assert reasons == ["v2/era5_19910101_20201231: differs in region_geometry_sha256"]


@pytest.mark.parametrize("time_units", [None, "days since 1950-01-02"])
def test_sliced_store_has_the_content_of_a_fresh_extraction(
    tmp_path, region_file, time_units
):
    """The WF3 seed hashes this content, so reuse must not change it.

    ``None`` is the window-anchored encoding (chirps, yearly era5 netCDFs);
    the fixed epoch is the era5 zarr encoding.
    """
    from blueearth_cst.climate_analysis.extract_historical_climate import (
        write_reused_store,
    )

    wide = tmp_path / "v2" / "store"
    _write_fresh(
        _dataset("1991-01-01", "2020-12-31"),
        wide / "extract_historical.nc",
        region_file,
        time_units=time_units,
    )
    target = tmp_path / "trial" / "store"
    fresh = _write_fresh(
        _dataset("1995-01-01", "2014-12-31"),
        target / "fresh.nc",
        region_file,
        time_units=time_units,
    )
    from blueearth_cst.shared.provenance import file_sha256

    sliced = target / "extract_historical.nc"
    write_reused_store(
        wide,
        sliced,
        starttime="1995-01-01T00:00:00",
        endtime="2014-12-31T00:00:00",
        clim_source="chirps",
        bbox=tuple(REGION.total_bounds),
        region_sha256=file_sha256(region_file),
        region_source=region_file,
        enforce_min_years=False,
    )

    assert netcdf_content_sha256(sliced) == netcdf_content_sha256(fresh)


def test_raw_slice_is_adopted_only_when_it_matches(tmp_path):
    own = tmp_path / "trial" / "outputs"
    raw = Path(
        "data/climate/projections/cmip6/raw/cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1.nc"
    )
    good, bad = tmp_path / "a" / "outputs", tmp_path / "b" / "outputs"
    for project, text in ((bad, "stale"), (good, "match")):
        (project / raw).parent.mkdir(parents=True)
        (project / raw).write_text(text, encoding="utf-8")

    candidates = data_reuse.raw_slice_candidates(
        [bad, good, tmp_path / "c"], own, own / raw
    )
    adopted = data_reuse.adopt_matching_file(
        candidates, own / raw, lambda path: path.read_text(encoding="utf-8") == "match"
    )

    assert candidates == [bad / raw, good / raw]
    assert adopted == good / raw
    assert (own / raw).read_text(encoding="utf-8") == "match"


def test_record_round_trips(tmp_path):
    catalog = _catalog(tmp_path / "catalog.yml", "x")
    record = _record(catalog)
    path = data_reuse.write_store_record(tmp_path / "store", record)
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert set(data_reuse.MATCH_FIELDS) <= set(record)


def test_project_label_names_trial_and_project_dir(tmp_path):
    path = tmp_path / "gabon" / "v2" / "outputs" / "data" / "climate" / "historical" / "x"
    assert data_reuse.project_label(path) == "v2/outputs"
