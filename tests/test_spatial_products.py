"""Tests for composing and serializing the neutral spatial contract."""

from __future__ import annotations

import inspect
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import LineString, box

from blueearth_cst.spatial import products as products_module
from blueearth_cst.spatial.config import SpatialConfig, SpatialSources
from blueearth_cst.spatial.hydrography import prepare_hydrography
from blueearth_cst.spatial.products import (
    _delineate_spatial_units,
    _parent_basins,
    _snap_gauge_points,
    prepare_spatial_products,
    validate_written_spatial_products,
    write_spatial_products,
)


def _source_dataset(size: int = 6) -> tuple[xr.Dataset, gpd.GeoDataFrame]:
    """Return a projected synthetic hydrography and one parent geometry."""
    elevation = np.add.outer(
        np.arange(size, 0, -1, dtype=float),
        np.arange(size, 0, -1, dtype=float),
    )
    transform = Affine(1000, 0, 0, 0, -1000, size * 1000)
    flow = pyflwdir.from_dem(elevation, transform=transform, outlets="min")
    x = np.arange(500, size * 1000, 1000)
    y = np.arange(size * 1000 - 500, 0, -1000)
    coords = {"y": y, "x": x}
    source = xr.Dataset(
        {
            "flwdir": xr.DataArray(
                flow.to_array(ftype="d8"), dims=("y", "x"), coords=coords
            ),
            "elevtn": xr.DataArray(elevation, dims=("y", "x"), coords=coords),
            "uparea": xr.DataArray(
                flow.upstream_area("km2"), dims=("y", "x"), coords=coords
            ),
        }
    )
    source.raster.set_crs(3857)
    source["flwdir"].raster.set_nodata(247)
    source["elevtn"].raster.set_nodata(-9999.0)
    region = gpd.GeoDataFrame(
        {"basin_name": ["synthetic"]},
        geometry=[box(0, 0, size * 1000, size * 1000)],
        crs=3857,
    )
    return source, region


def _base_maps():
    source, region = _source_dataset()
    maps, flow = prepare_hydrography(source, region, 1000, 0.1)
    basin_map, basins, outlets = _parent_basins(maps, flow, region)
    maps["basin_id"] = basin_map
    return maps, flow, basins, outlets


def _gauge_at(maps: xr.Dataset, index: int, name: str, role: str = "control"):
    x, y = maps.raster.idx_to_xy([index])
    projected = gpd.GeoDataFrame(
        {
            "station_name": [name],
            "x": [float(x[0])],
            "y": [float(y[0])],
            "location_role": [role],
            "provided_wflow_id": [pd.NA],
        },
        geometry=gpd.points_from_xy(x, y),
        crs=maps.raster.crs,
    )
    geographic = projected.to_crs(4326)
    geographic["x"] = geographic.geometry.x
    geographic["y"] = geographic.geometry.y
    return geographic


class _FakeCatalog:
    """Small HydroMT-like catalog used to isolate spatial composition."""

    def __init__(self, source: xr.Dataset, rivers: gpd.GeoDataFrame):
        self.source = source
        self.rivers = rivers

    def get_rasterdataset(self, name: str, **_kwargs):
        if name == "hydro":
            return self.source
        template = self.source[["elevtn"]].rename({"elevtn": "value"})
        value = {"lulc": 2, "lai": 3.5, "soil": 0.4}[name]
        template["value"] = xr.full_like(template["value"], value)
        return template

    def get_geodataframe(self, name: str, **_kwargs):
        assert name == "rivers"
        return self.rivers.copy()


def _config() -> SpatialConfig:
    return SpatialConfig(
        region={"basin": [0, 0]},
        resolution=1000,
        hydrography="hydro",
        basin_index=None,
        gauge_points_path=None,
        max_automatic_subbasins=3,
        gauge_snap_tolerance_m=1500,
        river_uparea_km2=0.1,
        sources=SpatialSources(rivers="rivers", lulc="lulc", lai="lai", soil="soil"),
    )


def test_no_gauges_selects_automatic_fallback_and_complete_registry():
    """An absent gauge file still creates bounded units and primary locations."""
    maps, flow, basins, outlets = _base_maps()
    empty = products_module.read_gauge_points(None)
    snapped = _snap_gauge_points(empty, maps, flow, 1000)

    subbasin_map, subbasins, catchments, locations, registry, methods = (
        _delineate_spatial_units(maps, flow, basins, outlets, snapped, 3)
    )

    assert methods == {1: "automatic"}
    assert 1 <= len(subbasins) <= 3
    assert len(catchments) == len(subbasins)
    assert len(registry) == len(subbasins)
    assert registry["is_primary"].all()
    assert (registry["wflow_id"] == registry["subbasin_id"]).all()
    assert set(np.unique(subbasin_map)) - {0} == set(subbasins["subbasin_id"])
    assert locations.crs.to_epsg() == 4326


def test_internal_control_uses_gauge_partition_and_is_row_order_invariant():
    """A distinct internal control determines the hierarchy independent of CSV order."""
    maps, flow, basins, outlets = _base_maps()
    outlet = outlets[1]
    internal = int(
        next(
            index
            for index in np.flatnonzero(maps["river_mask"].values.ravel())
            if index != outlet
        )
    )
    gauges = pd.concat(
        [_gauge_at(maps, outlet, "Outlet"), _gauge_at(maps, internal, "Internal")],
        ignore_index=True,
    )
    gauges = gpd.GeoDataFrame(gauges, geometry="geometry", crs=4326)

    registries = []
    for candidate in (gauges, gauges.iloc[::-1].reset_index(drop=True)):
        snapped = _snap_gauge_points(candidate, maps, flow, 1000)
        result = _delineate_spatial_units(
            maps, flow, basins, outlets, snapped, max_automatic_subbasins=3
        )
        assert result[-1] == {1: "gauge"}
        registries.append(result[4].sort_values("location_code").reset_index(drop=True))

    pd.testing.assert_frame_equal(registries[0], registries[1])


def test_duplicate_snapped_controls_are_rejected():
    """Two controls cannot define the same outlet cell."""
    maps, flow, _, outlets = _base_maps()
    gauges = pd.concat(
        [_gauge_at(maps, outlets[1], "A"), _gauge_at(maps, outlets[1], "B")],
        ignore_index=True,
    )
    gauges = gpd.GeoDataFrame(gauges, geometry="geometry", crs=4326)

    with pytest.raises(ValueError, match="duplicate river cells"):
        _snap_gauge_points(gauges, maps, flow, 1000)


def test_outlet_only_control_uses_automatic_fallback():
    """An outlet point names a primary location but does not force a subdivision."""
    maps, flow, basins, outlets = _base_maps()
    gauges = _gauge_at(maps, outlets[1], "Outlet")
    snapped = _snap_gauge_points(gauges, maps, flow, 1000)

    result = _delineate_spatial_units(maps, flow, basins, outlets, snapped, 3)

    assert result[-1] == {1: "automatic"}
    assert "Outlet" in set(result[4]["station_name"])


def test_fallback_is_resolved_independently_for_multiple_parent_features():
    """Every parent survives and shares one global automatic-unit budget."""
    source, region = _source_dataset()
    directions = np.tile(np.asarray([2, 4, 8, 2, 4, 8], dtype="uint8"), (6, 1))
    directions[-1] = np.asarray([1, 0, 16, 1, 0, 16], dtype="uint8")
    source["flwdir"] = xr.DataArray(
        directions,
        dims=("y", "x"),
        coords=source.raster.coords,
    )
    source["flwdir"].raster.set_nodata(247)
    region = gpd.GeoDataFrame(
        {"basin_name": ["west", "east"]},
        geometry=[box(0, 0, 3000, 6000), box(3000, 0, 6000, 6000)],
        crs=3857,
    )
    maps, flow = prepare_hydrography(source, region, 1000, 0.1)
    basin_map, basins, outlets = _parent_basins(maps, flow, region)
    maps["basin_id"] = basin_map
    gauges = _snap_gauge_points(
        products_module.read_gauge_points(None), maps, flow, 1000
    )

    result = _delineate_spatial_units(maps, flow, basins, outlets, gauges, 4)

    assert result[-1] == {1: "automatic", 2: "automatic"}
    assert set(result[1]["basin_id"]) == {1, 2}
    assert 2 <= len(result[1]) <= 4


def test_product_writer_round_trips_every_catalog_entry(tmp_path, monkeypatch):
    """The generated catalog is portable and its core ID joins remain valid."""
    source, region = _source_dataset()
    rivers = gpd.GeoDataFrame(
        {"river_name": ["synthetic"]},
        geometry=[LineString([(0, 6000), (6000, 0)])],
        crs=3857,
    )
    monkeypatch.setattr(products_module, "_region_geometry", lambda *_: region)

    product = prepare_spatial_products(_config(), _FakeCatalog(source, rivers))
    output_dir = tmp_path / "spatial"
    write_spatial_products(product, output_dir)
    validate_written_spatial_products(output_dir)

    expected = {
        output_dir / "spatial_maps.nc",
        output_dir / "spatial_catalog.yml",
        output_dir / "spatial_report.yml",
        output_dir / "location_registry.csv",
        *(output_dir / "geoms" / f"{name}.geojson" for name in (
            "basins", "subbasins", "catchments", "rivers", "locations"
        )),
    }
    assert all(path.is_file() for path in expected)
    reopened = xr.open_dataset(output_dir / "spatial_maps.nc")
    assert reopened.attrs["spatial_contract"] == "blueearth-cst-spatial-v1"
    assert {"land_cover", "leaf_area_index", "soil_value"}.issubset(reopened)


def test_products_module_is_wflow_independent():
    """The P1 composer cannot acquire a Wflow dependency accidentally."""
    source = inspect.getsource(products_module)
    assert "hydromt_wflow" not in source
    assert "WflowSbmModel" not in source
    assert "wflow_sbm.toml" not in source


def test_catalog_uris_are_relative_to_the_generated_catalog():
    """Moving a complete spatial directory does not invalidate its catalog."""
    catalog = products_module._catalog_dict()

    assert all(not Path(entry["uri"]).is_absolute() for entry in catalog.values())
