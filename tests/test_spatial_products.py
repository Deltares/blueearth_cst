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
    HYDROGRAPHY_SEAM_NAME,
    _delineate_spatial_units,
    _parent_basins,
    _snap_gauge_points,
    _validate_flow_topology,
    prepare_spatial_maps,
    prepare_spatial_units,
    read_hydrography_seam,
    spatial_report,
    validate_written_spatial_products,
    write_spatial_maps,
    write_spatial_units,
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
        if name == "lai":
            template = template.expand_dims(dim0=np.arange(12))
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
        max_subbasins_per_basin=3,
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
    # ADR 0003 §12: a primary is `basin*1000 + local_subbasin*10`, no longer
    # its own subbasin_id. What still holds is that it ends in 0.
    assert (registry["wflow_id"] % 10 == 0).all()
    rows = registry["snapped_row"].astype(int).to_numpy()
    cols = registry["snapped_col"].astype(int).to_numpy()
    assert maps["river_mask"].values[rows, cols].all()
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
            maps, flow, basins, outlets, snapped, max_subbasins_per_basin=3
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


def _two_parent_maps():
    """A synthetic grid resolving to TWO parent basins, west and east."""
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
    return maps, flow, basins, outlets, gauges


def test_the_automatic_ceiling_applies_to_each_parent_INDEPENDENTLY():
    """ADR 0003 §11: a per-parent ceiling, not one budget split across parents.

    Before §11 the number was a GLOBAL budget: every fallback parent got one
    unit and `allocate_automatic_subbasin_budgets` spread the remainder by
    upstream area, so a parent's partition depended on how many OTHER parents
    the project had. Now each parent is capped at the same number on its own,
    which is what makes two projects' partitions comparable.
    """
    maps, flow, basins, outlets, gauges = _two_parent_maps()

    result = _delineate_spatial_units(maps, flow, basins, outlets, gauges, 4)

    assert result[-1] == {1: "automatic", 2: "automatic"}
    subbasins = result[1]
    assert set(subbasins["basin_id"]) == {1, 2}
    per_parent = subbasins.groupby("basin_id").size()
    # The bound is PER PARENT. Under the old global budget of 4 the two parents
    # together could not exceed 4; now each may reach it, so a `len() <= 4`
    # assertion would pass for the wrong reason.
    assert (per_parent <= 4).all(), per_parent.to_dict()
    assert (per_parent >= 1).all(), per_parent.to_dict()


def test_more_parents_than_the_ceiling_no_longer_raises():
    """The failure mode §11 removes, as a test rather than as prose.

    `allocate_automatic_subbasin_budgets` raised outright when
    `len(parent_areas) > max_count` -- a project with more parent basins than
    the global budget could not be delineated at all. With a per-parent ceiling
    there is nothing to run out of: a ceiling of 1 simply gives each parent one
    subbasin.
    """
    maps, flow, basins, outlets, gauges = _two_parent_maps()

    result = _delineate_spatial_units(maps, flow, basins, outlets, gauges, 1)

    subbasins = result[1]
    assert set(subbasins["basin_id"]) == {1, 2}
    assert subbasins.groupby("basin_id").size().eq(1).all()


def test_the_derived_report_methods_match_the_delineation():
    maps, flow, basins, outlets, gauges = _two_parent_maps()
    result = _delineate_spatial_units(maps, flow, basins, outlets, gauges, 4)

    # ADR 0003 §8: `methods` used to cross into the report in memory. The
    # raster half now derives it from `subbasins`, which is only sound while
    # the two agree -- checked on the MULTI-BASIN case, since a one-basin
    # fixture cannot tell a per-basin mapping from a constant.
    #
    # Compared as ITEMS, not as dicts. `yaml.safe_dump(sort_keys=False)` writes
    # insertion order, so a key reordering is a byte change in
    # spatial_report.yml that dict equality cannot see.
    _subbasin_map, subbasins, _catchments, _locations, registry, methods = result
    derived = spatial_report(basins, subbasins, registry)[
        "delineation_method_by_basin"
    ]
    assert list(derived.items()) == list(methods.items())


def test_the_report_rejects_a_parent_with_two_delineation_methods():
    """Guard on the derivation above: the uniformity it rests on is checked."""
    maps, flow, basins, outlets = _base_maps()
    empty = products_module.read_gauge_points(None)
    snapped = _snap_gauge_points(empty, maps, flow, 1000)
    _map, subbasins, _c, _l, registry, _m = _delineate_spatial_units(
        maps, flow, basins, outlets, snapped, 3
    )
    if len(subbasins) < 2:
        pytest.skip("fixture produced a single subbasin")
    subbasins = subbasins.copy()
    subbasins.loc[subbasins.index[0], "delineation_method"] = "gauge"

    with pytest.raises(ValueError, match="more than one delineation method"):
        spatial_report(basins, subbasins, registry)


def _units_kwargs(config: SpatialConfig) -> dict:
    """The `shared.basin` fields the vector half takes, from a full config.

    Mirrors what `snake_utils.spatial_units_rule` puts in the rule's params --
    which deliberately excludes the thematic source names (ADR 0003 §8b).
    """
    return {
        "hydrography": config.hydrography,
        "resolution": config.resolution,
        "river_uparea_km2": config.river_uparea_km2,
        "rivers_source": config.sources.rivers,
        "gauge_points_path": config.gauge_points_path,
        "gauge_snap_tolerance_m": config.gauge_snap_tolerance_m,
        "max_subbasins_per_basin": config.max_subbasins_per_basin,
    }


def _run_both_halves(output_dir, catalog, config):
    """Drive the split exactly as rules 1.01c and 1.02 drive it, through disk."""
    units = prepare_spatial_units(catalog, "region.geojson", **_units_kwargs(config))
    write_spatial_units(units, output_dir)
    units.maps.close()

    seam = read_hydrography_seam(output_dir / HYDROGRAPHY_SEAM_NAME)
    basins = gpd.read_file(output_dir / "geoms" / "basins.geojson")
    maps = prepare_spatial_maps(seam, basins, config, catalog)
    report = spatial_report(
        basins,
        gpd.read_file(output_dir / "geoms" / "subbasins.geojson"),
        pd.read_csv(output_dir / "location_registry.csv"),
    )
    return units, seam, maps, report


def test_both_halves_round_trip_every_catalog_entry(tmp_path, monkeypatch):
    """The generated catalog is portable and its core ID joins remain valid.

    Drives the ADR 0003 §8 split end to end: the vector half writes, the seam
    goes to disk, the raster half reads it back. Before the split this was one
    in-memory call, and the seam is exactly what the split has to carry
    without loss.
    """
    source, region = _source_dataset()
    rivers = gpd.GeoDataFrame(
        {"river_name": ["synthetic"]},
        geometry=[LineString([(0, 6000), (6000, 0)])],
        crs=3857,
    )
    monkeypatch.setattr(products_module, "_region_geometry", lambda *_: region)
    catalog = _FakeCatalog(source, rivers)
    config = _config()
    output_dir = tmp_path / "spatial"

    # ADR 0003: the region arrives as a declared input path; _region_geometry
    # is stubbed above, so the value is only passed through.
    units, seam, maps, report = _run_both_halves(output_dir, catalog, config)
    assert units.subbasins.geometry.is_valid.all()
    assert units.catchments.geometry.is_valid.all()
    # Exercise CF decoding of integer nodata: xarray reopens _FillValue=0 as
    # NaN, which must not be mistaken for another spatial-unit identifier.
    ids, counts = np.unique(maps["subbasin_id"].values, return_counts=True)
    multi_cell_id = int(ids[np.argmax(counts)])
    row, col = np.argwhere(maps["subbasin_id"].values == multi_cell_id)[0]
    maps["subbasin_id"].values[row, col] = 0
    write_spatial_maps(maps, report, output_dir)
    validate_written_spatial_products(output_dir)

    expected = {
        output_dir / "spatial_maps.nc",
        output_dir / "spatial_catalog.yml",
        output_dir / "spatial_report.yml",
        output_dir / "location_registry.csv",
        output_dir / HYDROGRAPHY_SEAM_NAME,
        *(output_dir / "geoms" / f"{name}.geojson" for name in (
            "basins", "subbasins", "catchments", "rivers", "locations"
        )),
    }
    assert all(path.is_file() for path in expected)
    reopened = xr.open_dataset(output_dir / "spatial_maps.nc")
    assert reopened.attrs["spatial_contract"] == "blueearth-cst-spatial-v1"
    assert {"land_cover", "leaf_area_index", "soil_value"}.issubset(reopened)
    assert reopened["leaf_area_index"].dims == ("month", "y", "x")
    assert reopened["month"].values.tolist() == list(range(1, 13))
    reopened.close()
    seam.close()


def test_the_seam_carries_the_grid_stack_with_its_dtypes(tmp_path, monkeypatch):
    """The property the split rests on, and the one netCDF quietly breaks.

    Every layer stores its nodata as a `_FillValue` ATTRIBUTE, so xarray's
    default CF decoding recasts the array to float and puts NaN where the fill
    is. `basin_id` and `subbasin_id` would come back float64 and
    `spatial_maps.nc` would ship float identifier rasters -- silently, since
    the values still compare equal. `read_hydrography_seam` reads with
    `mask_and_scale=False` for exactly this.
    """
    source, region = _source_dataset()
    rivers = gpd.GeoDataFrame(
        {"river_name": ["synthetic"]},
        geometry=[LineString([(0, 6000), (6000, 0)])],
        crs=3857,
    )
    monkeypatch.setattr(products_module, "_region_geometry", lambda *_: region)
    config = _config()
    output_dir = tmp_path / "spatial"

    units = prepare_spatial_units(
        _FakeCatalog(source, rivers), "region.geojson", **_units_kwargs(config)
    )
    expected = {name: units.maps[name].dtype for name in units.maps.data_vars}
    write_spatial_units(units, output_dir)
    units.maps.close()

    seam = read_hydrography_seam(output_dir / HYDROGRAPHY_SEAM_NAME)
    assert dict(seam.dtypes) == expected
    # ADR 0003 §8a: the WHOLE stack crosses the seam, not only the six layers
    # the section enumerates -- `spatial_maps.nc` holds these too, and
    # re-deriving them in the raster half is the second hydrography read §8a
    # rejects.
    assert set(seam.data_vars) >= {
        "flow_direction", "flow_accumulation", "upstream_area", "river_mask",
        "basin_id", "subbasin_id", "cell_area", "river_order", "elevation",
        "slope",
    }
    assert seam.raster.crs.to_epsg() == 3857
    seam.close()


def test_the_seam_is_not_advertised_as_a_product():
    """It is an intermediate, not a catalog entry (ADR 0003 §8a).

    `build_wflow_model` resolves every model input through this catalog, so an
    entry here would make the seam part of the model-build interface.
    """
    catalog = products_module._catalog_dict()
    uris = {entry["uri"] for entry in catalog.values()}
    assert HYDROGRAPHY_SEAM_NAME not in uris
    assert "hydrography" not in catalog


def test_products_module_is_wflow_independent():
    """The P1 composer cannot acquire a Wflow dependency accidentally."""
    source = inspect.getsource(products_module)
    assert "hydromt_wflow" not in source
    assert "WflowSbmModel" not in source
    assert "wflow_sbm.toml" not in source


def test_flow_validator_rejects_decreasing_downstream_accumulation():
    """A spatial file cannot pass when accumulation contradicts its D8 graph."""
    maps, flow, _, _ = _base_maps()
    active = np.flatnonzero(maps["basin_id"].values.ravel() > 0)
    upstream = next(index for index in active if flow.idxs_ds[index] != index)
    downstream = int(flow.idxs_ds[upstream])
    corrupted = maps.copy(deep=True)
    corrupted["flow_accumulation"].values.ravel()[upstream] = 100
    corrupted["flow_accumulation"].values.ravel()[downstream] = 1

    with pytest.raises(ValueError, match="decreases"):
        _validate_flow_topology(corrupted)


def test_catalog_uris_are_relative_to_the_generated_catalog():
    """Moving a complete spatial directory does not invalidate its catalog."""
    catalog = products_module._catalog_dict()

    assert all(not Path(entry["uri"]).is_absolute() for entry in catalog.values())
