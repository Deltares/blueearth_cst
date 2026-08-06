"""Compose and write the versioned spatial-foundation products.

Two halves since ADR 0003 §8, split at the thematic-raster seam:

* the **vector half** (``prepare_spatial_units`` / ``write_spatial_units``) —
  region polygon + hydrography catalog + ``shared.basin.gauge_points`` into the
  five vector layers, the location registry, and the hydrography grid stack.
  Model-free, engine-neutral, and declared by ALL THREE workflows so WF2 and
  WF3 can obtain basin and subbasin boundaries without resampling a single
  thematic raster;
* the **raster half** (``prepare_spatial_maps`` / ``write_spatial_maps``) —
  reads that seam back, adds the LULC/LAI/soil layers, and writes
  ``spatial_maps.nc`` with its catalog and report. **WF1 only**: it exists to
  parameterise Wflow.

The seam is NOT free, and the split is not a pure decomposition (ADR 0003 §8a):
what crossed it in memory is the whole hydrography grid stack, so the vector
half writes it as an intermediate the raster half declares as an input.
Recomputing it instead would make WF1 read the hydrography twice with two grids
that can drift.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from hydromt import DataCatalog
from hydromt.gis import flw
from pyflwdir import FlwdirRaster

from blueearth_cst.spatial.config import SpatialConfig
from blueearth_cst.spatial.delineate_region import read_region
from blueearth_cst.spatial.delineation import (
    downstream_steps,
    find_parent_outlet,
    full_catchment,
    incremental_subbasins,
    select_automatic_subbasins,
)
from blueearth_cst.spatial.hydrography import prepare_hydrography
from blueearth_cst.spatial.identity import (
    assign_basin_ids,
    assign_location_ids,
    assign_subbasin_ids,
)

SPATIAL_CONTRACT_VERSION = "blueearth-cst-spatial-v1"

#: The seam intermediate's filename, relative to the spatial directory.
#:
#: A SEAM INTERMEDIATE, NOT A PRODUCT (ADR 0003 §8a): it is deliberately absent
#: from ``spatial_catalog.yml``, which is the model-build interface. It carries
#: the WHOLE hydrography grid stack -- not only the six layers §8a enumerates --
#: because ``spatial_maps.nc`` also holds ``cell_area``, ``river_order``,
#: ``elevation`` and ``slope``, and recomputing those in the raster half is the
#: second hydrography read §8a rejects.
HYDROGRAPHY_SEAM_NAME = "hydrography.nc"


@dataclass
class SpatialUnits:
    """The vector half's in-memory result: layers, registry, and the seam grid.

    ``maps`` is the hydrography grid stack the raster half consumes; the rest
    are the six declared vector artifacts, already in EPSG:4326.
    """

    maps: xr.Dataset
    basins: gpd.GeoDataFrame
    subbasins: gpd.GeoDataFrame
    catchments: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    locations: gpd.GeoDataFrame
    location_registry: pd.DataFrame


def _vectorize_ids(data: xr.DataArray, id_column: str) -> gpd.GeoDataFrame:
    """Vectorize positive raster IDs and name the value column explicitly."""
    vector = data.raster.vectorize().rename(columns={"value": id_column})
    vector[id_column] = vector[id_column].astype("int32")
    vector = vector.loc[vector[id_column] > 0]
    dissolved = vector.dissolve(by=id_column, as_index=False).reset_index(drop=True)
    # Corner-connected raster cells can polygonize as self-touching rings.
    # Preserve their exact footprint as valid Polygon/MultiPolygon geometry.
    dissolved.geometry = dissolved.geometry.make_valid()
    return dissolved


def _region_geometry(region_fn: str | os.PathLike[str]) -> gpd.GeoDataFrame:
    """Read the shared region artifact and split it into parent features.

    This used to call ``parse_region_basin`` itself. Since ADR 0003 the
    delineation happens once per project, in rule ``delineate_region``, and this
    reads its declared output — so 1.02 and the climate store can no longer
    disagree about the region, and WF2/WF3 can obtain the polygon without
    running either producer.

    Every validation stays: ``read_region`` rechecks non-empty and CRS, and the
    explode + non-overlap check below is unchanged. This stopped delineating,
    not checking.
    """
    geometry = read_region(region_fn)
    geometry = geometry.explode(index_parts=False).reset_index(drop=True)
    for left in range(len(geometry)):
        for right in range(left + 1, len(geometry)):
            overlap = geometry.geometry.iloc[left].intersection(
                geometry.geometry.iloc[right]
            )
            if not overlap.is_empty and overlap.area > 0:
                raise ValueError(
                    "resolved parent basins overlap; separate parent features must "
                    "be non-overlapping"
                )
    return geometry


def _parent_basins(
    maps: xr.Dataset, flwdir: FlwdirRaster, region_geom: gpd.GeoDataFrame
) -> tuple[xr.DataArray, gpd.GeoDataFrame, dict[int, int]]:
    """Rasterize parent features and assign deterministic parent identities."""
    if region_geom.crs != maps.raster.crs:
        region_geom = region_geom.to_crs(maps.raster.crs)
    records: list[dict[str, Any]] = []
    masks: dict[int, np.ndarray] = {}
    source_name_column = next(
        (name for name in ("basin_name", "name") if name in region_geom.columns),
        None,
    )
    for source_feature, row in region_geom.iterrows():
        feature = gpd.GeoDataFrame(geometry=[row.geometry], crs=region_geom.crs)
        mask = maps.raster.geometry_mask(feature).values.astype(bool)
        mask &= maps["flow_direction"].values != maps["flow_direction"].raster.nodata
        outlet_index = find_parent_outlet(flwdir, mask, maps["upstream_area"].values)
        outlet_row, outlet_col = np.unravel_index(outlet_index, flwdir.shape)
        records.append(
            {
                "source_feature": int(source_feature),
                "basin_name": row[source_name_column] if source_name_column else None,
                "upstream_area": float(maps["upstream_area"].values.ravel()[outlet_index]),
                "outlet_row": int(outlet_row),
                "outlet_col": int(outlet_col),
                "outlet_index": outlet_index,
            }
        )
        masks[int(source_feature)] = mask

    basin_table = assign_basin_ids(pd.DataFrame(records))
    basin_values = np.zeros(maps.raster.shape, dtype="int32")
    outlet_by_basin: dict[int, int] = {}
    for row in basin_table.itertuples(index=False):
        mask = masks[row.source_feature]
        collision = (basin_values > 0) & mask
        if collision.any():
            raise ValueError("resolved parent basins occupy the same grid cells")
        basin_values[mask] = row.basin_id
        outlet_by_basin[int(row.basin_id)] = int(row.outlet_index)

    basin_map = xr.DataArray(
        basin_values,
        coords=maps.raster.coords,
        dims=maps.raster.dims,
        name="basin_id",
        attrs={"long_name": "parent basin identifier", "units": "1", "_FillValue": 0},
    )
    basin_map.raster.set_crs(maps.raster.crs)
    basin_map.raster.set_nodata(0)
    basins = _vectorize_ids(basin_map, "basin_id").merge(
        basin_table[["basin_id", "basin_code", "basin_name"]],
        on="basin_id",
        how="left",
        validate="one_to_one",
    )
    return basin_map, basins, outlet_by_basin


def read_gauge_points(path: str | os.PathLike[str] | None) -> gpd.GeoDataFrame:
    """Read optional EPSG:4326 gauge/control points with explicit roles."""
    columns = ["station_name", "x", "y", "location_role", "provided_wflow_id"]
    if path is None:
        return gpd.GeoDataFrame(columns=columns + ["geometry"], geometry="geometry", crs=4326)
    frame = pd.read_csv(path, sep=",")
    missing = sorted({"station_name", "x", "y"}.difference(frame.columns))
    if missing:
        raise ValueError(f"gauge_points is missing required columns: {missing}")
    if frame.empty:
        return gpd.GeoDataFrame(columns=columns + ["geometry"], geometry="geometry", crs=4326)
    if frame[["x", "y"]].isna().any().any():
        raise ValueError("gauge_points x/y coordinates cannot be missing")
    frame["station_name"] = frame["station_name"].astype("string").str.strip()
    if frame["station_name"].isna().any() or frame["station_name"].eq("").any():
        raise ValueError("gauge_points station_name values cannot be empty")
    if "location_role" not in frame:
        frame["location_role"] = "control"
    frame["location_role"] = (
        frame["location_role"].fillna("control").astype("string").str.strip().str.lower()
    )
    invalid_roles = sorted(set(frame["location_role"]).difference({"control", "observation"}))
    if invalid_roles:
        raise ValueError(
            f"gauge_points location_role values must be control or observation: {invalid_roles}"
        )
    frame["provided_wflow_id"] = frame["wflow_id"] if "wflow_id" in frame else pd.NA
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["x"], frame["y"]),
        crs=4326,
    )


def _snap_gauge_points(
    gauges: gpd.GeoDataFrame,
    maps: xr.Dataset,
    flwdir: FlwdirRaster,
    tolerance_m: float,
) -> gpd.GeoDataFrame:
    """Snap configured points to rivers and attach parent/grid coordinates."""
    if gauges.empty:
        result = gauges.copy()
        for column in (
            "original_x",
            "original_y",
            "snapped_x",
            "snapped_y",
            "snapped_distance_m",
            "snapped_index",
            "snapped_row",
            "snapped_col",
            "basin_id",
        ):
            result[column] = pd.Series(dtype="float64")
        return result
    projected = gauges.to_crs(maps.raster.crs)
    initial = maps.raster.xy_to_idx(
        xs=projected.geometry.x.values, ys=projected.geometry.y.values
    )
    snapped, distances = flwdir.snap(
        idxs=initial, mask=maps["river_mask"].values, unit="m"
    )
    too_far = distances > tolerance_m
    if too_far.any():
        names = gauges.loc[too_far, "station_name"].tolist()
        raise ValueError(
            f"gauge points exceed gauge_snap_tolerance_m={tolerance_m}: {names}"
        )
    duplicate_points = pd.Series(snapped).duplicated(keep=False)
    if duplicate_points.any():
        duplicate_indices = sorted(set(snapped[duplicate_points].tolist()))
        raise ValueError(
            "gauge points snap to duplicate river cells: "
            f"{duplicate_indices}"
        )

    snapped_x, snapped_y = maps.raster.idx_to_xy(snapped)
    snapped_projected = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(snapped_x, snapped_y), crs=maps.raster.crs
    )
    snapped_wgs84 = snapped_projected.to_crs(4326)
    result = gauges.copy()
    result["original_x"] = gauges.geometry.x.values
    result["original_y"] = gauges.geometry.y.values
    result["snapped_x"] = snapped_wgs84.geometry.x.values
    result["snapped_y"] = snapped_wgs84.geometry.y.values
    result["snapped_distance_m"] = distances
    result["snapped_index"] = snapped.astype("int64")
    result["snapped_row"], result["snapped_col"] = np.unravel_index(
        snapped, maps.raster.shape
    )
    basin_ids = maps["basin_id"].values.ravel()[snapped]
    if (basin_ids <= 0).any():
        names = result.loc[basin_ids <= 0, "station_name"].tolist()
        raise ValueError(f"gauge points fall outside resolved parent basins: {names}")
    result["basin_id"] = basin_ids.astype("int32")
    result.geometry = snapped_wgs84.geometry
    result.set_crs(4326, allow_override=True, inplace=True)
    return result


def _mask_flwdir(
    maps: xr.Dataset, parent_mask: np.ndarray
) -> FlwdirRaster:
    """Create a parent-local flow network on the shared analysis grid."""
    mask = xr.DataArray(
        parent_mask,
        coords=maps.raster.coords,
        dims=maps.raster.dims,
    )
    return flw.flwdir_from_da(maps["flow_direction"], ftype="d8", mask=mask)


def _catchment_geometry(
    maps: xr.Dataset, catchment: np.ndarray, temporary_label: int
) -> gpd.GeoDataFrame:
    """Vectorize one full contributing catchment."""
    data = xr.DataArray(
        np.where(catchment, temporary_label, 0).astype("int32"),
        coords=maps.raster.coords,
        dims=maps.raster.dims,
        name="temporary_label",
    )
    data.raster.set_crs(maps.raster.crs)
    data.raster.set_nodata(0)
    vector = _vectorize_ids(data, "temporary_label")
    return vector


def _delineate_spatial_units(
    maps: xr.Dataset,
    flwdir: FlwdirRaster,
    basins: gpd.GeoDataFrame,
    outlet_by_basin: dict[int, int],
    gauges: gpd.GeoDataFrame,
    max_subbasins_per_basin: int,
) -> tuple[xr.DataArray, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, dict[int, str]]:
    """Resolve gauge-driven or automatic partitions independently per parent.

    ADR 0003 §11: the ceiling is PER PARENT, so every automatic parent gets the
    same one and there is nothing to allocate.
    ``allocate_automatic_subbasin_budgets`` — which handed each parent one unit,
    spread the remainder by largest-remainder weighted on upstream area, and
    RAISED outright when the project had more parent basins than the global
    budget — is deleted rather than adapted. Its per-parent upstream areas went
    with it: they existed only to weight that split.

    Dropping the area weighting costs less than it sounds. ``select_automatic_
    subbasins`` treats the ceiling as an UPPER BOUND, binary-searching for the
    most detailed partition within it, so a small parent already yielded fewer
    units than a large one at the same ceiling.
    """
    methods: dict[int, str] = {}
    for basin_id, parent_outlet in outlet_by_basin.items():
        controls = gauges.loc[
            gauges["basin_id"].eq(basin_id) & gauges["location_role"].eq("control")
        ]
        has_internal = bool((controls["snapped_index"] != parent_outlet).any())
        methods[basin_id] = "gauge" if has_internal else "automatic"

    temporary_map = np.zeros(maps.raster.shape, dtype="int32")
    subbasin_records: list[dict[str, Any]] = []
    catchment_parts: list[gpd.GeoDataFrame] = []
    primary_seeds: list[dict[str, Any]] = []
    for basin_id in sorted(outlet_by_basin):
        parent_mask = maps["basin_id"].values == basin_id
        parent_flwdir = _mask_flwdir(maps, parent_mask)
        parent_outlet = outlet_by_basin[basin_id]
        controls = gauges.loc[
            gauges["basin_id"].eq(basin_id) & gauges["location_role"].eq("control")
        ]
        if methods[basin_id] == "gauge":
            internal = sorted(
                set(
                    controls.loc[
                        controls["snapped_index"] != parent_outlet, "snapped_index"
                    ].astype(int)
                )
            )
            outlets = np.asarray([parent_outlet, *internal], dtype=np.int64)
            partition = incremental_subbasins(parent_flwdir, outlets)
        else:
            partition, outlets = select_automatic_subbasins(
                parent_flwdir,
                maps["upstream_area"].values,
                max_subbasins_per_basin,
                outlet_mask=maps["river_mask"].values,
            )
            if parent_outlet not in outlets:
                raise RuntimeError(
                    f"automatic partition for basin {basin_id} omitted its outlet"
                )

        for local_label, outlet_index in enumerate(outlets, start=1):
            temporary_label = basin_id * 1000 + local_label
            temporary_map[partition == local_label] = temporary_label
            row, col = np.unravel_index(outlet_index, maps.raster.shape)
            matching_control = controls.loc[
                controls["snapped_index"].eq(outlet_index)
            ]
            supplied_name = (
                matching_control.iloc[0]["station_name"]
                if not matching_control.empty
                else None
            )
            subbasin_records.append(
                {
                    "temporary_label": temporary_label,
                    "basin_id": basin_id,
                    "downstream_steps": downstream_steps(
                        parent_flwdir, int(outlet_index), parent_outlet
                    ),
                    "upstream_area": float(
                        maps["upstream_area"].values.ravel()[outlet_index]
                    ),
                    "outlet_row": int(row),
                    "outlet_col": int(col),
                    "outlet_index": int(outlet_index),
                    "subbasin_name": supplied_name,
                    "delineation_method": methods[basin_id],
                }
            )
            catchment = full_catchment(parent_flwdir, int(outlet_index))
            catchment_parts.append(
                _catchment_geometry(maps, catchment, temporary_label)
            )
            if matching_control.empty:
                snapped_x, snapped_y = maps.raster.idx_to_xy([outlet_index])
                point = gpd.GeoSeries(
                    gpd.points_from_xy(snapped_x, snapped_y), crs=maps.raster.crs
                ).to_crs(4326).iloc[0]
                primary_seeds.append(
                    {
                        "temporary_label": temporary_label,
                        "station_name": None,
                        "location_role": "automatic_outlet",
                        "is_primary": True,
                        "original_x": point.x,
                        "original_y": point.y,
                        "snapped_x": point.x,
                        "snapped_y": point.y,
                        "snapped_row": int(row),
                        "snapped_col": int(col),
                        "provided_wflow_id": pd.NA,
                    }
                )
            else:
                gauge = matching_control.iloc[0]
                primary_seeds.append(
                    {
                        "temporary_label": temporary_label,
                        "station_name": gauge["station_name"],
                        "location_role": "control",
                        "is_primary": True,
                        "original_x": gauge["original_x"],
                        "original_y": gauge["original_y"],
                        "snapped_x": gauge["snapped_x"],
                        "snapped_y": gauge["snapped_y"],
                        "snapped_row": int(gauge["snapped_row"]),
                        "snapped_col": int(gauge["snapped_col"]),
                        "provided_wflow_id": gauge["provided_wflow_id"],
                    }
                )

    subbasin_table = assign_subbasin_ids(pd.DataFrame(subbasin_records))
    subbasin_values = np.zeros_like(temporary_map, dtype="int32")
    for row in subbasin_table.itertuples(index=False):
        subbasin_values[temporary_map == row.temporary_label] = row.subbasin_id
    subbasin_map = xr.DataArray(
        subbasin_values,
        coords=maps.raster.coords,
        dims=maps.raster.dims,
        name="subbasin_id",
        attrs={"long_name": "incremental subbasin identifier", "units": "1", "_FillValue": 0},
    )
    subbasin_map.raster.set_crs(maps.raster.crs)
    subbasin_map.raster.set_nodata(0)
    subbasins = _vectorize_ids(subbasin_map, "subbasin_id").merge(
        subbasin_table[
            [
                "subbasin_id",
                "basin_id",
                "subbasin_code",
                "subbasin_name",
                "delineation_method",
                "upstream_area",
            ]
        ],
        on="subbasin_id",
        how="left",
        validate="one_to_one",
    )

    catchments = pd.concat(catchment_parts, ignore_index=True)
    catchments = gpd.GeoDataFrame(catchments, geometry="geometry", crs=maps.raster.crs)
    catchments = catchments.merge(
        subbasin_table[
            [
                "temporary_label",
                "subbasin_id",
                "basin_id",
                "subbasin_code",
                "subbasin_name",
            ]
        ],
        on="temporary_label",
        how="left",
        validate="one_to_one",
    ).drop(columns="temporary_label")

    seeds = pd.DataFrame(primary_seeds)
    assigned_primary = set(
        zip(seeds["temporary_label"], seeds["snapped_row"], seeds["snapped_col"])
    )
    extra_seeds: list[dict[str, Any]] = []
    for gauge in gauges.itertuples(index=False):
        temporary_label = int(temporary_map.ravel()[gauge.snapped_index])
        key = (temporary_label, int(gauge.snapped_row), int(gauge.snapped_col))
        if key in assigned_primary and gauge.location_role == "control":
            continue
        extra_seeds.append(
            {
                "temporary_label": temporary_label,
                "station_name": gauge.station_name,
                "location_role": gauge.location_role,
                "is_primary": False,
                "original_x": gauge.original_x,
                "original_y": gauge.original_y,
                "snapped_x": gauge.snapped_x,
                "snapped_y": gauge.snapped_y,
                "snapped_row": int(gauge.snapped_row),
                "snapped_col": int(gauge.snapped_col),
                "provided_wflow_id": gauge.provided_wflow_id,
            }
        )
    if extra_seeds:
        seeds = pd.concat([seeds, pd.DataFrame(extra_seeds)], ignore_index=True)
    seeds = seeds.merge(
        subbasin_table[
            [
                "temporary_label",
                "basin_id",
                "subbasin_id",
                "subbasin_code",
                "subbasin_name",
                # ADR 0003 §12 builds wflow_id from this, so it has to reach
                # `assign_location_ids` rather than stop at the subbasin table.
                "local_subbasin_number",
            ]
        ],
        on="temporary_label",
        how="left",
        validate="many_to_one",
    ).merge(
        basins[["basin_id", "basin_code", "basin_name"]],
        on="basin_id",
        how="left",
        validate="many_to_one",
    )
    synthetic = seeds["station_name"].isna()
    seeds.loc[synthetic, "station_name"] = seeds.loc[synthetic, "subbasin_name"]
    registry = assign_location_ids(seeds)
    locations = gpd.GeoDataFrame(
        registry.copy(),
        geometry=gpd.points_from_xy(registry["snapped_x"], registry["snapped_y"]),
        crs=4326,
    )
    return subbasin_map, subbasins, catchments, locations, registry, methods


def _as_dataset(data: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Normalize one catalog raster result to a dataset."""
    if isinstance(data, xr.DataArray):
        name = data.name or "value"
        return data.rename(name).to_dataset()
    return data


def _resample_source(
    catalog: DataCatalog,
    source_name: str,
    basin_geom: gpd.GeoDataFrame,
    grid: xr.DataArray,
    prefix: str,
    method: str,
) -> xr.Dataset:
    """Read, clip, regrid, and namespace one model-neutral thematic source."""
    source = catalog.get_rasterdataset(
        source_name,
        geom=basin_geom,
        buffer=2,
        single_var_as_array=False,
    )
    if source is None:
        raise ValueError(f"catalog source {source_name!r} returned no raster data")
    source_ds = _as_dataset(source)
    if source_ds.raster.crs is None:
        raise ValueError(f"catalog source {source_name!r} has no CRS")
    reprojected = source_ds.raster.reproject_like(grid, method=method)
    if (
        prefix == "leaf_area_index"
        and "dim0" in reprojected.dims
        and reprojected.sizes["dim0"] == 12
    ):
        reprojected = reprojected.rename(dim0="month").assign_coords(
            month=np.arange(1, 13, dtype="int16")
        )
        reprojected["month"].attrs.update(
            long_name="calendar month", units="1"
        )
    renamed: dict[str, str] = {}
    names = list(reprojected.data_vars)
    for name in names:
        if prefix == "land_cover" and len(names) == 1:
            renamed[name] = "land_cover"
        elif prefix == "leaf_area_index" and len(names) == 1:
            renamed[name] = "leaf_area_index"
        else:
            renamed[name] = f"{prefix}_{name}"
    reprojected = reprojected.rename(renamed)
    for name in reprojected.data_vars:
        data = reprojected[name]
        nodata = data.raster.nodata
        if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
            nodata = -1 if method == "nearest" else -9999.0
            if method == "nearest":
                data = data.fillna(nodata)
            else:
                data = data.astype("float32").fillna(nodata)
            reprojected[name] = data
            reprojected[name].raster.set_nodata(nodata)
        reprojected[name].encoding.pop("_FillValue", None)
        reprojected[name].attrs.update(
            source=source_name,
            resampling=method,
            resolution=float(abs(grid.raster.res[0])),
            units=reprojected[name].attrs.get("units", "source-native"),
            _FillValue=nodata,
        )
    return reprojected


def _thematic_maps(
    catalog: DataCatalog,
    config: SpatialConfig,
    basins: gpd.GeoDataFrame,
    grid: xr.DataArray,
) -> xr.Dataset:
    """Load raw/analysis-ready land-cover, LAI, and soil layers."""
    return xr.merge(
        [
            _resample_source(
                catalog, config.sources.lulc, basins, grid, "land_cover", "nearest"
            ),
            _resample_source(
                catalog,
                config.sources.lai,
                basins,
                grid,
                "leaf_area_index",
                "average",
            ),
            _resample_source(
                catalog, config.sources.soil, basins, grid, "soil", "average"
            ),
        ],
        compat="override",
    )


def prepare_spatial_units(
    catalog: DataCatalog,
    region_fn: str | os.PathLike[str],
    *,
    hydrography: str,
    resolution: float,
    river_uparea_km2: float,
    rivers_source: str,
    gauge_points_path: str | None,
    gauge_snap_tolerance_m: float,
    max_subbasins_per_basin: int,
) -> SpatialUnits:
    """Build the model-free vector foundation and the hydrography grid seam.

    The vector half of ADR 0003 §8. Every argument is a field
    ``parse_spatial_config`` resolves from ``shared.basin`` ALONE — spelled out
    rather than taken as a ``SpatialConfig`` so §8b's requirement is visible in
    the signature: the rule is declared by all three workflows, so anything that
    could differ per invoking workflow (``workflows.model_creation``, the
    ``--configfile`` path) cannot reach it. The thematic source names are
    deliberately absent: they belong to the raster half.

    ``region_fn`` is the rule's declared ``region_geojson`` input — the one
    project region artifact (ADR 0003 §1), not a path this function derives.
    """
    region = _region_geometry(region_fn)
    source = catalog.get_rasterdataset(
        hydrography,
        geom=region,
        buffer=10,
        single_var_as_array=False,
    )
    if source is None:
        raise ValueError(f"catalog source {hydrography!r} returned no data")
    source_ds = _as_dataset(source)
    maps, flwdir = prepare_hydrography(
        source_ds,
        region,
        resolution,
        river_uparea_km2,
    )
    basin_map, basins, outlet_by_basin = _parent_basins(maps, flwdir, region)
    maps["basin_id"] = basin_map
    gauges = _snap_gauge_points(
        read_gauge_points(gauge_points_path),
        maps,
        flwdir,
        gauge_snap_tolerance_m,
    )
    (
        subbasin_map,
        subbasins,
        catchments,
        locations,
        registry,
        _methods,
    ) = _delineate_spatial_units(
        maps,
        flwdir,
        basins,
        outlet_by_basin,
        gauges,
        max_subbasins_per_basin,
    )
    maps["subbasin_id"] = subbasin_map
    for name in maps.data_vars:
        maps[name].attrs.setdefault("source", hydrography)
        maps[name].attrs.setdefault("resolution", float(abs(maps.raster.res[0])))
    maps.raster.set_crs(source_ds.raster.crs)

    rivers = catalog.get_geodataframe(rivers_source, geom=basins)
    if rivers is None or rivers.empty:
        raise ValueError(f"catalog source {rivers_source!r} returned no rivers")
    if rivers.crs is None:
        raise ValueError(f"catalog source {rivers_source!r} has no CRS")
    rivers = rivers.to_crs(4326)
    for frame in (basins, subbasins, catchments):
        if frame.crs is None:
            raise ValueError("generated spatial geometry has no CRS")
    return SpatialUnits(
        maps=maps,
        basins=basins.to_crs(4326),
        subbasins=subbasins.to_crs(4326),
        catchments=catchments.to_crs(4326),
        rivers=rivers,
        locations=locations,
        location_registry=registry,
    )


def read_hydrography_seam(seam_fn: str | os.PathLike[str]) -> xr.Dataset:
    """Read the seam intermediate back as the raster half's starting grid.

    Two properties this reader exists for, both measured on the synthetic
    fixture (2026-08-06) against the pre-split ``spatial_maps.nc``:

    * ``mask_and_scale=False``. Every layer carries ``_FillValue`` in its
      ATTRS, and the CF mask decoder would move it into ``encoding`` and recast
      the array to float — ``basin_id`` and ``subbasin_id`` would come back as
      float64 with NaN where the fill is, and ``spatial_maps.nc`` would ship
      float identifier rasters;
    * COORDS FIRST. netCDF stores variables in creation order, and
      ``open_dataset`` yields data variables before coordinates while the
      pre-split dataset had ``y``/``x``/``spatial_ref`` first. Rebuilding the
      dataset coords-first is what makes the written file byte-identical rather
      than merely equivalent.

    The CRS re-anchor below needs an authority code. A hydrography source whose
    CRS has none keeps the WKT as stored: correct values and a correct
    projection, but ``spatial_maps.nc`` will not be byte-identical to what the
    unsplit rule wrote. No source in the shipped catalogs is in that position.
    """
    raw = xr.open_dataset(seam_fn, mask_and_scale=False)
    try:
        maps = xr.Dataset(coords=raw.coords, attrs=dict(raw.attrs))
        for name in raw.data_vars:
            maps[name] = raw[name]
        # Materialize before the handle closes: `open_dataset` is lazy, and the
        # caller writes these arrays out again.
        maps.load()
    finally:
        raw.close()
    # Re-anchor the CRS on its authority code. Reading `spatial_ref` back gives
    # pyproj a CRS built from the stored WKT, whose re-emitted WKT is POORER
    # than the catalog's -- `DATUM[...]` where the catalog had
    # `ENSEMBLE[... MEMBER ...]`, and `CONVERSION["unnamed"]`. Same coordinate
    # system, but a different string in the file, which is the difference
    # between byte-identical and merely equivalent. Measured 2026-08-06: this
    # was the ONLY difference left between the split and unsplit outputs.
    epsg = maps.raster.crs.to_epsg() if maps.raster.crs is not None else None
    if epsg is not None:
        maps.raster.set_crs(epsg)
    return maps


def prepare_spatial_maps(
    maps: xr.Dataset,
    basins: gpd.GeoDataFrame,
    config: SpatialConfig,
    catalog: DataCatalog,
) -> xr.Dataset:
    """Fold the thematic layers onto the seam grid — the raster half, WF1 only.

    ``basins`` arrives from the declared ``basins.geojson`` input in EPSG:4326
    and is reprojected onto the grid's CRS, which is what the unsplit rule
    passed: a no-op whenever the hydrography is geographic (merit_hydro is).
    """
    if basins.crs != maps.raster.crs:
        basins = basins.to_crs(maps.raster.crs)
    thematic = _thematic_maps(catalog, config, basins, maps["flow_direction"])
    merged = xr.merge([maps, thematic], compat="override")
    merged.raster.set_crs(maps.raster.crs)
    merged.attrs.update(
        spatial_contract=SPATIAL_CONTRACT_VERSION,
        hydrography_source=config.hydrography,
        river_source=config.sources.rivers,
        lulc_source=config.sources.lulc,
        lai_source=config.sources.lai,
        soil_source=config.sources.soil,
    )
    return merged


def spatial_report(
    basins: gpd.GeoDataFrame,
    subbasins: gpd.GeoDataFrame,
    registry: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize the written vector foundation.

    ``delineation_method_by_basin`` is DERIVED from ``subbasins`` rather than
    carried across the seam: the method is a per-parent property that every one
    of that parent's subbasins already records, so recovering it needs no extra
    channel. The uniformity that makes the recovery valid is checked, not
    assumed.
    """
    per_basin = subbasins.groupby("basin_id", sort=True)["delineation_method"]
    ambiguous = sorted(int(value) for value in per_basin.nunique().loc[lambda s: s > 1].index)
    if ambiguous:
        raise ValueError(
            f"parent basins carry more than one delineation method: {ambiguous}"
        )
    methods = {int(basin): str(method) for basin, method in per_basin.first().items()}
    return {
        "contract": SPATIAL_CONTRACT_VERSION,
        "parent_basins": len(basins),
        "subbasins": len(subbasins),
        "locations": len(registry),
        "delineation_method_by_basin": methods,
        "automatic_subbasins": int(
            subbasins["delineation_method"].eq("automatic").sum()
        ),
        "gauge_subbasins": int(subbasins["delineation_method"].eq("gauge").sum()),
        "basin_id_range": [int(basins["basin_id"].min()), int(basins["basin_id"].max())],
        "subbasin_id_range": [
            int(subbasins["subbasin_id"].min()),
            int(subbasins["subbasin_id"].max()),
        ],
        "wflow_id_range": [int(registry["wflow_id"].min()), int(registry["wflow_id"].max())],
    }


def _catalog_dict() -> dict[str, Any]:
    """Return the portable HydroMT catalog for a written spatial product.

    ``hydrography.nc`` is deliberately absent: it is the seam between the two
    halves, not a product (ADR 0003 §8a). Adding it would advertise an
    intermediate to ``build_wflow_model``, which resolves every model input
    through this catalog.
    """
    entries: dict[str, Any] = {
        "spatial_maps": {
            "data_type": "RasterDataset",
            "uri": "spatial_maps.nc",
            "driver": {"name": "raster_xarray"},
            "metadata": {"category": "topography", "contract": SPATIAL_CONTRACT_VERSION},
        },
        "location_registry": {
            "data_type": "DataFrame",
            "uri": "location_registry.csv",
            "driver": {"name": "pandas"},
            "metadata": {"category": "hydrography", "contract": SPATIAL_CONTRACT_VERSION},
        },
    }
    for name in ("basins", "subbasins", "catchments", "rivers", "locations"):
        entries[name] = {
            "data_type": "GeoDataFrame",
            "uri": f"geoms/{name}.geojson",
            "driver": {"name": "pyogrio"},
            "metadata": {
                "category": "hydrography",
                "contract": SPATIAL_CONTRACT_VERSION,
                "crs": 4326,
            },
        }
    return entries


def _write_dataset(data: xr.Dataset, path: Path) -> None:
    """Serialize one dataset through a temporary sibling, then swap it in."""
    temporary_path = path.parent / f"{path.stem}.tmp.nc"
    data.to_netcdf(temporary_path)
    temporary_path.replace(path)


def write_spatial_units(units: SpatialUnits, output_dir: str | Path) -> None:
    """Serialize the six vector artifacts and the hydrography grid seam.

    ``spatial_catalog.yml`` is NOT written here. It is the model-build
    interface, it lists the raster entries too, and a file with two writers is
    the R7-1 anti-pattern — so it stays whole in the raster half (ADR 0003 §8,
    *Neutral*).
    """
    output_dir = Path(output_dir)
    geoms_dir = output_dir / "geoms"
    geoms_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset(units.maps, output_dir / HYDROGRAPHY_SEAM_NAME)
    for name in ("basins", "subbasins", "catchments", "rivers", "locations"):
        frame = getattr(units, name)
        frame.to_file(geoms_dir / f"{name}.geojson", driver="GeoJSON")
    units.location_registry.to_csv(output_dir / "location_registry.csv", index=False)


def write_spatial_maps(
    maps: xr.Dataset, report: dict[str, Any], output_dir: str | Path
) -> None:
    """Serialize the thematic map stack, its HydroMT catalog, and the report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset(maps, output_dir / "spatial_maps.nc")
    with (output_dir / "spatial_catalog.yml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_catalog_dict(), stream, sort_keys=False)
    with (output_dir / "spatial_report.yml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)


def _validate_flow_topology(maps: xr.Dataset) -> None:
    """Validate D8 codes and non-decreasing accumulation downstream."""
    direction = np.asarray(maps["flow_direction"].values)
    finite_direction = np.isfinite(direction)
    codes = {int(value) for value in np.unique(direction[finite_direction])}
    valid_codes = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    if not codes.issubset(valid_codes):
        raise ValueError(f"flow_direction contains invalid ArcGIS D8 codes: {codes}")

    basin_values = np.asarray(maps["basin_id"].values)
    active = np.isfinite(basin_values) & (basin_values > 0)
    if np.any(active & ~finite_direction):
        raise ValueError("active basin cells contain missing flow direction")
    flow_da = xr.DataArray(
        np.where(finite_direction, direction, 247).astype("uint8"),
        coords=maps["flow_direction"].coords,
        dims=maps["flow_direction"].dims,
    )
    flow_da.raster.set_crs(maps.raster.crs)
    flow_da.raster.set_nodata(247)
    flow_network = flw.flwdir_from_da(
        flow_da,
        ftype="d8",
        mask=xr.DataArray(active, coords=flow_da.coords, dims=flow_da.dims),
    )
    active_indices = np.flatnonzero(active.ravel())
    downstream = flow_network.idxs_ds[active_indices]
    internal = (downstream >= 0) & (downstream != active_indices)
    internal &= active.ravel()[downstream]
    accumulation = np.asarray(maps["flow_accumulation"].values).ravel()
    if not np.isfinite(accumulation[active_indices]).all():
        raise ValueError("active basin cells contain missing flow accumulation")
    if np.any(accumulation[downstream[internal]] < accumulation[active_indices[internal]]):
        raise ValueError("flow accumulation decreases along a downstream D8 edge")


def _validate_vector_relations(geoms: Mapping[str, gpd.GeoDataFrame]) -> None:
    """Validate incremental-unit disjointness and catchment containment."""
    for name, frame in geoms.items():
        if frame.empty or not frame.geometry.is_valid.all():
            raise ValueError(f"written {name} geometry is empty or invalid")

    subbasins = geoms["subbasins"].to_crs(6933)
    catchments = geoms["catchments"].to_crs(6933)
    if not subbasins["subbasin_id"].is_unique:
        raise ValueError("subbasin vector identifiers are not unique")
    if not catchments["subbasin_id"].is_unique:
        raise ValueError("catchment vector identifiers are not unique")
    union_area = float(subbasins.geometry.union_all().area)
    summed_area = float(subbasins.geometry.area.sum())
    tolerance = max(union_area * 1e-9, 0.01)
    if summed_area - union_area > tolerance:
        raise ValueError("incremental subbasin polygons overlap")

    pairs = subbasins[["subbasin_id", "geometry"]].merge(
        catchments[["subbasin_id", "geometry"]],
        on="subbasin_id",
        how="outer",
        suffixes=("_subbasin", "_catchment"),
        validate="one_to_one",
        indicator=True,
    )
    if not pairs["_merge"].eq("both").all():
        raise ValueError("subbasin and catchment vector identifiers disagree")
    outside = [
        int(row.subbasin_id)
        for row in pairs.itertuples(index=False)
        if not row.geometry_subbasin.covered_by(row.geometry_catchment.buffer(0.01))
    ]
    if outside:
        raise ValueError(f"subbasins fall outside their full catchments: {outside}")


def validate_written_spatial_products(output_dir: str | Path) -> None:
    """Open every generated catalog entry and verify the core ID joins."""
    output_dir = Path(output_dir)
    catalog = DataCatalog(data_libs=str(output_dir / "spatial_catalog.yml"))
    maps = catalog.get_rasterdataset("spatial_maps", single_var_as_array=False)
    registry = catalog.get_dataframe("location_registry")
    geoms = {
        name: catalog.get_geodataframe(name)
        for name in ("basins", "subbasins", "catchments", "rivers", "locations")
    }
    subbasins = geoms["subbasins"]
    if maps is None or maps.raster.crs is None:
        raise ValueError("written spatial_maps has no readable CRS")
    if registry is None or subbasins is None:
        raise ValueError("written registry or subbasins catalog entry is unreadable")
    unreadable = [name for name, frame in geoms.items() if frame is None or frame.crs is None]
    if unreadable:
        raise ValueError(f"written spatial geometries are unreadable: {unreadable}")
    if any(frame.crs.to_epsg() != 4326 for frame in geoms.values()):
        raise ValueError("written spatial geometries must use EPSG:4326")
    for name, data in maps.data_vars.items():
        missing = sorted(
            {"source", "resolution", "units"}.difference(data.attrs)
        )
        if missing:
            raise ValueError(f"written spatial map {name!r} lacks metadata: {missing}")
        if data.raster.nodata is None:
            raise ValueError(f"written spatial map {name!r} lacks nodata metadata")
    raster_values = np.asarray(maps["subbasin_id"].values)
    valid_raster_values = raster_values[
        np.isfinite(raster_values) & (raster_values > 0)
    ]
    raster_ids = {int(value) for value in np.unique(valid_raster_values)}
    vector_ids = set(subbasins["subbasin_id"].astype(int))
    registry_ids = set(registry["subbasin_id"].astype(int))
    if raster_ids != vector_ids or not registry_ids.issubset(vector_ids):
        raise ValueError(
            "subbasin IDs disagree between raster, vector, and location registry"
        )
    required_registry_columns = {
        "basin_id",
        "basin_code",
        "basin_name",
        "subbasin_id",
        "subbasin_code",
        "subbasin_name",
        "location_id",
        "location_code",
        "station_name",
        "wflow_id",
        "location_role",
        "original_x",
        "original_y",
        "snapped_x",
        "snapped_y",
    }
    missing_registry = sorted(required_registry_columns.difference(registry.columns))
    if missing_registry:
        raise ValueError(f"location registry lacks columns: {missing_registry}")
    _validate_flow_topology(maps)
    _validate_vector_relations(geoms)
    maps.close()
