"""Prepare model-neutral hydrography and topography on the analysis grid."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pyflwdir
import xarray as xr
from hydromt.gis import flw
from pyflwdir import FlwdirRaster, core_d8


def _validate_source(source_ds: xr.Dataset) -> None:
    """Validate the minimum hydrography source contract."""
    missing = sorted({"flwdir", "elevtn"}.difference(source_ds.data_vars))
    if missing:
        raise ValueError(f"hydrography source is missing variables: {missing}")
    if source_ds.raster.crs is None:
        raise ValueError("hydrography source has no CRS")
    if any(size < 2 for size in source_ds.raster.shape):
        raise ValueError("hydrography source must contain at least 2 cells per axis")


def _target_resolution(source_ds: xr.Dataset, requested: float) -> tuple[float, int]:
    """Resolve target resolution and integer upscale ratio."""
    source_resolution = abs(float(source_ds.raster.res[0]))
    ratio = requested / source_resolution
    if ratio < 0.75:
        raise ValueError(
            f"target resolution {requested} is finer than hydrography resolution "
            f"{source_resolution}"
        )
    if 0.75 <= ratio < 1.25:
        return source_resolution, 1
    integer_ratio = int(round(ratio))
    if not math.isclose(ratio, integer_ratio, rel_tol=0.0, abs_tol=0.05):
        raise ValueError(
            "target resolution must be an integer multiple of the hydrography "
            f"resolution; got ratio {ratio:.6g}"
        )
    return source_resolution * integer_ratio, integer_ratio


def _slope_from_elevation(elevation: xr.DataArray) -> xr.DataArray:
    """Derive non-negative land-surface slope while preserving raster metadata."""
    nodata = elevation.raster.nodata
    if nodata is None:
        nodata = -9999.0
    crs = elevation.raster.crs
    values = pyflwdir.dem.slope(
        elevtn=elevation.values,
        nodata=nodata,
        latlon=bool(crs.is_geographic),
        transform=elevation.raster.transform,
    )
    slope = xr.DataArray(
        values,
        coords=elevation.raster.coords,
        dims=elevation.raster.dims,
        name="slope",
    )
    slope.raster.set_crs(crs)
    slope.raster.set_nodata(nodata)
    return slope


def _topography(
    source_ds: xr.Dataset, grid: xr.DataArray, active_mask: xr.DataArray
) -> xr.Dataset:
    """Resample elevation and slope to the analysis grid."""
    elevation = source_ds["elevtn"]
    slope = source_ds["lndslp"] if "lndslp" in source_ds else _slope_from_elevation(elevation)
    topo = xr.merge(
        [
            elevation.raster.reproject_like(grid, method="average").rename("elevation"),
            slope.raster.reproject_like(grid, method="average").rename("slope"),
        ]
    )
    topo["elevation"] = topo["elevation"].where(active_mask, -9999.0)
    topo["slope"] = topo["slope"].clip(min=0).where(active_mask, -9999.0)
    topo["elevation"].attrs.update(
        long_name="mean land-surface elevation", units="m", _FillValue=-9999.0
    )
    topo["slope"].attrs.update(
        long_name="mean land-surface slope", units="m m-1", _FillValue=-9999.0
    )
    return topo


def prepare_hydrography(
    source_ds: xr.Dataset,
    region_geom: gpd.GeoDataFrame,
    resolution: float,
    river_uparea_km2: float,
    upscale_method: str = "ihu",
) -> tuple[xr.Dataset, FlwdirRaster]:
    """Create neutral D8, accumulation, river, elevation, and slope layers.

    Flow direction is emitted in ArcGIS D8 encoding. Wflow-specific LDD names
    and parameter maps are intentionally absent from this product.
    """
    _validate_source(source_ds)
    if region_geom.empty:
        raise ValueError("region geometry is empty")
    if region_geom.crs is None:
        raise ValueError("region geometry has no CRS")
    if region_geom.crs != source_ds.raster.crs:
        region_geom = region_geom.to_crs(source_ds.raster.crs)
    if resolution <= 0 or river_uparea_km2 <= 0:
        raise ValueError("resolution and river_uparea_km2 must be > 0")

    resolved_resolution, scale_ratio = _target_resolution(source_ds, resolution)
    clipped = source_ds.raster.clip_geom(
        region_geom, align=resolved_resolution, buffer=10
    )
    source_mask = clipped.raster.geometry_mask(region_geom)
    if not bool(source_mask.any()):
        raise ValueError("region geometry selects no hydrography cells")

    if scale_ratio == 1:
        flwdir_out = flw.flwdir_from_da(clipped["flwdir"], mask=source_mask)
        flow_grid = xr.DataArray(
            flwdir_out.to_array(ftype="d8"),
            coords=clipped.raster.coords,
            dims=clipped.raster.dims,
            name="flow_direction",
        )
        flow_grid.raster.set_crs(clipped.raster.crs)
        active_mask = source_mask.astype(bool)
    else:
        source_flwdir = flw.flwdir_from_da(clipped["flwdir"], mask=False)
        upscaled, _ = flw.upscale_flwdir(
            clipped,
            flwdir=source_flwdir,
            scale_ratio=scale_ratio,
            method=upscale_method,
            uparea_name="uparea" if "uparea" in clipped else None,
            flwdir_name="flwdir",
        )
        upscaled.raster.set_crs(clipped.raster.crs)
        mask_for_reprojection = source_mask.astype("float32")
        mask_for_reprojection.raster.set_nodata(-1.0)
        active_mask = (
            mask_for_reprojection.raster.reproject_like(upscaled, method="nearest")
            .fillna(0)
            .astype(bool)
        )
        flwdir_out = flw.flwdir_from_da(upscaled, ftype="d8", mask=active_mask)
        flow_grid = xr.DataArray(
            flwdir_out.to_array(ftype="d8"),
            coords=upscaled.raster.coords,
            dims=upscaled.raster.dims,
            name="flow_direction",
        )
        flow_grid.raster.set_crs(clipped.raster.crs)

    flow_values = np.where(active_mask.values, flow_grid.values, core_d8._mv).astype(
        "uint8"
    )
    flow_grid = xr.DataArray(
        flow_values,
        coords=flow_grid.raster.coords,
        dims=flow_grid.raster.dims,
        name="flow_direction",
        attrs={
            "long_name": "local drainage direction",
            "units": "1",
            "encoding": "ArcGIS D8 (1,2,4,8,16,32,64,128; pit=0)",
            "_FillValue": core_d8._mv,
        },
    )
    flow_grid.raster.set_crs(clipped.raster.crs)
    flow_grid.raster.set_nodata(core_d8._mv)

    upstream_area = flwdir_out.upstream_area("km2").astype("float32")
    accumulation = flwdir_out.upstream_area("cell").astype("int32")
    stream_order = flwdir_out.stream_order().astype("uint8")
    mask_values = active_mask.values.astype(bool)
    output_ds = flow_grid.to_dataset()
    output_ds["upstream_area"] = xr.DataArray(
        np.where(mask_values, upstream_area, -9999.0),
        coords=flow_grid.raster.coords,
        dims=flow_grid.raster.dims,
        attrs={"long_name": "upstream contributing area", "units": "km2", "_FillValue": -9999.0},
    )
    output_ds["flow_accumulation"] = xr.DataArray(
        np.where(mask_values, accumulation, 0),
        coords=flow_grid.raster.coords,
        dims=flow_grid.raster.dims,
        attrs={"long_name": "upstream contributing cell count", "units": "cell", "_FillValue": 0},
    )
    output_ds["river_order"] = xr.DataArray(
        np.where(mask_values, stream_order, 0),
        coords=flow_grid.raster.coords,
        dims=flow_grid.raster.dims,
        attrs={"long_name": "Strahler stream order", "units": "1", "_FillValue": 0},
    )
    output_ds["river_mask"] = xr.DataArray(
        (mask_values & (upstream_area >= river_uparea_km2)).astype("uint8"),
        coords=flow_grid.raster.coords,
        dims=flow_grid.raster.dims,
        attrs={
            "long_name": "analysis river mask",
            "units": "1",
            "_FillValue": 0,
            "upstream_area_threshold_km2": float(river_uparea_km2),
        },
    )
    output_ds = xr.merge(
        [output_ds, _topography(clipped, flow_grid, active_mask)], compat="override"
    )
    output_ds.raster.set_crs(clipped.raster.crs)
    output_ds.attrs.update(
        spatial_contract="blueearth-cst-spatial-v1",
        flow_direction_encoding="ArcGIS D8",
        resolution=float(abs(output_ds.raster.res[0])),
    )
    return output_ds, flwdir_out
