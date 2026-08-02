"""Tests for model-neutral hydrography and topography preparation."""

from __future__ import annotations

import inspect

import geopandas as gpd
import numpy as np
import pyflwdir
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import box

from blueearth_cst.spatial import hydrography
from blueearth_cst.spatial.hydrography import prepare_hydrography


def _source_dataset(size: int = 6) -> tuple[xr.Dataset, gpd.GeoDataFrame]:
    """Return a projected synthetic hydrography source and full-domain region."""
    elevation = np.add.outer(
        np.arange(size, 0, -1, dtype=float),
        np.arange(size, 0, -1, dtype=float),
    )
    transform = Affine(1000, 0, 0, 0, -1000, size * 1000)
    flwdir = pyflwdir.from_dem(elevation, transform=transform, outlets="min")
    x = np.arange(500, size * 1000, 1000)
    y = np.arange(size * 1000 - 500, 0, -1000)
    coords = {"y": y, "x": x}
    flow = xr.DataArray(
        flwdir.to_array(ftype="d8"), dims=("y", "x"), coords=coords, name="flwdir"
    )
    elev = xr.DataArray(elevation, dims=("y", "x"), coords=coords, name="elevtn")
    upstream = xr.DataArray(
        flwdir.upstream_area("km2"), dims=("y", "x"), coords=coords, name="uparea"
    )
    source = xr.merge([flow, elev, upstream])
    source.raster.set_crs(3857)
    source["flwdir"].raster.set_nodata(247)
    source["elevtn"].raster.set_nodata(-9999.0)
    region = gpd.GeoDataFrame(
        geometry=[box(0, 0, size * 1000, size * 1000)], crs=3857
    )
    return source, region


def test_neutral_hydrography_contract_has_expected_maps_and_metadata():
    """The base product is self-describing and contains no Wflow aliases."""
    source, region = _source_dataset()

    result, flwdir = prepare_hydrography(source, region, 1000, 2.0)

    assert flwdir.shape == result.raster.shape
    assert {
        "flow_direction",
        "flow_accumulation",
        "cell_area",
        "upstream_area",
        "river_mask",
        "river_order",
        "elevation",
        "slope",
    }.issubset(result.data_vars)
    assert result.raster.crs.to_epsg() == 3857
    assert result["flow_direction"].attrs["encoding"].startswith("ArcGIS D8")
    assert result["upstream_area"].attrs["units"] == "km2"
    assert result["cell_area"].attrs["units"] == "km2"
    assert result["elevation"].attrs["units"] == "m"
    assert result["slope"].attrs["units"] == "m m-1"
    assert "local_drain_direction" not in result
    assert "subcatchment" not in result


def test_hydrography_can_upscale_to_an_integer_multiple():
    """The neutral grid can be coarsened without importing the Wflow plugin."""
    source, region = _source_dataset()

    result, flwdir = prepare_hydrography(source, region, 2000, 2.0)

    assert result.raster.res == pytest.approx((2000.0, -2000.0))
    assert flwdir.shape == result.raster.shape
    assert all(size >= 2 for size in result.raster.shape)


def test_non_integer_upscale_ratio_is_rejected():
    """Grid alignment is explicit rather than rounded silently."""
    source, region = _source_dataset()

    with pytest.raises(ValueError, match="integer multiple"):
        prepare_hydrography(source, region, 1600, 2.0)


def test_analysis_grid_trims_inactive_alignment_border():
    """Grid alignment cannot retain an empty row or column around the basin."""
    source, _ = _source_dataset()
    region = gpd.GeoDataFrame(
        geometry=[box(1000, 1000, 6000, 6000)], crs=3857
    )

    result, flow = prepare_hydrography(source, region, 1000, 2.0)

    assert result.raster.shape == (5, 5)
    assert flow.shape == (5, 5)
    assert result.raster.bounds == pytest.approx((1000, 1000, 6000, 6000))


def test_spatial_hydrography_module_does_not_import_hydromt_wflow():
    """Import independence is a source-level falsifier for the P1 boundary."""
    assert "hydromt_wflow" not in inspect.getsource(hydrography)
