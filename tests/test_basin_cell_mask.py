"""`write_basin_cell_mask` — which store cells weathergenr averages over.

weathergenr resamples on an unweighted mean of every cell handed to it, and the
store is a bbox read plus `BUFFER_CELLS`, so without a mask the series steering
a stress test includes climate the basin never sees.

The selector is INTERSECTS with equal weight per cell (owner ruling
2026-08-10). Both halves matter and both are pinned here: a centre-in-polygon
test returns nothing for a sub-cell basin, and equal weighting is what makes
weathergenr's own unweighted mean correct for the subset.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from blueearth_cst.climate_analysis.extract_historical_climate import (
    BUFFER_CELLS,
    write_basin_cell_mask,
)


def _store(tmp_path, lats, lons):
    """A minimal store netCDF carrying only the coordinates the mask reads."""
    ds = xr.Dataset(
        {"precip": (("latitude", "longitude"), [[0.0] * len(lons)] * len(lats))},
        coords={"latitude": list(lats), "longitude": list(lons)},
    )
    path = tmp_path / "extract_historical.nc"
    ds.to_netcdf(path)
    return path


def _region(geom):
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def test_the_forcing_buffer_is_one_cell():
    """Keep one source cell on each side of the rounded basin bbox."""
    assert BUFFER_CELLS == 1


def test_one_cell_buffer_preserves_basin_cells_and_nearest_forcing(tmp_path):
    source = xr.DataArray(
        np.arange(77, dtype=np.float32).reshape(7, 11),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(31.5, 29.99, -0.25),
            "longitude": np.arange(28.5, 31.01, 0.25),
        },
        name="precip",
    )
    source.raster.set_crs(4326)
    region = _region(box(28.9, 30.475, 30.65, 31.233333))
    bounds = region.total_bounds
    reference = source.raster.clip_bbox(bounds, buffer=2)
    reduced = source.raster.clip_bbox(bounds, buffer=BUFFER_CELLS)

    assert reduced.shape == (5, 9)
    reference_mask = write_basin_cell_mask(
        _store(tmp_path, reference.latitude, reference.longitude),
        region,
        tmp_path / "reference.csv",
    )
    reduced_mask = write_basin_cell_mask(
        _store(tmp_path, reduced.latitude, reduced.longitude),
        region,
        tmp_path / "reduced.csv",
    )
    pd.testing.assert_frame_equal(reduced_mask, reference_mask)

    # A target point exactly on a source cell edge is a nearest-index TIE, and
    # the two clips break it differently once rounding differs (it did on the
    # ubuntu CI leg). Unshifted, every 25th longitude here lands on an edge, so
    # the grid is offset and its clearance asserted: only unambiguous points.
    latitude = np.linspace(bounds[3] - 0.005, bounds[1] + 0.005, 76)
    longitude = np.linspace(bounds[0] + 0.005, bounds[2] - 0.005, 175) + 0.003
    for coords, origin in ((latitude, 31.5), (longitude, 28.5)):
        edges = (coords - origin) / 0.25 - 0.5
        assert (np.abs(edges - np.round(edges)) * 0.25).min() > 1e-3
    target = xr.DataArray(
        np.zeros((76, 175), dtype=np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )
    target.raster.set_crs(4326)
    expected = reference.raster.reproject_like(target, method="nearest_index")
    actual = reduced.raster.reproject_like(target, method="nearest_index")
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.values, expected.values)


def test_a_sub_cell_basin_selects_the_cells_it_touches(tmp_path):
    """The gabon_1008 geometry, which is why the selector is not centre-based.

    That basin spans 0.80 x 0.53 ERA5 cells and contains NO cell centre, so a
    centre-in-polygon mask yields an empty subset and weathergenr gets nothing.
    """
    nc = _store(tmp_path, [0.50, 0.25], [9.50, 9.75, 10.00])
    region = _region(box(9.658, 0.350, 9.858, 0.483))

    frame = write_basin_cell_mask(nc, region, tmp_path / "basin_cells.csv")

    assert len(frame) == 2, "the basin touches two cells; a centre test finds none"
    assert sorted(frame["longitude"].unique()) == [9.75]
    assert sorted(frame["latitude"]) == [0.25, 0.50]


def test_cells_the_basin_never_touches_are_dropped(tmp_path):
    """The whole point: the buffer ring must not enter the average."""
    nc = _store(tmp_path, [0.50, 0.25], [9.50, 9.75, 10.00])
    region = _region(box(9.658, 0.350, 9.858, 0.483))

    frame = write_basin_cell_mask(nc, region, tmp_path / "basin_cells.csv")

    kept = set(zip(frame["latitude"], frame["longitude"]))
    assert (0.50, 9.50) not in kept and (0.50, 10.00) not in kept


def test_the_mask_is_written_with_coordinate_headers(tmp_path):
    """The R consumer matches on coordinates, never on index order."""
    nc = _store(tmp_path, [0.50, 0.25], [9.75])
    out = tmp_path / "basin_cells.csv"
    write_basin_cell_mask(nc, _region(box(9.70, 0.20, 9.80, 0.55)), out)

    assert list(pd.read_csv(out).columns) == ["latitude", "longitude"]


def test_a_basin_covering_everything_keeps_every_cell(tmp_path):
    """No mask is still a legitimate outcome for a basin larger than the grid."""
    nc = _store(tmp_path, [0.50, 0.25], [9.50, 9.75, 10.00])
    frame = write_basin_cell_mask(
        nc, _region(box(0.0, -10.0, 20.0, 10.0)), tmp_path / "basin_cells.csv"
    )
    assert len(frame) == 6


def test_the_mask_is_never_empty(tmp_path):
    """An empty mask would hand weathergenr no data at all, and it would fail
    twenty rules away from anything that could explain it."""
    nc = _store(tmp_path, [0.50], [9.75])
    frame = write_basin_cell_mask(
        nc, _region(box(50.0, 50.0, 50.1, 50.1)), tmp_path / "basin_cells.csv"
    )
    assert len(frame) == 1


@pytest.mark.parametrize("n_lat,n_lon", [(1, 1), (1, 3), (3, 1)])
def test_degenerate_grids_do_not_crash(tmp_path, n_lat, n_lon):
    """A single row or column gives a zero half-width on that axis, so the cell
    boxes collapse; the fallback has to carry it."""
    nc = _store(
        tmp_path,
        [0.50 - 0.25 * i for i in range(n_lat)],
        [9.50 + 0.25 * i for i in range(n_lon)],
    )
    frame = write_basin_cell_mask(
        nc, _region(box(9.658, 0.350, 9.858, 0.483)), tmp_path / "basin_cells.csv"
    )
    assert len(frame) >= 1
