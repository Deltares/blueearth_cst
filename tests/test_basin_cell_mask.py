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


def test_the_forcing_buffer_is_two_cells():
    """hydromt reads precip for a model region with buffer=2
    (hydromt_wflow/wflow_sbm.py, setup_precip_forcing). A store built narrower
    is one ring short of what that reader sees."""
    assert BUFFER_CELLS == 2


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
