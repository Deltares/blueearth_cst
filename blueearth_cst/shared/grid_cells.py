"""Which cells of a climate grid a polygon covers, and the masks built from it.

ONE definition of "the cells inside this geometry", used wherever a gridded
climate field is reduced to a per-area series: the basin average and the
per-subbasin averages behind WF0's figures.

**The predicate is INTERSECTS on the cell's own box**, not a centre-in-polygon
test, and that is inherited rather than invented — ``extract_historical_climate.
write_basin_cell_mask`` established it for the file weathergenr averages over,
with the reason recorded there: a basin smaller than one ERA5 cell contains NO
cell centre, so a centre test picks zero cells exactly on the small basins CST
exists for. A subbasin is smaller than its basin by construction, so the
argument only gets stronger here.

Every selected cell counts EQUALLY (owner ruling 2026-08-10, same origin). No
fractional-area weighting: a cell either meets the polygon or it does not.

**Why this is not imported from ``extract_historical_climate``.** That module is
rule 0.04's ``script:``, and Snakemake hashes a script's file content to decide
whether the rule is stale — so editing it to export a helper re-fires every
extraction in every project tree, and in ``test_case/test_local`` that cascades
through the model build. The duplication is one function's worth and it is
deliberate; unifying the two belongs to the producer fix already planned on
``dev/tasks/t2608161450``, which edits that file anyway.
"""

# NO `from __future__ import annotations`: this module is imported by `script:`
# modules, whose Snakemake preamble displaces the first statement.
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.shared.snake_utils import log_row

#: The grid's coordinate names. The store contract is ``latitude``/``longitude``
#: (CHIRPS's ``lat``/``lon`` are normalised at extraction), and a mask is built
#: against those names because ``basin_cells.csv`` writes those two headers.
LAT, LON = "latitude", "longitude"

#: Rounding used to match coordinates against ``basin_cells.csv``, matching
#: ``weathergen/generate_weather.R``'s ``mask_key`` so the two consumers of that
#: file cannot disagree about which cells the basin touches.
CELL_KEY_DECIMALS = 6


def has_grid(ds: xr.Dataset) -> bool:
    """Does this dataset spell its grid the way a mask can be built against?"""
    return {LAT, LON} <= set(ds.dims)


def _half_steps(lats, lons):
    """Half a cell in each direction, for building cell boxes."""
    half_lat = abs(lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.0
    half_lon = abs(lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.0
    return half_lat, half_lon


def cells_touching(geometry, lats, lons) -> list:
    """The ``(lat, lon)`` centres whose CELL BOX meets ``geometry``.

    Returns them in row-major order. Empty when the geometry misses the grid
    entirely — callers decide what that means, since it is a real answer for a
    subbasin outside the extraction and a defect for a basin inside it.
    """
    from shapely.geometry import box

    half_lat, half_lon = _half_steps(lats, lons)
    return [
        (la, lo)
        for la in lats
        for lo in lons
        if geometry.intersects(
            box(lo - half_lon, la - half_lat, lo + half_lon, la + half_lat)
        )
    ]


def _mask_from_keys(ds: xr.Dataset, keys: set) -> Optional[xr.DataArray]:
    lats = [round(float(v), CELL_KEY_DECIMALS) for v in ds[LAT].values]
    lons = [round(float(v), CELL_KEY_DECIMALS) for v in ds[LON].values]
    values = np.array([[(la, lo) in keys for lo in lons] for la in lats])
    if not values.any():
        return None
    return xr.DataArray(values, dims=(LAT, LON), coords={LAT: ds[LAT], LON: ds[LON]})


def geometry_mask(ds: xr.Dataset, geometry) -> Optional[xr.DataArray]:
    """Boolean mask of the cells ``geometry`` covers, or ``None`` if it covers none."""
    if not has_grid(ds) or geometry is None or geometry.is_empty:
        return None
    lats = [float(v) for v in ds[LAT].values]
    lons = [float(v) for v in ds[LON].values]
    keys = {
        (round(la, CELL_KEY_DECIMALS), round(lo, CELL_KEY_DECIMALS))
        for la, lo in cells_touching(geometry, lats, lons)
    }
    return _mask_from_keys(ds, keys)


def cells_csv_mask(ds: xr.Dataset, cells_csv: Optional[Union[str, Path]]):
    """Boolean mask from a store's ``basin_cells.csv``, or ``None``.

    Preferred over re-deriving the basin's cells from ``basins.geojson``, and
    not merely to save the arithmetic: this file is rule 0.04's declared output
    and the one weathergenr averages over, so reading it makes the figures, the
    weather generator and the stress test agree on what "the basin" is. Two
    derivations from two polygons would be two answers.
    """
    if cells_csv is None or not Path(cells_csv).is_file():
        return None
    if not has_grid(ds):
        log_row(
            "store does not spell its grid latitude/longitude; the basin mask "
            "cannot be matched and the full extraction is used",
            module="cells",
            level="WARNING",
        )
        return None
    frame = pd.read_csv(cells_csv)
    keys = {
        (round(float(lat), CELL_KEY_DECIMALS), round(float(lon), CELL_KEY_DECIMALS))
        for lat, lon in zip(frame[LAT], frame[LON])
    }
    mask = _mask_from_keys(ds, keys)
    if mask is None:
        log_row(
            f"{Path(cells_csv).name} matched no cell in the store; the full "
            "extraction is used instead",
            module="cells",
            level="WARNING",
        )
    return mask


#: The column carrying a subbasin's identifier in the shared vector foundation
#: (rule 0.03's ``subbasins.geojson``). It is what the figure filename's
#: ``subbasin_<id>_avg`` scope is built from, so it must be stable and unique.
SUBBASIN_ID_COLUMN = "subbasin_id"


def subbasin_masks(ds: xr.Dataset, subbasins, id_column: str = SUBBASIN_ID_COLUMN):
    """``{subbasin_id: mask}`` for every subbasin that covers at least one cell.

    Ordered by id, so a figure set is written and declared in one stable order.

    A subbasin covering no cell is OMITTED rather than given an empty mask: at
    ERA5's 0.25 degrees a whole subbasin can be finer than the grid, and an
    all-NaN series would draw an empty axis under a confident filename. The
    caller reports the omission.

    The converse case is expected and is NOT an error: several subbasins of a
    small basin can select the SAME cell, so their averages coincide exactly.
    That is the coarse source failing to resolve the subbasins, which is a
    finding worth seeing rather than a defect to hide.
    """
    if subbasins is None or not len(subbasins) or not has_grid(ds):
        return {}
    if id_column not in subbasins.columns:
        log_row(
            f"subbasin layer carries no {id_column!r} column; per-subbasin "
            "figures are skipped",
            module="cells",
            level="WARNING",
        )
        return {}
    masks = {}
    for _, row in subbasins.iterrows():
        mask = geometry_mask(ds, row.geometry)
        if mask is not None:
            masks[str(row[id_column])] = mask
    return dict(sorted(masks.items()))


def masked(ds: xr.Dataset, mask) -> xr.Dataset:
    """``ds`` outside ``mask`` set to NaN, or ``ds`` unchanged when there is none.

    The spatial means downstream skip NaN, so masking here is what makes
    ``annual_series``/``monthly_spread`` an average over the polygon rather than
    over the extraction's buffered bbox.
    """
    return ds if mask is None else ds.where(mask)
