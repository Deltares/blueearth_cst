# Current guidance: dev/reference/toolbox-spatial-foundation.md
"""Delineate the project region once from basin settings and the data catalog.

Shared workflow declarations use ``region_rule`` to produce the same region
artifact. Climate extraction and spatial products consume it independently of
model construction. No built model or climate-store extraction is needed to
obtain the polygon.
"""

# NO `from __future__ import annotations` here, deliberately: Snakemake's
# `script:` directive executes this module through a generated wrapper that
# PREPENDS its own preamble, and a future import is then no longer the first
# statement — "SyntaxError: from __future__ imports must occur at the beginning
# of the file", at rule run time rather than at import. Every other `script:`
# module in this repo omits it for the same reason.
import ast
import gc
import os
from numbers import Real
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import hydromt
from hydromt.model.processes.region import parse_region_basin

from blueearth_cst.shared.snake_utils import (
    DEFAULT_BASIN_INDEX,
    DEFAULT_HYDROGRAPHY,
    log_row,
)


def delineate_region(
    model_region,
    data_libs: Union[str, Path],
    *,
    hydrography: str = DEFAULT_HYDROGRAPHY,
    basin_index: str = DEFAULT_BASIN_INDEX,
    region_out: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """Delineate the project region from the region spec + catalog.

    ``hydrography``/``basin_index`` are catalog ENTRY NAMES, not paths — hydromt
    resolves them against ``data_libs`` itself (verified on the pinned hydromt
    1.3.1). They default to the shipped build template's ``setup_basemaps``
    values; rule 1.01 raises if the two ever disagree.

    Parameters
    ----------
    model_region : str | dict
        ``shared.basin.region``. A Python-dict-literal string (the form the
        project config carries, e.g. ``"{'subbasin': [9.666, 0.4476],
        'uparea': 100}"``) is parsed with ``ast.literal_eval``, matching
        ``prepare_build_config.merge_build_config``.
    data_libs : str | Path
        Data catalog(s) to resolve the hydrography sources against.
    hydrography, basin_index : str
        Catalog entry names for the flow-direction data and its basin index.
    region_out : str | Path, optional
        When given, the delineated GeoDataFrame is written there as GeoJSON
        (parents created) — the rule's ``region_geojson`` output.

    Returns
    -------
    geopandas.GeoDataFrame
        The delineated region; ``.total_bounds`` is the extraction bbox.
    """
    if isinstance(model_region, str):
        model_region = ast.literal_eval(model_region)

    data_catalog = hydromt.DataCatalog(data_libs=data_libs)
    try:
        log_row(
            f"Delineating basin from {model_region} using {hydrography}",
            module="spatial",
        )
        try:
            gdf = parse_region_basin(
                model_region,
                data_catalog=data_catalog,
                hydrography_path=hydrography,
                basin_index_path=basin_index,
            )
        except ValueError as error:
            if str(error).startswith(
                (
                    "Invalid raster: less than 2 cells in x_dim",
                    "Invalid raster: less than 2 cells in y_dim",
                )
            ):
                message = (
                    "Basin delineation failed: the selected raster extent is too small "
                    f"from basin.region={model_region!r}. Check the outlet lies on "
                    "the intended drainage channel and the hydrography/index pairing; "
                    "consider a bounded subbasin outlet search area."
                )
                log_row(message, module="spatial", level="WARNING")
                raise ValueError(message) from error
            raise
        if gdf.empty:
            raise ValueError("shared.basin.region resolved to no parent basins")
        if gdf.crs is None:
            raise ValueError("resolved parent-basin geometry has no CRS")
        selection = model_region.get("subbasin")
        if (
            isinstance(selection, (list, tuple))
            and "xy" not in model_region
            and len(selection) == 4
            and all(isinstance(value, Real) for value in selection)
        ):
            log_row(
                f"Outlet selected within search area {selection}; inspect the basin boundary "
                "and output locations because no outlet point was supplied",
                module="spatial",
                level="WARNING",
            )
        if region_out is not None:
            parent = os.path.dirname(os.fspath(region_out))
            if parent:
                os.makedirs(parent, exist_ok=True)
            gdf.to_file(region_out, driver="GeoJSON")
            log_row(f"Basin boundary written → {region_out}", module="spatial")
        return gdf
    finally:
        # Drop the catalog and collect while the interpreter is still HEALTHY.
        # `parse_region_basin` opens the hydrography glob
        # (`merit_hydro_ihu/30sec/*.tif`) and the basin-index GeoPackage through
        # this catalog, and nothing else releases them -- so every GDAL/rasterio
        # handle survived to interpreter finalization, where tearing them all
        # down at once on Windows makes a stderr write fail. CPython's excepthook
        # cannot run that late (module globals are already gone), so it prints a
        # bare `Error in sys.excepthook:` / `Original exception was:` pair with
        # empty bodies, repeatedly, after a rule that SUCCEEDED.
        #
        # Same guard, same reason, as `prepare_spatial_maps.py`'s close + collect.
        # Measured on the tests fixture: 14 cascade lines before, 0 after.
        # In a `finally` because the raises above are failure paths that also
        # end the process.
        #
        # NOTE: this is not universal -- `extract_historical_climate.py` records
        # `ds.close()` NOT helping there (14 lines before and after), so treat
        # the release as something to measure per rule, never to assume.
        del data_catalog
        gc.collect()


def read_region(region_fn: Union[str, Path]) -> gpd.GeoDataFrame:
    """Read the shared region artifact, with the producer's guarantees rechecked.

    Consumers read a DECLARED input, so the file exists whenever Snakemake runs
    them. The two content guarantees are rechecked rather than assumed: a
    hand-edited or truncated region is a misconfigured project, and the failure
    should name the region rather than surface as an empty bbox twenty rules
    downstream.
    """
    gdf = gpd.read_file(region_fn)
    if gdf.empty:
        raise ValueError(f"{region_fn} holds no geometry")
    if gdf.crs is None:
        raise ValueError(f"{region_fn} has no CRS")
    return gdf


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            delineate_region(
                sm.params.model_region,
                sm.input.catalog,
                hydrography=sm.params.hydrography,
                basin_index=sm.params.basin_index,
                region_out=sm.output.region_geojson,
            )
