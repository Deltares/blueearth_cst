"""Delineate ``shared.basin.region`` once per project (ADR 0003).

Rule ``delineate_region``'s script — the SINGLE producer of
``spatial/geoms/region.geojson``, declared identically in all three workflows
(``1.01b`` / ``2.03b`` / ``3.01b``) from ``snake_utils.region_spec``.

Before ADR 0003 the same polygon was delineated TWICE from the same inputs:
once by rule 1.02 (``spatial/products.py::_region_geometry``, on its way to
``basins.geojson``) and once per climate-store key by the store producer
(``store_region.geojson``). Measured on a real project, the two agreed exactly
— agreement maintained by coincidence, since nothing compared them. Worse, WF2
declared the entire climate-store producer purely to obtain that polygon, so a
projections-only run paid for a multi-decade climate extraction to learn a
basin outline.

The extraction stays **model-free**, which is the property R07 B1 bought and
this must not spend: the region comes from ``shared.basin.region`` plus the
data catalog, never from a built model's ``staticmaps.nc`` or
``staticgeoms/region.geojson``.
"""

# NO `from __future__ import annotations` here, deliberately: Snakemake's
# `script:` directive executes this module through a generated wrapper that
# PREPENDS its own preamble, and a future import is then no longer the first
# statement — "SyntaxError: from __future__ imports must occur at the beginning
# of the file", at rule run time rather than at import. Every other `script:`
# module in this repo omits it for the same reason.
import ast
import os
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
    values; rule 1.02 raises if the two ever disagree.

    Parameters
    ----------
    model_region : str | dict
        ``shared.basin.region``. A Python-dict-literal string (the form the
        snake config carries, e.g. ``"{'subbasin': [9.666, 0.4476],
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
    log_row(f"Delineating region {model_region} on {hydrography}", module="spatial")
    gdf = parse_region_basin(
        model_region,
        data_catalog=data_catalog,
        hydrography_path=hydrography,
        basin_index_path=basin_index,
    )
    if gdf.empty:
        raise ValueError("shared.basin.region resolved to no parent basins")
    if gdf.crs is None:
        raise ValueError("resolved parent-basin geometry has no CRS")
    if region_out is not None:
        parent = os.path.dirname(os.fspath(region_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        gdf.to_file(region_out, driver="GeoJSON")
        log_row(f"Wrote region: {region_out}", module="spatial")
    return gdf


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
