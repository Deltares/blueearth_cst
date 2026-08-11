"""Snakemake entry point for Workflow 1's thematic spatial map stack.

The RASTER half of ADR 0003 §8, and WF1 only: it exists to parameterise Wflow.
The vector layers and the location registry it used to produce alongside
``spatial_maps.nc`` now come from rule ``delineate_spatial_units``, which all
three workflows declare — this rule consumes them, adds the LULC/LAI/soil
layers, and writes the catalog and report.
"""

import gc
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from hydromt import DataCatalog

from blueearth_cst.spatial.config import parse_spatial_config
from blueearth_cst.spatial.products import (
    prepare_spatial_maps,
    read_hydrography_seam,
    spatial_report,
    validate_written_spatial_products,
    write_spatial_maps,
)


def _catalog_paths(value: str | os.PathLike[str] | Sequence[object]) -> list[str]:
    """Normalize one or several Snakemake catalog inputs."""
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(item) for item in value]


def run_prepare_spatial_maps(
    basin_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    data_catalogs: str | os.PathLike[str] | Sequence[object],
    output_dir: str | os.PathLike[str],
    hydrography_fn: str | os.PathLike[str],
    basins_fn: str | os.PathLike[str],
    subbasins_fn: str | os.PathLike[str],
    location_registry_fn: str | os.PathLike[str],
) -> None:
    """Fold the thematic layers onto the seam grid, then reopen and validate.

    ``hydrography_fn`` is the seam intermediate rule
    ``delineate_spatial_units`` wrote (ADR 0003 §8a) — the grid stack that used
    to cross this boundary in memory. The three vector paths are that rule's
    declared outputs, read back for the thematic clip geometry and the report.
    """
    config = parse_spatial_config(basin_config, model_config)
    catalog = DataCatalog(data_libs=_catalog_paths(data_catalogs))
    seam = read_hydrography_seam(hydrography_fn)
    basins = gpd.read_file(basins_fn)
    maps = prepare_spatial_maps(seam, basins, config, catalog)
    try:
        report = spatial_report(
            basins,
            gpd.read_file(subbasins_fn),
            pd.read_csv(location_registry_fn),
        )
        write_spatial_maps(maps, report, output_dir)
        validate_written_spatial_products(output_dir)
    finally:
        # Rasterio/GDAL-backed lazy arrays otherwise survive until interpreter
        # shutdown on Windows and can emit a large benign sys.excepthook cascade.
        #
        # `del catalog` is not decoration: this collect ran with `catalog` still
        # BOUND until 2026-08-11, so the object holding the GDAL handles was
        # still reachable and the collector could not claim it — the comment
        # above promised a mitigation the code could not deliver.
        # `tee_to_log` now collects again on the way out, which covers whatever
        # is released after this point; this one still runs first, so the
        # handles go while the rule is at its healthiest.
        maps.close()
        del catalog
        gc.collect()


if __name__ == "__main__" and "snakemake" in globals():
    sm = globals()["snakemake"]
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        run_prepare_spatial_maps(
            basin_config=sm.params.basin_config,
            model_config=sm.params.model_config,
            data_catalogs=sm.input.data_catalogs,
            output_dir=Path(sm.output.spatial_maps).parent,
            hydrography_fn=sm.input.hydrography,
            basins_fn=sm.input.basins,
            subbasins_fn=sm.input.subbasins,
            location_registry_fn=sm.input.location_registry,
        )
