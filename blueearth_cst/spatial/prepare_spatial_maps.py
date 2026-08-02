"""Snakemake entry point for the engine-neutral Workflow 1 spatial product."""

import gc
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hydromt import DataCatalog

from blueearth_cst.spatial.config import parse_spatial_config
from blueearth_cst.spatial.products import (
    prepare_spatial_products,
    validate_written_spatial_products,
    write_spatial_products,
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
) -> None:
    """Build, write, and reopen the complete neutral spatial contract."""
    config = parse_spatial_config(basin_config, model_config)
    catalog = DataCatalog(data_libs=_catalog_paths(data_catalogs))
    products = prepare_spatial_products(config, catalog)
    try:
        write_spatial_products(products, output_dir)
        validate_written_spatial_products(output_dir)
    finally:
        # Rasterio/GDAL-backed lazy arrays otherwise survive until interpreter
        # shutdown on Windows and can emit a large benign sys.excepthook cascade.
        products.maps.close()
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
        )
