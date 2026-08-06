"""Delineate the shared vector foundation once per project (ADR 0003 §8).

Rule ``delineate_spatial_units``'s script — the SINGLE producer of
``spatial/geoms/{basins,subbasins,catchments,rivers,locations}.geojson``,
``spatial/location_registry.csv`` and the hydrography grid seam, declared
identically in all three workflows (``1.01c`` / ``2.03c`` / ``3.01f``) from
``snake_utils.spatial_units_rule``.

Before this split the layers were by-products of rule 1.02, whose real product
is ``spatial_maps.nc`` and the thematic raster stack behind it. WF2 and WF3
want basin and subbasin boundaries for figures and metrics; declaring the
unsplit rule would have made a projections-only run resample ``vito``,
``modis_lai`` and ``soilgrids`` to draw a subbasin outline — the same trade
ADR 0003 exists to have removed, one level down. Measured 2026-08-06, the split
avoids ~71% of what the unsplit alternative would add.

Model-free by construction, the property R07 B1 bought and this must not spend:
the inputs are the data catalog, the shared region polygon, and the optional
gauge-point file. Nothing here reads a built model.
"""

# NO `from __future__ import annotations` here, deliberately: Snakemake's
# `script:` directive executes this module through a generated wrapper that
# PREPENDS its own preamble, and a future import is then no longer the first
# statement. Every other `script:` module in this repo omits it for the reason
# spelled out in `delineate_region.py`.
import gc
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from hydromt import DataCatalog

from blueearth_cst.spatial.products import (
    prepare_spatial_units,
    write_spatial_units,
)


def _catalog_paths(value):
    """Normalize one or several Snakemake catalog inputs."""
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(item) for item in value]


def run_delineate_spatial_units(
    data_catalogs: Union[str, os.PathLike, Sequence[object]],
    region_fn: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    *,
    hydrography: str,
    resolution: float,
    river_uparea_km2: float,
    rivers_source: str,
    gauge_points_path: Optional[str],
    gauge_snap_tolerance_m: float,
    max_subbasins_per_basin: int,
) -> None:
    """Build and write the vector foundation plus the hydrography grid seam.

    Every keyword is a ``shared.basin`` field resolved by
    ``parse_spatial_config`` in the declaring Snakefile — spelled out one by
    one because ADR 0003 §8b requires the rule's params to be a pure function
    of ``project`` + ``shared.basin``. In particular the deprecated
    ``workflows.model_creation.output_locations`` fallback CANNOT feed this
    rule: the five projections-only configs carry no ``workflows.model_creation``
    section at all, so a params payload that depended on one would differ per
    invoking workflow — the input/params asymmetry ``ext1-02`` forbade for the
    climate store. ``shared.basin.gauge_points`` is the only source here.
    """
    catalog = DataCatalog(data_libs=_catalog_paths(data_catalogs))
    units = prepare_spatial_units(
        catalog,
        region_fn,
        hydrography=hydrography,
        resolution=resolution,
        river_uparea_km2=river_uparea_km2,
        rivers_source=rivers_source,
        gauge_points_path=gauge_points_path,
        gauge_snap_tolerance_m=gauge_snap_tolerance_m,
        max_subbasins_per_basin=max_subbasins_per_basin,
    )
    try:
        write_spatial_units(units, output_dir)
    finally:
        # Rasterio/GDAL-backed lazy arrays otherwise survive until interpreter
        # shutdown on Windows and can emit a large benign sys.excepthook cascade.
        units.maps.close()
        gc.collect()


if __name__ == "__main__" and "snakemake" in globals():
    sm = globals()["snakemake"]
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        run_delineate_spatial_units(
            sm.input.data_catalogs,
            sm.input.region_geojson,
            Path(sm.output.location_registry).parent,
            hydrography=sm.params.hydrography,
            resolution=sm.params.resolution,
            river_uparea_km2=sm.params.river_uparea_km2,
            rivers_source=sm.params.rivers_source,
            gauge_points_path=getattr(sm.input, "gauge_points", None),
            gauge_snap_tolerance_m=sm.params.gauge_snap_tolerance_m,
            max_subbasins_per_basin=sm.params.max_subbasins_per_basin,
        )
