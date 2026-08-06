"""R07 B1: the store's region and the built model agree on the basin.

Successor to the retired ``test_extract_climate_wf1.py``
``test_staticmaps_bbox_within_two_cells_of_region_bounds``. That test compared
the two *pre-R07* bbox derivations (``staticmaps.nc`` raster bounds vs
``staticgeoms/region.geojson``); both are now gone from the extraction path,
which delineates model-free from ``shared.basin`` + the catalog.

What survives it is the configuration-independent invariant: the store's
delineated region (``spatial/geoms/region.geojson``) and the model grid
(``staticmaps.nc``) must describe the same basin. Raster bounds are snapped
outward to the model grid, so the tolerance is 2 x model resolution per edge —
the same band the retired test used. On the seed fixture the two agree far more
tightly than that (the 2026-07-28 bounds probe put them bit-identical); the band
exists so the assertion holds for any region/resolution, not just the seed.

Fixture-gated exactly as its predecessor was: it needs a completed run under
``test_case/test_local``. One skip retired, one skip added — net zero.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SNAKEDIR = Path(__file__).resolve().parents[1]
from blueearth_cst.shared.snake_utils import region_rule  # noqa: E402

SEED_CONFIG = SNAKEDIR / "config" / "workflows" / "snake_config_model_test.yml"


def _seed_paths():
    """(region.geojson, staticmaps.nc, resolution) for the seed fixture.

    ADR 0003: the polygon is the one project artifact under
    ``spatial/geoms/``, not a per-store-key copy. The claim under test is
    unchanged — the delineated region and the model grid must agree.
    """
    cfg = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    project_dir = SNAKEDIR / cfg["project"]["project_dir"]
    basin_cfg = cfg["shared"]["basin"]
    spec = region_rule(
        project_dir=project_dir.as_posix(),
        model_region=basin_cfg["region"],
        data_sources=cfg["project"]["data_sources"],
    )
    return (
        Path(spec.region_geojson),
        project_dir / "hydrology_model" / "staticmaps.nc",
        basin_cfg.get("resolution", 0.00833333),
    )


def test_store_region_agrees_with_the_model_grid():
    # Config read lazily inside the test (repo convention): a config-schema
    # change must not be able to break suite COLLECTION.
    region_fn, staticmaps_fn, resolution = _seed_paths()
    if not (region_fn.exists() and staticmaps_fn.exists()):
        pytest.skip(
            "needs a completed run under test_case/test_local "
            "(spatial/geoms/region.geojson + staticmaps.nc)"
        )

    import geopandas as gpd
    import hydromt  # noqa: F401 -- registers the xarray .raster accessor
    import xarray as xr

    region_bounds = tuple(gpd.read_file(region_fn).total_bounds)
    with xr.open_dataset(staticmaps_fn) as ds:
        model_bounds = tuple(ds.raster.bounds)

    offsets = [abs(a - b) for a, b in zip(region_bounds, model_bounds)]
    tol = 2 * resolution
    assert all(offset <= tol for offset in offsets), (
        f"per-edge offsets {offsets} exceed 2*model_resolution={tol} "
        f"(store region {region_bounds} vs staticmaps {model_bounds})"
    )
    # Recorded for the baseline note (visible with pytest -s):
    print(f"store-region vs model-grid per-edge offsets (deg): {offsets}; tol={tol}")
