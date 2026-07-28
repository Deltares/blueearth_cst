"""Tests for the wf1 raw-climate extraction wrapper + parse-time eobs guard.

- bbox derivation (design ext1-5): the staticmaps-derived bounds must sit
  within 2 x model_resolution of the region file's tight bounds per edge
  (outward snapping of the model grid). Fixture-dependent: skipped when the
  untracked test_case/test_local tree is absent.
- eobs exclusion (design ext2-3): `clim_historical: eobs` must red the wf1
  dry-run at DAG-parse time with the named configuration error.
"""
import os
import subprocess
import sys
from os.path import dirname, join, realpath

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blueearth_cst.model.extract_climate_wf1 import staticmaps_bbox  # noqa: E402

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

_FIXTURE_STATICMAPS = join(SNAKEDIR, "test_case", "test_local", "hydrology_model", "staticmaps.nc")
_FIXTURE_REGION = join(
    SNAKEDIR, "test_case", "test_local", "hydrology_model", "staticgeoms", "region.geojson"
)
_FIXTURE_CONFIG = join(SNAKEDIR, "config", "workflows", "snake_config_model_test.yml")


@pytest.mark.skipif(
    not (os.path.exists(_FIXTURE_STATICMAPS) and os.path.exists(_FIXTURE_REGION)),
    reason="untracked test_case/test_local fixture tree not present",
)
def test_staticmaps_bbox_within_two_cells_of_region_bounds():
    import geopandas as gpd

    with open(_FIXTURE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    resolution = cfg["shared"]["basin"].get("resolution", 0.00833333)

    bbox_sm = staticmaps_bbox(_FIXTURE_STATICMAPS)
    bbox_region = tuple(gpd.read_file(_FIXTURE_REGION).geometry.total_bounds)

    offsets = [abs(a - b) for a, b in zip(bbox_sm, bbox_region)]
    tol = 2 * resolution
    assert all(off <= tol for off in offsets), (
        f"per-edge offsets {offsets} exceed 2*model_resolution={tol} "
        f"(staticmaps {bbox_sm} vs region {bbox_region})"
    )
    # Recorded for the baseline note (visible with pytest -s):
    print(f"bbox per-edge offsets (deg): {offsets}; tol={tol}")


def test_eobs_config_fails_wf1_dry_run_at_parse_time(tmp_path):
    with open(join(TESTDIR, "snake_config_model_test.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["shared"]["clim_historical"] = "eobs"
    cfg_path = tmp_path / "snake_config_eobs.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    os.chdir(SNAKEDIR)
    result = subprocess.run(
        f"snakemake all -c 1 -s Snakefile_model_creation --configfile {cfg_path} --dry-run",
        shell=True,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, "eobs config must fail the wf1 dry-run"
    assert (
        "clim_historical: eobs is not supported by the P3-2a wf1 raw-climate "
        "path; supported sources: era5, chirps, chirps_global"
    ) in combined, combined
