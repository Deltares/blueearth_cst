"""DAG and boundary tests for Workflow 1's neutral spatial target."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SNAKEFILE = REPO / "Snakefile_model_creation"
CONFIG = REPO / "tests" / "snake_config_model_test.yml"


def _rule_block(name: str) -> str:
    text = SNAKEFILE.read_text(encoding="utf-8")
    match = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert match, f"rule {name} not found"
    return match.group(1)


def test_prepare_spatial_maps_declares_the_file_contract():
    """The product is visible to Snakemake as files, not a directory sentinel."""
    block = _rule_block("prepare_spatial_maps")

    for path in (
        "spatial_maps.nc",
        "geoms/basins.geojson",
        "geoms/subbasins.geojson",
        "geoms/catchments.geojson",
        "geoms/rivers.geojson",
        "geoms/locations.geojson",
        "location_registry.csv",
        "spatial_catalog.yml",
        "spatial_report.yml",
    ):
        assert path in block
    assert "directory(" not in block
    assert "config_snake = config_path" in block
    assert "data_catalogs = DATA_SOURCES" in block
    assert "**_locations_input" in block
    assert "log:" in block and "benchmark:" in block


def test_prepare_spatial_maps_rule_and_script_are_wflow_independent():
    """The P1 execution surface contains no Wflow model operation."""
    block = _rule_block("prepare_spatial_maps")
    script = (REPO / "blueearth_cst" / "spatial" / "prepare_spatial_maps.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "hydromt_wflow",
        "WflowSbmModel",
        "wflow_sbm.toml",
        "build_wflow_model",
    )
    executable_block = "\n".join(
        line for line in block.splitlines() if not line.lstrip().startswith("#")
    )

    assert not any(token in executable_block for token in forbidden)
    assert not any(token in script for token in forbidden)
    assert "from __future__" not in script, (
        "Snakemake prepends a script preamble, so future imports are not first"
    )


def test_spatial_only_dry_run_has_no_wflow_edge():
    """A direct target schedules exactly P1, not the existing model build."""
    result = subprocess.run(
        [
            "snakemake",
            "prepare_spatial_maps",
            "-c",
            "1",
            "-s",
            str(SNAKEFILE),
            "--configfile",
            str(CONFIG),
            "--dry-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.returncode == 0, combined[-3000:]
    assert "prepare_spatial_maps" in combined
    for forbidden_rule in (
        "build_wflow_model",
        "add_reservoirs_lakes_glaciers",
        "add_gauges_and_outputs",
    ):
        assert forbidden_rule not in combined, combined[-3000:]
