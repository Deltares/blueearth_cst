"""DAG and boundary tests for Workflow 1's neutral spatial target."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SNAKEFILE = REPO / "Snakefile_model_creation"
CONFIG = REPO / "tests" / "snake_config_model_test.yml"


def _rule_block(name: str) -> str:
    text = SNAKEFILE.read_text(encoding="utf-8")
    match = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert match, f"rule {name} not found"
    return match.group(1)


def test_delineate_spatial_units_declares_the_vector_file_contract():
    """The six vector artifacts and the seam are files, not a directory sentinel.

    They moved here from `prepare_spatial_maps` with ADR 0003 §8, and every
    content-determining field is splatted from `snake_utils.spatial_units_rule`
    so the three workflow declarations cannot drift -- which is why this test
    asserts the SPLAT rather than the paths, and
    `tests/test_spatial_units_rule.py` asserts what the splat contains.
    """
    block = _rule_block("delineate_spatial_units")

    for splat in (
        "**SPATIAL_UNITS.inputs",
        "**SPATIAL_UNITS.params",
        "**SPATIAL_UNITS.outputs",
    ):
        assert splat in block
    assert "directory(" not in block
    assert "script: SPATIAL_UNITS.script" in block
    assert "log:" in block and "benchmark:" in block
    # The §8b consequence, as a property of the DECLARATION: the config path
    # differs between a full config and a projections-only one, so declaring it
    # would thrash the shared rule on every WF1/WF2 alternation.
    executable = "\n".join(
        line for line in block.splitlines() if not line.lstrip().startswith("#")
    )
    assert "config_snake" not in executable


def test_prepare_spatial_maps_declares_the_raster_file_contract():
    """The product is visible to Snakemake as files, not a directory sentinel.

    Three outputs since ADR 0003 §8, not nine: the raster half keeps
    `spatial_maps.nc` and the model-build interface, and consumes the vector
    layers rather than writing them.
    """
    block = _rule_block("prepare_spatial_maps")

    for path in (
        "spatial_maps.nc",
        "spatial_catalog.yml",
        "spatial_report.yml",
    ):
        assert path in block
    for consumed in (
        'SPATIAL_UNITS.outputs["hydrography"]',
        'SPATIAL_UNITS.outputs["basins"]',
        'SPATIAL_UNITS.outputs["subbasins"]',
        'SPATIAL_UNITS.outputs["catchments"]',
        'SPATIAL_UNITS.outputs["rivers"]',
        'SPATIAL_UNITS.outputs["locations"]',
        'SPATIAL_UNITS.outputs["location_registry"]',
    ):
        assert consumed in block
    assert "directory(" not in block
    assert "config_snake = config_path" in block
    assert "data_catalogs = DATA_SOURCES" in block
    assert "log:" in block and "benchmark:" in block


def test_the_vector_artifacts_have_exactly_one_producing_rule():
    """A path declared as an output twice is a Snakemake ambiguity, not a share.

    The split moved six `output:` entries from one rule to another; leaving a
    copy behind is the mistake this catches.
    """
    raster = _rule_block("prepare_spatial_maps")
    outputs = raster.split("output:", 1)[1].split("params:", 1)[0]
    for path in (
        "geoms/basins.geojson",
        "geoms/subbasins.geojson",
        "geoms/catchments.geojson",
        "geoms/rivers.geojson",
        "geoms/locations.geojson",
        "location_registry.csv",
    ):
        assert path not in outputs


@pytest.mark.parametrize(
    "rule_name,module",
    [
        ("prepare_spatial_maps", "prepare_spatial_maps.py"),
        ("delineate_spatial_units", "delineate_spatial_units.py"),
    ],
)
def test_spatial_rules_and_scripts_are_wflow_independent(rule_name, module):
    """The P1 execution surface contains no Wflow model operation."""
    block = _rule_block(rule_name)
    script = (REPO / "blueearth_cst" / "spatial" / module).read_text(encoding="utf-8")
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
    # Executable lines only: both modules CARRY a comment explaining why the
    # future import is absent, and a raw substring test reads that explanation
    # as the offence it warns about.
    executable_script = [
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    ]
    assert not any("from __future__" in line for line in executable_script), (
        "Snakemake prepends a script preamble, so future imports are not first"
    )


@pytest.mark.slow
@pytest.mark.workflow_contract
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
    # ...and it schedules its own producer, which is what a split rule must do.
    assert "delineate_spatial_units" in combined
    for forbidden_rule in (
        "build_wflow_model",
        "add_reservoirs_lakes_glaciers",
        "declare_wflow_outputs",
    ):
        assert forbidden_rule not in combined, combined[-3000:]
