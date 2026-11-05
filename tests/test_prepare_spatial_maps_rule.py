"""DAG and boundary tests for Workflow 1's neutral spatial target."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config
from tests.conftest import parse_workflow, write_config

pytestmark = pytest.mark.usefixtures("isolated_snakemake")


REPO = Path(__file__).resolve().parents[1]
SNAKEFILE = REPO / "build_model.smk"
CONFIG = REPO / "tests" / "project_config_fixture.yml"


def _rule_block(name: str) -> str:
    text = SNAKEFILE.read_text(encoding="utf-8")
    match = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert match, f"rule {name} not found"
    return match.group(1)


def test_delineate_spatial_units_declares_the_vector_file_contract():
    """The six vector artifacts and the seam are files, not a directory sentinel.

    They moved here from `prepare_land_and_soil_maps` with ADR 0003 §8, and every
    content-determining field is splatted from `snake_utils.spatial_units_rule`
    so the three workflow declarations cannot drift -- which is why this test
    asserts the SPLAT rather than the paths, and
    `tests/test_spatial_units_rule.py` asserts what the splat contains.
    """
    block = _rule_block("delineate_subbasins_and_rivers")

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
    block = _rule_block("prepare_land_and_soil_maps")

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
        # Read through the spatial catalog; undeclared, an old tree never
        # re-ran the producer and this rule failed as missing DATA.
        'SPATIAL_UNITS.outputs["river_attributes"]',
        'SPATIAL_UNITS.outputs["locations"]',
        'SPATIAL_UNITS.outputs["location_registry"]',
    ):
        assert consumed in block
    assert "directory(" not in block
    assert "config_snake =" not in block
    assert "config_workflows =" not in block
    assert "basin_config = basin_cfg" in block
    assert "data_catalogs = DATA_SOURCES" in block
    assert "log:" in block and "benchmark:" in block


@pytest.fixture(scope="module")
def model_workflow():
    return parse_workflow("build_model.smk", CONFIG)


@pytest.mark.workflow_contract
def test_evaluation_changes_preserve_thematic_parameters(tmp_path, model_workflow):
    """Observation and figure settings must not rebuild the model foundation."""
    cfg = load_composed_config(CONFIG)
    cfg["workflows"]["build_model"].update(
        observations={"river discharge": "tests/data/new_observations.csv"},
        evaluation={"subbasin_plots": True},
    )
    changed = parse_workflow("build_model.smk", write_config(tmp_path, cfg))
    original_rule = model_workflow.get_rule("prepare_land_and_soil_maps")
    changed_rule = changed.get_rule("prepare_land_and_soil_maps")
    assert dict(changed_rule.params.items()) == dict(original_rule.params.items())
    assert set(changed_rule.params.model_config) <= {"output_locations"}
    original_evaluation = model_workflow.get_rule("plot_model_evaluation")
    changed_evaluation = changed.get_rule("plot_model_evaluation")
    assert changed_evaluation.params.subbasin_plots is True
    assert changed_evaluation.input != original_evaluation.input


@pytest.mark.workflow_contract
def test_thematic_parameter_projection_preserves_legacy_gauge_rejection(tmp_path):
    cfg = load_composed_config(CONFIG)
    cfg["basin"]["output_locations"] = "tests/data/canonical_gauges.csv"
    cfg["workflows"]["build_model"]["output_locations"] = "tests/data/other_gauges.csv"
    with pytest.raises(ValueError, match="'output_locations' belongs in the project"):
        parse_workflow("build_model.smk", write_config(tmp_path, cfg))


@pytest.mark.workflow_contract
def test_native_simulation_csv_is_retained_for_evaluation(model_workflow):
    simulation = model_workflow.get_rule("run_historical_simulation")
    evaluation = model_workflow.get_rule("plot_model_evaluation")
    tables = model_workflow.get_rule("export_simulation_tables")
    assert simulation.output.csv_path not in simulation.temp_output
    assert simulation.output.csv_path in evaluation.input
    assert simulation.output.csv_path in tables.input


@pytest.mark.workflow_contract
@pytest.mark.parametrize(
    "rule_name,expected",
    [
        (
            "prepare_land_and_soil_maps",
            {
                "spatial/config.py",
                "spatial/products.py",
            },
        ),
        (
            "add_climate_forcing",
            {
                "climate_analysis/prepare_climate_data_catalog.py",
                "shared/setup_time_horizon.py",
                "shared/climate_window.py",
            },
        ),
        ("add_reservoirs_lakes_glaciers", {"shared/wflow_write.py"}),
        (
            "run_historical_simulation",
            {"model/run_wflow.jl", "shared/wflow_progress.jl"},
        ),
        (
            "declare_gauges_and_outputs",
            {"shared/wflow_write.py", "shared/wflow_outputs.py"},
        ),
    ],
)
def test_model_producers_track_their_scientific_helpers(
    model_workflow, rule_name, expected
):
    rule = model_workflow.get_rule(rule_name)
    declared = {
        str(path).replace("\\", "/").split("blueearth_cst/", 1)[1]
        for path in rule.input.scientific_code
    }
    assert declared == expected
    assert all((REPO / "blueearth_cst" / path).is_file() for path in declared)


def test_the_vector_artifacts_have_exactly_one_producing_rule():
    """A path declared as an output twice is a Snakemake ambiguity, not a share.

    The split moved six `output:` entries from one rule to another; leaving a
    copy behind is the mistake this catches.
    """
    raster = _rule_block("prepare_land_and_soil_maps")
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
        ("prepare_land_and_soil_maps", "prepare_spatial_maps.py"),
        ("delineate_subbasins_and_rivers", "delineate_spatial_units.py"),
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


@pytest.mark.workflow_contract
def test_spatial_only_dry_run_has_no_wflow_edge():
    """A direct target schedules exactly P1, not the existing model build."""
    result = subprocess.run(
        [
            "snakemake",
            "prepare_land_and_soil_maps",
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
    assert "prepare_land_and_soil_maps" in combined
    # ...and it schedules its own producer, which is what a split rule must do.
    assert "delineate_subbasins_and_rivers" in combined
    for forbidden_rule in (
        "build_wflow_model",
        "add_reservoirs_lakes_glaciers",
        "declare_gauges_and_outputs",
    ):
        assert forbidden_rule not in combined, combined[-3000:]
