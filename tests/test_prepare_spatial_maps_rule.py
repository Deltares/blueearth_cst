"""DAG and boundary tests for Workflow 1's neutral spatial target."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config
from tests.conftest import parse_workflow, write_config
from tests.snakemake_state import dry_run_job_stats

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
        plotting={"location": True},
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
@pytest.mark.parametrize("enabled", [False, True])
def test_optional_figures_preserve_spatial_maps_and_numerical_outputs(
    tmp_path, enabled
):
    cfg = load_composed_config(CONFIG)
    cfg["project"]["project_dir"] = (tmp_path / "project").as_posix()
    settings = cfg["workflows"]["build_model"]
    if enabled:
        settings["plotting"] = {"forcing": True, "location": True}
    workflow = parse_workflow("build_model.smk", write_config(tmp_path, cfg))
    missing = workflow.globals["_missing_figure_outputs"](None)
    forcing = workflow.get_rule("plot_model_forcing")
    evaluation = workflow.get_rule("plot_model_evaluation")
    assert any(path.endswith("elevation_basin.png") for path in missing)
    assert bool(set(forcing.output) & set(missing)) is enabled
    assert bool(evaluation.output.station_plots) is enabled
    assert evaluation.params.station_plots is enabled
    assert evaluation.output.indicators_csv in workflow.globals["WF1_TERMINALS"]
    assert any(
        path.endswith("output_q.csv") for path in workflow.globals["WF1_TERMINALS"]
    )


@pytest.mark.workflow_contract
@pytest.mark.parametrize(
    "field", ["forcing_plots", "evaluation.station_plots", "evaluation.subbasin_plots"]
)
def test_optional_plot_settings_reject_non_booleans(tmp_path, field):
    cfg = load_composed_config(CONFIG)
    settings = cfg["workflows"]["build_model"]
    if field.startswith("evaluation."):
        settings["evaluation"] = {field.split(".")[1]: "false"}
    else:
        settings[field] = "false"
    with pytest.raises(ValueError, match=rf"{field} must be a boolean"):
        parse_workflow("build_model.smk", write_config(tmp_path, cfg))


@pytest.mark.workflow_contract
def test_spatial_figures_can_be_disabled_without_removing_model_inputs(tmp_path):
    cfg = load_composed_config(CONFIG)
    cfg["project"]["project_dir"] = (tmp_path / "project").as_posix()
    cfg["workflows"]["build_model"]["plotting"] = {"spatial": False}
    workflow = parse_workflow("build_model.smk", write_config(tmp_path, cfg))
    spatial = workflow.get_rule("plot_basin_map")
    requested = set(workflow.globals["WF1_TERMINALS"])
    missing = set(workflow.globals["_missing_figure_outputs"](None))
    assert not set(spatial.output) & (requested | missing)
    assert any(path.endswith("spatial_catalog.yml") for path in requested)
    assert workflow.get_rule("run_historical_simulation").output.csv_path in (
        workflow.get_rule("plot_model_evaluation").input
    )


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
            {"model/run_wflow.jl", "shared/wflow_progress.jl", "shared/wflow_grid.jl"},
        ),
        (
            "declare_gauges_and_outputs",
            {
                "shared/wflow_write.py",
                "shared/wflow_outputs.py",
                "shared/grid_output.py",
            },
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
def test_spatial_only_dry_run_has_no_wflow_edge(tmp_path):
    """A direct target schedules exactly P1, not the existing model build.

    Scheduled rules come from Snakemake's RUN_INFO: the console's work plan
    also lists up-to-date rules, so its text cannot prove a rule is absent.
    """
    result, counts = dry_run_job_stats(
        tmp_path / "job_stats.json",
        [
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
    )
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.returncode == 0, combined[-3000:]
    assert counts.get("prepare_land_and_soil_maps"), counts
    # ...and it schedules its own producer, which is what a split rule must do.
    assert counts.get("delineate_subbasins_and_rivers"), counts
    for forbidden_rule in (
        "build_wflow_model",
        "add_reservoirs_lakes_glaciers",
        "declare_gauges_and_outputs",
    ):
        assert forbidden_rule not in counts, counts
