"""ADR 0003: one region artifact, one producer contract, three declarations.

``shared.basin.region`` used to be delineated twice per project — by rule 1.02
on its way to ``basins.geojson``, and once per climate-store key as
``store_region.geojson`` — and WF2 declared the entire climate-store producer
just to obtain the polygon, paying for a multi-decade extraction it never read.

These pin the replacement: ``snake_utils.region_rule`` owns the contract, and
the three workflow declarations of ``delineate_region`` may differ only in
``message`` / ``log`` / ``benchmark``. Same shape, and the same reason, as
``tests/test_climate_store_contract.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blueearth_cst.shared import snake_utils as su

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_model_test.yml"

RULE_NAME = "delineate_region"


def _spec(**overrides):
    kwargs = dict(
        project_dir="/proj",
        model_region="{'subbasin': [9.666, 0.4476], 'uparea': 100}",
        data_sources="config/catalogs/deltares_data.yml",
    )
    kwargs.update(overrides)
    return su.region_rule(**kwargs)


def test_the_artifact_sits_with_the_other_project_geoms():
    """One polygon per project, beside basins/catchments/locations."""
    spec = _spec()
    assert spec.region_geojson == "/proj/data/spatial/geoms/region.geojson"
    assert spec.outputs == {"region_geojson": spec.region_geojson}


def test_the_single_input_is_the_catalog():
    """Model-free: the delineation reads config + catalog, never a built model.

    That is the property R07 B1 bought for the climate store, and moving the
    delineation here must not spend it. A second input — staticmaps, a model
    root, anything under ``hydrology_model/`` — would.
    """
    spec = _spec()
    assert spec.inputs == {"catalog": "config/catalogs/deltares_data.yml"}


def test_params_carry_the_region_and_the_catalog_entry_names():
    spec = _spec()
    assert set(spec.params) == {"model_region", "hydrography", "basin_index"}
    assert spec.params["model_region"] == (
        "{'subbasin': [9.666, 0.4476], 'uparea': 100}"
    )


def test_defaults_match_the_spatial_contract():
    """One model-neutral source default across delineation and P1."""
    from blueearth_cst.spatial.config import parse_spatial_config

    spec = _spec()
    spatial = parse_spatial_config({"region": {"basin": [0, 0]}}, {})

    assert spec.params["hydrography"] == spatial.hydrography == "merit_hydro_ihu"
    assert spec.params["basin_index"] == spatial.basin_index == "merit_hydro_index"


def test_overrides_are_carried_through():
    spec = _spec(hydrography="merit_hydro_1k", basin_index="my_index")
    assert spec.params["hydrography"] == "merit_hydro_1k"
    assert spec.params["basin_index"] == "my_index"


def test_script_is_relative_to_the_repo_root():
    """One relative path serves all three Snakefiles (`script:` → basedir)."""
    spec = _spec()
    assert spec.script == "blueearth_cst/spatial/delineate_region.py"
    assert (SNAKEDIR / spec.script).is_file()


# ---------------------------------------------------------------------------
# The three declarations
# ---------------------------------------------------------------------------

_SNAKEFILES = (
    ("wf1", "Snakefile_model_creation"),
    ("wf2", "Snakefile_climate_projections"),
    ("wf3", "Snakefile_climate_experiment"),
)


def _parse_workflow(snakefile: str, config_path):
    """Parse a Snakefile in-process and return its ``Workflow``.

    Same helper, same pinning caveat, as ``test_climate_store_contract.py``:
    rules are built exactly as a real invocation builds them, and
    ``wf_api._workflow`` is private on the pinned Snakemake.
    """
    import snakemake.api as api

    with api.SnakemakeApi() as sa:
        wf_api = sa.workflow(
            resource_settings=api.ResourceSettings(cores=1),
            config_settings=api.ConfigSettings(configfiles=[Path(config_path)]),
            storage_settings=api.StorageSettings(),
            workflow_settings=api.WorkflowSettings(),
            snakefile=SNAKEDIR / snakefile,
            workdir=SNAKEDIR,
        )
        workflow = wf_api._workflow
        workflow.include(workflow.main_snakefile, overwrite_default_target=True)
        return workflow


@pytest.fixture(scope="module")
def declarations():
    """The built ``delineate_region`` rule from all three workflows."""
    return {
        label: _parse_workflow(snakefile, CONFIG_FN).get_rule(RULE_NAME)
        for label, snakefile in _SNAKEFILES
    }


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_every_workflow_declares_the_rule(declarations):
    for label, rule in declarations.items():
        assert rule is not None, f"{label} has no {RULE_NAME} rule"
        assert rule.name == RULE_NAME


@pytest.mark.workflow_contract
def test_declarations_are_identical(declarations):
    """Compared pairwise: with three declarations, one left/right comparison
    would let two agree while the third drifted."""
    import itertools

    def _fields(rule):
        return {
            "input": sorted(str(path) for path in rule.input),
            "output": sorted(str(path) for path in rule.output),
            "input_keys": sorted(rule.input.keys()),
            "output_keys": sorted(rule.output.keys()),
            "params": sorted(f"{k}={rule.params[k]!r}" for k in rule.params.keys()),
            "script": str(rule.script),
        }

    differences = []
    for left, right in itertools.combinations(sorted(declarations), 2):
        for field, left_value in _fields(declarations[left]).items():
            right_value = _fields(declarations[right])[field]
            if left_value != right_value:
                differences.append(
                    f"{field} ({left} vs {right}):\n"
                    f"    {left} = {left_value}\n    {right} = {right_value}"
                )
    assert not differences, (
        f"{RULE_NAME} differs across the three workflows on "
        f"{len(differences)} comparison(s). Only message/log/benchmark may "
        "differ; everything else must come from region_rule.\n" + "\n".join(differences)
    )


@pytest.mark.workflow_contract
def test_the_output_is_the_shared_artifact(declarations):
    for label, rule in declarations.items():
        paths = [str(path) for path in rule.output]
        assert len(paths) == 1, f"{label}: {paths}"
        assert (
            paths[0].replace("\\", "/").endswith("/data/spatial/geoms/region.geojson")
        ), f"{label}: {paths[0]}"


@pytest.mark.workflow_contract
def test_no_workflow_delineates_the_region_twice():
    """The whole point: ``parse_region_basin`` has exactly one caller left.

    Rule 1.02 and the climate store both read the artifact now. A second call
    site would restore the silent-agreement-by-coincidence this record removed.
    """
    callers = sorted(
        path.relative_to(SNAKEDIR).as_posix()
        for path in (SNAKEDIR / "blueearth_cst").rglob("*.py")
        if "parse_region_basin(" in path.read_text(encoding="utf-8")
    )
    assert callers == ["blueearth_cst/spatial/delineate_region.py"], callers
