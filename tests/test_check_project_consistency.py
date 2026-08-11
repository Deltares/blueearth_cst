"""Gate 2 (a-h): pure script-unit tests for the wf3 drift-guard comparator.

These call ``compare_project_consistency`` directly on staged config/snapshot
pairs — NO Snakemake, no rerun-triggers (that is gate 2b,
``test_guard_invalidation.py``). Design: dev/milestones/p31/experiment-structure-design.md
§7 gate 2, §3b.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from blueearth_cst.experiment.check_project_consistency import (  # noqa: E402
    compare_project_consistency,
)

# A minimal but structurally faithful full config (R01 sectioned schema),
# mirroring test_case/snake_config_model_test.yml at the guarded
# sections. The experiment sections that vary per experiment are present but
# deliberately NOT guarded.
_BASE_CFG = {
    "project": {
        "project_dir": "test_case/test_local",
        "static_dir": "config",
        "data_sources": "config/catalogs/deltares_data.yml",
        "data_sources_climate": "config/catalogs/cmip6_data.yml",
    },
    "shared": {
        "basin": {
            "region": "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
            "resolution": 0.00833,
        },
        "historical_window": {
            "starttime": "2000-01-01T00:00:00",
            "endtime": "2020-12-31T00:00:00",
        },
        "clim_historical": "era5",
    },
    "workflows": {
        "model_creation": {
            "enabled": True,
            "model_build_config": "config/defaults/wflow_build_model.yml",
            "waterbodies_config": "config/defaults/wflow_update_waterbodies.yml",
            "wflow_outvars": ["river discharge"],
        },
        "climate_projections": {
            "enabled": True,
            "clim_project": "cmip6",
            "models": ["NOAA-GFDL/GFDL-ESM4"],
            "scenarios": ["ssp245", "ssp585"],
        },
        "climate_experiment": {
            "experiment_name": "experiment",
            "realizations_num": 2,
        },
    },
}


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def snapshots(tmp_path):
    """Write matching wf1 + wf2 snapshots from _BASE_CFG; return their paths."""
    wf1 = _write(tmp_path / "snake_config_model_creation.yml", _BASE_CFG)
    wf2 = _write(tmp_path / "snake_config_climate_projections.yml", _BASE_CFG)
    return wf1, wf2


def test_a_identical_sections_pass(snapshots):
    wf1, wf2 = snapshots
    diffs = compare_project_consistency(copy.deepcopy(_BASE_CFG), wf1, wf2)
    assert diffs == []


def test_b_mutated_basin_resolution_fails_naming_key(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["shared"]["basin"]["resolution"] = 0.05
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("shared.basin" in d and "resolution" in d for d in diffs)


def test_c_mutated_model_creation_fails(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["model_creation"]["wflow_outvars"] = ["actual evapotranspiration"]
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("workflows.model_creation" in d for d in diffs)


def test_d_flat_vs_binned_paths_pass(tmp_path):
    """Symmetric normalization: a flat-path experiment config vs a binned
    snapshot converges (design §3b, gate 2d)."""
    # Snapshot uses NEW binned catalog paths; experiment config uses OLD flat
    # paths for the mapped keys. Symmetric normalization makes them equal.
    snapshot_cfg = copy.deepcopy(_BASE_CFG)
    wf1 = _write(tmp_path / "snake_config_model_creation.yml", snapshot_cfg)

    live = copy.deepcopy(_BASE_CFG)
    live["project"]["data_sources"] = "config\\deltares_data.yml"
    live["project"]["data_sources_climate"] = "config\\cmip6_data.yml"
    live["workflows"]["model_creation"]["model_build_config"] = (
        "config\\wflow_build_model.yml"
    )
    live["workflows"]["model_creation"]["waterbodies_config"] = (
        "config\\wflow_update_waterbodies.yml"
    )

    diffs = compare_project_consistency(live, wf1, wf2_snapshot_path=None)
    assert diffs == []


def test_e_missing_wf1_snapshot_fails_with_run_first_message(tmp_path):
    missing = tmp_path / "does_not_exist.yml"
    diffs = compare_project_consistency(copy.deepcopy(_BASE_CFG), missing)
    assert diffs
    assert any(
        "run Snakefile_model_creation first" in d.lower()
        or "run snakefile_model_creation first" in d.lower()
        for d in diffs
    )


def test_f_mutated_historical_window_passes_not_guarded(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["shared"]["historical_window"]["endtime"] = "2010-12-31T00:00:00"
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs == []


def test_g_mutated_climate_projections_with_wf2_snapshot_fails(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["climate_projections"]["scenarios"] = ["ssp126"]
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("workflows.climate_projections" in d for d in diffs)


def test_h_mutated_climate_projections_without_wf2_snapshot_passes(snapshots):
    wf1, _ = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["climate_projections"]["scenarios"] = ["ssp126"]
    # No wf2 snapshot -> projections section unchecked + logged, passes.
    diffs = compare_project_consistency(live, wf1, wf2_snapshot_path=None)
    assert diffs == []
