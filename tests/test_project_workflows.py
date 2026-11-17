"""The split `workflows` + `workflow_configs` form composes like legacy stanzas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.config_composition import compose_config
from blueearth_cst.shared.project_workflows import (
    legacy_workflow_stanzas,
    normalize_project_workflows,
    set_workflow_config_path,
    workflow_config_path,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "project_config_fixture.yml"


def _split(project: dict) -> dict:
    split = {key: value for key, value in project.items() if key != "workflows"}
    stanzas = project["workflows"]
    split["workflows"] = {
        name: stanza["enabled"]
        for name, stanza in stanzas.items()
        if "enabled" in stanza
    }
    split["workflow_configs"] = {
        name: stanza["config_path"]
        for name, stanza in stanzas.items()
        if "config_path" in stanza
    }
    return split


def test_split_form_returns_legacy_stanzas():
    project = {
        "workflows": {"build_model": True, "simulate_system": False},
        "workflow_configs": {"build_model": "a.yml", "generate_scenarios": "b.yml"},
    }
    assert legacy_workflow_stanzas(project) == {
        "build_model": {"enabled": True, "config_path": "a.yml"},
        "simulate_system": {"enabled": False},
        "generate_scenarios": {"config_path": "b.yml"},
    }
    normalized = normalize_project_workflows(project)
    assert "workflow_configs" not in normalized
    assert workflow_config_path(project, "build_model") == "a.yml"
    assert workflow_config_path(project, "simulate_system") is None


def test_legacy_form_passes_through_and_absent_adds_nothing():
    legacy = {"workflows": {"build_model": {"enabled": True, "config_path": "a.yml"}}}
    assert normalize_project_workflows(legacy) == legacy
    assert "workflows" not in normalize_project_workflows({"project": {}})


@pytest.mark.parametrize(
    "project",
    [
        {"workflows": {"build_model": {"enabled": True}, "simulate_system": True}},
        {
            "workflows": {"build_model": {"enabled": True}},
            "workflow_configs": {"build_model": "a.yml"},
        },
        {"workflows": {"build_model": "yes"}},
        {"workflows": {"build_model": True}, "workflow_configs": {"build_model": ""}},
        {"workflows": {"build_model": True}, "workflow_configs": ["a.yml"]},
    ],
)
def test_mixed_or_malformed_forms_are_refused(project):
    with pytest.raises(ValueError):
        normalize_project_workflows(project)


def test_set_config_path_keeps_the_form():
    split = {
        "workflows": {"build_model": True},
        "workflow_configs": {"build_model": "a"},
    }
    set_workflow_config_path(split, "build_model", "b")
    assert split == {
        "workflows": {"build_model": True},
        "workflow_configs": {"build_model": "b"},
    }
    legacy = {"workflows": {"build_model": {"enabled": True, "config_path": "a"}}}
    set_workflow_config_path(legacy, "build_model", "b")
    assert legacy["workflows"]["build_model"]["config_path"] == "b"


@pytest.mark.parametrize(
    "entry", ["build_model", "generate_scenarios", "simulate_system"]
)
def test_split_form_composes_identically(entry):
    legacy = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    expected = compose_config(legacy, FIXTURE, entry)
    assert compose_config(_split(legacy), FIXTURE, entry) == expected


def test_workflow_file_may_not_declare_workflow_configs(tmp_path):
    legacy = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    body = yaml.safe_load(
        (FIXTURE.parent / legacy["workflows"]["build_model"]["config_path"]).read_text(
            encoding="utf-8"
        )
    )
    body["workflow_configs"] = {"build_model": "x.yml"}
    workflow = tmp_path / "build_model.yml"
    workflow.write_text(yaml.safe_dump(body), encoding="utf-8")
    project = _split(legacy)
    project["workflow_configs"] = {
        **project["workflow_configs"],
        "build_model": str(workflow),
    }
    with pytest.raises(ValueError, match="workflow_configs"):
        compose_config(project, FIXTURE, "build_model")
