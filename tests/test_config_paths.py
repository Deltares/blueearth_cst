"""Portable configuration behavior across movement, launchers and legacy inputs."""

import copy
import os
import shutil
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.config_composition import compose_config, load_composed_config
from blueearth_cst.shared.config_paths import (
    TOOLBOX_ROOT,
    resolve_path,
    resolve_project_paths,
    resolve_workflow_paths,
)
from blueearth_cst.shared.provenance import effective_config_digest
from scripts.run_workflows import _project_dir, _read_config


def test_case_relocation_and_launch_directory(tmp_path, monkeypatch):
    original = tmp_path / "original" / "applications" / "basin"
    (original / "data").mkdir(parents=True)
    (original / "settings").mkdir()
    (original / "data" / "locations.csv").write_text("id,x,y\n1,2,3\n")
    project = {
        "schema_version": 3,
        "project": {
            "project_dir": "../../runs/basin",
            "catalog": [
                "toolbox:config/catalogs/deltares_data_pdrive.yml",
                "artifact_data",
            ],
        },
        "basin": {"output_locations": "data/locations.csv"},
        "climate": {"selected": "era5"},
        "model": {"outvars": ["river discharge"]},
        "workflows": {
            "build_model": {"enabled": True, "config_path": "settings/model.yml"}
        },
    }
    model = {
        "observations": {"river discharge": "../data/observed.csv"},
        "engine": {"build_config": "toolbox:config/defaults/wflow_build_model.yml"},
    }
    (original / "project.yml").write_text(yaml.safe_dump(project))
    (original / "settings/model.yml").write_text(yaml.safe_dump(model))
    relocated = tmp_path / "moved"
    shutil.copytree(original.parents[1], relocated)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    path = relocated / "applications/basin/project.yml"
    config = load_composed_config(path)
    assert Path(config["project"]["project_dir"]) == relocated / "runs/basin"
    assert Path(config["basin"]["output_locations"]).read_text() == "id,x,y\n1,2,3\n"
    assert (
        Path(config["workflows"]["build_model"]["observations"]["river discharge"])
        == path.parent / "data/observed.csv"
    )
    assert config["project"]["catalog"] == [
        (TOOLBOX_ROOT / "config/catalogs/deltares_data_pdrive.yml").as_posix(),
        "artifact_data",
    ]
    assert _project_dir(_read_config(str(path)), str(path)) == relocated / "runs/basin"
    # Repeated reads from another cwd preserve runtime settings and their digest.
    monkeypatch.chdir(tmp_path)
    assert effective_config_digest(
        load_composed_config(path), {}
    ) == effective_config_digest(config, {})
    assert yaml.safe_load(path.read_text()) == project


def test_legacy_values_and_identity_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = {
        "schema_version": 2,
        "project": {"project_dir": "outputs", "catalog": "config/catalog.yml"},
        "basin": {"output_locations": "data/points.csv"},
        "workflows": {},
    }
    expected = copy.deepcopy(project)
    actual, _ = compose_config(project, tmp_path / "project.yml")
    assert actual == expected
    assert effective_config_digest(actual, {}) == effective_config_digest(expected, {})


@pytest.mark.parametrize(
    "value",
    ["toolbox:", "toolbox:../escape", "toolbox:/absolute", "toolbox:C:/absolute"],
)
def test_invalid_toolbox_paths_are_refused(tmp_path, value):
    with pytest.raises(ValueError, match="toolbox"):
        resolve_path(value, tmp_path / "project.yml")


def test_urls_catalog_names_and_absolute_paths(tmp_path):
    source = tmp_path / "project.yml"
    assert (
        resolve_path("https://example.org/catalog.yml", source, catalog=True)
        == "https://example.org/catalog.yml"
    )
    assert resolve_path("artifact_data", source, catalog=True) == "artifact_data"
    assert (
        resolve_path(str(tmp_path / "data.csv"), source)
        == (tmp_path / "data.csv").as_posix()
    )
    assert (
        resolve_path("data\\observations.csv", source)
        == (tmp_path / "data/observations.csv").as_posix()
    )
    if os.name != "nt":
        with pytest.raises(ValueError, match="platform"):
            resolve_path("C:/data/file.csv", source)


def test_each_workflow_path_and_geometry(tmp_path):
    source = tmp_path / "nested" / "workflow.yml"
    for workflow, document, expected in [
        (
            "analyze_projections",
            {"catalog": "../catalog.yml"},
            {"catalog": (tmp_path / "catalog.yml").as_posix()},
        ),
        (
            "generate_scenarios",
            {"weathergen_config": "generator.yml"},
            {"weathergen_config": (source.parent / "generator.yml").as_posix()},
        ),
        (
            "simulate_system",
            {
                "scenario_collection": {"manifest_path": "../collection.json"},
                "indicator_sets": [{"set": "custom"}],
                "indicator_set_definitions": {"custom": {"config_path": "custom.yml"}},
            },
            {
                "scenario_collection": {
                    "manifest_path": (tmp_path / "collection.json").as_posix()
                },
                "indicator_sets": [{"set": "custom"}],
                "indicator_set_definitions": {
                    "custom": {"config_path": (source.parent / "custom.yml").as_posix()}
                },
            },
        ),
    ]:
        assert resolve_workflow_paths(document, source, workflow, 3) == expected
        assert resolve_workflow_paths(document, source, workflow, 2) == document
    project = {
        "schema_version": 3,
        "basin": {"region": {"subbasin": "../basin.gpkg", "uparea": 100}},
    }
    assert resolve_project_paths(project, source)["basin"]["region"] == {
        "subbasin": (tmp_path / "basin.gpkg").as_posix(),
        "uparea": 100,
    }


def test_future_schema_is_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match="unsupported schema_version"):
        compose_config({"schema_version": 4}, tmp_path / "project.yml")


def test_schema3_archive_stages_original_relative_inputs(tmp_path, monkeypatch):
    from blueearth_cst.shared import workflow_archive_launch as launch
    from blueearth_cst.shared.workflow_config_snapshot import read_archive

    source = tmp_path / "case"
    (source / "settings").mkdir(parents=True)
    (source / "data").mkdir()
    catalog = source / "catalog.yml"
    catalog.write_text("meta: {roots: [/data]}\nera5: {uri: data.nc}\n")
    observations = source / "data/observed.csv"
    observations.write_text("time,value\n2000-01-01,1\n")
    root = tmp_path / "outputs"
    project = {
        "schema_version": 3,
        "project": {"project_dir": str(root), "catalog": "catalog.yml"},
        "basin": {"region": "{'subbasin': [1, 2]}"},
        "climate": {"selected": "era5"},
        "model": {"outvars": ["river discharge"]},
        "workflows": {
            "build_model": {"enabled": True, "config_path": "settings/model.yml"}
        },
    }
    workflow = {"observations": {"river discharge": "../data/observed.csv"}}
    path = source / "project.yml"
    path.write_text(yaml.safe_dump(project))
    (source / "settings/model.yml").write_text(yaml.safe_dump(workflow))
    original_bytes = path.read_bytes()
    monkeypatch.setattr(launch, "toolbox_identity", lambda: {"commit": None})
    monkeypatch.setattr(launch, "environment_file_hashes", lambda: {})
    monkeypatch.chdir(tmp_path)

    def prepare(config_path):
        return launch.prepare_workflow(
            "build_model", config_path, root, command=["test"], targets=["all"]
        )[0]

    execution = prepare(path)
    loaded = load_composed_config(execution)
    observed = Path(
        loaded["workflows"]["build_model"]["observations"]["river discharge"]
    )
    assert observed.read_bytes() == observations.read_bytes()
    assert observed.is_relative_to(root)
    assert Path(loaded["project"]["project_dir"]) == root
    record = read_archive(root, "build_model", "config/build_model")
    captured = next(item for item in record["source_files"] if item["id"] == "project")
    assert (
        root / "config/build_model" / captured["archived_path"]
    ).read_bytes() == original_bytes
    moved = tmp_path / "moved-case"
    shutil.copytree(source, moved)
    assert prepare(moved / "project.yml") == execution
