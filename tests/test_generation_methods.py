"""Integrated configuration preserves producer inputs and explicit selection."""

import pytest
import yaml

from blueearth_cst.experiment.delta_generation import capture_delta
from blueearth_cst.experiment.delta_scenarios import generate_collection
from blueearth_cst.shared.config_composition import load_composed_config
from blueearth_cst.shared.generation_config import normalize_generation
from tests.test_config_composition import write_split
from tests.test_delta_scenarios import historical, monthly, settings


@pytest.mark.parametrize(
    "document",
    [
        {"stochastic": {}},
        {"method": "other"},
        {"method": "stochastic", "stochastic": {}, "seed": 3},
        {"method": "stochastic", "stochastic": {}, "delta_change": {}},
        {"method": "stochastic", "stochastic": {"typo": 3}},
    ],
)
def test_invalid_methods_refused(document):
    with pytest.raises(ValueError):
        normalize_generation(document, 3)


def test_nested_stochastic_composes_exactly_like_legacy(tmp_path):
    flat = {
        "seed": 42,
        "n_realizations": 2,
        "simulation_window": {"start": 2000, "end": 2001},
    }
    path = write_split(
        tmp_path, sections={}, bodies={"generate_scenarios": flat}, schema_version=3
    )
    old = load_composed_config(path)
    workflow = tmp_path / "project_config_test_generate_scenarios.yml"
    workflow.write_text(yaml.safe_dump({"method": "stochastic", "stochastic": flat}))
    assert load_composed_config(path) == old
    with pytest.raises(ValueError, match="schema_version: 3"):
        normalize_generation({"method": "stochastic", "stochastic": flat}, 2)


def delta_project(tmp_path):
    """Use the delta producer's existing numerical fixture at the real boundary."""
    historical().to_netcdf(tmp_path / "historical.nc")
    monthly().to_netcdf(tmp_path / "reference.nc")
    monthly().to_netcdf(tmp_path / "future.nc")
    path = write_split(
        tmp_path,
        sections={},
        schema_version=3,
        bodies={
            "generate_scenarios": {"method": "delta_change", "delta_change": settings()}
        },
    )
    project = yaml.safe_load(path.read_text())
    for name in (
        "analyze_climate",
        "build_model",
        "analyze_projections",
        "simulate_system",
    ):
        project["workflows"][name] = {"enabled": False}
    path.write_text(yaml.safe_dump(project))
    return path


def test_delta_capture_preview_equivalence_and_reuse(tmp_path, monkeypatch):
    path = delta_project(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    operation = capture_delta(path, ["--dry-run"])
    root = tmp_path / "proj"
    preview = operation.run(root, preview=True)
    assert not root.exists()
    actual = operation.run(root, preview=False)
    assert preview == actual
    before = actual.read_bytes()
    standalone = tmp_path / "delta.yml"
    standalone.write_text(yaml.safe_dump(settings()))
    assert generate_collection(standalone, root) == actual
    assert actual.read_bytes() == before
    assert operation.provenance()["method"] == "delta_change"


@pytest.mark.parametrize("changed", ["config", "input"])
def test_delta_capture_rechecks_sources(tmp_path, changed):
    path = delta_project(tmp_path)
    operation = capture_delta(path, [])
    target = path if changed == "config" else tmp_path / "reference.nc"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        operation.run(tmp_path / "proj", preview=False)
    assert not (tmp_path / "proj").exists()


def test_delta_partial_output_and_flags_refused(tmp_path):
    path = delta_project(tmp_path)
    with pytest.raises(ValueError, match="does not support"):
        capture_delta(path, ["--unlock"])
    operation = capture_delta(path, [])
    root = tmp_path / "proj"
    preview = operation.run(root, preview=True)
    preview.parent.mkdir(parents=True)
    for dry in (True, False):
        with pytest.raises(ValueError, match="incomplete or claimed"):
            operation.run(root, preview=dry)
    assert not preview.exists()


@pytest.mark.parametrize("preview_flag", ["--dry-run", "--dryrun", "-n"])
def test_runner_delta_only_preview_execute_and_failure(tmp_path, preview_flag):
    from scripts import run_workflows

    path = delta_project(tmp_path)
    root = tmp_path / "proj"
    assert run_workflows.run(str(path), 1, [preview_flag]) == 0
    assert not (root / "scenarios").exists()
    assert run_workflows.run(str(path), 1, []) == 0
    manifests = list((root / "scenarios/_engine/collections").glob("*/collection.json"))
    assert len(manifests) == 1
    # The real publisher and collection reader, not a mocked calculation, ran.
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

    assert len(read_collection_v2(manifests[0])["series"]) == 5


def test_runner_delta_with_missing_selection_fails_before_generation(tmp_path):
    from scripts import run_workflows

    path = delta_project(tmp_path)
    project = yaml.safe_load(path.read_text())
    project["workflows"]["simulate_system"] = {
        "enabled": True,
        "config_path": "simulation.yml",
    }
    project["climate"] = {"selected": "era5", "window": {"start": 2000, "end": 2000}}
    (tmp_path / "simulation.yml").write_text("operation: simulate-and-indicators\n")
    path.write_text(yaml.safe_dump(project))
    with pytest.raises(ValueError, match="explicit existing scenario_collection"):
        run_workflows.run(str(path), 1, [])
    assert not (tmp_path / "proj/scenarios").exists()


def test_runner_delta_failure_does_not_launch_wf4(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from blueearth_cst.experiment import (
        delta_generation,
        reservation_context,
        simulation_runner,
    )
    from scripts import run_workflows

    path = delta_project(tmp_path)
    root = tmp_path / "proj"
    selected = capture_delta(path, []).run(root, preview=False)
    project = yaml.safe_load(path.read_text())
    project["workflows"]["simulate_system"] = {
        "enabled": True,
        "config_path": "simulation.yml",
    }
    project["climate"] = {"selected": "era5", "window": {"start": 2000, "end": 2000}}
    (tmp_path / "simulation.yml").write_text(
        yaml.safe_dump(
            {
                "operation": "simulate-and-indicators",
                "scenario_collection": {"manifest_path": str(selected)},
            }
        )
    )
    path.write_text(yaml.safe_dump(project))
    monkeypatch.setattr(
        simulation_runner, "simulation_command", lambda *args, **kwargs: None
    )
    observed = []

    def admit(*args, **kwargs):
        observed.append(kwargs["execute_generation"])
        return SimpleNamespace(name="test", action="create", recheck=lambda: None)

    monkeypatch.setattr(reservation_context, "admit_context_owned", admit)
    monkeypatch.setattr(run_workflows, "_consumer_forcing_rows", lambda *args: ())

    def fail(*args, **kwargs):
        raise ValueError("injected delta publication failure")

    monkeypatch.setattr(delta_generation.DeltaGeneration, "run", fail)

    def unexpected(*args, **kwargs):
        pytest.fail("WF4 child launched after delta failure")

    monkeypatch.setattr(run_workflows, "run_project_child", unexpected)
    with pytest.raises(ValueError, match="injected delta publication failure"):
        run_workflows.run(str(path), 1, [])
    assert observed == [True]


def test_explicit_delta_reservation_capture_does_not_plan_stochastic(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from blueearth_cst.experiment import reservation_request
    from blueearth_cst.experiment.reservation import validate_request

    path = delta_project(tmp_path)
    root = tmp_path / "proj"
    selected = capture_delta(path, []).run(root, preview=False)
    project = yaml.safe_load(path.read_text())
    catalog = tmp_path / "catalog.yml"
    catalog.write_text("{}\n")
    project["project"]["catalog"] = [str(catalog)]
    project["climate"] = {"selected": "era5", "window": {"start": 2000, "end": 2000}}
    project["workflows"]["simulate_system"] = {
        "enabled": True,
        "config_path": "simulation.yml",
    }
    (tmp_path / "simulation.yml").write_text(
        yaml.safe_dump({"scenario_collection": {"manifest_path": str(selected)}})
    )
    path.write_text(yaml.safe_dump(project))

    def unexpected(*args, **kwargs):
        pytest.fail("explicit delta selection entered stochastic generation planning")

    monkeypatch.setattr(reservation_request, "generation_request", unexpected)
    request, captures = reservation_request.capture_request(
        path, Path.cwd(), execute_generation=False
    )
    validate_request(request, root)
    assert request["generation"] is None
    assert request["selection"]["mode"] == "explicit-manifest"
    assert "generate_scenarios" not in {role for role, _, _ in captures}


def v3_project(tmp_path, monkeypatch, *, name=None):
    """A composed case with WF2 products and the derived historical location."""
    from tests.test_delta_wf2 import wf2_project, wf2_settings

    (tmp_path / "wf2").mkdir(parents=True)
    project_root, *_ = wf2_project(tmp_path / "wf2", monkeypatch)
    body = wf2_settings(**({"name": name} if name else {}))
    path = write_split(
        tmp_path,
        sections={
            "basin": {"region": "{'bbox': [0.0, 0.0, 1.0, 1.0]}"},
            "climate": {
                "sources": ["era5"],
                "selected": "era5",
                "window": {"start": 1980, "end": 2010},
            },
        },
        schema_version=3,
        bodies={"generate_scenarios": {"method": "delta_change", "delta_change": body}},
    )
    project = yaml.safe_load(path.read_text())
    project["project"]["project_dir"] = project_root.as_posix()
    project["project"]["catalog"] = "catalog.yml"
    for workflow in ("analyze_climate", "build_model", "analyze_projections"):
        project["workflows"][workflow] = {"enabled": False}
    path.write_text(yaml.safe_dump(project))
    return path, project_root


def test_v3_capture_derives_baseline_pending_and_indexes(tmp_path, monkeypatch):
    from tests.test_delta_wf2 import era5_historical

    path, root = v3_project(tmp_path, monkeypatch)
    store = root / "data/climate/historical/era5_19800101_20101231"
    extract = store / "extract_historical.nc"
    with pytest.raises(
        ValueError, match="historical record missing.*extract_historical"
    ):
        capture_delta(path, [], project_dir=root)
    pending = capture_delta(path, [], project_dir=root, historical_producer=True)
    assert pending.historical == {
        "path": extract,
        "source": "era5",
        "recipe": None,
        "pet_method": "debruin",
    }
    assert pending.run(root, preview=True) is None
    with pytest.raises(ValueError, match="not produced by this invocation"):
        pending.run(root, preview=False)
    store.mkdir(parents=True)
    incomplete = era5_historical().drop_vars("kin")
    incomplete.to_netcdf(extract)
    with pytest.raises(ValueError, match="lacks era5 variables \\['kin'\\]"):
        pending.run(root, preview=False)
    era5_historical().to_netcdf(extract)
    manifest = pending.run(root, preview=False)
    pending.index(root, manifest)
    assert manifest == root / "scenarios/delta-cmip6-near-far/_engine/collection.json"
    # A renamed rerun of identical content reuses the set and records an alias.
    renamed, _ = v3_project(tmp_path / "renamed", monkeypatch, name="mine")
    project = yaml.safe_load(renamed.read_text())
    project["project"]["project_dir"] = root.as_posix()
    renamed.write_text(yaml.safe_dump(project))
    operation = capture_delta(renamed, [], project_dir=root)
    assert operation.run(root, preview=False) == manifest
    operation.index(root, manifest)
    index = (root / "scenarios/collections.csv").read_text().splitlines()
    assert index[1].startswith("delta-cmip6-near-far,mine,delta_change,")


def test_wf4_default_selects_the_delta_set_of_the_current_settings(
    tmp_path, monkeypatch
):
    from blueearth_cst.experiment.collection_resolution import (
        GeneratedCollectionUnavailable,
        resolve_default_delta,
    )
    from tests.test_delta_wf2 import era5_historical

    path, root = v3_project(tmp_path, monkeypatch)
    store = root / "data/climate/historical/era5_19800101_20101231"
    store.mkdir(parents=True)
    with pytest.raises(
        GeneratedCollectionUnavailable, match="historical record missing"
    ):
        resolve_default_delta(path, root)
    era5_historical().to_netcdf(store / "extract_historical.nc")
    with pytest.raises(GeneratedCollectionUnavailable, match="run scenario generation"):
        resolve_default_delta(path, root)
    manifest = capture_delta(path, [], project_dir=root).run(root, preview=False)
    selection, marker = resolve_default_delta(path, root)
    assert selection["manifest_path"] == manifest.resolve().as_posix()
    assert selection["resolution_mode"] == "project-generation"
    config = manifest.parent.parent / "config"
    assert sorted(p.name for p in config.iterdir()) == sorted(
        [path.name, "project_config_test_generate_scenarios.yml"]
    )
    intent = yaml.safe_load(manifest.with_name("collection_intent.json").read_text())
    assert intent["settings"]["historical_recipe"] is None
    # Changed WF2 products no longer match the published set: stale, not reused.
    inventory = root / "data/climate/projections/cmip6/analysis/gridded_changes"
    member = next(inventory.glob("member_*.nc"))
    member.write_bytes(member.read_bytes() + b"extra")
    with pytest.raises(
        GeneratedCollectionUnavailable, match="differs from its inventory"
    ):
        resolve_default_delta(path, root)


def test_pending_delta_request_is_stable_and_seedless(tmp_path, monkeypatch):
    from blueearth_cst.experiment.reservation import validate_request
    from blueearth_cst.experiment.reservation_request import delta_request
    from blueearth_cst.shared.config_composition import load_composed_config

    path, root = v3_project(tmp_path, monkeypatch)
    config = load_composed_config(path, entry="generate_scenarios")
    before = delta_request(config, root)
    assert before["historical"] == {
        "path": "data/climate/historical/era5_19800101_20101231/extract_historical.nc",
        "source": "era5",
        "recipe": None,
    }
    assert "name" not in before["settings"]
    # Producing the extract within the invocation never changes the request.
    from tests.test_delta_wf2 import era5_historical

    extract = root / before["historical"]["path"]
    extract.parent.mkdir(parents=True)
    era5_historical().to_netcdf(extract)
    assert delta_request(config, root) == before
    request = {
        "schema_version": "experiment-request/2",
        "generation": None,
        "selection": {"mode": "delta-generation", "manifest": None, "delta": before},
        "input_catalogs": [],
        "simulation": {},
    }
    try:
        validate_request(request, root)
    except ValueError as error:
        assert "delta" not in str(error) and "selection" not in str(error)
    bad = {**request, "selection": {**request["selection"], "delta": {"x": 1}}}
    with pytest.raises(ValueError, match="delta"):
        validate_request(bad, root)
