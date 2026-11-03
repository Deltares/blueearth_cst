"""WF4 selects a checked v2 marker without invoking WF3 producers."""

import io
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueearth_cst.experiment import collection_resolution, simulation_runner
from blueearth_cst.experiment.content_identity import atomic_record, content_sha256


@pytest.mark.parametrize("published", [True, False])
def test_launcher_verdict_follows_result_publication(tmp_path, monkeypatch, published):
    """A failed export cannot leave a successful workflow verdict behind it."""
    from blueearth_cst.experiment import indicator_export, reservation_context
    from blueearth_cst.shared.run_log_core import log_row
    from scripts import simulate_system as launcher

    config = tmp_path / "project.yml"
    config.write_text("{}", encoding="utf-8")
    record_path = tmp_path / "logs/_engine/invocations/example.json"
    record = {
        "invocation_id": "example",
        "execution_directory": "logs/example",
        "configuration": {},
    }
    project = {"project": {"project_dir": str(tmp_path)}}
    selection = SimpleNamespace(sets=[], indicators=True)
    monkeypatch.setattr(
        launcher.invocation_history, "start", lambda *a, **kw: (record_path, record)
    )
    monkeypatch.setattr(launcher.invocation_history, "update", lambda *a, **kw: None)
    monkeypatch.setattr(launcher.invocation_history, "finish", lambda *a, **kw: None)
    monkeypatch.setattr(launcher.resource_tracking, "phase", lambda *a, **kw: None)
    monkeypatch.setattr(
        launcher,
        "simulation_selection",
        lambda *a: (project, {"operation": "indicators-only"}, selection),
    )
    monkeypatch.setattr(launcher, "simulation_command", lambda *a, **kw: (["fake"], {}))
    monkeypatch.setattr(launcher, "archive_lock", lambda *a: nullcontext())
    context = SimpleNamespace(name="experiment_01", environment=lambda: {})
    monkeypatch.setattr(
        reservation_context, "admit_context_owned", lambda *a, **kw: context
    )
    monkeypatch.setattr(
        simulation_runner, "validate_retained_indicator_sets", lambda *a: None
    )

    def child(*args, **kwargs):
        assert kwargs["env"]["CST_SIMULATION_FINALIZE"] == "1"
        log_row("Calculation warning", module="indicators", level="WARNING")
        return 0

    def publication(*args):
        log_row(
            "Results saved" if published else "Publication failed",
            module="indicators",
            level="WARNING" if published else "ERROR",
        )
        return published

    monkeypatch.setattr(launcher, "run_project_child", child)
    monkeypatch.setattr(indicator_export, "publish_selected_sets", lambda *a: [])
    monkeypatch.setattr(indicator_export, "report_publication", publication)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.delenv("CST_LOG_LEVEL", raising=False)
    previous_tally = "previous-warning-tally"
    monkeypatch.setenv("CST_WARNING_TALLY", previous_tally)
    assert launcher.main(["--config", str(config), "--project-dir", str(tmp_path)]) == (
        0 if published else 1
    )
    output = stream.getvalue()
    verdict = "completed" if published else "FAILED"
    assert (
        output.index("Calculation warning")
        < output.index("Results saved" if published else "Publication failed")
        < output.index(f"wf4 simulate_system {verdict}")
    )
    assert "2 warnings" in output
    assert (
        output.rstrip().endswith("- DONE all")
        if published
        else "DONE all" not in output
    )
    assert launcher.os.environ["CST_WARNING_TALLY"] == previous_tally


def test_explicit_selector_uses_v2_reader(tmp_path, monkeypatch):
    marker = tmp_path / "scenarios/_engine/collections/abc/collection.json"
    selection = {
        "resolution_mode": "explicit-manifest",
        "manifest_path": str(marker),
        "collection_id": "a" * 64,
        "collection_revision": "b" * 64,
    }
    monkeypatch.setattr(
        simulation_runner,
        "simulation_settings",
        lambda _: ({}, {"scenario_collection": {"manifest_path": marker}}),
    )
    monkeypatch.setattr(
        simulation_runner, "verify_selected_collection_source", lambda *_: "era5"
    )
    monkeypatch.setattr(
        collection_resolution,
        "resolve_explicit_collection_v2",
        lambda selector: (
            (selection, {"schema_version": "scenario-collection/3"})
            if selector == {"manifest_path": marker}
            else pytest.fail("wrong selector")
        ),
    )
    selected, record = simulation_runner.resolve_selected_collection("unused", tmp_path)
    assert selected == selection
    assert record["schema_version"] == "scenario-collection/3"


def test_project_selector_checks_pinned_candidate(tmp_path, monkeypatch):
    from blueearth_cst.experiment import generation_plan, scenario_collection_v2
    from blueearth_cst.shared import config_composition

    project = tmp_path / "project"
    request = {"schema_version": "generation-request/2", "fixture": "v2"}
    request_id = content_sha256(request)
    request_dir = project / "scenarios/_engine/requests" / request_id[:12]
    request_dir.mkdir(parents=True)
    pointer = request_dir / "request.json"
    atomic_record(
        pointer,
        {
            "schema_version": "scenario-request/2",
            "generation_request_id": request_id,
            "plan_path": f"{'b' * 64}.json",
            "path_base": "request_directory",
            "plan_sha256": "b" * 64,
        },
    )
    marker = project / "scenarios/_engine/collections" / ("c" * 12) / "collection.json"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"{}\n")
    settings = {
        "request_path": str(pointer),
        "project_dir": str(project),
        "request": request,
    }
    plan = {
        "generation_request_id": request_id,
        "collection_id": "c" * 64,
        "decision": "create",
        "outputs": {"record_root": f"scenarios/_engine/collections/{'c' * 12}"},
    }
    plan_digest = content_sha256(plan)
    plan_path = request_dir / f"{plan_digest}.json"
    atomic_record(plan_path, plan)
    atomic_record(
        pointer,
        {
            "schema_version": "scenario-request/2",
            "generation_request_id": request_id,
            "plan_path": plan_path.name,
            "path_base": "request_directory",
            "plan_sha256": plan_digest,
        },
        replace=True,
    )
    monkeypatch.setattr(simulation_runner, "simulation_settings", lambda _: ({}, {}))
    monkeypatch.setattr(
        simulation_runner, "verify_selected_collection_source", lambda *_: "era5"
    )
    monkeypatch.setattr(
        config_composition, "compose_config", lambda *_args, **_kwargs: ({}, None)
    )
    monkeypatch.setattr(
        generation_plan, "generation_configuration", lambda *_: settings
    )
    monkeypatch.setattr(generation_plan, "read_pinned_plan", lambda *_: plan)
    monkeypatch.setattr(
        generation_plan,
        "build_candidate_intent",
        lambda _: pytest.fail("WF4 must not capture WF3 source snapshots"),
    )
    monkeypatch.setattr(
        scenario_collection_v2,
        "read_collection_v2",
        lambda path: (
            {"collection_id": "c" * 64, "collection_revision": "d" * 64}
            if Path(path) == marker
            else pytest.fail("wrong marker")
        ),
    )
    selection, collection = simulation_runner.resolve_selected_collection(
        pointer, tmp_path
    )
    assert selection["manifest_path"] == marker.as_posix()
    assert selection["collection_revision"] == collection["collection_revision"]
    selected, record, reference = simulation_runner.resolve_collection_with_plan(
        pointer, tmp_path
    )
    assert (selected, record) == (selection, collection)
    assert reference["sha256"] == plan_digest
    assert project / reference["path"] == plan_path
    settings["request"] = {"schema_version": "generation-request/2", "fixture": "drift"}
    with pytest.raises(ValueError, match="project request pointer differs"):
        simulation_runner.resolve_selected_collection(pointer, tmp_path)


def test_selected_collection_climate_source_matches_creator(tmp_path, monkeypatch):
    from blueearth_cst.shared import workflow_config_snapshot

    marker_path = (
        tmp_path / "scenarios/_engine/collections" / ("a" * 12) / "collection.json"
    )
    marker_path.parent.mkdir(parents=True)
    marker_path.write_bytes(b"{}\n")
    selection = {"manifest_path": str(marker_path)}
    marker = {"collection_id": "a" * 64, "archive": {"path": "unused"}}
    archive_path = tmp_path / "archive/run_record.yml"
    monkeypatch.setattr(
        workflow_config_snapshot,
        "resolve_file_reference",
        lambda *_args: archive_path,
    )
    monkeypatch.setattr(
        workflow_config_snapshot,
        "validate_archive",
        lambda _: {
            "owner": {"kind": "scenario_collection", "id": "a" * 64},
            "loaded_config": {"climate": {"selected": "era5"}},
        },
    )
    assert (
        simulation_runner.verify_selected_collection_source(
            {
                "project": {"project_dir": str(tmp_path)},
                "climate": {"selected": "era5"},
            },
            selection,
            marker,
        )
        == "era5"
    )
    with pytest.raises(ValueError, match="differs from collection creator"):
        simulation_runner.verify_selected_collection_source(
            {
                "project": {"project_dir": str(tmp_path)},
                "climate": {"selected": "chirps"},
            },
            selection,
            marker,
        )
    with pytest.raises(ValueError, match="outside the WF4 project root"):
        simulation_runner.verify_selected_collection_source(
            {
                "project": {"project_dir": str(tmp_path / "other")},
                "climate": {"selected": "era5"},
            },
            selection,
            marker,
        )


def _legacy_selection(monkeypatch, operation):
    from blueearth_cst.experiment import indicator_sets

    settings = {"operation": operation}
    monkeypatch.setattr(
        simulation_runner,
        "simulation_selection",
        lambda _: ({}, settings, indicator_sets.legacy_selection(settings)),
    )


def test_p6_target_matrix_uses_v2_contract(monkeypatch, tmp_path):
    from blueearth_cst.experiment import indicator_sets

    _legacy_selection(monkeypatch, "simulate-and-indicators")
    assert (
        simulation_runner.validate_targets("unused", ["all"])
        == "simulate-and-indicators"
    )
    # `responses` is the retired spelling, accepted for one release.
    for target in ("simulations_only", "responses"):
        assert (
            simulation_runner.validate_targets("unused", [target])
            == "simulate-and-indicators"
        )
    for target in ("simulations_and_indicators", "indicators"):
        with pytest.raises(ValueError, match="UnsupportedOperationTarget"):
            simulation_runner.validate_targets("unused", [target])

    selected = tmp_path / "_engine/indicator_sets/abc123/indicators.json"
    _legacy_selection(monkeypatch, "indicators-only")
    default = indicator_sets.resolve_set(
        indicator_sets.legacy_selection({"operation": "indicators-only"}).sets[0],
        ["gwr"],
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.indicator_plan.indicators_only_configuration",
        lambda _: (tmp_path, (default,), "YE-DEC"),
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.indicator_plan.current_indicator_request",
        lambda *_: {"request": "v2"},
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.indicator_plan.build_indicator_plan",
        lambda *_: {"targets": {"manifest": selected.as_posix()}, "tables": {}},
    )
    for target in ("simulations_and_indicators", "indicators"):
        assert (
            simulation_runner.validate_targets("unused", [target]) == "indicators-only"
        )
    assert (
        simulation_runner.validate_targets("unused", [selected.as_posix()])
        == "indicators-only"
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.indicator_plan.build_indicator_plan",
        lambda *_: {
            "targets": {
                "manifest": (
                    tmp_path / "_engine/indicator_sets/other/indicators.json"
                ).as_posix()
            },
            "tables": {},
        },
    )
    with pytest.raises(ValueError, match="UnsupportedOperationTarget"):
        simulation_runner.validate_targets("unused", [selected.as_posix()])


def test_indicators_only_reads_v2_collection_from_simulation_intent(
    tmp_path, monkeypatch
):
    from blueearth_cst.experiment import indicator_plan, reservation_context

    # This projection unit uses a synthetic marker; admission/integrity are
    # exercised with real retained records by the reservation integration tests.
    monkeypatch.setattr(
        reservation_context, "resolve_experiment_name", lambda _root, name: name
    )

    project = tmp_path / "project"
    root = project / "experiments/experiment_rapid"
    (root / "_engine").mkdir(parents=True)
    (root / "_engine/simulation.json").write_bytes(b"{}\n")
    workflow = tmp_path / "indicators.yml"
    workflow.write_text(
        "experiment_name: experiment_rapid\noperation: indicators-only\n",
        encoding="utf-8",
    )
    config = tmp_path / "project.yml"
    config.write_text(
        "project:\n"
        f"  project_dir: {project.as_posix()}\n"
        "workflows:\n"
        "  simulate_system:\n"
        "    enabled: true\n"
        "    config_path: indicators.yml\n",
        encoding="utf-8",
    )
    intent = {
        "collection": {
            "collection_id": "a" * 64,
            "collection_revision": "b" * 64,
            "manifest": {"path": "scenarios/_engine/collections/a/collection.json"},
        },
        "documents": {"response_request": {"variables": [{"variable": "q"}]}},
    }
    monkeypatch.setattr(
        indicator_plan,
        "read_simulation_v2",
        lambda path: {"schema_version": "simulation/3"},
    )
    monkeypatch.setattr(
        indicator_plan, "read_simulation_intent_v2", lambda path: intent
    )

    actual_root, sets, anchor = indicator_plan.indicators_only_configuration(config)

    assert actual_root == root
    assert {
        item.declaration.required_responses.variable for item in sets[0].indicators
    } == {"q"}
    assert anchor == "YE-DEC"


def test_retired_target_names_reach_snakemake_as_current_rule_names():
    assert simulation_runner.current_targets(["responses", "indicators", "all"]) == [
        "simulations_only",
        "simulations_and_indicators",
        "all",
    ]


def _named_selection(monkeypatch, tmp_path, settings, project=None):
    from blueearth_cst.experiment import indicator_sets

    selection = indicator_sets.parse_selection(settings, tmp_path / "wf4.yml")
    operation = (
        "indicators-only"
        if selection.operations == ("indicators",)
        else "simulate-and-indicators"
    )
    monkeypatch.setattr(
        simulation_runner,
        "simulation_selection",
        lambda _: (project or {}, {**settings, "operation": operation}, selection),
    )


@pytest.mark.parametrize(
    ("operations", "expected"),
    [
        (["simulations", "indicators"], ["all"]),
        (["indicators", "simulations"], ["all"]),
        (["simulations"], ["simulations_only"]),
    ],
)
def test_named_operations_infer_their_target(
    monkeypatch, tmp_path, operations, expected
):
    _named_selection(monkeypatch, tmp_path, {"operations": operations})
    assert simulation_runner.resolve_targets("unused", ["all"]) == (
        "simulate-and-indicators",
        expected,
    )
    with pytest.raises(ValueError, match="omit --target"):
        simulation_runner.resolve_targets("unused", ["simulations_only"])


def test_named_simulation_preflight_refuses_unavailable_outputs(monkeypatch, tmp_path):
    (tmp_path / "q_only.yml").write_text(
        "schema_version: 1\nid: q_only\ntitle: t\ndescription: d\n"
        "selection:\n  indicators:\n    - definition: aet_annual_total\n",
        encoding="utf-8",
    )
    _named_selection(
        monkeypatch,
        tmp_path,
        {
            "indicator_sets": ["q_only"],
            "indicator_set_definitions": {"q_only": {"config_path": "q_only.yml"}},
        },
        project={"model": {"outvars": ["river discharge"]}},
    )
    with pytest.raises(
        ValueError,
        match=r"q_only:aet_annual_total \(annual_summary/1\) needs response aet",
    ):
        simulation_runner.resolve_targets("unused", ["all"])
