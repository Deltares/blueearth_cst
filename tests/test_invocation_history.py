"""P3 common invocation states and launcher failure boundaries."""

import hashlib
import json
from pathlib import Path

import pytest

from blueearth_cst.shared import invocation_history, workflow_archive_launch
from blueearth_cst.shared.invocation_summary import summary_path
from scripts import run_workflows, simulate_system


def _records(root: Path) -> list[dict]:
    folder = root / "config/_engine/invocations"
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("*.json")
    ]


def test_direct_dry_run_is_a_nonproductive_attempt(tmp_path, monkeypatch):
    from blueearth_cst.shared import windows_job

    config = tmp_path / "project_config_test.yml"
    config.write_text(
        "workflows:\n  analyze_climate:\n    enabled: true\n", encoding="utf-8"
    )
    monkeypatch.setattr(windows_job, "run_project_child", lambda *_args, **_kwargs: 0)
    result = workflow_archive_launch.run_workflow(
        "analyze_climate", config, tmp_path / "project", dry_run=True
    )
    assert result == 0
    (record,) = _records(tmp_path / "project")
    assert record["schema_version"] == "invocation/1"
    assert record["parent_invocation_id"] is None
    assert record["mode"] == "dry_run"
    assert record["work_performed"] == "no"
    assert record["status"] == "succeeded"
    assert record["configuration"]["run_record"] is None
    assert record["elapsed_seconds"] >= 0
    assert (
        "dry_run"
        in summary_path(
            next((tmp_path / "project/config/_engine/invocations").glob("*.json"))
        ).read_text()
    )


def test_unobserved_hard_termination_stays_running(tmp_path):
    path, record = invocation_history.start(
        tmp_path,
        workflow="generate_scenarios",
        entry_point="scripts/run_workflows.py",
        command=["snakemake"],
        targets=["all"],
        mode="execute",
        contract_mode="legacy_wf3_interim",
    )
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"
    assert record["ended_at_utc"] is None
    assert record["work_performed"] == "unknown"
    assert record["elapsed_seconds"] is None
    assert "unfinished (last observed running)" in summary_path(path).read_text()


@pytest.mark.parametrize("explicit_root", [False, True])
def test_parent_capture_failure_records_no_fabricated_child(
    tmp_path, monkeypatch, explicit_root
):
    config = tmp_path / "project_config_test.yml"
    root = tmp_path / "project"
    config.write_text(
        f"project:\n  project_dir: {root}\nworkflows:\n"
        + "".join(
            f"  {name}:\n    enabled: {'true' if name == 'analyze_climate' else 'false'}\n"
            for name in run_workflows.WORKFLOW_ORDER
        ),
        encoding="utf-8",
    )

    def refuse(*_args, **_kwargs):
        raise ValueError("pre-parse capture failed")

    monkeypatch.setattr(workflow_archive_launch, "prepare_workflow", refuse)
    with pytest.raises(ValueError, match="pre-parse capture failed"):
        run_workflows.run(
            str(config), cores=1, extra=[], project_dir=root if explicit_root else None
        )
    (parent,) = _records(root)
    assert parent["workflow"] is None
    assert parent["status"] == "failed"
    assert parent["children"] == []
    assert parent["launch_failures"][0]["workflow"] == "analyze_climate"
    assert parent["execution_directory"].endswith("__analyze-climate")
    report = next((root / "logs").glob("*/summary.md")).read_text()
    assert "launch failed (no child record)" in report
    assert "Stop boundary:" in report


def test_explicit_root_records_config_failure_before_child(tmp_path):
    root = tmp_path / "project"
    with pytest.raises(run_workflows.ConfigError):
        run_workflows.run(
            str(tmp_path / "missing.yml"), cores=1, extra=[], project_dir=root
        )
    (parent,) = _records(root)
    assert parent["status"] == "failed"
    assert parent["children"] == []
    assert parent["error"]["type"] == "ConfigError"
    assert parent["execution_directory"].endswith("__workflows")
    assert (root / parent["execution_directory"] / "summary.md").is_file()


def test_simulation_explicit_root_records_startup_failure(tmp_path):
    root = tmp_path / "project"
    with pytest.raises((OSError, ValueError)):
        simulate_system.main(
            ["--config", str(tmp_path / "missing.yml"), "--project-dir", str(root)]
        )
    (record,) = _records(root)
    assert record["workflow"] == "simulate_system"
    assert record["status"] == "failed"
    assert record["children"] == []
    assert record["configuration"]["run_record"] is None


def test_interim_wf3_retains_exact_config_source_bytes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "test_case/project_config_rapid.yml"
    digest = run_workflows._capture_legacy_wf3_attempt(source, tmp_path, "attempt")
    copied = list(
        (tmp_path / "config/_engine/invocations/attempt/config/sources").rglob(
            source.name
        )
    )
    assert len(copied) == 1
    assert copied[0].read_bytes() == source.read_bytes()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.fixture
def reporting_run(tmp_path, monkeypatch):
    """Exercise launcher records with child execution and archive capture substituted."""
    import yaml

    config = tmp_path / "project_config_test.yml"
    root = tmp_path / "project"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"project_dir": str(root)},
                "workflows": {
                    name: {"enabled": name in run_workflows.WORKFLOW_ORDER[:3]}
                    for name in run_workflows.WORKFLOW_ORDER
                },
            }
        ),
        encoding="utf-8",
    )
    outcomes = [0, 0, 0]
    monkeypatch.setattr(
        workflow_archive_launch,
        "prepare_workflow",
        lambda _name, path, _root, **kwargs: (
            path,
            _root / "config/_engine/execution-configs/launch-test.json",
        ),
    )
    monkeypatch.setattr(run_workflows, "_bind_workflow_archive", lambda *args: None)
    monkeypatch.setattr(
        run_workflows, "run_project_child", lambda *args, **kwargs: outcomes.pop(0)
    )
    return config, root, outcomes


def test_multiworkflow_summary_matches_children_and_preserves_history(reporting_run):
    config, root, outcomes = reporting_run
    assert run_workflows.run(str(config), cores=1, extra=[]) == 0
    reports = list((root / "logs").glob("*/summary.md"))
    assert len(reports) == 1  # One parent report, not a competing report per child.
    original = reports[0].read_bytes()
    parent = next(record for record in _records(root) if record["workflow"] is None)
    assert parent["elapsed_seconds"] >= 0
    for child in parent["children"]:
        path = root / child["record"]["path"]
        saved = json.loads(path.read_text())
        assert saved["elapsed_seconds"] >= 0
        assert saved["status"] == "succeeded"
        assert f"({saved['elapsed_seconds']:.3f} s)" in original.decode()
        assert saved["work_performed"] == "unknown"
    outcomes.extend([0, 0, 0])
    assert run_workflows.run(str(config), cores=1, extra=[]) == 0
    assert len(list((root / "logs").glob("*/summary.md"))) == 2
    assert reports[0].read_bytes() == original


def test_summary_preserves_success_failure_and_stop_boundary(reporting_run):
    config, root, outcomes = reporting_run
    outcomes[:] = [0, 7]
    assert run_workflows.run(str(config), cores=1, extra=[]) == 7
    report = next((root / "logs").glob("*/summary.md")).read_text()
    assert "analyze_climate | succeeded (exit 0)" in report
    assert "build_model | failed (exit 7)" in report
    assert "analyze_projections | not invoked" in report
    assert "generate_scenarios | disabled" in report
    assert "Stop boundary:" in report


@pytest.mark.parametrize("exit_code", [0, 9])
def test_report_error_preserves_execution_outcome(
    reporting_run, monkeypatch, capsys, exit_code
):
    from blueearth_cst.shared import invocation_summary

    def denied(*args):
        raise OSError("report unavailable")

    monkeypatch.setattr(invocation_summary, "write_summary", denied)
    config, root, outcomes = reporting_run
    outcomes[:] = [0, exit_code, 0]
    assert run_workflows.run(str(config), cores=1, extra=[]) == exit_code
    parent = next(record for record in _records(root) if record["workflow"] is None)
    assert parent["exit_code"] == exit_code
    assert "invocation summary unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("no_op", [False, True])
def test_summary_distinguishes_dry_run_and_no_op(reporting_run, no_op):
    import yaml

    config, root, _ = reporting_run
    if no_op:
        document = yaml.safe_load(config.read_text())
        for workflow in document["workflows"].values():
            workflow["enabled"] = False
        config.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert (
        run_workflows.run(str(config), cores=1, extra=[] if no_op else ["--dry-run"])
        == 0
    )
    parent = next(record for record in _records(root) if record["workflow"] is None)
    report = (root / parent["execution_directory"] / "summary.md").read_text()
    assert parent["work_performed"] == "no"
    if no_op:
        assert not parent["children"]
        assert "disabled" in report
    else:
        assert "Mode: dry_run" in report
        assert all(record["work_performed"] == "no" for record in _records(root))
