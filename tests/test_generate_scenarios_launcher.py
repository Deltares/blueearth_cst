"""Direct owned WF3 invocations are recorded at every launch outcome."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueearth_cst.shared import output_capture
from scripts import generate_scenarios


@pytest.mark.parametrize("flag", ["--unlock", "--dry-run", "-n"])
@pytest.mark.parametrize("exit_code", [0, 9])
def test_preview_stops_after_snakemake(tmp_path, monkeypatch, flag, exit_code):
    config = tmp_path / "project.yml"
    config.write_text("{}", encoding="utf-8")
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs))
        return exit_code

    def unexpected(*args, **kwargs):
        pytest.fail("Preview entered generation setup")

    monkeypatch.setattr(generate_scenarios, "run_project_child", child)
    monkeypatch.setattr(generate_scenarios, "_run_source_phase_quietly", unexpected)
    monkeypatch.setattr(generate_scenarios, "compose_config", unexpected)
    assert (
        generate_scenarios.run_owned(config, tmp_path, 1, [flag], invocation_id="test")
        == exit_code
    )
    assert len(calls) == 1
    assert flag in calls[0][0]
    assert calls[0][1]["writing"] is False


@pytest.mark.parametrize("ready,exit_code", [(False, 0), (False, 9), (True, 0)])
def test_success_and_ready_reuse_export_the_selected_collection(
    tmp_path, monkeypatch, ready, exit_code, capsys
):
    from blueearth_cst.experiment import reservation_generation

    config = tmp_path / "project.yml"
    config.write_text("{}", encoding="utf-8")
    plan = {
        "outputs": {"record_root": "scenarios/_engine/collections/selected"},
        "collection_id": "selected",
        "rows": [
            {"run_id": "run_001", "rlz": "1", "st_id": ""},
            {"run_id": "run_002", "rlz": "1", "st_id": "1"},
        ],
    }
    monkeypatch.setattr(
        generate_scenarios, "_run_source_phase_quietly", lambda *a, **kw: 0
    )
    monkeypatch.setattr(
        generate_scenarios,
        "compose_config",
        lambda *a, **kw: (
            {
                "project": {"project_dir": str(tmp_path)},
                "workflows": {"generate_scenarios": {"enabled": True}},
            },
            {},
        ),
    )
    monkeypatch.setattr(
        reservation_generation,
        "select_reserved_plan_owned",
        lambda *a: (
            {"request_path": str(tmp_path / "request.json")},
            plan,
            tmp_path / "plan.json",
            "hash",
            ready,
        ),
    )
    monkeypatch.setattr(
        generate_scenarios,
        "initialize_generation",
        lambda *a, **kw: tmp_path / "receipt.json",
    )
    children = []
    monkeypatch.setattr(
        generate_scenarios,
        "run_project_child",
        lambda *a, **kw: children.append(a) or exit_code,
    )
    exports = []
    monkeypatch.setattr(
        generate_scenarios, "export_scenario_collection", lambda *a: exports.append(a)
    )
    context = SimpleNamespace(
        reservation_id="reserved", request={}, name="experiment", recheck=lambda: None
    )
    assert (
        generate_scenarios.run_owned(
            config, tmp_path, 1, [], invocation_id="test", reservation_context=context
        )
        == exit_code
    )
    assert len(children) == (0 if ready else 1)
    captured = capsys.readouterr()
    console = captured.out
    if ready:
        assert captured.err == "\nUp to date\n"
        assert "Scenario request" not in console
        assert "Reusing scenario collection" not in console
        assert plan["collection_id"] not in console
        assert "_engine" not in console
    else:
        assert "1 stochastic realization across 1 perturbation point" in console
        assert "planned collection: 2 runs" in console
        assert "Up to date" not in captured.err
    assert exports == (
        [(tmp_path, tmp_path / plan["outputs"]["record_root"] / "collection.json")]
        if exit_code == 0
        else []
    )


def test_source_phase_hides_console_chatter_only_on_success(capfd, monkeypatch):
    def noisy_child(_command, **_kwargs):
        # A genuine child process, not an in-process print: only a real OS-level
        # fd write proves the redirect (Python's own buffered streams do not
        # reliably observe a nested os.dup2 the way an inherited child fd does).
        subprocess.run(
            [sys.executable, "-c", "print('Using workflow specific profile ...')"],
            check=False,
        )
        return 0

    monkeypatch.setattr(generate_scenarios, "run_project_child", noisy_child)
    code = generate_scenarios._run_source_phase_quietly(
        ["snakemake", "--quiet", "all"], cwd=None, env={}
    )
    assert code == 0
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_source_phase_never_silences_warnings_on_success(capfd, monkeypatch):
    def warning_child(_command, **_kwargs):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "print('routine chatter'); print('WARNING: catalog lacks units')",
            ],
            check=False,
        )
        return 0

    monkeypatch.setattr(generate_scenarios, "run_project_child", warning_child)
    assert generate_scenarios._run_source_phase_quietly([], cwd=None, env={}) == 0
    captured = capfd.readouterr()
    assert "WARNING: catalog lacks units" in captured.err
    assert "routine chatter" not in captured.err


def test_in_process_writes_are_captured_and_errors_surface(capfd):
    with pytest.raises(RuntimeError, match="boom"):
        with output_capture.captured_output():
            print("in-process chatter")
            raise RuntimeError("boom")
    assert "in-process chatter" in capfd.readouterr().err


def test_source_phase_replays_console_chatter_on_failure(capfd, monkeypatch):
    def noisy_failing_child(_command, **_kwargs):
        subprocess.run(
            [sys.executable, "-c", "print('Using workflow specific profile ...')"],
            check=False,
        )
        return 7

    monkeypatch.setattr(generate_scenarios, "run_project_child", noisy_failing_child)
    code = generate_scenarios._run_source_phase_quietly(
        ["snakemake", "--quiet", "all"], cwd=None, env={}
    )
    assert code == 7
    captured = capfd.readouterr()
    assert "Using workflow specific profile" in captured.err


def test_source_phase_replay_reaches_a_real_terminal_fd():
    """Regression: pytest's ``capfd`` masks a replay-ordering bug that a real
    console does not. ``capfd`` redirects fd 1/2 for the whole test process
    before it starts, so ``_run_source_phase_quietly``'s own nested dup2/restore
    cycle still lands its replay write on a descendant of that same pipe either
    way -- the two prior tests above pass whether the replay runs before or
    after the restore. Only a fresh, unwrapped interpreter distinguishes them:
    before the fix, the replay wrote into the already-discarded capture temp
    file instead of the real fd, so a genuine failure surfaced no diagnostic
    at all (the exact silent-exit symptom this regression guards against).
    """
    script = (
        "import subprocess, sys\n"
        "sys.path.insert(0, '.')\n"
        "from scripts import generate_scenarios\n"
        "def noisy_failing_child(_command, **_kwargs):\n"
        "    subprocess.run([sys.executable, '-c', \"print('boom')\"], check=False)\n"
        "    return 7\n"
        "generate_scenarios.run_project_child = noisy_failing_child\n"
        "code = generate_scenarios._run_source_phase_quietly(['snakemake'], cwd=None, env={})\n"
        "sys.exit(code)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "boom" in result.stderr


@pytest.mark.parametrize("exit_code", [0, 9])
def test_direct_dry_run_records_terminal_invocation(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(
        generate_scenarios,
        "export_scenario_collection",
        lambda *a: pytest.fail("dry-run cannot export"),
    )
    config = tmp_path / "project_config.yml"
    config.write_text("project: {}\n", encoding="utf-8")
    project = tmp_path / "project"
    observed = []

    def fake_child(command, **kwargs):
        observed.append((command, kwargs))
        return exit_code

    monkeypatch.setattr(generate_scenarios, "run_project_child", fake_child)
    assert (
        generate_scenarios.main(
            [
                "--config",
                str(config),
                "--project-dir",
                str(project),
                "--cores",
                "2",
                "--",
                "--dry-run",
            ]
        )
        == exit_code
    )
    (path,) = (project / "config/_engine/invocations").glob("*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["workflow"] == "generate_scenarios"
    assert record["contract_mode"] == "new_schema"
    assert record["mode"] == "dry_run"
    assert record["status"] == ("succeeded" if exit_code == 0 else "failed")
    assert record["exit_code"] == exit_code
    assert len(observed) == 1
    assert observed[0][1]["writing"] is False


def test_direct_launch_exception_is_terminal(tmp_path, monkeypatch):
    config = tmp_path / "project_config.yml"
    config.write_text("project: {}\n", encoding="utf-8")
    project = tmp_path / "project"

    def fail_child(*_args, **_kwargs):
        raise OSError("snakemake unavailable")

    monkeypatch.setattr(generate_scenarios, "run_project_child", fail_child)
    with pytest.raises(OSError, match="snakemake unavailable"):
        generate_scenarios.main(
            [
                "--config",
                str(config),
                "--project-dir",
                str(project),
                "--cores",
                "2",
                "--",
                "--dry-run",
            ]
        )
    (path,) = (project / "config/_engine/invocations").glob("*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["type"] == "OSError"


def test_disabled_wf3_cannot_initialize_collection(tmp_path, monkeypatch):
    config = tmp_path / "project_config.yml"
    config.write_text("project: {}\n", encoding="utf-8")
    project = tmp_path / "project"
    calls = []
    monkeypatch.setattr(
        generate_scenarios,
        "run_project_child",
        lambda command, **_kwargs: calls.append(command) or 0,
    )
    monkeypatch.setattr(
        generate_scenarios,
        "compose_config",
        lambda *_args, **_kwargs: (
            {"workflows": {"generate_scenarios": {"enabled": False}}},
            {},
        ),
    )
    with pytest.raises(ValueError, match="WF3 is disabled"):
        generate_scenarios.main(
            [
                "--config",
                str(config),
                "--project-dir",
                str(project),
                "--cores",
                "2",
            ]
        )
    assert len(calls) == 1  # source preparation only; no generation child
    assert calls[0][-2:] == ["--quiet", "all"]
    assert not (project / "scenarios").exists()


def test_console_log_handlers_are_captured_and_restored(capfd):
    import logging

    logger = logging.getLogger("cst-test-capture")
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with output_capture.captured_output():
            assert handler.stream is not sys.__stderr__
            logger.warning("WARNING: captured through the handler")
        assert handler.stream is sys.stderr
    finally:
        logger.removeHandler(handler)
    err = capfd.readouterr().err
    assert "WARNING: captured through the handler" in err
    assert "Logging error" not in err
