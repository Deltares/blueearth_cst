"""Resource observations remain attributable and cannot replace run outcomes."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueearth_cst.shared import invocation_history, resource_tracking


@pytest.mark.parametrize(
    "scenario, expected_cpu",
    [
        ("new", 7),
        ("existing", 4),
        ("reused_pid", 7),
        ("denied_census", 4),
        ("denied_counters", 4),
    ],
)
def test_cpu_baseline_uses_initial_census(
    tmp_path, monkeypatch, scenario, expected_cpu
):
    """Rounded creation times must not erase a new child's observed CPU."""
    monkeypatch.setattr(resource_tracking.time, "time", lambda: 100.005)
    cpu = {"child": 3}
    created = {"child": 100.0}
    children = []

    def process(pid, name):
        return SimpleNamespace(
            pid=pid,
            create_time=lambda: created[name] if name == "child" else 1.0,
            cpu_times=lambda: SimpleNamespace(user=cpu.get(name, 20), system=0),
            memory_info=lambda: SimpleNamespace(rss=1024, vms=2048),
            io_counters=lambda: SimpleNamespace(read_bytes=0, write_bytes=0),
            name=lambda: name,
            cpu_affinity=lambda: [0],
            nice=lambda: 0,
            children=lambda recursive: list(children),
        )

    root = process(1, "root")
    child = process(2, "child")
    monkeypatch.setattr(resource_tracking.psutil, "Process", lambda: root)
    if scenario in {"existing", "reused_pid", "denied_counters"}:
        children.append(child)
    if scenario == "denied_census":

        def denied(recursive):
            raise resource_tracking.psutil.AccessDenied()

        root.children = denied
    if scenario == "denied_counters":

        def denied_counters():
            raise resource_tracking.psutil.AccessDenied()

        child.cpu_times = denied_counters
    monitor = resource_tracking.ResourceMonitor(tmp_path / "diagnostics", interval=60)
    monitor.start()
    try:
        root.children = lambda recursive: [child]
        child.cpu_times = lambda: SimpleNamespace(user=cpu["child"], system=0)
        if scenario == "reused_pid":
            created["child"] = 100.001
        # Observe the child before and after additional work, then after exit.
        monitor.sample()
        cpu["child"] = 7
        monitor.sample()
        root.children = lambda recursive: []
    finally:
        summary = monitor.finish()
    assert summary["sampled_tree_cpu_s"] == expected_cpu


def test_trace_tracks_child_and_retains_cpu_after_exit(tmp_path):
    monitor = resource_tracking.ResourceMonitor(tmp_path / "diagnostics", interval=60)
    monitor.start()
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; data=bytearray(16*1024*1024); sum(range(1000000)); "
            "print('ready', flush=True); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        monitor.mark("snakemake")
        monitor.sample()
        identity = next(key for key in monitor.processes if key[0] == child.pid)
        observed_cpu = monitor.processes[identity]["last"]["cpu_s"]
        child.communicate(timeout=10)
        monitor.mark("indicator_exports")
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
        summary = monitor.finish()
    rows = [
        json.loads(line)
        for line in (monitor.directory / "resources.jsonl").read_text().splitlines()
    ]
    assert rows[0]["schema_version"] == "wf4-resources/1"
    assert rows[-1]["kind"] == "summary"
    assert summary["sampled_tree_cpu_s"] >= observed_cpu
    assert summary["sampled_peak_tree_rss_bytes"] >= 16 * 1024 * 1024
    assert summary["minimum_available_memory_bytes"] > 0
    assert [phase["name"] for phase in summary["phases"]] == [
        "preparation",
        "snakemake",
        "indicator_exports",
    ]
    assert all(phase["wall_s"] >= 0 for phase in summary["phases"])
    assert not any("command" in row for row in rows)
    assert any(row["kind"] == "process" and row["pid"] == child.pid for row in rows)


def test_unsupported_metrics_are_null_and_counted(tmp_path, monkeypatch):
    def denied():
        raise resource_tracking.psutil.AccessDenied()

    monkeypatch.setattr(resource_tracking.psutil, "cpu_freq", denied)
    monitor = resource_tracking.ResourceMonitor(tmp_path / "diagnostics", interval=60)
    monitor.start()
    summary = monitor.finish()
    rows = [
        json.loads(line)
        for line in (monitor.directory / "resources.jsonl").read_text().splitlines()
    ]
    assert all(
        row["cpu_frequency_mhz"] is None for row in rows if row["kind"] == "sample"
    )
    assert summary["measurement_errors"]["cpu_freq"] >= 1


@pytest.mark.parametrize("mode", ["execute", "dry_run"])
def test_invocation_finishes_tracking_even_on_workflow_failure(tmp_path, mode):
    path, record = invocation_history.start(
        tmp_path,
        workflow="simulate_system",
        entry_point="test",
        command=["test"],
        targets=["all"],
        mode=mode,
        contract_mode="test",
    )
    resource_tracking.phase(path, "snakemake")
    invocation_history.finish(path, record, exit_code=7)
    saved = json.loads(path.read_text())
    assert saved["exit_code"] == 7
    assert saved["status"] == "failed"
    assert path not in resource_tracking._MONITORS
    if mode == "execute":
        assert saved["resources"]["status"] == "finished"
        assert (
            Path(saved["resources"]["directory"]).joinpath("resources.jsonl").is_file()
        )
    else:
        assert "resources" not in saved


def test_tracking_failure_does_not_prevent_invocation(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise PermissionError("diagnostics denied")

    monkeypatch.setattr(resource_tracking, "ResourceMonitor", unavailable)
    with pytest.warns(UserWarning, match="resource tracking unavailable"):
        path, record = invocation_history.start(
            tmp_path,
            workflow="simulate_system",
            entry_point="test",
            command=[],
            targets=[],
            mode="execute",
            contract_mode="test",
        )
    invocation_history.finish(path, record, exit_code=0)
    assert record["status"] == "succeeded"
    assert record["resources"]["status"] == "unavailable"


def test_batch_decision_records_requested_threads_and_members(tmp_path, monkeypatch):
    monkeypatch.setenv(resource_tracking.DIRECTORY_ENV, str(tmp_path))
    resource_tracking.write_batch_decision(
        "0",
        {"plan": {"available_memory_bytes": 123}},
        threads=2,
        run_ids=["run_01", "run_02"],
    )
    (path,) = tmp_path.glob("batch-*.json")
    saved = json.loads(path.read_text())
    assert saved["plan"]["available_memory_bytes"] == 123
    assert saved["rule_requested_threads"] == 2
    assert saved["run_ids"] == ["run_01", "run_02"]
