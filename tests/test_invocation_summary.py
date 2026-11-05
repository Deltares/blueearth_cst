"""Views of old, partial and timed records must not invent evidence."""

import json
import os

import pytest

from blueearth_cst.shared import invocation_history as history
from blueearth_cst.shared import invocation_summary as summary


def _start(root, **kwargs):
    return history.start(
        root,
        workflow="analyze_climate",
        entry_point="test",
        command=[],
        targets=[],
        mode="execute",
        contract_mode="test",
        **kwargs,
    )


def test_monotonic_duration_is_numeric_without_a_saved_clock_origin(
    tmp_path, monkeypatch
):
    ticks = iter([100.0, 104.25])
    monkeypatch.setattr(history.time, "monotonic", lambda: next(ticks))
    path, record = _start(tmp_path)
    history.finish(path, record, exit_code=0)
    saved = json.loads(path.read_text())
    assert saved["elapsed_seconds"] == 4.25
    assert not any("monotonic" in key for key in saved)
    assert saved["work_performed"] == "unknown"
    assert "4.250 s" in summary.summary_path(path).read_text()


def test_foreign_process_never_uses_another_process_timer(tmp_path, monkeypatch):
    path, record = _start(tmp_path)
    pid = os.getpid()
    monkeypatch.setattr(history.os, "getpid", lambda: pid + 1)
    history.finish(path, record, exit_code=0)
    assert record["elapsed_seconds"] is None
    assert "Elapsed: unavailable" in summary.render_summary(path)


def test_supervisor_can_supply_observed_duration(tmp_path):
    path, record = _start(tmp_path)
    history._STARTED.pop(path)
    history.finish(path, record, exit_code=3, elapsed_seconds=12.5)
    assert json.loads(path.read_text())["elapsed_seconds"] == 12.5


@pytest.mark.parametrize("finished", [False, True])
def test_old_or_unfinished_record_has_no_invented_duration(tmp_path, finished):
    path, record = _start(tmp_path)
    record.pop("elapsed_seconds")
    if finished:
        record.update(
            status="succeeded", exit_code=0, ended_at_utc="2026-01-01T02:00:00Z"
        )
    path.write_text(json.dumps(record), encoding="utf-8")
    before = path.read_bytes()
    text = summary.write_summary(path).read_text()
    assert "Elapsed: unavailable" in text
    assert "Work performed: unknown" in text
    if not finished:
        assert "unfinished (last observed running)" in text
        assert "Ended (UTC): not observed" in text
    assert path.read_bytes() == before


def test_wf4_diagnostics_and_warning_counts_share_the_run_summary(tmp_path):
    path, record = _start(tmp_path)
    record["warning_count"] = 2
    record["resources"] = {
        "status": "finished",
        "directory": str(path.with_suffix("") / "diagnostics"),
        "sampled_mean_cpu_cores": 1.1,
        "sampled_peak_tree_rss_bytes": 2048,
    }
    history.update(path, record)
    history.finish(path, record, exit_code=0)
    text = summary.render_summary(path)
    assert "| 2 | unknown |" in text
    assert "Resource trace and Julia/batch diagnostics" in text
    assert "Sampled mean CPU cores: 1.1" in text
