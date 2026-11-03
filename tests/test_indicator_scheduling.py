"""Checkpoint expansion reuses identities without caching incomplete work."""

import os
from types import SimpleNamespace

import pytest
from snakemake.exceptions import IncompleteCheckpointException

from blueearth_cst.experiment import indicator_plan
from blueearth_cst.experiment.content_identity import atomic_record
from tests.simulation_workflow_fixture import parse_simulation_workflow


@pytest.fixture
def scheduling(tmp_path, monkeypatch):
    workflow = parse_simulation_workflow(tmp_path, monkeypatch)
    return workflow.globals["_current_indicator_requests"].__globals__


@pytest.mark.parametrize(
    "operation",
    ["simulate-and-indicators", "indicators-only", "reuse", "simulations-only"],
)
def test_simulation_startup_details_wait_for_header(
    scheduling, operation, monkeypatch, capsys
):
    from blueearth_cst.shared import console_style

    monkeypatch.setattr(console_style, "_CONSOLE_STYLE_ACTIVE", True)
    monkeypatch.setattr(console_style, "_RUN_HEADER", None)
    monkeypatch.setitem(
        scheduling,
        "OPERATION",
        "indicators-only"
        if operation == "indicators-only"
        else "simulate-and-indicators",
    )
    monkeypatch.setitem(scheduling, "_simulation_complete", operation == "reuse")
    monkeypatch.setitem(scheduling, "stored", {"simulation_id": "retained-id"})
    monkeypatch.setitem(scheduling["COLLECTION"], "collection_id", "collection-id")
    if operation == "simulations-only":
        monkeypatch.setitem(scheduling, "INDICATOR_SETS", ())
    if operation == "indicators-only":
        for name in ("RUN_IDS", "SIM_WINDOW_START", "SIM_WINDOW_END"):
            monkeypatch.delitem(scheduling, name)
    capsys.readouterr()
    scheduling["_header"]()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    details = console_style._RUN_HEADER[3]
    assert console_style._RUN_HEADER[2] == scheduling["config_path"]
    header = console_style.run_header(*console_style._RUN_HEADER[:3], **details)
    paths = header.splitlines()[header.splitlines().index("Paths") + 1 :]
    assert paths[0].split()[0] == "config"
    assert "compute" not in details and "operation" not in details
    assert details["water year"].startswith("Starts in")
    if operation == "simulations-only":
        assert details["analysis"] == "Hydrological simulation"
        assert "indicator sets" not in details
    else:
        assert details["indicator sets"]
    if operation == "indicators-only":
        assert details["simulation results"] == "Using saved simulations"
        assert "simulation period" not in details
        assert "parallel simulations" not in details
    else:
        assert details["scenario series"]
        assert details["simulation period"] == "2046–2054"
        if operation == "reuse":
            assert details["simulation results"] == "Reusing saved simulations"


def test_requests_wait_for_checkpoint_and_resolve_only_once(scheduling, monkeypatch):
    state = {"ready": False, "observations": 0, "checkpoint_calls": 0}

    def frozen(_):
        state["checkpoint_calls"] += 1
        if not state["ready"]:
            raise IncompleteCheckpointException(None, "intent.json")

    def resolve(root, calculations, anchor):
        state["observations"] += 1
        return [{"calculations": records} for records in calculations]

    monkeypatch.setitem(scheduling, "_frozen_simulation", frozen)
    monkeypatch.setitem(
        scheduling,
        "INDICATOR_CALCULATIONS",
        [[{"indicator": "a"}], [{"indicator": "b"}]],
    )
    monkeypatch.setitem(scheduling, "INDICATOR_SETS", ())
    monkeypatch.setattr(indicator_plan, "current_indicator_requests", resolve)
    current = scheduling["_current_indicator_requests"]
    with pytest.raises(IncompleteCheckpointException):
        current()
    assert state["observations"] == 0
    assert scheduling["_INDICATOR_REQUESTS"] is None

    state["ready"] = True
    expected = current()
    assert len(expected) == 2
    for _ in range(5):
        assert current() == expected
    assert state == {"ready": True, "observations": 1, "checkpoint_calls": 7}


def test_only_successful_exact_checkpoint_plans_are_reused(
    scheduling, tmp_path, monkeypatch
):
    path = tmp_path / "plan.json"
    marker = tmp_path / "not-published.json"
    atomic_record(path, {"revision": 1})
    state = {"valid": False, "verifications": 0}

    def verify(*_, _observed_code):
        assert _observed_code == []
        state["verifications"] += 1
        if not state["valid"]:
            raise indicator_plan.IndicatorPlanStale("stale responses")
        return {"targets": {"manifest": str(marker)}}

    monkeypatch.setitem(
        scheduling,
        "_current_indicator_requests",
        lambda _: {"id": {"code_inventory": []}},
    )
    monkeypatch.setitem(
        scheduling,
        "checkpoints",
        SimpleNamespace(
            prepare_indicator_plan=SimpleNamespace(
                get=lambda **_: SimpleNamespace(output=[str(path)])
            )
        ),
    )
    monkeypatch.setattr(indicator_plan, "verify_indicator_plan", verify)
    outputs = scheduling["_selected_indicator_outputs"]
    with pytest.raises(indicator_plan.IndicatorPlanStale, match="stale responses"):
        outputs(None)
    state["valid"] = True
    assert outputs(None) == [str(marker)]
    assert outputs(None) == [str(marker)]
    assert state["verifications"] == 2

    # Content changes invalidate the cache even if a checkpoint keeps its mtime.
    stat = path.stat()
    atomic_record(path, {"revision": 2}, replace=True)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert outputs(None) == [str(marker)]
    assert state["verifications"] == 3
