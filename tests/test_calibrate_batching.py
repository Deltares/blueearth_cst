"""The calibration script decides on predicted WF4 time, not probe wall time."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_batching.py"
_spec = importlib.util.spec_from_file_location("calibrate_batching", SCRIPT)
calibrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate)


def test_cold_is_the_first_member_and_warm_the_rest():
    assert calibrate._cold_warm([[68.0, 3.0], [70.0, 4.0, 5.0]]) == (69.0, 4.0)


def test_prediction_splits_runs_evenly_across_the_batches():
    # 14 members, 2 at once -> 7 per batch: one cold, six warm.
    assert calibrate._predict(58.0, 12.5, 14, 2) == 58.0 + 6 * 12.5


def test_warm_time_decides_when_runs_are_many():
    """Short probes rank cold noise; the prediction weights warm by run count."""
    one_thread = calibrate._predict(68.0, 3.0, 14, 1)
    two_threads = calibrate._predict(55.0, 4.5, 14, 1)
    assert one_thread < two_threads


@pytest.mark.parametrize(
    ("cores", "widths"),
    [
        (None, [1]),
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (8, [1, 2, 4, 8]),
        (10, [1, 2, 4, 8, 10]),
        (12, [1, 2, 4, 8, 12]),
    ],
)
def test_default_widths_double_up_to_the_core_count(cores, widths):
    assert calibrate._default_widths(cores) == widths


def test_machine_cores_falls_back_to_logical(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "cpu_count", lambda logical=True: 6 if logical else None
    )
    assert calibrate._machine_cores() == 6


def test_each_width_runs_full_batches_of_the_member_count():
    """No member is dropped: width w is w batches of --members, not a split total."""
    configs = dict(calibrate._width_configs([1, 3, 8], 4, 2))
    assert configs == {
        "width1": [(4, 2)],
        "width3": [(4, 2)] * 3,
        "width8": [(4, 2)] * 8,
    }


def test_default_widths_above_the_memory_limit_are_dropped():
    kept, note = calibrate._guard_widths([1, 2, 4, 8, 10], 4, explicit=False)
    assert kept == [1, 2, 4]
    assert note.startswith("skipping width(s) 8, 10")


def test_explicit_widths_are_kept_with_a_warning():
    kept, note = calibrate._guard_widths([2, 6], 4, explicit=True)
    assert kept == [2, 6]
    assert note.startswith("warning: width(s) 6")


def test_widths_within_the_limit_print_nothing():
    assert calibrate._guard_widths([1, 2], 4, explicit=False) == ([1, 2], None)
    assert calibrate._guard_widths([1, 64], None, explicit=False) == ([1, 64], None)


def test_memory_limit_is_the_batch_plan_memory_rule(monkeypatch):
    """Room for exactly five batches in 80% of available memory gives five."""
    from types import SimpleNamespace

    import psutil

    from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS

    settings = ADVANCED_SETTINGS["batching"]
    cells = 1000
    per_batch = (
        settings["memory_per_batch_gb"] * (1 << 30)
        + cells * settings["memory_per_cell_kb"] * 1024
    )
    available = int(5 * per_batch / 0.8) + 1
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    assert calibrate._memory_limit(cells) == 5
