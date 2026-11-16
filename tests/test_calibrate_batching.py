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
