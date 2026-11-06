"""Independent event examples, including unknown boundaries and period overlap."""

import numpy as np
import pytest

from blueearth_cst.indicators.climate import Event, detect_events


def test_missing_observations_break_runs_and_mark_censoring():
    # Runs: edge-censored 2, complete 1, gap-censored 2, edge-censored 1.
    condition = np.array([True, True, False, True, False, True, True, True, True])
    valid = np.array([True, True, True, True, True, True, True, False, True])
    assert detect_events(condition, valid) == (
        Event(0, 2, True, False),
        Event(3, 4, False, False),
        Event(5, 7, False, True),
        Event(8, 9, True, True),
    )
    assert [
        e.duration for e in detect_events(condition, valid, minimum_duration=2)
    ] == [
        2,
        2,
    ]


def test_period_boundary_does_not_split_detection():
    condition = np.array([False, True, True, True, True, True, True, False])
    (event,) = detect_events(condition, np.ones(8, dtype=bool), minimum_duration=6)
    assert event == Event(1, 7, False, False)
    assert event.duration == 6
    assert event.overlap(0, 4) == 3
    assert event.overlap(4, 8) == 3
    assert event.overlap(8, 10) == 0


@pytest.mark.parametrize("length", [0, 1, 5])
def test_no_events_for_false_or_unknown_days(length):
    false = np.zeros(length, dtype=bool)
    true = np.ones(length, dtype=bool)
    assert detect_events(false, true) == ()
    assert detect_events(true, false) == ()


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True, np.bool_(True)])
def test_invalid_duration_is_rejected(minimum):
    with pytest.raises(ValueError, match="positive integer"):
        detect_events(np.array([True]), np.array([True]), minimum_duration=minimum)


@pytest.mark.parametrize(
    "condition,valid",
    [([1, 0], [True, True]), ([True], [1]), ([[True]], [[True]]), ([True], [])],
)
def test_invalid_masks_are_not_coerced(condition, valid):
    with pytest.raises(ValueError, match="Boolean arrays"):
        detect_events(np.asarray(condition), np.asarray(valid))


def test_reversed_reporting_interval_is_rejected():
    with pytest.raises(ValueError, match="start <= stop"):
        Event(1, 4, False, False).overlap(5, 2)
