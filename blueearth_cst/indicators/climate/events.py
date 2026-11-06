"""Continuous event detection without reporting-period or missing-data bridging."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Event:
    """An observed run on a contiguous daily axis.

    ``start`` is inclusive and ``stop`` exclusive. Censoring indicates an unknown
    adjacent day or the record boundary; duration is then a lower bound on the
    full physical event. Dates and location identities belong to the caller.
    """

    start: int
    stop: int
    left_censored: bool
    right_censored: bool

    @property
    def duration(self) -> int:
        """Number of observed qualifying days."""
        return self.stop - self.start

    def overlap(self, start: int, stop: int) -> int:
        """Count observed event days in a half-open reporting interval."""
        if start > stop:
            raise ValueError("Reporting interval requires start <= stop")
        return max(0, min(self.stop, stop) - max(self.start, start))


def detect_events(
    condition: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    minimum_duration: int = 1,
) -> tuple[Event, ...]:
    """Find qualifying observed runs, retaining censoring at gaps and edges.

    Args:
        condition: One-dimensional Boolean predicate on a contiguous daily axis.
        valid: Boolean observation mask on the same axis. Missing calendar days
            must be inserted as invalid before calling; false conditions alone
            cannot represent missing observations.
        minimum_duration: Minimum observed run length, a positive integer.

    Returns:
        Events in time order. Missing values break runs; they never count as
        qualifying or as known nonqualifying days. Censored runs shorter than
        the minimum are omitted because their observed length does not establish
        qualification. No period clipping or event-count allocation is applied.

    Raises:
        ValueError: Masks are not aligned one-dimensional Boolean arrays, or the
            minimum duration is not a positive integer.
    """
    condition = np.asarray(condition)
    valid = np.asarray(valid)
    if (
        condition.ndim != 1
        or valid.shape != condition.shape
        or condition.dtype != np.dtype(bool)
        or valid.dtype != np.dtype(bool)
    ):
        raise ValueError("condition and valid require aligned 1D Boolean arrays")
    if (
        isinstance(minimum_duration, (bool, np.bool_))
        or not isinstance(minimum_duration, (int, np.integer))
        or minimum_duration < 1
    ):
        raise ValueError("minimum_duration requires a positive integer")
    active = condition & valid
    edges = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return tuple(
        Event(
            start=int(start),
            stop=int(stop),
            left_censored=bool(start == 0 or not valid[start - 1]),
            right_censored=bool(stop == len(valid) or not valid[stop]),
        )
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_duration
    )
