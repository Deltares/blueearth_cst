"""WF4 batch plan: threads per batch and batches at once (t2609242342)."""

from __future__ import annotations

import pytest

from blueearth_cst.experiment.batch_sizing import (
    BYTES_PER_GB,
    resolve_batch_plan,
    resolve_batch_size,
)
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS

SETTINGS = dict(ADVANCED_SETTINGS["batching"])
PLENTY = 64 * BYTES_PER_GB


def test_a_small_basin_gets_the_measured_regime():
    plan = resolve_batch_plan(257, 12, SETTINGS, available_memory_bytes=PLENTY)
    assert (plan.threads, plan.max_parallel, plan.regime) == (1, 2, "small basin")


def test_a_large_basin_gets_the_provisional_regime():
    plan = resolve_batch_plan(50_000, 12, SETTINGS, available_memory_bytes=PLENTY)
    assert (plan.threads, plan.max_parallel, plan.regime) == (4, 1, "large basin")


def test_an_unknown_size_takes_the_conservative_regime():
    plan = resolve_batch_plan(None, 12, SETTINGS)
    assert (plan.threads, plan.max_parallel) == (4, 1)
    assert plan.regime == "unknown size"


def test_a_project_override_wins_and_is_named():
    plan = resolve_batch_plan(
        257, 12, SETTINGS, {"julia_threads": 2, "max_parallel_batches": 3}, PLENTY
    )
    assert (plan.threads, plan.max_parallel, plan.regime) == (2, 3, "project")


def test_a_numeric_setting_beats_the_regime():
    settings = {**SETTINGS, "threads": 3}
    plan = resolve_batch_plan(257, 12, settings, available_memory_bytes=PLENTY)
    assert (plan.threads, plan.regime) == (3, "settings + small basin")


def test_cores_cap_batches_at_once():
    plan = resolve_batch_plan(
        257, 4, SETTINGS, {"julia_threads": 2, "max_parallel_batches": 4}, PLENTY
    )
    assert plan.max_parallel == 2 and plan.regime.endswith("cores")


def test_memory_caps_batches_at_once():
    plan = resolve_batch_plan(
        257, 12, SETTINGS, available_memory_bytes=2 * BYTES_PER_GB
    )
    assert plan.max_parallel == 1 and plan.regime.endswith("memory")
    assert plan.available_memory_bytes == 2 * BYTES_PER_GB
    assert plan.requested_parallel == 2
    assert plan.core_limit == 12
    assert plan.memory_limit == 1
    assert plan.estimated_memory_per_batch_bytes == int(
        SETTINGS["memory_per_batch_gb"] * BYTES_PER_GB
        + 257 * SETTINGS["memory_per_cell_kb"] * 1024
    )


def test_the_summary_says_what_chose_it():
    plan = resolve_batch_plan(257, 12, SETTINGS, available_memory_bytes=PLENTY)
    assert plan.summary() == (
        "2 batches at once, 1 Julia thread each (small basin, 257 cells)"
    )


@pytest.mark.parametrize(("members", "slots", "size"), [(14, 2, 7), (40, 2, 20)])
def test_without_a_cap_each_slot_gets_one_batch(members, slots, size):
    """No `batch_size_max` means no extra cold starts: one batch per slot."""
    sizing = resolve_batch_size(member_count=members, cores=slots, batch_size_max=None)
    assert sizing.batch_size == size
