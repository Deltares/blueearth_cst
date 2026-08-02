"""Tests for deterministic, ceiling-bounded subbasin partitioning."""

from __future__ import annotations

import numpy as np
import pyflwdir
import pytest
from affine import Affine

from blueearth_cst.spatial.delineation import (
    allocate_automatic_subbasin_budgets,
    downstream_steps,
    find_parent_outlet,
    full_catchment,
    incremental_subbasins,
    select_automatic_subbasins,
)


def _flow_network():
    """Create a small deterministic dendritic network draining southeast."""
    elevation = np.asarray(
        [
            [9, 8, 7, 6, 5],
            [8, 7, 6, 5, 4],
            [7, 6, 5, 4, 3],
            [6, 5, 4, 3, 2],
            [5, 4, 3, 2, 1],
        ],
        dtype=float,
    )
    return pyflwdir.from_dem(
        elevation,
        transform=Affine(1000, 0, 0, 0, -1000, 5000),
        outlets="min",
    )


def test_budget_allocation_respects_global_ceiling_and_parent_minimum():
    """The ceiling is global while every fallback parent receives one unit."""
    budgets = allocate_automatic_subbasin_budgets({3: 20.0, 1: 70.0, 2: 10.0}, 8)

    assert budgets == {1: 5, 2: 1, 3: 2}
    assert sum(budgets.values()) == 8


def test_budget_allocation_fails_when_parents_alone_exceed_ceiling():
    """No parent basin is silently dropped to satisfy the global limit."""
    with pytest.raises(ValueError, match="exceed"):
        allocate_automatic_subbasin_budgets({1: 1.0, 2: 1.0, 3: 1.0}, 2)


def test_parent_outlet_and_downstream_steps_follow_flow_network():
    """Outlet choice and ordering use topology rather than input order."""
    flwdir = _flow_network()
    mask = np.ones(flwdir.shape, dtype=bool)
    area = flwdir.upstream_area("km2")
    outlet = find_parent_outlet(flwdir, mask, area)

    assert outlet == np.ravel_multi_index((4, 4), flwdir.shape)
    assert downstream_steps(flwdir, np.ravel_multi_index((0, 0), flwdir.shape), outlet) > 0


def test_incremental_partition_and_full_catchments_have_distinct_semantics():
    """Incremental units do not overlap; full nested catchments can."""
    flwdir = _flow_network()
    upstream = np.ravel_multi_index((2, 2), flwdir.shape)
    outlet = np.ravel_multi_index((4, 4), flwdir.shape)

    partition = incremental_subbasins(flwdir, np.asarray([outlet, upstream]))
    outlet_catchment = full_catchment(flwdir, outlet)
    upstream_catchment = full_catchment(flwdir, upstream)

    assert set(np.unique(partition)) == {1, 2}
    assert np.all(upstream_catchment <= outlet_catchment)
    assert np.any(upstream_catchment & outlet_catchment)


@pytest.mark.parametrize("ceiling", [1, 2, 4, 20])
def test_automatic_partition_never_exceeds_ceiling(ceiling):
    """Binary-search coarsening enforces the configured maximum."""
    flwdir = _flow_network()
    area = flwdir.upstream_area("km2")

    partition, outlets = select_automatic_subbasins(flwdir, area, ceiling)

    assert 1 <= outlets.size <= ceiling
    assert len(np.unique(partition[partition > 0])) == outlets.size


def test_automatic_partition_outlets_respect_eligibility_mask():
    """Automatic locations can be constrained to active model river cells."""
    flwdir = _flow_network()
    area = flwdir.upstream_area("km2")
    outlet_mask = area >= 6

    _, outlets = select_automatic_subbasins(
        flwdir, area, max_count=20, outlet_mask=outlet_mask
    )

    assert outlets.size < 20
    assert outlet_mask.ravel()[outlets].all()


def test_duplicate_control_cells_are_rejected():
    """Two controls cannot silently compete for the same spatial unit."""
    flwdir = _flow_network()
    outlet = np.ravel_multi_index((4, 4), flwdir.shape)

    with pytest.raises(ValueError, match="unique"):
        incremental_subbasins(flwdir, np.asarray([outlet, outlet]))
