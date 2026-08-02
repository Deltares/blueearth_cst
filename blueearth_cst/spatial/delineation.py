"""Model-neutral subbasin partitioning helpers built on public PyFlwDir APIs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from pyflwdir import FlwdirRaster


def allocate_automatic_subbasin_budgets(
    parent_areas: Mapping[int, float], max_count: int
) -> dict[int, int]:
    """Allocate a global automatic-subbasin ceiling across fallback parents.

    Every fallback parent receives one unit. Remaining units follow the
    largest-remainder method weighted by parent upstream area; basin ID breaks
    exact remainder ties.
    """
    if max_count < 1:
        raise ValueError("max_count must be >= 1")
    if not parent_areas:
        return {}
    if any(basin_id < 1 for basin_id in parent_areas):
        raise ValueError("parent basin IDs must be positive integers")
    if any(not np.isfinite(area) or area <= 0 for area in parent_areas.values()):
        raise ValueError("parent basin areas must be finite and > 0")
    if len(parent_areas) > max_count:
        raise ValueError(
            f"{len(parent_areas)} fallback parent basins exceed the global "
            f"automatic-subbasin ceiling of {max_count}"
        )

    ordered_ids = sorted(parent_areas)
    budgets = {basin_id: 1 for basin_id in ordered_ids}
    remaining = max_count - len(ordered_ids)
    if remaining == 0:
        return budgets

    total_area = sum(parent_areas.values())
    quotas = {
        basin_id: remaining * parent_areas[basin_id] / total_area
        for basin_id in ordered_ids
    }
    floors = {basin_id: int(np.floor(quotas[basin_id])) for basin_id in ordered_ids}
    for basin_id, floor in floors.items():
        budgets[basin_id] += floor
    left = remaining - sum(floors.values())
    remainder_order = sorted(
        ordered_ids,
        key=lambda basin_id: (-(quotas[basin_id] - floors[basin_id]), basin_id),
    )
    for basin_id in remainder_order[:left]:
        budgets[basin_id] += 1
    return budgets


def find_parent_outlet(
    flwdir: FlwdirRaster, parent_mask: np.ndarray, upstream_area: np.ndarray
) -> int:
    """Return the deterministic downstream outlet cell of one parent mask."""
    mask = np.asarray(parent_mask, dtype=bool)
    area = np.asarray(upstream_area)
    if mask.shape != flwdir.shape or area.shape != flwdir.shape:
        raise ValueError("parent mask, upstream area, and flow direction must align")
    cells = np.flatnonzero(mask.ravel())
    if cells.size == 0:
        raise ValueError("parent basin mask contains no cells")
    downstream = flwdir.idxs_ds[cells]
    leaves_parent = (downstream == cells) | (downstream < 0)
    valid_downstream = downstream >= 0
    leaves_parent[valid_downstream] |= ~mask.ravel()[downstream[valid_downstream]]
    candidates = cells[leaves_parent]
    if candidates.size == 0:
        raise ValueError("parent basin has no downstream outlet cell")

    rows, cols = np.unravel_index(candidates, flwdir.shape)
    candidate_areas = area.ravel()[candidates]
    order = np.lexsort((cols, rows, -candidate_areas))
    return int(candidates[order[0]])


def downstream_steps(
    flwdir: FlwdirRaster, start_index: int, target_index: int
) -> int:
    """Return downstream cell steps from a control point to its parent outlet."""
    target = np.zeros(flwdir.shape, dtype=bool)
    target.ravel()[target_index] = True
    paths, _ = flwdir.path(idxs=np.asarray([start_index]), mask=target, unit="cell")
    path = paths[0]
    if path.size == 0 or int(path[-1]) != target_index:
        raise ValueError(
            f"cell {start_index} does not drain to parent outlet {target_index}"
        )
    return int(path.size - 1)


def incremental_subbasins(
    flwdir: FlwdirRaster, outlet_indices: np.ndarray
) -> np.ndarray:
    """Return a non-overlapping temporary partition for selected outlets."""
    outlets = np.asarray(outlet_indices, dtype=np.int64)
    if outlets.ndim != 1 or outlets.size == 0:
        raise ValueError("at least one subbasin outlet index is required")
    if np.unique(outlets).size != outlets.size:
        raise ValueError("subbasin outlet indices must be unique")
    labels = np.arange(1, outlets.size + 1, dtype=np.int32)
    return flwdir.basins(idxs=outlets, ids=labels).astype(np.int32)


def full_catchment(flwdir: FlwdirRaster, outlet_index: int) -> np.ndarray:
    """Return the full contributing catchment of one control point."""
    return flwdir.basins(
        idxs=np.asarray([outlet_index]), ids=np.asarray([1], dtype=np.int32)
    ).astype(bool)


def select_automatic_subbasins(
    flwdir: FlwdirRaster,
    upstream_area: np.ndarray,
    max_count: int,
    outlet_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the most detailed area partition that respects ``max_count``.

    Candidate thresholds are the unique upstream-area values on valid cells.
    PyFlwDir's area partition count is monotonic as that threshold grows, so a
    binary search finds the smallest threshold whose outlet count is within the
    configured budget. Returned labels are temporary; the identity contract
    replaces them after hydrologic ordering.
    """
    if max_count < 1:
        raise ValueError("max_count must be >= 1")
    area = np.asarray(upstream_area, dtype=float)
    if area.shape != flwdir.shape:
        raise ValueError("upstream area and flow direction must align")
    valid = flwdir.mask.reshape(flwdir.shape) & np.isfinite(area) & (area > 0)
    if outlet_mask is not None:
        eligible = np.asarray(outlet_mask, dtype=bool)
        if eligible.shape != flwdir.shape:
            raise ValueError("outlet mask and flow direction must align")
        valid &= eligible
    thresholds = np.unique(area[valid])
    if thresholds.size == 0:
        raise ValueError(
            "automatic partitioning requires at least one eligible cell with "
            "positive upstream area"
        )

    selected_outlets = None
    low = 0
    high = thresholds.size - 1
    while low <= high:
        middle = (low + high) // 2
        _, candidate_outlets = flwdir.subbasins_area(
            area_min=float(thresholds[middle]), uparea=area
        )
        if candidate_outlets.size <= max_count:
            selected_outlets = candidate_outlets
            high = middle - 1
        else:
            low = middle + 1

    if selected_outlets is None or selected_outlets.size == 0:
        raise RuntimeError(
            f"unable to derive an automatic partition within max_count={max_count}"
        )
    if selected_outlets.size > max_count:
        raise RuntimeError("automatic partition exceeded its configured ceiling")
    partition = incremental_subbasins(flwdir, selected_outlets)
    partition[~flwdir.mask.reshape(flwdir.shape)] = 0
    return partition, np.asarray(selected_outlets, dtype=np.int64)
