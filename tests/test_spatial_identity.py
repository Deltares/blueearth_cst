"""Tests for deterministic basin, subbasin, and location identities."""

from __future__ import annotations

import pandas as pd
import pytest

from blueearth_cst.spatial.identity import (
    assign_basin_ids,
    assign_location_ids,
    assign_subbasin_ids,
)


def test_basin_ids_are_deterministic_under_row_shuffle():
    """Parent IDs depend on hydrologic/grid attributes, never input row order."""
    basins = pd.DataFrame(
        [
            {"source": "small", "upstream_area": 20, "outlet_row": 4, "outlet_col": 8},
            {"source": "large", "upstream_area": 80, "outlet_row": 9, "outlet_col": 2},
        ]
    )

    first = assign_basin_ids(basins).set_index("source")
    shuffled = assign_basin_ids(basins.sample(frac=1, random_state=7)).set_index("source")

    pd.testing.assert_series_equal(first["basin_id"], shuffled["basin_id"])
    assert first.loc["large", "basin_code"] == "B001"
    assert first.loc["large", "basin_name"] == "basin_001"
    assert first.loc["small", "basin_code"] == "B002"


def test_subbasin_ids_follow_downstream_then_branch_tie_breaks():
    """Numbering is downstream-first, then area, row, and column."""
    subbasins = pd.DataFrame(
        [
            {"name": "branch_b", "basin_id": 1, "downstream_steps": 2, "upstream_area": 10, "outlet_row": 2, "outlet_col": 1},
            {"name": "outlet", "basin_id": 1, "downstream_steps": 0, "upstream_area": 100, "outlet_row": 9, "outlet_col": 9},
            {"name": "branch_a", "basin_id": 1, "downstream_steps": 2, "upstream_area": 20, "outlet_row": 8, "outlet_col": 4},
            {"name": "middle", "basin_id": 1, "downstream_steps": 1, "upstream_area": 60, "outlet_row": 5, "outlet_col": 5},
        ]
    )

    resolved = assign_subbasin_ids(subbasins).set_index("name")

    assert resolved.loc["outlet", "subbasin_id"] == 101
    assert resolved.loc["middle", "subbasin_id"] == 102
    assert resolved.loc["branch_a", "subbasin_id"] == 103
    assert resolved.loc["branch_b", "subbasin_id"] == 104
    assert resolved.loc["branch_b", "subbasin_code"] == "B001-S04"


def _locations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "basin_id": 1,
                "basin_code": "B001",
                "basin_name": "basin_001",
                "subbasin_id": 101,
                "subbasin_code": "B001-S01",
                "subbasin_name": "outlet",
                "station_name": "Secondary",
                "location_role": "observation",
                "is_primary": False,
                "original_x": 1.1,
                "original_y": 2.1,
                "snapped_x": 1.0,
                "snapped_y": 2.0,
                "snapped_row": 3,
                "snapped_col": 4,
            },
            {
                "basin_id": 1,
                "basin_code": "B001",
                "basin_name": "basin_001",
                "subbasin_id": 101,
                "subbasin_code": "B001-S01",
                "subbasin_name": "outlet",
                "station_name": "Primary",
                "location_role": "control",
                "is_primary": True,
                "original_x": 1.0,
                "original_y": 2.0,
                "snapped_x": 1.0,
                "snapped_y": 2.0,
                "snapped_row": 3,
                "snapped_col": 4,
            },
        ]
    )


def test_location_ids_follow_primary_and_reserved_additional_ranges():
    """Primary locations inherit subbasin IDs; extras use the reserved range."""
    resolved = assign_location_ids(_locations()).set_index("station_name")

    assert resolved.loc["Primary", "location_code"] == "B001-S01-L01"
    assert resolved.loc["Primary", "wflow_id"] == 101
    assert resolved.loc["Secondary", "location_code"] == "B001-S01-L02"
    assert resolved.loc["Secondary", "wflow_id"] == 1_010_102


def test_location_ids_are_deterministic_under_row_shuffle():
    """Registry IDs and codes do not depend on gauge file row order."""
    first = assign_location_ids(_locations()).sort_values("station_name")
    shuffled = assign_location_ids(
        _locations().sample(frac=1, random_state=9)
    ).sort_values("station_name")

    pd.testing.assert_frame_equal(first.reset_index(drop=True), shuffled.reset_index(drop=True))


def test_supplied_wflow_id_must_match_resolved_registry():
    """A stale user ID cannot silently override the deterministic hierarchy."""
    locations = _locations()
    locations["provided_wflow_id"] = [999, 101]

    with pytest.raises(ValueError, match="disagree"):
        assign_location_ids(locations)


def test_non_numeric_supplied_wflow_id_is_not_treated_as_missing():
    """A populated invalid ID fails instead of being silently regenerated."""
    locations = _locations()
    locations["provided_wflow_id"] = ["stale-id", 101]

    with pytest.raises(ValueError, match="must be integers"):
        assign_location_ids(locations)


def test_each_subbasin_requires_exactly_one_primary_location():
    """The subbasin-to-location identity edge is unambiguous."""
    locations = _locations()
    locations["is_primary"] = False

    with pytest.raises(ValueError, match="exactly one primary"):
        assign_location_ids(locations)
