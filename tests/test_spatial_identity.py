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
    shuffled = assign_basin_ids(basins.sample(frac=1, random_state=7)).set_index(
        "source"
    )

    pd.testing.assert_series_equal(first["basin_id"], shuffled["basin_id"])
    assert first.loc["large", "basin_code"] == "B001"
    assert first.loc["large", "basin_name"] == "basin_001"
    assert first.loc["small", "basin_code"] == "B002"


def test_subbasin_ids_follow_downstream_then_branch_tie_breaks():
    """Numbering is downstream-first, then area, row, and column."""
    subbasins = pd.DataFrame(
        [
            {
                "name": "branch_b",
                "basin_id": 1,
                "downstream_steps": 2,
                "upstream_area": 10,
                "outlet_row": 2,
                "outlet_col": 1,
            },
            {
                "name": "outlet",
                "basin_id": 1,
                "downstream_steps": 0,
                "upstream_area": 100,
                "outlet_row": 9,
                "outlet_col": 9,
            },
            {
                "name": "branch_a",
                "basin_id": 1,
                "downstream_steps": 2,
                "upstream_area": 20,
                "outlet_row": 8,
                "outlet_col": 4,
            },
            {
                "name": "middle",
                "basin_id": 1,
                "downstream_steps": 1,
                "upstream_area": 60,
                "outlet_row": 5,
                "outlet_col": 5,
            },
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
                "local_subbasin_number": 1,
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
                "local_subbasin_number": 1,
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


def test_location_ids_decode_as_basin_subbasin_member():
    """ADR 0003 §12: `wflow_id = basin_id*1000 + local_subbasin*10 + m`.

    This REPLACES two unrelated formulas that shared one column — a primary got
    its three-digit `subbasin_id`, any additional point got a seven-digit
    `1_000_000 + subbasin_id*100 + n` — so sibling points sat four orders of
    magnitude apart. `location_code` is deliberately unchanged: codes are for
    reading, `wflow_id` is the integer for joining and for scanning a CSV
    header.
    """
    resolved = assign_location_ids(_locations()).set_index("station_name")

    assert resolved.loc["Primary", "location_code"] == "B001-S01-L01"
    assert resolved.loc["Primary", "wflow_id"] == 1010
    assert resolved.loc["Secondary", "location_code"] == "B001-S01-L02"
    assert resolved.loc["Secondary", "wflow_id"] == 1011
    # The decode every consumer relies on: a primary ends in 0.
    assert resolved.loc["Primary", "wflow_id"] % 10 == 0
    assert resolved.loc["Secondary", "wflow_id"] % 10 != 0


def test_wflow_ids_group_by_basin_and_order_by_subbasin():
    """The property §12 exists for: ids sort into per-basin blocks."""
    rows = []
    for basin_id in (1, 2):
        for local_subbasin in (1, 2):
            rows.append(
                {
                    "basin_id": basin_id,
                    "basin_code": f"B{basin_id:03d}",
                    "basin_name": f"basin_{basin_id:03d}",
                    "subbasin_id": basin_id * 100 + local_subbasin,
                    "subbasin_code": f"B{basin_id:03d}-S{local_subbasin:02d}",
                    "subbasin_name": "unit",
                    "local_subbasin_number": local_subbasin,
                    "station_name": f"P{basin_id}{local_subbasin}",
                    "location_role": "control",
                    "is_primary": True,
                    "original_x": 1.0,
                    "original_y": 2.0,
                    "snapped_x": 1.0,
                    "snapped_y": 2.0,
                    "snapped_row": 3,
                    "snapped_col": 4,
                }
            )
    resolved = assign_location_ids(pd.DataFrame(rows))

    assert resolved["wflow_id"].tolist() == [1010, 1020, 2010, 2020]
    assert resolved["wflow_id"].is_unique


def test_a_tenth_additional_location_in_one_subbasin_raises():
    """The collision boundary, not a preference: m=10 lands on the next
    subbasin's primary (`sub*10 + 10 == (sub+1)*10`)."""
    base = _locations().iloc[1].to_dict()  # the primary
    extra = _locations().iloc[0].to_dict()  # an additional point
    rows = [base]
    for n in range(10):  # ten additional -> m = 1..10
        row = dict(extra)
        row["station_name"] = f"Extra{n:02d}"
        row["snapped_row"] = 10 + n
        rows.append(row)

    with pytest.raises(ValueError, match="additional locations"):
        assign_location_ids(pd.DataFrame(rows))


def test_location_ids_are_deterministic_under_row_shuffle():
    """Registry IDs and codes do not depend on gauge file row order."""
    first = assign_location_ids(_locations()).sort_values("station_name")
    shuffled = assign_location_ids(
        _locations().sample(frac=1, random_state=9)
    ).sort_values("station_name")

    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), shuffled.reset_index(drop=True)
    )


def test_supplied_wflow_id_must_match_resolved_registry():
    """A stale user ID cannot silently override the deterministic hierarchy."""
    locations = _locations()
    locations["provided_wflow_id"] = [999, 1010]

    with pytest.raises(ValueError, match="disagree"):
        assign_location_ids(locations)


def test_non_numeric_supplied_wflow_id_is_not_treated_as_missing():
    """A populated invalid ID fails instead of being silently regenerated."""
    locations = _locations()
    locations["provided_wflow_id"] = ["stale-id", 1010]

    with pytest.raises(ValueError, match="must be integers"):
        assign_location_ids(locations)


def test_each_subbasin_requires_exactly_one_primary_location():
    """The subbasin-to-location identity edge is unambiguous."""
    locations = _locations()
    locations["is_primary"] = False

    with pytest.raises(ValueError, match="exactly one primary"):
        assign_location_ids(locations)
