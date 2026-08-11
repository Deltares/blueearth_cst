"""Deterministic identities for parent basins, subbasins, and locations."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

INT32_MAX = 2_147_483_647
MAX_LOCAL_SUBBASIN_NUMBER = 99

#: Additional (non-primary) locations allowed inside one subbasin (ADR 0003
#: §12). The id reserves ONE digit for the within-subbasin member, so `m` runs
#: 0–9 with 0 reserved for the primary. A tenth additional location would carry
#: `m = 10` and land exactly on the NEXT subbasin's primary, so this is a
#: collision boundary rather than a preference.
#:
#: If a real deployment ever needs more, the `basin_id*10000 +
#: local_subbasin_number*100 + m` variant lifts the limit to 99 at five digits
#: (ADR 0003 §12, *Cost, accepted*).
MAX_ADDITIONAL_LOCATIONS = 9


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """Raise when a tabular identity input omits required columns."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_positive_int32(values: pd.Series, label: str) -> None:
    """Validate that identifier values fit Wflow's positive int32 namespace."""
    numeric = pd.to_numeric(values, errors="raise")
    if not ((numeric > 0) & (numeric <= INT32_MAX) & (numeric % 1 == 0)).all():
        raise ValueError(f"{label} values must be positive integers <= {INT32_MAX}")


def assign_basin_ids(basins: pd.DataFrame) -> pd.DataFrame:
    """Assign stable parent-basin IDs by area, then outlet grid position."""
    _require_columns(
        basins,
        ("upstream_area", "outlet_row", "outlet_col"),
        "basins",
    )
    ordered = basins.copy()
    if ordered.empty:
        raise ValueError("basins must contain at least one parent basin")
    if ordered[["upstream_area", "outlet_row", "outlet_col"]].isna().any().any():
        raise ValueError("basin ordering fields cannot contain missing values")
    duplicate_outlets = ordered.duplicated(["outlet_row", "outlet_col"], keep=False)
    if duplicate_outlets.any():
        raise ValueError("parent basins cannot share the same outlet grid cell")
    ordered = ordered.sort_values(
        ["upstream_area", "outlet_row", "outlet_col"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ordered["basin_id"] = pd.Series(range(1, len(ordered) + 1), dtype="int32")
    ordered["basin_code"] = ordered["basin_id"].map(lambda value: f"B{value:03d}")

    if "basin_name" not in ordered:
        ordered["basin_name"] = None
    fallback = ordered["basin_id"].map(lambda value: f"basin_{value:03d}")
    supplied = ordered["basin_name"].astype("string").str.strip()
    ordered["basin_name"] = supplied.where(supplied.notna() & supplied.ne(""), fallback)
    return ordered


def assign_subbasin_ids(subbasins: pd.DataFrame) -> pd.DataFrame:
    """Number subbasins downstream-to-upstream with deterministic branch ties."""
    _require_columns(
        subbasins,
        (
            "basin_id",
            "downstream_steps",
            "upstream_area",
            "outlet_row",
            "outlet_col",
        ),
        "subbasins",
    )
    ordered = subbasins.copy()
    if ordered.empty:
        raise ValueError("subbasins must contain at least one spatial unit")
    ordering = [
        "basin_id",
        "downstream_steps",
        "upstream_area",
        "outlet_row",
        "outlet_col",
    ]
    if ordered[ordering].isna().any().any():
        raise ValueError("subbasin ordering fields cannot contain missing values")
    _validate_positive_int32(ordered["basin_id"], "basin_id")
    duplicate_outlets = ordered.duplicated(
        ["basin_id", "outlet_row", "outlet_col"], keep=False
    )
    if duplicate_outlets.any():
        raise ValueError(
            "subbasins in one parent basin cannot share an outlet grid cell"
        )
    ordered = ordered.sort_values(
        ordering,
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ordered["local_subbasin_number"] = (
        ordered.groupby("basin_id", sort=False).cumcount() + 1
    )
    if ordered["local_subbasin_number"].max() > MAX_LOCAL_SUBBASIN_NUMBER:
        raise ValueError(
            f"a parent basin cannot contain more than {MAX_LOCAL_SUBBASIN_NUMBER} "
            "subbasins under the Bnnn-Snn identity scheme"
        )
    ordered["subbasin_id"] = (
        ordered["basin_id"].astype("int64") * 100 + ordered["local_subbasin_number"]
    )
    _validate_positive_int32(ordered["subbasin_id"], "subbasin_id")
    ordered["subbasin_id"] = ordered["subbasin_id"].astype("int32")
    ordered["subbasin_code"] = ordered.apply(
        lambda row: (
            f"B{int(row['basin_id']):03d}-S{int(row['local_subbasin_number']):02d}"
        ),
        axis=1,
    )

    if "subbasin_name" not in ordered:
        ordered["subbasin_name"] = None
    supplied = ordered["subbasin_name"].astype("string").str.strip()
    fallback = ordered["local_subbasin_number"].map(lambda value: f"auto_{value:02d}")
    ordered["subbasin_name"] = supplied.where(
        supplied.notna() & supplied.ne(""), fallback
    )
    return ordered


def assign_location_ids(locations: pd.DataFrame) -> pd.DataFrame:
    """Assign location codes and Wflow-compatible IDs within each subbasin."""
    _require_columns(
        locations,
        (
            "basin_id",
            "basin_code",
            "basin_name",
            "subbasin_id",
            "subbasin_code",
            "subbasin_name",
            # ADR 0003 §12: the wflow_id is built from the LOCAL subbasin
            # number, not from `subbasin_id`. Carried explicitly rather than
            # recovered as `subbasin_id - basin_id*100`, so the id scheme does
            # not silently depend on how subbasin_id happens to be composed.
            "local_subbasin_number",
            "station_name",
            "location_role",
            "is_primary",
            "original_x",
            "original_y",
            "snapped_x",
            "snapped_y",
            "snapped_row",
            "snapped_col",
        ),
        "locations",
    )
    ordered = locations.copy()
    if ordered.empty:
        raise ValueError("locations must contain at least one point")
    _validate_positive_int32(ordered["subbasin_id"], "subbasin_id")
    ordered["is_primary"] = ordered["is_primary"].astype(bool)
    primary_counts = ordered.groupby("subbasin_id")["is_primary"].sum()
    invalid = primary_counts[primary_counts != 1]
    if not invalid.empty:
        raise ValueError(
            "each subbasin must have exactly one primary location; invalid "
            f"subbasin IDs: {invalid.index.astype(int).tolist()}"
        )

    ordered = ordered.sort_values(
        [
            "basin_id",
            "subbasin_id",
            "is_primary",
            "station_name",
            "snapped_row",
            "snapped_col",
            "original_x",
            "original_y",
        ],
        ascending=[True, True, False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ordered["local_location_number"] = (
        ordered.groupby("subbasin_id", sort=False).cumcount() + 1
    )
    ordered["location_id"] = ordered["local_location_number"].astype("int32")
    ordered["location_code"] = ordered.apply(
        lambda row: f"{row['subbasin_code']}-L{int(row['local_location_number']):02d}",
        axis=1,
    )
    # ADR 0003 §12: `wflow_id = basin_id*1000 + local_subbasin_number*10 + m`,
    # `m = 0` for the subbasin's primary location and 1–9 for additional points
    # inside it. Basin 1 reads 1010, 1011, 1020, 1030…; basin 2 reads 2010, …
    #
    # This REPLACES two unrelated formulas that shared one column: a primary got
    # `subbasin_id` (three digits) while any additional point got
    # `1_000_000 + subbasin_id*100 + n` (seven), so points a user thinks of as
    # siblings sat four orders of magnitude apart. The new id groups by basin,
    # orders by subbasin, and keeps the subbasin legible in the flat integer.
    #
    # `m` is `local_location_number - 1`, and the sort above places the primary
    # first within each subbasin, so the primary always lands on 0. Asserted
    # rather than assumed — it is the property every consumer decodes by.
    member = ordered["local_location_number"].astype("int64") - 1
    if not ordered.loc[ordered["is_primary"], "local_location_number"].eq(1).all():
        raise ValueError("each subbasin's primary location must sort first")
    too_many = member > MAX_ADDITIONAL_LOCATIONS
    if too_many.any():
        crowded = sorted(set(ordered.loc[too_many, "subbasin_code"].astype(str)))
        raise ValueError(
            f"a subbasin cannot hold more than {MAX_ADDITIONAL_LOCATIONS} "
            f"additional locations beside its primary under the ADR 0003 §12 "
            f"wflow_id scheme; crowded subbasins: {crowded}"
        )
    ordered["wflow_id"] = (
        ordered["basin_id"].astype("int64") * 1000
        + ordered["local_subbasin_number"].astype("int64") * 10
        + member
    )
    _validate_positive_int32(ordered["wflow_id"], "wflow_id")

    if "provided_wflow_id" in ordered:
        raw = ordered["provided_wflow_id"]
        supplied = raw.notna() & raw.astype("string").str.strip().ne("")
        provided = pd.to_numeric(raw, errors="coerce")
        invalid = supplied & provided.isna()
        if invalid.any():
            raise ValueError("supplied wflow_id values must be integers when populated")
        mismatch = supplied & (provided != ordered["wflow_id"])
        if mismatch.any():
            rows = ordered.loc[
                mismatch, ["station_name", "provided_wflow_id", "wflow_id"]
            ].to_dict("records")
            raise ValueError(
                "supplied wflow_id values disagree with the resolved location "
                f"registry: {rows}"
            )
    ordered["wflow_id"] = ordered["wflow_id"].astype("int32")
    if not ordered["wflow_id"].is_unique:
        raise ValueError("resolved wflow_id values are not globally unique")

    columns = [
        "basin_id",
        "basin_code",
        "basin_name",
        "subbasin_id",
        "subbasin_code",
        "subbasin_name",
        "location_id",
        "location_code",
        "station_name",
        "wflow_id",
        "location_role",
        "original_x",
        "original_y",
        "snapped_x",
        "snapped_y",
        "snapped_row",
        "snapped_col",
        "is_primary",
    ]
    return ordered.loc[:, columns]
