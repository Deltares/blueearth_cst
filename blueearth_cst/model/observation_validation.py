"""Validate observed-discharge station IDs against the spatial registry."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd


def validate_observation_station_ids(
    observations_path: str | Path,
    registry_path: str | Path,
) -> pd.DataFrame:
    """Return the registry after validating the observation CSV header.

    User-provided control and observation locations must have one series each.
    Synthetic automatic outlets may have a series but are not required to.
    """
    observations_path = Path(observations_path)
    registry_path = Path(registry_path)
    with observations_path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=";"), None)
    if not header:
        raise ValueError(f"observation timeseries is empty: {observations_path}")
    if len(header) == 1 and "," in header[0]:
        raise ValueError(
            "observation timeseries must be semicolon-separated with header "
            "time;<wflow_id>;..."
        )
    if header[0].strip().lower() != "time":
        raise ValueError("observation timeseries first column must be named time")

    labels = [label.strip() for label in header[1:]]
    if not labels or any(not label for label in labels):
        raise ValueError(
            "observation timeseries must contain non-empty station columns"
        )
    invalid = sorted({label for label in labels if not label.isdecimal()})
    if invalid:
        raise ValueError(
            f"observation station columns must be integer wflow_id values: {invalid}"
        )
    ids = [int(label) for label in labels]
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)

    registry = pd.read_csv(registry_path)
    required = {"wflow_id", "location_role"}
    missing_columns = sorted(required.difference(registry.columns))
    if missing_columns:
        raise ValueError(f"location_registry lacks columns: {missing_columns}")
    if registry["wflow_id"].duplicated().any():
        duplicate_registry = sorted(
            registry.loc[registry["wflow_id"].duplicated(False), "wflow_id"]
            .astype(int)
            .unique()
        )
        raise ValueError(
            f"location_registry contains duplicate wflow_id values: {duplicate_registry}"
        )

    registry_ids = set(registry["wflow_id"].astype(int))
    expected_ids = set(
        registry.loc[
            registry["location_role"].astype(str).ne("automatic_outlet"), "wflow_id"
        ].astype(int)
    )
    observed_ids = set(ids)
    missing = sorted(expected_ids.difference(observed_ids))
    unexpected = sorted(observed_ids.difference(registry_ids))
    problems = []
    if missing:
        problems.append(f"missing={missing}")
    if duplicates:
        problems.append(f"duplicate={duplicates}")
    if unexpected:
        problems.append(f"unexpected={unexpected}")
    if problems:
        raise ValueError(
            "observation station IDs do not match location_registry: "
            + "; ".join(problems)
        )
    return registry
