"""Independent calendar-loop arithmetic for crop-water calculations."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


def _one(indicator, responses) -> pd.Series:
    by_location = defaultdict(dict)
    for item in responses:
        by_location[item.location_id][item.variable] = dict(zip(item.time, item.values))
    params = indicator.calculation_parameters
    method = indicator.calculation_method
    answer = {}
    for location, variables in by_location.items():
        dates = sorted(next(iter(variables.values())))
        year_values = []
        for year in range(dates[0].year - 1, dates[-1].year + 2):
            if method == "annual_recharge/1":
                start, end = pd.Timestamp(year - 1, 10, 1), pd.Timestamp(year, 9, 30)
            elif params["season"] == "winter":
                start, end = pd.Timestamp(year - 1, 11, 1), pd.Timestamp(year, 5, 31)
            else:
                start, end = pd.Timestamp(year, 5, 1), pd.Timestamp(year, 10, 31)
            calendar = pd.date_range(start, end, freq="D")
            if not all(
                date in data and math.isfinite(float(data[date]))
                for data in variables.values()
                for date in calendar
            ):
                continue
            data = {
                name: [float(mapping[date]) for date in calendar]
                for name, mapping in variables.items()
            }
            if method == "seasonal_climatic_deficit/1":
                value = sum(
                    max(pet - p, 0) for p, pet in zip(data["precip"], data["pet"])
                )
            elif method == "seasonal_longest_dry_spell/1":
                lengths, current = [], 0
                for p in data["precip"]:
                    current = current + 1 if p < params["dry_day_mm"] else 0
                    lengths.append(current)
                value = max(lengths)
            elif method == "seasonal_low_flow_days/1":
                threshold = params["thresholds"][location]["value"]
                value = sum(q < threshold for q in data["q"])
            elif method == "annual_recharge/1":
                value = sum(data["gwr"])
            elif method == "seasonal_relative_et_deficit/1":
                pet = sum(data["pet"])
                if pet <= 0:
                    continue
                value = 100 * (1 - sum(data["aet"]) / pet)
            elif method == "seasonal_et_limitation_days/1":
                eligible = [
                    (aet, pet)
                    for aet, pet in zip(data["aet"], data["pet"])
                    if pet >= params["pet_min_mm"]
                ]
                if not eligible:
                    continue
                value = sum(aet / pet < params["ratio"] for aet, pet in eligible)
            else:
                raise ValueError(f"unknown crop-water method {method}")
            year_values.append(value)
        answer[location] = float(np.mean(year_values)) if year_values else np.nan
    return pd.Series(answer)


def expected(indicator, responses, context, *, group=False) -> pd.Series:
    if not group:
        return _one(indicator, responses)
    members = [
        _one(indicator, [item for item in responses if item.run_id == run])
        for run in context.expected_run_ids
    ]
    if not members or any(
        set(member.index) != set(members[0].index) for member in members
    ):
        raise ValueError("crop-water check requires matching member locations")
    return pd.Series(
        {
            location: float(np.mean([member[location] for member in members]))
            for location in members[0].index
        }
    )


def check_result(indicator, responses, result, context, *, group=False) -> None:
    if group:
        return
    tables = {table.name: table.data for table in result.diagnostics}
    if set(tables) != {"period_values", "period_summary"}:
        raise ValueError("crop-water period diagnostics are incomplete")
    periods = tables["period_values"]
    for location in result.values.index:
        rows = periods[periods.location_id == location]
        usable = rows.loc[rows.status == "valid", "value"]
        if not np.isclose(
            result.values[location],
            usable.mean() if not usable.empty else np.nan,
            equal_nan=True,
        ):
            raise ValueError("crop-water scalar differs from valid period values")
