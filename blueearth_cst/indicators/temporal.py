"""Neutral temporal semantics and explicit month references."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from blueearth_cst.indicators.contract import IndicatorDeclaration
from blueearth_cst.indicators.series import (
    ResponseSeries,
    compatible_bundle,
    validate_responses,
)


def block_bins(anchor: str) -> str:
    """How a water-year anchor closes and labels its resample blocks."""
    if anchor.startswith("YS"):
        return "left-closed, left-labelled"
    return "right-closed, right-labelled"


def _block_interval(label: pd.Timestamp, anchor: str):
    """The calendar interval one resample block covers, for its label."""
    offset = pd.tseries.frequencies.to_offset(anchor)
    if anchor.startswith("YS"):
        return label, label + offset
    return label - offset, label


def series_coverage(item: ResponseSeries, anchor: str) -> dict:
    """Immutable per-series block coverage under the resolved water-year anchor.

    Derived from the retained index and missing mask, never from requested
    dates. A complete label block has every expected daily label of its
    Gregorian anchor interval (leap days included); it may still hold NaNs.
    """
    labels = pd.Series(item.missing, index=item.time)
    blocks = []
    for block_label, group in labels.resample(anchor):
        if group.empty:
            continue
        start, end = _block_interval(block_label, anchor)
        expected = int((end - start) / item.timestep)
        observed = len(group)
        missing = int(group.sum())
        blocks.append(
            {
                "block_label": str(block_label.date()),
                "first_label": str(group.index[0].date()),
                "last_label": str(group.index[-1].date()),
                "observed": observed,
                "expected": expected,
                "missing": missing,
                "all_missing": missing == observed,
            }
        )
    return {
        "run_id": item.run_id,
        "variable": item.variable,
        "location_id": item.location_id,
        "calendar": item.calendar,
        "timestep_seconds": int(item.timestep.total_seconds()),
        "time_label": item.time_label,
        "anchor": anchor,
        "block_bins": block_bins(anchor),
        "first_label": str(item.time[0].date()),
        "last_label": str(item.time[-1].date()),
        "blocks": blocks,
        "block_count": len(blocks),
        "complete_label_blocks": sum(b["observed"] == b["expected"] for b in blocks),
        "missing_total": sum(b["missing"] for b in blocks),
        "all_missing_blocks": sum(b["all_missing"] for b in blocks),
    }


def validate_requirement(
    indicator: IndicatorDeclaration, series: Sequence[ResponseSeries]
) -> None:
    """Refuse metadata mismatches before an indicator performs any reduction."""
    validate_responses(series, evaluated_runs={item.run_id for item in series})
    requirements = {item.variable: item for item in indicator.requirements}
    if len(requirements) != len(indicator.requirements):
        raise ValueError(f"{indicator.name}: duplicate response requirements")
    by_run = {}
    for item in series:
        requirement = requirements.get(item.variable)
        if requirement is None or (
            item.variable,
            item.units,
            item.timestep.total_seconds(),
            item.time_label,
        ) != (
            requirement.variable,
            requirement.units,
            requirement.timestep_seconds,
            requirement.time_label,
        ):
            raise ValueError(
                f"{indicator.name}: incompatible response requirement at {item.key}"
            )
        by_run.setdefault(item.run_id, {}).setdefault(item.variable, {})[
            item.location_id
        ] = item
    for run, variables in by_run.items():
        if set(variables) != set(requirements):
            raise ValueError(f"{indicator.name}: missing paired variables in {run}")
        first = variables[indicator.required_responses.variable]
        for variable in variables.values():
            if set(variable) != set(first):
                raise ValueError(f"{indicator.name}: paired location sets differ")
            for location, item in variable.items():
                primary = first[location]
                if (
                    item.calendar != primary.calendar
                    or item.timestep != primary.timestep
                    or item.time_label != primary.time_label
                    or not item.time.equals(primary.time)
                    or item.location_ordinal != primary.location_ordinal
                ):
                    raise ValueError(
                        f"{indicator.name}: paired response support differs"
                    )


@dataclass(frozen=True)
class MonthReference:
    """Resolved Class-C reference; selection is never repeated per response."""

    run_ids: tuple[str, ...]
    reference_location_id: str
    month: int
    which: str
    counts: tuple[tuple[str, str, str, int, int], ...]
    selection_statistic: str = "monthly sum followed by argmax/argmin"
    tie_rule: str = "first label in ascending calendar-month order"


def resolve_month_reference(
    series: Sequence[ResponseSeries],
    *,
    expected_run_ids: Sequence[str],
    which: str,
) -> MonthReference:
    """Resolve the legacy first-native-gauge rule into an explicit location id."""
    if which not in {"wet", "dry"} or not expected_run_ids:
        raise ValueError(
            "Class-C reference requires nonempty explicit members and wet/dry selection"
        )
    if len(set(expected_run_ids)) != len(expected_run_ids):
        raise ValueError("Class-C reference has duplicate members")
    validate_responses(series, evaluated_runs=set(expected_run_ids))
    if any(
        (item.variable, item.units, item.time_label, item.timestep.total_seconds())
        != ("q", "m3 s-1", "interval_end", 86400)
        for item in series
    ):
        raise ValueError(
            "Class-C reference requires daily interval-end discharge responses"
        )
    if {item.run_id for item in series} != set(expected_run_ids):
        raise ValueError("Class-C reference membership is incomplete")
    locations = [
        {item.location_id for item in series if item.run_id == run}
        for run in expected_run_ids
    ]
    if any(observed != locations[0] for observed in locations):
        raise ValueError("Class-C reference location sets differ")
    for location in locations[0]:
        compatible_bundle([item for item in series if item.location_id == location])
    first = [
        item for item in series if item.variable == "q" and item.location_ordinal == 0
    ]
    if (
        len(first) != len(expected_run_ids)
        or len({item.location_id for item in first}) != 1
    ):
        raise ValueError(
            "Class-C reference needs the same first native q location in every member"
        )
    compatible_bundle(first)
    location = first[0].location_id
    ordered = sorted(first, key=lambda item: item.run_id)
    pooled = pd.concat([pd.Series(item.values, index=item.time) for item in ordered])
    monthly = pooled.groupby(pooled.index.month).sum()
    month = int(monthly.idxmax() if which == "wet" else monthly.idxmin())
    counts = tuple(
        (
            item.run_id,
            str(item.time[0]),
            str(item.time[-1]),
            int(item.time[~item.missing].year.nunique()),
            int(item.time[~item.missing].to_period("M").nunique()),
        )
        for item in ordered
    )
    return MonthReference(
        tuple(item.run_id for item in ordered), location, month, which, counts
    )
