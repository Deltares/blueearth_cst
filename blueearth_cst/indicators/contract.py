"""Neutral calculation and result contracts, independent of workflow storage."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from blueearth_cst.indicators.series import ResponseSeries


@dataclass(frozen=True)
class ResponseRequirement:
    """Physical and temporal interpretation required before a reduction."""

    variable: str
    units: str
    timestep_seconds: int = 86400
    time_label: str = "interval_end"
    coverage: str = "entire retained response; preserve partial water years"
    location_grain: str = "location"
    missingness: str = "native NaN; preserve pandas skipna and rolling completeness"
    calendars: tuple[str, ...] = ("standard", "gregorian", "proleptic_gregorian")


@dataclass(frozen=True)
class IndicatorDeclaration:
    """Resolved scientific meaning and optional set-local callable identity."""

    name: str
    statistic: str
    grain: str
    required_responses: ResponseRequirement
    bundle_by: str | None
    reference: str | None
    reference_location: str | None
    return_period: int | None
    blocks_per_return_period: float | None
    implementation_revision: str
    value_validity: str
    group_reduction: str
    calculation_set: str | None = None
    calculation_method: str | None = None
    calculation_parameters: dict | None = None
    additional_responses: tuple[ResponseRequirement, ...] = ()
    output_variable: str | None = None

    @property
    def requirements(self) -> tuple[ResponseRequirement, ...]:
        return (self.required_responses, *self.additional_responses)

    @property
    def table_variable(self) -> str:
        return self.output_variable or self.required_responses.variable


@dataclass(frozen=True)
class EvaluationContext:
    """Explicit temporal, reference and membership inputs, never policy switches."""

    anchor: str
    reference: Any = None
    expected_run_ids: tuple[str, ...] = ()
    validation: Mapping | None = None
    prepared: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class DiagnosticTable:
    """Named table with string columns and scalar cells only."""

    name: str
    data: pd.DataFrame


@dataclass(frozen=True)
class IndicatorResult:
    """In-memory location values, statuses and method-owned diagnostics."""

    values: pd.Series
    diagnostics: tuple[DiagnosticTable, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    statuses: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Indicator:
    """Set-owned registration; a new callable needs no global statistic entry."""

    id: str
    calculate: Callable
    parameters: Mapping
    input: ResponseRequirement
    units: str
    description: str
    statistic: str = "local"
    grain: str = "run"
    group_reduction: str = "equal-weight-run-reductions-propagate-nan/1"
    reference: str | None = None
    reference_location: str | None = None
    return_period: int | None = None
    blocks_per_return_period: float | None = None
    method: str | None = None
    additional_inputs: tuple[ResponseRequirement, ...] = ()
    output_variable: str | None = None


Series = ResponseSeries
