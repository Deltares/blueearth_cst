"""Indicator declarations and reductions over neutral response series only."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from blueearth_cst.indicators import loader
from blueearth_cst.indicators.contract import IndicatorDeclaration, ResponseRequirement
from blueearth_cst.indicators.extremes import (
    EXTRACTION_POLICY as EXTRACTION_POLICY,
)
from blueearth_cst.indicators.extremes import (
    MISSINGNESS_POLICY as MISSINGNESS_POLICY,
)
from blueearth_cst.indicators.extremes import (
    PARTIAL_BLOCK_POLICY as PARTIAL_BLOCK_POLICY,
)
from blueearth_cst.indicators.extremes import (
    InsufficientReturnLevelBlocks as InsufficientReturnLevelBlocks,
)
from blueearth_cst.indicators.extremes import (
    InvalidReturnLevelFit as InvalidReturnLevelFit,
)
from blueearth_cst.indicators.extremes import (
    ReturnLevelEvidence as ReturnLevelEvidence,
)
from blueearth_cst.indicators.extremes import (
    ReturnLevelProbabilityNotCovered as ReturnLevelProbabilityNotCovered,
)
from blueearth_cst.indicators.extremes import (
    _shape_coverage as _shape_coverage,
)
from blueearth_cst.indicators.temporal import (
    MonthReference as MonthReference,
)
from blueearth_cst.indicators.temporal import (
    block_bins as block_bins,
)
from blueearth_cst.indicators.temporal import (
    resolve_month_reference as resolve_month_reference,
)
from blueearth_cst.indicators.temporal import (
    series_coverage as series_coverage,
)
from blueearth_cst.indicators.temporal import (
    validate_requirement as validate_requirement,
)
from blueearth_cst.shared import indicator_tables as tables
from blueearth_cst.shared import indicators_definition as md

GEV_SCREENING_BLOCK_FLOOR = 10
SCREENING_POLICY_STATUS = "provisional_operational"

# Reviewed run-reduction statistics that have equal-weight realization means.
# A newly declared statistic needs a deliberate group policy before publication.
GROUP_MEAN_STATISTICS = {
    "mean",
    "sum",
    "max",
    "min",
    "q95",
    "Q7day_max",
    "Q7day_min",
    "BaseFlowIndex",
    "wetmonth_mean",
    "drymonth_mean",
    "Q30day_max",
    "Q30day_min",
    "Q90day_min",
    "rise_rate",
    "fall_rate",
}


@dataclass(frozen=True)
class DeclaredBundle:
    """Provider-resolved membership with an explicit canonical grouping key."""

    bundle_by: str
    canonical_key: str
    member_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class UnitIndexRow:
    """One logical row of the future three-column indicator-unit index."""

    unit_id: str
    grain: str
    member_run_id: str


def indicator_unit_index(
    all_run_ids: Sequence[str],
    evaluated_run_ids: Sequence[str],
    bundles: Sequence[DeclaredBundle],
    *,
    unit_id_capacity: int,
) -> tuple[UnitIndexRow, ...]:
    """Plan logical run/bundle membership without persisting a P2 artifact."""
    from blueearth_cst.experiment.scenario_rows import UnitNamespaceCapacityError

    if type(unit_id_capacity) is not int or unit_id_capacity < 1:
        raise ValueError("unit_id_capacity must be a positive integer")
    width = len(str(unit_id_capacity))
    if tuple(all_run_ids) != tuple(
        f"run_{number:0{width}d}" for number in range(1, len(all_run_ids) + 1)
    ):
        raise ValueError(
            "run ids must preserve the complete collection's order and width"
        )
    if (
        not evaluated_run_ids
        or len(set(evaluated_run_ids)) != len(evaluated_run_ids)
        or not set(evaluated_run_ids) <= set(all_run_ids)
    ):
        raise ValueError("evaluated run ids must be unique collection members")
    keys = [(bundle.bundle_by, bundle.canonical_key) for bundle in bundles]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate declared grouping key")
    required = max(len(all_run_ids), len(bundles))
    if required > unit_id_capacity:
        raise UnitNamespaceCapacityError(
            f"capacity={unit_id_capacity}, runs={len(all_run_ids)}, bundles={len(bundles)}, "
            f"required={required}, minimum replacement={required}"
        )
    result = [UnitIndexRow(run, "run", run) for run in sorted(evaluated_run_ids)]
    for number, bundle in enumerate(
        sorted(bundles, key=lambda item: (item.bundle_by, item.canonical_key)),
        start=1,
    ):
        members = bundle.member_run_ids
        if (
            not bundle.bundle_by
            or not members
            or len(set(members)) != len(members)
            or not set(members) <= set(evaluated_run_ids)
        ):
            raise ValueError("bundle must name unique evaluated members and a grouping")
        result.extend(
            UnitIndexRow(f"group_{number:0{width}d}", "bundle", run)
            for run in sorted(members)
        )
    return tuple(result)


def _revision(set_id: str = "default") -> str:
    """Hash the legacy adapter and contributing calculation set only."""
    digest = hashlib.sha256()
    for module in (
        Path(__file__),
        Path(md.__file__),
        Path(tables.__file__),
        *loader.execution_sources([set_id]),
    ):
        digest.update(module.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _run_declaration(name: str, statistic: str, revision: str) -> IndicatorDeclaration:
    return IndicatorDeclaration(
        name=name,
        statistic=statistic,
        grain="run",
        required_responses=ResponseRequirement("q", "m3 s-1"),
        bundle_by=None,
        reference=None,
        reference_location=None,
        return_period=None,
        blocks_per_return_period=None,
        implementation_revision=revision,
        value_validity="finite or native-reduction NaN; refuse infinity; no clipping",
        group_reduction="equal-weight-run-reductions-propagate-nan/1",
    )


def declarations(tokens: Sequence[str]) -> tuple[IndicatorDeclaration, ...]:
    """Declare selected indicators using the existing authoritative vocabulary."""
    revision = _revision()
    result = []
    for token in tokens:
        if token == "q":
            entries = tables.Q_INDICATOR_SUFFIXES.items()
            units = "m3 s-1"
        else:
            suffix, reduction = tables.BASIN_INDICATOR_SUFFIXES[token]
            entries = [(reduction, (suffix, "A"))]
            units = (
                "m3 s-1"
                if token == "overland_flow"
                else (
                    "degC"
                    if token == "temp"
                    else "mm"
                    if token == "snow"
                    else "mm dt-1"
                )
            )
        for statistic, (suffix, kind) in entries:
            period = None
            if kind == "B":
                period = (
                    tables.RETURN_PERIOD_PEAK_YR
                    if statistic == "return_level_max"
                    else tables.RETURN_PERIOD_LOW_YR
                )
            result.append(
                IndicatorDeclaration(
                    name=f"{token}_{suffix}",
                    statistic=statistic,
                    grain="bundle" if kind == "B" else "run",
                    required_responses=ResponseRequirement(token, units),
                    bundle_by="same_design_point" if kind == "B" else None,
                    reference="unperturbed_runs" if kind == "C" else None,
                    reference_location="class_c_first_gauge" if kind == "C" else None,
                    return_period=period,
                    blocks_per_return_period=1.0 if kind == "B" else None,
                    implementation_revision=revision,
                    value_validity=(
                        "finite required; reject invalid GEV fit; no clipping"
                        if kind == "B"
                        else "finite or native-reduction NaN; refuse infinity; no clipping"
                    ),
                    group_reduction=(
                        "pooled-member-local-blocks/1"
                        if kind == "B"
                        else "equal-weight-run-reductions-propagate-nan/1"
                    ),
                )
            )
    return tuple(result)


def optional_declarations() -> tuple[IndicatorDeclaration, ...]:
    """Declare the registered q indicators outside the ``current_default`` preset."""
    revision = _revision("flow_regime")
    return tuple(
        _run_declaration(name, statistic, revision)
        for statistic, (name, _, _) in OPTIONAL_Q_CALCULATIONS.items()
    )


#: Shared temporal semantics of every initial method (design v3, P2).
_TEMPORAL = (
    "daily interval_end labels binned directly without shifting to interval "
    "starts; the first label is starttime+step"
)
_PARTIAL = "partial edge blocks retained; each block weighted equally; no duration normalization"


@dataclass(frozen=True)
class MethodContract:
    """One registered calculation method: closed parameters and declared policies."""

    method_id: str
    parameters: dict[str, dict[str, tuple[str | int, ...]]]
    block_assignment: str
    partial_blocks: str
    missingness: str
    validity: str


#: Registered method vocabulary. ``parameters`` maps each response variable to
#: the closed choices of every required parameter; wrappers around existing
#: reductions only, so the whitelists span exactly the registered definitions.
METHODS = {
    "annual_summary/1": MethodContract(
        "annual_summary/1",
        {
            "q": {"statistic": ("mean", "max", "min", "p95")},
            "aet": {"statistic": ("sum",)},
            "pet": {"statistic": ("sum",)},
            "gwr": {"statistic": ("sum",)},
            "precip": {"statistic": ("sum",)},
            "snow": {"statistic": ("max",)},
            "overland_flow": {"statistic": ("mean",)},
            "temp": {"statistic": ("mean",)},
        },
        f"water-year blocks; {_TEMPORAL}",
        _PARTIAL,
        "sum skips NaN with min_count=0 (an all-missing block sums to 0); mean/max/min "
        "and quantile skip NaN and keep NaN for an all-missing block; the mean over "
        "blocks skips NaN blocks",
        "finite or native-reduction NaN; infinity refused; no clipping",
    ),
    "annual_rolling7_extreme/1": MethodContract(
        "annual_rolling7_extreme/1",
        {"q": {"extreme": ("min", "max")}},
        f"rolling 7-day mean over the whole member, then water-year blocks; {_TEMPORAL}; "
        "windows cross water-year boundaries but never member boundaries",
        _PARTIAL,
        "a rolling window needs seven nonmissing values; block extrema and the mean "
        "over blocks skip NaN",
        "finite or native-reduction NaN; infinity refused; no clipping",
    ),
    "annual_min7_mean_ratio/1": MethodContract(
        "annual_min7_mean_ratio/1",
        {"q": {}},
        f"annual minimum of the rolling 7-day mean over the annual mean; {_TEMPORAL}",
        _PARTIAL,
        "rolling windows need seven nonmissing values; 0/0 gives NaN for that block; "
        "the mean over blocks skips NaN",
        "infinite final value refused (a zero annual mean with a positive minimum); "
        "no clipping",
    ),
    "reference_month_mean/1": MethodContract(
        "reference_month_mean/1",
        {"q": {"which": ("wet", "dry")}},
        "fixed calendar month from pooled monthly SUMS of unperturbed evaluated members "
        "at the first native q location (not the highest/lowest daily mean rate; the "
        "choice depends on month length, member coverage and missingness; first month "
        f"on ties); that month's water-year means averaged; {_TEMPORAL}",
        _PARTIAL,
        "monthly selection sums skip NaN; block means and the mean over blocks skip NaN",
        "finite or native-reduction NaN; infinity refused; no clipping",
    ),
    "pooled_return_level/1": MethodContract(
        "pooled_return_level/1",
        {"q": {"kind": ("annual_max", "rolling7_annual_min")}},
        "member-local annual maxima or rolling 7-day annual minima pooled per design "
        f"group; L-moment GEV; fixed 10/2-year periods; {_TEMPORAL}",
        "partial member-local blocks retained; pooling weights each usable block equally",
        "nonfinite blocks dropped; at least max(ceil(T), 10) usable blocks",
        "finite required; invalid GEV fit and untested probabilities refused; no "
        "clipping; screening is provisional operational policy",
    ),
    "annual_rolling_extreme/1": MethodContract(
        "annual_rolling_extreme/1",
        {"q": {"window": (30, 90), "extreme": ("min", "max")}},
        "rolling N-day mean (N = window) over the whole member, then water-year blocks; "
        f"{_TEMPORAL}; windows cross water-year boundaries but never member boundaries",
        _PARTIAL,
        "a rolling window needs N nonmissing values; block extrema and the "
        "mean over blocks skip NaN",
        "finite or native-reduction NaN; infinity refused; no clipping",
    ),
    "annual_daily_change/1": MethodContract(
        "annual_daily_change/1",
        {"q": {"direction": ("rise", "fall")}},
        "day-to-day differences over the whole member, each in its later day's "
        f"water-year block; {_TEMPORAL}; differences cross water-year boundaries "
        "but never member boundaries",
        _PARTIAL,
        "a difference needs both days nonmissing; zero changes are neither rises "
        "nor falls; a block without one is NaN; the mean over blocks skips NaN",
        "finite or native-reduction NaN; infinity refused; no clipping",
    ),
}

#: Registered definition id -> (method id, parameters, registry statistic).
#: Built from the existing vocabulary so ids and suffixes keep one owner.
_Q_CALCULATIONS = {
    "mean": ("annual_summary/1", {"variable": "q", "statistic": "mean"}),
    "max": ("annual_summary/1", {"variable": "q", "statistic": "max"}),
    "min": ("annual_summary/1", {"variable": "q", "statistic": "min"}),
    "q95": ("annual_summary/1", {"variable": "q", "statistic": "p95"}),
    "Q7day_max": ("annual_rolling7_extreme/1", {"extreme": "max"}),
    "Q7day_min": ("annual_rolling7_extreme/1", {"extreme": "min"}),
    "BaseFlowIndex": ("annual_min7_mean_ratio/1", {}),
    "wetmonth_mean": ("reference_month_mean/1", {"which": "wet"}),
    "drymonth_mean": ("reference_month_mean/1", {"which": "dry"}),
    "return_level_max": ("pooled_return_level/1", {"kind": "annual_max"}),
    "return_level_7day_min": ("pooled_return_level/1", {"kind": "rolling7_annual_min"}),
}
#: Registered q calculations OUTSIDE the ``current_default`` preset: a set file
#: selects them, but ``declarations()`` never expands them, so ``default`` keeps
#: its membership. IHA-style (eflowrisk amax30d, amin30d, amin90d,
#: mean_rate_rise, mean_rate_fall), averaged over water years rather than the
#: eflowrisk median. statistic -> (definition id, method, parameters).
OPTIONAL_Q_CALCULATIONS = {
    "Q30day_max": (
        "q_mean_annual_30day_max",
        "annual_rolling_extreme/1",
        {"window": 30, "extreme": "max"},
    ),
    "Q30day_min": (
        "q_mean_annual_30day_min",
        "annual_rolling_extreme/1",
        {"window": 30, "extreme": "min"},
    ),
    "Q90day_min": (
        "q_mean_annual_90day_min",
        "annual_rolling_extreme/1",
        {"window": 90, "extreme": "min"},
    ),
    "rise_rate": ("q_mean_rise_rate", "annual_daily_change/1", {"direction": "rise"}),
    "fall_rate": ("q_mean_fall_rate", "annual_daily_change/1", {"direction": "fall"}),
}
DEFINITIONS = (
    {
        tables.q_indicator_name(statistic): (method, parameters, statistic)
        for statistic, (method, parameters) in _Q_CALCULATIONS.items()
    }
    | {
        tables.basin_indicator_name(token): (
            "annual_summary/1",
            {"variable": token, "statistic": reduction},
            reduction,
        )
        for token, (_, reduction) in tables.BASIN_INDICATOR_SUFFIXES.items()
    }
    | {
        name: (method, parameters, statistic)
        for statistic, (name, method, parameters) in OPTIONAL_Q_CALCULATIONS.items()
    }
)

#: Readable registry title, description and citations per definition. Units and
#: policies come from declarations and METHODS, never from this text.
DEFINITION_TEXT = {
    "q_annual_mean": (
        "Mean annual discharge",
        "Water-year mean discharge averaged over years.",
    ),
    "q_mean_annual_max": (
        "Mean annual maximum discharge",
        "Water-year maximum daily discharge averaged over years.",
    ),
    "q_mean_annual_min": (
        "Mean annual minimum discharge",
        "Water-year minimum daily discharge averaged over years.",
    ),
    "q_mean_annual_p95": (
        "Mean annual 95th-percentile discharge",
        "Water-year 0.95 non-exceedance quantile (a HIGH flow, not conventional low-flow Q95) averaged over years.",
    ),
    "q_mean_annual_7day_max": (
        "Mean annual 7-day maximum discharge",
        "Water-year maximum of the rolling 7-day mean averaged over years.",
    ),
    "q_mean_annual_7day_min": (
        "Mean annual 7-day minimum discharge",
        "Water-year minimum of the rolling 7-day mean averaged over years.",
    ),
    "q_baseflow_index": (
        "Baseflow index",
        "Water-year ratio of the 7-day minimum to the annual mean, averaged over years.",
    ),
    "q_wettest_month_mean": (
        "Wettest-month mean discharge",
        "Mean discharge in the reference wet month (largest pooled monthly sum), averaged over years.",
    ),
    "q_driest_month_mean": (
        "Driest-month mean discharge",
        "Mean discharge in the reference dry month (smallest pooled monthly sum), averaged over years.",
    ),
    "q_return_level_10yr_max": (
        "10-year return level of annual maximum discharge",
        "GEV (L-moments) 10-year level of pooled member-local annual maxima.",
    ),
    "q_return_level_2yr_7day_min": (
        "2-year return level of annual 7-day minimum discharge",
        "GEV (L-moments) 2-year level of pooled member-local annual 7-day minima.",
    ),
    "aet_annual_total": (
        "Mean water-year actual evapotranspiration depth",
        "Daily depths summed per water-year block, averaged with equal block weight.",
    ),
    "pet_annual_total": (
        "Mean water-year potential evapotranspiration depth",
        "Daily depths summed per water-year block, averaged with equal block weight.",
    ),
    "gwr_annual_total": (
        "Mean water-year groundwater recharge depth",
        "Daily depths summed per water-year block, averaged with equal block weight.",
    ),
    "precip_annual_total": (
        "Mean water-year precipitation depth",
        "Daily depths summed per water-year block, averaged with equal block weight.",
    ),
    "snow_annual_max": (
        "Mean annual maximum snowpack liquid water depth",
        "Water-year maximum depth averaged over years.",
    ),
    "overland_flow_annual_mean": (
        "Mean annual overland flow",
        "Water-year mean overland flow averaged over years.",
    ),
    "temp_annual_mean": (
        "Mean annual air temperature",
        "Water-year mean air temperature averaged over years; degrees Celsius.",
    ),
    "q_mean_annual_30day_max": (
        "Mean annual 30-day maximum discharge",
        "Water-year maximum of the rolling 30-day mean averaged over years.",
    ),
    "q_mean_annual_30day_min": (
        "Mean annual 30-day minimum discharge",
        "Water-year minimum of the rolling 30-day mean averaged over years.",
    ),
    "q_mean_annual_90day_min": (
        "Mean annual 90-day minimum discharge",
        "Water-year minimum of the rolling 90-day mean averaged over years.",
    ),
    "q_mean_rise_rate": (
        "Mean rate of rise",
        "Water-year mean of positive day-to-day discharge changes, averaged over years.",
    ),
    "q_mean_fall_rate": (
        "Mean rate of fall",
        "Water-year mean magnitude of negative day-to-day discharge changes, averaged over years.",
    ),
}
#: Registry citations per definition. Empty until a source the repository
#: actually retains is cited; never filled from memory.
DEFINITION_REFERENCES: dict[str, tuple[str, ...]] = {}
if set(DEFINITION_TEXT) != set(DEFINITIONS):
    raise ValueError("DEFINITION_TEXT must cover exactly the registered definitions")


def output_units(declaration: IndicatorDeclaration) -> str:
    """Declared output units; never inferred from input units alone."""
    if declaration.calculation_set:
        return loader.registration(declaration).units
    if declaration.statistic == "BaseFlowIndex":
        return "1"
    if declaration.statistic in {"rise_rate", "fall_rate"}:
        return "m3 s-1 d-1"
    if declaration.statistic == "sum":
        # Mean water-year block depth: mm per block, not a duration-normalized rate.
        return "mm"
    return declaration.required_responses.units


def quantity(declaration: IndicatorDeclaration) -> str:
    """The quantity a value measures, stated for the readable dictionary."""
    if declaration.calculation_set:
        return loader.registration(declaration).description
    if declaration.statistic == "sum":
        return (
            "mean water-year block depth: daily mm dt-1 values summed within each "
            "block and averaged with equal block weight; mean annual depth only for "
            "complete nonmissing blocks; not a duration-normalized rate"
        )
    if declaration.statistic == "BaseFlowIndex":
        return "dimensionless ratio"
    if declaration.statistic in {"rise_rate", "fall_rate"}:
        return "mean over water years of the annual mean day-to-day change magnitude"
    if declaration.grain == "bundle":
        return f"{declaration.return_period}-year return level"
    return "mean over water years of an annual statistic"


def validate_calculation(method: str, parameters) -> tuple[str, dict]:
    """Validate an authored calculation; return its registered definition id.

    Parameters are closed: extras, omissions and values outside the method's
    whitelist are refused, with no implicit numerical defaults.
    """
    contract = METHODS.get(method)
    if contract is None:
        raise ValueError(
            f"unknown indicator method {method!r}; known: {sorted(METHODS)}"
        )
    if not isinstance(parameters, dict):
        raise ValueError(f"{method}: parameters must be a mapping")
    normalized = dict(sorted(parameters.items()))
    for definition, (registered, registered_parameters, _) in DEFINITIONS.items():
        if registered == method and normalized == dict(
            sorted(registered_parameters.items())
        ):
            return definition, normalized
    variables = sorted(contract.parameters)
    raise ValueError(
        f"{method}: unsupported parameters {normalized!r}; required keys and "
        f"choices per variable: { ({v: contract.parameters[v] for v in variables}) }"
    )


def declaration_for(definition: str) -> IndicatorDeclaration:
    """The authoritative declaration of one registered definition id."""
    return declaration_catalog([definition])[definition]


def declaration_catalog(definitions: Sequence[str]) -> dict[str, IndicatorDeclaration]:
    """Observe contributing source revisions once for this declaration batch."""
    for definition in definitions:
        if definition not in DEFINITIONS:
            raise ValueError(f"unknown registered indicator definition {definition!r}")
    selected = set(definitions)
    optional = {name for name, _, _ in OPTIONAL_Q_CALCULATIONS.values()}
    tokens = sorted(
        {DEFINITIONS[name][1].get("variable", "q") for name in selected - optional}
    )
    catalog = list(declarations(tokens)) if tokens else []
    if selected & optional:
        catalog.extend(optional_declarations())
    return {item.name: item for item in catalog if item.name in selected}


def _reduce_optional(frame: pd.DataFrame, statistic: str, anchor: str) -> pd.Series:
    """Compatibility entry point for the existing optional formula callers."""
    _, method, parameters = OPTIONAL_Q_CALCULATIONS[statistic]
    if method == "annual_rolling_extreme/1":
        blocks = frame.rolling(parameters["window"]).mean().resample(anchor)
        return getattr(blocks, parameters["extreme"])().mean()
    change = frame.diff()
    change = (
        change.where(change > 0)
        if parameters["direction"] == "rise"
        else (-change).where(change < 0)
    )
    return change.resample(anchor).mean().mean()


def reduce_run(indicator, series, *, anchor, reference=None) -> pd.Series:
    """Compatibility adapter to the independent set-owned run reduction."""
    if indicator.grain != "run":
        raise ValueError(f"{indicator.name}: expected a run-grain declaration")
    return loader.evaluate(indicator, series, anchor=anchor, reference=reference).values


def reduce_group_mean(
    indicator, series, *, expected_run_ids, anchor, reference=None
) -> pd.Series:
    """Compatibility adapter preserving equal member weights and NaN propagation."""
    return loader.evaluate(
        indicator,
        series,
        anchor=anchor,
        reference=reference,
        expected_run_ids=expected_run_ids,
        group=True,
    ).values


def reduce_bundle(indicator, series, *, expected_run_ids, anchor, validation=None):
    """Resolve workflow qualification outside the neutral calculation module."""
    from blueearth_cst.experiment import return_level_validation

    validation = (
        return_level_validation.build_declaration()
        if validation is None
        else validation
    )
    validation = dict(
        validation,
        tested_domain=dict(
            validation.get("tested_domain", {}), shapes_c=_declared_shapes(validation)
        ),
    )
    result = loader.evaluate(
        indicator,
        series,
        anchor=anchor,
        expected_run_ids=expected_run_ids,
        validation=validation,
    )
    return result.values, result.metadata["evidence"]


def _declared_shapes(validation):
    """Compatibility access to qualified shape reporting; never a fit gate."""
    from blueearth_cst.experiment import return_level_validation
    from blueearth_cst.indicators.extremes import _declared_shapes as neutral_shapes

    declaration = (
        return_level_validation.build_declaration()
        if validation is None
        else validation
    )
    return neutral_shapes(declaration)


def _declared_probabilities(validation):
    """Compatibility access to the qualification declaration's probability gate."""
    from blueearth_cst.experiment import return_level_validation
    from blueearth_cst.indicators.extremes import (
        _declared_probabilities as neutral_probabilities,
    )

    declaration = (
        return_level_validation.build_declaration()
        if validation is None
        else validation
    )
    return neutral_probabilities(declaration)
