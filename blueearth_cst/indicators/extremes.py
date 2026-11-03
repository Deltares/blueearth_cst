"""Qualified pooled return-level extraction and screening without workflow state."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from blueearth_cst.indicators import gev_lmoments
from blueearth_cst.indicators.contract import EvaluationContext, IndicatorDeclaration
from blueearth_cst.indicators.series import (
    ResponseSeries,
    annual_blocks,
    compatible_bundle,
)
from blueearth_cst.indicators.temporal import validate_requirement

GEV_SCREENING_BLOCK_FLOOR = 10


class InsufficientReturnLevelBlocks(ValueError):
    """Operational screening refused a location before fitting."""


class InvalidReturnLevelFit(ValueError):
    """The estimator refused this location; its structured record is attached.

    Only a refusal reaches this exception. An unexpected fault keeps its own
    traceback and propagates, because a fault silently converted into a refusal
    row would be indistinguishable from a location the estimator genuinely
    declined (D3).
    """

    def __init__(self, message: str, *, fit_result=None):
        super().__init__(message)
        self.fit_result = fit_result


class ReturnLevelProbabilityNotCovered(ValueError):
    """A requested probability lies outside the verified benchmark domain (D4).

    This is a configuration and evidence-contract fault, not an estimator
    refusal: it fires before the library is invoked and aborts publication.
    Changing a return period must not silently inherit a bounded declaration,
    so the domain is never clamped, substituted or expanded to accommodate it.
    """


#: Identifiers for the extraction rules D1 preserves, recorded per location so a
#: retained fit says which sampling produced it rather than leaving it implied.
EXTRACTION_POLICY = {
    "return_level_max": "member_local_annual_max/1",
    "return_level_7day_min": "member_local_rolling7_annual_min/1",
}
MISSINGNESS_POLICY = "drop_nonfinite_blocks/1"
PARTIAL_BLOCK_POLICY = "retain_partial_period_blocks/1"


@dataclass(frozen=True)
class ReturnLevelEvidence:
    """Usable-block counts, coverage and the retained fit for one location."""

    location_id: str
    member_counts: tuple[tuple[str, int, str, str], ...]
    total: int
    required: int
    parameters: tuple[float, ...]
    member_count: int = 0
    extraction_policy: str = ""
    missingness_policy: str = MISSINGNESS_POLICY
    partial_block_policy: str = PARTIAL_BLOCK_POLICY
    fit: dict | None = None
    shape_coverage: dict | None = None


def _declared_probabilities(validation) -> list:
    """Read the verified declaration's tested probabilities once per reduction.

    Resolved outside the per-location loop because building the declaration
    reads and re-verifies the whole shipped report; the membership check itself
    still runs before every individual fit.
    """

    declaration = {} if validation is None else validation
    declared = declaration.get("tested_domain", {}).get("probabilities")
    if not isinstance(declared, list):
        raise ReturnLevelProbabilityNotCovered(
            "the retained return-level validation declares no tested probabilities"
        )
    return declared


def _declared_shapes(validation) -> list | None:
    """Read the declaration's tested shape values, or None if it states none.

    Unlike :func:`_declared_probabilities` this never raises. Shape coverage is
    reported, never enforced -- see :func:`_shape_coverage` for why.
    """

    declaration = {} if validation is None else validation
    declared = declaration.get("tested_domain", {}).get("shapes_c")
    return declared if isinstance(declared, list) and declared else None


def _shape_coverage(shape: float, declared: list | None) -> dict | None:
    """Report whether a FITTED shape falls inside the assessed domain (D4).

    This is deliberately **not** a guard, and it is not the shape-shaped twin of
    :func:`_covered_probability`. Two reasons, both structural:

    A probability is a property of the REQUEST, known before any fit, so
    refusing an untested one costs nothing and is unambiguous. A shape is a
    property of the RESULT: it does not exist until the fit does, and at the
    block counts this reduction produces its sampling spread is comparable to
    the width of the tested domain itself. Gating on it would refuse fits whose
    true shape is inside the domain and admit fits whose true shape is outside.

    And a refusal here is not local. ``InvalidReturnLevelFit`` is caught nowhere
    in this package, so one refused fit aborts the whole indicator set -- every
    other location, every other indicator, including statistics that are fully
    inside the assessed domain. A reporting field costs a consumer nothing and
    keeps that blast radius at zero.

    ``excess`` is recorded alongside the boolean on purpose: a fitted shape just
    outside the tested range is consistent with a true shape inside it, so a
    consumer needs the distance to apply its own threshold rather than trusting
    a bare flag. ``None`` means the declaration states no tested shapes --
    unavailable, not "covered".
    """
    if declared is None:
        return None
    low, high = min(declared), max(declared)
    inside = low <= shape <= high
    return {
        "tested_shapes_c": list(declared),
        "tested_range_c": [low, high],
        "fitted_c": shape,
        "within_tested_range": inside,
        "excess_beyond_tested_range": 0.0
        if inside
        else min(abs(shape - low), abs(shape - high)),
    }


def _covered_probability(probability: float, declared: list) -> None:
    """Refuse a probability the verified benchmark domain does not cover (D4).

    Exact numeric membership, for the fixed .9 and .5 values the declaration
    names. No clamping, substitution or widening.
    """
    if not any(probability == value for value in declared):
        raise ReturnLevelProbabilityNotCovered(
            f"requested probability {probability!r} is outside the verified "
            f"benchmark domain {declared!r}; the declaration is bounded and is "
            "not expanded to cover a changed return period"
        )


def reduce_bundle(
    indicator: IndicatorDeclaration,
    series: Sequence[ResponseSeries],
    *,
    expected_run_ids: Sequence[str],
    anchor: str,
    validation=None,
    _prepared=None,
) -> tuple[pd.Series, tuple[ReturnLevelEvidence, ...]]:
    """Extract blocks within each member, screen and fit without splicing runs.

    `validation` is the verified return-level declaration whose tested domain
    bounds the probabilities that may be requested. The publication path passes
    the declaration retained in its plan; omitting it resolves the shipped one.
    """
    if (
        indicator.grain != "bundle"
        or indicator.return_period is None
        or indicator.blocks_per_return_period is None
    ):
        raise ValueError(
            f"{indicator.name}: expected an explicit return-level bundle declaration"
        )
    if not expected_run_ids or len(set(expected_run_ids)) != len(expected_run_ids):
        raise ValueError(f"{indicator.name}: bundle needs unique evaluated members")
    if {item.run_id for item in series} != set(expected_run_ids):
        raise ValueError(
            f"{indicator.name}: bundle membership differs from declared grouping"
        )
    validate_requirement(indicator, series)
    by_run = {
        run_id: [item for item in series if item.run_id == run_id]
        for run_id in expected_run_ids
    }
    location_sets = [
        {item.location_id for item in members} for members in by_run.values()
    ]
    if any(locations != location_sets[0] for locations in location_sets):
        raise ValueError(f"{indicator.name}: bundle location sets differ")
    required = max(
        math.ceil(indicator.blocks_per_return_period * indicator.return_period),
        GEV_SCREENING_BLOCK_FLOOR,
    )
    maximum = indicator.statistic == "return_level_max"
    quantile = (
        1 - 1 / indicator.return_period if maximum else 1 / indicator.return_period
    )
    declared = _declared_probabilities(validation)
    declared_shapes = _declared_shapes(validation)
    preparation_context = EvaluationContext(
        anchor, prepared={} if _prepared is None else _prepared
    )
    blocks_by_run = {
        run: annual_blocks(
            members,
            preparation_context,
            "max" if maximum else "min",
            window=None if maximum else 7,
        )
        if all(item.time.equals(members[0].time) for item in members)
        else None
        for run, members in by_run.items()
    }
    values, evidence = {}, []
    for location in sorted(location_sets[0]):
        members = sorted(
            [item for item in series if item.location_id == location],
            key=lambda item: item.run_id,
        )
        compatible_bundle(members)
        samples, counts = [], []
        for item in members:
            if blocks_by_run[item.run_id] is None:
                # Locations may have distinct grids; only members of one
                # location must align. Preserve that standalone bundle contract.
                daily = pd.Series(item.values, index=item.time)
                blocks = (
                    daily.resample(anchor).max()
                    if maximum
                    else daily.rolling(7).mean().resample(anchor).min()
                )
            else:
                blocks = blocks_by_run[item.run_id][location]
            usable = blocks.to_numpy(dtype="float64")
            usable = usable[np.isfinite(usable)]
            samples.append(usable)
            counts.append(
                (item.run_id, len(usable), str(item.time[0]), str(item.time[-1]))
            )
        sample = np.concatenate(samples)
        context = (
            f"indicator={indicator.name}, unit_members={tuple(expected_run_ids)}, location={location}, "
            f"T={indicator.return_period}, required={required}, actual={len(sample)}, per_member={counts}"
        )
        if len(sample) < required:
            raise InsufficientReturnLevelBlocks(context)
        context += (
            f", sample_range={(float(sample.min()), float(sample.max()))}, "
            f"estimator={gev_lmoments.ESTIMATOR_ID}/"
            f"lmoments3-{gev_lmoments.DEPENDENCY_VERSION}, "
            f"implementation_revision={indicator.implementation_revision}"
        )
        # A constant sample is no longer screened out here: it reaches the
        # adapter and comes back as a structured invalid_range refusal (D1).
        _covered_probability(quantile, declared)
        result = gev_lmoments.fit_case(sample, (quantile,))
        if not result.accepted:
            raise InvalidReturnLevelFit(
                f"{context}: estimator refused: {', '.join(result.refusal_reasons)}",
                fit_result=result,
            )
        value = result.value(quantile)
        parameters = result.physical_parameters
        values[location] = value
        evidence.append(
            ReturnLevelEvidence(
                location,
                tuple(counts),
                len(sample),
                required,
                tuple(parameters[name] for name in ("c", "loc", "scale")),
                member_count=len(members),
                extraction_policy=EXTRACTION_POLICY[indicator.statistic],
                fit=result.record(),
                shape_coverage=_shape_coverage(parameters["c"], declared_shapes),
            )
        )
    return pd.Series(values), tuple(evidence)
