"""Opt-in discharge hazard quantiles under the production qualification gate."""

import math

from blueearth_cst.indicators.contract import IndicatorResult
from blueearth_cst.indicators.extremes import reduce_bundle


def flood_return_level(indicator, series, context) -> IndicatorResult:
    """Fit member-local maxima; never widen the supplied qualification domain."""
    values, evidence = reduce_bundle(
        indicator,
        series,
        expected_run_ids=context.expected_run_ids,
        anchor=context.anchor,
        validation=context.validation,
    )
    return IndicatorResult(
        values, metadata={"evidence": evidence, "estimator": "gf15-lmoments-c/1"}
    )


def _return_period(value) -> bool:
    """must be a finite return period greater than one year"""
    return type(value) in (int, float) and math.isfinite(value) and value > 1


PARAMETERS = {flood_return_level: {"return_period": _return_period}}
