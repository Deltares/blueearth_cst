"""Independent flow-regime expectations, including annual missingness."""

import numpy as np

from blueearth_cst.indicators.checking import member_expectation
from blueearth_cst.indicators.sets.default import checks as default_checks
from blueearth_cst.indicators.sets.default.registry import METHODS as DEFAULT_METHODS


def expected(indicator, series, context, *, group=False):
    """Calculate scalar locations independently of production formula functions."""

    if indicator.calculation_method in DEFAULT_METHODS:
        return default_checks.expected(indicator, series, context, group=group)
    if indicator.calculation_method not in {
        "full_period_mean/1",
        "annual_rolling_extreme/1",
        "annual_daily_change/1",
    }:
        raise ValueError(
            f"missing independent check for calculation {indicator.calculation_method!r}"
        )

    def scalar(daily):
        parameters = indicator.calculation_parameters
        if indicator.calculation_method == "full_period_mean/1":
            finite = [value for value in daily.to_numpy() if not np.isnan(value)]
            return sum(finite) / len(finite) if finite else np.nan
        if indicator.calculation_method == "annual_rolling_extreme/1":
            daily = daily.rolling(parameters["window"]).mean()
            annual = [
                getattr(block, parameters["extreme"])()
                for _, block in daily.resample(context.anchor)
            ]
        else:
            difference = daily.diff()
            positive = (
                difference[difference > 0]
                if parameters["direction"] == "rise"
                else -difference[difference < 0]
            )
            # Reindex keeps empty water-year blocks rather than silently changing coverage.
            positive = positive.reindex(daily.index)
            annual = [block.mean() for _, block in positive.resample(context.anchor)]
        usable = [value for value in annual if not np.isnan(value)]
        return float(np.mean(usable)) if usable else np.nan

    return member_expectation(indicator, series, context, scalar, group=group)


def check_result(indicator, series, result, context, *, group=False):
    """Inherited default fits retain their independent estimator-identity check."""
    if indicator.calculation_method in DEFAULT_METHODS:
        default_checks.check_result(indicator, series, result, context, group=group)
