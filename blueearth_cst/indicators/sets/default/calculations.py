"""Default formulas with native pandas missingness and explicit water years."""

from blueearth_cst.indicators.contract import IndicatorResult
from blueearth_cst.indicators.extremes import reduce_bundle
from blueearth_cst.indicators.series import annual_blocks, prepared_frame


def annual_summary(indicator, series, context) -> IndicatorResult:
    """Average the declared water-year summary, retaining partial blocks."""
    statistic = indicator.calculation_parameters["statistic"]
    blocks = annual_blocks(series, context, statistic)
    return IndicatorResult(blocks.mean())


def annual_rolling7_extreme(indicator, series, context) -> IndicatorResult:
    """Reduce complete trailing seven-day windows, including across blocks."""
    blocks = annual_blocks(
        series, context, indicator.calculation_parameters["extreme"], window=7
    )
    return IndicatorResult(blocks.mean())


def annual_min7_mean_ratio(indicator, series, context) -> IndicatorResult:
    """Average each water year's minimum-seven-day / mean-flow ratio."""
    minimum = annual_blocks(series, context, "min", window=7)
    return IndicatorResult((minimum / annual_blocks(series, context, "mean")).mean())


def reference_month_mean(indicator, series, context) -> IndicatorResult:
    """Use one explicitly resolved reference month, never select per response."""
    which = indicator.calculation_parameters["which"]
    reference = context.reference
    if (
        reference is None
        or reference.which != which
        or reference.month not in range(1, 13)
    ):
        raise ValueError("explicit resolved wet/dry month reference required")
    frame = prepared_frame(series, context)["frame"]
    return IndicatorResult(
        frame[frame.index.month == reference.month]
        .resample(context.anchor)
        .mean()
        .mean()
    )


def pooled_return_level(indicator, series, context) -> IndicatorResult:
    """Use the qualified estimator and preserve screening/refusal exceptions."""
    values, evidence = reduce_bundle(
        indicator,
        series,
        expected_run_ids=context.expected_run_ids,
        anchor=context.anchor,
        validation=context.validation,
        _prepared=context.prepared,
    )
    return IndicatorResult(
        values, metadata={"evidence": evidence, "estimator": "gf15-lmoments-c/1"}
    )


PARAMETERS = {
    annual_summary: {
        "variable": (
            "q",
            "aet",
            "pet",
            "gwr",
            "precip",
            "snow",
            "overland_flow",
            "temp",
        ),
        "statistic": ("mean", "sum", "max", "min", "p95"),
    },
    annual_rolling7_extreme: {"extreme": ("min", "max")},
    annual_min7_mean_ratio: {},
    reference_month_mean: {"which": ("wet", "dry")},
    pooled_return_level: {"kind": ("annual_max", "rolling_annual_min")},
}


def validate(function, parameters) -> None:
    """Conditional variable controls remain owned by the formula's set."""
    if function is annual_summary:
        accepted = {
            "q": ("mean", "max", "min", "p95"),
            "aet": ("sum",),
            "pet": ("sum",),
            "gwr": ("sum",),
            "precip": ("sum",),
            "snow": ("max",),
            "overland_flow": ("mean",),
            "temp": ("mean",),
        }
        if parameters["statistic"] not in accepted[parameters["variable"]]:
            raise ValueError(
                f"statistic: accepted for {parameters['variable']}: {accepted[parameters['variable']]}"
            )


def normalize_parameters(function, parameters):
    """Keep the historic fixed-seven-day selector as a local declarative alias."""
    if function is pooled_return_level and parameters == {
        "kind": "rolling7_annual_min"
    }:
        return {"kind": "rolling_annual_min"}
    return parameters
