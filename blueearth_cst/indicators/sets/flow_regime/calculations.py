"""Optional flow-regime formulas, with production annual means."""

from blueearth_cst.indicators.contract import IndicatorResult
from blueearth_cst.indicators.series import response_frame


def annual_rolling_extreme(indicator, series, context) -> IndicatorResult:
    """Mean water-year extreme of complete trailing N-day means."""
    parameters = indicator.calculation_parameters
    blocks = (
        response_frame(series)
        .rolling(parameters["window"])
        .mean()
        .resample(context.anchor)
    )
    return IndicatorResult(getattr(blocks, parameters["extreme"])().mean())


def annual_daily_change(indicator, series, context) -> IndicatorResult:
    """Mean water-year rise/fall magnitude; zero differences are excluded."""
    change = response_frame(series).diff()
    if indicator.calculation_parameters["direction"] == "rise":
        change = change.where(change > 0)
    else:
        change = (-change).where(change < 0)
    return IndicatorResult(change.resample(context.anchor).mean().mean())


def full_period_mean(indicator, series, context) -> IndicatorResult:
    """Arithmetic mean over retained days, without equal annual block weighting."""
    return IndicatorResult(response_frame(series).mean())


PARAMETERS = {
    annual_rolling_extreme: {"window": (30, 90), "extreme": ("min", "max")},
    annual_daily_change: {"direction": ("rise", "fall")},
    full_period_mean: {},
}


def validate(function, parameters) -> None:
    """The production flow-regime suite includes only the three reviewed windows."""
    if function is annual_rolling_extreme and parameters == {
        "window": 90,
        "extreme": "max",
    }:
        raise ValueError(
            "window/extreme: accepted combinations: (30,min), (30,max), (90,min)"
        )
