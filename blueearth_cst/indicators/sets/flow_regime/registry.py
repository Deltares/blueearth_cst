"""Five optional flow-regime indicators, preserving production units and reductions."""

from dataclasses import replace

from blueearth_cst.indicators.contract import Indicator, ResponseRequirement
from blueearth_cst.indicators.sets.default import registry as default
from blueearth_cst.indicators.sets.flow_regime import calculations as c

ASSUMPTIONS = {
    "annual_blocks": "configured water-year anchor; partial blocks retained with equal block weight",
    "rolling": "complete trailing windows across water-year boundaries, never across members",
    "changes": "positive rise or positive fall magnitude; zeros excluded; later day's water year",
    "group_mean": "equal member weights with NaN propagation",
}

_PRODUCTION_ADDITIONS = tuple(
    Indicator(
        id=name,
        calculate=function,
        parameters=parameters,
        input=ResponseRequirement("q", "m3 s-1"),
        units="m3 s-1 d-1" if statistic in {"rise_rate", "fall_rate"} else "m3 s-1",
        description=name.replace("_", " "),
        statistic=statistic,
    )
    for name, function, parameters, statistic in (
        (
            "q_mean_annual_30day_max",
            c.annual_rolling_extreme,
            {"window": 30, "extreme": "max"},
            "Q30day_max",
        ),
        (
            "q_mean_annual_30day_min",
            c.annual_rolling_extreme,
            {"window": 30, "extreme": "min"},
            "Q30day_min",
        ),
        (
            "q_mean_annual_90day_min",
            c.annual_rolling_extreme,
            {"window": 90, "extreme": "min"},
            "Q90day_min",
        ),
        ("q_mean_rise_rate", c.annual_daily_change, {"direction": "rise"}, "rise_rate"),
        ("q_mean_fall_rate", c.annual_daily_change, {"direction": "fall"}, "fall_rate"),
    )
)
_ALIASES = {
    "q_annual_mean": "q_annual_mean",
    "q_baseflow_index": "q_baseflow_index",
    "q_return_level_10yr_max": "flood_peak_10yr",
    "q_mean_annual_p95": "high_flow_p95",
    "q_mean_annual_7day_min": "low_flow_7day",
    "gwr_annual_total": "recharge_depth",
}
INDICATORS = (
    *(
        replace(item, id=_ALIASES[item.id])
        for item in default.INDICATORS
        if item.id in _ALIASES
    ),
    Indicator(
        id="whole_period_mean",
        calculate=c.full_period_mean,
        parameters={},
        input=ResponseRequirement("q", "m3 s-1"),
        units="m3 s-1",
        description="Arithmetic mean of all retained daily samples; no annual weighting.",
    ),
    *_PRODUCTION_ADDITIONS,
)
METHODS = {
    **default.METHODS,
    "annual_rolling_extreme/1": c.annual_rolling_extreme,
    "annual_daily_change/1": c.annual_daily_change,
    "full_period_mean/1": c.full_period_mean,
}
