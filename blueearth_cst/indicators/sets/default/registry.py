"""Default registrations and fixed scientific assumptions; formulas live locally."""

from blueearth_cst.indicators.contract import Indicator, ResponseRequirement
from blueearth_cst.indicators.sets.default import calculations as c

ASSUMPTIONS = {
    "annual_blocks": "configured water-year anchor; retain partial edge blocks with equal block weight",
    "missingness": "native NaN, pandas skipna reductions and complete rolling windows",
    "warmup": "already removed by caller before retained input; no additional removal",
    "group_mean": "equal member weights, propagating undefined run values",
    "month_reference": "explicit pooled monthly sum selection at the configured reference location",
    "estimator": "gf15-lmoments-c/1; supplied benchmark bounds probabilities; max(ceil(T),10) usable blocks",
}


def _indicator(
    name,
    function,
    parameters,
    statistic,
    *,
    variable="q",
    grain="run",
    reference=None,
    period=None,
):
    """Keep registration values explicit while sharing unvarying annotations."""
    units = {
        "q": "m3 s-1",
        "snow": "mm",
        "overland_flow": "m3 s-1",
        "temp": "degC",
    }.get(variable, "mm dt-1")
    return Indicator(
        id=name,
        calculate=function,
        parameters=parameters,
        input=ResponseRequirement(variable, units),
        units="1"
        if statistic == "BaseFlowIndex"
        else ("mm" if statistic == "sum" else units),
        description=name.replace("_", " "),
        statistic=statistic,
        grain=grain,
        group_reduction="pooled-member-local-blocks/1"
        if grain == "bundle"
        else "equal-weight-run-reductions-propagate-nan/1",
        reference=reference,
        reference_location="class_c_first_gauge" if reference else None,
        return_period=period,
        blocks_per_return_period=1.0 if period else None,
    )


INDICATORS = (
    _indicator(
        "q_annual_mean",
        c.annual_summary,
        {"variable": "q", "statistic": "mean"},
        "mean",
    ),
    _indicator(
        "q_mean_annual_max",
        c.annual_summary,
        {"variable": "q", "statistic": "max"},
        "max",
    ),
    _indicator(
        "q_mean_annual_min",
        c.annual_summary,
        {"variable": "q", "statistic": "min"},
        "min",
    ),
    _indicator(
        "q_mean_annual_p95",
        c.annual_summary,
        {"variable": "q", "statistic": "p95"},
        "q95",
    ),
    _indicator(
        "q_mean_annual_7day_max",
        c.annual_rolling7_extreme,
        {"extreme": "max"},
        "Q7day_max",
    ),
    _indicator(
        "q_mean_annual_7day_min",
        c.annual_rolling7_extreme,
        {"extreme": "min"},
        "Q7day_min",
    ),
    _indicator("q_baseflow_index", c.annual_min7_mean_ratio, {}, "BaseFlowIndex"),
    _indicator(
        "q_wettest_month_mean",
        c.reference_month_mean,
        {"which": "wet"},
        "wetmonth_mean",
        reference="unperturbed_runs",
    ),
    _indicator(
        "q_driest_month_mean",
        c.reference_month_mean,
        {"which": "dry"},
        "drymonth_mean",
        reference="unperturbed_runs",
    ),
    _indicator(
        "q_return_level_10yr_max",
        c.pooled_return_level,
        {"kind": "annual_max"},
        "return_level_max",
        grain="bundle",
        period=10,
    ),
    _indicator(
        "q_return_level_2yr_7day_min",
        c.pooled_return_level,
        {"kind": "rolling_annual_min"},
        "return_level_7day_min",
        grain="bundle",
        period=2,
    ),
    _indicator(
        "aet_annual_total",
        c.annual_summary,
        {"variable": "aet", "statistic": "sum"},
        "sum",
        variable="aet",
    ),
    _indicator(
        "pet_annual_total",
        c.annual_summary,
        {"variable": "pet", "statistic": "sum"},
        "sum",
        variable="pet",
    ),
    _indicator(
        "gwr_annual_total",
        c.annual_summary,
        {"variable": "gwr", "statistic": "sum"},
        "sum",
        variable="gwr",
    ),
    _indicator(
        "precip_annual_total",
        c.annual_summary,
        {"variable": "precip", "statistic": "sum"},
        "sum",
        variable="precip",
    ),
    _indicator(
        "snow_annual_max",
        c.annual_summary,
        {"variable": "snow", "statistic": "max"},
        "max",
        variable="snow",
    ),
    _indicator(
        "overland_flow_annual_mean",
        c.annual_summary,
        {"variable": "overland_flow", "statistic": "mean"},
        "mean",
        variable="overland_flow",
    ),
    _indicator(
        "temp_annual_mean",
        c.annual_summary,
        {"variable": "temp", "statistic": "mean"},
        "mean",
        variable="temp",
    ),
)

METHODS = {
    "annual_summary/1": c.annual_summary,
    "annual_rolling7_extreme/1": c.annual_rolling7_extreme,
    "annual_min7_mean_ratio/1": c.annual_min7_mean_ratio,
    "reference_month_mean/1": c.reference_month_mean,
    "pooled_return_level/1": c.pooled_return_level,
}
