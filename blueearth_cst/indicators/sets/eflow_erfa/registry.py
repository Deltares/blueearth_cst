"""The 70 fixed-default eflowrisk median/IQR registrations."""

from blueearth_cst.indicators.contract import Indicator, ResponseRequirement
from blueearth_cst.indicators.sets.eflow_erfa import calculations as c

ASSUMPTIONS = {
    "source": "eflowrisk calculate_flow_metrics and summarize_flow_metrics fixed defaults",
    "annual_blocks": "configured water-year anchor; at least 0.9 finite fraction of calendar days; leap days count",
    "monthly_blocks": "calendar month means require at least 0.8 finite fraction of calendar days",
    "thresholds": "per-record global type-7 0.9/0.1 quantiles; at least 10 finite values; strict >/< pulses",
    "zero_flow": "discharge <= 0; no events means zero count/duration and undefined start",
    "windows": "complete trailing 3/7/30/90-day means over full record; assigned by window end",
    "changes": "full-record daily difference; zero deadband; no smoothing; falls are positive magnitudes",
    "runs": "missing values and water-year boundaries break runs; first longest run wins ties",
    "timing": "one-based water-year day; first extreme tie; linear median and IQR, not circular",
    "summary": "metric-local missing exclusion; median and type-7 IQR; reject no retained years",
    "warmup": "already removed by caller before retained input; no additional removal",
    "group_mean": "equal member weights, propagating undefined run values",
    "calendar": "complete Gregorian daily grid required by response contract; gaps rejected rather than warned",
}

INDICATORS = tuple(
    Indicator(
        id=f"q_{metric}_{statistic}",
        calculate=c.annual_statistic,
        method="erfa_annual_statistic/1",
        parameters={"metric": metric, "statistic": statistic},
        input=ResponseRequirement(
            "q",
            "m3 s-1",
            coverage="water years with >=90% finite calendar days; report retained/dropped coverage",
            missingness="native NaN; >=80% monthly completeness; complete rolling windows; missing breaks runs",
        ),
        units=units,
        description=f"{group}: {label}; {'median' if statistic == 'med' else 'type-7 IQR'} across retained water years",
    )
    for metric, (group, label, units) in c.metric_definitions().items()
    for statistic in ("med", "iqr")
)

METHODS = {"erfa_annual_statistic/1": c.annual_statistic}
