"""Opt-in flood discharge hazards; these do not represent exposure or damages."""

from blueearth_cst.indicators.contract import Indicator, ResponseRequirement
from blueearth_cst.indicators.sets.flood_risk import calculations as c

ASSUMPTIONS = {
    "annual_blocks": "member-local maxima under configured water-year anchor; retain partial blocks",
    "estimator": "gf15-lmoments-c/1 with qualified source digests and supplied probability domain",
    "screening": "at least max(ceil(T),10) usable blocks; failed fits are refused",
    "qualification": "shipped GF15 evidence covers p=.5 and .9 only; other periods need new evidence",
}
INDICATORS = tuple(
    Indicator(
        id=f"q_return_level_{period}yr_max",
        calculate=c.flood_return_level,
        parameters={"return_period": period},
        input=ResponseRequirement("q", "m3 s-1"),
        units="m3 s-1",
        description=f"{period}-year annual maximum discharge return level",
        statistic="return_level_max",
        grain="bundle",
        return_period=period,
        blocks_per_return_period=1.0,
        group_reduction="pooled-member-local-blocks/1",
    )
    for period in (2, 5, 10, 15, 20)
)
METHODS = {"flood_return_level/1": c.flood_return_level}


def resolve_indicator(function, parameters):
    """Periods outside examples retain this method's explicit sample/qualification gates."""
    from dataclasses import replace

    return replace(
        INDICATORS[0], parameters=parameters, return_period=parameters["return_period"]
    )
