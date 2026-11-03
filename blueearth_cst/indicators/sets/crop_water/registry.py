"""Ten declarations for six crop-water screening methods."""

from blueearth_cst.indicators.contract import Indicator, ResponseRequirement
from blueearth_cst.indicators.sets.crop_water import calculations as c


def _input(variable: str, units: str) -> ResponseRequirement:
    return ResponseRequirement(
        variable,
        units,
        coverage="complete explicit winter, summer or October-September periods",
        missingness="native NaN invalidates the full period; partial edges excluded",
    )


P = _input("precip", "mm dt-1")
PET = _input("pet", "mm dt-1")
AET = _input("aet", "mm dt-1")
Q = _input("q", "m3 s-1")
R = _input("gwr", "mm dt-1")

ASSUMPTIONS = {
    "season_labels": "daily interval ends: winter 1 Nov-31 May, summer 1 May-31 Oct",
    "recharge_year": "1 Oct-30 Sep, labelled by ending year regardless of project anchor",
    "periods": "complete finite daily coverage; partial edges and missing periods excluded",
    "summary": "arithmetic mean of valid period values; no valid period is undefined",
    "spatial": "daily subcatchment means before calculation; Q at named outlet",
    "et": "reference PET and landscape AET are screening proxies, not calibrated crop stress",
    "qcrit": "fixed location thresholds from historical baseline; never refit per scenario",
}

INDICATORS = tuple(
    indicator
    for season in ("winter", "summer")
    for indicator in (
        Indicator(
            id=f"{season}_climatic_deficit",
            calculate=c.climatic_deficit,
            method="seasonal_climatic_deficit/1",
            parameters={"season": season},
            input=P,
            additional_inputs=(PET,),
            units="mm/season",
            description=f"Daily positive PET minus precipitation summed over {season}",
        ),
        Indicator(
            id=f"{season}_longest_dry_spell",
            calculate=c.dry_spell,
            method="seasonal_longest_dry_spell/1",
            parameters={"season": season, "dry_day_mm": 1.0},
            input=P,
            units="days",
            description=f"Longest strict dry-day run within {season}",
        ),
        Indicator(
            id=f"{season}_relative_et_deficit",
            calculate=c.relative_et_deficit,
            method="seasonal_relative_et_deficit/1",
            parameters={"season": season},
            input=AET,
            additional_inputs=(PET,),
            units="percent",
            description=f"Ratio-of-sums AET/PET deficit in {season}",
        ),
        Indicator(
            id=f"{season}_et_limitation_days",
            calculate=c.et_limitation_days,
            method="seasonal_et_limitation_days/1",
            parameters={"season": season, "pet_min_mm": 1.0, "ratio": 0.5},
            input=AET,
            additional_inputs=(PET,),
            units="days",
            description=f"Days with strict AET/PET limitation in {season}",
        ),
    )
) + (
    Indicator(
        id="summer_low_flow_days",
        calculate=c.low_flow,
        method="seasonal_low_flow_days/1",
        parameters={"season": "summer", "thresholds": {}},
        input=Q,
        units="days/season",
        description="Summer discharge days strictly below fixed historical Qcrit",
    ),
    Indicator(
        id="annual_recharge",
        calculate=c.recharge,
        method="annual_recharge/1",
        parameters={},
        input=R,
        units="mm/year",
        description="Signed net saturated-zone recharge, October-September",
    ),
)

METHODS = {
    "seasonal_climatic_deficit/1": c.climatic_deficit,
    "seasonal_longest_dry_spell/1": c.dry_spell,
    "seasonal_low_flow_days/1": c.low_flow,
    "annual_recharge/1": c.recharge,
    "seasonal_relative_et_deficit/1": c.relative_et_deficit,
    "seasonal_et_limitation_days/1": c.et_limitation_days,
}
