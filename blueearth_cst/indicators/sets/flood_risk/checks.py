"""Independent L-moment inverse checks with explicit estimator attribution."""

from blueearth_cst.indicators.checking import return_level_expected


def expected(indicator, series, context, *, group=False):
    """Check annual maximum extraction, fit sample and requested probability."""
    if indicator.calculation_method != "flood_return_level/1":
        raise ValueError(
            f"missing independent check for calculation {indicator.calculation_method!r}"
        )
    return return_level_expected(indicator, series, context)


def check_result(indicator, series, result, context, *, group=False):
    """Do not attribute another estimator's matching scalar to GF15 qualification."""
    if result.metadata.get("estimator") != "gf15-lmoments-c/1":
        raise ValueError("estimator metadata must identify gf15-lmoments-c/1")
