"""Closed WF0-owned diagnostic settings, resolved without config mutation."""

import math

DEFAULTS = {
    "reference_period": None,
    "comparison_period": None,
    "max_missing_days_month": 3,
    "max_missing_days_year": 15,
    "wet_day_threshold_mm": 1.0,
    "spi_distribution": "gamma",
    "pt_temperature_source": "era5",
}


def parse_settings(block, window, sources):
    """Resolve defaults and reject unknown keys/types at DAG construction."""
    block = {} if block is None else block
    if not isinstance(block, dict) or set(block) - set(DEFAULTS):
        raise ValueError("WF0 diagnostics must be a mapping with registered keys")
    settings = DEFAULTS | block
    for key in ("reference_period", "comparison_period"):
        value = settings[key]
        if value is None:
            value = (
                window if key == "reference_period" else settings["reference_period"]
            )
        if not isinstance(value, dict) or set(value) != {"start", "end"}:
            raise ValueError(f"{key} requires {{start: year, end: year}}")
        if (
            any(type(v) is not int for v in value.values())
            or value["start"] > value["end"]
        ):
            raise ValueError(f"{key} requires ordered integer years")
        settings[key] = dict(value)
    for key in ("max_missing_days_month", "max_missing_days_year"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} requires a nonnegative integer")
    wet = settings["wet_day_threshold_mm"]
    if (
        isinstance(wet, bool)
        or not isinstance(wet, (int, float))
        or not math.isfinite(wet)
        or wet <= 0
    ):
        raise ValueError("wet_day_threshold_mm must be finite and positive")
    if settings["spi_distribution"] not in {"gamma", "pearson3"}:
        raise ValueError("spi_distribution must be gamma or pearson3")
    carrier = settings["pt_temperature_source"]
    if "pt_temperature_source" in block and (
        carrier not in sources or not carries_temp(carrier)
    ):
        raise ValueError(
            "pt_temperature_source must name a declared genuine temperature carrier"
        )
    settings["subbasin_figures"] = (
        False  # resolved from the sibling WF0 key by the caller
    )
    settings["analysis_window"] = dict(window)
    return settings


def carries_temp(source):
    """Precipitation-only candidates never acquire borrowed temperature identity."""
    from blueearth_cst.climate_analysis.source_capabilities import source_climate_vars

    return "temp" in source_climate_vars(source)
