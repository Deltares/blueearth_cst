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
    "climate_indices": None,
}


def parse_settings(block, window, sources):
    """Resolve defaults and reject unknown keys/types at DAG construction."""
    block = {} if block is None else block
    if not isinstance(block, dict) or set(block) - set(DEFAULTS):
        raise ValueError("WF0 diagnostics must be a mapping with registered keys")
    settings = DEFAULTS | block
    indices = settings["climate_indices"]
    indices = {} if indices is None else indices
    if not isinstance(indices, dict) or set(indices) - {
        "enabled",
        "seasons",
        "reference_minimum_fraction",
        "reference_period",
        "mean_temperature_threshold_degc",
    }:
        raise ValueError(
            "climate_indices accepts enabled, seasons, reference_period, reference_minimum_fraction "
            "and mean_temperature_threshold_degc"
        )
    indices = {
        "enabled": False,
        "seasons": {},
        "reference_minimum_fraction": 0.9,
        "reference_period": None,
        "mean_temperature_threshold_degc": 25.0,
    } | indices
    reference = (
        window if indices["reference_period"] is None else indices["reference_period"]
    )
    if (
        not isinstance(reference, dict)
        or set(reference) != {"start", "end"}
        or any(type(v) is not int for v in reference.values())
        or reference["start"] > reference["end"]
    ):
        raise ValueError(
            "climate_indices.reference_period requires ordered calendar years"
        )
    indices["reference_period"] = dict(reference)
    threshold = indices["mean_temperature_threshold_degc"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
    ):
        raise ValueError("mean_temperature_threshold_degc must be a finite number")
    if type(indices["enabled"]) is not bool:
        raise ValueError("climate_indices.enabled requires a boolean")
    fraction = indices["reference_minimum_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction <= 1
    ):
        raise ValueError("reference_minimum_fraction must be in (0, 1]")
    if not isinstance(indices["seasons"], dict):
        raise ValueError(
            "climate_indices.seasons requires a mapping of names to month lists"
        )
    from blueearth_cst.indicators.climate.periods import seasonal_periods

    for name, months in indices["seasons"].items():
        if not isinstance(months, list):
            raise ValueError("Season months require an ordered list")
        seasonal_periods(2000, 2000, name=name, months=tuple(months))
    settings["climate_indices"] = indices
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
    if indices["enabled"] and reference["end"] <= reference["start"]:
        raise ValueError("Climate indices require at least two reference years")
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
