# -*- coding: utf-8 -*-
"""Pure WF2 suite folder/product contracts and configured-horizon parsing.

Folder declarations use distinct resolved analysis windows. Availability
tables decide which products are drawn; captions retain configured labels.
Scenario is the suite color channel; artists reuse WF0 publication tokens."""

from __future__ import annotations

import re
from collections.abc import Sequence

# Re-export for existing callers; the shared YAML owns the IPCC mapping.
from blueearth_cst.shared.plot_colors import SCENARIO_COLORS as SCENARIO_COLORS

# Logical product keys remain independent of their public output filenames.
FIGURE_FILENAMES = {
    "availability.png": "cmip6_ensemble_coverage_basin_avg.png",
    "trajectory-temperature.png": "cmip6_temp_annual_anomaly_ts_basin_avg.png",
    "trajectory-precipitation.png": "cmip6_precip_annual_anomaly_ts_basin_avg.png",
    "annual-temperature.png": "cmip6_temp_annual_overview_basin_avg.png",
    "annual-precipitation.png": "cmip6_precip_annual_overview_basin_avg.png",
    "change-temperature.png": "cmip6_temp_annual_seasonal_change_box_basin_avg.png",
    "change-precipitation-pct.png": "cmip6_precip_annual_seasonal_change_pct_box_basin_avg.png",
    "values-precipitation-total.png": "cmip6_precip_seasonal_total_box_basin_avg.png",
    "values-temperature-mean.png": "cmip6_temp_seasonal_mean_box_basin_avg.png",
    "seasonal-cycle-temperature.png": "cmip6_temp_monthly_clim_basin_avg.png",
    "seasonal-cycle-precipitation.png": "cmip6_precip_monthly_clim_basin_avg.png",
    "monthly-change-temperature.png": "cmip6_temp_monthly_change_box_basin_avg.png",
    "monthly-change-precipitation.png": "cmip6_precip_monthly_change_pct_box_basin_avg.png",
    "values-temperature-sd.png": "cmip6_temp_seasonal_sd_box_basin_avg.png",
    "values-precipitation-sd.png": "cmip6_precip_seasonal_sd_box_basin_avg.png",
    "values-precipitation-cv.png": "cmip6_precip_seasonal_cv_box_basin_avg.png",
    "variability-member-ratios.png": "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png",
    "change-factor-cloud.png": "cmip6_precip_temp_annual_change_cloud_basin_avg.png",
    "sign-agreement.png": "cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png",
    "change-factor-cloud-combined.png": "cmip6_precip_temp_annual_multi_period_change_cloud_basin_avg.png",
}
FIGURE_KEYS = {filename: key for key, filename in FIGURE_FILENAMES.items()}


def figure_key(filename: str) -> str:
    """Resolve a canonical filename to the renderer's logical product key."""
    return FIGURE_KEYS[filename]


def period_folder(window_id: str) -> str:
    """Canonical folder for a resolved start/end analysis-window identity."""
    return f"period_{window_id.replace('-', '_')}"


_SUITE_PRODUCT_KEYS = {
    "F3.1": {"folder_kind": "full_period", "filenames": ("availability.png",)},
    "F3.4": {
        "folder_kind": "full_period",
        "filenames": ("trajectory-temperature.png", "trajectory-precipitation.png"),
    },
    "F3.4b": {
        "folder_kind": "full_period",
        "filenames": ("annual-temperature.png", "annual-precipitation.png"),
    },
    "F3.2": {
        "folder_kind": "period",
        "filenames": (
            "change-temperature.png",
            "change-precipitation-pct.png",
            "values-precipitation-total.png",
            "values-temperature-mean.png",
        ),
    },
    "F3.3": {
        "folder_kind": "period",
        "filenames": (
            "seasonal-cycle-temperature.png",
            "seasonal-cycle-precipitation.png",
            "monthly-change-temperature.png",
            "monthly-change-precipitation.png",
        ),
    },
    "F3.5a": {
        "folder_kind": "period",
        "filenames": (
            "values-temperature-sd.png",
            "values-precipitation-sd.png",
            "values-precipitation-cv.png",
        ),
    },
    "F3.5c": {"folder_kind": "period", "filenames": ("variability-member-ratios.png",)},
    "F3.6b": {"folder_kind": "period", "filenames": ("change-factor-cloud.png",)},
    "F5.1": {"folder_kind": "period", "filenames": ("sign-agreement.png",)},
    "combined-cloud": {
        "folder_kind": "full_period",
        "filenames": ("change-factor-cloud-combined.png",),
    },
}

SUITE_PRODUCTS = {
    key: {
        **product,
        "filenames": tuple(FIGURE_FILENAMES[name] for name in product["filenames"]),
    }
    for key, product in _SUITE_PRODUCT_KEYS.items()
}


def suite_folders(horizons, water_year_start="Jan", reference_window=(1985, 2014)):
    """One owned folder for each distinct resolved analysis window."""
    from blueearth_cst.projections.get_change_climate_proj import analysis_windows

    records = analysis_windows(horizons, water_year_start, reference_window)
    return [
        "full_period",
        *dict.fromkeys(period_folder(r["window_id"]) for r in records.values()),
    ]


def parse_horizon_period(period: Sequence[int] | str) -> tuple[int, int]:
    """Return one inclusive ``(start_year, end_year)`` horizon pair.

    R01 delivers ``future_horizons`` as lists (``[2070, 2090]``); pre-R01
    configs delivered comma-separated strings (``"2070, 2090"``). Both are
    accepted, because a project config written against either is still valid.
    """
    values = period.split(",") if isinstance(period, str) else list(period)
    if len(values) != 2:
        raise ValueError(f"horizon period must contain two years, got {period!r}")
    try:
        start, end = (int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"horizon years must be integers, got {period!r}") from exc
    if start > end:
        raise ValueError(f"horizon start must not exceed end, got {start}-{end}")
    return start, end


def sanitize_horizon_name(name: str) -> str:
    """Sanitize a configured horizon name into a portable directory name.

    A horizon name is free text in the config, and it reaches the filesystem
    here. Anything outside ``[a-z0-9]`` collapses to a single underscore, so a name
    that would otherwise produce an unopenable path on win-64 is caught at parse
    time rather than at the moment a figure is written.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if not slug:
        raise ValueError(f"horizon name has no portable characters: {name!r}")
    return slug


def horizon_directory(name: str, period: Sequence[int] | str) -> str:
    """Return ``<sanitized_name>_<start>_<end>`` for one horizon.

    The years are in the directory name, not only the horizon's label, because
    a project that redefines ``far`` from 2070-2090 to 2060-2080 would otherwise
    overwrite the old figure in place and leave nothing saying the window moved.
    """
    start, end = parse_horizon_period(period)
    return f"{sanitize_horizon_name(name)}_{start}_{end}"
