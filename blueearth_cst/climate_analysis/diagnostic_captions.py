"""Unnumbered scientific captions reused by Markdown, JSON and optional PNG bands."""

from blueearth_cst.climate_analysis.diagnostic_figures import View

DESCRIPTIONS = {
    "coverage": (
        "Monthly data coverage by dataset. Each timeline cell represents one month in reporting-year "
        "order. Pale cells have complete records, darker cells have usable totals with gaps, and gray "
        "cells are missing or unusable. Annual diagnostics use years meeting the specified data "
        "completeness threshold."
    ),
    "seasonal": (
        "Monthly precipitation climatology. Lines show the mean monthly total; bands span the "
        "10th–90th percentiles across available years. Months follow the reporting-year order."
    ),
    "temperature": (
        "Monthly temperature climatology. The line shows mean monthly temperature and the "
        "band spans the 10th–90th percentiles across available years."
    ),
    "timing": (
        "Timing of cumulative annual precipitation reaching 25%, 50% and 75% of each year's total. "
        "Unfilled boxes span the interquartile range, whiskers the observed range, vertical lines the median, "
        "and numeric labels the mean day of the reporting year."
    ),
    "anomaly": (
        "Monthly precipitation departures from the calendar-month mean in the reference period. "
        "Brown indicates drier and teal wetter months; gray marks unavailable monthly totals. "
        "The continuous colour scale is symmetric about zero, spans all pooled finite departures, "
        "and is shared across sources and years. No temporal smoothing is applied."
    ),
    "spi": (
        "Standardized Precipitation Index at 3-, 6- and 12-month accumulation scales. "
        "Red indicates dryness, blue wetness and gray unavailable values; the distribution is fitted "
        "by calendar month using the reference period."
    ),
    "drought-events": (
        "Drought-event duration and accumulated SPI deficit for SPI-3 and SPI-12. "
        "Events are consecutive months below zero that reach SPI ≤ −1. Filled markers show complete "
        "events; hatched markers show truncated events that touch a data gap or "
        "record boundary, so their full duration and accumulated deficit are unknown."
    ),
    "extreme-Rx1day": (
        "Annual maximum one-day precipitation (Rx1day). Points show annual values; "
        "gaps indicate unavailable estimates."
    ),
    "extreme-Rx5day": (
        "Annual maximum five-day precipitation (Rx5day), calculated from complete five-day "
        "windows within each reporting year. Gaps indicate years without a valid estimate."
    ),
    "extreme-SDII": (
        "Simple daily intensity index (SDII): mean precipitation on observed wet days in "
        "each valid reporting year. Gaps indicate years without a valid estimate."
    ),
    "extreme-Wet days": (
        "Annual wet-day count, scaled from observed days to the full reporting year. "
        "Gaps indicate years without a valid estimate."
    ),
    "spell": (
        "Exceedance probability of dry-spell length on a logarithmic probability axis. "
        "A dry day falls below the wet-day threshold; spells touching gaps or record edges "
        "are excluded."
    ),
    "acf": (
        "Autocorrelation of monthly precipitation anomalies at lags of 1–12 months. "
        "Pearson correlations use only pairs with usable monthly totals. Shading shows approximate "
        "pointwise 95% limits under no serial correlation (±1.96/√n, where n is the valid pair count "
        "at each lag); these limits are not adjusted for multiple comparisons."
    ),
    "slopes": (
        "Theil–Sen trends in annual precipitation, wet days and Rx1day. Points are slope estimates "
        "and whiskers are 95% intervals adjusted for lag-1 serial dependence; filled points "
        "have a two-sided Mann–Kendall p < 0.05."
    ),
    "series": (
        "Annual precipitation totals and descriptive Theil–Sen trend lines. End labels give "
        "the slope per decade; totals are shown only for valid reporting years."
    ),
    "pt": (
        "Seasonal precipitation anomalies against temperature anomalies. Precipitation is relative "
        "to its reference-season mean and temperature is an absolute departure; each point is a "
        "reporting year. The precipitation axis varies by season."
    ),
}


TITLES = {
    "coverage": "Monthly climate data coverage",
    "seasonal": "Monthly precipitation climatology",
    "temperature": "Monthly temperature climatology",
    "timing": "Seasonal timing of precipitation accumulation",
    "anomaly": "Monthly precipitation anomalies",
    "spi": "Standardized Precipitation Index",
    "drought-events": "Drought-event duration and accumulated deficit",
    "extreme-Rx1day": "Annual maximum one-day precipitation",
    "extreme-Rx5day": "Annual maximum five-day precipitation",
    "extreme-SDII": "Annual wet-day precipitation intensity",
    "extreme-Wet days": "Annual wet-day count",
    "spell": "Dry-spell length exceedance",
    "acf": "Monthly precipitation anomaly autocorrelation",
    "slopes": "Annual precipitation and extreme-index trends",
    "series": "Annual precipitation totals and trends",
    "pt": "Seasonal precipitation and temperature anomalies",
    "spi-checks": "Reference-period SPI normality checks",
    "sensitivity": "Sensitivity of climate trends to the fitting period",
    "map-precip": "Native-grid annual precipitation climatology",
    "map-temp": "Native-grid annual temperature climatology",
    "map-pet": "Native-grid annual potential evapotranspiration climatology",
}


def title(name: str, sources: list[str], period, support: str) -> str:
    """Use the WF0 plot — dataset (years) format; support stays in captions."""
    return f"{TITLES[name]} — {', '.join(sources)} ({period[0]}–{period[-1]})"


def caption(name: str, view: View) -> str:
    """Return a brief journal-style caption without a figure-number prefix."""
    plotted = (
        [s for s in view.sources if view.results[s]["tclim"] is not None]
        if name == "temperature"
        else view.sources
    )
    sources = ", ".join(view.labels[s] for s in plotted)
    identity = f"{sources}; "
    period = f"{view.years[0]}–{view.years[-1]}"
    support = (
        view.support.replace("_", " ")
        .replace("avg", "average")
        .replace("equal-weight cells", "equally weighted grid cells")
    )
    context = f"{identity}{period}; {support}."
    detail = DESCRIPTIONS[name]
    if name.startswith("extreme-"):
        detail += (
            " Dashed lines show Theil–Sen trends; source-colored labels give slopes "
            "per decade."
        )
    if name == "anomaly":
        detail += f" Requested reference: {view.reference[0]}–{view.reference[1]}."
        for source in plotted:
            month_support = view.results[source]["anomaly_reference_support"]
            usable = [row for row in month_support if row["count"]]
            if usable:
                start = min(row["start"] for row in usable)
                end = max(row["end"] for row in usable)
                counts = [row["count"] for row in month_support]
                detail += (
                    f" {view.labels[source]} usable reference totals: {start}–{end}; "
                    f"{min(counts)}–{max(counts)} observations per calendar month "
                    "(span may contain gaps)."
                )
            else:
                detail += f" {view.labels[source]}: no usable reference totals."
    if name == "spi":
        if view.reference[0] == view.years[0] and view.reference[1] == view.years[-1]:
            context = f"{identity}{period} (reference period); {support}."
        else:
            detail += f" Reference: {view.reference[0]}–{view.reference[1]}."
    if name == "spi":
        distribution = view.results[view.sources[0]]["spi_dist"]
        detail += (
            f" Fit: {'Pearson III' if distribution == 'pearson3' else distribution}."
        )
    if name in {"extreme-SDII", "extreme-Wet days", "spell"}:
        detail += f" Wet day: ≥{view.wet_mm:g} mm."
    if name == "coverage":
        detail += (
            f" Usable months permit ≤{view.max_missing_month} missing days; valid years "
            f"permit ≤{view.max_missing_year} in total."
        )
    if name == "spell":
        counts = ", ".join(
            f"{view.labels[s]} {view.results[s]['n_spells']}" for s in view.sources
        )
        detail += f" Complete spells: {counts}."
    if name == "acf":
        pairs = []
        for s in view.sources:
            values = view.results[s]["acf"]["pairs"]
            pairs.append(f"{view.labels[s]} {values.min()}–{values.max()}")
        detail += f" Valid monthly observation pairs per lag: {', '.join(pairs)}."
    if name == "pt":
        t_source = next(
            (
                view.results[s]["t_source"]
                for s in view.sources
                if view.results[s]["pt"] is not None
            ),
            None,
        )
        if t_source:
            detail += f" Temperature source: {t_source}."
    return f"{detail} {context}"
