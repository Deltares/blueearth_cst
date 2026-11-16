"""Recoverable captions for table-driven WF2 projection figures.

Captions are concise journal figure legends: quantity, units and periods, the
encoding a key does not explain, and the per-figure facts a reader needs
(model support, omitted scenarios, shifted or joined windows, hatching,
variability bands). Suite-wide caveats and derivation methods are stated once
in ``SUITE_NOTE``; thresholds and per-model detail stay in the analysis record
and the caption manifest.
"""

from blueearth_cst.climate_analysis.diagnostic_figures import plural
from blueearth_cst.projections.projection_figures import figure_key

TITLES = {
    "trajectory-tdew.png": "Annual dew-point anomalies",
    "seasonal-cycle-tdew.png": "Monthly dew-point climatology",
    "change-tdew.png": "Annual and seasonal dew-point change",
    "trajectory-wind.png": "Annual 10 m wind anomalies",
    "seasonal-cycle-wind.png": "Monthly 10 m wind climatology",
    "change-wind.png": "Annual and seasonal 10 m wind change",
    "trajectory-pet.png": "Annual PET anomalies",
    "seasonal-cycle-pet.png": "Monthly PET climatology",
    "change-pet-pct.png": "Annual and seasonal PET change (%)",
    "change-pet.png": "Annual and seasonal PET change",
    "availability.png": "CMIP6 ensemble composition",
    "trajectory-temperature.png": "Annual temperature anomalies",
    "trajectory-precipitation.png": "Annual precipitation anomalies",
    "annual-temperature.png": "Annual temperature by scenario",
    "annual-precipitation.png": "Annual precipitation by scenario",
    "change-temperature.png": "Seasonal temperature change",
    "change-precipitation-pct.png": "Seasonal precipitation change (%)",
    "values-precipitation-total.png": "Seasonal precipitation totals",
    "values-temperature-mean.png": "Seasonal mean temperature",
    "values-temperature-sd.png": "Interannual variability of seasonal temperature",
    "values-precipitation-sd.png": "Interannual variability of seasonal precipitation",
    "values-precipitation-cv.png": "Coefficient of variation of seasonal precipitation",
    "seasonal-cycle-temperature.png": "Monthly temperature climatology",
    "seasonal-cycle-precipitation.png": "Monthly precipitation climatology",
    "monthly-change-temperature.png": "Monthly temperature change",
    "monthly-change-precipitation.png": "Monthly precipitation change (%)",
    "variability-member-ratios.png": "Annual interannual variability ratios by ensemble member",
    "change-factor-cloud.png": "Annual temperature and precipitation change",
    "sign-agreement.png": "Model agreement on the sign of annual and seasonal change",
    "change-factor-cloud-combined.png": "Annual temperature and precipitation change across future periods",
}

#: Stated once for the whole suite, at the head of ``figure_captions.md``.
SUITE_NOTE = (
    "All figures show basin-average quantities from the CMIP6 ensemble. Spatial averages use the "
    "selected buffered bounding box with spherical cell-area weights, including cells outside the "
    "basin polygon; they are not polygon integrals. Spatial change maps keep native GCM cells, "
    "clipped to the basin outline for display, and show the cell-wise median of paired members "
    "within each model/scenario. Unless stated otherwise, each model is the mean of its available "
    "ensemble members and models are weighted equally. Seasonal and annual temperatures are "
    "day-weighted means; precipitation totals are monthly rates times days per month; variability "
    "statistics use linearly detrended series. Inter-model spread describes structural model "
    "uncertainty; it is not a confidence interval, and models that share code ancestry are not "
    "independent. Historical spread reflects model differences in present-day climate; biases "
    "against observations are not assessed. Derived-variable methods (PET, dew point, wind), "
    "window definitions, eligibility thresholds and per-figure support and exclusion counts are "
    "recorded in _engine/analysis_record.json and _engine/plots/figure_captions.json."
)

#: Encoding of the seasonal grids (points, median, interquartile range).
_GRID = (
    "Points: models; bars: multi-model median; boxes: interquartile range (≥10 models)."
)
_AMOUNTS = "Panel titles give reference-period seasonal totals."


def title(filename, record, window_id=""):
    """Name a projection product using its resolved period, not a horizon alias."""
    name = TITLES[figure_key(filename)]
    if window_id:
        window = next(w for w in record["windows"] if w["window_id"] == window_id)
        name += f" — {window['used'][0]}–{window['used'][1]} relative to {window['reference'][0]}–{window['reference'][1]}"
    return f"{name}; CMIP6 basin average"


def _span(pair):
    return f"{pair[0]}–{pair[1]}"


def _period(window):
    return f"{_span(window['used'])} relative to {_span(window['reference'])}"


def _window_notes(window):
    """Only the window facts a reader could not infer: shifted or joined series."""
    notes = []
    for horizon, supplied in window.get("horizon_windows", {"window": window}).items():
        label = "" if horizon == "window" else f" ({horizon})"
        if supplied.get("shifted"):
            notes.append(
                f"Window{label} shifted from {supplied['requested']} to "
                f"{supplied['used']} to stay within the scenario runs."
            )
        if supplied.get("joined"):
            notes.append(f"Window{label} joins historical and scenario runs.")
    return notes


def _support(counts):
    """Models per group (a range where seasons differ) and omitted scenarios."""
    from blueearth_cst.projections.projection_suite_figures import _label

    models = counts.get("models", {})
    notes = []
    if models:
        parts = [
            f"{'Historical' if s == 'historical' else _label(s)}, "
            + (f"{lo}" if lo == hi else f"{lo}–{hi}")
            for s, (lo, hi) in models.items()
        ]
        notes.append("Models: " + "; ".join(parts) + ".")
    if dropped := counts.get("dropped"):
        names = ", ".join(
            f"{_label(s)} ({plural(n, 'model')})" for s, n in dropped.items()
        )
        notes.append(f"Not shown (fewer than ten models): {names}.")
    return notes


def _exclusions(counts):
    """Exclusion counts, only when nonzero; the manifest always records them."""
    excluded, zero = counts.get("n_excluded", 0), counts.get("n_zero_future", 0)
    parts = [
        f"{plural(excluded, 'degenerate ratio')} {'is' if excluded == 1 else 'are'} excluded"
        if excluded
        else "",
        f"{plural(zero, 'ratio')} with zero future variability {'is' if zero == 1 else 'are'} excluded"
        if zero
        else "",
    ]
    text = " and ".join(part for part in parts if part)
    return [text[0].upper() + text[1:] + "."] if text else []


def _band(counts):
    if not counts.get("variability_band"):
        return []
    return [
        "Grey band: 90% range of window-mean differences expected from interannual "
        "variability alone."
    ]


def _dry(counts, cells="panels"):
    if not counts.get("mostly_dry"):
        return []
    unit = "months" if cells == "months" else "seasons"
    return [f"Hatched {cells}: {unit} where most models fall below the dry threshold."]


def _models(n):
    return f"Based on {plural(n, 'model')}."


def caption(filename, record, window_id="", counts=None):
    """A concise, self-contained figure legend for one product."""
    filename = figure_key(filename)
    counts = counts or {}
    window = next(
        (w for w in record.get("windows", []) if w["window_id"] == window_id), None
    )
    notes = _window_notes(window) if window else []
    derived = next((v for v in ("tdew", "wind", "pet") if v in filename), None)
    if derived:
        subject, unit = {
            "tdew": ("dew-point", "°C"),
            "wind": ("10 m wind", "m/s"),
            "pet": ("PET", "mm/day"),
        }[derived]
        if filename.startswith("seasonal-cycle-"):
            quantity = (
                f"Monthly {subject} climatology ({unit}), {_span(window['reference'])} (black) "
                f"and {_span(window['used'])} by scenario; multi-model medians, shading the "
                "reference inter-model range."
            )
        else:
            if derived == "pet":
                unit = "%" if "-pct" in filename else "mm, period integrals"
            quantity = f"Annual and seasonal {subject} change ({unit}), {_period(window)}. {_GRID}"
        support = dict(counts)
        support.setdefault(
            "models", {s: (n, n) for s, n in counts.get("by_scenario", {}).items()}
        )
        n_models = counts.get("n_models", len(record.get("models", [])))
        return " ".join([quantity, _models(n_models), *_support(support), *notes])

    n_models = counts.get("n_models", len(record.get("models", [])))
    reference = _span(record["reference"])
    temperature = "temperature" in filename
    variable = "temperature" if temperature else "precipitation"

    if filename == "availability.png":
        missing = counts.get("n_unretrieved", 0)
        text = [
            "CMIP6 ensemble composition. Shaded cells: models with at least one retrieved member "
            "per experiment; numerals: member counts above one. Families after Kuma et al. "
            "(2023); * provisional, – unassigned.",
            *(
                [f"{plural(missing, 'model')} with no retrieved member omitted."]
                if missing
                else []
            ),
        ]
    elif filename.startswith("trajectory-"):
        unit = "°C" if temperature else f"% of each member's {reference} mean"
        relative = "" if not temperature else f" relative to {reference}"
        text = [
            f"Annual {variable} anomalies ({unit}){relative}. Thick lines: multi-model median; "
            "shading: inter-model range; thin lines: ensemble members. Dashed line: "
            "historical/scenario transition (2014/2015); grey band: analysis window. "
            + _models(n_models)
        ]
    elif filename.startswith("annual-"):
        units = ("°C", "°C") if temperature else ("mm yr⁻¹", "%")
        text = [
            f"Annual {variable} ({units[0]}, top) and anomaly relative to {reference} "
            f"({units[1]}, bottom) by scenario. Thin lines: members (grey: historical); "
            "thick lines and shading: member median and range. Dashed line: 2014/2015 "
            "transition; grey band: analysis window."
        ]
    elif filename.startswith("change-factor-cloud"):
        windows = [
            w for w in record["windows"] if not window_id or w["window_id"] == window_id
        ]
        text = [
            "Annual temperature change (°C) versus precipitation change (%), "
            f"{'; '.join(_period(w) for w in windows)}. Each point is one model under one "
            "scenario; the cloud shows ensemble spread, not a joint probability."
        ]
        if "combined" in filename:
            text.append("Lines join each model and scenario across periods.")
    elif filename == "sign-agreement.png":
        text = [
            "Fraction of models sharing the sign of the multi-model median change in "
            f"temperature (a) and precipitation (b), {_period(window)} (blue: increase; red: "
            "decrease). Agreement is not significance.",
            *(
                ["Hatched cells: too few models above the dry threshold."]
                if counts.get("mostly_dry")
                else []
            ),
        ]
    elif filename.startswith("seasonal-cycle-"):
        unit = "°C" if temperature else "mm d⁻¹"
        text = [
            f"Monthly {variable} climatology ({unit}), {_span(window['reference'])} (black) and "
            f"{_span(window['used'])} by scenario; multi-model medians, shading the reference "
            "inter-model range."
        ]
    elif filename.startswith("monthly-change-"):
        unit = "°C" if temperature else "%"
        text = [
            f"Monthly {variable} change ({unit}), {_period(window)}. Bars: multi-model median; "
            "boxes: interquartile range (≥10 models); whiskers: full range.",
            *_band(counts),
            *_dry(counts, "months"),
            *_support(counts),
        ]
    elif filename == "change-temperature.png":
        text = [
            f"Seasonal temperature change (°C), {_period(window)}. {_GRID}",
            *_band(counts),
            *_support(counts),
        ]
    elif filename == "change-precipitation-pct.png":
        text = [
            f"Seasonal precipitation change (%), {_period(window)}. {_GRID}",
            _AMOUNTS,
            *_band(counts),
            *_dry(counts),
            *_support(counts),
        ]
    elif filename.startswith("values-"):
        ref, fut = _span(window["reference"]), _span(window["used"])
        quantity = {
            "values-precipitation-total.png": "Seasonal precipitation totals (mm)",
            "values-temperature-mean.png": "Seasonal mean temperature (°C)",
            "values-temperature-sd.png": "Interannual standard deviation of seasonal temperature (°C)",
            "values-precipitation-sd.png": "Interannual standard deviation of seasonal precipitation (mm)",
            "values-precipitation-cv.png": "Interannual coefficient of variation of seasonal precipitation (%)",
        }[filename]
        text = [
            f"{quantity}, Historical ({ref}) and {fut} by scenario. {_GRID}",
            *([_AMOUNTS] if filename == "values-precipitation-total.png" else []),
            *_dry(counts),
            *_support(counts),
        ]
    else:  # variability-member-ratios.png
        text = [
            "Future-to-reference ratio of interannual variability of annual temperature "
            "(standard deviation, a) and precipitation (standard deviation, b; coefficient of "
            f"variation, c), {_period(window)}. Faint points: members; bars, boxes and whiskers: "
            "median, interquartile range (≥10 models) and range of model geometric means. "
            "A ratio of 1 means no change.",
            *_exclusions(counts),
        ]
    return " ".join([*text, *notes])
