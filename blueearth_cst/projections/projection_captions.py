"""Recoverable captions for table-driven WF2 projection figures.

Captions are self-contained, in the register of journal figure legends: what
is shown (quantity, units, periods, basin-average CMIP6 ensemble), how it is
encoded where no key says so, and the interpretive caveat the figure needs
(inter-model spread is not a confidence interval, sign agreement is not
significance). Model support, omitted scenarios, shifted or joined windows
and exclusions are reported where they apply; thresholds and per-model
detail stay in the analysis record and the caption manifest.
"""

from blueearth_cst.climate_analysis.diagnostic_figures import plural
from blueearth_cst.projections.projection_figures import figure_key

TITLES = {
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
    "All figures show basin-average quantities from the CMIP6 ensemble. Unless stated otherwise, each "
    "model is represented by the mean of its available ensemble members and models are weighted "
    "equally. Inter-model spread describes structural model uncertainty; it is not a confidence "
    "interval, and models that share code ancestry are not independent. Window definitions, "
    "eligibility thresholds and per-figure support and exclusion counts are recorded in "
    "_engine/analysis_record.json and _engine/figure_captions.json."
)

#: Encoding of the seasonal grids (points, median, interquartile range).
_GRID = (
    "Each point represents one model (mean of its ensemble members); horizontal bars mark the "
    "multi-model median and shaded boxes the interquartile range, shown where at least ten models "
    "contribute."
)
_AMOUNTS = (
    "Panel titles give the reference-period seasonal total, the multi-model median of the monthly "
    "climatologies summed over the season."
)
_SPREAD = "Inter-model spread is not a confidence interval."
_HISTORICAL_SPREAD = (
    "The spread of the Historical group reflects differences in simulated present-day climate among "
    "models; model biases against observations are not assessed here."
)


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
                f"The analysis window{label} was shifted from {supplied['requested']} to "
                f"{supplied['used']} to remain within the scenario simulations."
            )
        if supplied.get("joined"):
            notes.append(
                f"Historical and scenario simulations are concatenated to complete the "
                f"window{label}."
            )
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
        notes.append("Number of models: " + "; ".join(parts) + ".")
    if dropped := counts.get("dropped"):
        names = ", ".join(
            f"{_label(s)} ({plural(n, 'model')})" for s, n in dropped.items()
        )
        notes.append(f"Scenarios with fewer than ten models are not shown: {names}.")
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
        "The grey band around zero spans the 90% range of differences between window means "
        "expected from interannual variability alone, estimated from the models' historical "
        "simulations (median across models) assuming serially independent years; autocorrelation "
        "would widen it."
    ]


def _dry(counts, cells="panels"):
    if not counts.get("mostly_dry"):
        return []
    return [
        f"Hatched {cells} {'are months' if cells == 'months' else 'mark seasons'} in which at least half of the models fall below the "
        "dry-precipitation threshold and are excluded; the remaining models are too few to be "
        "representative."
    ]


def caption(filename, record, window_id="", counts=None):
    """A self-contained figure legend for one product."""
    filename = figure_key(filename)
    counts = counts or {}
    window = next(
        (w for w in record.get("windows", []) if w["window_id"] == window_id), None
    )
    n_models = counts.get("n_models", len(record.get("models", [])))
    reference = _span(record["reference"])
    temperature = "temperature" in filename
    variable = "temperature" if temperature else "precipitation"
    notes = _window_notes(window) if window else []

    if filename == "availability.png":
        missing = counts.get("n_unretrieved", 0)
        text = [
            "CMIP6 ensemble composition. Shaded cells mark models with at least one retrieved "
            "member per experiment; numerals give member counts above one. Families follow Kuma et "
            "al. (2023): * provisional, – unassigned.",
            *(
                [f"{plural(missing, 'model')} with no retrieved member are omitted."]
                if missing
                else []
            ),
        ]
    elif filename.startswith("trajectory-"):
        unit = "°C" if temperature else "%"
        text = [
            f"Annual basin-average {variable} anomalies ({unit}) relative to {reference}. Thick "
            "lines show the multi-model median and shading the inter-model range for the historical "
            "simulations (black) and each SSP scenario (colours); thin lines show individual "
            "ensemble members. Anomalies are computed against each member's own reference-period "
            f"mean{' and expressed as a percentage of it' if not temperature else ''}. Annual values "
            "are unsmoothed, so the range includes interannual variability. The dashed line marks "
            "the transition from historical to scenario forcing (2014/2015) and grey shading the "
            f"future analysis window. Based on {plural(n_models, 'model')}. {_SPREAD}"
        ]
    elif filename.startswith("annual-"):
        units = ("°C", "°C") if temperature else ("mm yr⁻¹", "%")
        text = [
            f"Annual basin-average {variable} ({units[0]}, upper panels) and its anomaly relative to "
            f"each member's {reference} mean ({units[1]}, lower panels), by SSP scenario. Thin lines "
            "show individual ensemble members (grey: historical; colours: scenario), thick lines the "
            "member median and shading the member range, which is not a confidence interval. The "
            "dashed line marks the transition from historical to scenario forcing (2014/2015) and "
            "grey shading the future analysis window; missing years are left as gaps."
        ]
    elif filename.startswith("change-factor-cloud"):
        windows = [
            w for w in record["windows"] if not window_id or w["window_id"] == window_id
        ]
        text = [
            "Projected change in annual mean temperature (°C) against relative change in annual "
            f"precipitation (%) for {'; '.join(_period(w) for w in windows)}. Each point represents "
            "one model (mean of its ensemble members) under one SSP scenario (colours), pairing both "
            "variables from the same simulations. The point cloud describes the spread of the "
            "ensemble; it is not a joint probability distribution."
        ]
        if "combined" in filename:
            text.append(
                "Lines connect the same model and scenario across future periods."
            )
    elif filename == "sign-agreement.png":
        text = [
            f"Agreement on the sign of projected annual and seasonal change in temperature (a) and "
            f"precipitation (b), {_period(window)}. Colour gives the fraction of models sharing the "
            "sign of the multi-model median (blue: increase; red: decrease); agreement indicates "
            "consistency of direction, not statistical significance.",
            *(
                ["Hatched cells: too few models above the dry-precipitation threshold."]
                if counts.get("mostly_dry")
                else []
            ),
        ]
    elif filename.startswith("seasonal-cycle-"):
        unit = "°C" if temperature else "mm d⁻¹"
        text = [
            f"Monthly climatology of basin-average {variable} ({unit}) for the reference period "
            f"{_span(window['reference'])} (black) and for {_span(window['used'])} under each SSP "
            "scenario (colours), shown as the multi-model median of model climatologies. Grey "
            "shading spans the inter-model range of the reference climatology; the projected spread "
            "is shown in the corresponding monthly-change figure. The reference climatology uses the "
            "models of the first scenario shown."
        ]
    elif filename.startswith("monthly-change-"):
        unit = "°C" if temperature else "%"
        text = [
            f"Projected change in monthly basin-average {variable} ({unit}) for {_period(window)}, "
            "by SSP scenario. Bars mark the multi-model median, boxes the interquartile range (where at "
            "least ten models contribute) and whiskers the full inter-model range; each model contributes the mean of its ensemble "
            "members.",
            *_band(counts),
            *_dry(counts, "months"),
            *_support(counts),
            _SPREAD,
        ]
    elif filename == "change-temperature.png":
        text = [
            f"Projected change in seasonal basin-average temperature (°C) for {_period(window)}, by "
            f"SSP scenario. {_GRID}",
            *_band(counts),
            *_support(counts),
            _SPREAD,
        ]
    elif filename == "change-precipitation-pct.png":
        text = [
            f"Projected relative change in seasonal basin-average precipitation (%) for "
            f"{_period(window)}, by SSP scenario. {_GRID}",
            _AMOUNTS,
            *_band(counts),
            *_dry(counts),
            *_support(counts),
            _SPREAD,
        ]
    elif filename.startswith("values-"):
        ref, fut = _span(window["reference"]), _span(window["used"])
        quantity, method = {
            "values-precipitation-total.png": (
                "Seasonal basin-average precipitation totals (mm)",
                "Totals are derived from monthly mean precipitation rates multiplied by the number "
                "of days in each month.",
            ),
            "values-temperature-mean.png": (
                "Seasonal basin-average mean temperature (°C)",
                "Seasonal means are day-weighted averages of the monthly mean temperatures.",
            ),
            "values-temperature-sd.png": (
                "Interannual standard deviation of seasonal basin-average temperature (°C)",
                "Standard deviations are computed from linearly detrended seasonal series within "
                "each period, separately for each ensemble member.",
            ),
            "values-precipitation-sd.png": (
                "Interannual standard deviation of seasonal basin-average precipitation totals (mm)",
                "Standard deviations are computed from linearly detrended seasonal series within "
                "each period, separately for each ensemble member, and expressed as season totals.",
            ),
            "values-precipitation-cv.png": (
                "Interannual coefficient of variation of seasonal basin-average precipitation (%)",
                "The coefficient of variation is the standard deviation of the linearly detrended "
                "seasonal totals divided by their mean, computed separately for each ensemble "
                "member; seasons with a reference or future mean below the dry-precipitation "
                "threshold are excluded.",
            ),
        }[filename]
        text = [
            f"{quantity} simulated by each model for the reference period (Historical, {ref}) and "
            f"for {fut} under each SSP scenario. {_GRID} {method}",
            *([_AMOUNTS] if filename == "values-precipitation-total.png" else []),
            _HISTORICAL_SPREAD,
            *_dry(counts),
            *_support(counts),
        ]
    else:  # variability-member-ratios.png
        text = [
            "Ratio of future to reference interannual variability of annual basin-average "
            "temperature (standard deviation, a), precipitation (standard deviation, b) and "
            f"precipitation (coefficient of variation, c) for {_period(window)}. Faint points show "
            "individual ensemble members; bars mark the multi-model median, boxes the interquartile "
            "range (shown where at least ten models contribute) and whiskers the full range, after "
            "geometric averaging of member ratios within each model. Series are linearly detrended "
            "within each period; a ratio of 1 indicates no change. Members of one model are not "
            "independent, and the spread is not a confidence interval.",
            *_exclusions(counts),
        ]
    return " ".join([*text, *notes])
