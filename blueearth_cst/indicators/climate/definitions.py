"""Display identities for the first climate-index delivery."""

# Public product names, display labels and units. Availability is per product,
# source, period and location, not inferred from this inventory.
INDEX_METADATA = {
    "cdd": ("CDD", "days"),
    "cwd": ("CWD", "days"),
    "r10mm": ("R10mm", "days"),
    "r20mm": ("R20mm", "days"),
    "r95p_amount": ("R95p amount", "mm"),
    "r95p_fraction": ("R95p fraction", "%"),
    "txx": ("TXx", "degC"),
    "tnx": ("TNx", "degC"),
    "mean_temp_below_zero_days": ("Mean T < 0°C days", "days"),
    "mean_temp_above_threshold_days": ("Mean T above threshold days", "days"),
    "tx90p": ("TX90p", "%"),
    "tn90p": ("TN90p", "%"),
    "wsdi": ("WSDI (fixed reference)", "days"),
    "heatwave_count": ("Heatwave count", "events"),
    "heatwave_days": ("Heatwave days", "days"),
    "longest_heatwave": ("Longest heatwave overlap", "days"),
    "wet_spell_mean": ("Mean complete wet spell", "days"),
    "wet_spell_median": ("Median complete wet spell", "days"),
    "wet_spell_p90": ("90th percentile complete wet spell", "days"),
    "rx1day": ("Rx1day", "mm"),
    "rx5day": ("Rx5day", "mm"),
    "wet_days": ("Observed wet days", "days"),
    "prcptot": ("Wet-day precipitation", "mm"),
    "sdii": ("SDII", "mm/day"),
}

# Presentation scopes; unavailable metrics and wholly unavailable scopes are omitted.
INDEX_GROUPS = {
    "spells": (
        "Dry and wet spells",
        3,
        (
            "cdd",
            "cwd",
            "wet_spell_mean",
            "wet_spell_median",
            "wet_spell_p90",
        ),
    ),
    "heavy_rainfall": (
        "Heavy rainfall",
        2,
        (
            "r10mm",
            "r20mm",
            "rx1day",
            "rx5day",
            "r95p_amount",
            "r95p_fraction",
        ),
    ),
    "rainfall_characteristics": (
        "Rainfall amount and frequency",
        3,
        (
            "prcptot",
            "wet_days",
            "sdii",
        ),
    ),
    "temperature_extremes": (
        "Temperature extremes",
        2,
        (
            "txx",
            "tnx",
            "mean_temp_below_zero_days",
            "mean_temp_above_threshold_days",
            "tx90p",
            "tn90p",
        ),
    ),
    "heatwaves": (
        "Heatwaves",
        3,
        (
            "wsdi",
            "heatwave_days",
            "longest_heatwave",
            "heatwave_count",
        ),
    ),
}

INDEX_SERIES_LABELS = {
    "cdd": "Longest dry spell · CDD",
    "cwd": "Longest wet spell · CWD",
    "wet_spell_mean": "Mean wet-spell duration",
    "wet_spell_median": "Median wet-spell duration",
    "wet_spell_p90": "90th-percentile wet spell",
    "r10mm": "Days ≥10 mm · R10mm",
    "r20mm": "Days ≥20 mm · R20mm",
    "rx1day": "Maximum 1-day rain · Rx1day",
    "rx5day": "Maximum 5-day rain · Rx5day",
    "prcptot": "Wet-day rain · PRCPTOT",
    "wet_days": "Wet days",
    "sdii": "Wet-day intensity · SDII",
}
