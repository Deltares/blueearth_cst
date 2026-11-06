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
