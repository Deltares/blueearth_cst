"""One table describing each configurable wflow output.

Three things used to be keyed off a wflow output variable, in three places:

* its CSDMS name, for the TOML (``model/setup_gauges_and_outputs.WFLOW_VARS``);
* its resample rule and axis legend, for the figures
  (``shared/func_plot_signature.WFLOW_VARS``);
* its COLUMN NAME in ``output.csv``, which was the semantic label with
  ``_basavg`` glued on -- so a config asking for ``groundwater recharge``
  produced ``groundwater recharge_basavg_101``. Spaces in a column name, 32
  characters before the id, and a figure filename to match.

They are one table here, keyed by a short CODE. The code is what reaches the
csv, so the column is ``gwr_101``; the semantic label stays the config spelling,
because that is what a user writes in ``wflow_outvars``.

**The code is a contract.** It appears in ``output.csv`` headers, in the
variables hydromt derives from them (``<code>_subcatchment``), in the derived
tables from rule 1.14, and in figure filenames. Changing one renames all four.

Discharge is deliberately absent from ``CODES``: it does not travel the
basin-average path at all. It is emitted per-gauge and per-outlet with the fixed
header ``Q``, which ``shared/gauges.py`` keys on.
"""

#: Semantic label (what a config writes) -> short column code.
CODES = {
    "precipitation": "p",
    "potential evapotranspiration": "pet",
    "overland flow": "qof",
    "actual evapotranspiration": "aet",
    "groundwater recharge": "gwr",
    "snow": "swe",
    "temperature": "temp",
}

#: How each code aggregates to a month, and how its axis is labelled. Keyed by
#: CODE rather than by label, because the figure sees only what the csv carried.
PLOT_META = {
    "qof": {"resample": "mean", "legend": "Overland Flow (m$^3$s$^{-1}$)"},
    "aet": {
        "resample": "sum",
        "legend": "Actual Evapotranspiration (mm month$^{-1}$)",
    },
    "gwr": {"resample": "sum", "legend": "Groundwater Recharge (mm month$^{-1}$)"},
    "swe": {"resample": "mean", "legend": "Snowpack Liquid Water (mm)"},
    "temp": {"resample": "mean", "legend": "Air Temperature (°C)"},
    "p": {"resample": "sum", "legend": "Precipitation (mm month$^{-1}$)"},
    "pet": {
        "resample": "sum",
        "legend": "Potential Evapotranspiration (mm month$^{-1}$)",
    },
}

for _code, _meta in PLOT_META.items():
    _meta["kind"] = (
        "rate" if _code == "qof" else "stock" if _code == "swe" else "amount"
    )
    _meta["monthly_units"] = (
        "m³ s⁻¹" if _code == "qof" else "mm" if _code == "swe" else "mm month⁻¹"
    )
    _meta["annual_units"] = (
        "m³ s⁻¹" if _code == "qof" else "mm" if _code == "swe" else "mm yr⁻¹"
    )
    _meta["trend_units"] = (
        "m³ s⁻¹ decade⁻¹"
        if _code == "qof"
        else "mm decade⁻¹"
        if _code == "swe"
        else "mm yr⁻¹ decade⁻¹"
    )
del _code, _meta

# Temperature is a state in degrees Celsius; depth/flux units never apply.
PLOT_META["temp"].update(
    kind="state", monthly_units="°C", annual_units="°C", trend_units="°C decade⁻¹"
)

SUBBASIN_PLOT_DIRNAME = "subbasins"
BASAVG_FIGURE_STEMS = {
    "p": "precip",
    "pet": "pet",
    "qof": "overland_flow",
    "swe": "snow",
    "aet": "aet",
    "gwr": "gwr",
    "temp": "temp",
}

#: hydromt names a csv-derived variable ``<header>_<mapname>``. Basin-average
#: outputs use the ``subcatchment`` map, so this suffix identifies them —
#: independently of the header, which is exactly what made the old
#: ``"_basavg" in dvar`` test brittle once the header changed.
SUBCATCHMENT_SUFFIX = "_subcatchment"


def code_for(label: str) -> str:
    """Short column code for a ``wflow_outvars`` label."""
    return CODES[label]


def code_from_variable(variable: str) -> str:
    """Recover the code from a hydromt variable name (``gwr_subcatchment``)."""
    name = str(variable)
    if name.endswith(SUBCATCHMENT_SUFFIX):
        name = name[: -len(SUBCATCHMENT_SUFFIX)]
    return name


def is_basin_average(variable) -> bool:
    """Whether a hydromt output variable is a per-subcatchment average."""
    return str(variable).endswith(SUBCATCHMENT_SUFFIX)
