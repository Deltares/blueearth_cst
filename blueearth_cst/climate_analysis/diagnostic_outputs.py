"""The one declared/written inventory for canonical WF0 diagnostics."""

from pathlib import Path

from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.figure_naming import figure_filename

TABLES = (
    "coverage",
    "valid_years",
    "annual_indices",
    "monthly_values",
    "precip_climatology",
    "temp_climatology",
    "rainfall_timing",
    "spi",
    "spi_fit_checks",
    "drought_events",
    "dry_spells",
    "dry_spell_exceedance",
    "anomaly_acf",
    "trends",
    "trend_sensitivity",
    "pt_anomalies",
    "homogeneity",
)
FORMS = {
    "coverage": ("precip", "monthly_coverage"),
    "seasonal": ("precip", "monthly_clim_band"),
    "temperature": ("temp", "monthly_clim_band"),
    "timing": ("precip", "annual_timing"),
    "anomaly": ("precip", "monthly_anomaly"),
    "spi": ("precip", "monthly_spi"),
    "drought-events": ("precip", "spi_events"),
    "extreme-Rx1day": ("precip", "annual_rx1day_ts"),
    "extreme-Rx5day": ("precip", "annual_rx5day_ts"),
    "extreme-SDII": ("precip", "annual_sdii_ts"),
    "extreme-Wet days": ("precip", "annual_wet_days_ts"),
    "spell": ("precip", "dry_spell_exceedance"),
    "acf": ("precip", "monthly_anomaly_acf"),
    "slopes": ("precip", "annual_trend_interval"),
    "series": ("precip", "annual_trend_ts"),
    "pt": ("precip_temp", "seasonal_anomaly"),
    "spi-checks": ("precip", "monthly_spi_fit"),
    "sensitivity": ("precip", "annual_trend_sensitivity"),
}


def inventory(root, sources, settings, comparison=False):
    """Named compute/render files; unknown subbasin IDs use optional directories."""
    root = Path(root)
    scope = "comparison" if comparison else sources[0]
    temp = sum(carries_temp(s) for s in sources)
    forms = dict(FORMS)
    if temp < (2 if comparison else 1):
        forms.pop("temperature")
    binding = settings["pt_temperature_source"]
    if (comparison and (binding not in sources or not carries_temp(binding))) or (
        not comparison and not temp
    ):
        forms.pop("pt")
    figures = {
        k: str(root / "plots" / figure_filename(scope, *v, "basin_avg"))
        for k, v in forms.items()
    }
    if not comparison:
        for variable in ("precip", "temp", "pet") if temp else ("precip",):
            figures[f"map-{variable}"] = str(
                root
                / "plots"
                / figure_filename(scope, variable, "annual_clim_map", "basin_ext")
            )
    compute = {k: str(root / "tables" / f"{k}.csv") for k in TABLES}
    compute["metadata"] = str(root / "_engine" / "diagnostics.json")
    if comparison:
        compute["agreement"] = str(root / "tables" / "agreement.csv")
        compute["agreement_md"] = str(root / "tables" / "agreement.md")
    else:
        compute["daily"] = str(root / "tables" / "daily_basin.csv")
        compute["maps"] = str(root / "tables" / "annual_climatology.nc")
    render = {
        "titles_json": str(root / "plots" / "_engine" / "figure_titles.json"),
        "titles_md": str(root / "plots" / "figure_titles.md"),
        "captions_json": str(root / "plots" / "_engine" / "figure_captions.json"),
        "captions_md": str(root / "plots" / "figure_captions.md"),
    }
    render.update({f"figure_{k}": v for k, v in figures.items()})
    return {"root": str(root), "compute": compute, "render": render, "figures": figures}


def retarget_render(paths, render_root):
    """The same inventory with figures and caption/title files under ``render_root``.

    Compute paths keep pointing at the retained run, so a re-render reads exactly
    what the rule read and never overwrites the run's own plots.
    """
    root, render_root = Path(paths["root"]), Path(render_root)

    def move(path):
        return str(render_root / Path(path).relative_to(root))

    return {
        **paths,
        "render_root": str(render_root),
        "figures": {k: move(v) for k, v in paths["figures"].items()},
        "render": {k: move(v) for k, v in paths["render"].items()},
    }


def diagnostic_source_outputs(store_dir, source, settings):
    return inventory(store_dir, [source], settings)


def diagnostic_comparison_outputs(comparison_dir, sources, settings):
    if len(sources) < 2:
        raise ValueError("A comparison requires at least two sources")
    paths = inventory(comparison_dir, sources, settings, True)
    # These summaries describe extracted stores alongside their diagnostics.
    paths["source_summary"] = {
        f"dataset_comparison_{suffix}": str(
            Path(comparison_dir) / f"dataset_comparison.{suffix}"
        )
        for suffix in ("csv", "md")
    }
    return paths
