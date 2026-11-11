"""The one declared/written inventory for canonical WF0 diagnostics."""

from pathlib import Path

from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.figure_naming import (
    annual_anomaly_filename,
    figure_filename,
)


def parse_figure_settings(block: dict) -> dict[str, bool]:
    """Resolve presentation switches independently of numerical diagnostics."""
    from blueearth_cst.shared.workflow_plotting import workflow_plotting

    return workflow_plotting("analyze_climate", block)


def rain_gauge_outputs(root, tokens):
    """Explicit station evaluation compute and figure inventory."""
    root = Path(root)
    compute = {
        name: str(root / "tables" / f"{name}.csv")
        for name in (
            "stations",
            "exclusions",
            "coverage",
            "agreement",
            "monthly",
            "annual",
        )
    }
    compute.update(
        daily=str(root / "_engine/station_daily.nc"),
        metadata=str(root / "_engine/evaluation.json"),
    )
    render = {
        f"{token}_{form}": str(
            root
            / "plots"
            / figure_filename("comparison", "precip", form, f"station_{token}_avg")
        )
        for token in tokens
        for form in (
            "annual_ts",
            "monthly_clim_line",
            "daily_agreement",
        )
    }
    render.update(
        titles=str(root / "plots/figure_titles.md"),
        captions=str(root / "plots/figure_captions.md"),
        titles_json=str(root / "_engine/figure_titles.json"),
        captions_json=str(root / "_engine/figure_captions.json"),
    )
    return {"compute": compute, "render": render}


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
COMPARISON_RESULT_TABLES = {
    "annual_indices",
    "monthly_values",
    "precip_climatology",
    "temp_climatology",
    "drought_events",
    "trends",
    "agreement",
}
FORMS = {
    "seasonal": ("precip", "monthly_clim_band"),
    "temperature": ("temp", "monthly_clim_band"),
    "timing": ("precip", "annual_timing"),
    "anomaly": ("precip", "monthly_anomaly"),
    "spi": ("precip", "monthly_spi"),
    "drought-events": ("precip", "spi_events"),
    "extreme-intensity": ("precip", "annual_intensity_ts"),
    "extreme-Wet days": ("precip", "annual_wet_days_ts"),
    "spell": ("precip", "dry_spell_exceedance"),
    "acf": ("precip", "monthly_anomaly_acf"),
    "slopes": ("precip", "annual_trend_interval"),
    "series": ("precip", "annual_trend_ts"),
    "pt": ("precip_temp", "seasonal_anomaly"),
    "sensitivity": ("precip", "annual_trend_sensitivity"),
}


def inventory(root, sources, settings, comparison=False, comparison_sources=()):
    """Named compute/render files; unknown subbasin IDs use optional directories."""
    root = Path(root)
    scope = "comparison" if comparison else sources[0]
    temp = sum(carries_temp(s) for s in sources)
    forms = dict(FORMS)
    if temp < 1:
        forms.pop("temperature")
    binding = settings["pt_temperature_source"]
    if (comparison and (binding not in sources or not carries_temp(binding))) or (
        not comparison and not temp
    ):
        forms.pop("pt")
    if comparison and not settings.get("comparison_figures", True):
        forms.clear()
    if (
        not comparison
        and len(comparison_sources) > 1
        and not settings.get("source_figures", False)
    ):
        compared = inventory(root, comparison_sources, settings, comparison=True)
        duplicated = set(compared["figures"])
        # A non-bound temperature carrier's own P/T relationship is not shown
        # by a comparison that pairs every precipitation source with the binding.
        if sources[0] != binding:
            duplicated.discard("pt")
        forms = {key: value for key, value in forms.items() if key not in duplicated}
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
        if settings.get("climate_indices", {}).get("enabled", False):
            compute["climate_indices"] = str(root / "_engine" / "climate_indices.json")
    render = {
        "titles_json": str(root / "_engine" / "plots" / "figure_titles.json"),
        "titles_md": str(root / "plots" / "figure_titles.md"),
        "captions_json": str(root / "_engine" / "plots" / "figure_captions.json"),
        "captions_md": str(root / "plots" / "figure_captions.md"),
    }
    render.update({f"figure_{k}": v for k, v in figures.items()})
    paths = {
        "root": str(root),
        "compute": compute,
        "render": render,
        "figures": figures,
    }
    if not comparison and settings.get("annual_spatial_anomalies", False):
        from blueearth_cst.climate_analysis.annual_spatial_anomalies import (
            anomaly_paths,
            page_years,
        )

        paths["annual_spatial"] = anomaly_paths(root)
        pages = page_years(settings)
        for page, _ in enumerate(pages, 1):
            for variable in ("precip", "temp") if temp else ("precip",):
                key = f"annual-map-{variable}-{page:03d}"
                filename = str(
                    root
                    / "plots"
                    / annual_anomaly_filename(scope, variable, page, len(pages))
                )
                figures[key] = filename
                render[f"figure_{key}"] = filename
    if not comparison and settings.get("climate_indices", {}).get("enabled", False):
        # Available basin sheets and internal inventories have separate owned directories.
        render["index_products"] = str(root / "plots" / "indices")
        render["index_metadata"] = str(root / "_engine" / "plots" / "indices")
        paths["climate_indices_table"] = str(root / "tables" / "climate_indices.csv")
        paths["climate_index_plot_data"] = str(
            root / "_engine" / "wet_spell_exceedance.csv"
        )
    return paths


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


def diagnostic_source_outputs(store_dir, source, settings, comparison_sources=()):
    return inventory(
        store_dir, [source], settings, comparison_sources=comparison_sources
    )


def diagnostic_comparison_outputs(comparison_dir, sources, settings):
    if len(sources) < 2:
        raise ValueError("A comparison requires at least two sources")
    paths = inventory(comparison_dir, sources, settings, True)
    root = Path(comparison_dir)
    # Comparison views share one root; source-store layouts remain unchanged.
    for section, folder in (
        ("compute", "tables"),
        ("render", "plots"),
        ("figures", "plots"),
    ):
        for key, value in paths[section].items():
            relative = Path(value).relative_to(root)
            if relative.parts[0] == folder:
                paths[section][key] = str(
                    root / folder / "basin-average" / Path(*relative.parts[1:])
                )
    for key in ("titles_json", "captions_json"):
        paths["render"][key] = str(
            root / "_engine/plots/basin-average" / Path(paths["render"][key]).name
        )
    for key, path in paths["compute"].items():
        if key in TABLES or key == "agreement_md":
            if key not in COMPARISON_RESULT_TABLES:
                paths["compute"][key] = str(
                    root / "_engine/basin-average" / Path(path).name
                )
    # These summaries describe extracted stores alongside their diagnostics.
    paths["source_summary"] = {
        f"dataset_comparison_{suffix}": str(
            Path(comparison_dir) / f"dataset_comparison.{suffix}"
        )
        for suffix in ("csv", "md")
    }
    return paths


def retained_captions_path(root):
    """Prefer consolidated metadata while still reading existing retained runs."""
    root = Path(root)
    current = root / "_engine/plots/figure_captions.json"
    return current if current.is_file() else root / "plots/_engine/figure_captions.json"


def location_comparison_outputs(root):
    """Dynamic location inventories are bounded by declared output directories."""
    root = Path(root)
    return {
        "compute": {
            "locations": str(root / "locations.csv"),
            "tables": str(root / "tables/locations"),
            "daily": str(root / "_engine/locations/daily.nc"),
            "metadata": str(root / "_engine/locations/evaluation.json"),
            "exclusions": str(root / "_engine/locations/exclusions.csv"),
        },
        "render": {
            "plots": str(root / "plots/locations"),
            "manifests": str(root / "_engine/locations/figures"),
        },
    }
