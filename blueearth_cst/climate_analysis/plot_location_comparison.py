"""Location comparisons inspired by CRIDA annual overlays and monthly agreement panels."""

import json
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis.diagnostic_figures import source_styles
from blueearth_cst.climate_analysis.diagnostics import month_order
from blueearth_cst.climate_analysis.figure_naming import (
    figure_filename,
    location_name_token,
)
from blueearth_cst.climate_analysis.rain_gauges import matched_tables
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import figure_engine_dir, write_figure_titles
from blueearth_cst.plotting.fundamentals import (
    new_fig,
    time_series_grid,
    time_series_style,
    year_ticks,
)
from blueearth_cst.plotting.panel_layout import arrange_panels
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_preset import CST_PRESET


def location_table(root, name: str, location_id: str, variable: str | None = None):
    """One location's rows of a long table, or its per-location file before layout 3."""
    root = Path(root)
    current = root / "tables/locations" / f"{name}.csv"
    if current.is_file():
        frame = pd.read_csv(current, dtype={"location_id": str})
        rows = frame.location_id == location_id
        if variable is not None and "variable" in frame:
            rows &= frame.variable == variable
        return frame.loc[rows].drop(
            columns=["location_id", "variable"], errors="ignore"
        )
    prefix = "temp_" if variable == "temp" else ""
    legacy = pd.read_csv(
        root / "tables/locations" / location_id / f"{prefix}{name}.csv"
    )
    return legacy.rename(columns={"month": "period", "reporting_year": "period"})


def _monthly_metrics(root, location_id: str, variable: str) -> pd.DataFrame:
    """Monthly agreement metrics of one location and variable, either layout."""
    root = Path(root)
    if (root / "tables/locations/agreement.csv").is_file():
        frame = location_table(root, "agreement", location_id, variable)
        return frame.loc[frame.scope == "monthly"].drop(columns="scope")
    prefix = "temp_" if variable == "temp" else ""
    return pd.read_csv(
        root / "tables/locations" / location_id / f"{prefix}monthly_agreement.csv"
    )


def agreement_figure(
    table: pd.DataFrame,
    value: str,
    styles: dict,
    labels: dict,
    unit: str,
    variable: str,
    metrics: pd.DataFrame,
):
    """One measured square panel per grid product, with shared limits and a 1:1 line."""
    wide = table.pivot(index="period", columns="source", values=value)
    sources = [source for source in styles if source != "observed" and source in wide]
    metrics = metrics.set_index("source")
    finite = wide.to_numpy()[np.isfinite(wide.to_numpy())]
    low = (
        min(float(finite.min()), 0)
        if len(finite) and variable == "precip"
        else float(finite.min())
        if len(finite)
        else 0
    )
    high = float(finite.max()) if len(finite) else 1
    padding = max((high - low) * 0.05, 1)
    limits = (0 if variable == "precip" else low - padding, high + padding)

    def draw(ax, source):
        paired = wide[["observed", source]].dropna()
        color, marker = styles[source]
        ax.scatter(
            paired.observed,
            paired[source],
            color=color,
            marker=marker,
            s=12,
            alpha=0.65,
            linewidths=0.25,
            edgecolors="0.2",
        )
        ax.plot(limits, limits, color="0.5", linestyle="--", linewidth=0.7)
        r = metrics.loc[source, "pearson_r"]
        note = f"r = {r:.2f}" if np.isfinite(r) else "r unavailable"
        ax.text(
            0.04,
            0.96,
            f"{note}\nn = {len(paired)} months",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        if paired.empty:
            ax.text(
                0.5,
                0.45,
                "No complete paired months",
                ha="center",
                transform=ax.transAxes,
                fontsize=7,
            )
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel=f"Observed ({unit})",
            ylabel=f"{labels[source]} ({unit})",
            title=labels[source],
        )
        ax.grid(alpha=0.15)

    fig, _, _ = arrange_panels(
        [partial(draw, source=source) for source in sources],
        panel_width_mm=65,
        panel_height_mm=65,
        preferred_columns=(3, 2),
        max_width_mm=270,
        dpi=300,
    )
    return fig


def series_figure(
    table: pd.DataFrame,
    value: str,
    labels: dict,
    styles: dict,
    m0: int,
    form: str,
    variable: str,
):
    """Annual overlays or monthly climatologies, retaining unavailable gaps."""
    fig, axes = new_fig(80, width_mm=150)
    ax = axes[0, 0]
    monthly = form == "monthly_clim_line"
    order = month_order(m0)
    for source in labels:
        selected = table[table.source == source]
        if monthly:
            data = selected.groupby("calendar_month")[value].mean().reindex(order)
            x = range(12)
        else:
            data = selected.set_index("period")[value]
            x = pd.to_numeric(data.index)
        color, marker = styles[source]
        ax.plot(
            x,
            data,
            label=labels[source],
            marker=marker,
            **time_series_style(source, color),
        )
    if monthly:
        ax.set_xticks(
            range(12), [pd.Timestamp(2000, month, 1).strftime("%b") for month in order]
        )
    else:
        years = np.sort(pd.to_numeric(table.period.unique()))
        year_ticks(ax, years)
    ax.set(
        ylabel=("mm/month" if monthly else "mm/year") if variable == "precip" else "°C",
        xlabel="Calendar month" if monthly else "Year",
    )
    if not table[value].notna().any():
        ax.text(0.5, 0.5, "No complete periods", ha="center", transform=ax.transAxes)
    ax.legend(fontsize=7)
    time_series_grid(ax)
    return fig


def render(
    root: str | Path, destination: str | Path | None = None, annotation: str = "none"
) -> dict:
    """Render each retained location independently, with local captions and global manifests."""
    root = Path(root)
    destination = Path(destination) if destination else root
    record = json.loads(
        (root / "_engine/locations/evaluation.json").read_text(encoding="utf-8")
    )
    registry = pd.read_csv(
        root / "locations.csv",
        dtype={"id": str, "gauge_id": str},
        keep_default_na=False,
    )
    m0 = record["water_year_start_month"]
    records = {}
    with (
        xr.open_dataset(root / "_engine/locations/daily.nc") as ds,
        use_preset(CST_PRESET),
    ):
        name_counts = registry.name.map(location_name_token).value_counts()
        for location in registry.itertuples():
            name_token = location_name_token(location.name)
            meaningful_name = (
                location.name
                if name_token
                and not str(location.kind).startswith("automatic")
                and not name_token.startswith("auto_")
                and name_counts[name_token] == 1
                else None
            )
            target = destination / "plots/locations" / location.id
            target.mkdir(parents=True, exist_ok=True)
            local = {}
            for variable in ("precip", "temp"):
                sources = (
                    record["sources"]
                    if variable == "precip"
                    else record["temperature_sources"]
                )
                if not sources:
                    continue
                has_observed = bool(getattr(location, f"observed_{variable}"))
                names = [*(["observed"] if has_observed else []), *sources]
                labels = {
                    name: {
                        "observed": "Observed",
                        "chirps": "CHIRPS",
                        "chirps_global": "CHIRPS global",
                    }.get(name, name.upper())
                    for name in names
                }
                styles = source_styles(sources)
                if variable == "temp":
                    temperature_labels = record.get("temperature_labels", {})
                    native_styles = source_styles(
                        [temperature_labels.get(source, source) for source in sources]
                    )
                    for source in sources:
                        carrier = temperature_labels.get(source, source)
                        labels[source] = (
                            carrier.upper()
                            if carrier == source
                            else f"{carrier.upper()} (on {source.upper()} grid)"
                        )
                        styles[source] = native_styles[carrier]
                styles["observed"] = ("#333333", "o")
                monthly = location_table(root, "monthly", location.id, variable)
                value = "total_mm" if variable == "precip" else "mean_degC"
                # Overlays retain each source's full complete-year record.
                # Agreement tables and monthly climatologies keep paired support.
                if variable == "temp":
                    # Keep the HydroMT sampling imports out of drawing-only use.
                    from blueearth_cst.climate_analysis.location_comparison import (
                        temperature_tables,
                    )

                frame = ds[variable].sel(location=location.id, source=names).to_pandas()
                annual = pd.concat(
                    [
                        matched_tables(frame[[source]], m0, observed=False)["annual"]
                        if variable == "precip"
                        else temperature_tables(frame[[source]], m0)["annual"]
                        for source in names
                    ],
                    ignore_index=True,
                )
                if variable == "precip":
                    annual = annual.rename(columns={"reporting_year": "period"})
                forms = ["monthly_clim_line", "annual_ts"]
                if has_observed:
                    forms.append("monthly_agreement")
                for form in forms:
                    if form == "monthly_agreement":
                        metrics = _monthly_metrics(root, location.id, variable)
                        fig = agreement_figure(
                            monthly,
                            value,
                            styles,
                            labels,
                            "mm/month" if variable == "precip" else "°C",
                            variable,
                            metrics,
                        )
                    else:
                        fig = series_figure(
                            monthly if form == "monthly_clim_line" else annual,
                            value,
                            labels,
                            styles,
                            m0,
                            form,
                            variable,
                        )
                    filename = figure_filename(
                        "comparison",
                        variable,
                        form,
                        f"location_{location.id.removeprefix('location_')}_point",
                        location_name=meaningful_name,
                    )
                    title = (
                        f"{variable.title()} {form.replace('_', ' ')} — {location.name} "
                        f"[{location.id}] ({record['comparison_period']['start']}–{record['comparison_period']['end']})"
                    )
                    support_note = (
                        "No infill: each source shows its own complete reporting years; "
                        "missing or incomplete years remain gaps. Retained agreement tables "
                        "continue to use matched support. "
                        if form == "annual_ts"
                        else "No infill: totals and temperature means require complete matched months or reporting years. "
                        "Climatologies average the same complete months across series. "
                    )
                    caption = (
                        f"{title}. Sources: {', '.join(labels.values())}. "
                        "Retained grid cells sampled at original location coordinates; point/grid support differs. "
                        "Midnight UTC interval ends are attributed to the preceding day. "
                        f"{support_note}"
                        "Comparison diagnoses consistency; correlation alone does not establish accuracy. "
                        "Gridded products may incorporate gauge observations."
                    )
                    if form == "monthly_agreement":
                        caption += " Each panel shows complete monthly pairs, Pearson r and n; the dashed line is 1:1. Shared axis limits apply to every dataset."
                    if variable == "temp" and any(
                        record.get("temperature_labels", {}).get(source, source)
                        != source
                        for source in sources
                    ):
                        caption += " Companion temperature is sampled on the retained extraction grid after the configured forcing preparation, rather than its original native grid."
                    methods_detail = caption
                    caption = f"{title}. Sources: {', '.join(labels.values())}. " + (
                        "Points show matched monthly values; the dashed line denotes agreement."
                        if form == "monthly_agreement"
                        else "Gaps indicate incomplete reporting years."
                        if form == "annual_ts"
                        else "Monthly means are calculated from complete matched months."
                    )
                    export_figure(
                        fig,
                        target / filename,
                        dpi=300,
                        annotation=annotation,
                        title=title,
                        caption=caption,
                        facecolor="white",
                    )
                    plt.close(fig)
                    key = (target / filename).relative_to(destination).as_posix()
                    local[key] = {
                        "title": title,
                        "caption": caption,
                        "methods_detail": methods_detail,
                        "location_id": location.id,
                        "gauge_id": location.gauge_id,
                        "variable": variable,
                        "sources": names,
                        "support": "retained grid cell at point",
                    }
                    if form == "monthly_agreement":
                        local[key]["monthly_agreement"] = json.loads(
                            metrics.to_json(orient="records")
                        )
            records.update(local)
            (target / "figure_captions.md").write_text(
                "# Figure captions\n\n"
                + "\n\n".join(
                    f"## {key}\n\n{item['caption']}" for key, item in local.items()
                )
                + "\n",
                encoding="utf-8",
            )
            write_figure_titles(
                target,
                local,
                manifest_path=figure_engine_dir(target) / "figure_titles.json",
            )
    engine = figure_engine_dir(destination / "plots/locations")
    engine.mkdir(parents=True, exist_ok=True)
    (engine / "figure_captions.json").write_text(
        json.dumps({"schema_version": 1, "figures": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    (engine / "figure_titles.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "figures": {
                    key: {"title": item["title"]} for key, item in records.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        render(sm.params.root, annotation=sm.params.annotation)
