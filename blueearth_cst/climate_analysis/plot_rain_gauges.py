"""Render station comparisons entirely from retained evaluation products."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis.diagnostics import month_order
from blueearth_cst.climate_analysis.figure_naming import figure_filename
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.fundamentals import add_caption, new_fig
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_colors import DATASET_COLORS, DEFAULT_COLOR_CYCLE
from blueearth_cst.shared.plot_preset import CST_PRESET


def render(
    root: str | Path, destination: str | Path | None = None, annotation: str = "none"
) -> None:
    """Draw all four figure families, with visible unavailable-data explanations."""
    root = Path(root)
    destination = Path(destination) if destination else root
    plots = destination / "plots"
    engine = destination / "_engine"
    plots.mkdir(parents=True, exist_ok=True)
    engine.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(
        root / "tables/stations.csv",
        dtype={"id": str, "token": str},
        keep_default_na=False,
    ).drop_duplicates("id")
    monthly = pd.read_csv(
        root / "tables/monthly.csv",
        dtype={"id": str},
        keep_default_na=False,
        na_values={"total_mm": [""]},
    )
    annual = pd.read_csv(
        root / "tables/annual.csv",
        dtype={"id": str},
        keep_default_na=False,
        na_values={"total_mm": [""]},
    )
    record = json.loads((root / "_engine/evaluation.json").read_text())
    captions = {}
    with (
        xr.open_dataset(root / "_engine/station_daily.nc") as ds,
        use_preset(CST_PRESET),
    ):
        labels = list(map(str, ds.source.values))
        colors = {
            label: DATASET_COLORS.get(
                label, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)]
            )
            for i, label in enumerate(labels)
        }
        colors["observed"] = "#333333"
        display = {
            label: {
                "era5": "ERA5",
                "chirps": "CHIRPS",
                "chirps_global": "CHIRPS global",
                "observed": "Observed",
            }.get(label, label)
            for label in labels
        }
        for station in stations.itertuples():
            available = station.id in ds.station.values
            frame = (
                ds.precip.sel(station=station.id).to_pandas()
                if available
                else pd.DataFrame(columns=labels)
            )
            joint = frame.notna().all(axis=1)
            paired = frame.where(joint)
            months = monthly[monthly.id == station.id]
            years = annual[annual.id == station.id]
            for form in (
                "annual_ts",
                "monthly_clim_line",
                "daily_agreement",
            ):
                fig, ax = new_fig(75, width_mm=150)
                ax = np.asarray(ax).ravel()[0]
                filename = figure_filename(
                    "comparison", "precip", form, f"station_{station.token}_avg"
                )
                if not available or not joint.any():
                    ax.text(
                        0.5,
                        0.5,
                        "No common valid days; see exclusions and coverage tables",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        wrap=True,
                    )
                    ax.set_axis_off()
                elif form in ("annual_ts", "monthly_clim_line"):
                    for i, label in enumerate(labels):
                        if form == "annual_ts":
                            data = (
                                years[years.source == label]
                                .set_index("reporting_year")
                                .total_mm
                            )
                        else:
                            data = (
                                months[months.source == label]
                                .groupby("calendar_month")
                                .total_mm.mean()
                                .reindex(month_order(record["water_year_start_month"]))
                            )
                        ax.plot(
                            data.index if form == "annual_ts" else range(12),
                            data.values,
                            marker="o",
                            ms=2,
                            linestyle=("--", "-", ":", "-.")[i % 4],
                            color=colors[label],
                            label=display[label],
                        )
                    ax.set(
                        ylabel="mm",
                        xlabel="Reporting year (start)"
                        if form == "annual_ts"
                        else "Calendar month",
                    )
                    if form == "annual_ts":
                        ticks = sorted(years.reporting_year.unique())
                        ax.set_xlim(ticks[0] - 0.5, ticks[-1] + 0.5)
                        ax.set_xticks(ticks[:: max(1, len(ticks) // 10)])
                    else:
                        order = month_order(record["water_year_start_month"])
                        ax.set_xticks(
                            range(12),
                            [
                                pd.Timestamp(2000, month, 1).strftime("%b")
                                for month in order
                            ],
                        )
                    if (
                        not (years if form == "annual_ts" else months)
                        .total_mm.notna()
                        .any()
                    ):
                        ax.text(
                            0.5,
                            0.5,
                            "No complete matched periods",
                            ha="center",
                            transform=ax.transAxes,
                        )
                else:
                    for i, label in enumerate(labels[1:]):
                        ax.scatter(
                            paired.observed,
                            paired[label],
                            s=8,
                            alpha=0.5,
                            edgecolors="0.2",
                            linewidths=0.25,
                            marker=("o", "s", "^")[i % 3],
                            color=colors[label],
                            label=display[label],
                        )
                    upper = float(np.nanmax(paired.to_numpy()))
                    ax.plot(
                        [0, upper],
                        [0, upper],
                        color="0.4",
                        linestyle="--",
                        linewidth=0.6,
                    )
                    ax.set(xlabel="Observed (mm/day)", ylabel="Native cell (mm/day)")
                    limit = max(upper * 1.05, 1.0)
                    ax.set(xlim=(0, limit), ylim=(0, limit))
                    ax.set_aspect("equal", adjustable="box")
                if ax.has_data():
                    ax.legend(fontsize=6)
                    ax.grid(axis="y", alpha=0.15)
                title = f"{form.replace('_', ' ')} — station {station.id} ({record['comparison_period']['start']}–{record['comparison_period']['end']})"
                caption = (
                    f"Station {station.id}: {form.replace('_', ' ')} for {', '.join(labels)}. "
                    f"Daily agreement uses {int(joint.sum())} common valid days. "
                    "Midnight UTC interval-end labels are attributed to the preceding day. "
                    "Totals require complete matched months or reporting years; missing totals remain unavailable. "
                    "Monthly climatology averages complete matched months; counts are retained in monthly.csv. "
                    "Gridded precipitation represents the native source cell, not a gauge average. "
                    "Point/grid support differs and products may incorporate these gauges; this is diagnostic comparison."
                )
                if annotation == "caption":
                    add_caption(fig, caption)
                elif annotation == "title":
                    fig.suptitle(title, fontsize=7)
                export_figure(fig, plots / filename, dpi=300, facecolor="white")
                plt.close(fig)
                captions[filename] = {"title": title, "caption": caption}
    (engine / "figure_captions.json").write_text(
        json.dumps(
            {"schema_version": 1, "path_base": "../plots", "figures": captions},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (plots / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{value['caption']}" for key, value in captions.items()
        )
        + "\n",
        encoding="utf-8",
    )
    write_figure_titles(plots, captions, manifest_path=engine / "figure_titles.json")


if __name__ == "__main__":
    sm = globals()["snakemake"]
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        render(sm.params.root, annotation=sm.params.annotation)
