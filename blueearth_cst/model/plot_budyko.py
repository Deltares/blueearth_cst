"""Render the observed water-balance diagram from retained tables."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.fundamentals import add_caption, new_fig
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_preset import CST_PRESET


def render(
    root: str | Path, destination: str | Path | None = None, annotation: str = "none"
) -> None:
    """Draw retained ratios with water/energy bounds and explicit exclusions."""
    root = Path(root)
    destination = Path(destination) if destination else root
    plots, engine = destination / "plots", destination / "_engine" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    engine.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "tables/summary.csv", keep_default_na=False)
    available = summary.loc[summary.available.eq(True)]
    filename = "observed_runoff_coefficient_budyko_stations.png"
    title = "Observed catchment water balance against selected historical forcing"
    caption = (
        "Runoff coefficient Q/P versus aridity PET/P over complete matched water years. "
        "Labels give station and retained year count. Ratios use totals over the same years; "
        "annual.csv retains exclusions and coverage. Q support is user-declared daily mean over UTC days. "
        "Dashed lines show water and energy limits under long-term equilibrium assumptions. "
        "Short or discontinuous records, storage change, transfers, regulation, area and input errors "
        "can explain departures; this diagram supplies no pass/fail or forcing ranking. "
        "The residual 1-Q/P is not measured evapotranspiration."
    )
    with use_preset(CST_PRESET):
        fig, axes = new_fig(height_mm=100, width_mm=127)
        ax = axes[0, 0]
        xmax = (
            max(2.0, float(available.aridity_index.max()) * 1.15)
            if len(available)
            else 2.0
        )
        ymax = (
            max(1.1, float(available.runoff_coefficient.max()) * 1.15)
            if len(available)
            else 1.1
        )
        phi = np.linspace(0, xmax, 300)
        ax.plot(phi, np.ones_like(phi), "--", color="0.4", label="Water limit")
        ax.plot(phi, np.maximum(1 - phi, 0), ":", color="0.4", label="Energy limit")
        previous_label_y = -np.inf
        for station in available.sort_values("runoff_coefficient").itertuples():
            ax.scatter(
                station.aridity_index,
                station.runoff_coefficient,
                s=36,
                alpha=0.7,
                edgecolors="0.2",
            )
            label_y = max(station.runoff_coefficient, previous_label_y + 0.045 * ymax)
            previous_label_y = label_y
            ax.annotate(
                f"{station.station_name} ({station.year_count} y)",
                (station.aridity_index, station.runoff_coefficient),
                xytext=(station.aridity_index + 0.03 * xmax, label_y),
                textcoords="data",
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.4},
                fontsize=6,
            )
        unavailable = summary.loc[~summary.available.eq(True)]
        if len(unavailable):
            ax.text(
                0.98,
                0.95,
                "Unavailable:\n"
                + "\n".join(
                    f"{r.station_name}: {r.reason}" for r in unavailable.itertuples()
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
            )
        ax.set(
            xlim=(0, xmax),
            ylim=(0, ymax),
            xlabel="Aridity index PET/P (–)",
            ylabel="Runoff coefficient Q/P (–)",
        )
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.15)
        if annotation == "caption":
            add_caption(fig, caption)
        elif annotation == "title":
            fig.suptitle(title, fontsize=7)
        export_figure(fig, plots / filename, dpi=300, facecolor="white")
        plt.close(fig)
    figures = {filename: {"title": title, "caption": caption}}
    (plots / "figure_captions.md").write_text(
        f"# Figure captions\n\n## {filename}\n\n{caption}\n", encoding="utf-8"
    )
    (engine / "figure_captions.json").write_text(
        json.dumps(
            {"schema_version": 1, "path_base": "../../plots", "figures": figures},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_figure_titles(plots, figures, manifest_path=engine / "figure_titles.json")
