"""Render the observed water-balance diagram from retained tables."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blueearth_cst.plotting import fundamentals
from blueearth_cst.plotting.figure_titles import (
    figure_engine_dir,
    figure_inventory_paths,
    write_figure_titles,
)
from blueearth_cst.plotting.location_labels import map_display_label
from blueearth_cst.plotting.prepared_charts import TREND_DASH
from blueearth_cst.plotting.preset import active, use_preset
from blueearth_cst.shared import plot_style
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.snake_utils import save_figure

FILENAME = "budyko_observed_runoff_coefficient_stations.png"
SINGLE_SERIES_COLOR = "#000000"


def render(
    summary_path: str | Path,
    plots_dir: str | Path,
    annotation: str = "none",
    forcing_label: str | None = None,
) -> None:
    """Draw retained ratios with water/energy limits and explicit exclusions.

    The figure joins any caption inventory already in ``plots_dir``: the WF1
    evaluation rule writes it beside the basin-average figures it owns.
    """
    plots = Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path, keep_default_na=False)
    available = summary.loc[summary.available.eq(True)]
    forcing = forcing_label or "the selected historical forcing"
    title = f"Observed catchment water balance — {forcing} precipitation and PET"
    caption = (
        f"Observed runoff coefficient (Q/P) against aridity index (PET/P) per gauged catchment, "
        f"with P and PET from {forcing} over complete water years matched to the discharge record. "
        "Labels give the station and number of years. The dashed line is the water limit (Q/P = 1) "
        "and the dotted line the energy limit (Q/P = 1 − PET/P), both assuming long-term equilibrium. "
        "Departures can reflect short records, storage change, transfers, regulation, catchment area "
        "or input errors; the diagram does not rank forcings or test the model."
    )
    with use_preset(CST_PRESET):
        preset = active()
        fig, axes = fundamentals.new_fig(preset.heights["single"], width_mm=preset.widths["panel"])
        ax = axes[0, 0]
        xmax = max(1.2, float(available.aridity_index.max()) * 1.25) if len(available) else 1.2
        # Headroom above the water limit keeps the key clear of the Q/P = 1 line.
        ymax = max(1.3, float(available.runoff_coefficient.max()) * 1.3) if len(available) else 1.3
        reference = {"color": preset.colors["muted"], "lw": preset.lines["ref"]}
        ax.plot([0, xmax], [1, 1], ls=TREND_DASH, label="Water limit", **reference)
        ax.plot([0, 1], [1, 0], ls=":", label="Energy limit", **reference)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        single = len(available) == 1
        previous_label_y = -np.inf
        ordered = available.sort_values("runoff_coefficient")
        for index, station in enumerate(ordered.itertuples()):
            ax.scatter(
                station.aridity_index,
                station.runoff_coefficient,
                s=36,
                alpha=0.7,
                color=SINGLE_SERIES_COLOR if single else colors[index % len(colors)],
                edgecolors="0.2",
                linewidths=0.7,
                zorder=3,
            )
            label_y = max(station.runoff_coefficient, previous_label_y + 0.06 * ymax)
            previous_label_y = label_y
            ax.annotate(
                f"{map_display_label(str(station.station_name))} ({station.year_count} y)",
                (station.aridity_index, station.runoff_coefficient),
                xytext=(station.aridity_index + 0.04 * xmax, label_y),
                textcoords="data",
                va="center",
                arrowprops={"arrowstyle": "-", "color": preset.colors["muted"], "lw": 0.4},
                fontsize=preset.fonts["tick"],
                bbox=dict(preset.label_box),
            )
        unavailable = summary.loc[~summary.available.eq(True)]
        if len(unavailable):
            ax.text(
                0.98,
                0.04,
                "Unavailable:\n"
                + "\n".join(
                    f"{map_display_label(str(r.station_name))}: {r.reason}"
                    for r in unavailable.itertuples()
                ),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=preset.fonts["tick"],
                bbox=dict(preset.label_box),
            )
        ax.set(
            xlim=(0, xmax),
            ylim=(0, ymax),
            xlabel="Aridity index PET/P (–)",
            ylabel="Runoff coefficient Q/P (–)",
        )
        fundamentals.time_series_grid(ax)
        handles, _ = ax.get_legend_handles_labels()
        fundamentals.inside_legend(fig, ax, handles, lines=handles)
        save_figure(
            str(plots / FILENAME),
            fig=fig,
            annotation=annotation,
            title=title,
            caption=caption,
            dpi=plot_style.RASTER_DPI,
        )
        plt.close(fig)
    _merge_inventory(plots, FILENAME, {"title": title, "caption": caption})


def _merge_inventory(plots: Path, name: str, record: dict) -> None:
    """Add one figure to the directory's readable and machine inventories."""
    engine = figure_engine_dir(plots)
    engine.mkdir(parents=True, exist_ok=True)
    path = engine / "figure_captions.json"
    inventory = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema_version": 1, "figures": {}}
    )
    inventory["figures"][name] = record
    inventory["produced_paths"] = list(inventory["figures"]) + figure_inventory_paths(plots)
    path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_figure_titles(plots, inventory["figures"])
    (plots / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{value['caption']}" for key, value in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )
