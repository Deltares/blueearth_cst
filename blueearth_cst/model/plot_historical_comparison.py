"""Render historical comparison tables with the shared prepared plotting API."""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blueearth_cst.model.historical_comparison import read_comparison
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.prepared_charts import PreparedLine, series_panel
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_colors import DEFAULT_COLOR_CYCLE, EVALUATION_COLORS
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.plot_style import RASTER_DPI


def render(
    root: Path, destination: Path | None = None, annotation: str = "none"
) -> None:
    """Draw every station and support from hash-verified retained tables."""
    root = Path(root)
    destination = Path(destination) if destination else root
    metadata, tables = read_comparison(root)
    plots, engine = destination / "plots", destination / "_engine"
    plots.mkdir(parents=True, exist_ok=True)
    engine.mkdir(parents=True, exist_ok=True)
    colors = [
        color for color in DEFAULT_COLOR_CYCLE if color != EVALUATION_COLORS["observed"]
    ]
    markers = ("", "o", "s", "^", "D", "v", "+", "x")
    styles = {
        name: (
            colors[i % len(colors)],
            ("-", "--", ":", "-.")[i % 4],
            markers[i % len(markers)],
        )
        for i, name in enumerate(metadata["runs"])
    }
    styles["__observed__"] = (EVALUATION_COLORS["observed"], "-", "")
    start_month = metadata["water_year_start_month"]
    months = list(range(start_month, 13)) + list(range(1, start_month))
    figures = {}
    supports = ["simulated_common"] + (
        ["observation_paired"] if metadata["observations_supplied"] else []
    )
    with use_preset(CST_PRESET):
        for station in tables["stations"].itertuples():
            station_label = str(getattr(station, "station_name", station.wflow_id))
            for support in supports:
                names = metadata["runs"] + (
                    ["__observed__"] if support == "observation_paired" else []
                )
                for table_name, form, xlabel in (
                    ("daily", "hydrograph", "Response timestamp"),
                    ("seasonal", "seasonal", "Month"),
                    ("flow_duration", "fdc", "Exceedance probability (%)"),
                ):
                    data = tables[table_name]
                    data = data.loc[
                        data.wflow_id.eq(station.wflow_id) & data.support.eq(support)
                    ]
                    lines = []
                    for name in names:
                        selected = data.loc[data.series.eq(name)]
                        if table_name == "seasonal":
                            values = (
                                selected.set_index("month")
                                .discharge.reindex(months)
                                .to_numpy()
                            )
                            x = np.arange(12)
                        else:
                            values = selected.discharge.to_numpy()
                            x = (
                                pd.to_datetime(selected.time).to_numpy()
                                if table_name == "daily"
                                else selected.exceedance.to_numpy()
                            )
                        color, linestyle, marker = styles[name]
                        lines.append(
                            PreparedLine(
                                "Observed" if name == "__observed__" else name,
                                x,
                                values,
                                color,
                                linestyle,
                                marker,
                            )
                        )
                    ticks = (
                        (np.arange(12), [calendar.month_abbr[m] for m in months])
                        if table_name == "seasonal"
                        else None
                    )
                    limits = (0, 100) if table_name == "flow_duration" else (-0.5, 11.5)
                    if table_name == "daily":
                        lo, hi = (
                            pd.Timestamp(metadata["requested_start"]),
                            pd.Timestamp(metadata["requested_end"]),
                        )
                        limits = (
                            (lo, hi)
                            if lo < hi
                            else (
                                lo - pd.Timedelta(hours=12),
                                hi + pd.Timedelta(hours=12),
                            )
                        )
                    fig = series_panel(
                        lines,
                        xlabel=xlabel,
                        ylabel="Discharge (m³/s)",
                        ticks=ticks,
                        xlim=limits,
                    )
                    filename = (
                        f"comparison_discharge_{form}_{station.wflow_id}_{support}.png"
                    )
                    description = {
                        "daily": "hydrograph",
                        "seasonal": "monthly seasonal curve",
                        "flow_duration": "flow-duration curve",
                    }[table_name]
                    title = f"{station_label} ({station.wflow_id}): historical discharge {description}"
                    support_text = (
                        "All simulations and observations share finite nonnegative daily support."
                        if support == "observation_paired"
                        else "All simulations share finite nonnegative daily support, independently of observations. No skill claim is made."
                    )
                    coverage = (
                        tables["coverage"].set_index("wflow_id").loc[station.wflow_id]
                    )
                    count = int(
                        coverage[
                            "observation_paired_days"
                            if support == "observation_paired"
                            else "simulated_common_days"
                        ]
                    )
                    support_text += f" Usable daily samples: {count}."
                    if (
                        support == "observation_paired"
                        and not coverage.observations_available
                    ):
                        support_text = "Observations are unavailable at this station; paired diagnostics are unavailable."
                    method = {
                        "daily": "Gaps are preserved; no values are filled.",
                        "seasonal": "Seasonal values average available monthly means equally by calendar month; incomplete months are retained.",
                        "flow_duration": "Daily values are sorted descending; exceedance is rank/(count+1). Zero flows are retained on a linear axis.",
                    }[table_name]
                    warmup = (
                        f"Warm-up excludes response dates before {metadata['warmup_exclusive_end']}."
                        if metadata.get("warmup_applied")
                        else "No warm-up exclusion applies to these runs of at most 365 samples."
                    )
                    caption = (
                        f"{title}. {metadata['requested_start']} to {metadata['requested_end']}, inclusive interval-end timestamps. "
                        f"{support_text} {method} {warmup} "
                        "Different styles identify retained runs; their recipes are recorded in inventory.csv."
                    )
                    try:
                        export_figure(
                            fig,
                            plots / filename,
                            annotation=annotation,
                            title=title,
                            caption=caption,
                            dpi=RASTER_DPI,
                            facecolor="white",
                        )
                    finally:
                        plt.close(fig)
                    figures[filename] = {"title": title, "caption": caption}
    (plots / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {name}\n\n{entry['caption']}" for name, entry in figures.items()
        )
        + "\n",
        encoding="utf-8",
    )
    (engine / "figure_captions.json").write_text(
        json.dumps(
            {"schema_version": 1, "path_base": "../plots", "figures": figures}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    write_figure_titles(plots, figures, manifest_path=engine / "figure_titles.json")
