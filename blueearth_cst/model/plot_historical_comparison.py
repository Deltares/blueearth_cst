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

from blueearth_cst.climate_analysis.diagnostic_figures import (
    SOURCE_STYLE,
    dataset_styles,
)
from blueearth_cst.model.historical_comparison import read_comparison
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.prepared_charts import PreparedLine, series_panel
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_evaluation import (
    COLOR_OBSERVED,
    DASH_OBSERVED,
    LINE_WIDTH,
)
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.plot_style import RASTER_DPI


def render(
    root: Path, destination: Path | None = None, annotation: str = "none"
) -> None:
    """Draw every station and support from hash-verified comparison tables."""
    root = Path(root)
    destination = Path(destination) if destination else root
    metadata, tables = read_comparison(root)
    plots, engine = destination / "plots", destination / "_engine" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    engine.mkdir(parents=True, exist_ok=True)
    # Canonical dataset colours and markers (WF0's source registry); a lone run
    # is black. Observations keep the WF1 evaluation style: blue, dashed.
    styles = {
        name: (color, "-", marker)
        for name, (color, marker) in dataset_styles(metadata["runs"]).items()
    }
    styles["__observed__"] = (COLOR_OBSERVED, DASH_OBSERVED, "")
    labels = {
        name: name.upper() if name.lower() in SOURCE_STYLE else name
        for name in metadata["runs"]
    }
    labels["__observed__"] = "Observed"
    start_month = metadata["water_year_start_month"]
    months = list(range(start_month, 13)) + list(range(1, start_month))
    figures = {}
    coverage_table = tables["coverage"].set_index("wflow_id")
    with use_preset(CST_PRESET):
        for station in tables["stations"].itertuples():
            station_label = str(getattr(station, "station_name", station.wflow_id))
            # One figure set per station: observation-paired where the station
            # has observations, otherwise simulations on their common support.
            observed_here = metadata["observations_supplied"] and bool(
                coverage_table.loc[station.wflow_id].observations_available
            )
            for support in (
                ["observation_paired"] if observed_here else ["simulated_common"]
            ):
                names = metadata["runs"] + (
                    ["__observed__"] if support == "observation_paired" else []
                )
                for table_name, form, xlabel in (
                    ("daily", "daily", "Year"),
                    ("seasonal", "monthly_cycle", ""),
                    ("flow_duration", "flow_duration", "Exceedance probability (%)"),
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
                        observed = name == "__observed__"
                        lines.append(
                            PreparedLine(
                                labels[name],
                                x,
                                values,
                                color,
                                linestyle,
                                # Dense daily and duration curves are plain
                                # lines, as on the station sheets; only the
                                # 12-point seasonal curve carries markers.
                                marker if table_name == "seasonal" else "",
                                LINE_WIDTH if observed else None,
                            )
                        )
                    zero_share = {
                        line.label: float(np.mean(np.asarray(line.y, float) <= 0))
                        for line in lines
                        if len(line.y)
                    }
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
                        ylabel="m³/s",
                        ticks=ticks,
                        xlim=limits,
                        log_y=table_name == "flow_duration",
                    )
                    # figure-naming.md: <dataset>_<subject>_<diagnostic>_<location>;
                    # the support (paired or common) is stated in the caption.
                    filename = (
                        f"comparison_discharge_{form}_station{station.wflow_id}.png"
                    )
                    description = {
                        "daily": "daily hydrograph",
                        "seasonal": "mean monthly discharge",
                        "flow_duration": "flow-duration curve",
                    }[table_name]
                    title = f"{station_label} ({station.wflow_id}): {description}"
                    coverage = coverage_table.loc[station.wflow_id]
                    count = int(
                        coverage[
                            "observation_paired_days"
                            if support == "observation_paired"
                            else "simulated_common_days"
                        ]
                    )
                    start, end = metadata["requested_start"], metadata["requested_end"]
                    period = (
                        f"{start[:4]}–{end[:4]}"
                        if start[5:] == "01-01" and end[5:] == "12-31"
                        else f"{start} to {end}"
                    )
                    runs = [labels[name] for name in metadata["runs"]]
                    forcing = (
                        runs[0]
                        if len(runs) == 1
                        else ", ".join(runs[:-1]) + " and " + runs[-1]
                    )
                    lead = {
                        "daily": "Daily discharge",
                        "seasonal": "Mean monthly discharge",
                        "flow_duration": "Flow-duration curves of daily discharge",
                    }[table_name]
                    paired = support == "observation_paired"
                    caption = (
                        f"{lead} at {station_label} ({station.wflow_id}), {period}: "
                        f"Wflow simulations forced by {forcing}"
                        + (" against observations" if paired else "")
                        + f", on days with all series available (n = {count})."
                    )
                    if table_name == "flow_duration":
                        zeros = [
                            f"{label} {share:.0%}"
                            for label, share in zero_share.items()
                            if share > 0
                        ]
                        caption += " Log discharge axis; exceedance is rank/(n + 1)."
                        if zeros:
                            caption += (
                                f" Zero flows not drawn ({', '.join(zeros)} of days)."
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
            {"schema_version": 1, "path_base": "../../plots", "figures": figures},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_figure_titles(plots, figures, manifest_path=engine / "figure_titles.json")
