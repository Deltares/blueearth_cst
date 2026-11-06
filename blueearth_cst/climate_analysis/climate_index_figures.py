"""Render climate-index products exclusively from verified retained evidence."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from blueearth_cst.climate_analysis.climate_indices import load_publication
from blueearth_cst.climate_analysis.diagnostic_figures import source_styles
from blueearth_cst.indicators.climate.definitions import INDEX_METADATA
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.fundamentals import new_fig, year_ticks
from blueearth_cst.plotting.prepared_maps import (
    MapFrame,
    MapScale,
    RasterMap,
    draw_raster_map,
)
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.map_reference import (
    drawing_frame,
    field_edges,
    load_reference,
    reference_arguments,
)
from blueearth_cst.shared.plot_preset import CST_PRESET


def _series(data, *, seasonal=False, color=None):
    """Draw all declared indices with unavailable panels retained."""
    columns = 3
    rows = int(np.ceil(len(INDEX_METADATA) / columns))
    fig, axes = new_fig(rows * 37, rows, columns, width_mm=180)
    dates = pd.DatetimeIndex(data.period_start.values)
    years = np.asarray(dates.year)
    season_names = np.array([str(label).split(":")[0] for label in data.period.values])
    for index, (metric, (label, units)) in enumerate(INDEX_METADATA.items()):
        ax = axes.flat[index]
        ax.set_title(textwrap.fill(f"{chr(97 + index)}) {label}", 30), loc="left")
        ax.set_ylabel(units)
        if metric not in data or not np.isfinite(data[metric].values).any():
            ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center")
        elif seasonal:
            for season in dict.fromkeys(season_names):
                selected = season_names == season
                ax.plot(
                    years[selected],
                    data[metric].values[selected],
                    marker="o",
                    label=season,
                )
            if index == 0:
                ax.legend()
        else:
            ax.plot(years, data[metric].values, marker="o", color=color)
        year_ticks(ax, np.unique(years))
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(INDEX_METADATA) :]:
        ax.set_visible(False)
    return fig


def render_indices(
    pointer,
    paths,
    *,
    map_reference=None,
    overlays=None,
    annotation="none",
    reference_record=None,
    map_reference_params=None,
):
    """Write declared maps, series, spell distributions and recoverable captions."""
    folder, manifest = load_publication(pointer)
    with xr.open_dataset(folder / "annual_basin.nc") as opened:
        basin = opened.load()
    with xr.open_dataset(folder / "annual_grid.nc") as opened:
        grid = opened.load()
    captions = {}
    source = manifest["identity"]["source"]
    hue = source_styles([source])[source][0]
    options = manifest["identity"]["settings"]["climate_indices"]
    reference = options["reference_period"]
    window = manifest["identity"]["settings"]["analysis_window"]
    common = (
        f"{source.upper()}; reference calendar years {reference['start']}–{reference['end']}. "
        f"Reporting start years {window['start']}–{window['end']}; "
        f"annual start month {manifest['identity']['start_month']}. "
        f"Minimum reference sample fraction {options['reference_minimum_fraction']:g}. "
        "TX90p/TN90p use in-reference bootstrap; WSDI and heatwaves use fixed thresholds. "
        "Missing values indicate unavailable inputs, reference support or period coverage. "
        "Censored observed durations are lower bounds; short references are exploratory."
    )

    def save(fig, key, title, explanation):
        path = Path(paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        export_figure(
            fig,
            path,
            annotation=annotation,
            title=title,
            caption=explanation + " " + common,
        )
        plt.close(fig)
        captions[path.name] = {
            "title": title,
            "caption": explanation + " " + common,
            "evaluation_id": manifest["evaluation_id"],
        }

    with use_preset(CST_PRESET):
        save(
            _series(basin, color=hue),
            "index_series",
            f"{source.upper()} annual climate indices",
            "Indices of daily equal-weight basin-cell means, not spatial means of cell indices. "
            "Years are labelled by start year. Durations count overlapping days within each period.",
        )
        if "index_seasons" in paths:
            with xr.open_dataset(folder / "seasonal_basin.nc") as opened:
                seasonal = opened.load()
            save(
                _series(seasonal, seasonal=True),
                "index_seasons",
                f"{source.upper()} seasonal climate indices",
                f"Explicit seasons {options['seasons']}; labels use start years. Basin-series calculation.",
            )
        exceedance = pd.read_csv(folder / "wet_spell_exceedance_basin.csv")
        fig, axes = new_fig(65, width_mm=100)
        ax = axes[0, 0]
        if len(exceedance):
            ax.step(
                exceedance.duration,
                exceedance.exceedance_probability,
                where="pre",
                color=hue,
            )
        else:
            ax.text(
                0.5, 0.5, "No complete wet spells", transform=ax.transAxes, ha="center"
            )
        ax.set(xlabel="days", ylabel="P(duration ≥ days)", ylim=(0, 1))
        save(
            fig,
            "index_spells",
            f"{source.upper()} complete wet-spell exceedance",
            "Empirical survival probabilities for complete basin-series wet spells over the retained record. "
            "Gap- and edge-censored segments are excluded; this can bias duration distributions.",
        )
        # Use a two-dimensional index field, never a scalar support counter.
        template = (
            grid["rx1day"]
            if "rx1day" in grid
            else grid["txx"]
            if "txx" in grid
            else grid["tnx"]
        )
        template = template.mean("period").transpose("latitude", "longitude")
        x, y = field_edges(template)
        if reference_record is not None:
            frame = drawing_frame(reference_record)
        elif map_reference is not None:
            frame = drawing_frame(
                load_reference(
                    map_reference, **reference_arguments(map_reference_params)
                )
            )
        else:
            frame = MapFrame(
                (float(x.min()), float(x.max()), float(y.min()), float(y.max())),
                np.linspace(x.min(), x.max(), 3),
                np.linspace(y.min(), y.max(), 3),
                1 / np.cos(np.deg2rad(float(template.latitude.mean()))),
            )
        cmap = LinearSegmentedColormap.from_list("source_quantity", ["#f5f5f5", hue])
        for metric, (label, units) in INDEX_METADATA.items():
            field = (
                grid[metric].mean("period").transpose("latitude", "longitude")
                if metric in grid
                else xr.full_like(template, np.nan)
            )
            finite = field.values[np.isfinite(field.values)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
                boundaries = np.linspace(lo, hi, 6)
                status = "available"
            else:
                boundaries = []
                status = "unavailable"
            scale = MapScale(
                units,
                boundaries,
                boundaries,
                cmap(np.linspace(0.1, 1, 5)) if finite.size else [],
                cmap(0.0),
                cmap(1.0),
                "#dddddd",
                status,
            )
            fig = draw_raster_map(
                RasterMap(field.values, x, y),
                frame,
                scale,
                overlays or {},
                colorbar_title=f"{label}\n({units})",
            )
            if not finite.size:
                fig.axes[0].text(
                    0.5,
                    0.5,
                    f"{textwrap.fill(label, 30)}\nUnavailable",
                    transform=fig.axes[0].transAxes,
                    ha="center",
                    va="center",
                )
            count = int(np.isfinite(field.values).sum())
            save(
                fig,
                f"index_map_{metric}",
                f"{source.upper()} {label}",
                f"Mean of available annual cellwise {label} values; {count} finite native basin cells. "
                "Independent colour scales describe this source and metric; they do not establish cross-source agreement.",
            )
    Path(paths["index_captions"]).write_text(
        json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
