"""Render climate-index products exclusively from verified retained evidence."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator

from blueearth_cst.climate_analysis.climate_indices import load_publication
from blueearth_cst.climate_analysis.diagnostic_figures import source_styles
from blueearth_cst.climate_analysis.figure_naming import figure_filename
from blueearth_cst.indicators.climate.definitions import (
    INDEX_GROUPS,
    INDEX_METADATA,
    INDEX_SERIES_LABELS,
)
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.fundamentals import (
    new_fig,
    time_series_grid,
    time_series_style,
    year_ticks,
)
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


def _series(data, metadata, *, seasonal=False, color=None, source="", columns=3):
    """Draw one available scope with compact row-unit and column-year labels."""
    metadata = {
        key: item
        for key, item in metadata.items()
        if key in data and np.isfinite(data[key].values).any()
    }
    if not metadata:
        raise ValueError("No available indices in the requested scope")
    columns = min(columns, len(metadata))
    rows = int(np.ceil(len(metadata) / columns))
    fig, axes = new_fig(
        rows * 42 + 11, rows, columns, width_mm=180 if columns > 1 else 100, sharex=True
    )
    fig.get_layout_engine().set(
        w_pad=0.8 / 25.4, h_pad=0.6 / 25.4, wspace=0.05, hspace=0.03
    )
    dates = pd.DatetimeIndex(data.period_start.values)
    years = np.asarray(dates.year)
    valid = np.any(
        np.stack([np.isfinite(data[key].values) for key in metadata]), axis=0
    )
    display_years = np.unique(years[valid])
    season_names = np.array([str(label).split(":")[0] for label in data.period.values])
    seasons = list(dict.fromkeys(season_names))
    season_styles = source_styles(seasons) if seasonal else {}
    units_by_panel = [units for _, units in metadata.values()]
    handles = []
    for index, (metric, (label, units)) in enumerate(metadata.items()):
        ax = axes.flat[index]
        row, column = divmod(index, columns)
        ax.set_title(
            textwrap.fill(f"{chr(97 + index)}) {label}", 30), loc="left", pad=4
        )
        row_units = units_by_panel[row * columns : (row + 1) * columns]
        if len(set(row_units)) != 1 or column == 0:
            ax.set_ylabel("°C" if units == "degC" else units)
        if seasonal:
            for season in seasons:
                selected = season_names == season
                (line,) = ax.plot(
                    years[selected],
                    data[metric].values[selected],
                    marker="o",
                    label=season,
                    **time_series_style(season, season_styles[season][0]),
                )
                if index == 0:
                    handles.append(line)
        else:
            ax.plot(
                years,
                data[metric].values,
                marker="o",
                **time_series_style(source, color),
            )
        year_ticks(ax, display_years)
        step = (
            10
            if len(display_years) > 20
            else 5
            if len(display_years) > 10
            else 2
            if len(display_years) > 4
            else 1
        )
        ax.set_xticks(
            np.arange(
                int(np.ceil(display_years[0] / step)) * step,
                display_years[-1] + 1,
                step,
            )
        )
        ax.xaxis.set_minor_locator(NullLocator())
        bottom = row == (len(metadata) - 1 - column) // columns
        ax.set_xlabel("Year" if bottom else "")
        ax.tick_params(
            axis="x",
            which="major",
            bottom=bottom,
            top=False,
            labelbottom=bottom,
            length=2.5,
            width=0.5,
        )
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.margins(y=0.1)
        time_series_grid(ax)
    for ax in list(axes.flat)[len(metadata) :]:
        ax.set_visible(False)
    if handles:
        fig.legend(handles=handles, loc="outside upper center", ncols=len(handles))
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
    omissions = {}
    output_dir = Path(paths["index_products"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source = manifest["identity"]["source"]
    hue = source_styles([source])[source][0]
    options = manifest["identity"]["settings"]["climate_indices"]
    threshold = options.get("mean_temperature_threshold_degc", 25.0)
    metadata = INDEX_METADATA | {
        "mean_temp_above_threshold_days": (f"Mean T > {threshold:g}°C days", "days")
    }
    reference = options["reference_period"]
    window = manifest["identity"]["settings"]["analysis_window"]
    common = (
        f"{source.upper()}; reference calendar years {reference['start']}–{reference['end']}. "
        f"Reporting start years {window['start']}–{window['end']}; "
        f"annual start month {manifest['identity']['start_month']}. "
        "Mean-temperature counts use T < 0°C and "
        f"T > {options.get('mean_temperature_threshold_degc', 25.0):g}°C, without missing-day scaling. "
        f"Minimum reference sample fraction {options['reference_minimum_fraction']:g}. "
        "TX90p/TN90p use in-reference bootstrap; WSDI and heatwaves use fixed thresholds. "
        "Missing values indicate unavailable inputs, reference support or period coverage. "
        "Censored observed durations are lower bounds; short references are exploratory."
    )

    def save(fig, key, title, explanation):
        path = Path(paths[key]) if key in paths else output_dir / key
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
        for period in ("annual", "seasonal") if options["seasons"] else ("annual",):
            if period == "seasonal":
                with xr.open_dataset(folder / "seasonal_basin.nc") as opened:
                    series = opened.load()
            else:
                series = basin
            omissions[period] = [
                metric
                for metric in metadata
                if metric not in series or not np.isfinite(series[metric].values).any()
            ]
            for scope, (scope_title, columns, metrics) in INDEX_GROUPS.items():
                subset = {
                    metric: (
                        INDEX_SERIES_LABELS.get(metric, metadata[metric][0]),
                        metadata[metric][1],
                    )
                    for metric in metrics
                    if metric not in omissions[period]
                }
                if not subset:
                    continue
                filename = figure_filename(
                    source, "climate_indices", f"{period}_{scope}_ts", "basin_avg"
                )
                save(
                    _series(
                        series,
                        subset,
                        seasonal=period == "seasonal",
                        color=hue,
                        source=source,
                        columns=columns,
                    ),
                    filename,
                    f"{source.upper()} {period} climate indices — {scope_title}",
                    "Indices of daily equal-weight basin-cell means, not spatial means of cell indices. "
                    "Only available metrics are shown; y scales are metric-specific and unit titles "
                    "are shared within compatible rows. Years are labelled by start year, with empty "
                    "boundary periods omitted and internal gaps retained. Durations count overlapping "
                    f"days within each period. Configured seasons: {options['seasons']}.",
                )
        exceedance = pd.read_csv(folder / "wet_spell_exceedance_basin.csv")
        if len(exceedance):
            fig, axes = new_fig(65, width_mm=100)
            ax = axes[0, 0]
            ax.step(
                exceedance.duration,
                exceedance.exceedance_probability,
                where="pre",
                **time_series_style(source, hue),
            )
            ax.set(xlabel="days", ylabel="P(duration ≥ days)", ylim=(0, 1))
            time_series_grid(ax)
            save(
                fig,
                figure_filename(
                    source, "climate_indices", "wet_spell_exceedance", "basin_avg"
                ),
                f"{source.upper()} complete wet-spell exceedance",
                "Empirical survival probabilities for complete basin-series wet spells over the retained record. "
                "Gap- and edge-censored segments are excluded; this can bias duration distributions.",
            )
        else:
            omissions["wet_spell_exceedance"] = "no complete wet spells"
        # Use a two-dimensional index field, never a scalar support counter.
        template = next(grid[metric] for metric in INDEX_METADATA if metric in grid)
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
        for metric, (label, units) in metadata.items():
            field = (
                grid[metric].mean("period").transpose("latitude", "longitude")
                if metric in grid
                else xr.full_like(template, np.nan)
            )
            finite = field.values[np.isfinite(field.values)]
            if not finite.size:
                omissions.setdefault("maps", []).append(metric)
                continue
            lo, hi = float(finite.min()), float(finite.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            boundaries = np.linspace(lo, hi, 6)
            scale = MapScale(
                units,
                boundaries,
                boundaries,
                cmap(np.linspace(0.1, 1, 5)),
                cmap(0.0),
                cmap(1.0),
                "#dddddd",
                "available",
            )
            fig = draw_raster_map(
                RasterMap(field.values, x, y),
                frame,
                scale,
                overlays or {},
                colorbar_title=f"{label}\n({units})",
            )
            count = int(np.isfinite(field.values).sum())
            save(
                fig,
                figure_filename(source, metric, "annual_clim_map", "basin_ext"),
                f"{source.upper()} {label}",
                f"Mean of available annual cellwise {label} values; {count} finite native basin cells. "
                "Independent colour scales describe this source and metric; they do not establish cross-source agreement.",
            )
    write_figure_titles(output_dir, captions)
    (output_dir / "figure_availability.json").write_text(
        json.dumps(
            {
                "evaluation_id": manifest["evaluation_id"],
                "omitted": omissions,
                "figures": list(captions),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "figure_captions.json").write_text(
        json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
