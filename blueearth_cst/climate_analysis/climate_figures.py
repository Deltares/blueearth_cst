"""ONE canonical climate figure set, applied to every gridded climate dataset.

WF1 holds two gridded climate products and until now each had its own plotting
code and its own idea of what a climate figure is:

=====================  =========================================  ==============
Dataset (``dataset``)  Home                                       Producer
=====================  =========================================  ==============
``source``             ``data/climate/historical/<key>/plots/``   rule 1.15
``forcing``            ``models/hydrology/wflow/forcing/plots/``  rule 1.13
=====================  =========================================  ==============

They are the SAME climate at two stages — raw on the extraction grid, and
downscaled/corrected onto the model grid — so the useful question is what
changed between them. That question is unanswerable while the two are drawn
differently, which is why this module exists: identical figures, identical
aggregation, identical layout, differing only in the data and the label. The
redundancy is deliberate and cheap (these are aggregations of arrays already in
memory).

The set is a CROSS-PRODUCT, ``variable x kind``, and both axes are meant to
grow. Adding a kind is one entry in ``FIGURE_KINDS`` plus its branch in
``_render``; adding a variable is one entry in ``CLIMATE_VARS``. Because the
product is config-invariant, ``figure_names()`` lets the Snakefile DECLARE every
figure (O-24) instead of writing some of them invisibly — keep it that way when
extending, or the new figures become undeclared outputs.

Deliberately plain matplotlib: no cartopy basemap tiles, so neither rule needs
NETWORK access. Rule 1.13 used ``cartopy.io.img_tiles.QuadtreeTiles`` before
this module and therefore made a live tile request mid-workflow; the basin/river
context it bought is available offline through ``overlays`` (drawn from the
model's own geometries) and, at higher fidelity, from rule 1.12's
``basin_area.png``.

A third climate-figure family — the model-parity plots under
``models/hydrology/wflow/evaluation/plots/`` (rule 1.11) — is NOT part of this
set. It answers a different question (per-subcatchment climate as the model sees
it, beside the discharge it produced) and is keyed by station rather than by
grid.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from blueearth_cst.shared.cartographic_map import RASTER_DPI
from blueearth_cst.shared.snake_utils import log_row, save_figure

#: One entry per variable: label and unit for the axes, and how the variable
#: aggregates in time. ``sum`` is a flux that accumulates (a yearly TOTAL);
#: ``mean`` is a state that averages. Getting this wrong is not cosmetic -- a
#: summed temperature is meaningless and a meaned rainfall understates by ~365x.
CLIMATE_VARS = {
    "precip": {
        "label": "precipitation",
        "unit": "mm",
        "how": "sum",
        "style": "precip",
    },
    "temp": {
        "label": "air temperature",
        "unit": "$\\degree$C",
        "how": "mean",
        "style": "temp",
    },
    "pet": {
        "label": "potential evaporation",
        "unit": "mm",
        "how": "sum",
        "style": "pet",
    },
}

#: One entry per figure kind. ``map`` is the spatial view (climatological
#: field), ``annual`` and ``monthly`` are the temporal views (domain mean).
FIGURE_KINDS = ("map", "annual", "monthly")

#: Datasets this set is applied to. The key is the filename prefix AND the
#: title, so a figure copied out of its directory still says what it is -- the
#: reason the source figures carried a ``source_`` prefix before this module,
#: now applied to both sides so the two directories are directly comparable.
DATASETS = {
    "source": "source grid (raw extraction)",
    "forcing": "model grid (wflow forcing)",
}

#: A year is plotted as a TOTAL only if it is essentially complete; below this
#: fraction of the modal timestep count it is dropped instead. A truncated first
#: or last year otherwise draws a dip that looks like climate and is calendar.
_COMPLETE_YEAR_FRACTION = 0.9


def figure_names(dataset: str) -> list[str]:
    """Every filename this module writes for ``dataset``, in a stable order.

    The Snakefile calls this to declare the outputs, so it is the single source
    of the naming scheme ``<dataset>_<variable>_<kind>.png``. Nothing else may
    build those names by hand.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        )
    return [
        f"{dataset}_{var}_{kind}.png"
        for var in CLIMATE_VARS
        for kind in FIGURE_KINDS
    ]


def _space_dims(da: xr.DataArray) -> list[str]:
    """The non-time dimensions of ``da`` (the spatial ones, on any grid)."""
    return [d for d in da.dims if d != "time"]


def _yearly(series: xr.DataArray, how: str) -> xr.DataArray:
    """Per-year aggregate of a 1-D time series, incomplete years dropped.

    Only ``sum`` needs the completeness filter -- a mean over a partial year is
    still a valid mean of what was observed, while a total is not a total.
    """
    grouped = series.resample(time="YE")
    values = grouped.sum("time") if how == "sum" else grouped.mean("time")
    if how != "sum":
        return values.compute()
    # .compute() BEFORE the mask, and not merely for speed: `where(..., drop=True)`
    # indexes with the boolean array, and xarray refuses to index with a DASK
    # one ("this will result in a dask array of unknown shape"). PET arrives
    # dask-backed from the meteo workflow while precip and temp come straight
    # off the netCDF, so leaving this lazy fails on the PET figures only --
    # which is exactly how it presented: six figures written, then a KeyError.
    values = values.compute()
    counts = series.resample(time="YE").count("time").compute()
    if counts.size:
        modal = int(np.median(counts.values))
        values = values.where(counts >= modal * _COMPLETE_YEAR_FRACTION, drop=True)
    return values


def _climatological_field(da: xr.DataArray, how: str) -> xr.DataArray:
    """The map panel's field: per-year aggregate, then averaged over years."""
    grouped = da.resample(time="YE")
    field = (grouped.sum("time") if how == "sum" else grouped.mean("time")).mean("time")
    if how == "sum":
        # Zero-accumulation cells are outside the domain, not dry.
        field = field.where(field > 0)
    return field.compute()


def _footer(fig, caveat: Optional[str]) -> None:
    if caveat:
        fig.text(0.01, 0.01, caveat, fontsize=6.5, color="dimgray", va="bottom")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()


#: Column carrying a point's human-readable label, if the layer has one.
_LABEL_COLUMN = "station_name"


def _label_points(ax, gdf) -> None:
    """Annotate a point overlay with its station names.

    A marker with no name answers "something is here" but not "which one",
    which is the question a reader brings to a multi-gauge basin. Rule 1.12's
    basin_area.png has labelled its gauges since R07; this brings the canonical
    climate maps into line rather than leaving one figure family mute.

    Silently does nothing for a layer without the column (the model's own
    ``outlets`` has none) — those markers are self-explanatory in context.
    """
    if _LABEL_COLUMN not in getattr(gdf, "columns", []):
        return
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        ax.annotate(
            text=str(row[_LABEL_COLUMN]),
            xy=(geometry.x, geometry.y),
            xytext=(3.0, 3.0),
            textcoords="offset points",
            fontsize=5,
            fontweight="bold",
            zorder=6,
        )


def _render_map(da, spec, title, caveat, overlays):
    """Climatological field as a cartographic map.

    A caller of ``shared.cartographic_map.plot_raster_map``, so this figure carries
    same furniture as rule 1.12's basin map: graticule and frame, latitude-
    corrected scale bar, north arrow, locator inset, and the side panel holding
    the colourbar over the vector legend. Only the raster and its palette
    differ, which is the point of the template — a new quantity is an entry in
    ``RASTER_STYLES``, not another plotting function.
    """
    from blueearth_cst.shared.cartographic_map import (
        RASTER_STYLES,
        RasterStyle,
        plot_raster_map,
        resolve_temperature_style,
    )
    from blueearth_cst.shared.plot_map import _basin_outline

    how, label, unit = spec["how"], spec["label"], spec["unit"]
    field = _climatological_field(da, how)
    axis_unit = f"{unit} y$^{{-1}}$" if how == "sum" else unit

    base = RASTER_STYLES[spec["style"]]
    # The unit belongs to the DATA, not to the style: `how` decides whether the
    # field is a yearly total or a mean, so the label is built here.
    style = RasterStyle(
        label=f"{label.capitalize()} ({axis_unit})",
        palette=base.palette,
        classification=base.classification,
        clip_quantiles=base.clip_quantiles,
        zero_baseline=base.zero_baseline,
        relief=base.relief,
        interpolation=base.interpolation,
        diverging_center=base.diverging_center,
    )
    if spec["style"] == "temp":
        style = resolve_temperature_style(field, style)

    # Every overlay is optional. The source-grid figures (rule 1.15) are drawn
    # before any model exists, so this function has to produce a bare raster map
    # as readily as a fully-dressed one.
    overlays = overlays or {}
    basins = overlays.get("basins")
    has_basins = basins is not None and len(basins) > 0
    fig, _ = plot_raster_map(
        field,
        overlays.get("rivers"),
        _basin_outline(basins) if has_basins else None,
        subbasins=basins if has_basins and len(basins) > 1 else None,
        gauges=overlays.get("gauges"),
        outlets=overlays.get("outlets"),
        style=style,
        # No figure title. A published figure carries its title in the caption,
        # and nothing is lost here: the colourbar names the quantity and the
        # footnote names the dataset. ``title`` stays available on the template
        # for a caller that renders outside a document.
        caveat=caveat,
        # The field is a derived aggregate whose units this function sets, so
        # the raster's own `units` attribute says nothing useful about it — the
        # wflow forcing labels both temp and pet "m". Skip the check rather
        # than warn on every run about metadata nothing here reads.
        expected_units=(),
    )
    return fig


def _render_annual(da, spec, title, caveat, overlays):
    """Domain-mean value per year, with the period mean for reference."""
    how, label, unit = spec["how"], spec["label"], spec["unit"]
    series = _yearly(da.mean(dim=_space_dims(da)), how).compute()
    axis_unit = f"{unit} y$^{{-1}}$" if how == "sum" else unit
    years = series["time"].dt.year.values
    values = series.values

    fig, ax = plt.subplots(figsize=(8, 4))
    colour = "steelblue" if how == "sum" else "firebrick"
    ax.plot(years, values, color=colour, marker="o", lw=1.1, ms=3.5)
    if values.size:
        mean = float(np.nanmean(values))
        ax.axhline(mean, color="dimgray", lw=0.9, ls="--")
        ax.annotate(
            f"period mean {mean:,.1f}",
            xy=(years[-1], mean),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="dimgray",
        )
    ax.set_xlabel("year")
    ax.set_ylabel(f"{label} [{axis_unit}]")
    ax.set_title(f"{label} — annual series, domain mean\n{title}", fontsize=9)
    # Years are integers; the default locator happily labels them 2002.5.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    _footer(fig, caveat)
    return fig


def _render_monthly(da, spec, title, caveat, overlays):
    """Monthly climatology of the domain mean."""
    how, label, unit = spec["how"], spec["label"], spec["unit"]
    domain = da.mean(dim=_space_dims(da)).resample(time="ME")
    monthly = domain.sum("time") if how == "sum" else domain.mean("time")
    monthly = monthly.groupby("time.month").mean("time").compute()
    months = np.arange(1, 13)
    values = monthly.reindex(month=months).values
    axis_unit = f"{unit} month$^{{-1}}$" if how == "sum" else unit

    fig, ax = plt.subplots(figsize=(6.5, 4))
    if how == "sum":
        ax.bar(months, values, color="steelblue")
    else:
        ax.plot(months, values, color="firebrick", marker="o", lw=0.9, ms=3)
    ax.set_xticks(months)
    ax.set_xlabel("month")
    ax.set_ylabel(f"{label} [{axis_unit}]")
    ax.set_title(f"{label} — monthly climatology, domain mean\n{title}", fontsize=9)
    ax.grid(alpha=0.3)
    _footer(fig, caveat)
    return fig


_RENDERERS = {
    "map": _render_map,
    "annual": _render_annual,
    "monthly": _render_monthly,
}


def plot_climate_figures(
    ds: xr.Dataset,
    plot_dir: Union[str, Path],
    dataset: str,
    *,
    caveat: Optional[str] = None,
    overlays: Optional[dict] = None,
) -> list[Path]:
    """Write the canonical figure set for one gridded climate dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded climate carrying every key of :data:`CLIMATE_VARS` on a
        ``time`` + spatial grid. A PLAIN dataset on purpose: the raw side has no
        model to load and the forcing side has already loaded one, so the model
        coupling stays in the callers and this function stays testable without a
        model (the P4 property ``tests/test_plot_climate_source.py`` pins).
    plot_dir : str | Path
        Destination directory. Created if absent.
    dataset : str
        A key of :data:`DATASETS` — the filename prefix and the subtitle.
    caveat : str, optional
        Footnote rendered on every figure, so it survives the file being copied
        out of its directory.
    overlays : dict, optional
        ``{"basins": gdf, "rivers": gdf, ...}`` drawn on the MAP figures only.
        Absent or empty entries are skipped, so a caller without a model simply
        passes nothing.

    Returns
    -------
    list[Path]
        The figures written, in :func:`figure_names` order.

    Raises
    ------
    ValueError
        If ``dataset`` is unknown or ``ds`` lacks a variable. Loud on purpose:
        the rules declare these figures, so a silent skip would resurface as an
        opaque ``MissingOutputException`` at the end of the job.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        )
    missing = [var for var in CLIMATE_VARS if var not in ds]
    if missing:
        raise ValueError(
            f"plot_climate_figures: dataset {dataset!r} is missing {missing}; "
            f"the canonical set needs {list(CLIMATE_VARS)}"
        )

    plot_dir = Path(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    title = DATASETS[dataset]
    written = []
    for var, spec in CLIMATE_VARS.items():
        da = ds[var]
        for kind in FIGURE_KINDS:
            out_path = plot_dir / f"{dataset}_{var}_{kind}.png"
            fig = _RENDERERS[kind](da, spec, title, caveat, overlays)
            save_figure(out_path, dpi=RASTER_DPI)
            plt.close(fig)
            written.append(out_path)
    log_row(
        f"Wrote {len(written)} canonical climate figures ({dataset}) to {plot_dir}",
        module="plot",
    )
    return written
