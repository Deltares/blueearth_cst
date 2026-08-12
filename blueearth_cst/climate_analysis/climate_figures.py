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

import json
import os
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from blueearth_cst.shared.plot_style import RASTER_DPI
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
        f"{dataset}_{var}_{kind}.png" for var in CLIMATE_VARS for kind in FIGURE_KINDS
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


#: Stream-order column names, in the order they are tried. wflow writes
#: ``strord``; ``spatial/geoms/rivers.geojson`` writes ``order``.
_RIVER_ORDER_COLUMNS = ("strord", "order")


def _river_order_column(rivers) -> Optional[str]:
    """The stream-order column this river layer actually carries, if any."""
    if rivers is None or not hasattr(rivers, "columns"):
        return None
    return next((c for c in _RIVER_ORDER_COLUMNS if c in rivers.columns), None)


#: The vector layers BOTH climate map families draw, keyed by the overlay name
#: ``_render_map`` expects. One source for both, so the source-grid and
#: model-grid maps differ only in the raster underneath — which is the whole
#: point of drawing them as one set.
#:
#: Deliberately the ENGINE-NEUTRAL products from rule 1.03, not the wflow
#: model's staticgeoms: rule 1.05 runs off the climate store and must stay
#: independent of the model build (1.07), so the shared foundation is the only
#: layer set both callers can reach. The model's separate ``outlets.geojson``
#: is not in it — but its point IS: the basin outlet is one of ``locations``,
#: so it is still drawn, as a point of interest rather than its own symbol.
_SPATIAL_OVERLAYS = {
    "basins": "basins",
    "subbasins": "subbasins",
    "rivers": "rivers",
    "gauges": "locations",
}


def load_spatial_overlays(geoms_dir: Optional[Union[str, Path]]) -> dict:
    """Read ``data/spatial/geoms/`` into the overlays the map renderer takes.

    Returns an empty dict when ``geoms_dir`` is absent, and skips any single
    layer that is missing — a climate map with no vectors on it is still
    a correct figure, so refusing to plot one would trade a complete figure for
    no figure.
    """
    if geoms_dir is None:
        return {}
    geoms_dir = Path(geoms_dir)
    overlays = {}
    for name, stem in _SPATIAL_OVERLAYS.items():
        path = geoms_dir / f"{stem}.geojson"
        if path.is_file():
            overlays[name] = gpd.read_file(path)
        else:
            log_row(f"spatial overlay absent, skipped: {path}", module="plot")
    return overlays


def _render_map(da, spec, title, caveat, overlays, levels=None, levels_out=None):
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
        extent_from_layer,
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
    style = base.replace(label=f"{label.capitalize()} ({axis_unit})")
    if spec["style"] == "temp":
        style = resolve_temperature_style(field, style)
    if levels is not None:
        style.levels = levels

    # Every overlay is optional, and the two datasets supply them from
    # different products: the FORCING maps take the wflow model's staticgeoms
    # (one polygon per subcatchment in ``basins``), the SOURCE maps take the
    # engine-neutral ``data/spatial/geoms/`` from rule 1.03 (a dissolved
    # ``basins`` plus a separate ``subbasins``). Accepting both shapes is what
    # lets one renderer serve both without either caller reshaping its layers.
    overlays = overlays or {}
    basins = overlays.get("basins")
    subbasins = overlays.get("subbasins")
    rivers = overlays.get("rivers")
    has_basins = basins is not None and len(basins) > 0
    if subbasins is not None and len(subbasins) > 0:
        divides = subbasins
    elif has_basins and len(basins) > 1:
        divides = basins
    else:
        divides = None
    fig, _ = plot_raster_map(
        field,
        rivers,
        _basin_outline(basins) if has_basins else None,
        subbasins=divides,
        gauges=overlays.get("gauges"),
        outlets=overlays.get("outlets"),
        # wflow spells stream order ``strord``; the shared vector foundation
        # spells it ``order``. Naming both keeps the river widths scaled on
        # either product instead of silently flattening to one weight.
        river_order_column=_river_order_column(rivers),
        style=style,
        # Framed on the BASIN, not on each raster's own footprint. The forcing
        # is masked to the basin and the source extraction is a few reanalysis
        # cells reaching far past it, so raster-framed the pair cannot be read
        # side by side — which is the one thing these two families exist to
        # support.
        extent=extent_from_layer(basins) if has_basins else None,
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
    if levels_out is not None:
        # Report back what this bar ended up using, so a later figure of the
        # same quantity can be pinned to it. Read from the style when it was
        # handed in, and recomputed from the FRAMED raster otherwise — the same
        # restriction plot_raster_map applies, so the two cannot disagree.
        levels_out[:] = (
            list(levels)
            if levels is not None
            else [float(v) for v in _levels_actually_used(field, style, basins)]
        )
    return fig


def _levels_actually_used(field, style, basins):
    """The class boundaries ``plot_raster_map`` would derive for this figure."""
    from blueearth_cst.shared.cartographic_map import (
        _class_levels,
        _raster_within,
        extent_from_layer,
    )

    extent = extent_from_layer(basins) if basins is not None and len(basins) else None
    framed = _raster_within(field, extent) if extent is not None else field
    return _class_levels(framed, style)


#: Month labels for the seasonal chart. Initials alone are ambiguous (J/J/J);
#: three letters fit at this width and read at a glance.
MONTH_LABELS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

#: In-plot annotations: the period mean, the trend, the box-plot key.
FONT_SIZE_ANNOTATION = 6.0


def _style_series_axes(ax) -> None:
    """The axis treatment every non-map figure in this set shares.

    An L-frame with a horizontal-only grid: on a time series the vertical
    gridlines compete with the data for the reader's eye, and the top and right
    spines close a box around nothing.
    """
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _series_style(spec):
    """The style this variable's non-map figures take, label and colour."""
    from blueearth_cst.shared.cartographic_map import RASTER_STYLES, style_series_color

    base = RASTER_STYLES[spec["style"]]
    return base, style_series_color(base)


def _series_axes(caveat, aspect=0.42):
    """A figure sized and styled like the maps, with the caveat in the layout.

    Constrained layout, not ``tight_layout``: the maps are built on it, and a
    figure family that mixes the two cannot be made to agree on margins. It is
    also what reserves room for the footnote instead of overprinting the axis.
    """
    from blueearth_cst.shared.cartographic_map import (
        _publication_rc,
        series_figure_size,
    )
    from blueearth_cst.shared.plot_style import COLOR_CAVEAT, FONT_SIZE_CAVEAT

    with plt.rc_context(_publication_rc()):
        fig = plt.figure(figsize=series_figure_size(aspect), layout="constrained")
        ax = fig.add_subplot()
        if caveat:
            fig.supxlabel(
                caveat, fontsize=FONT_SIZE_CAVEAT, color=COLOR_CAVEAT, wrap=True
            )
    return fig, ax


def _decadal_trend(years, values):
    """Least-squares slope per decade, or ``None`` when it cannot be fitted.

    Deliberately plain OLS and deliberately unlabelled as significant: on the
    two decades these figures cover, the slope is a description of what the
    record did, not evidence about climate. Reported per decade because per
    year is unreadably small for rainfall.
    """
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return None, None
    slope, intercept = np.polyfit(years[finite], values[finite], 1)
    return float(slope), float(intercept)


def _render_annual(da, spec, title, caveat, overlays, **_):
    """Domain-mean value per year, with its trend and the period mean."""
    how, label, unit = spec["how"], spec["label"], spec["unit"]
    series = _yearly(da.mean(dim=_space_dims(da)), how).compute()
    axis_unit = f"{unit} y$^{{-1}}$" if how == "sum" else unit
    years = series["time"].dt.year.values.astype(float)
    values = series.values.astype(float)
    _, colour = _series_style(spec)

    fig, ax = _series_axes(caveat)
    ax.plot(years, values, color=colour, marker="o", lw=1.1, ms=3.5, zorder=3)

    if values.size:
        mean = float(np.nanmean(values))
        ax.axhline(mean, color="0.45", lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.annotate(
            f"period mean {mean:,.1f}",
            xy=(years[0], mean),
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE_ANNOTATION,
            color="0.35",
        )
        slope, intercept = _decadal_trend(years, values)
        if slope is not None:
            ax.plot(
                years,
                slope * years + intercept,
                color=colour,
                lw=1.4,
                ls=(0, (6, 2.5)),
                alpha=0.85,
                zorder=4,
            )
            ax.annotate(
                f"trend {slope * 10:+,.1f} {axis_unit.split(' ')[0]}/decade",
                xy=(years[-1], slope * years[-1] + intercept),
                xytext=(-4, -4),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=FONT_SIZE_ANNOTATION,
                color=colour,
            )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"{label.capitalize()} ({axis_unit})")
    # Years are integers; the default locator happily labels them 2002.5.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_series_axes(ax)
    return fig


def _render_monthly(da, spec, title, caveat, overlays, **_):
    """Monthly climatology of the domain mean, and its year-to-year spread.

    The mean alone answered "when is the wet season?" and nothing about how
    reliably — two basins with the same climatology and very different
    interannual spread drew the same figure. The boxes are the distribution
    ACROSS YEARS for each calendar month, so the reader sees both.
    """
    how, label, unit = spec["how"], spec["label"], spec["unit"]
    domain = da.mean(dim=_space_dims(da)).resample(time="ME")
    per_month = (domain.sum("time") if how == "sum" else domain.mean("time")).compute()
    months = np.arange(1, 13)
    grouped = per_month.groupby("time.month")
    spread = [
        np.asarray(grouped[m].values, dtype=float)
        if m in grouped.groups
        else np.array([])
        for m in months
    ]
    spread = [values[np.isfinite(values)] for values in spread]
    axis_unit = f"{unit} month$^{{-1}}$" if how == "sum" else unit
    _, colour = _series_style(spec)

    fig, ax = _series_axes(caveat, aspect=0.40)
    populated = [i for i, values in enumerate(spread) if values.size]
    if populated:
        ax.boxplot(
            [spread[i] for i in populated],
            positions=[months[i] for i in populated],
            widths=0.62,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="white", lw=1.1),
            boxprops=dict(facecolor=colour, edgecolor=colour, lw=0.6),
            whiskerprops=dict(color=colour, lw=0.8),
            capprops=dict(color=colour, lw=0.8),
        )
        means = [float(np.mean(spread[i])) for i in populated]
        ax.plot(
            [months[i] for i in populated],
            means,
            color="0.2",
            marker="D",
            ms=2.6,
            lw=0.9,
            ls="-",
            zorder=5,
        )
        # No legend. Box-whiskers over a monthly axis are a convention the
        # audience reads without a key, and the caption carries what the boxes
        # are — the same reason the figures carry no title.
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlim(0.4, 12.6)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{label.capitalize()} ({axis_unit})")
    _style_series_axes(ax)
    return fig


_RENDERERS = {
    "map": _render_map,
    "annual": _render_annual,
    "monthly": _render_monthly,
}


#: Sidecar written beside the SOURCE figures and read by the FORCING ones, so a
#: variable's two maps carry the same colourbar and can be read against each
#: other. Direction matters and is one-way: the source rule (1.05) is
#: independent of the model build and writes; the forcing rule (1.13) is
#: downstream of it anyway and reads. The reverse would make the source figures
#: wait on a wflow model, which they exist to precede.
LEVELS_FILENAME = "climate_levels.json"


def read_shared_levels(levels_file: Optional[Union[str, Path]]) -> dict:
    """Class boundaries recorded by an earlier figure set, keyed by variable.

    Returns an empty dict when the file is absent or unreadable — a figure with
    its own bar is still correct, and refusing to plot because a convenience
    sidecar is missing would be the wrong trade.
    """
    if levels_file is None:
        return {}
    path = Path(levels_file)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log_row(f"shared levels unreadable, ignored: {path}", module="plot")
        return {}
    return {k: v for k, v in loaded.items() if isinstance(v, list) and len(v) > 1}


def write_shared_levels(levels_file: Union[str, Path], levels: dict) -> None:
    """Record the class boundaries this figure set used, for its pair to adopt."""
    path = Path(levels_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(levels, indent=2, sort_keys=True), encoding="utf-8")
    log_row(f"Wrote shared colourbar levels: {path}", module="plot")


def plot_climate_figures(
    ds: xr.Dataset,
    plot_dir: Union[str, Path],
    dataset: str,
    *,
    caveat: Optional[str] = None,
    overlays: Optional[dict] = None,
    levels_file: Optional[Union[str, Path]] = None,
    write_levels: bool = False,
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
    shared = {} if write_levels else read_shared_levels(levels_file)
    written = []
    for var, spec in CLIMATE_VARS.items():
        da = ds[var]
        captured = []
        for kind in FIGURE_KINDS:
            out_path = plot_dir / f"{dataset}_{var}_{kind}.png"
            fig = _RENDERERS[kind](
                da,
                spec,
                title,
                caveat,
                overlays,
                levels=shared.get(var),
                levels_out=captured if write_levels else None,
            )
            save_figure(out_path, dpi=RASTER_DPI)
            plt.close(fig)
            written.append(out_path)
        if write_levels and captured:
            shared[var] = list(captured)
    if write_levels and levels_file is not None:
        write_shared_levels(levels_file, shared)
    log_row(
        f"Wrote {len(written)} canonical climate figures ({dataset}) to {plot_dir}",
        module="plot",
    )
    return written
