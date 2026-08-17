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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from blueearth_cst.climate_analysis.figure_naming import (
    figure_filename,
    map_spatial_scope,
)
from blueearth_cst.shared.grid_cells import masked
from blueearth_cst.shared.plot_style import RASTER_DPI
from blueearth_cst.shared.snake_utils import (
    DEFAULT_WATER_YEAR_ANCHOR,
    PRECIP_ONLY_SOURCES,
    log_row,
    save_figure,
)

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

#: How each family FRAMES its maps, and it is not the same answer for both.
#:
#: ``"basin"`` crops to the basin's bounding box; ``"raster"`` shows the field's
#: own full footprint. The forcing is masked to the basin, so for it the two are
#: nearly the same box and ``"basin"`` is simply the tidier one.
#:
#: The SOURCE side has been both, and is BACK on ``"basin"`` as of 2026-08-17
#: (owner's ruling), reversing the 2026-08-16 ruling that had moved it to
#: ``"raster"`` the day before. Both directions are recorded here because the
#: trade is real and neither answer is free.
#:
#: ``"raster"`` was chosen because a raw extraction is a handful of reanalysis
#: cells reaching well past the catchment — at ERA5's 0.25 degrees a small basin
#: sits inside one or two of them — so cropping to the basin shows a single flat
#: class and calls it a map of the climate. Measured on the rapid fixture: the
#: whole era5 extraction is 4x5 cells and the basin is 0.200 x 0.133 degrees,
#: under one cell wide.
#:
#: ``"basin"`` is chosen anyway, because the question the figure pair answers is
#: what downscaling changed, and that comparison needs both families framed on
#: the same footprint. A flat source panel is then INFORMATION — it says the
#: source grid does not resolve this basin — rather than a defect of the figure.
#: On a fine source (CHIRPS at 0.05 degrees) the same frame is genuinely
#: informative.
#:
#: Two consequences, both live:
#:
#: * ``_raster_within`` crops the field to the frame before the colourbar
#:   classifies it, so the bar describes what is shown. That already existed for
#:   the caller-supplied-extent path and is what makes this switch safe.
#: * wf0's shared scale (rule 0.04b, ``climate_levels.json``) is pooled from the
#:   FULL stores, so it now describes a wider footprint than any source panel
#:   draws. The scale stays comparable ACROSS sources, which is its job, but it
#:   is no longer the range of the drawn cells.
MAP_EXTENT = {
    "source": "basin",
    "forcing": "basin",
}

#: A year is plotted as a TOTAL only if it is essentially complete; below this
#: fraction of the modal timestep count it is dropped instead. A truncated first
#: or last year otherwise draws a dip that looks like climate and is calendar.
_COMPLETE_YEAR_FRACTION = 0.9


def source_climate_vars(clim_source: str) -> tuple[str, ...]:
    """The :data:`CLIMATE_VARS` keys ``clim_source`` can HONESTLY be drawn for.

    A precipitation-only source (:data:`PRECIP_ONLY_SOURCES`) gets ``precip``
    and nothing else. Its store does carry temperature, radiation and pressure
    -- the extraction borrows them from era5, because the model cannot be forced
    without them -- but those values are era5's, regridded. Drawing them under
    this source's name would answer "how do the two sources differ?" with a
    panel that cannot differ, in the one workflow whose job is that comparison.

    Ruled 2026-08-16 (owner): a dataset missing a variable gets NO output for
    it, rather than an output silently filled from another dataset.
    """
    if clim_source in PRECIP_ONLY_SOURCES:
        return ("precip",)
    return tuple(CLIMATE_VARS)


#: Which ``plot_context`` token of the WF0 grammar each figure kind carries.
#: ``monthly`` is ``monthly_box`` HERE because this family draws the
#: year-to-year distribution as boxes; the comparison family draws the same
#: months as lines and names itself ``monthly_clim_line`` accordingly.
KIND_PLOT_CONTEXTS = {
    "map": "annual_clim_map",
    "annual": "annual_ts",
    "monthly": "monthly_box",
}

#: The kinds that REDUCE the field to a series over an area, and therefore exist
#: once per spatial scope. ``map`` is not one of them: it draws the field itself,
#: so a per-subbasin map would be the same raster cropped twelve ways.
AGGREGATED_KINDS = tuple(k for k in FIGURE_KINDS if k != "map")


def scope_caveat(caveat: Optional[str], spatial_scope: str, mask) -> Optional[str]:
    """Add the AREA a series figure reduces over to its footnote.

    Without this the per-source caveat still reads "on the source extraction
    grid" while the values are a basin-cell mean — true of the map beside it and
    false of the series itself.

    The CELL COUNT is the part that earns its place. At ERA5's 0.25 degrees a
    small basin's subbasins can all select the same one or two cells, so several
    per-subbasin figures come out identical; a reader who sees "2 cells" reads
    that as the source failing to resolve the subbasin, which it is, instead of
    suspecting the figures were mislabelled.
    """
    if mask is None:
        area = "the full extraction grid, which reaches past the basin"
    else:
        cells = int(mask.values.sum())
        where = (
            "the basin"
            if spatial_scope == "basin_avg"
            else "subbasin " + spatial_scope[len("subbasin_") : -len("_avg")]
        )
        area = f"{where}'s {cells} cell{'s' if cells != 1 else ''}"
    line = f"Domain mean over {area}."
    return f"{caveat}\n{line}" if caveat else line


def _legacy_figure_name(dataset: str, var: str, kind: str) -> str:
    """``<dataset>_<variable>_<kind>.png`` — the pre-grammar spelling.

    Retained for the FORCING family only. ``wf0-figure-filename-rule.md`` stages
    the new grammar at WF0 first, and the forcing figures have their own
    consumers and their own directory; migrating them is a separate step with
    its own rename sweep.
    """
    return f"{dataset}_{var}_{kind}.png"


def source_figure_names(
    clim_source: str,
    variables: Optional[Sequence[str]] = None,
    spatial_scopes: Sequence[str] = ("basin_avg",),
) -> list[str]:
    """Every filename the SOURCE family writes for one source, in a stable order.

    The WF0 grammar (``climate_analysis.figure_naming``), so the
    ``dataset_scope`` is the source id itself — ``era5_precip_annual_ts_basin_avg.png``
    — rather than the literal word ``source``. A figure copied out of its
    directory then still says which dataset it came from, which the old
    ``source_precip_annual.png`` did not.

    ``spatial_scopes`` lists the areas the AGGREGATED kinds are drawn for; the
    map is drawn once, framed per :data:`MAP_EXTENT`. Pass only the scopes whose
    count is knowable at DAG-parse time — the per-subbasin scopes are not, and
    the Snakefile declares those through a ``directory()`` instead.

    It must be the SAME list handed to :func:`plot_climate_figures`: narrow only
    the declaration and the extra files are undeclared, narrow only the drawing
    and the job ends in ``MissingOutputException``.
    """
    map_scope = map_spatial_scope(MAP_EXTENT["source"])
    names = []
    for var in _resolve_variables(variables):
        for kind in FIGURE_KINDS:
            context = KIND_PLOT_CONTEXTS[kind]
            if kind == "map":
                names.append(figure_filename(clim_source, var, context, map_scope))
                continue
            names.extend(
                figure_filename(clim_source, var, context, scope)
                for scope in spatial_scopes
            )
    return names


def figure_names(dataset: str, variables: Optional[Sequence[str]] = None) -> list[str]:
    """Every filename this module writes for ``dataset``, in a stable order.

    The FORCING family's declaration (rule 1.13), on the legacy spelling. The
    source family is named by :func:`source_figure_names` under the WF0 grammar;
    the two spellings coexist deliberately while that migration is staged.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        )
    return [
        _legacy_figure_name(dataset, var, kind)
        for var in _resolve_variables(variables)
        for kind in FIGURE_KINDS
    ]


def _resolve_variables(variables: Optional[Sequence[str]]) -> tuple[str, ...]:
    """Validate a variable subset, defaulting to the full canonical set.

    Order follows :data:`CLIMATE_VARS`, not the caller's, so two callers passing
    the same variables in a different order still declare the same filenames in
    the same sequence.
    """
    if variables is None:
        return tuple(CLIMATE_VARS)
    requested = set(variables)
    unknown = sorted(requested - set(CLIMATE_VARS))
    if unknown:
        raise ValueError(
            f"unknown climate variables {unknown}; expected a subset of "
            f"{list(CLIMATE_VARS)}"
        )
    if not requested:
        raise ValueError("variables must name at least one climate variable")
    return tuple(var for var in CLIMATE_VARS if var in requested)


def _space_dims(da: xr.DataArray) -> list[str]:
    """The non-time dimensions of ``da`` (the spatial ones, on any grid)."""
    return [d for d in da.dims if d != "time"]


def _yearly(
    series: xr.DataArray, how: str, anchor: str = DEFAULT_WATER_YEAR_ANCHOR
) -> xr.DataArray:
    """Per-year aggregate of a 1-D time series, incomplete years dropped.

    Only ``sum`` needs the completeness filter -- a mean over a partial year is
    still a valid mean of what was observed, while a total is not a total.
    """
    grouped = series.resample(time=anchor)
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
    counts = series.resample(time=anchor).count("time").compute()
    if counts.size:
        modal = int(np.median(counts.values))
        values = values.where(counts >= modal * _COMPLETE_YEAR_FRACTION, drop=True)
    return values


def _climatological_field(
    da: xr.DataArray, how: str, anchor: str = DEFAULT_WATER_YEAR_ANCHOR
) -> xr.DataArray:
    """The map panel's field: per-water-year aggregate, averaged over years."""
    grouped = da.resample(time=anchor)
    field = (grouped.sum("time") if how == "sum" else grouped.mean("time")).mean("time")
    if how == "sum":
        # Zero-accumulation cells are outside the domain, not dry.
        field = field.where(field > 0)
    return field.compute()


def map_field(da: xr.DataArray, spec: dict, anchor: str = DEFAULT_WATER_YEAR_ANCHOR):
    """The values the ``map`` figure draws."""
    return _climatological_field(da, spec["how"], anchor)


def annual_series(
    da: xr.DataArray, spec: dict, anchor: str = DEFAULT_WATER_YEAR_ANCHOR
):
    """The values the ``annual`` figure draws."""
    return _yearly(da.mean(dim=_space_dims(da)), spec["how"], anchor).compute()


def monthly_spread(da: xr.DataArray, spec: dict) -> list:
    """The per-calendar-month distributions the ``monthly`` box plot draws."""
    domain = da.mean(dim=_space_dims(da)).resample(time="ME")
    how = spec["how"]
    per_month = (domain.sum("time") if how == "sum" else domain.mean("time")).compute()
    grouped = per_month.groupby("time.month")
    spread = [
        np.asarray(grouped[m].values, dtype=float)
        if m in grouped.groups
        else np.array([])
        for m in np.arange(1, 13)
    ]
    return [values[np.isfinite(values)] for values in spread]


#: The three derivations above, by figure kind. The SINGLE definition of what
#: each figure plots -- the renderers draw from these and
#: ``climate_levels`` pools them across datasets to derive a shared scale, so a
#: scale cannot be computed over one quantity and applied to another.
VALUE_DERIVATIONS = {
    "map": map_field,
    "annual": annual_series,
    "monthly": lambda da, spec, anchor=None: monthly_spread(da, spec),
}


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


def _render_map(
    da,
    spec,
    title,
    caveat,
    overlays,
    extent_policy="basin",
    anchor=DEFAULT_WATER_YEAR_ANCHOR,
    scale=None,
    **_,
):
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
    field = map_field(da, spec, anchor)
    axis_unit = f"{unit} y$^{{-1}}$" if how == "sum" else unit

    base = RASTER_STYLES[spec["style"]]
    # The unit belongs to the DATA, not to the style: `how` decides whether the
    # field is a yearly total or a mean, so the label is built here.
    style = base.replace(label=f"{label.capitalize()} ({axis_unit})")
    if spec["style"] == "temp":
        style = resolve_temperature_style(field, style)
    if scale:
        # Pinned to the scale every other source's figure for this variable
        # draws against, so a difference between two maps is READABLE off the
        # colours. `RasterStyle.levels` bypasses the per-raster classifier,
        # which is what it exists for. AFTER resolve_temperature_style, whose
        # diverging branch would otherwise re-derive boundaries and discard
        # these.
        style = style.replace(levels=list(scale), diverging_center=None)

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
        # Per family — see MAP_EXTENT for why the two differ. ``None`` lets the
        # template frame on the raster's own footprint, which is what the source
        # side wants; the forcing side crops to the basin as before.
        extent=(
            extent_from_layer(basins)
            if extent_policy == "basin" and has_basins
            else None
        ),
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


#: Where a series figure's footnote starts, in figure coordinates.
#:
#: LEFT-ALIGNED (owner ruling 2026-08-17): a footnote is read as prose, and
#: prose starts where the reader's eye already is — under the y-axis label,
#: flush with the left edge of the sheet. ``supxlabel``'s default centring puts
#: it under the axes midpoint, which is not an edge the reader can see and which
#: MOVES with the caveat's own length, so a one-line and a two-line footnote sit
#: differently on two figures of the same family.
#:
#: The MAP family is right-aligned instead, deliberately and for a reason that
#: does not apply here: it flushes to the side panel's measured right edge
#: (``cartographic_map._align_caveat_to_panel``), which IS a visible edge shared
#: by the locator inset and the legend. A series figure has no side panel.
CAVEAT_X = 0.012


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
            # `x` and `ha` together: constrained layout rewrites a supxlabel's Y
            # on every draw but carries the X through, which is what makes this
            # survive the save -- the same property `_align_caveat_to_panel`
            # relies on for the maps.
            fig.supxlabel(
                caveat,
                fontsize=FONT_SIZE_CAVEAT,
                color=COLOR_CAVEAT,
                wrap=True,
                x=CAVEAT_X,
                ha="left",
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


def _render_annual(
    da, spec, title, caveat, overlays, anchor=DEFAULT_WATER_YEAR_ANCHOR, scale=None, **_
):
    """Domain-mean value per year, with its trend and the period mean."""
    how, label, unit = spec["how"], spec["label"], spec["unit"]
    series = annual_series(da, spec, anchor)
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
    _apply_scale(ax, scale)
    _style_series_axes(ax)
    return fig


def _apply_scale(ax, scale) -> None:
    """Pin a series axis to the shared y-range, with headroom for the ink.

    A hair of padding, because the plotted extremes are the scale's own
    endpoints by construction: without it the highest marker or whisker cap sits
    exactly on the frame and reads as clipped.
    """
    if not scale:
        return
    lower, upper = (float(v) for v in scale)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return
    pad = (upper - lower) * 0.04
    ax.set_ylim(lower - pad, upper + pad)


def _render_monthly(da, spec, title, caveat, overlays, scale=None, **_):
    """Monthly climatology of the domain mean, and its year-to-year spread.

    The mean alone answered "when is the wet season?" and nothing about how
    reliably — two basins with the same climatology and very different
    interannual spread drew the same figure. The boxes are the distribution
    ACROSS YEARS for each calendar month, so the reader sees both.
    """
    how, label, unit = spec["how"], spec["label"], spec["unit"]
    months = np.arange(1, 13)
    spread = monthly_spread(da, spec)
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
    _apply_scale(ax, scale)
    _style_series_axes(ax)
    return fig


_RENDERERS = {
    "map": _render_map,
    "annual": _render_annual,
    "monthly": _render_monthly,
}


# A `climate_levels.json` sidecar lived here until 2026-08-16: the SOURCE
# figures recorded their class boundaries and the FORCING figures adopted them,
# so a variable's two maps carried one colourbar and could be read against each
# other. It went with the extent ruling above (see MAP_EXTENT). Once the two
# families frame different footprints a shared bar is not a convenience but a
# defect — it would be classified on the source extraction and applied to a
# basin-cropped forcing field, so the forcing map would spend most of its ramp
# on values that occur only outside the catchment. Each family now classifies
# what it actually draws.
#
# What this gave up, stated plainly: the two maps of a variable no longer share
# a scale, so a source/forcing difference can no longer be read off the colours
# alone. Rules 1.05 and 1.13 lost the DAG edge that carried the file.


def plot_climate_figures(
    ds: xr.Dataset,
    plot_dir: Union[str, Path],
    dataset: str,
    *,
    caveat: Optional[str] = None,
    overlays: Optional[dict] = None,
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
    variables: Optional[Sequence[str]] = None,
    levels: Optional[Mapping] = None,
    clim_source: Optional[str] = None,
    area_masks: Optional[Mapping] = None,
    subbasin_dir: Optional[Union[str, Path]] = None,
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
    clim_source : str, optional
        The source id. Present ⇒ the SOURCE family, named under the WF0 grammar
        with this as the ``dataset_scope``; absent ⇒ the legacy
        ``<dataset>_<var>_<kind>.png`` spelling the forcing family keeps.
    area_masks : mapping, optional
        ``{spatial_scope: mask_or_None}`` for the AGGREGATED kinds — typically
        ``{"basin_avg": <basin mask>}`` plus one entry per subbasin. Each mask
        restricts the domain mean to that area's cells; ``None`` means the full
        extraction, which is what a caller with no basin geometry gets. The map
        kind ignores this entirely: it draws the field, not a reduction of it.
    subbasin_dir : str | Path, optional
        Where the per-subbasin figures land. They are written to their own
        directory because their COUNT is a property of the delineation and so is
        unknown when the Snakefile declares its outputs — the rule declares this
        directory instead, which is the same trade rule 1.15 records for its
        per-station figures. Defaults to ``plot_dir``.

    Returns
    -------
    list[Path]
        The figures written, in :func:`source_figure_names` order for the source
        family and :func:`figure_names` order for the forcing one.

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
    draw = _resolve_variables(variables)
    missing = [var for var in draw if var not in ds]
    if missing:
        raise ValueError(
            f"plot_climate_figures: dataset {dataset!r} is missing {missing}; "
            f"the requested set needs {list(draw)}"
        )

    plot_dir = Path(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    title = DATASETS[dataset]
    extent_policy = MAP_EXTENT[dataset]
    # One scope, unmasked, when the caller supplied none -- which is what keeps
    # the forcing family (rule 1.13) drawing exactly what it drew before.
    scopes = dict(area_masks) if area_masks else {"basin_avg": None}
    subbasin_dir = Path(subbasin_dir) if subbasin_dir is not None else plot_dir
    if any(scope.startswith("subbasin_") for scope in scopes):
        os.makedirs(subbasin_dir, exist_ok=True)
    written = []
    for var in draw:
        spec = CLIMATE_VARS[var]
        for kind in FIGURE_KINDS:
            context = KIND_PLOT_CONTEXTS[kind]
            if kind == "map":
                # Drawn ONCE, from the whole field: a map is the spatial view,
                # so reducing it to an area is what the other kinds are for.
                fig = _RENDERERS[kind](
                    ds[var],
                    spec,
                    title,
                    caveat,
                    overlays,
                    extent_policy=extent_policy,
                    anchor=anchor,
                    # Absent for this (var, kind) -> the renderer classifies
                    # from its own data, right for a single-source run.
                    scale=(levels or {}).get(var, {}).get(kind),
                )
                name = (
                    figure_filename(
                        clim_source, var, context, map_spatial_scope(extent_policy)
                    )
                    if clim_source
                    else _legacy_figure_name(dataset, var, kind)
                )
                out_path = plot_dir / name
                save_figure(out_path, dpi=RASTER_DPI)
                plt.close(fig)
                written.append(out_path)
                continue

            for scope, mask in scopes.items():
                fig = _RENDERERS[kind](
                    masked(ds, mask)[var],
                    spec,
                    title,
                    # The AREA belongs on the figure, not only in the filename:
                    # a file copied out of its directory keeps its footnote and
                    # loses its path.
                    scope_caveat(caveat, scope, mask) if clim_source else caveat,
                    overlays,
                    extent_policy=extent_policy,
                    anchor=anchor,
                    scale=(levels or {}).get(var, {}).get(kind),
                )
                if clim_source:
                    name = figure_filename(clim_source, var, context, scope)
                    # Per-subbasin figures sit in their own bin; see
                    # `subbasin_dir`. The basin-level ones stay beside the maps.
                    destination = (
                        subbasin_dir if scope.startswith("subbasin_") else plot_dir
                    )
                else:
                    name = _legacy_figure_name(dataset, var, kind)
                    destination = plot_dir
                out_path = destination / name
                save_figure(out_path, dpi=RASTER_DPI)
                plt.close(fig)
                written.append(out_path)
    # The directory is deliberately NOT repeated here: every figure above came
    # through `save_figure`, whose first row of the group already states it.
    # What this row adds is the COUNT -- "did I get all of them" without
    # counting rows -- and which dataset they belong to.
    log_row(
        f"Wrote {len(written)} canonical climate figures "
        f"({clim_source or dataset}, {len(scopes)} spatial scope(s))",
        module="plot",
    )
    return written
