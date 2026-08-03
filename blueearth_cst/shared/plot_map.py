# -*- coding: utf-8 -*-
"""Plot the wflow basin, rivers, gauges/outlets and DEM on a map (rule 1.12).

Created 2022-01-13 (@author: bouaziz); refactored in R3 into a guarded
function so the rule's output can be tee'd to its log and the module stays
importable; rebuilt in 2026-08 as a publication-grade figure.

What changed in the 2026-08 pass, and why each was a defect rather than a
matter of taste:

* **No basemap tiles.** ``cartopy.io.img_tiles.QuadtreeTiles`` fetched live
  satellite tiles mid-run. That is a NETWORK dependency inside WF1, a tile
  licence/attribution question for anything submitted to a journal, and — since
  the server may re-render at any time — a figure that cannot be reproduced.
  The terrain context it bought is now drawn from the model's OWN DEM as a
  hillshade, which is reproducible, offline, and higher-resolution than the
  tiles were at the zoom level we hardcoded.
* **Final physical size.** Width is a declared millimetre constant, not a
  guessed ``figsize``, and height follows the basin's own aspect ratio — so the
  figure lands at a journal column width for ANY basin instead of being tuned
  to one.
* **A colourblind-safe elevation ramp.** ``plt.cm.terrain`` is not CVD-safe and
  its green-to-brown reads as land cover, which elevation is not.
* **A vector deliverable.** PDF (embedded TrueType, editable text) alongside
  the PNG preview: a 300-dpi raster of a line-art map is not a submittable
  figure.
* **Cartographic furniture.** A graticule, a latitude-corrected scale bar and a
  north arrow, replacing a ``seaborn-whitegrid`` panel grid — gridlines behind
  a map are a cartographic error, not a style.

The figure still depicts the MODEL, and is still read from ``WflowSbmModel``.
Rendering it from the engine-neutral ``spatial/`` products instead was
considered and rejected: waterbodies come from rule 1.04 and the gauge layer
from 1.05, and ``SpatialProducts`` carries neither, so an artifact-driven
version would silently drop layers this figure exists to show.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors, rc_context
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from shapely.geometry import box as shapely_box

from blueearth_cst.shared.snake_utils import save_figure

# ===========================================================================
# TUNABLE CONSTANTS
# ===========================================================================
# Everything a reader might want to adjust lives in this block; nothing below
# it hardcodes a size, weight, colour or position. Values are grouped by what
# they control, and each says what it affects so a change can be made without
# reading the drawing code.
#
# Two rules the whole block depends on, worth knowing before changing anything:
#
# * Lengths are in PHYSICAL units (mm / inches / points), not pixels. The
#   figure is built at its final printed size, so a font size is the size it
#   will be on the page. Raising PREVIEW_DPI makes the PNG bigger, NOT the type
#   smaller.
# * Positions inside the map are AXES FRACTIONS (0 = left/bottom, 1 =
#   right/top) and can exceed 1 to sit outside the map — that is how the side
#   panel works. Positions of map furniture drawn in data space (the scale bar)
#   are fractions of the map's own extent, so they hold for any basin.
# ---------------------------------------------------------------------------

# --- page and export -------------------------------------------------------

#: Figure width in MILLIMETRES — converted once, here, and never re-guessed
#: downstream. 180 mm is the double-column width that Elsevier (190), AGU (190)
#: and Copernicus (170) all accept without downscaling. Set this to your target
#: journal's column width; every other size is chosen to work at it.
FIGURE_WIDTH_MM = 180.0
MM_PER_INCH = 25.4

#: Raster preview resolution. The PDF is the deliverable; the PNG is what the
#: workflow's other consumers (baseline fingerprint, quick review) read. 300 is
#: the usual journal minimum for raster figures.
PREVIEW_DPI = 400

# --- typography ------------------------------------------------------------

#: Type sizes in POINTS at the printed width above. Applied through
#: ``rc_context`` so the process-wide rcParams the other plotting rules inherit
#: are left untouched. Raise every value together to scale the labelling; raise
#: one to re-balance it.
#: Fallback for anything not named below — which, as the figure currently
#: stands, is NOTHING: every text element carries its own size, the title is
#: empty and the axes have no labels. Changing it alone therefore renders
#: identical bytes. It matters only once a text element is added without a size
#: of its own. Verified 2026-08-03 with ``dev/scripts/preview_basin_map.py``.
FONT_SIZE_BASE = 8.0
FONT_SIZE_TICK = 7.0  #: coordinate tick labels
FONT_SIZE_LEGEND = 7.0  #: legend entries and its title
FONT_SIZE_COLORBAR_LABEL = 8.0  #: "elevation [m a.s.l.]"
FONT_SIZE_GAUGE_LABEL = 5.5  #: the wflow_id beside each gauge marker
FONT_SIZE_SCALE_BAR = 6.0  #: the 0 / 2.5 / 5 km numbers
FONT_SIZE_NORTH_ARROW = 7.5  #: the "N"

#: Font family. ``None`` keeps matplotlib's default (DejaVu Sans, which embeds
#: cleanly in the PDF). Set e.g. ``"Arial"`` or ``["Helvetica", "Arial"]`` to
#: match a manuscript — but check the exported PDF, because a missing family
#: falls back SILENTLY.
FONT_FAMILY = None

# --- layout ----------------------------------------------------------------

#: Vertical room (inches) constrained layout needs for the x tick labels and the
#: axes furniture, on top of the map panel itself. Measured need is ~0.16 in of
#: tick labels; the rest is margin. Over-allowing here shows up directly as dead
#: space above and below an aspect-locked map.
_FURNITURE_HEIGHT_IN = 0.32

#: Horizontal room (inches) the y tick labels take on the left. Raise it if a
#: basin's coordinates need more decimal places than the default formatting.
_TICK_LABEL_WIDTH_IN = 0.5

#: Constrained layout owns the figure only up to here; the strip to the right is
#: a SIDE PANEL holding the colourbar and, beneath it, the legend. A GeoAxes has
#: a LOCKED aspect, so it does not fill its layout cell vertically — and
#: ``fig.colorbar(ax=ax)`` sizes to the CELL, which is what made the bar overhang
#: the map top and bottom. Both panel items are therefore anchored in AXES
#: coordinates (so they track the map exactly) and this rect reserves the room.
#: LOWER it to widen the panel (a longer legend entry needs more), RAISE it to
#: give the map more width.
_LAYOUT_RIGHT = 0.78

#: Keep pathological basin shapes from producing an unusable page. A basin
#: narrower or taller than these renders with whitespace rather than being
#: squashed or running off the figure.
_MIN_MAP_ASPECT, _MAX_MAP_ASPECT = 0.45, 1.45

# --- side panel: colourbar and legend --------------------------------------

#: Left edge of the side panel, in axes fractions (>1 = outside the map). The
#: colourbar and the legend BOTH start here — ONE value, so they cannot drift
#: out of alignment. Raise it to push the panel further from the map.
_PANEL_LEFT = 1.03

#: Colourbar geometry in axes fractions: (bottom, width, height). Height 0.5
#: spans the upper half of the map's height; the legend occupies the rest.
_COLORBAR_BOTTOM = 0.5
_COLORBAR_WIDTH = 0.025
_COLORBAR_HEIGHT = 0.5
_COLORBAR_OUTLINE_WIDTH = 0.5
#: Upper and lower quantiles of the DEM the ramp spans. The upper clip stops a
#: single high pixel flattening the rest of the basin to one colour.
_ELEVATION_CLIP_QUANTILES = (0.0, 0.98)

#: Top of the legend box, in axes fractions — just below the colourbar's lower
#: end (``_COLORBAR_BOTTOM``). Lower it to open a gap between the two.
_LEGEND_TOP = 0.44
_LEGEND_FRAME_ALPHA = 0.85  #: 1.0 = opaque, 0.0 = no fill
_LEGEND_FRAME_WIDTH = 0.5  #: border weight, points
_LEGEND_BORDER_PAD = 0.4  #: padding inside the frame, in font units
_LEGEND_HANDLE_LENGTH = 1.4  #: length of the sample line/marker, in font units
_LEGEND_TITLE = "Legend"  #: set to None to drop the title row

# --- colours ---------------------------------------------------------------

#: One place for every hue on the figure. The blue is used for BOTH the rivers
#: and the user's gauges, which is deliberate: it ties a gauge to the network it
#: sits on and separates it from the model's own outlets, which stay black.
COLOR_RIVER = "#2c6fad"
COLOR_GAUGE = "#2c6fad"
COLOR_OUTLET = "k"
COLOR_BASIN_OUTLINE = "k"
COLOR_SUBCATCHMENT = "0.45"
COLOR_GRATICULE = "0.4"
COLOR_MARKER_EDGE = "white"
#: Halo drawn behind furniture text so it stays legible over any terrain.
COLOR_HALO = "white"

#: Waterbody fills, as (facecolor, edgecolor). Keyed by the staticgeoms layer.
WATERBODY_COLORS = {
    "lakes": ("#a8d0e6", "#3d5a6c"),
    "reservoirs": ("#2c6fad", "#173d5e"),
    "glaciers": ("#d9d9d9", "#8c8c8c"),
}

#: A monotonic-lightness elevation ramp, hand-built rather than imported: the
#: perceptually-uniform terrain colormaps (cmcrameri, cmocean) are not in the
#: pixi env and adding a dependency for one figure is not warranted. Lightness
#: falls monotonically from low to high ground, so the ramp survives greyscale
#: printing AND every dichromacy — the two failure modes `terrain` had. Replace
#: it only with another ramp whose lightness is monotonic; a test enforces that.
_DEM_ANCHORS = ("#f6f2ea", "#e3d5ba", "#c9aa7d", "#a07f52", "#6f5533", "#46351f")

# --- line weights (points) -------------------------------------------------

#: River width scales with Strahler stream order, between these two bounds. The
#: minimum is what a headwater gets, the maximum the trunk — widen the gap for a
#: more dramatic network, narrow it for a flatter, more uniform one.
RIVER_WIDTH_MIN = 0.2
RIVER_WIDTH_MAX = 1.2
#: Used when every river shares one stream order, so there is nothing to scale.
RIVER_WIDTH_UNIFORM = 0.6

WIDTH_BASIN_OUTLINE = 0.9  #: the dissolved outer boundary — the map's key line
WIDTH_SUBCATCHMENT = 0.35  #: internal divides, deliberately much lighter
WIDTH_WATERBODY_EDGE = 0.5
WIDTH_MARKER_EDGE = 0.4
WIDTH_AXES_SPINE = 0.6
WIDTH_GRATICULE = 0.3
#: Dash pattern for the subcatchment divides, matplotlib ``(offset, (on, off))``.
DASH_SUBCATCHMENT = (0, (4, 2))
#: Halo stroke widths, points. The halo must exceed the line it protects.
HALO_WIDTH_TEXT = 2.5
HALO_WIDTH_GAUGE_LABEL = 1.8

# --- markers ---------------------------------------------------------------

MARKER_SHAPE = "d"  #: diamond, for both outlets and gauges
MARKER_SIZE = 18  #: matplotlib points-squared, as geopandas expects
#: Offset of a gauge's label from its marker, in points (x, y).
GAUGE_LABEL_OFFSET = (2.5, 2.5)

# --- graticule -------------------------------------------------------------

GRATICULE_ALPHA = 0.5
GRATICULE_LINESTYLE = ":"
#: Upper bound on tick count per axis; the locator picks round values under it.
GRATICULE_MAX_TICKS = 6
TICK_LENGTH = 2.5  #: points
TICK_PAD = 2.0  #: gap between tick and label, points

# --- scale bar -------------------------------------------------------------

#: Alternating filled/open segments, the conventional cartographic scale bar.
#: Must be EVEN for the midpoint label to land on a segment boundary.
_SCALE_BAR_SEGMENTS = 4
#: Bar height as a fraction of the map's latitude span.
_SCALE_BAR_HEIGHT = 0.011
#: Target bar length as a fraction of the map width, before rounding to a 1/2/5
#: value. Raise it for a longer, more precisely readable bar.
_SCALE_BAR_WIDTH_FRACTION = 0.25
#: Inset of the bar from its chosen corner, as a fraction of the map extent.
_SCALE_BAR_INSET = 0.06
#: Gap between the bar and its numbers, as a fraction of the latitude span.
_SCALE_BAR_LABEL_GAP = 0.008
_SCALE_BAR_EDGE_WIDTH = 0.5

# --- north arrow -----------------------------------------------------------

#: Arrow position in axes fractions: (x, tip y, tail y). The "N" sits at the
#: tail. Exactly vertical is correct here because PlateCarree's north is up.
_NORTH_ARROW_POSITION = (0.955, 0.94, 0.83)
_NORTH_ARROW_STYLE = "-|>"
_NORTH_ARROW_WIDTH = 0.8
#: The arrow's corner, kept clear of the scale bar. With the legend in the side
#: panel, this is the only reserved corner left on the map.
_NORTH_ARROW_CORNER = "upper right"

# --- furniture placement ---------------------------------------------------

#: Lower-left corner of each candidate furniture box, as a fraction of the map
#: extent. Names are matplotlib ``legend(loc=...)`` values verbatim.
_CORNER_BOX = 0.30
_CORNERS = {
    "lower left": (0.0, 0.0),
    "lower right": (1.0 - _CORNER_BOX, 0.0),
    "upper left": (0.0, 1.0 - _CORNER_BOX),
    "upper right": (1.0 - _CORNER_BOX, 1.0 - _CORNER_BOX),
}

# --- hillshade -------------------------------------------------------------

#: Illumination: light from the north-west at 45 deg, the convention readers'
#: relief perception is calibrated to (lit NW = ridge, shaded SE = valley;
#: reverse it and terrain visually inverts).
_AZIMUTH_DEG, _ALTITUDE_DEG = 315.0, 45.0

#: Target 90th-percentile terrain slope AFTER exaggeration, ~19 deg — steep
#: enough to read as relief, shallow enough not to fabricate mountains. The
#: exaggeration factor is derived per basin: CST runs on lowland deltas and on
#: Himalayan headwaters from the same code, and any FIXED factor renders one of
#: them featureless (a flat basin at exag 3) or blown out (an alpine basin at
#: exag 200). Raise for more dramatic relief, lower for a flatter, calmer map.
_TARGET_SLOPE = 0.35
_MAX_VERT_EXAG = 500.0
#: How the ramp and the shading combine. "soft" keeps colour; "overlay" is
#: higher contrast; "hsv" is the most dramatic and the least faithful.
_SHADE_BLEND_MODE = "soft"

# --- data / labels ---------------------------------------------------------

#: Gauge marker label. ``wflow_id`` is what the wflow output columns
#: (``Q_101``) and the observation file's rows are keyed on, so it is the label
#: that lets a reader join this map to a hydrograph. ``station_name`` is longer,
#: collides more, and answers a question the caption can answer instead — swap
#: it here if the names matter more than the join.
_GAUGE_LABEL_COLUMN = "wflow_id"

#: Padding around the model's own bounding box, in degrees, so the basin does
#: not touch the frame.
_EXTENT_BUFFER_DEG = 0.02

#: Drawing order. Every artist names one of these rather than a bare number, so
#: the stack is legible and reorderable in one place.
Z_RELIEF = 1
Z_RIVER = 3
Z_WATERBODY = 4
Z_SUBCATCHMENT = 5
Z_BASIN_OUTLINE = 6
Z_MARKER = 7
Z_FURNITURE = 8

_EARTH_RADIUS_M = 6_371_000.0

# ===========================================================================
# DERIVED VALUES
# ===========================================================================
# Anything assembled FROM the block above is derived in a function, never
# frozen into a module-level constant. A constant would snapshot its inputs at
# import time, so overriding e.g. FONT_SIZE_BASE afterwards would change
# nothing — which is precisely how `dev/scripts/preview_basin_map.py` drives
# this module. Keep that property when adding a value: derive it here.
# ---------------------------------------------------------------------------


def _colorbar_inset():
    """[x0, y0, width, height] for ``ax.inset_axes``, in axes fractions."""
    return (_PANEL_LEFT, _COLORBAR_BOTTOM, _COLORBAR_WIDTH, _COLORBAR_HEIGHT)


def _publication_rc():
    """The rcParams the figure is drawn under, from the FONT_SIZE_*/WIDTH_*."""
    return {
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_BASE + 1.0,
        "axes.labelsize": FONT_SIZE_BASE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.title_fontsize": FONT_SIZE_LEGEND,
        "axes.linewidth": WIDTH_AXES_SPINE,
        "xtick.major.width": WIDTH_AXES_SPINE,
        "ytick.major.width": WIDTH_AXES_SPINE,
        # 42 = TrueType. The default (Type 3) is not editable in Illustrator and
        # is rejected outright by several publishers' preflight.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        **({"font.family": FONT_FAMILY} if FONT_FAMILY else {}),
    }


def _metres_per_degree(latitude_deg):
    """Metres per degree of longitude and of latitude at ``latitude_deg``.

    Both the hillshade and the scale bar need this. The model grid is
    EPSG:4326, where a cell's ``dx`` is an ANGLE: feeding degrees to a gradient
    that expects metres exaggerates relief by ~10^5, and a scale bar drawn as a
    fixed number of degrees is wrong everywhere except the equator.
    """
    metres_per_degree_lat = np.pi * _EARTH_RADIUS_M / 180.0
    metres_per_degree_lon = metres_per_degree_lat * np.cos(np.radians(latitude_deg))
    return float(metres_per_degree_lon), float(metres_per_degree_lat)


def _nice_round_length(value_km):
    """Round a scale-bar length down to the nearest 1/2/5 x 10^n."""
    if value_km <= 0:
        return 1.0
    exponent = np.floor(np.log10(value_km))
    fraction = value_km / 10.0**exponent
    step = 5.0 if fraction >= 5.0 else (2.0 if fraction >= 2.0 else 1.0)
    return float(step * 10.0**exponent)


def _elevation_colormap():
    """The CVD-safe elevation ramp as a matplotlib colormap."""
    return colors.LinearSegmentedColormap.from_list("dem_cvd", _DEM_ANCHORS, N=256)


def _vertical_exaggeration(elevation, dx_metres, dy_metres, valid=None):
    """Exaggeration that renders THIS basin's relief legibly.

    Scales the DEM's own 90th-percentile slope onto ``_TARGET_SLOPE``, so the
    hillshade reads the same whether the basin drops 130 m over 24 km or
    3000 m over 20 km.

    ``valid`` masks the cells INSIDE the basin. Measuring the slope over the
    whole bounding box instead would let the same basin shade differently
    depending on how much nodata its box happens to contain — the flat fill
    drags the percentile down, and the exaggeration up.
    """
    gradient_y, gradient_x = np.gradient(elevation, dy_metres, dx_metres)
    slope = np.hypot(gradient_x, gradient_y)
    if valid is not None:
        slope = np.where(valid, slope, np.nan)
        if not np.any(np.isfinite(slope)):
            return 1.0
    typical_slope = float(np.nanpercentile(slope, 90))
    if not np.isfinite(typical_slope) or typical_slope <= 0.0:
        return 1.0
    return float(np.clip(_TARGET_SLOPE / typical_slope, 1.0, _MAX_VERT_EXAG))


def _shaded_relief(da, cmap, norm, latitude_deg):
    """Drape the elevation ramp over a hillshade of the same DEM.

    Returns an RGBA ``DataArray``: this replaces the satellite basemap, so it
    has to carry the terrain context on its own.
    """
    # LightSource reads the array as (row, column) = (y, x) and takes dx/dy in
    # that order, so put the DEM in y-major order before touching it rather than
    # assuming the model wrote it that way.
    da = da.transpose(da.raster.y_dim, da.raster.x_dim)
    resolution_x, resolution_y = (abs(float(value)) for value in da.raster.res)
    metres_per_degree_lon, metres_per_degree_lat = _metres_per_degree(latitude_deg)
    light = colors.LightSource(azdeg=_AZIMUTH_DEG, altdeg=_ALTITUDE_DEG)
    # LightSource cannot see NaN; fill with the basin minimum so the boundary
    # does not shade as a cliff, then restore the mask through alpha.
    values = da.values
    inside_basin = ~np.isnan(values)
    filled = np.where(inside_basin, values, float(np.nanmin(values)))
    dx_metres = resolution_x * metres_per_degree_lon
    dy_metres = resolution_y * metres_per_degree_lat
    rgba = light.shade(
        filled,
        cmap=cmap,
        norm=norm,
        blend_mode=_SHADE_BLEND_MODE,
        dx=dx_metres,
        dy=dy_metres,
        vert_exag=_vertical_exaggeration(filled, dx_metres, dy_metres, inside_basin),
    )
    rgba[..., 3] = inside_basin.astype(float)
    # Carry the DEM's OWN dimension names through: hydromt spells them
    # latitude/longitude here, not y/x, and hardcoding y/x raises KeyError.
    return xr.DataArray(
        rgba,
        dims=(*da.dims, "band"),
        coords={dim: da[dim] for dim in da.dims if dim in da.coords},
    )


def _figure_size(extent):
    """Figure size in inches: declared width, height from the basin's aspect."""
    lon_min, lon_max, lat_min, lat_max = extent
    width_in = FIGURE_WIDTH_MM / MM_PER_INCH
    # cartopy locks a PlateCarree panel to equal DEGREES, so the rendered map
    # aspect is the extent's own ratio -- not the true ground aspect.
    span_lon = max(float(lon_max - lon_min), 1e-9)
    aspect = float(lat_max - lat_min) / span_lon
    aspect = float(np.clip(aspect, _MIN_MAP_ASPECT, _MAX_MAP_ASPECT))
    # Height follows the MAP PANEL, not the full page: sizing off the figure
    # width leaves the aspect-locked panel floating in an over-tall cell.
    panel_width_in = width_in * _LAYOUT_RIGHT - _TICK_LABEL_WIDTH_IN
    return width_in, panel_width_in * aspect + _FURNITURE_HEIGHT_IN


def _coordinate_format(span_degrees):
    """Decimal places that suit the basin's size, not every basin's."""
    if span_degrees > 5.0:
        return ".0f"
    return ".1f" if span_degrees > 1.0 else ".2f"


def _graticule_ticks(extent, max_ticks=GRATICULE_MAX_TICKS):
    """Shared tick positions for the grid LINES and the axis LABELS."""
    lon_min, lon_max, lat_min, lat_max = extent
    locator = MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10])
    inside = lambda ticks, low, high: [t for t in ticks if low <= t <= high]
    return (
        inside(locator.tick_values(lon_min, lon_max), lon_min, lon_max),
        inside(locator.tick_values(lat_min, lat_max), lat_min, lat_max),
    )


def _add_graticule(ax, extent):
    """A light graticule, labelled through the normal tick machinery.

    The labels are REAL matplotlib ticks rather than ``gridlines(draw_labels=
    True)``. Cartopy's Gridliner labels are invisible to constrained layout,
    which reserves no room for them: observed here as latitude labels placed at
    x = -160 px, i.e. silently clipped off the canvas, on a figure whose
    longitude labels rendered fine. Ticks report their extent to the layout
    engine, so the room is reserved. Both consume the same tick list, so the
    lines and the labels cannot drift apart.
    """
    lon_ticks, lat_ticks = _graticule_ticks(extent)
    ax.gridlines(
        xlocs=lon_ticks,
        ylocs=lat_ticks,
        draw_labels=False,
        linewidth=WIDTH_GRATICULE,
        color=COLOR_GRATICULE,
        alpha=GRATICULE_ALPHA,
        linestyle=GRATICULE_LINESTYLE,
    )
    plate_carree = ccrs.PlateCarree()
    ax.set_xticks(lon_ticks, crs=plate_carree)
    ax.set_yticks(lat_ticks, crs=plate_carree)
    lon_min, lon_max, lat_min, lat_max = extent
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(number_format=_coordinate_format(lon_max - lon_min))
    )
    ax.yaxis.set_major_formatter(
        LatitudeFormatter(number_format=_coordinate_format(lat_max - lat_min))
    )
    # The formatters already spell out E/N, so an axis label would only repeat
    # them — the panel grid and the "longitude [degree east]" labels this
    # replaces were the two things making the old figure read as a plot of
    # coordinates rather than a map.
    ax.tick_params(length=TICK_LENGTH, pad=TICK_PAD)

    # An L-frame on the labelled sides only. A GeoAxes draws its box through the
    # single ``geo`` spine, so hiding that is what lets the four ordinary spines
    # be controlled individually — setting top/right invisible while ``geo`` is
    # still on leaves the full box drawn.
    ax.spines["geo"].set_visible(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(WIDTH_AXES_SPINE)
        spine.set_color(COLOR_BASIN_OUTLINE)


def _corner_occupancy(basin_geometry, extent):
    """Fraction of each corner box the basin covers.

    Fixed corners are only safe for basins shaped like the one they were tuned
    on. A basin that fills its bounding box, or simply carries mass in the
    south-west, gets the scale bar and the opaque legend frame drawn over its
    own rivers.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    span_lon, span_lat = lon_max - lon_min, lat_max - lat_min
    occupancy = {}
    for name, (x_fraction, y_fraction) in _CORNERS.items():
        corner = shapely_box(
            lon_min + x_fraction * span_lon,
            lat_min + y_fraction * span_lat,
            lon_min + (x_fraction + _CORNER_BOX) * span_lon,
            lat_min + (y_fraction + _CORNER_BOX) * span_lat,
        )
        area = corner.area
        occupancy[name] = (
            corner.intersection(basin_geometry).area / area if area > 0 else 1.0
        )
    return occupancy


def _scale_bar_corner(basin_geometry, extent, excluded=_NORTH_ARROW_CORNER):
    """The emptiest corner left for the scale bar, ties broken toward the bottom.

    ``excluded`` is the north arrow's corner. The legend used to be excluded too,
    but it now lives in the side panel rather than on the map, which gives the
    bar back a lower corner it previously had to yield.

    Ties are rounded before ranking so "equally empty" really does fall through
    to the bottom preference, and the corner name breaks the last tie so the
    figure never depends on dict iteration order.
    """
    occupancy = _corner_occupancy(basin_geometry, extent)
    candidates = [name for name in _CORNERS if name != excluded]
    return min(
        candidates,
        key=lambda name: (
            round(occupancy[name], 3),
            0 if name.startswith("lower") else 1,
            name,
        ),
    )


def _add_scale_bar(ax, extent, corner="lower left"):
    """A scale bar in kilometres, corrected for the basin's latitude."""
    lon_min, lon_max, lat_min, lat_max = extent
    metres_per_degree_lon, _ = _metres_per_degree(0.5 * (lat_min + lat_max))
    span_lon, span_lat = lon_max - lon_min, lat_max - lat_min
    map_width_km = span_lon * metres_per_degree_lon / 1000.0
    length_km = _nice_round_length(_SCALE_BAR_WIDTH_FRACTION * map_width_km)
    length_deg = length_km * 1000.0 / metres_per_degree_lon

    if corner.endswith("right"):
        x_start = lon_max - _SCALE_BAR_INSET * span_lon - length_deg
    else:
        x_start = lon_min + _SCALE_BAR_INSET * span_lon
    if corner.startswith("upper"):
        y_bar = lat_max - (_SCALE_BAR_INSET + 0.04) * span_lat
    else:
        y_bar = lat_min + _SCALE_BAR_INSET * span_lat

    # Alternating filled and open segments — the conventional bar, which lets a
    # reader step off a distance rather than only read the total.
    height = _SCALE_BAR_HEIGHT * span_lat
    segment_deg = length_deg / _SCALE_BAR_SEGMENTS
    halo = [pe.withStroke(linewidth=HALO_WIDTH_TEXT, foreground=COLOR_HALO)]
    for index in range(_SCALE_BAR_SEGMENTS):
        ax.add_patch(
            mpatches.Rectangle(
                (x_start + index * segment_deg, y_bar),
                segment_deg,
                height,
                facecolor=COLOR_BASIN_OUTLINE if index % 2 == 0 else "white",
                edgecolor=COLOR_BASIN_OUTLINE,
                linewidth=_SCALE_BAR_EDGE_WIDTH,
                zorder=Z_FURNITURE,
            )
        )

    segment_km = length_km / _SCALE_BAR_SEGMENTS
    # Label the ends and the midpoint only: a tick under every segment boundary
    # crowds at this size, and the midpoint is what makes the segments countable.
    for step in (0, _SCALE_BAR_SEGMENTS // 2, _SCALE_BAR_SEGMENTS):
        value = step * segment_km
        ax.text(
            x_start + step * segment_deg,
            y_bar + height + _SCALE_BAR_LABEL_GAP * span_lat,
            f"{value:g}" if step < _SCALE_BAR_SEGMENTS else f"{value:g} km",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_SCALE_BAR,
            zorder=Z_FURNITURE,
            path_effects=halo,
        )


def _add_north_arrow(ax):
    """A north arrow — exactly vertical, which PlateCarree guarantees.

    Top-right is always free: the legend is pinned bottom-right and the scale
    bar never takes an upper corner unless both lower ones are occupied.
    """
    x_fraction, tip_y, tail_y = _NORTH_ARROW_POSITION
    ax.annotate(
        "N",
        xy=(x_fraction, tip_y),
        xytext=(x_fraction, tail_y),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE_NORTH_ARROW,
        fontweight="bold",
        zorder=Z_FURNITURE,
        arrowprops=dict(
            arrowstyle=_NORTH_ARROW_STYLE,
            facecolor=COLOR_BASIN_OUTLINE,
            edgecolor=COLOR_BASIN_OUTLINE,
            linewidth=_NORTH_ARROW_WIDTH,
        ),
        path_effects=[pe.withStroke(linewidth=HALO_WIDTH_TEXT, foreground=COLOR_HALO)],
    )


def _basin_outline(gdf_bas):
    """The basin's OUTER boundary, dissolved to a single polygon.

    ``mod.basins`` returns one polygon PER SUBCATCHMENT once gauges are burned
    into the subcatchment map — four for a four-gauge model, one for a model
    with no user gauges. Drawing them all at boundary weight makes an internal
    divide indistinguishable from the basin outline, which is the one line on
    this figure a reader has to be able to trust. Observed 2026-08-03 on a real
    four-gauge project; the single-basin fixture cannot surface it.
    """
    return gdf_bas.dissolve()


def _river_linewidths(gdf_riv):
    """Stream order rescaled to publication line weights.

    ``strord / 2`` was tuned to a 10x8-inch canvas; at 180 mm it draws an
    8th-order river as a 4 pt band that swallows the basin.
    """
    order = gdf_riv["strord"].astype(float).to_numpy()
    lowest, highest = float(np.nanmin(order)), float(np.nanmax(order))
    if not np.isfinite(lowest) or highest <= lowest:
        return np.full(order.shape, RIVER_WIDTH_UNIFORM)
    span = RIVER_WIDTH_MAX - RIVER_WIDTH_MIN
    return RIVER_WIDTH_MIN + span * (order - lowest) / (highest - lowest)


def plot_basin_map(project_dir, gauges_fn, plot_dir=None):
    """Render basin_area.{pdf,png} (DEM + rivers + basin + outlets/waterbodies).

    The gauge layer is resolved from the MODEL (``shared.gauges``), not from the
    configured filename: hydromt_wflow renames ``output_locations`` to
    ``output-locations``, and deriving the name here is what silently dropped
    the gauges from this figure (2026-08-01).
    """
    from hydromt_wflow import WflowSbmModel

    from blueearth_cst.shared.gauges import gauges_layer_name

    if plot_dir is None:
        # R07 B10: basin_area depicts the MODEL, not its evaluation, so it
        # sits at the model root's plots/ — not under evaluation/ (P1).
        plot_dir = f"{project_dir}/hydrology_model/plots"
    root = f"{project_dir}/hydrology_model"

    mod = WflowSbmModel(root, mode="r")

    # read and mask the model elevation
    da = mod.staticmaps.data["land_elevation"].raster.mask_nodata()
    da.attrs.update(long_name="elevation", units="m")
    # read/derive river geometries
    gdf_riv = mod.rivers
    # read/derive model basin boundary
    gdf_bas = mod.basins
    geoms = mod.geoms.data
    # we assume the model maps are in the geographic CRS EPSG:4326
    proj = ccrs.PlateCarree()
    extent = np.array(
        da.raster.box.buffer(_EXTENT_BUFFER_DEG).total_bounds
    )[[0, 2, 1, 3]]
    centre_latitude = 0.5 * float(extent[2] + extent[3])

    with rc_context(_publication_rc()):
        fig = plt.figure(figsize=_figure_size(extent), layout="constrained")
        fig.get_layout_engine().set(rect=(0.0, 0.0, _LAYOUT_RIGHT, 1.0))
        ax = fig.add_subplot(projection=proj)
        ax.set_extent(extent, crs=proj)

        # --- elevation, as shaded relief -------------------------------------
        vmin, vmax = (
            float(value)
            for value in da.quantile(list(_ELEVATION_CLIP_QUANTILES)).compute()
        )
        cmap = _elevation_colormap()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        _shaded_relief(da, cmap, norm, centre_latitude).plot.imshow(
            ax=ax,
            x=da.raster.x_dim,
            y=da.raster.y_dim,
            transform=proj,
            zorder=Z_RELIEF,
            add_labels=False,
        )
        # imshow of an RGBA array carries no mappable, so the colourbar needs an
        # explicit one — the ramp is the same object either way.
        colorbar_axes = ax.inset_axes(_colorbar_inset())
        # The side panel lives OUTSIDE the map axes but is anchored to it. Left
        # in the layout, its footprint inflates the axes' tight bbox, and
        # constrained layout answers by shrinking the map — in BOTH directions,
        # because the aspect is locked. Measured cost before this line: 0.69 in
        # of dead space above AND below a map 1.2 in narrower than its cell.
        # ``rect`` already reserves the panel's room, so it must not be counted
        # twice.
        colorbar_axes.set_in_layout(False)
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axes)
        colorbar.set_label("elevation [m a.s.l.]", fontsize=FONT_SIZE_COLORBAR_LABEL)
        colorbar.outline.set_linewidth(_COLORBAR_OUTLINE_WIDTH)
        colorbar_axes.tick_params(
            labelsize=FONT_SIZE_TICK, length=TICK_LENGTH, pad=TICK_PAD
        )

        # --- hydrography ------------------------------------------------------
        gdf_riv.plot(
            ax=ax,
            linewidth=_river_linewidths(gdf_riv),
            color=COLOR_RIVER,
            zorder=Z_RIVER,
            label="river",
        )
        # Subcatchment divides first and lighter, then the outline over them, so
        # the two are never confusable at the same weight.
        subcatchment_handles = []
        if len(gdf_bas) > 1:
            divide_style = dict(
                color=COLOR_SUBCATCHMENT,
                linewidth=WIDTH_SUBCATCHMENT,
                linestyle=DASH_SUBCATCHMENT,
            )
            gdf_bas.boundary.plot(ax=ax, zorder=Z_SUBCATCHMENT, **divide_style)
            subcatchment_handles.append(
                Line2D([], [], label="subcatchments", **divide_style)
            )
        _basin_outline(gdf_bas).boundary.plot(
            ax=ax,
            color=COLOR_BASIN_OUTLINE,
            linewidth=WIDTH_BASIN_OUTLINE,
            zorder=Z_BASIN_OUTLINE,
        )

        # Resolved against what the model actually holds; warns loudly (never
        # skips silently) when output_locations is set but no layer matches.
        gauges_name = gauges_layer_name(geoms, gauges_fn)
        if "outlets" in geoms:
            geoms["outlets"].plot(
                ax=ax,
                marker=MARKER_SHAPE,
                markersize=MARKER_SIZE,
                facecolor=COLOR_OUTLET,
                edgecolor=COLOR_MARKER_EDGE,
                linewidth=WIDTH_MARKER_EDGE,
                zorder=Z_MARKER,
                label="outlets",
            )
        if gauges_name is not None:
            geoms[gauges_name].plot(
                ax=ax,
                marker=MARKER_SHAPE,
                markersize=MARKER_SIZE,
                facecolor=COLOR_GAUGE,
                edgecolor=COLOR_MARKER_EDGE,
                linewidth=WIDTH_MARKER_EDGE,
                zorder=Z_MARKER,
                label="output locs",
            )
            if _GAUGE_LABEL_COLUMN in geoms[gauges_name].columns:
                geoms[gauges_name].apply(
                    lambda x: ax.annotate(
                        text=str(x[_GAUGE_LABEL_COLUMN]),
                        xy=x.geometry.coords[0],
                        xytext=GAUGE_LABEL_OFFSET,
                        textcoords="offset points",
                        fontsize=FONT_SIZE_GAUGE_LABEL,
                        fontweight="bold",
                        color=COLOR_BASIN_OUTLINE,
                        zorder=Z_MARKER,
                        path_effects=[
                            pe.withStroke(
                                linewidth=HALO_WIDTH_GAUGE_LABEL,
                                foreground=COLOR_HALO,
                            )
                        ],
                    ),
                    axis=1,
                )

        # --- waterbodies ------------------------------------------------------
        # manual patches for legend (geopandas/geopandas#660)
        patches = []
        for name, (face, edge) in WATERBODY_COLORS.items():
            if name not in geoms:
                continue
            kwargs = dict(
                facecolor=face,
                edgecolor=edge,
                linewidth=WIDTH_WATERBODY_EDGE,
                label=name,
            )
            geoms[name].plot(ax=ax, zorder=Z_WATERBODY, **kwargs)
            patches.append(mpatches.Patch(**kwargs))

        # --- cartographic furniture -------------------------------------------
        # The legend sits in the side panel, so it no longer competes for a map
        # corner. The scale bar is placed against the basin's ACTUAL footprint,
        # so it does not land on a basin that reaches into a bottom corner.
        bar_corner = _scale_bar_corner(gdf_bas.union_all(), extent)
        _add_graticule(ax, extent)
        _add_scale_bar(ax, extent, bar_corner)
        _add_north_arrow(ax)
        ax.set_title("")
        legend = ax.legend(
            handles=[
                *ax.get_legend_handles_labels()[0],
                *subcatchment_handles,
                *patches,
            ],
            title=_LEGEND_TITLE,
            # Anchored to the SAME x as the colourbar, in axes coordinates, so
            # their left edges align by construction rather than by tuning.
            loc="upper left",
            bbox_to_anchor=(_PANEL_LEFT, _LEGEND_TOP),
            borderaxespad=0.0,
            alignment="left",
            frameon=True,
            framealpha=_LEGEND_FRAME_ALPHA,
            edgecolor=COLOR_BASIN_OUTLINE,
            facecolor="white",
            borderpad=_LEGEND_BORDER_PAD,
            handlelength=_LEGEND_HANDLE_LENGTH,
        )
        legend.get_frame().set_linewidth(_LEGEND_FRAME_WIDTH)
        # Same reason as the colourbar: the panel's room is reserved by ``rect``,
        # so letting the layout engine also see the legend costs the map size.
        legend.set_in_layout(False)

        # --- deliverables ------------------------------------------------------
        # No bbox_inches="tight": it re-crops to the drawn content, which throws
        # away the declared 180 mm width. Constrained layout already fits the
        # furniture inside that width.
        save_figure(
            os.path.join(plot_dir, "basin_area.pdf"),
            fig=fig,
            # Drop the timestamp so two identical runs produce identical bytes.
            metadata={"CreationDate": None},
        )
        save_figure(
            os.path.join(plot_dir, "basin_area.png"),
            fig=fig,
            dpi=PREVIEW_DPI,
            # Same reason: the default embeds the matplotlib version, which
            # would move the baseline fingerprint on every env bump.
            metadata={"Software": None},
        )
        plt.close(fig)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            plot_basin_map(
                project_dir=sm.params.project_dir,
                gauges_fn=getattr(sm.input, "output_locations", None),
            )
