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

Two entry points, split along reading-vs-drawing:

* ``plot_basin_map(dem, rivers, basin, *, subbasins=..., gauges=..., ...)`` takes
  each map layer as its OWN argument, touches no filesystem, and returns
  ``(fig, ax)`` without saving. Every layer but the DEM, the rivers and the
  basin outline is optional. It has no wflow in it: any DEM and any set of
  GeoDataFrames plots, so the function is usable in another project — or
  shareable on its own — rather than only against a model directory.
* ``plot_basin_map_from_model(project_dir, gauges_fn, plot_dir)`` resolves a
  wflow model on disk into those arguments and writes ``basin_area.{pdf,png}``.
  This is what the Snakemake rule runs.

The split is what keeps the wflow-specific knowledge — where ``staticmaps.nc``
lives, that ``basins`` holds one polygon per subcatchment, that hydromt_wflow
renames ``output_locations`` — in ONE function, out of the drawing code.

The figure still depicts the MODEL, but it no longer needs ``hydromt`` to read
one. ``load_basin_layers`` opens the model's own files directly — ``xarray`` for
``staticmaps.nc``, ``geopandas`` for ``staticgeoms/*.geojson`` — so the whole
render path imports neither ``hydromt`` nor ``hydromt_wflow``, and anyone
holding those two artifacts can call it.

Verified equivalent before the switch, not assumed: rendering the same model
through ``WflowSbmModel`` and through the files produced byte-identical images
(0 differing pixels of 3,754,080). ``mod.rivers`` / ``mod.basins`` simply return
``geoms["rivers"]`` / ``geoms["basins"]`` when those layers exist on disk, which
is every model WF1 writes, and ``.raster.mask_nodata()`` is a no-op once xarray
has decoded ``_FillValue``.

What is genuinely given up: ``mod.rivers`` and ``mod.basins`` are FALLBACK
properties (``hydromt_wflow/wflow_base.py``), reconstructing the network from
the flow-direction raster via ``pyflwdir`` when the geojson is absent. A model
whose ``staticgeoms/`` lacks them now raises instead of being rebuilt — a
deliberate trade, and the reason ``load_basin_layers`` names the missing layers
rather than failing obscurely downstream.

Rendering from the engine-neutral ``spatial/`` products was a DIFFERENT proposal,
considered and rejected: waterbodies come from rule 1.04 and the gauge layer from
1.05, and ``SpatialUnits`` carries neither, so an artifact-driven version would
silently drop layers this figure exists to show. Reading ``staticgeoms/`` keeps
every one of them — they are all written there.
"""

import os
from pathlib import Path

import geopandas as gpd
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

#: Draw passes run before the figure is returned, so its layout is already
#: settled. Constrained layout converges in two here (measured); a third costs
#: ~0.2 s and changes nothing. Raise it only if a label still lands off-canvas.
_LAYOUT_SETTLE_PASSES = 2

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

#: Where the colourbar's label goes. ``"right"`` is matplotlib's own placement
#: for a vertical bar: alongside it, rotated 90°. ``"top"`` puts it above the
#: bar, HORIZONTAL and left-aligned to the bar's left edge — which is
#: ``_PANEL_LEFT``, the legend's anchor too, so the two line up. Prefer "top"
#: for a long label: rotated text is slower to read and a unit string in
#: brackets reads badly on its side.
COLORBAR_LABEL_POSITION = "top"

#: Gap between the bar and a "top" label, in points.
_COLORBAR_TITLE_PAD = 5.0

#: Height given up to a "top" label, in axes fractions PER LINE of it. The
#: bar's BOTTOM is pinned (``_COLORBAR_BOTTOM``), so this shortens it from
#: above — without it the bar reaches 1.0 and the label renders off the canvas.
#: Per line, because the label wraps: a two-line label needs twice the room, and
#: a fixed value would either clip the second line or leave a gap above a
#: one-line one.
_COLORBAR_TOP_LABEL_HEADROOM = 0.055

#: The values ``COLORBAR_LABEL_POSITION`` accepts.
_COLORBAR_LABEL_POSITIONS = ("right", "top")
#: Upper and lower quantiles of the DEM the ramp spans. The upper clip stops a
#: single high pixel flattening the rest of the basin to one colour.
_ELEVATION_CLIP_QUANTILES = (0.0, 0.98)

#: Target number of colour CLASSES. The ramp is stepped rather than continuous:
#: a reader cannot resolve a shade back to a number off a smooth ramp, but can
#: off a class, and stepped classes survive the greyscale print that a
#: continuous ramp turns to mush. The count is a target — the class WIDTH is
#: rounded to a 1/2/5 value first, so the boundaries are numbers worth printing
#: and the count lands near this rather than on it.
_COLORBAR_LEVELS = 7

#: Start the ramp at 0 m rather than at the basin's own lowest cell. Elevation
#: is measured from a datum, so a bar starting at 4 m invites the reader to
#: treat the basin floor as the zero of the scale. Set False to always spend the
#: whole ramp on the basin's actual range.
_ELEVATION_STARTS_AT_ZERO = True

#: ...but not when it would cost the map its resolution. A 1900-1960 m plateau
#: zeroed gets classes 0/500/1000/1500/2000 — the ENTIRE basin lands in one of
#: them and the map renders as a single flat colour. So the baseline drops to
#: zero only while the basin's own range stays at least this fraction of the
#: zero-based range; below it, the ramp starts at the basin's floor instead.
#: CST runs on lowland deltas and Himalayan headwaters from the same code, and
#: a rule tuned on one of them is not a rule.
_ZERO_BASELINE_MIN_SPAN_FRACTION = 0.35

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
#: 0.2 pt is below what most printers hold and vanishes on screen at any
#: reasonable zoom, so the headwaters of the network simply were not there.
RIVER_WIDTH_MIN = 0.5
RIVER_WIDTH_MAX = 1.4
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

#: Separate shapes for the two point layers. They were both thin diamonds,
#: separated by colour alone — which fails in greyscale, fails for a
#: dichromat, and is hard to tell apart at 5 pt anyway. Shape is the redundant
#: channel that fixes all three. Circle reads as a measurement point, square as
#: a structural one; swap them if a convention says otherwise.
MARKER_SHAPE_GAUGE = "o"
MARKER_SHAPE_OUTLET = "s"
#: matplotlib points-squared, as geopandas expects. 18 was ~4.2 pt across at
#: 180 mm, which disappears against the relief; 44 is ~6.6 pt.
MARKER_SIZE = 44
#: Offset of a gauge's label from its marker, in points (x, y). Must clear the
#: marker's RADIUS: at MARKER_SIZE 44 that is ~3.3 pt, and the old (2.5, 2.5)
#: put the text inside the symbol.
GAUGE_LABEL_OFFSET = (4.5, 3.5)

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
#: Tucked into the map's own top-right CORNER — it used to float short of it,
#: which read as an artist adrift over the basin rather than as furniture.
_NORTH_ARROW_POSITION = (0.975, 0.985, 0.885)
_NORTH_ARROW_STYLE = "-|>"
_NORTH_ARROW_WIDTH = 0.8
#: The arrow's corner, kept clear of the scale bar. With the legend in the side
#: panel, this is the only reserved corner left on the map.
_NORTH_ARROW_CORNER = "upper right"

# --- locator inset ---------------------------------------------------------
# A small map in a corner saying WHERE this basin is: land and sea, country
# lines, a few major cities, and a mark on the basin. It is an INSET rather
# than a widened frame on purpose — the elevation map keeps the whole panel and
# its own scale, and a basin with nothing within 500 km still gets an answer,
# which a zoomed-out background could not give.

#: Draw it at all. The layers come from a vendored Natural Earth extract
#: (``config/basemap/``); with that file absent the inset is skipped and a note
#: is printed, so a copy of this module taken to another project still renders.
LOCATOR_ENABLED = True

#: Half-width of the locator's window, in degrees, around the basin's centre.
#: 8 deg is roughly national-to-regional: big enough to reach a coast or a
#: capital in most of the world, small enough that the basin mark is not a
#: speck. Raise it for a continental frame, lower it for a provincial one.
_LOCATOR_SPAN_DEG = 8.0

#: The inset's width as a fraction of the map panel's width, and its inset from
#: the corner. Its HEIGHT is derived so the box comes out square on the page —
#: a square window drawn into a non-square box would otherwise letterbox.
_LOCATOR_WIDTH = 0.22
_LOCATOR_MARGIN = 0.025

#: Which corner it sits in, or ``"auto"`` for the emptiest one the north arrow
#: is not using. Auto by default and it matters more here than for the scale
#: bar: the inset is OPAQUE, so a fixed corner does not merely crowd the basin,
#: it hides part of it — observed covering a gauge on the first render. Ties go
#: to an upper corner, where a locator is conventionally read.
_LOCATOR_CORNER = "auto"

COLOR_LOCATOR_OCEAN = "#eef2f5"  #: sea; the palest thing on the figure
COLOR_LOCATOR_LAND = "#dcdcd8"  #: land, a shade darker so the coast reads
COLOR_LOCATOR_COAST = "0.55"  #: the land polygons' own edge IS the coastline
COLOR_LOCATOR_BORDER = "0.7"  #: country lines, lighter still
COLOR_LOCATOR_CITY = "0.35"
#: The "you are here" mark. The one warm accent on the figure, and the only
#: place red appears — it has to win against grey without competing with the
#: elevation ramp, which owns every brown.
COLOR_LOCATOR_BASIN = "#c0392b"

WIDTH_LOCATOR_COAST = 0.35
WIDTH_LOCATOR_BORDER = 0.3
WIDTH_LOCATOR_FRAME = 0.6

#: Cities are filtered by Natural Earth's own prominence rank (0 = most
#: prominent), then the largest few by population are kept. Both limits matter:
#: the rank keeps towns out, the count keeps a dense region from filling up.
_LOCATOR_CITY_MAX_SCALERANK = 3
_LOCATOR_MAX_CITIES = 5
_LOCATOR_CITY_MARKER_SIZE = 5
FONT_SIZE_LOCATOR_CITY = 4.5

#: Basin mark size in points-squared, and the label offset for a city name.
_LOCATOR_BASIN_MARKER_SIZE = 26
_LOCATOR_CITY_LABEL_OFFSET = (2.5, -1.0)

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
# These three are the DEFAULTS of ``plot_basin_map`` parameters, so a caller
# overrides them per call rather than by patching the module.

#: Gauge marker label. ``wflow_id`` is what the wflow output columns
#: (``Q_101``) and the observation file's rows are keyed on, so it is the label
#: that lets a reader join this map to a hydrograph. ``station_name`` is longer,
#: collides more, and answers a question the caption can answer instead — swap
#: it here if the names matter more than the join.
GAUGE_LABEL_COLUMN = "wflow_id"

#: Column whose values scale the river line weights. ``strord`` is wflow's
#: Strahler stream order. Any numeric column works; ``None``, or a column the
#: frame does not carry, draws every reach at ``RIVER_WIDTH_UNIFORM``.
RIVER_ORDER_COLUMN = "strord"

#: Colourbar label. Units are the DEM's, so change it with the DEM. Broken over
#: two lines: on top of a 0.025-wide bar, one long line runs far past the side
#: panel; the quantity above its unit is also the conventional way to set it.
ELEVATION_LABEL = "elevation\n[m a.s.l.]"

#: Padding around the model's own bounding box, in degrees, so the basin does
#: not touch the frame.
_EXTENT_BUFFER_DEG = 0.02

# ---------------------------------------------------------------------------
# Model layout on disk. These four names are hydromt_wflow's write conventions,
# not ours -- they are stated here, as constants, precisely BECAUSE this module
# now reads the files itself instead of asking hydromt where they are. If a
# future hydromt_wflow changes a name, this block is the whole blast radius.
# ---------------------------------------------------------------------------
#: Model root, relative to ``project_dir``. R9 P2 commit 1 moved it under
#: ``models/``. Kept as a default only -- the RULE passes ``model_dir``, and
#: this constant serves standalone callers (dev/scripts/preview_basin_map.py)
#: that have no rule to ask.
MODEL_DIRNAME = "models/hydrology/wflow"
#: Gridded model parameters; carries the DEM this figure shades.
STATICMAPS_FILENAME = "staticmaps.nc"
#: Vector layers, one GeoJSON per layer, stem == layer name.
STATICGEOMS_DIRNAME = "staticgeoms"
#: The DEM variable inside ``staticmaps.nc`` (a CSDMS Standard Name).
ELEVATION_VARIABLE = "land_elevation"

#: The vendored Natural Earth extract the locator inset draws. Provenance,
#: licence and the rebuild recipe are in that folder's README. Committed rather
#: than fetched so the figure needs no network — see the module docstring.
BASEMAP_PATH = Path(__file__).resolve().parents[2] / "config" / "basemap" / "natural_earth_50m.gpkg"
#: Layers the figure cannot be drawn without; everything else is optional.
REQUIRED_GEOM_LAYERS = ("rivers", "basins")

#: Dimension names treated as easting/northing, lowercased. hydromt's ``.raster``
#: accessor sniffed these for us; reading the file directly means saying which
#: spellings count. wflow writes ``latitude``/``longitude``.
_X_DIM_NAMES = ("x", "longitude", "lon")
_Y_DIM_NAMES = ("y", "latitude", "lat")

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


def _colorbar_label_position():
    """``COLORBAR_LABEL_POSITION``, validated.

    Raises rather than falling back: a typo that silently keeps the default
    placement is the failure mode a tuning knob must not have — the figure
    still renders, so nothing tells you the value did nothing.
    """
    if COLORBAR_LABEL_POSITION not in _COLORBAR_LABEL_POSITIONS:
        raise ValueError(
            f"COLORBAR_LABEL_POSITION={COLORBAR_LABEL_POSITION!r}; expected one "
            f"of {_COLORBAR_LABEL_POSITIONS}"
        )
    return COLORBAR_LABEL_POSITION


def _colorbar_inset(label_lines=1):
    """[x0, y0, width, height] for ``ax.inset_axes``, in axes fractions.

    A "top" label is drawn ABOVE the bar, so the bar gives up the room for it.
    The bottom edge is pinned, so the height shrinks and the top comes down.
    ``label_lines`` is how many lines that label wraps to, since each one costs
    the same again.
    """
    height = _COLORBAR_HEIGHT
    if _colorbar_label_position() == "top":
        height -= _COLORBAR_TOP_LABEL_HEADROOM * label_lines
    return (_PANEL_LEFT, _COLORBAR_BOTTOM, _COLORBAR_WIDTH, height)


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


def _elevation_colormap(levels=None):
    """The CVD-safe elevation ramp, continuous or cut into ``levels`` classes."""
    ramp = colors.LinearSegmentedColormap.from_list("dem_cvd", _DEM_ANCHORS, N=256)
    return ramp if levels is None else ramp.resampled(levels)


def _nice_step_up(value):
    """Round a class width UP to the nearest 1/2/5 x 10^n.

    Up, not down: ``_nice_round_length`` rounds down, which for a class width
    means MORE classes than asked for, and a bar of twelve near-identical
    browns is the thing the discretisation exists to avoid.
    """
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / 10.0**exponent
    step = 1.0 if fraction <= 1.0 else (2.0 if fraction <= 2.0 else (5.0 if fraction <= 5.0 else 10.0))
    return float(step * 10.0**exponent)


def _elevation_levels(dem):
    """Class boundaries for the ramp: round numbers, first and last included.

    The lowest is 0 by default (``_ELEVATION_STARTS_AT_ZERO``) and the highest
    is the first round boundary at or above the clipped top of the DEM, so the
    bar's two end labels are real bounds rather than the basin's own extremes
    printed to a decimal.
    """
    lower, upper = (
        float(value)
        for value in dem.quantile(list(_ELEVATION_CLIP_QUANTILES)).compute()
    )
    if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
        upper = lower + 1.0
    if _ELEVATION_STARTS_AT_ZERO and lower > 0.0:
        # Only while the basin still gets most of the ramp; see the constant.
        if (upper - lower) / upper >= _ZERO_BASELINE_MIN_SPAN_FRACTION:
            lower = 0.0
    step = _nice_step_up((upper - lower) / max(_COLORBAR_LEVELS, 1))
    # Put the boundaries on multiples of the step, so a basin floor of 1903 m
    # labels as 1900 rather than carrying its own arbitrary value up the bar.
    lower = float(np.floor(lower / step) * step)
    # ceil, then +1 boundary: the top class must CONTAIN the highest cell, not
    # end at it, or the summit renders as nodata.
    count = int(np.ceil((upper - lower) / step))
    return lower + step * np.arange(count + 1)


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


def spatial_dim_names(da):
    """The ``(x_dim, y_dim)`` names of a 2-D geographic ``DataArray``.

    Replaces ``da.raster.x_dim`` / ``.y_dim``. Raises rather than guessing: a
    silently wrong axis produces a transposed map, which is far harder to notice
    than an exception.
    """
    lowered = {str(dim).lower(): dim for dim in da.dims}
    x_dim = next((lowered[n] for n in _X_DIM_NAMES if n in lowered), None)
    y_dim = next((lowered[n] for n in _Y_DIM_NAMES if n in lowered), None)
    if x_dim is None or y_dim is None:
        raise ValueError(
            f"cannot identify the spatial dimensions of {da.dims}; expected one "
            f"of {_X_DIM_NAMES} and one of {_Y_DIM_NAMES}"
        )
    return x_dim, y_dim


def pixel_resolution(da):
    """Signed ``(res_x, res_y)`` cell size in degrees, as ``da.raster.res`` gives.

    ``res_y`` is negative for the north-up ordering wflow writes; callers that
    only need a magnitude take ``abs()``.
    """
    x_dim, y_dim = spatial_dim_names(da)
    resolutions = []
    for dim in (x_dim, y_dim):
        coord = da[dim].values
        if coord.size < 2:
            raise ValueError(f"cannot derive a resolution from a {dim} of length {coord.size}")
        resolutions.append(float(coord[1] - coord[0]))
    return tuple(resolutions)


def map_extent(da, buffer_deg=_EXTENT_BUFFER_DEG):
    """``[lon_min, lon_max, lat_min, lat_max]`` covering the DEM, plus padding.

    Replaces ``da.raster.box.buffer(...).total_bounds``. Coordinates are cell
    CENTRES, so the box reaches half a cell beyond them on each side — dropping
    that half-cell shrinks the frame by one pixel row and column.
    """
    x_dim, y_dim = spatial_dim_names(da)
    res_x, res_y = pixel_resolution(da)
    half_x, half_y = abs(res_x) / 2.0, abs(res_y) / 2.0
    x, y = da[x_dim].values, da[y_dim].values
    return np.array(
        [
            x.min() - half_x - buffer_deg,
            x.max() + half_x + buffer_deg,
            y.min() - half_y - buffer_deg,
            y.max() + half_y + buffer_deg,
        ]
    )


def _mask_nodata(da):
    """NaN out the fill value, as ``da.raster.mask_nodata()`` does.

    Normally a no-op: xarray decodes ``_FillValue`` to NaN when it opens the
    file. It earns its place for a DataArray opened with ``mask_and_scale=False``
    or one carrying the fill value only as an attribute.
    """
    fill = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
    if fill is None or (isinstance(fill, float) and np.isnan(fill)):
        return da
    return da.where(da != fill)


def load_basin_layers(model_dir):
    """Read a wflow model's DEM and vector layers straight off disk.

    No hydromt: ``xarray`` for the grid, ``geopandas`` for the geometries. Any
    directory holding ``staticmaps.nc`` and a ``staticgeoms/`` of GeoJSON works,
    so a caller does not need a model object — or the packages to build one.

    Parameters
    ----------
    model_dir : str | Path
        The model root (the folder containing ``staticmaps.nc``).

    Returns
    -------
    (elevation, rivers, basins, geoms)
        ``geoms`` maps layer name to GeoDataFrame for EVERY GeoJSON found, so
        optional layers (waterbodies, gauges) resolve by name exactly as they did
        through ``mod.geoms.data``.
    """
    model_dir = Path(model_dir)
    staticmaps_path = model_dir / STATICMAPS_FILENAME
    staticgeoms_dir = model_dir / STATICGEOMS_DIRNAME
    if not staticmaps_path.is_file():
        raise FileNotFoundError(f"no {STATICMAPS_FILENAME} in {model_dir}")
    if not staticgeoms_dir.is_dir():
        raise FileNotFoundError(f"no {STATICGEOMS_DIRNAME}/ in {model_dir}")

    with xr.open_dataset(staticmaps_path) as dataset:
        if ELEVATION_VARIABLE not in dataset:
            raise KeyError(
                f"{staticmaps_path} has no {ELEVATION_VARIABLE!r}; it holds "
                f"{sorted(dataset.data_vars)}"
            )
        # load() before the file closes -- the render touches values repeatedly.
        elevation = _mask_nodata(dataset[ELEVATION_VARIABLE].load())

    geoms = {
        path.stem: gpd.read_file(path)
        for path in sorted(staticgeoms_dir.glob("*.geojson"))
    }
    missing = [name for name in REQUIRED_GEOM_LAYERS if name not in geoms]
    if missing:
        # Named explicitly: hydromt would have REBUILT these from the flow
        # direction raster, so their absence is the one case where dropping
        # hydromt changes behaviour. Say so here rather than fail downstream.
        raise FileNotFoundError(
            f"{staticgeoms_dir} is missing {missing}; found {sorted(geoms)}. "
            "hydromt_wflow would derive these from staticmaps; this reader does not."
        )
    return elevation, geoms["rivers"], geoms["basins"], geoms


def _shaded_relief(da, cmap, norm, latitude_deg):
    """Drape the elevation ramp over a hillshade of the same DEM.

    Returns an RGBA ``DataArray``: this replaces the satellite basemap, so it
    has to carry the terrain context on its own.
    """
    # LightSource reads the array as (row, column) = (y, x) and takes dx/dy in
    # that order, so put the DEM in y-major order before touching it rather than
    # assuming the model wrote it that way.
    x_dim, y_dim = spatial_dim_names(da)
    da = da.transpose(y_dim, x_dim)
    resolution_x, resolution_y = (abs(value) for value in pixel_resolution(da))
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
    """Shared tick positions for the grid LINES and the axis LABELS.

    The latitude window is clamped to +/-90 BEFORE the ticks are chosen.
    ``map_extent`` pads the DEM bounds by ``_EXTENT_BUFFER_DEG`` plus half a cell
    and clamps nothing, so a basin near a pole yields ``lat_max > 90`` and the
    graticule would label a latitude that does not exist.

    Longitude is deliberately NOT clamped: past +/-180 is a legitimate way to
    express a basin spanning the antimeridian, whereas past +/-90 is always
    meaningless.

    ``cartopy.mpl.ticker.LatitudeLocator`` looks like the ready-made answer and
    is not. It subdivides in degrees/minutes/seconds, so on a sub-degree basin it
    returns 0.33/0.36/0.39/0.42/0.45/0.48 where ``MaxNLocator`` returns
    0.35/0.40/0.45/0.50 -- uglier, and SIX ticks against a ``max_ticks`` of five.
    Measured on the fixture 2026-08-07, which is why the graticule half of the
    abandoned feat/outputs-figures branch was not carried over.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lat_min, lat_max = max(float(lat_min), -90.0), min(float(lat_max), 90.0)
    locator = MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10])

    def inside(ticks, low, high):
        return [t for t in ticks if low <= t <= high]

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


def _locator_drawn():
    """Whether the inset will actually be drawn, so corners can be budgeted."""
    return LOCATOR_ENABLED and BASEMAP_PATH.is_file()


def _locator_corner(basin_geometry, extent):
    """The corner the locator takes: the emptiest, or the configured one.

    Returns ``None`` when there is no inset, which hands the corner back to the
    scale bar rather than reserving it for something that never appears.
    """
    if not _locator_drawn():
        return None
    if _LOCATOR_CORNER != "auto":
        return _LOCATOR_CORNER
    occupancy = _corner_occupancy(basin_geometry, extent)
    candidates = [name for name in _CORNERS if name != _NORTH_ARROW_CORNER]
    return min(
        candidates,
        key=lambda name: (
            round(occupancy[name], 3),
            0 if name.startswith("upper") else 1,
            name,
        ),
    )


def _scale_bar_corner(basin_geometry, extent, excluded=None):
    """The emptiest corner left for the scale bar, ties broken toward the bottom.

    ``excluded`` defaults to the corners the north arrow and the locator inset
    hold. The legend is not among them — it lives in the side panel rather than
    on the map, which gives the bar back a lower corner it used to yield.

    Ties are rounded before ranking so "equally empty" really does fall through
    to the bottom preference, and the corner name breaks the last tie so the
    figure never depends on dict iteration order.
    """
    if excluded is None:
        excluded = {_NORTH_ARROW_CORNER}
    elif isinstance(excluded, str):
        excluded = {excluded}
    occupancy = _corner_occupancy(basin_geometry, extent)
    candidates = [name for name in _CORNERS if name not in excluded] or list(_CORNERS)
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


def _locator_window(extent):
    """The locator's own extent: a square window centred on the basin.

    Square in DEGREES, so the inset box can be made square on the page and the
    window fills it without letterboxing. Clamped to the world, and re-centred
    rather than clipped at the poles so a high-latitude basin still gets a full
    window rather than a half-empty one.
    """
    centre_lon = 0.5 * (extent[0] + extent[1])
    centre_lat = 0.5 * (extent[2] + extent[3])
    span = _LOCATOR_SPAN_DEG
    centre_lat = float(np.clip(centre_lat, -90.0 + span, 90.0 - span))
    return [
        centre_lon - span,
        centre_lon + span,
        max(centre_lat - span, -90.0),
        min(centre_lat + span, 90.0),
    ]


def _locator_box(extent, corner):
    """[x0, y0, w, h] in axes fractions, square ON THE PAGE, in its corner.

    The map axes is not square — PlateCarree locks it to the extent's own
    degree ratio — so a box with equal fractional width and height comes out
    as stretched as the panel is. Correcting by that ratio is what makes the
    inset a square rather than a slot.
    """
    lon_span = max(float(extent[1] - extent[0]), 1e-9)
    lat_span = max(float(extent[3] - extent[2]), 1e-9)
    width = _LOCATOR_WIDTH
    height = width * lon_span / lat_span
    # A tall, narrow basin makes the panel tall: the square would then overflow
    # the map vertically, so cap it and take the width back down to match.
    if height > 1.0 - 2.0 * _LOCATOR_MARGIN:
        height = 1.0 - 2.0 * _LOCATOR_MARGIN
        width = height * lat_span / lon_span
    left = (
        _LOCATOR_MARGIN if corner.endswith("left") else 1.0 - _LOCATOR_MARGIN - width
    )
    bottom = (
        1.0 - _LOCATOR_MARGIN - height
        if corner.startswith("upper")
        else _LOCATOR_MARGIN
    )
    return [left, bottom, width, height]


def _read_basemap(layer, window):
    """One vendored Natural Earth layer, clipped to the locator's window.

    ``bbox`` pushes the filter down into the driver, so a render reads the few
    hundred features it draws rather than the global layer.
    """
    return gpd.read_file(BASEMAP_PATH, layer=layer, bbox=tuple(
        (window[0], window[2], window[1], window[3])
    ))


def _locator_cities(window):
    """The few most prominent cities inside the window, largest first."""
    places = _read_basemap("places", window)
    if places.empty:
        return places
    places = places[places["scalerank"] <= _LOCATOR_CITY_MAX_SCALERANK]
    return places.sort_values("pop_max", ascending=False).head(_LOCATOR_MAX_CITIES)


def _add_locator_inset(ax, extent, basin, corner):
    """A small "where is this" map: land, sea, borders, cities, and the basin.

    Skips itself, with a note, when the vendored basemap is absent — a copy of
    this module taken to another project should still render a basin map, just
    without the inset. Silence would be worse: an inset that quietly never
    appears reads as a layout bug.
    """
    if not LOCATOR_ENABLED:
        return None
    if corner is None or not BASEMAP_PATH.is_file():
        if LOCATOR_ENABLED:
            print(f"note: locator inset skipped, no basemap at {BASEMAP_PATH}")
        return None

    window = _locator_window(extent)
    inset = ax.inset_axes(_locator_box(extent, corner), projection=ccrs.PlateCarree())
    # Outside the layout for the same reason as the side panel: its footprint
    # would inflate the map's tight bbox and shrink the map to make room for
    # something drawn INSIDE the map.
    inset.set_in_layout(False)
    inset.set_extent(window, crs=ccrs.PlateCarree())
    inset.set_facecolor(COLOR_LOCATOR_OCEAN)  # sea is whatever land is not

    land = _read_basemap("land", window)
    if not land.empty:
        land.plot(
            ax=inset,
            facecolor=COLOR_LOCATOR_LAND,
            edgecolor=COLOR_LOCATOR_COAST,
            linewidth=WIDTH_LOCATOR_COAST,
        )
    borders = _read_basemap("borders", window)
    if not borders.empty:
        borders.plot(
            ax=inset, color=COLOR_LOCATOR_BORDER, linewidth=WIDTH_LOCATOR_BORDER
        )

    halo = [pe.withStroke(linewidth=HALO_WIDTH_GAUGE_LABEL, foreground=COLOR_HALO)]
    for _, city in _locator_cities(window).iterrows():
        inset.plot(
            city.geometry.x,
            city.geometry.y,
            marker="o",
            markersize=_LOCATOR_CITY_MARKER_SIZE ** 0.5,
            color=COLOR_LOCATOR_CITY,
            transform=ccrs.PlateCarree(),
        )
        inset.annotate(
            city["name"],
            xy=(city.geometry.x, city.geometry.y),
            xytext=_LOCATOR_CITY_LABEL_OFFSET,
            textcoords="offset points",
            fontsize=FONT_SIZE_LOCATOR_CITY,
            color=COLOR_LOCATOR_CITY,
            va="center",
            path_effects=halo,
        )

    # The basin itself. At an 8 deg window a basin is a fraction of a degree, so
    # a mark reads where the outline would be a dot of noise.
    centroid = basin.union_all().centroid
    inset.plot(
        centroid.x,
        centroid.y,
        marker="o",
        markersize=_LOCATOR_BASIN_MARKER_SIZE ** 0.5,
        markerfacecolor=COLOR_LOCATOR_BASIN,
        markeredgecolor=COLOR_MARKER_EDGE,
        markeredgewidth=WIDTH_MARKER_EDGE,
        transform=ccrs.PlateCarree(),
        zorder=Z_FURNITURE,
    )

    inset.spines["geo"].set_linewidth(WIDTH_LOCATOR_FRAME)
    inset.spines["geo"].set_edgecolor(COLOR_BASIN_OUTLINE)
    return inset


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


def _river_linewidths(gdf_riv, column=RIVER_ORDER_COLUMN):
    """Stream order rescaled to publication line weights.

    ``strord / 2`` was tuned to a 10x8-inch canvas; at 180 mm it draws an
    8th-order river as a 4 pt band that swallows the basin.

    A river layer from outside wflow may carry no order column at all, so a
    missing (or ``None``) ``column`` falls back to one uniform weight rather
    than raising: a network drawn at a single width is a legitimate map, and
    refusing to plot it would be the wrong answer for the commonest
    non-wflow input.
    """
    if column is None or column not in gdf_riv.columns:
        return RIVER_WIDTH_UNIFORM
    order = gdf_riv[column].astype(float).to_numpy()
    lowest, highest = float(np.nanmin(order)), float(np.nanmax(order))
    if not np.isfinite(lowest) or highest <= lowest:
        return np.full(order.shape, RIVER_WIDTH_UNIFORM)
    span = RIVER_WIDTH_MAX - RIVER_WIDTH_MIN
    return RIVER_WIDTH_MIN + span * (order - lowest) / (highest - lowest)


def _draw_relief(fig, ax, dem, centre_latitude, elevation_label):
    """Shaded relief plus its colourbar in the side panel."""
    levels = _elevation_levels(dem)
    cmap = _elevation_colormap(len(levels) - 1)
    norm = colors.BoundaryNorm(levels, cmap.N)
    x_dim, y_dim = spatial_dim_names(dem)
    _shaded_relief(dem, cmap, norm, centre_latitude).plot.imshow(
        ax=ax,
        x=x_dim,
        y=y_dim,
        transform=ccrs.PlateCarree(),
        zorder=Z_RELIEF,
        add_labels=False,
    )
    # imshow of an RGBA array carries no mappable, so the colourbar needs an
    # explicit one — the ramp is the same object either way.
    colorbar_axes = ax.inset_axes(_colorbar_inset(elevation_label.count("\n") + 1))
    # The side panel lives OUTSIDE the map axes but is anchored to it. Left
    # in the layout, its footprint inflates the axes' tight bbox, and
    # constrained layout answers by shrinking the map — in BOTH directions,
    # because the aspect is locked. Measured cost before this line: 0.69 in
    # of dead space above AND below a map 1.2 in narrower than its cell.
    # ``rect`` already reserves the panel's room, so it must not be counted
    # twice.
    colorbar_axes.set_in_layout(False)
    # ``ticks=levels`` labels every class BOUNDARY, which is what puts the first
    # and the last on the bar; matplotlib's own locator drops both ends.
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axes,
        ticks=levels,
        spacing="proportional",
    )
    if _colorbar_label_position() == "top":
        # A TITLE, not the label: ``set_label`` on a vertical bar always lands
        # alongside it, rotated. ``loc="left"`` is what keeps a label wider than
        # the 0.025-wide bar from hanging off both of its sides — it starts at
        # the bar's left edge, which is the legend's anchor too.
        colorbar_axes.set_title(
            elevation_label,
            fontsize=FONT_SIZE_COLORBAR_LABEL,
            pad=_COLORBAR_TITLE_PAD,
            loc="left",
        )
    else:
        colorbar.set_label(elevation_label, fontsize=FONT_SIZE_COLORBAR_LABEL)
    colorbar.outline.set_linewidth(_COLORBAR_OUTLINE_WIDTH)
    colorbar_axes.tick_params(labelsize=FONT_SIZE_TICK, length=TICK_LENGTH, pad=TICK_PAD)


def _draw_points(ax, layer, facecolor, label, marker, label_column=None):
    """One point layer, optionally annotated with a column's values."""
    if layer is None or len(layer) == 0:
        return
    layer.plot(
        ax=ax,
        marker=marker,
        markersize=MARKER_SIZE,
        facecolor=facecolor,
        edgecolor=COLOR_MARKER_EDGE,
        linewidth=WIDTH_MARKER_EDGE,
        zorder=Z_MARKER,
        label=label,
    )
    if label_column is None or label_column not in layer.columns:
        return
    layer.apply(
        lambda row: ax.annotate(
            text=str(row[label_column]),
            xy=row.geometry.coords[0],
            xytext=GAUGE_LABEL_OFFSET,
            textcoords="offset points",
            fontsize=FONT_SIZE_GAUGE_LABEL,
            fontweight="bold",
            color=COLOR_BASIN_OUTLINE,
            zorder=Z_MARKER,
            path_effects=[
                pe.withStroke(linewidth=HALO_WIDTH_GAUGE_LABEL, foreground=COLOR_HALO)
            ],
        ),
        axis=1,
    )


def _draw_waterbodies(ax, layers):
    """Fill each present waterbody layer; return its legend patches.

    geopandas' polygon handle does not survive into a legend usably
    (geopandas/geopandas#660), so the patches are built by hand.

    The ``label`` goes on the PATCH ONLY, never on the ``plot`` call. Passing it
    to both — which this did until 2026-08-03 — put each waterbody in the legend
    twice: geopandas does register the labelled collection, so
    ``get_legend_handles_labels`` returns it AND the hand-built patch.
    """
    patches = []
    for name, layer in layers.items():
        if layer is None or len(layer) == 0:
            continue
        face, edge = WATERBODY_COLORS[name]
        style = dict(facecolor=face, edgecolor=edge, linewidth=WIDTH_WATERBODY_EDGE)
        layer.plot(ax=ax, zorder=Z_WATERBODY, **style)
        patches.append(mpatches.Patch(label=name, **style))
    return patches


def plot_basin_map(
    dem,
    rivers,
    basin,
    *,
    subbasins=None,
    gauges=None,
    outlets=None,
    lakes=None,
    reservoirs=None,
    glaciers=None,
    extent=None,
    gauge_label_column=GAUGE_LABEL_COLUMN,
    river_order_column=RIVER_ORDER_COLUMN,
    elevation_label=ELEVATION_LABEL,
):
    """Draw a basin map: shaded relief, rivers, boundaries, points, waterbodies.

    Every layer is its own argument, and every argument but the first three is
    optional — so this plots ANY basin, from any source, not only a wflow model
    on disk. It reads no files, writes no files and returns the figure; saving
    is the caller's decision. ``plot_basin_map_from_model`` is the wrapper that
    supplies these arguments from a wflow model directory.

    Parameters
    ----------
    dem : xarray.DataArray
        2-D elevation on a GEOGRAPHIC grid (EPSG:4326). Its coordinates set the
        default extent and its values drive both the colour ramp and the
        hillshade, whose vertical exaggeration is derived per basin. Cells
        outside the basin should be NaN — they are drawn transparent.
    rivers : geopandas.GeoDataFrame
        The river network (LineStrings). Line weight scales with
        ``river_order_column`` when the frame carries it.
    basin : geopandas.GeoDataFrame
        The OUTER boundary, already dissolved to what should be drawn as the
        map's heaviest line. Pass ``_basin_outline(subcatchments)`` if you hold
        one polygon per subcatchment — drawing those at boundary weight makes an
        internal divide indistinguishable from the basin outline, which is the
        one line on this figure a reader has to be able to trust.
    subbasins : geopandas.GeoDataFrame, optional
        Internal subcatchment divides, drawn lighter and dashed beneath the
        outline. Omit for a basin with no meaningful internal division; nothing
        is drawn and no legend entry appears.
    gauges, outlets : geopandas.GeoDataFrame, optional
        Point layers. Gauges take the river colour (they sit on the network);
        outlets stay black (they belong to the model). Gauges are annotated with
        ``gauge_label_column`` when the frame has it.
    lakes, reservoirs, glaciers : geopandas.GeoDataFrame, optional
        Filled polygon layers, each with its own colours from
        ``WATERBODY_COLORS``.
    extent : sequence of float, optional
        ``[lon_min, lon_max, lat_min, lat_max]``. Defaults to the DEM's own
        bounding box plus ``_EXTENT_BUFFER_DEG``. Set it to frame several basins
        alike, or to crop.
    gauge_label_column : str or None
        Column annotated beside each gauge; ``None`` draws the markers unlabelled.
    river_order_column : str or None
        Numeric column scaling the river widths; ``None`` or an absent column
        draws every reach at ``RIVER_WIDTH_UNIFORM``.
    elevation_label : str
        Colourbar label. Change it with the DEM's units.

    Returns
    -------
    (matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxes)
        Nothing has been saved. The figure is sized in millimetres
        (``FIGURE_WIDTH_MM``), so ``savefig`` without ``bbox_inches="tight"``
        preserves the declared width.
    """
    if extent is None:
        extent = map_extent(dem)
    # we assume the layers are in the geographic CRS EPSG:4326
    proj = ccrs.PlateCarree()
    centre_latitude = 0.5 * float(extent[2] + extent[3])

    with rc_context(_publication_rc()):
        fig = plt.figure(figsize=_figure_size(extent), layout="constrained")
        fig.get_layout_engine().set(rect=(0.0, 0.0, _LAYOUT_RIGHT, 1.0))
        ax = fig.add_subplot(projection=proj)
        ax.set_extent(extent, crs=proj)

        # --- elevation, as shaded relief -------------------------------------
        _draw_relief(fig, ax, dem, centre_latitude, elevation_label)

        # --- hydrography ------------------------------------------------------
        # The draw order below IS the legend order: handles come back from
        # ``get_legend_handles_labels`` in the order their artists were added.
        rivers.plot(
            ax=ax,
            linewidth=_river_linewidths(rivers, river_order_column),
            color=COLOR_RIVER,
            zorder=Z_RIVER,
            label="river",
        )
        # Subcatchment divides first and lighter, then the outline over them, so
        # the two are never confusable at the same weight.
        subcatchment_handles = []
        if subbasins is not None and len(subbasins) > 0:
            divide_style = dict(
                color=COLOR_SUBCATCHMENT,
                linewidth=WIDTH_SUBCATCHMENT,
                linestyle=DASH_SUBCATCHMENT,
            )
            subbasins.boundary.plot(ax=ax, zorder=Z_SUBCATCHMENT, **divide_style)
            subcatchment_handles.append(
                Line2D([], [], label="subcatchments", **divide_style)
            )
        basin.boundary.plot(
            ax=ax,
            color=COLOR_BASIN_OUTLINE,
            linewidth=WIDTH_BASIN_OUTLINE,
            zorder=Z_BASIN_OUTLINE,
        )

        _draw_points(ax, outlets, COLOR_OUTLET, "outlets", MARKER_SHAPE_OUTLET)
        _draw_points(
            ax, gauges, COLOR_GAUGE, "output locs", MARKER_SHAPE_GAUGE,
            gauge_label_column,
        )

        # --- waterbodies ------------------------------------------------------
        patches = _draw_waterbodies(
            ax, {"lakes": lakes, "reservoirs": reservoirs, "glaciers": glaciers}
        )

        # --- cartographic furniture -------------------------------------------
        # The legend sits in the side panel, so it no longer competes for a map
        # corner. The scale bar is placed against the basin's ACTUAL footprint,
        # so it does not land on a basin that reaches into a bottom corner.
        footprint = basin.union_all()
        # Corners are budgeted in one place, in priority order: the arrow's is
        # fixed, the locator takes the emptiest of what is left because it is
        # opaque, and the scale bar — which is see-through — takes the emptiest
        # of the remainder.
        locator_corner = _locator_corner(footprint, extent)
        bar_corner = _scale_bar_corner(
            footprint, extent, {_NORTH_ARROW_CORNER, locator_corner}
        )
        _add_graticule(ax, extent)
        _add_scale_bar(ax, extent, bar_corner)
        _add_north_arrow(ax)
        _add_locator_inset(ax, extent, basin, locator_corner)
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

        # Constrained layout is ITERATIVE, and one pass is not enough here: the
        # first draw leaves the y tick labels at x0 = -7.7 px — off the canvas,
        # so "0.45°N" prints as "45°N" — and the second settles them at +4.2.
        # The workflow path never saw this because it saves twice (PDF, then
        # PNG) and the second save inherits a settled layout. A caller doing one
        # savefig would not, so the figure is settled BEFORE it is handed back.
        for _ in range(_LAYOUT_SETTLE_PASSES):
            fig.draw_without_rendering()

    return fig, ax


def plot_basin_map_from_model(project_dir, gauges_fn, plot_dir=None, model_dir=None):
    """Render basin_area.{pdf,png} for a wflow model on disk.

    The file-reading half of the figure: it resolves a model directory into the
    layers ``plot_basin_map`` takes, then saves the result. This is what the
    Snakemake rule calls; anything that already holds the layers should call
    ``plot_basin_map`` directly.

    The gauge layer is resolved from the MODEL (``shared.gauges``), not from the
    configured filename: hydromt_wflow renames ``output_locations`` to
    ``output-locations``, and deriving the name here is what silently dropped
    the gauges from this figure (2026-08-01).
    """
    from blueearth_cst.shared.gauges import gauges_layer_name

    # The model root is the RULE's fact when a rule is calling; the constant is
    # the fallback for standalone callers (R9 P2 commit 1).
    root = str(model_dir) if model_dir else f"{project_dir}/{MODEL_DIRNAME}"
    if plot_dir is None:
        # basin_area depicts the MODEL, not its evaluation, so it sits at the
        # model root's plots/ -- not under evaluation/ (P1).
        plot_dir = f"{root}/plots"

    # Read straight off disk -- no model object, no hydromt (see module docstring).
    dem, rivers, basins, geoms = load_basin_layers(root)
    # ``basins`` is one polygon PER SUBCATCHMENT once gauges are burned into the
    # subcatchment map, and a single polygon otherwise. Split it into the two
    # roles the figure draws: a dissolved outline, and the divides — which exist
    # only when there is more than one subcatchment to divide.
    gauges_name = gauges_layer_name(geoms, gauges_fn)
    fig, _ = plot_basin_map(
        dem,
        rivers,
        _basin_outline(basins),
        subbasins=basins if len(basins) > 1 else None,
        # Resolved against what the model actually holds; ``gauges_layer_name``
        # warns loudly (never skips silently) when output_locations is set but
        # no layer matches.
        gauges=geoms.get(gauges_name) if gauges_name is not None else None,
        outlets=geoms.get("outlets"),
        lakes=geoms.get("lakes"),
        reservoirs=geoms.get("reservoirs"),
        glaciers=geoms.get("glaciers"),
    )

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
            plot_basin_map_from_model(
                project_dir=sm.params.project_dir,
                model_dir=sm.params.model_dir,
                gauges_fn=getattr(sm.input, "output_locations", None),
            )
