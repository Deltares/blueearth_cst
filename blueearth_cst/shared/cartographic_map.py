# -*- coding: utf-8 -*-
"""A cartographic template: any geographic raster, with map furniture.

One drawing function, ``plot_raster_map``, serves every map this toolbox makes.
What differs between a DEM, a rainfall field and a temperature field is a
``RasterStyle`` — palette, label, classification, whether the values are a
surface worth hillshading. Everything else a map needs is shared and lives
here: the graticule and frame, a latitude-corrected scale bar, a north arrow,
an auto-sized locator inset, and a side panel that stacks the colourbar over
the vector legend.

Adding a quantity is an entry in ``RASTER_STYLES``, not another plotting
function. That is the whole point of the split: rule 1.12's basin map and rule
1.13's three forcing maps were separate code with separate ideas of what a map
is, and the forcing maps had no furniture at all.

The two entry points, split along reading-vs-drawing, are elsewhere:
``shared.plot_map`` resolves a wflow model on disk into the layers this module
draws. Nothing here reads a file except the vendored basemap, and nothing here
knows what wflow is — so any raster on a geographic grid, from any source,
plots through it.

Three rules the whole module depends on, worth knowing before changing anything:

* Lengths are in PHYSICAL units (mm / inches / points). The figure is built at
  its final printed size, so a font size is the size it will be on the page.
  Raising ``RASTER_DPI`` makes the PNG bigger, NOT the type smaller.
* Positions inside the map are AXES FRACTIONS and may exceed 1 to sit outside
  it — that is how the side panel works. Positions of map furniture drawn in
  data space (the scale bar) are fractions of the map's own extent, so they
  hold for any basin.
* Anything assembled FROM a tunable is derived in a FUNCTION, never frozen into
  a module-level constant. A constant snapshots its inputs at import, so
  overriding the input afterwards would silently do nothing — which is exactly
  how ``dev/scripts/preview_basin_map.py`` drives this module.
"""

import warnings
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import colors, rc_context
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from shapely.geometry import box as shapely_box

from blueearth_cst.shared import plot_style

# The toolbox-wide page/typography settings. Bound HERE, in this module's
# globals, so the drawing code below reads them by bare name and
# `dev/scripts/preview_basin_map.py --set` can rebind them -- see the tunable
# block and `plot_style`'s docstring.
#
# RASTER_DPI is a deliberate re-export, not a leftover: `shared/plot_map.py`
# imports it from here, and that file is a Snakemake `script:` target
# (build_model.smk rule 1.12), so editing it would fire the `code`
# rerun trigger on every project_dir for no behavioural gain. It moves to a
# direct `plot_style` import in the plotting sweep, alongside a change to that
# file that is worth the invalidation.
from blueearth_cst.shared.plot_style import (
    COLOR_CAVEAT,
    FIGURE_WIDTH_MM,
    FONT_FAMILY,
    FONT_SIZE_CAVEAT,
    FONT_SIZE_TITLE,
    MM_PER_INCH,
    RASTER_DPI,  # noqa: F401
)

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
#   will be on the page. Raising RASTER_DPI makes the PNG bigger, NOT the type
#   smaller.
# * Positions inside the map are AXES FRACTIONS (0 = left/bottom, 1 =
#   right/top) and can exceed 1 to sit outside the map — that is how the side
#   panel works. Positions of map furniture drawn in data space (the scale bar)
#   are fractions of the map's own extent, so they hold for any basin.
# ---------------------------------------------------------------------------

# --- page and export -------------------------------------------------------
#
# FIGURE_WIDTH_MM, MM_PER_INCH, RASTER_DPI, the FONT_SIZE_* names below marked
# "(shared)" and FONT_FAMILY are the TOOLBOX-WIDE settings, imported from
# `shared/plot_style.py` so every figure family agrees on them. Change one
# THERE to move every figure; rebind one here (which is what
# `dev/scripts/preview_basin_map.py --set` does) to move only the map. The
# import binds each name in THIS module's globals, so the drawing code below
# and the preview's `setattr` both keep working unchanged.
#
# What stays here is map furniture — the scale bar, north arrow, gauge labels
# and colourbar sizes, which no non-map figure reads. `plot_style`'s docstring
# carries the test for which side a new constant belongs on.

# --- typography ------------------------------------------------------------

#: Type sizes in POINTS at the printed width above. Applied through
#: ``rc_context`` so the process-wide rcParams the other plotting rules inherit
#: are left untouched. Raise every value together to scale the labelling; raise
#: one to re-balance it.
#:
#: **Every size here is map-scoped, including the three that `plot_style` also
#: defines.** Those three (BASE / TICK / LEGEND) were imported from there until
#: 2026-08-16 and are now declared HERE instead, one step smaller across the
#: board — because the sizes that read correctly on a series plot are too heavy
#: on a map, where the labelling competes with the picture rather than
#: annotating an empty axes. Owner's call after reading a real basin's output:
#: the graticule, the colourbar title and the legend all crowded the map.
#:
#: Declaring rather than shrinking `plot_style` is what keeps the change
#: SCOPED. Those constants are page-level and every figure family reads them,
#: so lowering them there would silently shrink the projections and evaluation
#: figures too — families nobody asked about and nobody would re-render to
#: check. The blast radius of this block is the cartographic template plus the
#: climate series that share its `_publication_rc`, which is the one family
#: these sizes were judged against.
#:
#: `FONT_SIZE_CAVEAT` deliberately stays shared: `climate_figures` imports it
#: straight from `plot_style` for the series footnote, so map-scoping it would
#: print the same sentence at two sizes in one folder.
FONT_SIZE_BASE = 6.5  #: fallback size; the axes title is derived from it (+1)
FONT_SIZE_TICK = 6.0  #: the coordinate graticule labels
FONT_SIZE_LEGEND = 6.0  #: legend entries and the legend's own title
FONT_SIZE_COLORBAR_LABEL = 6.5  #: the colourbar's title
FONT_SIZE_COLORBAR_TICK = 5.5  #: the numbers beside the colourbar
FONT_SIZE_GAUGE_LABEL = 5.5  #: the wflow_id beside each gauge marker
FONT_SIZE_SCALE_BAR = 5.5  #: the 0 / 2.5 / 5 km numbers
FONT_SIZE_NORTH_ARROW = 7.0  #: the "N"

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

#: Colourbar geometry in axes fractions. ``HEIGHT`` is the BAR's own drawn
#: length, so it reads straight against a brief that asks for "60-70% of the
#: axis"; it is a MAXIMUM, since the bar gives way when the panel's other two
#: blocks need the room. There is no ``_COLORBAR_BOTTOM`` any more: the bar's
#: position is DERIVED from what the locator above it and the legend below it
#: leave, which is the only way the three can be guaranteed to fit the map's
#: height without a hand-tuned constant per combination.
_COLORBAR_WIDTH = 0.030
_COLORBAR_HEIGHT = 1.50
#: The bar never shrinks below this, even if that overruns the panel. A bar too
#: short to carry its own tick labels is not a smaller bar, it is a broken one —
#: better to overflow visibly than to render something unreadable.
_COLORBAR_MIN_HEIGHT = 0.18
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

#: Height reserved for a "top" label, in axes fractions PER LINE of it. The bar
#: keeps its declared length and is pushed DOWN when the label would otherwise
#: run off the canvas — the earlier behaviour shortened the bar instead, which
#: silently broke the "the bar is N% of the axis" contract the height states.
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
#: rounded to a ladder value first, so the boundaries are numbers worth printing
#: and the count lands near this rather than on it.
_COLORBAR_LEVELS = 6

#: Hard cap on the number of TICK LABELS on the bar — one per class boundary, so
#: a bar of N classes carries N+1 of them. The target above is a wish; this is
#: enforced, by widening the step until the count fits. Both exist because
#: rounding the class width to a readable number means the count cannot be
#: dialled in exactly: asking for 4 classes over a 0-140 m range yields 4 or 3
#: depending on which ladder rung the width lands on, and only the cap
#: guarantees the bar never comes back with eight.
_COLORBAR_MAX_TICKS = 9

#: How many times the step may be widened chasing that cap before the result is
#: taken as it stands. A bound, not a tuning knob: each rung roughly doubles the
#: step, so this covers a range of 10^3 and exists only so a pathological DEM
#: cannot spin here.
_STEP_WIDEN_ATTEMPTS = 12

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
#: Raised from 0.35 in 2026-08: at 0.35 a 1900-4200 m headwater still zeroed,
#: which spent the lowest class on ground the basin does not contain and left
#: 44% of its cells in one class. Measured on synthetic archetypes, not guessed.
_ZERO_BASELINE_MIN_SPAN_FRACTION = 0.70

# --- elevation classification ----------------------------------------------
# Equal-interval classes are the right default and the wrong answer on a skewed
# DEM. The fixture is the case: 1.5-215 m, median 10 m, so 0/20/40/.../140 puts
# 73% of the basin in ONE class and leaves three classes empty — the map renders
# as blank paper and the ramp does no work. Equal-AREA (quantile) classes fix
# that but read as arbitrary numbers, so the breaks are snapped onto a readable
# ladder. Which rule applies is decided per basin from the DEM's own histogram,
# because CST runs on deltas and on headwaters from this same code.

#: Which rule sets the class boundaries.
#:
#: * ``"equal_interval"`` — evenly spaced classes on a round step, always. The
#:   bar is then a linear scale: a reader can step up it in equal metres, and
#:   two classes are always the same number of metres apart. This is the
#:   default, and it is what most readers assume an elevation bar is.
#: * ``"auto"`` — equal-interval unless the DEM's own histogram says it is not
#:   working (see ``_MAX_CLASS_AREA_SHARE``), then equal-AREA classes snapped to
#:   a readable ladder. Differentiates a skewed basin far better; costs the
#:   linear reading.
#:
#: The trade is real in both directions and depends on the basin, which is why
#: both rules are kept rather than one being deleted.
ELEVATION_CLASSIFICATION = "equal_interval"
_ELEVATION_CLASSIFICATIONS = ("equal_interval", "auto")

#: Equal-interval is kept while no class holds more than this share of the
#: basin's cells AND no class is empty. Above it, the equal-area rule takes
#: over. Consulted only when ``ELEVATION_CLASSIFICATION == "auto"``.
_MAX_CLASS_AREA_SHARE = 0.40

#: A class holding less than this share counts as empty for that test.
_EMPTY_CLASS_AREA_SHARE = 0.01

#: Never fall to fewer classes than this. Snapping equal-area breaks onto the
#: ladder can collapse neighbours on a tightly-clustered DEM; below this count
#: the equal-interval breaks are the better of two imperfect answers.
_MIN_CLASSES = 4

#: Mantissas of the readable break ladder, spanning one decade. ~30% spacing:
#: fine enough that a quantile lands near a rung, coarse enough that every rung
#: is a number worth printing on a bar. Equal-area breaks snap onto it.
_BREAK_LADDER = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

#: Gap between the colourbar's lower end and the top of the legend box, in axes
#: fractions. The legend's position is DERIVED from the bar's rather than being
#: a second hand-tuned constant: the two used to be pinned independently, so
#: changing the bar's length silently moved it into the legend.
_LEGEND_GAP = 0.05

#: Row pitch of the legend, as a multiple of ``FONT_SIZE_LEGEND``. Used to
#: PREDICT the legend's height before it is drawn, so the colourbar above it can
#: be sized to what is left. matplotlib does not report a legend's extent until
#: after a draw, and by then the bar's inset axes is already placed — so the
#: height is estimated from the entry count instead. Raise it if a long legend
#: starts to crowd the bar; the estimate only has to be close.
_LEGEND_ROW_FACTOR = 1.55
_LEGEND_FRAME_ALPHA = 0.85  #: 1.0 = opaque, 0.0 = no fill
_LEGEND_FRAME_WIDTH = 0.5  #: border weight, points
_LEGEND_BORDER_PAD = 0.4  #: padding inside the frame, in font units
_LEGEND_HANDLE_LENGTH = 1.4  #: length of the sample line/marker, in font units
#: ``None`` drops the title row. Titled since 2026-08-16, and the wording is
#: why: "Legend" would label the obvious and cost a row of the panel's height
#: for nothing, which is why the row was dropped originally. What the row earns
#: its place saying is what KIND of thing the entries are — the panel stacks a
#: colourbar over this block, and both are keys, so the reader is otherwise
#: left to infer that one describes the raster and the other the linework over
#: it. "Map features" names the second half of that pair.
_LEGEND_TITLE = "Map features"

#: Legend wording, one place. These are what the READER sees, so they are the
#: domain's words rather than the model's internals: "output locs" was the
#: config key ``output_locations`` abbreviated to fit, which names a wflow
#: concept and not a thing on the ground. Keep them short — the side panel is
#: about 40 mm wide and a longer entry pushes ``_LAYOUT_RIGHT``.
LABEL_RIVER = "River network"
LABEL_BASIN = "Basin boundary"
LABEL_SUBCATCHMENT = "Subcatchments"
LABEL_OUTLET = "Basin outlet"
LABEL_GAUGE = "Points of interest"

# --- colours ---------------------------------------------------------------

#: One place for every hue on the figure. The blue is used for BOTH the rivers
#: and the user's gauges, which is deliberate: it ties a gauge to the network it
#: sits on and separates it from the model's own outlets, which stay black.
COLOR_RIVER = "#2c6fad"
COLOR_GAUGE = "#2c6fad"
COLOR_OUTLET = "k"
COLOR_BASIN_OUTLINE = "k"
#: Subcatchment divides. Darkened from "0.45": a light grey hairline is legible
#: over white and disappears over the mid-browns of the elevation ramp, which is
#: most of the map. Contrast alone was not enough — the divides also carry a
#: halo (``HALO_WIDTH_SUBCATCHMENT``), which is what makes a 0.6 pt line hold
#: against ANY terrain colour underneath it rather than only against pale ones.
COLOR_SUBCATCHMENT = "0.15"
COLOR_GRATICULE = "0.4"
COLOR_MARKER_EDGE = "white"
#: Halo drawn behind furniture text so it stays legible over any terrain.
COLOR_HALO = "white"

#: Fill and wording for cells of a CATEGORICAL raster whose code the style's
#: table does not declare. They are drawn, in one neutral grey, and warned
#: about — dropping them would take real ground off the map and leave a hole a
#: reader parses as nodata. See ``RasterStyle.categories``.
COLOR_UNCLASSIFIED = "#bdbdbd"
LABEL_UNCLASSIFIED = "Unclassified"

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
#: Internal divides. Still lighter than the basin outline — that hierarchy is
#: the point — but no longer a hairline: at 0.35 pt the divides were below what
#: reads on screen at all over textured ground.
WIDTH_SUBCATCHMENT = 0.7
WIDTH_WATERBODY_EDGE = 0.5
WIDTH_MARKER_EDGE = 0.4
#: Outline around a categorical class swatch in the legend. Thin, but never
#: zero: the Copernicus legend contains near-white classes (snow, moss) that
#: would otherwise be an invisible gap in the swatch column.
_CATEGORY_SWATCH_EDGE_WIDTH = 0.4
WIDTH_AXES_SPINE = 0.6
WIDTH_GRATICULE = 0.3
#: Dash pattern for the subcatchment divides, matplotlib ``(offset, (on, off))``.
#: Longer dashes than the old (4, 2): a short dash at this weight reads as a
#: dotted line, and the divides have to be told apart from the graticule, which
#: IS dotted.
DASH_SUBCATCHMENT = (0, (6, 2.5))
#: Halo stroke widths, points. The halo must exceed the line it protects.
HALO_WIDTH_TEXT = 2.5
HALO_WIDTH_GAUGE_LABEL = 1.8
#: Halo behind the subcatchment divides. Narrowed from 1.6 once the template
#: started drawing SATURATED rasters: over the pale elevation ramp a wide halo
#: is invisible, but over a dark red temperature field it became the thing you
#: see — the divides read as WHITE dashes with dark edges, the inverse of the
#: intended styling. It must stay wide enough to separate the line and narrow
#: enough that the line, not the ring, is what reads.
HALO_WIDTH_SUBCATCHMENT = 1.25

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

#: ``"box"`` closes the map on all four sides — the frame a reader expects
#: around a map panel, and what keeps the DEM's own edge from reading as the
#: panel's edge. ``"L"`` draws only the two labelled sides, which is the plot
#: convention rather than the map one.
_MAP_FRAME = "box"

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
#: VERTICAL inset of the bar from its chosen corner, as a fraction of the map
#: extent. The horizontal one is ``_FURNITURE_INSET_X``, shared with the north
#: arrow — see there.
_SCALE_BAR_INSET_Y = 0.06
#: Which corner the bar takes, or ``"auto"`` for the emptiest one left after the
#: north arrow and the locator have been placed. Pinned to lower left so the
#: figure's furniture sits where a reader expects it on EVERY basin — an
#: auto-placed bar that moves corner between two basins makes two maps of the
#: same study harder to compare, which is the cost the auto rule was not paying
#: attention to. Set "auto" to get the old behaviour back.
_SCALE_BAR_CORNER = "lower left"
#: Gap between the bar and its numbers, as a fraction of the latitude span.
_SCALE_BAR_LABEL_GAP = 0.008
_SCALE_BAR_EDGE_WIDTH = 0.5

# --- north arrow -----------------------------------------------------------

#: Horizontal inset of the map's left-hand furniture — the north arrow and the
#: scale bar — as a fraction of the map extent. ONE value for both, so the two
#: line up on the same vertical by construction. They were on 0.035 and 0.06,
#: close enough to look like a mistake rather than a choice; a shared constant
#: is what stops them drifting apart again the next time one is nudged.
#: (An axes fraction and an extent fraction are the same thing here: the panel
#: is PlateCarree with an explicit extent.)
_FURNITURE_INSET_X = 0.05

#: Arrow position in axes fractions: (tip y, tail y). The "N" sits at the tail;
#: the arrow's x comes from ``_FURNITURE_INSET_X``. Exactly vertical is correct
#: here because PlateCarree's north is up. Tucked into the map's own top-LEFT
#: corner: the locator inset owns the top right.
_NORTH_ARROW_POSITION = (0.985, 0.885)
_NORTH_ARROW_STYLE = "-|>"
_NORTH_ARROW_WIDTH = 0.8
#: The arrow's corner. Stated separately from the position because the corner
#: budget reads it — the position is where the artist is drawn, this is which
#: corner is spoken for.
_NORTH_ARROW_CORNER = "upper left"

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

#: Half-width of the locator's window, in degrees, around the basin's centre —
#: or ``"auto"`` to derive it from the basin's own size.
#:
#: Auto, because a fixed value cannot serve two basins of different sizes now
#: that the inset draws the real polygon rather than a centroid mark. The window
#: has to be wide enough to place the basin against something a reader knows,
#: and narrow enough that the basin is more than a few points across — and where
#: that balance falls depends entirely on how big the basin is. A fixed 8 deg
#: put this fixture's 0.24 deg basin at 1.5% of the frame, a speck; the same
#: 8 deg would make a 6 deg basin fill the window and show no context at all.
#: Set a number to pin it.
_LOCATOR_SPAN_DEG = "auto"

#: What "well distinguishable" means, as the basin's long axis over the window's
#: width. 3% is the measured answer on this fixture: it is what a 4 deg
#: half-width gives, and 4 deg is the width that reads as a shape while keeping
#: the coast, the country and the nearest capital in frame.
_LOCATOR_TARGET_BASIN_FRACTION = 0.03

#: Half-widths the auto rule may choose, in degrees. A ladder rather than a
#: continuous value so the window lands on a round number, and the same basin
#: re-rendered after a small extent change does not shift its frame slightly.
_LOCATOR_SPAN_LADDER = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)

#: The basin must FIT its window with room to spare, whatever the target says.
#: Binds for a large basin, where the target alone would choose a window
#: narrower than the basin itself.
_LOCATOR_MIN_SPAN_MARGIN = 0.75

#: The inset's width as a fraction of the map panel's width, and its inset from
#: the corner. Its HEIGHT is derived so the box comes out square on the page —
#: a square window drawn into a non-square box would otherwise letterbox.
_LOCATOR_WIDTH = 0.22
_LOCATOR_MARGIN = 0.025

#: Which corner it sits in, or ``"auto"`` for the emptiest one the north arrow
#: is not using. Pinned to upper right, the conventional place to read a
#: locator, so it does not move between two basins of the same study. The cost
#: is real and was the reason for "auto": the inset is OPAQUE, so a pinned
#: corner can cover basin the reader wanted — on this fixture it lands on the
#: basin's highest ground. ``_LOCATOR_PLACEMENT = "panel"`` is the way out that
#: keeps the position fixed AND covers nothing.
_LOCATOR_CORNER = "upper right"

#: Where the inset is drawn: ``"map"`` puts it in a corner OF THE MAP, over the
#: basin; ``"panel"`` puts it at the top of the side panel, above the colourbar,
#: where it covers nothing. "panel" costs the colourbar some length and reads
#: slightly less as "this window contains that map"; "map" costs whatever basin
#: it lands on. Both are defensible — this is the knob to compare, not to guess.
#: "panel" chosen 2026-08-09: on the map it covered the fixture's highest ground,
#: and the panel stacks locator/colourbar/legend into the map's own height.
_LOCATOR_PLACEMENT = "panel"

#: The inset's width in the side panel, as an axes fraction, when
#: ``_LOCATOR_PLACEMENT == "panel"``. Anchored at ``_PANEL_LEFT`` like the
#: colourbar and the legend, so all three share one left edge.
_LOCATOR_PANEL_WIDTH = 0.315
#: Gap between the panel inset's bottom and the colourbar's label, axes fractions.
#: Trimmed with the +10% inset: the panel's three blocks are stacked to fit the
#: map's height exactly, so the room the inset gains has to come from somewhere,
#: and a gap is cheaper to give up than the colourbar's length.
_LOCATOR_PANEL_GAP = 0.05
#: Cap on the panel inset's height. A wide, short basin makes the square inset
#: TALL in axes fractions (the box is square on the page, not in the fraction
#: space), and an uncapped one would take half the panel from the colourbar.
#: This is what actually SETS the inset's size on a basin wider than it is tall
#: (the fixture is one), so it and ``_LOCATOR_PANEL_WIDTH`` are raised together
#: — changing only the width would do nothing there.
_LOCATOR_PANEL_MAX_HEIGHT = 0.315
#: The colourbar's length when the locator shares the panel with it. The panel
#: cannot hold a full-length bar, a legend AND an inset — this is the price of
#: ``_LOCATOR_PLACEMENT = "panel"``, stated rather than discovered by clipping.
_COLORBAR_HEIGHT_PANEL = 0.272

COLOR_LOCATOR_OCEAN = "#eef2f5"  #: sea; the palest thing on the figure
COLOR_LOCATOR_LAND = "#dcdcd8"  #: land, a shade darker so the coast reads
COLOR_LOCATOR_COAST = "0.55"  #: the land polygons' own edge IS the coastline
COLOR_LOCATOR_BORDER = "0.7"  #: country lines, lighter still
COLOR_LOCATOR_CITY = "0.35"
#: The "you are here" shape — the basin's own outline, filled. The one warm
#: accent on the figure, and the only place red appears: it has to win against
#: grey without competing with the elevation ramp, which owns every brown.
COLOR_LOCATOR_BASIN = "#c0392b"
#: Its edge, a darker shade of the same red. Darker rather than white, because
#: at this size a light edge eats into a shape only a few points across and
#: leaves less of it than it outlines.
COLOR_LOCATOR_BASIN_EDGE = "#7d2318"

WIDTH_LOCATOR_COAST = 0.35
WIDTH_LOCATOR_BORDER = 0.3
WIDTH_LOCATOR_FRAME = 0.6

#: Cities are filtered by Natural Earth's own prominence rank (0 = most
#: prominent), then the largest few by population are kept. Both limits matter:
#: the rank keeps towns out, the count keeps a dense region from filling up.
_LOCATOR_CITY_MAX_SCALERANK = 3
_LOCATOR_MAX_CITIES = 5
_LOCATOR_CITY_MARKER_SIZE = 5
#: City labels. Dropped with the inset's +10%: the inset grew, so the names no
#: longer need the size to stay readable, and a smaller label leaves more of the
#: window showing the coast and borders the inset exists to show.
FONT_SIZE_LOCATOR_CITY = 4.0

#: Weight of the basin outline in the inset. Heavier than the coastline so the
#: basin reads as the subject and the basemap as context.
WIDTH_LOCATOR_BASIN = 0.5
#: White ring drawn behind the basin shape so it separates from land, borders
#: and city labels alike. Must exceed ``WIDTH_LOCATOR_BASIN``.
HALO_WIDTH_LOCATOR_BASIN = 1.6
#: Label offset for a city name.
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

#: How the DEM raster is resampled onto the page. ``"none"`` draws the model
#: grid as it is — honest, and on a coarse grid visibly blocky (the fixture is
#: 16x24 cells at ~900 m, so every cell is ~5 mm on a 180 mm figure).
#: ``"bilinear"`` smooths it into something that READS as terrain but is partly
#: invented: nothing between cell centres was ever measured. Use "none" when the
#: figure has to defend the model's actual resolution, "bilinear" when the DEM
#: is background context and the colourbar carries the numbers. This is a
#: presentation choice with an honesty cost, so it is stated rather than tuned.
DEM_INTERPOLATION = "none"

# --- data / labels ---------------------------------------------------------
# These three are the DEFAULTS of ``plot_basin_map`` parameters, so a caller
# overrides them per call rather than by patching the module.

#: Gauge marker label. ``wflow_id`` is what the wflow output columns
#: (``Q_101``) and the observation file's rows are keyed on, so it is the label
#: that lets a reader join this map to a hydrograph. ``station_name`` is longer,
#: collides more, and answers a question the caption can answer instead — swap
#: it here if the names matter more than the join.
GAUGE_LABEL_COLUMN = "wflow_id"

#: Clip the river network to the basin before drawing it. The engine-neutral
#: river layer is the REGIONAL network, so unclipped it runs well past the
#: study area — context the map did not ask for, and on a study-area figure
#: it reads as the basin outline failing to contain its own hydrography.
#: The raster is already clipped to the basin; this makes the vectors agree.
#: A no-op where the layer is already in-basin, as the wflow model's is.
CLIP_RIVERS_TO_BASIN = True

#: Column whose values scale the river line weights. ``strord`` is wflow's
#: Strahler stream order. Any numeric column works; ``None``, or a column the
#: frame does not carry, draws every reach at ``RIVER_WIDTH_UNIFORM``.
RIVER_ORDER_COLUMN = "strord"

#: Colourbar label. Units are the DEM's, so change it with the DEM. One line:
#: the label is left-aligned to the bar's own left edge, which is the panel's
#: anchor, so it runs into the panel's width rather than past it — and a
#: one-line label gives the bar back the axes fraction the second line cost.
ELEVATION_LABEL = "Elevation (m a.s.l.)"

#: Padding around the mapped footprint, as a fraction of its longitude and
#: latitude spans. Each axis is padded independently so a long, narrow basin
#: does not acquire excessive whitespace along its short axis.
_EXTENT_BUFFER_FRACTION = 0.05
#: Absolute padding floor, in degrees, for very small or degenerate footprints.
_EXTENT_BUFFER_MIN_DEG = 0.01
#: Optional fixed-degree override retained for previews and explicit tuning.
#: ``None`` uses the proportional rule above.
_EXTENT_BUFFER_DEG = None

# --- raster styles ---------------------------------------------------------
# What makes one raster map differ from another. Everything ELSE on the figure
# — furniture, panel stack, legend, graticule, locator — is shared, which is
# what makes this a template rather than four similar functions.
#
# Palettes were verified at the real class count, not at 256 samples: monotonic
# CIE L*, minimum adjacent dL* in greyscale, and order preserved under all three
# dichromacies (measured 2026-08-09). A DIVERGING palette deliberately fails the
# monotonic-lightness test — it is light at the midpoint by construction — so it
# is used only where a midpoint is real, and never as a default.


#: Palette and centre used for temperature that crosses freezing. Below 0 is
#: ice and above is not, so the midpoint is physical rather than chosen — which
#: is the only thing that licenses a diverging ramp. Named constants rather than
#: literals in the style below, because ``climate_figures`` and the tests both
#: assert against them.
TEMPERATURE_DIVERGING_PALETTE = "RdBu_r"
TEMPERATURE_DIVERGING_CENTER = 0.0


class RasterStyle:
    """How one quantity is drawn: palette, label, classification, relief.

    Deliberately a plain object rather than a dataclass, to stay consistent
    with a module that carries no type annotations.
    """

    def __init__(
        self,
        label,
        palette,
        classification="auto",
        clip_quantiles=(0.0, 0.98),
        zero_baseline=False,
        relief=False,
        interpolation="none",
        diverging_center=None,
        diverging_at=None,
        diverging_palette=None,
        reserve_low_for=None,
        low_clip=0.45,
        high_clip=None,
        step_ladder=None,
        levels=None,
        series_color=None,
        categories=None,
    ):
        #: Colourbar label, including units. The caller owns it, because the
        #: units belong to the data rather than to the style.
        self.label = label
        #: A matplotlib colormap NAME, or a tuple of hex anchors to build one
        #: from. Anchors exist for the elevation ramp, which is hand-built
        #: because the perceptually-uniform terrain maps are not in the env.
        self.palette = palette
        self.classification = classification
        self.clip_quantiles = clip_quantiles
        #: Drop the lowest class boundary to zero when the data can afford it.
        #: True for a quantity measured from a datum (elevation, rainfall
        #: depth); False for one that is not (temperature).
        self.zero_baseline = zero_baseline
        #: Drape the ramp over a hillshade of the raster itself. Meaningful
        #: only where the raster IS a surface — elevation. A hillshade of a
        #: precipitation field would render gradients as topography.
        self.relief = relief
        self.interpolation = interpolation
        #: The ACTIVE centre. Set only by :func:`resolve_diverging_style`, and
        #: only for a field that actually straddles the midpoint. Everything
        #: downstream — the class boundaries, the colour sampling — keys off
        #: this one field, so "is this figure diverging?" has exactly one
        #: answer and an unresolved style cannot half-behave like one.
        self.diverging_center = diverging_center
        #: The DECLARED candidate midpoint: a value that is physical rather than
        #: chosen — 0 degC (ice or not), pH 7 (acid or alkaline). Declaring it
        #: does not make the figure diverging; straddling it does.
        #:
        #: The two are separate fields on purpose. A diverging ramp is only
        #: honest where the midpoint is INSIDE the data: a field entirely below
        #: the centre would spend half its palette on values that do not occur
        #: and compress the ones that do. So a style declares the midpoint it
        #: WOULD centre on, and the resolver decides per raster.
        self.diverging_at = diverging_at
        #: The ramp to use once the centre is active. The sequential ``palette``
        #: stays as the fallback, which is what a non-straddling field gets.
        self.diverging_palette = diverging_palette
        #: Reserve the palette's palest end for values at or below this.
        #: Precipitation is the case: white reads as DRY, so a basin whose
        #: lowest class is 2725 mm/y must not paint that class white — it
        #: says 'no rain here' about the wettest ground on the map. When the
        #: class floor sits above this value the ramp starts at ``low_clip``
        #: instead of at 0, so white stays available for what it means.
        #: ``None`` uses the whole ramp always.
        self.reserve_low_for = reserve_low_for
        #: Class-width rungs for this quantity, overriding ``_STEP_LADDER``.
        #: Temperature takes its own so a bar steps in 0.25/0.5/1 degC rather
        #: than picking up 0.15 or 0.2 from the general ladder.
        self.step_ladder = step_ladder
        #: The one colour this quantity takes in a NON-map figure — its annual
        #: series, its monthly climatology. ``None`` derives it from the map's
        #: own ramp, which is the point: a reader meets precipitation as blue on
        #: the map and must meet it as the same blue on the line beside it.
        #: Deriving rather than declaring is what stops the two drifting when a
        #: palette changes.
        self.series_color = series_color
        #: Explicit class boundaries, bypassing the classifier entirely. This is
        #: what lets two figures share one bar: the first computes them, the
        #: second is handed them. ``None`` classifies from the data.
        self.levels = levels
        #: Where the ramp starts when the low end is reserved, 0-1. 0.32 is
        #: far enough up Blues that the driest class reads as a light BLUE
        #: rather than as an off-white a reader still parses as 'dry'. At
        #: 0.45 the driest swatch is L*=72, clearly blue; 0.32 measured L*=82,
        #: which still reads white against a white page.
        self.low_clip = low_clip
        #: Where the ramp STOPS, 0-1 — the mirror of ``low_clip``, and the
        #: answer to a raster so dark the linework over it cannot be read.
        #: ``None`` derives it per palette (:func:`_readable_ramp_ceiling`) so
        #: the darkest class stays above :data:`_MIN_RASTER_LIGHTNESS`; a float
        #: pins it; ``1.0`` restores the full ramp.
        self.high_clip = high_clip
        #: ``((code, colour, label), ...)`` when the raster is NOMINAL — land
        #: cover, a soil taxonomy, a subbasin identifier. Setting it switches the
        #: whole encoding: no ramp, no classifier, no colourbar. The classes go
        #: in the legend as swatches instead, because a colourbar asserts an
        #: ORDER, and "urban" is not between "cropland" and "water".
        #:
        #: Declare it from the SOURCE PRODUCT's published legend wherever one
        #: exists (Copernicus CGLS-LC100 ships per-code colours). A reader who
        #: knows the product then recognises the map, which no palette of ours
        #: can buy. Codes the raster carries but the table does not are drawn in
        #: one grey and WARNED about — never dropped, which would erase real
        #: ground from the map without saying so.
        self.categories = None if categories is None else tuple(categories)

    def replace(self, **changes):
        """A copy of this style with ``changes`` applied.

        Use this instead of rebuilding a style field by field. A caller that
        listed the fields by hand dropped ``step_ladder`` and
        ``reserve_low_for`` the moment they were added, so temperature bars
        stepped in 0.15 degC and rainfall reserved no white — both silently,
        because a missing field looks exactly like a default.
        """
        fields = dict(
            label=self.label,
            palette=self.palette,
            classification=self.classification,
            clip_quantiles=self.clip_quantiles,
            zero_baseline=self.zero_baseline,
            relief=self.relief,
            interpolation=self.interpolation,
            diverging_center=self.diverging_center,
            diverging_at=self.diverging_at,
            diverging_palette=self.diverging_palette,
            reserve_low_for=self.reserve_low_for,
            low_clip=self.low_clip,
            high_clip=self.high_clip,
            step_ladder=self.step_ladder,
            levels=self.levels,
            series_color=self.series_color,
            categories=self.categories,
        )
        fields.update(changes)
        return RasterStyle(**fields)


#: The styles this toolbox ships. Add a quantity by adding an entry, not by
#: writing another plotting function.
RASTER_STYLES = {
    "elevation": RasterStyle(
        label="Elevation (m a.s.l.)",
        palette=None,  # falls back to _DEM_ANCHORS
        # LINEAR, deliberately, and the one style that is. Elevation is the
        # quantity a reader is most likely to do arithmetic on — how far above
        # that gauge, how much fall to the outlet — and equal-area classes give
        # that up to show more spatial structure. The climate variables are read
        # as patterns rather than as differences, so they take the adaptive rule;
        # elevation keeps the scale you can subtract on. Owner's call, 2026-08-09.
        classification="equal_interval",
        zero_baseline=True,
        relief=True,
    ),
    #: Light-to-dark blue: the conventional wet ramp, and the one whose hue
    #: cannot be confused with the warm ramps the other two climate variables
    #: use. dL* 12.0 greyscale, 11.3 under CVD.
    "precip": RasterStyle(
        label="Precipitation (mm y$^{-1}$)",
        palette="Blues",
        zero_baseline=True,
        # White means DRY on a rainfall map. A basin whose driest class is
        # 2725 mm/y has no dry ground on it, so the ramp starts at a pale blue
        # instead and white stays reserved for the basins that do.
        reserve_low_for=0.0,
    ),
    #: Warm sequential for absolute temperature, which has no meaningful
    #: midpoint. ``_temperature_style`` swaps in a diverging palette centred on
    #: 0 degC when the field actually straddles freezing — that is where a
    #: midpoint becomes real. dL* 10.9 greyscale, 9.0 under CVD.
    "temp": RasterStyle(
        label="Air temperature ($\\degree$C)",
        palette="YlOrRd",
        zero_baseline=False,
        # 0.25 / 0.5 / 1 degC and their decades — the steps a reader expects of
        # a temperature bar. The general ladder would also offer 0.15 and 0.2,
        # which read as arbitrary on a thermometer.
        step_ladder=(2.5, 5.0, 10.0),
        # Below 0 is ice and above is not, so the midpoint is physical rather
        # than chosen — which is the only thing that licenses a diverging ramp.
        # Declared, not active: a basin that never freezes keeps the warm
        # sequential ramp. See ``resolve_diverging_style``.
        diverging_at=TEMPERATURE_DIVERGING_CENTER,
        diverging_palette=TEMPERATURE_DIVERGING_PALETTE,
    ),
    #: Evaporative demand: warm, but a different hue family from temperature so
    #: the two figures are not confused at a glance. dL* 10.7 greyscale, 10.0
    #: under CVD — the best of the warm-earth ramps available here.
    "pet": RasterStyle(
        label="Potential evaporation (mm y$^{-1}$)",
        palette="Oranges",
        zero_baseline=True,
    ),
}

#: Optional figure title and footnote, drawn INSIDE the constrained-layout
#: budget. The climate figures carry both; the basin map carries neither —
#: which is why FONT_SIZE_TITLE / FONT_SIZE_CAVEAT / COLOR_CAVEAT are shared
#: rather than map furniture, and live in ``plot_style`` (imported at the top).
#: They are re-exported here because ``climate_figures`` still reaches for them
#: through this module; that import moves in the plotting sweep.


#: The vendored Natural Earth extract the locator inset draws. Provenance,
#: licence and the rebuild recipe are in that folder's README. Committed rather
#: than fetched so the figure needs no network — see the module docstring.
BASEMAP_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "basemap"
    / "natural_earth_50m.gpkg"
)

#: Dimension names treated as easting/northing, lowercased. hydromt's ``.raster``
#: accessor sniffed these for us; reading the file directly means saying which
#: spellings count. wflow writes ``latitude``/``longitude``.
_X_DIM_NAMES = ("x", "longitude", "lon")
_Y_DIM_NAMES = ("y", "latitude", "lat")

#: The CRS every layer has to be in. This is not a preference: the panel is
#: ``ccrs.PlateCarree()``, and BOTH the scale bar and the hillshade convert
#: degrees to metres through ``_metres_per_degree``. Hand them a projected layer
#: — metres, or feet — and nothing raises: the map draws, the scale bar reports
#: a distance out by five orders of magnitude, and the relief is shaded as if
#: the basin were flat. A wrong figure that renders is worse than one that
#: fails, so the assumption is now checked instead of stated in a comment.
REQUIRED_CRS_EPSG = 4326

#: Elevation units the ``ELEVATION_LABEL`` default is honest about. A DEM in
#: feet plots perfectly well and gets a bar labelled "m a.s.l.".
_ELEVATION_UNITS = ("m", "meter", "meters", "metre", "metres")

#: Drawing order. Every artist names one of these rather than a bare number, so
#: the stack is legible and reorderable in one place.
Z_RELIEF = 1
Z_RIVER = 3
Z_WATERBODY = 4
Z_SUBCATCHMENT = 5
Z_BASIN_OUTLINE = 6
Z_MARKER = 7
#: The basin outlet sits ABOVE the other point layer. A gauge is routinely
#: snapped to the same cell as the outlet, and whichever layer was drawn second
#: hid the other — the outlet is the one the map cannot afford to lose, so it
#: wins. Draw order alone would not settle it: the outlet is drawn first so its
#: legend entry reads before the gauges', and zorder is what separates "read in
#: this order" from "drawn on top".
Z_OUTLET = 8
Z_FURNITURE = 9

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


def _colorbar_inset(label_lines=1, reserved_top=0.0, band_bottom=0.0):
    """[x0, y0, width, height] for ``ax.inset_axes``, in axes fractions.

    The bar is placed in the band the panel's OTHER blocks leave it:
    ``reserved_top`` is what the locator inset takes off the top, ``band_bottom``
    what the legend takes off the bottom. It keeps ``_COLORBAR_HEIGHT`` while
    that band can hold it and gives way when it cannot, down to
    ``_COLORBAR_MIN_HEIGHT`` — so the three blocks stack inside the map's own
    height instead of the bar being pinned and the legend running off the page.
    ``label_lines`` is how many lines the "top" label wraps to, since each one
    costs the same again.
    """
    label = (
        _COLORBAR_TOP_LABEL_HEADROOM * label_lines
        if _colorbar_label_position() == "top"
        else 0.0
    )
    band_top = 1.0 - reserved_top
    band = max(band_top - band_bottom, 0.0)
    height = min(_colorbar_height(), max(band - label, _COLORBAR_MIN_HEIGHT))
    # Pinned to the TOP of what is free, so the bar sits directly under the
    # locator and any slack falls between the bar and the legend rather than
    # opening a gap under the inset.
    bottom = max(band_top - label - height, band_bottom)
    return (_PANEL_LEFT, bottom, _COLORBAR_WIDTH, height)


def _colorbar_height():
    """The bar's length, shortened when the locator shares its panel."""
    if _LOCATOR_PLACEMENT == "panel" and _locator_drawn():
        return _COLORBAR_HEIGHT_PANEL
    return _COLORBAR_HEIGHT


def _panel_available_width():
    """Usable width to the right of ``_PANEL_LEFT``, in axes fractions.

    The side panel is the strip ``_LAYOUT_RIGHT`` leaves, measured in INCHES;
    the inset is placed in axes fractions. This converts between them, so the
    inset can be clamped to a panel it cannot see.
    """
    panel_in = FIGURE_WIDTH_MM / MM_PER_INCH * (1.0 - _LAYOUT_RIGHT)
    return panel_in / max(_map_width_inches(), 1e-6) - (_PANEL_LEFT - 1.0)


def _panel_locator_box(extent, target_width=None):
    """[x0, y0, w, h] for the locator inset when it sits in the SIDE PANEL.

    Square on the page, like the on-map box, and pinned to the panel's top so
    the reader meets "where is this" before the elevation scale.

    ``target_width`` is normally the LEGEND'S measured width, which is what
    makes the panel's top and bottom blocks come out the same size. The legend
    drives rather than the inset because a legend's width is set by its text and
    cannot be dictated — matplotlib has no width parameter for one, and
    ``mode="expand"`` does not constrain it either (tried: it stretched the
    legend to 1.29 axes fractions). An inset's width is free, so the inset
    yields.

    Two limits still bind, and which one depends on the basin. A WIDE basin
    makes the map panel short, so a square inset is tall in axes fractions and
    the height cap binds. A TALL basin makes the panel tall, the fractions
    shrink, and the panel's own WIDTH binds instead. Either one overrides the
    target, so the two blocks can end up slightly unequal — an inset cropped by
    the canvas or overrunning the colourbar is the worse failure.
    """
    lon_span = max(float(extent[1] - extent[0]), 1e-9)
    lat_span = max(float(extent[3] - extent[2]), 1e-9)
    width = target_width if target_width and target_width > 0 else _LOCATOR_PANEL_WIDTH
    width = min(width, _panel_available_width())
    height = width * lon_span / lat_span
    if height > _LOCATOR_PANEL_MAX_HEIGHT:
        height = _LOCATOR_PANEL_MAX_HEIGHT
        # Back-derive the width so the box stays square on the page.
        width = min(height * lat_span / lon_span, _panel_available_width())
    return [_PANEL_LEFT, 1.0 - height, width, height]


def _publication_rc():
    """The rcParams the figure is drawn under, from the FONT_SIZE_*/WIDTH_*.

    The page-level half comes from ``plot_style.rcparams`` so every figure
    family shares it; ``WIDTH_AXES_SPINE`` is map furniture and is passed in.

    Every value is handed over EXPLICITLY rather than letting ``plot_style``
    read its own globals. The names below resolve in this module's namespace,
    which is what ``dev/scripts/preview_basin_map.py --set`` rebinds; a value
    read inside ``plot_style`` would be that module's own copy, which no
    override touches — the figure would silently ignore the knob.
    """
    return plot_style.rcparams(
        font_size_base=FONT_SIZE_BASE,
        font_size_tick=FONT_SIZE_TICK,
        font_size_legend=FONT_SIZE_LEGEND,
        font_family=FONT_FAMILY,
        axes_linewidth=WIDTH_AXES_SPINE,
    )


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


#: The darkest a SEQUENTIAL raster is allowed to get, in CIE L*. Every map this
#: template draws carries black linework over the raster — the basin outline,
#: the subcatchment divides, the gauge labels' text — and a sequential palette
#: run to its full extent puts that linework on ground too dark to read it
#: against. Observed on a real basin's PET map, where the field is near-constant
#: across the extent so every visible cell landed in the top class and the whole
#: map rendered as one dark brown.
#:
#: Related to :data:`_DARK_RASTER_LIGHTNESS` and deliberately LOWER. That one is
#: a threshold on the area-weighted MEAN, deciding after the fact which way to
#: draw the divides; this is a floor on the DARKEST CLASS, applied to the ramp
#: before anything is drawn. Both answer "can a line be read on this?", at
#: different stages — fix the ground first, then cope with what is left.
#:
#: **45, not 55, and the number is a measured trade rather than a preference.**
#: Clipping the ramp buys linework contrast by spending class separation, so it
#: was swept rather than picked (2026-08-16, six classes, all four styles):
#:
#:     target   darkest L*   min adjacent dL*
#:     none      21 - 29        8.2 - 9.9
#:     55            55         5.1 - 6.2
#:     45            45         6.3 - 7.1
#:     35            35         7.4 - 8.3
#:
#: At 55 the ramp keeps barely half the separation it was chosen for, which
#: trades an unreadable overlay for an unreadable ramp. 45 lifts the darkest
#: class out of the range where black linework disappears (L* 21-29) while
#: holding ~75% of the separation, and 7 dL* is still well clear of the ~3 a
#: reader needs to tell two adjacent large patches apart.
_MIN_RASTER_LIGHTNESS = 45.0

#: How finely :func:`_readable_ramp_ceiling` scans for that stop. 64 steps
#: resolves the ceiling to ~1.5% of the ramp, well inside the width of one
#: class.
_RAMP_CEILING_STEPS = 64


def _readable_ramp_ceiling(ramp, min_lightness=None):
    """The highest 0-1 stop on ``ramp`` still lighter than ``min_lightness``.

    Derived per palette rather than pinned to one fraction, because the same
    fraction means different things on different ramps: 0.85 of ``Blues``
    measures L*=41 while 0.85 of ``Oranges`` measures L*=47, so a single
    hardcoded cap would over-clip one and under-clip the other. Scanning for the
    LIGHTNESS the figure actually needs is the same measured-not-guessed rule
    the palettes themselves were chosen under.

    Returns 1.0 when even the ramp's darkest end is light enough (nothing to
    clip), and never returns below the midpoint — a palette dark throughout
    would otherwise be clipped to a sliver and lose the class separation it
    exists to provide. Better a slightly dark map than an unreadable ramp; the
    divides' own contrast flip is what covers that residue.
    """
    target = _MIN_RASTER_LIGHTNESS if min_lightness is None else float(min_lightness)
    stops = np.linspace(0.5, 1.0, _RAMP_CEILING_STEPS)
    lightness = np.array([_relative_luminance(ramp(float(s))) for s in stops])
    acceptable = stops[lightness >= target]
    return float(acceptable.max()) if acceptable.size else 0.5


def _style_colormap(style, levels=None, floor=None):
    """The style's palette, continuous or cut into ``levels`` classes.

    A style may name a matplotlib colormap or supply its own hex anchors; the
    elevation ramp is anchors because the perceptually-uniform terrain maps are
    not in this environment and one figure does not justify a dependency.

    ``floor`` is the lowest class boundary. When the style reserves its pale end
    for low values and the floor sits above that threshold, the ramp starts at
    ``style.low_clip`` instead of at its palest colour — so white keeps meaning
    "none" rather than "least of a lot".

    The ramp also STOPS short of its darkest end, at ``style.high_clip`` or at
    the derived :func:`_readable_ramp_ceiling`, so the linework drawn over the
    raster stays readable. Applied here rather than to the drawn image so the
    colourbar is built from the same clipped ramp — an alpha on the map would
    have left the bar advertising colours the map never paints.

    **A resolved DIVERGING style is never clipped at the top**, and the guard is
    here rather than at the call site because ``_diverging_colormap`` builds on
    this function — it asks for the base ramp and then samples it itself, so
    routing by caller would not have caught it. Clipping one end of a diverging
    ramp moves its pale middle off the centre, which is the whole encoding:
    ``tests/test_plot_map.py::test_the_pale_middle_of_a_diverging_ramp_lands_on_the_centre``
    caught exactly that when this shipped without the guard, with the above-zero
    class coming out fractionally BLUER than the one below it.
    """
    palette = getattr(style, "palette", None)
    if palette is None:
        ramp = _elevation_colormap(None)
    elif isinstance(palette, str):
        ramp = matplotlib.colormaps[palette]
    else:
        ramp = colors.LinearSegmentedColormap.from_list("style", tuple(palette), N=256)

    start = 0.0
    reserve = getattr(style, "reserve_low_for", None)
    if reserve is not None and floor is not None and float(floor) > float(reserve):
        start = float(getattr(style, "low_clip", 0.0))
    if getattr(style, "diverging_center", None) is not None:
        stop = 1.0  # see the docstring: clipping would move the pale middle
    else:
        stop = getattr(style, "high_clip", None)
        stop = _readable_ramp_ceiling(ramp) if stop is None else float(stop)
    if (start, stop) != (0.0, 1.0):
        ramp = colors.LinearSegmentedColormap.from_list(
            f"{getattr(ramp, 'name', 'style')}_clipped",
            ramp(np.linspace(start, stop, 256)),
            N=256,
        )
    return ramp if levels is None else ramp.resampled(levels)


def _diverging_colormap(style, levels, extend):
    """One colour per class, sampled where that class SITS on the ramp.

    The sequential path asks the colormap for N evenly-spaced colours
    (``ramp.resampled(N)``), which is right when the only thing that matters is
    that adjacent classes differ. It is not enough here: a diverging ramp's pale
    middle has to land on the centre, and ``resampled`` knows nothing about
    where the centre is. Worse, ``_colorbar_extend`` adds a colour at one or
    both ends, so an evenly-resampled ramp shifts by half a class the moment one
    arrow appears — the centre would move depending on whether the basin
    happened to contain an outlier.

    So each class takes the colour at its OWN midpoint's position along a domain
    that IS symmetric about the centre — ``centre +/- half``, where the half-span
    is whichever side of the bar reaches further. Symmetry of the domain rather
    than of the boundaries is what the encoding needs: the two classes either
    side of the centre then sample just below and just above 0.5, so the pale
    middle falls exactly between them, while the boundaries stay free to cover
    the data at a useful resolution.

    The consequence on a one-sided-ish field is correct rather than unfortunate.
    A basin spanning -2 to 30 degC samples the cold half only in its palest few
    percent, because it barely goes below freezing — which is what the reader
    should see.
    """
    ramp = _style_colormap(style)
    centre = float(style.diverging_center)
    half = max(centre - float(levels[0]), float(levels[-1]) - centre, 1e-12)
    step = float(levels[1] - levels[0]) if len(levels) > 1 else half

    def at(value):
        return ramp(float(np.clip((value - (centre - half)) / (2.0 * half), 0.0, 1.0)))

    boundaries = np.asarray(levels, dtype=float)
    colours = [at(value) for value in (boundaries[:-1] + boundaries[1:]) / 2.0]
    # The arrows continue the ramp rather than jumping to its extreme end: one
    # more half-step out, which is where the next class's midpoint would sit.
    if extend in ("min", "both"):
        colours.insert(0, at(boundaries[0] - step / 2.0))
    if extend in ("max", "both"):
        colours.append(at(boundaries[-1] + step / 2.0))
    return colors.ListedColormap(colours)


def _classified_colormap(style, levels, extend):
    """The colour ramp for one classified raster, whichever rule applies."""
    if getattr(style, "diverging_center", None) is not None:
        return _diverging_colormap(style, levels, extend)
    extra = {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]
    return _style_colormap(style, len(levels) - 1 + extra, floor=levels[0])


def resolve_diverging_style(raster, style):
    """Activate a style's declared midpoint, but only for a field that spans it.

    A diverging ramp asserts that its pale middle is a MEANINGFUL value. That is
    true of 0 degC (ice or not) and pH 7 (acid or alkaline), and it is true only
    while the field actually reaches both sides of it: a basin whose soil runs
    4.7-5.5 has no alkaline ground, so an absolute pH ramp would spend half its
    colours on values that do not occur and compress the ones that do into two
    classes. Such a field keeps the sequential ramp, which is not a compromise —
    it is the honest encoding of a one-sided quantity.

    Centring a diverging ramp on the data's own mean instead is the standard way
    this rule gets broken. The centre here is always the DECLARED physical
    value, never derived, and the only decision this function makes is whether
    to use it at all.

    Idempotent: a style already resolved carries the diverging palette and still
    declares the same ``diverging_at``, so resolving it again returns the same
    thing. That is what lets ``plot_raster_map`` resolve unconditionally without
    caring whether its caller already did.
    """
    centre, palette = style.diverging_at, style.diverging_palette
    if centre is None or palette is None:
        return style
    values = np.asarray(raster.values)
    values = values[np.isfinite(values)]
    if values.size == 0 or values.min() >= centre or values.max() <= centre:
        return style
    return style.replace(
        palette=palette,
        diverging_center=float(centre),
        # Both belong to a one-sided scale and mean nothing on a centred one:
        # the baseline IS the centre, and the pale end is the middle rather
        # than an end to reserve.
        zero_baseline=False,
        reserve_low_for=None,
    )


def resolve_temperature_style(raster, style=None):
    """The temperature style, switched to diverging when the field crosses 0.

    Kept as its own name because ``climate_figures`` has called it since the
    2026-08 map sweep and because temperature is where the rule was first
    written down. The rule itself is now general — see
    :func:`resolve_diverging_style`; this only supplies the default style.

    It used to rebuild the style field by field and dropped ``step_ladder`` in
    the process, so a diverging temperature bar silently stepped in 0.15 degC
    off the general ladder instead of the 0.25/0.5/1 the sequential one uses.
    That is the exact defect ``replace`` exists to prevent, and the resolver
    goes through it.
    """
    return resolve_diverging_style(
        raster, RASTER_STYLES["temp"] if style is None else style
    )


#: Mantissas a class WIDTH may take, as a decade ladder. Wider than the 1/2/5
#: used for the scale bar: with a capped tick count the step is what controls
#: how many classes the bar gets, and 1/2/5 leaves gaps too coarse to land in.
#:
#: These rungs give the steps a reader expects of each quantity — 25/50/100/
#: 150/200 mm for rainfall, 0.25/0.5/1 degC for temperature — at every decade,
#: because the ladder is multiplicative. ``4`` was dropped when the tick cap
#: rose to 9: it existed only so a five-tick bar over 0-140 m could land on 40,
#: and it put 40 mm and 0.4 degC in reach of quantities that never want them.
#: A style may override the ladder with its own (``RasterStyle.step_ladder``).
_STEP_LADDER = (1.0, 1.5, 2.0, 2.5, 5.0, 10.0)


def _nice_step_up(value, ladder=None):
    """Round a class width UP to the next ``_STEP_LADDER`` rung x 10^n.

    Up, not down: ``_nice_round_length`` rounds down, which for a class width
    means MORE classes than asked for, and a bar of twelve near-identical
    browns is the thing the discretisation exists to avoid.
    """
    if value <= 0:
        return 1.0
    rungs = tuple(ladder) if ladder else _STEP_LADDER
    exponent = np.floor(np.log10(value))
    fraction = value / 10.0**exponent
    step = next((rung for rung in rungs if fraction <= rung), 10.0)
    return float(step * 10.0**exponent)


def _next_step_up(step, ladder=None):
    """The ladder rung above ``step`` — used to force the tick count down."""
    return _nice_step_up(step * (1.0 + 1e-9), ladder)


def _zero_baseline(lower, upper, enabled=None):
    """``lower``, dropped to 0 when the data can afford it. See the constants."""
    if enabled is None:
        enabled = _ELEVATION_STARTS_AT_ZERO
    if not enabled or lower <= 0.0 or upper <= 0.0:
        return lower
    if (upper - lower) / upper >= _ZERO_BASELINE_MIN_SPAN_FRACTION:
        return 0.0
    return lower


def _equal_interval_levels(lower, upper, zero_baseline=None, ladder=None):
    """Evenly spaced class boundaries on a round step — the default rule.

    The step comes from ``_COLORBAR_LEVELS`` but is then widened, a ladder rung
    at a time, until the boundary count fits ``_COLORBAR_MAX_TICKS``. Rounding
    the width to a readable number means the class count cannot be requested
    exactly, so the target sets the ambition and the cap sets the limit.
    """
    baseline = _zero_baseline(lower, upper, zero_baseline)
    step = _nice_step_up((upper - baseline) / max(_COLORBAR_LEVELS, 1), ladder)
    max_ticks = max(int(_COLORBAR_MAX_TICKS), 2)
    for _ in range(_STEP_WIDEN_ATTEMPTS):
        # Put the boundaries on multiples of the step, so a basin floor of
        # 1903 m labels as 1900 rather than carrying its own arbitrary value up
        # the bar.
        floor = float(np.floor(baseline / step) * step)
        # ceil, then +1 boundary: the top class must CONTAIN the highest cell,
        # not end at it, or the summit renders as nodata.
        count = int(np.ceil((upper - floor) / step))
        if count + 1 <= max_ticks:
            return floor + step * np.arange(count + 1)
        step = _next_step_up(step, ladder)
    return floor + step * np.arange(count + 1)


def _diverging_levels(lower, upper, centre, ladder=None):
    """Class boundaries on a round step, counted OUT FROM ``centre``.

    Its own function rather than a branch inside ``_equal_interval_levels``,
    because that rule is hostile to a pinned value: it drops the baseline to
    zero, and it snaps the floor to ``floor(baseline/step)*step``, which moves
    the centre off a boundary the moment the centre is not a multiple of the
    step. pH 7 on a 0.15 step is exactly that case.

    The one invariant: ``centre`` IS a boundary. A diverging palette's pale
    middle sits at the join between two classes, so if the centre is not that
    join the pale colour lands somewhere else and the map says freezing is at
    3 degC. Counting the boundaries out from the centre rather than up from a
    floor is what guarantees it, for any centre and any step.

    The levels are NOT forced symmetric about the centre, and the first version
    of this made them so. Measured on a field spanning -2 to 30 degC — a warm
    basin with a few frosty cells — symmetry gave a bar running -50 to +50 in
    four classes, because the half-span is set by whichever side reaches
    further. That is a worse figure than the sequential one it replaced. The
    boundaries here cover the DATA, so the same field gets a 5 degC step across
    its own range; what stays symmetric is the COLOUR DOMAIN, which is where
    symmetry was actually doing the work — see ``_diverging_colormap``.
    """
    span = max(upper - lower, 1e-12)
    step = _nice_step_up(span / max(_COLORBAR_LEVELS, 1), ladder)
    max_ticks = max(int(_COLORBAR_MAX_TICKS), 3)
    low_index, high_index = -1, 1
    for _ in range(_STEP_WIDEN_ATTEMPTS):
        # floor/ceil so the outermost classes CONTAIN the extremes rather than
        # ending at them; min/max so the centre is always inside the range even
        # for a field that barely reaches across it.
        low_index = min(int(np.floor((lower - centre) / step)), -1)
        high_index = max(int(np.ceil((upper - centre) / step)), 1)
        if high_index - low_index + 1 <= max_ticks:
            break
        step = _next_step_up(step, ladder)
    return float(centre) + step * np.arange(low_index, high_index + 1)


def _ladder_values(lower, upper):
    """Every readable break value spanning ``[lower, upper]``, ascending.

    The ladder is multiplicative (``_BREAK_LADDER`` x 10^n), which is what lets
    one rule serve a 1-200 m coastal basin and a 400-3800 m alpine one: the
    rungs get coarser as the numbers do, so a break is always about as precise
    as the reader can use.
    """
    lower = max(float(lower), 1e-9)
    exponents = range(
        int(np.floor(np.log10(lower))) - 1,
        int(np.ceil(np.log10(max(upper, lower)))) + 1,
    )
    return np.unique(
        np.concatenate([np.array(_BREAK_LADDER) * 10.0**e for e in exponents])
    )


def _snap_to_ladder(value, rungs, mode="near"):
    """``value`` moved onto the nearest ladder rung, or the one below/above it."""
    if mode == "down":
        below = rungs[rungs <= value]
        return float(below[-1]) if below.size else float(rungs[0])
    if mode == "up":
        above = rungs[rungs >= value]
        return float(above[0]) if above.size else float(rungs[-1])
    return float(rungs[np.argmin(np.abs(rungs - value))])


def _weighted_quantiles(values, weights, probabilities):
    """Quantiles of ``values`` weighted by ``weights``, ascending.

    ``np.quantile`` takes no weights, and the weights here are cell AREAS — the
    thing "equal-area classes" is named after. Standard definition: sort, take
    the cumulative weight at each sample's midpoint, interpolate.
    """
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    cumulative = np.cumsum(sorted_weights)
    total = cumulative[-1]
    if total <= 0:
        return np.quantile(values, probabilities)
    positions = (cumulative - 0.5 * sorted_weights) / total
    return np.interp(probabilities, positions, sorted_values)


def _equal_area_levels(values, lower, upper, weights=None, zero_baseline=None):
    """Quantile class boundaries, snapped onto the readable ladder.

    Each class holds roughly the same AREA, so a skewed DEM spends the whole
    ramp on ground the basin actually has. Snapping is what keeps the bar
    labelled with numbers rather than with the basin's own percentiles; a snap
    that collides with its neighbour is dropped, so the count comes out at or
    below ``_COLORBAR_LEVELS`` and the caller checks it against ``_MIN_CLASSES``.
    """
    inside = (values >= lower) & (values <= upper)
    if not np.any(inside):
        return np.array([lower, upper])
    probabilities = np.linspace(0.0, 1.0, _COLORBAR_LEVELS + 1)
    if weights is None:
        quantiles = np.quantile(values[inside], probabilities)
    else:
        quantiles = _weighted_quantiles(values[inside], weights[inside], probabilities)
    rungs = _ladder_values(lower, upper)
    floor = _zero_baseline(lower, upper, zero_baseline)
    levels = [0.0 if floor <= 0.0 else _snap_to_ladder(floor, rungs, "down")]
    for value in quantiles[1:-1]:
        rung = _snap_to_ladder(value, rungs, "near")
        if rung > levels[-1]:
            levels.append(rung)
    top = _snap_to_ladder(upper, rungs, "up")
    if top <= levels[-1]:
        top = _snap_to_ladder(levels[-1] * 1.3, _ladder_values(lower, upper * 2), "up")
    levels.append(top)
    return np.array(levels)


def _finite_cells(dem):
    """The DEM's valid cells as ``(values, area_weights)``, both flattened.

    The weights are ``cos(latitude)``, which is what a cell's ground area is
    proportional to on a lat-lon grid. Counting cells instead treats a cell at
    60 deg as the equal of one at the equator, and it is half the area — so
    "equal-AREA" classes built from counts are not equal-area, and neither is
    the largest-class test that decides whether to use them.

    Negligible on a basin spanning a fraction of a degree (this fixture varies
    by 0.001%) and not negligible on the continental-scale basins the same code
    has to serve, which is the whole reason it is weighted rather than argued
    about per basin.
    """
    x_dim, y_dim = spatial_dim_names(dem)
    values = dem.transpose(y_dim, x_dim).values
    latitudes = np.asarray(dem[y_dim].values, dtype=float)
    weights = np.repeat(
        np.cos(np.radians(np.clip(latitudes, -89.9999, 89.9999)))[:, None],
        values.shape[1],
        axis=1,
    )
    finite = np.isfinite(values)
    return values[finite], weights[finite]


def _class_area_shares(values, levels, weights=None):
    """Fraction of the basin's AREA falling in each class."""
    totals, _ = np.histogram(values, bins=levels, weights=weights)
    total = float(totals.sum())
    return totals / total if total > 0 else totals


def _class_levels(raster, style):
    """Class boundaries for any quantity, following ``style``.

    The generalisation of ``_elevation_levels``: same two rules and same
    readable-ladder snapping, with the clip quantiles, zero baseline and
    classification mode read from the style rather than from elevation's
    constants. See ``_elevation_levels`` for what the rules are and why.
    """
    if getattr(style, "levels", None) is not None:
        # Handed in, not derived: the caller is pinning this bar to another
        # figure's. See ``climate_figures`` and its shared-levels sidecar.
        return np.asarray(style.levels, dtype=float)
    ladder = getattr(style, "step_ladder", None)
    lower, upper = (
        float(value) for value in raster.quantile(list(style.clip_quantiles)).compute()
    )
    if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
        upper = lower + 1.0

    # An ACTIVE centre takes its own rule, before the classification branch:
    # both rules below optimise the class widths for the data's own histogram,
    # and neither can keep a boundary pinned to a value while doing it. Only
    # `resolve_diverging_style` sets this, and only for a field that spans the
    # centre, so an unresolved style never reaches here.
    centre = getattr(style, "diverging_center", None)
    if centre is not None:
        return _diverging_levels(lower, upper, centre, ladder)

    if style.classification not in _ELEVATION_CLASSIFICATIONS:
        raise ValueError(
            f"classification={style.classification!r}; expected one of "
            f"{_ELEVATION_CLASSIFICATIONS}"
        )
    equal_interval = _equal_interval_levels(lower, upper, style.zero_baseline, ladder)
    if style.classification == "equal_interval":
        return equal_interval

    values, weights = _finite_cells(raster)
    if values.size == 0:
        return equal_interval
    shares = _class_area_shares(values, equal_interval, weights)
    if (
        shares.max() <= _MAX_CLASS_AREA_SHARE
        and not (shares < _EMPTY_CLASS_AREA_SHARE).any()
    ):
        return equal_interval

    equal_area = _equal_area_levels(values, lower, upper, weights, style.zero_baseline)
    # A field with only a handful of distinct values — the wflow forcing on a
    # basin driven by a 2x3 reanalysis grid is one — collapses most of its
    # snapped breaks into each other. Equal interval is then the better of two
    # imperfect answers, which is what this guard is for.
    if len(equal_area) - 1 < _MIN_CLASSES:
        return equal_interval
    return equal_area


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
            raise ValueError(
                f"cannot derive a resolution from a {dim} of length {coord.size}"
            )
        resolutions.append(float(coord[1] - coord[0]))
    return tuple(resolutions)


def _pad_extent(extent, buffer_deg=None):
    """Pad an extent proportionally, or by an explicit fixed degree value."""
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in extent)
    fixed_buffer_deg = _EXTENT_BUFFER_DEG if buffer_deg is None else buffer_deg
    if fixed_buffer_deg is None:
        lon_pad = max(
            (lon_max - lon_min) * _EXTENT_BUFFER_FRACTION,
            _EXTENT_BUFFER_MIN_DEG,
        )
        lat_pad = max(
            (lat_max - lat_min) * _EXTENT_BUFFER_FRACTION,
            _EXTENT_BUFFER_MIN_DEG,
        )
    else:
        lon_pad = lat_pad = float(fixed_buffer_deg)
    return np.array(
        [
            lon_min - lon_pad,
            lon_max + lon_pad,
            lat_min - lat_pad,
            lat_max + lat_pad,
        ]
    )


def map_extent(da, buffer_deg=None):
    """``[lon_min, lon_max, lat_min, lat_max]`` covering the DEM, plus padding.

    Replaces ``da.raster.box.buffer(...).total_bounds``. Coordinates are cell
    CENTRES, so the box reaches half a cell beyond them on each side — dropping
    that half-cell shrinks the frame by one pixel row and column. Padding is
    proportional to each axis unless ``buffer_deg`` explicitly fixes it.
    """
    x_dim, y_dim = spatial_dim_names(da)
    res_x, res_y = pixel_resolution(da)
    half_x, half_y = abs(res_x) / 2.0, abs(res_y) / 2.0
    x, y = da[x_dim].values, da[y_dim].values
    return _pad_extent(
        [
            x.min() - half_x,
            x.max() + half_x,
            y.min() - half_y,
            y.max() + half_y,
        ],
        buffer_deg=buffer_deg,
    )


def check_geographic_inputs(raster, layers, value_label=None, expected_units=None):
    """Raise unless every layer is geographic and the DEM's units match its label.

    Three assumptions this figure has always made and never tested:

    * every vector layer is in ``EPSG:4326``;
    * the DEM's coordinates are degrees, not projected units;
    * the DEM's values are metres, which is what ``ELEVATION_LABEL`` says.

    All three are readable from the data — the fixture's DEM carries
    ``units: "m"`` and a ``spatial_ref``, and every GeoJSON carries its CRS — so
    the cost of checking is a few comparisons, against a failure mode that
    produces a plausible-looking map with a scale bar out by 10^5.

    Units are a WARNING, not an error: a DEM in feet is still a valid map once
    the caller passes a ``value_label`` that says so, and refusing to plot
    it would be the wrong answer. The CRS is an error, because no label can
    repair a scale bar computed from the wrong kind of number.
    """
    wrong_crs = {
        name: str(layer.crs)
        for name, layer in layers.items()
        if layer is not None
        and len(layer) > 0
        and layer.crs is not None
        and layer.crs.to_epsg() != REQUIRED_CRS_EPSG
    }
    if wrong_crs:
        raise ValueError(
            f"plot_basin_map needs every layer in EPSG:{REQUIRED_CRS_EPSG}; got "
            f"{wrong_crs}. Reproject with .to_crs(epsg={REQUIRED_CRS_EPSG}) first "
            "— the scale bar and the hillshade both convert degrees to metres."
        )

    x_dim, y_dim = spatial_dim_names(raster)
    x, y = raster[x_dim].values, raster[y_dim].values
    if np.nanmax(np.abs(x)) > 360.0 or np.nanmax(np.abs(y)) > 90.0:
        raise ValueError(
            f"the DEM's coordinates do not look geographic: {x_dim} reaches "
            f"{np.nanmax(np.abs(x)):.4g}, {y_dim} reaches {np.nanmax(np.abs(y)):.4g}. "
            f"Reproject the grid to EPSG:{REQUIRED_CRS_EPSG}."
        )

    units = str(raster.attrs.get("units", "")).strip().lower()
    label = ELEVATION_LABEL if value_label is None else value_label
    expected = _ELEVATION_UNITS if expected_units is None else expected_units
    if units and expected and units not in expected:
        warnings.warn(
            f"the DEM declares units={units!r} but the colourbar is labelled "
            f"{label!r}. Pass value_label= to match the data.",
            stacklevel=2,
        )


def extent_from_layer(layer, buffer_deg=None):
    """``[lon_min, lon_max, lat_min, lat_max]`` covering a vector layer.

    The companion to :func:`map_extent`, which frames a map on its RASTER. Two
    maps of the same area drawn from different grids frame differently that way
    — the wflow forcing is masked to the basin, the source extraction is a
    handful of reanalysis cells reaching far beyond it — so the pair cannot be
    read side by side. Framing both on the same VECTOR layer is what makes them
    comparable, and the basin is the subject of both. Padding is proportional
    to each axis unless ``buffer_deg`` explicitly fixes it.
    """
    if not _present(layer):
        return None
    lon_min, lat_min, lon_max, lat_max = layer.total_bounds
    return _pad_extent(
        [lon_min, lon_max, lat_min, lat_max],
        buffer_deg=buffer_deg,
    )


def _raster_within(raster, extent):
    """``raster`` cut to ``extent``, keeping the cells that touch its edges.

    The colourbar classifies whatever it is handed, so a raster reaching past
    the frame puts values the reader cannot see into the class breaks — the
    source-grid extraction spans several degrees around a basin a fraction of a
    degree wide, and framed on the basin its bar described mostly off-frame
    cells. A bar must describe the map it sits on.

    Cells are kept when they INTERSECT the frame — half a cell of padding on
    each side — because a cell straddling the edge is genuinely part of the
    picture. A centre-in-frame test was tried and is wrong here: on a 0.25 deg
    reanalysis grid under a 0.15 deg basin exactly one centre falls inside, so
    the range collapsed to a single value and the classifier invented 0.2 mm
    steps around it.
    """
    try:
        x_dim, y_dim = spatial_dim_names(raster)
        res_x, res_y = (abs(v) / 2.0 for v in pixel_resolution(raster))
    except ValueError:
        return raster
    lon_min, lon_max, lat_min, lat_max = (float(v) for v in extent)
    x, y = raster[x_dim], raster[y_dim]
    touching = raster.sel(
        {
            x_dim: x[(x >= lon_min - res_x) & (x <= lon_max + res_x)],
            y_dim: y[(y >= lat_min - res_y) & (y <= lat_max + res_y)],
        }
    )
    return touching if touching.size else raster


#: Where along a style's ramp its series colour is taken from. High enough to
#: read as a line on white, low enough not to be the near-black end of a
#: sequential palette.
_SERIES_COLOR_POSITION = 0.72


def style_series_color(style):
    """The single colour this quantity takes outside a map.

    Drawn from the style's own ramp so a variable keeps one identity across the
    figure set: the blue a reader meets on the precipitation map is the blue of
    its annual series. A style may pin ``series_color`` instead.
    """
    if getattr(style, "series_color", None) is not None:
        return style.series_color
    return _style_colormap(style)(_SERIES_COLOR_POSITION)


def series_figure_size(aspect=0.42):
    """Figure size in inches for a NON-map figure at the shared page width.

    Same declared width as every map, so a report setting these side by side
    gets one column width rather than three.
    """
    width_in = FIGURE_WIDTH_MM / MM_PER_INCH
    return width_in, width_in * aspect


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


def _map_aspect(extent):
    """The map panel's height/width ratio, clamped to the usable range.

    cartopy locks a PlateCarree panel to equal DEGREES, so the rendered aspect
    is the extent's own ratio -- not the true ground aspect.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    span_lon = max(float(lon_max - lon_min), 1e-9)
    aspect = float(lat_max - lat_min) / span_lon
    return float(np.clip(aspect, _MIN_MAP_ASPECT, _MAX_MAP_ASPECT))


def _map_width_inches():
    """Width of the map panel itself, once the side panel and ticks are taken."""
    return FIGURE_WIDTH_MM / MM_PER_INCH * _LAYOUT_RIGHT - _TICK_LABEL_WIDTH_IN


def _map_height_inches(extent):
    """Height of the map panel in inches — the axes an axes FRACTION is of.

    Needed to turn a legend's height in POINTS into the fraction the panel
    stack is measured in.
    """
    return _map_width_inches() * _map_aspect(extent)


def _figure_size(extent):
    """Figure size in inches: declared width, height from the basin's aspect."""
    # Height follows the MAP PANEL, not the full page: sizing off the figure
    # width leaves the aspect-locked panel floating in an over-tall cell.
    return (
        FIGURE_WIDTH_MM / MM_PER_INCH,
        _map_height_inches(extent) + _FURNITURE_HEIGHT_IN,
    )


def _legend_height_fraction(extent, entries):
    """Predicted legend height, as a fraction of the map's height.

    The FALLBACK for ``_measure_legend``, used only when no renderer can be
    obtained. Predicts from the entry count and the font size, which is close
    enough to stack three blocks that must not overlap but not close enough to
    align two edges — which is why the measured path exists.
    """
    rows = max(int(entries), 0) + (1 if _LEGEND_TITLE else 0)
    if rows == 0:
        return 0.0
    points = FONT_SIZE_LEGEND * (rows * _LEGEND_ROW_FACTOR + 2.0 * _LEGEND_BORDER_PAD)
    return (points / 72.0) / max(_map_height_inches(extent), 1e-6)


def _measure_legend(fig, legend, extent):
    """The legend's (width, height) in AXES FRACTIONS, measured not predicted.

    Alignment is the reason this is measured. Sizing the locator inset to a
    PREDICTED legend width leaves the two panel blocks visibly unequal whenever
    the prediction is a few percent out, and a character-count estimate is
    always a few percent out — it cannot know the font's real advance widths.

    Measured in INCHES and divided by the map panel's DESIGN size, not read off
    ``ax.transAxes``. The legend is built before the DEM, the ticks and the
    furniture, and constrained layout resizes the axes as each of those arrives:
    the same legend measured through ``transAxes`` reported 0.198 axes fractions
    early and 0.215 at the end, which is exactly the 8% error that made the two
    blocks visibly unequal. Its PHYSICAL size never changes, and the design size
    is what the figure was built to — measured within 0.5% of the settled axes.

    Returns ``None`` when no renderer can be had, so the caller can fall back to
    the estimate rather than crash on a backend that cannot measure text.
    """
    renderer = None
    for attempt in (0, 1):
        try:
            renderer = fig.canvas.get_renderer()
            break
        except AttributeError:
            if attempt:
                return None
            # Some backends only build a renderer once something has been laid
            # out; a layout pass is cheap here, before the DEM is drawn.
            fig.draw_without_rendering()
    if renderer is None:
        return None
    box = legend.get_window_extent(renderer)
    return (
        float(box.width / fig.dpi) / max(_map_width_inches(), 1e-6),
        float(box.height / fig.dpi) / max(_map_height_inches(extent), 1e-6),
    )


def _panel_layout(extent, label_lines=1, legend_size=(0.0, 0.0)):
    """Where the side panel's three blocks go, stacked to fit the map's height.

    Top to bottom: the locator inset (when it is in the panel), the colourbar
    with its label, then the legend. The locator is pinned to the top and the
    legend to the BOTTOM — anchoring the legend at y=0 and letting it grow
    upward is what makes a bottom overflow impossible, rather than something
    caught by re-tuning a constant after looking at a render.

    ``legend_size`` is the legend's MEASURED (width, height) in axes fractions.
    Both matter: the height is what the colourbar is sized against, and the
    width is what the locator is matched to.
    """
    legend_width, legend_height = legend_size
    box = None
    if _LOCATOR_PLACEMENT == "panel" and _locator_drawn():
        box = _panel_locator_box(extent, legend_width)
    reserved_top = box[3] + _LOCATOR_PANEL_GAP if box else 0.0
    band_bottom = legend_height + _LEGEND_GAP if legend_height else 0.0
    return {
        "colorbar": _colorbar_inset(label_lines, reserved_top, band_bottom),
        "locator": box,
    }


def _coordinate_format(span_degrees):
    """Decimal places that suit the basin's size, not every basin's."""
    if span_degrees > 5.0:
        return ".0f"
    return ".1f" if span_degrees > 1.0 else ".2f"


def _graticule_ticks(extent, max_ticks=GRATICULE_MAX_TICKS):
    """Shared tick positions for the grid LINES and the axis LABELS.

    The latitude window is clamped to +/-90 BEFORE the ticks are chosen.
    ``map_extent`` pads the DEM bounds proportionally plus half a cell and clamps
    nothing, so a basin near a pole yields ``lat_max > 90`` and the graticule
    would label a latitude that does not exist.

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

    # A GeoAxes draws its box through the single ``geo`` spine; the four
    # ordinary spines are what an L-frame is built from. Exactly one of the two
    # mechanisms is used, so they cannot both draw the left edge.
    if _MAP_FRAME == "box":
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        frame = ax.spines["geo"]
        frame.set_visible(True)
        frame.set_linewidth(WIDTH_AXES_SPINE)
        frame.set_edgecolor(COLOR_BASIN_OUTLINE)
    else:
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
    if _LOCATOR_PLACEMENT == "panel":
        # It is not on the map, so it claims no map corner — and the scale bar
        # gets that corner back rather than avoiding something that is not there.
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
        x_start = lon_max - _FURNITURE_INSET_X * span_lon - length_deg
    else:
        # The same inset the north arrow uses, so the bar's left end sits
        # directly under the arrow when both are on the left.
        x_start = lon_min + _FURNITURE_INSET_X * span_lon
    if corner.startswith("upper"):
        y_bar = lat_max - (_SCALE_BAR_INSET_Y + 0.04) * span_lat
    else:
        y_bar = lat_min + _SCALE_BAR_INSET_Y * span_lat

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


def _locator_span(extent):
    """Half-width of the locator window, in degrees — the auto rule or the pin.

    Sized so the basin's LONG axis comes out at
    ``_LOCATOR_TARGET_BASIN_FRACTION`` of the window's width, then snapped UP to
    the next ladder rung. Up rather than nearest: erring toward a wider window
    costs some of the basin's apparent size and keeps the context that makes the
    inset worth drawing, while erring narrow can crop the very coastline the
    reader is orienting against.
    """
    if _LOCATOR_SPAN_DEG != "auto":
        return float(_LOCATOR_SPAN_DEG)
    basin_span = max(float(extent[1] - extent[0]), float(extent[3] - extent[2]), 1e-9)
    half_width = basin_span / (2.0 * max(_LOCATOR_TARGET_BASIN_FRACTION, 1e-6))
    # A basin bigger than the target implies must still fit inside its window.
    half_width = max(half_width, basin_span * _LOCATOR_MIN_SPAN_MARGIN)
    rung = next((rung for rung in _LOCATOR_SPAN_LADDER if rung >= half_width), None)
    # Off the ladder rather than clamped to its top rung: falling back to the
    # largest rung is what let a basin wider than 2 x 12 deg overflow its own
    # locator window. The ladder exists to keep the window on a round number,
    # which is a nicety; containing the subject is not.
    return float(rung) if rung is not None else float(half_width)


def _locator_window(extent):
    """The locator's own extent: a square window centred on the basin.

    Square in DEGREES, so the inset box can be made square on the page and the
    window fills it without letterboxing. Clamped to the world, and re-centred
    rather than clipped at the poles so a high-latitude basin still gets a full
    window rather than a half-empty one.
    """
    centre_lon = 0.5 * (extent[0] + extent[1])
    centre_lat = 0.5 * (extent[2] + extent[3])
    span = _locator_span(extent)
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
    left = _LOCATOR_MARGIN if corner.endswith("left") else 1.0 - _LOCATOR_MARGIN - width
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
    return gpd.read_file(
        BASEMAP_PATH,
        layer=layer,
        bbox=tuple((window[0], window[2], window[1], window[3])),
    )


def _locator_cities(window):
    """The few most prominent cities inside the window, largest first."""
    places = _read_basemap("places", window)
    if places.empty:
        return places
    places = places[places["scalerank"] <= _LOCATOR_CITY_MAX_SCALERANK]
    return places.sort_values("pop_max", ascending=False).head(_LOCATOR_MAX_CITIES)


def _add_locator_inset(ax, extent, basin, corner, box=None):
    """A small "where is this" map: land, sea, borders, cities, and the basin.

    ``box`` is the inset's [x0, y0, w, h] in axes fractions, normally from
    ``_panel_layout`` so its width matches the legend's. Omitted, it is derived
    from the constants alone.

    Skips itself, with a note, when the vendored basemap is absent — a copy of
    this module taken to another project should still render a basin map, just
    without the inset. Silence would be worse: an inset that quietly never
    appears reads as a layout bug.
    """
    if not LOCATOR_ENABLED:
        return None
    if not BASEMAP_PATH.is_file():
        print(f"note: locator inset skipped, no basemap at {BASEMAP_PATH}")
        return None
    in_panel = _LOCATOR_PLACEMENT == "panel"
    if corner is None and not in_panel:
        return None

    window = _locator_window(extent)
    if box is None:
        box = _panel_locator_box(extent) if in_panel else _locator_box(extent, corner)
    inset = ax.inset_axes(box, projection=ccrs.PlateCarree())
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
            markersize=_LOCATOR_CITY_MARKER_SIZE**0.5,
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

    # The basin's OWN outline, filled, rather than a mark standing in for it.
    # It is small at this window — the fixture spans 0.24 deg in a 16 deg
    # frame — but its footprint, elongation and orientation are real information
    # a centroid dot cannot carry, and the filled shape stays findable at the
    # size the edge alone would not.
    basin.plot(
        ax=inset,
        facecolor=COLOR_LOCATOR_BASIN,
        edgecolor=COLOR_LOCATOR_BASIN_EDGE,
        linewidth=WIDTH_LOCATOR_BASIN,
        zorder=Z_FURNITURE,
        # A halo, for the same reason the scale bar's numbers carry one: the
        # basin is a few points across and lands wherever it lands — on land, on
        # a border, or under the label of the city it sits next to, which is the
        # commonest case because basins and cities share rivers. The white ring
        # is what separates it from all three without enlarging it.
        path_effects=[
            pe.withStroke(linewidth=HALO_WIDTH_LOCATOR_BASIN, foreground=COLOR_HALO)
        ],
    )

    inset.spines["geo"].set_linewidth(WIDTH_LOCATOR_FRAME)
    inset.spines["geo"].set_edgecolor(COLOR_BASIN_OUTLINE)
    return inset


def _add_north_arrow(ax):
    """A north arrow — exactly vertical, which PlateCarree guarantees.

    Sits top-left, on the same vertical as the scale bar below it. The legend
    and the locator both live in the side panel, so the map's left edge is the
    only furniture column and the two items on it share one inset.
    """
    tip_y, tail_y = _NORTH_ARROW_POSITION
    # Shared with the scale bar; see _FURNITURE_INSET_X.
    x_fraction = _FURNITURE_INSET_X
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


def _clip_to_basin(layer, basin):
    """``layer`` cut to the basin's footprint, or unchanged if that fails.

    Returns the original layer on any geometry error rather than dropping it:
    a river network drawn slightly too far is a lesser fault than one silently
    missing, and an invalid polygon is the caller's data problem, not this
    figure's to resolve.
    """
    if not _present(layer) or not _present(basin):
        return layer
    try:
        clipped = gpd.clip(layer, basin)
    except Exception:
        return layer
    return clipped if len(clipped) else layer


def _divide_linework(subbasins):
    """The subcatchment divides as ONE merged linework, not one ring per polygon.

    ``subbasins.boundary`` returns a closed ring per subcatchment, so every
    INTERNAL divide — which by definition belongs to the two subcatchments on
    either side of it — gets drawn twice. Two dashed lines at the same place
    with different phase interleave, and the gaps of one are filled by the dashes
    of the other: the shared edges render SOLID while the outer ones stay dashed.
    Observed at 600 dpi on the fixture, where it read as the divides being
    styled inconsistently rather than as double-drawing.

    ``union_all`` nodes the rings and merges the duplicated segments into single
    lines, so every divide is drawn once and the dash pattern means the same
    thing everywhere. It also stops the halo being laid down twice.
    """
    return gpd.GeoSeries([subbasins.geometry.boundary.union_all()], crs=subbasins.crs)


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


def _wrap_label(fig, text, max_width_inches, fontsize):
    """The colourbar title: the quantity on one line, its unit on the next.

    **The unit break is UNCONDITIONAL**, which is the whole of the 2026-08-16
    change here. It used to fire only once the whole label overflowed the
    panel, so a folder of sibling maps came out inconsistent by label length
    alone: "Potential evaporation (mm y-1)" was long enough to split and
    "Precipitation (mm y-1)" was not, leaving two figures a reader compares
    side by side with their bars at different heights and their titles in
    different shapes. Splitting always costs one line on the short ones and
    buys a family that looks like a family.

    It is also the right break on its own merits. A quantity's name and its
    unit are the two things a reader parses separately, so "Precipitation"
    over "(mm y-1)" reads as one label on two lines — where greedy wrapping
    gives "Precipitation (mm" over "y-1)", splitting the unit itself into two
    half-thoughts.

    Each part is then wrapped to ``max_width_inches`` in its own right, since
    a long quantity name still has to fit the panel it is left-aligned to.
    Wrapping is MEASURED rather than guessed at a character count, because the
    width that matters is the panel's and a character estimate cannot know the
    font's advance widths.

    A label with no parenthesised unit is wrapped and nothing more — it yields
    one line when it fits. Every style in ``RASTER_STYLES`` carries a unit, so
    that path is for a caller-supplied label (``plot_basin_map``'s
    ``elevation_label``) that omits one; inventing a second line for it would
    mean printing an empty one.

    Falls back to the unwrapped text when no renderer is available; a too-wide
    title is better than no figure.
    """
    text = str(text)
    if max_width_inches <= 0:
        return text
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return text

    def width_of(line):
        artist = mtext.Text(0, 0, line, fontsize=fontsize, figure=fig)
        return artist.get_window_extent(renderer).width / fig.dpi

    def wrapped(part):
        """``part`` greedily broken so no line exceeds the panel width."""
        words = part.split()
        if len(words) < 2 or width_of(part) <= max_width_inches:
            return part
        lines, current = [], words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if width_of(candidate) <= max_width_inches:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines)

    head, opener, tail = text.partition("(")
    head = head.rstrip()
    if head and tail:
        return f"{wrapped(head)}\n{wrapped(f'{opener}{tail}')}"
    return wrapped(text)


def _colorbar_extend(dem, levels):
    """Which ends of the bar need an "and beyond" arrow.

    ``_ELEVATION_CLIP_QUANTILES`` deliberately cuts the top of the DEM so one
    summit pixel cannot flatten the rest of the basin to a single colour. Those
    cells still get drawn, in the darkest class — so the bar has to SAY that its
    top number is a threshold and not the basin's maximum. An arrow is the
    cartographic convention for exactly that, and it costs one glyph.
    """
    values = np.asarray(dem.values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "neither"
    below = float(values.min()) < float(levels[0])
    above = float(values.max()) > float(levels[-1])
    if below and above:
        return "both"
    return "max" if above else ("min" if below else "neither")


#: Lightness (CIE L*) above which the raster under the overlays counts as PALE.
#: Below it the divides flip to a light line with a dark halo. 55 sits near the
#: middle of the L* range and is where a mid-grey line stops reading either way.
_DARK_RASTER_LIGHTNESS = 55.0


def _relative_luminance(rgb):
    """CIE L* of an sRGB triple, 0 (black) to 100 (white)."""
    channels = np.asarray(rgb, dtype=float)[..., :3]
    linear = np.where(
        channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4
    )
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return float(
        np.where(luminance > 0.008856, 116 * np.cbrt(luminance) - 16, 903.3 * luminance)
    )


def _overlay_contrast(raster, style):
    """``(line_colour, halo_colour)`` for boundaries drawn over ``raster``.

    A dark line over a dark fill has no contrast, so the halo ends up doing all
    the work and the divide renders as two white rails with an invisible core —
    observed on the precipitation map, where 90% of the basin sits in the
    darkest class. Which way round to draw depends on what the reader actually
    sees, so it is measured: the AREA-WEIGHTED mean lightness of the classes
    actually painted, not the palette's midpoint. A palette that is pale overall
    can still paint a basin dark if the data crowds its top class.
    """
    try:
        levels = _class_levels(raster, style)
        values, weights = _finite_cells(raster)
        shares = _class_area_shares(values, levels, weights)
        cmap = _classified_colormap(style, levels, "neither")
        lightness = np.array(
            [_relative_luminance(cmap(i)) for i in range(len(levels) - 1)]
        )
        mean_lightness = float((lightness * shares).sum() / max(shares.sum(), 1e-9))
    except (ValueError, KeyError, IndexError):
        # Any failure here is cosmetic; fall back to the pale-ground styling
        # rather than let a contrast heuristic stop a figure rendering.
        return COLOR_SUBCATCHMENT, COLOR_HALO
    if mean_lightness >= _DARK_RASTER_LIGHTNESS:
        return COLOR_SUBCATCHMENT, COLOR_HALO
    return COLOR_HALO, COLOR_SUBCATCHMENT


def _extent_frame(extent):
    """The map extent as a one-row GeoDataFrame, for a map with no basin layer.

    Every vector layer is optional — a raster on its own is a legitimate map,
    and the source-grid climate figures are drawn before any model exists. The
    furniture still needs a subject: something for the locator inset to mark and
    for the corner budget to measure against. The extent box is the honest
    stand-in, since it is exactly what the figure shows.
    """
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in extent)
    return gpd.GeoDataFrame(
        geometry=[shapely_box(lon_min, lat_min, lon_max, lat_max)],
        crs=f"EPSG:{REQUIRED_CRS_EPSG}",
    )


def _basin_clip_path(basin):
    """The dissolved basin as a matplotlib ``Path``, for clipping the raster.

    Needed only when the DEM is resampled. The basin edge is carried by the
    RGBA image's ALPHA channel, and any interpolation resamples alpha with
    everything else — so a bilinear DEM fades out ACROSS the basin boundary and
    visibly spills past the outline that is supposed to bound it. Clipping to
    the real polygon puts the edge back where the geometry says it is, and lets
    the smoothing act only on the inside.
    """
    vertices, codes = [], []
    for geometry in basin.geometry:
        for polygon in getattr(geometry, "geoms", [geometry]):
            for ring in (polygon.exterior, *polygon.interiors):
                coordinates = np.asarray(ring.coords)
                if len(coordinates) < 3:
                    continue
                vertices.append(coordinates)
                codes.append(
                    [MplPath.MOVETO]
                    + [MplPath.LINETO] * (len(coordinates) - 2)
                    + [MplPath.CLOSEPOLY]
                )
    if not vertices:
        return None
    return MplPath(np.concatenate(vertices), np.concatenate(codes))


def _draw_raster(
    fig,
    ax,
    raster,
    style,
    centre_latitude,
    colorbar_box=None,
    basin=None,
    label_width_inches=None,
):
    """Shaded relief plus its colourbar in the side panel.

    ``colorbar_box`` is the bar's [x0, y0, w, h] in axes fractions, normally
    from ``_panel_layout`` so the bar lands in what the locator and the legend
    leave. ``basin`` clips the raster, and matters only when
    ``DEM_INTERPOLATION`` resamples it.
    """
    levels = _class_levels(raster, style)
    extend = _colorbar_extend(raster, levels)
    # BoundaryNorm wants one colour per class, PLUS one per extended end — the
    # arrow is a colour, not a decoration, so the ramp has to carry it.
    cmap = _classified_colormap(style, levels, extend)
    norm = colors.BoundaryNorm(levels, cmap.N, extend=extend)
    x_dim, y_dim = spatial_dim_names(raster)
    field = (
        _shaded_relief(raster, cmap, norm, centre_latitude) if style.relief else raster
    )
    image = field.plot.imshow(
        ax=ax,
        x=x_dim,
        y=y_dim,
        transform=ccrs.PlateCarree(),
        zorder=Z_RELIEF,
        add_labels=False,
        interpolation=style.interpolation,
        **({} if style.relief else {"cmap": cmap, "norm": norm, "add_colorbar": False}),
    )
    if style.interpolation != "none" and basin is not None:
        clip = _basin_clip_path(basin)
        if clip is not None:
            image.set_clip_path(clip, transform=ax.transData)
    # imshow of an RGBA array carries no mappable, so the colourbar needs an
    # explicit one — the ramp is the same object either way.
    # The bar's title is wrapped to the panel's width. Most quantity names with
    # their units are longer than a 40 mm panel, and the title is left-aligned
    # to that panel — so unwrapped it runs off the figure.
    label = style.label
    if label_width_inches:
        label = _wrap_label(fig, label, label_width_inches, FONT_SIZE_COLORBAR_LABEL)
    if colorbar_box is None:
        colorbar_box = _colorbar_inset(label.count("\n") + 1)
    colorbar_axes = ax.inset_axes(colorbar_box)
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
    # ``spacing="uniform"`` gives every CLASS the same length of bar. With
    # equal-interval breaks that is what "proportional" already did; with
    # equal-area breaks it is the difference between a readable bar and one
    # whose top class — the widest in metres, by construction — eats two thirds
    # of it. The bar shows classes, so the classes get equal room.
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axes,
        ticks=levels,
        spacing="uniform",
    )
    if _colorbar_label_position() == "top":
        # A TITLE, not the label: ``set_label`` on a vertical bar always lands
        # alongside it, rotated. ``loc="left"`` is what keeps a label wider than
        # the 0.025-wide bar from hanging off both of its sides — it starts at
        # the bar's left edge, which is the legend's anchor too.
        colorbar_axes.set_title(
            label,
            fontsize=FONT_SIZE_COLORBAR_LABEL,
            pad=_COLORBAR_TITLE_PAD,
            loc="left",
        )
    else:
        colorbar.set_label(label, fontsize=FONT_SIZE_COLORBAR_LABEL)
    colorbar.outline.set_linewidth(_COLORBAR_OUTLINE_WIDTH)
    colorbar_axes.tick_params(
        labelsize=FONT_SIZE_COLORBAR_TICK, length=TICK_LENGTH, pad=TICK_PAD
    )


def category_entries(raster, style):
    """The classes ``raster`` actually contains, as ``[(codes, colour, label)]``.

    Two jobs, both about honesty rather than looks.

    * Only classes PRESENT are returned. A global land-cover legend declares 23
      classes and a single basin carries a handful; listing the other seventeen
      in the legend would describe ground the map does not show.
    * Codes present but NOT declared are collected into one grey entry and
      warned about. Silently dropping them renders real ground transparent,
      which reads exactly like nodata — the one failure this figure must not
      have. ``codes`` is a TUPLE for that reason: the catch-all entry stands for
      several codes at once.
    """
    values = np.asarray(raster.values, dtype="float64")
    present = {int(value) for value in np.unique(values[np.isfinite(values)])}
    declared = tuple(style.categories or ())
    entries = [
        ((int(code),), colour, label)
        for code, colour, label in declared
        if int(code) in present
    ]
    unlisted = tuple(sorted(present.difference(int(code) for code, _, _ in declared)))
    if unlisted:
        warnings.warn(
            f"categorical raster carries codes the style does not declare: "
            f"{list(unlisted)}; drawn as {LABEL_UNCLASSIFIED.lower()}",
            RuntimeWarning,
            stacklevel=2,
        )
        listed = ", ".join(str(code) for code in unlisted)
        entries.append(
            (unlisted, COLOR_UNCLASSIFIED, f"{LABEL_UNCLASSIFIED} ({listed})")
        )
    return entries


def _category_handles(entries):
    """Legend swatches for a categorical raster's classes, in declared order.

    Patches rather than a colourbar, because a bar asserts an ORDER and these
    classes have none: "urban" does not sit between "cropland" and "water".

    A class whose label is ``None`` is DRAWN but not listed. That is how a table
    with more classes than a 40 mm panel can carry asks for the fill without the
    key — the alternative, a legend running off the top of the page, explains
    nothing a reader wanted.
    """
    return [
        mpatches.Patch(
            facecolor=colour,
            edgecolor=COLOR_BASIN_OUTLINE,
            linewidth=_CATEGORY_SWATCH_EDGE_WIDTH,
            label=label,
        )
        for _, colour, label in entries
        if label is not None
    ]


def _draw_categorical_raster(ax, raster, entries):
    """Paint a nominal raster from its class table — no ramp, no colourbar.

    The codes are remapped onto contiguous indices before drawing. Handing the
    raw codes to a ``BoundaryNorm`` instead would space the classes by their
    NUMERIC gaps, so land cover 30/40/50/80/90/112/114/116/126 would be drawn as
    if 90 and 112 were nearly the same class and 50 and 80 far apart. Codes are
    labels; the arithmetic on them is meaningless.

    Cells matching no listed code stay NaN and render transparent, which is
    correct: after ``category_entries`` the only cells left over are nodata.
    """
    values = np.asarray(raster.values, dtype="float64")
    indexed = np.full(values.shape, np.nan)
    for index, (codes, _, _) in enumerate(entries):
        for code in codes:
            indexed[values == code] = index
    field = xr.DataArray(indexed, coords=raster.coords, dims=raster.dims)
    cmap = colors.ListedColormap([colour for _, colour, _ in entries])
    norm = colors.BoundaryNorm(np.arange(len(entries) + 1) - 0.5, cmap.N)
    x_dim, y_dim = spatial_dim_names(raster)
    field.plot.imshow(
        ax=ax,
        x=x_dim,
        y=y_dim,
        transform=ccrs.PlateCarree(),
        zorder=Z_RELIEF,
        add_labels=False,
        # Never resampled. A nominal raster interpolated between classes invents
        # codes that do not exist, and the invented ones land on whatever class
        # sits between them in the index table.
        interpolation="none",
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    )


def _categorical_overlay_contrast(raster, entries):
    """``_overlay_contrast``'s answer for a nominal raster.

    Same question — is the ground under the divides dark or pale? — measured the
    same way, over the class colours actually painted rather than over a ramp
    the categorical path never builds.
    """
    values = np.asarray(raster.values, dtype="float64")
    weights = np.array([float(np.isin(values, codes).sum()) for codes, _, _ in entries])
    total = weights.sum()
    if total <= 0:
        return COLOR_SUBCATCHMENT, COLOR_HALO
    lightness = np.array(
        [_relative_luminance(colors.to_rgba(colour)) for _, colour, _ in entries]
    )
    if float((lightness * weights).sum() / total) >= _DARK_RASTER_LIGHTNESS:
        return COLOR_SUBCATCHMENT, COLOR_HALO
    return COLOR_HALO, COLOR_SUBCATCHMENT


def _point_handle(facecolor, label, marker):
    """A legend entry for a point layer, built by hand.

    Not ``label=`` on the ``plot`` call. geopandas registers the labelled
    ``PathCollection``, and matplotlib's ``HandlerPathCollection`` draws a
    collection's paths stretched across ``handlelength`` — so "outlets" came out
    as a filled black BLOCK rather than as the square marker it labels. Passing
    the label to both the plot and a hand-built handle is the same double-entry
    bug ``_draw_waterbodies`` documents. One handle, built here, is the fix for
    both symptoms.
    """
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        markerfacecolor=facecolor,
        markeredgecolor=COLOR_MARKER_EDGE,
        markeredgewidth=WIDTH_MARKER_EDGE,
        # Line2D takes a marker SIZE in points; geopandas takes points-squared.
        markersize=MARKER_SIZE**0.5,
        label=label,
    )


def _draw_points(ax, layer, facecolor, marker, label_column=None, zorder=Z_MARKER):
    """One point layer, optionally annotated with a column's values.

    Draws only — the legend entry comes from ``_point_handle``, so nothing here
    registers a labelled artist.
    """
    if layer is None or len(layer) == 0:
        return
    layer.plot(
        ax=ax,
        marker=marker,
        markersize=MARKER_SIZE,
        facecolor=facecolor,
        edgecolor=COLOR_MARKER_EDGE,
        linewidth=WIDTH_MARKER_EDGE,
        zorder=zorder,
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


def _present(layer):
    """Whether an optional layer has anything to draw."""
    return layer is not None and len(layer) > 0


def _waterbody_entries(layers):
    """``[(name, layer, style)]`` for each waterbody layer that has features.

    Split out from the drawing so the legend can be BUILT before anything is
    drawn — its size is now measured to place the rest of the panel, so it has
    to exist first.
    """
    return [
        (
            name,
            layer,
            dict(
                facecolor=WATERBODY_COLORS[name][0],
                edgecolor=WATERBODY_COLORS[name][1],
                linewidth=WIDTH_WATERBODY_EDGE,
            ),
        )
        for name, layer in layers.items()
        if _present(layer)
    ]


def _draw_waterbodies(ax, entries):
    """Fill each waterbody layer from ``_waterbody_entries``.

    No ``label`` on the ``plot`` call, ever. geopandas registers a labelled
    collection, so labelling here AND building a patch — which this did until
    2026-08-03 — puts every waterbody in the legend twice. The patch built in
    ``_legend_handles`` is the only labelled artist; geopandas' polygon handle
    does not survive into a legend usably anyway (geopandas/geopandas#660).
    """
    for _, layer, style in entries:
        layer.plot(ax=ax, zorder=Z_WATERBODY, **style)


def _legend_handles(styles, *, rivers, basin, subbasins, outlets, gauges, waterbodies):
    """Every legend entry, in reading order, built without drawing anything.

    Shares its style dicts with the artists themselves, so a legend swatch
    cannot drift from the line it stands for.
    """
    handles = []
    if _present(rivers):
        handles.append(Line2D([], [], label=LABEL_RIVER, **styles["river"]))
    if _present(basin):
        # The outline is the map's key line and had no legend entry at all until
        # 2026-08, which left the heaviest thing on the figure unexplained.
        handles.append(Line2D([], [], label=LABEL_BASIN, **styles["basin"]))
    if _present(subbasins):
        handles.append(Line2D([], [], label=LABEL_SUBCATCHMENT, **styles["divide"]))
    if _present(outlets):
        handles.append(_point_handle(COLOR_OUTLET, LABEL_OUTLET, MARKER_SHAPE_OUTLET))
    if _present(gauges):
        handles.append(_point_handle(COLOR_GAUGE, LABEL_GAUGE, MARKER_SHAPE_GAUGE))
    handles.extend(
        mpatches.Patch(label=name, **style) for name, _, style in waterbodies
    )
    return handles


def _layer_styles():
    """The style dicts the map artists and their legend handles both use."""
    return {
        "river": dict(color=COLOR_RIVER, linewidth=RIVER_WIDTH_MAX),
        "basin": dict(color=COLOR_BASIN_OUTLINE, linewidth=WIDTH_BASIN_OUTLINE),
        "divide": dict(
            color=COLOR_SUBCATCHMENT,
            linewidth=WIDTH_SUBCATCHMENT,
            linestyle=DASH_SUBCATCHMENT,
        ),
    }


def _add_legend(ax, handles):
    """The legend, at its final anchor — the panel's lower-left corner.

    Created BEFORE the map is drawn, because its measured size is what places
    the colourbar and sizes the locator inset. Its own position depends on
    nothing else, so building it first costs nothing.
    """
    legend = ax.legend(
        handles=handles,
        title=_LEGEND_TITLE,
        # Anchored by its LOWER left to the panel's floor: the legend grows
        # upward toward the colourbar, so a long one cannot run off the bottom
        # of the figure.
        loc="lower left",
        bbox_to_anchor=(_PANEL_LEFT, 0.0),
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
    # The panel's room is reserved by the layout engine's ``rect``, so letting
    # the engine also see the legend costs the map size.
    legend.set_in_layout(False)
    return legend


def _align_caveat_to_panel(fig, footnote, panel_items):
    """Right-align the footnote to the SIDE PANEL's right edge.

    Centred under the whole figure — matplotlib's default for ``supxlabel`` —
    the line sits under the map-plus-panel midpoint, which is nothing the reader
    can see: slightly right of the map's centre, slightly left of the panel's,
    aligned to neither. The panel's right edge IS a visible edge, shared by the
    locator inset and the legend, so a footnote flushed to it reads as belonging
    to the sheet rather than as having drifted.

    The edge is MEASURED off the drawn panel items, not computed from
    ``_PANEL_LEFT`` plus the panel's available width. That was the first
    version and it overshot: with no legend to measure, the available width is
    the whole strip ``_LAYOUT_RIGHT`` leaves, which is wider than anything
    actually drawn in it — so the line ran past the locator's edge, which is the
    one edge the reader can see. What is drawn is the only thing worth aligning
    to.

    Only ``x`` and the alignment are set. Constrained layout repositions a
    ``supxlabel`` on every draw, but it rewrites the Y ONLY and carries the x
    through, so this survives the saves that follow.

    Leaves the default centring when nothing measurable is in the panel (no
    locator, no legend) or when the backend cannot produce a renderer.
    """
    if footnote is None:
        return
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return
    edges = [
        item.get_window_extent(renderer).x1 for item in panel_items if item is not None
    ]
    if not edges:
        return
    right_edge = fig.transFigure.inverted().transform((max(edges), 0.0))[0]
    footnote.set_x(float(right_edge))
    footnote.set_horizontalalignment("right")


def plot_raster_map(
    raster,
    rivers=None,
    basin=None,
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
    style=None,
    title=None,
    caveat=None,
    expected_units=None,
    vector_legend=True,
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
        bounding box plus proportional padding. Set it to frame several basins
        alike, or to crop.
    gauge_label_column : str or None
        Column annotated beside each gauge; ``None`` draws the markers unlabelled.
    river_order_column : str or None
        Numeric column scaling the river widths; ``None`` or an absent column
        draws every reach at ``RIVER_WIDTH_UNIFORM``.
    style : RasterStyle, optional
        Palette, label, classification and relief for the quantity being drawn.
        Defaults to ``RASTER_STYLES["elevation"]``. This is the one argument
        that makes the figure a template: everything else on it — furniture,
        panel stack, legend, graticule, locator — is the same for any raster.
    title, caveat : str, optional
        Figure title and footnote, drawn through ``suptitle``/``supxlabel`` so
        constrained layout reserves room for them.
    expected_units : sequence of str, optional
        Units the label claims. A mismatch warns; see ``check_geographic_inputs``.
    vector_legend : bool
        Whether the OVERLAY layers — rivers, boundary, divides, points — get
        legend entries. The layers are drawn either way; this governs only
        whether they are explained.

        ``False`` is for a SET of figures over one basin, where the overlay is
        identical on every sheet and one figure in the set already carries the
        key. Repeating four entries on thirteen maps costs panel height that a
        land-cover legend needs and teaches the reader nothing after the first
        sheet. A raster's own classes are never suppressed by this — they are
        the figure's subject, not its furniture.

    Returns
    -------
    (matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxes)
        Nothing has been saved. The figure is sized in millimetres
        (``FIGURE_WIDTH_MM``), so ``savefig`` without ``bbox_inches="tight"``
        preserves the declared width.
    """
    # Checked, not assumed: the panel is PlateCarree and two of the furniture
    # items convert degrees to metres, so a projected layer renders a wrong map
    # rather than failing. See ``check_geographic_inputs``.
    style = RASTER_STYLES["elevation"] if style is None else style
    # Resolved HERE, unconditionally, rather than left to the caller: whether a
    # figure diverges depends on the raster in front of it, so a caller can only
    # get it right by asking the same question this function is about to ask.
    # Idempotent, so a caller that already resolved (climate_figures does) loses
    # nothing, and a caller that forgot cannot ship a mis-centred ramp.
    style = resolve_diverging_style(raster, style)
    check_geographic_inputs(
        raster,
        {
            "rivers": rivers,
            "basin": basin,
            "subbasins": subbasins,
            "gauges": gauges,
            "outlets": outlets,
            "lakes": lakes,
            "reservoirs": reservoirs,
            "glaciers": glaciers,
        },
        style.label,
        expected_units,
    )
    if extent is None:
        extent = map_extent(raster)
    else:
        # An extent the CALLER chose can be far smaller than the raster — the
        # source-grid climate framed on the basin is. Classify what is shown.
        raster = _raster_within(raster, extent)
    proj = ccrs.PlateCarree()
    centre_latitude = 0.5 * float(extent[2] + extent[3])

    with rc_context(_publication_rc()):
        fig = plt.figure(figsize=_figure_size(extent), layout="constrained")
        fig.get_layout_engine().set(rect=(0.0, 0.0, _LAYOUT_RIGHT, 1.0))
        ax = fig.add_subplot(projection=proj)
        ax.set_extent(extent, crs=proj)

        # --- the legend, first ------------------------------------------------
        # Built and placed before anything is drawn, because its MEASURED size
        # is what the rest of the panel is laid out against: its height sizes
        # the colourbar, and its width sizes the locator inset so the panel's
        # top and bottom blocks come out the same. Its own anchor depends on
        # nothing, so there is no circularity — only this ordering.
        # Clipped BEFORE the legend is built, so an entry cannot describe a
        # layer the map does not show.
        if CLIP_RIVERS_TO_BASIN:
            rivers = _clip_to_basin(rivers, basin)
        styles = _layer_styles()
        waterbodies = _waterbody_entries(
            {"lakes": lakes, "reservoirs": reservoirs, "glaciers": glaciers}
        )
        # A nominal raster puts its classes in the LEGEND, not on a colourbar,
        # so they have to be resolved here — before the legend is built, because
        # the legend's measured size is what the rest of the panel is laid out
        # against. Nine land-cover classes make a legend twice the height of the
        # vector-only one, and the layout has to see that.
        categories = category_entries(raster, style) if style.categories else []
        handles = list(_category_handles(categories))
        if vector_legend:
            handles += _legend_handles(
                styles,
                rivers=rivers,
                basin=basin,
                subbasins=subbasins,
                outlets=outlets,
                gauges=gauges,
                waterbodies=waterbodies,
            )
        # No handles at all — a continuous raster with the overlay key
        # suppressed — means NO legend, not an empty one. matplotlib draws the
        # frame and the title for a legend with nothing in it, so the panel
        # would carry a blank box under the colourbar.
        legend = _add_legend(ax, handles) if handles else None

        measured = _measure_legend(fig, legend, extent) if legend else (0.0, 0.0)
        if measured is None:
            measured = (0.0, _legend_height_fraction(extent, len(handles)))
        # The title wraps to the PANEL ITEM width — what the legend and the
        # locator inset span — not to the panel's full available width. That is
        # the edge the reader sees the three blocks share, so a title running
        # past it is what reads as overflowing even while it is still on canvas.
        panel_width_axes = measured[0] or _panel_available_width()
        panel_width_in = panel_width_axes * _map_width_inches()
        # Wrapped BEFORE the layout is computed: how many lines the colourbar's
        # title takes is what the bar is positioned against, so measuring it
        # afterwards would place the bar against a title that no longer fits.
        wrapped_label = _wrap_label(
            fig, style.label, panel_width_in, FONT_SIZE_COLORBAR_LABEL
        )
        label_lines = wrapped_label.count("\n") + 1
        layout = _panel_layout(extent, label_lines, measured)

        # --- the raster ------------------------------------------------------
        # Two encodings, one figure. A nominal raster took its whole legend
        # above and needs no colourbar; everything else takes the classified
        # ramp and the side panel's bar.
        if categories:
            _draw_categorical_raster(ax, raster, categories)
        else:
            _draw_raster(
                fig,
                ax,
                raster,
                style,
                centre_latitude,
                layout["colorbar"],
                basin if _present(basin) else None,
                label_width_inches=panel_width_in,
            )

        # --- hydrography ------------------------------------------------------
        if _present(rivers):
            rivers.plot(
                ax=ax,
                linewidth=_river_linewidths(rivers, river_order_column),
                color=COLOR_RIVER,
                zorder=Z_RIVER,
            )
        # Subcatchment divides first and lighter, then the outline over them, so
        # the two are never confusable at the same weight.
        if _present(subbasins):
            divide_color, divide_halo = (
                _categorical_overlay_contrast(raster, categories)
                if categories
                else _overlay_contrast(raster, style)
            )
            styles["divide"]["color"] = divide_color
            _divide_linework(subbasins).plot(
                ax=ax,
                zorder=Z_SUBCATCHMENT,
                path_effects=[
                    pe.withStroke(
                        linewidth=HALO_WIDTH_SUBCATCHMENT, foreground=divide_halo
                    )
                ],
                **styles["divide"],
            )
        if _present(basin):
            basin.boundary.plot(ax=ax, zorder=Z_BASIN_OUTLINE, **styles["basin"])

        _draw_points(ax, outlets, COLOR_OUTLET, MARKER_SHAPE_OUTLET, zorder=Z_OUTLET)
        _draw_points(ax, gauges, COLOR_GAUGE, MARKER_SHAPE_GAUGE, gauge_label_column)

        # --- waterbodies ------------------------------------------------------
        _draw_waterbodies(ax, waterbodies)

        # --- cartographic furniture -------------------------------------------
        # The legend sits in the side panel, so it no longer competes for a map
        # corner. The scale bar is placed against the basin's ACTUAL footprint,
        # so it does not land on a basin that reaches into a bottom corner.
        # Every vector layer is optional — the raster alone is a map. Without a
        # basin the locator falls back to the raster's own extent, and the
        # corner budget to the extent box, so the furniture still places.
        subject = basin if _present(basin) else _extent_frame(extent)
        footprint = subject.union_all()
        # Corners are budgeted in one place, in priority order: the arrow's is
        # fixed, then the locator's, then the bar's. Each of the three may be
        # pinned to a named corner or left on "auto", in which case it takes the
        # emptiest corner none of its predecessors claimed. Pinned by default
        # now — furniture that moves between two basins of one study is harder
        # to compare than furniture that occasionally sits on a river.
        locator_corner = _locator_corner(footprint, extent)
        bar_corner = (
            _SCALE_BAR_CORNER
            if _SCALE_BAR_CORNER != "auto"
            else _scale_bar_corner(
                footprint, extent, {_NORTH_ARROW_CORNER, locator_corner}
            )
        )
        _add_graticule(ax, extent)
        _add_scale_bar(ax, extent, bar_corner)
        _add_north_arrow(ax)
        # Sized to the legend's measured width, so the panel's top and bottom
        # blocks share a right edge as well as a left one.
        # Kept, not discarded: it is the panel's widest item, so it is what the
        # source footnote is right-aligned to.
        locator_axes = _add_locator_inset(
            ax, extent, subject, locator_corner, layout["locator"]
        )
        ax.set_title("")
        # Title and footnote go through the FIGURE-level artists that
        # constrained layout knows about. The climate figures previously drew
        # their caveat with fig.text + fig.tight_layout(), and tight_layout
        # DISABLES constrained layout — which is what the whole side-panel
        # stack is built on, so it would have arrived as a broken panel
        # rather than as a misplaced caption.
        if title:
            fig.suptitle(title, fontsize=FONT_SIZE_TITLE)
        footnote = None
        if caveat:
            footnote = fig.supxlabel(
                caveat, fontsize=FONT_SIZE_CAVEAT, color=COLOR_CAVEAT, wrap=True
            )

        # Constrained layout is ITERATIVE, and one pass is not enough here: the
        # first draw leaves the y tick labels at x0 = -7.7 px — off the canvas,
        # so "0.45°N" prints as "45°N" — and the second settles them at +4.2.
        # The workflow path never saw this because it saves twice (PDF, then
        # PNG) and the second save inherits a settled layout. A caller doing one
        # savefig would not, so the figure is settled BEFORE it is handed back.
        for _ in range(_LAYOUT_SETTLE_PASSES):
            fig.draw_without_rendering()

        # AFTER the passes, because it measures the panel items where they
        # FINALLY landed — constrained layout is still moving them until here.
        _align_caveat_to_panel(fig, footnote, (locator_axes, legend))

    return fig, ax
