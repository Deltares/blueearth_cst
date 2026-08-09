# -*- coding: utf-8 -*-
"""Render a basin map from files on disk. Edit the values below, then run:

    pixi run python dev/scripts/basin_map_example.py

Every layer is read from its own file — the DEM from a netCDF, the vectors from
GeoJSON — so any of them can be swapped for a file that never came from wflow.
"""

import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# F401: `cartographic_map` reads as unused because every line that touches it
# is COMMENTED OUT by design -- the tunable block below is a menu of
# `cartographic_map.FONT_SIZE_BASE = ...` overrides you uncomment. Deleting the
# import would break the file the moment it is used as intended.
#
# The tunables live in `cartographic_map`, not in `plot_map`: the cartography
# moved out in 2026-08. Setting a name on `plot_map` would set a copy and
# change nothing.
from blueearth_cst.shared import cartographic_map  # noqa: E402,F401
from blueearth_cst.shared.plot_map import plot_basin_map  # noqa: E402


def primary_checkout():
    """The main working tree, which is where test_case/ lives.

    A git worktree does not have test_case/, so resolving the default model
    against the current directory alone leaves it dead exactly where figure
    work happens. Only matters for the default below.
    """
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return REPO_ROOT
    return (REPO_ROOT / common).resolve().parent


# --- input files -----------------------------------------------------------
# Point MODEL_DIR at any folder holding staticmaps.nc and staticgeoms/.
# Set any of the optional vector paths to None to leave that layer off the map.

MODEL_DIR = primary_checkout() / "test_case" / "basin_map_fixture" / "hydrology_model"
GEOMS_DIR = MODEL_DIR / "staticgeoms"

STATICMAPS_PATH = MODEL_DIR / "staticmaps.nc"
DEM_VARIABLE = "land_elevation"

RIVERS_PATH = GEOMS_DIR / "rivers.geojson"
BASINS_PATH = GEOMS_DIR / "basins.geojson"  # one polygon per subcatchment
SUBBASINS_PATH = GEOMS_DIR / "subbasins.geojson"
GAUGES_PATH = GEOMS_DIR / "gauges_locations.geojson"
OUTLETS_PATH = GEOMS_DIR / "outlets.geojson"
LAKES_PATH = None
RESERVOIRS_PATH = None
GLACIERS_PATH = None

OUT_PATH = REPO_ROOT / ".tmp" / "basin_map.png"
DPI = 300

# --- plotting parameters ---------------------------------------------------
# EVERY tunable constant in plot_map.py, with its default. Uncomment one and
# change it. The comment under each is its own comment from plot_map.py,
# shortened -- read that file for the reasoning behind a value.
#
# Lengths are PHYSICAL (mm / inches / points), not pixels: the figure is built
# at its final printed size, so a font size is the size it will be on the page.
# Positions inside the map are axes fractions (0 = left/bottom, 1 = right/top)
# and may exceed 1 to sit outside it -- that is how the side panel works.

# -- page and export ----------------------------------------------------------
#   Figure width in MILLIMETRES — converted once, here, and never re-guessed
#   downstream. 180 mm is the double-column width that Elsevier (190), AGU
#   (190) and Copernicus (170) all accept without downscaling. Set this to
#   your target journal's column width; every other size is chosen to work at
#   it.
# cartographic_map.FIGURE_WIDTH_MM = 180.0
#   Raster preview resolution. The PDF is the deliverable; the PNG is what the
#   workflow's other consumers (baseline fingerprint, quick review) read. 300
#   is the usual journal minimum for raster figures.
# cartographic_map.PREVIEW_DPI = 400

# -- typography ---------------------------------------------------------------
#   Type sizes in POINTS at the printed width above. Applied through
#   rc_context so the process-wide rcParams the other plotting rules inherit
#   are left untouched. Raise every value together to scale the labelling;
#   raise one to re-balance it. Fallback for anything not named below — which,
#   as the figure currently stands, is NOTHING: every text element carries its
#   own size, the title is empty and the axes have no labels. Changing it
#   alone therefore renders identical bytes. It matters only once a text
#   element is added without a size of its own. Verified 2026-08-03 with
#   dev/scripts/preview_basin_map.py.
# cartographic_map.FONT_SIZE_BASE = 8.0
#   coordinate tick labels
# cartographic_map.FONT_SIZE_TICK = 7.0
#   legend entries and its title
# cartographic_map.FONT_SIZE_LEGEND = 7.0
#   "elevation [m a.s.l.]"
# cartographic_map.FONT_SIZE_COLORBAR_LABEL = 8.0
#   the wflow_id beside each gauge marker
# cartographic_map.FONT_SIZE_GAUGE_LABEL = 5.5
#   the 0 / 2.5 / 5 km numbers
# cartographic_map.FONT_SIZE_SCALE_BAR = 6.0
#   the "N"
# cartographic_map.FONT_SIZE_NORTH_ARROW = 7.5
#   Font family. None keeps matplotlib's default (DejaVu Sans, which embeds
#   cleanly in the PDF). Set e.g. "Arial" or ["Helvetica", "Arial"] to match a
#   manuscript — but check the exported PDF, because a missing family falls
#   back SILENTLY.
# cartographic_map.FONT_FAMILY = None

# -- layout -------------------------------------------------------------------
#   Vertical room (inches) constrained layout needs for the x tick labels and
#   the axes furniture, on top of the map panel itself. Measured need is ~0.16
#   in of tick labels; the rest is margin. Over-allowing here shows up
#   directly as dead space above and below an aspect-locked map.
# cartographic_map._FURNITURE_HEIGHT_IN = 0.32
#   Horizontal room (inches) the y tick labels take on the left. Raise it if a
#   basin's coordinates need more decimal places than the default formatting.
# cartographic_map._TICK_LABEL_WIDTH_IN = 0.5
#   Constrained layout owns the figure only up to here; the strip to the right
#   is a SIDE PANEL holding the colourbar and, beneath it, the legend. A
#   GeoAxes has a LOCKED aspect, so it does not fill its layout cell
#   vertically — and fig.colorbar(ax=ax) sizes to the CELL, which is what made
#   the bar overhang the map top and bottom. Both panel items are therefore
#   anchored in AXES coordinates (so they track the map exactly) and this rect
#   reserves the room. LOWER it to widen the panel (a longer legend entry
#   needs more), RAISE it to give the map more width.
# cartographic_map._LAYOUT_RIGHT = 0.78
#   Keep pathological basin shapes from producing an unusable page. A basin
#   narrower or taller than these renders with whitespace rather than being
#   squashed or running off the figure.
# cartographic_map._MIN_MAP_ASPECT = 0.45
# cartographic_map._MAX_MAP_ASPECT = 1.45
#   Draw passes run before the figure is returned, so its layout is already
#   settled. Constrained layout converges in two here (measured); a third
#   costs ~0.2 s and changes nothing. Raise it only if a label still lands
#   off-canvas.
# cartographic_map._LAYOUT_SETTLE_PASSES = 2

# -- side panel: colourbar and legend -----------------------------------------
#   Left edge of the side panel, in axes fractions (>1 = outside the map). The
#   colourbar and the legend BOTH start here — ONE value, so they cannot drift
#   out of alignment. Raise it to push the panel further from the map.
# cartographic_map._PANEL_LEFT = 1.03
#   Colourbar geometry in axes fractions: (bottom, width, height). Height 0.5
#   spans the upper half of the map's height; the legend occupies the rest.
# cartographic_map._COLORBAR_BOTTOM = 0.5
# cartographic_map._COLORBAR_WIDTH = 0.025
# cartographic_map._COLORBAR_HEIGHT = 0.5
# cartographic_map._COLORBAR_OUTLINE_WIDTH = 0.5
#   Where the colourbar's label goes. "right" is matplotlib's own placement
#   for a vertical bar: alongside it, rotated 90°. "top" puts it above the
#   bar, HORIZONTAL and left-aligned to the bar's left edge — which is
#   _PANEL_LEFT, the legend's anchor too, so the two line up. Prefer "top" for
#   a long label: rotated text is slower to read and a unit string in brackets
#   reads badly on its side.
# cartographic_map.COLORBAR_LABEL_POSITION = 'top'
#   Gap between the bar and a "top" label, in points.
# cartographic_map._COLORBAR_TITLE_PAD = 5.0
#   Height given up to a "top" label, in axes fractions PER LINE of it. The
#   bar's BOTTOM is pinned (_COLORBAR_BOTTOM), so this shortens it from above
#   — without it the bar reaches 1.0 and the label renders off the canvas. Per
#   line, because the label wraps: a two-line label needs twice the room, and
#   a fixed value would either clip the second line or leave a gap above a
#   one-line one.
# cartographic_map._COLORBAR_TOP_LABEL_HEADROOM = 0.055
#   Upper and lower quantiles of the DEM the ramp spans. The upper clip stops
#   a single high pixel flattening the rest of the basin to one colour.
# cartographic_map._ELEVATION_CLIP_QUANTILES = (0.0, 0.98)
#   Target number of colour CLASSES. The ramp is stepped rather than
#   continuous: a reader cannot resolve a shade back to a number off a smooth
#   ramp, but can off a class, and stepped classes survive the greyscale print
#   that a continuous ramp turns to mush. The count is a target — the class
#   WIDTH is rounded to a 1/2/5 value first, so the boundaries are numbers
#   worth printing and the count lands near this rather than on it.
# cartographic_map._COLORBAR_LEVELS = 7
#   Start the ramp at 0 m rather than at the basin's own lowest cell.
#   Elevation is measured from a datum, so a bar starting at 4 m invites the
#   reader to treat the basin floor as the zero of the scale. Set False to
#   always spend the whole ramp on the basin's actual range.
# cartographic_map._ELEVATION_STARTS_AT_ZERO = True
#   ...but not when it would cost the map its resolution. A 1900-1960 m
#   plateau zeroed gets classes 0/500/1000/1500/2000 — the ENTIRE basin lands
#   in one of them and the map renders as a single flat colour. So the
#   baseline drops to zero only while the basin's own range stays at least
#   this fraction of the zero-based range; below it, the ramp starts at the
#   basin's floor instead. CST runs on lowland deltas and Himalayan headwaters
#   from the same code, and a rule tuned on one of them is not a rule.
# cartographic_map._ZERO_BASELINE_MIN_SPAN_FRACTION = 0.35
#   Top of the legend box, in axes fractions — just below the colourbar's
#   lower end (_COLORBAR_BOTTOM). Lower it to open a gap between the two.
# cartographic_map._LEGEND_TOP = 0.44
#   1.0 = opaque, 0.0 = no fill
# cartographic_map._LEGEND_FRAME_ALPHA = 0.85
#   border weight, points
# cartographic_map._LEGEND_FRAME_WIDTH = 0.5
#   padding inside the frame, in font units
# cartographic_map._LEGEND_BORDER_PAD = 0.4
#   length of the sample line/marker, in font units
# cartographic_map._LEGEND_HANDLE_LENGTH = 1.4
#   set to None to drop the title row
# cartographic_map._LEGEND_TITLE = 'Legend'

# -- colours ------------------------------------------------------------------
#   One place for every hue on the figure. The blue is used for BOTH the
#   rivers and the user's gauges, which is deliberate: it ties a gauge to the
#   network it sits on and separates it from the model's own outlets, which
#   stay black.
# cartographic_map.COLOR_RIVER = '#2c6fad'
# cartographic_map.COLOR_GAUGE = '#2c6fad'
# cartographic_map.COLOR_OUTLET = 'k'
# cartographic_map.COLOR_BASIN_OUTLINE = 'k'
# cartographic_map.COLOR_SUBCATCHMENT = '0.45'
# cartographic_map.COLOR_GRATICULE = '0.4'
# cartographic_map.COLOR_MARKER_EDGE = 'white'
#   Halo drawn behind furniture text so it stays legible over any terrain.
# cartographic_map.COLOR_HALO = 'white'
#   Waterbody fills, as (facecolor, edgecolor). Keyed by the staticgeoms
#   layer.
# cartographic_map.WATERBODY_COLORS = {'lakes': ('#a8d0e6', '#3d5a6c'), 'reservoirs': ('#2c6fad', '#173d5e'), 'glaciers': ('#d9d9d9', '#8c8c8c')}
#   A monotonic-lightness elevation ramp, hand-built rather than imported: the
#   perceptually-uniform terrain colormaps (cmcrameri, cmocean) are not in the
#   pixi env and adding a dependency for one figure is not warranted.
#   Lightness falls monotonically from low to high ground, so the ramp
#   survives greyscale printing AND every dichromacy — the two failure modes
#   `terrain` had. Replace it only with another ramp whose lightness is
#   monotonic; a test enforces that.
# cartographic_map._DEM_ANCHORS = ('#f6f2ea', '#e3d5ba', '#c9aa7d', '#a07f52', '#6f5533', '#46351f')

# -- line weights (points) ----------------------------------------------------
#   River width scales with Strahler stream order, between these two bounds.
#   The minimum is what a headwater gets, the maximum the trunk — widen the
#   gap for a more dramatic network, narrow it for a flatter, more uniform
#   one. 0.2 pt is below what most printers hold and vanishes on screen at any
#   reasonable zoom, so the headwaters of the network simply were not there.
# cartographic_map.RIVER_WIDTH_MIN = 0.5
# cartographic_map.RIVER_WIDTH_MAX = 1.4
#   Used when every river shares one stream order, so there is nothing to
#   scale.
# cartographic_map.RIVER_WIDTH_UNIFORM = 0.6
#   the dissolved outer boundary — the map's key line
# cartographic_map.WIDTH_BASIN_OUTLINE = 0.9
#   internal divides, deliberately much lighter
# cartographic_map.WIDTH_SUBCATCHMENT = 0.35
# cartographic_map.WIDTH_WATERBODY_EDGE = 0.5
# cartographic_map.WIDTH_MARKER_EDGE = 0.4
# cartographic_map.WIDTH_AXES_SPINE = 0.6
# cartographic_map.WIDTH_GRATICULE = 0.3
#   Dash pattern for the subcatchment divides, matplotlib (offset, (on, off)).
# cartographic_map.DASH_SUBCATCHMENT = (0, (4, 2))
#   Halo stroke widths, points. The halo must exceed the line it protects.
# cartographic_map.HALO_WIDTH_TEXT = 2.5
# cartographic_map.HALO_WIDTH_GAUGE_LABEL = 1.8

# -- markers ------------------------------------------------------------------
#   Separate shapes for the two point layers. They were both thin diamonds,
#   separated by colour alone — which fails in greyscale, fails for a
#   dichromat, and is hard to tell apart at 5 pt anyway. Shape is the
#   redundant channel that fixes all three. Circle reads as a measurement
#   point, square as a structural one; swap them if a convention says
#   otherwise.
# cartographic_map.MARKER_SHAPE_GAUGE = 'o'
# cartographic_map.MARKER_SHAPE_OUTLET = 's'
#   matplotlib points-squared, as geopandas expects. 18 was ~4.2 pt across at
#   180 mm, which disappears against the relief; 44 is ~6.6 pt.
# cartographic_map.MARKER_SIZE = 44
#   Offset of a gauge's label from its marker, in points (x, y). Must clear
#   the marker's RADIUS: at MARKER_SIZE 44 that is ~3.3 pt, and the old (2.5,
#   2.5) put the text inside the symbol.
# cartographic_map.GAUGE_LABEL_OFFSET = (4.5, 3.5)

# -- graticule ----------------------------------------------------------------
# cartographic_map.GRATICULE_ALPHA = 0.5
# cartographic_map.GRATICULE_LINESTYLE = ':'
#   Upper bound on tick count per axis; the locator picks round values under
#   it.
# cartographic_map.GRATICULE_MAX_TICKS = 6
#   points
# cartographic_map.TICK_LENGTH = 2.5
#   gap between tick and label, points
# cartographic_map.TICK_PAD = 2.0

# -- scale bar ----------------------------------------------------------------
#   Alternating filled/open segments, the conventional cartographic scale bar.
#   Must be EVEN for the midpoint label to land on a segment boundary.
# cartographic_map._SCALE_BAR_SEGMENTS = 4
#   Bar height as a fraction of the map's latitude span.
# cartographic_map._SCALE_BAR_HEIGHT = 0.011
#   Target bar length as a fraction of the map width, before rounding to a
#   1/2/5 value. Raise it for a longer, more precisely readable bar.
# cartographic_map._SCALE_BAR_WIDTH_FRACTION = 0.25
#   Inset of the bar from its chosen corner, as a fraction of the map extent.
# cartographic_map._SCALE_BAR_INSET = 0.06
#   Gap between the bar and its numbers, as a fraction of the latitude span.
# cartographic_map._SCALE_BAR_LABEL_GAP = 0.008
# cartographic_map._SCALE_BAR_EDGE_WIDTH = 0.5

# -- north arrow --------------------------------------------------------------
#   Arrow position in axes fractions: (x, tip y, tail y). The "N" sits at the
#   tail. Exactly vertical is correct here because PlateCarree's north is up.
#   Tucked into the map's own top-right CORNER — it used to float short of it,
#   which read as an artist adrift over the basin rather than as furniture.
# cartographic_map._NORTH_ARROW_POSITION = (0.975, 0.985, 0.885)
# cartographic_map._NORTH_ARROW_STYLE = '-|>'
# cartographic_map._NORTH_ARROW_WIDTH = 0.8
#   The arrow's corner, kept clear of the scale bar. With the legend in the
#   side panel, this is the only reserved corner left on the map.
# cartographic_map._NORTH_ARROW_CORNER = 'upper right'

# -- locator inset ------------------------------------------------------------
#   Draw it at all. The layers come from a vendored Natural Earth extract
#   (config/basemap/); with that file absent the inset is skipped and a note
#   is printed, so a copy of this module taken to another project still
#   renders.
# cartographic_map.LOCATOR_ENABLED = True
#   Half-width of the locator's window, in degrees, around the basin's centre.
#   8 deg is roughly national-to-regional: big enough to reach a coast or a
#   capital in most of the world, small enough that the basin mark is not a
#   speck. Raise it for a continental frame, lower it for a provincial one.
# cartographic_map._LOCATOR_SPAN_DEG = 8.0
#   The inset's width as a fraction of the map panel's width, and its inset
#   from the corner. Its HEIGHT is derived so the box comes out square on the
#   page — a square window drawn into a non-square box would otherwise
#   letterbox.
# cartographic_map._LOCATOR_WIDTH = 0.22
# cartographic_map._LOCATOR_MARGIN = 0.025
#   Which corner it sits in, or "auto" for the emptiest one the north arrow is
#   not using. Auto by default and it matters more here than for the scale
#   bar: the inset is OPAQUE, so a fixed corner does not merely crowd the
#   basin, it hides part of it — observed covering a gauge on the first
#   render. Ties go to an upper corner, where a locator is conventionally
#   read.
# cartographic_map._LOCATOR_CORNER = 'auto'
#   sea; the palest thing on the figure
# cartographic_map.COLOR_LOCATOR_OCEAN = '#eef2f5'
#   land, a shade darker so the coast reads
# cartographic_map.COLOR_LOCATOR_LAND = '#dcdcd8'
#   the land polygons' own edge IS the coastline
# cartographic_map.COLOR_LOCATOR_COAST = '0.55'
#   country lines, lighter still
# cartographic_map.COLOR_LOCATOR_BORDER = '0.7'
# cartographic_map.COLOR_LOCATOR_CITY = '0.35'
#   The "you are here" mark. The one warm accent on the figure, and the only
#   place red appears — it has to win against grey without competing with the
#   elevation ramp, which owns every brown.
# cartographic_map.COLOR_LOCATOR_BASIN = '#c0392b'
# cartographic_map.WIDTH_LOCATOR_COAST = 0.35
# cartographic_map.WIDTH_LOCATOR_BORDER = 0.3
# cartographic_map.WIDTH_LOCATOR_FRAME = 0.6
#   Cities are filtered by Natural Earth's own prominence rank (0 = most
#   prominent), then the largest few by population are kept. Both limits
#   matter: the rank keeps towns out, the count keeps a dense region from
#   filling up.
# cartographic_map._LOCATOR_CITY_MAX_SCALERANK = 3
# cartographic_map._LOCATOR_MAX_CITIES = 5
# cartographic_map._LOCATOR_CITY_MARKER_SIZE = 5
# cartographic_map.FONT_SIZE_LOCATOR_CITY = 4.5
#   Basin mark size in points-squared, and the label offset for a city name.
# cartographic_map._LOCATOR_BASIN_MARKER_SIZE = 26
# cartographic_map._LOCATOR_CITY_LABEL_OFFSET = (2.5, -1.0)

# -- furniture placement ------------------------------------------------------
#   Lower-left corner of each candidate furniture box, as a fraction of the
#   map extent. Names are matplotlib legend(loc=...) values verbatim.
# cartographic_map._CORNER_BOX = 0.3
# cartographic_map._CORNERS = {'lower left': (0.0, 0.0), 'lower right': (0.7, 0.0), 'upper left': (0.0, 0.7), 'upper right': (0.7, 0.7)}

# -- hillshade ----------------------------------------------------------------
#   Illumination: light from the north-west at 45 deg, the convention readers'
#   relief perception is calibrated to (lit NW = ridge, shaded SE = valley;
#   reverse it and terrain visually inverts).
# cartographic_map._AZIMUTH_DEG = 315.0
# cartographic_map._ALTITUDE_DEG = 45.0
#   Target 90th-percentile terrain slope AFTER exaggeration, ~19 deg — steep
#   enough to read as relief, shallow enough not to fabricate mountains. The
#   exaggeration factor is derived per basin: CST runs on lowland deltas and
#   on Himalayan headwaters from the same code, and any FIXED factor renders
#   one of them featureless (a flat basin at exag 3) or blown out (an alpine
#   basin at exag 200). Raise for more dramatic relief, lower for a flatter,
#   calmer map.
# cartographic_map._TARGET_SLOPE = 0.35
# cartographic_map._MAX_VERT_EXAG = 500.0
#   How the ramp and the shading combine. "soft" keeps colour; "overlay" is
#   higher contrast; "hsv" is the most dramatic and the least faithful.
# cartographic_map._SHADE_BLEND_MODE = 'soft'

# -- data / labels ------------------------------------------------------------
#   Padding around the model's own bounding box, in degrees, so the basin does
#   not touch the frame.
# cartographic_map._EXTENT_BUFFER_DEG = 0.02

# -- drawing order ------------------------------------------------------------
#   Drawing order. Every artist names one of these rather than a bare number,
#   so the stack is legible and reorderable in one place.
# cartographic_map.Z_RELIEF = 1
# cartographic_map.Z_RIVER = 3
# cartographic_map.Z_WATERBODY = 4
# cartographic_map.Z_SUBCATCHMENT = 5
# cartographic_map.Z_BASIN_OUTLINE = 6
# cartographic_map.Z_MARKER = 7
# cartographic_map.Z_FURNITURE = 8

# --- read the files --------------------------------------------------------


def read_vector(path):
    return gpd.read_file(path) if path is not None else None


# load() before the file closes -- the render touches the values repeatedly.
with xr.open_dataset(STATICMAPS_PATH) as dataset:
    dem = dataset[DEM_VARIABLE].load()

rivers = read_vector(RIVERS_PATH)
# The map wants ONE outer boundary at its heaviest weight; basins.geojson holds
# one polygon per subcatchment, so dissolve them. The divides come in
# separately, lighter and dashed.
basin = read_vector(BASINS_PATH).dissolve()

# --- render ----------------------------------------------------------------

fig, ax = plot_basin_map(
    dem,
    rivers,
    basin,
    subbasins=read_vector(SUBBASINS_PATH),
    gauges=read_vector(GAUGES_PATH),
    outlets=read_vector(OUTLETS_PATH),
    lakes=read_vector(LAKES_PATH),
    reservoirs=read_vector(RESERVOIRS_PATH),
    glaciers=read_vector(GLACIERS_PATH),
    extent=None,  # [lon_min, lon_max, lat_min, lat_max]
    gauge_label_column="wflow_id",
    river_order_column="strord",
    elevation_label="elevation\n[m a.s.l.]",  # \n wraps it; the bar makes room
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=DPI)
print(f"wrote {OUT_PATH}")
