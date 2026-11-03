# -*- coding: utf-8 -*-
"""The thematic map family drawn from ``data/spatial/spatial_maps.nc``.

``elevation_basin`` shows one layer of the spatial foundation — elevation. This
module draws the rest of the set: the delineation, land cover, leaf area and the
soil profile, each through the SAME cartographic template, so a basin report is
one visual family rather than ten unrelated pictures.

:func:`plot_spatial_figure_set` draws BOTH — it is rule 1.11's entry point, and
the two halves are one deliverable rather than two that happen to share a
folder. See its docstring.

Three things live here and deliberately not in the drawing code
(``plotting/prepared_maps.py``) or the classifier (``shared/map_classes.py``):

* the **layer registry** (:data:`SPATIAL_MAP_FIGURES`) — which variables get a
  figure, in what order, under what filename;
* the **styles**, because they name data sources. The template's docstring says
  nothing in it knows what wflow is; by the same rule nothing in it should know
  what SoilGrids or Copernicus are, so this family declares every style it
  draws, elevation included;
* the **class tables** for the nominal layers, taken from the source product's
  published legend where one exists.

Two things every figure here does NOT have, and both are deliberate:

* **no title.** The colourbar label or the class legend already names the
  quantity, and a title over the map costs panel height on every sheet. The
  FILENAME is what names the figure, which is why the stems read as the
  quantity (``soil_ph_basin``) rather than as a position in a set. The one
  thing the title did carry and a filename cannot — the product credit — moved
  to the caption; see ``map_frames.SOURCE_CREDITS``.
* **no overlay key.** Rivers, the basin outline, the divides and the points of
  interest are still DRAWN — that is what ties ten rasters to one basin — but
  they get no legend entries. The styling reads without one (black outline,
  dashed divides, blue rivers stepped by order, labelled points), and a key on
  every sheet would spend the room the land-cover legend needs.

What is NOT drawn, and why, is as much a decision as what is:

* ``elevation`` — it is ``elevation_basin.png``, which lands in this same folder.
  Drawing it twice under two names would put two different-looking figures of
  one quantity in front of a reader.
* ``slope``, ``upstream_area``, ``river_order`` — drawn in the first draft and
  cut on review (owner's call, 2026-08-10): terrain derivatives a first-order
  basin assessment does not read, where ``elevation_basin`` already carries the
  relief and the river layer already carries the network.
* ``flow_accumulation`` — proportional to ``upstream_area`` (the cells are
  equal-area to within 0.002% here), so it is the same map in different units.
* ``river_mask`` — the river vector layer already draws it, on every figure.
* ``flow_direction`` — D8 codes are CYCLIC, not ordinal and not really nominal
  either; a useful QA raster, not a report figure.
* anything CONSTANT over the basin — ``basin_id`` on a single-parent project,
  ``cell_area``, and ``soil_soilthickness`` where the source is flat. Skipped at
  RENDER time with a printed reason rather than by an exclusion list, because
  which layers are degenerate depends on the basin, not on the code.

Soil is drawn at the TOPSOIL slice (``sl1``, 0-5 cm) only. SoilGrids ships seven
depth slices of six properties; forty-two near-identical maps is not a
deliverable, and the topsoil is the slice that governs infiltration and the
land-surface exchange this toolbox is built around. A deeper slice is one
registry entry away.

**No ``from __future__ import annotations`` here**, and none may be added:
Snakemake PREPENDS a preamble to a ``script:`` module, so the future import is
no longer at the top of the file and the job dies with SyntaxError before
running a line. It costs nothing on the pinned Python — PEP 585 and 604
annotations are native — and ``tests/test_model_reference.py`` enforces it.
"""

import os
from pathlib import Path

from blueearth_cst.plotting.figure_captions import write_figure_captions

# NOTHING HEAVY AT MODULE SCOPE, deliberately. `build_model.smk` imports this
# module at PARSE time, for `figure_paths` alone -- a pure string function --
# and every dry-run of WF1 then paid for whatever this block pulled in. It used
# to pull cartopy, geopandas, xarray and matplotlib (t2608202307 measured the
# cost; t2608210029 removed it). The rule is: a name used only inside a function
# is imported inside that function.
#
# The two module-scope imports below are what the DECLARATIONS in this file
# need, and both are cheap:
#   * `raster_style` is pure Python -- it exists so the five module-level
#     `RasterStyle` constants below cost nothing to declare;
#   * `plot_style` is the page/typography contract, stdlib-only, and owns
#     `RASTER_DPI` directly.
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.shared.plot_style import RASTER_DPI
from blueearth_cst.shared.raster_style import RasterStyle
from blueearth_cst.shared.snake_utils import save_figure

# ---------------------------------------------------------------------------
# WHERE THE INPUTS ARE
# ---------------------------------------------------------------------------
#: Project-relative home of the shared spatial products (rules 1.02 / 1.05).
SPATIAL_DIRNAME = "data/spatial"
#: The thematic raster stack this family draws.
SPATIAL_MAPS_FILENAME = "spatial_maps.nc"
#: Vector layers drawn over every figure, mapped to the template's arguments.
#: The same four ``elevation_basin`` uses, for the same reason: the overlay is what
#: makes ten rasters read as ten views of ONE basin. Drawn, but not explained —
#: see the module docstring on the missing key.
SPATIAL_MAP_LAYERS = {
    "basins": "basins",
    "subbasins": "subbasins",
    "rivers": "rivers",
    "gauges": "locations",
}
#: The layer whose positive values define "inside the basin". The thematic
#: rasters are clipped to the basin's BOUNDING BOX plus a buffer, not to the
#: basin, so without this a land-cover map paints a full rectangle and the basin
#: outline reads as a box drawn on top of a bigger dataset.
BASIN_MASK_VARIABLE = "subbasin_id"

#: Where the figures are written, relative to the spatial directory. The same
#: folder ``elevation_basin`` uses — this family is the rest of that figure's set.
PLOTS_DIRNAME = "plots"


# ---------------------------------------------------------------------------
# CLASS TABLES FOR THE NOMINAL LAYERS
# ---------------------------------------------------------------------------

#: The Copernicus Global Land Cover discrete-classification legend, verbatim.
#:
#: Codes AND colours are the product's own (Buchhorn et al. 2020; the table
#: published with CGLS-LC100, reproduced in the Earth Engine catalog entry).
#: They are not a palette choice and must not be "improved": a reader who knows
#: the product recognises this map at a glance, which no ramp of ours can buy,
#: and the same basin drawn here and in any other CGLS map then matches.
#:
#: All 23 classes are declared even though a single basin carries a handful —
#: ``category_entries`` drops the absent ones from the legend, and an undeclared
#: code would be drawn grey and warned about. Declaring the whole legend is what
#: keeps that warning meaning "the source changed", not "this basin is elsewhere".
#:
#: Labels are shortened from the product's sentence-long definitions, and the
#: twelve forest classes WRAP onto two lines. The side panel is about 40 mm
#: wide and the legend does not participate in constrained layout, so a label
#: longer than that does not widen the panel — it runs off the page, which is
#: what "Closed forest, evergreen broadleaf" did on the first render. The codes
#: are unchanged, so the mapping back to the product stays checkable.
LAND_COVER_CLASSES = (
    (0, "#282828", "No data"),
    (20, "#ffbb22", "Shrubs"),
    (30, "#ffff4c", "Herbaceous vegetation"),
    (40, "#f096ff", "Cropland"),
    (50, "#fa0000", "Urban / built-up"),
    (60, "#b4b4b4", "Bare / sparse vegetation"),
    (70, "#f0f0f0", "Snow and ice"),
    (80, "#0032c8", "Permanent water"),
    (90, "#0096a0", "Herbaceous wetland"),
    (100, "#fae6a0", "Moss and lichen"),
    (111, "#58481f", "Closed forest,\nevergreen needleleaf"),
    (112, "#009900", "Closed forest,\nevergreen broadleaf"),
    (113, "#70663e", "Closed forest,\ndeciduous needleleaf"),
    (114, "#00cc00", "Closed forest,\ndeciduous broadleaf"),
    (115, "#4e751f", "Closed forest, mixed"),
    (116, "#007800", "Closed forest, other"),
    (121, "#666000", "Open forest,\nevergreen needleleaf"),
    (122, "#8db400", "Open forest,\nevergreen broadleaf"),
    (123, "#8d7400", "Open forest,\ndeciduous needleleaf"),
    (124, "#a0dc00", "Open forest,\ndeciduous broadleaf"),
    (125, "#929900", "Open forest, mixed"),
    (126, "#648c00", "Open forest, other"),
    (200, "#000080", "Open sea"),
)

#: The GlobCover 2009 (ESA / UCLouvain) legend: codes and OFFICIAL colours,
#: labels shortened from the product's definitions and wrapped like the table
#: above. GlobCover reuses code numbers Copernicus gives other meanings — 40 is
#: broadleaved evergreen forest here and cropland there — so a GlobCover raster
#: drawn with the Copernicus table mislabels most of a tropical basin.
GLOBCOVER_CLASSES = (
    (11, "#aaf0f0", "Irrigated cropland"),
    (14, "#ffff64", "Rainfed cropland"),
    (20, "#dcf064", "Mosaic cropland /\nvegetation"),
    (30, "#cdcd66", "Mosaic vegetation /\ncropland"),
    (40, "#006400", "Broadleaved evergreen\nforest"),
    (50, "#00a000", "Closed broadleaved\ndeciduous forest"),
    (60, "#aac800", "Open broadleaved\ndeciduous forest"),
    (70, "#003c00", "Closed needleleaved\nevergreen forest"),
    (90, "#286400", "Open needleleaved\nforest"),
    (100, "#788200", "Mixed broadleaved /\nneedleleaved forest"),
    (110, "#8ca000", "Mosaic forest-shrubland /\ngrassland"),
    (120, "#be9600", "Mosaic grassland /\nforest-shrubland"),
    (130, "#966400", "Shrubland"),
    (140, "#ffb432", "Herbaceous vegetation"),
    (150, "#ffebaf", "Sparse vegetation"),
    (160, "#00785a", "Flooded forest,\nfresh water"),
    (170, "#009678", "Flooded forest,\nsaline water"),
    (180, "#00dc82", "Flooded grassland /\nwoody vegetation"),
    (190, "#c31400", "Artificial surfaces"),
    (200, "#fff5d7", "Bare areas"),
    (210, "#0046c8", "Water bodies"),
    (220, "#ffffff", "Permanent snow and ice"),
    (230, "#000000", "No data"),
)

#: Class table per land-cover product, keyed by the catalog source-name prefix
#: (``globcover``, ``globcover_2009_v2.3``; ``vito``, ``vito_2019_v3.0.1``).
_LAND_COVER_TABLES = (("globcover", GLOBCOVER_CLASSES), ("vito", LAND_COVER_CLASSES))


def land_cover_classes(layer):
    """The class table of the product the layer came from, read from ``source``.

    The table must follow the data: products share code numbers with different
    meanings. An unrecognised source gets NO table, with a warning, so its cells
    draw as unclassified rather than under another product's labels.
    """
    import warnings

    source = str(layer.attrs.get("source", "")).lower()
    for prefix, table in _LAND_COVER_TABLES:
        if source.startswith(prefix):
            return table
    warnings.warn(
        f"Land-cover map drawn without class labels: no classification table is registered for {source!r}",
        RuntimeWarning,
        stacklevel=2,
    )
    # Not ``()``: an empty table reads as "not nominal" and would draw the codes
    # on a continuous ramp. One never-present code keeps the layer nominal, and
    # ``category_entries`` sends every real code to its unclassified entry.
    return ((-1, "#000000", None),)


#: Classes covering less of the basin than this are drawn but left out of the
#: key, and named in the caption: a handful of cells (six snow-and-ice cells in
#: a desert delta) is product noise a reader cannot find on the map.
CLASS_KEY_MIN_SHARE = 0.001

#: Legend titles for the nominal figures, by figure stem.
LEGEND_TITLES = {"land_cover_basin": "Land cover", "subbasins": "Subbasins"}
FIGURE_TITLES = {
    "subbasins": "Basin and subbasin delineation",
    "land_cover_basin": "Basin land cover",
    "lai_clim_basin": "Basin leaf area index climatology",
    "soil_clay_basin": "Basin topsoil clay content",
    "soil_silt_basin": "Basin topsoil silt content",
    "soil_sand_basin": "Basin topsoil sand content",
    "soil_organic_carbon_basin": "Basin topsoil organic carbon",
    "soil_ph_basin": "Basin topsoil pH",
    "soil_bulk_density_basin": "Basin topsoil bulk density",
    "soil_depth_to_bedrock_basin": "Basin depth to bedrock",
}

#: Subbasin fills: ColorBrewer Set3 without its grey, a light qualitative set
#: that keeps rivers and divides readable over it, drawn at ``SUBBASIN_ALPHA``.
#: Colours are assigned so touching subbasins never share one
#: (``_neighbour_aware_colours``).
SUBBASIN_COLORS = (
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
)
SUBBASIN_ALPHA = 0.6

#: How many subbasins still get a legend entry each. The side panel is about
#: 40 mm wide and its legend is anchored at the map's floor and grows upward, so
#: a long one runs into the locator inset rather than off the bottom — and on a
#: wide basin, whose map panel is short, that happens sooner than the class
#: count suggests.
#:
#: It applies to the subbasin identifiers and deliberately NOT to the land-cover
#: legend. The difference is what dropping the key costs: subbasin swatches say
#: "unit 104" and the map is readable without them, while land-cover swatches
#: are the only thing that says which green is which. There, a crowded legend is
#: the better of two bad answers.
_MAX_LEGEND_SUBBASINS = 15


def subbasin_classes(raster):
    """A class table for the subbasin identifier raster, built from its codes.

    Derived rather than declared: the identifiers are assigned per project by
    ``spatial.identity``, so no constant could list them.

    Past ``_MAX_LEGEND_SUBBASINS`` the swatches are dropped and only the fill is
    drawn. The colours still separate the units on the map — which is what the
    figure is for — while a forty-row legend beside a 40 mm panel would run off
    the top of the page and explain nothing a reader wanted. An empty table is
    the template's own way of saying "no key", so this needs no extra flag.
    """
    import numpy as np

    values = np.asarray(raster.values, dtype="float64")
    codes = [int(value) for value in np.unique(values[np.isfinite(values)])]
    labelled = len(codes) <= _MAX_LEGEND_SUBBASINS
    if not labelled:
        print(
            f"Subbasin boundaries drawn for all {len(codes)} subbasins; "
            "class legend omitted because it would be too large"
        )
    colour_of = _neighbour_aware_colours(values, codes, SUBBASIN_COLORS)
    return tuple(
        (code, colour_of[code], f"Subbasin {code}" if labelled else None)
        for code in codes
    )


def _neighbour_aware_colours(values, codes, palette):
    """``{code: colour}`` so that no two touching units share a colour.

    Adjacency is read off the raster (cells sharing an edge). Units are coloured
    greedily, most-connected first, each taking the least-used palette colour
    none of its neighbours holds; the fallback (reuse the least-used colour)
    only fires for a unit touching more units than the palette has colours.
    """
    import numpy as np

    neighbours = {code: set() for code in codes}
    for a, b in ((values[:, :-1], values[:, 1:]), (values[:-1, :], values[1:, :])):
        mask = np.isfinite(a) & np.isfinite(b) & (a != b)
        for x, y in set(zip(a[mask].astype(int), b[mask].astype(int))):
            if x in neighbours and y in neighbours:
                neighbours[x].add(y)
                neighbours[y].add(x)
    used = dict.fromkeys(palette, 0)
    colour_of = {}
    for code in sorted(codes, key=lambda c: (-len(neighbours[c]), c)):
        taken = {colour_of[n] for n in neighbours[code] if n in colour_of}
        free = [c for c in palette if c not in taken] or list(palette)
        choice = min(free, key=lambda c: (used[c], palette.index(c)))
        colour_of[code] = choice
        used[choice] += 1
    return colour_of


# ---------------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------------
# Every palette below is a matplotlib built-in: ColorBrewer sequential schemes,
# which are monotonic in CIE lightness and colour-vision-deficiency safe by
# construction. Nothing here needs `cmcrameri` or `cmocean`, which are not in
# the pixi env and would be a new dependency for a figure family.
#
# The rule applied throughout: take the SOURCE PRODUCT's own legend where it has
# one (land cover), otherwise the discipline's convention (green = vegetation,
# brown = organic matter), otherwise a neutral ramp rather than a hue that would
# assert something the data does not say.

#: Vegetation density. Green is the one hue a reader will not misread here.
LEAF_AREA_INDEX_STYLE = RasterStyle(
    label="Mean leaf area index (m$^2$ m$^{-2}$)",
    # Fixed 1-unit classes, 0-6: LAI has a physical scale (bare ground ~0, dense
    # tropical forest ~6), so equal steps read the same on every basin.
    levels=(0, 1, 2, 3, 4, 5, 6),
    palette="YlGn",
    zero_baseline=True,
)

#: The three particle-size fractions SHARE one ramp on purpose. They are the
#: same quantity — mass % of the same fine-earth total, summing to 100 — so a
#: reader compares the three maps directly, and a different hue per fraction
#: would say they are different kinds of thing. Red-orange is the mineral family
#: the USDA texture triangle already puts clay in.
TEXTURE_PALETTE = "OrRd"

#: Organic carbon. Pale straw to dark brown is the SOC convention (ISRIC's own
#: legends, the FAO GSOCmap), and it is the one soil property whose colour
#: meaning is genuinely established.
#:
#: MASS PERCENT, not the source's raw integers. SoilGrids 2017 stores ORCDRC in
#: g/kg and the catalog's ``unit_mult`` of 0.1 converts it on read, which is the
#: unit ``hydromt_wflow.workflows.ptf`` documents its pedotransfer functions as
#: taking ("organic carbon [%]"). Checked rather than inferred from the range —
#: a soil map labelled in the wrong unit is wrong in a way that still looks
#: plausible.
ORGANIC_CARBON_STYLE = RasterStyle(
    label="Organic carbon (mass %)",
    palette="YlOrBr",
    zero_baseline=True,
)

#: Bulk density. No colour convention exists, so this takes a neutral hue rather
#: than borrowing one that would imply wetness or fertility. NOT zero-based: a
#: bulk density of 0 is not a physical baseline a reader measures from.
BULK_DENSITY_STYLE = RasterStyle(
    label="Bulk density (g cm$^{-3}$)",
    palette="Purples",
    zero_baseline=False,
)

#: Soil pH, on whichever of two encodings the basin earns.
#:
#: pH 7 is a PHYSICAL midpoint — acid below, alkaline above — which is what
#: licenses a diverging ramp, and ``RdYlBu`` puts acid in red and alkaline in
#: blue, the near-universal soil convention. But it is only honest for a basin
#: that reaches both sides: one running 4.7-5.5 has no alkaline ground, so an
#: absolute ramp would spend half its colours on values that do not occur and
#: flatten the real variation into two classes.
#:
#: So the midpoint is DECLARED and activated per raster
#: (``resolve_diverging_style``). A basin spanning pH 7 gets the diverging map
#: with white pinned to neutral; a one-sided basin keeps ``YlGnBu``, pale (more
#: acidic) to dark blue-green (less), which runs in the same direction as the
#: convention without claiming a midpoint the data does not contain.
SOIL_PH_STYLE = RasterStyle(
    label="Soil pH (H$_2$O)",
    palette="YlGnBu",
    zero_baseline=False,
    diverging_at=7.0,
    diverging_palette="RdYlBu",
)

#: Depth to bedrock. Neutral, distinct from every other soil ramp in the family
#: so a depth map is not mistaken for a texture map at thumbnail size.
#:
#: Drawn in METRES. SoilGrids stores it in centimetres, which put 4000-16000 on
#: the bar for a basin whose bedrock is 40-160 m down — legible, but not a
#: number anyone working on this basin would write. The conversion is declared
#: on the registry entry beside the label, so the two cannot drift.
SOIL_DEPTH_STYLE = RasterStyle(
    label="Depth to bedrock (m)",
    palette="BuPu",
    zero_baseline=True,
)


#: Elevation, the basin overview. LINEAR, deliberately, and the one style that
#: is: elevation is the quantity a reader does arithmetic on (how far above that
#: gauge, how much fall to the outlet), so it keeps a scale you can subtract on
#: rather than equal-area classes. Owner's call, 2026-08-09. The ramp falls back
#: to the monotonic-lightness elevation anchors; drawn over its own hillshade.
ELEVATION_STYLE = RasterStyle(
    label="Elevation (m a.s.l.)",
    palette=None,
    classification="equal_interval",
    zero_baseline=True,
    relief=True,
)


def _texture_style(fraction):
    """One particle-size fraction's style, on the shared texture ramp."""
    return RasterStyle(
        label=f"{fraction} content (mass %)",
        palette=TEXTURE_PALETTE,
        zero_baseline=True,
    )


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------


class SpatialFigure:
    """One entry of the family: which variable, drawn how, saved as what.

    A plain object for the same reason ``RasterStyle`` is one — this package
    carries no type annotations in its plotting layer.
    """

    def __init__(
        self,
        variable,
        stem,
        style=None,
        classes=None,
        mask_to_basin=True,
        expected_units=("source-native",),
        scale=1.0,
        guaranteed=True,
    ):
        #: Variable name in ``spatial_maps.nc``.
        self.variable = variable
        #: Output filename stem; the figure is written as ``<stem>.{png,pdf}``.
        #:
        #: These figures carry NO TITLE — a title over the map panel is furniture
        #: the colourbar label and the legend already provide, and it costs panel
        #: height on every sheet. The filename is what names the figure, so the
        #: stem has to read as the quantity on its own: ``soil_ph_basin``, not
        #: ``fig_07``. Soil stems carry no depth token: only the surface slice
        #: (``sl1``) is registered, so ``soil_<property>_basin`` is unambiguous.
        #: A deeper slice would need its depth in the stem.
        self.stem = stem
        #: A continuous style, or ``None`` for a nominal layer.
        self.style = style
        #: For a nominal layer: the class table, or a callable taking the raster
        #: and returning one (the subbasin identifiers have no fixed codes).
        self.classes = classes
        #: Clip to the basin. True for the thematic layers, which arrive clipped
        #: to the bounding box; the hydrography layers are already basin-shaped
        #: and masking them again is a no-op.
        self.mask_to_basin = mask_to_basin
        #: What the layer's own ``units`` attribute is expected to say. The
        #: template warns when the raster's declared units and the bar's label
        #: disagree, which is how a map of feet labelled in metres gets caught —
        #: but it defaults to ELEVATION's units, so every non-elevation figure
        #: has to state its own or warn on every render.
        #:
        #: ``"source-native"`` is not a cop-out: it is literally what
        #: ``spatial.products._resample_source`` writes when the source declares
        #: no units, so a source that STARTS declaring them still trips the
        #: warning and the label gets re-checked against the real unit.
        self.expected_units = expected_units
        #: Multiplier applied before drawing, for a layer whose SOURCE unit is
        #: not the unit a reader wants on the bar (depth to bedrock is stored in
        #: centimetres). It sits here, one line from ``style.label``, precisely
        #: because a scale factor and a unit string that disagree produce a
        #: figure that is wrong and looks right. Never use it to rescale for
        #: appearance — that is the classifier's job.
        self.scale = scale
        #: Whether the source variable exists for EVERY shipped catalog source,
        #: and so whether rule 1.11 may declare this figure as an output.
        #:
        #: Most are: ``prepare_land_and_soil_maps`` writes ``land_cover`` and
        #: ``leaf_area_index`` under fixed names, ``subbasin_id`` always exists,
        #: and the six soil ``sl1`` properties keep their names across both
        #: shipped soil sources because the catalogs rename them onto the same
        #: targets. ``soil_BDTICM_M_250m_ll`` is the exception: it is NOT in the
        #: catalog's rename map, so it carries the source's own filename, and
        #: ``soilgrids_2020`` has no equivalent entry at all.
        #:
        #: A declared output that the data cannot produce fails the RULE, not the
        #: figure — the run stops with "missing output files" on a project whose
        #: only sin was choosing the other soil source. So the source-specific
        #: figures are drawn and left undeclared; ``data/spatial/plots/`` is a
        #: directory in the tree inventory, so they are still accounted for.
        self.guaranteed = guaranteed


#: The family, in the order a reader should meet it: terrain, then the
#: delineation, then what is on the ground, then what is under it.
SPATIAL_MAP_FIGURES = (
    SpatialFigure(
        "subbasin_id",
        "subbasins",
        classes=subbasin_classes,
        mask_to_basin=False,
        expected_units=("1",),
    ),
    SpatialFigure("land_cover", "land_cover_basin", classes=land_cover_classes),
    SpatialFigure("leaf_area_index", "lai_clim_basin", LEAF_AREA_INDEX_STYLE),
    SpatialFigure("soil_clyppt_sl1", "soil_clay_basin", _texture_style("Clay")),
    SpatialFigure("soil_sltppt_sl1", "soil_silt_basin", _texture_style("Silt")),
    SpatialFigure("soil_sndppt_sl1", "soil_sand_basin", _texture_style("Sand")),
    SpatialFigure("soil_oc_sl1", "soil_organic_carbon_basin", ORGANIC_CARBON_STYLE),
    SpatialFigure("soil_ph_sl1", "soil_ph_basin", SOIL_PH_STYLE),
    SpatialFigure("soil_bd_sl1", "soil_bulk_density_basin", BULK_DENSITY_STYLE),
    SpatialFigure(
        "soil_BDTICM_M_250m_ll",
        "soil_depth_to_bedrock_basin",
        SOIL_DEPTH_STYLE,
        # cm in the file, metres on the bar. See SpatialFigure.scale.
        scale=0.01,
        # soilgrids v1.0 only; see SpatialFigure.guaranteed.
        guaranteed=False,
    ),
)


def figure_paths(plots_dir, formats=("png",), declared_only=True):
    """The figure files this family writes under ``plots_dir``.

    Rule 1.11's ``output:`` comes from here rather than restating the stems, the
    same way rule 1.12's does from ``climate_figures.figure_names`` — a registry
    edit then reaches the Snakefile without a second place to remember.

    ``declared_only`` keeps the source-specific figures out of the rule's
    promise; see :attr:`SpatialFigure.guaranteed` for why that is not a
    shortcut. It does NOT stop them being drawn.

    PNG ONLY since 2026-08-10 (owner's call, applied first to ``elevation_basin`` and
    then to the whole deliverable). The PDF was the vector, embedded-font
    publication copy and nothing in the toolbox or the platform read it; at
    600 dpi and 180 mm the PNG carries the figure everywhere it is used. It also
    halved this rule's render time, which serialised every figure twice.
    ``formats`` stays a parameter because a caller preparing a manuscript can
    still ask for one.
    """
    return [
        f"{plots_dir}/{figure.stem}.{extension}"
        for figure in SPATIAL_MAP_FIGURES
        if figure.guaranteed or not declared_only
        for extension in formats
    ]


# ---------------------------------------------------------------------------
# READING
# ---------------------------------------------------------------------------


def load_spatial_map_layers(spatial_dir):
    """Open ``spatial_maps.nc`` and the vector layers drawn over it.

    Returns ``(dataset, layers)``. The dataset is loaded into memory and its
    handle closed — every figure reads from it, and holding a netCDF open across
    a dozen renders is what makes a Windows run trip over its own file lock.
    """
    spatial_dir = Path(spatial_dir)
    maps_path = spatial_dir / SPATIAL_MAPS_FILENAME
    geoms_dir = spatial_dir / "geoms"
    if not maps_path.is_file():
        raise FileNotFoundError(f"no {SPATIAL_MAPS_FILENAME} in {spatial_dir}")

    # mask_and_scale=False for the same reason ``read_hydrography_seam`` uses it:
    # every layer carries _FillValue in its ATTRS, and the CF decoder would move
    # it into encoding and recast the identifier rasters to float. The fills are
    # applied explicitly, per layer, by ``_mask_nodata``.
    import xarray as xr

    with xr.open_dataset(maps_path, mask_and_scale=False) as dataset:
        maps = dataset.load()

    layers = {}
    for argument, stem in SPATIAL_MAP_LAYERS.items():
        path = geoms_dir / f"{stem}.geojson"
        if path.is_file():
            import geopandas as gpd

            layers[argument] = gpd.read_file(path)
    missing = [name for name in ("basins", "rivers") if name not in layers]
    if missing:
        raise FileNotFoundError(f"{geoms_dir} is missing {missing}")
    return maps, layers


#: The elevation grid the basin overview draws, in the shared hydrography.
HYDROGRAPHY_FILENAME = "hydrography.nc"
SPATIAL_ELEVATION_VARIABLE = "elevation"


def load_spatial_basin_layers(spatial_dir):
    """The elevation grid from ``hydrography.nc`` and the vector layers.

    Model-free: everything comes from ``data/spatial/``, so the overview can be
    drawn before a model exists. Optional vector layers are read when present.
    """
    import geopandas as gpd
    import xarray as xr

    from blueearth_cst.shared.map_classes import _mask_nodata

    spatial_dir = Path(spatial_dir)
    hydrography = spatial_dir / HYDROGRAPHY_FILENAME
    geoms_dir = spatial_dir / "geoms"
    if not hydrography.is_file():
        raise FileNotFoundError(f"no {HYDROGRAPHY_FILENAME} in {spatial_dir}")
    with xr.open_dataset(hydrography) as dataset:
        if SPATIAL_ELEVATION_VARIABLE not in dataset:
            raise KeyError(
                f"{hydrography} has no {SPATIAL_ELEVATION_VARIABLE!r}; it holds "
                f"{sorted(dataset.data_vars)}"
            )
        elevation = _mask_nodata(dataset[SPATIAL_ELEVATION_VARIABLE].load())
    layers = {}
    for argument, stem in SPATIAL_MAP_LAYERS.items():
        path = geoms_dir / f"{stem}.geojson"
        if path.is_file():
            layers[argument] = gpd.read_file(path)
    missing = [name for name in ("basins", "rivers") if name not in layers]
    if missing:
        raise FileNotFoundError(f"{geoms_dir} is missing {missing}")
    return elevation, layers


def _outer_boundary(basins):
    """The basin's OUTER boundary, dissolved to a single polygon.

    ``plot_map._basin_outline`` is the same two lines and was imported at first.
    It is duplicated rather than imported because that module is a Snakemake
    ``script:`` target: reaching a private name out of it points this family's
    dependency at the wflow-model reader, and any later need to touch it there
    fires the ``code`` rerun trigger on every ``project_dir``.

    Why it exists at all: ``basins`` is one polygon per parent, and drawing them
    all at boundary weight makes an internal divide indistinguishable from the
    outline — the one line on this figure a reader has to be able to trust.
    """
    return basins.dissolve()


def _basin_mask(maps):
    """Cells inside the delineated basin, as a boolean array, or ``None``.

    Built from the subbasin identifiers rather than from the basin polygon: it
    is on the same grid, so there is no rasterisation step that could disagree
    with the raster it masks by half a cell.
    """
    import numpy as np

    from blueearth_cst.shared.map_classes import _mask_nodata

    if BASIN_MASK_VARIABLE not in maps:
        return None
    layer = _mask_nodata(maps[BASIN_MASK_VARIABLE])
    values = np.asarray(layer.values, dtype="float64")
    return np.isfinite(values) & (values > 0)


def prepare_layer(maps, figure, basin_mask=None):
    """The 2-D field one figure draws: fills masked, extra dims reduced, clipped.

    Reduction is a MEAN over anything that is not a spatial dimension, which
    today is the leaf area index's 12 monthly steps. That is stated in the
    figure's title ("annual mean"), because a silently averaged seasonal cycle
    is the kind of thing a reader assumes did not happen.
    """
    from blueearth_cst.shared.map_classes import _mask_nodata

    layer = _mask_nodata(maps[figure.variable]).astype("float64")
    extra = [
        dim
        for dim in layer.dims
        if dim not in ("x", "y", "lat", "lon", "latitude", "longitude")
    ]
    if extra:
        layer = layer.mean(dim=extra, skipna=True)
    if figure.mask_to_basin and basin_mask is not None:
        layer = layer.where(basin_mask)
    if figure.scale != 1.0:
        # Attributes survive the multiply on purpose: ``source`` credits the
        # product in the title, and ``units`` still records what the FILE says,
        # which is what the units check should be comparing against.
        layer = (layer * figure.scale).assign_attrs(layer.attrs)
    return layer


def _is_degenerate(layer):
    """``True`` when the field carries no spatial information worth a figure.

    A constant raster renders as one flat colour with a one-value colourbar,
    which looks like a broken figure rather than like a flat field. Whether a
    layer is constant depends on the BASIN — ``cell_area`` varies with latitude
    and ``soil_soilthickness`` is flat over some regions and not others — so
    this is decided per render and reported, never hard-coded as an exclusion.
    It REPORTS; it no longer decides. Skipping was the first design and it made
    the rule's promise conditional on the data — a project whose bedrock depth
    happens to be uniform would have failed with "missing output files", which
    is a workflow crash over a flat soil property. A flat figure with a note in
    the log is the better of the two, and the note is the part that was actually
    wanted.
    """
    import numpy as np

    values = np.asarray(layer.values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True, "no valid cells"
    spread = float(finite.max() - finite.min())
    scale = max(abs(float(finite.max())), abs(float(finite.min())), 1e-12)
    if spread <= 1e-9 * scale:
        return True, f"constant at {finite[0]:g}"
    return False, ""


#: Mathtext in style labels and its plain-text caption form.
_PLAIN_UNITS = (
    ("$^2$", "²"),
    ("$^{-2}$", "⁻²"),
    ("$^{-3}$", "⁻³"),
    ("$_2$", "₂"),
)


def _caption_lead(title, style):
    """The figure title with the key's unit in plain text, e.g. ``(mass %)``."""
    if style is None or " (" not in style.label:
        return title
    unit = style.label.rpartition(" (")[2].rstrip(")")
    for math, plain in _PLAIN_UNITS:
        unit = unit.replace(math, plain)
    return f"{title} ({unit})"


def _spatial_caption(lead, layer, note=None):
    """The shared one-sentence map caption for a spatial layer."""
    from blueearth_cst.shared.map_frames import map_caption

    return map_caption(lead, layer.attrs.get("source"), [note] if note else [])


def _unit_break(label):
    """The quantity on one line and its parenthesised unit on the next."""
    head, sep, unit = label.rpartition(" (")
    return f"{head}\n({unit}" if sep else label


def _key_for(figure, layer):
    """The template key for one figure: ``MapClasses`` or a classified ``MapScale``.

    Classes come from ``shared/map_classes``: the diverging midpoint resolved
    per raster, values beyond the outer breaks clipped into the end classes
    (no extensions).
    """
    from matplotlib.colors import to_hex

    from blueearth_cst.plotting.prepared_maps import MapClasses, MapScale
    from blueearth_cst.shared import map_classes as classes
    from blueearth_cst.shared.plot_colors import MAP_FEATURE_COLORS

    missing = MAP_FEATURE_COLORS["missing"]
    if figure.classes is not None:
        table = figure.classes(layer) if callable(figure.classes) else figure.classes
        return MapClasses(
            title=LEGEND_TITLES.get(figure.stem, figure.stem),
            codes=[code for code, _, _ in table],
            colors=[color for _, color, _ in table],
            labels=[label or str(code) for code, _, label in table],
            missing_color=missing,
            unclassified_color=MAP_FEATURE_COLORS["unclassified"],
            alpha=SUBBASIN_ALPHA if figure.stem == "subbasins" else 1.0,
            key_min_share=CLASS_KEY_MIN_SHARE,
        )
    style = classes.resolve_diverging_style(layer, figure.style)
    levels = [float(level) for level in classes._class_levels(layer, style)]
    cmap = classes._classified_colormap(style, levels, "neither")
    colors = [to_hex(cmap(i)) for i in range(cmap.N)]
    return MapScale(
        units=_unit_break(style.label),
        boundaries=levels,
        ticks=levels,
        colors=colors,
        under_color=colors[0],
        over_color=colors[-1],
        missing_color=missing,
        status="available",
        extend="neither",
    )


def spatial_locator(extent):
    """The overview locator from the vendored Natural Earth basemap, or ``None``.

    Window and city selection follow the spatial profile. A city is kept only
    when its label fits inside the window, largest first.
    """
    from blueearth_cst.plotting.prepared_maps import PANEL_LOCATOR_WIDTH, LocatorMap
    from blueearth_cst.shared import map_classes as classes
    from blueearth_cst.shared.map_frames import features

    if not classes.BASEMAP_PATH.is_file():
        print(f"note: locator inset skipped, no basemap at {classes.BASEMAP_PATH}")
        return None
    window = classes._locator_window(extent)
    cities = classes._locator_cities(window)
    limit = classes._LOCATOR_MAX_CITIES
    west, east, south, north = window
    # Label width in degrees at the inset's 4.5 pt text on a ~150 mm map frame.
    deg_per_pt = (east - west) / (PANEL_LOCATOR_WIDTH * 150 / 25.4 * 72)
    margin = 0.03 * (east - west)
    places = []
    for _, city in cities.iterrows():
        x, y = city.geometry.x, city.geometry.y
        label_deg = (2 + 0.6 * 4.5 * len(city["name"])) * deg_per_pt
        if (
            west + margin <= x
            and x + label_deg <= east - margin
            and south + margin <= y <= north - margin
        ):
            places.append((x, y, city["name"]))
        if len(places) >= limit:
            break
    return LocatorMap(
        tuple(window),
        land=features(classes._read_basemap("land", window)),
        borders=features(classes._read_basemap("borders", window)),
        places=places,
    )


def _spatial_context(maps, layers, basin_mask):
    """What every map of the set shares: frame, overlays, locator, basin mask."""
    import numpy as np

    from blueearth_cst.shared.map_frames import (
        as_lat_lon,
        basin_frame,
        features,
        overlay_features,
    )

    outline = _outer_boundary(layers["basins"])
    frame = basin_frame(layers["basins"])
    return {
        "outline": outline,
        "frame": frame,
        "overlay": overlay_features(
            {
                "basins": features(outline),
                "subbasins": features(layers.get("subbasins")),
                "rivers": features(layers.get("rivers")),
                "locations": features(layers.get("gauges")),
            }
        ),
        "locator": spatial_locator(frame.extent),
        "inside": (
            None
            if basin_mask is None
            else np.asarray(
                as_lat_lon(maps[BASIN_MASK_VARIABLE].copy(data=basin_mask)).values
            )
        ),
    }


def _export_map(fig, plot_dir, stem, record, annotation, formats=("png",), dpi=None):
    """Save one map, tight-cropped with a 2 mm border, and record its caption.

    The crop matches the climate maps' export (``diagnostic_render``); with an
    embedded caption the crop happens inside ``add_caption`` instead.
    """
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.fundamentals import add_caption
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared.plot_preset import CST_PRESET

    crop = {"bbox_inches": "tight", "pad_inches": 2 / 25.4}
    if annotation != "none":
        if annotation not in ("caption", "title"):
            raise ValueError("annotation must be none, caption, or title")
        with (
            use_preset(CST_PRESET, rcparams=False),
            plt.rc_context({"savefig.dpi": dpi or RASTER_DPI}),
        ):
            add_caption(
                fig,
                record[annotation],
                above=annotation == "title",
                crop=True,
                pad_inches=crop["pad_inches"],
            )
        crop = {}
    written = []
    for extension in formats:
        path = os.path.join(str(plot_dir), f"{stem}.{extension}")
        # PDF takes no dpi and needs its timestamp scrubbed; PNG takes dpi and
        # needs the matplotlib version scrubbed, so identical runs give
        # identical bytes.
        scrub = (
            {"metadata": {"CreationDate": None}}
            if extension == "pdf"
            else {"dpi": dpi or RASTER_DPI, "metadata": {"Software": None}}
        )
        save_figure(path, fig=fig, **scrub, **crop)
        written.append(Path(path))
        write_figure_captions(plot_dir, {f"{stem}.{extension}": record})
    plt.close(fig)
    return written


def plot_elevation_basin(spatial_dir, plot_dir=None, annotation="none"):
    """``elevation_basin.png`` on the map template: hillshaded elevation classes.

    The basin overview: the elevation ramp (equal-interval, zero-based) draped
    over a hillshade of the same grid, keyed by its colourbar, with the locator
    inside the map. The relief is prepared here, the template only draws it.
    """
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from blueearth_cst.plotting.prepared_maps import RasterMap, draw_raster_map
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared import map_classes as classes
    from blueearth_cst.shared.map_frames import as_lat_lon, grid_edges
    from blueearth_cst.shared.plot_preset import CST_PRESET

    spatial_dir = Path(spatial_dir)
    plot_dir = Path(plot_dir) if plot_dir is not None else spatial_dir / PLOTS_DIRNAME
    elevation, layers = load_spatial_basin_layers(spatial_dir)
    maps, _ = load_spatial_map_layers(spatial_dir)
    context = _spatial_context(maps, layers, _basin_mask(maps))
    maps.close()
    figure = SpatialFigure("elevation", "elevation_basin", ELEVATION_STYLE)
    scale = _key_for(figure, elevation)
    cmap = ListedColormap(scale.colors)
    norm = BoundaryNorm(scale.boundaries, cmap.N, clip=True)
    south, north = context["frame"].extent[2:]
    relief = classes._shaded_relief(elevation, cmap, norm, (south + north) / 2)
    raster = RasterMap(
        np.asarray(as_lat_lon(elevation).values, dtype=float),
        *grid_edges(elevation),
        inside=context["inside"],
        cell_edges=False,
        rgba=np.asarray(relief.values, dtype=float),
    )
    with use_preset(CST_PRESET):
        fig = draw_raster_map(
            raster,
            context["frame"],
            scale,
            context["overlay"],
            locator=context["locator"],
            locator_in_map=True,
        )
    record = {
        "title": "Basin elevation and drainage network",
        "caption": _spatial_caption(
            "Basin elevation (m a.s.l.) over hillshaded relief", elevation
        ),
    }
    return _export_map(fig, plot_dir, "elevation_basin", record, annotation)


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------


def plot_spatial_maps(
    spatial_dir,
    plot_dir=None,
    variables=None,
    dpi=None,
    formats=("png",),
    *,
    annotation="none",
):
    """Render the family into ``<spatial_dir>/plots``, returning what it wrote.

    ``variables`` selects a subset by variable name, and also REACHES layers the
    default family leaves out — a deeper soil slice, ``flow_direction`` — as long
    as the registry declares them. It is a filter over the registry, not a way to
    plot an undeclared layer, because an undeclared layer has no style and no
    class table and would be drawn on a default ramp that means nothing.

    Layers that are constant over this basin are skipped with a printed reason.
    A skip is reported, never silent: "the figure is not there" and "the figure
    was not worth drawing" are different facts and a reader cannot tell them
    apart from an empty folder.
    """
    import numpy as np

    from blueearth_cst.plotting.prepared_maps import (
        MapClasses,
        RasterMap,
        draw_raster_map,
        unkeyed_classes,
    )
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared.map_classes import check_geographic_inputs
    from blueearth_cst.shared.map_frames import (
        as_lat_lon,
        grid_edges,
    )
    from blueearth_cst.shared.plot_preset import CST_PRESET

    spatial_dir = Path(spatial_dir)
    plot_dir = Path(plot_dir) if plot_dir is not None else spatial_dir / PLOTS_DIRNAME

    # Checked BEFORE anything is opened: a typo in --variable should not cost a
    # read of the whole raster stack before it is reported.
    unknown = sorted(
        set(variables or ()).difference(f.variable for f in SPATIAL_MAP_FIGURES)
    )
    if unknown:
        raise KeyError(
            f"{unknown} are not in the spatial map registry; declare them in "
            "SPATIAL_MAP_FIGURES with a style before asking for them"
        )
    selected = [
        figure
        for figure in SPATIAL_MAP_FIGURES
        if variables is None or figure.variable in set(variables)
    ]

    maps, layers = load_spatial_map_layers(spatial_dir)
    basin_mask = _basin_mask(maps)
    context = _spatial_context(maps, layers, basin_mask)

    written = []
    for figure in selected:
        if figure.variable not in maps:
            # The ONLY skip. It can reach a figure the rule declared as an
            # output, which would fail the rule -- and that is the right
            # failure: it means the registry and ``prepare_land_and_soil_maps``
            # disagree about what the foundation contains.
            print(
                f"Spatial figure not written: {figure.stem} requires {figure.variable!r}, which is missing from {SPATIAL_MAPS_FILENAME}"
            )
            continue
        layer = prepare_layer(maps, figure, basin_mask)
        degenerate, reason = _is_degenerate(layer)
        if degenerate:
            # Noted, then drawn anyway. See ``_is_degenerate``.
            print(
                f"{figure.stem}: {figure.variable} within the basin: {reason}; rendering the map"
            )

        check_geographic_inputs(
            layer,
            {
                "rivers": layers.get("rivers"),
                "basin": context["outline"],
                "subbasins": layers.get("subbasins"),
            },
            value_label=(
                figure.style.label
                if figure.style is not None
                else LEGEND_TITLES.get(figure.stem, figure.stem)
            ),
            expected_units=figure.expected_units,
        )
        # The template (toolbox-plot-style.md, Spatial map template): no title,
        # the source credited in the caption, the key in the side panel. The
        # subbasin overview carries no key and its locator sits inside the map.
        overview = figure.stem == "subbasins"
        raster = RasterMap(
            np.asarray(as_lat_lon(layer).values, dtype=float),
            *grid_edges(layer),
            inside=context["inside"],
            cell_edges=False,
        )
        key = _key_for(figure, layer)
        with use_preset(CST_PRESET):
            fig = draw_raster_map(
                raster,
                context["frame"],
                key,
                context["overlay"],
                locator=context["locator"] if overview else None,
                show_key=not overview,
                locator_in_map=overview,
            )
        unkeyed = (
            unkeyed_classes(raster, context["frame"], key)
            if isinstance(key, MapClasses) and not overview
            else []
        )
        note = (
            f"Classes under {CLASS_KEY_MIN_SHARE:.1%} of the basin are drawn but not listed ("
            + ", ".join(label.replace("\n", " ") for label in unkeyed)
            + ")"
            if unkeyed
            else None
        )
        record = {
            "title": FIGURE_TITLES[figure.stem],
            "caption": _spatial_caption(
                _caption_lead(FIGURE_TITLES[figure.stem], figure.style), layer, note
            ),
        }
        written += _export_map(
            fig, plot_dir, figure.stem, record, annotation, formats, dpi
        )

    maps.close()
    titles = {path.name: {"title": FIGURE_TITLES[path.stem]} for path in written}
    if (Path(plot_dir) / "elevation_basin.png").is_file():
        titles["elevation_basin.png"] = {
            "title": "Basin elevation and drainage network"
        }
    write_figure_titles(plot_dir, titles)
    return written


def plot_spatial_maps_from_project(project_dir, plot_dir=None, **kwargs):
    """``plot_spatial_maps`` for a project directory rather than a spatial one."""
    return plot_spatial_maps(
        Path(project_dir) / SPATIAL_DIRNAME, plot_dir=plot_dir, **kwargs
    )


def plot_spatial_figure_set(spatial_dir, plot_dir=None, **kwargs):
    """Every figure the spatial foundation supports: ``elevation_basin`` and the family.

    Rule 1.11's entry point. The two are drawn by ONE rule because they are one
    deliverable — the same basin, the same overlay, the same style profile and
    the same folder.

    ``elevation_basin`` is drawn FIRST for the same reason it is listed first
    everywhere else: it is the sheet a reader meets before the thematic ones.
    """
    plot_elevation_basin(
        spatial_dir, plot_dir=plot_dir, annotation=kwargs.get("annotation", "none")
    )
    return plot_spatial_maps(spatial_dir, plot_dir=plot_dir, **kwargs)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            plot_spatial_figure_set(
                spatial_dir=sm.params.spatial_dir,
                annotation=getattr(sm.params, "annotation", "none"),
            )
