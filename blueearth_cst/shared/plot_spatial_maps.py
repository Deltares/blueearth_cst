# -*- coding: utf-8 -*-
"""The thematic map family drawn from ``data/spatial/spatial_maps.nc``.

``basin_area`` shows one layer of the spatial foundation — elevation. This
module draws the rest of the set: the delineation, land cover, leaf area and the
soil profile, each through the SAME cartographic template, so a basin report is
one visual family rather than ten unrelated pictures.

:func:`plot_spatial_figure_set` draws BOTH — it is rule 1.12's entry point, and
the two halves are one deliverable rather than two that happen to share a
folder. See its docstring.

Three things live here and deliberately not in ``shared.cartographic_map``:

* the **layer registry** (:data:`SPATIAL_MAP_FIGURES`) — which variables get a
  figure, in what order, under what filename;
* the **styles**, because they name data sources. The template's docstring says
  nothing in it knows what wflow is; by the same rule nothing in it should know
  what SoilGrids or Copernicus are. ``RASTER_STYLES`` holds the source-neutral
  quantities (elevation, precipitation, temperature, PET); a family tied to
  specific products holds its own;
* the **class tables** for the nominal layers, taken from the source product's
  published legend where one exists.

Two things every figure here does NOT have, and both are deliberate:

* **no title.** The colourbar label or the class legend already names the
  quantity, and a title over the map costs panel height on every sheet. The
  FILENAME is what names the figure, which is why the stems read as the
  quantity (``soil_ph_topsoil``) rather than as a position in a set. The one
  thing the title did carry and a filename cannot — the product credit — moved
  to a footnote under the axes; see :data:`SOURCE_CREDITS`.
* **no overlay key.** Rivers, the basin outline, the divides and the points of
  interest are still DRAWN — that is what ties ten rasters to one basin — but
  they get no legend entries. ``basin_area``, in this same folder, carries the
  key that explains them once; repeating it ten times spends the panel height
  the land-cover legend needs and teaches a reader nothing after the first sheet.

What is NOT drawn, and why, is as much a decision as what is:

* ``elevation`` — it is ``basin_area.png``, which lands in this same folder.
  Drawing it twice under two names would put two different-looking figures of
  one quantity in front of a reader.
* ``slope``, ``upstream_area``, ``river_order`` — drawn in the first draft and
  cut on review (owner's call, 2026-08-10): terrain derivatives a first-order
  basin assessment does not read, where ``basin_area`` already carries the
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

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from blueearth_cst.shared.cartographic_map import (
    RASTER_DPI,
    RasterStyle,
    _mask_nodata,
    plot_raster_map,
)
from blueearth_cst.shared.snake_utils import save_figure

# ---------------------------------------------------------------------------
# WHERE THE INPUTS ARE
# ---------------------------------------------------------------------------
#: Project-relative home of the shared spatial products (rules 1.03 / 1.06).
SPATIAL_DIRNAME = "data/spatial"
#: The thematic raster stack this family draws.
SPATIAL_MAPS_FILENAME = "spatial_maps.nc"
#: Vector layers drawn over every figure, mapped to the template's arguments.
#: The same four ``basin_area`` uses, for the same reason: the overlay is what
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
#: folder ``basin_area`` uses — this family is the rest of that figure's set.
PLOTS_DIRNAME = "plots"

#: Product credits, keyed by the catalog entry name each layer records in its
#: own ``source`` attribute. The attr is OUR plumbing ("vito"); a figure credits
#: the PRODUCT, with the reference the catalog already carries
#: (``metadata.paper_ref`` / ``source_version``), so the footnote and the
#: catalog cannot say different things about the same layer.
#:
#: This is the one piece of provenance the figures still show. It used to ride
#: on the title; dropping titles took it with them, and a filename cannot carry
#: it, so it moved to a footnote under the axes — smaller, out of the way, and
#: still on the sheet when the sheet is the only thing a reader has.
SOURCE_CREDITS = {
    "merit_hydro_ihu": "MERIT Hydro IHU (Eilander et al., 2020)",
    "vito": "Copernicus Global Land Cover v2.0.2 (Buchhorn et al., 2020)",
    "modis_lai": "MODIS MCD15A3H V006 (Myneni et al., 2015)",
    "soilgrids": "SoilGrids (ISRIC, 2017)",
}

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

#: Okabe-Ito, the qualitative set designed to stay separable under all three
#: dichromacies (Okabe & Ito 2008). Used for NOMINAL identifiers — subbasins —
#: where the numbering carries no order and a sequential ramp would invent one.
#: Black is left out: it is the basin outline's colour on every figure here.
_QUALITATIVE_COLORS = (
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
)

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
    values = np.asarray(raster.values, dtype="float64")
    codes = [int(value) for value in np.unique(values[np.isfinite(values)])]
    labelled = len(codes) <= _MAX_LEGEND_SUBBASINS
    if not labelled:
        print(
            f"note: {len(codes)} subbasins is more than a legend can carry; "
            "drawing the delineation without a class key"
        )
    return tuple(
        (
            code,
            _QUALITATIVE_COLORS[index % len(_QUALITATIVE_COLORS)],
            f"Subbasin {code}" if labelled else None,
        )
        for index, code in enumerate(codes)
    )


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
    label="Leaf area index (m$^2$ m$^{-2}$)",
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

#: Soil pH. Sequential, running pale (more acidic) to dark blue-green (less), so
#: the direction matches the universal acid-red / alkaline-blue semantics
#: without pretending to a pinned midpoint.
#:
#: A DIVERGING ramp centred on pH 7 is the textbook encoding and is deliberately
#: not used: ``RasterStyle.diverging_center`` is carried but not yet consumed by
#: the classifier, so the centre would land wherever the data's own range put
#: it. Centring a diverging ramp on the data instead of on the physical midpoint
#: is exactly the failure the temperature style documents. Revisit when the
#: classifier honours the centre.
SOIL_PH_STYLE = RasterStyle(
    label="Soil pH (H$_2$O)",
    palette="YlGnBu",
    zero_baseline=False,
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
        #: stem has to read as the quantity on its own: ``soil_ph_topsoil``, not
        #: ``fig_07``. It is also the only place the depth slice is recorded,
        #: which is why every soil stem carries ``_topsoil``.
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
        #: and so whether rule 1.12 may declare this figure as an output.
        #:
        #: Most are: ``prepare_spatial_maps`` writes ``land_cover`` and
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
        "subbasin_delineation",
        classes=subbasin_classes,
        mask_to_basin=False,
        expected_units=("1",),
    ),
    SpatialFigure("land_cover", "land_cover", classes=LAND_COVER_CLASSES),
    SpatialFigure(
        "leaf_area_index", "leaf_area_index_annual_mean", LEAF_AREA_INDEX_STYLE
    ),
    SpatialFigure("soil_clyppt_sl1", "soil_clay_topsoil", _texture_style("Clay")),
    SpatialFigure("soil_sltppt_sl1", "soil_silt_topsoil", _texture_style("Silt")),
    SpatialFigure("soil_sndppt_sl1", "soil_sand_topsoil", _texture_style("Sand")),
    SpatialFigure("soil_oc_sl1", "soil_organic_carbon_topsoil", ORGANIC_CARBON_STYLE),
    SpatialFigure("soil_ph_sl1", "soil_ph_topsoil", SOIL_PH_STYLE),
    SpatialFigure("soil_bd_sl1", "soil_bulk_density_topsoil", BULK_DENSITY_STYLE),
    SpatialFigure(
        "soil_BDTICM_M_250m_ll",
        "soil_depth_to_bedrock",
        SOIL_DEPTH_STYLE,
        # cm in the file, metres on the bar. See SpatialFigure.scale.
        scale=0.01,
        # soilgrids v1.0 only; see SpatialFigure.guaranteed.
        guaranteed=False,
    ),
)


def figure_paths(plots_dir, formats=("png", "pdf"), declared_only=True):
    """The figure files this family writes under ``plots_dir``.

    Rule 1.12's ``output:`` comes from here rather than restating the stems, the
    same way rule 1.13's does from ``climate_figures.figure_names`` — a registry
    edit then reaches the Snakefile without a second place to remember.

    ``declared_only`` keeps the source-specific figures out of the rule's
    promise; see :attr:`SpatialFigure.guaranteed` for why that is not a
    shortcut. It does NOT stop them being drawn.
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
    with xr.open_dataset(maps_path, mask_and_scale=False) as dataset:
        maps = dataset.load()

    layers = {}
    for argument, stem in SPATIAL_MAP_LAYERS.items():
        path = geoms_dir / f"{stem}.geojson"
        if path.is_file():
            layers[argument] = gpd.read_file(path)
    missing = [name for name in ("basins", "rivers") if name not in layers]
    if missing:
        raise FileNotFoundError(f"{geoms_dir} is missing {missing}")
    return maps, layers


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
    layer = _mask_nodata(maps[figure.variable]).astype("float64")
    extra = [dim for dim in layer.dims if dim not in ("x", "y", "lat", "lon", "latitude", "longitude")]
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
    values = np.asarray(layer.values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True, "no valid cells"
    spread = float(finite.max() - finite.min())
    scale = max(abs(float(finite.max())), abs(float(finite.min())), 1e-12)
    if spread <= 1e-9 * scale:
        return True, f"constant at {finite[0]:g}"
    return False, ""


def source_caveat(layer):
    """``"Source: <product>."`` for the footnote, or ``None`` if unattributed.

    An UNMAPPED source falls back to the catalog key rather than dropping the
    line. "Source: vito." is our plumbing showing through and reads badly, but a
    figure that silently loses its attribution because a catalog gained an entry
    is worse — the ugly version is visible and gets fixed, the missing one is
    not. ``None`` is only for a layer that records no source at all.
    """
    source = layer.attrs.get("source")
    if not source:
        return None
    return f"Source: {SOURCE_CREDITS.get(source, source)}."


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------


def plot_spatial_maps(spatial_dir, plot_dir=None, variables=None, dpi=None, formats=("png", "pdf")):
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
    spatial_dir = Path(spatial_dir)
    plot_dir = Path(plot_dir) if plot_dir is not None else spatial_dir / PLOTS_DIRNAME

    # Checked BEFORE anything is opened: a typo in --variable should not cost a
    # read of the whole raster stack before it is reported.
    unknown = sorted(set(variables or ()).difference(f.variable for f in SPATIAL_MAP_FIGURES))
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
    outline = _outer_boundary(layers["basins"])
    subbasins = layers.get("subbasins")

    written = []
    for figure in selected:
        if figure.variable not in maps:
            # The ONLY skip. It can reach a figure the rule declared as an
            # output, which would fail the rule -- and that is the right
            # failure: it means the registry and ``prepare_spatial_maps``
            # disagree about what the foundation contains.
            print(f"skip {figure.stem}: {SPATIAL_MAPS_FILENAME} has no {figure.variable!r}")
            continue
        layer = prepare_layer(maps, figure, basin_mask)
        degenerate, reason = _is_degenerate(layer)
        if degenerate:
            # Noted, then drawn anyway. See ``_is_degenerate``.
            print(f"note {figure.stem}: {figure.variable} is {reason} over this basin")

        style = figure.style
        if figure.classes is not None:
            classes = figure.classes(layer) if callable(figure.classes) else figure.classes
            # The label is unused on a nominal figure — there is no colourbar to
            # put it on — but it is what ``check_geographic_inputs`` names when
            # the units disagree, so it stays the quantity rather than empty.
            style = RasterStyle(label=figure.stem, palette=None, categories=classes)

        fig, _ = plot_raster_map(
            layer,
            layers.get("rivers"),
            outline,
            subbasins=subbasins,
            gauges=layers.get("gauges"),
            style=style,
            expected_units=figure.expected_units,
            # The product credit, as a footnote under the axes. Read from the
            # LAYER's own `source` attr rather than from the registry entry, so
            # a catalog change cannot leave a figure crediting the wrong product.
            caveat=source_caveat(layer),
            # No title: the filename names the figure (see SpatialFigure.stem).
            # No overlay key either — the layers are still DRAWN, tying the set
            # to one basin, but ``basin_area`` in this same folder carries the
            # legend that explains them, and repeating four entries on ten maps
            # spends the panel height a land-cover legend needs.
            vector_legend=False,
        )
        for extension in formats:
            path = os.path.join(str(plot_dir), f"{figure.stem}.{extension}")
            save_figure(
                path,
                fig=fig,
                # Same reason as basin_area: drop the timestamp and the
                # matplotlib version so two identical runs produce identical
                # bytes and an env bump does not move a fingerprint.
                **(
                    {"metadata": {"CreationDate": None}}
                    if extension == "pdf"
                    else {"dpi": dpi or RASTER_DPI, "metadata": {"Software": None}}
                ),
            )
            written.append(Path(path))
        plt.close(fig)

    maps.close()
    return written


def plot_spatial_maps_from_project(project_dir, plot_dir=None, **kwargs):
    """``plot_spatial_maps`` for a project directory rather than a spatial one."""
    return plot_spatial_maps(Path(project_dir) / SPATIAL_DIRNAME, plot_dir=plot_dir, **kwargs)


def plot_spatial_figure_set(spatial_dir, plot_dir=None, **kwargs):
    """Every figure the spatial foundation supports: ``basin_area`` and the family.

    Rule 1.12's entry point. The two are drawn by ONE rule because they are one
    deliverable — the same basin, the same overlay, the same folder — and because
    they are the reason each other reads: the family suppresses the overlay key
    precisely because ``basin_area`` carries it, so a run that produced one
    without the other would ship ten maps whose linework nothing explains.

    ``basin_area`` is drawn FIRST for the same reason it is listed first
    everywhere else: it is the sheet a reader meets before the thematic ones.
    """
    from blueearth_cst.shared.plot_map import plot_basin_map_from_spatial

    plot_basin_map_from_spatial(spatial_dir, plot_dir=plot_dir)
    return plot_spatial_maps(spatial_dir, plot_dir=plot_dir, **kwargs)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            plot_spatial_figure_set(spatial_dir=sm.params.spatial_dir)
