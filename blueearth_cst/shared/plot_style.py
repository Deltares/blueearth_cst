# -*- coding: utf-8 -*-
"""Page, typography and export settings shared by every figure the toolbox draws.

One basin assessment ships figures from four different producers. Before this
module each decided its own printed width, resolution and type sizes
independently, so the deliverable a reader actually sees did not look like
itself: 180 mm beside ``figsize=(16/2.54, 22/2.54)`` beside ``figsize=(8, 6)``,
and 600 dpi beside 300. This is the one place those are decided.

**Scope: the page, not the picture.** What belongs here is anything that would
be the same whether the figure is a map, a time series or a scatter — width,
resolution, type sizes, font family, the rcParams every figure is drawn under.
What does NOT belong here is anything a single figure family owns: the basin
map's scale bar, north arrow and gauge labels stay in
``shared/cartographic_map.py`` beside the code that draws them. The test is
whether a second family would ever read the value, not whether it happens to
be a font size.

**Moved by evidence, not by family.** The constants here are the ones a second
consumer already imported (``RASTER_DPI``, ``FONT_SIZE_CAVEAT``,
``COLOR_CAVEAT`` — all three were reached for by ``climate_figures.py`` during
the 2026-08 map sweep) plus the page-level ones that are shared by nature.
``FONT_SIZE_COLORBAR_LABEL`` / ``FONT_SIZE_COLORBAR_TICK`` were deliberately
LEFT in ``cartographic_map``: both map families draw colourbars, but no
non-map figure does, so promoting them would assert a sharing that does not
exist yet. Move a constant here when a second family reads it, not in
anticipation.

Overriding
----------
``dev/scripts/preview_basin_map.py`` drives the basin map by rebinding names in
``cartographic_map``'s module globals (``setattr``), so a value imported there
from this module stays overridable: the drawing code looks the name up in its
OWN globals and finds the rebound one.

That property is why :func:`rcparams` takes every setting as an explicit
argument defaulting to ``None`` rather than reading this module's globals
directly. A function living here would read *this* module's copy, which no
override touches — the value would appear to change and the figure would not,
which is the exact failure mode a tuning knob must never have. Callers pass
their own (rebindable) names in; see ``cartographic_map._publication_rc``.

For the same reason, nothing here is assembled into a module-level constant.
A constant snapshots its inputs at import.
"""

from __future__ import annotations

# ===========================================================================
# PAGE AND EXPORT
# ===========================================================================

#: Figure width in MILLIMETRES — converted once, here, and never re-guessed
#: downstream. 180 mm is the double-column width that Elsevier (190), AGU (190)
#: and Copernicus (170) all accept without downscaling. Set this to your target
#: journal's column width; every other size is chosen to work at it.
FIGURE_WIDTH_MM = 180.0

#: Millimetres per inch. matplotlib thinks in inches; the block above thinks in
#: millimetres, because that is what a journal's author guide states.
MM_PER_INCH = 25.4

#: Resolution of the PNG. The figure is built at its final PHYSICAL size, so
#: this changes the pixel count and nothing else — type, line weights and the
#: map all stay the size they will be on the page. Raising it makes the PNG
#: bigger, NOT the type smaller.
#:
#: 600 because the map is COMBINATION artwork: a raster (the DEM) under line art
#: and type. Publishers ask more of that than of either alone — 300 dpi is the
#: pure-halftone minimum, while Elsevier asks 500 and AGU/Wiley 600 for
#: combination figures. 600 clears all of them, and 180 mm at 600 dpi is
#: 4252 px, which is still a small file for a map this sparse.
#:
#: It also covers PowerPoint. matplotlib writes the resolution into the PNG's
#: ``pHYs`` chunk, so PowerPoint and Word insert the image at its true 180 mm
#: width instead of assuming 96 dpi and dropping in something 750 mm wide. That
#: is the part that actually breaks when a figure is exported without dpi
#: metadata, and it is verified rather than assumed (checked 2026-08-09).
#:
#: The PDF remains the deliverable for print: it is vector, so it has no
#: resolution to get wrong.
RASTER_DPI = 600

# ===========================================================================
# TYPOGRAPHY
# ===========================================================================
# Type sizes in POINTS at the printed width above. Applied through
# ``rc_context`` so the process-wide rcParams that other plotting rules inherit
# are left untouched. Raise every value together to scale the labelling; raise
# one to re-balance it.
# ---------------------------------------------------------------------------

#: Fallback for any text element that does not carry its own size, and the base
#: the axes title is derived from (+1 pt).
FONT_SIZE_BASE = 8.0

#: Axis tick labels — on a map, the coordinate graticule labels.
FONT_SIZE_TICK = 7.0

#: Legend entries and the legend's own title.
FONT_SIZE_LEGEND = 6.5

#: A figure-level ``suptitle``. One point above base so it reads as a title
#: without shouting at a 180 mm width.
FONT_SIZE_TITLE = 9.0

#: The provenance/caveat footnote a figure carries under its axes — the "these
#: are simulated, not observed" line. Deliberately the smallest type on the
#: page: it must be legible when looked for and invisible when not.
FONT_SIZE_CAVEAT = 6.0

#: Font family. ``None`` keeps matplotlib's default (DejaVu Sans, which embeds
#: cleanly in the PDF). Set e.g. ``"Arial"`` or ``["Helvetica", "Arial"]`` to
#: match a manuscript — but check the exported PDF, because a missing family
#: falls back SILENTLY.
FONT_FAMILY = None

# ===========================================================================
# COLOUR
# ===========================================================================

#: The caveat footnote's ink. Grey rather than black so it sits below the
#: figure's own content in the reading order.
COLOR_CAVEAT = "0.35"

#: Where a caveat footnote STARTS, in figure fractions.
#:
#: LEFT-ALIGNED everywhere (owner ruling 2026-08-17): a footnote is read as
#: prose, and prose starts where the reader's eye already is. ``supxlabel``'s
#: default centring puts it under the axes midpoint — not an edge the reader can
#: see, and one that MOVES with the caveat's own length, so a one-line and a
#: two-line footnote sit differently on two figures of the same family.
#:
#: Lives here rather than in either plotting module because BOTH families use
#: it: the series figures (``climate_analysis.climate_figures``) and the
#: cartographic template (``shared.cartographic_map``). The map family
#: right-aligned to its side panel's measured edge until this ruling.
CAVEAT_X = 0.012

# ===========================================================================
# PDF TEXT ENCODING
# ===========================================================================

#: 42 = TrueType. matplotlib's default (Type 3) is not editable in Illustrator
#: and is rejected outright by several publishers' preflight, so every figure
#: this toolbox exports sets it.
PDF_FONTTYPE = 42


# ===========================================================================
# DERIVED VALUES
# ===========================================================================
# Functions, never constants — see the module docstring on why.
# ---------------------------------------------------------------------------


def mm_to_inches(millimetres: float) -> float:
    """Millimetres to inches, the one conversion matplotlib needs."""
    return millimetres / MM_PER_INCH


def figure_width_inches(width_mm: float | None = None) -> float:
    """The standard printed width in inches.

    ``width_mm`` overrides :data:`FIGURE_WIDTH_MM` for a caller that has its own
    (overridable) copy of the name — pass it rather than relying on this
    module's global, for the reason in the module docstring.
    """
    return mm_to_inches(FIGURE_WIDTH_MM if width_mm is None else width_mm)


def rcparams(
    *,
    font_size_base: float | None = None,
    font_size_tick: float | None = None,
    font_size_legend: float | None = None,
    font_family=None,
    axes_linewidth: float | None = None,
) -> dict:
    """The rcParams a publication figure is drawn under, as an ``rc_context`` dict.

    Every argument defaults to ``None`` and is resolved against this module's
    constants inside the body — NOT in the signature, where the default would be
    evaluated once at import and freeze the value. Callers whose own globals can
    be rebound at runtime (``cartographic_map``, driven by
    ``dev/scripts/preview_basin_map.py``) pass their names in explicitly so an
    override actually reaches the figure.

    ``axes_linewidth`` is left to the caller with no fallback of its own beyond
    matplotlib's, because line weight is picture-level rather than page-level:
    the basin map owns a spine weight tuned to its own furniture. Omit it and
    matplotlib's default stands.
    """
    base = FONT_SIZE_BASE if font_size_base is None else font_size_base
    tick = FONT_SIZE_TICK if font_size_tick is None else font_size_tick
    legend = FONT_SIZE_LEGEND if font_size_legend is None else font_size_legend
    family = FONT_FAMILY if font_family is None else font_family

    params = {
        "font.size": base,
        "axes.titlesize": base + 1.0,
        "axes.labelsize": base,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "legend.title_fontsize": legend,
        "pdf.fonttype": PDF_FONTTYPE,
        "ps.fonttype": PDF_FONTTYPE,
    }
    if axes_linewidth is not None:
        params.update(
            {
                "axes.linewidth": axes_linewidth,
                "xtick.major.width": axes_linewidth,
                "ytick.major.width": axes_linewidth,
            }
        )
    if family:
        params["font.family"] = family
    return params
