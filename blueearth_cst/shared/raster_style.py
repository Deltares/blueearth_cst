# -*- coding: utf-8 -*-
"""``RasterStyle`` — how one quantity is drawn, with nothing needed to draw it.

A module that only DECLARES styles must not import the modules that classify
or draw them: ``plot_spatial_maps`` builds its ``RasterStyle`` constants at
module scope and is imported by ``build_model.smk`` at PARSE time for
:func:`~blueearth_cst.shared.plot_spatial_maps.figure_paths` alone, so a heavy
import here would cost every WF1 dry-run (t2608210029).

Nothing here draws, classifies or resolves anything — it is a declaration and
its ``replace``. The classifier, the colormap builders and the diverging
resolvers live in ``shared/map_classes.py``, where a style is interpreted.

Keep this module free of third-party imports. That property is the whole point
of the split, and it is not enforced by anything but this sentence.
"""


class RasterStyle:
    """How one quantity is drawn: palette, label, classification, relief.

    Deliberately a plain object rather than a dataclass: the fields carry long
    explanatory comments a dataclass field list would push into ``field(...)``
    metadata, and this module states no type annotations, matching
    ``map_classes``, which interprets these styles.
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
