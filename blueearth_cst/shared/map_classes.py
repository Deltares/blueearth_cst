"""Raster classification, relief and locator basemap for the map adapters.

Moved verbatim from the retired ``shared/cartographic_map.py`` (2026-10-29), with
the reviewed spatial-profile locator settings as defaults. It decides WHAT a map
shows -- class boundaries, ramp colours, the diverging midpoint, the hillshade,
the locator's window and cities -- and leaves drawing to the template,
``plotting/prepared_maps.py`` (toolbox-plot-style.md, Spatial map template).
Adapters: ``shared/plot_spatial_maps.py`` and ``shared/map_frames.py``.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import xarray as xr
from matplotlib import colors

from blueearth_cst.shared.plot_colors import ELEVATION_COLORS

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


#: A monotonic-lightness elevation ramp, hand-built rather than imported: the
#: perceptually-uniform terrain colormaps (cmcrameri, cmocean) are not in the
#: pixi env and adding a dependency for one figure is not warranted. Lightness
#: falls monotonically from low to high ground, so the ramp survives greyscale
#: printing AND every dichromacy — the two failure modes `terrain` had. Replace
#: it only with another ramp whose lightness is monotonic; a test enforces that.
_DEM_ANCHORS = ELEVATION_COLORS


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
# Raised from 0.03 (tuned on a 0.24 deg test basin, which put a 2 deg basin in
# a +/-35 deg window) so the study area stays visible at any basin size.
_LOCATOR_TARGET_BASIN_FRACTION = 0.12


#: Half-widths the auto rule may choose, in degrees. A ladder rather than a
#: continuous value so the window lands on a round number, and the same basin
#: re-rendered after a small extent change does not shift its frame slightly.
_LOCATOR_SPAN_LADDER = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)


#: The basin must FIT its window with room to spare, whatever the target says.
#: Binds for a large basin, where the target alone would choose a window
#: narrower than the basin itself.
_LOCATOR_MIN_SPAN_MARGIN = 0.75


#: Cities are filtered by Natural Earth's own prominence rank (0 = most
#: prominent), then the largest few by population are kept. Both limits matter:
#: the rank keeps towns out, the count keeps a dense region from filling up.
_LOCATOR_CITY_MAX_SCALERANK = 4


_LOCATOR_MAX_CITIES = 8


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


_EARTH_RADIUS_M = 6_371_000.0


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
            f"Maps need every layer in EPSG:{REQUIRED_CRS_EPSG}; got "
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
    # Not capped here: the inset drops cities whose labels do not fit first,
    # then keeps the largest ``_LOCATOR_MAX_CITIES`` of the rest.
    return places.sort_values("pop_max", ascending=False)


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


def _present(layer):
    """Whether an optional layer has anything to draw."""
    return layer is not None and len(layer) > 0
