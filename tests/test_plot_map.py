# -*- coding: utf-8 -*-
"""Unit tests for the cartographic raster-map template and the basin map.

The figures are only rendered in full by a WF1 run, so what is testable here is
the arithmetic that decides whether they are CORRECT rather than merely
produced: the latitude correction, the per-basin exaggeration, the declared
physical width, the classification, and the ramps' accessibility properties.
Each of these was a real defect in the pre-2026-08 figures.

Since 2026-08 one drawing function serves four quantities (``plot_raster_map``
plus ``RASTER_STYLES``). The cartography lives in ``shared.cartographic_map``
and the wflow-specific reading half in ``shared.plot_map``; this file covers
both, because every test here is ultimately about one of the two figures they
combine to produce.
"""

import numpy as np
import pytest
import xarray as xr
from matplotlib import colors

from blueearth_cst.shared import cartographic_map as carto
from blueearth_cst.shared import plot_map
from blueearth_cst.shared.cartographic_map import (
    _CORNERS,
    _EXTENT_BUFFER_DEG,
    _NORTH_ARROW_CORNER,
    FIGURE_WIDTH_MM,
    GRATICULE_MAX_TICKS,
    MM_PER_INCH,
    RASTER_STYLES,
    RIVER_WIDTH_UNIFORM,
    RasterStyle,
    _class_levels,
    _colorbar_inset,
    _corner_occupancy,
    _divide_linework,
    _elevation_colormap,
    _equal_interval_levels,
    _figure_size,
    _finite_cells,
    _graticule_ticks,
    _locator_span,
    _mask_nodata,
    _metres_per_degree,
    _nice_round_length,
    _nice_step_up,
    _overlay_contrast,
    _publication_rc,
    _river_linewidths,
    _scale_bar_corner,
    _style_colormap,
    _weighted_quantiles,
    _wrap_label,
    check_geographic_inputs,
    map_extent,
    pixel_resolution,
    plot_raster_map,
    resolve_temperature_style,
    spatial_dim_names,
)
from blueearth_cst.shared.plot_map import (
    _basin_outline,
    _elevation_levels,
    load_basin_layers,
    plot_basin_map,
)


def _dem(x_name="longitude", y_name="latitude", res=0.01, nx=4, ny=3):
    """A minimal north-up geographic DEM, as wflow writes them."""
    x = np.arange(nx) * res
    y = -np.arange(ny) * res
    return xr.DataArray(
        np.arange(ny * nx, dtype="float32").reshape(ny, nx),
        dims=(y_name, x_name),
        coords={y_name: y, x_name: x},
    )


# --- the EPSG:4326 correction ------------------------------------------------
# The model grid is geographic, so a degree of longitude is not a distance.
# Ignoring that gave a scale bar that was right only at the equator and a
# hillshade gradient off by ~10^5.


def test_degrees_are_square_at_the_equator():
    metres_lon, metres_lat = _metres_per_degree(0.0)
    assert metres_lon == pytest.approx(metres_lat)


@pytest.mark.parametrize("latitude, ratio", [(60.0, 0.5), (45.0, 0.7071), (0.0, 1.0)])
def test_longitude_degree_shrinks_with_the_cosine_of_latitude(latitude, ratio):
    metres_lon, metres_lat = _metres_per_degree(latitude)
    assert metres_lon / metres_lat == pytest.approx(ratio, abs=1e-4)


def test_latitude_degree_does_not_vary():
    assert _metres_per_degree(0.0)[1] == pytest.approx(_metres_per_degree(70.0)[1])


# --- scale bar ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_km, expected_km",
    [(37.0, 20.0), (6.1, 5.0), (0.42, 0.2), (1.0, 1.0), (99.0, 50.0)],
)
def test_scale_bar_rounds_down_to_one_two_or_five(raw_km, expected_km):
    assert _nice_round_length(raw_km) == expected_km


def test_scale_bar_length_is_never_zero_or_negative():
    assert _nice_round_length(0.0) > 0
    assert _nice_round_length(-5.0) > 0


# --- physical size ------------------------------------------------------------


@pytest.mark.parametrize(
    "extent",
    [
        (5.0, 6.0, 45.0, 45.5),  # wide
        (9.6, 9.9, 0.3, 0.5),  # small equatorial basin
        (80.0, 89.0, 25.0, 35.0),  # large, tall
    ],
)
def test_width_is_the_declared_millimetre_constant_for_any_basin(extent):
    width_in, height_in = _figure_size(np.asarray(extent))
    assert width_in * MM_PER_INCH == pytest.approx(FIGURE_WIDTH_MM)
    assert height_in > 0


def test_a_taller_basin_yields_a_taller_figure():
    _, short = _figure_size(np.array([0.0, 1.0, 0.0, 0.5]))
    _, tall = _figure_size(np.array([0.0, 1.0, 0.0, 1.0]))
    assert tall > short


def test_degenerate_extent_does_not_divide_by_zero():
    width_in, height_in = _figure_size(np.array([5.0, 5.0, 45.0, 45.0]))
    assert np.isfinite(width_in) and np.isfinite(height_in)


@pytest.mark.parametrize("aspect", [0.05, 0.15, 3.0, 12.0])
def test_extreme_basin_shapes_stay_inside_the_aspect_clamps(aspect):
    """A ribbon delta and a narrow headwater must both give a usable page."""
    width_in, height_in = _figure_size(np.array([0.0, 1.0, 0.0, aspect]))
    assert np.isfinite(height_in) and height_in > 0
    assert height_in < 2.0 * width_in


# --- graticule ----------------------------------------------------------------
# `map_extent` pads the DEM bounds and clamps nothing, so the tick chooser is the
# last thing standing between a polar basin and a latitude label past the pole.


def test_ticks_stay_inside_the_basin_extent():
    lon, lat = _graticule_ticks(np.array([9.638, 9.878, 0.33, 0.503]))
    assert lon and lat
    assert all(9.638 <= t <= 9.878 for t in lon)
    assert all(0.33 <= t <= 0.503 for t in lat)


@pytest.mark.parametrize(
    "extent",
    [
        (20.0, 25.0, 80.0, 92.0),  # padding pushed lat_max past the north pole
        (20.0, 25.0, -93.0, -80.0),  # and past the south pole
    ],
)
def test_no_latitude_tick_is_placed_past_the_pole(extent):
    """A latitude beyond +/-90 does not exist, so it must never be labelled."""
    _, lat = _graticule_ticks(np.array(extent))
    assert lat, "clamping must not empty the graticule"
    assert all(-90.0 <= t <= 90.0 for t in lat)


def test_longitude_past_the_antimeridian_is_preserved():
    """Unlike latitude, past +/-180 is a legitimate antimeridian-spanning basin."""
    lon, _ = _graticule_ticks(np.array([176.0, 184.0, 60.0, 64.0]))
    assert max(lon) > 180.0


@pytest.mark.parametrize(
    "extent",
    [
        (9.638, 9.878, 0.33, 0.503),  # sub-degree
        (5.0, 6.0, 45.0, 45.5),
        (80.0, 89.0, 25.0, 35.0),
    ],
)
def test_tick_count_respects_the_declared_maximum(extent):
    """Regression guard: cartopy's LatitudeLocator overshoots this on a
    sub-degree basin, which is why the graticule does not use it."""
    lon, lat = _graticule_ticks(np.array(extent))
    assert len(lon) <= GRATICULE_MAX_TICKS + 1
    assert len(lat) <= GRATICULE_MAX_TICKS + 1


# --- furniture placement ------------------------------------------------------
# Fixed corners are only safe for the basin shape they were tuned on.


def _basin_covering(x0, y0, x1, y1):
    from shapely.geometry import box

    return box(x0, y0, x1, y1)


_UNIT_EXTENT = (0.0, 1.0, 0.0, 1.0)


def test_occupancy_is_one_for_a_basin_filling_the_box():
    occupancy = _corner_occupancy(_basin_covering(0, 0, 1, 1), _UNIT_EXTENT)
    assert set(occupancy) == set(_CORNERS)
    assert all(value == pytest.approx(1.0) for value in occupancy.values())


def test_occupancy_is_zero_where_the_basin_is_absent():
    # basin hugging the top-right only
    occupancy = _corner_occupancy(_basin_covering(0.75, 0.75, 1, 1), _UNIT_EXTENT)
    assert occupancy["lower left"] == pytest.approx(0.0)
    assert occupancy["upper right"] > 0.5


def test_scale_bar_avoids_the_occupied_corner():
    """A basin in the south-west must not get the scale bar drawn on top of it."""
    basin = _basin_covering(0.0, 0.0, 0.45, 0.45)
    assert _scale_bar_corner(basin, _UNIT_EXTENT) != "lower left"


def test_scale_bar_never_takes_the_north_arrow_corner():
    """The arrow's corner is reserved, even when it is the emptiest."""
    basin = _basin_covering(0.0, 0.0, 0.9, 0.9)  # leaves only upper right free
    assert _scale_bar_corner(basin, _UNIT_EXTENT) != _NORTH_ARROW_CORNER


def test_scale_bar_prefers_a_lower_corner_among_EQUALLY_empty_ones():
    """The bottom preference breaks ties; it does not override emptiness."""
    central = _basin_covering(0.35, 0.35, 0.65, 0.65)  # touches no corner
    assert _scale_bar_corner(central, _UNIT_EXTENT) == "lower left"


def test_emptiness_outranks_the_bottom_preference():
    """A basin on lower left sends the bar to the OTHER lower corner.

    The legend used to occupy lower right, which forced the bar upward here.
    Now that the legend sits in the side panel, the bar keeps a bottom corner.
    """
    basin = _basin_covering(0.0, 0.0, 0.45, 0.45)
    assert _scale_bar_corner(basin, _UNIT_EXTENT) == "lower right"


def test_placement_is_deterministic_for_a_symmetric_basin():
    """Ties must not make the figure depend on dict iteration order."""
    basin = _basin_covering(0.35, 0.35, 0.65, 0.65)  # touches no corner
    assert _scale_bar_corner(basin, _UNIT_EXTENT) == _scale_bar_corner(
        basin, _UNIT_EXTENT
    )


def test_a_basin_filling_its_box_still_yields_a_valid_corner():
    assert _scale_bar_corner(_basin_covering(0, 0, 1, 1), _UNIT_EXTENT) in _CORNERS


def test_the_scale_bar_yields_the_corner_the_locator_took():
    """Two artists in one corner is the collision this budgeting prevents."""
    central = _basin_covering(0.35, 0.35, 0.65, 0.65)  # every corner equally free
    taken = _scale_bar_corner(central, _UNIT_EXTENT, {_NORTH_ARROW_CORNER})
    assert (
        _scale_bar_corner(central, _UNIT_EXTENT, {_NORTH_ARROW_CORNER, taken}) != taken
    )


def test_reserving_every_corner_still_returns_one():
    """Better a crowded bar than a crash: the fallback must not empty the list."""
    assert (
        _scale_bar_corner(
            _basin_covering(0.35, 0.35, 0.65, 0.65), _UNIT_EXTENT, set(_CORNERS)
        )
        in _CORNERS
    )


# --- locator inset ------------------------------------------------------------
# The "where in the world is this" panel. Its window and its box are pure
# geometry and testable; whether it LOOKS right is a question for the render.


@pytest.fixture
def restore_locator():
    names = (
        "LOCATOR_ENABLED",
        "_LOCATOR_CORNER",
        "_LOCATOR_WIDTH",
        "_LOCATOR_PLACEMENT",
        "_LOCATOR_SPAN_DEG",
    )
    original = {name: getattr(carto, name) for name in names}
    yield
    for name, value in original.items():
        setattr(carto, name, value)


def test_the_locator_window_is_square_and_centred_on_the_basin():
    window = carto._locator_window([9.65, 9.86, 0.35, 0.50])
    assert (window[1] - window[0]) == pytest.approx(window[3] - window[2])
    assert 0.5 * (window[0] + window[1]) == pytest.approx(9.755)


def test_a_polar_basin_gets_a_full_window_rather_than_half_of_one():
    """Clipping at the pole would leave the inset half empty; re-centre instead."""
    window = carto._locator_window([20.0, 21.0, 88.0, 89.5])
    assert (window[1] - window[0]) == pytest.approx(window[3] - window[2])
    assert window[3] <= 90.0


@pytest.mark.parametrize(
    "extent", [(0.0, 1.0, 0.0, 1.0), (0.0, 4.0, 0.0, 1.0), (0.0, 1.0, 0.0, 3.0)]
)
def test_the_locator_box_comes_out_square_on_the_page(extent):
    """Equal fractions in a non-square panel would render a slot, not a square."""
    lon_span, lat_span = extent[1] - extent[0], extent[3] - extent[2]
    _, _, width, height = carto._locator_box(np.asarray(extent), "upper left")
    assert width * lon_span == pytest.approx(height * lat_span)
    assert 0 < width <= 1 and 0 < height <= 1


@pytest.mark.parametrize(
    ("corner", "expect_left", "expect_upper"),
    [
        ("upper left", True, True),
        ("lower right", False, False),
        ("upper right", False, True),
        ("lower left", True, False),
    ],
)
def test_the_locator_box_lands_in_the_corner_it_is_given(
    corner, expect_left, expect_upper
):
    x0, y0, width, height = carto._locator_box(np.array([0.0, 1.0, 0.0, 1.0]), corner)
    assert (x0 < 0.5) is expect_left
    assert (y0 + height > 0.5) is expect_upper


def test_no_corner_is_reserved_when_the_locator_is_off(restore_locator):
    carto.LOCATOR_ENABLED = False
    assert carto._locator_corner(_basin_covering(0, 0, 1, 1), _UNIT_EXTENT) is None


def test_the_locator_never_takes_the_north_arrow_corner(restore_locator):
    carto.LOCATOR_ENABLED = True
    carto._LOCATOR_CORNER = "auto"
    basin = _basin_covering(0.0, 0.0, 0.9, 0.9)  # only upper right is free
    assert carto._locator_corner(basin, _UNIT_EXTENT) != _NORTH_ARROW_CORNER


def test_an_explicit_locator_corner_is_honoured(restore_locator):
    carto.LOCATOR_ENABLED = True
    carto._LOCATOR_PLACEMENT = "map"
    carto._LOCATOR_CORNER = "lower right"
    assert (
        carto._locator_corner(_basin_covering(0, 0, 1, 1), _UNIT_EXTENT)
        == "lower right"
    )


def test_a_panel_locator_claims_no_map_corner(restore_locator):
    """It is not on the map, so the scale bar gets that corner back."""
    carto.LOCATOR_ENABLED = True
    carto._LOCATOR_PLACEMENT = "panel"
    carto._LOCATOR_CORNER = "lower right"
    assert carto._locator_corner(_basin_covering(0, 0, 1, 1), _UNIT_EXTENT) is None


# --- the vendored basemap -----------------------------------------------------
# It is committed rather than downloaded, so the thing to check is that the
# committed file is intact and holds what the inset asks it for.


def test_the_vendored_basemap_is_present():
    assert carto.BASEMAP_PATH.is_file(), (
        f"{carto.BASEMAP_PATH} is missing; see config/basemap/README.md"
    )


@pytest.mark.parametrize("layer", ["land", "borders", "places"])
def test_each_basemap_layer_is_readable_and_not_empty(layer):
    import geopandas as gpd

    assert len(gpd.read_file(carto.BASEMAP_PATH, layer=layer)) > 0


def test_the_places_layer_carries_the_columns_the_inset_filters_on():
    import geopandas as gpd

    places = gpd.read_file(carto.BASEMAP_PATH, layer="places")
    assert {"name", "pop_max", "scalerank"} <= set(places.columns)


# --- basin outline vs subcatchment divides ------------------------------------
# `mod.basins` returns one polygon per SUBCATCHMENT once gauges are burned in.


def _basins_frame(*boxes):
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"value": list(range(len(boxes)))},
        geometry=[box(*bounds) for bounds in boxes],
        crs="EPSG:4326",
    )


def test_a_multi_gauge_model_dissolves_to_one_outline():
    """Four gauge subcatchments must still draw ONE basin boundary."""
    four = _basins_frame((0, 0, 1, 1), (1, 0, 2, 1), (0, 1, 1, 2), (1, 1, 2, 2))
    assert len(four) == 4
    assert len(_basin_outline(four)) == 1


def test_the_dissolved_outline_drops_the_internal_divides():
    two = _basins_frame((0, 0, 1, 1), (1, 0, 2, 1))
    outline = _basin_outline(two).geometry.iloc[0]
    # the shared edge at x == 1 is interior to the union, so the ring is the
    # 0..2 rectangle rather than the two squares' perimeters
    assert outline.bounds == (0.0, 0.0, 2.0, 1.0)
    assert outline.boundary.length == pytest.approx(6.0)


def test_a_single_basin_survives_the_dissolve_unchanged():
    one = _basins_frame((0, 0, 1, 1))
    assert len(_basin_outline(one)) == 1
    assert _basin_outline(one).geometry.iloc[0].bounds == (0.0, 0.0, 1.0, 1.0)


# --- river line weights -------------------------------------------------------


class _FakeRivers:
    """Minimal stand-in for the ``strord`` column ``_river_linewidths`` reads."""

    def __init__(self, orders, column="strord"):
        self._orders = np.asarray(orders, dtype=float)
        self.columns = [column] if column is not None else []
        self._column = column

    def __getitem__(self, key):
        assert key == self._column
        return self

    def astype(self, dtype):
        return self

    def to_numpy(self):
        return self._orders


def test_river_widths_are_publication_scale_and_increase_with_order():
    widths = _river_linewidths(_FakeRivers([1, 2, 4, 8]))
    assert np.all(np.diff(widths) > 0)
    # The old `strord / 2` drew an 8th-order river as a 4 pt band at 180 mm.
    assert widths.max() < 1.5


def test_uniform_stream_order_does_not_divide_by_zero():
    widths = _river_linewidths(_FakeRivers([3, 3, 3]))
    assert np.all(np.isfinite(widths)) and np.all(widths > 0)


def test_a_river_layer_without_an_order_column_still_draws():
    """Rivers from outside wflow carry no `strord`; that is not an error."""
    assert _river_linewidths(_FakeRivers([1, 2], column="order")) == RIVER_WIDTH_UNIFORM
    assert _river_linewidths(_FakeRivers([1, 2]), None) == RIVER_WIDTH_UNIFORM


def test_an_alternative_order_column_is_honoured():
    widths = _river_linewidths(_FakeRivers([1, 4], column="order"), "order")
    assert np.all(np.diff(widths) > 0)


# --- the figure is still colour-independent -----------------------------------


def test_ramp_endpoints_are_distinguishable_under_dichromacy():
    """Deuteranope-simulated endpoints must still differ in luminance.

    A cheap proxy for the real check: collapse the red-green channel pair, which
    is what deuteranopia does, and confirm the ends have not converged.
    """
    ramp = _elevation_colormap()
    low, high = np.array(ramp(0.0)[:3]), np.array(ramp(1.0)[:3])

    def collapse(rgb):
        return np.array([rgb[:2].mean(), rgb[:2].mean(), rgb[2]])

    assert np.abs(collapse(low) - collapse(high)).max() > 0.3


def test_colormap_is_a_matplotlib_colormap():
    assert isinstance(_elevation_colormap(), colors.Colormap)


def test_the_ramp_can_be_cut_into_discrete_classes():
    assert _elevation_colormap(6).N == 6


# --- elevation classes --------------------------------------------------------
# The bar is stepped, its boundaries are round numbers, and both ends are
# labelled. The trap is the zero baseline: it must not flatten a highland basin.


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.9, 1.0),
        (1.0, 1.0),
        (1.2, 1.5),
        (1.7, 2.0),
        (2.4, 2.5),
        (3.0, 5.0),
        (16.9, 20.0),
    ],
)
def test_a_class_width_rounds_UP_to_a_ladder_rung(value, expected):
    """Down would give MORE classes than asked for -- the opposite of the point.

    The rungs give each quantity the steps a reader expects -- 25/50/100/150/
    200 mm of rainfall, 0.25/0.5/1 degC -- at every decade, because the ladder
    is multiplicative. ``4`` was dropped when the tick cap rose to 9: it existed
    only so a five-tick bar over 0-140 m could land on 40, and it put 40 mm and
    0.4 degC in reach of quantities that never want them.
    """
    assert _nice_step_up(value) == expected


def test_a_style_may_carry_its_own_ladder():
    """Temperature steps in 0.25/0.5/1, not in the general ladder's 0.15."""
    assert _nice_step_up(0.19, RASTER_STYLES["temp"].step_ladder) == 0.25
    assert _nice_step_up(0.19) == 0.2


def test_replacing_a_style_field_keeps_every_other_one():
    """Rebuilding a style by hand dropped whatever was added later.

    That is how the temperature ladder never reached the figure and rainfall
    reserved no white: a missing field looks exactly like a default.
    """
    original = RASTER_STYLES["precip"]
    copy = original.replace(label="Other (mm)")
    assert copy.label == "Other (mm)"
    for field in (
        "palette",
        "classification",
        "zero_baseline",
        "reserve_low_for",
        "low_clip",
        "step_ladder",
        "relief",
    ):
        assert getattr(copy, field) == getattr(original, field), field


def test_explicit_levels_bypass_the_classifier():
    """The sidecar's whole mechanism: a bar handed in, not derived."""
    pinned = [0.0, 10.0, 20.0, 30.0]
    style = RASTER_STYLES["precip"].replace(levels=pinned)
    assert list(_class_levels(_ramp(2700.0, 2900.0), style)) == pinned


def _dem_spanning(low, high):
    values = np.linspace(low, high, 400)
    side = 20
    return xr.DataArray(values.reshape(side, side), dims=("latitude", "longitude"))


def test_a_lowland_basin_gets_a_ramp_starting_at_zero():
    levels = _elevation_levels(_dem_spanning(4.0, 137.0))
    assert levels[0] == 0.0
    assert levels[-1] >= 137.0


def test_a_plateau_basin_keeps_its_own_baseline():
    """Zeroed, a 1900-1960 m basin lands entirely in ONE class and goes flat."""
    levels = _elevation_levels(_dem_spanning(1900.0, 1960.0))
    assert levels[0] > 0.0
    # the basin must span most of the ramp, not a single band of it
    assert (1960.0 - 1900.0) / (levels[-1] - levels[0]) > 0.5


def test_a_mountain_basin_still_zeroes_because_it_can_afford_to():
    levels = _elevation_levels(_dem_spanning(800.0, 6200.0))
    assert levels[0] == 0.0


def test_the_top_class_contains_the_highest_cell_rather_than_ending_at_it():
    """A boundary exactly at the summit renders the summit as nodata."""
    assert _elevation_levels(_dem_spanning(0.0, 100.0))[-1] >= 100.0


def test_boundaries_sit_on_multiples_of_the_class_width():
    """A floor of 1903 m must label as 1900, not carry itself up the bar."""
    levels = _elevation_levels(_dem_spanning(1903.0, 2410.0))
    step = levels[1] - levels[0]
    assert np.allclose(np.mod(levels, step), 0.0)


def test_classes_are_evenly_spaced_and_increasing():
    levels = _elevation_levels(_dem_spanning(4.0, 137.0))
    assert np.all(np.diff(levels) > 0)
    assert np.allclose(np.diff(levels), levels[1] - levels[0])


def test_a_flat_dem_does_not_produce_a_degenerate_ramp():
    levels = _elevation_levels(_dem_spanning(12.0, 12.0))
    assert len(levels) >= 2 and np.all(np.diff(levels) > 0)


def test_a_below_sea_level_basin_is_not_clipped_at_zero():
    levels = _elevation_levels(_dem_spanning(-30.0, 90.0))
    assert levels[0] <= -30.0


# --- the tunable block stays live ---------------------------------------------
# Anything assembled from the tunables is derived in a function, not frozen into
# a module constant. A constant would snapshot its inputs at import, so
# `dev/scripts/preview_basin_map.py` would set a value and change nothing — the
# worst failure mode for a tuning tool, because the figure still renders.


@pytest.fixture
def restore_tunables():
    """Put the module globals back, whatever a test does to them."""
    names = ("FONT_SIZE_TICK", "WIDTH_AXES_SPINE", "_PANEL_LEFT", "_COLORBAR_WIDTH")
    original = {name: getattr(carto, name) for name in names}
    yield
    for name, value in original.items():
        setattr(carto, name, value)


def test_rcparams_follow_a_font_size_override(restore_tunables):
    carto.FONT_SIZE_TICK = 99.0
    carto.WIDTH_AXES_SPINE = 3.0
    assert _publication_rc()["xtick.labelsize"] == 99.0
    assert _publication_rc()["ytick.labelsize"] == 99.0
    assert _publication_rc()["axes.linewidth"] == 3.0


def test_colorbar_inset_follows_the_panel_position(restore_tunables):
    carto._PANEL_LEFT = 1.5
    carto._COLORBAR_WIDTH = 0.1
    left, _, width, _ = _colorbar_inset()
    assert (left, width) == (1.5, 0.1)


@pytest.fixture
def restore_colorbar_label_position():
    original = carto.COLORBAR_LABEL_POSITION
    yield
    carto.COLORBAR_LABEL_POSITION = original


def test_a_right_label_leaves_the_colorbar_at_full_height(
    restore_colorbar_label_position,
):
    carto.COLORBAR_LABEL_POSITION = "right"
    assert _colorbar_inset()[3] == pytest.approx(carto._colorbar_height())


def test_a_top_label_moves_the_bar_down_rather_than_shortening_it(
    restore_colorbar_label_position,
):
    """``_COLORBAR_HEIGHT`` is a promise about the bar's LENGTH.

    A brief is written against it ("60-70% of the axis"), so the label's room
    comes out of the bar's position, not its height. The earlier behaviour
    shortened the bar instead, which broke that promise silently.
    """
    carto.COLORBAR_LABEL_POSITION = "top"
    _, bottom, _, height = _colorbar_inset()
    assert height == pytest.approx(carto._colorbar_height())
    assert bottom + height + carto._COLORBAR_TOP_LABEL_HEADROOM <= 1.0 + 1e-9


def test_each_extra_label_line_pushes_the_bar_down_the_same_again(
    restore_colorbar_label_position,
):
    """A fixed headroom would clip a two-line label or gap above a one-line one."""
    carto.COLORBAR_LABEL_POSITION = "top"
    one, two = _colorbar_inset(1)[1], _colorbar_inset(2)[1]
    assert one - two == pytest.approx(carto._COLORBAR_TOP_LABEL_HEADROOM)


def test_the_bar_gives_way_when_the_panel_cannot_hold_it(
    restore_colorbar_label_position,
):
    """Position first, length only when the band genuinely cannot fit it."""
    carto.COLORBAR_LABEL_POSITION = "top"
    squeezed = _colorbar_inset(1, reserved_top=0.5, band_bottom=0.3)
    assert squeezed[3] < carto._colorbar_height()
    assert squeezed[3] >= carto._COLORBAR_MIN_HEIGHT


def test_label_lines_do_not_move_a_right_hand_label(
    restore_colorbar_label_position,
):
    carto.COLORBAR_LABEL_POSITION = "right"
    assert _colorbar_inset(1) == _colorbar_inset(3)


def test_an_unknown_label_position_raises_rather_than_silently_defaulting(
    restore_colorbar_label_position,
):
    """A knob that reads as set but does nothing is the worst failure here."""
    carto.COLORBAR_LABEL_POSITION = "above"
    with pytest.raises(ValueError, match="COLORBAR_LABEL_POSITION"):
        _colorbar_inset()


def test_the_pdf_stays_truetype_whatever_the_sizes_are():
    """Type 3 is rejected by several publishers' preflight; 42 is TrueType."""
    assert _publication_rc()["pdf.fonttype"] == 42


# --- reading the model without hydromt ---------------------------------------
# These four replace what hydromt's ``.raster`` accessor and ``WflowSbmModel``
# used to do. The accessor sniffed dimension names and computed the bounding box
# for us; doing it here means the sniffing needs its own tests, because a wrong
# axis silently transposes the map rather than raising.


@pytest.mark.parametrize(
    ("x_name", "y_name"),
    [("longitude", "latitude"), ("lon", "lat"), ("x", "y"), ("LONGITUDE", "LATITUDE")],
)
def test_spatial_dims_are_found_under_every_accepted_spelling(x_name, y_name):
    assert spatial_dim_names(_dem(x_name, y_name)) == (x_name, y_name)


def test_unrecognised_dims_raise_rather_than_guess():
    """A guessed axis transposes the map silently; an exception does not."""
    with pytest.raises(ValueError, match="cannot identify the spatial dimensions"):
        spatial_dim_names(_dem("easting", "northing"))


def test_resolution_is_signed_and_matches_the_coordinate_spacing():
    """``.raster.res`` is signed: negative y for the north-up order wflow writes."""
    res_x, res_y = pixel_resolution(_dem(res=0.25))
    assert res_x == pytest.approx(0.25)
    assert res_y == pytest.approx(-0.25)


def test_a_single_cell_axis_raises_instead_of_returning_nonsense():
    with pytest.raises(ValueError, match="length 1"):
        pixel_resolution(_dem(nx=1))


def test_extent_reaches_half_a_cell_beyond_the_outermost_centres():
    """Coordinates are cell CENTRES; dropping the half-cell crops a row and column."""
    res = 0.01
    dem = _dem(res=res, nx=4, ny=3)
    lon_min, lon_max, lat_min, lat_max = map_extent(dem, buffer_deg=0.0)
    assert lon_min == pytest.approx(0.0 - res / 2)
    assert lon_max == pytest.approx(3 * res + res / 2)
    assert lat_min == pytest.approx(-2 * res - res / 2)
    assert lat_max == pytest.approx(0.0 + res / 2)


def test_extent_buffer_pads_every_side_and_defaults_to_the_constant():
    dem = _dem()
    tight = map_extent(dem, buffer_deg=0.0)
    padded = map_extent(dem)
    assert padded[0] == pytest.approx(tight[0] - _EXTENT_BUFFER_DEG)
    assert padded[1] == pytest.approx(tight[1] + _EXTENT_BUFFER_DEG)
    assert padded[2] == pytest.approx(tight[2] - _EXTENT_BUFFER_DEG)
    assert padded[3] == pytest.approx(tight[3] + _EXTENT_BUFFER_DEG)


def test_an_undecoded_fill_value_is_masked_to_nan():
    """xarray usually decodes _FillValue for us; mask_and_scale=False does not."""
    dem = _dem()
    dem.values[0, 0] = -9999.0
    dem.attrs["_FillValue"] = -9999.0
    assert np.isnan(_mask_nodata(dem).values[0, 0])


def test_a_nan_fill_value_leaves_the_array_untouched():
    dem = _dem()
    dem.attrs["_FillValue"] = np.nan
    assert not np.isnan(_mask_nodata(dem).values).any()


def test_a_missing_staticmaps_names_the_file(tmp_path):
    (tmp_path / "staticgeoms").mkdir()
    with pytest.raises(FileNotFoundError, match="staticmaps.nc"):
        load_basin_layers(tmp_path)


def test_a_missing_staticgeoms_names_the_directory(tmp_path):
    (tmp_path / "staticmaps.nc").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="staticgeoms"):
        load_basin_layers(tmp_path)


def test_missing_required_layers_are_named_because_hydromt_would_have_derived_them(
    tmp_path,
):
    """The one behavioural loss versus ``mod.rivers``/``mod.basins`` -- say so."""
    _dem().to_dataset(name=plot_map.ELEVATION_VARIABLE).to_netcdf(
        tmp_path / plot_map.STATICMAPS_FILENAME
    )
    (tmp_path / plot_map.STATICGEOMS_DIRNAME).mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_basin_layers(tmp_path)
    message = str(excinfo.value)
    assert "rivers" in message and "basins" in message
    assert "derive" in message


def test_an_absent_elevation_variable_lists_what_the_file_does_hold(tmp_path):
    _dem().to_dataset(name="something_else").to_netcdf(
        tmp_path / plot_map.STATICMAPS_FILENAME
    )
    (tmp_path / plot_map.STATICGEOMS_DIRNAME).mkdir()
    with pytest.raises(KeyError, match="something_else"):
        load_basin_layers(tmp_path)


# --- the layer-in plotting function -------------------------------------------
# `plot_basin_map` takes each map layer as its own argument and returns the
# figure. What needs covering is the OPTIONALITY: every layer but the first
# three may be absent, and an absent layer must drop out of the drawing AND out
# of the legend rather than raising or leaving a dangling entry. The figure's
# appearance is not asserted here -- it is verified by rendering it and looking
# at it (AGENTS.md, "Figures are terminal artifacts").


@pytest.fixture(autouse=True, scope="module")
def _headless_backend():
    """No display in CI, and no interactive window locally."""
    import matplotlib

    matplotlib.use("Agg", force=True)


def _frame(geometries):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"value": list(range(len(geometries)))},
        geometry=list(geometries),
        crs="EPSG:4326",
    )


def _layers():
    """A DEM and one of every vector layer, in one geographic neighbourhood."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point, box

    dem = _dem(nx=12, ny=10, res=0.01)
    rivers = gpd.GeoDataFrame(
        {"strord": [1, 3]},
        geometry=[
            LineString([(0.01, -0.01), (0.05, -0.05)]),
            LineString([(0.05, -0.05), (0.10, -0.08)]),
        ],
        crs="EPSG:4326",
    )
    gauges = gpd.GeoDataFrame(
        {"wflow_id": [101, 102]},
        geometry=[Point(0.05, -0.05), Point(0.09, -0.07)],
        crs="EPSG:4326",
    )
    return dict(
        dem=dem,
        rivers=rivers,
        basin=_frame([box(0.0, -0.09, 0.11, 0.0)]),
        subbasins=_frame([box(0.0, -0.05, 0.11, 0.0), box(0.0, -0.09, 0.11, -0.05)]),
        gauges=gauges,
        outlets=_frame([Point(0.10, -0.08)]),
        lakes=_frame([box(0.02, -0.03, 0.04, -0.01)]),
        reservoirs=_frame([box(0.06, -0.07, 0.08, -0.05)]),
        glaciers=_frame([box(0.01, -0.085, 0.03, -0.065)]),
    )


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


# Asserted through the LABEL_* constants rather than through their current
# wording: what the legend must get right is which entries appear and in what
# order, not the words. Pinning the strings made a copy edit look like a
# regression.
_RIVER = carto.LABEL_RIVER
_BASIN = carto.LABEL_BASIN
_DIVIDES = carto.LABEL_SUBCATCHMENT
_OUTLET = carto.LABEL_OUTLET
_GAUGE = carto.LABEL_GAUGE


def test_the_minimal_call_draws_a_figure_and_writes_nothing(tmp_path):
    """DEM + rivers + basin is the whole requirement; the rest is optional."""
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(layers["dem"], layers["rivers"], layers["basin"])
    try:
        assert fig is not None and ax is not None
        # The basin outline earns an entry too: it is the heaviest line on
        # the figure and went unexplained until 2026-08.
        assert _legend_labels(ax) == [_RIVER, _BASIN]
        assert not list(tmp_path.iterdir())  # saving is the caller's decision
    finally:
        plt.close(fig)


def test_the_source_footnote_is_flush_with_the_panel_not_centred_under_the_page():
    """Centred, a ``supxlabel`` lands under the map-plus-panel midpoint, which is
    an edge nothing on the figure has. The panel's right edge is one the reader
    can see, so the footnote is flushed to it.

    Measured against the LEGEND's own right edge rather than a constant: the
    first version computed the edge from ``_PANEL_LEFT`` plus the panel's
    AVAILABLE width and overshot whatever was actually drawn there.
    """
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(
        layers["dem"], layers["rivers"], layers["basin"], caveat="Source: X (Y, 2020)."
    )
    try:
        footnote = fig._supxlabel
        assert footnote.get_horizontalalignment() == "right"
        renderer = fig.canvas.get_renderer()
        legend_right = ax.get_legend().get_window_extent(renderer).x1
        footnote_right = fig.transFigure.transform((footnote.get_position()[0], 0.0))[0]
        assert footnote_right == pytest.approx(legend_right, abs=2.0)
    finally:
        plt.close(fig)


def test_a_figure_without_a_caveat_grows_no_footnote():
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, _ = plot_basin_map(layers["dem"], layers["rivers"], layers["basin"])
    try:
        assert fig._supxlabel is None
    finally:
        plt.close(fig)


def test_every_layer_together_renders():
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(
        layers.pop("dem"), layers.pop("rivers"), layers.pop("basin"), **layers
    )
    try:
        # A list, not a set: a duplicated entry is exactly the defect the
        # hand-built waterbody patches caused until 2026-08-03.
        # Reading order, not draw order: network, boundaries, points,
        # waterbodies. A list, not a set -- a duplicated entry is exactly the
        # defect the hand-built waterbody patches caused until 2026-08-03.
        assert _legend_labels(ax) == [
            _RIVER,
            _BASIN,
            _DIVIDES,
            _OUTLET,
            _GAUGE,
            "lakes",
            "reservoirs",
            "glaciers",
        ]
    finally:
        plt.close(fig)


def test_omitting_subbasins_drops_the_divides_and_their_legend_entry():
    """The split's whole point: an outline without divides is a valid map."""
    import matplotlib.pyplot as plt

    layers = _layers()
    common = (layers["dem"], layers["rivers"], layers["basin"])
    with_divides, ax_with = plot_basin_map(*common, subbasins=layers["subbasins"])
    without, ax_without = plot_basin_map(*common, subbasins=None)
    try:
        assert _DIVIDES in _legend_labels(ax_with)
        assert _DIVIDES not in _legend_labels(ax_without)
    finally:
        plt.close(with_divides)
        plt.close(without)


def test_an_empty_layer_is_treated_as_an_absent_one():
    """An empty frame must not leave a legend entry for something undrawn."""
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(
        layers["dem"],
        layers["rivers"],
        layers["basin"],
        subbasins=layers["subbasins"].iloc[:0],
        lakes=layers["lakes"].iloc[:0],
        outlets=layers["outlets"].iloc[:0],
    )
    try:
        assert _legend_labels(ax) == [_RIVER, _BASIN]
    finally:
        plt.close(fig)


def test_each_waterbody_enters_on_its_own_argument():
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(
        layers["dem"],
        layers["rivers"],
        layers["basin"],
        reservoirs=layers["reservoirs"],
    )
    try:
        assert _legend_labels(ax) == [_RIVER, _BASIN, "reservoirs"]
    finally:
        plt.close(fig)


def test_gauge_labels_can_be_switched_off_without_dropping_the_markers():
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(
        layers["dem"],
        layers["rivers"],
        layers["basin"],
        gauges=layers["gauges"],
        gauge_label_column=None,
    )
    try:
        assert _GAUGE in _legend_labels(ax)
        assert not [text for text in ax.texts if text.get_text() in {"101", "102"}]
    finally:
        plt.close(fig)


def test_the_returned_figure_is_already_laid_out():
    """One savefig must be enough — the caller does not owe the figure a draw.

    Constrained layout is iterative: before this was settled inside the
    function, the first draw left the y tick labels at x0 < 0, i.e. off the
    canvas, so "0.45°N" saved as "45°N". The workflow path hid it by saving
    twice (PDF then PNG).
    """
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(layers["dem"], layers["rivers"], layers["basin"])
    try:
        assert min(t.get_window_extent().x0 for t in ax.get_yticklabels()) >= 0.0
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("position", "expect_title", "expect_ylabel"),
    [("top", True, False), ("right", False, True)],
)
def test_the_colorbar_label_is_drawn_where_the_position_says(
    position, expect_title, expect_ylabel, restore_colorbar_label_position
):
    """ "top" is a horizontal title above the bar; "right" is the rotated label."""
    import matplotlib.pyplot as plt

    carto.COLORBAR_LABEL_POSITION = position
    layers = _layers()
    fig, ax = plot_basin_map(
        layers["dem"], layers["rivers"], layers["basin"], elevation_label="depth [m]"
    )
    try:
        colorbar_axes = ax.child_axes[0]  # an inset axes, so NOT in fig.axes
        assert (colorbar_axes.get_title(loc="left") == "depth [m]") is expect_title
        assert (colorbar_axes.get_ylabel() == "depth [m]") is expect_ylabel
    finally:
        plt.close(fig)


def test_an_explicit_extent_overrides_the_dem_bounding_box():
    import matplotlib.pyplot as plt

    layers = _layers()
    extent = [0.0, 0.05, -0.04, 0.0]
    fig, ax = plot_basin_map(
        layers["dem"], layers["rivers"], layers["basin"], extent=extent
    )
    try:
        assert list(ax.get_extent(crs=ax.projection)) == pytest.approx(extent)
    finally:
        plt.close(fig)


# =============================================================================
# THE TEMPLATE
# =============================================================================
# Four quantities share one drawing function. What must not regress is the
# handful of rules that differ per quantity, and the guards that stop a wrong
# figure rendering rather than failing.


def _field(values, lat0=0.0):
    """A geographic raster from a 2-D array, as the template expects one."""
    values = np.asarray(values, dtype="float32")
    ny, nx = values.shape
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={
            "latitude": lat0 + np.arange(ny) * 0.01,
            "longitude": np.arange(nx) * 0.01,
        },
    )


def _ramp(low, high, ny=20, nx=20, lat0=0.0):
    return _field(np.tile(np.linspace(low, high, nx), (ny, 1)), lat0=lat0)


def test_every_shipped_style_names_a_resolvable_palette():
    """A typo in a colormap name would surface as a KeyError mid-render."""
    for name, style in RASTER_STYLES.items():
        assert _style_colormap(style, 5).N == 5, name


def test_elevation_stays_linear_while_the_climate_styles_adapt():
    """Owner's call 2026-08-09: elevation is the quantity read by subtraction."""
    assert RASTER_STYLES["elevation"].classification == "equal_interval"
    for name in ("precip", "temp", "pet"):
        assert RASTER_STYLES[name].classification == "auto", name


def test_only_elevation_asks_for_a_hillshade():
    """A hillshade of a rainfall field would render gradients as topography."""
    assert RASTER_STYLES["elevation"].relief is True
    for name in ("precip", "temp", "pet"):
        assert RASTER_STYLES[name].relief is False, name


def test_precipitation_keeps_white_for_genuinely_dry_ground():
    """White reads as DRY. A basin whose driest class is 2725 mm/y has none."""
    style = RASTER_STYLES["precip"]
    at_zero = _style_colormap(style, 6, floor=0.0)(0)
    lifted = _style_colormap(style, 6, floor=2725.0)(0)
    assert sum(lifted[:3]) < sum(at_zero[:3]) - 0.2


def test_a_style_without_a_reserved_low_end_uses_its_whole_ramp():
    style = RasterStyle(label="x", palette="Blues")
    assert _style_colormap(style, 5, floor=999.0)(0) == _style_colormap(style, 5)(0)


def test_temperature_goes_diverging_only_when_it_crosses_freezing():
    """0 degC is a physical midpoint; a basin mean is not."""
    warm = resolve_temperature_style(_ramp(12.0, 28.0))
    assert warm.diverging_center is None
    assert warm.palette == RASTER_STYLES["temp"].palette

    crossing = resolve_temperature_style(_ramp(-8.0, 14.0))
    assert crossing.diverging_center == 0.0
    assert crossing.palette == carto.TEMPERATURE_DIVERGING_PALETTE


def test_a_diverging_temperature_ramp_is_centred_on_zero_not_on_the_data():
    """Centring on the data's own mean is how this rule usually gets broken."""
    assert resolve_temperature_style(_ramp(-2.0, 30.0)).diverging_center == 0.0


def test_the_diverging_style_keeps_the_quantitys_own_step_ladder():
    """Rebuilding a style field by field is how `step_ladder` got dropped, so a
    diverging temperature bar stepped in 0.15 degC off the general ladder."""
    crossing = resolve_temperature_style(_ramp(-8.0, 14.0))
    assert crossing.step_ladder == RASTER_STYLES["temp"].step_ladder


def test_a_declared_midpoint_does_nothing_until_the_field_spans_it():
    """A one-sided field would spend half its palette on values that do not
    occur. The centre is DECLARED on the style and ACTIVATED per raster."""
    style = RasterStyle(
        label="pH", palette="YlGnBu", diverging_at=7.0, diverging_palette="RdYlBu"
    )
    acidic = carto.resolve_diverging_style(_ramp(4.7, 5.5), style)
    assert acidic.diverging_center is None and acidic.palette == "YlGnBu"

    both = carto.resolve_diverging_style(_ramp(4.2, 8.9), style)
    assert both.diverging_center == 7.0 and both.palette == "RdYlBu"


def test_resolving_an_already_resolved_style_changes_nothing():
    """`plot_raster_map` resolves unconditionally; climate_figures resolves too."""
    style = RASTER_STYLES["temp"]
    once = carto.resolve_diverging_style(_ramp(-8.0, 14.0), style)
    twice = carto.resolve_diverging_style(_ramp(-8.0, 14.0), once)
    assert (twice.palette, twice.diverging_center) == (
        once.palette,
        once.diverging_center,
    )


# --- diverging class boundaries ----------------------------------------------
# The whole encoding rests on ONE invariant: the centre is a class BOUNDARY. A
# diverging palette's pale middle sits at the join between two classes, so a
# centre that is not that join puts the pale colour somewhere else -- and the
# map then says freezing is at 3 degC while looking perfectly reasonable.


@pytest.mark.parametrize(
    "low, high, centre, ladder",
    [
        (-8.0, 14.0, 0.0, (2.5, 5.0, 10.0)),
        (-2.0, 30.0, 0.0, (2.5, 5.0, 10.0)),  # barely crosses
        (-40.0, 5.0, 0.0, (2.5, 5.0, 10.0)),  # barely crosses, other side
        (4.2, 8.9, 7.0, None),  # centre is not a step multiple
        (6.93, 7.04, 7.0, None),  # a hair either side
        (3.1, 9.7, 7.0, None),
    ],
)
def test_the_centre_is_always_a_class_boundary(low, high, centre, ladder):
    levels = carto._diverging_levels(low, high, centre, ladder)
    assert np.isclose(levels, centre).any()


@pytest.mark.parametrize(
    "low, high, centre, ladder",
    [(-8.0, 14.0, 0.0, (2.5, 5.0, 10.0)), (4.2, 8.9, 7.0, None), (3.1, 9.7, 7.0, None)],
)
def test_the_diverging_boundaries_cover_the_data(low, high, centre, ladder):
    """Not symmetric about the centre — deliberately. Forcing that gave a
    -50..+50 four-class bar for a field spanning -2..30, which is a worse figure
    than the sequential one it replaced. Symmetry lives in the COLOUR domain.
    """
    levels = carto._diverging_levels(low, high, centre, ladder)
    assert levels[0] <= low and levels[-1] >= high


def test_the_diverging_tick_cap_is_enforced():
    assert len(carto._diverging_levels(-400.0, 900.0, 0.0)) <= carto._COLORBAR_MAX_TICKS


def test_the_pale_middle_of_a_diverging_ramp_lands_on_the_centre():
    """The two classes touching the centre must sample either side of the
    ramp's midpoint. If they do not, the pale colour is somewhere else.
    """
    style = RASTER_STYLES["temp"].replace(
        palette=carto.TEMPERATURE_DIVERGING_PALETTE, diverging_center=0.0
    )
    levels = carto._diverging_levels(-8.0, 14.0, 0.0, style.step_ladder)
    cmap = carto._diverging_colormap(style, levels, "neither")
    centre_index = int(np.argmin(np.abs(np.asarray(levels) - 0.0)))
    below = np.array(cmap(centre_index - 1)[:3])
    above = np.array(cmap(centre_index)[:3])
    # RdBu_r runs blue -> red, so the class below freezing is the BLUER of the
    # two and the one above is the redder. Both pale, being next to the join.
    assert below[2] > below[0], "sub-zero class should be the blue side"
    assert above[0] > above[2], "above-zero class should be the red side"
    # Pale means close to the ramp's own midpoint colour.
    midpoint = np.array(carto._style_colormap(style)(0.5)[:3])
    assert np.abs(below - midpoint).max() < 0.35
    assert np.abs(above - midpoint).max() < 0.35


def test_a_one_sided_field_never_reaches_the_diverging_classifier():
    """The sequential path must be untouched: elevation and precipitation draw
    through the equal-interval / equal-area rules this branch sits in front of.
    """
    warm = resolve_temperature_style(_ramp(12.0, 28.0))
    assert warm.diverging_center is None
    levels = _class_levels(_ramp(12.0, 28.0), warm)
    assert not np.isclose(levels, 0.0).any()


# --- classification ----------------------------------------------------------


@pytest.mark.parametrize(
    "low, high",
    [(1.5, 140.0), (1900.0, 4200.0), (0.0, 12.0), (0.2, 1400.0), (0.0, 819.0)],
)
def test_the_tick_cap_is_enforced_by_widening_the_step(low, high):
    """The target is a wish; the cap is enforced.

    Both exist because rounding a class width to a readable number means the
    count cannot be requested exactly.
    """
    assert len(_equal_interval_levels(low, high)) <= carto._COLORBAR_MAX_TICKS


def test_a_skewed_field_switches_to_equal_area_under_auto():
    """The transform exists for exactly this: most of the data in one class."""
    skewed = _field(np.tile(np.linspace(0.0, 10.0, 20) ** 3, (20, 1)))
    auto = RasterStyle(label="x", palette="Blues", classification="auto")
    linear = RasterStyle(label="x", palette="Blues", classification="equal_interval")
    values, weights = _finite_cells(skewed)
    modal = {
        name: carto._class_area_shares(
            values, _class_levels(skewed, style), weights
        ).max()
        for name, style in (("auto", auto), ("linear", linear))
    }
    assert modal["auto"] < modal["linear"]


def test_equal_interval_is_kept_when_it_is_already_working():
    """auto is not always-equal-area: a uniform field keeps the linear bar."""
    flat = _ramp(0.0, 100.0)
    auto = RasterStyle(label="x", palette="Blues", classification="auto")
    linear = RasterStyle(label="x", palette="Blues", classification="equal_interval")
    assert np.allclose(_class_levels(flat, auto), _class_levels(flat, linear))


def test_a_field_with_two_distinct_values_falls_back_rather_than_collapsing():
    """The wflow forcing on a 2x3 reanalysis grid is exactly this input."""
    values = np.where(np.arange(400).reshape(20, 20) < 200, 2727.0, 2820.0)
    style = RasterStyle(label="x", palette="Blues", classification="auto")
    levels = _class_levels(_field(values), style)
    assert len(levels) - 1 >= carto._MIN_CLASSES
    assert np.all(np.diff(levels) > 0)


def test_class_shares_are_weighted_by_area_not_by_cell_count():
    """cos(latitude): a cell at 60 deg is half the ground area of one at 0."""
    _, weights = _finite_cells(_ramp(0.0, 1.0, ny=40, lat0=20.0))
    assert weights.max() > weights.min()


def test_weighted_quantiles_reduce_to_plain_ones_under_equal_weights():
    values = np.linspace(0.0, 100.0, 501)
    probabilities = [0.1, 0.5, 0.9]
    assert np.allclose(
        _weighted_quantiles(values, np.ones_like(values), probabilities),
        np.quantile(values, probabilities),
        atol=0.3,
    )


# --- guards ------------------------------------------------------------------


def test_a_projected_vector_layer_is_refused_not_silently_drawn():
    """The scale bar and the hillshade convert DEGREES to metres.

    A projected layer renders a plausible map whose scale bar is wrong by five
    orders of magnitude, which is worse than a figure that fails.
    """
    import geopandas as gpd
    from shapely.geometry import box

    layer = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_crs(3857)
    with pytest.raises(ValueError, match="EPSG:4326"):
        check_geographic_inputs(_dem(), {"rivers": layer})


def test_a_projected_grid_is_refused():
    dem = _dem()
    with pytest.raises(ValueError, match="geographic"):
        check_geographic_inputs(dem.assign_coords(longitude=dem["longitude"] * 1e5), {})


def test_a_unit_mismatch_warns_rather_than_refusing():
    """A DEM in feet is a valid map once the caller's label says so."""
    dem = _dem()
    dem.attrs["units"] = "ft"
    with pytest.warns(UserWarning, match="units"):
        check_geographic_inputs(dem, {}, "Elevation (m a.s.l.)")


def test_a_caller_that_owns_its_units_can_opt_out_of_the_check():
    """The climate fields are derived aggregates; the raster's attrs say nothing."""
    import warnings as _warnings

    dem = _dem()
    dem.attrs["units"] = "ft"
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        check_geographic_inputs(dem, {}, "Precipitation (mm)", expected_units=())


# --- overlays over any raster ------------------------------------------------


def test_overlay_contrast_flips_between_a_pale_and_a_dark_raster():
    """A dark line over a dark fill leaves the halo doing all the work.

    The divides then render as white rails with a hollow core, which is the
    inverse of the intended styling.
    """
    style = RasterStyle(label="x", palette="Blues", classification="equal_interval")
    mostly_dark = _field(np.full((20, 20), 99.0) - (np.arange(400).reshape(20, 20) < 8))
    mostly_pale = _field(np.full((20, 20), 1.0) + (np.arange(400).reshape(20, 20) < 8))
    dark_line, dark_halo = _overlay_contrast(mostly_dark, style)
    pale_line, pale_halo = _overlay_contrast(mostly_pale, style)
    assert (dark_line, dark_halo) == (pale_halo, pale_line)


def test_divides_are_merged_so_a_shared_edge_is_drawn_once():
    """Two dashed strokes at one place interleave into an apparent solid line."""
    import geopandas as gpd
    from shapely.geometry import box

    touching = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(0, 1, 1, 2)], crs="EPSG:4326"
    )
    merged = _divide_linework(touching)
    assert len(merged) == 1
    # Outer ring of the union (6) plus the internal divide counted ONCE (1).
    # Two un-merged rings would total 8.
    assert merged.geometry.iloc[0].length == pytest.approx(7.0)


# --- locator window ----------------------------------------------------------


def test_the_locator_window_is_sized_from_the_basin(restore_locator):
    """A fixed span cannot serve two basins of different sizes."""
    carto._LOCATOR_SPAN_DEG = "auto"
    small = _locator_span(np.array([0.0, 0.25, 0.0, 0.2]))
    large = _locator_span(np.array([0.0, 6.0, 0.0, 5.0]))
    assert small < large
    assert small in carto._LOCATOR_SPAN_LADDER


def test_a_pinned_locator_span_overrides_the_auto_rule(restore_locator):
    carto._LOCATOR_SPAN_DEG = 3.0
    assert _locator_span(np.array([0.0, 0.25, 0.0, 0.2])) == 3.0


def test_a_basin_always_fits_inside_its_own_locator_window(restore_locator):
    """The target alone would choose a window narrower than a large basin."""
    carto._LOCATOR_SPAN_DEG = "auto"
    extent = np.array([0.0, 30.0, 0.0, 25.0])
    assert 2.0 * _locator_span(extent) >= (extent[1] - extent[0])


# --- the whole figure, for a non-elevation quantity --------------------------


def test_a_climate_style_renders_through_the_same_function():
    """The template's reason to exist: one figure function, four quantities."""
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_raster_map(
        layers["dem"],
        layers["rivers"],
        layers["basin"],
        subbasins=layers["subbasins"],
        style=RASTER_STYLES["precip"],
    )
    try:
        assert _legend_labels(ax) == [_RIVER, _BASIN, _DIVIDES]
    finally:
        plt.close(fig)


def test_a_long_bar_title_wraps_to_the_panel_width():
    """Most quantity names with their units are wider than a 40 mm panel."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 4.0))
    try:
        wrapped = _wrap_label(fig, "Potential evaporation (mm per year)", 0.8, 7.0)
        assert "\n" in wrapped
        assert wrapped.replace("\n", " ") == "Potential evaporation (mm per year)"
    finally:
        plt.close(fig)


def test_a_single_word_title_is_never_wrapped():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 4.0))
    try:
        assert _wrap_label(fig, "Elevation", 0.01, 7.0) == "Elevation"
    finally:
        plt.close(fig)
