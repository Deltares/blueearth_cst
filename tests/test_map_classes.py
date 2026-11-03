"""Unit tests for the map classifier, relief, extent, locator and input guards.

Moved with the code from ``test_plot_map.py`` when ``shared/cartographic_map.py``
was retired (2026-10-29); the functions now live in ``shared/map_classes.py``.
Tests of the retired drawing (scale bar, figure sizing, side panel, title
wrapping, legacy styles) went with it.
"""

import numpy as np
import pytest
import xarray as xr
from matplotlib import colors

from blueearth_cst.shared import map_classes as classes
from blueearth_cst.shared.map_classes import (
    _class_levels,
    _elevation_colormap,
    _equal_interval_levels,
    _finite_cells,
    _locator_span,
    _mask_nodata,
    _metres_per_degree,
    _nice_step_up,
    _style_colormap,
    _weighted_quantiles,
    check_geographic_inputs,
    extent_from_layer,
    pixel_resolution,
    spatial_dim_names,
)
from blueearth_cst.shared.plot_spatial_maps import ELEVATION_STYLE
from blueearth_cst.shared.raster_style import RasterStyle


def _elevation_classes(dem):
    """Class boundaries of the elevation ramp, as the basin overview draws it."""
    return _class_levels(dem, ELEVATION_STYLE)


@pytest.fixture
def restore_locator():
    """Put back the locator span a test pins."""
    original = classes._LOCATOR_SPAN_DEG
    yield
    classes._LOCATOR_SPAN_DEG = original


def _dem(x_name="longitude", y_name="latitude", res=0.01, nx=4, ny=3):
    """A minimal north-up geographic DEM, as wflow writes them."""
    x = np.arange(nx) * res
    y = -np.arange(ny) * res
    return xr.DataArray(
        np.arange(ny * nx, dtype="float32").reshape(ny, nx),
        dims=(y_name, x_name),
        coords={y_name: y, x_name: x},
    )


def _basins_frame(*boxes):
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"value": list(range(len(boxes)))},
        geometry=[box(*bounds) for bounds in boxes],
        crs="EPSG:4326",
    )


def _dem_spanning(low, high):
    values = np.linspace(low, high, 400)
    side = 20
    return xr.DataArray(values.reshape(side, side), dims=("latitude", "longitude"))


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


def test_degrees_are_square_at_the_equator():
    metres_lon, metres_lat = _metres_per_degree(0.0)
    assert metres_lon == pytest.approx(metres_lat)


@pytest.mark.parametrize("latitude, ratio", [(60.0, 0.5), (45.0, 0.7071), (0.0, 1.0)])
def test_longitude_degree_shrinks_with_the_cosine_of_latitude(latitude, ratio):
    metres_lon, metres_lat = _metres_per_degree(latitude)
    assert metres_lon / metres_lat == pytest.approx(ratio, abs=1e-4)


def test_latitude_degree_does_not_vary():
    assert _metres_per_degree(0.0)[1] == pytest.approx(_metres_per_degree(70.0)[1])


def test_the_locator_window_is_square_and_centred_on_the_basin():
    window = classes._locator_window([9.65, 9.86, 0.35, 0.50])
    assert (window[1] - window[0]) == pytest.approx(window[3] - window[2])
    assert 0.5 * (window[0] + window[1]) == pytest.approx(9.755)


def test_a_polar_basin_gets_a_full_window_rather_than_half_of_one():
    """Clipping at the pole would leave the inset half empty; re-centre instead."""
    window = classes._locator_window([20.0, 21.0, 88.0, 89.5])
    assert (window[1] - window[0]) == pytest.approx(window[3] - window[2])
    assert window[3] <= 90.0


def test_the_vendored_basemap_is_present():
    assert classes.BASEMAP_PATH.is_file(), (
        f"{classes.BASEMAP_PATH} is missing; see config/basemap/README.md"
    )


@pytest.mark.parametrize("layer", ["land", "borders", "places"])
def test_each_basemap_layer_is_readable_and_not_empty(layer):
    import geopandas as gpd

    assert len(gpd.read_file(classes.BASEMAP_PATH, layer=layer)) > 0


def test_the_places_layer_carries_the_columns_the_inset_filters_on():
    import geopandas as gpd

    places = gpd.read_file(classes.BASEMAP_PATH, layer="places")
    assert {"name", "pop_max", "scalerank"} <= set(places.columns)


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


def test_a_lowland_basin_gets_a_ramp_starting_at_zero():
    levels = _elevation_classes(_dem_spanning(4.0, 137.0))
    assert levels[0] == 0.0
    assert levels[-1] >= 137.0


def test_a_plateau_basin_keeps_its_own_baseline():
    """Zeroed, a 1900-1960 m basin lands entirely in ONE class and goes flat."""
    levels = _elevation_classes(_dem_spanning(1900.0, 1960.0))
    assert levels[0] > 0.0
    # the basin must span most of the ramp, not a single band of it
    assert (1960.0 - 1900.0) / (levels[-1] - levels[0]) > 0.5


def test_a_mountain_basin_still_zeroes_because_it_can_afford_to():
    levels = _elevation_classes(_dem_spanning(800.0, 6200.0))
    assert levels[0] == 0.0


def test_the_top_class_contains_the_highest_cell_rather_than_ending_at_it():
    """A boundary exactly at the summit renders the summit as nodata."""
    assert _elevation_classes(_dem_spanning(0.0, 100.0))[-1] >= 100.0


def test_boundaries_sit_on_multiples_of_the_class_width():
    """A floor of 1903 m must label as 1900, not carry itself up the bar."""
    levels = _elevation_classes(_dem_spanning(1903.0, 2410.0))
    step = levels[1] - levels[0]
    assert np.allclose(np.mod(levels, step), 0.0)


def test_classes_are_evenly_spaced_and_increasing():
    levels = _elevation_classes(_dem_spanning(4.0, 137.0))
    assert np.all(np.diff(levels) > 0)
    assert np.allclose(np.diff(levels), levels[1] - levels[0])


def test_a_flat_dem_does_not_produce_a_degenerate_ramp():
    levels = _elevation_classes(_dem_spanning(12.0, 12.0))
    assert len(levels) >= 2 and np.all(np.diff(levels) > 0)


def test_a_below_sea_level_basin_is_not_clipped_at_zero():
    levels = _elevation_classes(_dem_spanning(-30.0, 90.0))
    assert levels[0] <= -30.0


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


def test_layer_extent_uses_the_layer_size_for_proportional_padding():
    basin = _basins_frame((0.0, 0.0, 4.0, 1.0))

    np.testing.assert_allclose(
        extent_from_layer(basin),
        [-0.2, 4.2, -0.05, 1.05],
    )


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


def test_a_style_without_a_reserved_low_end_uses_its_whole_ramp():
    style = RasterStyle(label="x", palette="Blues")
    assert _style_colormap(style, 5, floor=999.0)(0) == _style_colormap(style, 5)(0)


def test_a_declared_midpoint_does_nothing_until_the_field_spans_it():
    """A one-sided field would spend half its palette on values that do not
    occur. The centre is DECLARED on the style and ACTIVATED per raster."""
    style = RasterStyle(
        label="pH", palette="YlGnBu", diverging_at=7.0, diverging_palette="RdYlBu"
    )
    acidic = classes.resolve_diverging_style(_ramp(4.7, 5.5), style)
    assert acidic.diverging_center is None and acidic.palette == "YlGnBu"

    both = classes.resolve_diverging_style(_ramp(4.2, 8.9), style)
    assert both.diverging_center == 7.0 and both.palette == "RdYlBu"


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
def test_diverging_boundaries_include_the_centre_and_cover_the_data(
    low, high, centre, ladder
):
    levels = classes._diverging_levels(low, high, centre, ladder)
    assert np.isclose(levels, centre).any()
    # Coverage need not be symmetric: symmetry belongs to the colour domain.
    assert levels[0] <= low and levels[-1] >= high


def test_the_diverging_tick_cap_is_enforced():
    assert (
        len(classes._diverging_levels(-400.0, 900.0, 0.0))
        <= classes._COLORBAR_MAX_TICKS
    )


@pytest.mark.parametrize(
    "low, high",
    [(1.5, 140.0), (1900.0, 4200.0), (0.0, 12.0), (0.2, 1400.0), (0.0, 819.0)],
)
def test_the_tick_cap_is_enforced_by_widening_the_step(low, high):
    """The target is a wish; the cap is enforced.

    Both exist because rounding a class width to a readable number means the
    count cannot be requested exactly.
    """
    assert len(_equal_interval_levels(low, high)) <= classes._COLORBAR_MAX_TICKS


def test_a_skewed_field_switches_to_equal_area_under_auto():
    """The transform exists for exactly this: most of the data in one class."""
    skewed = _field(np.tile(np.linspace(0.0, 10.0, 20) ** 3, (20, 1)))
    auto = RasterStyle(label="x", palette="Blues", classification="auto")
    linear = RasterStyle(label="x", palette="Blues", classification="equal_interval")
    values, weights = _finite_cells(skewed)
    modal = {
        name: classes._class_area_shares(
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
    assert len(levels) - 1 >= classes._MIN_CLASSES
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


def test_the_locator_window_is_sized_from_the_basin(restore_locator):
    """A fixed span cannot serve two basins of different sizes."""
    classes._LOCATOR_SPAN_DEG = "auto"
    small = _locator_span(np.array([0.0, 0.25, 0.0, 0.2]))
    large = _locator_span(np.array([0.0, 6.0, 0.0, 5.0]))
    assert small < large
    assert small in classes._LOCATOR_SPAN_LADDER


def test_a_pinned_locator_span_overrides_the_auto_rule(restore_locator):
    classes._LOCATOR_SPAN_DEG = 3.0
    assert _locator_span(np.array([0.0, 0.25, 0.0, 0.2])) == 3.0


def test_a_basin_always_fits_inside_its_own_locator_window(restore_locator):
    """The target alone would choose a window narrower than a large basin."""
    classes._LOCATOR_SPAN_DEG = "auto"
    extent = np.array([0.0, 30.0, 0.0, 25.0])
    assert 2.0 * _locator_span(extent) >= (extent[1] - extent[0])
