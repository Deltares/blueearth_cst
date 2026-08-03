# -*- coding: utf-8 -*-
"""Unit tests for the basin-map figure geometry and colour ramp (rule 1.12).

The figure itself is only rendered by a full WF1 run, so what is testable here
is the arithmetic that decides whether it is CORRECT rather than merely
produced: the latitude correction, the per-basin exaggeration, the declared
physical width, and the ramp's accessibility property. Each of these was a real
defect in the pre-2026-08 figure.
"""

import numpy as np
import pytest
import xarray as xr
from matplotlib import colors

from blueearth_cst.shared import plot_map
from blueearth_cst.shared.plot_map import (
    FIGURE_WIDTH_MM,
    MM_PER_INCH,
    _CORNERS,
    _EXTENT_BUFFER_DEG,
    _NORTH_ARROW_CORNER,
    RIVER_WIDTH_UNIFORM,
    _basin_outline,
    _colorbar_inset,
    _corner_occupancy,
    _elevation_colormap,
    _elevation_levels,
    _figure_size,
    _nice_step_up,
    _mask_nodata,
    _publication_rc,
    _scale_bar_corner,
    _metres_per_degree,
    _nice_round_length,
    _river_linewidths,
    load_basin_layers,
    map_extent,
    pixel_resolution,
    plot_basin_map,
    spatial_dim_names,
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
    four = _basins_frame(
        (0, 0, 1, 1), (1, 0, 2, 1), (0, 1, 1, 2), (1, 1, 2, 2)
    )
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
    "value, expected", [(0.9, 1.0), (1.0, 1.0), (1.7, 2.0), (3.0, 5.0), (16.9, 20.0)]
)
def test_a_class_width_rounds_UP_to_one_two_or_five(value, expected):
    """Down would give MORE classes than asked for -- the opposite of the point."""
    assert _nice_step_up(value) == expected


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
    original = {name: getattr(plot_map, name) for name in names}
    yield
    for name, value in original.items():
        setattr(plot_map, name, value)


def test_rcparams_follow_a_font_size_override(restore_tunables):
    plot_map.FONT_SIZE_TICK = 99.0
    plot_map.WIDTH_AXES_SPINE = 3.0
    assert _publication_rc()["xtick.labelsize"] == 99.0
    assert _publication_rc()["ytick.labelsize"] == 99.0
    assert _publication_rc()["axes.linewidth"] == 3.0


def test_colorbar_inset_follows_the_panel_position(restore_tunables):
    plot_map._PANEL_LEFT = 1.5
    plot_map._COLORBAR_WIDTH = 0.1
    left, _, width, _ = _colorbar_inset()
    assert (left, width) == (1.5, 0.1)


@pytest.fixture
def restore_colorbar_label_position():
    original = plot_map.COLORBAR_LABEL_POSITION
    yield
    plot_map.COLORBAR_LABEL_POSITION = original


def test_a_right_label_leaves_the_colorbar_at_full_height(
    restore_colorbar_label_position,
):
    plot_map.COLORBAR_LABEL_POSITION = "right"
    assert _colorbar_inset()[3] == pytest.approx(plot_map._COLORBAR_HEIGHT)


def test_a_top_label_shortens_the_bar_to_make_room(restore_colorbar_label_position):
    """Without this the bar reaches 1.0 and the label renders off the canvas."""
    plot_map.COLORBAR_LABEL_POSITION = "top"
    assert _colorbar_inset()[3] == pytest.approx(
        plot_map._COLORBAR_HEIGHT - plot_map._COLORBAR_TOP_LABEL_HEADROOM
    )


def test_each_extra_label_line_costs_the_bar_the_same_again(
    restore_colorbar_label_position,
):
    """A fixed headroom would clip a two-line label or gap above a one-line one."""
    plot_map.COLORBAR_LABEL_POSITION = "top"
    one, two = _colorbar_inset(1)[3], _colorbar_inset(2)[3]
    assert one - two == pytest.approx(plot_map._COLORBAR_TOP_LABEL_HEADROOM)


def test_label_lines_do_not_shorten_a_right_hand_label(
    restore_colorbar_label_position,
):
    plot_map.COLORBAR_LABEL_POSITION = "right"
    assert _colorbar_inset(1)[3] == _colorbar_inset(3)[3]


def test_an_unknown_label_position_raises_rather_than_silently_defaulting(
    restore_colorbar_label_position,
):
    """A knob that reads as set but does nothing is the worst failure here."""
    plot_map.COLORBAR_LABEL_POSITION = "above"
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


def test_the_minimal_call_draws_a_figure_and_writes_nothing(tmp_path):
    """DEM + rivers + basin is the whole requirement; the rest is optional."""
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(layers["dem"], layers["rivers"], layers["basin"])
    try:
        assert fig is not None and ax is not None
        assert _legend_labels(ax) == ["river"]
        assert not list(tmp_path.iterdir())  # saving is the caller's decision
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
        assert _legend_labels(ax) == [
            "river",
            "outlets",
            "output locs",
            "subcatchments",
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
        assert "subcatchments" in _legend_labels(ax_with)
        assert "subcatchments" not in _legend_labels(ax_without)
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
        assert _legend_labels(ax) == ["river"]
    finally:
        plt.close(fig)


def test_each_waterbody_enters_on_its_own_argument():
    import matplotlib.pyplot as plt

    layers = _layers()
    fig, ax = plot_basin_map(
        layers["dem"], layers["rivers"], layers["basin"], reservoirs=layers["reservoirs"]
    )
    try:
        assert _legend_labels(ax) == ["river", "reservoirs"]
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
        assert "output locs" in _legend_labels(ax)
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
    """"top" is a horizontal title above the bar; "right" is the rotated label."""
    import matplotlib.pyplot as plt

    plot_map.COLORBAR_LABEL_POSITION = position
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
