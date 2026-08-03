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
from matplotlib import colors

from blueearth_cst.shared.plot_map import (
    FIGURE_WIDTH_MM,
    MM_PER_INCH,
    _CORNERS,
    _add_north_arrow,
    _basin_outline,
    _coordinate_format,
    _corner_occupancy,
    _elevation_colormap,
    _figure_size,
    _furniture_corners,
    _graticule_ticks,
    _metres_per_degree,
    _nice_round_length,
    _river_linewidths,
    _vertical_exaggeration,
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


def test_furniture_avoids_the_occupied_corner():
    """A basin in the south-west must not get the legend on top of it."""
    basin = _basin_covering(0.0, 0.0, 0.45, 0.45)
    legend_loc, bar_corner = _furniture_corners(basin, _UNIT_EXTENT)
    assert legend_loc != "lower left"
    assert bar_corner != "lower left"
    assert bar_corner != legend_loc


def test_scale_bar_prefers_a_lower_corner():
    basin = _basin_covering(0.0, 0.0, 0.45, 0.45)
    _, bar_corner = _furniture_corners(basin, _UNIT_EXTENT)
    assert bar_corner.startswith("lower")


def test_placement_is_deterministic_for_a_symmetric_basin():
    """Ties must not make the figure depend on set iteration order."""
    basin = _basin_covering(0.35, 0.35, 0.65, 0.65)  # touches no corner
    assert _furniture_corners(basin, _UNIT_EXTENT) == _furniture_corners(
        basin, _UNIT_EXTENT
    )


def test_a_basin_filling_its_box_still_yields_valid_corners():
    legend_loc, bar_corner = _furniture_corners(
        _basin_covering(0, 0, 1, 1), _UNIT_EXTENT
    )
    assert legend_loc in _CORNERS and bar_corner in _CORNERS


def test_north_arrow_moves_when_the_legend_takes_the_top_right():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = {}
    for legend_loc in ("upper right", "lower right"):
        _, ax = plt.subplots()
        _add_north_arrow(ax, legend_loc)
        positions[legend_loc] = ax.texts[-1].get_position()[0]
        plt.close("all")
    assert positions["upper right"] < 0.5 < positions["lower right"]


# --- hillshade exaggeration ---------------------------------------------------
# A fixed factor renders a lowland basin featureless or an alpine one blown out;
# CST runs both from this one function.


def _ramp_dem(total_relief_m, cells=64):
    row = np.linspace(0.0, total_relief_m, cells)
    return np.tile(row, (cells, 1))


def test_flat_basin_gets_a_large_exaggeration():
    dem = _ramp_dem(130.0)  # 130 m over 24 km, the test fixture's basin
    assert _vertical_exaggeration(dem, 375.0, 375.0) > 10.0


def test_mountainous_basin_is_not_blown_out():
    dem = _ramp_dem(3000.0)
    assert _vertical_exaggeration(dem, 375.0, 375.0) < 5.0


def test_exaggeration_is_bounded_and_finite_on_a_perfectly_flat_dem():
    exaggeration = _vertical_exaggeration(np.zeros((32, 32)), 375.0, 375.0)
    assert np.isfinite(exaggeration) and exaggeration >= 1.0


def test_steeper_terrain_never_gets_more_exaggeration_than_flatter():
    steep = _vertical_exaggeration(_ramp_dem(2000.0), 375.0, 375.0)
    gentle = _vertical_exaggeration(_ramp_dem(50.0), 375.0, 375.0)
    assert gentle >= steep


def test_nodata_padding_does_not_change_the_exaggeration():
    """The same basin must shade the same however much nodata its box holds.

    Slope is measured inside the basin only; without the mask, the flat fill
    outside it drags the percentile down and the exaggeration up, so a basin
    that happens to sit in a roomier bounding box would shade differently.
    """
    dem = _ramp_dem(500.0, cells=48)
    inside = np.ones_like(dem, dtype=bool)

    padded = np.full((96, 96), float(dem.min()))
    padded[:48, :48] = dem
    padded_inside = np.zeros_like(padded, dtype=bool)
    padded_inside[:48, :48] = True

    tight = _vertical_exaggeration(dem, 375.0, 375.0, inside)
    roomy = _vertical_exaggeration(padded, 375.0, 375.0, padded_inside)
    assert tight == pytest.approx(roomy, rel=0.05)


def test_a_fully_masked_dem_falls_back_rather_than_raising():
    dem = _ramp_dem(500.0)
    assert _vertical_exaggeration(dem, 375.0, 375.0, np.zeros_like(dem, bool)) == 1.0


# --- the colour ramp ----------------------------------------------------------
# `plt.cm.terrain`, which this replaced, is neither greyscale- nor CVD-safe.


def test_elevation_ramp_luminance_is_strictly_monotonic():
    samples = _elevation_colormap()(np.linspace(0.0, 1.0, 64))[:, :3]
    luminance = samples @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luminance) < 0.0), "ramp must survive greyscale printing"


def test_elevation_ramp_spans_a_usable_luminance_range():
    samples = _elevation_colormap()(np.linspace(0.0, 1.0, 64))[:, :3]
    luminance = samples @ np.array([0.2126, 0.7152, 0.0722])
    assert luminance.max() - luminance.min() > 0.5


def test_elevation_ramp_is_not_matplotlib_terrain():
    import matplotlib.pyplot as plt

    ours = _elevation_colormap()(np.linspace(0.0, 1.0, 16))
    terrain = plt.cm.terrain(np.linspace(0.0, 1.0, 16))
    assert not np.allclose(ours, terrain)


# --- axis labelling -----------------------------------------------------------


@pytest.mark.parametrize(
    "span, expected", [(0.2, ".2f"), (0.9, ".2f"), (3.0, ".1f"), (12.0, ".0f")]
)
def test_coordinate_precision_follows_the_basin_span(span, expected):
    assert _coordinate_format(span) == expected


def test_graticule_ticks_stay_inside_the_extent():
    extent = (9.646, 9.872, 0.330, 0.503)
    lon_ticks, lat_ticks = _graticule_ticks(extent)
    assert lon_ticks and lat_ticks
    assert all(extent[0] <= t <= extent[1] for t in lon_ticks)
    assert all(extent[2] <= t <= extent[3] for t in lat_ticks)


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

    def __init__(self, orders):
        self._orders = np.asarray(orders, dtype=float)

    def __getitem__(self, key):
        assert key == "strord"
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


# --- the figure is still colour-independent -----------------------------------


def test_ramp_endpoints_are_distinguishable_under_dichromacy():
    """Deuteranope-simulated endpoints must still differ in luminance.

    A cheap proxy for the real check: collapse the red-green channel pair, which
    is what deuteranopia does, and confirm the ends have not converged.
    """
    ramp = _elevation_colormap()
    low, high = np.array(ramp(0.0)[:3]), np.array(ramp(1.0)[:3])
    collapse = lambda rgb: np.array([rgb[:2].mean(), rgb[:2].mean(), rgb[2]])
    assert np.abs(collapse(low) - collapse(high)).max() > 0.3


def test_colormap_is_a_matplotlib_colormap():
    assert isinstance(_elevation_colormap(), colors.Colormap)
