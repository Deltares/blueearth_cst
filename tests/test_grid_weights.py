"""Grid-geometry tests for step 5a (design D10).

Each test names the falsifier it discharges from
``dev/r08/2026-07-30_wf2-5a-falsifier.md``, which was written before any 5a
code existed. The point of that ordering is that these can fail for the right
reason: a test that merely exercises the code confirms it runs, while a test tied
to a stated falsifier confirms a *claim*.
"""

import numpy as np
import pytest

from blueearth_cst.projections.grid_weights import (
    WEIGHTING_SCHEME,
    GridGeometryError,
    cell_area_weights,
    check_axis,
    latitude_weights,
    longitude_weights,
    midpoint_edges,
)


# --- F1: strict generalization -- uniform grid must reproduce cos(lat) ---------


@pytest.mark.parametrize("step", [0.5, 1.0, 2.5, 5.0])
def test_F1_uniform_grid_weights_are_proportional_to_cos_latitude(step):
    """The claim that lets D10 REPLACE cos-lat rather than compete with it.

    sin(phi+d/2) - sin(phi-d/2) = 2*sin(d/2)*cos(phi); the constant cancels in a
    weighted mean. A failure here is an edge/extrapolation bug, and would mean
    every uniform-grid diff attributed to "weighting" is attributing a bug.
    """
    lat = np.arange(-60.0, 60.0 + step, step)
    w = latitude_weights(lat)
    cos = np.cos(np.deg2rad(lat))
    np.testing.assert_allclose(w / w.sum(), cos / cos.sum(), rtol=1e-12)


# --- F2: on a NON-uniform grid it must NOT reduce to cos(lat) ------------------


def test_F2_gaussian_like_grid_weights_differ_from_cos_latitude():
    """If these matched, the scheme ext2-02 rejected was reimplemented.

    Gaussian latitudes are unevenly spaced, so the per-cell latitude WIDTH varies
    -- the term cos(lat) alone omits. The weights must reflect it.
    """
    # Uneven, strictly monotonic, denser near the equator (Gaussian-like).
    lat = np.array([-70.0, -45.0, -25.0, -10.0, -3.0, 3.0, 10.0, 25.0, 45.0, 70.0])
    w = latitude_weights(lat)
    cos = np.cos(np.deg2rad(lat))

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(w / w.sum(), cos / cos.sum(), rtol=1e-3)

    # And the difference is substantial, not floating-point noise: the wide polar
    # cells gain weight relative to the narrow equatorial ones.
    assert np.max(np.abs(w / w.sum() - cos / cos.sum())) > 0.01


# --- F3: the weights must be an AREA, checked without trusting the formula -----


def test_F3_global_grid_weights_sum_to_the_area_of_the_sphere():
    """The one check that tests the boundary extrapolation rather than the algebra.

    A grid partitioning the sphere must have cell areas summing to 4*pi
    steradians. An interior-only check cannot see a mis-sized boundary cell.
    """
    lat = np.arange(-88.75, 90.0, 2.5)      # centers; edges land exactly on +-90
    lon = np.arange(-179.375, 180.0, 1.25)  # centers spanning the full circle
    total = cell_area_weights(lat, lon).sum()
    np.testing.assert_allclose(total, 4.0 * np.pi, rtol=1e-12)


def test_F3_boundary_cells_use_symmetric_extrapolation():
    centers = np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(midpoint_edges(centers), [5.0, 15.0, 25.0, 35.0])


def test_F3_latitude_edges_are_clamped_at_the_pole():
    """Extrapolation past +-90 would make sin turn back and yield a NEGATIVE area."""
    lat = np.array([-89.0, -88.0])  # first edge extrapolates to -89.5, fine
    assert np.all(latitude_weights(lat) > 0)
    lat_polar = np.array([-89.9, -89.0, -88.0])  # extrapolates below -90
    assert np.all(latitude_weights(lat_polar) > 0)


# --- F4: refuse exactly the unrepresentable class, in BOTH directions ----------


def test_F4_two_dimensional_coordinates_are_refused_naming_the_source():
    lat2d = np.array([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(GridGeometryError, match="2-D or curvilinear"):
        check_axis(lat2d, "latitude", source="cmip6_X_ssp245_r1i1p1f1")
    with pytest.raises(GridGeometryError, match="cmip6_X_ssp245_r1i1p1f1"):
        check_axis(lat2d, "latitude", source="cmip6_X_ssp245_r1i1p1f1")


@pytest.mark.parametrize(
    "axis",
    [
        [0.0, 1.0, 1.0, 2.0],        # repeated value
        [0.0, 2.0, 1.0, 3.0],        # unordered
        [170.0, 175.0, -175.0],      # dateline-wrapped subset
    ],
)
def test_F4_non_monotonic_axes_are_refused(axis):
    with pytest.raises(GridGeometryError, match="not strictly monotonic"):
        check_axis(axis, "longitude")


def test_F4_non_finite_axis_is_refused():
    with pytest.raises(GridGeometryError, match="non-finite"):
        check_axis([0.0, np.nan, 2.0], "latitude")


def test_F4_non_uniform_1d_axis_is_ACCEPTED_not_refused():
    """Over-refusing is as wrong as under-refusing -- the failure that looks cautious.

    A Gaussian axis raising here would narrow coverage relative to today and
    contradict the strict-generalization framing (falsifier F4, second direction).
    """
    gaussian_like = np.array([-70.0, -45.0, -25.0, -10.0, 10.0, 25.0, 45.0, 70.0])
    w = latitude_weights(gaussian_like)     # must not raise
    assert w.shape == gaussian_like.shape
    assert np.all(w > 0)


def test_F4_descending_axis_is_accepted():
    """Strictly monotonic includes DEcreasing -- CMIP6 latitudes are often N-to-S."""
    ascending = latitude_weights(np.array([10.0, 20.0, 30.0]))
    descending = latitude_weights(np.array([30.0, 20.0, 10.0]))
    np.testing.assert_allclose(descending, ascending[::-1], rtol=1e-12)


# --- F5: degenerate axis -- the ordinary small-basin path, not an edge case ----


def test_F5_length_one_axis_takes_the_degenerate_weight_one():
    """At Amon resolution a buffered small-basin bbox can select a single cell."""
    np.testing.assert_array_equal(latitude_weights(np.array([12.5])), [1.0])
    np.testing.assert_array_equal(longitude_weights(np.array([9.75])), [1.0])

    single = cell_area_weights(np.array([12.5]), np.array([9.75]))
    assert single.shape == (1, 1)
    assert single[0, 0] == 1.0


def test_F5_single_row_grid_keeps_the_other_axis_weighted():
    w = cell_area_weights(np.array([12.5]), np.array([9.0, 10.0, 11.0]))
    assert w.shape == (1, 3)
    assert np.all(w > 0)
    # One latitude row: the weights vary only with longitude width, which is
    # uniform here, so all three are equal.
    np.testing.assert_allclose(w[0], w[0][0])


# --- shape and provenance ------------------------------------------------------


def test_cell_area_weights_are_separable_and_correctly_shaped():
    lat = np.array([0.0, 10.0, 20.0])
    lon = np.array([100.0, 110.0])
    w = cell_area_weights(lat, lon)
    assert w.shape == (3, 2)
    np.testing.assert_allclose(
        w, np.outer(latitude_weights(lat), longitude_weights(lon)), rtol=1e-15
    )


def test_weighting_scheme_label_is_the_one_the_design_specifies():
    """A correct number under a stale provenance label survives review (F7)."""
    assert WEIGHTING_SCHEME == "spherical_cell_area_midpoint_edges"
