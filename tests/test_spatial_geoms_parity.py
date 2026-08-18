"""Contract tests for `shared/spatial_geoms_parity` (board item t2608071203 / R9-1).

Two layers, deliberately:

- SYNTHETIC cases pin the semantics -- what counts as contained, what counts
  as a copy, and that the tolerance is a tolerance rather than an escape
  hatch. These run everywhere, including a bare-checkout CI leg.
- Two FIXTURE cases assert the relationships on a really-built project. Be
  precise about what they can catch: the fixture is a static tree, so a hydromt
  upgrade does NOT move it on its own. These fire when the fixture is REBUILT
  under a new hydromt -- a model rebuild or a baseline re-record -- which is
  the moment a changed `GridComponent._region_data` would first reach disk.
  They skip without the fixture, and AGENTS.md records that a worktree lacking
  `test_case/` downgrades rather than fails, so a green run here is not
  evidence the fixture leg ran at all.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blueearth_cst.shared.spatial_geoms_parity import (  # noqa: E402
    CONTAINED_LAYERS,
    COPIED_LAYERS,
    GEOMETRY_ATOL_DEG,
    INCOMPARABLE_LAYERS,
    SHARED_BASENAMES,
    validate_contained_layer,
    validate_copied_layer,
    validate_spatial_geoms_parity,
)

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely.geometry")

FIXTURE = Path(__file__).resolve().parents[1] / "test_case" / "test_local"
OURS_DIR = FIXTURE / "data" / "spatial" / "geoms"
THEIRS_DIR = FIXTURE / "models" / "hydrology" / "wflow" / "staticgeoms"


def _frame(geoms, crs="EPSG:4326", **cols):
    return gpd.GeoDataFrame(dict(cols), geometry=list(geoms), crs=crs)


def _box(x0, y0, x1, y1):
    return shapely.box(x0, y0, x1, y1)


# --- the layer sets are consistent with each other ---------------------------


def test_shared_basenames_is_exactly_the_three_relationship_classes():
    """No layer is silently unclassified, and none is in two classes."""
    classes = [set(CONTAINED_LAYERS), set(INCOMPARABLE_LAYERS), set(COPIED_LAYERS)]
    union = set().union(*classes)
    assert union == set(SHARED_BASENAMES)
    assert sum(len(c) for c in classes) == len(union), "a layer is in two classes"


def test_every_incomparable_layer_states_a_reason():
    """The reasons ARE the contract for these two -- an empty one says nothing."""
    for layer, reason in INCOMPARABLE_LAYERS.items():
        assert reason.strip(), f"{layer} is listed incomparable with no reason"


# --- containment -------------------------------------------------------------


def test_contained_passes_when_ours_is_strictly_inside():
    ours = _frame([_box(0, 0, 1, 1)])
    theirs = _frame([_box(-1, -1, 2, 2)])
    assert validate_contained_layer(ours, theirs, "region") == []


def test_contained_passes_on_equal_extents():
    """Equality is containment. It would be a SURPRISE, but it is not a breach."""
    ours = _frame([_box(0, 0, 1, 1)])
    assert validate_contained_layer(ours, ours, "region") == []


def test_contained_fails_when_ours_escapes_and_says_how_much():
    ours = _frame([_box(0, 0, 2, 1)])  # half of it outside
    theirs = _frame([_box(0, 0, 1, 1)])
    report = validate_contained_layer(ours, theirs, "region")
    assert len(report) == 1
    assert "NOT contained" in report[0]
    assert "50.00%" in report[0], report[0]


def test_contained_tolerates_a_boundary_sliver():
    """The 1.4e-08 GeoJSON round-trip sliver must not read as a breach."""
    ours = _frame([_box(0, 0, 1, 1 + 1e-9)])
    theirs = _frame([_box(0, 0, 1, 1)])
    assert validate_contained_layer(ours, theirs, "region") == []


def test_contained_reports_a_crs_difference():
    ours = _frame([_box(0, 0, 1, 1)], crs="EPSG:3857")
    theirs = _frame([_box(-1, -1, 2, 2)], crs="EPSG:4326")
    assert any("crs" in r for r in validate_contained_layer(ours, theirs, "region"))


# --- copies ------------------------------------------------------------------


def test_copy_passes_on_identical_frames():
    gdf = _frame([_box(0, 0, 1, 1)], value=[1])
    assert validate_copied_layer(gdf, gdf, "subbasins") == []


def test_copy_tolerates_a_geojson_round_trip_displacement():
    """4.7e-07 deg is what a real round trip costs; it is not a divergence."""
    d = 4.7e-7
    ours = _frame([_box(0, 0, 1, 1)])
    theirs = _frame([_box(d, d, 1 + d, 1 + d)])
    assert validate_copied_layer(ours, theirs, "subbasins") == []


def test_copy_fails_beyond_the_tolerance():
    d = GEOMETRY_ATOL_DEG * 10
    ours = _frame([_box(0, 0, 1, 1)])
    theirs = _frame([_box(d, d, 1 + d, 1 + d)])
    report = validate_copied_layer(ours, theirs, "subbasins")
    assert len(report) == 1
    assert "hausdorff" in report[0]


def test_copy_catches_a_point_displacement_that_has_no_area():
    """`locations` is points -- a symmetric-difference-area test cannot see this.

    This is why the tolerant test is a DISTANCE. Two points a kilometre apart
    have a symmetric difference of area 0, so an areal comparator passes them.
    """
    ours = _frame([shapely.Point(0, 0)])
    theirs = _frame([shapely.Point(0.01, 0.01)])
    report = validate_copied_layer(ours, theirs, "locations")
    assert len(report) == 1
    assert "hausdorff" in report[0]


def test_copy_reports_a_schema_difference():
    ours = _frame([_box(0, 0, 1, 1)], subbasin_code=["a"])
    theirs = _frame([_box(0, 0, 1, 1)], value=[1])
    assert any("columns" in r for r in validate_copied_layer(ours, theirs, "subbasins"))


def test_copy_reports_differing_column_values():
    ours = _frame([_box(0, 0, 1, 1)], subbasin_code=["a"])
    theirs = _frame([_box(0, 0, 1, 1)], subbasin_code=["b"])
    report = validate_copied_layer(ours, theirs, "subbasins")
    assert any("subbasin_code" in r and "values differ" in r for r in report)


def test_copy_reports_a_feature_count_difference_and_stops():
    ours = _frame([_box(0, 0, 1, 1), _box(2, 2, 3, 3)])
    theirs = _frame([_box(0, 0, 1, 1)])
    report = validate_copied_layer(ours, theirs, "subbasins")
    assert report == ["subbasins: feature count 2 vs 1"]


# --- the whole-mapping entry point -------------------------------------------


def _synthetic_pair():
    ours, theirs = {}, {}
    for layer in SHARED_BASENAMES:
        ours[layer] = _frame([_box(0, 0, 1, 1)])
        theirs[layer] = _frame([_box(0, 0, 1, 1)])
    theirs["region"] = _frame([_box(-1, -1, 2, 2)])  # grid extent circumscribes
    return ours, theirs


def test_parity_passes_on_a_well_formed_pair():
    ours, theirs = _synthetic_pair()
    assert validate_spatial_geoms_parity(ours, theirs) == []


def test_parity_reports_an_absent_layer_rather_than_skipping_it():
    """An empty mapping must not read as a pass -- the whole point is silence
    being indistinguishable from correctness elsewhere in this area."""
    report = validate_spatial_geoms_parity({}, {})
    assert len(report) == 2 * len(SHARED_BASENAMES)
    assert all("absent" in r for r in report)


def test_parity_does_not_compare_the_incomparable_layers():
    """`basins` and `rivers` share a basename and nothing else.

    Made concrete: give them wildly different content and confirm the report
    stays empty. A future edit that "helpfully" compares them turns this red.
    """
    ours, theirs = _synthetic_pair()
    for layer in INCOMPARABLE_LAYERS:
        ours[layer] = _frame([_box(0, 0, 1, 1)], mine=[1])
        theirs[layer] = _frame(
            [_box(50, 50, 60, 60), _box(70, 70, 80, 80)], theirs=[1, 2]
        )
    assert validate_spatial_geoms_parity(ours, theirs) == []


# --- the actual watcher ------------------------------------------------------


@pytest.mark.skipif(
    not (OURS_DIR.exists() and THEIRS_DIR.exists()),
    reason="needs the built test_case/test_local fixture",
)
def test_the_relationships_hold_on_a_really_built_project():
    """The hydromt watch (exposure 3 of R9-1).

    `ours <= theirs` for `region` holds because
    `GridComponent._region_data` returns `box(*self.bounds)`
    (hydromt/model/components/grid.py:269, hydromt 1.3.1) and the grid is built
    to circumscribe the delineated basin.

    This notices an upgrade that changed that derivation only once the fixture
    is REBUILT under the new hydromt -- the tree here is static, so the check
    is armed at rebuild time rather than at upgrade time. That is still the
    right place: a rebuild is exactly when the new derivation first lands on
    disk, and before then nothing downstream has consumed it.
    """
    ours = {n: gpd.read_file(OURS_DIR / f"{n}.geojson") for n in SHARED_BASENAMES}
    theirs = {n: gpd.read_file(THEIRS_DIR / f"{n}.geojson") for n in SHARED_BASENAMES}
    assert validate_spatial_geoms_parity(ours, theirs) == []


@pytest.mark.skipif(
    not (OURS_DIR.exists() and THEIRS_DIR.exists()),
    reason="needs the built test_case/test_local fixture",
)
def test_the_incomparable_layers_really_are_incomparable_on_the_fixture():
    """Pin the PREMISE, not just the conclusion.

    If a hydromt upgrade ever made `basins` or `rivers` agree across the two
    trees, the module's central claim -- that these share a basename and
    nothing else -- would be quietly false, and the right response would be to
    reclassify them rather than to keep skipping them.
    """
    for layer in INCOMPARABLE_LAYERS:
        ours = gpd.read_file(OURS_DIR / f"{layer}.geojson")
        theirs = gpd.read_file(THEIRS_DIR / f"{layer}.geojson")
        same_shape = len(ours) == len(theirs) and [
            c for c in ours.columns if c != "geometry"
        ] == [c for c in theirs.columns if c != "geometry"]
        assert not same_shape, (
            f"{layer} now agrees across both trees; it is no longer "
            f"incomparable and should be reclassified"
        )
