"""ADR 0003 §8: one vector foundation, one producer contract, three declarations.

The vector layers used to be by-products of rule 1.02, whose real product is
``spatial_maps.nc`` and the thematic raster stack behind it. WF2 and WF3 could
therefore only reach basin and subbasin boundaries by declaring a rule that
resamples ``vito``, ``modis_lai`` and ``soilgrids`` — the trade ADR 0003 exists
to have removed, one level down.

These pin the replacement: ``snake_utils.spatial_units_rule`` owns the contract,
and the three workflow declarations of ``delineate_spatial_units`` may differ
only in ``message`` / ``log`` / ``benchmark``. Same shape, and the same reason,
as ``tests/test_region_spec.py`` and ``tests/test_climate_store_contract.py`` —
the third and last member of that family.
"""
from __future__ import annotations

from pathlib import Path

from blueearth_cst.shared import snake_utils as su
from blueearth_cst.spatial.config import parse_spatial_config

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent


def _rule(basin_overrides=None, **overrides):
    basin = {"region": "{'subbasin': [9.666, 0.4476], 'uparea': 100}"}
    basin.update(basin_overrides or {})
    kwargs = dict(
        project_dir="/proj",
        spatial_config=parse_spatial_config(basin),
        data_sources="config/catalogs/deltares_data.yml",
    )
    kwargs.update(overrides)
    return su.spatial_units_rule(**kwargs)


# ---------------------------------------------------------------------------
# The helper's shape
# ---------------------------------------------------------------------------

def test_the_six_vector_artifacts_keep_their_paths():
    """The split moves the PRODUCER, never the products (ADR 0003 §8)."""
    rule = _rule()
    assert rule.outputs["basins"] == "/proj/data/spatial/geoms/basins.geojson"
    assert rule.outputs["subbasins"] == "/proj/data/spatial/geoms/subbasins.geojson"
    assert rule.outputs["catchments"] == "/proj/data/spatial/geoms/catchments.geojson"
    assert rule.outputs["rivers"] == "/proj/data/spatial/geoms/rivers.geojson"
    assert rule.outputs["locations"] == "/proj/data/spatial/geoms/locations.geojson"
    assert rule.outputs["location_registry"] == "/proj/data/spatial/location_registry.csv"


def test_the_seventh_output_is_the_seam_intermediate():
    """§8a: the whole hydrography grid stack crosses the seam as a file."""
    rule = _rule()
    assert rule.hydrography_nc == "/proj/data/spatial/hydrography.nc"
    assert rule.outputs["hydrography"] == rule.hydrography_nc
    assert len(rule.outputs) == 7


def test_the_inputs_are_the_catalog_and_the_shared_region():
    """Model-free: config + catalog + the one project polygon, nothing else.

    A built model, `staticmaps.nc`, or the `--configfile` path as an input
    would each break the property this rule is shared for.
    """
    rule = _rule()
    assert rule.inputs == {
        "data_catalogs": "config/catalogs/deltares_data.yml",
        "region_geojson": "/proj/data/spatial/geoms/region.geojson",
    }


def test_the_region_path_comes_from_the_region_helper():
    """One owner for the polygon's path, so the two helpers cannot disagree."""
    rule = _rule()
    region = su.region_spec(
        "/proj",
        "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
        "config/catalogs/deltares_data.yml",
    )
    assert rule.inputs["region_geojson"] == region.region_geojson


def test_gauge_points_are_a_declared_input_only_when_configured():
    """An unset key contributes no entry at all -- rule 1.02's own shape.

    Declared as an INPUT rather than a param: as a param Snakemake compares the
    path, so renumbering the FILE would leave the registry on the old ids in
    silence.
    """
    assert "gauge_points" not in _rule().inputs
    assert "gauge_points" not in _rule(basin_overrides={"gauge_points": None}).inputs
    # The legacy "None" sentinel spelling is unset too.
    assert "gauge_points" not in _rule(basin_overrides={"gauge_points": "None"}).inputs
    configured = _rule(basin_overrides={"gauge_points": "C:/data/gauges.csv"})
    assert configured.inputs["gauge_points"] == "C:/data/gauges.csv"


def test_params_carry_only_shared_basin_fields():
    """§8b: a pure function of `project` + `shared.basin`, thematic-free.

    The thematic source names belong to the raster half. Carrying them here
    would make an edit to `spatial_sources.lulc` re-run the vector rule in all
    three workflows for a layer none of them reads.
    """
    rule = _rule()
    assert set(rule.params) == {
        "hydrography",
        "resolution",
        "river_uparea_km2",
        "rivers_source",
        "gauge_snap_tolerance_m",
        "max_automatic_subbasins",
    }
    assert rule.params["hydrography"] == "merit_hydro_ihu"
    assert rule.params["rivers_source"] == "rivers_lin2019_v1"


def test_the_deprecated_model_creation_fallback_cannot_reach_the_rule():
    """§8b's stated consequence, pinned rather than left to be discovered.

    `resolve_gauge_points_path` still accepts
    `workflows.model_creation.output_locations` for one compatibility release,
    but the shared rule is resolved WITHOUT a model section -- the five
    projections-only configs have none, so a params payload drawn from it would
    differ per invoking workflow.
    """
    legacy_only = parse_spatial_config(
        {"region": {"basin": [0, 0]}}, {"output_locations": "C:/data/legacy.csv"}
    )
    assert legacy_only.gauge_points_path == "C:/data/legacy.csv"
    shared = parse_spatial_config({"region": {"basin": [0, 0]}})
    assert shared.gauge_points_path is None
    assert "gauge_points" not in su.spatial_units_rule(
        "/proj", shared, "cat.yml"
    ).inputs


def test_overrides_are_carried_through():
    rule = _rule(basin_overrides={
        "hydrography": "merit_hydro_1k",
        "resolution": 0.05,
        "river_uparea_km2": 50.0,
        "gauge_snap_tolerance_m": 2500.0,
        "automatic_subbasins": {"max_count": 7},
        "spatial_sources": {"rivers": "my_rivers"},
    })
    assert rule.params["hydrography"] == "merit_hydro_1k"
    assert rule.params["resolution"] == 0.05
    assert rule.params["river_uparea_km2"] == 50.0
    assert rule.params["gauge_snap_tolerance_m"] == 2500.0
    assert rule.params["max_automatic_subbasins"] == 7
    assert rule.params["rivers_source"] == "my_rivers"


def test_script_is_relative_to_the_repo_root():
    """One relative path serves all three Snakefiles (`script:` -> basedir)."""
    rule = _rule()
    assert rule.script == "blueearth_cst/spatial/delineate_spatial_units.py"
    assert (SNAKEDIR / rule.script).is_file()


def test_shared_does_not_import_spatial():
    """`spatial/` imports `shared/`; the dependency must not also run back.

    `spatial_units_rule` reads its `SpatialConfig` attribute-wise for this
    reason -- a real import would be a cycle, and a function-local one would
    hide it.
    """
    import ast

    module = SNAKEDIR / "blueearth_cst" / "shared" / "snake_utils.py"
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    offenders = {name for name in imported if name.startswith("blueearth_cst.spatial")}
    assert not offenders, offenders


