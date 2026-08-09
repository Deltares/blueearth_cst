"""The POST-MIGRATION project-tree inventory (`build_project_tree_rules`).

`[R10-11]`: the R9 path map runs one way — pre-R9 paths to post-R9 ones — so a
tree in the layout R9 delivered matches none of its old-side patterns and
`tree-check` returned exit 1 on every CORRECT tree. The map was never wrong; it
was being asked about an era that has passed.

The inventory answers the question that outlives the migration: **does this tree
hold anything nobody declared?** That is the property that caught
`region.geojson` (R9 phase-1 F1a) and that ADR 0003 §8a's seam intermediate
needed a row for.

Two instruments here, and the second matters more than the first:

1. row-driven coverage — every shape a clean three-workflow run produces
   resolves as IDENTITY;
2. the NON-CATCH-ALL guard — an artifact nobody declared still resolves to
   UNMAPPED. Without it the report is empty by construction and the gate passes
   unconditionally, which is the hazard
   `test_a_catch_all_config_prefix_would_empty_the_report` demonstrates on the
   R9 map.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import semantic_tree_diff as std  # noqa: E402

E = "experiment"
KEY = "era5_20000101_20201231"
CP = "cmip6"
INVENTORY = std.build_project_tree_rules(E, KEY, CP)


def _kind(rel: str) -> str:
    return std.classify_path_map([rel], INVENTORY)[0][2]


# ---------------------------------------------------------------------------
# 1. Coverage — the shapes a clean run produces
# ---------------------------------------------------------------------------

#: Taken from a clean three-workflow run on the seed config (2026-08-06, 186
#: paths), collapsed to distinct shapes. Grouped by destination root.
COVERED: dict[str, list[str]] = {
    "root": [
        "logs/wf1_model_creation.log",
        "logs/wf2_climate_projections.log",
        "logs/_parts/1.01b_delineate_region.log",
        "logs/dag/test_wf1_dag.png",
        "benchmarks/wf1_benchmarks.md",
        "benchmarks/wf2_benchmarks.md",
        "benchmarks/_parts/1.02_prepare_spatial_maps.tsv",
    ],
    "config": [
        "config/runs/snake_config_model_creation.yml",
        "config/runs/snake_config_climate_projections.yml",
        "config/runs/model_creation/1a22a14838f3/effective.yml",
        "config/runs/model_creation/1a22a14838f3/referenced-files.json",
        "config/runs/model_creation/1a22a14838f3/source.yml",
        "config/runs/model_creation/1a22a14838f3/.snakemake_timestamp",
        "config/runs/model_creation/1a22a14838f3/files/catalogs/459d6135261e-deltares_data.yml",
        "config/runs/climate_projections/61868971c618/effective.yml",
        "config/catalogs/deltares_data.yml",
        "config/templates/wflow_build_model.yml",
        "config/observations/output_locations.csv",
    ],
    "data": [
        "data/spatial/spatial_maps.nc",
        "data/spatial/spatial_catalog.yml",
        "data/spatial/spatial_report.yml",
        "data/spatial/location_registry.csv",
        # ADR 0003 §8a — the seam intermediate this inventory has to cover.
        "data/spatial/hydrography.nc",
        "data/spatial/geoms/region.geojson",
        "data/spatial/geoms/basins.geojson",
        "data/spatial/geoms/subbasins.geojson",
        # ADR 0007: basin_area depicts elevation, so it sits with the data.
        "data/spatial/plots/basin_area.png",
        "data/spatial/plots/basin_area.pdf",
        f"data/climate/historical/{KEY}/extract_historical.nc",
        f"data/climate/historical/{KEY}/.guard_ok",
        f"data/climate/historical/{KEY}/plots/source_precip_map.png",
        # A SECOND store key is legitimate: the key is a cache key, so a project
        # with an era5 and a chirps store holds both.
        "data/climate/historical/chirps_19900101_20101231/extract_historical.nc",
        f"data/climate/projections/{CP}/raw/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc",
        f"data/climate/projections/{CP}/scalar/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc",
        f"data/climate/projections/{CP}/summary/cmip6_change_factors_annual.csv",
        f"data/climate/projections/{CP}/summary/provenance.json",
        f"data/climate/projections/{CP}/plots/cmip6_change_factor_cloud.png",
        f"data/climate/projections/{CP}/report.md",
    ],
    "models": [
        "models/hydrology/wflow/staticmaps.nc",
        "models/hydrology/wflow/wflow_sbm.toml",
        "models/hydrology/wflow/hydromt.log",
        "models/hydrology/wflow/hydromt_data.yml",
        "models/hydrology/wflow/.model_built",
        "models/hydrology/wflow/.outputs_configured",
        # ADR 0004's terminal build sentinel.
        "models/hydrology/wflow/.model_final",
        "models/hydrology/wflow/config/build_historical_forcing.yml",
        "models/hydrology/wflow/forcing/inmaps_historical.nc",
        "models/hydrology/wflow/forcing/plots/forcing_precip_map.png",
        "models/hydrology/wflow/staticgeoms/outlet_index.csv",
        "models/hydrology/wflow/staticgeoms/gauges_locations.geojson",
        "models/hydrology/wflow/run_default/output.csv",
        "models/hydrology/wflow/run_default/log.txt",
        "models/hydrology/wflow/run_default/outstate/outstates.nc",
        "models/hydrology/wflow/evaluation/performance_metrics.csv",
        "models/hydrology/wflow/evaluation/plots/hydro_wflow_1.png",
    ],
    "experiments": [
        f"experiments/{E}/.project_consistency_ok",
        f"experiments/{E}/config/snake_config_climate_experiment.yml",
        f"experiments/{E}/config/model_reference.yml",
        f"experiments/{E}/config/experiment.yml",
        f"experiments/{E}/config/catalogs/data_catalog_climate_experiment.yml",
        f"experiments/{E}/config/runs/climate_experiment/278159763309/effective.yml",
        f"experiments/{E}/logs/wf3_climate_experiment.log",
        f"experiments/{E}/logs/_parts/3.11_generate_weather_realizations.log",
        f"experiments/{E}/benchmarks/wf3_benchmarks.md",
        f"experiments/{E}/results/q_indicators.csv",
        f"experiments/{E}/results/basin_indicators.csv",
        f"experiments/{E}/climate/weathergenr/output/sim_dates.csv",
        f"experiments/{E}/climate/weathergenr/config/weathergen_config.yml",
        f"experiments/{E}/climate/weathergenr/_work/st_4.csv",
        f"experiments/{E}/climate/weathergenr/plots/obs_power_spectra.png",
        f"experiments/{E}/hydrology/wflow/config/rlz_1_st_2.toml",
        f"experiments/{E}/hydrology/wflow/output/rlz_1_st_2.csv",
        f"experiments/{E}/hydrology/wflow/output/rlz_1_st_2.log",
    ],
}

ALL_COVERED = [(section, rel) for section, rows in COVERED.items() for rel in rows]


@pytest.mark.parametrize(
    "section,rel", ALL_COVERED, ids=[f"{s}:{r}" for s, r in ALL_COVERED]
)
def test_every_produced_shape_is_covered(section, rel):
    """A clean run must classify entirely as IDENTITY — zero unmapped."""
    new, matched = std.apply_path_map_matched(rel, INVENTORY)
    assert matched, f"{rel} is UNMAPPED — the inventory does not cover it"
    assert new == rel, f"the inventory must be identity, got {new}"


def test_coverage_is_not_trivially_satisfied():
    """Guard on the guard: every destination root is exercised."""
    assert set(COVERED) == {"root", "config", "data", "models", "experiments"}
    assert len(ALL_COVERED) >= 60


# ---------------------------------------------------------------------------
# 2. The non-catch-all guard — the property the inventory exists for
# ---------------------------------------------------------------------------

#: Artifacts no rule declares. Each must report UNMAPPED: this is the whole
#: point, and each one sits under a root the inventory DOES cover, so a prefix
#: written one level too broad would swallow it silently.
UNDECLARED = [
    "data/spatial/leftover_intermediate.nc",  # a settled dir: enumerated
    "data/spatial/spatial_maps.tmp.nc",  # a crashed write
    "models/hydrology/wflow/stray_output.nc",
    "models/hydrology/wflow/forcing/inmaps_2050.nc",
    "config/runs/something_new.yml",  # not the two contract paths
    "config/whatever_new_thing.yml",
    f"experiments/{E}/orphan_table.csv",
    f"experiments/{E}/indicators/Qstats.csv",  # the pre-R9 name, now retired
    "climate_historical/era5_20000101_20201231/extract_historical.nc",  # pre-R9
    "hydrology_model/staticmaps.nc",  # pre-R9
    "spatial/geoms/basins.geojson",  # pre-R9
    "unknown_root/anything.txt",
]


@pytest.mark.parametrize("rel", UNDECLARED)
def test_undeclared_artifacts_are_reported(rel):
    """An artifact nobody declared must not be absorbed by a broad prefix.

    Includes PRE-R9 paths deliberately: on a migrated tree those are leftovers
    from before the move, and reporting them is how a stale copy gets noticed.
    """
    assert _kind(rel) == "UNMAPPED", f"{rel} was silently absorbed"


def test_a_broad_data_prefix_would_empty_the_report():
    """Demonstrates the hazard rather than asserting its absence.

    Same argument as `test_a_catch_all_config_prefix_would_empty_the_report` on
    the R9 map: with `data/` -> `data/` an unknown artifact reads as a
    deliberate identity, so the unmapped report goes empty by construction.
    """
    unknown = "data/spatial/leftover_intermediate.nc"
    assert _kind(unknown) == "UNMAPPED"
    catch_all = [("data/", "data/")] + INVENTORY
    assert std.classify_path_map([unknown], catch_all)[0][2] == "IDENTITY"


# ---------------------------------------------------------------------------
# 3. The two maps are different questions
# ---------------------------------------------------------------------------


def test_the_r09_map_and_the_inventory_answer_different_eras():
    """`[R10-11]`'s finding, pinned so the two cannot be conflated again.

    A post-R9 path is covered by the inventory and UNMAPPED under the migration
    map; a pre-R9 path is the exact inverse. Neither map is wrong — they answer
    about different eras, and `tree-check` was asking the wrong one.
    """
    r09 = std.build_r09_path_map(E, KEY, CP)
    post = "data/spatial/spatial_maps.nc"
    pre = "spatial/spatial_maps.nc"

    assert std.apply_path_map_matched(post, INVENTORY)[1] is True
    assert std.apply_path_map_matched(post, r09)[1] is False
    assert std.apply_path_map_matched(pre, INVENTORY)[1] is False
    assert std.apply_path_map_matched(pre, r09) == (post, True)


def test_other_experiments_are_covered_but_not_the_project_root():
    """A tree may hold several experiments; all are legitimate.

    The catch-all is scoped INSIDE `experiments/`, so it cannot reach anything
    else — checked by the undeclared cases above, which include a project-root
    stray.
    """
    assert _kind("experiments/another_run/results/q_indicators.csv") == "IDENTITY"
    assert _kind(f"experiments/{E}/results/q_indicators.csv") == "IDENTITY"


def test_the_inventory_is_identity_everywhere():
    """No rule may MOVE a path — this map describes, it does not migrate."""
    moved = [
        (rel, std.apply_path_map(rel, INVENTORY))
        for _, rel in ALL_COVERED
        if std.apply_path_map(rel, INVENTORY) != rel
    ]
    assert not moved, moved


def test_rules_are_well_formed():
    """Every rule is a (pattern, template) pair the applier can use."""
    for old, new in INVENTORY:
        assert isinstance(old, (str, re.Pattern))
        assert isinstance(new, str) and new
