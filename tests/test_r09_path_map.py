"""Unit tests for the R09 project-tree path map (dev/scripts/semantic_tree_diff.py).

The map is the migration contract (`dev/milestones/r09/migration_project-tree.md`);
this module is its regression detector, and it exists BEFORE the move because a
map that is wrong in the same direction as the migration is undetectable
afterwards (master brief, human gate 1).

Four instruments, in increasing coverage:

1. per-relocation-class tests -- readable, and they name the property each class
   asserts;
2. the two named precedence hazards -- each must resolve to the NARROW
   destination, not to the general rule that would otherwise swallow it;
3. `test_every_map_row_resolves` -- the map doc's four destination sections,
   row by row, as test data. A per-class test does not reach every row and the
   declared-tier falsifier cannot reach the undeclared engine artifacts
   (`hydromt.log`, `staticgeoms/*`, `run_default/*`, `evaluation/*`, `_work/*`,
   Wflow's `log.txt`); this is the only instrument that does;
4. the declared-tier falsifier against the committed inventory, asserting the
   EXACT set of paths the map does not cover.
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import semantic_tree_diff as std  # noqa: E402

E = "experiment"
KEY = "era5_20000101_20201231"
MAP = std.build_r09_path_map(E, KEY)
MAP_WITH_GAPS = MAP + std.build_r09_gap_rules(E)
DELETIONS = std.build_r09_deletions(E)

REPO = Path(__file__).resolve().parents[1]
DECLARED_INVENTORY = REPO / "dev" / "milestones" / "r09" / "declared_inventory.txt"


def _map(rel, path_map=MAP):
    return std.apply_path_map(rel, path_map)


# ---------------------------------------------------------------------------
# 1. One test per relocation class
# ---------------------------------------------------------------------------

def test_wflow_member_index_moves_from_directory_into_filename():
    """The class R09 exists for, and the exact inverse of R07's B5 rules.

    R07 moved `rlz_<r>` out of the filename and into a directory level; R09
    removes that level and puts the index back into the filename. Neither a
    prefix nor an exact rule can express it -- these are regex rules, and the
    `inmaps_` / `outstates_` prefixes have to survive the round trip.
    """
    cases = {
        f"experiments/{E}/hydrology_runs/rlz_2/config/cst_3.toml":
            f"experiments/{E}/hydrology/wflow/config/rlz_2_cst_3.toml",
        f"experiments/{E}/hydrology_runs/rlz_2/forcing/inmaps_cst_3.nc":
            f"experiments/{E}/hydrology/wflow/forcing/inmaps_rlz_2_cst_3.nc",
        f"experiments/{E}/hydrology_runs/rlz_10/output/cst_7.csv":
            f"experiments/{E}/hydrology/wflow/output/rlz_10_cst_7.csv",
        f"experiments/{E}/hydrology_runs/rlz_1/output/outstates_cst_12.nc":
            f"experiments/{E}/hydrology/wflow/output/outstates_rlz_1_cst_12.nc",
    }
    for old, new in cases.items():
        assert _map(old) == new, old


def test_weathergenr_subtree_relocates_whole_directories():
    cases = {
        f"experiments/{E}/weather_generator/output/rlz_1_cst_0.nc":
            f"experiments/{E}/climate/weathergenr/output/rlz_1_cst_0.nc",
        f"experiments/{E}/weather_generator/output/sim_dates.csv":
            f"experiments/{E}/climate/weathergenr/output/sim_dates.csv",
        f"experiments/{E}/weather_generator/config/weathergen_config.yml":
            f"experiments/{E}/climate/weathergenr/config/weathergen_config.yml",
        f"experiments/{E}/weather_generator/_work/cst_4.csv":
            f"experiments/{E}/climate/weathergenr/_work/cst_4.csv",
        f"experiments/{E}/weather_generator/plots/warm_annual_precip.png":
            f"experiments/{E}/climate/weathergenr/plots/warm_annual_precip.png",
    }
    for old, new in cases.items():
        assert _map(old) == new, old


def test_result_tables_are_the_only_rule_all_renames():
    """naming.md §7's rename record is exactly two files across the whole map."""
    assert _map(f"experiments/{E}/indicators/Qstats.csv") == \
        f"experiments/{E}/results/q_indicators.csv"
    assert _map(f"experiments/{E}/indicators/basin.csv") == \
        f"experiments/{E}/results/basin_indicators.csv"


def test_rt_tables_are_classified_deleted_not_mapped():
    """`indicators/RT_*.csv` is deleted, not migrated (v2 decision 3).

    Encoding a destination for it would invent one; leaving it out of the map
    entirely would make it read as an uncovered artifact. It is classified
    separately, so the row is covered and the map stays honest.
    """
    rows = std.classify_path_map(
        [f"experiments/{E}/indicators/RT_10.csv"], MAP, DELETIONS)
    assert rows == [(f"experiments/{E}/indicators/RT_10.csv", "", "DELETED")]
    # ...and without the deletion list it is UNMAPPED, never a silent identity.
    assert std.classify_path_map(
        [f"experiments/{E}/indicators/RT_10.csv"], MAP)[0][2] == "UNMAPPED"


def test_hydrology_model_relocates_to_the_models_root():
    cases = {
        "hydrology_model/staticmaps.nc": "models/hydrology/wflow/staticmaps.nc",
        "hydrology_model/wflow_sbm.toml": "models/hydrology/wflow/wflow_sbm.toml",
        "hydrology_model/.model_built": "models/hydrology/wflow/.model_built",
        "hydrology_model/staticgeoms/region.geojson":
            "models/hydrology/wflow/staticgeoms/region.geojson",
        "hydrology_model/forcing/inmaps_historical.nc":
            "models/hydrology/wflow/forcing/inmaps_historical.nc",
        "hydrology_model/forcing/plots/forcing_precip_map.png":
            "models/hydrology/wflow/forcing/plots/forcing_precip_map.png",
        "hydrology_model/run_default/output.csv":
            "models/hydrology/wflow/run_default/output.csv",
        "hydrology_model/evaluation/performance_metrics.csv":
            "models/hydrology/wflow/evaluation/performance_metrics.csv",
    }
    for old, new in cases.items():
        assert _map(old) == new, old


def test_climate_store_keeps_its_cache_key():
    """Finding 3: `<source>_<window>` is a CACHE KEY and is retained verbatim."""
    assert _map(f"climate_historical/{KEY}/extract_historical.nc") == \
        f"data/climate/historical/{KEY}/extract_historical.nc"
    assert _map(f"climate_historical/{KEY}/.guard_ok") == \
        f"data/climate/historical/{KEY}/.guard_ok"
    # The row is keyed by a variable, so an unfamiliar key still maps.
    assert _map("climate_historical/chirps_19900101_20101231/plots/x.png") == \
        "data/climate/historical/chirps_19900101_20101231/plots/x.png"


def test_projection_tiers_relocate_under_data():
    cases = {
        "climate_projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1.nc":
            "data/climate/projections/cmip6/raw/"
            "cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1.nc",
        "climate_projections/cmip6/scalar/cmip6_INM_INM-CM4-8_historical_r1i1p1f1.nc":
            "data/climate/projections/cmip6/scalar/"
            "cmip6_INM_INM-CM4-8_historical_r1i1p1f1.nc",
        "climate_projections/cmip6/summary/cmip6_change_factors_annual.csv":
            "data/climate/projections/cmip6/summary/cmip6_change_factors_annual.csv",
        "climate_projections/cmip6/report.md":
            "data/climate/projections/cmip6/report.md",
    }
    for old, new in cases.items():
        assert _map(old) == new, old
    # Verbatim CMIP model IDs are tier-1 identifiers, never normalized.
    assert "NOAA-GFDL_GFDL-ESM4" in _map(list(cases)[0])


def test_rule_3_11_rename_touches_only_transient_parts():
    """The one rule identifier R09 changes; its path effect is parts only."""
    assert _map(f"experiments/{E}/logs/_parts/3.11_export_wflow_results.log") == \
        f"experiments/{E}/logs/_parts/3.11_derive_wflow_indicators.log"
    assert _map(
        f"experiments/{E}/benchmarks/_parts/3.11_export_wflow_results.tsv"
    ) == f"experiments/{E}/benchmarks/_parts/3.11_derive_wflow_indicators.tsv"
    # The rename rule must beat the `_parts/` identity row registered after it.
    assert _map(f"experiments/{E}/logs/_parts/3.09_other.log") == \
        f"experiments/{E}/logs/_parts/3.09_other.log"


def test_identity_rows_are_matched_rules_not_fall_through():
    """The distinction the whole falsifier rests on.

    Every one of these resolves to itself -- and so does an unmapped path. Only
    `apply_path_map_matched` can tell them apart, which is why identity is
    enumerated per row instead of written as a `config/` catch-all.
    """
    for rel in (
        "config/runs/snake_config_model_creation.yml",
        "config/runs/snake_config_climate_projections.yml",
        "config/runs/model_creation/1a22a14838f3/snake_config.yml",
        "config/catalogs/deltares_data.yml",
        "config/templates/wflow_build_model.yml",
        "config/observations/discharge.csv",
        "logs/wf1_model_creation.log",
        "logs/_parts/1.01_snapshot_config.log",
        "benchmarks/wf2_benchmarks.md",
        f"experiments/{E}/.project_consistency_ok",
        f"experiments/{E}/config/snake_config_climate_experiment.yml",
    ):
        new, matched = std.apply_path_map_matched(rel, MAP)
        assert new == rel, rel
        assert matched, f"{rel} resolves to itself by FALL-THROUGH, not by rule"


def test_a_catch_all_config_prefix_would_empty_the_report():
    """Pins the hazard the map doc calls out, by demonstrating it.

    With a broad `config/` -> `config/` rule an unknown `config/` artifact is
    reported as a deliberate identity, so the unmapped-path report goes empty
    by construction and the falsifier passes unconditionally.
    """
    unknown = "config/whatever_new_thing.yml"
    assert std.classify_path_map([unknown], MAP)[0][2] == "UNMAPPED"
    catch_all = [("config/", "config/")] + MAP
    assert std.classify_path_map([unknown], catch_all)[0][2] == "IDENTITY"


# ---------------------------------------------------------------------------
# 2. The two named precedence hazards
# ---------------------------------------------------------------------------

def test_hazard_generated_build_yaml_beats_the_config_identity_rows():
    """`config/generated/*` is routed to the MODEL root, not left under config/."""
    assert _map("config/generated/wflow_build_model_run.yml") == \
        "models/hydrology/wflow/config/build_model.yml"
    assert _map("config/generated/wflow_build_forcing_historical.yml") == \
        "models/hydrology/wflow/config/build_historical_forcing.yml"


def test_hazard_wflow_log_beats_the_run_config_regex():
    """Wflow's `log.txt` goes to `output/`, not to `config/` with the TOMLs.

    The member index is NOT recoverable from the old path -- one log per
    realization becomes N per-member logs -- so the destination keeps the map
    doc's `<c>` placeholder. That is a one-to-many SPLIT, reported as a finding
    against the map; what this test pins is the PRECEDENCE, which is what a
    later `config/(.*)` rule would silently break.
    """
    assert _map(f"experiments/{E}/hydrology_runs/rlz_3/config/log.txt") == \
        f"experiments/{E}/hydrology/wflow/output/rlz_3_cst_<c>.log"
    # The general run-config rule still owns the TOMLs in the same directory.
    assert _map(f"experiments/{E}/hydrology_runs/rlz_3/config/cst_5.toml") == \
        f"experiments/{E}/hydrology/wflow/config/rlz_3_cst_5.toml"


def test_narrower_source_pattern_is_registered_first():
    """The invariant stated as an ordering property, not just as outcomes.

    Each pair is (narrow, general): the narrow rule's index must be lower, or
    the general one consumes its paths first (`apply_path_map` is first match
    wins).
    """
    def index_of(pattern_src: str) -> int:
        for i, (old, _) in enumerate(MAP):
            src = old.pattern if isinstance(old, re.Pattern) else old
            if src == pattern_src:
                return i
        raise AssertionError(f"rule not found: {pattern_src}")

    exp = re.escape(E)
    pairs = [
        (rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/config/log\.txt",
         rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/config/cst_(\d+)\.toml"),
        (rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/output/"
         rf"outstates_cst_(\d+)\.nc",
         rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/output/cst_(\d+)\.csv"),
        (f"experiments/{E}/logs/_parts/3.11_export_wflow_results.log",
         f"experiments/{E}/logs/"),
        (f"climate_historical/{KEY}/", r"climate_historical/([^/]+)/(.*)"),
        ("hydrology_model/forcing/inmaps_historical.nc",
         "hydrology_model/forcing/plots/"),
    ]
    for narrow, general in pairs:
        assert index_of(narrow) < index_of(general), (narrow, general)


# ---------------------------------------------------------------------------
# 3. Row-driven: every row of the map doc's four destination sections
# ---------------------------------------------------------------------------

#: (old, new) verbatim from dev/milestones/r09/migration_project-tree.md.
#: Glob/`**` rows are instantiated with one representative path each; rows whose
#: old side carries `<r>`/`<c>`/`<digest>`/`<store_key>` are instantiated with
#: concrete indices. Grouped by the doc's own sections, in the doc's own order --
#: which is deliberately NOT the rule order.
MAP_ROWS: dict[str, list[tuple[str, str]]] = {
    # --- section: -> models/hydrology/wflow/ ---
    "models": [
        ("hydrology_model/staticmaps.nc",
         "models/hydrology/wflow/staticmaps.nc"),
        ("hydrology_model/wflow_sbm.toml",
         "models/hydrology/wflow/wflow_sbm.toml"),
        ("hydrology_model/hydromt.log",
         "models/hydrology/wflow/hydromt.log"),
        ("hydrology_model/hydromt_data.yml",
         "models/hydrology/wflow/hydromt_data.yml"),
        ("hydrology_model/staticgeoms/outlets.geojson",
         "models/hydrology/wflow/staticgeoms/outlets.geojson"),
        ("hydrology_model/forcing/inmaps_historical.nc",
         "models/hydrology/wflow/forcing/inmaps_historical.nc"),
        ("hydrology_model/forcing/plots/forcing_temp_map.png",
         "models/hydrology/wflow/forcing/plots/forcing_temp_map.png"),
        ("hydrology_model/run_default/output.csv",
         "models/hydrology/wflow/run_default/output.csv"),
        ("hydrology_model/evaluation/plots/hydro_wflow_1.png",
         "models/hydrology/wflow/evaluation/plots/hydro_wflow_1.png"),
        ("hydrology_model/plots/basin_area.png",
         "models/hydrology/wflow/plots/basin_area.png"),
        ("hydrology_model/plots/basin_area.pdf",
         "models/hydrology/wflow/plots/basin_area.pdf"),
        ("hydrology_model/.model_built",
         "models/hydrology/wflow/.model_built"),
        ("hydrology_model/.outputs_configured",
         "models/hydrology/wflow/.outputs_configured"),
        ("config/generated/wflow_build_model_run.yml",
         "models/hydrology/wflow/config/build_model.yml"),
        ("config/generated/wflow_build_forcing_historical.yml",
         "models/hydrology/wflow/config/build_historical_forcing.yml"),
    ],
    # --- section: -> data/ ---
    "data": [
        ("spatial/spatial_maps.nc", "data/spatial/spatial_maps.nc"),
        ("spatial/spatial_catalog.yml", "data/spatial/spatial_catalog.yml"),
        ("spatial/spatial_report.yml", "data/spatial/spatial_report.yml"),
        ("spatial/location_registry.csv", "data/spatial/location_registry.csv"),
        ("spatial/geoms/basins.geojson", "data/spatial/geoms/basins.geojson"),
        ("spatial/geoms/catchments.geojson",
         "data/spatial/geoms/catchments.geojson"),
        ("spatial/geoms/locations.geojson",
         "data/spatial/geoms/locations.geojson"),
        ("spatial/geoms/rivers.geojson", "data/spatial/geoms/rivers.geojson"),
        ("spatial/geoms/subbasins.geojson",
         "data/spatial/geoms/subbasins.geojson"),
        (f"climate_historical/{KEY}/extract_historical.nc",
         f"data/climate/historical/{KEY}/extract_historical.nc"),
        (f"climate_historical/{KEY}/store_region.geojson",
         f"data/climate/historical/{KEY}/store_region.geojson"),
        (f"climate_historical/{KEY}/plots/source_precip_map.png",
         f"data/climate/historical/{KEY}/plots/source_precip_map.png"),
        (f"climate_historical/{KEY}/.guard_ok",
         f"data/climate/historical/{KEY}/.guard_ok"),
        ("climate_projections/cmip6/raw/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc",
         "data/climate/projections/cmip6/raw/"
         "cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc"),
        ("climate_projections/cmip6/scalar/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc",
         "data/climate/projections/cmip6/scalar/"
         "cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc"),
        ("climate_projections/cmip6/summary/provenance.json",
         "data/climate/projections/cmip6/summary/provenance.json"),
        ("climate_projections/cmip6/plots/cmip6_change_factor_cloud.png",
         "data/climate/projections/cmip6/plots/cmip6_change_factor_cloud.png"),
        ("climate_projections/cmip6/report.md",
         "data/climate/projections/cmip6/report.md"),
    ],
    # --- section: -> experiments/<id>/ ---
    "experiments": [
        (f"experiments/{E}/weather_generator/output/rlz_1_cst_2.nc",
         f"experiments/{E}/climate/weathergenr/output/rlz_1_cst_2.nc"),
        (f"experiments/{E}/weather_generator/config/weathergen_config.yml",
         f"experiments/{E}/climate/weathergenr/config/weathergen_config.yml"),
        (f"experiments/{E}/weather_generator/_work/"
         f"weathergen_config_rlz_1_cst_2.yml",
         f"experiments/{E}/climate/weathergenr/_work/"
         f"weathergen_config_rlz_1_cst_2.yml"),
        (f"experiments/{E}/weather_generator/plots/obs_power_spectra.png",
         f"experiments/{E}/climate/weathergenr/plots/obs_power_spectra.png"),
        (f"experiments/{E}/weather_generator/output/sim_dates.csv",
         f"experiments/{E}/climate/weathergenr/output/sim_dates.csv"),
        (f"experiments/{E}/weather_generator/output/resampled_dates.csv",
         f"experiments/{E}/climate/weathergenr/output/resampled_dates.csv"),
        (f"experiments/{E}/hydrology_runs/rlz_1/config/cst_2.toml",
         f"experiments/{E}/hydrology/wflow/config/rlz_1_cst_2.toml"),
        (f"experiments/{E}/hydrology_runs/rlz_1/forcing/inmaps_cst_2.nc",
         f"experiments/{E}/hydrology/wflow/forcing/inmaps_rlz_1_cst_2.nc"),
        (f"experiments/{E}/hydrology_runs/rlz_1/output/cst_2.csv",
         f"experiments/{E}/hydrology/wflow/output/rlz_1_cst_2.csv"),
        (f"experiments/{E}/hydrology_runs/rlz_1/output/outstates_cst_2.nc",
         f"experiments/{E}/hydrology/wflow/output/outstates_rlz_1_cst_2.nc"),
        # One-to-many split: `<c>` is not recoverable from the old path.
        (f"experiments/{E}/hydrology_runs/rlz_1/config/log.txt",
         f"experiments/{E}/hydrology/wflow/output/rlz_1_cst_<c>.log"),
        (f"experiments/{E}/indicators/Qstats.csv",
         f"experiments/{E}/results/q_indicators.csv"),
        (f"experiments/{E}/indicators/basin.csv",
         f"experiments/{E}/results/basin_indicators.csv"),
        (f"experiments/{E}/data_catalog_climate_experiment.yml",
         f"experiments/{E}/config/catalogs/data_catalog_climate_experiment.yml"),
        (f"experiments/{E}/.project_consistency_ok",
         f"experiments/{E}/.project_consistency_ok"),
        (f"experiments/{E}/logs/wf3_climate_experiment.log",
         f"experiments/{E}/logs/wf3_climate_experiment.log"),
        (f"experiments/{E}/benchmarks/wf3_benchmarks.md",
         f"experiments/{E}/benchmarks/wf3_benchmarks.md"),
        (f"experiments/{E}/config/snake_config_climate_experiment.yml",
         f"experiments/{E}/config/snake_config_climate_experiment.yml"),
        (f"experiments/{E}/config/catalogs/data_catalog_climate_experiment.yml",
         f"experiments/{E}/config/catalogs/data_catalog_climate_experiment.yml"),
    ],
    # --- section: -> config/ ---
    "config": [
        ("config/runs/snake_config_model_creation.yml",
         "config/runs/snake_config_model_creation.yml"),
        ("config/runs/snake_config_climate_projections.yml",
         "config/runs/snake_config_climate_projections.yml"),
        ("config/runs/model_creation/1a22a14838f3/snake_config.yml",
         "config/runs/model_creation/1a22a14838f3/snake_config.yml"),
        ("config/catalogs/cmip6_data.yml", "config/catalogs/cmip6_data.yml"),
        ("config/templates/wflow_build_model.yml",
         "config/templates/wflow_build_model.yml"),
        ("config/observations/output_locations.csv",
         "config/observations/output_locations.csv"),
        ("config/generated/wflow_build_model_run.yml",
         "models/hydrology/wflow/config/build_model.yml"),
        ("config/generated/wflow_build_forcing_historical.yml",
         "models/hydrology/wflow/config/build_historical_forcing.yml"),
    ],
    # --- section: -> project root ---
    "root": [
        ("logs/wf1_model_creation.log", "logs/wf1_model_creation.log"),
        ("logs/wf2_climate_projections.log", "logs/wf2_climate_projections.log"),
        ("logs/_parts/1.01_snapshot_config.log",
         "logs/_parts/1.01_snapshot_config.log"),
        ("benchmarks/wf1_benchmarks.md", "benchmarks/wf1_benchmarks.md"),
        ("benchmarks/wf2_benchmarks.md", "benchmarks/wf2_benchmarks.md"),
        ("benchmarks/_parts/1.01_snapshot_config.tsv",
         "benchmarks/_parts/1.01_snapshot_config.tsv"),
        ("logs/dag/test_wf1_dag.png", "logs/dag/test_wf1_dag.png"),
    ],
    # --- section: Rule rename carried by R9 ---
    "rule_rename": [
        ("logs/_parts/3.11_export_wflow_results.log",
         "logs/_parts/3.11_derive_wflow_indicators.log"),
        ("benchmarks/_parts/3.11_export_wflow_results.tsv",
         "benchmarks/_parts/3.11_derive_wflow_indicators.tsv"),
    ],
}

ALL_ROWS = [(section, old, new)
            for section, rows in MAP_ROWS.items() for old, new in rows]


@pytest.mark.parametrize("section,old,new", ALL_ROWS,
                         ids=[f"{s}:{o}" for s, o, _ in ALL_ROWS])
def test_every_map_row_resolves(section, old, new):
    """Row-driven: each (old, new) pair from the map doc, as test data.

    This is the only instrument covering the rows the declared-tier falsifier
    cannot reach -- the undeclared engine artifacts, which no `output:`
    declaration names and `--dry-run` structurally cannot see.
    """
    got, matched = std.apply_path_map_matched(old, MAP)
    assert matched, f"{old} is UNMAPPED (fall-through), not resolved by a rule"
    assert got == new, old


def test_row_coverage_is_not_trivially_satisfied():
    """Guard on the guard: the row table must exercise every section."""
    assert set(MAP_ROWS) == {"models", "data", "experiments", "config",
                             "root", "rule_rename"}
    assert len(ALL_ROWS) >= 60


# ---------------------------------------------------------------------------
# 4. Declared-tier falsifier (dev/milestones/r09/declared_inventory.txt)
# ---------------------------------------------------------------------------

def _declared_paths() -> list[str]:
    text = DECLARED_INVENTORY.read_text(encoding="utf-8")
    paths = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert paths, "declared inventory is empty"
    return paths


def test_declared_inventory_holds_project_relative_paths_only():
    """A stray absolute path would read as a phantom uncovered artifact."""
    for rel in _declared_paths():
        assert not rel.startswith("/"), rel
        assert not re.match(r"^[A-Za-z]:[\\/]", rel), rel
        assert "\\" not in rel, rel


#: The digest-keyed config bundles are named by a hash over the PARSED CONFIG,
#: which includes `project.project_dir` -- so regenerating the inventory into a
#: different temp dir changes exactly these path components and nothing else
#: (measured 2026-08-04: `climate_projections/61868971c618` -> `407f4256c490`).
#: The assertions below normalize them, or a faithful regeneration would fail
#: for a reason that has nothing to do with the map.
_DIGEST_SEGMENT = re.compile(r"(config/runs/[a-z_]+)/[0-9a-f]{8,}(?=/|$)")


def _digest_agnostic(rel: str) -> str:
    return _DIGEST_SEGMENT.sub(r"\1/<digest>", rel)


#: The EXACT set of declared-tier paths the migration map does not cover, as an
#: exact set rather than a count: a count still passes when one gap is fixed and
#: a different one appears. Each is a finding against the map (phase-1 report);
#: `build_r09_gap_rules` carries the proposed rules, and the owner rules on them
#: at gate 1.
KNOWN_UNMAPPED = {
    "config/runs/climate_projections/<digest>",
    "experiments/experiment/config/runs/climate_experiment/<digest>",
    "spatial/geoms/region.geojson",
}


def test_declared_tier_unmapped_set_is_exactly_the_known_gaps():
    rows = std.classify_path_map(_declared_paths(), MAP, DELETIONS)
    unmapped = {_digest_agnostic(old)
                for old, _, kind in rows if kind == "UNMAPPED"}
    assert unmapped == KNOWN_UNMAPPED, std.format_path_map_report(rows)


def test_the_digest_normalizer_only_touches_config_run_bundles():
    """Guard on the normalizer: it must not blunt any other assertion."""
    assert _digest_agnostic("config/runs/model_creation/1a22a14838f3") == \
        "config/runs/model_creation/<digest>"
    assert _digest_agnostic(
        "experiments/experiment/config/runs/climate_experiment/278159763309/x.yml"
    ) == "experiments/experiment/config/runs/climate_experiment/<digest>/x.yml"
    for untouched in (
        "config/runs/snake_config_model_creation.yml",
        "spatial/geoms/region.geojson",
        "climate_projections/cmip6/raw/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc",
    ):
        assert _digest_agnostic(untouched) == untouched


def test_declared_tier_has_zero_unmapped_with_the_gap_rules():
    """The falsifier's headline: zero unmapped once the gaps are accepted."""
    rows = std.classify_path_map(_declared_paths(), MAP_WITH_GAPS, DELETIONS)
    unmapped = [old for old, _, kind in rows if kind == "UNMAPPED"]
    assert unmapped == [], std.format_path_map_report(rows)


def test_gap_rules_are_appended_and_change_no_map_row():
    """Gap rules are APPENDED, never interleaved.

    Each is disjoint from, or strictly broader than, every map rule, so a path
    the strict map already resolves must resolve identically with them
    appended. Checked over the whole declared inventory plus every row of the
    map doc, which is what makes 'appending is safe' a tested property rather
    than an assertion in a comment.
    """
    for rel in _declared_paths() + [old for _, old, _ in ALL_ROWS]:
        new, matched = std.apply_path_map_matched(rel, MAP)
        if matched:
            assert std.apply_path_map(rel, MAP_WITH_GAPS) == new, rel


def test_declared_tier_reports_moved_identity_and_deleted_separately():
    """Gate 1's evidence is the classified table, not a single count.

    `config/catalogs/x.yml -> config/catalogs/x.yml` is indistinguishable from
    a fall-through TO A READER, so the report has to label which rows are
    identity-by-rule.
    """
    rows = std.classify_path_map(_declared_paths(), MAP_WITH_GAPS, DELETIONS)
    kinds = {kind for _, _, kind in rows}
    assert "MOVED" in kinds and "IDENTITY" in kinds
    text = std.format_path_map_report(rows)
    assert "identity (by rule)" in text
    assert "MAP CLEAN" in text
