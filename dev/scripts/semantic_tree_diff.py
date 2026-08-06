"""Milestone full-tree semantic diff for the R06 structural refactor (design §9).

Walks a `project_dir` output tree and compares every file against a reference
tree, dispatching by extension to per-type comparators. This is the *un-manifested
slice* gate: it covers wf2/wf3 staticmaps, `wflow_sbm.toml`, and change-factor
NetCDFs that `check_baseline.py`'s thin `TARGETS` list never fingerprints.

Design contract (dev/milestones/r06/structural-refactor-design.md §9, rows ext1-04 / ext2-01
/ ext2-02):

- `.nc`  : ELEMENT-WISE comparator (dims; coordinate labels+order with NO
           realignment; exact NaN masks; per-element `_within_tol`; non-volatile
           attrs) -- NOT the aggregate `fingerprint_nc`/`diff_nc` stats.
- `.toml`: parse-and-normalize compare (structural, key-order/comment-insensitive).
- `.yml`/`.yaml` under `{project_dir}/config/` : the copied-config
           NORMALIZE-THEN-COMPARE policy (ext2-01) -- parse both sides, apply only
           the documented old->new path map to the reference, require everything
           else deep-equal.
- `.csv`, `.png`, discharge `output.csv` : REUSED verbatim from `check_baseline.py`
           (imported, never modified).

`check_baseline.py` is imported for its comparators; its own P3-1 edits (the
TARGETS repoint, G1 scope amendment) live in that file, not here. The
CSV/PNG/discharge comparators and `VOLATILE_NC_ATTRS` come from it by import.

P3-1 layer (dev/milestones/p31/experiment-structure-design.md §6a, commit 5):

- **Path map** -- an ORDERED list of directory-prefix rewrite rules on
  project-root-relative paths (NOT a per-file table; the prefix form also covers
  in-toml pointer targets that are `temp()`-deleted and exist in neither tree).
  Ref (old-layout) relpaths are translated old->new before pairing with the
  current tree, so a pure move is content-diffed instead of degrading to a
  MISSING+EXTRA pair.
- **Allowlist gate contract (risk-4)** -- after translation and content-diffing
  of translated pairs, the residual MISSING and EXTRA sets must be EMPTY modulo
  an explicitly enumerated allowlist (each entry justified in
  dev/milestones/p31/migration_experiment-structure.md). A nonempty unexplained
  MISSING/EXTRA is a gate FAILURE, not a pass.
- **Path-aware toml comparator (§6a step 3, ext1-3)** -- for each path-valued
  toml field: (1) lexical resolve against its own toml's dir (normpath+join,
  never `.resolve()`); (2) translate to project-root-relative by stripping that
  side's root; (3) apply the prefix map to the REF side's target; (4) compare --
  equal => the pointer move is behavior-neutral (PASS), different => a real
  failure naming the field (a mis-repoint is caught, not hidden).

CLI (self-contained; no snakemake global)::

    python dev/scripts/semantic_tree_diff.py --ref <dir> --cur <dir> [--tolerance 1e-9]
        [--experiment-name experiment] [--dataset-key era5_20000101_20201231]
        [--no-path-map] [--allow <relpath> ...]

Exit 0 = clean (every file equal under its comparator, residual MISSING/EXTRA
empty modulo the allowlist), 1 = at least one FAIL or unexplained
missing/extra file. A clean self-comparison (`--ref X --cur X`) is the smoke.

Path-map falsifier mode (R09 phase 1) -- no trees, one path list::

    python dev/scripts/semantic_tree_diff.py --check-map <pathlist> \
        --milestone r09 --experiment-name experiment \
        --dataset-key era5_20000101_20201231 [--r09-gap-rules]

Classifies every project-relative path as MOVED / IDENTITY / DELETED /
UNMAPPED and exits 1 if anything is UNMAPPED. This is what makes the claim
"the map covers every artifact" testable: `apply_path_map` alone cannot
distinguish an identity rule from a fall-through, so `apply_path_map_matched`
reports whether a rule actually fired.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

try:  # tomllib is stdlib >=3.11; the pixi env is 3.12
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib  # type: ignore

# Reuse check_baseline.py comparators by import; NEVER edit that file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_baseline as cb  # noqa: E402

# The package is the source of truth for what WE write, so the attr set below
# is imported rather than restated -- one definition, as with the leaf set.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blueearth_cst.projections.series_identity import (  # noqa: E402
    INHERITED_SINGLE_SOURCE_ATTRS,
)

VOLATILE_NC_ATTRS = cb.VOLATILE_NC_ATTRS

#: Path classes whose files carry CMIP6 global attrs inherited from ONE member
#: of a multi-variable merge. SCOPED, not folded into VOLATILE_NC_ATTRS: that
#: frozenset is global to every netCDF comparison in both tools, and masking
#: `variable_id` everywhere would drop it from artifacts where it does describe
#: the file. Here it is dropped only where it provably cannot (R9 P2 F4).
#:
#: Needed even after the writers stopped emitting these attrs, and that is the
#: point: every reference tree recorded before that fix still carries them, so a
#: new-vs-old comparison would report them present on one side and absent on the
#: other. Retire this only once no reference tree in use predates the fix.
_INHERITED_ATTR_PATH_MARKERS = (
    "climate_projections/cmip6/raw/",
    "climate_projections/cmip6/scalar/",
    "climate/projections/cmip6/raw/",
    "climate/projections/cmip6/scalar/",
)


def _volatile_attrs_for(*paths: str) -> frozenset:
    """Volatile attrs for a comparison, widened for the CMIP6 merge classes.

    Widened if EITHER side is in the class, since the whole point is comparing a
    post-fix tree against a pre-fix reference.
    """
    joined = " ".join(p.replace("\\", "/") for p in paths)
    if any(marker in joined for marker in _INHERITED_ATTR_PATH_MARKERS):
        return VOLATILE_NC_ATTRS | INHERITED_SINGLE_SOURCE_ATTRS
    return VOLATILE_NC_ATTRS

# ---------------------------------------------------------------------------
# Copied-config normalize map (ext2-01). The documented old->new path map that
# commit 2 rewrote INSIDE every orchestration config -- so the copied snapshot
# under {project_dir}/config/ legitimately differs from a pre-R6 recording only
# in exactly these path values. FOUR keys (data_sources_climate is included --
# the design's 3-key list predates the as-built inventory, which also rewrote
# data_sources_climate; see commit 2). Any OTHER difference is a real FAIL.
#
# Each entry: config-key -> {old-path-value: new-path-value, ...}. Only an
# exact OLD-value match is normalized; any other value is left untouched and
# will fail the equality step.
COPIED_CONFIG_PATH_MAP: dict[str, dict[str, str]] = {
    "data_sources": {
        "config/deltares_data.yml": "config/catalogs/deltares_data.yml",
        "config/deltares_data_linux.yml": "config/catalogs/deltares_data_linux.yml",
        "config/deltares_data_climate_projections.yml":
            "config/catalogs/deltares_data_climate_projections.yml",
        "config/deltares_data_climate_projections_linux.yml":
            "config/catalogs/deltares_data_climate_projections_linux.yml",
        "config/cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    "data_sources_climate": {
        "config/cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    "model_build_config": {
        "config/wflow_build_model.yml": "config/templates/wflow_build_model.yml",
    },
    "waterbodies_config": {
        "config/wflow_update_waterbodies.yml":
            "config/templates/wflow_update_waterbodies.yml",
    },
    # --- R07 additions (migration_project-layout.md §2d) -------------------
    # Without these the phase-B gate goes red on the copied config snapshots
    # for pure path bookkeeping, which is indistinguishable from a real
    # content regression (repo-6, arch-11a).
    "project_dir": {
        "examples/test_local": "test_case/test_local",   # O-20
        "examples/Gabon": "test_case/gabon",             # O-20
        # O-21 retargets snake_config.template.yml to an outside-the-tree
        # placeholder in commit 6. That template is not a copied *snapshot*
        # (no run writes it into project_dir), so it needs no entry here;
        # add one only if a snapshot of it ever appears in a reference tree.
    },
    # O-01 retires the tracked data/ tree; both observation keys fall back to
    # the "None" STRING sentinel (unquoted None in YAML -> Python str, never
    # YAML null -- the existence guards downstream depend on that).
    "output_locations": {
        "data/observations/output-locations-test.csv": "None",
    },
    "observations_timeseries": {
        "data/observations/observations_timeseries_test.csv": "None",
    },
    # B2 (commit 8) moved the wflow forcing into the engine subtree, so the
    # GENERATED forcing build config carries a new pointer. It lives under
    # config/generated/, which `_is_copied_config` sweeps, so it is normalized
    # here rather than allowlisted -- the value change is documented and
    # expected, and normalizing keeps a REAL content regression detectable.
    "input.path_forcing": {
        "../climate_historical/wflow_data/inmaps_historical.nc":
            "forcing/inmaps_historical.nc",
    },
}

# Directories whose contents are compared as ABSENT (non-deterministic wall
# times / timestamps / snakemake metadata) -- never byte-diffed (design §9).
EXCLUDED_DIR_NAMES = frozenset({"logs", "benchmarks", ".snakemake"})

DEFAULT_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# P3-1 path map (design §6a step 2, ext1-3). An ORDERED list of rewrite rules
# on project-root-relative POSIX paths. Two rule kinds:
#   - directory-prefix rule: old ends with "/" -- rewrites any path under it
#     (load-bearing for temp() targets that exist in NEITHER tree, e.g. the
#     per-realization forcing inmaps consumed by path_forcing);
#   - exact-file rule: old does not end with "/" -- rewrites that one relpath.
# First match wins. Direction is OLD (pre-P3-1 reference) -> NEW (current).
# ---------------------------------------------------------------------------

def build_p31_path_map(
    experiment_name: str, dataset_key: str | None
) -> list[tuple[str, str]]:
    """The P3-1 old->new relocation rules for one experiment (design §6a).

    Covers the five content-bearing relocation classes: results CSVs + the
    experiment subtree (rule 3), the wf3 config snapshot (rule 1), the run-dir
    tomls/output CSVs (rule 2), and the keyed extraction netCDF (rule 4,
    only when the dataset key is known).
    """
    rules: list[tuple[str, str]] = [
        (
            "config/snake_config_climate_experiment.yml",
            f"experiments/{experiment_name}/config/snake_config_climate_experiment.yml",
        ),
        (
            f"hydrology_model/run_climate_{experiment_name}/",
            f"experiments/{experiment_name}/model_runs/",
        ),
        (
            f"climate_{experiment_name}/",
            f"experiments/{experiment_name}/",
        ),
    ]
    if dataset_key:
        rules.append(
            ("climate_historical/raw_data/", f"climate_historical/{dataset_key}/")
        )
    return rules


def build_p31_allowlist(
    experiment_name: str, dataset_key: str | None
) -> list[str]:
    """EXTRA-by-design current-tree relpaths (risk-4 presence exemptions ONLY).

    Justifications live in dev/milestones/p31/migration_experiment-structure.md: the
    per-experiment guard sentinel and the key-level guard artifact are new
    gate outputs with no pre-P3-1 counterpart; neither carries scientific
    content. There is no wf3 plots producer, so nothing is MISSING-by-design.
    """
    allow = [f"experiments/{experiment_name}/.project_consistency_ok"]
    if dataset_key:
        allow.append(f"climate_historical/{dataset_key}/.guard_ok")
    return allow


# ---------------------------------------------------------------------------
# R07 path map / allowlist / merge class (dev/milestones/r07/migration_project-layout.md
# §2a, §2b, §2e, §4 -- that map is the path authority; this is its executable
# form and `check_baseline.TARGETS` is rewritten from the same source).
# ---------------------------------------------------------------------------

def build_r07_path_map(
    experiment_name: str,
    dataset_key: str | None,
    clim_project: str = "cmip6",
) -> list[tuple[str | re.Pattern, str]]:
    """The R07 old->new relocation rules (migration map §2a + §2b).

    Excludes B1's climate-store collapse: that is many-to-one and CANNOT be a
    path-map rule (it would raise the path-map collision `diff_trees` guards
    against). It is declared separately -- see `build_r07_merges`.
    """
    e = experiment_name
    rules: list[tuple[str | re.Pattern, str]] = [
        # -- B9: the project config snapshot splits four ways (commit 10) ----
        ("config/snake_config_model_creation.yml",
         "config/runs/snake_config_model_creation.yml"),
        ("config/snake_config_climate_projections.yml",
         "config/runs/snake_config_climate_projections.yml"),
        ("config/deltares_data.yml", "config/catalogs/deltares_data.yml"),
        ("config/cmip6_data.yml", "config/catalogs/cmip6_data.yml"),
        ("config/wflow_build_model.yml",
         "config/templates/wflow_build_model.yml"),
        ("config/wflow_update_waterbodies.yml",
         "config/templates/wflow_update_waterbodies.yml"),
        # generated at run time, not verbatim template snapshots (P3 rule)
        ("config/wflow_build_model_run.yml",
         "config/generated/wflow_build_model_run.yml"),
        ("config/wflow_build_forcing_historical.yml",
         "config/generated/wflow_build_forcing_historical.yml"),
        # -- B2: wflow forcing into the engine subtree (commit 8) ------------
        ("climate_historical/wflow_data/", "hydrology_model/forcing/"),
        # The run_default TOML is a copy of the model TOML placed one level
        # deeper, but wflow resolves `path_forcing` against the MODEL ROOT, not
        # against the toml's own directory -- which is why the pointer works at
        # runtime (proved: the run completes and discharge is bit-identical).
        # `compare_toml` resolves lexically against the toml's own dir, so for
        # this one file it produces a FICTIONAL target on both sides: pre-R07
        # `hydrology_model/climate_historical/wflow_data/...`, post-R07
        # `hydrology_model/run_default/forcing/...`. Neither exists; they only
        # matched before because both sides were equally fictional. This rule
        # keeps the two namespaces aligned so the comparator still catches a
        # REAL mis-repoint of this pointer instead of failing on the artifact.
        ("hydrology_model/climate_historical/wflow_data/",
         "hydrology_model/run_default/forcing/"),
        # -- B10: wf1 figures leave the project-level plots/ tree (commit 12) -
        # Split by DEPICTED subject (P1), so these are per-file, not a prefix.
        ("plots/wflow_model_performance/precip.png",
         "hydrology_model/forcing/plots/precip.png"),
        ("plots/wflow_model_performance/temp.png",
         "hydrology_model/forcing/plots/temp.png"),
        ("plots/wflow_model_performance/pet.png",
         "hydrology_model/forcing/plots/pet.png"),
        ("plots/wflow_model_performance/hydro_wflow_1.png",
         "hydrology_model/evaluation/plots/hydro_wflow_1.png"),
        ("plots/wflow_model_performance/clim_wflow_1_month.png",
         "hydrology_model/evaluation/plots/clim_wflow_1_month.png"),
        ("plots/wflow_model_performance/clim_wflow_1_year.png",
         "hydrology_model/evaluation/plots/clim_wflow_1_year.png"),
        # depicts the MODEL, not its evaluation
        ("plots/wflow_model_performance/basin_area.png",
         "hydrology_model/plots/basin_area.png"),
        # a CSV leaves plots/ entirely (P1: plots/ holds figures only)
        ("plots/wflow_model_performance/performance_metrics.csv",
         "hydrology_model/evaluation/performance_metrics.csv"),
        # -- S8-04/05/06/07: the result surface moved. Without these rows a
        # whole-tree diff against any pre-S8-04 reference reports ~14 deletions
        # plus ~14 additions instead of comparing element-wise -- the gate still
        # runs but stops discriminating exactly where the most changed.
        (f"climate_projections/{clim_project}/change_factors/annual.csv",
         f"climate_projections/{clim_project}/summary/{clim_project}_change_factors_annual.csv"),
        (f"climate_projections/{clim_project}/change_factors/monthly.csv",
         f"climate_projections/{clim_project}/summary/{clim_project}_change_factors_monthly.csv"),
        (f"climate_projections/{clim_project}/provenance.json",
         f"climate_projections/{clim_project}/summary/provenance.json"),
        (f"climate_projections/{clim_project}/plots/projected_climate_statistics.png",
         f"climate_projections/{clim_project}/plots/{clim_project}_change_factor_cloud.png"),
        # The eight scalar figures: {precipitation,temperature} x {anomaly,monthly}
        # x {abs,anom} -> {precip,temp} x {annual,monthly} x {absolute,change}.
        # "anomaly" was the ANNUAL view, not the anomaly quantity -- which is the
        # contradiction S8-07 fixed, so the mapping is not name-for-name.
        *[
            (f"climate_projections/{clim_project}/plots/{old_var}_{old_view}_projections_{old_q}.png",
             f"climate_projections/{clim_project}/plots/{clim_project}_{new_var}_{new_view}_{new_q}.png")
            for old_var, new_var in (("precipitation", "precip"), ("temperature", "temp"))
            for old_view, new_view in (("anomaly", "annual"), ("monthly", "monthly"))
            for old_q, new_q in (("abs", "absolute"), ("anom", "change"))
        ],
        # -- S8-03: the reduced tier is `scalar/`, not `series/`. A DIRECTORY
        # prefix rule -- the filename grammar is unchanged, so every key maps
        # one-to-one and a pre-rename reference tree still compares element-wise
        # instead of reporting nine deletions and nine additions.
        (f"climate_projections/{clim_project}/series/",
         f"climate_projections/{clim_project}/scalar/"),
        # -- B3: only the THREE summary files move; the PNGs stay (commit 9) -
        (f"climate_projections/{clim_project}/gcm_timeseries.nc",
         f"climate_projections/{clim_project}/timeseries/gcm_timeseries.nc"),
        (f"climate_projections/{clim_project}/annual_change_scalar_stats_summary.nc",
         f"climate_projections/{clim_project}/summary/annual_change_scalar_stats_summary.nc"),
        (f"climate_projections/{clim_project}/annual_change_scalar_stats_summary.csv",
         f"climate_projections/{clim_project}/summary/annual_change_scalar_stats_summary.csv"),
        (f"climate_projections/{clim_project}/annual_change_scalar_stats_summary_mean.csv",
         f"climate_projections/{clim_project}/summary/annual_change_scalar_stats_summary_mean.csv"),
        # -- B7: model_results/ -> indicators/ (commit 11) -------------------
        (f"experiments/{e}/model_results/", f"experiments/{e}/indicators/"),
        # B5 moved the weathergen root down one level. This is a LEAF value
        # inside weathergen_config.yml (generateWeatherSeries.output.path), not
        # a file path -- so it must match the experiment dir EXACTLY. A prefix
        # rule here would swallow the whole experiment subtree; a regex rule
        # uses fullmatch, so it fires on this one value and on no real file.
        (re.compile(rf"experiments/{re.escape(e)}/"),
         f"experiments/{e}/weather_generator/"),
        # -- B5/B6: the experiment splits into two engine subtrees (commit 11)
        # realization_<r>/ dissolves; the index migrates from the FILENAME into
        # a DIRECTORY for the wflow-side artifacts, so these must be regexes.
        (re.compile(rf"experiments/{re.escape(e)}/realization_(\d+)/"
                    rf"inmaps_rlz_\1_cst_(\d+)\.nc"),
         rf"experiments/{e}/hydrology_runs/rlz_\1/forcing/inmaps_cst_\2.nc"),
        (re.compile(rf"experiments/{re.escape(e)}/realization_(\d+)/"
                    rf"weathergen_config_rlz_\1_cst_(\d+)\.yml"),
         rf"experiments/{e}/weather_generator/_work/"
         rf"weathergen_config_rlz_\1_cst_\2.yml"),
        # `(.*)`, not `(.+)`: the same rule has to translate the BARE directory
        # string `experiments/<id>/realization_<r>/`, which is what the
        # per-member weagen configs carry as `imposeClimateChanges.output.path`
        # and which `compare_yaml`'s cross-root leaf normalization feeds through
        # this map. With `(.+)` that leaf maps to itself and reads as a content
        # regression.
        (re.compile(rf"experiments/{re.escape(e)}/realization_(\d+)/(.*)"),
         rf"experiments/{e}/weather_generator/output/\2"),
        (re.compile(rf"experiments/{re.escape(e)}/model_runs/"
                    rf"wflow_sbm_rlz_(\d+)_cst_(\d+)\.toml"),
         rf"experiments/{e}/hydrology_runs/rlz_\1/config/cst_\2.toml"),
        (re.compile(rf"experiments/{re.escape(e)}/model_runs/"
                    rf"output_rlz_(\d+)_cst_(\d+)\.csv"),
         rf"experiments/{e}/hydrology_runs/rlz_\1/output/cst_\2.csv"),
        (re.compile(rf"experiments/{re.escape(e)}/model_runs/"
                    rf"outstates_rlz_(\d+)_cst_(\d+)\.nc"),
         rf"experiments/{e}/hydrology_runs/rlz_\1/output/outstates_cst_\2.nc"),
        # cst_*.csv is RETAINED under _work/, not deleted -- it is the only
        # record of precip_variance and of monthly structure (B6 note).
        (f"experiments/{e}/stress_test/",
         f"experiments/{e}/weather_generator/_work/"),
        (f"experiments/{e}/weathergen_config.yml",
         f"experiments/{e}/weather_generator/config/weathergen_config.yml"),
        (f"experiments/{e}/resampled_dates.csv",
         f"experiments/{e}/weather_generator/output/resampled_dates.csv"),
        (f"experiments/{e}/sim_dates.csv",
         f"experiments/{e}/weather_generator/output/sim_dates.csv"),
    ]
    for png in ("obs_power_spectra", "warm_annual_precip",
                "warm_annual_stats", "warm_annual_wavelet"):
        rules.append((f"experiments/{e}/{png}.png",
                      f"experiments/{e}/weather_generator/plots/{png}.png"))
    return rules


def build_r07_allowlist(
    experiment_name: str, dataset_key: str | None
) -> list[str]:
    """EXTRA-by-design relpaths for R07 (migration map §4).

    A FULL set per gate invocation, not an increment: R07 retires none of
    P3-1's entries, so they are carried forward here. Every entry is
    justified in the migration map; an entry not listed fails the gate.
    """
    allow = list(build_p31_allowlist(experiment_name, dataset_key))
    # Rule 1.03's completion sentinel (R7-1). A gate artifact with no
    # scientific content, same class as P3-1's .guard_ok and
    # .project_consistency_ok: it exists so a rebuild of the model re-fires the
    # rules that write wflow_sbm.toml in place. No pre-R07 counterpart.
    allow.append("hydrology_model/.model_built")
    if dataset_key:
        # B1's second declared output -- the model-free delineation the store
        # bbox came from; no pre-R07 counterpart.
        allow.append(f"climate_historical/{dataset_key}/store_region.geojson")
        # B4's new producer (rule 1.15). Additive; source-grid PET did not
        # previously exist. Named source_* so a file copied out of its
        # directory is still distinguishable from the model-grid figures.
        allow += [
            f"climate_historical/{dataset_key}/plots/source_precip.png",
            f"climate_historical/{dataset_key}/plots/source_temp.png",
            f"climate_historical/{dataset_key}/plots/source_pet.png",
        ]
    return allow


def build_r07_merges(
    dataset_key: str | None, clim_source: str | None = None
) -> list[tuple[str, list[str]]]:
    """B1's declared many-to-one collapse (migration map §2e).

    Returns (survivor_relpath, [source_relpath, ...]) in the CURRENT and
    REFERENCE namespaces respectively. The survivor is compared against EVERY
    source and the merge passes only if ALL comparisons pass -- allowlisting
    one side as MISSING was rejected, because it lets the gate go green while
    proving nothing about the store that disappeared.
    """
    if not dataset_key:
        return []
    merges = [(
        f"climate_historical/{dataset_key}/extract_historical.nc",
        [
            "climate_historical/wf1_raw/extract_historical.nc",
            f"climate_historical/{dataset_key}/extract_historical.nc",
        ],
    )]
    if clim_source in ("chirps", "chirps_global"):
        # The sidecar exists only on this branch, and the two stores name it
        # differently -- the collapse standardises on the clim_source-
        # independent `orography.nc` (repo-1).
        merges.append((
            f"climate_historical/{dataset_key}/orography.nc",
            [
                "climate_historical/wf1_raw/orography.nc",
                f"climate_historical/{dataset_key}/{clim_source}_orography.nc",
            ],
        ))
    return merges


# ---------------------------------------------------------------------------
# R09 path map (dev/milestones/r09/migration_project-tree.md -- that map is the
# path authority; this is its executable form). Target tree:
# dev/milestones/r09/project-tree-design.md v10.
#
# Direction is the INVERSE of R07's: R07 moved the realization index out of the
# filename and into a directory (`rlz_<r>/output/cst_<c>.csv`); R09 moves it
# back into the filename (`output/rlz_<r>_cst_<c>.csv`) and drops the per-
# realization directory level.
#
# RULE ORDER IS NOT SECTION ORDER. The map doc groups its tables by destination
# root because that is how a human reads them; `apply_path_map` is first match
# wins, so the rules below are registered NARROWER SOURCE PATTERN FIRST (map
# doc, *Rule precedence -- the tables are not the rule order*).
#
# IDENTITY IS ENUMERATED PER ROW, never as a catch-all prefix. A broad
# `config/` -> `config/` rule would satisfy every `config/` row at once and
# empty the unmapped-path report by construction, since a fall-through and an
# identity match would then look the same to `apply_path_map_matched`.
# ---------------------------------------------------------------------------

def build_r09_path_map(
    experiment_name: str,
    dataset_key: str | None = None,
    clim_project: str = "cmip6",
) -> list[tuple[str | re.Pattern, str]]:
    """The R09 old->new relocation rules, one per row of the migration map.

    ONLY rows the map doc states are encoded here. Artifacts the map does not
    cover fall through and are reported as UNMAPPED -- that is the phase-1
    falsifier, and finding one is a finding against the map, not a licence to
    improvise. Candidate additions live in `build_r09_gap_rules`, which the
    caller opts into explicitly.

    Parameters
    ----------
    experiment_name : str
        `workflows.climate_experiment.experiment_name` -- the `<id>` in
        `experiments/<id>/`.
    dataset_key : str | None
        The historical climate-store key, `<clim_source>_<start>_<end>`. Only
        registers a narrower keyed prefix ahead of the generic store rule; the
        map is complete without it (the row is keyed by a variable).
    clim_project : str
        `workflows.climate_projections.clim_project` -- the subdirectory under
        `climate_projections/`.
    """
    e = experiment_name
    exp = re.escape(e)
    cp = clim_project
    ident: list[tuple[str | re.Pattern, str]] = []

    rules: list[tuple[str | re.Pattern, str]] = [
        # -- 1. The named precedence hazard, registered first -----------------
        # `config/generated/*` appears twice in the map doc -- once under its
        # destination root and once under its source root -- precisely to signal
        # that it must precede the `config/**` rows. A LATENT hazard in this
        # encoding (there is no `config/` catch-all below, by design), but the
        # order is kept so that widening a later rule cannot silently reopen it.
        #
        # `wflow_build_model_run.yml` had a row here until 2026-08-04. The
        # observed-tier run showed nothing in the codebase writes it any more --
        # only the forcing config is generated -- so the row described an
        # artifact that cannot appear, and the map doc's SECOND named hazard was
        # a hazard about a retired file. Dropped (phase-1 report, G2).
        ("config/generated/wflow_build_forcing_historical.yml",
         "models/hydrology/wflow/config/build_historical_forcing.yml"),
        # Wflow's own `log.txt`, which `[logging] path_log` writes beside the
        # TOML. It must precede any `hydrology_runs/rlz_(\d+)/config/(.*)`
        # rule, which would otherwise consume it and route it to `config/`.
        #
        # THE MEMBER INDEX IS NOT RECOVERABLE FROM THE OLD PATH. One old
        # `log.txt` per realization becomes N per-member logs after P2 sets
        # `path_log` per member, so this row is a one-to-many SPLIT -- the
        # inverse of `build_r07_merges` -- and no path-map rule can express it
        # as a function. The destination therefore keeps the map doc's own
        # `<c>` placeholder verbatim. This never reaches `diff_trees`:
        # `_is_excluded` drops `.log` and `log.txt` before mapping.
        (re.compile(rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/config/log\.txt"),
         rf"experiments/{e}/hydrology/wflow/output/rlz_\1_cst_<c>.log"),

        # -- 2. The index relocation: directory -> filename -------------------
        (re.compile(rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/config/"
                    rf"cst_(\d+)\.toml"),
         rf"experiments/{e}/hydrology/wflow/config/rlz_\1_cst_\2.toml"),
        (re.compile(rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/forcing/"
                    rf"inmaps_cst_(\d+)\.nc"),
         rf"experiments/{e}/hydrology/wflow/forcing/inmaps_rlz_\1_cst_\2.nc"),
        # `outstates_` before the bare output rule: same directory, narrower
        # source pattern.
        (re.compile(rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/output/"
                    rf"outstates_cst_(\d+)\.nc"),
         rf"experiments/{e}/hydrology/wflow/output/outstates_rlz_\1_cst_\2.nc"),
        (re.compile(rf"experiments/{exp}/hydrology_runs/rlz_(\d+)/output/"
                    rf"cst_(\d+)\.csv"),
         rf"experiments/{e}/hydrology/wflow/output/rlz_\1_cst_\2.csv"),

        # -- 3. Rule 3.11's rename, before the `_parts/` identity rows --------
        # `export_wflow_results` -> `derive_wflow_indicators`. Transient parts
        # only; both are merged and deleted every run. Registered at experiment
        # scope (where WF3's parts live) and at project root (the scope the map
        # doc's table writes them at).
        (f"experiments/{e}/logs/_parts/3.11_export_wflow_results.log",
         f"experiments/{e}/logs/_parts/3.11_derive_wflow_indicators.log"),
        (f"experiments/{e}/benchmarks/_parts/3.11_export_wflow_results.tsv",
         f"experiments/{e}/benchmarks/_parts/3.11_derive_wflow_indicators.tsv"),
        ("logs/_parts/3.11_export_wflow_results.log",
         "logs/_parts/3.11_derive_wflow_indicators.log"),
        ("benchmarks/_parts/3.11_export_wflow_results.tsv",
         "benchmarks/_parts/3.11_derive_wflow_indicators.tsv"),

        # -- 4. The experiment result surface + its catalog -------------------
        # The only two `rule all` filename renames in the whole map (naming.md
        # §7 rename record). `indicators/RT_*.csv` is NOT migrated -- see
        # `build_r09_deletions`.
        (f"experiments/{e}/indicators/Qstats.csv",
         f"experiments/{e}/results/q_indicators.csv"),
        (f"experiments/{e}/indicators/basin.csv",
         f"experiments/{e}/results/basin_indicators.csv"),
        (f"experiments/{e}/data_catalog_climate_experiment.yml",
         f"experiments/{e}/config/catalogs/data_catalog_climate_experiment.yml"),

        # -- 5. weather_generator/ -> climate/weathergenr/ --------------------
        # One prefix per map row's directory. These are RELOCATIONS, not
        # identities, so a prefix cannot be confused with a fall-through.
        # `output/` carries both the generated series and the date tables
        # (two map rows, one destination directory).
        (f"experiments/{e}/weather_generator/output/",
         f"experiments/{e}/climate/weathergenr/output/"),
        (f"experiments/{e}/weather_generator/config/",
         f"experiments/{e}/climate/weathergenr/config/"),
        (f"experiments/{e}/weather_generator/_work/",
         f"experiments/{e}/climate/weathergenr/_work/"),
        (f"experiments/{e}/weather_generator/plots/",
         f"experiments/{e}/climate/weathergenr/plots/"),
        # The BARE directory string, registered AFTER the four subdirectory
        # rules so it can only catch what they do not.
        #
        # Not a file path -- it is a directory-valued LEAF inside
        # weathergen_config.yml (`generateWeatherSeries.output.path`), and
        # `compare_yaml`'s cross-root normalization feeds such leaves through
        # this map. Without the rule the leaf falls through unmapped and reads
        # as a content regression on a file whose every other value matches.
        #
        # R07 hit exactly this and carries
        # `test_r07_bare_realization_dir_maps_to_the_generator_output_dir` for
        # it; R9's equivalent was missing until the P2 whole-tree gate found it
        # (phase-2 report F3). A prefix, not a regex: `apply_path_map` matches
        # prefixes on the raw string, so this fires on the bare directory and,
        # harmlessly, on anything else directly beneath it.
        (f"experiments/{e}/weather_generator/",
         f"experiments/{e}/climate/weathergenr/"),
    ]

    # -- 6. experiments/<id>/ identity rows, ONE RULE PER MAP ROW -------------
    for same in (
        f"experiments/{e}/.project_consistency_ok",
        f"experiments/{e}/config/snake_config_climate_experiment.yml",
        f"experiments/{e}/config/catalogs/",   # row: `config/catalogs/*`
        # Row added 2026-08-04 by the P1 falsifier's F1c ruling: WF3 emits an
        # experiment-scoped digest bundle that no row and no design-tree line
        # covered. Ruled toward the code under principle P9.
        f"experiments/{e}/config/runs/",       # row: `config/runs/<workflow>/<digest>/**`
        f"experiments/{e}/logs/",              # rows: `logs/*` + new `logs/dag/`
        f"experiments/{e}/benchmarks/",        # row: `benchmarks/*`
    ):
        ident.append((same, same))

    # -- 7. hydrology_model/ -> models/hydrology/wflow/ -----------------------
    wflow = "models/hydrology/wflow"
    rules += [
        (f"hydrology_model/{leaf}", f"{wflow}/{leaf}")
        for leaf in (
            "staticmaps.nc",
            "wflow_sbm.toml",
            "hydromt.log",
            "hydromt_data.yml",
            ".model_built",          # sentinel rule
            ".outputs_configured",
            # NO ROW for ADR 0004's `.model_final`, deliberately. This list is
            # pre-R09 -> post-R09 RELOCATION, and `.model_final` has no pre-R09
            # form: any tree carrying it is already post-R09, so a row here
            # could never fire. It surfaces instead as an EXTRA in a whole-tree
            # diff against a pre-ADR-0004 reference, which the r09 milestone
            # handles through `--allow` (there is no build_r09_allowlist).
            "forcing/inmaps_historical.nc",
            "plots/basin_area.png",
            "plots/basin_area.pdf",
        )
    ]
    rules += [
        # `forcing/plots/` before `forcing/`'s file row is immaterial (they are
        # disjoint), but the directory rows below are each a `*`/`**` row in the
        # map doc, so a prefix IS the faithful per-row encoding.
        ("hydrology_model/staticgeoms/", f"{wflow}/staticgeoms/"),
        ("hydrology_model/forcing/plots/", f"{wflow}/forcing/plots/"),
        ("hydrology_model/run_default/", f"{wflow}/run_default/"),
        ("hydrology_model/evaluation/", f"{wflow}/evaluation/"),
    ]

    # -- 8. data/ -------------------------------------------------------------
    rules += [
        (f"spatial/{leaf}", f"data/spatial/{leaf}")
        for leaf in (
            "spatial_maps.nc",
            "spatial_catalog.yml",
            "spatial_report.yml",
            "location_registry.csv",
            # ADR 0003 §8a, added 2026-08-06. The SEAM INTERMEDIATE between
            # rule 1.01c (the vector foundation, shared by all three workflows)
            # and rule 1.02 (the thematic stack, WF1-only): the hydrography
            # grid stack that used to cross that boundary in memory.
            #
            # It has no pre-move counterpart -- the same situation as
            # `region.geojson`, which the F1a ruling of 2026-08-04 covered by
            # turning the geoms row into a DIRECTORY row. A flat file under
            # `spatial/` has no such row to widen, so it is enumerated here
            # instead. Note it is NOT in `spatial_catalog.yml` and never should
            # be: covered by the path map is not the same as advertised as a
            # product (tests/test_spatial_products.py pins the exclusion).
            "hydrography.nc",
        )
    ]
    # A DIRECTORY row since the F1a ruling of 2026-08-04: it enumerated the five
    # layers rule `prepare_spatial_maps` writes, which missed `region.geojson`
    # from rule `delineate_region` (ADR 0003). A sixth layer would have reopened
    # the same gap, so the row is now the directory.
    rules.append(("spatial/geoms/", "data/spatial/geoms/"))
    if dataset_key:
        # Narrower than the generic store rule below, so it is registered first.
        rules.append((f"climate_historical/{dataset_key}/",
                      f"data/climate/historical/{dataset_key}/"))
    # The store key is RETAINED (map doc, Finding 3): it is a cache key, not
    # multi-window support. The row is keyed by a variable, so the rule is too;
    # this covers extract_historical.nc, store_region.geojson, plots/* and
    # .guard_ok in one go, which is exactly the four rows' shared destination.
    rules.append((re.compile(r"climate_historical/([^/]+)/(.*)"),
                  r"data/climate/historical/\1/\2"))
    proj = f"data/climate/projections/{cp}"
    rules += [
        (f"climate_projections/{cp}/raw/", f"{proj}/raw/"),
        (f"climate_projections/{cp}/scalar/", f"{proj}/scalar/"),
        (f"climate_projections/{cp}/summary/", f"{proj}/summary/"),
        (f"climate_projections/{cp}/plots/", f"{proj}/plots/"),
        (f"climate_projections/{cp}/report.md", f"{proj}/report.md"),
    ]

    # -- 8b. the wrapper's invocation manifest --------------------------------
    # Added 2026-08-05. `scripts/run_workflows.py` wrote one immutable manifest
    # per invocation to `provenance/runs/`, a SEVENTH project root that no R9
    # instrument could see: the declared tier reads the Snakefiles' `output:`
    # declarations and the wrapper is not a rule, the observed tier came from
    # direct `snakemake` invocations so the wrapper never ran, and the
    # whole-tree diff compared two trees that both lacked it. The follow-up
    # ruled it under `config/runs/`, where the per-run generated provenance
    # already lives; see the destination row in section 9.
    rules.append(("provenance/runs/", "config/runs/invocations/"))

    # -- 9. config/ identity rows, ONE RULE PER MAP ROW -----------------------
    # The two `config/runs/snake_config_*.yml` entries are CONTRACT PATHS --
    # declared inputs of WF3's rule 3.00b drift guard -- which is why option (A)
    # kept the snapshot under `config/` at all (map doc, Finding 1).
    for same in (
        "config/runs/snake_config_model_creation.yml",
        "config/runs/snake_config_climate_projections.yml",
        "config/catalogs/",              # row: `config/catalogs/*.yml`
        "config/templates/",             # row: `config/templates/*.yml`
        "config/observations/",          # row: `config/observations/*`
    ):
        ident.append((same, same))
    # Row: `config/runs/invocations/**`, added 2026-08-05 by the R9 follow-up
    # that moved the wrapper's invocation manifest off its own `provenance/`
    # root (relocation rule in section 8b). Registered BEFORE the workflow
    # regex below even though both would yield identity: `invocations` is NOT a
    # workflow, and it only matches `[a-z_]+` by coincidence. Tightening that
    # regex to the three real workflow names -- a reasonable future edit --
    # would otherwise drop this path to UNMAPPED with nothing to say why.
    ident.append(("config/runs/invocations/", "config/runs/invocations/"))
    # Row: `config/runs/<workflow>/<digest>/**`. Generalised from
    # `model_creation` by the F1b ruling of 2026-08-04 — WF2 emits
    # `climate_projections/<digest>/` from the same producer class, and design
    # tree v10 already reads `<workflow>`. A regex, not a `config/runs/` prefix:
    # the prefix form would also swallow the two `snake_config_*.yml` CONTRACT
    # PATHS above, collapsing three enumerated rows into one catch-all. The
    # pattern needs the second `/`, so those two files cannot match it.
    ident.append((re.compile(r"(config/runs/[a-z_]+/.*)"), r"\1"))

    # -- 10. project root identity rows ---------------------------------------
    ident += [
        ("logs/_parts/", "logs/_parts/"),
        ("logs/dag/", "logs/dag/"),
        (re.compile(r"(logs/wf[12]_[^/]*\.log)"), r"\1"),
        ("benchmarks/_parts/", "benchmarks/_parts/"),
        (re.compile(r"(benchmarks/wf[12]_benchmarks\.md)"), r"\1"),
    ]
    return rules + ident


#: Candidate map rows for artifacts the migration map does not cover, found by
#: applying `build_r09_path_map` to an inventory. Kept OUT of the map so the
#: falsifier reports them; amending the map is an owner decision (phase-1 brief,
#: *Task constraints*). Each entry: (artifact, producing rule, authority).
#:
#: **EMPTY, and that is a result rather than an absence.** Five candidates were
#: raised and all five are closed:
#:
#: * three declared-tier gaps -- `spatial/geoms/region.geojson`,
#:   `config/runs/climate_projections/<digest>/`, and WF3's experiment-scoped
#:   bundle -- ruled by the owner on 2026-08-04 (phase-1 report F1a-F1c). The
#:   migration map was amended and the rules moved into `build_r09_path_map`;
#: * two more -- `hydrology_model/instate/` and a directory-wide
#:   `hydrology_model/plots/` -- were inferred from the design tree and never
#:   observed. The 2026-08-04 observed-tier run settled both NEGATIVELY:
#:   `instate/` does not exist, and `plots/` holds exactly
#:   `basin_area.{png,pdf}`, so the map's two-file row is right as written.
#:   Their rules matched nothing and were removed (phase-1 report F2).
#:
#: The mechanism is kept, empty, because the next inventory may raise a sixth:
#: it is what lets the falsifier report "N unmapped" and "0 once accepted" as
#: two numbers instead of quietly reporting 0.
R09_MAP_GAPS: tuple[tuple[str, str, str], ...] = ()


def build_r09_gap_rules(
    experiment_name: str,
) -> list[tuple[str | re.Pattern, str]]:
    """Proposed rules for the artifacts in `R09_MAP_GAPS` -- OPT-IN, now empty.

    APPENDED to `build_r09_path_map`, never interleaved: any rule here must be
    either disjoint from, or strictly broader than, every map rule, so appending
    cannot change how a map row resolves. `test_r09_path_map.py` pins that
    property against the declared-tier inventory.

    `experiment_name` is unused now that every candidate is closed; the
    parameter is kept so the call signature does not change under the caller,
    and so a future experiment-scoped candidate needs no signature change.
    """
    del experiment_name
    return []


def build_r09_deletions(experiment_name: str) -> list[re.Pattern]:
    """Paths the migration deliberately does NOT carry forward.

    `indicators/RT_*.csv` is deleted, not migrated (map doc, v2 decision 3).
    Encoding that as a path-map rule would invent a destination; classifying it
    separately keeps the row covered without polluting the map.
    """
    return [re.compile(rf"experiments/{re.escape(experiment_name)}/"
                       rf"indicators/RT_.*\.csv")]


def classify_path_map(
    paths, path_map, deleted: list[re.Pattern] | None = None
) -> list[tuple[str, str, str]]:
    """Classify every path as MOVED / IDENTITY / DELETED / UNMAPPED.

    IDENTITY means a rule fired and resolved the path to itself -- a
    deliberately unchanged artifact. UNMAPPED means no rule fired. The two are
    the same STRING and are only distinguishable through
    `apply_path_map_matched`, which is the whole reason that sibling exists.
    """
    deleted = list(deleted or [])
    out: list[tuple[str, str, str]] = []
    for rel in paths:
        rel = rel.replace("\\", "/")
        if any(p.fullmatch(rel) for p in deleted):
            out.append((rel, "", "DELETED"))
            continue
        new, matched = apply_path_map_matched(rel, path_map)
        if not matched:
            out.append((rel, new, "UNMAPPED"))
        elif new == rel:
            out.append((rel, new, "IDENTITY"))
        else:
            out.append((rel, new, "MOVED"))
    return out


def format_path_map_report(rows: list[tuple[str, str, str]]) -> str:
    """Render `classify_path_map` output: the old->new table, then the counts."""
    lines = [f"{kind:<8} {old}" + (f"  ->  {new}" if kind == "MOVED" else "")
             for old, new, kind in rows]
    counts = {k: sum(1 for _, _, kind in rows if kind == k)
              for k in ("MOVED", "IDENTITY", "DELETED", "UNMAPPED")}
    unmapped = counts["UNMAPPED"]
    lines.append("")
    lines.append(
        f"{'MAP CLEAN' if not unmapped else 'UNMAPPED PATHS'}: {len(rows)} paths, "
        f"{counts['MOVED']} moved, {counts['IDENTITY']} identity (by rule), "
        f"{counts['DELETED']} deleted-by-design, {unmapped} unmapped"
    )
    return "\n".join(lines)


def apply_path_map_matched(
    rel: str, path_map: list[tuple[str | re.Pattern, str]] | None
) -> tuple[str, bool]:
    """`apply_path_map`, plus WHETHER a rule fired -- the R9 falsifier's basis.

    `apply_path_map` returns its input unchanged both when an identity rule
    fires and when nothing matches, so a fall-through is indistinguishable from
    a deliberate non-move. The R9 phase-1 brief authorizes this sibling because
    that ambiguity makes the property "the map covers every artifact"
    inexpressible: a map with NO rules would report every path as mapped.

    This function is the implementation; `apply_path_map` is a thin
    `[0]` projection of it, so the two can never drift apart. Behaviour for
    every existing caller (`build_r07_path_map` users, `compare_yaml`,
    `_normalize_tree_root_paths`) is unchanged bit-for-bit, including the
    backslash normalization applied to `rel` before matching.

    Returns
    -------
    (translated, matched)
        `matched` is False for an empty/None map and for a fall-through; True
        only when one of the three rule kinds actually fired.
    """
    rel = rel.replace("\\", "/")
    if not path_map:
        return rel, False
    for old, new in path_map:
        if isinstance(old, re.Pattern):
            m = old.fullmatch(rel)
            if m:
                return m.expand(new), True
        elif old.endswith("/"):
            if rel.startswith(old):
                return new + rel[len(old):], True
        elif rel == old:
            return new, True
    return rel, False


def apply_path_map(
    rel: str, path_map: list[tuple[str | re.Pattern, str]] | None
) -> str:
    """Translate one project-root-relative path through the ordered rule list.

    Three rule kinds, first match wins:
      - regex rule: `old` is a compiled pattern -- `new` is an expansion
        template (`\\1` backrefs). R07 needs this for B5, where the
        realization index migrates from the FILENAME into a DIRECTORY
        (`realization_2/inmaps_rlz_2_cst_3.nc` ->
        `hydrology_runs/rlz_2/forcing/inmaps_cst_3.nc`), which neither a
        prefix nor an exact rule can express;
      - directory-prefix rule: `old` ends with "/";
      - exact-file rule: otherwise.

    A path no rule matches is returned unchanged. When the caller needs to tell
    that apart from an identity rule, use `apply_path_map_matched`.
    """
    return apply_path_map_matched(rel, path_map)[0]


# ---------------------------------------------------------------------------
# Element-wise numeric tolerance (ext2-02). Distinct from check_baseline's
# _within_tol, which returns False for tol<=0; here tol==0 means EXACT (plain
# ==), and tol>0 uses the same relative rule vectorized over arrays.
# ---------------------------------------------------------------------------

def _values_within_tol(ref: np.ndarray, cur: np.ndarray, tol: float) -> np.ndarray:
    """Element-wise boolean mask of |c-r| / max(|r|,|c|,1e-300) <= tol.

    Applied only to the finite (non-NaN) positions; NaN masks are checked
    separately by the caller. tol==0 -> exact equality.
    """
    if tol <= 0:
        return ref == cur
    denom = np.maximum.reduce([np.abs(ref), np.abs(cur), np.full(ref.shape, 1e-300)])
    return np.abs(cur - ref) / denom <= tol


def _compare_array(name: str, ref: np.ndarray, cur: np.ndarray, tol: float) -> list[str]:
    """Positional (NO realignment) element-wise compare of two arrays."""
    diffs: list[str] = []
    if ref.shape != cur.shape:
        return [f"{name}: shape {list(cur.shape)} vs {list(ref.shape)}"]
    if ref.dtype != cur.dtype:
        diffs.append(f"{name}: dtype {cur.dtype} vs {ref.dtype}")
    if np.issubdtype(ref.dtype, np.floating) and np.issubdtype(cur.dtype, np.floating):
        ref_nan = np.isnan(ref)
        cur_nan = np.isnan(cur)
        if not np.array_equal(ref_nan, cur_nan):
            pos = np.argwhere(ref_nan != cur_nan)
            first = tuple(int(i) for i in pos[0]) if pos.size else ()
            return diffs + [f"{name}: NaN mask mismatch at {first}"]
        finite = ~ref_nan
        if finite.any():
            ok = _values_within_tol(ref[finite], cur[finite], tol)
            if not ok.all():
                # locate the first offending finite element in flat order
                finite_idx = np.argwhere(finite)
                bad = finite_idx[np.argmin(ok)]
                p = tuple(int(i) for i in bad)
                diffs.append(
                    f"{name}: value out of tolerance at {p} "
                    f"({cur[tuple(bad)]} vs {ref[tuple(bad)]})"
                )
    else:
        # non-float (int / datetime / string coords) -> exact positional equality
        if not np.array_equal(ref, cur):
            pos = np.argwhere(ref != cur)
            first = tuple(int(i) for i in pos[0]) if pos.size else ()
            diffs.append(f"{name}: value mismatch at {first}")
    return diffs


def compare_nc(ref_path: str, cur_path: str, tol: float = DEFAULT_TOLERANCE) -> list[str]:
    """ELEMENT-WISE NetCDF comparator (design §9 ext2-02).

    Dims (names+sizes), coordinate variables (labels AND stored order, no
    realignment), data variables (shape/dtype, exact NaN masks, per-element
    tolerance), and non-volatile attrs. Summary/aggregate stats are NOT an
    equality criterion here.
    """
    diffs: list[str] = []
    volatile = _volatile_attrs_for(ref_path, cur_path)
    with xr.open_dataset(ref_path) as ref, xr.open_dataset(cur_path) as cur:
        # Dimensions
        if dict(ref.sizes) != dict(cur.sizes):
            diffs.append(f"dims {dict(cur.sizes)} vs {dict(ref.sizes)}")
        # Coordinates: identical sets, compared labels+order (no sort/realign)
        if set(ref.coords) != set(cur.coords):
            diffs.append(
                f"coord set {sorted(cur.coords)} vs {sorted(ref.coords)}"
            )
        else:
            for name in sorted(ref.coords):
                diffs += _compare_array(
                    f"coord {name}",
                    np.asarray(ref.coords[name].values),
                    np.asarray(cur.coords[name].values),
                    tol,
                )
        # Data variables: identical sets, element-wise values
        if set(ref.data_vars) != set(cur.data_vars):
            diffs.append(
                f"variable set {sorted(cur.data_vars)} vs {sorted(ref.data_vars)}"
            )
        else:
            for name in sorted(ref.data_vars):
                diffs += _compare_array(
                    f"var {name}",
                    np.asarray(ref[name].values),
                    np.asarray(cur[name].values),
                    tol,
                )
                diffs += _compare_attrs(
                    f"var {name}", ref[name].attrs, cur[name].attrs, volatile
                )
        # Dataset-level attrs
        diffs += _compare_attrs("dataset", ref.attrs, cur.attrs, volatile)
    return diffs


def _compare_attrs(
    scope: str, ref_attrs: dict, cur_attrs: dict, volatile: frozenset | None = None
) -> list[str]:
    volatile = VOLATILE_NC_ATTRS if volatile is None else volatile
    ref_a = {k: str(v) for k, v in ref_attrs.items() if k not in volatile}
    cur_a = {k: str(v) for k, v in cur_attrs.items() if k not in volatile}
    if ref_a != cur_a:
        return [f"{scope} attrs {cur_a} vs {ref_a}"]
    return []


# ---------------------------------------------------------------------------
# TOML: parse-and-normalize structural compare, with the P3-1 path-aware
# pointer-field comparator (design §6a step 3, ext1-3).
# ---------------------------------------------------------------------------

# Path-valued run-toml fields resolved relative to the toml's own directory.
# The three fields the design names as legitimately changing string value are
# path_static / path_forcing / path_input; path_output and csv.path are
# included for the same treatment (their targets moved WITH the run dir, so
# raw strings are unchanged and the normalized compare is equally a PASS).
TOML_PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("input", "path_forcing"),
    ("input", "path_static"),
    ("state", "path_input"),
    ("state", "path_output"),
    ("csv", "path"),
    # R07 B5 correction. `("csv", "path")` above targets the wflow v0 layout;
    # every toml this repo writes on the pinned Wflow.jl carries the output CSV
    # pointer at `[output.csv] path`, so that tuple never resolves and the field
    # silently fell through to the RAW string diff. That was invisible while the
    # value was an unmoved bare filename; B5 moves its target into the run's
    # output/ dir, so without this entry a correct repoint reads as a content
    # regression. Both tuples are kept -- the stale one is inert.
    ("output", "csv", "path"),
)


def _get_nested(doc: dict, keys: tuple[str, ...]):
    node = doc
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _set_nested(doc: dict, keys: tuple[str, ...], value) -> None:
    node = doc
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _project_relative_target(toml_path: str, field_val: str, root: str) -> str:
    """§6a step 3 (1)+(2): lexical resolve against the toml's own dir, then
    strip that side's project root. Pure string arithmetic (normpath+join, NOT
    `.resolve()`), so it works after `temp()` targets are deleted."""
    v = field_val.replace("\\", "/")
    if os.path.isabs(v):
        resolved = os.path.normpath(v)
    else:
        toml_dir = os.path.dirname(os.path.abspath(toml_path))
        resolved = os.path.normpath(os.path.join(toml_dir, v))
    rel = os.path.relpath(resolved, os.path.abspath(root))
    return rel.replace("\\", "/")


def compare_toml(
    ref_path: str,
    cur_path: str,
    ref_root: str | None = None,
    cur_root: str | None = None,
    path_map: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Structural toml compare. When both project roots are given, the known
    path-valued fields are compared in a PROJECT-ROOT-RELATIVE namespace with
    the old->new path map applied to the ref side (§6a step 3): equal mapped
    targets => behavior-neutral pointer move (PASS); different => a real
    failure naming the field. Without roots: raw parsed-dict equality."""
    with open(ref_path, "rb") as f:
        ref = tomllib.load(f)
    with open(cur_path, "rb") as f:
        cur = tomllib.load(f)

    diffs: list[str] = []
    if ref_root is not None and cur_root is not None:
        for field in TOML_PATH_FIELDS:
            rv = _get_nested(ref, field)
            cv = _get_nested(cur, field)
            if not (isinstance(rv, str) and isinstance(cv, str)):
                continue  # absent on a side -> handled by the raw dict diff
            ref_target = _project_relative_target(ref_path, rv, ref_root)
            cur_target = _project_relative_target(cur_path, cv, cur_root)
            mapped_ref = apply_path_map(ref_target, path_map)  # step 3
            dotted = ".".join(field)
            if mapped_ref != cur_target:  # step 4
                diffs.append(
                    f"{dotted}: project-relative target {cur_target!r} vs ref "
                    f"{ref_target!r} (mapped -> {mapped_ref!r}) -- mis-repoint"
                )
            # Neutralize the field for the raw compare either way: an unequal
            # target is already reported above; an equal one is a PASS.
            _set_nested(ref, field, "<path-field-compared>")
            _set_nested(cur, field, "<path-field-compared>")

    if ref != cur:
        diffs += _dict_diff(ref, cur, prefix="")
    return diffs


def _dict_diff(ref, cur, prefix: str) -> list[str]:
    """First-difference reporter for two parsed structures (dicts/lists/scalars)."""
    if isinstance(ref, dict) and isinstance(cur, dict):
        diffs: list[str] = []
        for k in sorted(set(ref) | set(cur)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in ref:
                diffs.append(f"{p}: new in current")
            elif k not in cur:
                diffs.append(f"{p}: missing in current")
            elif ref[k] != cur[k]:
                diffs += _dict_diff(ref[k], cur[k], p)
        return diffs
    return [f"{prefix or '<root>'}: {cur!r} vs {ref!r}"]


# ---------------------------------------------------------------------------
# Copied-config YAML: normalize-then-compare (ext2-01).
# ---------------------------------------------------------------------------

def _normalize_config_paths(doc):
    """Apply COPIED_CONFIG_PATH_MAP to a parsed config, in place, recursively.

    Only rewrites a key's value when it equals a documented OLD path exactly.
    Any other value is left untouched (and will fail the equality step). Applied
    at any nesting depth so a mapped key inside `project:`/`workflows:` is caught.
    """
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k in COPIED_CONFIG_PATH_MAP and isinstance(v, str):
                doc[k] = COPIED_CONFIG_PATH_MAP[k].get(v, v)
            else:
                _normalize_config_paths(v)
    elif isinstance(doc, list):
        for item in doc:
            _normalize_config_paths(item)
    return doc


def compare_copied_config(ref_path: str, cur_path: str) -> list[str]:
    """Normalize-then-compare for a copied config snapshot (ext2-01).

    Parse both; apply the documented old->new path map to the REFERENCE
    (pre-R6) side; require deep structural equality. Any residual difference --
    an unmapped path, a changed non-path value, a missing/extra key -- FAILs.
    """
    ref = yaml.safe_load(Path(ref_path).read_text())
    cur = yaml.safe_load(Path(cur_path).read_text())
    # Reflexivity guard: identical inputs have no difference by definition. The
    # normalize step is DIRECTIONAL (ref = pre-R6 OLD paths -> NEW), which is not
    # reflexive -- normalizing one side of two identical OLD-path docs would
    # falsely mismatch. This guard makes a self-compare clean without loosening
    # the directional policy for real pre/post comparisons.
    if ref == cur:
        return []
    ref = _normalize_config_paths(ref)
    if ref != cur:
        return _dict_diff(ref, cur, prefix="")
    return []


# ---------------------------------------------------------------------------
# P3-1 commit-5b layer: cross-root path normalization for YAML string leaves.
#
# The milestone diff compares trees generated under DIFFERENT project roots,
# and several wf3-written YAMLs legitimately embed that root inside string
# values: the config snapshots record `project.project_dir` (the root itself),
# the weathergen configs carry root-prefixed output paths, and the experiment
# data catalog carries absolute `uri`s. Under a cross-root comparison every
# such leaf differs by construction -- the same behavior-neutral pointer-move
# class ext1-3 solved for the run tomls, in YAML. Parse-level adjudication of
# the 2026-07-23 milestone diff confirmed ALL leaf diffs in these files are
# path-only (dev/milestones/p31/baseline_diffs.md). Mechanism mirroring the toml
# comparator: each side's own root token becomes `<PROJECT_ROOT>` and the ref
# side's project-relative remainder goes through the old->new path map; equal
# normalized docs => behavior-neutral move (PASS); any non-path leaf diff
# still FAILs.
# ---------------------------------------------------------------------------

def _root_token_variants(root: str, extra: list[str] | None = None) -> list[str]:
    """Forward-slash string forms under which a tree's own project root can
    appear inside a written value: as given, absolute, plus any RECORDED
    tokens supplied by the caller. Longest first so the absolute form wins
    when both would match.

    `extra` exists because a reference tree can be READ from one location
    while the values inside it record a different project_dir -- which is
    exactly what a milestone that renames project_dir produces. R07's O-20
    renamed examples/ to test_case/, so the pre-R07 reference embeds
    `examples/test_local/...` no matter where the tree is now held. Without
    the recorded token, every root-embedded leaf fails the equality step for
    a reason that has nothing to do with the change under test."""
    p = Path(root)
    forms = {p.as_posix()}
    try:
        forms.add(p.resolve().as_posix())
    except OSError:
        pass
    forms.update(t.replace("\\", "/").rstrip("/") for t in (extra or []))
    return sorted(forms, key=len, reverse=True)


def _normalize_path_leaf(
    val: str, variants: list[str], path_map: list[tuple[str, str]] | None
) -> str:
    """Rewrite a string leaf that IS or is PREFIXED BY this side's project
    root; every other leaf is returned untouched (and fails the equality step
    if it diverges). Prefix-or-equality only -- no mid-string rewriting."""
    s = val.replace("\\", "/")
    for v in variants:
        if s == v:
            return "<PROJECT_ROOT>"
        if s.startswith(v + "/"):
            rest = s[len(v) + 1:]
            return "<PROJECT_ROOT>/" + apply_path_map(rest, path_map)
    return val


def _normalize_tree_root_paths(doc, variants, path_map):
    if isinstance(doc, dict):
        return {
            k: _normalize_tree_root_paths(v, variants, path_map)
            for k, v in doc.items()
        }
    if isinstance(doc, list):
        return [_normalize_tree_root_paths(v, variants, path_map) for v in doc]
    if isinstance(doc, str):
        return _normalize_path_leaf(doc, variants, path_map)
    return doc


def compare_yaml(
    ref_path: str,
    cur_path: str,
    rel: Path,
    ref_root: str | None = None,
    cur_root: str | None = None,
    path_map: list[tuple[str, str]] | None = None,
    ref_root_tokens: list[str] | None = None,
) -> list[str]:
    """Structural YAML compare: reflexivity guard, then the R6 directional
    copied-config normalization (config-dir snapshots only), then -- when both
    project roots are known -- the cross-root path-leaf normalization above.
    The path map is applied to the REF side only (old->new direction)."""
    ref = yaml.safe_load(Path(ref_path).read_text())
    cur = yaml.safe_load(Path(cur_path).read_text())
    if ref == cur:
        return []
    if _is_copied_config(rel):
        ref = _normalize_config_paths(ref)
        if ref == cur:
            return []
    if ref_root is not None and cur_root is not None:
        ref = _normalize_tree_root_paths(
            ref, _root_token_variants(ref_root, ref_root_tokens), path_map)
        cur = _normalize_tree_root_paths(cur, _root_token_variants(cur_root), None)
    if ref != cur:
        return _dict_diff(ref, cur, prefix="")
    return []


# ---------------------------------------------------------------------------
# Reused check_baseline.py comparators (imported, unchanged).
# ---------------------------------------------------------------------------

def compare_csv(ref_path: str, cur_path: str) -> list[str]:
    return cb.diff_hashed(cb.fingerprint_csv(ref_path), cb.fingerprint_csv(cur_path))


def compare_png(ref_path: str, cur_path: str) -> list[str]:
    return cb.diff_png(cb.fingerprint_png(ref_path), cb.fingerprint_png(cur_path))


def compare_discharge_csv(ref_path: str, cur_path: str) -> list[str]:
    ref_t, ref_q, _ = cb.read_discharge_series(ref_path)
    cur_t, cur_q, _ = cb.read_discharge_series(cur_path)
    report = cb.compare_discharge(ref_t, ref_q, cur_t, cur_q)
    return [] if report.get("ok") else cb._discharge_report_lines(report)


def compare_geojson(ref_path: str, cur_path: str, tol: float = DEFAULT_TOLERANCE) -> list[str]:
    """Compare two GeoJSON files by GEOMETRY, not by bytes.

    `.geojson` previously fell through to `compare_hashed`, which is
    byte-exact. That is wrong for this format: regenerating an identical
    model re-serializes the vectors with different coordinate formatting, so
    a byte hash reports a difference where the geometry is provably the same.
    Observed at R07 commit 8 -- `staticgeoms/basins.geojson` and
    `meta_basins_highres.geojson` differed in bytes while `geom_equals` was
    True, the symmetric-difference area was exactly 0.0, and both carried the
    same 65 vertices and the same attribute values. The byte hash only ever
    passed before because the reference tree and the current tree were the
    same never-regenerated files.

    Compares CRS, row count, non-geometry columns and their values, and then
    geometry via shapely's `equals` (topological, order-insensitive) with a
    symmetric-difference-area fallback so a shape difference is reported with
    its magnitude rather than as an opaque hash mismatch.
    """
    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover - geopandas is a hard dep here
        return compare_hashed(ref_path, cur_path)

    ref = gpd.read_file(ref_path)
    cur = gpd.read_file(cur_path)
    out: list[str] = []

    if str(ref.crs) != str(cur.crs):
        out.append(f"crs: {ref.crs} vs {cur.crs}")
    if len(ref) != len(cur):
        out.append(f"feature count: {len(ref)} vs {len(cur)}")
        return out

    ref_cols = [c for c in ref.columns if c != "geometry"]
    cur_cols = [c for c in cur.columns if c != "geometry"]
    if ref_cols != cur_cols:
        out.append(f"columns: {ref_cols} vs {cur_cols}")
    else:
        for col in ref_cols:
            if not ref[col].equals(cur[col]):
                out.append(f"column {col!r}: values differ")

    for i, (g_ref, g_cur) in enumerate(zip(ref.geometry, cur.geometry)):
        if g_ref is None or g_cur is None:
            if g_ref is not g_cur:
                out.append(f"feature {i}: one geometry is null")
            continue
        if g_ref.equals(g_cur):
            continue
        area = g_ref.symmetric_difference(g_cur).area
        out.append(
            f"feature {i}: geometry differs "
            f"(symmetric difference area {area:.6g}; ref area {g_ref.area:.6g})"
        )
    return out


def compare_hashed(ref_path: str, cur_path: str) -> list[str]:
    """Fallback for unrecognized extensions: normalized-hash (CRLF-stripped) compare."""
    return cb.diff_hashed(cb.fingerprint_csv(ref_path), cb.fingerprint_csv(cur_path))


# ---------------------------------------------------------------------------
# Walker + dispatch.
# ---------------------------------------------------------------------------

def _is_excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    # Run-log FILES outside the excluded logs/ dirs (hydromt.log, the Wflow
    # run-dir log.txt, run_default/log.txt): same non-content-bearing class as
    # the excluded dirs -- timestamp-laden by nature, never value-comparable.
    return rel.suffix.lower() == ".log" or rel.name == "log.txt"


def _is_copied_config(rel: Path) -> bool:
    """A copied-config snapshot: a YAML directly under a `config/` dir in the tree."""
    return rel.suffix in (".yml", ".yaml") and "config" in rel.parts


def dispatch(
    rel: Path,
    ref_path: str,
    cur_path: str,
    tol: float,
    ref_root: str | None = None,
    cur_root: str | None = None,
    path_map: list[tuple[str, str]] | None = None,
    ref_root_tokens: list[str] | None = None,
) -> list[str]:
    suffix = rel.suffix.lower()
    name = rel.name
    if suffix == ".nc":
        return compare_nc(ref_path, cur_path, tol)
    if suffix == ".toml":
        return compare_toml(ref_path, cur_path, ref_root, cur_root, path_map)
    if suffix in (".yml", ".yaml"):
        return compare_yaml(ref_path, cur_path, rel, ref_root, cur_root,
                            path_map, ref_root_tokens)
    if suffix == ".png":
        return compare_png(ref_path, cur_path)
    if suffix == ".csv":
        if name == "output.csv" and "run_default" in rel.parts:
            return compare_discharge_csv(ref_path, cur_path)
        return compare_csv(ref_path, cur_path)
    if suffix == ".geojson":
        return compare_geojson(ref_path, cur_path, tol)
    return compare_hashed(ref_path, cur_path)


def _list_files(root: Path) -> set[Path]:
    out: set[Path] = set()
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root)
            if not _is_excluded(rel):
                out.add(rel)
    return out


def diff_trees(
    ref_root: str,
    cur_root: str,
    tol: float = DEFAULT_TOLERANCE,
    path_map: list[tuple[str | re.Pattern, str]] | None = None,
    allowlist: list[str] | None = None,
    merges: list[tuple[str, list[str]]] | None = None,
    ref_root_tokens: list[str] | None = None,
    allow_content: list[str] | None = None,
) -> dict:
    """Compare two output trees file-by-file. Returns a report dict with
    `failures` (list of (relpath, [reasons])), `missing`, `extra`, `allowed`,
    `passed`.

    P3-1 semantics (§6a): every ref relpath is translated through `path_map`
    (old->new) before pairing with the current tree, so a mapped move is
    content-diffed (ref bytes vs cur bytes) rather than reported as
    MISSING+EXTRA. Residual MISSING/EXTRA entries matching `allowlist` are
    reported separately as `allowed` and do not fail the gate; any other
    residual entry FAILS it (risk-4)."""
    ref = Path(ref_root)
    cur = Path(cur_root)
    ref_files = _list_files(ref)
    cur_files = _list_files(cur)

    # Declared many-to-one merges are handled out of band: their sources are
    # withheld from `translated` (so they neither collide nor read as MISSING)
    # and their survivor from `raw_extra`. A merge is proven by comparing the
    # survivor against EVERY source -- see the merge block below.
    merges = list(merges or [])
    merge_sources = {src for _, srcs in merges for src in srcs}
    merge_survivors = {survivor for survivor, _ in merges}

    # Translate ref relpaths old->new (POSIX keys); keep the original for I/O.
    translated: dict[str, Path] = {}
    for p in ref_files:
        posix = p.as_posix()
        if posix in merge_sources:
            continue
        key = apply_path_map(posix, path_map)
        if key in translated:  # two ref files mapping onto one target
            raise ValueError(
                f"path map collision: {translated[key]} and {p} both map to "
                f"{key} -- if this is a deliberate many-to-one collapse, "
                f"declare it with --merge {key}={translated[key].as_posix()},{posix}"
            )
        translated[key] = p
    cur_keys = {p.as_posix(): p for p in cur_files}

    allow = set(allowlist or [])
    allow_content_set = set(allow_content or [])
    raw_missing = sorted(set(translated) - set(cur_keys))
    raw_extra = sorted(set(cur_keys) - set(translated) - merge_survivors)
    allowed = sorted(
        [f"MISSING allowed: {k}" for k in raw_missing if k in allow]
        + [f"EXTRA allowed: {k}" for k in raw_extra if k in allow]
    )
    missing = [
        (k if translated[k].as_posix() == k
         else f"{translated[k].as_posix()} (expected at {k})")
        for k in raw_missing if k not in allow
    ]
    extra = [k for k in raw_extra if k not in allow]
    failures: list[tuple[str, list[str]]] = []

    for key in sorted(set(translated) & set(cur_keys)):
        rel_ref = translated[key]
        rel_cur = cur_keys[key]
        reasons = dispatch(
            rel_cur, str(ref / rel_ref), str(cur / rel_cur), tol,
            ref_root=ref_root, cur_root=cur_root, path_map=path_map,
            ref_root_tokens=ref_root_tokens,
        )
        if reasons:
            label = (key if rel_ref.as_posix() == key
                     else f"{rel_ref.as_posix()} -> {key}")
            if key in allow_content_set:
                # An ADJUDICATED content difference: the reference side is
                # known-bad for this file and the exception is written down.
                # Reported, never silent -- a reader of the report sees it.
                allowed.append(f"CONTENT allowed: {label} ({len(reasons)} reason(s))")
            else:
                failures.append((label, reasons))

    # -- Declared merges: the survivor must match EVERY collapsed source -----
    merged: list[str] = []
    n_merge_compared = 0
    for survivor, sources in merges:
        if survivor not in cur_keys:
            failures.append((f"merge {survivor}",
                             [f"survivor missing from current tree: {survivor}"]))
            continue
        for src in sources:
            src_path = Path(src)
            if src_path not in ref_files:
                failures.append(
                    (f"merge {survivor} <- {src}",
                     [f"declared merge source missing from reference tree: {src}"])
                )
                continue
            n_merge_compared += 1
            reasons = dispatch(
                cur_keys[survivor], str(ref / src_path),
                str(cur / cur_keys[survivor]), tol,
                ref_root=ref_root, cur_root=cur_root, path_map=path_map,
                ref_root_tokens=ref_root_tokens,
            )
            if reasons:
                failures.append((f"merge {survivor} <- {src}", reasons))
            else:
                merged.append(f"merge OK: {survivor} <- {src}")

    passed = not (missing or extra or failures)
    return {
        "passed": passed,
        "missing": missing,
        "extra": extra,
        "allowed": allowed,
        "merged": merged,
        "failures": failures,
        "n_compared": len(set(translated) & set(cur_keys)) + n_merge_compared,
    }


def format_report(report: dict) -> str:
    lines: list[str] = []
    for path in report["missing"]:
        lines.append(f"MISSING (in ref, not cur): {path}")
    for path in report["extra"]:
        lines.append(f"EXTRA (in cur, not ref): {path}")
    for entry in report.get("allowed", []):
        lines.append(f"ALLOWED ({entry})")
    for entry in report.get("merged", []):
        lines.append(entry)
    for path, reasons in report["failures"]:
        lines.append(f"FAIL {path}")
        for r in reasons:
            lines.append(f"    - {r}")
    status = "CLEAN" if report["passed"] else "MISMATCH"
    lines.append(
        f"{status}: {report['n_compared']} files compared, "
        f"{len(report['failures'])} failed, {len(report['missing'])} missing, "
        f"{len(report['extra'])} extra, "
        f"{len(report.get('allowed', []))} allowlisted"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", help="reference (pre-move) project_dir tree")
    ap.add_argument("--cur", help="current (post-move) project_dir tree")
    ap.add_argument(
        "--check-map", metavar="PATHLIST",
        help="path-map falsifier mode: read project-relative paths (one per "
             "line, '#' comments ignored) and classify each as MOVED / "
             "IDENTITY / DELETED / UNMAPPED instead of diffing two trees. "
             "Exit 1 if any path is UNMAPPED. --ref/--cur are not used",
    )
    ap.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="relative tolerance for element-wise numeric compare (0 = exact)",
    )
    ap.add_argument(
        "--experiment-name", default="experiment",
        help="experiment_name for the P3-1 path map (default: experiment)",
    )
    ap.add_argument(
        "--dataset-key", default=None,
        help="historical-store dataset key, e.g. era5_20000101_20201231 "
             "(enables the climate_historical/raw_data/ -> <key>/ rule and the "
             ".guard_ok allowlist entry)",
    )
    ap.add_argument(
        "--no-path-map", action="store_true",
        help="disable the P3-1 path map + built-in allowlist (identical-relpath "
             "keying only, the pre-P3-1 behavior)",
    )
    ap.add_argument(
        "--allow", action="append", default=[],
        help="extra allowlisted MISSING/EXTRA relpath (repeatable; every entry "
             "must be justified in the migration note)",
    )
    ap.add_argument(
        "--allow-content", action="append", default=[], metavar="RELPATH",
        help="adjudicated CONTENT difference (repeatable): the file is still "
             "compared and the exception is printed in the report, but it does "
             "not fail the gate. Distinct from --allow, which covers "
             "MISSING/EXTRA presence. Every entry must be justified in the "
             "migration note",
    )
    ap.add_argument(
        "--ref-token", action="append", default=[], metavar="TOKEN",
        help="project_dir token as RECORDED inside the reference tree's own "
             "files, when it differs from where the tree is now read from "
             "(repeatable). R07's fixture rename makes this necessary: the "
             "pre-R07 reference embeds 'examples/test_local' wherever it is "
             "held",
    )
    ap.add_argument(
        "--milestone", choices=("p31", "r07", "r09"), default="p31",
        help="which built-in path map + allowlist to use (default: p31)",
    )
    ap.add_argument(
        "--r09-gap-rules", action="store_true",
        help="append the PROPOSED rules for artifacts the R09 migration map "
             "does not cover (semantic_tree_diff.R09_MAP_GAPS). Default OFF: "
             "the strict map is what the phase-1 falsifier reports against, "
             "and amending the map is an owner decision",
    )
    ap.add_argument(
        "--clim-project", default="cmip6",
        help="clim_project subdir under climate_projections/ (r07 B3 rules)",
    )
    ap.add_argument(
        "--clim-source", default=None,
        help="clim_source, e.g. era5 or chirps; the r07 orography merge is "
             "declared only on the chirps / chirps_global branch",
    )
    ap.add_argument(
        "--map", action="append", default=[], metavar="OLD=NEW",
        help="extra path-map rule, appended after the built-in rules "
             "(repeatable). A trailing '/' on OLD makes it a directory-prefix "
             "rule; otherwise it is an exact-file rule",
    )
    ap.add_argument(
        "--merge", action="append", default=[],
        metavar="SURVIVOR=SRC1,SRC2",
        help="declare a many-to-one collapse (repeatable): SURVIVOR is a "
             "current-tree relpath, SRC* are reference-tree relpaths. The "
             "survivor is compared against EVERY source and the merge passes "
             "only if all comparisons pass",
    )
    args = ap.parse_args(argv)

    extra_rules: list[tuple[str | re.Pattern, str]] = []
    for spec in args.map:
        if "=" not in spec:
            ap.error(f"--map expects OLD=NEW, got: {spec!r}")
        old, new = spec.split("=", 1)
        extra_rules.append((old, new))

    merges: list[tuple[str, list[str]]] = []
    for spec in args.merge:
        if "=" not in spec:
            ap.error(f"--merge expects SURVIVOR=SRC1,SRC2, got: {spec!r}")
        survivor, srcs = spec.split("=", 1)
        sources = [s for s in srcs.split(",") if s]
        if len(sources) < 2:
            ap.error(f"--merge needs at least two sources, got: {spec!r}")
        merges.append((survivor, sources))

    if args.no_path_map:
        path_map, allowlist = (extra_rules or None), list(args.allow)
    elif args.milestone == "r09":
        path_map = build_r09_path_map(
            args.experiment_name, args.dataset_key, args.clim_project
        )
        if args.r09_gap_rules:
            path_map = path_map + build_r09_gap_rules(args.experiment_name)
        path_map += extra_rules
        allowlist = list(args.allow)
    elif args.milestone == "r07":
        path_map = build_r07_path_map(
            args.experiment_name, args.dataset_key, args.clim_project
        ) + extra_rules
        allowlist = build_r07_allowlist(args.experiment_name, args.dataset_key)
        allowlist += list(args.allow)
        merges = build_r07_merges(args.dataset_key, args.clim_source) + merges
    else:
        path_map = build_p31_path_map(
            args.experiment_name, args.dataset_key
        ) + extra_rules
        allowlist = build_p31_allowlist(args.experiment_name, args.dataset_key)
        allowlist += list(args.allow)
    if args.check_map:
        paths = [
            line.strip()
            for line in Path(args.check_map).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        absolute = [p for p in paths
                    if p.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", p)]
        if absolute:
            ap.error(
                f"--check-map expects PROJECT-RELATIVE paths; {len(absolute)} "
                f"absolute path(s) found, first: {absolute[0]!r}"
            )
        deleted = (build_r09_deletions(args.experiment_name)
                   if args.milestone == "r09" and not args.no_path_map else None)
        rows = classify_path_map(paths, path_map, deleted)
        print(format_path_map_report(rows))
        return 0 if not any(kind == "UNMAPPED" for _, _, kind in rows) else 1

    if not (args.ref and args.cur):
        ap.error("--ref and --cur are required unless --check-map is given")
    report = diff_trees(args.ref, args.cur, args.tolerance,
                        path_map=path_map, allowlist=allowlist, merges=merges,
                        ref_root_tokens=list(args.ref_token),
                        allow_content=list(args.allow_content))
    print(format_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
