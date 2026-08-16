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

Path-map falsifier mode -- no trees, one path list::

    python dev/scripts/semantic_tree_diff.py --check-map <pathlist> \
        --experiment-name experiment --dataset-key era5_20000101_20201231

Classifies every project-relative path as MOVED / IDENTITY / DELETED /
UNMAPPED and exits 1 if anything is UNMAPPED. This is what makes the claim
"the map covers every artifact" testable: `apply_path_map` alone cannot
distinguish an identity rule from a fall-through, so `apply_path_map_matched`
reports whether a rule actually fired.

For a CURRENT tree this is not the entry point -- use
`dev/scripts/snapshot_project_tree.py`, which drives `build_project_tree_rules`
(the post-migration inventory) and is what `pixi run tree-check` runs.

**The R07 and R09 one-way migration maps were retired on 2026-08-11**
(`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §6a). They resolved
PRE-migration paths only, both milestones are sealed, and no pre-migration tree
survives to apply them to. Recover either verbatim from its tag -- `r07-layout`,
`r09-project-tree` -- if an archived tree ever needs re-verification. `p31`
stays as the `--milestone` default so a bare invocation, the form AGENTS.md's
validation ladder prescribes, behaves as it always has.
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
        "config/deltares_data_climate_projections.yml": "config/catalogs/deltares_data_climate_projections.yml",
        "config/deltares_data_climate_projections_linux.yml": "config/catalogs/deltares_data_climate_projections_linux.yml",
        "config/cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    "data_sources_climate": {
        "config/cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    # TWO old spellings each: these moved pre-R6 flat -> config/templates/ ->
    # config/defaults/ (2026-08-11 templates/ split). A reference tree recorded
    # in either earlier era must still normalize onto the current path.
    "model_build_config": {
        "config/wflow_build_model.yml": "config/defaults/wflow_build_model.yml",
        "config/templates/wflow_build_model.yml": "config/defaults/wflow_build_model.yml",
    },
    "waterbodies_config": {
        "config/wflow_update_waterbodies.yml": "config/defaults/wflow_update_waterbodies.yml",
        "config/templates/wflow_update_waterbodies.yml": "config/defaults/wflow_update_waterbodies.yml",
    },
    # --- R07 additions (migration_project-layout.md §2d) -------------------
    # Without these the phase-B gate goes red on the copied config snapshots
    # for pure path bookkeeping, which is indistinguishable from a real
    # content regression (repo-6, arch-11a).
    "project_dir": {
        "examples/test_local": "test_case/test_local",  # O-20
        "examples/Gabon": "test_case/gabon",  # O-20
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
        "../climate_historical/wflow_data/inmaps_historical.nc": "forcing/inmaps_historical.nc",
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
            # LEFT is the pre-P3-1 tree and keeps its historical spelling; only
            # RIGHT describes the tree as it is today, so only RIGHT follows the
            # 2026-08-14 workflow rename.
            "config/snake_config_climate_experiment.yml",
            f"experiments/{experiment_name}/config/snake_config_run_stress_test.yml",
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


def build_p31_allowlist(experiment_name: str, dataset_key: str | None) -> list[str]:
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


def build_project_tree_rules(
    experiment_name: str,
    dataset_key: str,
    clim_project: str = "cmip6",
) -> list[tuple[str | re.Pattern, str]]:
    """Identity rules for the CURRENT project tree — the post-R9 inventory.

    **Why this exists, and why it was not the R9 migration map.** That map ran
    ONE WAY: pre-R9 paths to post-R9 ones. A live tree holds only the post-R9
    side, so nothing matched the old-side patterns and every relocated artifact
    fell through as UNMAPPED — `pixi run tree-check` returned exit 1 on every
    CORRECTLY migrated tree (`dev/followups-archive.md` `[R10-11]`; measured 153
    of 186 unmapped on a clean run, and 165 of 203 on a tree that predated the
    artifact people first suspected). The map was never wrong; it was being
    asked about an era that has passed.

    That era has now passed far enough that the map itself is gone — retired
    2026-08-11 with its R07 predecessor, recoverable from tag `r09-project-tree`
    (`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §6a). What remains
    is the question that outlives any migration: **does this tree hold anything
    nobody declared?** That is the property that caught `region.geojson` (R9
    phase-1 F1a) and that the ADR 0003 §8a seam intermediate needed a row for.

    Every rule maps a path to ITSELF, so a covered path classifies as IDENTITY
    and an unknown one still classifies as UNMAPPED. That distinction only
    exists through `apply_path_map_matched`, which is why identity is enumerated
    rather than written as a `data/` → `data/` catch-all: a catch-all would make
    the report empty by construction and the gate would pass unconditionally
    (`test_a_catch_all_config_prefix_would_empty_the_report` demonstrated the
    hazard on the R9 map, and `test_a_broad_data_prefix_would_empty_the_report`
    keeps demonstrating it here).

    DIRECTORY PREFIXES are used only where the contents are genuinely open —
    `plots/`, `staticgeoms/`, a wflow run directory, the projection tiers. Where
    the set of files is fixed, they are enumerated, so a new file in a settled
    directory is reported rather than absorbed.

    Parameters
    ----------
    experiment_name, dataset_key, clim_project
        The same config-derived values the retired R9 map took, so one
        caller can build either.
    """
    e = experiment_name
    exp = re.escape(experiment_name)
    rules: list[tuple[str | re.Pattern, str]] = []

    def same(path: str) -> None:
        rules.append((path, path))

    def same_rx(pattern: str) -> None:
        rules.append((re.compile(f"({pattern})"), r"\1"))

    # -- project root ---------------------------------------------------------
    # All three workflows' run records live here since 2026-08-11. WF3's are
    # experiment-keyed in the FILENAME, so they get their own rows rather than
    # widening the wf1/wf2 ones to `wf[123]` — the experiment fragment is what
    # distinguishes them, and a row that does not say so would accept
    # `wf3_anything.log`. `[a-z0-9_]+` is validate_experiment_name's grammar.
    # The two `_parts/` prefixes already cover WF3's `<experiment>/` level.
    same_rx(r"logs/wf[012]_[^/]+\.log")
    same_rx(r"logs/wf3_run_stress_test_[a-z0-9_]+\.log")
    same("logs/_parts/")
    same("logs/dag/")
    same_rx(r"benchmarks/wf[012]_benchmarks\.md")
    same_rx(r"benchmarks/wf3_benchmarks_[a-z0-9_]+\.md")
    same("benchmarks/_parts/")

    # -- config/ --------------------------------------------------------------
    # The two snapshot CONTRACT PATHS are enumerated; the digest bundles are a
    # regex because the digest is content-derived. `files/` inside a bundle is a
    # prefix: its members are named `<hash>-<original>` per referenced input.
    same("config/runs/snake_config_analyze_climate.yml")
    same("config/runs/snake_config_build_model.yml")
    same("config/runs/snake_config_analyze_projections.yml")
    # `scripts/run_workflows.py`'s invocation manifest, one immutable file per
    # wrapper run. A PREFIX, because the set is genuinely open — the filename is
    # `<utc stamp>-<uuid12>.json`, so a new one appears every run.
    #
    # It needs its own row and cannot ride the workflow regex below: that regex
    # requires a `<digest>/` directory between the workflow name and the file,
    # and a manifest sits DIRECTLY under `invocations/`. `invocations` is not a
    # workflow anyway — it is a SIBLING of the `<workflow>/<digest>/` bundles,
    # since an invocation spans workflows (`run_workflows.py`, R9 follow-up
    # ruling of 2026-08-05). Registered BEFORE the regex to match the R9 map's
    # ordering and its reasoning, though here the two are disjoint outright.
    #
    # MISSED UNTIL 2026-08-11 because no tree the gate was pointed at had one:
    # the wrapper is not a rule, so the declared tier cannot see it, and
    # `test_local`'s runs were all direct `snakemake` calls. It surfaced the
    # first time a fixture (`test_rapid`) was rebuilt THROUGH the wrapper. The
    # R9 map has carried the equivalent row since 2026-08-05; this is the
    # mirror that was never made.
    same("config/runs/invocations/")
    # The content-addressed bundles are GONE (config-snapshot redesign,
    # 2026-08-13); the regex that matched
    # `config/runs/<workflow>/<digest>/...` went with them, so a surviving
    # bundle in an existing project now reports as undeclared. That is the
    # migration's own signal, and `dev/scripts/prune_config_snapshots.py` is
    # what clears it.
    #
    # What replaces them is ONE record per workflow, at an enumerated path --
    # no digest level, so nothing needs a regex.
    same("config/runs/analyze_climate/run_record.yml")
    same("config/runs/build_model/run_record.yml")
    same("config/runs/analyze_projections/run_record.yml")
    # The run journal. An UNDECLARED SIDE EFFECT and the only one in this tree:
    # it is written by the workflow's lifecycle handlers, not by a rule, so the
    # declared tier cannot see it and it has to be whitelisted here by hand. It
    # is deliberately not a rule output -- Snakemake deletes those before a job
    # runs, which would truncate the ledger to one line every time.
    same("config/runs/journal.jsonl")
    # The bin's own README, rewritten by rule X.01 on every run.
    same("config/runs/README.md")
    # Both still receive copies, but only of files the toolbox repository
    # cannot give back: a tracked, unmodified catalog or template is recorded
    # by git blob id instead. In a normal checkout these are usually EMPTY.
    same("config/catalogs/")
    same("config/templates/")
    same("config/basin_data/")
    # Generated build config lives in the model's own config/ (R9 design v10).

    # -- data/ ----------------------------------------------------------------
    for leaf in (
        "spatial_maps.nc",
        "spatial_catalog.yml",
        "spatial_report.yml",
        "location_registry.csv",
        # ADR 0003 §8a seam intermediate.
        "hydrography.nc",
    ):
        same(f"data/spatial/{leaf}")
    same("data/spatial/geoms/")
    # ADR 0007: basin_area depicts elevation, so it is drawn from the spatial
    # foundation and lands beside it rather than inside the model tree.
    same("data/spatial/plots/")
    # The climate store is keyed by <clim_source>_<window>; the key is a CACHE
    # KEY, so the rule is keyed by a variable exactly as the R9 map's is.
    same(f"data/climate/historical/{dataset_key}/")
    same_rx(r"data/climate/historical/[^/]+/.*")
    for tier in ("raw", "scalar", "summary", "plots"):
        same(f"data/climate/projections/{clim_project}/{tier}/")
    same(f"data/climate/projections/{clim_project}/report.md")

    # -- models/ --------------------------------------------------------------
    wflow = "models/hydrology/wflow"
    for leaf in (
        "staticmaps.nc",
        "wflow_sbm.toml",
        "hydromt.log",
        "hydromt_data.yml",
        ".model_built",
        ".outputs_configured",
        # ADR 0004's terminal build sentinel.
        ".model_final",
        "config/build_historical_forcing.yml",
        # The one-entry catalog pointing hydromt at the climate store — a
        # DECLARED, non-temp() output of rule 1.10 beside the build YAML from
        # the same rule (`build_model.smk`, `store_catalog`). Added
        # 2026-08-11: `test_local`'s WF1 predates the output, so no tree the
        # gate was pointed at held one until `test_rapid` was rebuilt.
        #
        # An ENUMERATED leaf, not a `config/` prefix — the model's `config/`
        # holds a fixed set, so a genuinely new file there should still report.
        # It had deliberately NO row in the R9 migration map either: no pre-R9
        # form, so a relocation row could never have fired — the same argument
        # that map already made for `.model_final`.
        "config/climate_store_catalog.yml",
        "forcing/inmaps_historical.nc",
        # The values hydromt was ACTUALLY handed, from rules 1.07 and 1.08
        # (config-snapshot redesign §5.6). Not copies of the build templates:
        # arguments are replaced with the P1 spatial products and some are
        # derived at call time, so these record what the template cannot.
        # Enumerated leaves for the same reason as the store catalog above --
        # the model root holds a fixed set, so a genuinely new file here
        # should still report.
        "hydromt_build_config.yml",
        "hydromt_update_waterbodies.yml",
    ):
        same(f"{wflow}/{leaf}")
    for directory in (
        "staticgeoms/",
        "forcing/plots/",
        "run_default/",
        "evaluation/",
    ):
        same(f"{wflow}/{directory}")
    # `{wflow}/plots/` is deliberately absent: ADR 0007 moved its only member,
    # basin_area, to `data/spatial/plots/`. A leftover directory there is stale
    # output from a pre-0007 run and SHOULD report as undeclared.

    # -- experiments/<id>/ ----------------------------------------------------
    for leaf in (
        ".project_consistency_ok",
        "config/snake_config_run_stress_test.yml",
        "config/model_reference.yml",
        "config/experiment.yml",
        # The stress-test parameter lookup, beside the config snapshot whose
        # settings produced it. Replaced `config/stress_test_design.csv` and
        # absorbed `climate/weathergenr/_work/st_<m>.csv` -- measured: without
        # this row the new artifact classifies UNMAPPED on every run.
        "config/stress_test_lookup.csv",
        # WF3's run record. It sits DIRECTLY in the experiment's config bin,
        # not under `config/runs/` like WF1's and WF2's: per arch-10 the WF3
        # snapshot stays inside the experiment, which IS the partition here.
        # So it needs its own row and cannot ride the `config/runs/` prefix
        # below.
        "config/run_record.yml",
    ):
        same(f"experiments/{e}/{leaf}")
    for directory in (
        # Still a prefix: it holds the GENERATED experiment catalog, and a
        # project may also have copies here from before the R4 predicate.
        "config/catalogs/",
        # `config/runs/` is GONE from this list (2026-08-13). WF3's record moved
        # to `config/run_record.yml` above -- the experiment IS the partition, so
        # it needs no per-workflow subdirectory -- and the bundles that were the
        # directory's only other occupants are retired. Keeping the prefix would
        # declare a directory nothing writes, which is how an orphan goes
        # unreported: the retired bundle under it stayed GREEN while its WF1 and
        # WF2 siblings correctly went red.
        "results/",
        "hydrology/wflow/",
    ):
        same(f"experiments/{e}/{directory}")
    # `climate/weathergenr/` is NARROWED to its three live subdirectories rather
    # than declared whole. The fixture holds exactly these plus the retired
    # `_work/`, so the narrowing is exact — and it is what makes a leftover
    # `_work/` report as undeclared instead of riding a whole-directory prefix.
    # Declaring the parent would accept the very orphan the migration creates,
    # which is the failure mode the `config/runs/` note above already records.
    for directory in ("config/", "output/", "plots/"):
        same(f"experiments/{e}/climate/weathergenr/{directory}")
    # `logs/` and `benchmarks/` are deliberately ABSENT since 2026-08-11: WF3's
    # run records moved to the project's own logs/ and benchmarks/, keyed by
    # experiment in the filename. A file under `experiments/<id>/logs/` is now
    # stale output from an earlier run and SHOULD report as undeclared — the same
    # reasoning `{wflow}/plots/` carries above.
    # Belt and braces on the experiment id: a tree may hold OTHER experiments
    # than the one this config names, and they are legitimate rather than
    # orphaned. Registered last so the named experiment's narrower rows win.
    rules.append((re.compile(rf"(experiments/(?!{exp}/)[^/]+/.*)"), r"\1"))
    return rules


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
    lines = [
        f"{kind:<8} {old}" + (f"  ->  {new}" if kind == "MOVED" else "")
        for old, new, kind in rows
    ]
    counts = {
        k: sum(1 for _, _, kind in rows if kind == k)
        for k in ("MOVED", "IDENTITY", "DELETED", "UNMAPPED")
    }
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
    every caller (the path-map users, `compare_yaml`,
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
                return new + rel[len(old) :], True
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


def _compare_array(
    name: str, ref: np.ndarray, cur: np.ndarray, tol: float
) -> list[str]:
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


def compare_nc(
    ref_path: str, cur_path: str, tol: float = DEFAULT_TOLERANCE
) -> list[str]:
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
            diffs.append(f"coord set {sorted(cur.coords)} vs {sorted(ref.coords)}")
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
            rest = s[len(v) + 1 :]
            return "<PROJECT_ROOT>/" + apply_path_map(rest, path_map)
    return val


def _normalize_tree_root_paths(doc, variants, path_map):
    if isinstance(doc, dict):
        return {
            k: _normalize_tree_root_paths(v, variants, path_map) for k, v in doc.items()
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
            ref, _root_token_variants(ref_root, ref_root_tokens), path_map
        )
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


def compare_geojson(
    ref_path: str, cur_path: str, tol: float = DEFAULT_TOLERANCE
) -> list[str]:
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
        return compare_yaml(
            ref_path, cur_path, rel, ref_root, cur_root, path_map, ref_root_tokens
        )
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
        (
            k
            if translated[k].as_posix() == k
            else f"{translated[k].as_posix()} (expected at {k})"
        )
        for k in raw_missing
        if k not in allow
    ]
    extra = [k for k in raw_extra if k not in allow]
    failures: list[tuple[str, list[str]]] = []

    for key in sorted(set(translated) & set(cur_keys)):
        rel_ref = translated[key]
        rel_cur = cur_keys[key]
        reasons = dispatch(
            rel_cur,
            str(ref / rel_ref),
            str(cur / rel_cur),
            tol,
            ref_root=ref_root,
            cur_root=cur_root,
            path_map=path_map,
            ref_root_tokens=ref_root_tokens,
        )
        if reasons:
            label = (
                key if rel_ref.as_posix() == key else f"{rel_ref.as_posix()} -> {key}"
            )
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
            failures.append(
                (
                    f"merge {survivor}",
                    [f"survivor missing from current tree: {survivor}"],
                )
            )
            continue
        for src in sources:
            src_path = Path(src)
            if src_path not in ref_files:
                failures.append(
                    (
                        f"merge {survivor} <- {src}",
                        [f"declared merge source missing from reference tree: {src}"],
                    )
                )
                continue
            n_merge_compared += 1
            reasons = dispatch(
                cur_keys[survivor],
                str(ref / src_path),
                str(cur / cur_keys[survivor]),
                tol,
                ref_root=ref_root,
                cur_root=cur_root,
                path_map=path_map,
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
        "--check-map",
        metavar="PATHLIST",
        help="path-map falsifier mode: read project-relative paths (one per "
        "line, '#' comments ignored) and classify each as MOVED / "
        "IDENTITY / DELETED / UNMAPPED instead of diffing two trees. "
        "Exit 1 if any path is UNMAPPED. --ref/--cur are not used",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="relative tolerance for element-wise numeric compare (0 = exact)",
    )
    ap.add_argument(
        "--experiment-name",
        default="experiment",
        help="experiment_name for the P3-1 path map (default: experiment)",
    )
    ap.add_argument(
        "--dataset-key",
        default=None,
        help="historical-store dataset key, e.g. era5_20000101_20201231 "
        "(enables the climate_historical/raw_data/ -> <key>/ rule and the "
        ".guard_ok allowlist entry)",
    )
    ap.add_argument(
        "--no-path-map",
        action="store_true",
        help="disable the P3-1 path map + built-in allowlist (identical-relpath "
        "keying only, the pre-P3-1 behavior)",
    )
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        help="extra allowlisted MISSING/EXTRA relpath (repeatable; every entry "
        "must be justified in the migration note)",
    )
    ap.add_argument(
        "--allow-content",
        action="append",
        default=[],
        metavar="RELPATH",
        help="adjudicated CONTENT difference (repeatable): the file is still "
        "compared and the exception is printed in the report, but it does "
        "not fail the gate. Distinct from --allow, which covers "
        "MISSING/EXTRA presence. Every entry must be justified in the "
        "migration note",
    )
    ap.add_argument(
        "--ref-token",
        action="append",
        default=[],
        metavar="TOKEN",
        help="project_dir token as RECORDED inside the reference tree's own "
        "files, when it differs from where the tree is now read from "
        "(repeatable). R07's fixture rename makes this necessary: the "
        "pre-R07 reference embeds 'examples/test_local' wherever it is "
        "held",
    )
    ap.add_argument(
        "--milestone",
        choices=("p31",),
        default="p31",
        help="which built-in path map + allowlist to use (default, and now "
        "only, p31). The r07 and r09 one-way maps were retired 2026-08-11; "
        "recover either from its tag if an archived tree needs them",
    )
    ap.add_argument(
        "--clim-project",
        default="cmip6",
        help="clim_project subdir under climate_projections/",
    )
    ap.add_argument(
        "--clim-source",
        default=None,
        help="clim_source, e.g. era5 or chirps; consulted by a --merge class "
        "declared only on the chirps / chirps_global branch",
    )
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="extra path-map rule, appended after the built-in rules "
        "(repeatable). A trailing '/' on OLD makes it a directory-prefix "
        "rule; otherwise it is an exact-file rule",
    )
    ap.add_argument(
        "--merge",
        action="append",
        default=[],
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
    else:
        path_map = (
            build_p31_path_map(args.experiment_name, args.dataset_key) + extra_rules
        )
        allowlist = build_p31_allowlist(args.experiment_name, args.dataset_key)
        allowlist += list(args.allow)
    if args.check_map:
        paths = [
            line.strip()
            for line in Path(args.check_map).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        absolute = [
            p for p in paths if p.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", p)
        ]
        if absolute:
            ap.error(
                f"--check-map expects PROJECT-RELATIVE paths; {len(absolute)} "
                f"absolute path(s) found, first: {absolute[0]!r}"
            )
        # No built-in DELETED class: the one that existed belonged to the
        # retired R09 map (`indicators/RT_*.csv`, deleted rather than
        # migrated). `classify_path_map` still takes the argument, and
        # `snapshot_project_tree.py` still passes one, so a future map can
        # declare deletions without a signature change.
        rows = classify_path_map(paths, path_map, None)
        print(format_path_map_report(rows))
        return 0 if not any(kind == "UNMAPPED" for _, _, kind in rows) else 1

    if not (args.ref and args.cur):
        ap.error("--ref and --cur are required unless --check-map is given")
    report = diff_trees(
        args.ref,
        args.cur,
        args.tolerance,
        path_map=path_map,
        allowlist=allowlist,
        merges=merges,
        ref_root_tokens=list(args.ref_token),
        allow_content=list(args.allow_content),
    )
    print(format_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
