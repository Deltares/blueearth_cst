# Repository layout

Last updated: 2026-11-05

Use this map to place code, configuration, documentation and outputs. The
[architecture reference](toolbox-architecture.md) explains workflow responsibilities and dependencies.

## blueearth_cst/

Modules are invoked by Snakemake `script:` (Python) or `Rscript --vanilla` `shell:` (R), not as CLIs.

Stages include `model/`, `projections/`, `climate_analysis/` and `experiment/`. `weathergen/` holds
the R weather generator; `shared/` holds cross-workflow helpers:

| Helper | Holds |
|---|---|
| `snake_utils.py` | config parsing for every Snakefile |
| `run_logged.py` | logged subprocess execution |
| `climate_parity.py` | the engine-neutral regrid/PET transform |
| `cross_workflow_leaves.py` | the wf1 outputs WF3 declares but cannot build |
| `spatial_geoms_parity.py` | which of the two geoms trees answers which question |
| `interchange_contracts.py` | structural checks for weather-generator and hydrological-model artifacts |

## config/

Six directories plus `advanced_settings.yml`: `basemap/`, `catalogs/`, `defaults/`,
`indicator_sets/`, `plotting/` and `templates/`.

Project configuration files live beside their project, as the seeds do under `test_case/`.
There is no `config/workflows/` directory.

**`defaults/` vs `templates/`:** rules read defaults, so changes to `wflow_build_model.yml`,
`wflow_update_waterbodies.yml` or `weathergen_config.yml` change runs. Templates are scaffolds to copy.

`<project_dir>/config/templates/` holds generated snapshots and is normally empty. It contains a file only
when a project config points to its own copy; shipped defaults are recorded by Git blob ID instead.
`wflow_sbm.reference.toml` is a reference, not a runtime input or a current build output.

**`indicator_sets/`** holds bundled WF4 sets (`default.yml`), resolved by set ID against the
installed repository. A change affects published indicators. Custom sets live beside the project
and can start from `templates/indicator_set.template.yml`.

**`plotting/`** holds shared figure configuration. The
[plotting component map](../../blueearth_cst/plotting/README.md) explains its scope and callers.

**`catalogs/`** holds HydroMT `-d` targets. `generate_cmip6_catalog.py` generates
`cmip6_data.yml` and `cmip6_store_index.json` from a live `gs://cmip6` listing; never hand-edit
them. Both carry the same crawl's `crawled_on`, which a consumer checks for equality.

**`basemap/`** holds `natural_earth_50m.gpkg`, the committed locator-inset asset. Vendoring avoids
Cartopy's first-use download and per-user cache differences; it does not set a precedent for basin
data. See [asset provenance and licence](../../config/basemap/README.md).

**`advanced_settings.yml`** has `constraints:` (hard limits such as `min_historical_years`),
`defaults:` (project-overridable values such as `julia_threads`) and `runtime:` (toolchain pins such
as `julia_version`). The Julia pin must agree with `pixi.toml` and `Manifest.toml`; see
`tests/test_julia_runtime.py`. `snake_utils` reads this file for every project; it is not a
`--configfile` target. Its schema is closed: add each key to `_ADVANCED_SETTINGS_SCHEMA` in
`blueearth_cst/shared/advanced_settings.py` together with the YAML. Keep the schema separate from
`wf3_science.py` so an edit does not re-key WF3 collections.

**Tracked `test_case/` inputs** are `project_config_*.yml` and two basin CSVs in `test_data/`.
Keep `project_config_` on new seeds; the Git allowlist excludes other names. To add another tracked
input, change `.gitignore` with a pattern that admits its parent directory and the file, then verify
with `git check-ignore` and `git ls-files`.

## scripts/

User-facing runners, indexed in `scripts/README.md`.

`suggest_experiment_name.py` previews selection without editing YAML or reserving a directory.

The simulation's `experiment_name` is optional. Unset, runners start at `stress_test_01`; plain reruns
select the highest allocation regardless of completion. Only `--new-experiment` advances numbering.
A custom base selects its own family; an exact numbered name pins it. Existing owned dated or unnumbered
experiments remain explicit legacy selections. The resolver lives in `experiment/reservation.py`.

## tests/

Public tests live under `tests/`. Keep their input and expected-result fixtures available in the
public checkout; a local-only data dependency does not supply CI coverage.

`tests/data/gev-lmoments/` holds frozen GEV parity controls and a README describing their provenance.

## dev/

`dev/` holds contributor references, templates, the baseline manifest and development helpers; it is
not shipped with the runtime package. Use the [reference index](README.md) to find the current topic
owner. Reusable reports and review forms belong in `dev/templates/`. The
[record lifecycle](dev-record-lifecycle.md) governs promotion and retention.

`scaffold_project_tree.py` supports static WF0–2 layouts only. It refuses WF3/4,
whose output identities require executed checkpoints; inspect a completed run
with `snapshot_project_tree.py --config <config> --project-dir <root>` instead.

`dev/README.md` is the folder map and contributor entry point. Keep disposable diagrams and render
previews in scratch; retain evidence only when a future reader needs it.

### Task temporary material

For an existing task, keep temporary notes, probes, diagrams, review pages and logs in its
`dev/tasks/<task-id>/.tmp/<purpose>/` folder. For standalone work, use
`.tmp/scratchpad/<purpose>/`. Create a folder only when needed and check `git check-ignore -q` before
writing. Keep configured tool state, including pytest and Ruff caches, in its configured location.
Before cleanup, verify the target is within the intended scratch root, inspect links and other writers,
and preserve evidence still needed by a reader or gate. Completion alone grants no deletion authority.

Three helpers report by default: `prune_series_cache.py` finds orphaned WF2 series;
`prune_climate_store.py` finds stale `<source>_<window>` stores; `snapshot_project_tree.py` lists paths
against the project-tree inventory (`pixi run tree-check`). Deletion requires `--delete`. Prune before
a reference snapshot so it excludes orphans. Directories routed wholesale by the path map also need
an mtime sweep; the pruners do not see every file.

The project-tree inventory is generated from `blueearth_cst/shared/layout_paths.py`, which declares every
path the toolbox writes, with its kind (`tests/test_layout_paths.py` checks placement). Change a
declaration in the same commit as its writer; only legacy shapes stay hand-written in
`semantic_tree_diff.build_project_tree_rules`.

## docs/

`docs/guide/` owns the canonical Markdown user guides and their assets.
The Quarto site stages ignored copies in `docs/site/guide/` during pre-render
and publishes them alongside `setup/`, `scientific-approach/`, and
`toolbox-reference/`. `notebooks/` holds source-only walkthroughs;
`references/` keeps the dated 2025
technical note and vendored HydroMT / HydroMT-Wflow / Wflow guides for offline
use. See `docs/README.md` for the map. Configs are not mirrored here;
`config/` is the single source.
Current user guidance lives in `docs/guide/`; older material is available through Git history.

## .github/workflows/ci.yml

`pytest tests/` on ubuntu + windows with `locked: true`, so `pixi.lock` drift fails CI.

CI covers bare-checkout tests. Fixture-dependent and `--run-integration` cases skip;
`check_baseline.py` and whole-tree `semantic_tree_diff.py` need untracked `test_case/test_local`.
CI success therefore does not establish baseline agreement.

The baseline covers data, not figures. `FIGURE_KINDS` are excluded by default: byte-size changes
from cosmetic edits would otherwise fail the gate without establishing a scientific defect.

When moving the project tree, search tests for old roots. Fixture-dependent tests may skip where
their local data is absent, especially behind `os.path.exists`; tree-shape gates do not inspect
reader code and cannot substitute for this check.

## Outputs

Outputs land under configured `project_dir`, outside the repository for production cases. Untracked
`test_case/test_local` is the development exception.

Execution diagnostics share `logs/<YYYY-MM-DD_HH-MM-SSZ>__<scope>[__dry-run]/`: summary, console transcript,
workflow logs and Markdown/TSV benchmarks. Child invocations share their parent's folder; each retry
has fresh part namespaces. Historical shared logs and benchmark files remain untouched. The maintained
layout and interpretation live in `docs/guide/running.md` § Execution logs and benchmarks.
The WF2 product additionally owns `_engine/execution/` (production attribution and retained copies of
contributing logs) and, after an import, `_engine/imported/`; `docs/guide/outputs.md` owns both.

For new output projects, `config/<workflow>/` holds `run_record.yml`, captured files,
`SOURCES.md`, and `catalogs/` for the latest WF0–WF2
configuration archives. `SOURCES.md` lists the original source paths within each workflow folder.
Immutable generations live in `config/_engine/snapshots/<workflow>/<record-sha256>/`;
attempt history references those exact records, while unchanged captures reuse their creator generation.
Shared execution inputs, archive locks and invocation history live in `config/_engine/`.
Completed replacement backups and child launch contexts are transient. Existing cases can be inspected
or migrated with `scripts/maintain_config_archives.py` (`--apply` opts into changes).
There is no intermediate `config/runs/` directory. Editable source configurations
remain beside the project, outside these generated archives.

`experiments/_engine/{allocation.json,transactions/}` owns project-wide numbering and creation
recovery. Each `experiments/<name>/_engine/` owns that experiment's reservation, simulation and indicator
records. Discovery skips the reserved collection-level `_engine/` directory. These paths replace
`config/experiments/` for new projects; existing output trees are not migrated.

## Three homes for executables

Keep runtime modules in `blueearth_cst/`, user execution entry points in `scripts/`, and maintenance
helpers in `dev/scripts/`. A workflow run must not depend on the development directory.

`dev/scripts/` also contains libraries imported by tests: `semantic_tree_diff.py` (tree comparison
and inventory), `cross_workflow_inputs.py` (WF1 leaves for WF2/WF3) and `sweep_common.py` (shared
sweep primitives). Treat their imported surfaces as test contracts.

`sweep_unread_config_keys.py` checks whether declared template keys reach run-path readers. It does
not edit. Its `sweep_common.py` checks fail closed: an unclassified hit or lost subject coverage is
reported, not counted as a clean tree.

When a `dev/scripts/` helper gains a run-path caller, move the shared part into `blueearth_cst/`.
Runtime code cannot rely on `dev/` being installed. Shared path definitions belong in
`blueearth_cst/shared/`; test-only staging stays in `dev/`.

`dev/scripts/console.py` is a vendored copy used by `stage_data.py` and `stage_cmip6.py`. Keep it in
sync with its upstream module instead of making a local-only fix. `console.__version__` identifies the
copied release; absence denotes an older copy. `[tool.ruff.format]` excludes it so formatting does not
obscure upstream comparison, while `ruff check` still lints it.
