# dev/scripts/

One-off and recurring helpers used outside the Snakemake workflows. Nothing
here is invoked by `Snakefile_*` rules — those live in `blueearth_cst/`.

## Env / install

| Script | What it does |
|---|---|
| [`install_weathergenr.R`](install_weathergenr.R) | Idempotent install of `weathergenr` v2.0.0 from GitHub via `remotes::install_github` (`dependencies=FALSE`, `upgrade="never"`). Invoked by `pixi run install-rdeps` (and transitively by `pixi run install`). Lands in `.libPaths()[1]` — the pixi env's R site-lib on both platforms. |
| [`check_r_packages.R`](check_r_packages.R) | Loops over the R packages the workflow needs (`weathergenr`, `dplyr`, `ggplot2`, `ncdf4`, `yaml`, …) and reports presence + version. Quick sanity check after env changes. |

## Data catalog / staging

| Script | What it does |
|---|---|
| [`migrate_data_catalog_v0_to_v1.py`](migrate_data_catalog_v0_to_v1.py) | Convert a hydromt 0.x catalog YAML to the 1.x schema (path → uri, meta → metadata, driver-string → driver-object, kwargs → driver.options, etc.). Used during M2b to migrate `config/cmip6_data.yml` and `config/deltares_data*.yml`. |
| [`stage_data.py`](stage_data.py) (config: [`stage_data.yml`](stage_data.yml)) | Mirror a bbox subset of a remote data root (P-drive zarr / netcdf) to a local SSD. The matching catalog YAML just needs its `meta.root` swapped to the local path after staging. |
| [`list_era5_vars.py`](list_era5_vars.py) | Print the data variables present in the staged era5_daily zarr (metadata only — no streaming). Useful when picking variable names for `stage_data.yml`'s `variables:` filter. |

## Diagnostics / probes

One-off scripts written to chase down a specific bug. Kept for reference;
re-run when a similar symptom appears. Not part of any workflow.

| Script | What it diagnosed |
|---|---|
| [`inspect_era5_nan.py`](inspect_era5_nan.py) | Where the NaN values around 2010-2011 in the staged era5_daily zarr come from (source vs `subset_zarr()` step). |
| [`probe_era5.py`](probe_era5.py) | Definitive NaN count on the staged era5 `t2m` plus sample reads — sanity-check against `inspect_era5_nan.py`. |
| [`inspect_spatial_ref.py`](inspect_spatial_ref.py) | Whether `spatial_ref.x_dim` / `y_dim` attrs propagate through weathergenr's `write_netcdf` (they don't — see the weathergenr items in `dev/tasks/` R5 section). |
| [`inspect_weathergenr.R`](inspect_weathergenr.R) | Lists the installed weathergenr's exported API and the signatures of functions called by `src/weathergen/generate_weather.R`. Used to detect signature drift between the package and the workflow. |

## Workflow inspection

| Script | What it does |
|---|---|
| [`rule_dag_levels.py`](rule_dag_levels.py) | Print a Snakefile's rules in **DAG order** with per-rule job counts (runnable vs already up to date). Snakemake's own `Job stats:` table is alphabetical and no flag re-sorts it, so it never shows what runs before what. Reads `--rulegraph dot` (structure) + `--dag dot` (job counts); executes nothing. Use `dot` and not `--d3dag` — on snakemake 9.6.2 the D3 JSON drops edges (48 vs the DOT graph's 73 on WF2), and `--rulegraph mermaid-js` emits self-edges. |
| [`prune_series_cache.py`](prune_series_cache.py) | Report (and with `--delete`, remove) orphaned WF2 series left behind by a key-grammar or config change. Dry run by default. Must run **before** any reference snapshot, or the snapshot bakes the orphans in. |

## Figure tuning

| Script | What it does |
|---|---|
| [`preview_basin_map.py`](preview_basin_map.py) | Render rule 1.12's basin figure against a model already on disk, with any constant in `plot_map.py`'s TUNABLE block overridden from the command line — no WF1 run. `--list` prints every tunable with its current value and its own comment; `--set NAME=VALUE` overrides one; `--sweep NAME=V1,V2,...` renders one figure per value, named after it, for side-by-side comparison (repeat `--sweep` for the cross-product). Writes only to `--out-dir` (gitignored `.tmp/basin_map_preview` by default), never into a project's `plots/`. Renders against `test_case/basin_map_fixture` by default — a kept five-subcatchment model with gauges (see its README); override with `--project-dir` or `$BASIN_MAP_PROJECT_DIR`. **A figure is verified by looking at it** — this is the tool the "figures are terminal artifacts" clause in `AGENTS.md` points at. |
| [`basin_map_example.py`](basin_map_example.py) | A plain script that calls the layer-in `plot_basin_map(dem, rivers, basin, ...)`: set the model directory, the output path and the plotting parameters at the top, run it, get a PNG. Edit and re-run to try a value. |

## Baseline / regression

| Script | What it does |
|---|---|
| [`check_baseline.py`](check_baseline.py) | Record / check fingerprints for `rule all` targets across the three Snakefiles. Manifest at `dev/baseline/manifest.json`. `record` overwrites the manifest; `check` recomputes and diffs (exits non-zero on drift). Per-variable summary stats for netCDF, normalized SHA256 for CSV/YAML, size-only for PNG. See `dev/milestones/phase-1/m02b/baseline_diffs.md` for the as-shipped M2b drift report. |

## Todo board

| Script | What it does |
|---|---|
| [`todoboard.py`](todoboard.py) | Run the `todoboard` CLI that backs `dev/tasks/` and the GENERATED `dev/TODO.md` — `python dev/scripts/todoboard.py render \| list \| add "Title" \| done <id>`. Every verb and flag is the CLI's own; this only locates it and delegates. The CLI ships inside the `todo-board` **skill bundle**, which is per-user, gitignored and symlinked, so its path cannot be committed and `todoboard` is on nobody's `PATH` — which is how a board note landed on 2026-08-12 with the table left a row stale: the CLI was simply unreachable and nothing said so. Searches `.claude/skills` → `.agents/skills` → the brain artifacts dir → `~/.claude/skills`. `TODOBOARD_SKILL_DIR` overrides it, and a wrong one is **refused rather than fallen back from**, so a deliberate override cannot silently resolve to a different skill version. `render` runs wherever the task runs, but two session slots must not run it concurrently — it regenerates `dev/TODO.md`, a file neither edits by hand. |

## Shared helpers

Two of these are imported by `tests/` (via `sys.path`), so they are **contract
surfaces with test consumers, not scratch helpers** — a bare-checkout CI run
imports them on both legs, and an import-time error there fails the suite.

| File | Purpose |
|---|---|
| [`cross_workflow_inputs.py`](cross_workflow_inputs.py) | **Library, not a script.** The one definition of the wf1 leaves WF2/WF3 declare and Snakemake will not satisfy on its own, plus the deliberate non-leaves some callers stage anyway. Consumed by `tests/test_cli.py`, `tests/test_guard_invalidation.py` and `scaffold_project_tree.py` — it replaced three hand-kept copies that drifted (R9 P5 F3). `tests/test_cross_workflow_inputs.py` proves the set complete and minimal against the real DAG, so a rule declaring a new cross-workflow input turns it red rather than surfacing later as an unrelated-looking failure. |
| [`semantic_tree_diff.py`](semantic_tree_diff.py) | Whole-tree comparator and the home of the project-tree INVENTORY (`build_project_tree_rules`) — the map that asks whether a tree holds anything nobody declared. Imported by `tests/test_semantic_tree_diff.py` and `tests/test_project_tree_inventory.py`, and driven by `snapshot_project_tree.py`. The R07 and R09 one-way migration maps it also held were retired 2026-08-11 (`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §6a); recover either from its tag. `build_p31_path_map` stays, as the `--milestone` default a bare invocation resolves through. |
| [`console.py`](console.py) | Vendored colour / glyph / banner helpers from the `console-formatting` skill. Used by other scripts here. Self-contained, no third-party deps. Keep in sync with the upstream brain copy. |
| [`open_shell.bat`](open_shell.bat) | Double-clickable launcher: opens a PowerShell at the repo root with the `cst` conda env activated. Pre-pixi convenience; mostly superseded by `pixi shell`. |
