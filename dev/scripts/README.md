# dev/scripts/

Tools that inspect or maintain the repository. None is part of a run: nothing
here is invoked by a Snakemake rule (those live in `blueearth_cst/`), and a run
never imports from `dev/`. One-off probes and migrations are removed once their
job is done; the 2026-09-24 sweep is recoverable from tag
`archive/dev-scripts-2026-09-24`.

`private/` groups the personal activation, board/navigation and repository synchronization helpers.
It is excluded from the public projection. Shared helpers remain in this directory and continue to
support public development and CI.

`private/ingest_hydromt_user_guide.py` is the retained one-off HydroMT RST-to-Markdown converter.
It expects the historical `_raw/` clone and pin file and is not part of the runtime or CI build.

## Libraries imported by tests

Contract surfaces, not scratch helpers: a bare-checkout CI run imports them on
both legs, so an import-time error fails the suite.

| File | Purpose |
|---|---|
| [`console.py`](console.py) | Vendored colour/glyph/banner helpers from the `console-formatting` skill. Fix defects upstream and re-copy. |
| [`cross_workflow_inputs.py`](cross_workflow_inputs.py) | The one definition of the WF1 leaves other workflows declare; proved complete and minimal by `tests/test_cross_workflow_inputs.py`. |
| [`semantic_tree_diff.py`](semantic_tree_diff.py) | Whole-tree comparator and the project-tree inventory (`build_project_tree_rules`). |
| [`sweep_common.py`](sweep_common.py) | Shared machinery for the report-only repository sweeps below. |

## Gates and board

| Script | What it does |
|---|---|
| [`check_baseline.py`](check_baseline.py) | `record` / `check` / `compare` fingerprints of the baseline targets in `dev/baseline/manifest.json`. Requires WF1's retained native CSV. |
| [`snapshot_project_tree.py`](snapshot_project_tree.py) | Project-tree snapshot behind `pixi run tree-check`. |

- `private/todoboard.py` (private-checkout-only citation): Runs the `todoboard` CLI against the canonical board in the
  `main` worktree, including its generated `dev/TODO.md`.

## Repository sweeps (report-only)

| Script | What it finds |
|---|---|
| [`sweep_unread_config_keys.py`](sweep_unread_config_keys.py) | Declared config keys no reader reads. |
| [`scan_console_encoding.py`](scan_console_encoding.py) | Non-ASCII in strings that reach a console (cp1252). |

## Environment

| Script | What it does |
|---|---|
| [`install_weathergenr.R`](install_weathergenr.R) | Idempotent install of `weathergenr` from GitHub; run by `pixi run install-rdeps`. |
| [`check_env.py`](check_env.py) | Whether a checkout can actually run the workflows (weathergenr, Julia, activation). |
| [`inspect_weathergenr.R`](inspect_weathergenr.R) | The installed weathergenr's exports and the signatures the workflow calls; run after an upgrade. |
| [`pixi_activate.bat`](pixi_activate.bat) | Activate the pixi env in a cmd shell. |

- Board view (private-checkout-only citation: `private/setup-board-view.ps1`): Preview or reconcile a session's
  canonical task projection.
- `private/reconcile_board_projection.py` (private-checkout-only citation): Adopted-board migration helper used by
  Board view; new sessions use Git-workflow provisioning.
- Navigation setup (private-checkout-only citation: `private/setup-worktree-navigation.ps1`): Preview or create
  shortcuts; sparse targets must be named.
- `private/check_activation.py` (private-checkout-only citation): Drift between
  `dev/reference/private/dev-agent-activation.md` and the untracked agent manifest.

Open main's `dev/TODO.md` for Work, or `dev/tasks/` in a configured session for canonical task records. See
board setup (private-checkout-only citation:
`../reference/private/dev-task-and-session-workflow.md#local-scratch-and-navigation`) for installation. Navigation
preview makes no writes. Apply sparse exclusion with `-Apply -ExcludeSnapshots -Worktree session-N`; selected
roots are preflighted before writing, failures are reported while valid roots continue, and stale shortcuts
are retained. See coordination (private-checkout-only citation:
`../reference/private/dev-task-and-session-workflow.md#worktree-coordination`) for ownership and safe legacy junction
removal with non-recursive `cmd /c rmdir`, never `git clean`.

## Data staging and catalogs

`inspect_era5_zarr_geometry.py` reports ERA5 Zarr-v2 metadata, bounded chunk samples, and modeled read
costs for fixed representative envelopes. It preserves the original probe's sampling and reports its
limits; candidate compression is modeled, not measured in a rechunked store.

```powershell
pixi run python dev/scripts/inspect_era5_zarr_geometry.py --help
pixi run python dev/scripts/inspect_era5_zarr_geometry.py --store <era5-zarr-directory>
```

The optional `--probe-current` performs a read-only 30-day Ntoum HydroMT read. The nine case-specific transfer
probes remain historical measurement evidence, described by the retained migration context
(private-checkout-only citation: `../records/migrations/legacy-development-migration-2026-10-09.md`).

| Script | What it does |
|---|---|
| [`stage_data.py`](stage_data.py) + [`stage_data.yml`](stage_data.yml) | Mirror a bbox subset of a remote data root to local disk. |
| [`stage_cmip6.py`](stage_cmip6.py) + [`stage_cmip6.yml`](stage_cmip6.yml) | Stage CMIP6 slices for one region, digest-stamped so WF2's rule 2.04 hits its cache. |
| [`generate_cmip6_catalog.py`](generate_cmip6_catalog.py) | Regenerate `config/catalogs/cmip6_data.yml` from a live crawl of gs://cmip6. |
| [`probe_cmip6_grids.py`](probe_cmip6_grids.py) | Which CMIP6 models hydromt's raster accessor refuses (non-uniform grids). |

## Project inspection and cleanup

| Script | What it does |
|---|---|
| [`rule_dag_levels.py`](rule_dag_levels.py) | A Snakefile's rules in DAG order with per-rule job counts; executes nothing. |
| [`region_bbox.py`](region_bbox.py) | Report, and optionally write, the project region's bounding box. |
| [`scaffold_project_tree.py`](scaffold_project_tree.py) + [`scaffold_extras.yml`](scaffold_extras.yml) | Dummy `project_dir` tree from the Snakefiles, for layout review. |
| [`prune_series_cache.py`](prune_series_cache.py) | Orphaned WF2 series; dry run unless `--delete`. |
| [`prune_climate_store.py`](prune_climate_store.py) | Orphaned historical-climate stores; dry run unless `--delete`. |
| [`estimate_batch_makespan.py`](estimate_batch_makespan.py) | LPT makespan estimate behind WF4's batch size; see board item `t2608071217`. |

The scaffolder requires a new or empty `--out` directory. Use `--keep` to add missing placeholders
while preserving existing files; it never deletes the destination. A failed workflow summary stops
with a nonzero exit status and diagnostics. Staged prerequisites may remain in the incomplete tree.

## Figures and console

| Script | What it does |
|---|---|
| [`capture_page.py`](capture_page.py) | Headless screenshots of a figure-review page at chosen widths; fails on console, HTTP or image errors. |
| [`preview_plots.py`](preview_plots.py) | Render the toolbox's figure families from a project tree without running a workflow. Being superseded by `scripts/render_figures.py`, which already owns WF0 and WF1 forcing figures. |
| [`plotting_standalone_demo.py`](plotting_standalone_demo.py) | Prove the plotting core's portable boundary: render WF0 results under `python -I` from a copy of `blueearth_cst/plotting/` with only numpy and matplotlib. |
| [`export_svg.py`](export_svg.py) | Lift an inline `<svg>` out of an HTML page into a standalone file. |
| [`render_console_sample.py`](render_console_sample.py) + [`console_sample_specs.py`](console_sample_specs.py) | A console transcript per workflow through the real handler. |

## Repository housekeeping

| Script | What it does |
|---|---|
| [`notebook_outputs.py`](notebook_outputs.py) | Keep rendered outputs out of tracked notebooks. |


## Environment procedures

[environment-notes.md](environment-notes.md) owns local installation fixes, data staging and developer
runtime borrowing recipes. It links session/validation policy rather than defining a second workflow.
User setup and recovery live in [the guide](../../docs/guide/install.md).
