# dev/scripts/

One-off and recurring helpers used outside the Snakemake workflows. Nothing
here is invoked by `Snakefile_*` rules — those live in `blueearth_cst/`.

## Env / install

| Script | What it does |
|---|---|
| [`install_weathergenr.R`](install_weathergenr.R) | Idempotent install of `weathergenr` v1.2.0 from GitHub via `remotes::install_github` (`dependencies=FALSE`, `upgrade="never"`). Invoked by `pixi run install-rdeps` (and transitively by `pixi run install`). Lands in `.libPaths()[1]` — the pixi env's R site-lib on both platforms. |
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
| [`inspect_spatial_ref.py`](inspect_spatial_ref.py) | Whether `spatial_ref.x_dim` / `y_dim` attrs propagate through weathergenr's `write_netcdf` (they don't — see the weathergenr items in `dev/followups.md` R5 section). |
| [`inspect_weathergenr.R`](inspect_weathergenr.R) | Lists the installed weathergenr's exported API and the signatures of functions called by `src/weathergen/generate_weather.R`. Used to detect signature drift between the package and the workflow. |

## Workflow inspection

| Script | What it does |
|---|---|
| [`rule_dag_levels.py`](rule_dag_levels.py) | Print a Snakefile's rules in **DAG order** with per-rule job counts (runnable vs already up to date). Snakemake's own `Job stats:` table is alphabetical and no flag re-sorts it, so it never shows what runs before what. Reads `--rulegraph dot` (structure) + `--dag dot` (job counts); executes nothing. Use `dot` and not `--d3dag` — on snakemake 9.6.2 the D3 JSON drops edges (48 vs the DOT graph's 73 on WF2), and `--rulegraph mermaid-js` emits self-edges. |
| [`prune_series_cache.py`](prune_series_cache.py) | Report (and with `--delete`, remove) orphaned WF2 series left behind by a key-grammar or config change. Dry run by default. Must run **before** any reference snapshot, or the snapshot bakes the orphans in. |

## Baseline / regression

| Script | What it does |
|---|---|
| [`check_baseline.py`](check_baseline.py) | Record / check fingerprints for `rule all` targets across the three Snakefiles. Manifest at `dev/baseline/manifest.json`. `record` overwrites the manifest; `check` recomputes and diffs (exits non-zero on drift). Per-variable summary stats for netCDF, normalized SHA256 for CSV/YAML, size-only for PNG. See `dev/phase-1/m02b/baseline_diffs.md` for the as-shipped M2b drift report. |

## Shared helpers

| File | Purpose |
|---|---|
| [`console.py`](console.py) | Vendored colour / glyph / banner helpers from the `console-formatting` skill. Used by other scripts here. Self-contained, no third-party deps. Keep in sync with the upstream brain copy. |
| [`open_shell.bat`](open_shell.bat) | Double-clickable launcher: opens a PowerShell at the repo root with the `cst` conda env activated. Pre-pixi convenience; mostly superseded by `pixi shell`. |
