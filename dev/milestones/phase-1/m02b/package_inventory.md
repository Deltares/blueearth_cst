# Package inventory (M2b)

Audit of every package declared in `pixi.toml`, `Project.toml`, and the
weathergenr install script vs. what the three Snakemake workflows
actually load. Goal: see what could move into pixi, what could be
dropped, and confirm the Rtools dependency status.

Generated 2026-05-08, against `milestone/02b-library-upgrades` HEAD.

## TL;DR

- **Python conda:** 23 declared. 13 directly imported, 4 are explicit
  declarations of runtime backends used via fsspec/snakemake/etc.
  (gdal, shapely, gcsfs, graphviz), **6 unused** (datrie, dvc,
  descartes, flit, jupyter, tabulate).
- **Python pypi:** 3 declared. snakemake is the workflow runner;
  **hydroengine and gwwapi are unused** (mentioned only in upstream
  hydromt docs).
- **R:** 17 declared. r-yaml and r-ncdf4 are called directly; 12 are
  transitive imports of weathergenr (correctly declared since
  weathergenr is sideloaded). **r-devtools and r-forecast are
  candidates for removal** (devtools no longer used; forecast is a
  weathergenr Suggests, not Imports).
- **Julia:** Wflow only — already minimal.
- **Outside pixi:** Julia binary (juliaup; conda-forge has no win-64
  build). weathergenr is installed via `pixi run install-rdeps` (pak
  from GitHub) but on Windows the artifact lands in the user-lib due
  to a conda r-base toolchain ABI issue.
- **Rtools:** **Not a declared dependency.** mingw is pulled
  transitively by conda's r-base on Windows; the only mentions in the
  repo are two R3 followup comments.

---

## Python — conda dependencies

Listed in `pixi.toml` `[dependencies]`. "Used in" columns reference
files relative to repo root.

| Package          | Status                              | Used in                                                                                                                                                  |
| ---------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python`         | ✅ runtime                           | (the interpreter)                                                                                                                                        |
| `pip`            | ✅ runtime                           | (used for the `[pypi-dependencies]` install path)                                                                                                        |
| `cartopy`        | ✅ direct import                     | `src/plot_map.py`, `src/plot_map_forcing.py`, `src/plot_proj_timeseries.py`                                                                              |
| `geopandas`      | ✅ direct import                     | `src/extract_historical_climate.py`, `src/get_region_preview.py`, `src/get_stats_climate_proj.py`, `dev/scripts/stage_data.py`                          |
| `hydromt`        | ✅ direct import                     | `src/extract_historical_climate.py`, `src/func_plot_signature.py`, `src/get_change_climate_proj.py`, `src/get_change_climate_proj_summary.py`, `src/get_region_preview.py`, `src/get_stats_climate_proj.py`, `src/plot_proj_timeseries.py`, `src/plot_results.py`, `src/prepare_climate_data_catalog.py`, `src/setup_reservoirs_lakes_glaciers.py` |
| `hydromt_wflow`  | ✅ direct import                     | `src/downscale_climate_forcing.py`, `src/plot_map.py`, `src/plot_map_forcing.py`, `src/plot_results.py`, `src/setup_gauges_and_outputs.py`, `src/setup_reservoirs_lakes_glaciers.py`, `src/setup_time_horizon.py` |
| `matplotlib`     | ✅ direct import                     | `src/func_plot_signature.py`, `src/get_change_climate_proj.py`, `src/get_stats_climate_proj.py`, `src/plot_map.py`, `src/plot_map_forcing.py`, `src/plot_proj_timeseries.py`, `src/plot_results.py` |
| `numpy`          | ✅ direct import                     | most src/ scripts; check_baseline; tests                                                                                                                 |
| `pandas`         | ✅ direct import                     | `src/export_wflow_results.py`, `src/func_plot_signature.py`, `src/get_change_climate_proj.py`, `src/get_region_preview.py`, `src/get_stats_climate_proj.py`, `src/plot_proj_timeseries.py`, `src/plot_results.py`, `src/prepare_cst_parameters.py`, `dev/scripts/inspect_era5_nan.py` |
| `pytest`         | ✅ direct import                     | `tests/conftest.py`, `tests/test_cli.py`, `tests/test_model_creation.py`                                                                                 |
| `scipy`          | ✅ direct import                     | `src/func_plot_signature.py` (`scipy.stats`)                                                                                                             |
| `seaborn`        | ✅ direct import                     | `src/get_change_climate_proj_summary.py`                                                                                                                 |
| `xarray`         | ✅ direct import                     | most src/ scripts; check_baseline                                                                                                                        |
| `xclim`          | ✅ direct import                     | `src/metrics_definition.py` (`from xclim.indices.stats import frequency_analysis`)                                                                       |
| `gdal`           | 🟡 declared as runtime backend       | No direct `from osgeo import gdal`. Provides the GDAL binaries that rasterio / fiona / geopandas link to. Removing breaks geo I/O.                       |
| `shapely`        | 🟡 declared as runtime backend       | No direct `import shapely` in workflow code; pulled transitively by geopandas. Explicit declaration pins the version.                                    |
| `gcsfs`          | 🟡 declared for fsspec backend       | No direct import. Provides the `gs://` filesystem backend that hydromt uses to fetch CMIP6 data (`gs://cmip6/...`). Removing breaks `Snakefile_climate_projections`. |
| `graphviz`       | 🟡 declared for snakemake DAG render | No direct Python import. Provides the `dot` CLI used by `snakemake --dag \| dot -Tpng`. Listed in `CLAUDE.md` workflow examples.                          |
| `zarr`           | 🟡 used by probe only                | Direct import in `dev/scripts/probe_era5.py:9` (one-off diagnostic). Hydromt uses zarr internally for `*.zarr` URIs but doesn't require explicit import. |
| `datrie`         | ❌ unused                            | No imports anywhere. Was added as a snakemake helper transitive; snakemake (PyPI) installs it itself.                                                    |
| `dvc`            | ❌ unused                            | No imports, no CLI invocations.                                                                                                                          |
| `descartes`      | ❌ unused                            | Only commented-out `# import descartes` in `src/plot_map.py:22`, `src/plot_map_forcing.py:23`. Package is also archived on PyPI.                          |
| `flit`           | ❌ unused                            | No `pyproject.toml` or build-system configured. Build tool with no consumer.                                                                             |
| `jupyter`        | ❌ unused at runtime                 | Three notebooks under `docs/notebooks/` are generated artifacts (output of upstream conversion). No `import jupyter` and no notebook is run by any rule. |
| `tabulate`       | ❌ unused                            | No imports.                                                                                                                                              |

### Implicit Python deps used but not pinned

These are imported directly by workflow code but rely on transitive
resolution from other deps. Pinning them in `pixi.toml` would lock
their versions explicitly:

| Package      | Imported in                                                                                                                                              | Comes via       |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `dask`       | `src/extract_historical_climate.py:11`, `src/get_stats_climate_proj.py:23` (`from dask.diagnostics import ProgressBar`)                                  | xarray          |
| `netCDF4`    | `dev/scripts/check_baseline.py:28`, `dev/scripts/inspect_spatial_ref.py:10`                                                                              | xarray / hydromt |
| `rasterio`   | `dev/scripts/stage_data.py:60-61`                                                                                                                        | gdal / hydromt  |
| `PyYAML`     | most `src/*.py` scripts (`import yaml`); tests; dev scripts                                                                                              | snakemake / hydromt |

---

## Python — PyPI dependencies

Listed in `pixi.toml` `[pypi-dependencies]`.

| Package        | Status                  | Used in                                                                                                                                |
| -------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `snakemake`    | ✅ workflow runner       | `Snakefile_*` (driven by `pixi run snakemake -s ...`); test invocations via `tests/test_cli.py`                                       |
| `hydroengine`  | ❌ unused                | No imports. Mentioned only in `docs/hydromt-user-guide/data_catalog/data_conventions.md:44` (upstream hydromt doc, not workflow code). |
| `gwwapi`       | ❌ unused                | No imports anywhere.                                                                                                                   |

---

## R packages

Listed in `pixi.toml` `[dependencies]`. The workflow's R layer
(`src/weathergen/*.R`) only directly references **weathergenr**,
**yaml**, and **ncdf4** — everything else is a transitive Imports
edge from weathergenr.

| Package           | Status                            | Used in                                                                                                                       |
| ----------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `r-base`          | ✅ runtime                         | (the R interpreter)                                                                                                           |
| `r-yaml`          | ✅ direct                          | `src/weathergen/generate_weather.R:6,11`, `src/weathergen/impose_climate_change.R:12`                                          |
| `r-ncdf4`         | ✅ direct (and transitive)         | `src/weathergen/generate_weather.R:78-88` (spatial_ref workaround); also a weathergenr Imports                                |
| `r-doparallel`    | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-dplyr`         | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-e1071`         | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-fitdistrplus`  | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-foreach`       | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-ggplot2`       | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-patchwork`     | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-rlang`         | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-scales`        | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-tibble`        | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-tidyr`         | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-waveslim`      | ✅ transitive via weathergenr     | weathergenr DESCRIPTION Imports                                                                                                |
| `r-forecast`      | ❓ candidate for removal           | weathergenr DESCRIPTION lists `forecast` only under `Suggests`, not `Imports` — runtime path doesn't load it.                  |
| `r-devtools`      | ❓ candidate for removal           | Only referenced in `dev/scripts/install_weathergenr.R`, which is now **verify-only** (no `devtools::install_*` calls left).    |

The `r-base` declaration also pulls in `parallel`, `stats`, `utils`
(R recommended packages) which the workflow uses transitively via
weathergenr — those don't need separate pixi entries.

---

## Julia packages

Listed in `Project.toml` `[deps]`.

| Package | Status           | Used in                                                                                                                              |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `Wflow` | ✅ runtime        | Inline `julia +1.11.7 ... -e "using Wflow; Wflow.run()"` in `Snakefile_model_creation` and `Snakefile_climate_experiment`            |

`Manifest.toml` resolves 127 transitive packages (Pkg.instantiate
handles them). No other top-level Julia deps to track.

---

## Not fully in pixi

| Item             | Status                                                                                                                                                                                                                                     | Notes                                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Julia 1.11.7      | Managed by `juliaup` outside pixi. Snakefiles call `julia +1.11.7 ...`.                                                                                                                                                                    | conda-forge does **not** ship win-64 julia (linux-64 / osx-64 only, and even there it skips 1.11.x — 1.10 and 1.12 only, 1.12 broken for Wflow.jl per #884). Staying on juliaup is the only working setup until conda-forge ships 1.11 for Windows. R3 followup.                                                                                              |
| weathergenr 1.2.0 | Installed via `pixi run install-rdeps` (calls `dev/scripts/install_weathergenr.R` → `pak::pkg_install("tanerumit/weathergenr@v1.2.0")`). Lands in `.libPaths()[1]` — user lib on Windows, conda site-lib on Linux. Idempotent on rerun. | The install **command** is pixi-driven (so "everything upfront via pixi" holds at the command level). On Windows the **package itself** ends up in user lib because the conda r-base toolchain hits `Mingw-w64 runtime failure` when byte-compiling weathergenr's namespace against conda r-* deps. Loaded at workflow runtime via R's default `.libPaths()`. |

---

## Rtools (and mingw) status

**Not a declared dependency anywhere.** No mention in `pixi.toml`,
`Project.toml`, `pyproject.toml` (which doesn't exist), or
`environment.yml`.

The only references are two M2b-era comments noting it as a future
followup:

- `dev/scripts/install_weathergenr.R:18` — comment about a possible R3
  refactor "build weathergenr against the conda toolchain (likely
  needs `m2w64-toolchain` in pixi.toml)".
- `src/weathergen/global.R:7` — comment explaining why we trust
  default `.libPaths()` (paraphrased: forcing conda site-lib first
  caused weathergenr's imports to load from a mismatched mingw
  runtime ABI).

mingw-w64 binaries DO appear in `pixi.lock` as transitive deps of
conda-forge `r-base` on Windows (e.g. `mingw-w64-ucrt-x86_64-*`), but
that's the conda r-base build's compiler runtime, not Rtools per se.
Conda's r-base on Windows ships its own toolchain.

**Bottom line:** Removing Rtools is a non-issue — it was never declared.
The mingw runtime that comes with conda r-base is sufficient for
loading R packages. The compile path that fails today is "build a
GitHub R package against the conda r-base toolchain at install time",
which is a separate problem from "Rtools-or-not".

---

## Summary of cleanup candidates

If you want to trim the env to just what's used, these can be removed
without breaking the three workflows:

**Python conda (6):** `datrie`, `dvc`, `descartes`, `flit`, `jupyter`,
`tabulate`.

**Python pypi (2):** `hydroengine`, `gwwapi`.

**R (2 candidates):** `r-devtools` (only used by a now-verify-only
script), `r-forecast` (weathergenr Suggests, not Imports — would need
a weathergenr smoke test to confirm the runtime path doesn't touch
forecast).

**Julia (0):** Already minimal.

If instead you want to **add explicit pins** for the implicit Python
runtime deps so version drift is locked: `dask`, `netCDF4`,
`rasterio`, `PyYAML`.

Net change if you act on both: −10 declarations, +4 pins, same
behavior.
