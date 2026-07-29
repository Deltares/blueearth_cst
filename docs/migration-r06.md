# MIGRATION — R06 structural refactor

Complete old → new path map for the R06 structural refactor (milestone
`r06-refactor`). Use this to rebase a downstream fork, a user-local config, or any
script that imported `from src.` or referenced a `config/` / runner path directly.

The refactor is **pure layout**: every entry below is a file move + mechanical
reference rewrite. No computational path changed. The only produced-output content
change is the copied-config snapshot path rewrite (see "Config path values" below),
which alters no computed value.

Reference tips: pre-R6 = `e33ee45` (main tip before the milestone); the map is
derived from `git diff --find-renames --name-status e33ee45..<milestone tip>` and
audited for completeness (every rename row present; nothing dropped).

---

## 1. Python package: `src/` → `blueearth_cst/`

The flat `src/` package became `blueearth_cst/`, split by workflow stage
(`model/`, `projections/`, `experiment/`) with a `shared/` submodule for
cross-cutting helpers and `weathergen/` for the R weather generator. **Import
rewrite:** `from src.<module> import ...` → `from blueearth_cst.<stage>.<module>
import ...`; `from src import <module>` → `from blueearth_cst.<stage> import
<module>` (stage per the table). Bare cross-module imports inside a `script:` body
(e.g. `from func_plot_signature import ...`) also become fully-qualified
`blueearth_cst.<stage>.<module>` — see the fix commit
`r06: fix bare sibling imports missed in package move`.

### shared/ (cross-cutting helpers)

| Old | New |
| --- | --- |
| `src/snake_utils.py` | `blueearth_cst/shared/snake_utils.py` |
| `src/run_logged.py` | `blueearth_cst/shared/run_logged.py` |
| `src/func_plot_signature.py` | `blueearth_cst/shared/func_plot_signature.py` |
| `src/plot_map.py` | `blueearth_cst/shared/plot_map.py` |
| `src/merge_logs.py` | `blueearth_cst/shared/merge_logs.py` |
| `src/merge_benchmarks.py` | `blueearth_cst/shared/merge_benchmarks.py` |
| `src/metrics_definition.py` | `blueearth_cst/shared/metrics_definition.py` |
| `src/setup_time_horizon.py` | `blueearth_cst/shared/setup_time_horizon.py` |

### model/ (workflow 1 — model creation)

| Old | New |
| --- | --- |
| `src/prepare_build_config.py` | `blueearth_cst/model/prepare_build_config.py` |
| `src/setup_gauges_and_outputs.py` | `blueearth_cst/model/setup_gauges_and_outputs.py` |
| `src/setup_reservoirs_lakes_glaciers.py` | `blueearth_cst/model/setup_reservoirs_lakes_glaciers.py` |
| `src/write_outlet_index.py` | `blueearth_cst/model/write_outlet_index.py` |
| `src/get_region_preview.py` | `blueearth_cst/model/get_region_preview.py` |
| `src/plot_results.py` | `blueearth_cst/model/plot_results.py` |
| `src/plot_map_forcing.py` | `blueearth_cst/model/plot_map_forcing.py` |
| `src/climate_forcing.py` | `blueearth_cst/model/climate_forcing.py` |
| `src/copy_config_files.py` | `blueearth_cst/model/copy_config_files.py` |

### projections/ (workflow 2 — climate projections)

| Old | New |
| --- | --- |
| `src/prepare_climate_data_catalog.py` | `blueearth_cst/projections/prepare_climate_data_catalog.py` |
| `src/get_stats_climate_proj.py` | `blueearth_cst/projections/get_stats_climate_proj.py` |
| `src/get_change_climate_proj.py` | `blueearth_cst/projections/get_change_climate_proj.py` |
| `src/get_change_climate_proj_summary.py` | `blueearth_cst/projections/get_change_climate_proj_summary.py` |
| `src/plot_proj_timeseries.py` | `blueearth_cst/projections/plot_proj_timeseries.py` |

### experiment/ (workflow 3 — climate experiment)

| Old | New |
| --- | --- |
| `src/prepare_cst_parameters.py` | `blueearth_cst/experiment/prepare_cst_parameters.py` |
| `src/prepare_weagen_config.py` | `blueearth_cst/experiment/prepare_weagen_config.py` |
| `src/extract_historical_climate.py` | `blueearth_cst/experiment/extract_historical_climate.py` |
| `src/downscale_climate_forcing.py` | `blueearth_cst/experiment/downscale_climate_forcing.py` |
| `src/export_wflow_results.py` | `blueearth_cst/experiment/export_wflow_results.py` |

### weathergen/ (R weather generator — moved verbatim)

| Old | New |
| --- | --- |
| `src/weathergen/generate_weather.R` | `blueearth_cst/weathergen/generate_weather.R` |
| `src/weathergen/impose_climate_change.R` | `blueearth_cst/weathergen/impose_climate_change.R` |
| `src/weathergen/global.R` | `blueearth_cst/weathergen/global.R` |

The two R `Rscript --vanilla` shell paths in `Snakefile_climate_experiment` and the
internal `source("./src/weathergen/global.R")` calls were rewritten to
`blueearth_cst/weathergen/...`.

### Package markers and the deleted stray root `__init__.py`

- The **stray root-level `__init__.py`** (empty, tracked at the repo root pre-R6)
  was **deleted** — it was not a package marker for anything and had no reason to
  sit at the root. (Git's rename detection paired it by content identity with
  `blueearth_cst/__init__.py`; that pairing is a byte-identity artifact, not
  intent.)
- `src/__init__.py` (empty) is **removed**; the package now carries a marker per
  subpackage: `blueearth_cst/__init__.py`, `blueearth_cst/shared/__init__.py`,
  `blueearth_cst/model/__init__.py`, `blueearth_cst/projections/__init__.py`,
  `blueearth_cst/experiment/__init__.py`, and
  `blueearth_cst/climate_analysis/__init__.py` (all empty; the last was
  omitted from this list until R07 / O-10). `blueearth_cst/weathergen/` has no
  marker -- it holds R sources, not an importable Python subpackage. (Git paired
  `src/__init__.py` with `blueearth_cst/experiment/__init__.py` by byte identity —
  again a detection artifact.)

The `sys.path.insert(0, str(Path(workflow.basedir)))` shim in each Snakefile and
`conftest.py`'s `parents[1]` insert are unchanged (repo root is the package parent,
so `blueearth_cst` imports exactly as `src` did). `run_logged.py`'s own bootstrap
gained one extra `dirname` level because it now sits two levels under the repo root.

---

## 2. Config split: `config/` → `config/{workflows,catalogs,templates}/`

The flat `config/` directory was split into three bins.

### workflows/ (snake configs — the `--configfile` targets)

| Old | New |
| --- | --- |
| `config/snake_config.template.yml` | `config/workflows/snake_config.template.yml` |
| `config/snake_config_model_test.yml` | `config/workflows/snake_config_model_test.yml` |
| `config/snake_config_model_test_linux.yml` | `config/workflows/snake_config_model_test_linux.yml` |
| `config/snake_config_projections_cmip5_full.yml` | `config/workflows/snake_config_projections_cmip5_full.yml` |
| `config/snake_config_projections_cmip5_full_linux.yml` | `config/workflows/snake_config_projections_cmip5_full_linux.yml` |
| `config/snake_config_projections_cmip6_full.yml` | `config/workflows/snake_config_projections_cmip6_full.yml` |
| `config/snake_config_projections_isimip3.yml` | `config/workflows/snake_config_projections_isimip3.yml` |
| `config/snake_config_projections_isimip3_linux.yml` | `config/workflows/snake_config_projections_isimip3_linux.yml` |

### catalogs/ (hydromt data catalogs — the `-d` targets)

| Old | New |
| --- | --- |
| `config/deltares_data.yml` | `config/catalogs/deltares_data.yml` |
| `config/deltares_data_linux.yml` | `config/catalogs/deltares_data_linux.yml` |
| `config/deltares_data_climate_projections.yml` | `config/catalogs/deltares_data_climate_projections.yml` |
| `config/deltares_data_climate_projections_linux.yml` | `config/catalogs/deltares_data_climate_projections_linux.yml` |
| `config/cmip6_data.yml` | `config/catalogs/cmip6_data.yml` |

Catalog-internal `uri:` entries are unaffected: `deltares_data*.yml` resolve against
an absolute `meta.roots`, and `cmip6_data.yml` uses location-independent `gs://`
URIs.

### templates/ (hydromt / wflow / weathergen build templates)

| Old | New |
| --- | --- |
| `config/wflow_build_model.yml` | `config/templates/wflow_build_model.yml` |
| `config/wflow_update_waterbodies.yml` | `config/templates/wflow_update_waterbodies.yml` |
| `config/weathergen_config.yml` | `config/templates/weathergen_config.yml` |
| `config/wflow_sbm.toml` | `config/templates/wflow_sbm.toml` |

**`config/wflow_sbm.toml` note.** This tracked file is not referenced by any live
config path (workflows build the run TOML via hydromt_wflow; the tracked copy is a
reference template). It was placed under `templates/` by kind. Provisional
placement — flagged for the milestone gate.

### Config path values rewritten inside configs (not just file moves)

Moving catalogs/templates required rewriting the path **values** that point at them.
The rewritten keys (old flat → new binned path), applied in the workflow configs:

| Key | Old value pattern | New value pattern |
| --- | --- | --- |
| `data_sources` | `config/deltares_data*.yml`, `config/cmip6_data.yml` | `config/catalogs/...` |
| `data_sources_climate` | `config/deltares_data_climate_projections*.yml`, `config/cmip6_data.yml` | `config/catalogs/...` |
| `model_build_config` | `config/wflow_build_model.yml` | `config/templates/wflow_build_model.yml` |
| `waterbodies_config` | `config/wflow_update_waterbodies.yml` | `config/templates/wflow_update_waterbodies.yml` |

`data_sources_climate` is a 4th catalog-path key beyond the design's original three
(`data_sources`, `model_build_config`, `waterbodies_config`); its rewrite is
included above. This table is kept **in lockstep** with
`dev/scripts/semantic_tree_diff.py`'s `COPIED_CONFIG_PATH_MAP` (same four keys, same
old→new values) — that map is what the run-relative baseline and full-tree diffs use
to normalize the copied-config snapshots. (The stale `docs/config/` mirror that
also carried these paths was deleted in R07 / O-05 — `config/` is now the single
source.)

Snakefile changes tied to the config split: the `static_dir`-based template default
expressions gained `/templates/` (`static_dir` itself stays `config`); the
`--dry-run`-blind `default_config` param at `Snakefile_climate_experiment:131`
became `config/templates/weathergen_config.yml`.

### `.gitignore` and user-local configs

- `.gitignore` already ignores `*_local.yml` (line 144) and `*_gabon.yml` (line 145)
  as unanchored globs, so both patterns cover the new `config/workflows/` paths — no
  `.gitignore` change was needed.
- **`config/snake_config_model_test_gabon.yml`** is an **untracked, gitignored**
  per-machine variant. It was **filesystem-moved** into `config/workflows/` (it is in
  no commit, so it appears in no rename audit). If you carry a local `*_gabon.yml` or
  `*_local.yml`, move it into `config/workflows/` yourself and rewrite its
  `data_sources` / `data_sources_climate` / `model_build_config` /
  `waterbodies_config` path values per the table above.
- **Recommendation:** rename `*_gabon.yml` variants to `*_local.yml` when relocating
  them, so a single gitignore convention (`*_local.yml`) covers all user-local
  configs uniformly. This is a suggestion, not a forced change.

---

## 3. Runners: repo root → `scripts/`

| Old | New |
| --- | --- |
| `run_snake_test.cmd` | `scripts/run_snake_test.cmd` |
| `run_snake_docker.sh` | `scripts/run_snake_docker.sh` |

Their internal `config/...` references were updated to `config/workflows/...`; the
per-workflow flags are preserved verbatim (`--keep-going` on the projections line
only). `scripts/run_workflows.py` (the new `enabled:`-aware wrapper) also lives here.

---

## 4. Test fixture and dev-script rewrites (in place — not moved)

These files stayed put but had references rewritten:

- `tests/conftest.py`, and the `tests/test_*.py` modules importing a moved target,
  were rewritten to the `blueearth_cst.<stage>.<module>` import forms.
- `tests/snake_config_model_test.yml` lines 23–24 (`model_build_config` /
  `waterbodies_config`) → `config/templates/...`.
- `dev/scripts/probe_attrs_chain.py:101` →
  `from blueearth_cst.projections.get_change_climate_proj_summary import ...`.
  (`dev/scripts/stage_data.py` was **not** a rewrite site — its `src` tokens are a
  local variable.)

---

## Import-prefix migration cheat-sheet

For a downstream script or fork:

| Old import | New import |
| --- | --- |
| `from src.snake_utils import ...` | `from blueearth_cst.shared.snake_utils import ...` |
| `from src.run_logged import ...` | `from blueearth_cst.shared.run_logged import ...` |
| `from src.<shared module> import ...` | `from blueearth_cst.shared.<module> import ...` |
| `from src.<model module> import ...` | `from blueearth_cst.model.<module> import ...` |
| `from src.<projections module> import ...` | `from blueearth_cst.projections.<module> import ...` |
| `from src.<experiment module> import ...` | `from blueearth_cst.experiment.<module> import ...` |
| `from src import <module>` | `from blueearth_cst.<stage> import <module>` |

Use the §1 tables above to look up the `<stage>` for any given module.
