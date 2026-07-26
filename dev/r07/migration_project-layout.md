# Migration — R07 project layout (old → new)

> Companion to `dev/r07/project-layout-design.md` (**DRAFT** — not yet accepted,
> not yet implemented). Required by `dev/conventions/naming.md` §7, which mandates
> a migration note for renames of `rule all` output filenames, checked-in example
> config keys, and fixture paths read by `tests/conftest.py` /
> `dev/scripts/check_baseline.py`.
>
> **Scope: paths move, values do not.** No computational path is edited. Two
> additive exceptions, both new artifacts rather than changed ones: the
> source-grid climate figures (design §B4) and the parse-time `project_dir`
> warning (O-22).
>
> Reference tip: pre-R07 = `75eb4d6`. Fill in the milestone tip on landing.

---

## 1. Repository paths

| Old | New | Note |
| --- | --- | --- |
| `data/observations/output-locations-test.csv` | *(deleted)* | Gabon test values; recoverable from git history |
| `data/observations/observations_timeseries_test.csv` | *(deleted)* | as above |
| — | `config/templates/observations/output_locations.csv` | NEW — header + 2 illustrative rows, comma-separated |
| — | `config/templates/observations/observations_timeseries.csv` | NEW — header + ~3 rows, **semicolon**-separated |
| — | `config/templates/observations/README.md` | NEW — schemas, the separator asymmetry, the `wflow_id` join rule |
| `docs/config/*` (16 files) | *(deleted)* | Pre-R6 duplicates of `config/` |
| `examples/` | `test_case/` | Gitignored fixture root |
| `examples/test_local/` | `test_case/test_local/` | The baseline seed's `project_dir` |
| `dag/` (repo root) | `<project_dir>/dag/` | Gitignored; `dag/` leaves `.gitignore` |
| `dag_model.png` (repo root) | *(deleted)* | Stale untracked artifact |

### Config key *values* that change

| File | Key | Old | New |
| --- | --- | --- | --- |
| `config/workflows/snake_config_model_test.yml` | `project.project_dir` | `examples/test_local` | `test_case/test_local` |
| `config/workflows/snake_config_model_test_linux.yml` | `project.project_dir` | `examples/Gabon` | `test_case/gabon` |
| `config/workflows/snake_config_model_test_linux.yml` | `…model_creation.output_locations` | `data/observations/output-locations-test.csv` | `None` |
| `config/workflows/snake_config_model_test_linux.yml` | `…model_creation.observations_timeseries` | `data/observations/observations_timeseries_test.csv` | `None` |
| `tests/snake_config_model_test.yml` | both observation keys | `tests/data/observations/…` *(dangling — no such dir)* | `None` |
| `config/workflows/snake_config.template.yml` | `project.project_dir` | `examples/test` | outside-the-tree placeholder |

**Sentinel warning.** Unquoted `None` parses to the Python **string** `"None"`,
not YAML `null`. `os.path.isfile("None")` / `os.path.exists("None")` are False, so
the guards at `setup_gauges_and_outputs.py:55` and `plot_results.py:127` work. Real
YAML `null` raises `TypeError` in both. Every `None` above must be byte-identical
to `snake_config_model_test.yml:36-37`.

### Reference sites to rewrite

`scripts/run_snake_test.cmd:32` (DAGDIR) · `scripts/run_snake_docker.sh:7` (drop
the `data/` mount) · `.gitignore:124,135-136` · `README.rst:204,211,218,254,259,269,285,298` ·
`AGENTS.md` (repo map, `docs/` description, the invocation-model line) ·
`MIGRATION.md:173` · `docs/notebooks/*.ipynb` (six DAG cells) ·
`docs/cst-toolbox-technical-note-2025.md:1278-1279` · `dev/workflows/model_creation.md:15-16,115` ·
`dev/scripts/check_baseline.py` · `dev/followups.md:171-184`.

---

## 2. `project_dir` paths

All paths project-root-relative. `<key>` = `<clim_source>_<YYYYMMDD>_<YYYYMMDD>`;
`<id>` = `experiment_name`; `<r>` / `<c>` = realization / stress-test index.

### 2a. Project level

| Class | Old | New |
| --- | --- | --- |
| wf1 config snapshot | `config/snake_config_model_creation.yml` | `config/runs/snake_config_model_creation.yml` |
| wf2 config snapshot | `config/snake_config_climate_projections.yml` | `config/runs/snake_config_climate_projections.yml` |
| Copied catalogs | `config/{deltares_data,cmip6_data}.yml` | `config/catalogs/…` |
| Copied wflow templates | `config/wflow_*.yml` (4 files) | `config/templates/…` |
| Raw climate store (wf1) | `climate_historical/wf1_raw/extract_historical.nc` | `climate_historical/<key>/extract_historical.nc` — **merged with the wf3 store** |
| Orography sidecar | `climate_historical/wf1_raw/orography.nc` | `climate_historical/<key>/orography.nc` |
| Raw climate store (wf3) | `climate_historical/<key>/extract_historical.nc` | *unchanged path; now the single store* |
| Climate figures | — | `climate_historical/<key>/plots/` — **NEW producer** (§B4) |
| wflow forcing | `climate_historical/wflow_data/inmaps_historical.nc` | `hydrology_model/forcing/inmaps_historical.nc` |
| Forcing figures | `plots/wflow_model_performance/{precip,temp,pet}.png` | `hydrology_model/forcing/plots/` |
| Model-evaluation figures | `plots/wflow_model_performance/{hydro_wflow_1,basin_area,clim_wflow_1_month,clim_wflow_1_year}.png` | `hydrology_model/evaluation/plots/` |
| Performance metrics | `plots/wflow_model_performance/performance_metrics.csv` | `hydrology_model/evaluation/performance_metrics.csv` — out of a `plots/` dir (P1) |
| wf2 processed | `climate_projections/<proj>/gcm_timeseries.nc` | `climate_projections/<proj>/timeseries/gcm_timeseries.nc` |
| wf2 summaries | `climate_projections/<proj>/annual_change_scalar_stats_summary*.{nc,csv}` | `climate_projections/<proj>/summary/…` |
| wf2 figures | `climate_projections/<proj>/plots/` | *unchanged* |

Unchanged by design: `hydrology_model/` model root itself (`wflow_sbm.toml`,
`staticmaps.nc`, `staticgeoms/`, `hydromt.log`, `hydromt_data.yml`),
`hydrology_model/run_default/`, project-level `logs/` and `benchmarks/`.

### 2b. Experiment subtree

| Class | Old | New |
| --- | --- | --- |
| Results | `experiments/<id>/model_results/{Qstats,basin,RT_*}.csv` | `experiments/<id>/indicators/…` |
| Result figures | — | `experiments/<id>/indicators/plots/` (reserved; no producer yet) |
| Run TOMLs | `experiments/<id>/model_runs/wflow_sbm_rlz_<r>_cst_<c>.toml` | `experiments/<id>/hydrology_runs/rlz_<r>/config/cst_<c>.toml` |
| Run outputs | `experiments/<id>/model_runs/output_rlz_<r>_cst_<c>.csv` | `experiments/<id>/hydrology_runs/rlz_<r>/output/cst_<c>.csv` |
| Outstates | `experiments/<id>/model_runs/outstates_*.nc` | `experiments/<id>/hydrology_runs/rlz_<r>/output/` |
| Weathergen base config | `experiments/<id>/weathergen_config.yml` | `experiments/<id>/weather_generator/config/weathergen_config.yml` |
| Per-member weathergen configs | `experiments/<id>/realization_<r>/weathergen_config_rlz_<r>_cst_<c>.yml` | `experiments/<id>/weather_generator/_work/` |
| Per-member forcing (`temp()`) | `experiments/<id>/realization_<r>/inmaps_rlz_<r>_cst_<c>.nc` | `experiments/<id>/weather_generator/output/` |
| Stress-test grid | `experiments/<id>/stress_test/cst_<c>.csv` | `experiments/<id>/weather_generator/_work/cst_<c>.csv` |
| Weathergen figures | `experiments/<id>/{obs_power_spectra,warm_annual_precip,warm_annual_stats,warm_annual_wavelet}.png` | `experiments/<id>/weather_generator/plots/` |
| Weathergen date CSVs | `experiments/<id>/{resampled_dates,sim_dates}.csv` | `experiments/<id>/weather_generator/output/` |

`experiments/<id>/realization_<r>/` **dissolves**: configs to `_work/`, netCDFs to
`output/`.

Unchanged: `experiments/<id>/config/snake_config_climate_experiment.yml`,
`data_catalog_climate_experiment.yml`, `.project_consistency_ok`,
`experiments/<id>/{logs,benchmarks}/`.

### In-TOML pointer strings that change value

Same targets, new depths; hydromt re-relativizes on write. Compare in a
project-root-relative namespace with the path-aware TOML comparator
(`dev/scripts/semantic_tree_diff.py`), never as raw strings.

| Key | Old | New |
| --- | --- | --- |
| `input.path_forcing` (project TOML) | `../climate_historical/wflow_data/inmaps_historical.nc` | `forcing/inmaps_historical.nc` |
| `input.path_static` (run TOMLs) | `../../../hydrology_model/staticmaps.nc` | recomputed for `hydrology_runs/rlz_<r>/config/` depth |
| `state.path_input` (run TOMLs) | `../../../hydrology_model/instate/instates.nc` | recomputed |
| `input.path_forcing` (run TOMLs) | `../realization_<r>/inmaps_….nc` | `../../../weather_generator/output/inmaps_….nc` |

---

## 3. Baseline manifest impact

Eighteen targets in `dev/baseline/manifest.json`. **Re-record once**, at the end of
the milestone.

| Target class | Count | Impact |
| --- | --- | --- |
| Copied config snapshots | 4 | **Path and content** — the `config/runs/` split, plus the embedded `project_dir` string changing with the `examples/` → `test_case/` rename |
| wf1 plots | 3 | Path — split across `hydrology_model/{forcing,evaluation}/plots/` |
| wf2 summary + plots | 6 | Path — the `summary/` tier |
| wf3 results | 2 | Path — `model_results/` → `indicators/` |
| wf2 netCDF summary | 1 | Path |
| wf1 discharge (`hydrology_model/run_default/output.csv`) | 1 | **Unchanged** |
| Newly declared plot outputs (O-24) | +4 | Additive: `temp.png`, `pet.png`, `clim_wflow_1_{month,year}.png` |

Also update: `dev/scripts/check_baseline.py` `TARGETS` templates ·
`dev/scripts/semantic_tree_diff.py` directory-prefix path map and TOML comparator.

## 4. MISSING/EXTRA allowlist

To be filled at the milestone gate, in the form of
`dev/p31/migration_experiment-structure.md` §2: every entry carries a written
justification, and an entry not listed fails the gate. Expected classes:

- **EXTRA** — the new source-grid climate figures (§B4); the four newly declared
  plot outputs (O-24).
- **MISSING** — none by design. `climate_historical/wf1_raw/` disappears by
  merging into `<key>/`, which the path map covers as a rename, not a deletion.
