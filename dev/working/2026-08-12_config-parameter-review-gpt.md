# Configuration parameter review

## Q1 · Reach — which declared parameters never reach the computation?

### Inventory remeasurement

The appendix is stale and internally inconsistent:

- It says 55 project leaves, but its own lists expand to 56: 4 `project` + 15 `shared` (the stated 13 plus `hydrography` and `basin_index`) + 7 model-creation + 10 projections + 20 experiment keys (`dev/working/parameter-placement.md:56-64`, `dev/working/parameter-placement.md:163-187`).
- The current template additionally exposes `shared.seed`, `shared.water_year_start`, `shared.julia_threads`, `relative_change.{min_reference,max_flagged_months}`, and `{batch_size,batch_size_max}` (`config/templates/snake_config.template.yml:41-67`, `config/templates/snake_config.template.yml:131-134`, `config/templates/snake_config.template.yml:166-167`).
- Conversely, `save_grids` is retired, and the appendix’s `stress_test.temp.variance.{min,max}` keys have no template entry or runtime reader (`blueearth_cst/projections/gridded_outputs.py:7-30`, `blueearth_cst/experiment/prepare_cst_parameters.py:92-100`).

The auditable universe is therefore:

- 60 current active project leaves;
- 1 retired compatibility key, `save_grids`;
- 2 appendix-only, silently ignored temperature-variance leaves;
- 63 names total covered below.

The six-file configuration-value scan was run over the template, all four `test_case/snake_config_*.yml` files, and `tests/snake_config_fixture.yml`. It found every active template/config leaf and was cross-checked against every `get_config`, direct mapping access, parser access, and wrapper access in the three Snakefiles, `blueearth_cst/**`, and `scripts/run_workflows.py`.

### Project-config classification

| Keys | Class | Evidence and terminal consumer |
|---|---:|---|
| `project.{project_dir,static_dir,data_sources,data_sources_climate}` | d | `project_dir` determines every project output root; the three workflows read it at `Snakefile_model_creation:50`, `Snakefile_climate_projections:34`, and `Snakefile_climate_experiment:37`. `static_dir` supplies WF1’s fallback template paths at `Snakefile_model_creation:115-116`. The two catalogs feed rules and catalog resolution at `Snakefile_model_creation:55`, `Snakefile_climate_projections:38-70`, and `Snakefile_climate_experiment:42`. |
| `shared.basin.{region,resolution,hydrography,basin_index,gauge_points,automatic_subbasins.max_per_basin,gauge_snap_tolerance_m,river_uparea_km2,spatial_sources.{rivers,lulc,lai,soil}}` | d | All are resolved in `parse_spatial_config` (`blueearth_cst/spatial/config.py:207-270`). They reach delineation through the shared rule payload (`blueearth_cst/shared/snake_utils.py:1225-1230`); the numerical consumers include grid resolution and river threshold at `blueearth_cst/spatial/hydrography.py:115-123,227`, gauge snapping at `blueearth_cst/spatial/products.py:682-723`, subbasin limits at `blueearth_cst/spatial/products.py:330-387`, and thematic source reads at `blueearth_cst/spatial/products.py:658-669,830-834`. |
| `shared.historical_window.{starttime,endtime}` | d | Parsed and validated at `blueearth_cst/shared/snake_utils.py:891-931,1366-1367`; it keys and slices the climate store and bounds the model simulation. |
| `shared.clim_historical` | d | Read by all workflows (`Snakefile_model_creation:88`, `Snakefile_climate_projections:78`, `Snakefile_climate_experiment:159`) and selects the historical climate source. |
| `shared.seed` | d | Resolved at `Snakefile_climate_experiment:96`, injected at `blueearth_cst/experiment/prepare_weagen_config.py:102-106`, and used by generation and perturbation (`blueearth_cst/weathergen/impose_climate_change.R:92-105`). |
| `shared.water_year_start` | d | Read by all workflows (`Snakefile_model_creation:46`, `Snakefile_climate_projections:117`, `Snakefile_climate_experiment:101`); it now reaches WF2 bounds and annual arithmetic at `blueearth_cst/projections/derive_change_factors.py:152-185` and is converted to weathergenr’s integer argument at `blueearth_cst/experiment/prepare_weagen_config.py:107-110`. |
| `shared.julia_threads` | d | Read at `Snakefile_model_creation:94` and passed into the Julia command at `blueearth_cst/shared/snake_utils.py:738-744`. |
| `workflows.*.enabled` ×3 | d | The Snakefiles intentionally do not read them; the orchestration wrapper validates each flag at `scripts/run_workflows.py:132-158` and skips or invokes workflows at `scripts/run_workflows.py:209-231`. |
| `workflows.model_creation.{model_build_config,waterbodies_config}` | d | Become declared rule inputs/params at `Snakefile_model_creation:514,564`; the first is parsed and applied at `blueearth_cst/model/build_wflow_model.py:294-320`, and the second is forwarded by method name and keyword arguments at `blueearth_cst/model/setup_reservoirs_lakes_glaciers.py:30-34,83-86`. |
| `workflows.model_creation.wflow_outvars` | d | Read at `Snakefile_model_creation:110`, passed to output declaration at `Snakefile_model_creation:584`, and determines output and indicator targets at `Snakefile_model_creation:805-831` and `Snakefile_climate_experiment:476-479`. |
| `workflows.model_creation.observations_timeseries` | d | Read at `Snakefile_model_creation:118`, conditionally becomes a real input at `Snakefile_model_creation:149-150`. |
| `workflows.model_creation.simulation_window.{starttime,endtime}` | d | Resolved and bounded at `blueearth_cst/shared/snake_utils.py:891-931`, then passed to forcing/model time at `Snakefile_model_creation:683-684`. |
| `workflows.climate_projections.{clim_project,models,scenarios,members}` | d | Read at `Snakefile_climate_projections:80-83` and jointly resolve the actual ensemble at `Snakefile_climate_projections:399-405`. |
| `workflows.climate_projections.variables` | d | Parsed into quantity, units, and change semantics at `Snakefile_climate_projections:84-94`; those semantics reach thresholds, reducers, and output metadata at `Snakefile_climate_projections:191,786,834,922`. |
| `workflows.climate_projections.{historical_year_range,future_horizons}` | d | Read at `Snakefile_climate_projections:118-119`; the historical range is clipped at `Snakefile_climate_projections:134-147`, and horizon values are read for each wildcard at `Snakefile_climate_projections:566`. |
| `workflows.climate_projections.stats` | d | Read at `Snakefile_climate_projections:179` and passed to change-factor derivation at `Snakefile_climate_projections:903`. |
| `workflows.climate_projections.relative_change.min_reference` | d | Resolved at `Snakefile_climate_projections:189-192` and used in change-factor and flagging arithmetic at `blueearth_cst/projections/derive_change_factors.py:414,641`. |
| `workflows.climate_projections.relative_change.max_flagged_months` | d | Resolved at `Snakefile_climate_projections:193-195` and controls combination flags and report disclaimers at `blueearth_cst/projections/derive_change_factors.py:627,642`. |
| `workflows.climate_experiment.experiment_name` | d | Explicit value or derived name selects the experiment tree at `Snakefile_climate_experiment:59-81`; it also participates in seed derivation and freeze identity. |
| `workflows.climate_experiment.realizations_num` | d | Read at `Snakefile_climate_experiment:103`; it sizes realization outputs and the run cross-product at `Snakefile_climate_experiment:843-890,953`. |
| `workflows.climate_experiment.horizontime_climate` | d | Read at `Snakefile_climate_experiment:235`; passed as `middle_year` at `Snakefile_climate_experiment:808` and used in generated-year arithmetic at `blueearth_cst/experiment/prepare_weagen_config.py:26-33`. |
| `workflows.climate_experiment.run_length` | d | Read at `Snakefile_climate_experiment:236`; passed both to weather-series sizing and Wflow forcing-window calculation at `Snakefile_climate_experiment:809,933`. |
| `workflows.climate_experiment.run_historical` | d | Read at `Snakefile_climate_experiment:128`; it changes `ST_START` and therefore whether `st_0` is run at `Snakefile_climate_experiment:129,1021`. |
| `stress_test.{temp,precip}.step_num` | d | Both are required and converted to inclusive grid counts at `blueearth_cst/shared/snake_utils.py:1435-1481`; the counts size the parameter linspaces at `blueearth_cst/experiment/prepare_cst_parameters.py:87-109`. |
| `stress_test.temp.mean.{min,max}` | d | Read at `blueearth_cst/experiment/prepare_cst_parameters.py:92-94` and used to build monthly temperature offsets at `blueearth_cst/experiment/prepare_cst_parameters.py:102-104`. |
| `stress_test.precip.mean.{min,max}` | d | Read and used at `blueearth_cst/experiment/prepare_cst_parameters.py:96-107`. |
| `stress_test.precip.variance.{min,max}` | d | Read and used at `blueearth_cst/experiment/prepare_cst_parameters.py:99-110`. |
| `stress_test.{temp,precip}.transient_change` | d | Required at `blueearth_cst/experiment/prepare_weagen_config.py:36-50`, injected at `blueearth_cst/experiment/prepare_weagen_config.py:123-129`, and passed into perturbation arithmetic at `blueearth_cst/weathergen/impose_climate_change.R:48-74`. |
| `stress_test.{dry_spell_factor,wet_spell_factor}` | d | Validated at `Snakefile_climate_experiment:119-125`, injected at `Snakefile_climate_experiment:812-813`, and included in the generator configuration at `blueearth_cst/experiment/prepare_weagen_config.py:111-115`. |
| `workflows.climate_experiment.{batch_size,batch_size_max}` | d | Parsed and validated at `Snakefile_climate_experiment:970-987`; they determine the member partition at `Snakefile_climate_experiment:989-990`. |
| `workflows.climate_projections.save_grids` | b | Retired compatibility read only. `false` is warned and ignored; `true` raises before execution (`blueearth_cst/projections/gridded_outputs.py:14-30`). It cannot affect a gridded branch because that branch no longer exists (`Snakefile_climate_projections:149-168`). |
| `stress_test.temp.variance.{min,max}` | a | Appendix-only pseudo-parameters. The sole stress-grid consumer reads temperature mean and precipitation mean/variance, never temperature variance (`blueearth_cst/experiment/prepare_cst_parameters.py:92-110`). They are also absent from the current template’s temperature block (`config/templates/snake_config.template.yml:143-149`). |

`project.static_dir` has a partial reach defect even though its global classification is d: WF1 uses it only as a fallback prefix (`Snakefile_model_creation:115-116`), while WF3 binds it and never refers to the variable again (`Snakefile_climate_experiment:41`).

### Engine-native `config/defaults/*.yml`

#### `weathergen_config.yml`

The actual checked-in surface is 34 leaves, not the appendix’s “2 + 16 + 15 + 5”:

- `run_weather_generator`: `eval_max_grids`, `log_messages` (`config/defaults/weathergen_config.yml:26-35`);
- `generate_weather`: `vars`, `warm_var`, `warm_signif`, `warm_pool_size`, `warm_filter_bounds`, `annual_knn_n`, `wet_q`, `extreme_q`, `parallel`, `n_cores`, `verbose`, `save_plots` (`config/defaults/weathergen_config.yml:39-87`);
- `apply_climate_perturbations`: all 15 leaves from `compute_pet` through `diagnostic` (`config/defaults/weathergen_config.yml:107-136`);
- `write_netcdf`: `calendar`, `compression`, `spatial_ref`, `signif_digits`, `verbose` (`config/defaults/weathergen_config.yml:145-150`).

All 34 are d. The generation mapping is passed as the wrapper’s `config` argument and the two wrapper keys are explicit arguments (`blueearth_cst/weathergen/generate_weather.R:115-125`); perturbation leaves are explicit arguments (`blueearth_cst/weathergen/impose_climate_change.R:65-105`); and netCDF leaves are explicit arguments in both R scripts (`blueearth_cst/weathergen/generate_weather.R:179-191`, `blueearth_cst/weathergen/impose_climate_change.R:110-125`). Eight generation keys and `write_netcdf.file_prefix` are injected later rather than declared in the defaults file (`blueearth_cst/experiment/prepare_weagen_config.py:96-121`).

#### `wflow_build_model.yml`

There are 25 leaves: `modeltype`; six `setup_rivers` arguments; one `setup_lulcmaps`; one `setup_laimaps`; two `setup_soilmaps`; and fourteen `setup_constant_pars` values (`config/defaults/wflow_build_model.yml:1-49`).

- Twenty-four are d. `modeltype` is a fail-loud discriminator (`blueearth_cst/model/build_wflow_model.py:45-51`); other kwargs reach HydroMT methods through `_apply_parameter_steps` (`blueearth_cst/model/build_wflow_model.py:207-240`).
- `steps.setup_laimaps.lai_fn` is c, forwarded then dropped. The parser retains it in `kwargs`, but `_apply_parameter_steps` explicitly removes it and supplies the already-built P1 LAI raster instead (`blueearth_cst/model/build_wflow_model.py:233-238`). The chain stops at line 234.

#### `wflow_update_waterbodies.yml`

All seven leaves are d: three reservoir-control arguments, two uncontrolled-reservoir arguments, and two glacier arguments (`config/defaults/wflow_update_waterbodies.yml:1-12`). Each section name becomes a method and every leaf is expanded as `**kwargs` (`blueearth_cst/model/setup_reservoirs_lakes_glaciers.py:30-34`).

### All `DEFAULT_*` constants

The tree contains 14 unique names but 15 definitions because `DEFAULT_ANCHOR` is defined twice. The draft obtained “14” by omitting `DEFAULT_RIVER_UPAREA_KM2` and counting the two anchor definitions separately (`dev/working/parameter-placement.md:197-205`).

| Identifier | Class | Definition and consumer |
|---|---:|---|
| `DEFAULT_ANCHOR` ×2 | d | Defined at `blueearth_cst/shared/metrics_definition.py:18` and `blueearth_cst/climate_analysis/climate_figures.py:120`; used as annual aggregation defaults at `metrics_definition.py:22-73` and `climate_figures.py:129-155`. |
| `DEFAULT_BASIN_INDEX` | d | `blueearth_cst/shared/snake_utils.py:1004`; consumed at `blueearth_cst/spatial/config.py:232`. |
| `DEFAULT_DECIMALS` | d | `blueearth_cst/shared/tidy_wflow_table.py:55`; controls result rounding at `blueearth_cst/shared/tidy_wflow_table.py:124,168`. |
| `DEFAULT_GAUGE_SNAP_TOLERANCE_M` | d | `blueearth_cst/spatial/config.py:25`; fallback at `blueearth_cst/spatial/config.py:258-260`. |
| `DEFAULT_HYDROGRAPHY` | d | `blueearth_cst/shared/snake_utils.py:1003`; fallback at `blueearth_cst/spatial/config.py:231`. |
| `DEFAULT_JULIA_THREADS` | d | Re-export of advanced settings at `blueearth_cst/shared/snake_utils.py:625`; consumed at `Snakefile_model_creation:94`. |
| `DEFAULT_MAX_FLAGGED_MONTHS` | d | `blueearth_cst/projections/dry_month.py:38`; fallback at `Snakefile_climate_projections:193-195`. |
| `DEFAULT_MAX_SUBBASINS_PER_BASIN` | d | `blueearth_cst/spatial/config.py:21`; fallback at `blueearth_cst/spatial/config.py:253-256`. |
| `DEFAULT_MIN_REFERENCE` | d | `blueearth_cst/projections/dry_month.py:32`; used by threshold resolution at `blueearth_cst/projections/dry_month.py:66-75`. |
| `DEFAULT_RIVER_UPAREA_KM2` | d | `blueearth_cst/spatial/config.py:26`; fallback at `blueearth_cst/spatial/config.py:262-264`. |
| `DEFAULT_SEED` | d | Re-export at `blueearth_cst/shared/snake_utils.py:647`; fallback at `blueearth_cst/shared/snake_utils.py:688`. |
| `DEFAULT_SPELL_FACTOR` | d | `blueearth_cst/shared/snake_utils.py:1485`; fallback at `blueearth_cst/shared/snake_utils.py:1499-1500`. |
| `DEFAULT_STATS` | d | `blueearth_cst/projections/get_change_climate_proj.py:104`; used at `blueearth_cst/projections/get_change_climate_proj.py:161,270`. |
| `DEFAULT_WATER_YEAR_START` | d | Re-export at `blueearth_cst/shared/snake_utils.py:708`; fallback at `blueearth_cst/shared/snake_utils.py:714`. |

### Inertness falsifier searches

PowerShell could not be started in the read-only sandbox (`CreateProcessAsUserW: access denied`), so these were actual recursive UTF-8 source scans through the available Node filesystem runtime. Each scan covered all three Snakefiles plus `blueearth_cst/**` and `scripts/run_workflows.py`, reporting `file:line`.

| Claim | Search run | Result |
|---|---|---|
| `temp.variance.*` has a consumer | Regex `temp.{0,40}variance\|variance.{0,40}temp\|\["temp"\].*\["variance"\]\|\$temp\$variance` | Only unrelated output-column/header mentions at `blueearth_cst/experiment/prepare_cst_parameters.py:43` and `blueearth_cst/shared/interchange_contracts.py:224`; no config access. |
| WF3 consumes `static_dir` | Exact `static_dir` | WF1 read/fallbacks at `Snakefile_model_creation:54,115-116`; WF3 only the binding at `Snakefile_climate_experiment:41`. |
| `lai_fn` reaches HydroMT | Exact `lai_fn` | `kwargs.pop("lai_fn", None)` then replacement with the P1 raster at `blueearth_cst/model/build_wflow_model.py:234,238`. |
| `save_grids` controls output | Exact `save_grids` | Only retired-key registry at `blueearth_cst/projections/gridded_outputs.py:7` and a comment at `Snakefile_climate_projections:183`. |
| Current code consumes `relax_priority` | Exact `relax_priority` | Only the deliberate-absence comment at `blueearth_cst/shared/interchange_contracts.py:270-273`. |
| Current code consumes `evaluate.model` | Regex `evaluate[._]model` | Only historical comments at `blueearth_cst/weathergen/generate_weather.R:110-114`. |
| Legacy T2 `start_month_hyd_year` is still honoured | Exact `start_month_hyd_year` | The T2 key is read solely to refuse it at `Snakefile_climate_projections:107-116`; current arithmetic receives `shared.water_year_start` through `blueearth_cst/projections/derive_change_factors.py:152-185`. |

Thus the additional inert parameters are the two appendix-only temperature-variance leaves and `wflow_build_model.yml`’s `setup_laimaps.lai_fn`. Gate 1’s override was applied: they are recorded here and the review continued.

### Tier-3 read-back finding

The draft’s “generated = record, never input” rule is false as a generalization:

- Generated `model_reference.yml` is an explicit input to the drift guard (`Snakefile_climate_experiment:610,661`).
- Generated `experiment.yml` is read back to enforce immutable experiment semantics (`blueearth_cst/experiment/write_experiment_config.py:113-139,164-181`).
- Generated `weathergen_config.yml` is output by rule 3.10 and input to rules 3.11/3.12 (`Snakefile_climate_experiment:797-803,841,875`).
- Generated `wflow_sbm.toml` is model output/state and subsequently an explicit WF3 input (`Snakefile_model_creation:527`, `Snakefile_climate_experiment:605,662`).

`config/runs/**/{source.yml,effective.yml,referenced-files.json}` remain write-only provenance records in runtime code: their only runtime occurrences are writes in `blueearth_cst/model/copy_config_files.py:116-122,185-191`.

---

## Q2 · Necessity — which live parameters should not be user-facing?

### Definite candidates

| Key | Values across template + four test-case configs + fixture | Finding |
|---|---|---|
| `project.static_dir` | `config` in all six (`config/templates/snake_config.template.yml:14`; `test_case/snake_config_rapid.yml:21`; `test_case/snake_config_baseline.yml:7`; `test_case/snake_config_baseline_linux.yml:9`; `test_case/snake_config_wf2_fast.yml:23`; `tests/snake_config_fixture.yml:7`) | Implementation-layout detail. It only supplies WF1 fallback paths (`Snakefile_model_creation:115-116`), while five configs explicitly repeat those paths and WF3 ignores the value. Remove it and use canonical toolbox paths; breaking for existing configs unless accepted as a deprecated ignored key for one release. |
| `workflows.model_creation.model_build_config` | Canonical path in template, rapid, baseline, wf2-fast, and fixture; omitted only in Linux (`config/templates/snake_config.template.yml:72`; `test_case/snake_config_rapid.yml:40`; `test_case/snake_config_baseline.yml:28`; `test_case/snake_config_wf2_fast.yml:42`; `tests/snake_config_fixture.yml:30`) | Advanced engine override, not a routine basin parameter. Keep override capability but move it into an explicitly advanced/customization block. Schema-breaking move. |
| `workflows.model_creation.waterbodies_config` | Same pattern and canonical path (`config/templates/snake_config.template.yml:73`; `test_case/snake_config_rapid.yml:41`; `test_case/snake_config_baseline.yml:29`; `test_case/snake_config_wf2_fast.yml:43`; `tests/snake_config_fixture.yml:31`) | Same recommendation and cost as `model_build_config`. |
| `workflows.climate_experiment.batch_size` | Absent from all executable configs; template comments `4` (`config/templates/snake_config.template.yml:166`) | Scheduler implementation detail. Runtime already derives it from members and cores (`Snakefile_climate_experiment:983-990`). Remove from ordinary project config or put it under an advanced execution block. Breaking only for projects that explicitly set it. |
| `workflows.climate_experiment.batch_size_max` | Absent from all executable configs; template comments `8` (`config/templates/snake_config.template.yml:167`) | Execution guard/tuning knob, not a scientific project parameter. Same recommendation. |
| `save_grids` | Absent from current files | Superseded, not merely unnecessary. Delete from the declared inventory; retain the current fail-loud migration reader temporarily (`blueearth_cst/projections/gridded_outputs.py:14-30`). |

### Live but correctly user-facing despite no shipped variation

Several keys are identical across shipped configs but have a documented admissible reason to vary:

- `clim_historical` is always `era5`, but WF1 explicitly supports `era5`, `chirps`, and `chirps_global` (`Snakefile_model_creation:99-107`).
- `enabled` is always true, but the wrapper’s purpose is to allow false and skip invocation (`scripts/run_workflows.py:205-231`).
- `realizations_num` is always 2, but it directly controls ensemble size and compute cost (`Snakefile_climate_experiment:103,843-890`).
- `temp.step_num` is always 1 and both transient flags are always true, but they define grid density and ramp-versus-step semantics (`blueearth_cst/shared/snake_utils.py:1455-1481`, `blueearth_cst/experiment/prepare_weagen_config.py:36-50`).
- Both spell factors are identity vectors everywhere, but they are documented perturbation controls and reach weathergenr (`config/templates/snake_config.template.yml:159-165`, `Snakefile_climate_experiment:119-125`).
- `shared.seed`, `shared.water_year_start`, and `shared.julia_threads` are absent from executable configs, but their project-specific reproducibility, hydrological-year, and resource purposes are explicit (`config/advanced_settings.yml:48-95`).
- Spatial source and tolerance defaults are mostly unvaried, but alternative catalogs, resolutions, gauge geometry, and delineation tolerances are basin-specific and have live numerical consumers (`blueearth_cst/spatial/config.py:240-270`).

`clim_project` is `cmip6` in all six current files (`config/templates/snake_config.template.yml:94`; `test_case/snake_config_rapid.yml:52`; `test_case/snake_config_baseline.yml:34`; `test_case/snake_config_baseline_linux.yml:33`; `test_case/snake_config_wf2_fast.yml:52`; `tests/snake_config_fixture.yml:49`). Archived configs contain `cmip5` and `isimip3` (`config/templates/archive/snake_config_projections_cmip5_full.yml:9`, `config/templates/archive/snake_config_projections_isimip3.yml:9`), but those configs are explicitly unmaintained. Whether those values remain admissible is an open question, not grounds for deletion.

### Correctly toolbox-only fixed values

Several engine-native values have only one workflow-compatible setting and are already outside project config:

- `modeltype` must be `wflow_sbm` (`blueearth_cst/model/build_wflow_model.py:45-48`);
- `generate_weather.warm_var` is pinned to precipitation (`config/defaults/weathergen_config.yml:43`);
- `run_weather_generator.log_messages` must remain false under the deterministic-tree policy (`config/defaults/weathergen_config.yml:28-35`);
- `apply_climate_perturbations.diagnostic` must remain false because true changes the return type and breaks the following write (`config/defaults/weathergen_config.yml:130-136`).

These should remain T1 implementation declarations, not become T2 user knobs.

---

## Q3 · Duplication — where is one concept declared more than once?

### Exact duplicates and divergent defaults

| Concept | Locations | Runtime winner and consequence |
|---|---|---|
| Annual anchor | `DEFAULT_ANCHOR = "YE-DEC"` at `blueearth_cst/shared/metrics_definition.py:18` and `blueearth_cst/climate_analysis/climate_figures.py:120` | Call-site defaults choose whichever module was imported. Workflow call sites now pass the configured anchor explicitly (`blueearth_cst/climate_analysis/plot_climate_source.py:256`, `blueearth_cst/experiment/export_wflow_results.py:601`), so current runs are protected, but standalone/default calls can drift. Single-source from `water_year_end_anchor(DEFAULT_WATER_YEAR_START)`. Non-breaking if values remain identical. |
| Spatial resolution default | Template `0.00833` (`config/templates/snake_config.template.yml:21`) versus parser fallback `0.00833333` (`blueearth_cst/spatial/config.py:242-245`) | Explicit template value wins in copied configs; Python wins when the key is omitted. The difference changes target-grid alignment and potentially basin cells. This is a behavioral break for configs that omit the key whichever value is chosen. |
| Realization default | Template `2` (`config/templates/snake_config.template.yml:139`) versus code fallback `1` (`Snakefile_climate_experiment:103`) | Explicit template wins; omission silently halves the ensemble. Establish one canonical default or require the key. |
| Batch-size default | Template’s “commented keys show defaults” contract (`config/templates/snake_config.template.yml:7`) shows `batch_size: 4` (`config/templates/snake_config.template.yml:166`), while runtime derives it from members and cores (`Snakefile_climate_experiment:983-987`) | Runtime derivation wins when omitted. The displayed “default” is false except for configurations that happen to derive four. |
| Seed default | Template shows `#seed: auto` (`config/templates/snake_config.template.yml:52`) while omission resolves to numeric 123 (`config/advanced_settings.yml:60-79`, `blueearth_cst/shared/snake_utils.py:647-688`) | Advanced setting wins when omitted. `auto` is an example override, not the default, contradicting the template’s line-7 contract. |
| Water-year default | Template shows `#water_year_start: Oct` (`config/templates/snake_config.template.yml:66`) while omission resolves to Jan (`config/advanced_settings.yml:81-95`) | Advanced setting wins. The prose says Jan, but the generic “commented-out values are defaults” statement makes the key line misleading. |

The duplicated but currently equal defaults are also numerous: subbasin cap, gauge tolerance, river threshold, spatial source names, HydroMT source defaults, Wflow output variables, run length, baseline inclusion, spell factors, statistics, relative-change thresholds, and batch cap are declared in both the template and Python (`config/templates/snake_config.template.yml:20-33,72-76,131-167`; `blueearth_cst/spatial/config.py:21-26,231-270`; `Snakefile_model_creation:110-116`; `Snakefile_climate_experiment:128,236,983`; `blueearth_cst/projections/dry_month.py:32-38`). Equality today does not remove the drift mechanism.

### One concept under multiple names or declarations

| Concept | Locations | Runtime winner |
|---|---|---|
| Water year | T2 `shared.water_year_start`; generated/upstream `generate_weather.year_start_month`; WF2 function argument `start_month_hyd_year` (`config/templates/snake_config.template.yml:53-66`; `blueearth_cst/experiment/prepare_weagen_config.py:107-110`; `blueearth_cst/projections/derive_change_factors.py:179-185`) | T2 is authoritative; conversion to integer or the legacy function-argument spelling occurs at the seam. These translations are forced by upstream/existing function contracts. The retired T2 `workflows.climate_projections.start_month_hyd_year` is now refused (`Snakefile_climate_projections:107-116`). |
| River threshold | T2 `shared.basin.river_uparea_km2: 32`; T1 HydroMT `setup_rivers.river_upa: 32` (`config/templates/snake_config.template.yml:28`; `config/defaults/wflow_build_model.yml:8-10`) | T2 creates the P1 river mask (`blueearth_cst/spatial/hydrography.py:227`); T1 is separately forwarded to `setup_rivers` (`blueearth_cst/model/build_wflow_model.py:216-223`). Both are live and can diverge. Whether HydroMT reapplies the threshold when explicit river geometry is supplied must be tested. |
| Land-cover source/mapping | T2 `spatial_sources.lulc`; T1 `setup_lulcmaps.lulc_fn` (`config/templates/snake_config.template.yml:31`; `config/defaults/wflow_build_model.yml:16-17`) | T2 supplies the P1 raster, but T1’s name wins when selecting the mapping table (`blueearth_cst/model/build_wflow_model.py:224-231`). A custom T2 source with unchanged T1 can apply the wrong mapping. |
| LAI source | T2 `spatial_sources.lai`; T1 `setup_laimaps.lai_fn` (`config/templates/snake_config.template.yml:32`; `config/defaults/wflow_build_model.yml:19-20`) | T2 wins; T1 is discarded at `blueearth_cst/model/build_wflow_model.py:234`. |
| Soil source | T2 `spatial_sources.soil`; T1 `setup_soilmaps.soil_fn` (`config/templates/snake_config.template.yml:33`; `config/defaults/wflow_build_model.yml:22-24`) | T2 populates the spatial product, while the Wflow step separately forwards T1 to HydroMT (`blueearth_cst/model/build_wflow_model.py:239-240`). They can describe different soils. |
| PET method | Weathergen perturbation uses `hargreaves` (`config/defaults/weathergen_config.yml:108-112`); Wflow forcing reconstruction chooses `debruin` for supported sources (`blueearth_cst/experiment/downscale_climate_forcing.py:39-45,147-153`) | The downstream `setup_temp_pet_forcing` calculation supplies Wflow’s forcing, so it appears to win for model results. The earlier PET calculation is at least duplicated work and may be discarded; confirm against the generated variables before removal. |
| WF2 overlay versus WF3 forcing window | `future_horizons` versus `horizontime_climate ± run_length/2` | Neither wins; they are computed independently. The baseline comment claims an exact match while declaring `[2070,2090]` against a stated WF3 window of `2070-2086` (`test_case/snake_config_baseline.yml:48-61`). This is coordination duplicated in prose, not a single runtime value. |
| Toolbox root versus explicit template paths | `project.static_dir` plus `model_build_config`/`waterbodies_config` (`config/templates/snake_config.template.yml:14,72-73`) | Explicit workflow paths win; `static_dir` only supplies omitted-value fallbacks (`Snakefile_model_creation:115-116`). |

### Derivable values

The code already derives several lower-tier values correctly and should retain that pattern:

- `experiment_name` may derive from `project_dir`, while remaining overridable for multiple experiments (`Snakefile_climate_experiment:59-81`).
- `n_years` derives from `horizontime_climate` and `run_length` (`blueearth_cst/experiment/prepare_weagen_config.py:26-33`).
- `year_start_month` derives from the shared month name (`blueearth_cst/experiment/prepare_weagen_config.py:107-110`).
- `ST_NUM` derives from both step counts (`blueearth_cst/shared/snake_utils.py:1479-1481`).
- Index widths derive from realization/stress counts (`blueearth_cst/shared/snake_utils.py:1518-1548`).

The spatial source/threshold duplicates should follow this model: derive engine arguments from the resolved spatial contract or its generated metadata rather than asking users to keep T1 and T2 synchronized.

---

## Q4 · Naming and documentation

### Formal convention

All current BlueEarth-owned project keys use snake_case, and all checked-in BlueEarth booleans use lowercase YAML booleans, satisfying `naming.md` §2 (`dev/reference/naming.md:42-53`).

The formal path-suffix convention is broken by file-path keys that do not end in `_path`: `data_sources`, `data_sources_climate`, `gauge_points`, `model_build_config`, `waterbodies_config`, and `observations_timeseries`. The rule reserves `_dir` for directory paths and `_path` for file paths (`dev/reference/naming.md:118-135`). These are grandfathered contracts, so renaming requires a migration note rather than a cosmetic sweep (`dev/reference/naming.md:7-11,205-224`).

`project_dir` and `static_dir` conform structurally, although `static_dir` is semantically misleading: a hydrologist can reasonably read it as the project’s static geospatial/model data rather than the repository configuration root.

### Where the convention is silent

The guide says nothing prescriptive about:

- units in config-key names;
- count word order (`realizations_num` versus `n_realizations` versus `RLZ_NUM`);
- qualifier position (`clim_historical` versus `historical_window`);
- whether a leaf may rely on its parent for meaning;
- predicate phrasing for booleans;
- English-word quality.

That silence explains why every current key can pass snake_case while `horizontime_climate`, `run_historical`, and `step_num` remain misleading. The guide’s stable vocabulary explicitly protects `st`, `rlz`, `wflow`, `gcm`, and `cmip6`, but not ad-hoc contractions such as `clim` or `outvars` (`dev/reference/naming.md:366-369`).

### Name-quality and reader test

| Keys | Assessment and plausible wrong reading |
|---|---|
| `project_dir` | Good: directory type and purpose are clear. |
| `static_dir` | Wrong, not merely unfashionable. Likely reading: static basin/model data directory. Actual: toolbox configuration root used for two fallback files (`Snakefile_model_creation:115-116`). |
| `data_sources`, `data_sources_climate` | Missing path suffix; qualifier order is inconsistent. Likely reading: loaded source objects or lists. Actual: catalog file path(s) (`Snakefile_climate_projections:38-50`). Prefer `data_catalog_path` and `climate_catalog_path`; breaking migration. |
| `basin.region` | Parent helps, but likely reading is a polygon/region object. Actual value may be a quoted HydroMT basin/subbasin specification (`config/templates/snake_config.template.yml:20`). |
| `basin.resolution` | Likely reading: model resolution in unspecified units. Actual: target hydrography/grid resolution in degrees; unit appears only in the comment (`config/templates/snake_config.template.yml:21`, `blueearth_cst/spatial/hydrography.py:26-43`). Rename to `grid_resolution_deg` if breaking migration is accepted. |
| `hydrography` | Likely reading: hydrography dataset/object. Actual: catalog entry name. `hydrography_source` would be clearer. |
| `basin_index` | Wrong without context: likely integer/index array; actual catalog source name (`blueearth_cst/spatial/config.py:231-238`). Prefer `basin_index_source`. |
| `gauge_points` | Wrong value class: likely point collection; actual CSV path, currently often named `output_locations.csv` (`config/templates/snake_config.template.yml:24`; `test_case/snake_config_baseline.yml:15`). Prefer `gauge_points_path`; the filename should then be aligned separately. |
| `automatic_subbasins.max_per_basin` | Good when fully qualified; meaningless as a bare `max_per_basin`. Parent-dependent naming is legitimate here because the full path reads naturally. Error/log messages should always use the full path, as the parser does (`blueearth_cst/spatial/config.py:253-256`). |
| `gauge_snap_tolerance_m`, `river_uparea_km2` | Good: quantity and units are in the name. `uparea` is established hydrological shorthand, though `upstream_area` would be clearer outside HydroMT vocabulary. |
| `spatial_sources.{rivers,lulc,lai,soil}` | Clear with the parent. `LULC` and `LAI` are established domain abbreviations for the intended audience. |
| `historical_window.{starttime,endtime}` | Clear path context; ISO type is in comments. `start_time`/`end_time` would be more idiomatic English, but current names mirror common HydroMT/Wflow spelling and are not wrong. |
| `clim_historical` | Wrong word order and unnecessary contraction. Likely reading: historical climatology object. Actual: catalog source identifier. Prefer `historical_climate_source`; breaking migration. |
| `seed`, `water_year_start`, `julia_threads` | Clear. Water-year type/range is documented; Julia thread scope is explained (`config/templates/snake_config.template.yml:41-67`). |
| `model_build_config`, `waterbodies_config`, `observations_timeseries` | Wrong value class: each looks like a loaded mapping/series but is a file path. Add `_path` if retained. |
| `wflow_outvars` | `wflow` is established; `outvars` is merely short. Likely reading: Wflow/CSDMS variable IDs. Actual: BlueEarth display labels mapped later (`Snakefile_model_creation:110,584`). Prefer `wflow_output_variables`. |
| `clim_project` | Likely reading: climate project directory or project name. Actual: catalog/dataset-family prefix such as `cmip6` (`Snakefile_climate_projections:399-405`). Prefer `climate_dataset_family` or `projection_source`. |
| `models`, `scenarios`, `members`, `variables`, `future_horizons`, `stats` | Clear under `climate_projections`. `variables` is especially well specified because each entry declares source, quantity class, units, and change semantics (`config/templates/snake_config.template.yml:123-130`). |
| `historical_year_range` | Clear and carries the unit in “year.” |
| `relative_change.min_reference` | Parent supplies semantics, but unit is implicit in each variable’s `units`, not stated beside the threshold (`config/templates/snake_config.template.yml:123-134`). Error/report text must include resolved units. |
| `relative_change.max_flagged_months` | Clear and unit-bearing. |
| `experiment_name` | Clear. |
| `realizations_num` | Meaning is guessable, but word order disagrees with upstream `n_realizations` and local `RLZ_NUM`. Prefer `realization_count`; rename is clarity-only, so defer unless bundled with a schema migration. |
| `horizontime_climate` | Wrong. It is not English and gives neither role nor unit. Likely readings include horizon duration, time step, or climate-window length. Actual: middle/target year of the Wflow forcing window (`Snakefile_climate_experiment:235`, `Snakefile_climate_experiment:808-809`). Prefer `target_year` or `forcing_mid_year`; migration is justified. |
| `run_length` | Likely reading: number of jobs, timesteps, or days. Actual: Wflow simulation duration in years (`config/templates/snake_config.template.yml:141`). Prefer `simulation_years`. |
| `run_historical` | Wrong and consequential. Likely reading: run the historical period/model. Actual: include the unperturbed `st_0` baseline in WF3 (`Snakefile_climate_experiment:128-129`). Prefer `include_baseline`; breaking migration justified because the wrong reading can omit metrics. |
| `stress_test.*.step_num` | Likely reading: number of grid points. Actual: number of intervals; runtime adds one to include endpoints (`blueearth_cst/shared/snake_utils.py:1455-1481`). Prefer `interval_count`, or document “N intervals, N+1 values” directly beside each key. |
| `stress_test.*.transient_change` | Adequate with the parent; comments correctly distinguish ramp versus step (`blueearth_cst/experiment/prepare_weagen_config.py:36-50`). |
| `stress_test.*.{mean,variance}.{min,max}` | Clear only as full paths. Units/types are documented in the template: temperature offset in °C and precipitation/variance multipliers (`config/templates/snake_config.template.yml:143-158`). |
| `dry_spell_factor`, `wet_spell_factor` | Clear with comments and monthly-vector explanation (`config/templates/snake_config.template.yml:159-165`). |
| `batch_size`, `batch_size_max` | Likely reading: number of jobs per batch. Actual: number of `(realization, stress)` members grouped into a dynamically created Snakemake job (`Snakefile_climate_experiment:949-990`). If retained, use `members_per_batch` and `max_members_per_batch`. |

### Units and types

Units are in the key for `gauge_snap_tolerance_m`, `river_uparea_km2`, `historical_year_range`, and `max_flagged_months`. They are only in comments for `resolution`, `horizontime_climate`, `run_length`, and monthly perturbation arrays (`config/templates/snake_config.template.yml:21,127-141,147-158`). `variables.*.units` is explicit data, which is stronger than a comment (`config/templates/snake_config.template.yml:123-126`).

The weakest cases are `horizontime_climate` and `run_length`: neither name reveals years. `step_num` also hides that it counts intervals, not resulting values.

### Documentation completeness

The earlier finding that `basin.hydrography` and `basin.basin_index` are absent from the template is now false: both are documented at `config/templates/snake_config.template.yml:22-23`. `stats` and relative-change defaults are also visible at `config/templates/snake_config.template.yml:131-134`.

Every one of the 60 active project leaves now appears in the template, including commented optional keys. The exceptions are not active keys:

- retired `save_grids`;
- unsupported `temp.variance.{min,max}`.

Default and optionality documentation remains defective:

1. The template says every commented-out key shows its default (`config/templates/snake_config.template.yml:7`), but `seed: auto`, `water_year_start: Oct`, `simulation_window`, and `batch_size: 4` are examples or overrides, not runtime defaults (`config/templates/snake_config.template.yml:52,66,88-90,166`; `config/advanced_settings.yml:79,95`; `Snakefile_climate_experiment:983-987`).
2. Many optional keys are active rather than commented—`resolution`, `gauge_points`, spatial defaults, model template paths, `wflow_outvars`, observations, realization count, run length, baseline, and spell factors—so active versus commented cannot consistently mean required versus optional (`config/templates/snake_config.template.yml:20-33,71-77,137-165`).
3. The baseline’s WF2 comment claims its horizon exactly matches WF3 while the same lines state `2070-2086` and configure `[2070,2090]` (`test_case/snake_config_baseline.yml:48-61`).
4. `generate_weather.R` says the retired `evaluate.model` “now governs” plots, but the actual current control is `save_plots` passed through `gw`; the historical comment has the wrong antecedent (`blueearth_cst/weathergen/generate_weather.R:102-125`).
5. `batch_size`’s effective default is invisible without reading the Snakefile (`Snakefile_climate_experiment:983-987`).

Any rename above is a checked-in project-config contract change and therefore requires the migration record mandated by `dev/reference/naming.md:205-224`. Renames should be limited to names that invite a materially wrong interpretation—`static_dir`, `gauge_points`, `basin_index`, `horizontime_climate`, `run_historical`, and arguably `step_num`—not every grandfathered name that could be stylistically improved.

---

## Q5 · Organisation — is the hierarchy right for a user?

### Are three tiers right?

The T1/T2 ownership distinction is sound: toolbox constraints/runtime/engine templates differ from per-project scientific choices (`config/advanced_settings.yml:9-27`). The third tier is not soundly described as “record, never input,” because generated files include:

- provenance records;
- immutable guard records that are read back;
- derived engine configurations;
- mutable model runtime state.

Keep the three ownership tiers, but add an orthogonal role classification for generated artifacts: `provenance record`, `guard/reference`, `derived runtime config`, and `engine state`. Calling all of them records directly contradicts the read-backs at `Snakefile_climate_experiment:610-662,797-875` and `blueearth_cst/experiment/write_experiment_config.py:113-139`.

### Is `project` / `shared` / `workflows.*` the right axis?

`project` and `workflows.*` are coherent. `shared` is partly a leftovers bin:

- `shared.basin` mixes basin identity (`region`, `resolution`), dataset bindings (`hydrography`, `basin_index`, `spatial_sources.*`), observation geometry (`gauge_points`), and delineation controls (`automatic_subbasins`, snap tolerance, river threshold) (`config/templates/snake_config.template.yml:18-33`).
- The remainder mixes climate-record definition, stochastic reproducibility, hydrological aggregation, and execution resources (`config/templates/snake_config.template.yml:34-67`).

Recommended future grouping:

```text
project:
  project_dir
  catalogs: ...

basin:
  definition: region, grid_resolution_deg
  sources: hydrography, basin_index, rivers, lulc, lai, soil
  delineation: automatic_subbasins, gauge_snap_tolerance_m, river_uparea_km2
  observations: gauge_points_path

climate_record:
  source
  window
  water_year_start

execution:
  seed
  julia_threads
  advanced batching overrides

workflows:
  model_creation
  climate_projections
  climate_experiment
```

This is a breaking schema redesign, not a cleanup. It requires a design document and migration plan before implementation.

### Is nesting depth justified?

`shared.basin.automatic_subbasins.max_per_basin` is four levels but semantically clear as a full path. The depth is justified if error messages always print the full path, as the parser does (`blueearth_cst/spatial/config.py:253-256`). Flattening it to `max_subbasins_per_basin` would reduce depth but mix delineation controls back into the same undifferentiated basin block.

The better fix is kind-based grouping, not blanket flattening. A three-level `basin.delineation.max_subbasins_per_basin` is clearer than either current four-level nesting or a flat `shared` namespace.

### New-basin user-oriented test

A user must set or review these values before a defensible new-basin run:

1. Project placement and catalogs: `project_dir`, `data_sources`, `data_sources_climate`; currently also `static_dir`.
2. Basin definition and sources: `region`, `resolution`, `hydrography`, `basin_index`, `spatial_sources.{rivers,lulc,lai,soil}`.
3. Basin delineation and observations: `gauge_points`, `automatic_subbasins.max_per_basin`, `gauge_snap_tolerance_m`, `river_uparea_km2`, `observations_timeseries`.
4. Historical climate: `historical_window.{starttime,endtime}`, `clim_historical`, `water_year_start`.
5. Model run: `simulation_window`, `wflow_outvars`; engine-template overrides only if intentionally customizing the model.
6. Projection overlay: `clim_project`, `models`, `scenarios`, `members`, `variables`, `historical_year_range`, `future_horizons`, `stats`, and relative-change thresholds.
7. Stress experiment: `experiment_name`, `realizations_num`, `horizontime_climate`, `run_length`, baseline inclusion, both stress axes and their monthly bounds/transient flags, and spell factors.
8. Execution: workflow enable flags, `julia_threads`; batching only for advanced troubleshooting.

The current template orders most of these reasonably by workflow, but basin setup is split between `project`, `shared.basin`, `shared.historical_window`, `workflows.model_creation`, and execution keys embedded in `shared` (`config/templates/snake_config.template.yml:12-90`). The “everything needed to establish a new basin and historical model” set is therefore not contiguous.

A contiguous replacement should put project location/catalogs, basin definition/sources/delineation, observations, historical climate, water year, and model simulation window together before projection and experiment choices. Workflow enable flags can remain with workflows; advanced execution knobs should not interrupt scientific configuration.

### Where should defaults be visible?

Recommendation: every optional project key’s effective default should be visible beside that key in the user scaffold, but declared once in a machine-readable source consumed by runtime validation. The template should be generated or checked mechanically against that source.

Arguments for:

- prevents the current `0.00833`/`0.00833333`, `2`/`1`, and batch-default divergences;
- gives users one place to understand an omitted key;
- enables closed-schema unknown-key rejection, which would catch `temp.variance.*`.

Argument against:

- introducing a canonical schema and template synchronization adds machinery;
- dynamic defaults such as batching and experiment naming cannot be represented as a simple scalar;
- dynamic mappings such as climate variables and future horizons require schema patterns rather than fixed leaf enumeration;
- existing configs need a migration/version policy.

`advanced_settings.yml` should remain the home of genuine toolbox-wide constraints, runtime pins, and universal fallbacks. Moving every project default there would centralize validation but force users to cross-reference two files, worsening the new-basin test (`config/advanced_settings.yml:9-19`). The exact schema/template mechanism is an architectural choice and should go through `design-document`; this review establishes the requirements, not the implementation.

### Mechanical Q1 feasibility

A complete static “declared keys ⊆ read keys” test is not feasible as a simple grep:

- Snakemake passes whole mappings through `params:` (`Snakefile_model_creation:501-504`);
- scripts sometimes reread the original YAML by path (`blueearth_cst/experiment/prepare_cst_parameters.py:67-72`);
- generated engine configs are consumed in R or by external engines;
- some reads are translations rather than literal key matches;
- `params:` declaration alone does not prove a script reads the value—the original water-year and `static_dir` defects demonstrate that.

The cheapest useful partial check is:

1. Maintain a canonical declared-key set.
2. Search exact leaf names across runtime sources, excluding configs, docs, tests, provenance serialization, and generic digest code.
3. Fail on zero runtime matches.
4. Separately compare each named Snakemake `params:` field with `snakemake.params`/`sm.params` accesses in its Python/R/Julia consumer.
5. Keep explicit engine-surface validators such as WG-3 (`blueearth_cst/shared/interchange_contracts.py:264-390`).
6. Add targeted dynamic tests for forwarded dictionaries where static matching cannot resolve the consumer.

That partial check would have caught `temp.variance.*`, `static_dir`’s WF3 path, and `lai_fn`. It would not by itself prove upstream `run_weather_generator` forwards every key, nor detect a value that reaches arithmetic whose result is later overwritten, such as the unresolved PET duplication.

---

## Ranked recommendations

| Rank | Finding | Consequence | Proposed action | Cost | Breaking? |
|---:|---|---|---|---|---|
| 1 | T2 spatial sources/thresholds are duplicated in the Wflow engine template | Divergent river masks, land-cover mappings, or soil parameters can produce wrong model numbers; `lai_fn` already reaches nothing | Derive Wflow step arguments/mappings from the resolved spatial contract or generated P1 metadata; remove independent duplicates after design review | Medium/high; engine-seam tests and baseline rerun | Potentially behavioral; T2 schema need not break |
| 2 | `resolution` has two unequal defaults | Omitting one key changes grid geometry and potentially basin cells/results | Choose one exact canonical default or require the key; encode it once | Low code cost, high validation cost | Yes for configs relying on omission |
| 3 | `run_historical` names the wrong behavior | A user can omit `st_0` and consequently omit baseline-derived metrics while believing only a historical run was disabled | Rename to `include_baseline`, fail loudly on the old key, and publish a migration note | Medium | Yes |
| 4 | Open project schema accepts appendix-only `temp.variance.*` silently | A user can configure temperature variance and receive unchanged results | Introduce closed project-schema validation; explicitly reject unsupported temperature variance | Medium/high; design document required | Yes for configs carrying unknown keys |
| 5 | PET is computed under two methods, with downstream Wflow forcing apparently recomputing it | Risk of inconsistent intermediate versus modeled PET; at minimum unnecessary computation | Establish which PET field is consumed, then remove or unify the unused calculation | Medium; requires data/run inspection | Possibly |
| 6 | Generated tier is incorrectly documented as “record, never input” | Maintainers may prune or treat guard/runtime files as inert records, breaking reproducibility or execution | Retain ownership tiers but document generated artifact roles and read-backs | Low documentation cost | No |
| 7 | Template and code duplicate defaults; several already diverge | Silent behavior differences between copied and minimal configs | Adopt one machine-readable default/schema source and mechanically check/render the template | High initial design; low recurring cost | Migration likely |
| 8 | `setup_laimaps.lai_fn` is a forwarded-then-dropped engine parameter | Misleads maintainers into believing changing it changes the model | Remove it from the template or explicitly replace it during generated config assembly with documented provenance | Low | T1 template behavior change, not T2 schema |
| 9 | `static_dir` is redundant and partially unused | Confusing implementation knob; WF3 requires a value it ignores | Remove it, use canonical toolbox paths, and retain a temporary migration reader if needed | Low/medium | Yes |
| 10 | `save_grids` remains in the inventory after retirement | Inflates and confuses the supported surface | Remove it from inventories; retain temporary fail-loud compatibility only | Low | No additional break |
| 11 | New-basin inputs are scattered and `shared` mixes kinds | Users can miss a basin-specific scientific choice and must scan unrelated execution settings | Redesign around basin definition, sources, delineation, climate record, execution, and workflows | High; design document and migration | Yes |
| 12 | Misleading names: `horizontime_climate`, `gauge_points`, `basin_index`, `step_num`, path keys | Wrong type/unit/behavior guesses; some can lead to wrong experiment design | Rename only materially wrong names, with explicit old→new migration mapping | Medium/high due broad config references | Yes |
| 13 | Required/optional/default documentation is inconsistent | Users cannot reliably infer omission behavior from the template | Mark each key explicitly as required, optional-static-default, optional-derived-default, or advanced | Low after canonical schema decision | No |
| 14 | No mechanical reach gate | Future inert parameters remain dependent on manual reading | Add declared-key zero-read scan, params-consumer matching, engine contract validators, and targeted dynamic checks | Medium | No |

## Open questions, assumptions, and residual risk

1. Does HydroMT-Wflow still apply `setup_rivers.river_upa` when both explicit hydrography and river geometry are supplied? Source inspection shows it is forwarded (`blueearth_cst/model/build_wflow_model.py:216-223`), but only an upstream-code inspection or controlled run will establish whether it changes the resulting parameters.
2. Is the Hargreaves PET written by `apply_climate_perturbations` consumed anywhere before `setup_temp_pet_forcing` recomputes PET with De Bruin? The downstream call strongly suggests De Bruin wins (`blueearth_cst/experiment/downscale_climate_forcing.py:147-153`), but generated-variable inspection is needed before removal.
3. Are `cmip5` and `isimip3` still supported values of `clim_project`, or only historical archive labels? The current resolver is generic (`Snakefile_climate_projections:399-405`), while the only maintained catalog/config is CMIP6.
4. Do production users intentionally override `model_build_config` or `waterbodies_config` per project? Repository examples do not; user/project evidence would decide whether to move these to an advanced block or remove the override surface.
5. Which canonical-default mechanism should be adopted—closed schema, generated template, or required explicit keys? This affects architecture, backward compatibility, and dynamic defaults and therefore requires a design document.
6. The reach review proves paths through repository code but cannot prove the internals of external HydroMT, Wflow, or weathergenr calls without executing those engines. The WG-3 contract asserts the intended forwarding surface (`blueearth_cst/shared/interchange_contracts.py:264-390`).
7. No numerical workflow or test suite was run, as required. No repository file was written. `git status` could not be executed because the sandbox could not start PowerShell; all successful operations were read-only filesystem scans. There is no Results delta.