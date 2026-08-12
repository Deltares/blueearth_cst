# Configuration parameter review — findings

Executed 2026-08-12 against `lane/devmeta` @ `8d9c548` (clean tree). Review
only; no runtime file was touched. Gate resolutions applied as dispatched:
Gate 1 — new inert parameters recorded, review continued; Gate 2 — breaking
schema changes may be *proposed*; Gate 3 — stop at this report. The brief's
`Progress` section was deliberately left unedited: the dispatch constrains
`git status` to this file alone, and that constraint outranks the brief's
"update the Progress section".

Method: every claim below carries `file:line` from this checkout. Every
inertness claim names the search that would refute it and reports its result
(§1.4). Coverage is complete per §1.2–§1.3: all project-config leaf keys as
re-measured (not the appendix's 55 — see §0) and all `DEFAULT_*` constants.

---

## 0. Re-measurement of the appendix inventory (corrections — findings in themselves)

The brief's appendix is a same-day snapshot and is already stale in six ways.
The draft (`dev/working/parameter-placement.md`) shares five of them, since the
two appendices are the same measurement — which is itself the duplication its
own note predicts ("If one is corrected, correct both or delete this one").

| # | Appendix claim | Measured state | Evidence |
|---|---|---|---|
| C1 | Project config = 55 leaves | **60 leaves** (counting rule: `variables` = 1 key; `simulation_window` = 2; `stress_test` fully expanded; legacy `output_locations` and the never-read `temp.variance.*` excluded). New since the snapshot: `shared.seed`, `shared.water_year_start`, `shared.julia_threads` (template `config/templates/snake_config.template.yml:52,66,67`; reads at `Snakefile_climate_experiment:96,101`, `Snakefile_model_creation:46,94`, `Snakefile_climate_projections:117`), `climate_projections.relative_change.{min_reference,max_flagged_months}` (`Snakefile_climate_projections:189–195`), `climate_experiment.{batch_size,batch_size_max}` (`Snakefile_climate_experiment:983–988`) | template + Snakefile lines cited |
| C2 | `save_grids` is a live key | **Refused at parse time**, both spellings (`blueearth_cst/projections/gridded_outputs.py:7`; `Snakefile_climate_projections:163–168`; `tests/test_gridded_outputs_removed.py`) | grep run, see §1.4-F5 |
| C3 | 14 `DEFAULT_*` constants | **15 definitions, 14 unique names**: `DEFAULT_RIVER_UPAREA_KM2 = 32.0` (`blueearth_cst/spatial/config.py:26`) is missing from brief *and* draft. It backs the config key `shared.basin.river_uparea_km2` (`config.py:262–265`). Likely cause: an inventory grep whose class `[A-Z_]` excludes digits — `KM2` breaks the match. A lesson for Q7: the mechanical check must not repeat it | `config.py:26` |
| C4 | Draft: `DEFAULT_MIN_REFERENCE` / `DEFAULT_MAX_FLAGGED_MONTHS` "no config surface, correctly constants" | **Wrong in this tree**: both back `relative_change.min_reference` / `.max_flagged_months` (`Snakefile_climate_projections:189–195`; `blueearth_cst/projections/dry_month.py:49–97`; template `:132–134`). The config-backed constant set is **9**, not 6 (M3 undercounts by 3: these two plus C3's) | cited lines |
| C5 | weathergen `generate_weather` = "16 set + 6 injected" | **12 set + 8 injected** (`config/defaults/weathergen_config.yml:39–101`): the spell factors moved to the project config the same day; `seed` and `year_start_month` are injected (`blueearth_cst/experiment/prepare_weagen_config.py:96–121`) | file read in full |
| C6 | Q4 premise: `basin.hydrography` / `basin.basin_index` "not in the template at all" | Both are now documented as commented optional keys with their defaults (`snake_config.template.yml:22–23`) | template read |

Counting note on "17 required": the appendix reaches 17 only by counting
`stress_test` as one starred key. Fully expanded, required-at-parse keys number
26 — and two more are *de facto* required at rule time despite optional
parse-time reads (§1, F7).

Shipped-config set in this tree (Q2's evidence base): the four
`test_case/snake_config_{baseline,baseline_linux,rapid,wf2_fast}.yml` plus
`tests/snake_config_fixture.yml` and the template. Note the file renames since
the AGENTS.md text quoted in the dispatch context (`model_test → baseline`,
`dev_fast → wf2_fast`).

---

## Q1 · Reach — classification of every declared parameter

### 1.1 Project-config leaf keys (60 + 2 ignored + 1 legacy)

Classes: **(a)** never read · **(b)** read, unused · **(c)** forwarded,
dropped · **(d)** live. Where a key is live in one workflow and dead in
another, the row says so — the per-workflow split is where three of the four
new findings live.

**`project` (4)**

| Key | Class | Chain / stop point |
|---|---|---|
| `project_dir`* | (d) | `Snakefile_model_creation:50`, `Snakefile_climate_projections:34`, `Snakefile_climate_experiment:37` → every output path |
| `static_dir`* | **(b) in WF3; (d)-narrow in WF1** | WF3 reads it required (`Snakefile_climate_experiment:41`) and never uses it — chain stops at the assignment. WF1 uses it only inside the *fallback defaults* for two keys every shipped config sets explicitly (`Snakefile_model_creation:115–116`). Confirms draft M1. Falsifier §1.4-F1 |
| `data_sources`* | (d) | WF1:55 (catalog input to nearly every rule); WF2:70 (`STORE_DATA_SOURCES` → region + spatial-units rules); WF3:42 |
| `data_sources_climate`* | (d) in WF2 only | `Snakefile_climate_projections:38` → catalog parse (`:389`), resolution (`:399`), fetch/reduce params. Unread by WF1/WF3 (correct: one config drives three Snakefiles) |

**`shared` (18)**

| Key | Class | Chain |
|---|---|---|
| `basin.region`* | (d) | WF1:60, WF2:74, WF3:167 → region/store/spatial params (`snake_utils.py:1089,1417`) |
| `basin.resolution` | (d) | `spatial/config.py:242–245` → spatial-units params (`snake_utils.py:1227`) → `delineate_spatial_units.py:112`. Code default **0.00833333** ≠ template value 0.00833 — see Q3-D5 |
| `basin.hydrography` | (d) | WF1 via `spatial_cfg` (:67), WF2:75, WF3:168 → region/store/spatial params; cross-checked against the build template at rule 1.06 (`Snakefile_model_creation:63–66`) |
| `basin.basin_index` | (d) | same chain (WF1:68, WF2:76, WF3:169) |
| `basin.gauge_points` | (d) | `spatial/config.py:99–116` → spatial-units *input* (`snake_utils.py:1206–1207`); WF1 rule 1.01/1.13/1.15 inputs (`Snakefile_model_creation:144–151`) |
| `basin.automatic_subbasins.max_per_basin` | (d) | `config.py:253–257` → `delineate_spatial_units.py:58,82` |
| `basin.gauge_snap_tolerance_m` | (d) | `config.py:258–261` → `delineate_spatial_units.py:57,81` |
| `basin.river_uparea_km2` | (d) | `config.py:262–265` → `delineate_spatial_units.py:54,78` |
| `basin.spatial_sources.rivers` | (d) | `config.py:267` → spatial-units param `rivers_source` (`snake_utils.py:1228`) |
| `basin.spatial_sources.{lulc,lai,soil}` | (d) | rule 1.06 params `basin_config` (`Snakefile_model_creation:502`) → `prepare_spatial_maps.py:54` re-parses via `parse_spatial_config` |
| `historical_window.starttime`* / `.endtime`* | (d) | WF1:69/78/87; store key (`snake_utils.py:1366–1372`); WF2:77/135; WF3:175 |
| `clim_historical`* | **(b) in WF2; (d) in WF1/WF3** | WF1:88/103 (store params, eobs refusal, plot params); WF3:159. **WF2 reads it required at `Snakefile_climate_projections:78` and never uses it** — the only occurrence of `clim_source` in that file is the assignment. Vestigial since ADR 0003 removed WF2's climate store. **New finding (Gate 1: recorded, continued).** Falsifier §1.4-F2 |
| `seed` | (d) (WF3 only, by design) | WF3:96 (`resolve_seed`, `snake_utils.py:678–695`) → rule 3.10 params → injected into weathergen config (`prepare_weagen_config.py:106`) |
| `water_year_start` | (d), all three | WF1:46 → rules 1.05/1.13 params (:1029,1096); WF2:117 → rule 2.06 param → `derive_change_factors.py:185`; WF3:101 → rule 3.10 (`year_start_month` seam, `prepare_weagen_config.py:110`) + rule 3.16 → `export_wflow_results.py:601` |
| `julia_threads` | **(d) in WF1; inert for WF3** | WF1:94 → `julia_prefix` (:97) → rule 1.14 shell (:754). **WF3's rule 3.15 hardcodes `--threads 4`** (`Snakefile_climate_experiment:1015`); the key appears nowhere in WF2/WF3 or `blueearth_cst/experiment/`. A user tuning `shared.julia_threads` for a large basin changes WF1's single historical run and *not* the `RLZ_NUM × ST_NUM` stress-test runs where nearly all wflow compute is. `advanced_settings.yml:48–58` describes the key as governing "Wflow.jl `--threads`" with no WF1-only caveat. **New finding (Gate 1: recorded, continued).** Falsifier §1.4-F3 |

**`workflows.model_creation` (7 + 1 legacy)**

| Key | Class | Chain |
|---|---|---|
| `enabled` | (d), wrapper-only | Read by `scripts/run_workflows.py:113–160,209,226` (hard error if missing/non-bool); read by **no Snakefile** (falsifier §1.4-F4) — documented at `run_workflows.py:10` |
| `model_build_config` | (d) | WF1:115 → rule 1.01 input (:423), rule 1.07 input (:514) |
| `waterbodies_config` | (d) | WF1:116 → rule 1.01 input (:425), rule 1.08 params (:564) |
| `wflow_outvars` | (d) — **but two defaults; see F6** | WF1:110 → rule 1.09 params (`setup_gauges_and_outputs.py:155`), `WFLOW_TABLE_PATHS` (:829), `_basavg_pngs` (:803); WF3:477–479 → `INDICATOR_TABLES` |
| `observations_timeseries` | (d) | WF1:118 → rule 1.01/1.15 inputs (:147–151) |
| `simulation_window.starttime/endtime` | (d) | `resolve_simulation_window` (`snake_utils.py:854–931`, containment-validated) → rule 1.10 params (:683–684) |
| *(legacy)* `output_locations` | refused alone; accepted only aliasing `gauge_points` | `spatial/config.py:100–133` |

**`workflows.climate_projections` (11)**

| Key | Class | Chain |
|---|---|---|
| `enabled` | (d), wrapper-only | as above |
| `clim_project`* | (d) | WF2:80 → output root (:202), series keys (:476,495), catalog entry keys (:544) — see Q2 (single admissible value today) |
| `models`* / `scenarios`* / `members`* | (d) | WF2:81–83 → `resolution.resolve` (:399); scenarios also rule 2.07 params (:982) |
| `variables`* (incl. `source`/`canonical`/`units`/`change` per variable) | (d), all four subkeys | WF2:91 → `variable_spec.py:87–108` (all four required, validated); `canonical` drives annual aggregation (`variable_spec.py:137–148`), `units` → `variable_units` params (WF2:786,834), `change` → stage B, `source` → catalog read |
| `historical_year_range`* | (d) | WF2:118 → clipped (:134) → rule 2.06 param `time_horizon_hist` (:902) |
| `future_horizons`* | (d) | WF2:119 → rules 2.06/2.07 params (:900,983) |
| `stats` | (d) | WF2:179 → 2.06 param (:903) → `derive_change_factors.py:411`; `None` → `DEFAULT_STATS` (`get_change_climate_proj.py:161,270`) |
| `relative_change.min_reference` | (d) | WF2:189–191 → `dry_month.resolve_thresholds` → 2.06 param (:906) → `derive_change_factors.py:414,641` |
| `relative_change.max_flagged_months` | (d) | WF2:193–195 → 2.06 param (:921) → `derive_change_factors.py:627,642` |

Refused legacies (correctly loud): `start_month_hyd_year`
(`Snakefile_climate_projections:107–116`), `save_grids`/`save_gridded`
(:163–168). **Silently ignored legacy:** `ensemble.min_sources` — retired by
design D6, mentioned only in a comment (:419–421); a config still carrying it
gets no error and no effect (grep: the string exists nowhere outside that
comment). Same class as the four known inert parameters, at legacy-key scale.

**`workflows.climate_experiment` (20 + 2 never-read)**

| Key | Class | Chain |
|---|---|---|
| `enabled` | (d), wrapper-only | as above |
| `experiment_name` | (d) | WF3:59–86 (`resolve_default_experiment_name` / `validate_experiment_name`) |
| `realizations_num` | (d) — **but see F7** | WF3:103 (`get_config(..., 1)`) → `RLZ_NUM` everywhere; **re-read from the config file at rule 3.10 with no default** (`prepare_weagen_config.py:101`, `experiment_cfg["realizations_num"]` → `KeyError`). A config omitting the key parses fine (RLZ_NUM=1) and then crashes rule 3.10 with a bare `KeyError` — the parse-time default is unreachable in a real run. Effectively required, misdocumented as defaulted |
| `horizontime_climate`* | (d) | WF3:235 → rule 3.10 `middle_year` (:808), rule 3.14 params (:932) |
| `run_length` | (d) | WF3:236 (default 20) → 3.10 `sim_years` (:809), 3.14 params (:933) |
| `run_historical` | (d) | WF3:128 → `ST_START` (:129) → member enumeration (:953–954), indicator inputs (:1021) |
| `batch_size` / `batch_size_max` | (d) | WF3:983–988 (validated positive; dynamic default `min(batch_size_max, ceil(K/cores))`) |
| `stress_test.temp.step_num`* / `.precip.step_num`* | (d) | WF3:111 (`stress_test_grid`, strict — `snake_utils.py:1448–1481`) + `prepare_cst_parameters.py:90` |
| `stress_test.{temp,precip}.transient_change`* | (d) | `prepare_weagen_config.py:36–51,126–129` (refuses missing) → weathergen config → `impose_climate_change.R` |
| `stress_test.temp.mean.{min,max}`* | (d) | `prepare_cst_parameters.py:93–94` (raw `KeyError` if absent — required in effect, unstarred nowhere) |
| `stress_test.precip.mean.{min,max}`* | (d) | `prepare_cst_parameters.py:97–98` |
| `stress_test.precip.variance.{min,max}`* | (d) | `prepare_cst_parameters.py:99–100` → `impose_climate_change.R:70` |
| **`stress_test.temp.variance.{min,max}`** | **(a) never read** | The appendix's own `{temp,precip}.{…variance.{min,max}}` braces declare it; nothing reads it. `prepare_cst_parameters.py` reads only precip variance (:99–100); its axis guard checks only *top-level* unknown keys (:77–86), so a `variance:` block under `temp:` passes silently; `impose_climate_change.R` consumes `precip_variance` only (:70). A user who writes it believes they perturbed temperature variance; results are computed without it. **New finding (Gate 1: recorded, continued).** Falsifier §1.4-F8 |
| `stress_test.dry_spell_factor` / `.wet_spell_factor` | (d) | WF3:119–126 (`validate_spell_factor`) → 3.10 params (:812–813) → injected (`prepare_weagen_config.py:114–115`) → `generate_weather.R` (pinned by `interchange_contracts.py:265–286`) |

Refused legacies: `aggregate_rlz`, `Tpeak`, `Tlow`
(`indicator_tables.py:349–381`, enforced at `Snakefile_climate_experiment:474`).

### 1.2 `config/defaults/*.yml` (engine-native; described, not restructured)

`weathergen_config.yml` — all four sections wired: `run_weather_generator`'s 2
keys reach `generate_weather.R:124–125`; the 12 `generate_weather` keys plus 8
injected are **pinned declared-⊆-wired by
`interchange_contracts.py:265–286`** (`relax_priority` deliberately absent —
the upstream wrapper does not forward it, documented at
`weathergen_config.yml:52–57`). `apply_climate_perturbations` (15) and
`write_netcdf` (5) are the R's read surface; `extreme_prob_threshold` /
`extreme_k` are documented as inert while `exaggerate_extremes: false`
(`weathergen_config.yml:119–123`) — declared inertness, acceptable. One stale
comment: `weathergen_config.yml:141` attributes the transient flags to "rule
3.04", which is `delineate_spatial_units` after the R10-5 renumber; the
injector is rule 3.10.

`wflow_build_model.yml` / `wflow_update_waterbodies.yml` — hydromt `setup_*`
schema verbatim; rule inputs (WF1:423–425, 514, 564). Out of scope to
restructure; nothing new found.

### 1.3 The `DEFAULT_*` constants (15 definitions)

| Constant | Definition | Class |
|---|---|---|
| `DEFAULT_JULIA_THREADS` | `snake_utils.py:625` | re-export of T1 `defaults.julia_threads` — but see F3: its documented scope exceeds its real one |
| `DEFAULT_SEED` | `snake_utils.py:647` | re-export of T1 `defaults.seed` |
| `DEFAULT_WATER_YEAR_START` | `snake_utils.py:708` | re-export of T1 `defaults.water_year_start` |
| `DEFAULT_HYDROGRAPHY` / `DEFAULT_BASIN_INDEX` | `snake_utils.py:1003–1004` | back config keys; deliberately duplicate the build template's `setup_basemaps` values, guarded loud at rule 1.06 |
| `DEFAULT_SPELL_FACTOR` | `snake_utils.py:1485` | backs the two spell-factor keys |
| `DEFAULT_MAX_SUBBASINS_PER_BASIN` | `spatial/config.py:21` | backs `max_per_basin` |
| `DEFAULT_GAUGE_SNAP_TOLERANCE_M` | `spatial/config.py:25` | backs `gauge_snap_tolerance_m` |
| `DEFAULT_RIVER_UPAREA_KM2` | `spatial/config.py:26` | backs `river_uparea_km2` — **missing from both prior inventories (C3)** |
| `DEFAULT_STATS` | `get_change_climate_proj.py:104` | backs `stats` |
| `DEFAULT_MIN_REFERENCE` | `dry_month.py:32` | backs `relative_change.min_reference` (C4) |
| `DEFAULT_MAX_FLAGGED_MONTHS` | `dry_month.py:38` | backs `relative_change.max_flagged_months` (C4) |
| `DEFAULT_ANCHOR` ×2 | `metrics_definition.py:18`, `climate_figures.py:120` | duplicated signature fallback shadowing a config-backed default (draft M2 — confirmed, still unfixed) |
| `DEFAULT_DECIMALS` | `tidy_wflow_table.py:55` | genuinely no config surface — the only clean member of the draft's "correctly constants" class left |

Config-backed total: **9** (draft M3 said 5–6). Beyond `DEFAULT_*`, four more
defaults hide as inline literals: `resolution` 0.00833333
(`spatial/config.py:243`), `realizations_num` 1 (WF3:103), `run_length` 20
(WF3:236), `batch_size_max` 8 (WF3:983), `wflow_outvars`
`['river discharge','actual evapotranspiration']` (WF1:110) — the last with a
*second, different* literal in WF3 (F6).

### 1.4 Falsifier searches run (inertness evidence)

- **F1 `static_dir`:** `grep -rn static_dir blueearth_cst/ scripts/ tests/ Snakefile_* config/templates/snake_config.template.yml` → WF1:54,115–116; WF3:41 (assignment only); no `sm.params.static_dir` anywhere; remaining hits are test fixtures/comments. WF3 claim stands.
- **F2 `clim_historical` in WF2:** `grep -n clim_source Snakefile_climate_projections` → line 78 only. No param, no rule body, no derived path uses it. Claim stands.
- **F3 `julia_threads` in WF2/WF3:** `grep -n julia_threads Snakefile_climate_projections Snakefile_climate_experiment blueearth_cst/experiment/*.py blueearth_cst/weathergen/*.R` → **zero matches**. Rule 3.15's shell literal at WF3:1015. Claim stands. (The julia *version* hardcode there is known and test-pinned as a shrink-target exception — `tests/test_julia_runtime.py:90–101`; the *threads* literal is pinned by nothing.)
- **F4 `enabled` in Snakefiles:** `grep -c enabled Snakefile_*` → 0/0/0. Wrapper-only, as documented.
- **F5 retired keys:** `grep -rn "save_grids|relax_priority|start_month_hyd_year" blueearth_cst/ scripts/ tests/ config/ Snakefile_*` → only refusal code, contract comments, archived configs, and WF2's internal `start_month_hyd_year` *function argument* (`get_change_climate_proj.py`, fed from `water_year_start` at `derive_change_factors.py:185` — the argument is live; the config key is gone).
- **F6 `wflow_outvars` dual default:** WF1:110 defaults to two variables; WF3:477–479 defaults to `[]` → `indicator_tables([])` = `{}` (`indicator_tables.py:488–492`) → zero indicator tables in `WF3_TARGETS`/rule 3.16, no error. `snake_config_baseline_linux.yml` *ships in that state* — its header says "Keys absent here (wflow_outvars, …) stay absent so the Snakefile defaults apply", which is true for WF1 and produces an indicator-less WF3.
- **F7 `realizations_num`:** `grep -n realizations_num blueearth_cst/experiment/prepare_weagen_config.py` → line 101, subscript with no default; the WF3:103 default `1` cannot survive to rule 3.10.
- **F8 `temp.variance`:** `grep -rn variance blueearth_cst/experiment/prepare_cst_parameters.py blueearth_cst/weathergen/*.R` → precip-variance reads only (`prepare_cst_parameters.py:99–100,109,141,184`; `impose_climate_change.R:70`). No temp-variance read exists.

### 1.5 Tier 3 read back as input?

Checked against the brief's one question for tier 3. Nothing reads a tier-3
file as *configuration*; four reads of tier-3 files exist and all are
records-used-as-records or deliberate, documented state resolution:

1. `resolve_default_experiment_name` lists `experiments/` to reuse a dated
   experiment (`allocate.py:87–131`) — run state resolving an *unset* config
   key; deliberate, and the idempotence rationale is sound.
2. Rule 3.01 reads the wf1/wf2 config snapshots as drift-guard comparands
   (`Snakefile_climate_experiment:359–362,531–549`).
3. Rules 3.05–3.07 read `model_reference.yml` / `experiment.yml` / the merged
   log as freeze/drift guards (:601–708).
4. WF3 reads WF1's model root (`basin_dir`, :242) — the interchange artifact,
   not configuration.

No finding of the "record silently used as configuration" kind.

---

## Q2 · Necessity — live keys that should not be user-facing

Value sets measured across the five shipped configs + template.

| Key | Evidence | Verdict |
|---|---|---|
| `project.static_dir` | `"config"` in all 6 documents; WF1 uses it only in fallback paths for two keys 4 of 6 documents set explicitly; the fallbacks resolve to in-repo toolbox files, so no other value can work without relocating tracked files | **Remove** (draft M1 confirmed). Cost: 2 Snakefiles, 5 configs + template, `tests/test_check_project_consistency.py:28`, `tests/test_plot_workflow_dag.py:30,121`, `tests/test_guard_invalidation.py:241` (uses it as its guarded-key example — needs a replacement key, not deletion of the test). Breaking for every config; pair with a schema-version bump |
| `workflows.climate_projections.clim_project` | `cmip6` in all 6; the generated catalog (`cmip6_data.yml`) is the only shipped source of matching entry keys (`{clim_project}_{model}_{scenario}_{member}`, WF2:544), so any other value fails resolution today; template annotates it "text — cmip6" | **Demote**: derive from the catalog or default it, keep an override for a future second projection archive. Breaking only if removed outright |
| `shared.basin.spatial_sources.*`, `hydrography`, `basin_index`, `gauge_snap_tolerance_m`, `river_uparea_km2`, `max_per_basin` | never set in any shipped config except the template's explicit-default block | Keep, but they are *advanced* keys interleaved with first-run keys — a presentation problem (Q5), not a necessity one. Note `hydrography`/`basin_index` are coupled to `wflow_build_model.yml`'s `setup_basemaps` (guarded at rule 1.06), so "user-facing" here means "user must edit two files consistently" — worth stating in the template |
| `workflows.*.enabled` ×3 | consumed by the wrapper only | Keep (the wrapper contract is documented and tested), but the template should say the Snakefiles ignore it — a user running `snakemake -s …` directly on an `enabled: false` workflow gets a full run |
| `shared.julia_threads`, `climate_experiment.batch_size{,_max}` | machine-shaped, not basin-shaped: their right value depends on the host, yet they live in the project config that `experiment.yml` freezes and the drift guard hashes (`batch_size` is *not* under a guarded section, but `shared` partially is via `shared.basin` only — threads escape the guard) | Tension worth a ruling: performance knobs in a scientific config. Not urgent; `julia_threads` first needs to actually reach WF3 (F3) |
| `stress_test.precip.variance.{min,max}` | identity (twelve 1.0s) in every shipped config, yet required in effect (KeyError when absent) | Make genuinely optional with the identity default — same posture as the spell factors (`snake_utils.py:1499–1500`). Non-breaking |

Superseded keys are all already handled (refused) except `ensemble.min_sources`
(silent — §Q1) and the never-read `temp.variance` (silent — §Q1).

---

## Q3 · Duplication — one concept declared more than once

| # | Duplication | Locations | Which wins at runtime |
|---|---|---|---|
| D1 | `DEFAULT_ANCHOR = "YE-DEC"` twice | `metrics_definition.py:18`, `climate_figures.py:120` | Each module its own — no winner, parallel drift possible. Draft M2 confirmed verbatim; still the cheapest fix (derive both from `water_year_end_anchor(DEFAULT_WATER_YEAR_START)`, `snake_utils.py:723–735`) |
| D2 | Julia **version**: `runtime.julia_version` vs literal `+1.11.7` | `advanced_settings.yml:123` vs `Snakefile_climate_experiment:1015` | The literal wins in rule 3.15. Known, deliberate, test-pinned as a shrink-target (`test_julia_runtime.py:90–101`) — drift *would* be caught (the settings value is asserted against pixi/Manifest, and any new literal fails the test), but a settings bump leaves 3.15 on the old version until the exception is cleared |
| D3 | Julia **threads**: `defaults.julia_threads` (+ `shared.julia_threads` override) vs literal `--threads 4` | `advanced_settings.yml:58` vs `Snakefile_climate_experiment:1015` | The literal wins for all WF3 runs; **no test ties the two** (F3). Unlike D2, the override key makes this user-visible: the documented knob half-works |
| D4 | `wflow_outvars` default, twice and differently | WF1:110 (`['river discharge','actual evapotranspiration']`) vs WF3:477–479 (`[]`) | Consumer-dependent: same absent key yields a two-variable model and a zero-table experiment (F6). The worst duplication found — it changes what a run *produces* |
| D5 | `resolution` default | template active value `0.00833` (`snake_config.template.yml:21`) vs code default `0.00833333` (`spatial/config.py:243`) | Config wins when set (always, today). A config that *omits* it builds a ~0.4 %-different grid from one that copies the template. Two spellings of "1/120 degree", neither exact |
| D6 | `hydrography`/`basin_index` defaults vs `wflow_build_model.yml` `setup_basemaps` | `snake_utils.py:1003–1004` vs the build template | By-design duplication with a loud runtime guard at rule 1.06 (WF1:63–66) — the acceptable pattern; D1–D5 lack exactly this guard |
| D7 | `spatial_sources` literals twice in one module | `spatial/config.py:33–36` (dataclass defaults) and `:267–270` (`_source_name` defaults) | Same values, same file; harmless today, classic future-drift shape |
| D8 | `2010` historical-end anchor twice in one module | `prepare_weagen_config.py:33` (`compute_nr_years`) and `:99` (`start_year`) | Both live; a change to one silently desynchronizes series span from series label |
| D9 | Template comments restating code defaults | e.g. `#stats: [mean, median, std]` (:131) vs `DEFAULT_STATS`; `#julia_threads: 4` (:67); **`#batch_size: 4` (:166) vs the dynamic default `min(batch_size_max, ceil(K/cores))` (WF3:986–988) — already wrong**, and `#seed: auto` / `#water_year_start: Oct` (:52,66) show *example* values where the header (:7) promises "commented-out keys … show their default" (actual defaults: 123, Jan) | Code wins; the comments are the only default surface a user sees, and nothing tests them |
| D10 | One quantity, two config keys, manual coupling: WF2 `future_horizons` vs WF3 `horizontime_climate ± run_length/2` | rapid config maintains them consistent by hand (`snake_config_rapid.yml:65–68,74–75`) | Independent by design (overlay vs experiment window) — not derivable, but the template says nothing about the relationship; record as documentation debt, not duplication |
| D11 | The inventory itself, twice | brief appendix + draft appendix | Both stale identically (§0) — the draft's own footnote predicted this; delete one when either is next corrected |

Water-year duplication (the brief's P3 example) is **resolved** in this tree:
one key, seams converting at the boundary (`prepare_weagen_config.py:110`,
`derive_change_factors.py:185`).

---

## Q4 · Naming and documentation

### 4.1 Where the convention is broken

Almost nowhere, mechanically. All current project-config keys are snake_case
with lowercase booleans (naming.md §2); no tracked config carries `TRUE/FALSE`.
The one arguable breach: naming.md §5 reserves `_dir`/`_path` for paths, and
`static_dir` honours it while six other path-valued keys carry no suffix
(`data_sources`, `data_sources_climate`, `gauge_points`,
`observations_timeseries`, `model_build_config`, `waterbodies_config`) — but §5
binds *variables*, not YAML keys, so this lands in 4.2.

### 4.2 Where the convention is silent (the larger finding)

naming.md governs config keys in exactly one clause (§2 YAML: case and
booleans). It is silent on, and the tree consequently disagrees about:

- **Units in names.** With unit: `gauge_snap_tolerance_m`, `river_uparea_km2`.
  Without: `resolution` (degrees), `run_length` (years), `horizontime_climate`
  (a year), `historical_year_range` (years), `min_historical_years` (has it —
  T1). No rule says which is right, so both exist.
- **Path keys.** Six path-valued keys, no suffix convention (above).
- **Abbreviation.** `clim_historical`, `clim_project`, `wflow_outvars`,
  `realizations_num` vs the spelled-out `historical_window`,
  `observations_timeseries`. §8b's "nouns are full words" governs rule names
  only.
- **Word order.** Qualifier-first (`historical_window`, `future_horizons`) vs
  qualifier-last (`clim_historical`, `batch_size_max`, `realizations_num`).
  Adjacent concepts read in opposite orders.
- **Count spelling.** `realizations_num`, `step_num` vs the internal
  style ruling `ST_NUM → stress_test_count` (naming.md §10) — the convention
  already prefers `_count` internally and says nothing about config keys.

### 4.3 One concept, several names across tiers

| Concept | T2 key | Internal | Engine (forced?) |
|---|---|---|---|
| realization count | `realizations_num` | `RLZ_NUM`, `rlz_count` (target style §10), `rlz_` token | `n_realizations` (weathergenr — forced) |
| water-year start | `water_year_start` (Oct) | `WATER_YEAR_START` | `year_start_month` (int; forced) — converted at seam, correct pattern |
| observed climate source | `clim_historical` | `clim_source` | — (ours both; two spellings for zero reason) |
| gauge file | `gauge_points` | `output_locations` ("the internal compatibility name", WF1:121–122) | file ships as `output_locations.csv` |
| run window centre | `horizontime_climate` | `middle_year` (rule 3.10 param, WF3:808) | — (ours both) |
| catalog path | `data_sources` | `DATA_SOURCES`, param names `data_catalogs` (1.01), `data_catalog` (1.08/1.09), `catalog` (region rule), `catalog_path` (2.04/2.05) | — five internal spellings, all ours |
| perturbation magnitudes | `stress_test.temp.mean` etc. | `temp_change` columns | `temp_delta`, `precip_mean_factor`, `precip_var_factor` (forced tier-1) |
| output variable | `wflow_outvars` labels | tokens (`aet`), codes (`gwr`) | CSDMS names (forced) — the "third spelling" is a documented, accepted cost (`indicator_tables.py:22–47`) |

Forced: the weathergenr and CSDMS columns. Ours to fix: `clim_historical` ↔
`clim_source`, `gauge_points` ↔ `output_locations`, `horizontime_climate` ↔
`middle_year`, and the five internal catalog-param spellings.

### 4.4 Units and types — where does the unit live?

In the name: `gauge_snap_tolerance_m`, `river_uparea_km2`. In a comment:
`resolution` ("numeric — degrees", template:21), `run_length` ("integer —
years", :141), `horizontime_climate` ("integer — year", :140). Nowhere: none
found — the template's type annotations are actually complete for the active
keys. The exposure is that comments are the only carrier and D9 shows comments
drift.

### 4.5 Documentation completeness (template as the user surface)

- **In the template:** all 60 current leaves appear (C6 corrected the brief's
  premise) *except* `output_locations` (legacy — correctly absent) and
  `stress_test.temp.variance` (absent because it does nothing — but nothing
  says so, and the precip block's presence invites symmetry; a one-line "temp
  has no variance axis" comment would pre-empt F8).
- **Defaults invisible without reading Python:** `realizations_num` (real
  default 1 — template shows 2 active), `run_length` (20 — shown active), the
  dynamic `batch_size` (comment claims 4 — wrong), `resolution` (code default
  differs from shown value, D5). The pointer pattern used for
  `seed`/`water_year_start` ("absent, it takes `defaults.X` from
  advanced_settings", :42–44, 54–55) is the honest form; the D9 rows are not.
- **Required-vs-optional marking:** conveyed *only* by which keys are
  commented out, and inconsistently — `realizations_num`, `run_length`,
  `run_historical` are optional but active; `precip.variance` is
  required-in-effect but indistinguishable from `temp`'s optional-looking
  block. No `# required` markers exist.
- **Stale comments (defect class per the brief):**
  `snake_config_wf2_fast.yml:44–47` claims "this is the baseline-recording
  seed" inside the config whose own header says "NOT a baseline seed" —
  copy-paste from the baseline config;
  `weathergen_config.yml:141` credits rule 3.04 with the transient-flag
  injection (it is 3.10); template:166 `#batch_size: 4` misstates a dynamic
  default; `advanced_settings.yml:48–58` describes `julia_threads` as
  governing Wflow generally while WF3 ignores it (F3).

### 4.6 Beyond conformance — is the name good? (reader test)

Recorded as *plausible wrong reading*, per the brief.

| Key | Wrong reading a hydrologist would plausibly take | Actual meaning |
|---|---|---|
| `horizontime_climate` | "length of the climate horizon, in years" (a 20–30-style number), or a WF2 horizon name | the *central year* of the WF3 run window (2050); also sets generated-series length via the 2010 anchor. Not English; states neither quantity nor unit. The worst name in the config |
| `stress_test.*.step_num: 1` | "one grid point on this axis" | *two* points — the count is `step_num + 1` (`snake_utils.py:1479–1481`). An off-by-one a user meets on their first grid. The template says "integer" and nothing else |
| `mean.min` / `mean.max` | validation bounds or clamps on the data | the *sweep endpoints* the grid interpolates between, per calendar month |
| `run_historical` | "run the historical (WF1) simulation" — collides with `clim_historical` and `historical_window` vocabulary | include the unperturbed `st_0` baseline member; and `false` silently drops 2 of 11 q-metrics (the R11 P3 case) — the one boolean whose wrong guess changes the result surface |
| `run_length` | length of *the* run — but which? plausibly WF1's | the WF3 per-member simulation length in years |
| `static_dir` | wflow `staticmaps`/static data directory (strong association in this domain) | the toolbox's in-repo `config/` root; should not exist (M1) |
| `clim_historical` | a window or a boolean ("use historical climate?") | catalog *entry name* of the observed source (`era5`) |
| `gauge_points` | fine — but it resolves to a file named `output_locations.csv`; one of the two nouns is wrong (the key is the better one; the fixture filename and the internal name carry the legacy) |
| `realizations_num` | guessable; reads Yoda-order; three sibling spellings across tiers (4.3) |
| `data_sources` (singular path) | "a list of sources" | one hydromt catalog path (a list is tolerated at WF1:167 but nowhere documented) |
| `max_per_basin` | meaningless alone — deliberate path-leaning (`automatic_subbasins.max_per_basin` is clear); fails when quoted bare in an error message, which `spatial/config.py:253–256` avoids by quoting the full path — the policy works but is nowhere stated as policy |
| `wflow_outvars` | guessable by a wflow user; "outvars" is merely short, not domain vocabulary — but values are the real interface and they are readable labels |
| `water_year_start` | none — a genuinely good name: domain term, value self-describing (`Oct`), unit-free by construction | — |

**Rename posture (naming.md §7 costs):** every key above is a checked-in
example-config key = a §7 contract surface. Only two are *wrong* rather than
not-what-you'd-choose: `horizontime_climate` (not a word, misleads on
quantity) and `step_num` (its literal reading is off by one). Recommend
renaming only those two if the Q5 regroup happens anyway (one migration, one
note), e.g. `horizon_year` and `steps_beyond_baseline`… the second is worse
than documenting the +1; for `step_num` prefer a comment + template fix over a
rename. Everything else: grandfather, fix the documentation.

---

## Q5 · Organisation

**Are three tiers right?** Yes — no fourth tier is hiding, and T3 is clean
(§1.5). What the split hides is the draft's P1, now measured bigger: the
*default* surface of T2 lives in four places — `advanced_settings.yml` (3),
`DEFAULT_*` constants (9), inline literals (5+), and template comments (the
only one a user sees, and the one that is wrong twice). The tiers are fine;
the default placement has no rule. This confirms the draft's problem statement
and sharpens M3's count.

**Is `project` / `shared` / `workflows.*` the right axis?** Yes, and `shared`
is *not* a leftovers bin — measured by consumers, every `shared` key is read
by ≥2 workflows except `seed` (WF3-only today, placed for the stated
one-seed-policy reason, `advanced_settings.yml:60–79`) and `julia_threads`
(WF1-only *by defect*, F3). The misfits are in `project`: `static_dir` (delete)
and arguably `data_sources_climate` (WF2-only; but catalogs are project-level
bindings, and moving it would break symmetry with `data_sources` for no user
gain — leave it).

**Nesting depth.** `shared.basin.automatic_subbasins.max_per_basin` is four
levels for one integer in a section holding exactly one key
(`spatial/config.py:207`). Collapse to `basin.max_automatic_subbasins` only as
part of a schema-version bump (breaking, legibility-only — same verdict as
draft M4). `stress_test.{axis}.mean.{min,max}` depth is justified: the shape
*is* the grid.

**Grouping by kind.** Draft P4 confirmed: `shared.basin` mixes basin identity
(`region`, `resolution`), catalog bindings (`hydrography`, `basin_index`,
`spatial_sources.*`), observation wiring (`gauge_points`), and delineation
tuning (`max_per_basin`, `gauge_snap_tolerance_m`, `river_uparea_km2`). The
practical cost shows up in the user test below.

**The user-oriented test — keys a new basin must touch:**
`project_dir` · `basin.region` · (`resolution`) · `basin.gauge_points` ·
`historical_window.{starttime,endtime}` · (`simulation_window`) ·
`observations_timeseries` · (`models`/`scenarios`/`members` if defaults don't
suit) · `future_horizons` · `experiment_name` · `horizontime_climate` ·
`run_length` · `stress_test` ranges · `run_historical`.

Is it contiguous? Mostly yes *within* sections — the real finding is inverted:
the file **opens** with the four `project` keys, of which three
(`static_dir`, both catalog paths) are plumbing a new user should never touch,
and the first section a user must actually edit (`shared.basin`) buries its
two required keys (`region`, `gauge_points`) among seven advanced knobs nobody
has ever varied (Q2). The contiguous grouping that would replace it: a
first-screen block of exactly the required keys (basin identity + window +
observations + experiment window + grid), with plumbing defaulted away
(`static_dir` deleted, catalogs defaulted to the shipped paths, `clim_project`
demoted) and advanced keys in a clearly-marked trailing block. That is a
template *reordering* plus 3 key deletions/demotions — the schema itself needs
no new axis.

**Where should a key's default be visible?** Recommendation: a single
declarative schema registry in Python — `(key, required, default | pointer,
unit, doc)` — from which (i) parse-time validation reads, (ii) the template's
comment block is *checked* (a test that the template's stated default equals
the registry's), and (iii) the docs table is generated. `advanced_settings`
keeps only genuinely toolbox-wide values, as today; per-key defaults stop
living in scattered `DEFAULT_*`/literals. Argument against: it is a real
mechanism (new module + tests + migration of 9 constants), it centralizes what
is currently local to each consumer, and a half-adopted registry is worse than
none — two homes for defaults again. The lighter alternative (move the 9 into
`advanced_settings`) fails its own file's charter: these are not "knobs a
normal project never touches", they are per-key defaults. If the registry is
rejected, the fallback is draft M3 + a template-comment test. This is the
architecture decision the brief says to name and stop at — `design-document`
owns it. **Stopping here per Non-goals.**

---

## 6. Ranked findings (consequence first)

Severity classes: **A** can produce a wrong or silently-empty number; **B**
misleads a user into a wrong decision; **C** untidy/drift-hazard.

| Rank | Finding | Consequence | Proposed action | Cost | Breaking? |
|---|---|---|---|---|---|
| 1 (A) | `wflow_outvars` has two defaults; WF3's `[]` yields a zero-indicator experiment with no error; `snake_config_baseline_linux.yml` ships in that state (F6, D4) | An experiment "succeeds" producing no response surface — the exact P2 class with a shipped reproducer | WF3 reads the same default WF1 uses (single-source the literal), or refuses an absent key at WF3 parse | ~5 lines + a test; linux config unchanged | No |
| 2 (A) | `stress_test.temp.variance.{min,max}` accepted and never read (F8) | User believes temperature variance is perturbed; results computed without it | Extend the axis guard in `prepare_cst_parameters.py` to refuse unknown *sub-keys* per axis (it already refuses unknown axes) | ~10 lines + test | Only for configs already carrying the dead block — which is the point |
| 3 (B) | `shared.julia_threads` ignored by WF3 rule 3.15; `--threads 4` and `+1.11.7` hardcoded (F3, D3) | Documented performance knob half-works; on a production basin the stress-test sweep — the dominant cost — cannot be tuned; threads do not change numbers, so B not A | Adopt `julia_prefix(julia_threads)` in 3.15 (the test at `test_julia_runtime.py:96–101` is already written to shrink); until then caveat `advanced_settings.yml:48` | Small; owner noted a concurrent worktree owns that file | No |
| 4 (B) | `realizations_num` "default 1" is unreachable — rule 3.10 KeyErrors on an absent key (F7) | A minimal config crashes mid-run with a bare KeyError instead of parsing loud or working | Pass `RLZ_NUM` as a rule 3.10 param instead of re-reading the YAML (also removes a duplicate read path), or make the script default match | ~4 lines | No |
| 5 (B) | `clim_historical` read-required-unused in WF2 (F2); `static_dir` read-required-unused in WF3 (M1) | Required keys that do nothing teach users wrong causality; M1's key admits only one value | WF2: drop the read (2 lines, non-breaking). `static_dir`: delete key toolbox-wide per draft M1 | M1 cost as §Q2 | M1 yes — schema bump; WF2 read no |
| 6 (B) | Template default-comments drift: `#batch_size: 4` false; `#seed: auto`/`#water_year_start: Oct` are examples despite the header's "commented keys show their default"; `resolution` two-value default (D5, D9) | The only default surface a user sees is wrong in one place already | Fix the three comments now; longer term the Q5 registry/test decides whether comments can be trusted | Minutes | No |
| 7 (B) | Required-vs-optional not marked in the template; `precip.variance` required-in-effect via raw KeyError; optional keys shown active | First-run users cannot tell the minimal config | Mark `# required` per key; give `precip.variance` the identity default (Q2) | Small | No |
| 8 (C) | `DEFAULT_ANCHOR` ×2 (D1/M2) | Water-year drift hazard if one is edited | Draft M2's fix, unchanged | Trivial | No |
| 9 (C) | `ensemble.min_sources` silently ignored if present | Legacy configs believe a floor exists | Add to a refused-keys list like WF3's `RETIRED_EXPERIMENT_KEYS` | ~5 lines | Only for configs carrying it |
| 10 (C) | Stale comments: wf2_fast "baseline-recording seed"; weathergen "rule 3.04"; `advanced_settings` julia_threads scope | Comments contradicting code are the brief's stated defect class | Fix the three lines | Minutes | No |
| 11 (C) | 9 config-backed defaults in Python + 5 inline literals; no placement rule (P1, M3 enlarged) | Every future key re-litigates placement; users read Python for defaults | The Q5 registry decision (design-document scope) | Medium | No (defaults keep values) |
| 12 (C) | Naming: `horizontime_climate` (wrong), `step_num` off-by-one reading, internal spelling pairs (`clim_source`, `output_locations`, `middle_year`, five catalog-param names) | Reader-test failures; none changes a number | Rename only `horizontime_climate` if the Q5 regroup lands (one §7 migration note); document `step_num`'s +1 in the template; converge internal spellings opportunistically | Rename: template+5 configs+WF3+docs | Rename yes; rest no |
| 13 (C) | Inventory duplicated brief↔draft, both stale (C1–C6, D11) | Two wrong maps outrank one | Correct the draft's appendix from §0 or delete it in favour of the brief's, per the draft's own footnote | Minutes | No |

---

## 7. P2 feasibility — can Q1's classification be produced mechanically?

**A single "declared keys ⊆ read keys" check is not feasible as one
mechanism.** Two measured obstacles:

1. **Provenance defeats access-tracking.** All three Snakefiles serialize the
   *entire* config into digests and snapshots
   (`snapshot_bundle_digest(config, …)` at `Snakefile_model_creation:175`,
   `Snakefile_climate_projections:54`; `guarded_sections_digest`'s
   `json.dumps(config…)` at `Snakefile_climate_experiment:338–350`). A
   tracking-Mapping proxy would observe every key being "read" on every parse.
   `temp.variance` *is* read — by the provenance code. Reach, not access, is
   the property, and access instrumentation cannot see the difference.
2. **Four distinct read sites.** Parse-time `get_config`/subscripts; rule
   `params:` consumed (or not) by scripts; scripts that re-read the YAML file
   at rule time (`prepare_cst_parameters.py:67–71`,
   `prepare_weagen_config.py:88–89` — invisible to any parse-time check); and
   the R/Julia seams. No one instrumentation point covers them.

**What is feasible — a four-part battery, each cheap, jointly covering every
observed failure in this class** (the four known + the four found here):

1. **`params:` ↔ script cross-reference test.** Parse each Snakefile's rule
   bodies (precedent already in-tree: `tests/test_climate_store_contract.py`
   and `test_region_rule.py` parse rule bodies across workflows), collect
   `(param_name, script_path)`, and assert the script's text references
   `params.<name>` (or the module's documented arg-passing shim). Catches the
   `start_month_hyd_year` class (c). Snakemake's `params:` indirection is not
   an obstacle at this level because the check is textual per rule, not
   semantic; lambdas and `**splat` params need a small whitelist (the three
   `_rule` dataclasses centralize most of them already).
2. **Assigned-but-unreferenced lint over Snakefile preambles.** The module
   level of a Snakefile is plain Python; a regex/AST pass flagging a name
   assigned once and never referenced again (f-strings included) catches
   `static_dir` in WF3 and `clim_source` in WF2 *today* — both fail exactly
   this test. Vulture/ruff cannot parse Snakefiles whole, but extracting the
   statements above the first `rule` keyword is mechanical.
3. **Engine-template key ⊆ wired-argument checks.** Already exists for the
   generate_weather section (`interchange_contracts.py:265–286` pins declared
   keys, with `relax_priority` deliberately unpinned) — this caught-class
   covers `relax_priority` and C34's `evaluate.model`. Extend the same pin to
   `apply_climate_perturbations`/`write_netcdf` (grep the R for each key).
4. **Sub-key closure where a section is structural.** The `stress_test` axis
   guard (`prepare_cst_parameters.py:77–86`) closes top-level keys; closing
   per-axis sub-keys (rank-2 fix) is the same pattern one level down. The
   general form — a closed schema for the whole project config, as
   `_ADVANCED_SETTINGS_SCHEMA` already does for T1 — catches user typos and
   ignored legacies (`ensemble.min_sources`) but *not* the static_dir class
   (a schema happily declares a key nothing reads), so it complements rather
   than replaces 1–2.

What stays manual: semantic reach through R and Julia (a key wired into the R
that the R then ignores), and value-level inertness (`extreme_k` inert while
`exaggerate_extremes: false`). Both are declared in comments today; no cheap
mechanical check exists.

Confidence: judgement with reasons, per the brief — checks 1–2 were not
prototyped here (review-only dispatch), but check 2's two positive cases and
check 1's negative space were verified by the greps in §1.4, and check 3 is
already running in CI, which is direct evidence of feasibility for its class.

---

## Open questions (not resolvable by this review)

1. **Registry vs advanced_settings for the 9 config-backed defaults** — the
   Q5 decision; needs an owner ruling and a design document. What would settle
   it: whether the template-comment test (registry option iii) is acceptable
   maintenance, and whether a second consumer of per-key metadata (docs
   generation, GUI schema for CST-frontend) is actually foreseen — noting the
   standing rule that web-app needs never constrain this repo.
2. **Should WF3 honour `shared.julia_threads`, or should the key be demoted to
   WF1-only with its doc corrected?** Owner intent unclear because the WF3
   pass was explicitly parked for a concurrent worktree
   (`test_julia_runtime.py:96–99`).
3. **Is `clim_project` a future extension point or dead generality?** If a
   second projections archive is on the roadmap, keep the key; if not, demote
   it (Q2). Settled by the roadmap owner, not by code.
4. **`enabled`'s contract for direct `snakemake` invocation** — should the
   Snakefiles warn when run with their own `enabled: false`? Cheap, but it
   couples Snakefiles to a wrapper-only key; needs a ruling.
5. **Whether `batch_size`/`julia_threads`-class machine knobs belong in a
   frozen-by-experiment project config at all** (Q2) — touches the
   `experiment.yml` freeze semantics.

## Assumptions and residual risk

- **Tree reviewed: `lane/devmeta` @ `8d9c548`.** `lane/pipeline` carries newer
  commits (seed-config CLI gate work); any key added or retired there is
  outside this snapshot. The classification is of this checkout, as dispatched.
- R and Julia consumers were verified at grep depth (argument wiring), not by
  execution; the `interchange_contracts` pins are the mechanical backstop for
  the deepest seam. A key wired into R but ignored *inside* weathergenr would
  not be visible here.
- Tier-3 evidence used the seeded `test_case/test_local/config/` tree
  (catalogs/observations/runs/templates present); the experiments subtree was
  not walked file-by-file — §1.5's conclusions rest on the Snakefile read
  sites, which are exhaustive for what the *workflows* read back.
- Leaf counting rules are stated in §0; a different rule (e.g. exploding
  `variables` per-variable) moves totals but no classification.
- The `git status` acceptance criterion holds: this file is the only change.
