# WF2 — rule-level overview (`Snakefile_climate_projections`)

Working aid for the planned efficiency/modularity rework of workflow 2. Records
**current** behavior, rule by rule: what each rule does, what it consumes, what
it writes, and which rule consumes that output.

Scope split — this file is strictly **rule/DAG level**. The behavioral contract
(owned config keys, the `precip`/`temp` unit split, `save_grids` semantics, the
known metadata regression, downstream-consumer semantics) lives in
`dev/workflows/climate_projections.md` and is **not** repeated here.

Grounded in `Snakefile_climate_projections`,
`blueearth_cst/projections/*.py`, `blueearth_cst/shared/merge_{logs,benchmarks}.py`,
`blueearth_cst/model/copy_config_files.py`, and the seed config
`config/workflows/snake_config_model_test.yml`.

Path shorthand used below:

- `PD` = `project.project_dir`
- `CPD` = `{PD}/climate_projections/{clim_project}` (seed: `.../cmip6`)
- `BD` = `{PD}/hydrology_model` (workflow-1 product)

---

## 1. Rule inventory

11 rules. "Banner" is the `W.NN` reference number in the message/log/benchmark
names — **definition order, not execution order**.

| Banner | Rule | Script | Fan-out |
| --- | --- | --- | --- |
| 2.00 | `all` | — | — |
| 2.01 | `fetch_gcm_raw` | `projections/fetch_gcm_raw.py` | `{series_key}` |
| 2.02 | `reduce_gcm_series` | `projections/get_stats_climate_proj.py` | `{series_key}` |
| 2.03 | `copy_config` | `model/copy_config_files.py` | — |
| 2.04 | `derive_change_factors` | `projections/derive_change_factors.py` | — (one job) |
| 2.06 | `plot_climate_proj_timeseries` | `projections/plot_proj_timeseries.py` | — (gather) |
| 2.07 | `gather_series_logs` | `shared/merge_logs.py` | — (gather) |
| 2.08 | `gather_raw_logs` | `shared/merge_logs.py` | — (gather) |
| 2.10 | `gather_benchmarks` | `shared/merge_benchmarks.py` | — (gather) |
| 2.11 | `extract_climate_grid` | `model/extract_historical_climate.py` | — (shared store) |

**Nine rules plus `all`, not the eight the design's §8 predicted.** The extra one
is `gather_raw_logs`, which arrived with the fetch/reduce split (design revision
6) after that count was written. Recorded rather than quietly reconciled.

What went, and where it went:

| Retired | Replaced by |
| --- | --- |
| `monthly_stats_hist`, `monthly_stats_fut` | `reduce_gcm_series` over `{series_key}` (step 3) |
| `monthly_change`, `monthly_change_scalar_merge` | `derive_change_factors`, one job (step 4d) |
| `gather_stats_hist_logs`, `gather_stats_fut_logs` | `gather_series_logs` (step 3) |
| `gather_change_logs` | nothing — stage B is one job and writes its own log (step 4d) |
| the `ruleorder:` directive | nothing — it named two rules that no longer exist (step 4d) |

**Job count on the seed config** (3 models × 2 scenarios × 1 horizon):
1 (`copy_config`) + 3 (2.02) + 6 (2.03) + 6 (2.04) + 1 (2.05) + 1 (2.06)
+ 3 (2.07–2.09) + 1 (2.10) = **22 jobs**, plus `all`.

---

## 2. Rule detail

### 2.00 `all` — target aggregator

Declares the workflow's terminal targets. Note that the merged logs and the
benchmark table are **targets**, not optional bookkeeping — WF2 is not complete
until 2.07–2.10 have run.

| Direction | Item | Producer |
| --- | --- | --- |
| in | `{CPD}/summary/annual_change_scalar_stats_summary.{nc,csv}`, `..._summary_mean.csv` | 2.05 |
| in | `{CPD}/plots/projected_climate_statistics.png` | 2.05 |
| in | `{CPD}/plots/precipitation_anomaly_projections_abs.png`, `temperature_anomaly_projections_abs.png` | 2.06 |
| in | `{PD}/config/runs/snake_config_climate_projections.yml` | 2.01 |
| in | `{PD}/logs/2.02_monthly_stats_hist.log`, `2.03_monthly_stats_fut.log`, `2.04_monthly_change.log` | 2.07/2.08/2.09 |
| in | `{PD}/benchmarks/wf2_benchmarks.md` | 2.10 |

### 2.01 `copy_config`

Snapshots the run's `--configfile` YAML plus the CMIP6 data catalog into the
project's config bins, so a finished project records what produced it.

- **In:** `config_path` (the `--configfile` YAML, from `workflow.configfiles[0]`).
- **Params:** `data_catalogs = project.data_sources_climate`, `workflow_name`,
  `config_dir = {PD}/config`.
- **Out:** `{PD}/config/runs/snake_config_climate_projections.yml` (YAML,
  verbatim copy). Side effect, **not declared as output**: each catalog in
  `data_sources_climate` copied to `{PD}/config/catalogs/`.
- **Connections:** none — an isolated leaf consumed only by `all`. No other WF2
  rule depends on it, so it never gates the compute chain.

### 2.02 `monthly_stats_hist`

Per `{model}`: opens the CMIP6 **historical** source from the hydromt catalog,
clips it to the basin bbox (+1° buffer), resamples to monthly
(`precip`→sum, else→mean), and averages over the grid into a monthly scalar
time series. Loops over `members` **inside the script** and merges them.

- **In:** `ancient({BD}/staticgeoms/region.geojson)` — GeoJSON clip mask from WF1.
- **Params:** `catalog_path`, `project_dir`, `name_scenario="historical"`,
  `name_members`, `name_model={model}`, `name_clim_project`, `variables`,
  `save_grids`.
- **External input (not in the DAG):** catalog source
  `{clim_project}_{model}_historical_{member}` — remote GCS/zarr. Absent source
  ⇒ empty `xr.Dataset()` written, not an error.
- **Out:** `temp({CPD}/historical_stats_time_{model}.nc)` — netCDF, dims
  `(clim_project, model, scenario, member, time)`, monthly, zlib.
- **Out, undeclared** (only when `save_grids: true`): `{CPD}/historical_stats_{model}.nc`
  (12-month climatology grid). The Snakefile's output line for it is commented
  out (line 96).
- **Log/benchmark:** `logs/_parts/2.02_monthly_stats_hist/{model}.log`,
  `benchmarks/_parts/2.02_monthly_stats_hist/{model}.tsv`.
- **Consumed by:** 2.03 (ordering only — see §3), 2.04 (`ancient`), 2.06, 2.07.

### 2.03 `monthly_stats_fut`

Same script and same computation as 2.02, per `{model}×{scenario}`, on the
future (SSP) sources instead of historical.

- **In:** `ancient({BD}/staticgeoms/region.geojson)`;
  `{CPD}/historical_stats_time_{model}.nc` — commented in the Snakefile as
  *"make sure starts with previous job"*.
- **Params:** as 2.02 with `name_scenario={scenario}`.
- **External input:** catalog source `{clim_project}_{model}_{scenario}_{member}`.
- **Out:** `temp({CPD}/stats_time-{model}_{scenario}.nc)` — netCDF, same shape as
  2.02's output.
- **Out, undeclared** (`save_grids: true`): `{CPD}/stats-{model}_{scenario}.nc`.
- **Log/benchmark:** `logs/_parts/2.03_monthly_stats_fut/{model}_{scenario}.log`,
  matching `.tsv`.
- **Consumed by:** 2.04 (`ancient`), 2.06, 2.08.

### 2.04 `monthly_change`

Per `{model}×{scenario}×{horizon}`: slices the hist series to
`historical_year_range` and the future series to that horizon's window,
aggregates each to hydrological years (`YS-{start_month_hyd_year}`), and reduces
to 8 statistics (`mean, std, var, median, q_90, q_75, q_10, q_25`) as
**change factors** — multiplicative % for `precip`, additive degC for `temp`.

- **In:** `ancient({CPD}/historical_stats_time_{model}.nc)`,
  `ancient({CPD}/stats_time-{model}_{scenario}.nc)`.
- **Params:** `clim_project_dir`, `start_month_hyd_year`, `name_{model,scenario,horizon}`,
  `time_horizon_hist`, `time_horizon_fut` (via the `get_horizon` wildcard
  function), `save_grids`, plus the two **grid paths passed as params, not
  inputs**: `stats_path_hist`, `stats_path`.
- **Out:** `temp({CPD}/annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc)`
  — netCDF, dims `(stats, clim_project, model, scenario, horizon, member)`.
  Empty future series ⇒ **dummy empty netCDF** so Snakemake sees the target.
- **Out, undeclared** (`save_grids: true`):
  `{CPD}/monthly_change_mean_grid-{model}_{scenario}_{horizon}.nc`.
- **Log/benchmark:** `logs/_parts/2.04_monthly_change/{model}_{scenario}_{horizon}.log`,
  matching `.tsv`.
- **Consumed by:** 2.05, 2.09; the undeclared grid netCDF by 2.06 (via params).
- **Raises** (fail-loud guards, t260720d) on asymmetric hist/clim variable sets
  or member sets.

### 2.05 `monthly_change_scalar_merge`

Merges every per-(model, scenario, horizon) change netCDF into one summary
dataset (dropping the dummy empties), writes it as netCDF + two CSVs, and draws
the joint precip/temp scatter used as the plausibility overlay.

- **In:** `ancient(expand(annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc, ...))`
  — the full 2.04 fan-out (seed: 6 files).
- **Params:** `clim_project_dir`, `horizons` (used to relabel each horizon by its
  mid-year in the plot).
- **Out:** `{CPD}/summary/annual_change_scalar_stats_summary.nc` (netCDF, zlib),
  `.../annual_change_scalar_stats_summary.csv`,
  `.../annual_change_scalar_stats_summary_mean.csv` (`stats="mean"` slice only),
  `{CPD}/plots/projected_climate_statistics.png` (seaborn `JointGrid`).
- **Log/benchmark:** `logs/2.05_monthly_change_scalar_merge.log` (**unmerged** —
  single job, so no `_parts`), `benchmarks/_parts/2.05_....tsv`.
- **Consumed by:** `all`; the summary `.nc` is also an input to 2.06 (declared,
  but the script never opens it — a pure ordering edge) and to 2.10.

### 2.06 `plot_climate_proj_timeseries`

Reopens all the monthly scalar series (hist + future), computes multi-model
5/50/95 percentile envelopes of absolute values and anomalies (annual and
monthly climatology), writes one merged timeseries netCDF, and renders the
projection figures.

- **In:** `{CPD}/summary/annual_change_scalar_stats_summary.nc` (declared, unread);
  all `historical_stats_time_{model}.nc`; all `stats_time-{model}_{scenario}.nc`.
- **Params:** `clim_project_dir`, `scenarios`, `horizons`, `save_grids`,
  `change_grids` (the undeclared 2.04 grid netCDFs, referenced by path).
- **Out (declared):** `{CPD}/plots/precipitation_anomaly_projections_abs.png`,
  `{CPD}/plots/temperature_anomaly_projections_abs.png`,
  `{CPD}/timeseries/gcm_timeseries.nc` (netCDF, declared under the misleading
  output label `timeseries_csv`).
- **Out (undeclared):** 6 further PNGs from the same loops —
  `precipitation_anomaly_projections_anom`, `temperature_anomaly_projections_anom.png`,
  `precipitation_monthly_projections_{abs,anom}.png`,
  `temperature_monthly_projections_{abs,anom}.png`; plus, when `save_grids: true`,
  4 gridded map PNGs per `{scenario}×{horizon}`.
- **Log/benchmark:** `logs/2.06_plot_climate_proj_timeseries.log`,
  `benchmarks/_parts/2.06_....tsv`.
- **Consumed by:** `all` (2 PNGs), 2.10 (one PNG as an ordering barrier).

### 2.07 / 2.08 / 2.09 `gather_*_logs`

Three structurally identical rules, one per fan-out stage. Each concatenates that
stage's per-job part logs into a single merged log, regenerated fresh each run.
The `.nc` inputs are **sync barriers only** — `merge_logs.py` reads
`params.parts`, never `input`.

| Rule | Banner | Input barrier (`.nc`) | `params.parts` | Output |
| --- | --- | --- | --- | --- |
| `gather_stats_hist_logs` | 2.07 | all 2.02 outputs | `logs/_parts/2.02_.../{model}.log` | `{PD}/logs/2.02_monthly_stats_hist.log` |
| `gather_stats_fut_logs` | 2.08 | all 2.03 outputs | `logs/_parts/2.03_.../{model}_{scenario}.log` | `{PD}/logs/2.03_monthly_stats_fut.log` |
| `gather_change_logs` | 2.09 | all 2.04 outputs | `logs/_parts/2.04_.../{model}_{scenario}_{horizon}.log` | `{PD}/logs/2.04_monthly_change.log` |

**Banner ≠ output filename here**: the rule's banner is 2.07/2.08/2.09 but the
merged log it writes is named after the *gathered* stage (2.02/2.03/2.04).

- These rules declare **no `log:` and no `benchmark:`** — they never appear in
  `wf2_benchmarks.md`.
- **Consumed by:** `all`.

### 2.10 `gather_benchmarks`

Globs `benchmarks/_parts/2.*` (prefix-filtered — all three workflows share
`_parts`), concatenates the per-job TSVs into one Markdown table with a `rule`
column and a `TOTAL` row (sum for time/IO/CPU, peak for memory, mean for load),
then **deletes the merged parts** and prunes empty part dirs.

- **In (barriers only):** `{CPD}/summary/annual_change_scalar_stats_summary.nc`,
  `{CPD}/plots/projected_climate_statistics.png`,
  `{CPD}/plots/precipitation_anomaly_projections_abs.png`.
- **Params:** `parts_dir = {PD}/benchmarks/_parts`, `workflow_num = 2`.
- **Out:** `{PD}/benchmarks/wf2_benchmarks.md` (Markdown).
- **Consumed by:** `all`.

---

## 3. DAG shape

```
config_path ─► 2.01 copy_config ─────────────────────────────────► all
                                                                    ▲
region.geojson (WF1, ancient)                                       │
   │                                                                │
   ├─► 2.02 monthly_stats_hist  {model}         ──┬──► 2.07 ────────┤
   │        │  historical_stats_time_{model}.nc  │                  │
   │        │ (ordering edge only)               │                  │
   │        ▼                                    │                  │
   └─► 2.03 monthly_stats_fut  {model,scenario} ─┼──► 2.08 ─────────┤
            │  stats_time-{model}_{scenario}.nc  │                  │
            ▼                                    │                  │
        2.04 monthly_change {model,scen,horizon}─┴──► 2.09 ─────────┤
            │  annual_change_scalar_stats-*.nc                      │
            ▼                                                       │
        2.05 monthly_change_scalar_merge ──► summary/*.nc,csv ──────┤
            │                             └─► plots/projected_*.png │
            ▼ (ordering edge; file unread)                          │
        2.06 plot_climate_proj_timeseries ─► plots/*.png, ──────────┤
            │                                timeseries/*.nc        │
            ▼                                                       │
        2.10 gather_benchmarks ─► benchmarks/wf2_benchmarks.md ─────┘
```

Effective parallel width on the seed config: 3 (2.02) → 6 (2.03) → 6 (2.04),
then everything downstream is single-job.

**`temp()` lifetime.** All three intermediate netCDF families are `temp(...)`,
so Snakemake deletes each only after *every* declared consumer has finished.
For `historical_stats_time_{model}.nc` that means 2.03, 2.04, 2.06 **and** 2.07;
for `stats_time-*.nc`, 2.04, 2.06 and 2.08. Because 2.06 sits at the very end of
the DAG, the full set of monthly series stays on disk for the whole run.

**`ruleorder`.** `monthly_stats_hist > monthly_stats_fut > monthly_change >
monthly_change_scalar_merge` is retained as stale insurance, not confirmed
load-bearing; a 2026-07 dry-run showed it constrains nothing on the tests
fixture. Status and removal conditions: `AGENTS.md` § Conventions and
`dev/r04/climate-projections-design.md` §3.

---

## 4. Observations relevant to efficiency / modularity

Findings from the code as it stands — recorded to think against, not proposals.

1. **2.03's dependency on 2.02's output is ordering-only.** The `__main__` block
   of `get_stats_climate_proj.py` reads only `input.region_path`; the historical
   netCDF is never opened by the future job. The edge serializes hist→fut across
   the entire fan-out for no data reason. *Hypothesis to test, not an established
   cause:* the script does `os.mkdir(folder_out)` guarded by `os.path.exists`
   (lines 179–180) rather than `makedirs(exist_ok=True)`, so concurrent hist/fut
   jobs would race on creating `{CPD}`.

2. **2.02 and 2.03 are the same script with different params.** The only
   structural difference is the output naming, which is inconsistent between
   them: `historical_stats_time_{model}.nc` (underscore separator, no scenario)
   vs `stats_time-{model}_{scenario}.nc` (dash separator). Any consolidation into
   a single `{model}×{scenario}` rule has to reconcile that naming first — the
   inconsistency is what forces two rules, not the computation.

3. **`members` is looped inside the script, not a wildcard.** The per-member
   remote reads (the dominant cost — remote GCS/zarr fetch + `.load()`) run
   serially within one job and cannot be scheduled by Snakemake. With the seed's
   single member this is invisible; with several it caps parallelism.

4. **`ancient()` on intra-workflow temps.** 2.04 marks both of its stats inputs
   `ancient`, and 2.05 marks its whole `expand()` `ancient`. Unlike the
   documented `region.geojson` case (a deliberate cross-workflow staleness
   exemption), these suppress rebuild propagation *inside* WF2: regenerated stats
   will not retrigger change factors, and regenerated change factors will not
   retrigger the merge. Combined with `temp()`, an incremental re-run is
   effectively an all-or-nothing re-run.

5. **A whole file-dependency layer sits outside the DAG when `save_grids: true`.**
   The grid netCDFs (`historical_stats_{model}.nc`, `stats-{model}_{scenario}.nc`,
   `monthly_change_mean_grid-*.nc`) are written by the scripts but declared
   nowhere; 2.04 and 2.06 receive their paths through `params`. Snakemake neither
   schedules, tracks, nor cleans them, and a missing one fails at runtime rather
   than at DAG build.

6. **The three `gather_*_logs` rules are identical modulo wildcards** — same
   script, same shape, differing only in which `expand()` feeds the barrier and
   which `_parts` dir feeds `params.parts`.

7. **Gather rules are invisible in the cost picture.** 2.07–2.09 declare neither
   `log:` nor `benchmark:`, so their contribution never reaches
   `wf2_benchmarks.md`. 2.10 deletes the parts it merged, so a later partial
   re-run produces a table covering only the rules that re-ran.

8. **`{model}` contains a `/`** (e.g. `NOAA-GFDL/GFDL-ESM4`), so the vendor
   becomes a real path segment in every output filename, log part, and benchmark
   part. Relevant to any change touching path templates or wildcard constraints.

9. **2.06 is a monolith.** One script does: reopen all series → anomaly
   statistics → write the merged timeseries netCDF → render 8 PNGs → optionally
   render the gridded maps. It declares 3 of those artifacts, and it is the last
   consumer of the `temp()` series (see §3). It is also the natural split point
   if figure generation should become independently re-runnable.

10. **Undeclared / mislabeled outputs.** `gcm_timeseries.nc` is declared under
    the label `timeseries_csv`; the precip anomaly figure is saved without an
    extension (`plots/precipitation_anomaly_projections_{n}`, matplotlib appends
    `.png`), while its temp counterpart passes `.png` explicitly.

11. **Doc discrepancy (not fixed here):** `dev/workflows/climate_projections.md`
    lists the config snapshot at `{PD}/config/snake_config_climate_projections.yml`;
    the Snakefile writes `{PD}/config/runs/snake_config_climate_projections.yml`.
    The Snakefile is authoritative.
