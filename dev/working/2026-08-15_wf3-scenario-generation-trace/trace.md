# WF3 scenario generation — config to `rlz_1_st_4.nc`, and where the time goes

Status: WORKING NOTE, written 2026-08-15 to support a rework of the execution
mechanics for efficiency. Descriptive of the tree at `e197502`; it makes no
proposal and takes no decision.

Author: Claude, 2026-08-15.

**Read this with** `dev/tasks/t2608082036-re-derive-the-wf3-v2-execution-design-against-the-post-r11-tree.md`
(R12's open design item, queue 1) and
`dev/tasks/t2608071216-make-the-wf3-batch-size.md`. R12 is the milestone whose
whole remit is *how WF3 executes* (`dev/roadmap.md` § Phase 8), so anything here
that becomes a decision belongs there rather than in this file.

---

## 1. The chain, step by step

Every path is relative to `project_dir`. `<exp>` is
`experiments/<experiment_name>`, `<wg>` is `<exp>/climate/weathergenr`.
Numbers in § 3 come from a real `snake_config_rapid.yml` run recorded in
`test_case/test_rapid/benchmarks/wf3_benchmarks_experiment_rapid.md`.

### Step 0 — what the config declares

`workflows.run_stress_test.stress_test` in the project config:

```yaml
stress_test:
  temp:
    step_num: 1
    mean: {min: [0.0 ×12], max: [3.0 ×12]}
  precip:
    step_num: 1
    mean:     {min: [0.7 ×12], max: [1.3 ×12]}
    variance: {min: [1.0 ×12], max: [1.0 ×12]}
  dry_spell_factor: [1.0 ×12]
  wet_spell_factor: [1.0 ×12]
```

Twelve values per entry because every axis is **monthly**. `step_num: 1` means
one step *between* the endpoints, so each axis yields `step_num + 1` = 2 levels.
The spell factors sit beside `temp:`/`precip:` rather than under them because
they are **not axes**: they add no grid member and no design-table column.

### 3.09 `prepare_stress_test_grid` — expand the envelopes into a grid

| | |
|---|---|
| script | `blueearth_cst/experiment/prepare_cst_parameters.py` |
| in | the config YAML; `<exp>/.project_consistency_ok` |
| out | `<wg>/_work/st_1.csv` … `st_<ST_NUM>.csv`, `<exp>/config/stress_test_design.csv` |

`stress_test_grid()` takes the cross product of the per-axis levels →
`ST_NUM` members. Each member is written as **twelve monthly rows**:

```
month,temp_mean,precip_mean,precip_variance      # st_4.csv
1,3.0,1.3,1.0
…                                                # all twelve months
```

**Two artifacts, deliberately.** The per-member CSV is what the generator
consumes and carries no id; `stress_test_design.csv` is what answers *"what is
run 4?"*:

```
st_id,temp_change,precip_change,precip_variance_change
0,0.0,0.0,0.0          ← st_0, the reserved unperturbed baseline
1,0.0,-30.0,0.0
2,0.0,30.0,0.0
3,3.0,-30.0,0.0
4,3.0,30.0,0.0
```

Both are written by the SAME loop, so the enumeration that names the members
and the one that describes them cannot disagree.

**Unit change to know:** precipitation is a **multiplier** in the member file
(`1.3`) and a **percent** in the design table (`+30.0`). Temperature is additive
degC in both.

`st_0` is not in the `_work/` set — it is the unperturbed baseline and is
produced by 3.11, not by perturbation.

### 3.10 `prepare_weathergen_config`

| | |
|---|---|
| script | `blueearth_cst/experiment/prepare_weagen_config.py` |
| out | `<wg>/config/weathergen_config.yml` |

One shared config for the generator: series length (anchored at 2010 and running
to `horizontime_climate` + `run_length`/2), water year, the spell factors, and
the `transient_change` flags. C29 retired the per-member variant of this file.

### 3.11 `generate_weather_realizations` — the stochastic baseline

| | |
|---|---|
| script | `blueearth_cst/weathergen/generate_weather.R` (weathergenr, R) |
| in | `data/climate/historical/<key>/extract_historical.nc`, `basin_cells.csv` (both `ancient`), the weathergen config |
| out | `<wg>/output/rlz_<n>_st_0.nc` for n in 1..`RLZ_NUM` — **`temp()`** |

weathergenr resamples the historical record into `RLZ_NUM` synthetic series that
are statistically like the observed climate without repeating it. **One job
produces all realizations**, not one job per realization.

### 3.12 `perturb_climate_realization` — where `rlz_1_st_4.nc` is created

| | |
|---|---|
| script | `blueearth_cst/weathergen/impose_climate_change.R` (R) |
| in | `<wg>/output/rlz_{rlz}_st_0.nc`, `<wg>/_work/st_{st}.csv`, the weathergen config |
| out | `<wg>/output/rlz_{rlz}_st_{st}.nc` — **`temp()`** |

One job per (realization × member). It applies the twelve monthly rows to the
baseline series: temperature **additively**, precipitation **multiplicatively**,
variance scaled separately. With `transient_change: true` the perturbation ramps
across the series instead of arriving as a step.

**The fan-out starts here:** `RLZ_NUM × (ST_NUM)` perturbed files, plus the
`RLZ_NUM` baselines from 3.11.

`wildcard_constraints: st_num` is restricted to ≥ 1 on this rule. Without it the
rule is a second eligible producer of `st_0.nc` — which is also its own input —
and the DAG self-loops (`CyclicGraphException`).

### 3.13 `write_climate_data_catalog`

| | |
|---|---|
| in | the scenario `.nc` set |
| out | `<exp>/config/catalogs/data_catalog_run_stress_test.yml` |

A hydromt catalog with one entry per member, so 3.14 can address them by name.

### 3.14 `downscale_climate_realization` — onto the model grid

| | |
|---|---|
| in | `rlz_{rlz}_st_{st}.nc`, the catalog, the built wflow model |
| out | `<exp>/hydrology/wflow/forcing/inmaps_rlz_{rlz}_st_{st}.nc` (**`temp()`**), `<exp>/hydrology/wflow/config/rlz_{rlz}_st_{st}.toml` |

Regrids each scenario from the climate grid to the wflow grid and writes the
per-member run TOML. One job per member, including `st_0`.

### 3.15 `run_wflow`

| | |
|---|---|
| in | the forcing NC + TOML per member |
| out | `<exp>/hydrology/wflow/output/rlz_{rlz}_st_{st}.csv`; `outstates_*.nc` (**`temp()`**) |

**Batched:** members are grouped into batches of `batch_size` and one Julia
process runs each batch, so the rule identifiers are `run_wflow_batch_<b>` and
the log/benchmark files are keyed by batch id, not by member. `batch_size`
defaults to `ceil(members / cores)` clamped to `batch_size_max` (8);
`batch_size: 1` restores one job per member.

### 3.16 `derive_wflow_indicators`

| | |
|---|---|
| in | every per-member run CSV |
| out | `<exp>/results/<token>_indicators.csv`, one per `wflow_outvars` entry |

Reduces the runs to the response surface. `temp_change` and `precip_change`
become **columns**, which is how every point on the surface traces back to one
row of `stress_test_design.csv`.

---

## 2. What survives a run

`temp()` covers the whole scenario chain: the baselines (3.11), the perturbed
scenarios (3.12), the downscaled forcing (3.14) and the warm states (3.15). A
completed run therefore leaves `<wg>/output/` **empty** — which is why
`test_case/test_rapid` shows no `.nc` there, and why `--notemp` exists.

What persists: `stress_test_design.csv`, the per-member `_work/st_*.csv`, the
run TOMLs, the per-member run CSVs, and the indicator tables.

---

## 3. Where the time actually goes

From a real `snake_config_rapid.yml` run — `RLZ_NUM` = 2, `ST_NUM` = 4 (+ `st_0`),
so 10 members. Source: `test_case/test_rapid/benchmarks/wf3_benchmarks_experiment_rapid.md`.

**Every rule, in execution order.** Rule numbers are also dependency order since
[R10-5], so reading down the table is reading the run. `jobs` is how many times
the rule executed; `peak RSS` is the largest single job's resident memory.

| rule | jobs | s/job | total s | share | peak RSS | load |
|---|---:|---:|---:|---:|---:|---:|
| 3.01 `check_project_consistency` | 1 | 0.44 | 0.4 | — | 6 MB | — |
| 3.02 `snapshot_config` | 1 | — | — | — | — | *no benchmark declared* |
| 3.03 `delineate_region` | 0 | — | — | — | — | *up to date from WF1* |
| 3.04 `delineate_spatial_units` | 0 | — | — | — | — | *up to date from WF1* |
| 3.05 `write_model_reference` | 1 | 0.44 | 0.4 | — | 3 MB | — |
| 3.06 `check_model_reference` | 1 | 0.57 | 0.6 | — | 43 MB | — |
| 3.07 `write_experiment_config` | 1 | 0.56 | 0.6 | — | 40 MB | — |
| 3.08 `extract_historical_climate` | 0 | — | — | — | — | *up to date from WF1* |
| 3.09 `prepare_stress_test_grid` | 1 | 2.62 | 2.6 | 0.2% | 96 MB | 26% |
| 3.10 `prepare_weathergen_config` | 1 | 0.51 | 0.5 | — | 5 MB | — |
| 3.11 `generate_weather_realizations` | 1 | 34.65 | **34.7** | 2.6% | 457 MB | 41% |
| 3.12 `perturb_climate_realization` | 8 | 18.9–19.2 | **152.2** | **11.5%** | 407 MB | ~92% |
| 3.13 `write_climate_data_catalog` | 1 | 9.23 | 9.2 | 0.7% | 319 MB | 57% |
| 3.14 `downscale_climate_realization` | 10 | 27.6–31.2 | **304.9** | **23.0%** | 303 MB | ~45% |
| 3.15 `run_wflow` | 5 batches | 148.6–166.8 | **813.3** | **61.2%** | 912 MB | ~82% |
| 3.16 `derive_wflow_indicators` | 1 | 9.11 | 9.1 | 0.7% | 257 MB | 55% |
| 3.16b `write_run_metadata` | 1 | — | — | — | — | *no benchmark declared* |
| 3.17 `gather_benchmarks` | 1 | — | — | — | — | *no benchmark declared* |
| 3.18 `gather_logs` | 1 | — | — | — | — | *no benchmark declared* |
| | | | **~1 329** | | | |

Wall-clock is lower than the sum because `-c 3` runs jobs concurrently; the sum
is the work, not the elapsed time.

**Three rules did not execute at all** — 3.03, 3.04 and 3.08 are the shared
producers, already built by WF1 and found up to date. In a WF3-only run against a
fresh project they would execute, and 3.08 in particular is a multi-decade
extraction. The gathers and `snapshot_config` declare no `benchmark:`, so their
absence from the table is a declaration gap, not a zero.

**Four observations, stated as facts rather than proposals:**

1. **The model run dominates, but not overwhelmingly.** 3.15 is 61%. The
   remaining 39% is scenario *preparation*, and 34.5% of the total (3.12 + 3.14)
   is spent producing files that are then deleted by `temp()`.

2. **3.12 and 3.14 are two full passes over the same per-member data.** Each
   opens a scenario netCDF, transforms it and writes another netCDF: ~19 s then
   ~30 s per member, plus the intermediate file in between. They are separate
   rules for good reasons — different languages (R, then Python/hydromt) and
   different invalidation causes — but the cost is a second serialization
   round-trip per member that nothing downstream reads.

3. **Memory, not time, is what bounds batching.** A `run_wflow` batch peaks near
   **912 MB** while holding its members' forcing and warm states, which is
   exactly why `batch_size` is clamped rather than set to `ceil(K/N)` — and it is
   the constraint `t2608071216` is about. The forcing files an on-the-fly
   approach would eliminate are a large part of that residency.

4. **Everything scales linearly with `RLZ_NUM × ST_NUM`** except 3.11, 3.13 and
   3.16. Raising both axes to `step_num: 3` gives 16 + 1 = 17 members ⇒ 34 members
   at `RLZ_NUM` 2, i.e. 3.4× the members and ~3.4× the 96% of cost that is
   per-member.

## 4. The prior art an efficiency rework should not rediscover

**The `fao` branch applies change factors ON THE FLY, at each timestep, and
materialises no perturbed forcing at all.** That is recorded in
`dev/reviews/2026-08-13_fao-branch-assessment.md` §4.2 as one of the two ideas
worth keeping regardless of the code, and the assessment explicitly flags that
it is *the opposite* of what WF3 does — WF3 writes per-member netCDFs and wraps
them in `temp()` — with the note that "if the on-the-fly approach is adopted
there, someone will reasonably ask why WF3 does not do the same."

This trace is the answer to how much that would be worth: the materialisation it
would remove is 3.12 + 3.14, which is **34% of WF3's job time** on the rapid
config, plus the disk that forces `batch_size` to be clamped in the first place.

Two cautions from the same assessment, both load-bearing:

- The `fao` Julia driver `run_wflow_change_factors.jl` **does not run on our
  stack**. §4.1 checked it symbol by symbol against Wflow.jl 1.0.2: every symbol
  it reaches for was renamed or removed. The mechanism is a good specification;
  the file is not a port.
- Wflow v1 offers **no native hook** either. Its `scale`/`offset` config keys
  apply only to *fixed* forcing values (`io.jl:83`, inside
  `load_fixed_forcing!`); `update_forcing!` applies no scaling to dynamic netCDF
  forcing. So on-the-fly perturbation means a custom driver, whichever way it is
  written.

**Also already tracked, do not re-open cold:**
`t2608071216` — the `batch_size` default is not disk-aware, which is the
constraint that currently ties batching to the size of the very files an
on-the-fly approach would eliminate.

---

## 5. What this note deliberately does NOT do

No proposal, no benchmark of an alternative, no decision. R12 owns *how WF3
executes* and `t2608082036` is its open design item; a change to these mechanics
should be designed there, with this trace as the cost baseline it argues
against.

One measurement worth taking before any design: these numbers are from the
**rapid** config on a 384-cell fixture. The 3.15 share almost certainly rises on
a production basin, because wflow time scales with cell count while the
per-member netCDF handling in 3.12/3.14 scales with the time axis. If the
production ratio is 85/15 rather than 61/39, removing materialisation buys much
less than it does here.
