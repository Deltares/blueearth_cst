# Migration — workflow 2 (R8, `v2.0` of the climate-projections workflow)

Workflow 2 was restructured in milestone R8. **Three config changes are breaking**
and fail loudly at DAG build; everything else is additive or a value change you
should know about.

Design rationale: `dev/reference/workflows/wf2-climate-analysis-v2-design.md`.
Step-by-step evidence: the falsifier notes under `dev/milestones/r08/`.

---

## Breaking config changes

### 1. `variables` is a mapping, not a list

```yaml
# before
variables: [precip, temp]

# after
variables:
  precip: {source: precip, canonical: rate,  units: mm/day, change: relative}
  temp:   {source: temp,   canonical: state, units: degC,   change: absolute}
```

`canonical` says whether the stored monthly series is a *rate* or a *state* — it
drives the annual aggregation. `change` says whether a change factor is a ratio or
a difference — it drives the arithmetic. Previously **both** were inferred from
the literal name `"precip"`, so any other relative variable was silently
differenced as if it were a temperature.

A `change: relative` variable outside the shipped set must also declare a
near-zero threshold (see the `relative_change` keys below).

### 2. `save_grids` / `save_gridded` are removed

```yaml
# delete this line, whichever spelling you have
save_gridded: false
```

The key was renamed `save_grids` → `save_gridded` during this milestone and then
removed outright; if you are migrating from before R8, skip the rename and just
delete the line. The gridded outputs are gone. `raw/{series_key}.nc` **is** the basin slice on the
source grid and is written on every run, so `grids/series/` would have been a
near-duplicate of a file you already have; `grids/change/` was never declared to
Snakemake by any rule.

`save_gridded: true` (or the older `save_grids: true`) now **raises** — it asks
for an output that no longer exists. `false` is accepted with a warning, because
it requests exactly what the workflow now always does; delete the key at your
convenience.

### 3. A model absent from the catalog now fails at DAG build

The CMIP6 catalog is generated from a live crawl of the store, so a model name
absent from it is absent from the store. That is a typo or a stale config, not
thin data, and it stops the run naming the model.

A model that is *present* but does not publish a requested scenario or member is
**not** an error: it is recorded in `composition.csv` with a status and the run
continues. See "New outputs" below.

---

## New config keys (all optional)

| Key | Default | What it does |
| --- | --- | --- |
| `stats` | `[mean, median, std]` | The statistic set. Tail quantiles are opt-in and, when emitted, labelled with their effective sample size (`q_90[n=21]`). |
| `relative_change.min_reference` | `{precip: 0.1}` mm/day | Below this, a month's *relative* change is undefined and is reported as `NaN` with `status = reference_below_threshold`; its absolute change is kept. |
| `relative_change.max_flagged_months` | `3` | More than this many flagged months flags the whole combination in the report. |

---

## Values changed

If you compare results across this milestone, expect differences from four
deliberate changes. Each landed as its own commit with its own gate.

| Change | Effect |
| --- | --- |
| **Spherical cell-area weighting** (5a) | The basin mean weights each cell by its true area. **No effect on an equatorial, latitude-symmetric basin** — where the weights are exactly uniform — and a growing effect with latitude. |
| **Calendar-aware month weighting** (5b) | Annual aggregates weight each month by its length *in the model's own calendar*. Zero effect on a `360_day` model; real on `noleap` and standard. |
| **Rounding dropped** (5c) | Stage A no longer rounds to 2 decimals — that was a 0.005 mm/day floor on every monthly value. Distinct from the 3 dp the summary CSVs are *written* at (see "Number formatting in the CSVs"): that is presentation, this was the stored series. |
| **Reference window off-by-one fixed** | A configured `[1990, 2010]` now yields **21** complete hydrological years, not 20. The final complete year was being discarded. This is the largest of the four. |

Two provenance corrections also landed: series now record the model's **true
calendar** (previously every series claimed `proleptic_gregorian`, which is false
for every `noleap` and `360_day` model), and the effective reference window is
reported consistently everywhere it appears.

---

## New outputs

| Path | What it is |
| --- | --- |
| `summary/cmip6_change_factors_annual.csv` | Long format: one row per (model, scenario, member, horizon, variable, statistic). Schema below. |
| `summary/cmip6_change_factors_monthly.csv` | The same, per calendar month — the seasonal shift an annual figure averages away. |
| `summary/composition.csv` | Every **requested** combination and how it resolved, including the ones that do not exist in the store. |
| `summary/provenance.json` | Sources with verified physical store paths, digests, windows, settings — enough to reconstruct the run. |
| `report.md` | The run with a disclaimer block: window clipping, alignment, weighting scheme and its approximation, the dry-month rule, catalog snapshot date, unresolved combinations. |

The wide `summary/annual_change_scalar_stats_summary*` files are **no longer
produced** — see "Rebuilt tables" below.

## Renamed paths

| Before | After | Why |
| --- | --- | --- |
| `series/{key}.nc` | `scalar/{key}.nc` | `series` said nothing about the files being spatially averaged. `scalar` is the word this codebase already uses for the quantity (`var_m_scalar` in the reducer), on the axis it already asserts — scalar vs grid. |
| `change_factors/annual.csv` | `summary/cmip6_change_factors_annual.csv` | Every result now lives under `summary/`, and the name identifies the file when it is detached from the tree. |
| `change_factors/monthly.csv` | `summary/cmip6_change_factors_monthly.csv` | |
| `provenance.json` | `summary/provenance.json` | Beside `composition.csv` — both are run-level records rather than results. `report.md` stays at the root as the single entry point. |
| `projected_climate_statistics.png` | `plots/cmip6_change_factor_cloud.png` | It is the ΔT/ΔP cloud, one point per combination. |
| `{precipitation,temperature}_{anomaly,monthly}_projections_{abs,anom}.png` | `plots/cmip6_{precip,temp}_{annual,monthly}_{absolute,change}.png` | The old names contradicted their contents — `precipitation_anomaly_projections_abs.png` plots absolute levels, so "anomaly" sat in the filename of the non-anomaly figure. |

The figure scheme is `{clim_project}_{variable}_{view}_{quantity}`, using the same
`precip`/`temp` names as the config and the tables, and the same
`absolute`/`change` distinction the tables draw.

`raw/` is unchanged, and **filenames are identical across both tiers**: the
directory carries the tier, the filename carries the identity. `grids/series/`
also keeps its name — it is the *gridded* counterpart, so `grids/scalar/` would
be a contradiction.

An existing project directory strands its old `series/` folder, since Snakemake
cannot clean a path it no longer declares. `dev/scripts/prune_series_cache.py`
now reports it as a legacy generation; delete it once (see "Post-migration
cleanup" below).

## Rebuilt tables

`summary/cmip6_change_factors_{annual,monthly}.csv` replace the long tables *and*
the three wide `summary/annual_change_scalar_stats_summary{,_mean}.{nc,csv}`
files, which are **no longer written**. Nothing outside the workflow read them.

Twenty columns became fourteen (fifteen for monthly, which adds `month`):

```
model,scenario,member,horizon,variable,statistic,
reference_value,absolute_value,units,relative_value,relative_units,
status,reference_window,horizon_window
```

- **`reference_value`** is the **baseline level** — e.g. `25.057` degC — in `units`.
- **`absolute_value`** is the **future level** — e.g. `26.235` degC — in `units`.
- **`relative_value`** is the change **against the reference window**, in
  `relative_units`: a difference for a variable declared `change: absolute`
  (`+1.179` degC), a percent for one declared `change: relative` (`+10.950`).
- **`reference_window`** and **`horizon_window`** are both the **effective**
  bounds — the complete hydrological years the arithmetic actually used, from
  `hydrological_year_bounds` — in one form:
  `1990-01-01 / 2010-12-01` and `2070-01-01 / 2090-12-01`.

  `horizon_window` previously reported the config's **nominal** year pair
  (`2070-2090`) beside an effective reference span, so two adjacent columns
  disagreed about both meaning and format. The effective horizon bounds were being
  computed all along (`get_change_annual_clim_proj` needs them to aggregate) and
  then discarded. If you parse this column, the value shape changed. The nominal
  horizons are unchanged in `report.md` and `provenance.json`, which is where the
  config echo belongs.

### Number formatting in the CSVs

Every float in every WF2 CSV is written **fixed to 3 decimal places**; integer
columns (`month`, `n_reference_years`) stay integral and text is untouched. This
replaces the full float64 representation, which ran to 17 significant digits
(`9.331056256303583`).

Two things it fixes. Excel prompted *"By default, Excel will perform the
following data conversion"* every time one of these files was opened — the long
floats were the only conversion trigger the files carried. And fixed-point never
emits an exponent, so a near-zero ratio can no longer arrive as `1.2345e-06`, the
form Excel converts most eagerly.

**This does not undo "Rounding dropped (5c)" above.** 5c removed quantisation from
the **stored** monthly series (`scalar/*.nc`), because that fed every downstream
calculation and put a 0.005 mm/day floor under it. This rounds only the
**presentation** CSV, at write time. The netCDFs keep full precision, and nothing
in the workflow reads these CSVs back (`plot_climate_proj_timeseries` declares the
annual table as an ordering edge and never opens it).

3 dp is 100× finer than the default `relative_change.min_reference` of
0.1 mm/day, so it cannot interact with the dry-month threshold. If you declare a
variable whose units make its values small in absolute terms, that is the case to
revisit — `CSV_DECIMALS` in `blueearth_cst/projections/change_factor_table.py` is
the single knob.

**Two corrections you should know about, because they change numbers you may
already have:**

1. The old `units` column was **wrong for every relative variable** — it reported
   the underlying variable's units (`mm/day`) beside a value that was a percent.
2. Model names longer than the first-merged one were **silently truncated** in the
   wide summary — `NOAA-GFDL/GFDL-ESM4` became `NOAA-GFDL/GFD`. The old `dataset`
   column carried the truncated name. Fixed; `model` now reports the full
   `source_id`.

Both levels are shipped so that every number in a row is recoverable from that
row: the change is `absolute_value - reference_value` (or that over
`reference_value`, as a percent). This also keeps the dry-month rule exact — a
flagged month drops the meaningless ratio and still carries the informative
difference.

`composition.csv` drops from 15 columns to 10 on the same principles: `model`
replaces `dataset`/`institution`/`source_id`, and the constant `catalog_crawled_on`
and reference-window columns move to the artifacts that own them.

**Precipitation is now reported in mm/day everywhere, figures included.** The
annual precipitation figure previously plotted mm/year, so it disagreed with every
table by a factor of 365.

## Removed output

`timeseries/gcm_timeseries.nc` is **no longer written**. A project directory from
an earlier run keeps its stale copy — Snakemake cannot clean an output no longer
declared — so delete `climate_projections/<proj>/timeseries/` by hand once; fresh
runs never create it. It merged the nine `scalar/*.nc` into one cube that
nothing consumed, while rounding to 2 decimals — re-imposing the quantisation
"Rounding dropped" above removes — and stripping every `cst_*` attribute, so it
carried no digest, region fingerprint or calendar and could not be traced.

If you were reading it, use `scalar/*.nc` for the full monthly timeseries (same
values, unrounded, with provenance) or the change-factor tables above.

## Fixed: the `units` attribute on the netCDFs

`raw/*.nc` and `scalar/*.nc` used to carry `units = "kg m-2 s-1"` on `precip` and
`"K"` on `temp` while the values were already mm/day and °C — the catalog's
`data_adapter` converts on read and does not rewrite the attribute. Both tiers now
carry the units declared in your `variables:` block. A `raw/` slice cached before
this change is repaired in place the next time the run touches it; no re-download.

## Post-migration cleanup

Snakemake cannot clean an output it no longer declares, so a project directory
from an earlier run keeps every superseded path. Delete these once, per project:

```
climate_projections/<proj>/series/          # renamed to scalar/
climate_projections/<proj>/timeseries/      # removed entirely
climate_projections/<proj>/change_factors/  # moved into summary/
climate_projections/<proj>/provenance.json  # moved into summary/
climate_projections/<proj>/summary/annual_change_scalar_stats_summary*
climate_projections/<proj>/plots/{precipitation,temperature}_*_projections_*.png
climate_projections/<proj>/plots/projected_climate_statistics.png
climate_projections/<proj>/grids/                # if you ever set save_gridded: true
logs/2.0{2,3,4,5}_{monthly_stats_hist,monthly_stats_fut,monthly_change,monthly_change_scalar_merge}.log
logs/2.0{1,2,4,6}_*.log, logs/2.11_extract_climate_grid.log   # see "One log per workflow"
logs/_parts/2.0{2,3,4}_{monthly_stats_hist,monthly_stats_fut,monthly_change}/
```

`dev/scripts/prune_series_cache.py` reports the stale `series/` generation;
the rest are a manual delete. Do this **before** recording any reference
snapshot, or the snapshot bakes in files the workflow no longer produces.

---

## One log per workflow (all three workflows)

Every workflow used to scatter its run across many log files — WF2 left five in
`logs/`, WF1 thirteen, and WF3 wrote hundreds (3.05/3.07/3.09 log one file per
`(realization, stress-test)` combination and 3.10 one per batch). Following a
single run meant opening all of them and knowing the order to read them in.

Each workflow now writes exactly one log:

| Workflow | Merged log | Gather rule |
| --- | --- | --- |
| `Snakefile_model_creation` | `logs/wf1_model_creation.log` | 1.16 `gather_logs` |
| `Snakefile_climate_projections` | `logs/wf2_climate_projections.log` | 2.07 `gather_logs` |
| `Snakefile_climate_experiment` | `<exp_dir>/logs/wf3_climate_experiment.log` | 3.13 `gather_logs` |

Every rule logs into `logs/_parts/`, the gather rule merges the parts, then
deletes them and prunes the emptied directories — so after a clean full run there
is no `logs/_parts/` either. Each merged file carries **one** provenance header at
the top (the per-part `# BlueEarth-CST | project: … | log: … | started …` blocks
are stripped), then one section per rule:

```
================================================================================
== 2.01  fetch_gcm_raw
================================================================================

-- cmip6_INM_INM-CM4-8_historical_r1i1p1f1 -------------------------------------
19:39:26 - fetch - INFO - raw cache_hit digest=b352f7bd93d0 (…)
```

Sections are in rule-number order (matching the workflow's `benchmarks/*.md`), and
a fan-out rule gets one `--` sub-header per member — the CMIP6 series above, or
WF3's `rlz_2_cst_1` / `batch_0`. Members sort naturally, so `rlz_2` precedes
`rlz_10`.

Two behaviours to know:

- After a **partial** re-run only the rules that actually re-ran have parts, so
  the rewritten log marks the others
  `# (no part from this run — rule was already up to date)`. The log describes the
  run that produced it, not an accumulated history.
- `extract_climate_grid` is one rule shared by all three workflows (1.10 = 2.11 =
  3.02). Run in order, WF1 builds the shared climate store and the others find it
  current, so that section normally carries the same marker in WF2's and WF3's
  logs — the work is recorded in `wf1_model_creation.log`, not missing.

Snakemake cannot clean logs it no longer declares, so an existing project keeps
the old per-rule files. Delete them once (they are listed under "Post-migration
cleanup" above for WF2; for WF1 and WF3 remove `logs/1.*.log` and
`<exp_dir>/logs/3.*` respectively).

---

## Figures

The anomaly and monthly projection figures no longer show a **multi-model median**
or a **5–95 % envelope**. They show **one labelled trace per (model, scenario,
member)**, because under this design each combination is one data point and
nothing is averaged across them. If you want an ensemble summary, it is a
downstream analysis over `change_factors/*` — deliberately not computed here.

---

## Recommended reference window

`snake_config.template.yml` now recommends **1985–2014**: thirty years ending at
the last year the CMIP6 historical experiment covers. The range is inclusive, and
with the default `start_month_hyd_year: Jan` that is thirty complete hydrological
years. Any other start month yields 29, with the partial years at both ends
dropped — every artifact reports the effective window and count beside the
nominal one, so the difference is never silent.

Test fixtures keep `[1990, 2010]` deliberately, so the recommendation change moves
no test number.
