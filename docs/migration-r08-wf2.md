# Migration — workflow 2 (R8, `v2.0` of the climate-projections workflow)

Workflow 2 was restructured in milestone R8. **Three config changes are breaking**
and fail loudly at DAG build; everything else is additive or a value change you
should know about.

Design rationale: `dev/workflows/wf2-climate-analysis-v2-design.md`.
Step-by-step evidence: the falsifier notes under `dev/working/2026-07-3*`.

---

## Breaking config changes

### 1. `save_grids` → `save_gridded`

```yaml
# before
save_grids: false
# after
save_gridded: false
```

The old key **raises** rather than being ignored. Ignoring it would silently give
a user who set `save_grids: true` the `false` behaviour with no signal.

### 2. `variables` is a mapping, not a list

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
near-zero threshold (see 4 below).

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
| **Rounding dropped** (5c) | Stage A no longer rounds to 2 decimals — that was a 0.005 mm/day floor on every monthly value. |
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
| `grids/series/`, `grids/change/` | The gridded products, when `save_gridded: true`. Same key as their reduced counterparts. |

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

- **`reference_value`** is the **baseline level** — e.g. `25.0567` degC — in `units`.
- **`absolute_value`** is the **future level** — e.g. `26.2354` degC — in `units`.
- **`relative_value`** is the change **against the reference window**, in
  `relative_units`: a difference for a variable declared `change: absolute`
  (`+1.1787` degC), a percent for one declared `change: relative` (`+10.95`).

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
logs/2.0{2,3,4,5}_{monthly_stats_hist,monthly_stats_fut,monthly_change,monthly_change_scalar_merge}.log
```

`dev/scripts/prune_series_cache.py` reports the stale `series/` generation;
the rest are a manual delete. Do this **before** recording any reference
snapshot, or the snapshot bakes in files the workflow no longer produces.

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
