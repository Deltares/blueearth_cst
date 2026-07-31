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
| `change_factors/annual.csv` | Long format: one row per (dataset, scenario, member, horizon, variable, statistic). |
| `change_factors/monthly.csv` | The same, per calendar month — the seasonal shift an annual figure averages away. |
| `summary/composition.csv` | Every **requested** combination and how it resolved, including the ones that do not exist in the store. |
| `provenance.json` | Sources with verified physical store paths, digests, windows, settings — enough to reconstruct the run. |
| `report.md` | The run with a disclaimer block: window clipping, alignment, weighting scheme and its approximation, the dry-month rule, catalog snapshot date, unresolved combinations. |
| `grids/series/`, `grids/change/` | The gridded products, when `save_gridded: true`. Same key as their reduced counterparts. |

The wide `summary/annual_change_scalar_stats_summary*` files are unchanged and
still produced.

## Renamed paths

| Before | After | Why |
| --- | --- | --- |
| `series/{key}.nc` | `scalar/{key}.nc` | `series` said nothing about the files being spatially averaged. `scalar` is the word this codebase already uses for the quantity (`var_m_scalar` in the reducer; `annual_change_scalar_stats_summary*`), on the axis it already asserts — scalar vs grid. |

`raw/` is unchanged, and **filenames are identical across both tiers**: the
directory carries the tier, the filename carries the identity. `grids/series/`
also keeps its name — it is the *gridded* counterpart, so `grids/scalar/` would
be a contradiction.

An existing project directory strands its old `series/` folder, since Snakemake
cannot clean a path it no longer declares. `dev/scripts/prune_series_cache.py`
now reports it as a legacy generation; delete it once (see "Post-migration
cleanup" below).

## Removed output

`timeseries/gcm_timeseries.nc` is **no longer written**. A project directory from
an earlier run keeps its stale copy — Snakemake cannot clean an output no longer
declared — so delete `climate_projections/<proj>/timeseries/` by hand once; fresh
runs never create it. It merged the nine `series/*.nc` into one cube that
nothing consumed, while rounding to 2 decimals — re-imposing the quantisation
"Rounding dropped" above removes — and stripping every `cst_*` attribute, so it
carried no digest, region fingerprint or calendar and could not be traced.

If you were reading it, use `series/*.nc` for the full monthly timeseries (same
values, unrounded, with provenance) or `change_factors/annual.csv` /
`monthly.csv` for the analysis-ready long form.

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
