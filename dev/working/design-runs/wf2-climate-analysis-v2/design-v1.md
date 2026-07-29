# WF2 v2.0 — Climate Data & Projections Analysis: Design (DRAFT)

```
Status:     draft — not admitted to a milestone; no implementation authorised
Date:       2026-07-29
Authors:    tanerumit (with Claude Code)
Supersedes: none
Revisions:
  - 2026-07-29: initial draft
  - 2026-07-29: corrected §5.3 — retiring `historical_year_range` moves the
    reference window by a decade and is value-CHANGING; it belongs in migration
    step 5, not the value-neutral step 1. G3 is available-by-construction after
    step 1, realized at step 5. Linked the steps 1–2 task brief from §8.
```

Companion documents:

- `dev/workflows/wf2_climate_projections_overview.md` — rule-level map of WF2 **as
  it is today** (the baseline this design changes).
- `dev/workflows/climate_projections.md` — the behavioral contract of WF2 as it is
  today (config keys, unit split, `save_grids`, downstream semantics).
- `dev/p32a/climate-analysis-design.md` — the sealed milestone that created
  `blueearth_cst/climate_analysis/` and the model-free climate store this design
  builds on.

---

## 1. Problem statement

WF2 today computes one product: a cloud of (ΔP%, ΔT°C) change factors per
(model, scenario, horizon) that situates the WF3 perturbation grid in projection
space. Three problems motivate a v2.

**(a) The structure is inverted relative to the cost.** The expensive operation
is remote CMIP6 zarr access — the code comments record ~5 h per source before an
eager `.load()` was added. The rule graph fans out 6 jobs for `monthly_change`
(milliseconds of arithmetic each, behind a full hydromt import) while serializing
the network-bound stages behind an ordering edge that carries no data. Every
intermediate is `temp()`, so re-running with one changed horizon re-downloads the
entire archive slice.

**(b) The scientific product is narrower than the data it already fetches.** WF2
computes basin-mean *monthly* series for every model/scenario, then discards the
seasonality and emits annual scalars only — even though WF3 perturbs with
12-element *monthly* arrays. The reference window is configured independently of
the project's baseline window, so the overlay can be displaced from the response
surface it is overlaid on. Ensemble statistics are computed as percentiles over
as few as three models, two of which may come from the same institution.

**(c) The workflow is not a climate-analysis workflow.** The user's stated
direction is for WF2 to become a general climate data and projections analysis
capability, not a delta-factor calculator. Today it cannot even run without a
built hydrological model, because its only cross-workflow input is
`hydrology_model/staticgeoms/region.geojson`.

**The enabling discovery.** P3-2a already built most of what (c) needs.
`blueearth_cst/climate_analysis/` is a model-independent subpackage, and
`snake_utils.climate_store_spec` defines a single producer rule —
already declared identically in `Snakefile_model_creation` (1.10) and
`Snakefile_climate_experiment` (3.02) — that writes a shared store:

```
{project_dir}/climate_historical/{clim_source}_{window}/
    extract_historical.nc     # gridded historical climate over the basin
    store_region.geojson      # the model-free delineated polygon
    orography.nc              # chirps branch only
```

The extent is derived from `shared.basin` + the data catalog via
`parse_region_basin` — **no built model required**. WF2 declaring that same rule
gets, in one move: a model-free region (removing the WF1 dependency), and the
observed/reanalysis historical climate on disk next to the projections. That
turns "climate analysis workflow" from a rewrite into a composition.

---

## 2. Goals / Non-goals

### Goals

- **G1.** One reducer, applied uniformly to observed, GCM-historical, and
  GCM-future data, producing a single tidy series store — the substrate every
  other product reads.
- **G2.** WF2 runs from `region + catalogs` alone, with no `hydrology_model/` on
  disk.
- **G3.** The projection reference window equals the project baseline window by
  construction, not by validation.
- **G4.** Change factors emitted at both annual and monthly resolution, in a
  tidy long-format table with explicit units and provenance.
- **G5.** Re-running with a changed horizon, statistic, plot, or report costs no
  network access.
- **G6.** Fan-out width is set by the network-bound stage; the pure-computation
  stages are single jobs.
- **G7.** Absent catalog sources are resolved at DAG-build time, not papered over
  with dummy empty netCDFs at runtime.
- **G8.** One report artifact replaces a directory of loose PNGs.
- **G9.** The extension surface for further climate analysis (trends, extremes,
  multi-dataset comparison, bias diagnostics) is a named slot, not a rewrite.

### Non-goals

- **N1.** Driving WF3 from CMIP6. The grid-vs-cloud check in §5.8 is a one-way
  advisory that emits a figure and a warning; it never writes or adjusts WF3
  config. (`AGENTS.md` § Background.)
- **N2.** Relocating `plot_climate_source` (WF1 rule 1.15) into WF2 — see §6.2.
- **N3.** A standalone 4th Snakefile entry point. Named as decision **D1** and
  deferred; P3-2a deferred it explicitly and the roadmap lists it in the Phase-4
  candidate pool.
- **N4.** New third-party dependencies. Three candidates are recorded in §10 as
  asks, and the design as specified needs none of them.
- **N5.** Re-engineering how hydromt resolves catalogs or reads rasters.
  (`AGENTS.md` § Hard Constraints.)
- **N6.** Bias correction or downscaling. Delta-change is bias-cancelling by
  construction; adding a bias-correction layer is a separate method decision.

---

## 3. Constraints (standing; restated because this design brushes each one)

- **C1 — `climate_store_spec` must be declared identically in every Snakefile
  that declares it.** Its docstring is explicit: *"The input set is exactly one
  entry — the catalog — in both DAGs. An asymmetric input set re-creates the
  wf1↔wf3 re-extraction oscillation (design P2(b) / ext1-02)."* A third
  declaration inherits that constraint verbatim; deviating resurrects a fixed
  bug. The catalog **file** is the store's freshness boundary and is declared
  plain, never `ancient()`.
- **C2 — WF2 is a plausibility overlay.** Nothing in WF3 consumes its outputs to
  drive stress-test runs.
- **C3 — hydromt / hydromt_wflow / Wflow conventions are consumed verbatim.**
  Catalog format, CSDMS naming, `setup_*` semantics are upstream's.
- **C4 — `test_case/test_local` is the local baseline gate.** CI cannot run
  `check_baseline.py` or a whole-tree `semantic_tree_diff`; green CI is not
  evidence the baseline held.
- **C5 — no new dependency without explicit approval.**
- **C6 — production `project_dir` lives outside the repository tree.**

---

## 4. Decision criteria

Applied when the alternatives in §6 are weighed:

1. **Value-neutrality is tiered.** A step that provably cannot move a manifested
   number ships without a re-record. A step that can, ships with a documented
   re-record and a characterized diff. The two are never mixed in one commit.
2. **Cost follows the network.** Structure that adds parallelism to
   pure-computation stages is overhead, not efficiency.
3. **Sealed acceptance gates are not reopened.** A change that invalidates a
   sealed milestone's pinned assertion needs a reason stronger than tidiness.
4. **Explicit beats inferred.** Behavior keyed on a string literal, an
   undeclared file, or a silently-empty dataset is a defect class regardless of
   whether it currently produces correct numbers.
5. **The extension surface is designed once.** A slot named now costs a
   paragraph; a slot retrofitted later costs a migration.

---

## 5. Selected approach

### 5.1 Architecture — three stages, fan-out only where it pays

| Stage | Rule(s) | Fan-out | Network | Purpose |
| --- | --- | --- | --- | --- |
| **0. Store** | `extract_climate_grid` (from `climate_store_spec`) | — | yes | Model-free region polygon + gridded observed/reanalysis climate |
| **A. Reduce** | `reduce_climate_series` | `{series_key}` | yes | One source → one basin monthly series |
| **B. Derive** | `derive_change_factors` | — | no | All series → annual + monthly change factors |
| **C. Report** | `climate_report` | — | no | Tables + figures + one report page |

Plus `copy_config` (unchanged) and `gather_benchmarks` (unchanged). The three
`gather_*_logs` rules disappear: only one stage fans out, so one gather remains.

Rule count drops from 11 to 6. Job count on the seed config
(3 models × 2 scenarios × 1 member, plus 1 observed source) goes from 22 to
**13** — 1 store + 9 reduce + 1 derive + 1 report + 1 config + 1 benchmark
gather — while the *concurrent* width at the network-bound stage rises from 3
to 9.

```
catalogs ──► 0. extract_climate_grid ──┬─► store_region.geojson ──┐
                                       └─► extract_historical.nc  │
                                                        │         │
                                                        ▼         ▼
                        A. reduce_climate_series {series_key}  (obs + GCM hist + GCM fut)
                                                        │
                                            series/*.nc (PERSISTENT, not temp)
                                                        │
                                                        ▼
                                          B. derive_change_factors
                                             ├─ change_factors_annual.csv
                                             ├─ change_factors_monthly.csv
                                             └─ change_factors.nc
                                                        │
                                                        ▼
                                              C. climate_report
                                                 ├─ report.md
                                                 ├─ plots/*.png
                                                 └─ provenance.json
```

### 5.2 The unified series store — the central idea

Stage A applies **one reducer** to every source, regardless of provenance:

```
reduce(source_entry, region_bounds, buffer, variables, window)
    -> monthly basin series, dims (time, variable), with source metadata as coords
```

Observed/reanalysis data enters through the store's `extract_historical.nc`;
CMIP6 enters through the catalog. Both leave as the same artifact shape. That
single unification is what converts WF2 from a delta-factor calculator into a
climate-analysis workflow: once observed and GCM series are reduced identically
into one tidy store, observed climatology, GCM-vs-observed historical
comparison, projection diagnostics, and the delta factors are all *reads* of the
same table rather than separate pipelines.

**Series key.** `{provenance}_{dataset}_{scenario}_{member}` with `/` sanitized
to `_` (CMIP6 model names carry a vendor path segment, e.g.
`NOAA-GFDL/GFDL-ESM4`). Examples: `obs_era5_historical_none`,
`cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1`.

**Layout.**

```
{project_dir}/climate/
    series/{series_key}.nc            # PERSISTENT — the cache
    change_factors/{annual,monthly}.csv
    change_factors/change_factors.nc
    plots/*.png
    report.md
    provenance.json
```

`climate_historical/` (the shared store) is untouched — it is co-owned with WF1
and WF3. Whether `climate/` supersedes today's `climate_projections/<clim_project>/`
or sits beside it is **OQ-3**.

**Spatial reduction.** Area-weighted (cos-latitude) mean over the bbox +
configurable buffer, replacing today's unweighted `mean([x_dim, y_dim])`.
Weighting is negligible for the equatorial seed basin and material at high
latitude. The buffer stays a *regional* sampling choice, deliberately: at Amon
resolution (~1–2°) a catchment-polygon mask on a small basin can select a single
cell, which is worse than a regional average for large-scale change factors. This
is now a documented, configurable decision rather than an accident of
`bbox + buffer=1`.

**Caching.** `series/*.nc` is **not** `temp()`. Each file is a few KB; the run
that produced it may have cost hours. The reducer's `params` carry a digest of
`(source entry, region bounds, buffer, variables, window, reducer_version)`, so
Snakemake's params rerun-trigger re-derives a series when any of those change and
reuses it otherwise. Changing a horizon, a statistic, a figure, or the report
then costs zero network access (**G5**).

### 5.3 Region and baseline — solved structurally, not by validation

WF2 declares `extract_climate_grid` from `climate_store_spec` (identically —
**C1**) and reads `store_region.geojson` instead of
`hydrology_model/staticgeoms/region.geojson`. Two consequences:

- **G2 falls out immediately, and is value-neutral.** The store's extent is a
  pure function of `shared.basin` + catalog; WF2 stops depending on WF1 entirely.
- **G3 becomes available, but is not free.** The store key is
  `{clim_historical}_{slugify_window(...)}` built from
  `shared.historical_window`, so once the observed reference series comes from
  the store, WF2's reference window **is** the project baseline window by
  construction, and today's `historical_year_range: [1990, 2010]` versus
  `shared.historical_window: 2000-01-01 … 2020-12-31` mismatch — roughly 0.3 °C
  of global warming between window midpoints, displacing the overlay from the
  response surface — cannot recur.

  **Retiring `historical_year_range` changes every change factor**, because the
  reference window moves by a decade. It is therefore *not* part of the
  value-neutral decoupling: the store declaration lands in migration step 1, and
  the window switch lands in step 5 with the other methodological changes, under
  a documented re-record. Structural-by-construction describes the end state,
  not a free step.

**Measured (2026-07-29, seed fixture).** The two polygons have identical bounds
`[9.658333, 0.35, 9.858333, 0.483333]`. WF2 consumes only
`geom.geometry.bounds`, so on this fixture the swap selects the identical GCM
cell set and cannot move a number. This is a fixture-level result: the migration
gate in §9 re-checks it rather than assuming it generalizes to a basin whose
hydromt_wflow delineation at `resolution: 0.00833` diverges from
`parse_region_basin`.

### 5.4 Variable specification — replacing name-based dispatch

Today `get_stats_climate_proj.py` dispatches on the literal string `"precip"`
(sum) versus everything-else (mean), and `get_change_climate_proj.py` dispatches
the same way for multiplicative versus additive change.

**Verified 2026-07-29:** the catalog sources are CMIP6 **`Amon`** — already
monthly means, with `pr` converted to mm/day (`unit_mult: 86400`) and `tas` to
°C (`unit_add: -273.15`). With one sample per month, `resample("MS").sum()` and
`.mean()` return the identical value: **the monthly aggregation dispatch is a
no-op.** `dev/workflows/climate_projections.md` describes it as "the
multiplicative/additive split the whole change-factor method depends on" and
warns that a mis-named precipitation variable would be aggregated with wrong
units. On Amon input that specific failure cannot occur; the real split lives
only in the change arithmetic downstream. The documented contract and the code
disagree about what the dispatch key controls — which is the strongest argument
for making it explicit:

```yaml
variables:
  precip: {source: pr,  aggregate: sum,  change: relative, units: mm/day}
  temp:   {source: tas, aggregate: mean, change: absolute, units: degC}
```

The reducer reads `aggregate`; the derive stage reads `change`. Neither infers
anything from a name. This removes the silent-wrong-answer path, makes the
workflow extensible to other catalog variables (the catalog already renames
`rsds`→`kin` and `psl`→`press_msl`), and makes the monthly/annual aggregation
question answerable from config rather than from source.

### 5.5 Change factors — annual and monthly

Stage B is one job, no network, reading `series/*.nc`.

- **Annual** (today's product): hydrological-year aggregation over the reference
  and horizon windows, then the configured statistics — relative % for `change:
  relative`, absolute delta for `change: absolute`.
- **Monthly** (new, **G4**): the 12-month change factor pattern per
  (dataset, scenario, horizon). This is the product directly comparable to how
  WF3 perturbs — `stress_test.{temp,precip}.mean.{min,max}` are 12-element
  arrays — and today it is computed only inside the `save_grids` branch and
  inside plotting code, never emitted as a table.

**Statistics.** The current eight (`mean, std, var, median, q_90, q_75, q_10,
q_25`) over a 20-year window mean `q_90` is effectively the second-highest of 20
values. Two changes, both recorded as value-changing (§8, step 5):

- Default the reference and horizon windows to **30 years** (WMO normal); warn
  below 20.
- Emit only statistics the sample supports by default (`mean`, `median`, `std`),
  with the tail quantiles opt-in and labelled with their effective sample size.

**Ensemble treatment.** The report states ensemble composition explicitly,
including institution counts — the seed ensemble is GFDL-ESM4, INM-CM4-8,
INM-CM5-0, i.e. two of three from one institution, currently treated as two
independent draws. Percentile envelopes across models are drawn only above a
configurable threshold (default 10); below it, individual model traces plus the
range. Weighting and institution-level dedup are **OQ-6**, not built here.

Month-length weighting on annual aggregates (~0.3 %, exact for 360-day-calendar
models) is corrected while the aggregation code is being touched.

### 5.6 Report stage

One job producing:

- `change_factors/{annual,monthly}.csv` — long format, one row per
  `(dataset, institution, scenario, member, horizon, period, variable, statistic)`
  with `value`, `units`, `reference_window`, `horizon_window`.
- `report.md` — the ΔT/ΔP cloud, seasonal change pattern, timeseries context,
  ensemble-composition table, and the resolution status of every requested
  source.
- `provenance.json` — resolved sources, missing sources, failed sources, windows,
  config digest, reducer version.
- `plots/*.png` — all **declared**. Today 6 of 8 figures are undeclared, one is
  saved without an extension, and `gcm_timeseries.nc` is declared under the label
  `timeseries_csv` (see the overview doc §2/§4).

A missing catalog source currently becomes an empty dataset, a dummy netCDF, and
a silent drop at merge — the user's 3-model ensemble quietly becomes 2. The
provenance file makes that visible; a run where fewer than a configurable minimum
of sources resolve fails loud.

### 5.7 Source resolution at DAG-build time

`config/catalogs/cmip6_data.yml` declares templated entries
(`cmip6_{model}_{scenario}_{member}`) with a `placeholders:` block listing valid
models and members. Source existence is therefore a **pure YAML lookup** — no
hydromt import, no network — cheap enough to run at DAG-build time on every
dry-run.

The `{series_key}` fan-out is built from the validated list. Absent combinations
never become jobs. This deletes: the dummy-empty-netCDF pattern, `filter_nonempty`,
and the three separate "did this file have data?" loops in the plotting code.

Catalog membership is not availability — a listed source can still fail at read
time. That path stays, but records a status in `provenance.json` instead of
emitting an empty file.

### 5.8 Extension slots (named, not built)

The unified series store makes each of these a read, not a pipeline:

| Slot | What it adds | Reads |
| --- | --- | --- |
| **S1 — observed climatology & trends** | Long-term means, seasonality, annual trends with CI | `series/obs_*` |
| **S2 — multi-dataset comparison** | ERA5 vs CHIRPS vs E-OBS over the same basin | several `series/obs_*` |
| **S3 — GCM historical bias diagnostics** | GCM-historical vs observed climatology (diagnostic only — delta-change already cancels bias) | `series/obs_*` + `series/cmip6_*_historical_*` |
| **S4 — extremes / indicator set** | Wet/dry spells, hot days, seasonality indices | `series/*` (needs sub-monthly input; see OQ-5) |
| **S5 — grid-vs-cloud advisory** | Does the configured WF3 perturbation grid envelope the projected cloud? Emits a figure and a warning. **Never writes WF3 config** (N1/C2). | `change_factors/*` + WF3 config, read-only |

S5 is the highest-value slot and the one closest to a hard constraint: it is a
one-way diagnostic a human reads, not the pipeline consuming projections.

---

## 6. Alternatives considered

### 6.1 Keep the current rule-per-combination fan-out; add caching only

Cheapest change: make the series persistent, leave the 11-rule structure alone.
**Not chosen** — it fixes G5 but not G1/G4/G6/G7, and the structure is what
prevents the observed side from entering at all. Preferable if the only complaint
were re-run cost.

### 6.2 Move `plot_climate_source` (WF1 rule 1.15) into WF2

Superficially tidier: all climate figures in the climate workflow.
**Not chosen.** `tests/test_plot_climate_source.py` pins the P3-2a P4 assertion —
those three figures build with neither `hydrology_model/` nor
`config/templates/wflow_build_model.yml` on disk. Relocating the rule invalidates
a sealed milestone's acceptance gate for a cosmetic gain. WF2 **composes**
instead: it declares the same store producer (C1) and reads the store's outputs,
leaving WF1's figure rule where it is. Preferable only if WF1 were being
restructured anyway for independent reasons.

### 6.3 A standalone 4th Snakefile versus extending `Snakefile_climate_projections`

A `Snakefile_climate_analysis` would match the widened purpose and give the
capability its own entry point. **Deferred (decision D1, recommendation: extend
in place for v2.0).** P3-2a explicitly deferred the standalone entry point
("subpackage now, standalone entry point deferred; platform surface unchanged")
and the roadmap lists it in the Phase-4 candidate pool. Extending in place keeps
the platform surface at three entry points and the `run_workflows.py`
`enabled:` contract unchanged; the rename becomes a separable follow-up once the
internal structure has settled. Preferable now if the workflow must be runnable
independently of a project that has WF1/WF3 configured at all.

### 6.4 Region source — store polygon versus a config-supplied region path

An explicit `region:` config key would decouple WF2 from *both* other workflows.
**Not chosen** — it adds a config surface for something the store already
computes model-free, and it would let WF2's region drift from the one WF1/WF3
extract against. The store polygon keeps a single delineation for the project.
Preferable if WF2 ever needs to run over a region the store was not built for.

### 6.5 Catalog generation via `intake-esm` instead of hand-enumerated entries

`cmip6_data.yml` hand-lists 23 models across scenarios. A query-based catalog
would remove that maintenance and enable ensembles large enough for real
percentile envelopes. **Not chosen for v2.0** — new dependency (C5). Recorded as
an ask in §10.

### 6.6 Long-format Parquet instead of netCDF for `series/`

Parquet is a better fit for tidy tables and pandas consumers. **Not chosen** —
netCDF is the repo's stated interchange format across R/Python/Julia, and the
series carry coordinate metadata that survives the netCDF round-trip. The
change-factor *tables* are CSV, which covers the tidy-consumer case.

---

## 7. Consequences and risks

**Observable consequences (falsifiable):**

1. WF2 runs to completion with no `hydrology_model/` directory on disk.
2. A second WF2 run with a changed `future_horizons` entry performs **zero**
   network reads; wall-clock drops to the derive+report stages.
3. `snakemake -n` on the seed config lists 13 jobs, not 22.
4. A config naming a model absent from the catalog fails at DAG build with a
   message naming the model — instead of silently producing a smaller ensemble.
   **This is a user-visible behavior change**, and the most likely source of
   "it used to work" reports.
5. `change_factors/monthly.csv` exists and has 12 rows per
   (dataset, scenario, horizon, variable, statistic).
6. `provenance.json` names every requested source and its resolution status.
7. Every figure WF2 writes is a declared output.

**Risks:**

- **R1 — third `climate_store_spec` declaration diverges (C1).** An asymmetric
  input set re-creates the wf1↔wf3 re-extraction oscillation. Mitigation: the
  declaration is generated from the shared helper, and a test asserts the three
  declarations produce identical input sets.
- **R2 — dropping the hist→fut ordering edge exposes a directory race.** The
  current `get_stats_climate_proj.py` uses `os.mkdir` guarded by
  `os.path.exists`; concurrent reduce jobs would race. Mitigation: the reducer
  creates its own directories with `makedirs(exist_ok=True)`. This is the likely
  original reason for the ordering edge.
- **R3 — window default change moves every number.** 20 → 30 years is a genuine
  methodological change. Mitigation: isolated in its own commit (§8 step 5) with
  a documented re-record and characterized diff; the default is a named decision
  the user signs off (**OQ-4**).
- **R4 — persistent series accumulate.** Each is KB-scale, but a large ensemble
  over many configs grows the tree. Mitigation: keyed by content digest, so
  re-runs overwrite rather than accumulate; stale-key pruning is a follow-up.
- **R5 — output-layout change breaks user muscle memory.** `climate/` versus
  `climate_projections/<clim_project>/` (**OQ-3**). The web app is explicitly not
  a constraint here, but a migration note is required.
- **R6 — the report becomes the only place a number is stated.** If the report
  generator fails, the run has no human-readable output. Mitigation: the CSV
  tables are separate declared outputs and do not depend on the report stage.

---

## 8. Migration + commit plan

Sequenced so every output-neutral step lands before the first value-changing one.
Steps 1–4 are checkable against the baseline manifest; step 5 onward changes
numbers deliberately.

| # | Commit | Value-neutral? | Gate |
| --- | --- | --- | --- |
| 1 | Declare `extract_climate_grid` in WF2 from `climate_store_spec`; read `store_region.geojson` | **Yes** (bounds measured identical on the fixture, §5.3) | `check_baseline.py`; re-verify bounds equality; identical-input-set test (R1) |
| 2 | Persistent `series/` store: drop `temp()`, add the params digest, `makedirs(exist_ok=True)`, drop the hist→fut ordering edge | **Yes** (same values, different lifetime) | `check_baseline.py`; second run performs no network reads |
| 3 | Collapse `monthly_stats_hist`/`monthly_stats_fut` into one `reduce_climate_series` over `{series_key}`; unify output naming; make `members` a wildcard | **Yes** | `check_baseline.py`; `pytest tests/test_cli.py` |
| 4 | Collapse `monthly_change` + `monthly_change_scalar_merge` into one `derive_change_factors` job; source validation at DAG build; delete the dummy-netCDF path | **Yes** for present sources; **behavior change** for absent ones (consequence 4) | `check_baseline.py`; new unit tests for the catalog validator |
| 5 | Variable spec; area weighting; month-length weighting; retire `historical_year_range` in favour of `shared.historical_window` (§5.3); 30-year window default | **No — value-changing** | Documented re-record + characterized diff; user sign-off on OQ-4 |
| 6 | Monthly change-factor table; tidy CSV schema; `provenance.json` | Additive | Schema tests; row-count assertions |
| 7 | Report stage; declare all figures; retire the loose-PNG set | Additive + layout | Visual QA; migration note |

Steps 1–2 are independently useful and could ship alone: together they deliver
G2 and G5, and put G3 within reach, without touching a single computed value.
Task brief: `dev/working/2026-07-29_wf2-v2-decouple-and-cache.md`.

---

## 9. Validation plan

**Per-commit gates.**

- `pytest tests/test_cli.py` — dry-runs all three Snakefiles (cheapest DAG check).
- `pytest tests/` — full suite; must stay green and additive.
- `snakemake -n` on the seed config before and after each structural commit; job
  count and rule set compared explicitly.
- `dev/scripts/check_baseline.py check` on `test_case/test_local` for steps 1–4.
  **CI cannot run this** (C4) — it is a local gate.
- `dev/scripts/semantic_tree_diff.py` whole-tree on the WF2 output subtree.

**Targeted checks.**

- **Region equality (step 1).** Re-run the bounds comparison of
  `store_region.geojson` versus `hydrology_model/staticgeoms/region.geojson`
  and assert the buffered bbox selects the same GCM cell set. If they diverge on
  a future basin, step 1 is reclassified as value-changing.
- **Identical input sets (R1).** A test asserting the three
  `climate_store_spec` declarations produce byte-identical input lists.
- **Cache correctness (step 2).** Run; touch nothing; re-run — assert zero
  reduce jobs. Change one digest component — assert exactly the affected series
  re-derive.
- **Reducer properties (step 5).** Relative change is invariant to a
  multiplicative unit rescale of the input; absolute change is invariant to an
  additive offset. Both are cheap property tests over synthetic series.
- **Catalog validator (step 4).** Unit tests over the templated-entry +
  `placeholders:` lookup, including a model absent from the list and a scenario
  template absent from the file.
- **Characterized diff (step 5).** Old versus new change factors on the seed
  config, with the difference attributed per cause (weighting, window,
  month-length) rather than reported as one aggregate delta.

**Not validated here.** Whether 30-year windows are the right default for a
given basin; whether the ensemble is adequate. Those are user judgements the
report surfaces, not gates.

---

## 10. Open questions

- **OQ-1 (D1).** Extend `Snakefile_climate_projections` in place, or open a 4th
  entry point (`Snakefile_climate_analysis`)? *Recommendation: extend in place
  for v2.0; treat the entry point as a separable follow-up (§6.3).*
- **OQ-2.** Does this open **Phase 4**, or land as an unnumbered milestone? The
  roadmap records Phase 3 complete with "the open question is whether to close
  the roadmap or open a Phase 4", and lists the 4th entry point in the candidate
  pool. This design plus the entry point would be a coherent Phase 4.
- **OQ-3.** Output layout: does `climate/` supersede
  `climate_projections/<clim_project>/`, or sit beside it? (R5.)
- **OQ-4.** Default window length — 30 years (WMO normal) versus keeping 20.
  Changes every number; needs an explicit call.
- **OQ-5.** Extremes indices (**S4**) need sub-monthly data. CMIP6 `day` tables
  exist in the same GCS store but are ~30× the volume. Does the scope include a
  daily branch, or do extremes stay observed-only?
- **OQ-6.** Ensemble weighting / institution-level dedup policy. Surfaced in the
  report; not applied.
- **OQ-7 (dependencies — asks, C5).** None required by this design.
  `xclim` (calendar/unit handling, standard indicators — relevant to S4),
  `regionmask` (fractional-area polygon masking — relevant if §5.2's regional
  sampling choice is revisited), `intake-esm` (query-based catalog generation —
  §6.5, and the precondition for ensembles large enough to justify percentile
  envelopes).
- **OQ-8.** Should `save_grids` survive? It currently gates an entire undeclared
  file layer passed between rules via `params`. Options: declare the outputs
  conditionally, move the grid branch to its own optional rule, or retire it.

---

## 11. Revision log

- **2026-07-29** — initial draft. Grounded in `Snakefile_climate_projections`,
  `blueearth_cst/projections/*.py`, `blueearth_cst/climate_analysis/*.py`,
  `blueearth_cst/shared/snake_utils.py` (`climate_store_spec`),
  `config/catalogs/cmip6_data.yml`, and `config/workflows/snake_config_model_test.yml`.
  Two facts measured rather than assumed: the two region polygons have identical
  bounds on the seed fixture (§5.3), and the CMIP6 sources are `Amon` so the
  monthly sum/mean dispatch is a no-op (§5.4).
