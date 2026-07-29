# WF2 v2.0 — GCM Projections Analysis: Design (DRAFT, revision 2)

```
Status:     draft — not admitted to a milestone; no implementation authorised
Date:       2026-07-29
Authors:    tanerumit (with Claude Code)
Supersedes: none
Revisions:
  - 2026-07-29: initial draft (design-v1.md)
  - 2026-07-29: corrected §5.3 — retiring `historical_year_range` moves the
    reference window by a decade and is value-CHANGING (design-v1.md)
  - 2026-07-29: revision 2 (this file) — authored against G1 rulings R1–R4
    after round-1 internal (Fable / `critical-thinker`) and external
    (gpt-5.6-sol) review. Scope narrowed to GCM projections analysis; the
    reference series is clipped, never spliced; `save_grids` becomes a
    first-class optional branch; the ensemble sampling unit is the unique
    model. Finding-by-finding disposition: `ledger.md`.
```

Companion documents:

- `dev/workflows/wf2_climate_projections_overview.md` — rule-level map of WF2 **as
  it is today** (the baseline this design changes).
- `dev/workflows/climate_projections.md` — the behavioral contract of WF2 as it is
  today (config keys, unit split, `save_grids`, downstream semantics).
- `dev/p32a/climate-analysis-design.md` — the sealed milestone that created
  `blueearth_cst/climate_analysis/` and the model-free climate store this design
  builds on.
- `dev/working/design-runs/wf2-climate-analysis-v2/` — the review run: `intake.md`,
  `design-v1.md`, the two round-1 reviews, `review-index.md`, `status.md`
  (G1 rulings R1–R4), and `ledger.md`.

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
12-element *monthly* arrays. Ensemble statistics are computed as percentiles over
as few as three models, two of which come from the same institution, and the
member axis is a config list rather than a declared dimension. The spatial
reduction is an unweighted `mean([x_dim, y_dim])`; the annual aggregation ignores
month length and each model's calendar.

**(c) The workflow cannot run without a built hydrological model.** Its only
cross-workflow input is `hydrology_model/staticgeoms/region.geojson`, so a
projections-only run requires a completed WF1.

**The enabling discovery.** P3-2a already built the model-free region.
`snake_utils.climate_store_spec` defines a single producer rule — declared
identically in `Snakefile_model_creation` (1.10) and
`Snakefile_climate_experiment` (3.02) — that writes:

```
{project_dir}/climate_historical/{clim_source}_{window}/
    extract_historical.nc     # gridded historical observed/reanalysis climate
    store_region.geojson      # the model-free delineated polygon
    orography.nc              # chirps branch only
```

The extent is derived from `shared.basin` + the data catalog via
`parse_region_basin` — **no built model required**. WF2 declaring that same rule
removes the WF1 dependency.

**What revision 2 changed about the problem framing.** design-v1 additionally
claimed that declaring the store turns WF2 into a general climate-analysis
workflow "by composition", because observed and GCM data would flow through one
reducer. Round-1 review established that the two sources are not
interchangeable at stage A: `extract_historical.nc` on the seed fixture is daily
(`freq: D`, 7671 steps), 7 variables, precip in `mm d**-1` and temp in **K**,
while CMIP6 `Amon` is monthly, 2 variables, temp already in °C via the catalog's
`unit_add`. Owner ruling **R4** resolves this by narrowing v2.0 to **monthly GCM
projections output analysis with room to expand to gridded results** and taking
**no observed comparison at this stage**. The store is still declared — for the
region polygon — but `extract_historical.nc` is **not reduced in v2.0**. §5.4
records the cost of that choice as a named decision.

---

## 2. Goals / Non-goals

### Goals

- **G1.** One reducer applied uniformly to **every GCM source** — historical
  experiment and every scenario experiment, every model, every member —
  producing a single tidy monthly series store that all downstream products
  read. (Narrowed from v1 by **R4**: observed data is not a stage-A input in
  v2.0.)
- **G2.** WF2 runs from `shared.basin` + catalogs alone, with no
  `hydrology_model/` on disk.
- **G3.** The overlay's reference window is **explicit, clipped to the source,
  recorded, and checked** against `shared.historical_window`, with a warning when
  they differ. (Restated from v1 by **R1**: an alignment check with a warning,
  not equality by construction.)
- **G4.** Change factors emitted at both annual and monthly resolution, in a
  tidy long-format table with explicit units, formulas, and provenance.
- **G5.** Re-running with a changed horizon, statistic, plot, or report costs no
  network access.
- **G6.** Fan-out width is set by the network-bound stage; the pure-computation
  stages are single jobs.
- **G7.** Absent catalog sources are resolved at DAG-build time, not papered over
  with dummy empty netCDFs at runtime.
- **G8.** One report artifact replaces a directory of loose PNGs, and every
  figure is a declared output.
- **G9.** The extension surface is **documented with the contract change each
  extension requires** — not advertised as a free read. (Restated from v1 by
  **ext1-06** and **R4**; see §5.10.)
- **G10.** Raw gridded products remain available on request as a **declared**
  optional branch, default off. (New, from **R2**.)

### Non-goals

- **N1.** Driving WF3 from CMIP6. The grid-vs-cloud check (S5, §5.10) is a
  one-way advisory that emits a figure and a warning; it never writes or adjusts
  WF3 config. (`AGENTS.md` § Background.)
- **N2.** Relocating `plot_climate_source` (WF1 rule 1.15) into WF2 — see §6.2.
- **N3.** A standalone 4th Snakefile entry point. Named as decision **D5** and
  deferred.
- **N4.** New third-party dependencies. Three candidates are recorded in §10 as
  asks; the design as specified needs none of them, and none is adopted.
- **N5.** Re-engineering how hydromt resolves catalogs or reads rasters.
  (`AGENTS.md` § Hard Constraints.)
- **N6.** Bias correction or downscaling.
- **N7.** *(New, R4.)* Observed-vs-GCM comparison, observed climatology, and
  GCM bias diagnostics. `extract_historical.nc` is **not** reduced, read, or
  plotted by any v2.0 WF2 rule. The store is declared for its region polygon.
- **N8.** *(New, R1.)* Splicing, gap-filling, or otherwise processing the
  2015–2020 interval between the CMIP6 historical experiment and the scenario
  experiments. No historical or scenario run is modified.
- **N9.** *(New, R3 residual.)* Ensemble weighting by model performance and
  institution-level de-duplication. The report states institution counts so a
  reader can judge the ensemble; the pipeline applies no such weighting.

---

## 3. Constraints (standing; restated because this design brushes each one)

- **C1 — `climate_store_spec` must be declared identically in every Snakefile
  that declares it.** Its docstring is explicit: *"The input set is exactly one
  entry — the catalog — in both DAGs. An asymmetric input set re-creates the
  wf1↔wf3 re-extraction oscillation (design P2(b) / ext1-02)."* A third
  declaration inherits that constraint verbatim. The catalog **file** is the
  store's freshness boundary and is declared plain, never `ancient()`.
- **C2 — WF2 is a plausibility overlay.** Nothing in WF3 consumes its outputs to
  drive stress-test runs.
- **C3 — hydromt / hydromt_wflow / Wflow conventions are consumed verbatim.**
- **C4 — `test_case/test_local` is the local baseline gate.** CI cannot run
  `check_baseline.py` or a whole-tree `semantic_tree_diff`; green CI is not
  evidence the baseline held.
- **C5 — no new dependency without explicit approval.**
- **C6 — production `project_dir` lives outside the repository tree.**

---

## 4. Decision criteria

Applied when the alternatives in §6 are weighed:

1. **Value-neutrality and manifest-cleanliness are different properties, and
   both are tiered.** *Value-neutral* means no manifested number moves.
   *Manifest-clean* means every pinned baseline target matches without a
   re-record. They are separable: the manifest pins a **verbatim sha256 of the
   seed config file** (§8), so a step that only adds a config key is
   value-neutral but not manifest-clean. A step that provably cannot move a
   number ships with, at most, a re-record of the config target and the diff
   shown. A step that can move a number ships with a full documented re-record
   and a characterized diff. The two are never mixed in one commit.
2. **Cost follows the network.** Structure that adds parallelism to
   pure-computation stages is overhead, not efficiency.
3. **Sealed acceptance gates are not reopened.** A change that invalidates a
   sealed milestone's pinned assertion needs a reason stronger than tidiness.
4. **Explicit beats inferred.** Behavior keyed on a string literal, an
   undeclared file, or a silently-empty dataset is a defect class regardless of
   whether it currently produces correct numbers.
5. **A gate must be executable where it is placed.** A policy check belongs at
   the point in the DAG where it can actually run and stop the run. (New in
   revision 2 — this is what ext1-02 falsified about v1's runtime failure
   handling.)
6. **The extension surface is designed once, and honestly.** A slot named now
   costs a paragraph; a slot retrofitted later costs a migration. A slot
   advertised as free when it is not costs credibility.

---

## 5. Selected approach

### 5.1 Scope of v2.0 (owner ruling R4)

**In scope.** Monthly GCM projections output analysis: acquire every configured
CMIP6 (model, experiment, member) source, reduce each to one basin monthly
series, derive annual and monthly change factors relative to a clipped reference
window, summarize the ensemble, and report. Plus an optional, declared gridded
branch (§5.8) for later advanced analysis.

**Out of scope for v2.0.** Any product that reads observed or reanalysis data.
Concretely: **`extract_historical.nc` is not reduced in v2.0.** No WF2 rule opens
it. The `extract_climate_grid` rule is declared solely so WF2 can read
`store_region.geojson` and stop depending on `hydrology_model/`; the cost of that
is the subject of decision **D2** (§5.4, §6.4).

**What this buys.** The four blocking and major findings that arose from mixing
observed and GCM data at stage A (ext1-03, ext1-06, and the observed half of
risk-09/ext1-10) are removed rather than mitigated. What remains is a single
source family with a single temporal semantics, which is what makes the reducer
contract in §5.3 specifiable in one page.

**What this costs.** The "general climate data & projections analysis workflow"
of the change request is reached in two moves, not one. §5.10 names the contract
change each remaining move requires so that v2.0's decisions do not have to be
undone.

### 5.2 Architecture — three stages, fan-out only where it pays

| Stage | Rule(s) | Fan-out | Network | Purpose |
| --- | --- | --- | --- | --- |
| **0. Store** | `extract_climate_grid` (from `climate_store_spec`) | — | yes | Model-free region polygon (consumed) + gridded observed climate (**not** consumed in v2.0, N7) |
| **A. Reduce** | `reduce_gcm_series` | `{series_key}` | yes | One GCM source → one basin monthly series (+ optional gridded climatology) |
| **B. Derive** | `derive_change_factors` | — | no | All series → annual + monthly change factors (+ optional gridded change) |
| **C. Report** | `climate_report` | — | no | Tables + figures + one report page + provenance |

Plus `copy_config` (unchanged), `gather_series_logs` (one gather, because one
stage fans out), and `gather_benchmarks` (unchanged).

**Rule count: 11 → 8**, both counts including the `all` target rule. New set:
`all`, `copy_config`, `extract_climate_grid`, `reduce_gcm_series`,
`derive_change_factors`, `climate_report`, `gather_series_logs`,
`gather_benchmarks`. The three `gather_*_logs` rules collapse to one; `ruleorder`
disappears with the rules it ordered.

**Job accounting on the seed config** (`config/workflows/snake_config_model_test.yml`:
3 models × 2 scenarios × 1 member × 1 horizon; `save_grids: false`).

Counting convention, stated because v1's arithmetic was wrong twice: counts
**exclude the `all` target job**, which is how v1's "22 today" was counted.
Today's WF2 under that convention is 1 `copy_config` + 3 `monthly_stats_hist` +
6 `monthly_stats_fut` + 6 `monthly_change` + 1 `monthly_change_scalar_merge` +
1 `plot_climate_proj_timeseries` + 3 `gather_*_logs` + 1 `gather_benchmarks` = **22**.

Under this design, on a **fresh** `project_dir`:

| Component | Jobs |
| --- | --- |
| `extract_climate_grid` | 1 |
| `reduce_gcm_series` — GCM historical (3 models × 1 member) | 3 |
| `reduce_gcm_series` — GCM scenario (3 models × 2 scenarios × 1 member) | 6 |
| `derive_change_factors` | 1 |
| `climate_report` | 1 |
| `copy_config` | 1 |
| `gather_series_logs` | 1 |
| `gather_benchmarks` | 1 |
| **Total** | **15** |

**14** when `climate_historical/era5_20000101_20201231/` already exists (WF1 has
run, or WF2 has run once) — the store rule is then not scheduled.

Reduce is **9, not 10**: risk-09 / ext1-10 both counted an observed series, which
was correct against v1's §5.2 but is removed by **R4** (N7). Reduce is **not**
`3 + 6 + 1`; it is `3 + 6`.

`save_grids: true` does **not** change any of these numbers — the gridded
artifacts are additional declared outputs of the same jobs (§5.8).

The concurrent width at the network-bound stage rises from 3 (today's
`monthly_stats_hist`, serialized ahead of `monthly_stats_fut` by a data-free
ordering edge) to 9.

```
catalogs ──► 0. extract_climate_grid ──► store_region.geojson ──┐
                                     └─► extract_historical.nc  │   (not read in v2.0 — N7)
                                                                │
                            ancient() edge: ordering only ──────┤
                                                                ▼
      A. reduce_gcm_series {series_key}   (GCM historical + GCM scenario, all members)
                     │
                     ├─► series/{series_key}.nc          PERSISTENT, digest-stamped
                     └─► grids/monthly_climatology_{series_key}.nc   [save_grids only]
                     │
                     ▼
      B. derive_change_factors            (explicit expanded input list — never a glob)
                     ├─► change_factors/annual.csv
                     ├─► change_factors/monthly.csv
                     ├─► change_factors/change_factors.nc
                     └─► grids/change_{dataset}_{scenario}_{member}_{horizon}.nc  [save_grids only]
                     │
                     ▼
      C. climate_report
                     ├─► report.md
                     ├─► plots/*.png        (all declared)
                     └─► provenance.json
```

### 5.3 The GCM series store — reducer, identity, caching

Stage A applies one reducer to every GCM source:

```
reduce(catalog_entry, region_bounds, buffer, variable_spec, acquisition_window)
    -> monthly basin series, dims (time, variable), source metadata as coords
```

**Series key.** `cmip6_{dataset}_{experiment}_{member}`, with `/` in the CMIP6
model name sanitized to `_` (the catalog's `placeholders.model` list carries a
vendor path segment, e.g. `NOAA-GFDL/GFDL-ESM4`). Examples:
`cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1`,
`cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1`. `member` is a wildcard, not a config list
folded inside one job, so a multi-member ensemble is a fan-out rather than a
loop.

**Layout** (decision **D3**, §6.10 — the WF2 root does *not* move in v2.0):

```
{project_dir}/climate_projections/{clim_project}/
    series/{series_key}.nc                 # PERSISTENT — the cache
    grids/…                                # save_grids only
    change_factors/{annual,monthly}.csv
    change_factors/change_factors.nc
    summary/…                              # retained; manifest-pinned (§8)
    plots/*.png
    report.md
    provenance.json
```

`climate_historical/` (the shared store) is untouched — it is co-owned with WF1
and WF3.

**Spatial reduction — a labelled approximation, with a geometry check.**
Basin-mean over the bbox + configurable buffer, weighted by cos(latitude),
replacing today's unweighted `mean([x_dim, y_dim])`. This is stated as an
**approximation to cell-area weighting, valid for 1-D rectilinear
monotonically-spaced lat/lon grids**, and not as an area weight in general.

The reason exact areas are unavailable is in the repo: `config/catalogs/cmip6_data.yml`
sets `drop_variables: [time_bnds, lat_bnds, lon_bnds, bnds]` on every CMIP6
entry, so cell edges are not retained and true cell areas cannot be derived from
what the reducer receives. The reducer therefore:

- checks that the latitude and longitude coordinates are 1-D and strictly
  monotonic, and **raises** naming the source if they are not (2-D or curvilinear
  coordinates are refused, not silently mis-weighted);
- records `cst_weighting_scheme = "cos_latitude"` and the geometry-check result
  as attributes on the series and as fields in `provenance.json`;
- states the scheme in the report.

The buffer stays a *regional* sampling choice, deliberately: at `Amon` resolution
(~1–2°) a catchment-polygon mask on a small basin can select a single cell, which
is worse than a regional average for large-scale change factors. It is a
documented, configurable decision rather than an accident of `bbox + buffer=1`.

Retaining bounds so exact areas can be computed is a live option — the catalog is
repo-owned — but it changes what hydromt reads for every entry and needs its own
evidence. Recorded as **OQ-10**.

**Precision.** Stage A stops rounding to 2 decimals (today's
`.round(decimals=2)` on a mm/day precipitation rate is a 0.005 mm/day floor,
~0.15 mm/month). Value-changing; lands in migration step 5c.

**Acquisition window — fixed per experiment class, independent of the analysis
windows.** This is the axis risk-02 / ext1-04 identified as underspecified, and
it is where G5 lives or dies.

| Experiment class | Acquisition window |
| --- | --- |
| `historical` | `1950-01-01` … `2014-12-31` |
| any `sspNNN` | `2015-01-01` … `2100-12-31` |

These are exactly the spans `get_stats_climate_proj.py` already hardcodes as
`time_tuple_all`, lifted out of a script branch into the reducer's declared
contract. **`future_horizons` is not an input to stage A and is not a digest
component.** All analysis-window selection — reference window, horizon windows —
happens in stage B, which reads local files. Changing a horizon therefore
schedules zero reduce jobs.

Each series records `cst_acquisition_window` and its actual first/last time step.
Stage B **fails** when a requested analysis window is not fully covered by the
series' recorded acquisition coverage, naming the series and both windows.

**Cache key.** `series/*.nc` is **not** `temp()`. Each file is KB-scale; the run
that produced it may have cost hours. The reducer's `params` carry a digest over:

1. the catalog **entry name**;
2. the catalog **entry as parsed** (uri template, driver options, `data_adapter`
   unit/rename maps, placeholders) — not the catalog file;
3. the **region specification** — the region-determining subset of
   `climate_store_spec`'s params (`model_region`, `hydrography`, `basin_index`),
   plus WF2's configured buffer. Note `shared.basin.resolution` is *not*
   included: it is the wflow build resolution and is not an input to
   `parse_region_basin`, so it does not determine the store polygon;
4. the **variable spec** (§5.5);
5. the **acquisition window** for the experiment class;
6. the **reducer source hash** (below).

**The catalog file is deliberately not a declared input of `reduce_gcm_series`.**
Declaring it would make any edit anywhere in `cmip6_data.yml` re-download every
series through Snakemake's mtime trigger, defeating G5. C1's file-level freshness
boundary is a property of the *shared* store rule — three Snakefiles co-own it
and a file-level boundary is the only cheap symmetric contract available there —
and `extract_climate_grid` keeps it verbatim. The series cache is WF2-private and
can afford the finer-grained entry-level digest, which is strictly stronger where
it applies: an edit to an unused entry invalidates nothing, an edit to a used
entry invalidates exactly that series. Recorded here as a deliberate, bounded
divergence, not an oversight.

**The region edge.** `store_region.geojson` is declared as an input of
`reduce_gcm_series` under `ancient()`, exactly as today's rules 2.02/2.03 declare
`hydrology_model/staticgeoms/region.geojson`. `ancient()` supplies the DAG
ordering edge without the mtime trigger — a WF1 re-run that rewrites a
byte-identical polygon must not re-download the archive. Correctness is carried
by item 3 of the digest: the region *specification* is config, known at DAG-build
time, and fully determines the polygon. The bounds actually read at runtime are
recorded in the series attributes and in `provenance.json`, so a divergence
between spec and polygon is auditable after the fact.

**Reducer-version staleness (risk-03).** `reducer_version` is **not** a
hand-bumped constant. Snakemake's code rerun-trigger tracks the rule's script
body, not the `blueearth_cst` modules it imports, so a hand-bumped constant is a
silent-wrong-numbers path with no failure signal. Two mechanisms, both stdlib:

1. **Mechanical version.** At DAG-build time the Snakefile hashes an
   **explicitly enumerated** list of reducer module files (the stage-A script and
   the reduction/weighting helpers it imports — named in the rule's params, so
   the enumeration is reviewable) with `hashlib.sha256`, and folds the result
   into the digest. Enumerated, not "all of `blueearth_cst`", so an unrelated
   edit does not invalidate the cache.
2. **Fail-loud on stale reads.** Each series carries its digest as the
   `cst_series_digest` attribute. Stage B recomputes the expected digest for each
   series key and **raises** on mismatch, naming the series and both digests. A
   series that survived a mechanism-1 miss — hand-copied, restored from a backup,
   produced by an older checkout — fails the run instead of quietly entering the
   ensemble.

**Stage B's input set is explicit (risk-06).** `derive_change_factors` declares
**exactly the expanded `{series_key}` list built from the validated config** —
never a directory glob. It additionally asserts that the set of series it opened
equals that list. A model removed from the config therefore cannot rejoin the
ensemble through a leftover file, and stale series in `series/` become a disk
hygiene question rather than a correctness one.

### 5.4 Region and reference window

#### D1 — reference-series construction: clip, never splice (owner ruling R1)

The change-factor reference is the **GCM historical experiment**, which ends
2014-12-31 (`config/catalogs/cmip6_data.yml` resolves
`cmip6_{model}_historical_{member}` under `gs://cmip6/CMIP6/CMIP/{model}/historical/`;
2015+ exists only under the per-scenario `ScenarioMIP` entries). The design does
**not** splice historical and scenario data to fill 2015–2020 (**N8**).

- **Clip rule.** The effective reference window is
  `requested ∩ [source start, 2014-12-31]`.
- **Warning, at three named sites — per condition, not lumped.** Two distinct
  conditions surface differently, because a signal that fires on every run is a
  signal nobody reads:

  | Condition | stderr at DAG build | `provenance.json` | `report.md` |
  | --- | --- | --- | --- |
  | **Clip** — requested reference overruns 2014-12-31 | **yes**, naming requested vs effective | `reference_window_requested`, `_effective`, `reference_window_clipped: true` | disclaimer line above every change-factor figure and table |
  | **Alignment** — effective reference ≠ `shared.historical_window` | **no** by default | `shared_historical_window`, `reference_alignment: "differs"` + both windows | disclaimer line |
  | **Short window** — effective length < 20 years | **yes** | `reference_window_years` | effective length printed beside every statistic derived from it |

  The alignment difference is deliberately *not* a stderr warning by default:
  the shipped seed config (`historical_year_range: [1990, 2010]` vs
  `shared.historical_window: 2000–2020`) differs, so it would fire on 100 % of
  runs and be filtered out. R1's wording is "the disclaimer is what surfaces
  that". It is promoted to stderr only when the two windows would have been
  equal but for the clip — the case where the user plausibly *intended*
  alignment and did not get it.

- **Never raises, with one exception.** A requested reference window lying
  **entirely** after 2014-12-31 is an error — there is nothing to clip to.
- **Horizon boundary rule.** A configured horizon window is clipped the same way
  against the scenario acquisition end (2100-12-31), with the clip row's
  treatment.

**G3 is an alignment check, not a construction.** WF2 keeps its own reference
window key (`historical_year_range`, grandfathered per
`dev/conventions/naming.md`); it is **not** retired in favour of
`shared.historical_window`. At DAG build the workflow compares the effective
reference window against `shared.historical_window` and records the result per
the alignment row above, naming both windows and stating that the overlay's
reference period is not the stress-test baseline period. The owner has accepted
that they remain different periods (R1); the `provenance.json` field and the
report disclaimer are what surface it.

**The tradeoff, stated plainly.** A user who sets the reference window to the
project baseline (2000-01-01 … 2020-12-31) gets an effective GCM reference of
2000–2014 — **15 years**, which trips the short-window warning above. That is the
honest surface of R1: the clip is visible, the number is smaller than the
20-year floor this design elsewhere recommends, and the disclaimer says so.
Today's `historical_year_range: [1990, 2010]` (21 years, entirely inside the
historical experiment) triggers no stderr warning at all: no clip, no
short window, and the alignment difference goes to `provenance.json` and the
report disclaimer only.

Whether the default reference window should become 30 years — which under the
clip rule means 1985–2014 — remains **OQ-4**.

#### D2 — declaring the store when nothing reads its gridded output

Under **R4**, declaring `extract_climate_grid` buys WF2 exactly one thing: the
model-free region polygon. It also puts a full gridded observed extraction on
WF2's critical path, because **C1** forbids declaring a subset of the store's
outputs. On the seed fixture that extraction is
`climate_historical/era5_20000101_20201231/extract_historical.nc` — 7 variables,
daily, 7671 time steps over the basin bbox. Magnitude on a production basin is
source- and extent-dependent; this design asserts no wall-clock figure.

Alternatives are enumerated in §6.4. **Selected: A1 — declare the full store and
accept the cost**, because:

- The gain is real and is the change request's most user-visible one: WF2 becomes
  runnable with no Wflow build at all, from `shared.basin` + catalogs.
- On any project where WF1 has run with the same `shared.clim_historical` +
  `shared.historical_window`, the store already exists and the rule is not
  scheduled (§5.2: 15 jobs fresh, 14 with the store present). WF2 is documented
  to run after WF1, so the cost is paid on the projections-only path only.
- The cost is confined to the **store rule**. No stage-A, B, or C rule reads
  `extract_historical.nc`, so an observed-source read failure blocks the run at a
  single identifiable rule and cannot corrupt or silently alter a change factor.
- The store is the precondition for the observed analysis R4 defers and for the
  gridded work R2 preserves, so on the intended trajectory it is not dead cost.
- A user who wants only the polygon can build the store once
  (`snakemake extract_climate_grid …`) and never pay again. There is no supported
  way to obtain the polygon without the gridded file, because C1 fixes the output
  set.

**Named fallback: A2** — keep reading `hydrology_model/staticgeoms/region.geojson`
and drop G2 from v2.0. The discriminating question is the owner's:
*is a first-run gridded observed extraction acceptable on a projections-only run?*
If it is not, A2 is the correct answer and v2.0 ships efficiency and monthly
factors without the WF1 decoupling.

**Region-equality evidence (2026-07-29, seed fixture).** The two polygons have
identical bounds `[9.658333, 0.35, 9.858333, 0.483333]`, and WF2 consumes only
`geom.geometry.bounds`, so on this fixture the swap selects the identical GCM
cell set and cannot move a number. This is a fixture-level result; the migration
gate in §9 re-checks it rather than assuming it generalizes to a basin whose
hydromt_wflow delineation at `resolution: 0.00833` diverges from
`parse_region_basin`.

### 5.5 Variable specification — declaring the quantity, not the aggregator

Today `get_stats_climate_proj.py` dispatches on the literal string `"precip"`
(sum) versus everything-else (mean), and `get_change_climate_proj.py` dispatches
the same way for multiplicative versus additive change.

**Verified 2026-07-29:** the catalog sources are CMIP6 **`Amon`** — monthly
means, with `pr` renamed to `precip` and converted to mm/day (`unit_mult: 86400`)
and `tas` renamed to `temp` and converted to °C (`unit_add: -273.15`). With one
sample per month, `resample("MS").sum()` and `.mean()` return the identical
value: **the monthly aggregation dispatch is a no-op on `Amon` input.** The real
split lives only in the change arithmetic downstream, and
`dev/workflows/climate_projections.md` describes the dispatch key as controlling
something it does not control.

design-v1 proposed replacing the name dispatch with an explicit
`aggregate: sum|mean`. Round-1 review (ext1-03) showed that `aggregate:` is
exactly the wrong axis: summing daily precipitation yields mm/month while summing
`Amon` preserves mm/day, so a variable-level aggregator cannot describe two
source frequencies. R4 removes the daily source from v2.0, but the spec should
not encode an axis that will not generalize. The spec therefore declares the
**canonical quantity** and the **change semantics**, not the aggregator:

```yaml
variables:
  precip: {source: precip, canonical: rate,  units: mm/day, change: relative}
  temp:   {source: temp,   canonical: state, units: degC,   change: absolute}
```

- `source` names the **post-rename** variable, because the catalog's
  `data_adapter.rename` maps `pr → precip` and `tas → temp` before the reducer
  sees the data.
- `canonical` states what the stored monthly series **is**: a monthly mean *rate*
  in the declared units, or a monthly mean *state*. Conversion from a source's
  native frequency to the canonical quantity is a property of the **source**, not
  the variable.
- In v2.0 there is exactly one source frequency (`Amon`, monthly mean), so that
  conversion is the identity — and it is **asserted, not inferred**: the reducer
  checks the decoded time axis is monthly and **raises** naming the source
  otherwise. A future daily branch adds a source-level conversion at that same
  assertion point (§5.10, S1) without touching the variable spec.
- `change` is read by stage B (§5.6). Nothing infers anything from a name.

This removes the silent-wrong-answer path, makes the workflow extensible to the
other variables the catalog already renames (`rsds → kin`, `psl → press_msl`),
and puts the frequency question where ext1-03 showed it belongs.

### 5.6 Change factors — formulas, edge cases, ensemble

Stage B is one job, no network, reading `series/*.nc` through the explicit
expanded list of §5.3.

**Notation.** For source *s* (a resolved `{series_key}`), variable *v*, and a
window *W* (the effective reference window *R* or an effective horizon window
*H*): `x̄(s,v,W)` is the month-length-weighted mean of the monthly series over
*W*, and `x̄ₘ(s,v,W)` is the mean over the month-of-year *m* ∈ 1…12 within *W*.

**Formulas.**

| `change:` | Annual | Monthly (per *m*) | Units |
| --- | --- | --- | --- |
| `absolute` | `x̄(s,v,H) − x̄(s,v,R)` | `x̄ₘ(s,v,H) − x̄ₘ(s,v,R)` | the variable's `units` |
| `relative` | `100 × (x̄(s,v,H)/x̄(s,v,R) − 1)` | `100 × (x̄ₘ(s,v,H)/x̄ₘ(s,v,R) − 1)` | `%` |

**Calendar-aware weighting.** `x̄` weights each month by its length in the
model's own calendar, taken from the decoded `cftime` axis (the catalog sets
`decode_times: true`). For a 360-day-calendar model every month is 30 days and
the weights are uniform; for standard and no-leap calendars they are not. This
corrects v1's noted ~0.3 % month-length error and makes annual means comparable
across models with different calendars rather than differing for procedural
reasons. The calendar name is recorded per series; stage B raises on a calendar
it cannot weight.

**Dry-month / near-zero denominator rule (risk-05, ext1-05).** Relative change is
undefined-in-practice when the reference month is near zero — a well-known
delta-method failure that the annual product largely avoids and that a monthly
product walks straight into on any basin with a dry season. Rule:

- Config key `relative_change.min_reference`, per variable, in the variable's
  canonical units. **Default value is OQ-9** — the rule below is complete and
  testable regardless of the number.
- When `x̄ₘ(s,v,R) < min_reference`: emit `value = NaN`,
  `status = "reference_below_threshold"`, and the corresponding **absolute**
  change in an `absolute_value` column, so the information is not lost.
- The report renders flagged months as gaps with a footnote naming the rule and
  the threshold.
- The S5 grid-vs-cloud advisory (§5.10) excludes flagged months from its envelope
  comparison rather than treating a `+4000 %` artifact as signal.
- A `(dataset, scenario, member, horizon, variable)` whose flagged-month count
  exceeds `relative_change.max_flagged_months` is itself flagged in the summary
  table.

**Coverage and partial-year policy (ext1-05).**

- Aggregation uses **complete hydrological years only**, where the hydrological
  year starts at `start_month_hyd_year`. A partial first or last year is dropped
  and the count of dropped years is recorded per series and window.
- A series must have **every month present** in the effective window. `Amon` zarr
  stores are contiguous, so a gap indicates a source problem; the run **fails**
  naming the series and the missing months (consistent with D4 below), rather
  than averaging a short month set.

**Statistics.** Today's eight (`mean, std, var, median, q_90, q_75, q_10, q_25`)
over a ~20-year window make `q_90` effectively the second-highest of 20 values.
v2.0 emits `mean`, `median`, `std` by default; the tail quantiles are opt-in and,
when emitted, are labelled with their effective sample size in both the CSV and
the report. Value-changing; migration step 5d. The *window-length* default remains
**OQ-4**.

**Ensemble treatment (owner ruling R3).**

- **Sampling unit is the unique model** (`dataset`, the CMIP6 source_id).
- **Members are averaged within a model first**, with equal weight per member,
  producing one series per (model, scenario) for model-level summaries. Model-level
  summaries then weight each model equally, so adding members to one model cannot
  give it disproportionate influence.
- **Members are shown hierarchically, never collapsed away.** Model is the
  primary grouping in every figure and table; member is a nested trace. Every
  member-level value is present in `change_factors/{annual,monthly}.csv`.
- **No aggregate envelope below the threshold.** Percentile bands across models
  are emitted only when the count of **unique models** with a resolved series for
  that (scenario, horizon) is at least `ensemble.min_models_for_envelope`
  (default 10, carried from design-v1). Below it the report emits individual
  model and member traces plus the min–max range, labelled as the range of a
  small ensemble — and no percentile envelope, no ±σ band.
- The report states composition explicitly: unique models, members per model,
  institutions and their model counts, and the count actually used for each
  summary. On the seed ensemble that reads: 3 unique models, 1 member each,
  2 institutions (INM contributes 2 of 3 models), envelope suppressed.
- Institution-level de-duplication and performance weighting are **not applied**
  (**N9**); the report gives a reader what they need to judge the ensemble.

### 5.7 Source resolution and failure semantics

**DAG-build validation (G7).** `config/catalogs/cmip6_data.yml` declares templated
entries (`cmip6_{model}_{scenario}_{member}`) with a `placeholders:` block listing
valid models and members. Existence of a requested source is therefore a **pure
YAML lookup** — no hydromt import, no network — cheap enough to run on every
dry-run. The `{series_key}` fan-out is built from the validated list; absent
combinations never become jobs. This deletes the dummy-empty-netCDF pattern,
`filter_nonempty`, and the three "did this file have data?" loops in the plotting
code.

**Keeping the validator from drifting (risk-08).** The validator is a second
implementation of something hydromt already does, so:

- its logic stays **minimal** — exact entry-name template match plus placeholder
  membership, nothing else;
- an entry containing a construct the validator does not model (variants,
  aliases, or any unrecognized top-level key on an entry it is asked about) is an
  **error naming the key**, not a silent accept or reject — drift becomes visible
  rather than wrong;
- one integration-marked test cross-checks the validator's accept list against
  `hydromt.DataCatalog(data_libs=…).sources` for representative entries (§9), so
  drift is caught by CI-adjacent tooling rather than by a user.

#### D4 — runtime source failure: fail-fast

ext1-02 established that design-v1's middle position is not implementable: a
`reduce` job that fails leaves its declared output absent, Snakemake halts, and
no downstream rule can write the failure record, enforce a minimum-source count,
or continue with survivors. Two coherent contracts exist (§6.8); v2.0 takes **fail-fast** (§4 criterion 5 —
a gate must be executable where it is placed):

- A `reduce_gcm_series` job that cannot read its source **raises**. Snakemake
  halts. The failing series key and the exception are in the job log and in the
  merged stage log.
- **No** dummy netCDF, **no** empty dataset, **no** silent ensemble shrink. A run
  that completes used exactly the validated source set.
- The **minimum-source check moves to DAG build**, where it can execute. Note
  what fail-fast does to this check's meaning: "fewer than N sources *resolved*"
  can no longer happen at runtime, because a resolution failure is now a run
  failure. The only question left is "did you configure enough models for a
  meaningful ensemble?", which is a **configuration** judgement, not a runtime
  one. Accordingly `ensemble.min_sources` is asserted against the validated
  source list before any job runs; a shortfall fails the DAG naming the resolved
  and missing sources. **Its default is `1`** — i.e. effectively unchecked —
  because any higher default would fail DAG build on the shipped seed config
  (3 models × 2 scenarios), which is also the tracked baseline seed. It exists
  for a user who wants a floor enforced, not as a policy this design imposes.
  Ensemble *adequacy* is signalled separately and non-fatally by
  `ensemble.min_models_for_envelope` (§5.6), which suppresses envelopes rather
  than failing the run. The two keys are not interchangeable: one is a hard
  configuration floor, the other a reporting threshold.
- `provenance.json` therefore describes the composition of a **successful** run.
  It is written by stage C, and by construction a run that produced it had every
  validated source resolve. It records resolved sources, not a failure ledger.
- **`--keep-going` still helps.** `scripts/run_workflows.py` already invokes WF2
  with `--keep-going`; under fail-fast that means all independent reduce jobs
  still run and every failure is reported in one pass — only the downstream
  stages are skipped. The user gets the full failure picture without the artifact
  contract having to model partial success.
- **Retry is cheap** because the series cache is persistent: a re-run re-derives
  only the sources that failed.

Rationale against the tolerant alternative is in §6.8; the evidence that would
reopen it is in **OQ-11**.

### 5.8 The optional gridded branch (owner ruling R2)

`save_grids` is **retained, default `false`**, and becomes a first-class branch
with **declared** outputs, replacing today's undeclared, params-passed file layer.

Today three file families are written but never declared, and are passed between
rules through `params`:

| Today (undeclared, via `params`) | v2.0 (declared) |
| --- | --- |
| `historical_stats_{model}.nc` (`params.stats_path_hist`) | `grids/monthly_climatology_{series_key}.nc` |
| `stats-{model}_{scenario}.nc` (`params.stats_path`) | `grids/monthly_climatology_{series_key}.nc` |
| `monthly_change_mean_grid-{model}_{scenario}_{horizon}.nc` (`params.change_grids`) | `grids/change_{dataset}_{scenario}_{member}_{horizon}.nc` |

Mechanism and properties:

- `save_grids` is a config value read when the Snakefile is **parsed**, so the
  extra entries are appended to the rules' `output:` lists at parse time. The DAG
  is fully determined before any job runs; no checkpoint or conditional-output
  machinery is needed.
- The gridded climatology is written from the **same network read** as the
  series, inside the same stage-A job. The gridded change fields are computed in
  the same stage-B job as the tabular change factors, expanded over
  (dataset, scenario, member, horizon). **`save_grids: true` therefore adds no
  jobs and no additional network access** — it adds disk and declared outputs.
- Grids are **not** `temp()`.
- `save_grids` does **not** enter the series digest: a series is byte-identical
  either way, and flipping the flag on re-derives the stage because the newly
  declared grid outputs are missing, which is Snakemake's normal mechanism.
- **Grids are an archive, not an analysis input in v2.0.** No v2.0 product reads
  them. That keeps the branch first-class — declared, gated, and covered by an
  existence test — without letting an optional artifact into the change-factor
  path, where it would make the tabular products depend on a flag.

This resolves ext1-09's actual point: stage A discards spatial dimensions, so
stage B could not have reconstructed the grids from `series/`. The grids are
produced where the spatial dimensions still exist.

### 5.9 Report stage

One job producing:

- `change_factors/{annual,monthly}.csv` — long format, one row per
  `(dataset, institution, scenario, member, horizon, period, variable, statistic)`
  with `value`, `absolute_value`, `units`, `status`, `reference_window`,
  `horizon_window`, `n_years`, `n_models_in_summary`.
- `change_factors/change_factors.nc` — the same content with coordinate metadata.
- `report.md` — the ΔT/ΔP cloud, the seasonal change pattern, timeseries context,
  the ensemble-composition table, and a **disclaimer block** carrying: requested
  vs effective reference window and whether it was clipped; the alignment result
  against `shared.historical_window`; the effective window length and any
  short-window warning; the spatial weighting scheme and its approximation label;
  the dry-month rule and threshold; and the envelope-suppression state.
- `provenance.json` — resolved sources (entry name, digest, acquisition window,
  actual coverage, calendar, grid-geometry check result, weighting scheme);
  requested and effective reference window plus the clip flag;
  `shared.historical_window` and the alignment-check result; horizon windows;
  reducer version hash; config digest; variable spec; ensemble composition;
  flagged months.
- `plots/*.png` — **all declared**. Today 6 of 8 figures are undeclared, one is
  saved without an extension, and `gcm_timeseries.nc` is declared under the label
  `timeseries_csv` (confirmed in `Snakefile_climate_projections` rule 2.06; see
  also the overview doc §2/§4).

The CSV tables are separate declared outputs and do not depend on the report
stage, so a report-generator failure does not leave the run without numbers.

**The legacy summary artifacts stay.** `summary/annual_change_scalar_stats_summary.{nc,csv}`
and `summary/annual_change_scalar_stats_summary_mean.csv` remain declared outputs
at their current paths throughout v2.0. They are the migration's evidence anchor
(§8): they are the only WF2 numbers the baseline manifest pins strictly, and
moving or retiring them would leave steps 1–4 self-matching rather than
evidenced. Superseding them with the `change_factors/` tables is a follow-up that
must carry its own manifest re-record; it is not part of this design.

### 5.10 Extension surface — contract changes, not free reads (G9)

design-v1 claimed every slot was "a read, not a pipeline". ext1-06 falsified that
and R4 narrowed the claim. Each remaining extension is listed with **the contract
change it requires**, so a future implementer is not surprised:

| Slot | What it adds | Contract change required |
| --- | --- | --- |
| **S1 — observed climatology & trends** | Long-term means, seasonality, annual trends | A second source class at stage A with a **source-level frequency conversion** at §5.5's assertion point (daily → canonical monthly rate/state, unit handling for temp in K); a `provenance` axis in the series key and cache; possibly an acquisition window longer than `shared.historical_window`, which the store key fixes |
| **S2 — multi-dataset observed comparison** | ERA5 vs CHIRPS vs E-OBS over the same basin | S1, plus more than one store instance — stage 0 builds exactly the one `shared.clim_historical` store — so either a multi-source store spec (a **C1-scope change**, co-owned with WF1/WF3) or a WF2-private observed acquisition rule |
| **S3 — GCM historical bias diagnostics** | GCM-historical vs observed climatology (diagnostic only; delta-change already cancels bias) | S1, plus a resolution-reconciliation step between a ~1–2° `Amon` grid and a fine observed grid, and a decision on what "comparable" means spatially |
| **S4 — extremes / indicator set** | Wet/dry spells, hot days, seasonality indices | A **daily CMIP6 acquisition branch** (`day` tables, ~30× volume), a temporal-resolution axis in the series key, cache, and layout, and probably a new dependency for standard indices (**OQ-7**, C5) |
| **S5 — grid-vs-cloud advisory** | Does the configured WF3 perturbation grid envelope the projected cloud? Emits a figure and a warning | **None.** A read of `change_factors/*` plus the WF3 config section, one-way. Never writes WF3 config (N1/C2). Excludes dry-month-flagged values (§5.6) |

S5 is the only slot that costs no contract change, and it is the highest-value
one. S1 is the gateway to S2 and S3; the v2.0 decisions that keep it cheap are
the `canonical:` variable spec (§5.5) and the source-level assertion point.

---

## 6. Alternatives considered

### 6.1 Keep the current rule-per-combination fan-out; add caching only

Cheapest change: make the series persistent, leave the 11-rule structure alone.
**Not chosen** — it fixes G5 but not G1/G4/G6/G7, and the structure is what
prevents any second source class from entering at all. Preferable if the only
complaint were re-run cost.

### 6.2 Move `plot_climate_source` (WF1 rule 1.15) into WF2

Superficially tidier: all climate figures in the climate workflow.
**Not chosen.** `tests/test_plot_climate_source.py` pins the P3-2a P4 assertion —
those three figures build with neither `hydrology_model/` nor
`config/templates/wflow_build_model.yml` on disk. Relocating the rule invalidates
a sealed milestone's acceptance gate for a cosmetic gain (criterion 3). Under
**R4/N7** it would also be the only WF2 rule reading observed data, contradicting
the scope narrowing. Preferable only if WF1 were being restructured anyway.

### 6.3 A standalone 4th Snakefile versus extending `Snakefile_climate_projections`

**Deferred (decision D5, recommendation: extend in place for v2.0).** P3-2a
explicitly deferred the standalone entry point and the roadmap lists it in the
Phase-4 candidate pool. Extending in place keeps the platform surface at three
entry points and the `run_workflows.py` `enabled:` contract unchanged. **R4
strengthens this**: with scope narrowed to GCM projections, the workflow's name
still describes it, so the rename loses its motivation for v2.0. Preferable once
observed analysis lands and the name stops matching. See **OQ-1**.

### 6.4 Region source — the D2 decision under R4

Four options, since R4 removed the analytical payoff v1 assumed:

- **A1 — declare the full `climate_store_spec`, accept the cost. SELECTED.**
  Preserves G2, keeps one delineation for the project, no-op when WF1 has run.
  Cost: a fresh projections-only run pays the gridded observed extraction and
  inherits its network failure surface, for a polygon. Reasoning in §5.4.
- **A2 — keep reading `hydrology_model/staticgeoms/region.geojson`. NAMED
  FALLBACK.** Drops G2 from v2.0 and defers the store declaration to whenever
  observed analysis lands and actually consumes the gridded file. Touches C1 not
  at all — WF2 simply does not declare the spec. Chosen if the owner answers "no"
  to §5.4's discriminating question.
- **A3 — a separate WF2-private region-only producer** writing just the polygon
  from `shared.basin` + catalog. **Not chosen.** On C1's letter this is *not* an
  asymmetric `climate_store_spec` declaration — it is a different rule at a
  different path — so the constraint's text does not forbid it. But it creates a
  second delineation code path that can drift from `store_region.geojson`, which
  is precisely the bug class C1 exists to prevent; and two rules that must agree
  about a polygon is the same failure mode as two rules that must agree about an
  input set. Rejected on the constraint's purpose, not its wording.
- **A4 — an explicit `region:` config key.** Adds a config surface for something
  the store computes model-free, and lets WF2's region drift from the one WF1/WF3
  extract against. Preferable only if WF2 must run over a region the store was
  not built for.

### 6.5 Reference series: splice historical + scenario for 2015–2020

The common delta-method convention when the baseline overruns the historical
experiment. **Rejected by owner ruling R1**, and the reasons are worth recording:
the spliced reference is **scenario-dependent** (which SSP fills 2015–2020?), so
each scenario would get a different baseline and the change factors would no
longer share one reference; it requires overlap, gap, and calendar reconciliation
between two experiments; and it modifies the interpretation of a run, which the
owner ruled out (N8). The clip rule (§5.4 D1) keeps a single common reference at
the price of a shorter window, surfaced by warnings.

### 6.6 Catalog generation via `intake-esm` instead of hand-enumerated entries

`cmip6_data.yml` hand-lists 23 models for the historical entry and smaller lists
per scenario. A query-based catalog would remove that maintenance and enable
ensembles large enough for the percentile envelopes §5.6 currently suppresses.
**Not chosen for v2.0** — new dependency (C5, N4). Recorded as an ask in §10.

### 6.7 Long-format Parquet instead of netCDF for `series/`

**Not chosen** — netCDF is the repo's stated interchange format across
R/Python/Julia, and the series carry coordinate metadata that survives the
netCDF round-trip. The change-factor *tables* are CSV, which covers the
tidy-consumer case.

### 6.8 Failure-tolerant status-artifact contract instead of fail-fast (D4)

Every reduce job emits an **always-written required status artifact** plus an
**optional data artifact**; stage B discovers the surviving data through a
Snakemake `checkpoint`; the minimum-source check runs between the checkpoint and
stage B. This is implementable, and it is the design ext1-02's `suggested_fix`
points at. **Not chosen for v2.0** because:

1. It adds a checkpoint and a second artifact class to a workflow whose whole
   point in this revision is fewer moving parts (11 rules → 8).
2. It makes ensemble composition depend on transient network state: two runs of
   the same config produce different change factors with no config difference and
   no error. That is a reproducibility defect, and the tolerant design has to
   carry provenance machinery specifically to make it auditable.
3. `--keep-going`, which WF2 is already invoked with, delivers the "see all
   failures in one pass" benefit without changing the artifact contract.

Revisitable; the evidence that settles it is **OQ-11**.

### 6.9 Retire `save_grids`

The simplest resolution of OQ-8: delete the branch, document the lost
functionality, migrate. **Rejected by owner ruling R2** — the raw grids are wanted
for advanced analysis at a later stage. §5.8 takes the other option and makes the
branch declared.

### 6.10 Relocate WF2 output to `{project_dir}/climate/` (D3)

design-v1 raised this as OQ-3 while step 3 simultaneously consumed it — the
sequencing contradiction risk-04 named. **Decision D3: keep
`{project_dir}/climate_projections/{clim_project}/` in v2.0.** The rename's
motivation was that "climate_projections" would understate a workflow that also
did observed analysis; **R4 removes that**, because v2.0 is projections analysis
and the existing name describes it. Keeping the root also keeps every
manifest-pinned path byte-identical through the migration (§8), which is what
makes the value-neutral steps evidenced rather than self-matching. The rename
becomes a follow-up tied to whenever observed analysis lands, alongside D5.
Owner-reversible; reversing it re-opens the manifest-evidence problem, which §8
would then have to solve with an explicit old→new path map.

---

## 7. Consequences and risks

**Observable consequences (falsifiable).**

1. WF2 runs to completion with no `hydrology_model/` directory on disk.
2. `snakemake -n` on the seed config lists **15** jobs on a fresh `project_dir`
   and **14** when `climate_historical/era5_20000101_20201231/` already exists —
   against **22** today. Counts exclude the `all` target job (§5.2).
3. A second run with a changed `future_horizons` entry schedules **zero**
   `reduce_gcm_series` jobs and performs zero network reads; wall clock drops to
   derive + report.
4. Editing `config/catalogs/cmip6_data.yml` in a way that does not change a
   *used* entry schedules zero reduce jobs; editing a used entry re-derives
   exactly that entry's series and no other.
5. Editing an enumerated reducer module re-derives **every** series without any
   manual version bump; editing an unrelated `blueearth_cst` module re-derives
   none.
6. A series file whose `cst_series_digest` attribute does not match the expected
   digest makes stage B **fail**, naming the series and both digests.
7. A config naming a model absent from the catalog fails at **DAG build** with a
   message naming the model. **User-visible behavior change**, and the most
   likely source of "it used to work" reports.
8. A remote read failure halts the run with the failing series key named. No run
   produces a smaller-than-configured ensemble. **User-visible behavior change** —
   today the 3-model ensemble silently becomes 2.
9. A reference window ending after 2014-12-31 produces a DAG-build warning naming
   requested vs effective window, `reference_window_clipped: true` in
   `provenance.json`, and a `report.md` disclaimer. It never raises. A reference
   window lying entirely after 2014-12-31 raises.
10. `change_factors/monthly.csv` exists with 12 rows per
    `(dataset, scenario, member, horizon, variable, statistic)`.
11. A monthly relative change whose reference month is below
    `relative_change.min_reference` has `value = NaN`,
    `status = "reference_below_threshold"`, and a populated `absolute_value`.
12. With fewer than `ensemble.min_models_for_envelope` unique models, no
    percentile envelope or ±σ band appears in any figure; individual traces and
    the min–max range do.
13. `provenance.json` names every resolved source, both reference windows, the
    alignment-check result, the weighting scheme, and the ensemble composition.
14. Every figure WF2 writes is a declared output.
15. `save_grids: true` adds declared `grids/*.nc` outputs and **no additional
    jobs** (still 15/14 on the seed config).
16. A source whose latitude/longitude coordinates are not 1-D and monotonic makes
    the reducer **raise**, naming the source.

**Risks.**

- **R1 — the third `climate_store_spec` declaration diverges (C1).** Mitigation:
  the declaration is generated from the shared helper, and a test asserts the
  three declarations produce identical input sets (§9).
- **R2 — dropping the hist→fut ordering edge exposes a directory race.** Today's
  `get_stats_climate_proj.py` uses `os.mkdir` guarded by `os.path.exists`;
  concurrent reduce jobs would race. This is the likely original reason for the
  ordering edge. Mitigation: `makedirs(exist_ok=True)`.
- **R3 — the methodological changes move every number.** Weighting, calendar,
  rounding, statistic set, coverage policy, ensemble treatment, and possibly
  window length. Mitigation: **one cause per commit** (§8 steps 5a–5e, 6b, 6c),
  each with its own re-record, so per-cause attribution is a property of the
  commit boundaries rather than of machinery the design would otherwise have to
  specify. Bundling them would leave §9's characterized-diff gate unable to
  execute (§4 criterion 5).
- **R4 — persistent series accumulate.** Correctness is no longer at stake
  (explicit input list + digest assertion, §5.3), so this is disk hygiene:
  stale-key pruning is a follow-up.
- **R5 — fail-fast turns transient network flakiness into a failed run.**
  Mitigation: the cache means a retry re-derives only what failed, and
  `--keep-going` surfaces all failures in one pass. Falsifiable trigger for
  revisiting: if observed remote-read failure rates make long runs impractical,
  §6.8's tolerant contract is the recorded fallback (**OQ-11**).
- **R6 — the report becomes the only place a number is stated.** Mitigation: the
  CSV and netCDF tables are separate declared outputs independent of the report
  stage.
- **R7 — a first WF2 run on a fresh project pays the climate-store build.**
  "WF2 no longer needs WF1" reads as a cost reduction; on a fresh
  projections-only project it is a **cost transfer** (§5.4 D2). Mitigation: no-op
  when WF1 has run with the same store key; the store is not on the analysis
  path, only the polygon is; the store can be pre-built once.
- **R8 — the geometry check converts a silent bias into a blocked source.** A
  non-1-D-rectilinear `Amon` source that today is (wrongly) averaged unweighted
  will, under §5.3, refuse to reduce. This design does not assert how many
  catalog models that affects — the check-and-fail contract does not require
  knowing. The failure is loud and names the source, so the cost is a visible
  blocked model rather than a hidden bias.
- **R9 — the DAG-build validator drifts from hydromt.** Mitigation: minimal
  logic, unknown constructs are errors, and an integration-marked cross-check
  test (§5.7, §9).
- **R10 — the baseline manifest pins a verbatim sha256 of the seed config file**,
  so any commit that adds or renames a config key fails that target even when
  every number is identical. Mitigation: §4 criterion 1's value-neutral /
  manifest-clean split; steps that add config keys ship with a re-record of the
  config target alone and the config diff shown (§8).

---

## 8. Migration + commit plan

### What the baseline manifest actually pins

`dev/baseline/manifest.json` pins exactly **7** WF2-relevant targets. **Six** are
under `climate_projections/cmip6/`; the seventh is the config snapshot under
`config/runs/`:

| Target (relative to `test_case/test_local/`) | Comparator |
| --- | --- |
| `climate_projections/cmip6/plots/precipitation_anomaly_projections_abs.png` | exists + `size_bytes` |
| `climate_projections/cmip6/plots/projected_climate_statistics.png` | exists + `size_bytes` |
| `climate_projections/cmip6/plots/temperature_anomaly_projections_abs.png` | exists + `size_bytes` |
| `climate_projections/cmip6/summary/annual_change_scalar_stats_summary.csv` | sha256 |
| `climate_projections/cmip6/summary/annual_change_scalar_stats_summary_mean.csv` | sha256 |
| `climate_projections/cmip6/summary/annual_change_scalar_stats_summary.nc` | per-variable summary statistics |
| `config/runs/snake_config_climate_projections.yml` | sha256 |

Three facts follow, and they are what risk-04 was about:

1. **Coverage is thin.** No monthly intermediates, no per-model stats, no
   `timeseries/gcm_timeseries.nc`. A green `check_baseline` constrains the final
   **annual scalar summary** and three PNG file sizes — nothing about the
   intermediates that steps 2–4 restructure. "check_baseline passed" is therefore
   weaker evidence of value-neutrality than design-v1 implied, and the plan below
   does not lean on it alone.
2. **The config snapshot is a verbatim copy of the seed config file**
   (`copy_config_files.py` reads `config_snake` and writes it unchanged). Its
   sha256 is the hash of `config/workflows/snake_config_model_test.yml`. Any
   commit that adds, removes, or renames a config key breaks that target *even
   when no number moves* — hence §4 criterion 1's value-neutral vs
   manifest-clean split.
3. **A rename severs the evidence chain.** If a step moves a pinned path,
   `check_baseline` either fails on a missing path or is re-recorded from the new
   run — and a re-record under a rename trivially matches itself. **D3** (§6.10)
   removes the problem for v2.0 by keeping the WF2 root and every pinned path
   byte-identical; no step below renames a manifest-pinned path.

### Commit plan

Sequenced so every value-neutral step lands before the first value-changing one.

§5.2's 8-rule set and 15/14 job counts describe the **end state**. Intermediate
commits carry a transitional rule set: in particular the existing
`plot_climate_proj_timeseries` rule survives until step 7, and
`projected_climate_statistics.png` moves from `monthly_change_scalar_merge` into
`derive_change_factors` at step 4 **keeping its path**. The `snakemake -n`
comparison in §9 is run per commit against that commit's expected set, not
against the end-state numbers.

| # | Commit | Value-neutral? | Manifest-clean? | Gate |
| --- | --- | --- | --- | --- |
| 1 | Declare `extract_climate_grid` in WF2 from `climate_store_spec` (D2/A1); read `store_region.geojson` under `ancient()` | **Yes** (bounds measured identical on the fixture, §5.4) | Yes | `check_baseline`; re-verify bounds equality; identical-input-set test (R1) |
| 2 | Persistent `series/`: drop `temp()`, add the entry+region+module digest and the `cst_series_digest` attribute, `makedirs(exist_ok=True)`, drop the hist→fut ordering edge, fix the acquisition-window contract | **Yes** (same values, different lifetime) | Yes | `check_baseline`; second run schedules zero reduce jobs; horizon-change and catalog-edit cache tests |
| 3 | Collapse `monthly_stats_hist`/`monthly_stats_fut` into `reduce_gcm_series` over `{series_key}`; `members` becomes a wildcard; collapse the three log gathers into one. Intermediate filenames change; **no manifest-pinned path moves** | **Yes** | Yes | `check_baseline`; `pytest tests/test_cli.py`; `semantic_tree_diff` with an explicit old→new map for intermediates |
| 4 | Collapse `monthly_change` + `monthly_change_scalar_merge` into `derive_change_factors`; DAG-build source validation + `min_sources`; delete the dummy-netCDF path and `filter_nonempty`; fail-fast (D4); explicit expanded input list + digest assertion | **Yes** for present sources; **behavior change** for absent/failing ones (consequences 7, 8) | Yes | `check_baseline`; validator unit tests + integration cross-check; stale-series digest test |
| 5a | Cos-latitude weighting + the 1-D/monotonic geometry check | **No — value-changing** | Yes | Re-record; diff **is** the weighting effect; grid-geometry tests |
| 5b | Calendar-aware month-length weighting on annual aggregates | **No — value-changing** | Yes | Re-record; diff is the calendar effect; 360-day vs standard synthetic tests |
| 5c | Drop the stage-A 2-decimal rounding | **No — value-changing** | Yes | Re-record; diff is the rounding floor |
| 5d | Default statistic set (`mean`, `median`, `std`; tail quantiles opt-in and sample-size-labelled) | **No — output-set change** | **No** — the summary CSVs lose columns | Re-record of the two summary CSVs with the column diff shown |
| 5e | Variable spec (`canonical`/`change`); reference-window clip + per-condition warnings + alignment check | **No — value-changing** if OQ-4 moves the window; otherwise output-neutral on the seed config (`[1990, 2010]` needs no clip) | **No** — adds config keys → config-target re-record | Reference-window tests; property tests; **OQ-4 sign-off** if the window default changes |
| 6a | Monthly change-factor table; tidy CSV schema; `provenance.json` | Additive | No | Schema and row-count tests |
| 6b | Dry-month rule + coverage/partial-year policy | **No — value-changing** (partial years now dropped) | No | Near-zero-reference, missing-month, partial-year synthetic tests |
| 6c | Ensemble treatment per R3 (unique-model unit, within-model member averaging, envelope suppression) | **No — value-changing** for every ensemble summary | No | Ensemble composition tests; re-record of `..._summary_mean.csv` |
| 7 | Report stage; declare every figure; declare the optional gridded branch (R2); retire the loose-PNG set | Additive + plot-set change | **No** — the three pinned PNGs change or disappear | Visual QA; migration note; re-record of the three PNG targets with the old/new figure set shown; `save_grids: true` declares-not-adds-jobs dry-run |

Steps 1–2 remain independently shippable: together they deliver G2 and G5 without
touching a single computed value.

**Why 5 and 6 are decomposed.** design-v1 bundled five independent
value-changing items into one step and gated it on "a diff attributed per
cause". That gate cannot execute — attributing a single diff to weighting versus
calendar versus rounding would need a flag matrix this design does not specify
(§4 criterion 5). Splitting the step makes each sub-commit's diff *be* its cause,
at the price of five re-records instead of one. Sub-steps 5a–5c are
manifest-clean because they change numbers inside the pinned summary files
without changing paths or columns; 5d and 5e are not, for the reasons in each
row.

**Log paths.** Collapsing three `gather_*_logs` rules into one (step 3) changes
which log files rule `all` requires — `logs/2.02_monthly_stats_hist.log`,
`2.03_monthly_stats_fut.log`, and `2.04_monthly_change.log` are replaced by the
single gathered series log. Log paths follow the rule declarations, not a
filename convention (per commit `1c3013c`). **No baseline manifest target is a
log file** (§8), so this is not a manifest concern; it is a change to rule
`all`'s input list and to `dev/workflows/` documentation.

**Derived-artifact re-check.** The existing task brief
`dev/working/2026-07-29_wf2-v2-decouple-and-cache.md` covers steps 1–2 and
**must be re-checked before dispatch** against three revision-2 changes: D2
(the store declaration is retained and its cost is now an explicit accepted
tradeoff), the step-2 digest contract (entry-level rather than file-level;
mechanical module hash; digest written as a series attribute), and the
`ancient()` treatment of `store_region.geojson`.

---

## 9. Validation plan

**Per-commit gates.**

- `pytest tests/test_cli.py` — dry-runs all three Snakefiles (cheapest DAG check).
- `pytest tests/` — full suite; must stay green and additive.
- `snakemake -n` on the seed config before and after each structural commit, with
  the job count and rule set compared explicitly. The **test derives the expected
  count from the resolved source manifest**, never from a hard-coded literal;
  §5.2's 15/14 are illustrative values for the seed config, not the assertion.
- `dev/scripts/check_baseline.py check` on `test_case/test_local` for steps 1–4.
  **CI cannot run this** (C4) — local gate.
- `dev/scripts/semantic_tree_diff.py` on the WF2 output subtree, with an explicit
  old→new path map for renamed intermediates in step 3.

**Targeted checks.**

- **Region equality (step 1).** Re-run the bounds comparison of
  `store_region.geojson` versus `hydrology_model/staticgeoms/region.geojson` and
  assert the buffered bbox selects the same GCM cell set. If they diverge on a
  future basin, step 1 is reclassified as value-changing.
- **Identical input sets (R1).** A test asserting the three `climate_store_spec`
  declarations produce byte-identical input lists.
- **Cache correctness (step 2).** Six cases: (a) run, touch nothing, re-run →
  zero reduce jobs; (b) change a `future_horizons` entry → zero reduce jobs;
  (c) edit an unused catalog entry → zero reduce jobs; (d) edit a used entry →
  exactly that series re-derives; (e) edit an enumerated reducer module → all
  series re-derive; (f) hand-plant a series whose `cst_series_digest` is wrong →
  stage B fails.
- **Coverage assertion.** Request an analysis window outside a series' recorded
  acquisition coverage → stage B fails naming both windows.
- **Reference window (step 5e).** Request a reference ending 2020-12-31 → warning,
  `reference_window_effective` = 2000–2014 in provenance, disclaimer in the
  report, no raise. Request a reference entirely after 2014 → raise. Request a
  horizon ending after 2100 → warning + clip.
- **Change-factor properties (steps 5a–5e).** Relative change is invariant to a
  multiplicative unit rescale of the input; absolute change is invariant to an
  additive offset. Cheap property tests over synthetic series.
- **Calendar and coverage (steps 5b, 6b).** Synthetic series on a 360-day calendar
  and on a standard calendar with the same underlying rate → month-length weights
  differ and the annual means match the analytic expectation; a partial first
  hydrological year is dropped and counted; a missing month fails loud; a leap
  year is weighted correctly.
- **Dry month (step 6b).** Synthetic series with a near-zero reference month →
  `value = NaN`, `status = "reference_below_threshold"`, `absolute_value`
  populated, report footnote present, S5 excludes it.
- **Ensemble (step 6c).** Two members of one model plus one member of another →
  members averaged within the model, unique-model count 2, envelope suppressed,
  composition table lists 2 models / 3 members / correct institution counts, and
  every member value present in the CSV.
- **Grid geometry (step 5a).** Synthetic source with 2-D lat/lon coordinates →
  reducer raises; 1-D non-monotonic → raises; 1-D monotonic → cos-latitude
  weights applied and `cst_weighting_scheme` recorded.
- **Catalog validator (step 4).** Unit tests over the templated-entry +
  `placeholders:` lookup, including a model absent from the list, a scenario
  template absent from the file, and an entry carrying an unrecognized construct
  (must error naming the key). One integration-marked test cross-checking the
  accept list against `hydromt.DataCatalog(...).sources`.
- **Fail-fast (step 4).** A reduce job forced to raise → the run halts, the
  series key appears in the log, no dummy netCDF is written, and no summary is
  produced. With `--keep-going`, all sibling reduce jobs still complete.
- **Minimum sources (step 4).** A config whose validated list is below
  `ensemble.min_sources` fails at DAG build naming resolved and missing sources.
- **Gridded branch (step 7).** `save_grids: true` dry-run → same job count as
  `false`, plus the declared `grids/*.nc` outputs; `save_grids: false` → those
  paths are absent from the DAG entirely.
- **Characterized diff (steps 5a–5e, 6b, 6c).** Old versus new change factors on
  the seed config, recorded **once per sub-step**. Attribution per cause —
  weighting, calendar/month length, rounding, statistic set, window, coverage
  policy, ensemble treatment — comes from the commit boundaries, not from a flag
  matrix: each sub-step's diff *is* its cause. This is why step 5 is decomposed
  in §8.

**Not validated here.** Whether 30-year windows are the right default for a given
basin; whether the ensemble is adequate; whether the cos-latitude approximation
error is material for any particular model (that is **OQ-10**). These are user
judgements the report surfaces, not gates.

---

## 10. Open questions

- **OQ-1 (D5).** Extend `Snakefile_climate_projections` in place, or open a 4th
  entry point? *Recommendation: extend in place for v2.0; R4 strengthens this
  because the existing name still describes the narrowed scope (§6.3).*
- **OQ-2.** Does this open **Phase 4**, or land as an unnumbered milestone? The
  roadmap records Phase 3 complete with the open question of whether to close the
  roadmap or open a Phase 4, and lists the 4th entry point in the candidate pool.
- **OQ-3 — CLOSED by D3 (§6.10).** WF2 output stays at
  `climate_projections/{clim_project}/` for v2.0. R4 removed the rename's
  motivation, and keeping the root preserves the manifest evidence chain (§8).
- **OQ-4.** Default reference-window length — 30 years (which under the R1 clip
  rule means 1985–2014) versus today's 21-year `[1990, 2010]`. Value-changing;
  needs an explicit owner call before step 5e.
- **OQ-5.** Extremes indices need sub-monthly data. Under R4 this is no longer a
  v2.0 scoping question but the S4 architecture change of §5.10; the question
  that remains is whether a daily CMIP6 branch is ever in scope given ~30× the
  volume.
- **OQ-6 — CLOSED by owner ruling R3** (§5.6). Residual policy — institution
  de-duplication and performance weighting — is recorded as **N9**, not applied.
- **OQ-7 (dependencies — asks, C5/N4).** **None is adopted.** The design as
  specified needs none. Standing candidates, recorded so the asks are visible:
  `xclim` (calendar/unit handling and standard indicators — relevant to S4),
  `regionmask` (fractional-area polygon masking — relevant if §5.3's regional
  sampling choice is revisited), `intake-esm` (query-based catalog generation —
  §6.6, and the precondition for ensembles large enough to justify the percentile
  envelopes §5.6 currently suppresses).
- **OQ-8 — CLOSED by owner ruling R2** (§5.8). `save_grids` is retained,
  default off, with declared outputs.
- **OQ-9 (new).** Default value of `relative_change.min_reference` per variable,
  in canonical units. The rule (§5.6) is complete without it; only the number is
  open. *Evidence that settles it:* the distribution of monthly reference-mean
  precipitation on the seed basin plus one strongly seasonal basin — choose the
  threshold below which the relative factor's sampling spread exceeds the factor
  itself.
- **OQ-10 (new).** Should `config/catalogs/cmip6_data.yml` stop dropping
  `lat_bnds` / `lon_bnds` / `bnds` so the reducer can compute true cell areas
  instead of the cos-latitude approximation? *Evidence that settles it:* measure
  the basin-mean difference between cos-latitude weights and bounds-derived cell
  areas for one catalog model on a non-uniformly-spaced grid at the seed basin;
  if the difference is below the reporting precision of the change factors, keep
  the approximation and the label.
- **OQ-11 (new).** Revisit fail-fast (D4) in favour of §6.8's status-artifact /
  checkpoint contract? *Evidence that settles it:* the observed remote-read
  failure rate across real runs, from `logs/` and `benchmarks/`. If a
  multi-hour ensemble run fails often enough that fail-fast forces repeated full
  restarts *despite* the series cache, the tolerant contract earns its
  complexity.

---

## 11. Revision log

- **2026-07-29 — revision 1** (`design-v1.md`). Initial draft, grounded in
  `Snakefile_climate_projections`, `blueearth_cst/projections/*.py`,
  `blueearth_cst/climate_analysis/*.py`, `blueearth_cst/shared/snake_utils.py`
  (`climate_store_spec`), `config/catalogs/cmip6_data.yml`, and
  `config/workflows/snake_config_model_test.yml`. Two facts measured rather than
  assumed: the two region polygons have identical bounds on the seed fixture, and
  the CMIP6 sources are `Amon` so the monthly sum/mean dispatch is a no-op.
- **2026-07-29 — revision 2** (this file). Authored against G1 rulings R1–R4
  following round-1 internal (Fable / `critical-thinker`) and external
  (gpt-5.6-sol) review; all 19 findings dispositioned in `ledger.md`.
  Substantive changes:
  - **Scope narrowed (R4).** v2.0 is monthly GCM projections analysis with an
    optional gridded branch; `extract_historical.nc` is explicitly not reduced
    (N7). G1 and G9 restated; §5.1 added; the extension slots of §5.10 rewritten
    as contract changes rather than free reads.
  - **Reference series: clip, never splice (R1).** §5.4 D1 adds the clip rule,
    three named warning sites, a horizon boundary rule, and a short-window
    warning; G3 becomes an alignment check; `historical_year_range` is retained
    rather than retired; §6.5 records why splicing was rejected. The 15-year
    effective reference implied by a 2000–2020 request is stated explicitly.
  - **The R4 open consequence resolved as a named decision.** §5.4 D2 / §6.4
    enumerate four options for the region source, select A1 with the cost stated
    and A2 as the named fallback, and surface the discriminating question for the
    owner.
  - **`save_grids` retained as a declared branch (R2).** §5.8 names the three
    file families that stop being params-passed, uses a parse-time output branch,
    and shows the branch adds no jobs.
  - **Ensemble treatment fixed (R3).** §5.6 sets the sampling unit to the unique
    model, averages members within a model, shows members hierarchically, and
    suppresses envelopes below the model threshold.
  - **Runtime failure semantics decided (ext1-02).** §5.7 D4 takes fail-fast and
    moves the minimum-source check to DAG build, where it can execute; §6.8
    records the tolerant alternative and OQ-11 its revisit evidence.
  - **Cache contract pinned (risk-02 / ext1-04 / risk-03 / risk-06).** Fixed
    acquisition windows per experiment class; `future_horizons` excluded from the
    digest; the catalog file deliberately not a reduce input; `store_region.geojson`
    under `ancient()` with the region *specification* in the digest; a mechanical
    module-hash reducer version plus a fail-loud digest attribute; stage B's
    input set explicit rather than a glob.
  - **Method edge cases specified (risk-05 / ext1-05 / ext1-08).** Normative
    change-factor formulas; the dry-month denominator rule; complete-hydrological-year
    and full-coverage requirements; calendar-aware month-length weighting; and
    cos-latitude weighting relabelled an approximation with a fail-loud geometry
    check, grounded in the catalog's `drop_variables`.
  - **Factual corrections (risk-09 / ext1-10 / risk-04).** Job arithmetic
    recomputed for the narrowed scope with a stated counting convention (22 → 15
    fresh, 14 with the store present) and an explanation of why reduce is 9 and
    not 10; the baseline manifest's coverage stated exactly (7 targets, 6 under
    `climate_projections/cmip6/` plus the config snapshot under `config/runs/`),
    with the value-neutral / manifest-clean split it forces added to §4.
  - **Commit plan decomposed.** §4 criterion 5 (new) falsified v1's own step-5
    gate: a per-cause diff cannot be attributed when five independent
    value-changing items land in one commit. Steps 5 and 6 are split into
    5a–5e and 6a–6c so each sub-commit's diff *is* its cause. Step 7's
    manifest impact on the three pinned PNGs is stated. Log-path reassignment
    from collapsing the three gathers is noted as a non-manifest concern.
  - **Warning surfaces assigned per condition**, not lumped: the alignment
    difference does not warn on stderr by default, because the shipped seed
    config differs and an always-firing warning is filtered out.
  - **`ensemble.min_sources` given a default of 1** and distinguished from
    `ensemble.min_models_for_envelope`: under fail-fast it is a configuration
    floor, not a runtime resolution check, and any higher default would fail
    DAG build on the tracked baseline seed.
  - **OQ-3 closed** (D3, keep the output root); **OQ-6 and OQ-8 closed** by
    rulings; **OQ-9, OQ-10, OQ-11 opened**, each with the evidence that would
    settle it. No new dependency is adopted (N4, OQ-7).
