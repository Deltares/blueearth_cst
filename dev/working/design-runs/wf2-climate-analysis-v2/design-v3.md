# WF2 v2.0 — GCM Projections Analysis: Design (DRAFT, revision 3)

```
Status:     draft — not admitted to a milestone; no implementation authorised
Date:       2026-07-29
Authors:    tanerumit (with Claude Code)
Supersedes: none
Revisions:
  - 2026-07-29: initial draft (design-v1.md)
  - 2026-07-29: corrected §5.3 — retiring `historical_year_range` moves the
    reference window by a decade and is value-CHANGING (design-v1.md)
  - 2026-07-29: revision 2 (design-v2.md) — authored against G1 rulings R1–R4
    after round-1 internal (Fable / `critical-thinker`) and external
    (gpt-5.6-sol) review. Scope narrowed to GCM projections analysis; the
    reference series is clipped, never spliced; `save_grids` becomes a
    first-class optional branch; the ensemble sampling unit is the unique
    model. Finding-by-finding disposition: `ledger.md`.
  - 2026-07-29: revision 3 (this file) — authored against owner rulings R3′,
    R3″ and R5, the D2→A1 and OQ-4 confirmations, and the **generated** CMIP6
    catalog (commit `f8194e8`). No aggregation at any level; cross-combination
    statistics deferred as ex-post; the monthly series (basin and, optionally,
    gridded) becomes a declared deliverable; source resolution is rewritten
    around the generated catalog and separates *not published* from *failed to
    read*. Ledger amendment: `ext1-07` becomes partially rejected by ruling.
```

Companion documents:

- `dev/workflows/wf2_climate_projections_overview.md` — rule-level map of WF2 **as
  it is today** (the baseline this design changes).
- `dev/workflows/climate_projections.md` — the behavioral contract of WF2 as it is
  today (config keys, unit split, `save_grids`, downstream semantics).
- `dev/workflows/wf2-cmip6-store-inventory.md` — the live crawl of `gs://cmip6`
  behind the generated catalog; the availability numbers this revision reasons
  from. **New input to revision 3.**
- `dev/scripts/generate_cmip6_catalog.py` — the generator that now owns
  `config/catalogs/cmip6_data.yml`. **New input to revision 3.**
- `dev/p32a/climate-analysis-design.md` — the sealed milestone that created
  `blueearth_cst/climate_analysis/` and the model-free climate store this design
  builds on.
- `dev/working/design-runs/wf2-climate-analysis-v2/` — the review run: `intake.md`,
  `design-v1.md`, `design-v2.md`, the two round-1 reviews, `review-index.md`,
  `status.md` (rulings R1–R5), and `ledger.md`.

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
12-element *monthly* arrays. The member axis is a config list folded inside one
job rather than a declared dimension. The spatial reduction is an unweighted
`mean([x_dim, y_dim])`; the annual aggregation ignores month length and each
model's calendar.

**(c) The workflow cannot run without a built hydrological model.** Its only
cross-workflow input is `hydrology_model/staticgeoms/region.geojson`, so a
projections-only run requires a completed WF1.

**The first enabling discovery.** P3-2a already built the model-free region.
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

**The second enabling discovery (new in revision 3).** Since commit `f8194e8`,
`config/catalogs/cmip6_data.yml` is **generated** by
`dev/scripts/generate_cmip6_catalog.py` from a live directory listing of
`gs://cmip6`. It carries **289 entries and 2 426 sources** — one entry per
(model, scenario), keyed `cmip6_{institution}/{source}_{experiment}_{member}`,
whose `placeholders.member` list is exactly the members that exist in the bucket
with both `pr` and `tas` at `Amon`. The previous hand-curated catalog exposed 23
of 64 historical models and pinned everything to `r1i1p1f1`.

Two consequences run through this revision. First, **existence became a fact the
repository owns**: a source name that resolves in the catalog is a store that was
observed to exist at `meta.crawled_on`, so the DAG-build validator (§5.7) becomes
a lookup rather than a re-implementation of hydromt's resolution. Second,
**availability became ragged and visible**: member counts differ per (model,
scenario) — CanESM5 publishes 65 historical members, INM-CM4-8 publishes one —
and scenarios are missing wholesale for many models. Ruling **R3′** settles what
the workflow does with that: it collects the union of what the store offers and
never treats raggedness as an error.

**What revision 3 changed about the problem framing.** design-v2 still contained
an aggregation layer: members averaged within a model, a model-count threshold
gating percentile envelopes, a min–max range. Rulings **R3′** and **R3″** remove
the whole layer. WF2 v2.0's deliverable is **the data points themselves** — one
(ΔT, ΔP) per (model, scenario, member, horizon) — plus the monthly series they
were computed from (**R5**) and a record of what the run actually resolved.
Everything that reduces *across* data points is ex-post and downstream.

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
  tidy long-format table with explicit units, formulas, and provenance — keyed on
  **(model, scenario, member, horizon)** crossed with (period, variable,
  statistic), and **never a collapsed group**: no row is a value computed over
  more than one (model, scenario, member) combination.
- **G5.** Re-running with a changed horizon, statistic, plot, or report costs no
  network access.
- **G6.** Fan-out width is set by the network-bound stage; the pure-computation
  stages are single jobs.
- **G7.** Absent catalog sources are resolved at DAG-build time, not papered over
  with dummy empty netCDFs at runtime — and **a source that is absent from the
  store is distinguished from a source that failed to read**.
- **G8.** One report artifact replaces a directory of loose PNGs, and every
  figure is a declared output.
- **G9.** The extension surface is **documented with the contract change each
  extension requires** — not advertised as a free read. (Restated from v1 by
  **ext1-06** and **R4**; see §5.10.)
- **G10.** Raw gridded products remain available on request as a **declared**
  optional branch, default off. (From **R2**.)
- **G11.** *(New, **R5**.)* The basin monthly series is a **user-facing product**
  with a documented schema, a stable name, and a retention rule — not an internal
  cache whose only consumer is stage B. When the gridded option is on, the
  monthly series **on the source grid** is a product on the same terms.

### Non-goals

- **N1.** Driving WF3 from CMIP6. (`AGENTS.md` § Background.)
- **N2.** Relocating `plot_climate_source` (WF1 rule 1.15) into WF2 — see §6.2.
- **N3.** A standalone 4th Snakefile entry point. Named as decision **D5** and
  deferred.
- **N4.** New third-party dependencies. Candidates are recorded in §10 as asks;
  the design as specified needs none of them, and none is adopted.
- **N5.** Re-engineering how hydromt resolves catalogs or reads rasters.
  (`AGENTS.md` § Hard Constraints.)
- **N6.** Bias correction or downscaling.
- **N7.** *(R4.)* Observed-vs-GCM comparison, observed climatology, and GCM bias
  diagnostics. `extract_historical.nc` is **not** reduced, read, or plotted by
  any v2.0 WF2 rule. The store is declared for its region polygon.
- **N8.** *(R1.)* Splicing, gap-filling, or otherwise processing the 2015–2020
  interval between the CMIP6 historical experiment and the scenario experiments.
  No historical or scenario run is modified.
- **N9.** *(Restated, **R3″**.)* Institution-level de-duplication and
  performance weighting are **downstream concerns**, not "not applied". They
  belong to the same ex-post layer as every other cross-combination statistic
  (**N10**), and v2.0 neither performs them nor decides them. The composition
  record (§5.7) gives a downstream consumer the institution and member counts
  needed to apply either.
- **N10.** *(New, **R3′** / **R3″**.)* **Aggregation across data points, at any
  level.** No averaging of members within a model, no model-level collapse, no
  percentile envelope, no ±σ band, no min–max range, no model-count threshold,
  no ensemble weighting. Statistics computed *over the set of* (model, scenario,
  member, horizon) points are ex-post and are computed downstream of WF2's tidy
  table. Interpretation of model similarity and correlation is explicitly out of
  scope.
  **The boundary, stated so the two senses of "statistic" cannot be conflated:**
  statistics computed on the annual series *within one* (model, scenario, member,
  horizon) — `mean`, `median`, `std`, tail quantiles — are **in scope**; they are
  what the change factor *is* (§5.6). Statistics computed *across* those tuples
  are **out**.

### Declared outputs — the v2.0 deliverable set (ruling R5)

R5 names four deliverables. This is the full declared output contract; every row
is a Snakemake `output:`, and nothing WF2 writes is undeclared.

| # | Product | Path (under `{project_dir}/climate_projections/{clim_project}/`) | Stage | Condition |
| --- | --- | --- | --- | --- |
| 1 | **Change-factor table** — one (ΔT, ΔP) per (model, scenario, member, horizon) | `change_factors/annual.csv`, `change_factors/monthly.csv`, `change_factors/change_factors.nc` | B | always |
| 2 | **Basin-averaged monthly series** — one per (model, scenario, member) | `series/{series_key}.nc` | A | always |
| 3 | **Composition record** — one row per *requested* combination with its resolution status | `composition.csv` | B | always |
| 4a | **Gridded monthly series** — the same series on the source grid, before spatial reduction | `grids/series/{series_key}.nc` | A | `save_grids: true` |
| 4b | **Gridded change fields** | `grids/change/{series_key}_{horizon}.nc` | B | `save_grids: true` |
| — | Report, figures, provenance | `report.md`, `plots/*.png`, `provenance.json` | C | always |
| — | Legacy annual summary (migration evidence anchor, §5.9/§8) | `summary/annual_change_scalar_stats_summary{,_mean}.{nc,csv}` | B | always in v2.0 |

Schemas: series in §5.3, change-factor table in §5.9, composition record in
§5.7, gridded products in §5.8.

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
- **C7 — *(New, revision 3.)* `config/catalogs/cmip6_data.yml` is a generated
  file.** Its header says *"do not hand-edit"*; the generator
  (`dev/scripts/generate_cmip6_catalog.py`) owns its format. Any change WF2
  needs from the catalog — a pinned version, a new table, an extra variable — is
  a change to the **generator**, and lands as a regeneration. Nothing in
  `blueearth_cst/` or a Snakefile may write to it.

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
   the point in the DAG where it can actually run and stop the run. (Added in
   revision 2 — this is what ext1-02 falsified about v1's runtime failure
   handling.)
6. **The extension surface is designed once, and honestly.** A slot named now
   costs a paragraph; a slot retrofitted later costs a migration. A slot
   advertised as free when it is not costs credibility.
7. ***(New, revision 3.)* Absence and failure are different classes and get
   different machinery.** A combination the store does not publish is **data
   about the store**: it is resolved at DAG build, recorded, reported, and the
   run proceeds. A published source that cannot be read is an **error**: the run
   halts (**D4**). A design that routes both through one check — as design-v2's
   `ensemble.min_sources` did — cannot make either meaningful. This criterion is
   what ruling **R3′** forces: raggedness is normal, and normal cannot share a
   code path with broken.

---

## 5. Selected approach

### 5.1 Scope of v2.0 (rulings R4, R3′, R3″, R5)

**In scope.** Monthly GCM projections output analysis: acquire every *resolved*
CMIP6 (model, experiment, member) source, reduce each to one basin monthly
series, derive annual and monthly change factors relative to a clipped reference
window, and report — **one data point per (model, scenario, member, horizon),
carried distinctly end to end**. Plus the composition record, and an optional,
declared gridded branch (§5.8).

**Out of scope for v2.0.**

- Any product that reads observed or reanalysis data (**R4** / **N7**).
  Concretely: **`extract_historical.nc` is not reduced in v2.0.** No WF2 rule
  opens it. The `extract_climate_grid` rule is declared solely so WF2 can read
  `store_region.geojson` and stop depending on `hydrology_model/`; the cost of
  that is decision **D2** (§5.4, §6.4).
- **Any statistic computed across data points** (**R3″** / **N10**). This is not
  a deferral of polish; it removes an entire layer design-v2 specified — member
  averaging, the model-count threshold, envelope suppression, the min–max range,
  and the `ensemble:` config block that carried them. §5.10 records where that
  layer lands (slot **S6**) and states that it needs no WF2 contract change,
  which is precisely why it can be deferred without cost.

**What this buys.** The reducer contract is specifiable in one page (one source
family, one temporal semantics), and the product is a table whose every row has a
provenance you can point at. There is no place in v2.0 where a number is a
function of *which other models happened to resolve* — which is what made
design-v2's ensemble section hard to specify and impossible to test stably
against a ragged store.

**What this costs.** The "general climate data & projections analysis workflow"
of the change request is reached in more than one move. §5.10 names the contract
change each remaining move requires so that v2.0's decisions do not have to be
undone.

### 5.2 Architecture — three stages, fan-out only where it pays

| Stage | Rule(s) | Fan-out | Network | Purpose |
| --- | --- | --- | --- | --- |
| **0. Store** | `extract_climate_grid` (from `climate_store_spec`) | — | yes | Model-free region polygon (consumed) + gridded observed climate (**not** consumed in v2.0, N7) |
| **A. Reduce** | `reduce_gcm_series` | `{series_key}` | yes | One GCM source → one basin monthly series (+ optional gridded series) |
| **B. Derive** | `derive_change_factors` | — | no | All series → annual + monthly change factors + composition record (+ optional gridded change) |
| **C. Report** | `climate_report` | — | no | Tables + figures + one report page + provenance |

Plus `copy_config` (unchanged), `gather_series_logs` (one gather, because one
stage fans out), and `gather_benchmarks` (unchanged).

**Rule count: 11 → 8**, both counts including the `all` target rule. New set:
`all`, `copy_config`, `extract_climate_grid`, `reduce_gcm_series`,
`derive_change_factors`, `climate_report`, `gather_series_logs`,
`gather_benchmarks`. The three `gather_*_logs` rules collapse to one; `ruleorder`
disappears with the rules it ordered.

**Job accounting is data-dependent (new in revision 3).** Under R3′'s union
semantics the fan-out width is a function of the *store*, not of the config
alone. The count is therefore given as a **formula**, with one measured example:

```
reduce jobs = |resolved scenario combinations|          (§5.7)
            + |distinct (model, member) references they require|
total jobs  = reduce jobs + 6      (store, derive, report, copy_config,
                                    gather_series_logs, gather_benchmarks)
            − 1 if the climate store already exists
```

Two corollaries worth stating because they are non-obvious:

- **The historical series set is derived, not configured.** A model's historical
  series is reduced only if some resolved scenario combination needs it as a
  reference (§5.7 **D7**). `NCC/NorCPM1` publishes 30 historical members and
  **zero** members in any SSP, so no config that requests it produces a single
  reduce job for it. Requesting a model is not the same as reducing it.
- **A reference is reduced once, however many scenarios share it.** The second
  term counts *distinct* (model, member) pairs, not one per scenario — this is
  the non-obvious half of the arithmetic. On the seed config three models × two
  scenarios need **3** references, not 6, which is why 6 + 3 = 9 rather than 12.
- **`save_grids: true` still changes no count** — the gridded products are
  additional declared outputs of the same jobs (§5.8).

**Measured example — the seed config** (`config/workflows/snake_config_model_test.yml`:
`models: [NOAA-GFDL/GFDL-ESM4, INM/INM-CM4-8, INM/INM-CM5-0]`,
`scenarios: [ssp245, ssp585]`, `members: [r1i1p1f1]`, `save_grids: false`).
Resolved against the generated catalog on 2026-07-29: all three models publish
`r1i1p1f1` for `historical`, `ssp245` and `ssp585`, so 6 scenario combinations
resolve and they require 3 distinct references.

| Component | Jobs |
| --- | --- |
| `extract_climate_grid` | 1 |
| `reduce_gcm_series` — references (3 models × 1 member, `historical`) | 3 |
| `reduce_gcm_series` — scenario combinations (3 × 2 × 1) | 6 |
| `derive_change_factors` | 1 |
| `climate_report` | 1 |
| `copy_config` | 1 |
| `gather_series_logs` | 1 |
| `gather_benchmarks` | 1 |
| **Total** | **15** |

**14** when `climate_historical/era5_20000101_20201231/` already exists.

Counting convention, stated because v1's arithmetic was wrong twice: counts
**exclude the `all` target job**, which is how v1's "22 today" was counted.
Today's WF2 under that convention is 1 `copy_config` + 3 `monthly_stats_hist` +
6 `monthly_stats_fut` + 6 `monthly_change` + 1 `monthly_change_scalar_merge` +
1 `plot_climate_proj_timeseries` + 3 `gather_*_logs` + 1 `gather_benchmarks` = **22**.

15/14 is a **measured value for this config against this catalog snapshot**, not
an invariant: §9's test derives the expected count from the resolved source
manifest and never from a literal. No count is asserted here for
`snake_config.template.yml`, whose four-label `members:` list resolves to a
different, catalog-dependent number.

The concurrent width at the network-bound stage rises from 3 (today's
`monthly_stats_hist`, serialized ahead of `monthly_stats_fut` by a data-free
ordering edge) to the full resolved set — 9 on the seed config.

```
catalogs ──► 0. extract_climate_grid ──► store_region.geojson ──┐
                                     └─► extract_historical.nc  │   (not read in v2.0 — N7)
                                                                │
                            ancient() edge: ordering only ──────┤
                                                                ▼
      A. reduce_gcm_series {series_key}   (resolved scenario combos + their references)
                     │
                     ├─► series/{series_key}.nc                  PRODUCT, digest-stamped
                     └─► grids/series/{series_key}.nc            PRODUCT [save_grids only]
                     │
                     ▼
      B. derive_change_factors            (explicit expanded input list — never a glob)
                     ├─► change_factors/annual.csv
                     ├─► change_factors/monthly.csv
                     ├─► change_factors/change_factors.nc
                     ├─► composition.csv
                     ├─► summary/annual_change_scalar_stats_summary*.{nc,csv}
                     └─► grids/change/{series_key}_{horizon}.nc  [save_grids only]
                     │
                     ▼
      C. climate_report
                     ├─► report.md
                     ├─► plots/*.png        (all declared)
                     └─► provenance.json
```

### 5.3 The GCM series store — product, identity, caching

Stage A applies one reducer to every resolved GCM source:

```
reduce(catalog_entry, region_bounds, buffer, variable_spec, acquisition_window)
    -> monthly basin series, dims (time,), source metadata as coords
```

**Series key.** `cmip6_{dataset}_{experiment}_{member}`, with `/` in the CMIP6
model name sanitized to `_` (the catalog entry name carries a vendor path
segment, e.g. `cmip6_NOAA-GFDL/GFDL-ESM4_ssp245_{member}`). Examples:
`cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1`,
`cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1`. `member` is a wildcard, not a config list
folded inside one job, so a multi-member ensemble is a fan-out rather than a
loop.

**Layout** (decision **D3**, §6.10 — the WF2 root does *not* move in v2.0):

```
{project_dir}/climate_projections/{clim_project}/
    series/{series_key}.nc                 # PRODUCT (R5 deliverable 2)
    grids/series/{series_key}.nc           # PRODUCT [save_grids only] (deliverable 4)
    grids/change/{series_key}_{horizon}.nc # [save_grids only]
    change_factors/{annual,monthly}.csv
    change_factors/change_factors.nc
    composition.csv
    summary/…                              # retained; manifest-pinned (§8)
    plots/*.png
    report.md
    provenance.json
```

`climate_historical/` (the shared store) is untouched — it is co-owned with WF1
and WF3.

#### Series schema (new in revision 3 — R5 promotes this to a product)

design-v2 treated `series/*.nc` as a cache whose only consumer was stage B and
specified no schema. R5 makes it deliverable 2, so it needs one. This is a
strengthening of the architecture, not a conflict with it: the cache **is** the
deliverable.

| Element | Contract |
| --- | --- |
| Dimensions | `time` only — monthly, one step per month, decoded `cftime` in the model's **own calendar** (the catalog sets `decode_times: true`) |
| Scalar coordinates | `dataset` (`institution/source_id`, unsanitized), `institution`, `source_id`, `experiment`, `member` |
| Data variables | one per configured variable, named per the variable spec (`precip`, `temp`), dims `(time,)`, `dtype` float64 |
| Variable attributes | `units` (canonical, §5.5), `canonical` (`rate` \| `state`), `long_name` |
| Global attributes | `cst_schema_version`; `cst_series_digest`; `cst_catalog_entry`; `cst_catalog_crawled_on`; `cst_acquisition_window`; `cst_time_first` / `cst_time_last`; `cst_calendar`; `cst_region_bounds`; `cst_buffer_degrees`; `cst_weighting_scheme`; `cst_geometry_check`; `cst_reducer_module_hash` |

- **Naming is stable within v2.0.** `series/{series_key}.nc` with the key grammar
  above. A change to the key grammar or to the attribute set is a
  `cst_schema_version` bump, and stage B rejects a schema version it does not
  know — the same fail-loud treatment as the digest.
- **Retention: persistent, never `temp()`, never auto-pruned.** A run only ever
  adds files. Correctness does not depend on the directory's contents: stage B
  reads an explicit expanded list and asserts each file's digest, so a stale or
  orphaned series cannot enter a product (§5.3 "Stage B's input set is
  explicit", and risk **R4** in §7). Removing stale keys is an explicit user
  action; a pruning helper is a follow-up, not a correctness fix.
- **Size.** One series is `n_months × n_variables × 8 bytes` plus metadata —
  1 032 months × 2 × 8 ≈ 17 KB for a full scenario span. The run that produced it
  may have cost hours. That asymmetry is the whole argument for persistence.

**Spatial reduction — a labelled approximation, with a geometry check.**
Basin-mean over the bbox + configurable buffer, weighted by cos(latitude),
replacing today's unweighted `mean([x_dim, y_dim])`. This is stated as an
**approximation to cell-area weighting, valid for 1-D rectilinear
monotonically-spaced lat/lon grids**, and not as an area weight in general.

The reason exact areas are unavailable is in the repo: the generated catalog sets
`drop_variables: [time_bnds, lat_bnds, lon_bnds, bnds]` on every CMIP6 entry (the
generator renders this into the shared anchor block, so it holds for all 289
entries), so cell edges are not retained and true cell areas cannot be derived
from what the reducer receives. The reducer therefore:

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

Retaining bounds so exact areas can be computed is a live option — but the
catalog is now **generated** (C7), so it is a change to
`generate_cmip6_catalog.py`'s `DEFAULTS_BLOCK` and a regeneration, not a hand
edit. Recorded as **OQ-10**.

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
series' recorded acquisition coverage, naming the series and both windows. (The
inventory records SSP spans running to 2300 for some models; the fixed
acquisition window normalises that, as today's hardcoded slice already does.)

**Cache key.** `series/*.nc` is **not** `temp()`. The reducer's `params` carry a
digest over:

1. the catalog **entry name**;
2. the **resolved URI** — the entry's `uri` with `{member}` substituted, so the
   digest identifies the store this series came from, not the template;
3. the entry's **driver options** and **`data_adapter`** maps (unit add/mult,
   rename) as parsed;
4. the **region specification** — the region-determining subset of
   `climate_store_spec`'s params (`model_region`, `hydrography`, `basin_index`),
   plus WF2's configured buffer. Note `shared.basin.resolution` is *not*
   included: it is the wflow build resolution and is not an input to
   `parse_region_basin`, so it does not determine the store polygon;
5. the **variable spec** (§5.5);
6. the **acquisition window** for the experiment class;
7. the **reducer source hash** (below).

**What is deliberately excluded from the digest, and why (revised in revision 3
for the generated catalog).**

- **`placeholders` is excluded.** design-v2 folded "the catalog entry as parsed,
  including placeholders" into the digest. Under the generated catalog that is a
  defect: regeneration routinely *adds* members as the store grows, so a member
  list change would re-derive every series of that model even though not one byte
  of the data it read changed. The placeholder list determines **which** sources
  exist — that is resolution (§5.7), not identity. The resolved member is already
  in the series key and in item 2's resolved URI.
- **`meta:` is excluded** — `crawled_on` changes on every regeneration by
  construction. It is recorded on the series as `cst_catalog_crawled_on` and in
  `provenance.json`, so the snapshot is auditable without being a cache trigger.
- **The catalog *file* is not a declared input of `reduce_gcm_series`.** Declaring
  it would make any regeneration re-download every series through Snakemake's
  mtime trigger, defeating G5. C1's file-level freshness boundary is a property
  of the *shared* store rule — three Snakefiles co-own it and a file-level
  boundary is the only cheap symmetric contract available there — and
  `extract_climate_grid` keeps it verbatim. The series cache is WF2-private and
  can afford the finer-grained entry-level digest. Recorded as a deliberate,
  bounded divergence, not an oversight.
- `save_grids` is excluded (§5.8).

Two falsifiable consequences follow, and both are cache tests (§9):

- Regenerating the catalog after the store gains a member re-derives **zero**
  series.
- Regenerating with a changed shared driver/adapter block — a new
  `drop_variables` entry, a changed `unit_mult` — re-derives **every** series.
  That is intended: the shared block changes what every read *means*.

The second holds because the generator emits the driver/adapter/metadata block
once as a YAML anchor and pulls it into the other 288 entries with a merge key.
**Verified 2026-07-29:** `yaml.safe_load` (PyYAML 6.0.3, the pinned env) resolves
`<<` merge keys, so a plain YAML parse of an entry yields the merged mapping —
the validator and the digest both see the effective entry without a hydromt
import. The validator is required to resolve merge keys; a parser that does not
would silently see entries with no `driver:` block, so §9 asserts the merged read
on a non-anchor entry.

**The region edge.** `store_region.geojson` is declared as an input of
`reduce_gcm_series` under `ancient()`, exactly as today's rules 2.02/2.03 declare
`hydrology_model/staticgeoms/region.geojson`. `ancient()` supplies the DAG
ordering edge without the mtime trigger — a WF1 re-run that rewrites a
byte-identical polygon must not re-download the archive. Correctness is carried
by item 4 of the digest: the region *specification* is config, known at DAG-build
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
   table.

**Stage B's input set is explicit (risk-06).** `derive_change_factors` declares
**exactly the expanded `{series_key}` list built from the resolved combination
set** — never a directory glob. It additionally asserts that the set of series it
opened equals that list. A model removed from the config therefore cannot rejoin
the run through a leftover file, and stale series in `series/` become a disk
hygiene question rather than a correctness one.

#### D8 — time-axis uniqueness (the multi-version glob)

*New in revision 3, forced by the generated catalog's inventory.* Each
entry's URI ends `.../Amon/{variable}/*/*`, where the two globbed segments are
`{grid_label}/{version}`. `dev/workflows/wf2-cmip6-store-inventory.md` §2 checked
this across all 69 previously declared combinations × {`pr`, `tas`}: 137 resolve
to a single `{grid}/{version}` pair and **one does not** — `NCC/NorCPM1`
historical `tas` publishes both `gn/v20190914` and `gn/v20200724`, so the glob
matches two zarr stores with fully overlapping time. The design's response:

- **The reducer asserts the decoded time axis is strictly increasing with no
  duplicate timestamps, and raises naming the source, the duplicate count, and
  the first duplicated timestamp.** This is a property of the data the reducer
  actually received, so it catches the multi-version case *and* any other cause
  of duplication, without the reducer needing to know how hydromt resolved the
  glob.
- **No silent de-duplication.** Today's `drop_duplicates(dim="time")` in
  `get_stats_climate_proj.py:232` runs only in the per-variable fallback path,
  i.e. only after the combined open has already raised — so it is not a policy,
  it is an accident of the error path. v2.0 does not carry it: dropping
  duplicates chooses one publication version arbitrarily and hides which.
- **Scope of the known case, stated honestly.** `NCC/NorCPM1` publishes zero
  members in every SSP (inventory §3), so under **D7**'s pairing rule it can
  never produce a data point and its historical series is never reduced (§5.2).
  The check therefore exists because the glob property is **general**, not
  because the one measured instance is reachable. That is the correct reason to
  have it, and it is falsifiable: if a future regeneration adds a second version
  under a reachable entry, the run fails loud naming it.
- Alternatives — pinning `{grid}/{version}` in the generator, or preferring the
  latest version — are in §6.12 with the evidence that would select them
  (**OQ-14**).

### 5.4 Region and reference window

#### D1 — reference-series construction: clip, never splice (owner ruling R1)

The change-factor reference is the **GCM historical experiment**, which ends
2014-12-31 (the generated catalog resolves `cmip6_{model}_historical_{member}`
under `gs://cmip6/CMIP6/CMIP/{model}/historical/`; 2015+ exists only under the
per-scenario `ScenarioMIP` entries, and the inventory confirms historical is
uniformly 1850-01…2014-12). The design does **not** splice historical and
scenario data to fill 2015–2020 (**N8**).

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

#### The reference window length — OQ-4 closed at 30 years, 1985–2014

The owner closed **OQ-4** on 2026-07-29: **30 years, 1985–2014.** Under D1's clip
that window sits entirely inside the historical experiment, so **no clip warning
fires** and it clears the 20-year short-window floor.

How that closure lands matters for §8, because `historical_year_range` is
**required, not defaulted**: `Snakefile_climate_projections:36` reads it with
`get_config(..., optional=False)`, and every shipped config sets it explicitly
(`snake_config.template.yml`: `[1980, 2010]`; `snake_config_model_test*.yml`:
`[1990, 2010]`). There is no default in code to change. The closure therefore
lands as:

- **`config/workflows/snake_config.template.yml` adopts `[1985, 2014]`**, and the
  documentation states 30 years ending 2014 as the recommendation with the reason
  (a full standard climatological normal that needs no clip).
- **The test fixtures keep `[1990, 2010]`.** Moving the seed's window would move
  every number in the manifest-pinned summaries *in the same commit as* the clip
  and warning machinery, which destroys the per-cause diff attribution §4
  criterion 5 exists to protect — and the seed is the baseline-recording fixture,
  deliberately minimal. It is a separate, value-changing, fully re-recorded
  commit if wanted.

**Driver-visible reading, flagged for correction.** The owner classified OQ-4 as
"value-changing, gates step 5e". That holds for any project adopting the new
window, and for the template. This design does **not** move the seed fixture's
window, so step 5e stays output-neutral on the seed; the seed adoption is carried
as its own migration row (§8, step 5f) so the owner can take it or leave it
without re-opening the rest. If the intent was that the seed moves too, 5f is the
row to enable, and it is value-changing and manifest-breaking on all four pinned
summary targets.

**The tradeoff, stated plainly.** A user who instead sets the reference window to
the project baseline (2000-01-01 … 2020-12-31) gets an effective GCM reference of
2000–2014 — **15 years**, which trips the short-window warning. That is the honest
surface of R1: the clip is visible, the number is smaller than the 20-year floor,
and the disclaimer says so.

#### D2 — declaring the store when nothing reads its gridded output

**Confirmed by the owner on 2026-07-29: A1.** Under **R4**, declaring
`extract_climate_grid` buys WF2 exactly one thing: the model-free region polygon.
It also puts a full gridded observed extraction on WF2's critical path, because
**C1** forbids declaring a subset of the store's outputs. On the seed fixture that
extraction is `climate_historical/era5_20000101_20201231/extract_historical.nc` —
**7 variables** (`precip`, `temp`, `temp_min`, `temp_max`, `kin`, `kout`,
`press_msl`), daily, **7671 time steps** over the basin bbox. *(Re-verified
against the fixture on 2026-07-29; this is the sole quantitative basis for the
accepted cost, so it is measured rather than inherited.)* Magnitude on a
production basin is source- and extent-dependent; this design asserts no
wall-clock figure.

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

**A2 remains the recorded fallback** (§6.4) should the discriminating question —
*is a first-run gridded observed extraction acceptable on a projections-only
run?* — ever be answered differently; the owner answered **yes**.

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

**Verified:** the catalog sources are CMIP6 **`Amon`** — monthly means, with `pr`
renamed to `precip` and converted to mm/day (`unit_mult: 86400`) and `tas`
renamed to `temp` and converted to °C (`unit_add: -273.15`). With one sample per
month, `resample("MS").sum()` and `.mean()` return the identical value: **the
monthly aggregation dispatch is a no-op on `Amon` input.** The real split lives
only in the change arithmetic downstream, and
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
  checks the decoded time axis is monthly (and unique, per D8) and **raises**
  naming the source otherwise. A future daily branch adds a source-level
  conversion at that same assertion point (§5.10, S1) without touching the
  variable spec.
- `change` is read by stage B (§5.6). Nothing infers anything from a name.

**Caveat forced by the generated catalog (new in revision 3).** The catalog's
`rename` block still maps `rsds → kin` and `psl → press_msl`, so those variables
are *nameable*. But the generator only guarantees presence of **`pr` and `tas`**:
its `REQUIRED_VARS = frozenset({"pr", "tas"})` is the membership test for the
`placeholders.member` list, and the catalog header says so explicitly —
*"Consumers of `kin` (rsds) or `press_msl` (psl) may find a listed member lacks
those variables."* Configuring `variables: [precip, temp, kin]` is therefore
**not** a free extension: a member that resolves for `pr`+`tas` may have no
`rsds` store, and the failure surfaces at **read time**, i.e. as a D4 fail-fast
halt naming the series. That is loud and correct, but it is a run failure, not a
resolution skip — resolution cannot see it, because the placeholder list does not
encode it. Widening the guaranteed variable set is a change to the generator's
`REQUIRED_VARS` (C7) and a regeneration; recorded as **OQ-15**.

### 5.6 Change factors — formulas, edge cases, and what is *not* computed

Stage B is one job, no network, reading `series/*.nc` through the explicit
expanded list of §5.3.

**Notation.** For a resolved combination *c* = (model, scenario, member), variable
*v*, and a window *W* (the effective reference window *R*, taken from *c*'s
**reference series** per D7, or an effective horizon window *H*, taken from *c*'s
own series): `y(c,v,W)` is the set of complete-hydrological-year values in *W*,
and `yₘ(c,v,W)` the set of per-year values of calendar month *m* ∈ 1…12 in *W*.
Each annual value is the month-length-weighted mean of that year's monthly
series.

**Formulas.** For each requested statistic σ ∈ {`mean`, `median`, `std`, …}:

| `change:` | Annual | Monthly (per *m*) | Units |
| --- | --- | --- | --- |
| `absolute` | `σ(y(c,v,H)) − σ(y(c,v,R))` | `σ(yₘ(c,v,H)) − σ(yₘ(c,v,R))` | the variable's `units` |
| `relative` | `100 × (σ(y(c,v,H)) / σ(y(c,v,R)) − 1)` | `100 × (σ(yₘ(c,v,H)) / σ(yₘ(c,v,R)) − 1)` | `%` |

design-v2 wrote these for the window mean only, leaving the interaction with the
statistic set readable two ways. Stated explicitly: **the statistic is computed
on the per-year values inside the window, and the `change:` arithmetic is then
applied to the statistic.** For `σ = mean` this is the familiar delta factor. For
a dispersion statistic (`std`) under `change: relative` the result is a **ratio
of dispersions, not a change in level**; the CSV's `statistic` column and the
report label it as such, and the report does not plot it on the same axis as a
level change.

**Statistics.** Today's eight (`mean, std, var, median, q_90, q_75, q_10, q_25`)
over a ~20-year window make `q_90` effectively the second-highest of 20 values.
v2.0 emits `mean`, `median`, `std` by default; the tail quantiles are opt-in and,
when emitted, are labelled with their effective sample size in both the CSV and
the report. Value-changing; migration step 5d. These are **per-series**
statistics — R3″ keeps them in scope precisely because they are what a data point
*is*.

**Calendar-aware weighting.** The annual value weights each month by its length in
the model's own calendar, taken from the decoded `cftime` axis (the catalog sets
`decode_times: true`). For a 360-day-calendar model every month is 30 days and
the weights are uniform; for standard and no-leap calendars they are not. The
inventory records the spread (`noleap` for GFDL-ESM4, INM-CM5-0, CanESM5;
`360_day` for UKESM1-0-LL; `proleptic_gregorian` for MPI-ESM1-2-LR, EC-Earth3),
and records that today's code sums or averages the 12 monthly values unweighted.
This corrects that and makes annual means comparable across models with different
calendars rather than differing for procedural reasons. The calendar name is
recorded per series; stage B raises on a calendar it cannot weight.

**Dry-month / near-zero denominator rule (risk-05, ext1-05).** Relative change is
undefined-in-practice when the reference month is near zero — a well-known
delta-method failure that the annual product largely avoids and that a monthly
product walks straight into on any basin with a dry season. Rule:

- Config key `relative_change.min_reference`, per variable, in the variable's
  canonical units. **Default value is OQ-9** — the rule below is complete and
  testable regardless of the number.
- When `σ(yₘ(c,v,R)) < min_reference`: emit `value = NaN`,
  `status = "reference_below_threshold"`, and the corresponding **absolute**
  change in an `absolute_value` column, so the information is not lost.
- The report renders flagged months as gaps with a footnote naming the rule and
  the threshold.
- A `(dataset, scenario, member, horizon, variable)` whose flagged-month count
  exceeds `relative_change.max_flagged_months` is itself flagged in the report's
  flagged-month table. (Named that way to avoid collision with
  `summary/annual_change_scalar_stats_summary*`, which is the unrelated legacy
  artifact of §5.9.)

**Coverage and partial-year policy (ext1-05).**

- Aggregation uses **complete hydrological years only**, where the hydrological
  year starts at `start_month_hyd_year`. A partial first or last year is dropped
  and the count of dropped years is recorded per series and window.
- A series must have **every month present** in the effective window. `Amon` zarr
  stores are contiguous, so a gap indicates a source problem; the run **fails**
  naming the series and the missing months (consistent with D4), rather than
  averaging a short month set.

**No aggregation, at any level (rulings R3′ and R3″).** This is the section
design-v2 got wrong, and the correction is a deletion, not a replacement:

- Every (model, scenario, member, horizon) is **one data point with its own ΔT
  and ΔP**, carried distinctly end to end and rendered as its own row, point, or
  trace. Nothing is averaged, collapsed, or grouped away.
- **Deleted from design-v2 §5.6:** "members are averaged within a model first";
  the unique-model sampling unit; the `ensemble.min_models_for_envelope`
  threshold and its envelope-suppression rule; the labelled min–max range; and
  the `ensemble:` config block that carried them. `ensemble.min_sources` goes
  with them (§5.7, **D6**).
- **Nothing is lost that v2.0 owed anyone**, because none of those keys ever
  shipped: neither `ensemble.min_sources` nor `ensemble.min_models_for_envelope`
  appears in any `config/workflows/*.yml`. They were introduced *by* design-v2
  and are removed *from* design-v3 — so this deletion costs no config-key
  removal, no seed-config sha256 change, and no manifest re-record.
- **What the run reports instead of an aggregate: the composition record**
  (§5.7). Unique models, members per (model, scenario), institutions, and — the
  part v2 had no artifact for — every combination that was *requested but not
  resolved*, with the reason. That is what a downstream consumer needs to compute
  any ex-post statistic it wants, including the ones R3″ defers.
- **Cross-combination statistics are ex-post** and live in slot **S6** (§5.10):
  envelopes, percentile bands, ±σ, model-count thresholds, weighting, and
  institution de-duplication. They read `change_factors/*` and `composition.csv`
  and need **no WF2 contract change**, which is exactly why deferring them costs
  nothing.

### 5.7 Source resolution and failure semantics

**Rewritten in revision 3.** design-v2's §5.7 described a catalog with templated
`cmip6_{model}_{scenario}_{member}` entries and a `placeholders:` block listing
valid *models and members*. That catalog no longer exists. The generated catalog
has **one entry per (model, scenario)** — key
`cmip6_{institution}/{source}_{experiment}_{member}` — and exactly one
placeholder axis, `member`, whose list is the members observed in the bucket with
both `pr` and `tas` at `Amon`.

**DAG-build validation is now a two-level lookup (G7).** For a requested
combination:

1. does the entry key `cmip6_{model}_{scenario}_{{member}}` exist in the parsed
   catalog?
2. is the requested member in that entry's `placeholders.member` list?

No hydromt import, no network — a plain `yaml.safe_load` with merge-key
resolution (§5.3), cheap enough to run on every dry-run. The `{series_key}`
fan-out is built from the resolved list; unresolved combinations never become
jobs. This deletes the dummy-empty-netCDF pattern, `filter_nonempty`, and the
three "did this file have data?" loops in the plotting code.

The generated catalog makes this lookup **stronger than v2's**, not just simpler.
Its header states the guarantee directly: *"a source name resolving means the
store is really there"*, because membership in the list is a live-crawl fact, not
a hand-curated claim. Under the old hand-curated catalog a placeholder
cross-product could assert combinations that did not exist; the generator's
per-(model, scenario) member lists exist precisely to stop that.

#### The `members:` contract under union semantics (ruling R3′)

`members:` is a **requested set of member labels**, and the resolved set for each
(model, scenario) is `requested ∩ published`. The run's data-point set is the
**union** of those per-combination resolutions. Differing member counts per model
and missing SSP scenarios are therefore normal outcomes of a correct run, not
errors — R3′ verbatim.

Concretely, on the shipped template list `[r1i1p1f1, r1i1p1f2, r1i1p1f3,
r1i1p2f1]`: GFDL-ESM4 (which publishes `f1` only) contributes its `r1i1p1f1`
points; UKESM1-0-LL (which publishes `f2` only) contributes its `r1i1p1f2`
points; neither is an error and neither is padded with an empty slice.

**"All available" is deliberately *not* expressible in v2.0.** `members:` accepts
a list of labels; there is no `all` token and no wildcard. The reason is the
fan-out, and it is measured, not hypothetical: EC-Earth3 publishes 96 `Amon`
members for ssp245 and CanESM5 publishes 65 for historical (inventory §3). A
config naming a handful of large-ensemble models across four tier-1 SSPs would
turn a one-word edit into **hundreds of network-bound reduce jobs** with no
warning — the exact failure class §4 criterion 2 exists to prevent. R3′'s union
semantics is about not *erroring* on raggedness; it is not an instruction to
maximise the request. Two recorded alternatives, both **OQ-13**: an `all` token
paired with a mandatory per-run cap, and a per-model `members:` mapping (which
`dev/workflows/wf2-cmip6-store-inventory.md` §4 already records as "for v2 rather
than done"). Neither is built here.

#### D7 — member pairing between a scenario point and its reference

Every scenario data point needs a reference. Under **N10** there is no ensemble
mean to reference, so the pairing must be one-to-one:

**The reference for (model, scenario, member) is the series (model, `historical`,
*the same member label*).** Strict; no substitution.

The member label encodes realization, initialization, physics variant and forcing
variant (`r`/`i`/`p`/`f`). Pairing an `r1i1p1f2` future against an `r1i1p1f1`
historical would difference two runs that differ in **forcing dataset** as well
as in scenario, so the resulting ΔT would conflate a variant difference with the
scenario response. Alternatives — a designated fallback member, or a
model-historical mean — are in §6.11; the second is barred by N10 outright.

**This deliberately replaces an existing guard, and the replacement is stated so
it is not read as a regression.** `get_change_climate_proj.py` currently raises
`asymmetric hist/clim members` when a model's historical and future member sets
differ (guard added at t260720d / D-MEM, specifically to stop xarray's inner join
from **silently** shrinking the ensemble). The inventory measures 18 raising
(model, scenario) pairs under the shipped template `members:` list. D7's pairing
*is* that intersection — so the guard's arithmetic is preserved. What changes is
where it happens and what it does:

| | today | v2.0 under D7 |
| --- | --- | --- |
| when | run time, in `get_change_climate_proj.py` | **DAG build**, in resolution |
| on asymmetry | **raises**, run stops | **records a skip** and proceeds |
| visibility | exception text | `composition.csv` row + DAG-build stderr summary + report |

The guard's *purpose* — no silent shrink — is fully served, because the shrink is
now enumerated in a declared artifact rather than being invisible. What is given
up is the forcing function: a user who wanted a raise on asymmetry no longer gets
one. That is the direct consequence of R3′ ("differing member counts … must not
be an error") and it is accepted deliberately, not overlooked. The 18 pairs the
inventory names go from failing the run to appearing in `composition.csv` with
`status = reference_member_unpublished`.

#### The resolution ladder — every way a requested combination can fail

Total over the ways resolution can end. `horizon` is not a resolution axis: it
selects a window in stage B and cannot make a source absent.

| Status | Condition | Class | Surfaced by |
| --- | --- | --- | --- |
| `resolved` | scenario entry exists, member in its list, **and** the model's `historical` entry publishes the same member | data point | series, change factors, `composition.csv` |
| `model_not_in_catalog` | no entry `cmip6_{model}_*` for any experiment | **config error — DAG build raises** | stderr + raise, naming the model and the closest catalog keys |
| `scenario_not_published` | model known; no entry for (model, scenario) | normal skip | `composition.csv` + DAG-build stderr summary |
| `member_not_published` | (model, scenario) entry exists; member absent from its `placeholders.member` | normal skip | same |
| `no_historical_entry` | model has scenario entries but no `historical` entry at all | normal skip | same |
| `reference_member_unpublished` | `historical` entry exists but does not publish this member (D7) | normal skip | same |
| `no_resolved_combinations` *(run-level)* | the resolved set is empty | **config error — DAG build raises** | stderr + raise, listing every requested combination and its status |

Two of the rows exist because the store is genuinely shaped that way:

- `no_historical_entry` is not hypothetical — `DKRZ/MPI-ESM1-2-HR` publishes
  ssp126 and ssp585 members and **zero** historical members (inventory §3). It is
  neither an unknown model nor a missing scenario, and design-v2's ladder had no
  place for it.
- `model_not_in_catalog` is the only *model-level* error, and it is justified by
  C7: the catalog now covers the store in full for `Amon` `pr`+`tas`, so a name
  absent from it is absent from the store — a typo or a stale config, not thin
  data. This is the one place where "declare the full store" buys a real gate.

#### D6 — the minimum-source check, replaced

design-v2 specified `ensemble.min_sources` (default `1`) asserted at DAG build.
That key conflated the two classes §4 criterion 7 separates: it counted resolved
sources, which under R3′ is a *property of the store*, and treated a shortfall as
a configuration failure. **It is deleted**, along with
`ensemble.min_models_for_envelope` (§5.6) and the `ensemble:` block. Neither ever
shipped in a config, so deletion costs nothing.

What replaces it is a **non-configurable rule**: a run with **zero** resolved
combinations raises at DAG build, listing every requested combination and its
status from the ladder above. That is exactly what `min_sources: 1` did, minus
the key that implied it was tunable. Ensemble *adequacy* — "are these enough
models to draw a conclusion?" — is a downstream judgement (**S6**, N10), informed
by `composition.csv`; v2.0 does not encode a threshold for it.

#### The composition record (R5 deliverable 3)

`composition.csv` is written by stage B and is the artifact that makes skips
auditable. One row per **requested** (model, scenario, member) — not per resolved
one, which is the point:

| Column | Content |
| --- | --- |
| `dataset` | `institution/source_id` as configured |
| `institution`, `source_id` | split form, so downstream de-duplication (N9) is possible without re-parsing |
| `scenario`, `member` | the requested combination |
| `status` | one of the ladder's status codes |
| `reason` | human-readable, naming the catalog entry or the missing member |
| `series_key`, `reference_series_key` | populated when `status = resolved`, empty otherwise |
| `catalog_entry`, `catalog_crawled_on` | the entry consulted and the snapshot date |

Run-level counts — resolved combinations, unique models, members per model,
institutions and their model counts — are derived from this table and repeated in
`provenance.json` and the report, so no consumer has to aggregate the CSV to see
the shape of the run.

#### D4 — runtime source failure: fail-fast (carried forward, and now sharper)

ext1-02 established that design-v1's middle position is not implementable: a
`reduce` job that fails leaves its declared output absent, Snakemake halts, and
no downstream rule can write the failure record or continue with survivors. Two
coherent contracts exist (§6.8); v2.0 takes **fail-fast** (§4 criterion 5):

- A `reduce_gcm_series` job that cannot read its source **raises**. Snakemake
  halts. The failing series key and the exception are in the job log and in the
  merged stage log.
- **No** dummy netCDF, **no** empty dataset, **no** silent shrink. A run that
  completes used exactly the resolved combination set — which is now written down
  in `composition.csv` before any job runs.
- **The distinction §4 criterion 7 requires is now structural, not a judgement
  call.** Not-published is decided at DAG build from the catalog and recorded;
  failed-to-read is decided at run time from the network and raises. A given
  combination cannot be both, because a combination that did not resolve never
  becomes a job. design-v2's `min_sources` sat across that line; nothing does now.
- **`--keep-going` still helps.** `scripts/run_workflows.py` already invokes WF2
  with `--keep-going`; under fail-fast that means all independent reduce jobs
  still run and every failure is reported in one pass — only the downstream
  stages are skipped.
- **Retry is cheap** because the series cache is persistent: a re-run re-derives
  only the sources that failed.

Rationale against the tolerant alternative is in §6.8; the evidence that would
reopen it is in **OQ-11**.

#### Validator drift, revisited under the generated catalog (risk-08)

risk-08's concern was that the validator re-implements hydromt's catalog
resolution and can drift from it. The generated catalog **weakens that risk and
changes its shape**, and the design should say so rather than carry v2's
mitigation unexamined:

- **What the validator now does is narrower.** It no longer models a placeholder
  cross-product over models and members. It resolves merge keys, looks up one
  key, and tests membership in one list. There is materially less surface to
  drift.
- **The counterparty changed.** The format is produced by
  `dev/scripts/generate_cmip6_catalog.py`, a repo-owned file, so a format change
  is a change *this repository makes to itself* — visible in a diff and testable
  — rather than an upstream library evolving underneath us. Drift against hydromt
  is now confined to whether hydromt still reads this shape at all, which the
  integration test covers.
- **Retained mitigations, unchanged in substance:** validator logic stays minimal
  (exact key match + placeholder membership); an entry carrying a construct the
  validator does not model — variants, aliases, any unrecognized top-level key —
  is an **error naming the key**, so drift becomes visible rather than wrong; and
  one integration-marked test cross-checks the accept list against
  `hydromt.DataCatalog(data_libs=…).sources` (§9).
- **A new mitigation the generated catalog makes cheap:** the validator asserts
  the file's `meta.generated_by` and the presence of `meta.crawled_on`. A catalog
  that is not the generator's output — a hand edit, a foreign file passed with
  `-d` — is refused by name rather than silently parsed under assumptions that
  may not hold. This is C7 made executable.
- **A risk the generated catalog *adds*, recorded as R11 (§7):** the catalog is a
  **snapshot** (`meta.crawled_on: 2026-07-29`). A store withdrawn or re-published
  after the crawl still resolves at DAG build and then fails at read time. That
  is the correct classification under D4 — it is a read failure, not an absence —
  and `cst_catalog_crawled_on` on every series plus `catalog_crawled_on` in
  `composition.csv` make the snapshot's age visible when it happens.

### 5.8 The optional gridded branch (rulings R2 and R5)

`save_grids` is **retained, default `false`**, and is a first-class branch with
**declared** outputs, replacing today's undeclared, params-passed file layer.

**What R5 changed, and how this design reconciles it.** design-v2 §5.8 carried
forward the three file families the current code writes, of which two —
`historical_stats_{model}.nc` and `stats-{model}_{scenario}.nc` — are 12-month
**climatologies**, not series. R5 asks for the gridded counterpart of deliverable
2: the **monthly series on the source grid** (`time × lat × lon`), retained
*before* spatial reduction. Those are different artifacts, and the reconciliation
is to take R5's and drop the other two:

| Today (undeclared, via `params`) | design-v2 | design-v3 (this revision) |
| --- | --- | --- |
| `historical_stats_{model}.nc` (12-month climatology) | `grids/monthly_climatology_{series_key}.nc` | **superseded** by `grids/series/{series_key}.nc` |
| `stats-{model}_{scenario}.nc` (12-month climatology) | `grids/monthly_climatology_{series_key}.nc` | **superseded** by `grids/series/{series_key}.nc` |
| `monthly_change_mean_grid-{model}_{scenario}_{horizon}.nc` | `grids/change_{dataset}_{scenario}_{member}_{horizon}.nc` | **retained** as `grids/change/{series_key}_{horizon}.nc` |

- **Why the climatologies are superseded rather than kept alongside.** The gridded
  series is a strict superset: the 12-month climatology is
  `ds.groupby("time.month").mean()` of it, one line, computable by any consumer
  from the declared product. Shipping both would declare a derived artifact
  beside the thing it derives from, for no consumer — nothing in v2.0 reads
  either (see below), and neither is manifest-pinned (§8 lists all 7 pinned
  targets; no gridded file is among them).
- **Why the change field is retained.** It is not derivable from the series
  alone. It is a **stage-B** product: producing it requires the effective
  reference and horizon windows, calendar weighting, the complete-year policy and
  the dry-month rule — i.e. re-implementing stage B. It stays a declared output
  of `derive_change_factors`, expanded over resolved (combination, horizon).

**Schema.** `grids/series/{series_key}.nc` carries the §5.3 series schema with
dimensions `(time, lat, lon)` instead of `(time,)`, the spatial coordinates as
read (pre-reduction), and the same global attributes — except
`cst_weighting_scheme`, which is `"none — pre-reduction"`, since no weighting has
been applied yet. That is the point of the artifact: it is what the reducer saw.

**Mechanism and properties:**

- `save_grids` is a config value read when the Snakefile is **parsed**, so the
  extra entries are appended to the rules' `output:` lists at parse time. The DAG
  is fully determined before any job runs; no checkpoint or conditional-output
  machinery is needed.
- The gridded series is written from the **same network read** as the basin
  series, in the same stage-A job, **before** the spatial reduction — it is
  literally the array the reducer then collapses. The gridded change fields are
  computed in the same stage-B job as the tabular change factors. **`save_grids:
  true` therefore adds no jobs and no additional network access** — it adds disk
  and declared outputs. (This argument holds *more* strongly for the series than
  it did for design-v2's climatologies, which required an extra reduction.)
- Grids are **not** `temp()`, and follow the same retention rule as `series/`.
- `save_grids` does **not** enter the series digest: the basin series is
  byte-identical either way, and flipping the flag on re-derives the stage because
  the newly declared grid outputs are missing, which is Snakemake's normal
  mechanism.
- **Grids are a declared product, consumed by no v2.0 rule.** No v2.0 analysis
  reads them; they exist for the later advanced analysis R2 and R5 name. That
  keeps the branch first-class — declared, gated, covered by tests — without
  letting an optional artifact into the change-factor path, where it would make
  the tabular products depend on a flag.

**Volume.** `n_cells × n_months × n_variables × 8 bytes`. The seed basin's bbox
(~0.2° × 0.13°) plus the 1° buffer spans ~2.2° × 2.1°, i.e. single-digit cells at
`Amon` resolution: ~9 × 1 032 × 2 × 8 ≈ **150 KB** per scenario run. The formula
is given rather than a bound because the cell count scales with basin extent; a
20° × 20° basin at 1° is ~400 cells, ~6 MB per run. Cheap either way — default-off
is a tidiness choice, not a cost one.

**Naming.** The owner wrote `save_gridded`; the existing config key, the current
`dev/workflows/climate_projections.md` contract, and every shipped config use
`save_grids`. This design **keeps `save_grids`** for continuity. Flagged as
**OQ-12**: the rename is one config key and a docs line, but it breaks the
manifest's config-file sha256 (R10), so it should ride with another config-key
commit rather than land alone.

### 5.9 Report stage

One job producing:

- `change_factors/{annual,monthly}.csv` — long format, **one row per
  `(dataset, institution, scenario, member, horizon, period, variable,
  statistic)`** with `value`, `absolute_value`, `units`, `status`,
  `reference_window`, `horizon_window`, `n_years`, `n_years_dropped`,
  `reference_series_key`. design-v2's `n_models_in_summary` column is **removed**
  — under N10 no row is a summary over models, so the column would be a
  contradiction in the schema.
- `change_factors/change_factors.nc` — the same content with coordinate metadata.
- `composition.csv` — §5.7. Written by stage B alongside the numbers, not by the
  report, so a report failure never leaves a table without its provenance.
- `report.md` — the ΔT/ΔP cloud, the seasonal change pattern, timeseries context,
  the composition table, and a **disclaimer block** carrying: requested vs
  effective reference window and whether it was clipped; the alignment result
  against `shared.historical_window`; the effective window length and any
  short-window warning; the spatial weighting scheme and its approximation label;
  the dry-month rule and threshold; the catalog snapshot date; and the count of
  requested-but-unresolved combinations by status.
- `provenance.json` — resolved sources (entry name, resolved URI, digest,
  acquisition window, actual coverage, calendar, grid-geometry check result,
  weighting scheme); requested and effective reference window plus the clip flag;
  `shared.historical_window` and the alignment-check result; horizon windows;
  reducer module hash; config digest; variable spec; catalog `meta`; run-level
  composition counts; flagged months.
- `plots/*.png` — **all declared**. Today 6 of 8 figures are undeclared, one is
  saved without an extension, and `gcm_timeseries.nc` is declared under the label
  `timeseries_csv` (confirmed in `Snakefile_climate_projections` rule 2.06).

**Figures under N10 — one point or trace per combination.** Three families, which
map onto the three currently pinned PNGs:

| Figure | Content under v2.0 | Today |
| --- | --- | --- |
| ΔT/ΔP cloud | one **point** per (model, scenario, member, horizon), styled by scenario, labelled by model | `projected_climate_statistics.png` (from `get_change_climate_proj_summary.py`) |
| Precipitation timeseries context | one **trace** per (model, scenario, member) | `precipitation_anomaly_projections_abs.png` |
| Temperature timeseries context | one **trace** per (model, scenario, member) | `temperature_anomaly_projections_abs.png` |

The two anomaly figures are today produced by `plot_proj_timeseries.py`, which
computes `quantile([0.05, 0.5, 0.95])` **across models** and renders them as a
`fill_between` band plus a "multi-model median" line. **That is exactly the
cross-combination statistic R3″ removes**, so those two figures change content —
envelope → per-combination traces — at the same paths. Both are manifest-pinned
by `size_bytes`, so this is a re-record; §8 carries it.

**The legacy summary artifacts stay, and R3″ does not touch them.**
`summary/annual_change_scalar_stats_summary.{nc,csv}` and
`..._summary_mean.csv` remain declared outputs at their current paths throughout
v2.0. Verified in `get_change_climate_proj_summary.py`: these are an
`open_mfdataset` **merge** across (model, scenario, horizon) carrying a `stats`
dimension of **per-series** statistics — `ds.sel(stats="mean")` selects a
per-series statistic, not a multi-model one. They contain no cross-combination
aggregation and therefore survive R3″ unchanged in kind. They are the migration's
evidence anchor (§8): the only WF2 numbers the baseline manifest pins strictly.
Superseding them with the `change_factors/` tables is a follow-up that must carry
its own manifest re-record; it is not part of this design.

### 5.10 Extension surface — contract changes, not free reads (G9)

design-v1 claimed every slot was "a read, not a pipeline". ext1-06 falsified that
and R4 narrowed the claim. Each extension is listed with **the contract change it
requires**:

| Slot | What it adds | Contract change required |
| --- | --- | --- |
| **S1 — observed climatology & trends** | Long-term means, seasonality, annual trends | A second source class at stage A with a **source-level frequency conversion** at §5.5's assertion point (daily → canonical monthly rate/state, unit handling for temp in K); a `provenance` axis in the series key and cache; possibly an acquisition window longer than `shared.historical_window`, which the store key fixes |
| **S2 — multi-dataset observed comparison** | ERA5 vs CHIRPS vs E-OBS over the same basin | S1, plus more than one store instance — stage 0 builds exactly the one `shared.clim_historical` store — so either a multi-source store spec (a **C1-scope change**, co-owned with WF1/WF3) or a WF2-private observed acquisition rule |
| **S3 — GCM historical bias diagnostics** | GCM-historical vs observed climatology (diagnostic only; delta-change already cancels bias) | S1, plus a resolution-reconciliation step between a ~1–2° `Amon` grid and a fine observed grid, and a decision on what "comparable" means spatially |
| **S4 — extremes / indicator set** | Wet/dry spells, hot days, seasonality indices | A **daily CMIP6 acquisition branch** — a generator change to `TABLE` (C7), ~30× volume, and 11 of the 46 monthly models lost (inventory §5.2) — a temporal-resolution axis in the series key, cache and layout, and probably a new dependency for standard indices (**OQ-7**, C5) |
| **S5 — grid-vs-cloud advisory** | Does the configured WF3 perturbation grid envelope the projected cloud? Emits a figure and a warning | **None in WF2.** But it *is* an ex-post statistic — it compares the cloud's extent against the perturbation grid — so under **R3″** it defers with S6 rather than being a v2.0 read. Reads `change_factors/*` + the WF3 config section, one-way; never writes WF3 config (N1/C2) |
| **S6 — ex-post ensemble statistics** *(new)* | Envelopes, percentile bands, ±σ, model-count thresholds, weighting, institution de-duplication, model-similarity interpretation | **None in WF2.** Reads `change_factors/*` and `composition.csv`. This is where everything R3′/R3″ removed from §5.6 lands, and the fact that it needs no contract change is the reason deferring it is free rather than a debt |

S5 and S6 are the only slots that cost no WF2 contract change — and both are now
out of v2.0, which is a coherent position rather than a coincidence: the cheap
extensions are cheap *because* they are pure reads of the tidy table, and R3″
defers exactly the pure reads. S1 is the gateway to S2 and S3; the v2.0 decisions
that keep it cheap are the `canonical:` variable spec (§5.5) and the source-level
assertion point.

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

- **A1 — declare the full `climate_store_spec`, accept the cost. SELECTED, and
  confirmed by the owner 2026-07-29.** Preserves G2, keeps one delineation for
  the project, no-op when WF1 has run. Cost: a fresh projections-only run pays
  the gridded observed extraction and inherits its network failure surface, for a
  polygon. Reasoning in §5.4.
- **A2 — keep reading `hydrology_model/staticgeoms/region.geojson`. RECORDED
  FALLBACK.** Drops G2 from v2.0 and defers the store declaration to whenever
  observed analysis lands and actually consumes the gridded file. Touches C1 not
  at all. Would be chosen if a first-run gridded observed extraction on a
  projections-only run were unacceptable; the owner answered that it is
  acceptable.
- **A3 — a separate WF2-private region-only producer** writing just the polygon
  from `shared.basin` + catalog. **Not chosen.** On C1's letter this is *not* an
  asymmetric `climate_store_spec` declaration — it is a different rule at a
  different path — so the constraint's text does not forbid it. But it creates a
  second delineation code path that can drift from `store_region.geojson`, which
  is precisely the bug class C1 exists to prevent. Rejected on the constraint's
  purpose, not its wording.
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

### 6.6 Catalog generation — how it was actually resolved

design-v2 recorded query-based catalog generation via `intake-esm` as a
**not-chosen** alternative on dependency grounds (C5, N4). It has since been
resolved a third way, outside this design: `dev/scripts/generate_cmip6_catalog.py`
performs a live `gcsfs` directory listing and renders the catalog, using a
library already in the environment (hydromt reads `gs://` through it) and adding
**no new dependency**. Coverage went from 69 hand-curated sources to 2 426.

Recorded here because it changes two judgements design-v2 made:

- The `intake-esm` candidate in **OQ-7 is withdrawn** — the need it addressed is
  met. (Its second stated benefit, "ensembles large enough to justify percentile
  envelopes", is moot under R3″ regardless.)
- The alternative that remains live is not *whether* to generate but *what to
  generate against*: a live crawl (ground truth, slow, snapshot-dated) versus the
  published `pangeo-cmip6.csv` index (fast, but measurably incomplete at store
  level — the inventory documents `NorESM2-LM ssp585 Amon pr` existing in the
  bucket and absent from the index). The generator takes the crawl and records
  `crawled_on`; §6.12 and **OQ-14** carry the staleness question that creates.

### 6.7 Long-format Parquet instead of netCDF for `series/`

**Not chosen** — netCDF is the repo's stated interchange format across
R/Python/Julia, and the series carry coordinate metadata that survives the
netCDF round-trip. Under R5 the series is user-facing, which strengthens rather
than weakens this: a consumer already reads netCDF for every other CST product.
The change-factor *tables* are CSV, which covers the tidy-consumer case.

### 6.8 Failure-tolerant status-artifact contract instead of fail-fast (D4)

Every reduce job emits an **always-written required status artifact** plus an
**optional data artifact**; stage B discovers the surviving data through a
Snakemake `checkpoint`. This is implementable, and it is the design ext1-02's
`suggested_fix` points at. **Not chosen for v2.0** because:

1. It adds a checkpoint and a second artifact class to a workflow whose whole
   point in this revision is fewer moving parts (11 rules → 8).
2. It makes the run's composition depend on transient network state: two runs of
   the same config produce different change factors with no config difference and
   no error. That is a reproducibility defect. Note this argument **survives
   R3′**: R3′ makes *store-level* raggedness normal and enumerable at DAG build,
   which is reproducible; it says nothing that makes *network-level* raggedness
   acceptable, and the two are now cleanly separated by §4 criterion 7.
3. `--keep-going`, which WF2 is already invoked with, delivers the "see all
   failures in one pass" benefit without changing the artifact contract.

Revisitable; the evidence that settles it is **OQ-11**.

### 6.9 Retire `save_grids`

The simplest resolution of the old OQ-8: delete the branch, document the lost
functionality, migrate. **Rejected by owner ruling R2**, and now doubly so by
**R5**, which makes the gridded series a named deliverable rather than a legacy
option. §5.8 takes the other option and makes the branch declared.

### 6.10 Relocate WF2 output to `{project_dir}/climate/` (D3)

design-v1 raised this as OQ-3 while step 3 simultaneously consumed it — the
sequencing contradiction risk-04 named. **Decision D3: keep
`{project_dir}/climate_projections/{clim_project}/` in v2.0.** The rename's
motivation was that "climate_projections" would understate a workflow that also
did observed analysis; **R4 removes that**. Keeping the root also keeps every
manifest-pinned path byte-identical through the migration (§8). The rename
becomes a follow-up tied to whenever observed analysis lands, alongside D5.
Owner-reversible; reversing it re-opens the manifest-evidence problem, which §8
would then have to solve with an explicit old→new path map.

### 6.11 Member pairing — alternatives to D7's strict same-member rule

- **Fall back to a designated historical member** (e.g. the model's `r1i1p1f1`,
  or its lowest-sorting member) when the scenario member has no historical
  counterpart. **Not chosen.** The member label encodes physics and forcing
  variant, so the resulting delta would difference two runs that differ in more
  than the scenario, and the row would silently mean something different from its
  neighbours. Preferable only if a downstream use needed maximum row count and
  could tolerate heterogeneous reference semantics — which is the opposite of
  what a change factor is for. If ever adopted it must carry its own status code
  in `composition.csv`, never be invisible.
- **Reference against the model's historical ensemble mean.** Would resolve every
  scenario member. **Barred by N10** (R3′: members are never averaged), and
  independently questionable: differencing one realization against a mean mixes
  internal variability treatments between numerator and denominator.
- **Keep today's raise on asymmetric member sets.** This is the status quo
  (`get_change_climate_proj.py`, guard t260720d / D-MEM). **Not chosen** because
  R3′ rules that raggedness must not be an error; the guard's *purpose* is
  preserved by making the intersection explicit in `composition.csv` (§5.7 D7).
  Preferable if a user needed a hard stop on any ensemble thinning — obtainable
  downstream by asserting on `composition.csv`, which is strictly more flexible.

### 6.12 Multi-version glob — alternatives to D8's uniqueness assertion

- **Pin `{grid_label}/{version}` in the generated URIs.** Removes the ambiguity at
  the source and makes a run exactly reproducible against a named publication
  version. **Not chosen for v2.0**: it is a generator change (C7) that freezes
  every entry to a snapshot, so a re-published store becomes a read failure
  rather than a transparent upgrade, and the catalog's staleness surface grows
  from "which members exist" to "which bytes exist". Recorded as **OQ-14** with
  the evidence that would select it.
- **Prefer the latest version when the glob matches several.** Plausible and
  cheap, but it makes the resolution rule depend on lexical version ordering
  inside a URI glob that hydromt, not this repo, expands — so it is not
  implementable without either re-implementing resolution (the exact drift risk-08
  warns about) or a generator change, in which case pinning is strictly better.
- **De-duplicate the time axis silently** (`drop_duplicates(dim="time")`).
  **Rejected**: it chooses one publication arbitrarily and records nothing —
  §4 criterion 4 verbatim.

### 6.13 Keep a cross-combination statistics layer inside v2.0

Emit percentile envelopes and model-count-gated summaries from WF2 itself, as
design-v2 specified and as `plot_proj_timeseries.py` does today.
**Rejected by owner rulings R3′ and R3″.** Recording why the rejection is
coherent rather than merely instructed: any such statistic is a function of
*which combinations happened to resolve*, which under a ragged store and a union
member rule is data-dependent — so the same config against a re-crawled catalog
can move an envelope with no code change and no config change. Deferring the
layer to S6 makes that dependence the consumer's explicit choice, over a table
that carries `composition.csv` beside it. Preferable to reverse only if WF2's
report were the terminal artifact for a user with no downstream step — which the
overlay's role (C2) says it is not.

---

## 7. Consequences and risks

**Observable consequences (falsifiable).**

1. WF2 runs to completion with no `hydrology_model/` directory on disk.
2. `snakemake -n` on the seed config lists **15** jobs on a fresh `project_dir`
   and **14** when `climate_historical/era5_20000101_20201231/` already exists —
   against **22** today. Counts exclude the `all` target job. The number is
   derived from the resolved combination set, not asserted (§5.2, §9).
3. A second run with a changed `future_horizons` entry schedules **zero**
   `reduce_gcm_series` jobs and performs zero network reads.
4. **Regenerating `config/catalogs/cmip6_data.yml` after the store gains members
   schedules zero reduce jobs**; regenerating with a changed shared
   driver/adapter block re-derives **every** series; changing one entry's URI
   re-derives exactly that entry's series.
5. Editing an enumerated reducer module re-derives **every** series without any
   manual version bump; editing an unrelated `blueearth_cst` module re-derives
   none.
6. A series file whose `cst_series_digest` or `cst_schema_version` does not match
   the expected value makes stage B **fail**, naming the series and both values.
7. A config naming a model absent from the catalog fails at **DAG build** naming
   the model. **User-visible behavior change.**
8. A config naming a (model, scenario) or a member the store does not publish
   **does not fail**: the combination appears in `composition.csv` with its status
   code, is summarized on stderr at DAG build, and the run proceeds with the rest.
   **User-visible behavior change** — today the combination silently produces an
   empty dataset and a thinner ensemble with no record.
9. A scenario member whose model does not publish the same member for
   `historical` is skipped with `status = reference_member_unpublished`. **This
   replaces a raise** (`asymmetric hist/clim members`); the inventory measures 18
   such (model, scenario) pairs under the shipped template `members:` list, all
   of which go from failing the run to appearing in `composition.csv`.
10. A remote read failure on a **resolved** source halts the run with the failing
    series key named. No run produces a smaller-than-resolved set.
11. A source whose glob resolved to more than one publication version — i.e. whose
    time axis contains duplicate timestamps — makes the reducer **raise**, naming
    the source and the first duplicated timestamp. Nothing is de-duplicated.
12. A reference window ending after 2014-12-31 produces a DAG-build warning naming
    requested vs effective window, `reference_window_clipped: true` in
    `provenance.json`, and a `report.md` disclaimer. It never raises. A reference
    window lying entirely after 2014-12-31 raises.
13. `change_factors/monthly.csv` exists with 12 rows per
    `(dataset, scenario, member, horizon, variable, statistic)`.
14. **No figure, table, or file WF2 writes contains a value aggregated across
    models, scenarios, or members.** Every row and every plotted element is one
    (model, scenario, member, horizon) combination. Grepping the v2.0 report code
    for a cross-combination reduction returns nothing.
15. `series/{series_key}.nc` exists after a run, survives a re-run, conforms to
    the §5.3 schema, and is readable without any WF2 code.
16. `save_grids: true` adds declared `grids/series/*.nc` and `grids/change/*.nc`
    outputs and **no additional jobs** (still 15/14 on the seed config);
    `save_grids: false` leaves those paths absent from the DAG entirely.
17. `provenance.json` and `composition.csv` together name every requested
    combination, its resolution status, both reference windows, the alignment
    result, the weighting scheme, and the catalog snapshot date.
18. Every figure WF2 writes is a declared output.
19. A source whose latitude/longitude coordinates are not 1-D and monotonic makes
    the reducer **raise**, naming the source.

**Risks.**

- **R1 — the third `climate_store_spec` declaration diverges (C1).** Mitigation:
  the declaration is generated from the shared helper, and a test asserts the
  three declarations produce identical input sets (§9).
- **R2 — dropping the hist→fut ordering edge exposes a directory race.** Today's
  `get_stats_climate_proj.py` uses `os.mkdir` guarded by `os.path.exists`;
  concurrent reduce jobs would race. Mitigation: `makedirs(exist_ok=True)`.
- **R3 — the methodological changes move every number.** Weighting, calendar,
  rounding, statistic set, coverage policy, and the removal of cross-combination
  figures. Mitigation: **one cause per commit** (§8 steps 5a–5f, 6a–6c), each
  with its own re-record.
- **R4 — persistent series accumulate.** Correctness is no longer at stake
  (explicit input list + digest + schema-version assertion, §5.3), so this is disk
  hygiene: stale-key pruning is a follow-up.
- **R5 — fail-fast turns transient network flakiness into a failed run.**
  Mitigation: the cache means a retry re-derives only what failed, and
  `--keep-going` surfaces all failures in one pass. Falsifiable trigger for
  revisiting: **OQ-11**.
- **R6 — the report becomes the only place a number is stated.** Mitigation: the
  CSV, netCDF and composition tables are stage-B outputs, independent of stage C.
- **R7 — a first WF2 run on a fresh project pays the climate-store build.**
  On a fresh projections-only project the WF1 decoupling is a **cost transfer**
  (§5.4 D2), owner-accepted. Mitigation: no-op when WF1 has run; the store is not
  on the analysis path; it can be pre-built once.
- **R8 — the geometry check converts a silent bias into a blocked source.** A
  non-1-D-rectilinear `Amon` source that today is (wrongly) averaged unweighted
  will refuse to reduce. This design does not assert how many of the catalog's
  289 entries that affects — the check-and-fail contract does not require
  knowing. The failure is loud and names the source. **The generated catalog
  raises the exposure**: the reachable model set went from 23 to 65 for
  historical, so grid heterogeneity that the hand-curated subset never met is now
  reachable. Falsifiable and cheap to measure: run the geometry check over the
  full catalog once and count refusals (that measurement is **not** a v2.0 gate;
  it is the evidence for OQ-10).
- **R9 — the DAG-build validator drifts from the catalog format.** Materially
  reduced by C7 (the format is repo-owned) and by the validator's narrower job
  (§5.7). Mitigations retained: minimal logic, unknown constructs are errors,
  `meta.generated_by` assertion, and an integration-marked cross-check against
  `hydromt.DataCatalog(...).sources`.
- **R10 — the baseline manifest pins a verbatim sha256 of the seed config file**,
  so any commit that adds or renames a config key fails that target even when
  every number is identical. Mitigation: §4 criterion 1's value-neutral /
  manifest-clean split (§8).
- **R11 — *(new)* the catalog is a dated snapshot.** `meta.crawled_on` is
  2026-07-29; the bucket changes. A withdrawn or re-published store resolves at
  DAG build and fails at read time. Correctly classified as a read failure (D4),
  but it means "resolves" guarantees *observed at crawl time*, not *exists now*.
  Mitigation: the snapshot date is recorded on every series, in
  `composition.csv`, in `provenance.json`, and in the report's disclaimer, so the
  age is visible at the moment it matters. Regeneration cadence is **OQ-14**.
- **R12 — *(new)* skips are quiet by design, and quiet can become invisible.**
  R3′ requires that raggedness not fail the run, so a config typo in a *scenario*
  name (unlike a model name) now yields skips rather than an error, and a user
  who ignores `composition.csv` sees a thinner run with no complaint. Mitigations:
  the DAG-build stderr summary prints counts by status before any job runs; the
  report's disclaimer block carries the unresolved count; and `composition.csv` is
  a required output, so it cannot be forgotten by the pipeline even if it is
  ignored by the reader. Residual risk accepted as the direct cost of R3′.
- **R13 — *(new)* the request surface got much larger.** The catalog exposes 65
  historical models and up to 96 members per (model, scenario). A config change
  of a few characters can multiply the network-bound fan-out by an order of
  magnitude. Mitigations: `members:` takes an explicit list only (no `all`,
  §5.7); the DAG-build stderr summary prints the resolved reduce-job count before
  any job runs, so the size of a run is visible from `--dry-run`. Not mitigated:
  nothing caps the count. A cap is **OQ-13**.

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
   intermediates that steps 2–4 restructure.
2. **The config snapshot is a verbatim copy of the seed config file**
   (`copy_config_files.py` reads `config_snake` and writes it unchanged). Its
   sha256 is the hash of `config/workflows/snake_config_model_test.yml`. Any
   commit that adds, removes, or renames a **seed** config key breaks that target
   *even when no number moves* — hence §4 criterion 1's split. Note the
   corollary: a change to `snake_config.template.yml` is **not** pinned, so
   template-only changes are manifest-clean.
3. **A rename severs the evidence chain.** **D3** (§6.10) removes the problem for
   v2.0 by keeping the WF2 root and every pinned path byte-identical; no step
   below renames a manifest-pinned path.

### Commit plan

Sequenced so every value-neutral step lands before the first value-changing one.

§5.2's 8-rule set and 15/14 job counts describe the **end state**. Intermediate
commits carry a transitional rule set: the existing `plot_climate_proj_timeseries`
rule survives until step 7, and `projected_climate_statistics.png` moves from
`monthly_change_scalar_merge` into `derive_change_factors` at step 4 **keeping its
path**. The `snakemake -n` comparison in §9 is run per commit against that
commit's expected set.

| # | Commit | Value-neutral? | Manifest-clean? | Gate |
| --- | --- | --- | --- | --- |
| 1 | Declare `extract_climate_grid` in WF2 from `climate_store_spec` (D2/A1); read `store_region.geojson` under `ancient()` | **Yes** (bounds measured identical on the fixture, §5.4) | Yes | `check_baseline`; re-verify bounds equality; identical-input-set test (R1) |
| 2 | Persistent `series/`: drop `temp()`, add the entry+resolved-URI+region+module digest and the `cst_series_digest` / `cst_schema_version` attributes, the §5.3 schema, `makedirs(exist_ok=True)`, drop the hist→fut ordering edge, fix the acquisition-window contract | **Yes** (same values, different lifetime and metadata) | Yes | `check_baseline`; second run schedules zero reduce jobs; the seven cache tests (§9) incl. catalog-regeneration invariance; the series-schema test |
| 3 | Collapse `monthly_stats_hist`/`monthly_stats_fut` into `reduce_gcm_series` over `{series_key}`; `members` becomes a wildcard; collapse the three log gathers into one. Intermediate filenames change; **no manifest-pinned path moves** | **Yes** | Yes | `check_baseline`; `pytest tests/test_cli.py`; `semantic_tree_diff` with an explicit old→new map for intermediates |
| 4 | Collapse `monthly_change` + `monthly_change_scalar_merge` into `derive_change_factors`; the §5.7 resolution ladder + **D7** pairing + `composition.csv`; delete the dummy-netCDF path and `filter_nonempty`; fail-fast (D4); the time-axis uniqueness assertion (D8); explicit expanded input list + digest assertion | **Yes** for combinations that resolve on the seed (all 6 do); **behavior change** for absent, unpairable, and failing ones (consequences 7–11) | Yes (`composition.csv` is a new path, not a pinned one) | `check_baseline`; resolution-ladder unit tests + integration cross-check; pairing tests; stale-series digest test; duplicate-time test |
| 5a | Cos-latitude weighting + the 1-D/monotonic geometry check | **No — value-changing** | Yes | Re-record; diff **is** the weighting effect; grid-geometry tests |
| 5b | Calendar-aware month-length weighting on annual aggregates | **No — value-changing** | Yes | Re-record; diff is the calendar effect; 360-day vs standard synthetic tests |
| 5c | Drop the stage-A 2-decimal rounding | **No — value-changing** | Yes | Re-record; diff is the rounding floor |
| 5d | Default statistic set (`mean`, `median`, `std`; tail quantiles opt-in and sample-size-labelled) | **No — output-set change** | **No** — the summary CSVs lose columns | Re-record of the two summary CSVs with the column diff shown |
| 5e | Variable spec (`canonical`/`change`); reference-window clip + per-condition warnings + alignment check | Output-neutral on the seed (`[1990, 2010]` needs no clip) | **No** — adds config keys → config-target re-record | Reference-window tests; property tests |
| 5f | **OQ-4's 30-year reference window (1985–2014) in `snake_config.template.yml` only**; documentation states the recommendation. Test fixtures unchanged | **Yes** — the seed fixture keeps `[1990, 2010]`, so no number moves | **Yes** — the manifest pins the *seed* config, not the template (§8 fact 2) | `check_baseline`; a clip-warning test on a 1985–2014 config confirming neither the clip nor the short-window warning fires |
| 6a | Monthly change-factor table; tidy CSV schema (no `n_models_in_summary`); `provenance.json` | Additive | No | Schema and row-count tests |
| 6b | Dry-month rule + coverage/partial-year policy | **No — value-changing** (partial years now dropped) | No | Near-zero-reference, missing-month, partial-year synthetic tests |
| 6c | **Remove cross-combination statistics (R3′/R3″)**: no member averaging, no model-level collapse, no percentile envelope, no ±σ, no min–max; the two anomaly figures become one trace per combination; composition reporting replaces the ensemble summary | **No — value-changing** for the two anomaly figures (the three `summary/*` targets are *unaffected*: they carry no cross-model statistic, §5.9) | **No** | Re-record scoped to the **two anomaly PNG targets only**, with old/new figures shown; the three `summary/*` targets and the config target must match **without** re-record, which is the check that the change really was confined to the figures. Plus: a test asserting no cross-combination reduction in the report path; composition-record tests |
| 7 | Report stage; declare every figure; declare the optional gridded branch (R2/R5) incl. `grids/series/`; retire the loose-PNG set | Additive + plot-set change | **No** — the pinned PNG set changes | Visual QA; migration note; re-record of the pinned PNG targets; `save_grids: true` declares-not-adds-jobs dry-run |

Steps 1–2 remain independently shippable: together they deliver G2 and G5 without
touching a single computed value.

**Why 5 and 6 are decomposed.** design-v1 bundled independent value-changing items
into one step and gated it on "a diff attributed per cause". That gate cannot
execute — attributing a single diff to weighting versus calendar versus rounding
would need a flag matrix this design does not specify (§4 criterion 5). Splitting
makes each sub-commit's diff *be* its cause.

**The seed-fixture variant of step 5f, kept out of the table deliberately.** If
the owner intends OQ-4's window to move the **test fixtures** as well as the
template, that is a *different* commit, not a wider version of 5f: it moves every
number in all three pinned `summary/*` targets and breaks the config target's
sha256, so it is value-changing **and** manifest-breaking and needs a full
documented re-record with a characterized diff. It is not listed as a row because
§8's own rule is that the two properties are never mixed in one commit, and a
row whose classification depends on an unmade decision cannot be checked against
§4 criterion 1. Enabling it is an owner call that adds one row after 5f.

**What is *not* in the commit plan, and why.** Removing
`ensemble.min_sources` / `ensemble.min_models_for_envelope` needs no commit row:
neither key exists in any `config/workflows/*.yml`. They were design-v2
specifications that were never implemented, so design-v3 deletes them from the
spec and nothing is deleted from the repo. No config-target re-record is owed for
them.

**Log paths.** Collapsing three `gather_*_logs` rules into one (step 3) changes
which log files rule `all` requires. Log paths follow the rule declarations, not a
filename convention (per commit `1c3013c`). **No baseline manifest target is a
log file**, so this is a change to rule `all`'s input list and to `dev/workflows/`
documentation.

**Derived-artifact re-check.** The existing task brief
`dev/working/2026-07-29_wf2-v2-decouple-and-cache.md` covers steps 1–2 and
**must be re-checked before dispatch** against: D2 (retained, cost accepted, owner
-confirmed), the step-2 digest contract (entry + **resolved URI**, `placeholders`
and `meta` **excluded**, mechanical module hash, digest and schema version written
as series attributes), the §5.3 series schema now that the series is a
deliverable, and the `ancient()` treatment of `store_region.geojson`.

---

## 9. Validation plan

**Per-commit gates.**

- `pytest tests/test_cli.py` — dry-runs all three Snakefiles (cheapest DAG check).
- `pytest tests/` — full suite; must stay green and additive.
- `snakemake -n` on the seed config before and after each structural commit, with
  the job count and rule set compared explicitly. The **test derives the expected
  count from the resolved combination set** (§5.2's formula), never from a
  hard-coded literal; 15/14 are measured values for the seed config against the
  2026-07-29 catalog snapshot, not the assertion.
- `dev/scripts/check_baseline.py check` on `test_case/test_local` for steps 1–4
  and 5f. **CI cannot run this** (C4) — local gate.
- `dev/scripts/semantic_tree_diff.py` on the WF2 output subtree, with an explicit
  old→new path map for renamed intermediates in step 3.

**Targeted checks.**

- **Region equality (step 1).** Re-run the bounds comparison of
  `store_region.geojson` versus `hydrology_model/staticgeoms/region.geojson` and
  assert the buffered bbox selects the same GCM cell set. If they diverge on a
  future basin, step 1 is reclassified as value-changing.
- **Identical input sets (R1).** A test asserting the three `climate_store_spec`
  declarations produce byte-identical input lists.
- **Cache correctness (step 2).** Seven cases: (a) run, touch nothing, re-run →
  zero reduce jobs; (b) change a `future_horizons` entry → zero reduce jobs;
  (c) edit an unused catalog entry → zero reduce jobs; (d) edit a used entry's
  URI → exactly that series re-derives; (e) edit an enumerated reducer module →
  all series re-derive; (f) hand-plant a series whose `cst_series_digest` or
  `cst_schema_version` is wrong → stage B fails; **(g) *(new)* add a member to a
  used entry's `placeholders.member` and bump `meta.crawled_on` → zero reduce
  jobs re-derive, while a change to the shared driver/adapter block re-derives
  all.**
- **Series schema (step 2).** Open a produced `series/*.nc` with plain xarray, no
  WF2 code on the path, and assert the §5.3 dimensions, scalar coordinates,
  variable attributes and global attributes are all present and typed.
- **Catalog parse (step 4).** Assert `yaml.safe_load` on
  `config/catalogs/cmip6_data.yml` yields a **merged** mapping for a non-anchor
  entry (`driver`, `data_adapter` and `metadata` present on an entry declared
  with `<<:`), and that `meta.generated_by` is the expected generator path.
- **Resolution ladder (step 4).** One test per status code in §5.7's table,
  including: an unknown model → raise; a model with no entry for a requested
  scenario → `scenario_not_published` skip; a member absent from a published
  entry → `member_not_published` skip; **a model with scenario entries and no
  historical entry** → `no_historical_entry` skip; a zero-resolution config →
  raise listing every requested combination. Plus one integration-marked test
  cross-checking the accept list against `hydromt.DataCatalog(...).sources`, and
  one asserting an entry with an unrecognized top-level construct errors naming
  the key.
- **Member pairing (step 4, D7).** A model publishing `r1i1p1f1` and `r1i1p1f2`
  for a scenario but only `r1i1p1f1` for historical → exactly one resolved
  combination, one `reference_member_unpublished` row, **no raise**, and the
  reference series key on the resolved row points at the same member.
- **Composition record (steps 4, 6c).** Every requested combination appears
  exactly once; status codes are drawn only from §5.7's table; resolved rows have
  both series keys populated and skipped rows have neither; run-level counts in
  `provenance.json` match the CSV.
- **Coverage assertion.** Request an analysis window outside a series' recorded
  acquisition coverage → stage B fails naming both windows.
- **Reference window (step 5e/5f).** Request a reference ending 2020-12-31 →
  warning, `reference_window_effective` = 2000–2014, disclaimer, no raise. Request
  a reference entirely after 2014 → raise. Request 1985–2014 → **no clip warning
  and no short-window warning** (the OQ-4 closure's acceptance check). Request a
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
  populated, report footnote present.
- **No aggregation (step 6c).** The strongest single test of R3′/R3″: two members
  of one model plus one member of another, all resolved → `change_factors/*.csv`
  contains exactly three rows per (scenario, horizon, variable, statistic), each
  member's value distinct and traceable; **no row's value equals the mean of any
  other rows**; no figure contains a filled band or a "multi-model" label; and
  `composition.csv` reports 2 models / 3 members / correct institution counts.
- **Time-axis uniqueness (step 4, D8).** Synthetic source with a duplicated
  monthly timestamp → reducer raises naming the source and the timestamp; nothing
  is de-duplicated.
- **Grid geometry (step 5a).** Synthetic source with 2-D lat/lon coordinates →
  reducer raises; 1-D non-monotonic → raises; 1-D monotonic → cos-latitude
  weights applied and `cst_weighting_scheme` recorded.
- **Fail-fast (step 4).** A reduce job forced to raise → the run halts, the
  series key appears in the log, no dummy netCDF is written, and no summary is
  produced. With `--keep-going`, all sibling reduce jobs still complete.
- **Gridded branch (step 7).** `save_grids: true` dry-run → same job count as
  `false`, plus declared `grids/series/*.nc` and `grids/change/*.nc`;
  `save_grids: false` → those paths absent from the DAG entirely. On a real small
  run, assert `grids/series/*.nc` has dims `(time, lat, lon)`, that its
  cos-latitude-weighted spatial mean reproduces the basin series to floating-point
  tolerance, and that `cst_weighting_scheme` is the pre-reduction value.
- **Characterized diff (steps 5a–5d, 6b, 6c).** Old versus new outputs on the seed
  config, recorded **once per sub-step**. Attribution per cause comes from the
  commit boundaries, not from a flag matrix: each sub-step's diff *is* its cause.

**Not validated here.** Whether the ensemble is adequate for a given decision
(that is S6/N10, downstream); whether the cos-latitude approximation error is
material for any particular model (**OQ-10**); whether 1985–2014 is the right
window for a given basin. These are user judgements the report surfaces, not
gates.

---

## 10. Open questions

- **OQ-1 (D5).** Extend `Snakefile_climate_projections` in place, or open a 4th
  entry point? *Recommendation: extend in place for v2.0 (§6.3).*
- **OQ-2.** Does this open **Phase 4**, or land as an unnumbered milestone?
- **OQ-3 — CLOSED by D3 (§6.10).** WF2 output stays at
  `climate_projections/{clim_project}/` for v2.0.
- **OQ-4 — CLOSED by the owner, 2026-07-29: 30 years, 1985–2014.** Lands in
  `snake_config.template.yml` (step 5f); the test fixtures deliberately keep
  `[1990, 2010]` (§5.4, with the reading flagged for correction).
- **OQ-5.** Whether a daily CMIP6 branch (S4) is ever in scope given ~30× the
  volume and the 11 monthly-only models it would drop (inventory §5.2).
- **OQ-6 — CLOSED by rulings R3′ and R3″.** The residual policy design-v2 recorded
  as N9 (institution de-duplication, performance weighting) is not "not applied";
  it is **downstream** (S6, N10).
- **OQ-7 (dependencies — asks, C5/N4). None is adopted.** `intake-esm` is
  **withdrawn** — catalog generation landed via a repo-owned `gcsfs` crawler with
  no new dependency (§6.6). Standing candidates: `xclim` (calendar/unit handling
  and standard indicators — relevant to S4); `regionmask` (fractional-area polygon
  masking — relevant if §5.3's regional sampling choice is revisited).
- **OQ-8 — CLOSED by ruling R2**, and reinforced by R5: `save_grids` is retained,
  default off, with declared outputs including the gridded series.
- **OQ-9.** Default value of `relative_change.min_reference` per variable, in
  canonical units. The rule (§5.6) is complete without it. *Evidence that settles
  it:* the distribution of monthly reference-mean precipitation on the seed basin
  plus one strongly seasonal basin — choose the threshold below which the relative
  factor's sampling spread exceeds the factor itself.
- **OQ-10.** Should the generator stop dropping `lat_bnds` / `lon_bnds` / `bnds`
  so the reducer can compute true cell areas instead of the cos-latitude
  approximation? Now a generator change (C7). *Evidence:* measure the basin-mean
  difference between cos-latitude weights and bounds-derived cell areas for one
  catalog model on a non-uniformly-spaced grid at the seed basin; if it is below
  the reporting precision of the change factors, keep the approximation and the
  label. The R8 refusal count over the full 289-entry catalog is the companion
  measurement.
- **OQ-11.** Revisit fail-fast (D4) in favour of §6.8's status-artifact /
  checkpoint contract? *Evidence:* the observed remote-read failure rate across
  real runs, from `logs/` and `benchmarks/`.
- **OQ-12 (new).** Rename `save_grids` → `save_gridded`, as the owner's R5 wording
  used? The design keeps `save_grids` for continuity with the config, the current
  behavioral contract doc, and every shipped config file. *Cost if taken:* one
  config key, one docs line — but it breaks the manifest's seed-config sha256
  (R10), so it should ride with another config-key commit (5e) rather than land
  alone. Owner call; no evidence needed.
- **OQ-13 (new).** How should a user express "many members" without a fan-out
  surprise? v2.0 takes explicit lists only (§5.7). Two candidates: an `all` token
  paired with a mandatory per-run job cap, and a **per-model `members:` mapping**
  (which `wf2-cmip6-store-inventory.md` §4 already records as a config-schema plus
  script change, deliberately not done). *Evidence that settles it:* whether real
  use asks for large single-model ensembles (favours the cap) or for maximal model
  coverage at one member each (favours the per-model mapping — the inventory
  measures the four-label list reaching 45 models with 18 unpairable pairs, which
  a per-model mapping would reduce to zero).
- **OQ-14 (new).** Catalog snapshot policy: how often is
  `generate_cmip6_catalog.py` re-run, and should the generated URIs pin
  `{grid_label}/{version}` (§6.12) so a run is reproducible against a named
  publication? *Evidence:* the observed rate of resolve-then-fail-at-read events
  (R11) and of multi-version globs (D8) across real runs. Pinning trades
  transparent upgrades for reproducibility; the measurement says which costs more.
- **OQ-15 (new).** Should the generator's `REQUIRED_VARS` widen beyond
  `{pr, tas}`? The catalog renames `rsds → kin` and `psl → press_msl`, so those
  variables are nameable but not guaranteed for a listed member (§5.5), which
  turns a config extension into a read-time failure rather than a resolution
  skip. *Evidence:* whether any planned product needs `kin` or `press_msl`; the
  inventory §5.3 already measures coverage (57 of 64 historical models for `rsds`,
  64 for `psl`), so the cost of requiring them is a known reduction in the model
  set.

---

## 11. Revision log

- **2026-07-29 — revision 1** (`design-v1.md`). Initial draft, grounded in
  `Snakefile_climate_projections`, `blueearth_cst/projections/*.py`,
  `blueearth_cst/climate_analysis/*.py`, `blueearth_cst/shared/snake_utils.py`
  (`climate_store_spec`), `config/catalogs/cmip6_data.yml`, and
  `config/workflows/snake_config_model_test.yml`. Two facts measured rather than
  assumed: the two region polygons have identical bounds on the seed fixture, and
  the CMIP6 sources are `Amon` so the monthly sum/mean dispatch is a no-op.
- **2026-07-29 — revision 2** (`design-v2.md`). Authored against G1 rulings R1–R4
  following round-1 internal (Fable / `critical-thinker`) and external
  (gpt-5.6-sol) review; all 19 findings dispositioned in `ledger.md`. Scope
  narrowed to GCM projections analysis (R4/N7); clip-never-splice reference (R1,
  D1); the store declaration accepted as a named cost (D2); `save_grids` as a
  declared branch (R2); ensemble treatment set to the unique model (R3);
  fail-fast runtime semantics (D4); the cache and acquisition contract pinned;
  method edge cases specified; job arithmetic and manifest coverage corrected;
  the commit plan decomposed into 5a–5e / 6a–6c.
- **2026-07-29 — revision 3** (this file). Authored against owner rulings **R3′**,
  **R3″** and **R5**, the **D2 → A1** and **OQ-4 → 1985–2014** confirmations, and
  the **generated** CMIP6 catalog (commit `f8194e8`, 289 entries / 2 426 sources).
  Substantive changes:
  - **All aggregation removed (R3′ / R3″).** §5.6's ensemble section is a
    deletion, not a rewrite: member averaging, the unique-model sampling unit,
    `ensemble.min_models_for_envelope`, envelope suppression, the min–max range
    and the whole `ensemble:` block are gone. **N10** states the rule; **N9** is
    restated from "not applied" to "downstream"; the per-series / cross-
    combination boundary is written into §2 so the two senses of "statistic"
    cannot be conflated. Neither deleted key ever shipped in a config, so no
    commit row and no manifest re-record is owed. §5.10 gains slot **S6** as the
    ex-post statistics' home, and **S5 defers with it** because a grid-vs-cloud
    comparison is itself a cross-combination statistic.
  - **The monthly series became a product (R5).** §2 gains a declared-output
    contract; §5.3 gains a full series schema, a stable naming rule and a
    retention rule. **R5's gridded ask is reconciled explicitly**: it is the
    monthly series *on the source grid* (`time × lat × lon`, pre-reduction), so
    §5.8 supersedes design-v2's two 12-month climatology families with
    `grids/series/{series_key}.nc` (a strict superset — the climatology is a
    `groupby("time.month").mean()` of it) and retains the change grid, which is a
    stage-B product not derivable from the series. The "adds no jobs, no extra
    network" argument holds more strongly for the series than it did for the
    climatologies. `save_grids` is kept over the owner's `save_gridded` wording;
    flagged as **OQ-12**.
  - **§5.7 rewritten around the generated catalog.** Entries are now one per
    (model, scenario) with `member` the only placeholder, so validation is a key
    lookup plus a membership test rather than a placeholder cross-product —
    simpler *and* stronger, because membership is a live-crawl fact. `members:`
    is specified as *requested ∩ published*, unioned across combinations (R3′),
    with **"all available" deliberately not expressible** and the measured
    fan-out reason (EC-Earth3 publishes 96 ssp245 members). A total **resolution
    ladder** replaces design-v2's implicit checks, including the
    `no_historical_entry` row the store actually exhibits
    (`DKRZ/MPI-ESM1-2-HR`). **D6** deletes `ensemble.min_sources` — which
    conflated absence with failure — in favour of a non-configurable
    zero-resolution error, and §4 gains **criterion 7** stating why. **D7** sets
    strict same-member pairing and states openly that it converts today's
    `asymmetric hist/clim members` raise into a recorded skip, preserving the
    guard's purpose (no silent shrink) via `composition.csv`. **D8** answers the
    `*/*` multi-version glob with a time-axis uniqueness assertion rather than
    silent de-duplication. risk-08's drift concern is revisited and reduced: the
    format is now repo-owned (**C7**), and a `meta.generated_by` assertion makes
    C7 executable. New risk **R11** records the snapshot's staleness surface.
  - **The composition record is a first-class output** (R5 deliverable 3),
    written by stage B beside the numbers, carrying one row per *requested*
    combination with a status code — the artifact that makes "not published"
    visible without making it an error, and the input any ex-post statistic (S6)
    needs.
  - **The digest was corrected for a generated catalog.** `placeholders` and
    `meta` are excluded and the **resolved** URI is included, so regenerating
    after the store gains a member re-derives zero series while a change to the
    shared driver/adapter block re-derives all. Both are cache tests (§9 case g).
    Merge-key resolution under `yaml.safe_load` was verified, not assumed.
  - **OQ-4 closed, with its mechanics stated.** `historical_year_range` is
    required, not defaulted, so the closure lands in the template (new step 5f,
    value-neutral and manifest-clean because the manifest pins the *seed* config)
    with the fixtures deliberately unmoved to protect per-cause diff attribution.
    Flagged for owner correction if the seed was meant to move.
  - **§8 and §9 recomputed.** Job counts become a formula plus one measured seed
    example (6 resolved combinations + 3 references = 9 reduce, 15/14 total,
    verified against the generated catalog), with the corollary that the
    historical series set is *derived* — `NCC/NorCPM1` produces zero jobs. Step
    6c is re-aimed from "ensemble treatment per R3" to "remove cross-combination
    statistics", and §5.9 records the measured consequence: the two pinned
    anomaly PNGs are exactly `plot_proj_timeseries.py`'s multi-model percentile
    bands, so they change content at the same paths, while the three pinned
    `summary/*` targets are *unaffected* because they carry only per-series
    statistics (verified in `get_change_climate_proj_summary.py`). §9 gains the
    schema, catalog-parse, resolution-ladder, pairing, composition,
    duplicate-time, catalog-regeneration and no-aggregation tests.
  - **§6 gains three real alternatives** — member pairing (6.11), multi-version
    glob handling (6.12), and keeping a statistics layer in v2.0 (6.13) — and
    §6.6 records how catalog generation was actually resolved, withdrawing the
    `intake-esm` candidate from OQ-7. **OQ-12, OQ-13, OQ-14, OQ-15 opened**, each
    with the evidence that would settle it. No new dependency is adopted (N4).
