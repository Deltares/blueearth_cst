# Post-R8 self-check — challenge register

Live register of the owner's own self-check after the **R8 WF2 v2.0** milestone
sealed. Opened 2026-07-31 against `da2e002` (`main`), during the first
post-refactor production run (`C:/TESTS/CST/gabonx`, Gabon test basin).

Unlike a defect log, this register is about **decisions**: places where an
earlier design or implementation choice — R8's or any earlier milestone's — now
looks questionable in hindsight. Each entry records the challenge, the reasoning
behind the original choice, the alternatives, and where it landed. Entries are
worked one at a time, in conversation, and written up here as they are resolved.

**This register is CLOSED.** The conventions below describe how it was run and
remain the template for the next one; the entries are a finished record, not a
live queue.

## Outcome — read this first

**Closed 2026-07-31.** All eight entries are `resolved` or `no-change`; nothing is
carried forward. This section is a derived overview, not a substitute for the
entries — every claim below is argued in the block it names.

The register began as one question about two same-sized files and ended as an
overhaul of everything WF2 hands a reader. The through-line: the output tree was
built for the workflow, not for whoever receives it. Directories named after
implementation stages, columns that were constant or empty or wrong, filenames
that contradicted their contents.

**The tree, before → after:**

```
climate_projections/cmip6/          climate_projections/cmip6/
├── raw/                            ├── raw/
├── series/                         ├── scalar/                    S8-03
├── timeseries/          ──┐        ├── plots/                     S8-07
├── change_factors/        │        │   └── 9 renamed figures
├── plots/                 │        ├── summary/
│   └── 9 contradictory    │        │   ├── cmip6_change_factors_annual.csv    S8-04
│       filenames          │        │   ├── cmip6_change_factors_monthly.csv
├── summary/               │ gone   │   ├── composition.csv        15 → 10 cols
│   ├── 3 wide files     ──┘ S8-02  │   └── provenance.json        S8-06
│   └── composition.csv      S8-05  └── report.md
├── provenance.json
└── report.md
```

| | Entry | Outcome |
|---|---|---|
| S8-01 | `raw/` ≈ `series/` in size | `no-change` — 2×2 cells on a coarse grid, and ~34 kB of identical HDF5 overhead. The split was never justified on payload; it is justified on **open cost** (1142 s vs 0.2 s) via the digest asymmetry. |
| S8-02 | Is `gcm_timeseries.nc` needed? | Deleted. Nothing read it, and it was a lossier, identity-stripped, ~48 % NaN copy of the tier it merged. |
| S8-03 | `series/` says nothing | → `scalar/`, the codebase's own word for the quantity (`var_m_scalar`). `raw/` kept by ruling. |
| S8-04 | Tables bloated, one column wrong | 20 → 14/15 columns, and a **correctness fix**: `units` labelled a percent as `mm/day` on every relative row. |
| S8-05 | Three wide files the tables supersede | Deleted; wide `.nc` becomes job-internal. Manifest **swapped**, not subtracted — 14 targets, better coverage. |
| S8-06 | `provenance.json` clutters the root | → `summary/`, beside `composition.csv`. |
| S8-07 | Figure names contradict contents | `{proj}_{var}_{view}_{quantity}.png`; precipitation forced to mm/day everywhere. |
| S8-08 | Four residuals | Units attribute fixed; stale comment rewritten; **the gridded branch and `save_gridded` removed**; rule detail and DAG diagram brought to the end state. |

**Five defects surfaced that nothing was looking for.** None was introduced by
this work; each was found by checking an artifact against what it claimed:

1. **Model names silently truncated** — `NOAA-GFDL/GFDL-ESM4` → `NOAA-GFDL/GFD`,
   because concatenating fixed-width numpy string coords truncates to the first
   file's width. It had been corrupting the wide summary's `dataset` column all
   along, and truncated rows also missed their per-combination window override.
2. **The annual level was in mm/year** while `units` said mm/day — a `rate`
   variable's annual aggregation is a duration-weighted *sum*.
3. **A baseline landing under the wrong scenario** — a companion emitted directly
   keeps the scalar `scenario='historical'` coord that arithmetic would have
   dropped; dropping it instead removed the *dimension*. Both stages produced
   plausible numbers.
4. **`report.md` advertised two figures that were never written** — the param
   feeding the report's listing is not an output declaration, so nothing failed.
5. **Two rules answering to banner `2.01`.**

**What caught them, and what did not.** Not the unit tests, and not
`check_baseline` — the wrong baseline and the truncated name are both *plausible*
values in the right shape. What caught them was checking each artifact against a
claim it makes: recompute the change from the two levels and compare it to the
change column (468/468); assert every figure `report.md` names exists on disk
(9/9); compare a column against the series it summarises. Worth reusing.

**Gates at close:** `pytest tests/` 819 passed · `check_baseline` 14/14 ·
`semantic_tree_diff --milestone r07` resolves every rename element-wise · wf2
runs end to end in 25 jobs.

**Residue cleared** from both project directories (20 superseded paths each). Note
for the next reference snapshot: pruning must happen *before* recording, or the
snapshot bakes in files the workflow no longer produces.

**One correction on the record:** the S8-02 entry originally claimed its
tree-diff row was "verified". That run used the tool's default `--milestone p31`,
under which the map is never built. Corrected in place.

---

**Scope and boundary.**

- **In scope:** any decision or implementation from Phase 2 (R1–R8) that the
  owner now wants to re-examine — not just R8's surface.
- **Out of scope / do not re-log:** items already in [`../followups.md`](../followups.md),
  the post-R6 observations in [`2026-07-25_post-r6-assessment.md`](2026-07-25_post-r6-assessment.md)
  (`O-nn`), and the post-R7 register [`2026-07-29_post-r7-self-check.md`](2026-07-29_post-r7-self-check.md)
  (`S7-nn`). Cross-reference those by ID; never duplicate them here.
- **Promotion:** an entry that needs work in a later milestone goes to
  `../followups.md`; one needing tracked multi-session work gets a `../TODO.md`
  row; one that changes a standing design position gets a `../decisions/` ADR.
  The **Disposition** field keeps the pointer, so nothing lives in two registers.

**ID prefix:** `S8-nn` (self-check, opened post-R8). Distinct from `S7-nn`,
`O-nn`, and `R7-nn`.

**Status vocabulary:** `open` (recorded, not yet discussed) · `discussed`
(reasoning and alternatives worked through, decision pending) · `resolved`
(decision made; disposition names the outcome) · `promoted` (routed to
followups/TODO/ADR) · `no-change` (challenge considered, original decision
stands — with the sharper reason recorded).

**How to add an entry.** Append the next `S8-nn` index row, then add the matching
detail block below with all five headings filled in. A block must read standalone
once the conversation that produced it is gone. (Detail blocks appear in the order
they were worked, not in ID order — S8-03…07 first, because they were decided as
one set. The index is the map.)

---

## Index

| ID | Challenge | Area | Status | Created | Updated | Disposition |
|---|---|---|---|---|---|---|
| S8-01 | `raw/` and `series/` files are nearly the same size — is the fetch/reduce split earning its second cache? | WF2 stage A | no-change | 2026-07-31 | 2026-07-31 | Sizes explained; split justified on open cost, not payload |
| S8-02 | Is `timeseries/gcm_timeseries.nc` needed at all, now that `series/` is already lat/lon-collapsed? | WF2 stage C | resolved | 2026-07-31 | 2026-07-31 | Deleted (option a); 2.06 is now figure-only |
| S8-03 | `series/` does not say the files are spatially averaged | WF2 layout | resolved | 2026-07-31 | 2026-07-31 | → `scalar/`; `raw/` unchanged; filenames identical across tiers |
| S8-04 | The change-factor tables are bloated and one column is wrong | WF2 stage B | resolved | 2026-07-31 | 2026-07-31 | Renamed + 13/14-column schema; `units` bug fixed |
| S8-05 | `summary/` carries three wide files the tidy tables supersede | WF2 stage B | resolved | 2026-07-31 | 2026-07-31 | Wide files deleted; tables move into `summary/`; manifest 15 → 14 |
| S8-06 | `provenance.json` clutters the run root | WF2 layout | resolved | 2026-07-31 | 2026-07-31 | → `summary/provenance.json` |
| S8-07 | Figure filenames are contradictory (`..._anomaly_..._abs.png` is the absolute plot) | WF2 stage C | resolved | 2026-07-31 | 2026-07-31 | `{proj}_{var}_{view}_{quantity}.png`; precip forced to mm/day |
| S8-08 | Four residual inconsistencies noticed in passing, none blocking | WF2 / docs | resolved | 2026-07-31 | 2026-07-31 | All four fixed; (c) removed the gridded branch and its config key |

---

## S8-03 … S8-07 — the output-surface overhaul

**Challenge.** Working through the first production run after R8, the WF2 output
tree read as workflow-internal rather than shareable: directory names that did not
say what they held, table columns that were constant, empty, or actively wrong,
and figure filenames that contradicted their own contents.

Five decisions, taken together on 2026-07-31 and landed as one sequence because
they touch overlapping surfaces. Recorded here **before** implementation so the
rationale survives the session that produced it.

### S8-03 — `series/` → `scalar/`

`raw/` **unchanged** (owner ruling); filenames stay identical across both tiers,
so the directory carries the tier and the filename carries the identity.

`scalar` is not a new coinage — it is this codebase's own word for the quantity:
`get_stats_climate_proj.py` writes `var_m_scalar = weighted_spatial_mean(...)`,
and the shipped `annual_change_scalar_stats_summary*` names use it in exactly this
sense, against the gridded `monthly_change_mean_grid-*`. So the scalar ↔ grid axis
already exists; `basin_series/` would have introduced a competing third axis, and
"scalar" stays true if the region is not strictly a basin.

Accepted cost: `raw/` and `scalar/` sit on different axes (processing stage vs
spatial shape). The internally consistent pairs would be `grid/`+`scalar/` or
`raw/`+`reduced/`. "Raw" reads as stage-zero and the owner chose to keep it; if it
ever grates, renaming `raw/` → `grid/` later is a strictly smaller change.
`grids/series/` keeps the old word deliberately — it is the *gridded* counterpart,
and `grids/scalar/` would be a contradiction.

### S8-04 — the change-factor tables

Renamed `change_factors/{annual,monthly}.csv` →
`summary/cmip6_change_factors_{annual,monthly}.csv`. The `{clim_project}` prefix
carries the archive identity that a constant `dataset` column used to repeat on
every row.

**A correctness fix, not only slimming.** `units` was populated from the variable
spec's declared units — the units of the *underlying variable*, not of the change.
Both change paths compute `(clim - hist)/hist*100` for a `relative` variable, so
every precipitation row labelled a **percent** as `mm/day`. Right for absolute
variables, wrong for relative ones: the same name-vs-semantics split 5e existed to
end, surviving one layer up in the metadata.

Three columns were structurally dead: `horizon_window_effective` (hardcoded `{}`),
`n_years_dropped` (hardcoded `""`), and `absolute_value` in the annual table (the
`__absolute` companion was only ever produced on the monthly path).

New schema — annual 14 columns, monthly 15 (from 20). `reference_value` was
added after the first cut, see "the 6b consequence, closed" below:

```
model,scenario,member,horizon,[month,]variable,statistic,
reference_value,absolute_value,units,relative_value,relative_units,
status,reference_window,horizon_window
```

- `reference_value` = the **baseline level** (25.0567 degC), in `units`.
- `absolute_value` = the **future level** (26.2354 degC), in `units`.
- `relative_value` = **relative to baseline**, per the spec's `change` field:
  a difference for `absolute` variables (+1.1787 degC), a percent for `relative`
  ones (+10.95 %). `relative_units` says which. No empty cells, nothing inferred
  from a variable name.
- This requires emitting the future level, which the change computation currently
  discards — a new companion on both paths.
- `reference_window` is the **effective** window, not the nominal one. An earlier
  draft proposed collapsing the two on the grounds that they were identical on
  every row; that was wrong. `tidy_rows`' own docstring records that the effective
  bounds and year count are properties of a *series*, not of the run — they are
  constant on this fixture, not structurally.
- `status` kept: it is the dry-month verdict and it varies per row. The two-value
  layout expresses 6b natively — a flagged month is `relative_value` empty,
  `absolute_value` present, `status = reference_below_threshold`.

Dropped: `dataset`, `institution` (→ `model`, unique in the CMIP6 controlled
vocabulary), `period` (constant in annual; becomes `month` in monthly),
`n_years`, `reference_series_key`, and the two dead columns.

Consequence accepted: the two tables stop sharing one schema, so they can no
longer be concatenated. That shared schema was `period`'s only purpose and no
consumer in the repo used it.

### S8-05 — `summary/` consolidation

`summary/annual_change_scalar_stats_summary.{nc,csv}` and `_mean.csv` are
**deleted**. Verified safe: `Snakefile_climate_experiment` and
`blueearth_cst/experiment/` reference them **zero** times, and rule 2.06 declares
the `.nc` as an input it never opens.

The wide `.nc` becomes a **job-internal intermediate** in the `TemporaryDirectory`
`derive_change_factors` already opens. That preserves the invariant the read-back
exists for — "the table must describe what was PERSISTED, so a reshape can never
disagree with the artifact it claims to reshape" — while removing the artifact.
`summary_climate_proj` stays, because it also renders the ΔT/ΔP figure.

**Manifest swap, not subtraction.** Removing three targets would drop coverage
from 15 to 12 and leave the change factors unfingerprinted. The two new CSVs are
added in their place: **14 targets, with strictly better coverage**. This
supersedes the earlier decision to widen the wide summary — there is no longer a
wide summary to widen.

`composition.csv` gets the same treatment, 15 → 10 columns:

```
model,scenario,member,status,reason,tier,
series_key,reference_series_key,catalog_entry,n_reference_years
```

`n_reference_years` is **kept** — it genuinely varies per series. Dropped:
`dataset`/`institution` (→ `model`), `catalog_crawled_on` (constant, already in
`provenance.json`), and both `reference_window_*` columns (now carried by the
change-factor tables, so one artifact owns the fact instead of three).

### S8-06 — `provenance.json` → `summary/`

Moved beside `composition.csv`; both are run-level records rather than results.
`report.md` stays at the run root as the single human entry point, and its
cross-reference at `report.py:164` updates.

Not buried further: `provenance.json` is a genuine in-workflow input — `report.py`
generates the report *from* it — and it is the audit trail (per-source digests,
true calendars, resolved store paths, catalog snapshot date). Dense, but the file
you hand a reviewer who asks which model version and which window backed a number.

### S8-07 — figure filenames

The old names contradict their contents: `precipitation_anomaly_projections_abs.png`
carries `plt.title("Annual precipitation")` and plots absolute levels — "anomaly"
sits in the filename of the non-anomaly figure.

New scheme, `{clim_project}_{variable}_{view}_{quantity}.png`, aligning three
axes with decisions already taken: `precip`/`temp` (matching the config keys and
the tables' `variable` column, where the plots were the only artifact using the
long forms), `annual`/`monthly`, and `absolute`/`change` (the same distinction as
`absolute_value`/`relative_value`).

`projected_climate_statistics.png` → `cmip6_change_factor_cloud.png`, picking up
the design's own phrase for the ΔT/ΔP cloud. The four `save_gridded` figures
follow the same scheme plus `grid_{scenario}_{horizon}`, dropping the
`-future-horizon` suffix (every horizon is a future horizon).

Sweeps up a live defect: `plot_proj_timeseries.py:284` calls `savefig` with **no
extension**, relying on matplotlib to append `.png`, while its temperature
counterpart passes it explicitly — the mislabel still recorded in the overview's
observation 10. Every path becomes explicit.

**Owner ruling on units:** the annual precipitation figure multiplied by 365 to
plot mm/year while every other artifact reports mm/day, so a reader comparing the
figure against the table saw ~2210 versus ~6.05 for one quantity. Forced to
**mm/day everywhere**.

**Disposition.** All five `resolved` and implemented in this session — S8-03 as
its own commit, S8-04/05/06/07 together (they touch overlapping surfaces and
splitting them would have produced commits that do not run).

### Two bugs the overhaul uncovered

Neither was introduced here; both were exposed by looking at the columns.

**String truncation in the wide merge.** `model` read `GFD` on the first run of
the new schema. The per-point netCDFs carry string coords as numpy fixed-width
dtypes whose width is set by the longest value *in that file*, and concatenating
files of different widths truncates every value to the FIRST file's width:
`NOAA-GFDL/GFDL-ESM4` (19) became `NOAA-GFDL/GFD` (13) whenever an
`INM/INM-CM4-8` file merged first. Silent, because a truncated model name is
still a plausible string. It had been corrupting the wide summary's `dataset`
column all along, and — because the per-combination window lookup is keyed on
that name — truncated rows also missed their effective-window override and
silently fell back to the run-level one. Fixed in `preprocess_coords` by casting
string coords to object dtype.

**The annual level was in mm/year.** For a `canonical: rate` variable the annual
aggregation is a duration-weighted **sum**, so the future level came out as an
annual total while `units` said `mm/day`. The level is now drawn from a
duration-weighted **mean** — the same integral divided by the year's length, so
the ratio that feeds `relative_value` is untouched and only the level's scale
changes. This is also what makes the owner's "mm/day everywhere" ruling true of
the tables, not just the figures.

### Verification

- `pytest tests/` (excluding the slow `test_model_creation.py`): **815 passed**,
  6 skipped, 1 xfailed. `test_change_factor_table.py` and
  `test_derive_change_factors.py` were rewritten for the new schema (26 tests);
  `test_get_change_climate_proj.py` and `test_report.py` had assertions updated.
- wf2 executed for real on the fixture after every change.
- `check_baseline`: **14 targets, OK**. The swap landed as planned — three wide
  summary files and three renamed figures out, two tidy tables and three renamed
  figures in. Coverage improves: the change factors are fingerprinted for the
  first time.
- `semantic_tree_diff --milestone r07` against `ref_wf2_pre_5f_fixed`, on a
  cleaned copy of the fixture: **all nine figure renames and `series/` → `scalar/`
  resolve element-wise**, leaving `4 missing` (exactly the three wide summary
  files plus `gcm_timeseries.nc` — the deletions) and `4 extra` (`report.md`,
  the two tables, `summary/provenance.json` — 6a/7 artifacts that postdate this
  reference). Without the added rows the same diff reported ~14 deletions plus
  ~14 additions and stopped discriminating where the most had changed.
- Every figure `report.md` advertises exists on disk (9/9), asserted directly.
  This caught a real miss: the Snakefile's `figures` param still named two
  pre-S8-07 figures, so `report.md` was advertising files that were never
  written. Nothing else would have caught it — that param feeds only the report's
  listing, not an output declaration, so the run succeeded; `check_baseline`
  fingerprints three PNGs, none of them the two; and `test_report.py` passes
  synthetic names.

### Residue

Every superseded path is stranded in existing project directories, because
Snakemake cannot clean an output it no longer declares: `series/`, `timeseries/`,
`change_factors/`, the root `provenance.json`, the three wide summary files, the
nine old figures, and four logs from rules retired before v2.0. Enumerated in
`docs/migration-r08-wf2.md` under "Post-migration cleanup", and left as an owner
action — deleting fixture state is not something a task should do implicitly, and
it must happen before the next reference snapshot.

### `reference_value` — the 6b consequence, closed

The first cut of the schema weakened the 6b dry-month rule. The rule drops the
meaningless ratio and **keeps the informative difference**; under a two-value
layout a flagged month had `relative_value` empty and only the future level
present, so the difference was no longer representable.

**Owner ruled: add `reference_value`** (the baseline level, in the same `units`).
Annual 14 columns, monthly 15. The difference is now
`absolute_value - reference_value` on every row, flagged or not, and every number
in a row is recoverable from that row alone.

Emitting it exposed a two-stage trap worth recording, because both stages produce
plausible numbers:

1. The historical source carries a scalar `scenario='historical'` coordinate from
   its `.sel`. `change` never shows it — DataArray arithmetic drops conflicting
   scalar coords — but a companion emitted **directly** keeps it and merges onto
   the `historical` label instead of the scenario it is the reference for.
2. Simply dropping that coordinate removes the scenario **dimension**, leaving the
   companion one rank short of its siblings, which the multi-file merge then
   mangles differently.

`broadcast_like(level_stat)` fixes both: identical dims, identical labels.

The check that caught it, and that neither the unit tests nor `check_baseline`
would have: **recompute the change from the two levels and compare it against the
change column, on every row.** The first version put `6.489` where `9.331`
belonged for GFDL-ESM4 precipitation — a wrong baseline that still looked like a
precipitation rate. 468/468 rows now agree.

---

## S8-01 — `raw/` and `series/` are nearly the same size

**Challenge.** `series/{key}.nc` collapses lat/lon to a basin scalar, yet on the
Gabon run it is 50.8 kB against `raw/`'s 74.8 kB. If the reduction removes almost
nothing, does the two-tier fetch/reduce split earn its keep?

**Finding.** Measured on `cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc`:

| | `raw/` | `series/` |
|---|---|---|
| dims | `time: 1032, lat: 2, lon: 2` | `…identity dims…, time: 1032` |
| payload (`ds.nbytes`) | 41.3 kB | 16.7 kB |
| on disk | 74.8 kB | 50.8 kB |
| structural overhead (disk − payload) | 33.5 kB | 34.1 kB |

Two effects stack. The Gabon bbox is **2×2 CMIP6 cells** at ~2°×1.5° with the
1.0° buffer, so the spatial collapse can only remove a factor of 4. And the fixed
netCDF4/HDF5 overhead (~34 kB, near-identical in both) swamps what is left; the
`time` coordinate alone (1032 × int64 = 8.3 kB) is carried unchanged in both and
is already half the series payload.

**Resolution.** No change. The split was never justified on payload size — it was
justified on **open cost**: per `dev/milestones/r08/2026-07-30_wf2-fetch-reduce-benchmark.md`,
opening one remote source costs 1142 s, transfer 19 s, and the reduction
arithmetic 0.2 s. The load-bearing mechanism is the digest asymmetry: `raw/`
carries `cst_raw_digest` and deliberately omits `cst_series_digest` /
`cst_reducer_module_hash`, so a reducer-formula edit invalidates only the series
and the fetch cache-hits. Comparable file sizes are the expected consequence of a
small basin on a coarse grid, not evidence against the split.

**Disposition.** `no-change`. Reasoning recorded here; nothing to promote.

---

## S8-02 — Does `timeseries/gcm_timeseries.nc` need to exist?

**Challenge.** `series/` already collapses lat/lon and carries identity
coordinates. Is there any advantage to also merging the nine series into one
`timeseries/gcm_timeseries.nc`?

**Finding.** As it stands the file is a lossier, untraceable duplicate of the
series tier, produced as a side effect of the plotting rule:

1. **No consumer.** Not in `rule all`'s inputs, not read by
   `Snakefile_climate_experiment`, not read by any script. The only other
   reference is `dev/scripts/semantic_tree_diff.py:278-279`, which maps its old
   path to its new one for baseline diffing — a tool that compares it, not a
   consumer that needs it. `dev/workflows/climate_projections.md:105` already
   classifies it as a "side-effect artifact (bookkeeping; no downstream reader)".
2. **Not a manifest target.** The design states this explicitly under "Coverage
   is thin" (`wf2-climate-analysis-v2-design.md` §2156).
3. **Lossier than its inputs.** `plot_proj_timeseries.py:255-256` rounds precip
   and temp to 2 decimals — the exact `.round(decimals=2)` that step 5c
   deliberately removed from stage A because it imposed a 0.005 mm/day floor on
   every value. The merged file re-imposes the quantisation the series just shed.
4. **No identity.** Lines 252-253 strip every `cst_*` attribute — correctly, since
   `xr.merge` would otherwise propagate one arbitrary series' digest to a
   nine-series file. The result (`attrs: {}`) has no digest, region fingerprint,
   source pins, or calendar, so it cannot be validated or traced.
5. **Hardcoded to precip/temp.** Lines 255-256 name the two variables literally,
   so the file does not generalise under the 5e `variables:` mapping.
6. **No compaction.** Dims `(model: 3, scenario: 3, member: 1, time: 1812)` span
   1950-01…2100-12 as a dense cube, but `historical` ends 2014 and the SSPs start
   2015, so ~48 % of it is NaN. 145 kB in memory against ~150 kB for the nine
   series it replaces.

**Alternatives.**

- **(a) Delete it.** The nine `series/*.nc` are the durable timeseries tier, and
  v2.0's new `change_factors/annual.csv` + `monthly.csv` are the analysis-ready
  long format. Retires a rounding path that contradicts 5c.
- **(b) Promote it to a real artifact.** Move it out of the plotting rule into its
  own rule, drop the rounding, replace stripped attrs with a per-source
  provenance block, and drive variable names off `VARIABLE_SPEC`.
- **(c) Keep as-is,** documented as an unvalidated convenience export.

**Disposition.** `resolved` — **owner ruled (a), delete**, 2026-07-31.

Landed as one commit. Surfaces changed:

- `Snakefile_climate_projections` — the `timeseries_nc` output dropped from rule
  2.06, which becomes a figure-only rule (8 PNGs, no netCDF).
- `blueearth_cst/projections/plot_proj_timeseries.py` — the merge/strip/round/
  write block removed, along with the `ds_fut` accumulator that existed only to
  feed it. `.load()` on both `open_mfdataset` calls is **retained**: it was added
  for the deadlock this write caused, but every dataset is converted straight to
  pandas anyway, so removing it would be an unrelated behaviour change. The stale
  justification comments were rewritten rather than left pointing at a write that
  no longer exists.
- `dev/workflows/wf2_climate_projections_overview.md` — §2 output list and
  observations 9–10.
- `docs/migration-r08-wf2.md` — a "Removed output" section naming the
  replacements (`series/*.nc`, `change_factors/*.csv`).

**Deliberately not changed:** `dev/scripts/semantic_tree_diff.py:278-279` keeps
its `gcm_timeseries.nc` rename row. That row asserts a *historical* path
equivalence for the R7 layout move; it is not a claim that the file is still
produced. Run against `ref_wf2_pre_5f_fixed` with a copy of the fixture whose
`timeseries/` was removed, the tool reports the file as one clean deletion:

```
> MISSING (in ref, not cur): climate_projections/cmip6/timeseries/gcm_timeseries.nc
```

The control run against the untouched fixture reports it missing zero times, with
otherwise identical counts — so this change contributes one deletion and nothing
else.

> **Correction (same day).** That first run used the tool's **default
> `--milestone p31`**, under which `build_r07_path_map` is never built — so the
> rename row was not actually exercised, and the entry originally claimed
> "verified, not assumed" on the strength of it. The rows live inside
> `build_r07_path_map` and require `--milestone r07`. Re-run with the flag during
> S8-04..07 (below), the whole map resolves as intended. The observed behaviour
> above was correct either way; the claim about *why* was not.

**Verification.** wf2 dry-run clean; rule 2.06 executed for real on both the
Gabon project and the tracked fixture; `pytest tests/test_cli.py` 9 passed;
`check_baseline check` **OK — 15/15 targets match**, which is the load-bearing
result: the two anomaly PNGs are manifest targets produced by this rule, so a
green gate proves the deletion is output-neutral for everything the baseline
constrains. `gcm_timeseries.nc` was never a manifest target, so nothing needed
re-recording.

**Known residue — action required before the next snapshot.** A project directory
from an earlier run keeps its stale `timeseries/gcm_timeseries.nc`; Snakemake
cannot clean an output that is no longer declared. Present in both
`test_case/test_local/` and `C:/TESTS/CST/gabonx/` at time of writing, and
documented in the migration note as a one-time manual delete for users.

The fixture copy is the one that matters. `check_baseline` does not fingerprint
it (never a manifest target), so nothing fails today — but AGENTS.md requires
pruning **before** any reference snapshot, or the snapshot bakes the orphan in
and every later whole-tree diff compares against a file the workflow no longer
produces. **Delete `test_case/test_local/climate_projections/cmip6/timeseries/`
before recording the next reference tree.** Left to the owner: deleting fixture
state is an explicit owner action, same principle as `prune_series_cache.py`.

---

## S8-08 — four residual inconsistencies

Noticed while working S8-01…S8-07. None blocks anything; each is logged here
rather than fixed in passing, so the commits that fixed the surfaces around them
stay attributable.

### a. The `units` netCDF attribute contradicts its own values

`raw/*.nc` and `scalar/*.nc` both carry `units = "kg m-2 s-1"` on `precip` and
`"K"` on `temp`, while the values are plainly mm/day and °C. The conversion is
done by the catalog's `data_adapter` (`unit_mult: precip: 86400`,
`unit_add: temp: -273.15`), which does not rewrite the attribute.

The units that reach `provenance.json` and the change-factor tables come from the
`variables:` spec and are correct — only the netCDF attribute disagrees. Not
verified whether anything downstream reads it. Fixing it means stamping the
declared units onto the arrays after the adapter runs; per the repo's hard
constraint, the fix belongs in our code, never in a vendored hydromt.

### b. A stale comment tells readers to use the config form that now raises

`get_stats_climate_proj.py:105-119` explains the monthly resample dispatch and
ends: *"The required config is `variables: [precip, temp]`"* — the bare list that
`variable_spec.parse` has raised on since 5e.

The code it documents is also the last place in stage A that reads a **name**:
the dispatch branches on the literal string `"precip"` to choose sum vs mean.
Inert today and documented as such (`variable_spec`'s docstring: one source
frequency, `Amon`, so the conversion is the identity — with one step per `MS`
group, sum and mean return the same element). It would bite the day a sub-monthly
source is added. The comment is the part that is actively wrong now.

### c. `grids/series/` will duplicate `raw/` for `Amon` sources

`grids/` is declared but never produced — the Snakefile has no `grids/` path at
all; only `get_stats_climate_proj.py` builds one, under `save_gridded`. If it is
ever switched on, `grids/series/{key}.nc` will be near-identical to
`raw/{key}.nc` for `Amon` input: same grid, same monthly steps, differing only in
attributes, because the monthly resample between them is the identity.

Worth deciding **before** enabling the gridded branch, not after — the same
question S8-02 answered for `gcm_timeseries.nc`.

### d. The overview's DAG diagram still describes the pre-v2.0 rule set

`dev/workflows/wf2_climate_projections_overview.md` opens by claiming it records
**current** behavior, and its §1 rule inventory was corrected at the v2.0 seal.
But the §2 detail blocks and the DAG diagram still describe `monthly_stats_hist`,
`monthly_stats_fut`, `monthly_change`, `monthly_change_scalar_merge` and the
`ruleorder:` directive — all retired, and all listed as retired in that same
file's own "what went, and where it went" table. The `all` target table and rules
2.04/2.06 were brought to the end state at S8-04…07; the rest was left, because
rewriting a whole document inside a rename commit destroys attributability.

A doc that contradicts itself one section apart is worse than a stale one: the
inventory says these rules are gone and the detail section explains how they work.

**Disposition.** `resolved` — all four fixed, 2026-07-31.

**(a) Units stamped where the values are converted.** The declared units from the
spec are written onto the arrays in both writers, keyed by post-rename source
name. A slice cached *before* the fix is repaired in place on its next cache hit
rather than left alone: `scalar/` is stamped on every reduce, so skipping the
repair would have made `raw/` the only artifact still misdescribing its own
values, and only on projects old enough to have a cache. Safe against identity —
`units` is a variable attribute and the digest covers neither it nor the values.

**(b) Comment rewritten** to say what is true: the dispatch is the last
name-reading site in stage A, it is inert because `Amon` is already monthly (one
element per `MS` group, so sum and mean agree), and it must move to
`canonical_kind` the day a sub-monthly source is added.

**(c) The gridded branch and its config key are gone.** Owner ruling: `raw/` is
already the basin slice on the source grid, so `grids/series/` would have been a
near-copy of a file every run writes. Removed: the branch in the reducer, stage B
and the plot script; `get_change_clim_projections` (the cellwise change — also the
last unconverted 5e site outside stage A's resample); the cartopy imports; the
`change_grids` and `stats_path*` params; and `save_gridded` from every shipped
config. Both spellings now **raise on `true`** and **warn on `false`** — a `true`
asks for something that no longer exists, a `false` asks for exactly what the
workflow always does, so breaking every config over it would be ceremony.

**(d) Rule detail and DAG diagram rewritten** to the end state. §4 gains a banner
marking it as pre-rework observations rather than current behaviour — it describes
rules §1's own table lists as retired, which is worse than merely stale.

**A collision found while fixing (d):** `copy_config` and `fetch_gcm_raw` both
answered to banner `2.01`. The numbers exist to disambiguate, so two rules sharing
one is self-defeating; the rule map already listed `copy_config` as 2.03, which it
is now. No path moved — that rule names its output explicitly and has no
banner-derived log or benchmark, so the collision only ever showed in the console.

**Verification.** `pytest tests/` 819 passed; wf2 ran end to end (25 jobs, the
count §1 now states); `check_baseline` FAILED on exactly one target — the config
snapshot, a verbatim copy of the seed config that lost the `save_gridded` line —
and every computed target matched, which is the signal that the removal is
output-neutral. Re-recorded, 14/14.
