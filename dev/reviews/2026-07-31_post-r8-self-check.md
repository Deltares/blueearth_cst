# Post-R8 self-check — challenge register

Live register of the owner's own self-check after the **R8 WF2 v2.0** milestone
sealed. Opened 2026-07-31 against `da2e002` (`main`), during the first
post-refactor production run (`C:/TESTS/CST/gabonx`, Gabon test basin).

Unlike a defect log, this register is about **decisions**: places where an
earlier design or implementation choice — R8's or any earlier milestone's — now
looks questionable in hindsight. Each entry records the challenge, the reasoning
behind the original choice, the alternatives, and where it landed. Entries are
worked one at a time, in conversation, and written up here as they are resolved.

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
once the conversation that produced it is gone.

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

New schema — annual 13 columns, monthly 14 (from 20):

```
model,scenario,member,horizon,[month,]variable,statistic,
absolute_value,units,relative_value,relative_units,
status,reference_window,horizon_window
```

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

**Disposition.** All five `resolved`; implementation follows in this session.

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
justified on **open cost**: per `dev/working/2026-07-30_wf2-fetch-reduce-benchmark.md`,
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
produced. **Verified, not assumed** — run against `ref_wf2_pre_5f_fixed` with a
copy of the fixture whose `timeseries/` was removed, the tool reports exactly:

```
> MISSING (in ref, not cur): climate_projections/cmip6/timeseries/gcm_timeseries.nc
> MISMATCH: 125 files compared, 15 failed, 1 missing, 4 extra, 0 allowlisted
```

The control run against the untouched fixture gives `126 files compared, 15
failed, 0 missing, 4 extra` — identical but for the one file. So the row resolves
the old path correctly and this change contributes one clean deletion and nothing
else. (The 15 failed / 4 extra are pre-existing deltas against a reference that
predates 5f's value change and 6a's new outputs; they are present with and
without this commit.)

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
