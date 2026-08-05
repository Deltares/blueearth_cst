# WF3 output change requests

Opened 2026-08-05, from observations on the first full WF3 run of the migrated
R9 tree. **Collect-then-implement**: the requests are specified here and built as
one batch, so the contract, the validators, the migration note and the baseline
re-record move once rather than per request.

Status: **CR-1 landed. CR-2 specified, NOT implemented — open questions below.**

---

## CR-1 — indicator-table axis columns (LANDED)

`tavg` → `temp_change`, `prcp` → `precip_change`. Full record:
`migration_indicator-axis-columns.md`. Commits `ba6f2b2`, `e159ca7`.

**Owes two gates**, both cleared by one WF3 run from the primary checkout:
`pytest tests/` (the fixture tree still holds pre-rename headers, so
`test_hm7_integration` and the 12 `test_gauge_identity_integration` cases fail
*correctly*) and `check_baseline.py check`. Tracked as followups R9-2.

**CR-2 supersedes this table's shape entirely.** CR-1 is not wasted work — it
fixed the vocabulary, and CR-2 carries the corrected names forward — but if CR-2
lands before the CR-1 re-run, do the WF3 run **once, after both**, and record a
single baseline move. Do not re-run twice.

---

## CR-2 — `q_indicators.csv` wide → long

### The shape

Seven fixed columns, independent of gauge count:

| Column | Domain |
| --- | --- |
| `variable` | `q` (only value today — see open Q3) |
| `metric` | the statistic (vocabulary under open Q4) |
| `temp_change` | perturbation axis, absolute degC |
| `precip_change` | perturbation axis, relative % |
| `realization_id` | **integer**; `1..RLZ_NUM`, or `0` = pooled over all realizations |
| `location` | gauge identifier (spelling under open Q2) |
| `value` | the metric value (precision under open Q1) |

Rows-not-columns for locations is the point: the header stops growing with gauge
count, which is what forces `validate_hm_gauge_column_identity`'s check 3 to
compare column *sets* today. Post-CR-2 that check compares the `location`
column's value set instead — same invariant, simpler expression.

### Decision — `realization_id = 0` means pooled

Ruled 2026-08-05. Keeps the column integer. Realizations are 1-indexed
(`rlz_1..rlz_N`), so `0` is free.

`cst_0` is NOT revisited. It is a genuine grid member (a real run, not a
sentinel) and is pinned in `naming.md` §4, in
`wildcard_constraints: st_num=[1-9][0-9]*` on `generate_climate_stress_test`, and
in `validate_wg5_catalog_grid`'s `m in 0..st_num`. Nothing about
`realization_id` justifies moving it.

**A numeric sentinel is safe here ONLY because of ruling (b1) below.** The hazard
of a numeric sentinel in a numeric key column is that `groupby("realization_id")`
folds pooled rows in as another realization. That requires both grains to
coexist *for one metric*, which under (b1) never happens — see the class table.
**If (b1) is ever reversed, the sentinel must become a string (`"all"`).** This
is a dependency, not a preference.

Since `0` cannot announce itself, `validate_hm7` asserts the invariant:
every class-B row has `realization_id == 0`; no class-A row does;
`realization_id` ∈ {0} ∪ `1..RLZ_NUM`.

### Decision — metrics split three ways

| Class | Metrics | Grain |
| --- | --- | --- |
| **A — linear in years** | `mean`, `max`, `min`, `q95`, `Q7day_max`, `Q7day_min`, `BaseFlowIndex` | Per realization (`1..N`). All are "annual statistic → mean over years"; realizations are equal-length, so per-realization values average back to the pooled value **exactly**. Nothing is lost. |
| **B — non-linear fit** | `returninterval`, `returninterval_min_7day` | Pooled (`0`) only. `frequency_analysis(freq="YS")` fits a GEV to annual blocks; pooling multiplies the block sample by `RLZ_NUM`. A per-realization fit over a short record is ill-conditioned — the owner's stated reason for this CR. |
| **C — selects a category** | `wetmonth_mean`, `drymonth_mean` | `groupby(index.month).sum().idxmax()` picks ONE month from the record, so different realizations can pick different months. Realization axis ruled: **pick once from the pooled record**. Stress-test axis still open (Q5). |

### Decision — retire `aggregate_rlz` (ruling b1)

Ruled 2026-08-05. In the long shape "aggregated" is no longer a *shape* choice —
which is the only reason the flag existed — so the table always carries the
finest grain (class A per realization, class B pooled) and downstream aggregates
as it likes.

Consequences:
- The key is read at `Snakefile_climate_experiment:811` via
  `get_config(my_cfg, "aggregate_rlz", True)` and appears in
  `snake_config.template.yml:194` and `snake_config_model_test.yml:82`.
- Workflow configs **silently ignore unread keys**, so a user's stale
  `aggregate_rlz` would quietly do nothing while they believed it was in effect.
  Removal policy under open Q7.
- One `aggr_rlz` parameter drives **both** tables, so `basin_indicators.csv`
  must change in the same commit — open Q6.

### Decision — CR-3 folded in: pool the sample, not the spliced series

Was a separate CR; merged 2026-08-05 after the worked example narrowed it.

`export_wflow_results.py:173-181` pools by `pd.concat`-ing realization series and
overwriting the index with a synthetic continuous `date_range` — butt-splicing
them into one fictitious record. A `rolling(7)` window then crosses each splice,
producing 7-day flows **that occurred in no realization**:

```
A's last 7 days:   20 20 20 20 | 2 2 2      -> A's true 7-day min = 12.3
B's first 7 days:   2  2  2  2 | 20 20 20   -> B's true 7-day min =  9.7
spliced window:              2 2 2 | 2 2 2 2 -> 2.0   <- never occurred
```

Each realization has a short 3-4 day low spell too brief to fill a 7-day window;
the splice manufactures a 7-day one. It becomes that year's annual minimum and
enters the GEV block sample for the low-flow RP.

**Scope is one metric, not the pooling method.** Which metrics touch `rolling(7)`
on the spliced series:

| Metric | Rolling? | Class | Fixed by |
| --- | --- | --- | --- |
| `returninterval` (high flow) | No — `frequency_analysis` on raw daily | B | Nothing needed; only block alignment matters |
| `returninterval_min_7day` | **Yes** | B | **This item** |
| `Q7day_max`, `Q7day_min`, `BaseFlowIndex` | Yes | A | **(b1) for free** — per-realization means no `concat`, no splice |
| `mean`, `max`, `min`, `q95` | No | A | n/a |

So the fix is: for `returninterval_min_7day`, extract each realization's annual
7-day minima **within** that realization, then pool the *blocks* and fit the GEV
on `RLZ_NUM × N` of them. The synthetic calendar disappears.

**Magnitude, stated honestly.** 6 contaminated windows per splice ×
`RLZ_NUM − 1` splices — 6 rows out of ~21,900 for 2 realizations × 30 years.
They matter only when a fabricated window becomes an annual extremum, which
needs the basin's low-flow season to coincide with the calendar year boundary.
Snowmelt and dry-winter basins: plausible. Wet-winter basins: inert. The
unconditional argument is different — both methods give `RLZ_NUM × N` blocks
**only if** every realization is a whole number of years on the same calendar
boundary, and nothing checks that. Sample-pooling does not depend on it.

---

## Open questions

| # | Question | Recommendation |
| --- | --- | --- |
| **Q1** | `value` precision. Requested: 2 digits. But precision is currently **per-statistic** — `.round(2)` for `mean/max/q95/returninterval/Q7day_max/wetmonth_mean`, `.round(4)` for `min/returninterval_min_7day/Q7day_min/drymonth_mean/BaseFlowIndex`. The 4-digit set is exactly the low-flow metrics plus BFI; at 2 digits a `Q7day_min` of `0.0034` becomes `0.00` and the low-flow half of the surface flattens to zero. | Keep `value` a real float and make precision a **write-time format**, not a data round — the long shape finally gives one homogeneous float column. Apply a per-metric format map: 2 dp everywhere it is meaningful, 4 dp for the low-flow set. |
| **Q2** | `location` spelling: `Q_130000086` or bare `130000086`? With `variable: q` present the `Q_` prefix is redundant, but it is also the literal HM-5 column name and the tie the gauge-identity validator checks. | Bare id; update the validator to strip the prefix explicitly rather than relying on the strings matching. |
| **Q3** | Is `variable: q` future-proofing for absorbing `basin_indicators`? A one-value column only earns its place as such. | Yes — see Q6. It becomes the discriminator. |
| **Q4** | Metric vocabulary. Current values are inconsistent (`mean`/`q95` lowercase, `Q7day_max` mixed, `BaseFlowIndex` PascalCase) and one is **misspelled and shipped**: `returninternval_min_7day` (`export_wflow_results.py:239`). | Harmonise now. `metric` becomes a *data value* users filter on — a typo in a header is ugly, a typo in a filter key is a support ticket. |
| **Q5** | Class C, stress-test axis. Ruling (a) fixed the month across realizations but not across stress tests. If each member picks its own wettest month, the surface compares different months — the same incomparability, one axis over. Live: the config supports monthly perturbation vectors. | Pick the month once from **`cst_0`, pooled over realizations**, and evaluate it everywhere. If the seasonal *shift* is interesting, that is a different indicator and the long shape carries it additively. |
| **Q6** | `basin_indicators.csv`, forced by (b1). | Same seven columns: `variable` ∈ {`actual_evapotranspiration`, `snow`, …}, `metric` ∈ {`annual_total`, `annual_max`}, `location` = `basin`, `realization_id` = `1..N` (all basavg metrics are class A). `_basavg` then disappears from the names because `location` says it. Two files, one shared schema — vs merging into a single `indicators.csv`, which is tidier but collapses two `rule all` targets. |
| **Q7** | Stale `aggregate_rlz` in an existing user config — silently ignored today. | Hard error naming the migration note, following the `variable_spec.parse` precedent (it refuses the pre-5e list shape and states the migration). |

---

## Cost, once the batch lands

- **§7 migration note** — required twice over: output-table column labels (the
  bullet added to `naming.md` §7 on 2026-08-05) and removal of a checked-in
  user-facing config key.
- **`validate_hm7`** — rewritten. Simpler: the header is now fixed and
  gauge-independent. Gains the class↔sentinel invariant.
- **`validate_hm_gauge_column_identity`** — check 3 moves from column-set
  equality to `location`-value-set equality.
- **Baseline** — both tables are byte-exact `sha256` entries in
  `dev/baseline/manifest.json`. One WF3 run from the primary checkout, then
  re-record. **Combine with CR-1's outstanding run** rather than running twice.
- **`pytest tests/`** from the primary checkout is the merge gate; it fails until
  that run lands.
