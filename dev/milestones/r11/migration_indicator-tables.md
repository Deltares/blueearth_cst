# Migration — WF3 indicator tables, wide → long (R11 P1)

> **The header moved again after this record was written.** On 2026-08-11 the
> seven columns were reordered identifier-first and `realization_id` was renamed
> `rlz_id`, giving
> `metric, location, st_id, rlz_id, temp_change, precip_change, value`.
> This document is the R11 era record and its header listings below are left as
> they were — they describe what R11 P1 produced, which is the shape a table from
> that era actually has. For the CURRENT header, read
> `dev/reference/contracts/hydrological-model-seam.md` §HM-7 or
> `blueearth_cst/shared/indicator_tables.py::INDICATOR_COLUMNS`. Everything else
> here — the wide→long reshape, the `aggregate_rlz` retirement, the grain
> classes, the no-migration support decision — is unaffected by that reorder.

Rename/reshape record per `naming.md` §7. Landed 2026-08-08 on
`feat/r11-p1-tables`. Specification: `dev/milestones/r09/wf3-change-requests.md`
CR-2 and its `### Decision —` blocks; scope and rulings:
`wf3-consolidation-scope.md`.

**Support decision: pre-existing experiments are not migrated.** An experiment
that has already run under the wide shape is re-run as a NEW experiment. This is
the same ruling R7 (GA-2) and R9 took, re-checked rather than inherited — see
the scope doc §9, and the note on `aggregate_rlz` below, which is what makes it
unavoidable rather than merely convenient.

---

## Files

| before | after |
| --- | --- |
| `<exp>/results/q_indicators.csv` | `<exp>/results/q_indicators.csv` *(name kept)* |
| `<exp>/results/basin_indicators.csv` | **gone** — replaced by one table per variable |
| — | `<exp>/results/<token>_indicators.csv`, one per `wflow_outvars` entry |

Tokens: `q`, `precip`, `aet`, `recharge`, `overland_flow`, `snow`. The set is
config-dependent, so `WF3_TARGETS`, rule 3.16's `output:` and
`check_baseline.py`'s targets all derive from
`workflows.model_creation.wflow_outvars` rather than listing paths.

## Columns

| before (`q_indicators.csv`) | after (every table) |
| --- | --- |
| `statistic, temp_change, precip_change, Q_<id>, Q_<id>, …` | `metric, temp_change, precip_change, realization_id, location, value` |

The header no longer grows with the gauge count — locations are rows. The two
perturbation-axis columns are unchanged.

## Metric names

`statistic` (a bare word) became `metric`, a composite `<token>_<statistic>`.

| before | after |
| --- | --- |
| `mean` | `q_annual_mean` |
| `max` | `q_mean_annual_max` |
| `min` | `q_mean_annual_min` |
| `q95` | `q_mean_annual_p95` |
| `Q7day_max` | `q_mean_annual_7day_max` |
| `Q7day_min` | `q_mean_annual_7day_min` |
| `wetmonth_mean` | `q_wettest_month_mean` |
| `drymonth_mean` | `q_driest_month_mean` |
| `BaseFlowIndex` | `q_baseflow_index` |
| `returninterval` | `q_return_level_<Tpeak>yr_max` |
| `returninternval_min_7day` *(shipped typo)* | `q_return_level_<Tlow>yr_7day_min` |

Two of these are not cosmetic:

- **`q95` → `_p95`.** Ours is the mean annual 95th percentile, a HIGH flow.
  Conventional *Q95* is the flow exceeded 95% of the time — a LOW-flow drought
  index. Propagating the established name would have been actively wrong.
- **The return levels absorb `Tpeak`/`Tlow`.** Those are config values that
  appeared in no column and no name, so two runs with different settings produced
  identical-looking rows meaning different things. Because the vocabulary is now
  partly config-derived, `validate_hm7` matches a **pattern**, not an enumeration.

Basin variables: `aet_annual_total`, `recharge_annual_total`,
`precip_annual_total`, `snow_annual_max`, `overland_flow_annual_mean`. Overland
flow reduces with a **mean**, not a sum — it is a volume flow rate, so summing
daily values yields a quantity in no useful unit (Q10).

## Values

- `value` is `float32` and **unrounded**. The old `.round(2)` / `.round(4)` was an
  accidental drift buffer; removing it is why the baseline comparator for these
  targets moves to a tolerance (Q8) — scheduled for P3, with the re-record.
- `location` is the **bare** gauge id (`130000086`, not `Q_130000086`), which is
  the subcatchment id wflow emits, so it joins `outlet_index.csv` with no
  crosswalk. `basin` is reserved for a basin-scalar value.
- `realization_id` is `0` for pooled, `1..RLZ_NUM` for one realization.

## Retired config key — `aggregate_rlz`

**Removed, and a config still carrying it is a HARD ERROR** (Q7, ruled
2026-08-07).

In the long shape "aggregated" is not a *shape* choice, which is the only reason
the flag existed: the table always carries the finest grain available and
downstream aggregates as it likes. Metrics linear in years are emitted per
realization; the two GEV fits and the two month-selecting metrics are pooled.

Refusing rather than ignoring, per the `variable_spec.parse` precedent: workflow
configs silently ignore unread keys, so a stale `aggregate_rlz` would leave a
user believing a setting is in effect while it does nothing at all. That is worse
than a refusal.

**To migrate:** delete the `aggregate_rlz:` line from
`workflows.climate_experiment`. Nothing replaces it.

**This is also why an existing experiment cannot continue.** `experiment.yml`
records the resolved `climate_experiment` section and is frozen at first
successful run, so removing the key changes the frozen document and
`check_not_frozen` refuses — correctly, because the table's grain really has
changed and the old results really do mean something different. Delete the
experiment and re-run, or start a new one.

## Consumers

- `validate_hm7` — rewritten for the long shape; asserts the header exactly, the
  metric-to-variable agreement, the vocabulary as a pattern, and the grain
  invariant.
- `validate_hm_gauge_column_identity` check 3 — compares the `location` value set
  instead of subtracting non-gauge columns out of the header.
- `dev/reference/contracts/hydrological-model-seam.md` — HM-7 rewritten, and it is
  where the token vocabulary is published.
- CST-API / GUI read these tables. Per `web-app-independence`, that consumption
  did not constrain the design; it is named here so the reshape is not a surprise.

## Not in this migration

`st_id` (C28) lands in **P2**, alongside the design table it must be checked
against. The header therefore gains a seventh column later; `validate_hm7` will
change with it.

---

# Migration — the WF3 member token, `cst_` → `st_` (R11 P2, commit 1)

Rename record per `naming.md` §7, appended here rather than given its own file
because P2's brief carries it in this record and the two migrations land in one
milestone and are re-recorded by one P3 baseline pass. Landed 2026-08-08 on
`refactor/r11-p2-rename`. Specification: `dev/milestones/r09/wf3-change-requests.md`
**C22**; boundary and rulings: `phase-2-run-identification-task-brief.md`.

**Why.** `cst` is the toolbox's own name, so it said nothing as a member token,
and every layer that mattered already said `st`: the `st_num` wildcard, `ST_NUM`,
`stress_test_grid()`, the `stress_test:` config section, the `st_csv_fns` rule
input. Only the filenames and the WG-5 catalog keys disagreed — one Snakefile
line built a `cst_` filename out of an `st_num` wildcard. This removes an
inconsistency; it invents no vocabulary. `rlz_` deliberately stays: it
abbreviates a *correct* term (CMIP's `r1i1p1f1`) and collides with nothing.

## Paths

| before | after |
| --- | --- |
| `<exp>/climate/weathergenr/_work/cst_<m>.csv` | `…/_work/st_<m>.csv` |
| `<exp>/climate/weathergenr/output/rlz_<n>_cst_0.nc` | `…/output/rlz_<n>_st_0.nc` |
| `<exp>/climate/weathergenr/output/rlz_<n>_cst_<m>.nc` | `…/output/rlz_<n>_st_<m>.nc` |
| `<exp>/hydrology/wflow/forcing/inmaps_rlz_<n>_cst_<m>.nc` | `…/forcing/inmaps_rlz_<n>_st_<m>.nc` |
| `<exp>/hydrology/wflow/config/rlz_<n>_cst_<m>.toml` | `…/config/rlz_<n>_st_<m>.toml` |
| `<exp>/hydrology/wflow/output/rlz_<n>_cst_<m>.csv` | `…/output/rlz_<n>_st_<m>.csv` |
| `<exp>/hydrology/wflow/output/rlz_<n>_cst_<m>.log` | `…/output/rlz_<n>_st_<m>.log` |
| `<exp>/hydrology/wflow/output/outstates_rlz_<n>_cst_<m>.nc` | `…/output/outstates_rlz_<n>_st_<m>.nc` |
| `<exp>/logs/_parts/3.1{2,4}_*/rlz_<n>_cst_<m>.log` | `…/rlz_<n>_st_<m>.log` |
| `<exp>/benchmarks/_parts/3.1{2,4}_*/rlz_<n>_cst_<m>.tsv` | `…/rlz_<n>_st_<m>.tsv` |

## Catalog keys

WG-5 (`<exp>/config/catalogs/data_catalog_climate_experiment.yml`): every entry
key `rlz_<n>_cst_<m>` → `rlz_<n>_st_<m>`, `m ∈ 0..ST_NUM`. The key is derived as
the realization NC's stem (`prepare_climate_data_catalog.py`), so it moved with
the file rather than being renamed separately; `validate_wg5_catalog_grid`'s
constructed expectation moved with it. This is the §7 clause on *hydromt
data-catalog source names* and is what makes this a recorded rename rather than
an internal tidy.

## Unchanged, deliberately

| surface | why |
| --- | --- |
| `blueearth_cst` (the package) | 885 occurrences; renaming breaks every import. Never a `cst_` rename target |
| `cst_calendar`, `cst_raw_digest`, `cst_source_paths`, `cst_series_digest`, `cst_schema_version`, `cst_acquisition_window`, `cst_time_*`, `cst_region_*`, `cst_crs`, `cst_members`, `cst_geometry_check`, `cst_weighting_scheme`, `cst_reducer_module_hash`, `cst_catalog_entry` | **WF2 netCDF provenance attributes** meaning "written by CST", in `blueearth_cst/projections/`. Renaming them corrupts WF2 output and breaks every cached store's identity check |
| `rlz_` | a correct abbreviation, colliding with nothing (C22) |
| `prepare_cst_parameters.py` / `prep_cst_parameters()` | module and function identifiers, not paths or keys — explicit P2 non-goal. Its *output* moved (`st_<m>.csv`); its own name did not |
| `cst_data` (R local), `_cst_df` / `_read_cst_csvs` (test helpers), `test_the_cst_index_…`, `_cst_pss_shim`, `cst_test_lib` | pure identifiers. None is a path or a key, so none is part of this migration; carried as a named residual for a later decision |
| `dev/scripts/semantic_tree_diff.py`, `tests/test_r09_path_map.py`, `tests/test_semantic_tree_diff.py` | they encode the P3-1 / R07 / R9 migration maps, whose eras used `cst_` on **both** sides. Renaming their expectations would make them assert a migration that never happened |
| `tests/test_wflow_log_attribution.py` | consumes `.cst_runs/r09_p2_post`, an R9-era tree outside the repo. Renaming its `MEMBER` regex and globs would match zero files and turn the test vacuous |
| every `dev/milestones/**`, `dev/reviews/**`, `docs/migration-r08-wf2.md`, `dev/reference/workflows/climate_experiment.md` | records of what was true when written (`AGENTS.md`, Conventions). `climate_experiment.md` is in `sealed-records.yml` |
| inline comments naming a PAST era — the R07 stem `cst_<m>`, the R07 batch tag, the retired C29 `weathergen_config_rlz_<n>_cst_<m>.yml` | they name files that existed under that token and, in C29's case, no longer exist at all. Renaming them invents a filename no tree ever held |

## Zero-padded member indices (R11 P2, commit 2)

C27 applied to the PATH as well as to the `st_id` column. Member indices are
padded to a width derived from their own count, so lexical order matches run
order in `ls`, a glob, an IDE tree and the WG-5 catalog's key order.

| count | width | members |
| --- | --- | --- |
| 1–9 | 1 | `st_1 … st_6` — **unpadded**, they already sort correctly |
| 10–99 | 2 | `st_01 … st_12`, baseline `st_00` |
| 100+ | 3 | `st_001 …` |

`rlz_` and `st_` pad INDEPENDENTLY, each from its own count: a 2-realization ×
100-member experiment is `rlz_1_st_001`, which is right rather than
inconsistent.

**On the tracked test config this changes nothing on disk.** `ST_NUM = 6`,
`RLZ_NUM = 2`, so both widths are 1 and every filename is byte-identical to
commit 1's. The fixture and the baseline therefore cannot exercise padding;
`tests/test_prepare_cst_parameters.py` and `tests/test_stress_test_grid.py`
carry that load with grids of ten or more, and a 12 × 12 dry-run was used to
prove the DAG (144 perturb jobs, `rlz_01_st_01 … rlz_12_st_12`, baseline
`rlz_<n>_st_00`, no `CyclicGraphException`).

**The wildcard constraint had to change, and it is load-bearing.** Rule 3.12's
`st_num=[1-9][0-9]*` forbids a leading zero — it would have rejected `st_01`
outright. It is now `member_index_regex(ST_WIDTH)`, which bars the all-zeros
baseline exactly as before AND rejects an *unpadded* `st_1` with
`MissingRuleException` rather than routing it silently. The laxer
`0*[1-9][0-9]*` was rejected for accepting both spellings: a producer that
forgot to pad would have agreed with the DAG invisibly, which is the failure
mode that cost this milestone two attempts.

**The regex is ANCHOR-FREE, and one wrong version shipped before the suite
caught it.** The natural spelling is `(?!0+$)[0-9]{W}`. It passes every
anchored unit check and is still wrong: Snakemake embeds a wildcard's
constraint in the regex for the WHOLE path, so `$` binds to the end of the path
rather than the end of the wildcard. With `.nc` always following, `0+$` can
never match, the lookahead always succeeds, and the constraint degenerates to
`[0-9]{W}` — admitting `st_00` and making rule 3.12 a second producer of the
baseline. A 12 × 12 `--dry-run` did NOT catch it, because where the baseline is
also reachable from its plural rule Snakemake prefers that one (fewer
wildcards) and the ambiguity stays hidden; it surfaced as a
`CyclicGraphException` in `test_cross_workflow_inputs` and
`test_guard_invalidation`, whose staged configs produce a DAG shape that forces
the choice. The shipped form spells not-all-zeros positionally instead —
`[1-9]` at width 1, `(?:[1-9][0-9]|0[1-9])` at width 2 — with no anchor and no
lookahead, and `test_the_member_regex_holds_when_EMBEDDED_in_a_path` pins it in
the position that actually broke.

**One new cross-language seam.** `generate_weather.R` composes its own output
filenames, so rule 3.11 now passes the two widths as CLI args 3 and 4 (arity
2 → 4). Re-deriving them in R would mean reimplementing `stress_test_grid`'s
arithmetic there, and a cross-language copy of a filename rule is invisible to
`--dry-run`.

## The stress-test design table (C23–C27)

`experiments/<id>/config/stress_test_design.csv` — one row per design point plus
a row for the `st_0` baseline with every change zero, beside the config snapshot
whose settings produced it (C25).

| column | meaning |
| --- | --- |
| `st_id` | the DESIGNED axis (C24), padded identically to the filename, so the two are the same token |
| `temp_change` | annual temperature change for that design point |
| `precip_change` | annual precipitation change factor |
| `precip_variance_change` | annual precipitation-variance factor |

`realization` is deliberately absent: it is the *sampled* axis, and a draw has no
design parameters to record. Run identity stays `(rlz, st)`.

**Written by rule 3.09, from the same loop that writes the member CSVs.** That is
C26's property — the enumeration which names the members and the enumeration
which describes them are one loop, so they cannot disagree about what run `m` is.
The values go through `annual_perturbation`, the SAME month-length-weighted
reduction the indicator tables use, because C28 (commit 3) will assert a results
row against the design table's row for its `st_id`; two independent collapses of
the same twelve monthly values would make that check fail on rounding.

**A third stress axis REFUSES.** `stress_test:` carrying anything beyond `temp`
and `precip` raises, naming C28 — a new dimension needs a design column and a
results column together, and one that merely went unrecorded would leave the
table describing a different experiment than the one that ran.

**Judgment call, easily reverted:** C23 says "one column per stress dimension",
which is two. `precip_variance_change` is a third column because the variance
genuinely varies per design point, and without it the table still does not fully
answer "what is run 37?" — the question C23 exists for. It is additive and does
not affect C28's check, which reads only the two axes.

**Not a baseline target.** It is declared in `WF3_TARGETS` so `rule all` demands
it (otherwise an experiment whose 3.09 is up to date would never regain a deleted
table — the F7 hazard), but it is deliberately NOT added to `check_baseline.py`:
that changes what P3 records, and the brief gates it. `build_project_tree_rules`
gained its inventory row, or `tree-check` would report it UNMAPPED;
`build_r09_path_map` stays frozen.

## `st_id` in the indicator tables (C28, R11 P2 commit 3)

The header is now **seven** columns:

    metric, st_id, temp_change, precip_change, realization_id, location, value

`st_id` is the design point's id, padded identically to the member filename, so
the results, the design table and the file on disk are one token. Ruled *"at
this stage"* — ALONGSIDE the perturbation columns rather than replacing them,
for plottable-without-a-join — which re-couples the header to the stress
dimension count. Two obligations hold that in place, and neither is optional:

1. **`validate_hm7(tables, rlz_num, design=…)` asserts the axis columns equal
   the design table's row for that `st_id`.** They are a cached copy; the writer
   derives them independently from the parameter files, so they really can drift.
   Skipped, never silently passed, when no design table is supplied.
2. **The writer REFUSES a design table carrying an axis the header cannot
   express**, naming C28. Rule 3.16 declares the design table as an input, so the
   refusal is reachable and the DAG edge is real.

**A units bug this forced into the open.** Commit 2's design table wrote the raw
precipitation FACTOR (`1.3`) while the results writer has always written a
PERCENT change (`30.0`). The two tables therefore disagreed *by construction*,
and C28's assertion would have failed on a unit rather than on a defect — the
exact way a consistency check rots into noise. Fixed by naming the derivation
once: `perturbation_axes()` in `export_wflow_results.py` is now the only place
either table computes an axis, and `precip_variance_change` follows the same
percent convention so one table does not mix two.

**`st_id` MUST be read as a string.** `pd.read_csv` with no `dtype` infers the
column as an integer, so `01` returns as `1` and the join to
`stress_test_design.csv` silently misses. Both tables carry the padded text on
disk — pinned against the file's BYTES, not a parsed frame, because a parsed
frame is exactly where the padding appears to vanish. Consumers (including
CST-API / the GUI) need `dtype={"st_id": str}`; recorded in the HM-7 seam
contract.

## Weathergenr config keys (C34, R11 P2 commit 4)

A checked-in example config key change, `naming.md` §7 tier 2. Full reasoning and
the one-decision-per-argument table: `c34-weathergenr-argument-decisions.md`.

| before | after |
| --- | --- |
| `generateWeatherSeries.evaluate.model: TRUE` | `generateWeatherSeries.save.plots: TRUE` |
| `generateWeatherSeries.evaluate.grid.num: 20` | **gone** |
| — | `generateWeatherSeries.pet.method: hargreaves` |

**Nothing is lost, because both removed keys were already dead.** weathergenr
1.2.0 split evaluation into its own exports, so `evaluate.model` and
`evaluate.grid.num` reached nothing: a project setting `evaluate.model: FALSE`
still got every plot. `save.plots` is the same stated intent wired to the
argument that actually controls it, and it defaults `TRUE`, so every tracked
config keeps today's behaviour.

**To migrate:** rename `evaluate.model` to `save.plots` in any project's
weathergen template copy and delete `evaluate.grid.num`. If neither is done the
generated config simply lacks `save.plots`, and `validate_wg3` reports it —
which is the point of pinning it.

**Also surfaced, without a config-key change:** the perturbation step now
receives the generator's own `seed` (it was unseeded while generation was
seeded — F15) and an explicit `pet_method` (PET is computed twice by two
methods, neither chosen — F16). **Both are UNEXECUTED here**: `weathergenr` is
absent from a `pixi install`-only worktree, there is no R test harness, and they
first run at rules 3.11/3.12. If the perturbation turns out to be stochastic,
seeding it moves numbers, and **P3's single re-record absorbs that** — flagged
so it is read as expected rather than as drift.

## Follow-on: rule 3.13's input keyword

`cst_nc` → `st_nc`, landed **after Gate 1** as its own commit, not part of the
atomic rename above. It is an identifier rather than a path or a key, so it sat
outside commit 1's scope — but unlike the other identifier residuals it is a
Snakefile↔script SEAM (`Snakefile_climate_experiment:770` declares it,
`prepare_climate_data_catalog.py:133` reads `sm.input.st_nc`), and its sibling
input on the next line has always been `rlz_nc`. Leaving it would have made
every future `cst_` grep report an oversight. Separated from commit 1 so the
atomic rename's blast radius stays exactly the paths and keys it claims.

## Support decision

Same as P1's, and for the same reason: **an experiment that has already run is
not migrated, it is re-run as a new experiment.** No rename shim, no dual-token
read path. Consistent with R7 GA-2 and R9, re-checked rather than inherited.

The `test_case/test_local` fixture is a pre-P2 tree and stays one until P3's
single WF3 re-run. The Layer-2 integration cases in
`tests/test_interchange_contracts.py` therefore assert the **post**-rename path
and skip on the specific pre-rename shape — `_member_artifact()` skips only when
the old-token twin exists exactly where the new artifact is missing, and
`test_wg5_catalog_grid_integration` skips only when every catalog key still
carries `_cst_`. Never a bare existence guard: that is how R9-4 turned a wrong
path into a silent pass. Precedent: P1's `test_hm7_integration`.

## Baseline

Every WF3 member artifact moved, so the baseline is **expected red** for those
targets. P3 owns the single re-record for the whole milestone; do not re-record
here. Targets that moved are exactly the ten path rows above.
