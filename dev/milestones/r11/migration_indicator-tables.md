# Migration — WF3 indicator tables, wide → long (R11 P1)

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
