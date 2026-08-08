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
