---
title: st_0 is not method-comparable with the stress-test surface
type: todo-item
status: backlog
effort: 2
area: wf3 / scenario generation + reporting
origin: R12
queue:
created: 2026-08-15
updated: 2026-08-15
---

> [!note] Overview
> **What** — st_0 is the raw generated series; every grid member is that series round-tripped through the perturbation in rule 3.12. Baseline and surface therefore differ by a PROCESSING STEP, not only by a perturbation, so the annotated baseline value reported beside a response surface is not comparable with it for most indicators.
> **Why** — Measured st_0 vs the grid's identity member: of eleven q metrics one is preserved (q_annual_mean +0.2%), five move by 20% or less, and five move by a FACTOR -- all five low-flow, worst q_mean_annual_min -69.7% and q_return_level_2yr_7day_min +127.9%. This is a live reporting property of the shipped pipeline, not a consequence of the lookup-table redesign.
> **Effort** — large

## The measurement

Found 2026-08-15 while verifying the alias precondition for `t2608152230`; that
item's ruling 6 was withdrawn on the same evidence. **No pipeline run was
needed** — `snake_config_baseline.yml` already contains an identity member, since
precip `step_num: 2` puts 1.0 on a level, and with the grid's
temp-outer/precip-inner order that is `st_2` (`stress_test_design.csv`:
`2,0.0,0.0,0.0`). Both it and `st_0` sit in `test_case/test_local`, with run
TOMLs identical apart from paths.

`results/q_indicators.csv`, `st_0` → `st_2`, mean over locations × realizations:

| metric | change | |
|---|---|---|
| `q_annual_mean` | +0.2% | preserved |
| `q_mean_annual_p95` | −1.2% | |
| `q_wettest_month_mean` | −3.7% | |
| `q_mean_annual_7day_max` | −10.0% | |
| `q_mean_annual_max` | −14.9% | |
| `q_return_level_10yr_max` | −18.4% | |
| `q_driest_month_mean` | −52.6% | |
| `q_baseflow_index` | −57.0% | |
| `q_mean_annual_7day_min` | −59.9% | |
| `q_mean_annual_min` | −69.7% | |
| `q_return_level_2yr_7day_min` | +127.9% (to +229.7%) | |

**Cause.** `weathergenr::apply_climate_perturbations` sends every grid cell
through `adjust_precipitation_qm(...)` **unconditionally** — no
`mean_factor == 1` short-circuit. That is empirical → fitted-Gamma quantile
mapping, so the daily series is replaced by its fitted-distribution image at any
factor. Probed on 1.2.0 directly: temperature *is* exactly the identity at
`temp_delta = 0`; precipitation is not. **All twelve monthly means are preserved
to +0.0000%** (`enforce_target_mean` is per-month), while the tail compresses —
single max day −32.9%, max 7-day sum −19.9%. The rainfall–runoff model then
amplifies that into the low-flow column above.

## Three things to carry

- **Why it has gone unnoticed.** `st_0` sits *inside* the member envelope for
  every metric — the grid spans ±30% precip and +3 °C, far wider than the
  artifact — so it never reads as an outlier. It is simply at the wrong place on
  the axis: `q_mean_annual_min` at location 101 is `st_0` = 0.0083 against a grid
  origin of ≈0.0025, i.e. annotated at **3.3×** the unperturbed grid point.
- **The option the owner declined for now**: route `st_0` through rule 3.12 with
  unit factors, so it becomes the true grid origin and is method-consistent with
  the surface. It fixes the root cause (and would make `t2608152230`'s withdrawn
  alias valid again), but it changes every baseline number, invalidates the two
  class-C metrics' current values, and forces a baseline re-record from
  `snake_config_baseline.yml` in the primary checkout with no other session live.
  The cheaper alternative is to leave the pipeline alone and caveat the
  annotation wherever it is reported.
- **The magnitudes need re-measuring before they are quoted.**
  `test_case/test_local` predates the 2026-08-12 weathergenr 1.2.0 rename (its
  `weathergen_config.yml` still carries the `generateWeatherSeries` schema), so
  the table above comes from the older `imposeClimateChanges`. The 1.2.0 probe
  agrees in direction and forcing-side magnitude, so the qualitative finding is
  version-independent; the numbers are not. See `t2608121258`.

## Refs

- `t2608152230` — the lookup-table redesign; its design note §5 carries the same
  evidence and the withdrawn ruling.
- `t2608121258` — propagate the post-R11 `test_local` fixture, which is what
  makes a version-current re-measurement cheap.
- `dev/reference/contracts/hydrological-model-seam.md` — HM-7.

## Progress

- [ ] Decide: fix the pipeline (unit-factor pass for `st_0`) or caveat the
      reporting. This is the gating call and it is R12's.
