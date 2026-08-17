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

## Independent corroboration: the v2 design already encoded it

Found 2026-08-15 while drawing the R12 boundary. `design-v4.md` §6.6 (on
`docs/wf3-redesign`, cited by path) gives `baseline: true` members their **own
branch of the member lifecycle**: `st_csv: null`, `precip_variance: null`, *no
spec write, no perturb step*, downscale directly from the persistent
`realizations[].baseline_nc`.

So the separate production path this item reports is not an accident of the
current rules — an independent design run formalized it as a first-class
lifecycle branch. What that run did **not** do is recognise it as a comparability
problem: a member that skips the perturbation is not on the same footing as one
that does not, and the design treats the branch purely as an execution shortcut.
That makes this item a live input to `t2608082036` rather than a bystander.

## Three things to carry

- **Why it has gone unnoticed.** `st_0` sits *inside* the member envelope for
  every metric — the grid spans ±30% precip and +3 °C, far wider than the
  artifact — so it never reads as an outlier. It is simply at the wrong place on
  the axis: `q_mean_annual_min` at location 101 is `st_0` = 0.0083 against a grid
  origin of ≈0.0025, i.e. annotated at **3.3×** the unperturbed grid point.
- **The option the owner declined on 2026-08-15 and ADOPTED on 2026-08-17**:
  route `st_0` through rule 3.12 with unit factors, so it becomes the true grid
  origin and is method-consistent with the surface. It fixes the root cause (and
  makes `t2608152230`'s withdrawn alias valid again), but it changes every
  baseline number, invalidates the two class-C metrics' current values, and
  forces a baseline re-record from `snake_config_baseline.yml` in the primary
  checkout with no other session live. The cheaper alternative — leave the
  pipeline alone and caveat the annotation wherever it is reported — is
  **rejected**; see the ruling below.
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

## RULED 2026-08-17 — fix the pipeline

> **Ruling (owner).** Route `st_0` through rule 3.12 with unit factors. Caveating
> the reporting was rejected: it makes every reader responsible for a correction
> the pipeline can make, and the caveat would have to be maintained in every
> figure, table and report that shows a baseline.

**What changed the cost calculus.** On 2026-08-15 this option was declined
because it forces a baseline re-record. That cost is now largely already owed
and should be paid ONCE, in one sitting in the primary checkout:

- `t2608121258` — neither fixture tree matches the manifest's WF3 `n_rows` (756
  expected, 630 from both), so a deliberate WF1+WF3 re-run on
  `snake_config_baseline.yml` with `--notemp` is owed regardless of this item.
- *The magnitudes need re-measuring* above — `test_case/test_local` predates the
  2026-08-12 weathergenr 1.2.0 rename, so the table in this note has to be
  re-measured on 1.2.0 before any number in it is quoted. Same run.
- `t2608090907a` — its only remaining step is `check_baseline.py check` after
  the next full re-run. Same run.

So the marginal cost of this ruling is the rule-3.12 change plus the judgment on
the resulting diff, not a re-record scheduled for its own sake. **Sequence the
`st_0` change BEFORE the re-run**, or the re-record bakes in the old baseline and
a second one is needed.

**The 756 → 630 question stays separate.** It predates this finding and neither
tree produces 756, so it is not evidence about the unit-factor change; rule on it
on its own terms (`t2608121258`) rather than absorbing it into this diff.

## Progress

- [x] Decide: fix the pipeline (unit-factor pass for `st_0`) or caveat the
      reporting — **fix, ruled 2026-08-17.**
- [ ] Implement the unit-factor pass for `st_0` in rule 3.12, so the baseline is
      produced by the same path as every grid member. Check first whether the
      right shape is a unit-factor member or a `mean_factor == 1` short-circuit
      in `weathergenr::apply_climate_perturbations` — the latter is upstream and
      out of scope per `AGENTS.md`, so the pass belongs on our side.
- [ ] Re-measure the `st_0` → identity-member table on weathergenr 1.2.0; the
      current numbers come from the older `imposeClimateChanges`.
- [ ] Re-record the baseline in the same primary-checkout sitting as
      `t2608121258` and `t2608090907a`, and state which metrics moved and why.
- [ ] Re-check `t2608152230`'s withdrawn alias, which this ruling makes valid
      again, and `t2608082036` §6.6, whose `baseline: true` lifecycle branch this
      contradicts — that design treats the separate path as an execution
      shortcut, and it is now a correctness question.
