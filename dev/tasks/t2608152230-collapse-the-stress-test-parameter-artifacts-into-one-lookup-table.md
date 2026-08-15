---
title: Collapse the stress-test parameter artifacts into one lookup table, and make the surface axis a parameter
type: todo-item
status: backlog
effort: 2
area: wf3 / stress-test design + reduction
origin: R12
queue: 1
created: 2026-08-15
updated: 2026-08-15
---

> [!note] Overview
> **What** — Merge `_work/st_*.csv` and `stress_test_design.csv` into one long
> `stress_test_lookup.csv` (`st_id × month × parameters`, percent), make it the
> source of truth, and move the response-surface axis from a baked reduction-time
> collapse to a declared post-processing parameter.
> **Why** — For a seasonal design the fixed annual collapse **misreports what was
> explored**: +30% in JJA reads as +7.6% on the axis. That is a correctness
> problem, not a missing feature.
> **Effort** — Large, and now **specified**. All six gaps that made this
> unimplementable are closed by an accepted design:
> `dev/milestones/r12/stress-test-lookup-design.md` (**ACCEPTED 2026-08-15**,
> D1–D35, both drop-in contract replacements, migration plan, validation plan
> V1–V23). This is implementation against a spec, not design work.

## The accepted design supersedes everything below

**Read `dev/milestones/r12/stress-test-lookup-design.md` first.** It is the
specification; the sections below are the design *conversation* that preceded it
and are kept for provenance, not for instruction. Where they differ, the accepted
design wins — and they do differ: ruling 6 was withdrawn on measurement, the
schema moved to WG-2, and D35 added a config constraint none of them anticipated.

**One prerequisite before the first implementation commit, and it is not a
commit:** re-record the baseline (**R1**) from `snake_config_baseline.yml`, in the
**primary checkout**, `--notemp` on WF1, no other session live. The gate fails by
design across this migration, and a comparison gate cannot be applied
retrospectively — every step landing before that re-record is permanently
ungateable. Two board items bear on it: `t2608131718` and `t2608121258`.

**Two facts the design pins that a reader will otherwise re-derive:** the
replaced artifact was never a baseline-manifest target, so `check_baseline` says
nothing about the replacement's correctness (coverage is unit tests plus
`validate_hm7`); and `test_cli.py` must run whenever a rule's declared input
changes, which this does.

The run's full argument — 35 findings, five reviewer verdicts, the ledger, the
process log — is `dev/milestones/r12/stress-test-lookup-review-record.md`.

**The implementation brief is `dev/working/wf3-stress-test-lookup-task-brief.md`** —
scope partition, the nine-step commit plan derived from the design's §8, the
validation ladder with per-rung frequencies, and three human gates. Gate 1 is the
one that cannot be recovered from: the baseline "before" must be recorded before
the first implementation commit.

## The design conversation that preceded it (provenance)

`dev/working/2026-08-15_wf3-scenario-generation-trace/` holds three documents:

- `stress-test-design-and-surface-axes.md` — the design *conversation*, with six
  questions ruled (§6 indexes them) and two same-day revisions. It said "read
  this first" while it was the only design there was; the accepted design has
  superseded it.
- `trace.md` — the run, config to scenario, with the measured cost profile.
- `wf3-rule-reference.md` — every WF3 rule, its scripts and file shapes.

Rulings, all owner, 2026-08-15:

1. **Units: percent everywhere**, keeping the names `temp_change` / `precip_change`.
2. **The lookup is the source of truth.** Indicator tables carry `st_id` + `value`;
   no baked axis. A surface is a declaration plus a figure — axis values are
   derived, never stored.
3. **No external consumer constrains this.** CST-API/frontend out of scope;
   `csthelpers` is parameterized and its owner updates it after the toolbox settles.
4. **Name: `stress_test_lookup.csv`** in `<exp>/config/`; `_work/` disappears.
5. **`st_0` is not a surface member** — baseline reference only, reported as an
   annotated value. It stays simulated: 2 of 11 `q` metrics derive from it.
   *(Stands, with a caveat added 2026-08-15 — see below.)*
6. ~~**Identity-member duplication: alias (option A)**~~ — **WITHDRAWN
   2026-08-15**, premise falsified by its own precondition test. There is no
   duplicate: the perturbation is not the identity at unit factors, so the
   identity member is a distinct scenario and is simulated like any other.
   Design §5 carries the measurement.

## Before implementing

- ~~Verify the alias premise~~ — **DONE 2026-08-15, and it failed.** Do not
  re-run it. No pipeline run was needed: `snake_config_baseline.yml` already has
  an identity member (`st_2`, since precip `step_num: 2` puts 1.0 on a level),
  and `test_case/test_local` holds both it and `st_0`.
  `apply_climate_perturbations` sends every cell through
  `adjust_precipitation_qm` with **no `mean_factor == 1` short-circuit**, so unit
  factors are mean-preserving (all twelve monthly means to +0.0000%) and
  tail-compressing. Measured `st_0` → `st_2`: of eleven `q` metrics **one is
  preserved** (`q_annual_mean` +0.2%), **five move ≤20%**, **five move by a
  factor** — all low-flow, worst `q_mean_annual_min` −69.7% and
  `q_return_level_2yr_7day_min` +127.9%. Full detail and the version caveat:
  design §5, obligation 1.
- **The `st_0` annotation now needs a health warning**, and that is a separate
  board item (`origin: R12`, admitted 2026-08-15): `st_0` is the raw generated
  series and every member is that series round-tripped through rule 3.12, so
  baseline and surface differ by a processing step. Live property of the shipped
  pipeline; **not** a blocker for this item.
- **Only linear statistics** may define an axis, or HM-7's evenly-spaced guarantee
  breaks and the surface stops being a regular grid.
- HM-7 and `validate_hm7` change: the cache-drift check retires with the cache,
  and the seven-column contract loses two columns.

## Refs

- `dev/reference/contracts/hydrological-model-seam.md` — HM-7.
- **t2608082036 depends on this item, and the order is ruled** (owner,
  2026-08-15): this lands first. R12's `member_hash` currently indexes members by
  `tavg`/`prcp`/`precip_variance` — *the annual collapse this item abolishes* —
  so its member-identity re-derivation must follow and key on the monthly lookup
  rows instead. That is why this item is queue 1 and `t2608082036` moved to 2.
  Full reasoning in the design note §7b.
- **t2608151154** — `st_0` is not method-comparable with the surface. Split out of
  this item on 2026-08-15 because it is a live pipeline property rather than part
  of the redesign. Not a blocker here; it is what the §5 caveat points at.
- Related, same territory: t2608071216 (batch size). `trace.md` § 3 is the cost
  baseline for it and for t2608082036.
- Noted in the design (§5d) and belonging to R12 rather than here: rule 3.09 takes
  `ancient(config_path)` with **no params**, so a `stress_test` edit does not
  re-fire the grid rule. `experiment.yml`'s freeze guard exists to compensate.
