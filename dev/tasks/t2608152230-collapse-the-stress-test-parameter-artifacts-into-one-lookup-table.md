---
title: Collapse the stress-test parameter artifacts into one lookup table, and make the surface axis a parameter
type: todo-item
status: backlog
effort: 2
area: wf3 / stress-test design + reduction
origin: R12
queue:
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
> **Effort** — Large. Design is done and ruled; this is implementation.

## The design is written and the questions are answered

`dev/working/2026-08-15_wf3-scenario-generation-trace/` holds three documents:

- `stress-test-design-and-surface-axes.md` — **the design**, with six questions
  ruled (§6 indexes them). Read this first.
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
6. **Identity-member duplication: alias (option A)** — keep `st_id` dense, reuse
   `st_0`'s result rather than re-simulating.

## Before implementing

- **Verify the alias premise — it is a precondition, not a nicety.** Option A
  *copies* `st_0`'s result into another member's slot, so if the two are not the
  same scenario it fabricates a result. Test: an even-`step_num` config,
  `--notemp`, compare the identity member's forcing and output CSV against
  `st_0`'s. A mismatch is itself a finding.
- **The saving is one member regardless of grid size** — 10% on a 3×3, 4% on a
  5×5, zero on the shipped rapid config. This is design cleanliness, not
  efficiency; do not sell it as the latter.
- **Only linear statistics** may define an axis, or HM-7's evenly-spaced guarantee
  breaks and the surface stops being a regular grid.
- HM-7 and `validate_hm7` change: the cache-drift check retires with the cache,
  and the seven-column contract loses two columns.

## Refs

- `dev/reference/contracts/hydrological-model-seam.md` — HM-7.
- Related, same territory: t2608082036 (R12's execution design, queue 1) and
  t2608071216 (batch size). `trace.md` § 3 is the cost baseline for both.
- Noted in the design (§5d) and belonging to R12 rather than here: rule 3.09 takes
  `ancient(config_path)` with **no params**, so a `stress_test` edit does not
  re-fire the grid rule. `experiment.yml`'s freeze guard exists to compensate.
