# Post-R6 assessment — observation register

Live register of observations from the owner's own assessment and testing of the
repo **after the R6 structural refactor**. Opened 2026-07-25 against `3c8c2a9`
(`main`). Each row is one observation, from intake through triage to disposition.

**Scope and boundary.** This file owns the owner's post-R6 assessment
observations only. Items that survive triage and belong to a later milestone are
promoted to [`../followups.md`](../followups.md); items needing tracked,
multi-session work get a [`../TODO.md`](../TODO.md) row. Either way the
**Disposition** column keeps the pointer, so nothing lives in two registers. The
pre-existing `## Post-R6` entries already in `followups.md` (surfaced 2026-07-23
during R6 milestone validation) stay where they are — do not re-log them here.

**How to add a row.**

1. Append an index row with the next `O-nn` ID; record `Created`, `Rev` (short
   sha the observation was made against), and `Status: open`.
2. Add a matching detail block below with the exact command, configfile, and
   observed-vs-expected — enough for a future session to confirm the issue still
   applies before acting on it.
3. On any status change, update `Updated` and, once routed, `Disposition`.

**Status vocabulary:** `open` (logged, not yet triaged) · `triaged` (cause
understood, routed) · `fixed` (landed; put the sha in Disposition) ·
`wontfix` (accepted as-is, with a reason) · `not-reproducible` (does not
reproduce under current pins) · `by-design` (expected behaviour, not a defect).

**Kind:** `defect` · `regression` (worked before R6, broken after) ·
`docs` · `usability` · `performance` · `question` (needs a decision, not a fix).

---

## Index

| ID | Observation | Area | Kind | Severity | Created | Updated | Rev | Status | Disposition |
|---|---|---|---|---|---|---|---|---|---|
| O-00 | _(example — delete once real rows exist)_ `wf2` dry-run warns about an unused config key | projections | docs | low | 2026-07-25 | 2026-07-25 | `3c8c2a9` | open | — |

Area labels are free-form; keep to the repo's vocabulary where one fits:
`wf1`/`model`, `wf2`/`projections`, `wf3`/`experiment`, `weathergen`, `shared`,
`config`, `tests`, `ci`, `env`, `docs`, `dev-tooling`.

Severity: `high` (blocks a workflow or produces wrong numbers) · `medium`
(works but wrong/awkward in a way users will hit) · `low` (cosmetic, noise,
wording).

---

## Details

### O-00 — _(example)_ `wf2` dry-run warns about an unused config key

- **Created:** 2026-07-25 · **Rev:** `3c8c2a9` · **Status:** open
- **Command:**
  ```powershell
  pixi run snakemake all -n -s Snakefile_climate_projections --configfile config/workflows/snake_config_model_test.yml
  ```
- **Observed:** _what actually happened — paste the relevant output lines._
- **Expected:** _what should have happened, and why._
- **Notes:** _scope, suspected cause, whether it predates R6, anything that
  narrows it. Delete this whole block with its index row once real
  observations land._

---

## Closure

When the assessment pass is done: promote surviving items to `followups.md` /
`TODO.md`, fill every `Disposition` cell, and add a short outcome summary at the
top of this section (what was checked, what held, what did not). This file then
stays as the durable record of the pass — it is a `dev/reviews/` artifact, not a
working note, so it is not deleted at closure.
