# TODO

Live task board. Unfinished work only -- move closed tasks to `dev/tasks/` and
delete their working notes. Row order expresses priority.

**Next review:** _not scheduled_ -- set a date or milestone for the first
periodic project-health review.

- **ID** -- date-based, `t<YYMMDD><letter>` (e.g. `t260716a`).
- **Status** -- `backlog`, `active`, or `blocked`.
- **Area** -- free project-specific label.
- **Updated** -- ISO date of the last status change.
- **Working note** -- optional link into `dev/working/`.

**No active campaign.** The last one — pre-R6 followups (`fix/pre-r6-followups`),
all waves DONE 2026-07-21 — is closed; its record moved to
[`tasks/2026-07-21_pre-r6-followups.md`](tasks/2026-07-21_pre-r6-followups.md)
on 2026-08-02. R6, R7, and R8 have since been sealed (`dev/roadmap.md`).

_No open backlog rows._ Note this board is **not** the whole picture:
`dev/followups.md` § Post-R7 still carries unmarked defect items that have
never been triaged into rows here. Reconciling the two is open work.

**Done this campaign (2026-07-21, `fix/pre-r6-followups`):** t260720a
(`variance.max` endpoint, `d2de843`), t260720c (D-CAL cftime, `c57eda0`),
t260720d (D-VAR/D-MEM fail-loud, `735cc20`), t260716a truncation warning
(`ce56bc3`), t260721a (wf1 tee wrapper, `d13ba37`), **t260719a** (CSDMS
constant-params restoration via [ADR 0001](decisions/0001-restore-wflow-constant-parameters.md);
gate all-13-PASS, discharge IMMATERIAL, wf1 baseline re-recorded; evidence
`dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md`), **t260716a′** (M1 warnings
re-triage — bucket 2/3 empty of defects; `extract_climate_grid` config-staleness
resolved by R5 params-wiring + verified; docs-only), **t260720e** (D-ATTRS —
confirmed does-not-reproduce under current pins: summary `.nc` + recorded manifest
both carry full CF attrs; no fix, no re-record; docs-only).

**Closed as already-done (verified 2026-07-21):** t260716a `test_cli` xfails
(R3+R5), t260716b `historical:` wiring (R5), t260716c outlet naming (R3).
Upstream weathergenr (t260716b tail) is a separate `tanerumit/weathergenr`
concern, out of this repo's board.

> **Detail lives in `followups.md`**, kept as the milestone-scoped backlog store
> (it carries reproducible context and is referenced by live tests). Promote an
> individual item to an `active` row here when its milestone starts.
