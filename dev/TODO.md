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

## Open

Triaged from `dev/followups.md` on 2026-08-02 — every Post-R7 item was read and
each is now either resolution-marked there or carries a row here.

| ID | Status | Area | Updated | Item |
|---|---|---|---|---|
| `t260802b` | backlog | wf1 rule shape | 2026-08-02 | `hydro_*`, `clim_*`, `signatures_*` figures are undeclarable at parse time, so `--delete-all-output` leaves stale figures on a config with extra gauges or observations. Needs a `checkpoint` or `directory()` output. `followups.md` R7-5 (basin-average half already fixed) |
| `t260802c` | backlog | test hygiene | 2026-08-02 | `tests/test_stage_data_incremental.py` fails intermittently under some orderings, passes in isolation. Test-isolation issue, not a product defect. `followups.md` R7-14 |
| `t260802d` | backlog | packaging | 2026-08-02 | O-14 decision 2 (real packaging) needs a superseding record in `dev/decisions/`; O-16 (flit) stays blocked until it lands. `followups.md` R7-16 |
| `t260802e` | backlog | importability | 2026-08-02 | `downscale_climate_forcing.py` is the last module reading the bare `snakemake` global at import; converting it makes it unit-testable and lets the `F821` per-file-ignore in `pyproject.toml` be **deleted**. Not mechanical — the body sits inside `tee_to_log(...)`. `followups.md` R7-22 |
| `t260802f` | blocked | formatting | 2026-08-02 | Adopt `ruff format`? Now **136** files / ~7.8k lines (118 at R7). A churn decision, not a defect — needs an owner ruling, then one mechanical commit and a baseline re-record. `followups.md` R7-23 |
| `t260802g` | backlog | housekeeping | 2026-08-02 | Retire the 48 MB pre-R7 reference tree at `~/workspace/.r07-reference/`. Its precondition — R7 sealed — was met 2026-07-29, so this has simply gone unexecuted. `followups.md` R7-20 |

**Closed since the triage:** `t260802a` (ruff gate red on `main`) — DONE
2026-08-07, `2c8f32a`; `pixi run ruff check .` is clean and `followups.md` R8-1
carries the remaining question (why a red gate went unnoticed).

Left deliberately without a row: R7-8 (wflow `log.txt` placement — gate-invisible
cosmetic whose verification needs a full wf3 run), R7-7 (working as intended),
R7-21 candidates (b)/(c) (conditional on misattribution recurring), and the
R7-15/17/18 parking rulings.

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
