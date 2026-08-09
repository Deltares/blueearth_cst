# LOG

Closure ledger. One row per board item closed since the board was adopted
(2026-08-07). Work closed BEFORE that lives in `followups-archive.md`, which is
the pre-board ledger — this file is deliberately not backfilled with IDs the
board never issued.

| Closed | ID | Item | Area |
| ------ | -- | ---- | ---- |
| 2026-08-09 | t2608071212 | Rule on adopting `ruff format`, then land it in one mechanical commit — ruled ADOPT (ADR 0005 accepted), and stage 1 landed: 122 files under `tests/`, `dev/` and `scripts/` reformatted, `ruff format --check` armed in CI. Two of the note's instructions were superseded: "one mechanical commit" by the ADR's two-stage split, and "re-record the baseline afterwards" — which was simply wrong, since formatting changes no artifact and the baseline compares equal. Stage 2 (the 51 `blueearth_cst/` modules, held out of the gate until a full re-run is already scheduled) continues as `t2608090907a` | formatting |
| 2026-08-09 | t2608081012 | Fix the pre-R9 fixture guard that has silently skipped the rebuild-cascade test since R9 — repointed; the case now EXECUTES and PASSES, so the cascade had not rotted. The sweep found a second instance, `test_workflow_model_creation.py`, stale three ways and ERRORING rather than skipping. Both in `5476960` | test hygiene |
| 2026-08-09 | t2608090806 | Write WF3 result CSVs in plain decimal notation, capped to 4 significant digits — the note's "needs a baseline re-record" was WRONG: the indicator target is compared with a tolerance, not hashed, and a 4-sig-digit cap sits 20x inside `INDICATOR_RTOL`. Landed in `4139217`; scope was the results writer only, never the `st_*.csv` inputs | wf3 results |
| 2026-08-09 | t2608080807 | Rename metrics_definition's returninterval functions to return_level, and fix the returninternval typo — closed by DELETING the three functions, which R11 had already left without callers; the typo half was discharged by R11's `03da1b7`. Rationale in `6f8e1f1` | wf3 metrics |
| 2026-08-08 | t2608071945 | Decide whether the tee_to_log traceback fix lands inside R11 or on its own | workflow ergonomics |
| 2026-08-07 | t2608071944 | Settle whether unit B's cst_ to st_ rename reaches the frozen experiment.yml | wf3 identification |
| 2026-08-07 | t2608071210 | Retire the 48 MB pre-R7 reference tree at `~/workspace/.r07-reference/` | housekeeping |
| 2026-08-07 | t2608071200 | Re-record the baseline manifest for the indicator-table axis-column rename | baseline |
