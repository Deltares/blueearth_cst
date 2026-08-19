---
title: Refresh test_local's WF2 slices to schema 6, so the digest check stops skipping
type: todo-item
status: backlog
effort: 1
area: wf2 projections / test fixtures
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Re-run WF2 stage A against `test_case/test_local` FROM THE PRIMARY
> CHECKOUT, so its 9 raw + 9 scalar slices are rewritten under
> `SCHEMA_VERSION = "6"`. `test_case/test_rapid` follows on its next run and
> needs no deliberate action.
> **Why** — The `fix/wf2-improvements` merge bumped the schema 5 → 6, which
> makes the fixture stale and sends
> `tests/test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote`
> back to SKIPPING. That test is the ONLY pin on `stage_cmip6`'s
> cache-compatibility claim — a Snakefile is not importable, so there is no
> non-fixture substitute. Until the refresh the claim is unchecked, not merely
> untested, and the suite reports green while saying nothing about it.
> **Effort** — Small. Measured at 24 jobs in 6:17 the last time it was done.

## Why this is a new item and not t2608191308 reopened

`t2608191308` closed truthfully: it refreshed the fixture to schema 5 and the
check ran again. The 5 → 6 bump is a **second** occurrence with a different
cause, so it gets its own ID rather than contradicting a closure row. Reviving
the old ID would leave it both closed in `LOG.md` and open in `tasks/`.

This is the second time one schema bump has stranded the fixture. If it happens
a third time, the item to raise is not another refresh — it is whether the
digest check should be able to fail rather than skip when the fixture is behind.

## Progress

- [ ] Re-run WF2 stage A against `test_case/test_local` from the primary checkout
- [ ] Confirm `tests/test_stage_cmip6.py` reports 0 skipped
- [ ] `check_baseline.py check` to confirm no number moved

## Links

[[t2608191308]], the same refresh done once already, for schema 5.
[[t2608191613]], the newest-version ruling whose fix bumped 5 → 6.
