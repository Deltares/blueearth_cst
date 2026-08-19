---
title: Refresh test_local's WF2 raw slices to the current schema
type: todo-item
status: backlog
effort: 1
area: wf2 projections
origin: 2026-08-19 t2608182238
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Re-run WF2 stage A against test_case/test_local FROM THE PRIMARY CHECKOUT, so its 9 raw + 9 scalar slices are rewritten under the CURRENT SCHEMA_VERSION. test_case/test_rapid's 4 + 4 follow on its next run and need no deliberate action.
> **Why** — t2608182238's bump left the fixture at schema 4, so tests/test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote SKIPS. That test is the ONLY pin on stage_cmip6's cache-compatibility claim -- DEFAULT_BUFFER_CELLS' own comment names it as what catches a value diverging from the pipeline's -- and there is no non-fixture substitute, because a Snakefile is not importable. Until the refresh the claim is unchecked, not merely untested.
> **Effort** — small

## Progress

- [ ] <first step>

## Links

[[t2608182238]], the rename and schema bump (4->5) that made the fixture stale.
[[t2608191613]], the newest-version ruling that bumped it again (5->6) before
this refresh was done — which is why the title no longer names a number. One
re-run clears both; there is no separate work per bump.

The fixture was never refreshed to 5, so nothing was wasted by the second bump
and nothing is owed twice.
