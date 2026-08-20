---
title: All 9 fixture slices fail the digest-reproduction check at schema 6
type: todo-item
status: backlog
effort: 1
area: wf2 projections / test fixtures
origin: 2026-08-20 t2608191613 closure gate
queue: 1
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — tests/test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote FAILS on main. Every one of the 9 test_local raw slices carries cst_schema_version = 6, so the whole-fixture skip does NOT engage, and the recomputed digest disagrees with the one WF2 wrote: cmip6_INM_INM-CM4-8_historical_r1i1p1f1 recomputes e629400635b2 against a recorded 03877288efa5.
> **Why** — This is the failure the test exists to catch: a staged slice would be re-fetched instead of reused, and the tool's recipe no longer reproduces what the pipeline writes. It is NOT a stale fixture -- the fixture is current (rewritten 2026-08-19 21:43 under schema 6 by t2608192113, which recorded the test as 29 passed / 0 skipped), and the slices in the primary and in session-1 are byte-identical. So something moved the recipe after that run WITHOUT a SCHEMA_VERSION bump, which is exactly the silent digest move the schema field exists to make loud.
> **Effort** — small

## Progress

- [ ] <first step>
