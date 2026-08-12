---
title: Propagate the post-R11 test_local fixture back to the primary checkout
type: todo-item
status: backlog
effort: 1
area: test fixtures
origin: 2026-08-12 standing-lane split
queue:
created: 2026-08-12
updated: 2026-08-12
---

> [!note] Overview
> **What** — The primary's test_case/test_local carries a pre-R11 q_indicators.csv, so every worktree seeded from it fails tests/test_interchange_contracts.py::test_hm7_integration. Propagate the regenerated fixture (currently only in the improvements worktree) to the primary, or re-run WF3 there, and decide how a regenerated fixture is meant to travel back.
> **Why** — worktree_seed copies the primary, so the primary's fixture age is inherited by every new lane. A red test that is really a stale fixture reads as a code defect, which is the most expensive kind of false signal.
> **Effort** — small

## Detail

Found 2026-08-12 while cutting the two standing lanes. `pixi run test-full` in a
freshly seeded `lane/pipeline` gave **1 failed, 2043 passed, 6 skipped** — the
one failure being `test_hm7_integration`, which **passes** in the older
`improvements` worktree:

```
primary       test_case/test_local/experiments/experiment/results/q_indicators.csv
              metric,st_id,temp_change,precip_change,realization_id,location,value   (mtime 2026-08-10)
improvements  same path
              metric,location,st_id,rlz_id,temp_change,precip_change,value           (mtime 2026-08-11)
```

The second is the R11 CR-2 shape the contract expects. So the WF3 re-run that
regenerated it happened **inside a worktree**, and `test_case/` is untracked, so
nothing carried it back. `worktree_seed` copies the primary, which means every
lane created from now on inherits the 08-10 fixture and fails this test.

The test itself is not at fault and needs no change: it skips only on the
*pre-R11* header (`statistic` first) and never on the file's absence, precisely
so a new-but-broken shape still fails. This fixture is neither — it is a third,
intermediate shape.

## Progress

- [ ] Decide the propagation rule: does a worktree that regenerates a fixture copy
      it back to the primary, or is the primary the only place WF3 may be re-run?
      `AGENTS.md` already reserves the primary for integration runs, which argues
      for the second.
- [ ] Refresh the primary's `test_local` accordingly (copy back, or re-run WF3
      there with `--notemp`), then reseed the two lanes
      (`worktree-session.py --cwd <lane> sync`) and re-run `test_hm7_integration`.
- [ ] Consider whether the untracked fixture needs a provenance marker, so its age
      is legible without diffing headers across checkouts.
