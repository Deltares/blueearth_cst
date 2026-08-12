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

### Copying the worktree's tree back is NOT the fix

Measured 2026-08-12, before attempting it. The two trees differ in **104 entries**
— 57 under `experiments/experiment`, 32 under `models/hydrology`, plus
`data/spatial/spatial_maps.nc`, an extra ERA5 climate store, and differing run
records. They are two different runs under different configs, not two versions of
one artifact. Against `dev/baseline/manifest.json`:

| Manifest target | expects | primary | improvements |
|---|---|---|---|
| `q_indicators.csv` columns | `metric, location, st_id, rlz_id, temp_change, precip_change, value` | **pre-R11 order** ✗ | matches ✓ |
| `q_indicators.csv` `n_rows` | 756 | 630 ✗ | 630 ✗ |
| `run_default/output.csv` `n_rows` | 7670 | **7670** ✓ | 6209 ✗ |
| `run_default/output.csv` `mean_ref` | 10.94766158 | **10.94766158** ✓ | 10.58012255 ✗ |

So each tree satisfies what the other fails, and **neither matches the manifest's
WF3 row count**. Copying the worktree's tree over the primary would fix
`test_hm7_integration` and break the one baseline target the primary currently
passes exactly — trading a red for a red while making provenance worse. The
worktree's shorter discharge series (6209 rows) is consistent with a run under a
shortened config, not with `snake_config_model_test.yml`.

## Progress

- [ ] **Do not copy either tree onto the other.** The fix is one deliberate
      re-run in the PRIMARY on `snake_config_model_test.yml` with `--notemp`
      (rule 1.14 declares the discharge target `temp()`), covering WF1 and WF3 so
      tree, code and manifest agree for the first time since R11.
- [ ] Then `check_baseline.py check`, and decide whether the WF3 `n_rows`
      756 → 630 change is an accepted result change (re-record) or a defect. It
      predates this finding — neither tree produces 756.
- [ ] Reseed the lanes afterwards (`worktree-session.py --cwd <lane> sync`, after
      deleting the stale seed) and confirm `test_hm7_integration` goes green.
- [ ] Decide the standing rule: is the primary the only place a fixture may be
      regenerated? `AGENTS.md` already reserves it for integration runs, which
      argues yes — and a worktree run that silently produces a divergent fixture
      is what created this.
