---
title: Fix the pre-R9 fixture guard that has silently skipped the rebuild-cascade test since R9
type: todo-item
status: backlog
effort: 1
area: test hygiene
origin: R11
queue:
created: 2026-08-08
updated: 2026-08-08
---

> [!note] Overview
> **What** — `tests/test_model_rebuild_cascade.py` guards its integration test on a directory that has not existed since the R9 project-tree migration, so the test skips on every machine, including one with the fixture fully present. Point the guard at the current path.
> **Why** — The guarded test is the only thing that runs a real `snakemake all -c 1` against the fixture and checks that editing the wflow model reschedules the whole TOML chain. That contract has been unverified for two milestones while the suite reported green over it — the failure mode is a rebuild that silently does not cascade, discovered by a user rather than by a gate.
> **Effort** — The one-line path fix is trivial; the work is confirming the test still passes once it actually runs (it drives a real Snakemake invocation, so it may have rotted in the two milestones it was dark), and sweeping the rest of `tests/` for guards naming other pre-R9 roots.

## Progress

- [ ] Point the guard at the post-R9 path (`models/hydrology/wflow`)
- [ ] Run the test on a seeded checkout and confirm it EXECUTES rather than skips
- [ ] Fix whatever the now-live test reports, or record why it is expected
- [ ] Grep `tests/` for other guards naming a pre-R9 root — the class matters more than this instance

## Refs

- Found 2026-08-08 during R11 P2 Gate 1, while accounting for the full-suite skip list.
  Not caused by P2: the file was last touched by `c058a02` (the R10 rule renames), and the
  path it names was moved by R9.
- `dev/followups-archive.md` `[R9-4]` — the archived instance of this exact shape.
- `AGENTS.md`, Repo Map (`.github/workflows/ci.yml` bullet) — already states that a wrong
  path behind an existence guard becomes a silent skip rather than a failure, and that
  tree-shape gates cannot catch it because they do not read the code that reads the tree.

## Detail

**The guard**, `tests/test_model_rebuild_cascade.py:67`:

```python
@pytest.mark.skipif(
    not (SNAKEDIR / "test_case" / "test_local" / "hydrology_model").is_dir(),
    reason="untracked test_case/test_local fixture tree not present",
)
@pytest.mark.workflow_contract
def test_rerunning_build_wflow_model_reschedules_the_whole_toml_chain():
```

`test_case/test_local/hydrology_model` is a **pre-R9** path. R9 moved the model to
`models/hydrology/wflow`, so the directory has not existed since that migration.

**Verified rather than reasoned about**, on a worktree seeded per `AGENTS.md`:
`test_case/test_local/` holds `benchmarks config data experiments logs models` — the
fixture is fully present — and the test skips anyway, reporting
`untracked test_case/test_local fixture tree not present`. The reason string states the
opposite of the truth, which is what makes this invisible in a skip list: it reads as the
ordinary bare-checkout skip that CI legitimately produces.

**Why it matters more than a normal stale path.** This is the R9-4 shape recurring:
a wrong path behind an `is_dir()` / `os.path.exists` guard turns into a SILENT SKIP rather
than a failure. R9 shipped 22 such failures and three silent skips; this one survived that
cleanup. The affected test is not a unit case — it invokes Snakemake for real and asserts
the model-rebuild cascade, which is exactly the kind of contract no static gate
(`semantic_tree_diff`, `check_baseline`, `tree-check`) can substitute for, because none of
them reads the code that reads the tree.

**Do not fix it inside an R11 phase.** R11 P2's commits are a bounded token rename; P3 owns
the milestone's single WF3 re-run and baseline re-record. Un-skipping a real Snakemake
integration test is likely to surface unrelated work and belongs on its own branch.
