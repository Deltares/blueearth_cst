---
title: Preflight the wf1 leaves when run_workflows drives wf3 without build_model
type: todo-item
status: backlog
effort: 1
area: orchestration / run_workflows
queue:
created: 2026-08-17
updated: 2026-08-17
---

> [!note] Overview
> **What** — Before invoking anything, `scripts/run_workflows.py` checks the three
> `dev/scripts/cross_workflow_inputs.py::LEAVES` against `project_dir` whenever
> `run_stress_test` is enabled and `build_model` is not, and exits with a message
> naming wf1 as the missing producer. It already has both halves: it parses every
> `enabled:` flag, and the leaf list is a shared definition with a test proving it
> complete and minimal.
> **Why** — A fresh project with `build_model.enabled: false` spends its whole run
> on wf0 and wf2 and only then discovers wf3 was never runnable. Measured
> 2026-08-17 on `C:/TESTS/CST/rapid`: 4:14 total, of which wf3 was 0:07.
> **Effort** — Small. One check in the wrapper plus its test; no Snakefile change.

## Progress

- [ ] Decide where the check lives — beside `_enabled_flags` at parse time, or in
      `run_workflows` just before the first invocation
- [ ] Import `LEAVES` from `dev/scripts/cross_workflow_inputs.py` rather than
      restating the three paths, per that module's own "one definition" argument
- [ ] Message names all three missing leaves and wf1 as their producer
- [ ] Test in `tests/test_run_workflows.py`, alongside the other contract clauses
- [ ] Update the module docstring's contract list — the check is part of the
      contract or it is not

## Why this is not the prerequisite check the docstring declines

`run_workflows.py` states the exclusion deliberately:

> no prerequisite-freshness check -- identical to invoking a single Snakefile
> directly today. A user who disables a prerequisite owns the staleness of what
> it consumes.

That is about **staleness**, and it should stand: deciding whether an existing
wf1 model is new enough for this wf3 run needs the DAG, and duplicating that
judgment in the wrapper is how the two drift apart.

Absence is a different claim, and a cheaper one. A stale artifact still resolves
the DAG and produces a defensible answer the user owns. A missing leaf makes the
DAG **unresolvable** — there is no run to own the staleness of. So the check
proposed here is an existence test over three paths, never a comparison, and it
can never disagree with Snakemake about whether a rule should re-run.

Whether that distinction is worth carrying in the wrapper is the actual decision,
and it is the owner's. The alternative — leave it, on the grounds that
`enabled: false` beside an enabled consumer is the user's error and the message
is merely unkind — is defensible and costs nothing to adopt.

## What the failure looks like today

Snakemake stops at rule 3.01 `check_project_consistency` because it is the
earliest rule to declare a leaf, and reports **one** missing file:

```
MissingInputException in rule check_project_consistency
    affected files:
        C:/TESTS/CST/rapid/config/runs/snake_config_build_model.yml
```

Three are missing. `LEAF_MODEL_TOML` and `LEAF_MODEL_READY` are declared by rule
3.01c `write_model_reference` and would fail next. The truncation is what makes
the message actively misleading rather than merely late: it invites hand-creating
the one named YAML, which buys exactly one rule of progress. Any message this
item adds should name the full set, which is the reason to read it from `LEAVES`
rather than to restate it.

## Related

- [[t2608131807-collapsing-the-per-workflow-config-copies-is-blocked-by-wf3-s-ancient-input]]
  — the same `ancient()` input, seen from the config-snapshot side. If that item
  ever removes the wf1 snapshot from wf3's inputs, the leaf set here shrinks to
  two and this check narrows with it.
