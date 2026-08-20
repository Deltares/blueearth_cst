---
title: Two Snakemake-running test files carry no tier marker
type: watch-item
area: testing / tiering
origin: xdist trial (2026-08-20)
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — tests/test_workflow_build_model.py and tests/test_interchange_contracts.py both invoke Snakemake against the shared test_case/test_local tree and carry neither workflow_contract nor process_isolation, so they run in test-fast -- now in parallel.
> **Why** — The markers exist to separate tests that take .snakemake locks and write a shared project_dir from those that do not. These two are mismarked by that rule. They pass today because file-grouping keeps each file on one worker, but they are the reason parallel test-full is unsafe: the writers must be in the serial tier.
> **Trigger** — An intermittent failure or lock error appears in either file under -n auto, or someone proposes parallelising test-full.
