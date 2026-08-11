---
title: Cover the two declared artifacts the post-R9 tree inventory misses
type: todo-item
status: backlog
effort: 1
area: test fixtures
origin: t2608111659 rapid rebuild
queue: 2
created: 2026-08-11
updated: 2026-08-11
---

> [!note] Overview
> **What** — build_project_tree_rules in dev/scripts/semantic_tree_diff.py does not cover models/hydrology/wflow/config/climate_store_catalog.yml (a DECLARED, non-temp output of rule 1.10, Snakefile_model_creation:641) or config/runs/invocations/*.json (written by scripts/run_workflows.py). Both report UNMAPPED, so tree-check exits 1 on a correctly built tree.
> **Why** — Neither is a leftover to prune, so the standing advice in the gate's own failure message sends the reader hunting for an orphan that does not exist. test_local has neither shape -- its WF1 predates the store catalog and its runs bypassed the wrapper -- which is exactly why the gap went unseen until test_rapid was rebuilt through run_workflows.py. AGENTS.md makes amending the inventory an owner decision, so this is boarded rather than applied.
> **Effort** — small

## Progress

- [ ] <first step>
