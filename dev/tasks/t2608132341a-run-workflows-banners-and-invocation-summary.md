---
title: Give run_workflows.py workflow-level banners and an end-of-invocation summary
type: todo-item
status: backlog
effort: 1
area: logging / console
origin: "console-output assessment #2 (2026-08-13), item 2"
queue:
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — The wrapper announces each workflow with a plain log_row carrying the raw argv (run_workflows.py:236) while individual rules get bold banners — the biggest structural transitions are styled least. Add a workflow banner ('wf2 climate_projections (2/3)', same isatty+NO_COLOR gate as rule_banner) and an end-of-invocation block: per-workflow verdict + elapsed, plus the invocation-manifest path under config/runs/invocations/, which is written but never announced.
> **Why** — Gives the console the same hierarchy as the run, and fixes the 'artifact invisible unless you already know it exists' problem for the invocation manifest — the same rationale run_summary documents for the merged log and benchmark table.
> **Effort** — small

## Progress

- [ ] Workflow-start banner per invocation — `wf2 climate_projections (2/3)`
      shape, reusing `rule_banner`'s isatty + `NO_COLOR` gating (extract the
      gate rather than duplicating it).
- [ ] End-of-invocation block: one line per workflow (verdict + elapsed,
      `format_elapsed` grammar), then the invocation-manifest path.
- [ ] Keep the argv line — it is the reproduction recipe — but demote it to
      follow the banner rather than be the announcement.
- [ ] Pin the new console shape in `tests/test_run_workflows.py` (the module
      docstring is the wrapper's contract surface; extend it too).

## Refs

- `scripts/run_workflows.py:231-247` — the current `log_row` announcements;
  `_initialize_manifest` writes `config/runs/invocations/*.json`, never announced.
- `blueearth_cst/shared/snake_utils.py` — `rule_banner` (color gate),
  `run_summary` (the per-workflow analogue and its "name the invisible
  artifacts" rationale), `format_elapsed`.
