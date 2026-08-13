---
title: Re-render the workflow notebooks when their banner sha falls behind
type: watch-item
area: docs / notebooks
origin: t2608131847 rot-control ruling (option C)
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — The three notebooks under `docs/notebooks/` commit their rendered outputs and carry a dated `rendered against <sha>` banner. They are currently stamped `ca4c9df` (2026-08-13). Nothing re-renders them, by design: CI cannot: a bare checkout has neither `test_case/test_rapid` nor data access. This item is the counterweight the ruling attached to that choice.
> **Why** — Option C of the rot-control ruling (`dev/reviews/2026-08-13_fao-branch-assessment.md` §6.3) trades prevention for visibility: outputs rot, but the banner says when they last did not. That trade is only honest if someone eventually looks. Without this item, "rots visibly" degrades into "rots", which is exactly how the previous three notebooks reached the state t2608131847 repaired — result paths three tree-moves stale, and no gate that could notice, because nothing executes these files.
> **Trigger** — Any of: a change to the pipeline that moves a path or a number the notebooks display; a rule renumbering or rename (the rule-by-rule tables name every rule by number and by the config keys that tune it); a config schema change (each notebook prints `test_case/snake_config_rapid.yml` in full and narrates its blocks); or simply the banner sha being far enough behind `main` that a reader would be misled. Recipe and its two traps: `docs/notebooks/README.md` § Re-rendering.

## Notes

The prose and the outputs rot **independently**, and only one of them is
mechanically detectable.

- **Prose** — rule tables, config-key narratives, result paths — is checked by
  nothing. It is the half that went stale before, and it goes stale on the
  ordinary schedule of any documentation, so it is covered by the standing
  "keep configuration references current" rule in `AGENTS.md`, not by this item.
- **Outputs** — figures, tables, run logs — are what the banner is about. They
  can only be refreshed by a real run from the primary checkout.

A re-render is therefore not automatically the right response to a trigger. If
the pipeline moved but the numbers did not, fixing the prose and **leaving the
banner alone** is correct: the banner describes the outputs, not the text.

Cost, measured 2026-08-13: a terminal-rules-only re-render (delete the `plots/`
directories, `performance_metrics.csv` and `experiments/*/results/`, then run)
is ~15 minutes and produces real Snakemake job logs in the run cells. A full
fresh rapid run is ~73 minutes of rule time and needs the CMIP6 store
reachable.

## Refs

- `docs/notebooks/README.md` § Committed outputs, and how they go stale.
- `dev/reviews/2026-08-13_fao-branch-assessment.md` §6.3 — the ruling.
- [[t2608131847-repair-and-extend-the-workflow-notebooks]] — the item that
  built them, and which this one exists to keep honest.
