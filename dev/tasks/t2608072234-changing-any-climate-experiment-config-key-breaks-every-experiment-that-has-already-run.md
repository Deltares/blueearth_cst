---
title: Changing any climate_experiment config key breaks every experiment that has already run
type: watch-item
area: wf3 experiment lifecycle
origin: R11
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — experiment.yml records the resolved workflows.climate_experiment section and is frozen at an experiment's first successful run. Adding or removing ANY key in that section makes check_not_frozen refuse, because it compares the union of keys and a missing one reads as changed. Confirmed on the fixture 2026-08-07: retiring aggregate_rlz raises ExperimentConfigFrozenError.
> **Why** — This is the freeze working as designed -- a changed setting really does redefine what existing results mean -- and R11 rules it per-milestone (accept the break, re-run as a new experiment). That ruling is cheap only while no production trees exist. Every future milestone touching this section pays it again.
> **Trigger** — A third milestone needs the same ruling, or a real project reports being unable to continue an experiment. At that point the freeze wants a schema-version concept rather than a per-milestone support decision.
