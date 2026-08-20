---
title: WF2's behavioural contract has no live home
type: watch-item
area: wf2 / dev records
origin: reference-doc cleanup (2026-08-20)
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — The WF2 behavioural contract — owned config keys, the precip/temp unit split, save_grids semantics, downstream-consumer semantics — exists only in dev/reference/workflows/climate_projections.md, which is SEALED. rule-index.md is rule/DAG level and does not carry it, and the overview that used to disclaim it ("lives in climate_projections.md and is not repeated here") was deleted 2026-08-20 as stale and self-contradictory.
> **Why** — A sealed record is kept because it was true when written, not because it is current. Reading a behavioural contract off one is the failure the seal exists to prevent, and sealed-records.yml now routes a reader to rule-index.md, which cannot answer the question.
> **Trigger** — Someone needs WF2's config-key or unit semantics and finds only the sealed doc; or a WF2 change makes the sealed contract actively wrong rather than merely old.

## Cause: measured

Measured 2026-08-20, not inferred. `climate_projections.md` is entry 1 of
`dev/reference/sealed-records.yml`, sealed 2026-07-31. Its `current_truth` named
`wf2_climate_projections_overview.md`, whose own text disclaimed the behavioural
contract: *"the behavioral contract ... lives in
`dev/reference/workflows/climate_projections.md` and is not repeated here"*.
That overview was deleted in the same commit as this note, so the pointer now
goes to `rule-index.md`, which is rule/DAG level by its own scope statement.

## Why the overview was deleted rather than fixed

It stated three different rule counts for one workflow — "11 rules", a
9-row inventory table, and "Eight rules plus `all`" — where
`analyze_projections.smk` declares ten. Its table omitted `2.02
delineate_region` and `2.03 delineate_spatial_units` while carrying a
struck-through section for a rule that no longer exists. Its opening line called
it a "working aid for the PLANNED efficiency/modularity rework", a rework sealed
as R8 on 2026-07-31. `rule-index.md` already covers WF2 with ten per-rule
sections matching the Snakefile.

## What would close this

Either a live WF2 behavioural-contract document, or a ruling that the Snakefile
plus `rule-index.md` are sufficient and the sealed doc needs no successor — in
which case say so in the registry rather than leaving the pointer to imply one.
