---
title: Long-running rules announce themselves only by rule name, not in plain language
type: watch-item
area: ergonomics
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — Long-running rules announce themselves only by rule name, not in plain language.
> **Why** — Parked as cosmetic — but on a multi-hour WF3 run the console gives no sense of where it is.
> **Trigger** — Someone runs the pipeline for a user who is watching it, or asks what a stalled run is doing.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.
- No `origin` recorded. It was migrated from the un-milestoned "Cross-cutting — workflow ergonomics"
  section; it was parked 2026-07-19 without being attached to a milestone.

## Detail

**[PARKED 2026-07-19] Per-rule progress messages.** Add `message:`
directives to the long-running rules so each announces itself in plain
language when it starts (e.g. "Building Wflow model from global data…"),
layered on top of Snakemake's built-in `N of M steps (X%) done` counter and
the per-rule timestamps. Snakemake cannot show progress *inside* an external
step (hydromt build, Julia) — only start/end — but the tool's own streamed
output (now visible via `tee`) covers the in-between. Cross-cutting: apply
across all three `Snakefile_*` as a consistent pattern; R4/R5 would inherit
it. Per-rule wall-clock is already captured by the `benchmark:` TSVs added in
R3. Deferred by choice, not a blocker — pick up when convenient (a natural
fit alongside R4/R5 Snakefile work or R6 polish).
