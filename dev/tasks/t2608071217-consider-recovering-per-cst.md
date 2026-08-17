---
title: Recover per-cst persistence isolation under batching
type: todo-item
status: backlog
effort: 2
area: wf3 batching
origin: P3-3
queue: 7
created: 2026-08-07
updated: 2026-08-17
---

> [!note] Overview
> **What** — Recover per-cst persistence isolation under batching.
> **Why** — One failing cst takes down its whole batch by design, so a single bad member costs B members of work.
> **Effort** — Large and design-shaped: C5 is degraded deliberately, so this reopens an accepted trade-off.

## Progress

- [ ] <first step>

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**Consider recovering per-cst persistence isolation under batching.** C5 is
DEGRADED by design (blast radius `B`): one failing cst causes Snakemake to
delete the `B−1` completed sibling CSVs and re-run the whole batch, and rule
3.11 is blocked sweep-wide until it succeeds. Measured exactly as documented
(`dev/milestones/p33/batching-results.md` GN-4). §6.1 names the mechanism worth probing:
the `--keep-incomplete` ↔ `--keep-going` interaction (does `--keep-incomplete`
preserve successfully-written sibling CSVs across a failed batch job, and does
the sweep then re-run only the failed cst?), with **accept-the-degradation as
the explicit fallback** if the probe fails. Only worth doing if the blast
radius actually bites in practice.
