---
title: Outlet stations are named by subcatchment ID, which is not a human-readable station name
type: watch-item
area: naming
origin: M02b
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — Outlet stations are named by subcatchment ID, which is not a human-readable station name.
> **Why** — It reads as arbitrary in every plot and result table, and renaming is a contract change nobody has ruled on.
> **Trigger** — The owner rules on a naming scheme, or a deliverable needs readable station names.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**Outlet station naming convention decision.** hydromt_wflow 1.x's
`setup_outlets` uses subcatchment IDs (e.g. `130000086`, `1`, `2`, …) for
outlet stations rather than the contiguous `1..N` of 0.x. The CSV column
also renamed `Q_gauges` → `Q_outlets`. M2b's `src/plot_results.py`
rebuilds `station_name` as `1..N` to keep `hydro_wflow_1.png` visually
stable; R3 should pick a consistent project-wide convention (real
subcatchment IDs vs `1..N` rebuild) and document it.
