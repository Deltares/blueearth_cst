---
title: Make the WF3 batch-size default actually disk-aware
type: todo-item
status: backlog
effort: 2
area: wf3 batching
origin: P3-3
queue: 6
created: 2026-08-07
updated: 2026-08-17
---

> [!note] Overview
> **What** — Make the WF3 batch-size default actually disk-aware.
> **Why** — The design names the disk ceiling as the binding constraint, and the default ignores it, so a large run can fill the disk mid-flight.
> **Effort** — Medium: the arithmetic is known, the unknown is how to read available disk portably.

## Progress

- [ ] <first step>

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**Make the wf3 batch-size default genuinely disk-aware.** Design §6.1 names
three ceilings on `B` and calls the **disk ceiling the BINDING constraint** on
large `RLZ_NUM×ST_NUM` runs, capped so `p × B × (forcing_size + state_size)`
stays inside a stated headroom. The landed default implements only the
*parallelism* ceiling (`ceil(K / -c N)`), which scales `B` **up** with sweep
size and therefore grows peak temp disk as the sweep grows — backwards from
what §6.1 asks. Commit `3392587` bounds it with an overridable
`batch_size_max` (default 8); that caps the blast radius but is a constant, not
a disk computation. A real cap needs (a) a stated disk-headroom config key and
(b) a per-run forcing+state size estimate, and (b) is the hard part: at parse
time the forcing NCs are `temp()` and do not exist yet, so the estimate has to
come from the wflow grid dimensions × run length × variable count, or from a
measured prior run recorded in config. Verified 2026-07-25: fixture (K=12,
`-c 3`) is unaffected — `min(ceil(12/3), 8) = 4`, so every P3-3 measurement
stands; the clamp only binds from K > 24 at `-c 3`. Confirm the hazard still
applies before fixing (it is scale-dependent and invisible on the seed
fixture, whose peak footprint is 120 MB).
