---
title: Rename the project-dir config/observations bin to config/basin_data
type: todo-item
status: backlog
effort: 1
area: naming
origin: owner question on the config bins (2026-08-14)
queue:
created: 2026-08-14
updated: 2026-08-14
---

> [!note] Overview
> **What** — The bin holds output_locations.csv (a specification of where the model reports) alongside observations_timeseries.csv (ground truth), so 'observations' contradicts one of its two files and blocks the future ones. Renamed to basin_data on the content-kind criterion the sibling bins already use.
> **Why** — Owner opened test_rapid/config/ and asked for a better name; the bin is not in the baseline manifest and is not a declared rule input, so unlike the runs/ proposals it is a cheap rename.
> **Effort** — small

## Progress

- [ ] <first step>
