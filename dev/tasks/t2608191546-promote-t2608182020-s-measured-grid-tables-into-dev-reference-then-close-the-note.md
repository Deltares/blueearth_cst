---
title: Promote t2608182020's measured grid tables into dev/reference/, then close the note
type: todo-item
status: backlog
effort: 1
area: wf2 projections / dev records
origin: 2026-08-19 board normalisation
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Move the two measured tables out of dev/tasks/t2608182020 into a durable dev/reference/ document (the 67-model Gaussian-grid sweep and the per-model dlat ranges), repoint the six code citations at it, then close the note with todoboard done so it leaves dev/tasks/ like every other closed item.
> **Why** — t2608182020 is the last note still closed by hand-edited status: done rather than through the ledger, and it is the one that CANNOT simply be deleted: it has no LOG.md row and four live modules cite it as the thing that HOLDS the measurement -- series_identity.py calls it 'the measured table', probe_cmip6_grids.py 'the full table and the options'. Deleting it and writing a summary row would make those comments false; leaving it makes dev/tasks/ mean two different things. A closed board note is not a reference surface, which is what dev/reference/ is for.
> **Effort** — small

## Progress

- [ ] <first step>

## Sites

The six citations to repoint, all bare-id (none is a path, so none is broken
today -- they resolve to a note that must stop existing):

- `blueearth_cst/projections/fetch_gcm_raw.py:102, 534`
- `blueearth_cst/projections/series_identity.py:59`
- `dev/scripts/probe_cmip6_grids.py:13, 70`
- `dev/scripts/stage_cmip6.py:490`

Plus the note itself, `dev/tasks/t2608182020-*.md` (218 lines, 8 sections; the
tables are under `## The sweep` and `## The measurement`).

## Ordering

**Do not start this while a session holds `dev/scripts/stage_cmip6.py`.** Three
of the four modules are wf2 code and the fourth is the staging tool, so the
write set overlaps wf2 territory almost exactly. Deferred from the 2026-08-19
normalisation for that reason -- session-3 had `stage_cmip6.py` and
`shared/snake_utils.py` open at the time. Check `git worktree list` and the
slot claims before claiming this.

## Not a blocker for

The other four hand-closed notes were normalised on 2026-08-19 (see
`dev/LOG.md`). This item is the residue, not the whole job.
