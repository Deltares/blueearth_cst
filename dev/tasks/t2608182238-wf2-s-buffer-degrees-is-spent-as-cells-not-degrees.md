---
title: WF2's buffer_degrees is spent as CELLS, not degrees
type: todo-item
status: backlog
effort: 1
area: wf2 projections
origin: 2026-08-18 t2608182020 design
queue:
created: 2026-08-18
updated: 2026-08-18
---

> [!note] Overview
> **What** — REGION_BUFFER_DEGREES = 1.0 (analyze_projections.smk:402) is
> passed as get_rasterdataset(buffer=...), and hydromt spends it in clip_bbox
> as int(np.round(cs.min() - buffer)) — a count of cells, not degrees. Every
> WF2 slice is buffered by one grid cell.
> **Why** — The value is silently resolution-dependent: one cell is 0.70° on
> EC-Earth3 and 2.77° on CanESM5, so the same "1.0" buys a footprint that
> varies ~4× across the ensemble. hydromt's own API is inconsistent about it
> (_parse_geom_bbox_buffer documents buffer as meters; clip_bbox as resolution
> multiplicity), which is how the misnomer got in.
> **Effort** — Small to change, but the ruling below is the work, not the code.

## Progress

- [ ] **Rule the cache invalidation before touching any of the sites.** The two
      fixes fail differently and neither is free (see The trap); pick one
      deliberately and record which, so the choice is not made by omission.
- [ ] Apply it across the sites listed below, in one change.

## The trap

Neither fix is free. buffer_degrees is a digest component
(series_identity.digest_components:557). Making it truly degrees changes the
footprint while the recorded value stays 1.0, so cst_raw_digest does not
move and raw/ goes silently mixed-provenance — the same failure mode ruled
on in t2608182020. Renaming the key to buffer_cells does move the digest
(key names are canonicalized into the hash), invalidating every cached raw
slice and series. Pick the invalidation deliberately; do not let it happen
by omission.

## Sites

analyze_projections.smk:402,594,853,905; fetch_gcm_raw.py:224,521;
get_stats_climate_proj.py:188,379; series_identity.py:518,557;
dev/scripts/stage_cmip6.py:44,108,161,179,325,507 and
stage_cmip6.yml:9,16,18; the cst_buffer_degrees attr in
dev/reference/workflows/wf2-climate-analysis-v2-design.md:498.

## Links

[[t2608182020]], whose irregular-grid branch must reproduce the one-cell
buffer rather than 1.0°, and is why this surfaced.
