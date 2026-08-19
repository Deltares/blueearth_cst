---
title: WF2's buffer_degrees is spent as CELLS, not degrees
type: todo-item
status: done
effort: 1
area: wf2 projections
origin: 2026-08-18 t2608182020 design
queue:
created: 2026-08-18
updated: 2026-08-19
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

- [x] **Rule the cache invalidation before touching any of the sites.** Ruled by
      the owner on 2026-08-19: **rename the key, and bump `SCHEMA_VERSION` 4->5
      to make the invalidation loud.** See The ruling.
- [x] Apply it across the sites listed below, in one change.

## The ruling

`buffer_degrees` -> `buffer_cells`, semantics untouched, `SCHEMA_VERSION` 4->5.

**Why cells and not degrees.** Three places already committed to the cell
reading before this note existed: the constant's own comment records that it
replaced a bare `buffer = 1`; `fetch_gcm_raw.bbox_index_slice` states that
`buffer` is in CELLS and that the irregular path must read it the same way or
the two would differ inside one ensemble; and t2608182020's branch had to
reproduce the one-cell buffer rather than 1.0 deg. So the NAME was the defect
and the value was never wrong. Making it truly degrees is a change to the
sampling footprint — a scientific decision with its own justification, not a
naming fix — and it is not effort-1, since hydromt spends `buffer` as
resolution multiplicity and degrees would need `ceil(deg/res)` per grid or a
self-expanded bbox with `buffer=0`, coherently on both the regular and the
irregular path. Raise it separately if the ~4x ensemble spread is unwanted.

**Why the bump, when the rename already moves the digest.** It moves it
*silently*: key names are canonicalized into the `sort_keys` JSON that is
hashed, so every raw and series digest changes and every cached slice simply
re-derives with nothing said. `SCHEMA_VERSION` is the repo's declared
invalidation lever (`cache_hit` refuses an unknown version by name, telling the
operator to delete the slice), and its own docstring scopes it to "the attribute
set, the digest recipe, or the key grammar" — this change is all three. The bump
converts a silent re-derive into a loud refusal.

**Third option rejected:** renaming the identifiers while freezing the digest
component key at `buffer_degrees` would have preserved every cached slice, at
the cost of a provenance record and a cache key that disagree about the field's
name — and it sidesteps the ruling rather than making it.

**Accepted cost, measured:** 9 raw + 9 scalar slices in `test_case/test_local`
(632 KB) and 4 + 4 in `test_case/test_rapid` re-fetch from `gs://cmip6`. No
number moves: the footprint is identical, so `dev/baseline/manifest.json` — which
covers the change-factor CSVs, not the netCDFs — must re-record identical.

**Follow-on, not yet done:** the `test_local` fixture still carries schema-4
slices, so `test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote`
SKIPS with a reason naming the refresh. It is skipped rather than failed because
a digest is only comparable within one schema — the pipeline rejects a stale
slice on the schema check before consulting any digest, so an inequality there
would say nothing about the staging tool. Re-run WF2 stage A against
`test_case/test_local` **from the primary checkout** to restore the check.

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
