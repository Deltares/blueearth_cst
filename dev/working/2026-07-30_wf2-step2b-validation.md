# WF2 v2.0 step 2b — validation record

Covers **commit 2b** (`891f583`, persistent series + content-based identity) and
its follow-up fix (`b7698b9`, the `update()` output flag). Companion to
`2026-07-29_wf2-step1-validation.md`.

Design: `dev/workflows/wf2-climate-analysis-v2-design.md` §5.3, D9, D12, §8 row 2b.
Milestone: Phase 5 / R8.

## Validation ladder — outcomes

| Rung | Command | Outcome |
|---|---|---|
| 1 Narrow | `pytest tests/test_cli.py` | **9 passed** |
| 2 New behavioural | `pytest tests/test_series_identity.py` | **32 passed**, offline |
| 3 DAG diff | `snakemake -n` | 22 jobs; all 9 reduce jobs scheduled — correct on a first post-2b run, since the pre-2b series carry no digest |
| 4 Full gate | `pytest tests/` | **564 passed, 6 skipped, 1 xfailed** (was 524; purely additive) |
| 5 Baseline | `check_baseline.py check` after a full fresh derivation | **OK — 15/15 match** |
| 5 Semantic diff | `semantic_tree_diff.py` | **NOT RUN** — still needs a pre-change reference tree (carry-forward from step 1) |

**Value-neutrality is measured.** All 9 series were re-derived from the network,
the downstream chain re-ran, and every manifest target reproduced. The new
`cst_*` attributes land on the series files, which are not manifest-pinned, and
they do not propagate through the change arithmetic — confirmed by the 15/15,
not assumed.

## The cache properties, measured

These are the point of 2b and could not be tested before it.

**G5 — changing a horizon performs zero network reads.** Added
`near: [2040, 2060]` to the seed config alongside `far`:

```
monthly_change                12   (6 combinations x 2 horizons)
monthly_stats_hist             0
monthly_stats_fut              0
```

Before 2b the `temp()` series were deleted after each run, so the same edit
re-downloaded all 9. Config restored afterwards.

**D9 — a store rerun revalidates offline.** `--forcerun extract_climate_grid`
rewrites the polygon byte-identically and schedules all 9 reduce jobs:

```
0 deriving,  9 cache_hit      -- zero network reads
```

That is design cache test (i): the property `ancient()` used to buy is preserved
by content revalidation instead of trigger suppression, at the cost of
"scheduled, no-op" rather than "not scheduled".

## The finding: D9 was unimplementable as written

**Design D9 item 3 could never have fired, and no unit test would have caught
it.** Snakemake's `Job.prepare()` calls `remove_existing_output()` before every
job — *"Clean up output before rules actually run"* (`snakemake/jobs.py:789`) —
so the series file the revalidation inspects is **deleted before the job body
runs**. `cache_hit` therefore always saw a missing output and always re-derived.

Observed directly: a `--forcerun` of a series whose `cst_series_digest` already
matched the expected value still logged `deriving` and re-read the archive. The
attributes were correct on disk (verified: 12 `cst_*` attributes, digest
`d3aade12935c…` matching); the file was simply gone by the time the job looked.

**The fix is in the same function.** `remove_existing_output()` skips outputs
flagged `"update"`, and that flag's docstring is exactly D9's semantics: *"A flag
for an output file that shall be updated instead of overwritten."* It is
registered into the Snakefile global namespace by
`snakemake.ioflags.register_in_globals`, and its only other DAG effect concerns
`before_update` **input** priorities (`dag.py:2358`), which nothing here uses —
so on an output it purely prevents removal.

Measured before and after, same command:

| | deriving | cache_hit | network reads |
|---|---|---|---|
| without `update()` | 9 | 0 | 9 |
| with `update()` | 0 | 9 | 0 |

**Carry-forward:** `update()` is load-bearing for any persistent, revalidated
Snakemake output. Step 3 renames these outputs to `series/{series_key}.nc` and
must carry the flag across. A test asserting the flag is present would be
cheaper than rediscovering this.

## A test of mine that was invalid, and what it exposed

My first D9 attempt rewrote the polygon byte-identically with Python and expected
the reduce jobs to be scheduled. They were not — the polygon was demonstrably
newer (00:37 vs 00:21 mtime) and Snakemake reported "nothing to be done".

That was **my test being wrong, not the design**: I edited the file behind
Snakemake's back, whereas D9 route (a) concerns a store-*rule* rerun, which marks
its output updated and propagates `needrun`. The `--forcerun` form above is the
valid test.

The invalid test still exposed something real: **an external edit to the polygon
does not schedule re-derivation via the mtime trigger.** Route (b) — the in-job
digest assertion in `get_change_climate_proj.py` — is what catches that case. The
design argued route (b) covers a series restored from a backup or built by an
older checkout; "or a polygon edited outside Snakemake" belongs on that list.

## Observed rather than predicted: the A1 cost

Forcing a store rerun triggers a **full gridded ERA5 re-extraction**, taking
minutes, because WF2 now declares `extract_climate_grid` (D2/A1). This is the
cost transfer risk-07 filed as minor and R4 made load-bearing. It is now a
measurement: a store rerun on a projections-only workflow is not free.

## Operational notes for the remaining steps

- **The `ssp585` reads exceed the 10-minute tool cap** with three in parallel,
  while 3 historical + 3 `ssp245` complete in ~6 minutes. Cause not diagnosed;
  it is a property of those remote stores. Drive long runs as bounded per-target
  calls rather than one `snakemake all`, and **put targets before the flags** —
  `--configfile` takes multiple values and will silently swallow a target path,
  then fail trying to YAML-parse a netCDF.
- A run killed mid-write leaves a manifest-pinned target flagged incomplete;
  `--rerun-incomplete` is the recovery. A `check_baseline` in that window reports
  drift caused by the interruption, not by the code.
