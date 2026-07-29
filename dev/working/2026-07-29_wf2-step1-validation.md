# WF2 v2.0 step 1 — validation record

Deliverable required by `dev/working/2026-07-29_wf2-v2-decouple-and-cache.md`
§ Output requirements. Covers **commit 1** (`dcd5459`, model-free region) and the
generator capability of **commit 2a** (`37b2e1f`).

Design: `dev/workflows/wf2-climate-analysis-v2-design.md` (ACCEPTED), §8 row 1.
Milestone: Phase 5 / R8.

## Validation ladder — outcomes

| Rung | Command | Outcome |
|---|---|---|
| 1 Narrow | `pytest tests/test_cli.py` | **9 passed** |
| 2 New behavioural | `pytest tests/test_climate_store_contract.py` | **16 passed** — extended from 2 declarations to 3, compared pairwise |
| 2 New behavioural | `pytest tests/test_generate_cmip6_catalog.py` | **8 passed** — offline, `FakeFS` for the bucket |
| 3 DAG diff | `snakemake -n` on the seed config | **23 → 24 jobs** (+1 `extract_climate_grid`); no other rule changed |
| 4 Full gate | `pytest tests/` | **524 passed, 6 skipped, 1 xfailed** — same skip/xfail profile, purely additive |
| 5 Baseline | `check_baseline.py check` after a **completed** WF2 run | **OK — 15 of 15 targets match manifest** |
| 5 Semantic diff | `semantic_tree_diff.py` | **NOT RUN — see limitation below** |
| 6 Region re-check | bounds comparison | **Gate 2 fired**; see below |
| 7 Decoupling | WF2 DAG built with `hydrology_model/` moved aside | **24 jobs, no MissingInputException** |

## Rung 5 is the real result

`check_baseline` passing is only meaningful **after** a full WF2 run regenerates
the manifested outputs. That run is now complete, so the 15/15 match is
empirical, not trivial: WF2 ran end-to-end reading the store's
`store_region.geojson` as a plain input instead of
`ancient(hydrology_model/staticgeoms/region.geojson)`, and every fingerprinted
target — including all 7 WF2 ones — reproduced.

**Commit 1's value-neutrality is therefore measured, not argued.**

Getting there took three attempts, recorded because the failure modes matter:

1. Background run killed at 12/24 jobs (mid-`ssp585` reads).
2. Background run killed again at the same stage.
3. Foreground run SIGTERMed at the tool's 10-minute cap — but it had completed
   all `ssp585` stats **and** change factors first, and left
   `summary/annual_change_scalar_stats_summary.nc` (a manifest-pinned target)
   flagged **incomplete** by Snakemake. `--rerun-incomplete` finished the last
   4 jobs cleanly.

The `ssp585` reads are the slow stage: three in parallel exceed 10 minutes, while
the three historical plus three `ssp245` reads together took ~6 minutes. Cause
not diagnosed — it is a property of those remote stores, not of this change.

**Operational note:** a partial run can leave a *manifested* target half-written.
Snakemake detects it and refuses to proceed, which is the right behavior, but a
`check_baseline` run between the kill and the `--rerun-incomplete` would have
reported drift caused by the interruption rather than by any code change.

## Gate 2 — fired, ratified by the owner

The two region polygons' bounds are **not identical**, contrary to the original
measurement:

```
model region.geojson : 9.658333          (stored rounded to 6 dp)
store_region.geojson : 9.65833333316084  (full precision)
max component delta  : 3.33e-07 deg  (~3.7 cm at the equator)
```

The original check compared *rounded* values, so "identical" was overstated and
had propagated into the design, the review record, the framing given to both
reviewers, and the task brief. All corrected in `04013fc`.

The conclusion survives on a bound rather than equality: WF2 consumes only
`geom.geometry.bounds`, so the difference matters only if a grid-cell edge falls
inside a 3.3e-07° interval. Checked across **36 (resolution, origin)
combinations** spanning CMIP6 Amon grids (0.9375°–2.8125°; origins at 0, −180,
and half-cell offsets): **zero** change cell selection. Owner ratified.

Rung 5's 15/15 match is now independent confirmation of the same conclusion.

## Limitation — the semantic tree diff was not run

`semantic_tree_diff.py` requires `--ref` (a reference output tree) and `--cur`;
it compares two trees, not a tree against the manifest. No pre-change snapshot of
`test_case/test_local` was taken, so this gate cannot be run retrospectively for
commit 1.

`check_baseline`'s 15 targets are therefore the whole of the numerical evidence,
and its WF2 coverage is thin by design — 3 PNGs (size only), the two summary
CSVs (sha256), the summary `.nc` (per-variable stats), and the config snapshot
(sha256). **No monthly intermediates are fingerprinted.**

**Carry-forward:** steps 5a–5e are value-*changing* and their gate is a
per-cause characterized diff. Each needs a reference tree copied **before** it
runs. Snapshot `test_case/test_local` first.

## Not applicable to commit 1

The brief's "second run performs zero network reads" criterion is a property of
the **persistent cache (commit 2b)**, not of the region swap. The series are
still `temp()` at this commit, so a re-run still re-downloads. Not claimed here.

## Commit 2a status

The generator capability landed with offline tests, but **the crawl has not been
run**, so `config/catalogs/cmip6_store_index.json` does not exist yet.
Deliberate: regenerating the catalog mid-run could have changed member lists
under the in-flight baseline run. The crawl is now unblocked.
