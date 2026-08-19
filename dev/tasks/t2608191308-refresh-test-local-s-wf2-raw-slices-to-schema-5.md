---
title: Refresh test_local's WF2 raw slices to schema 5
type: todo-item
status: done
effort: 1
area: wf2 projections
origin: 2026-08-19 t2608182238
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Re-run WF2 stage A against test_case/test_local FROM THE PRIMARY CHECKOUT, so its 9 raw + 9 scalar slices are rewritten under SCHEMA_VERSION 5 with the cst_buffer_cells attribute. test_case/test_rapid's 4 + 4 follow on its next run and need no deliberate action.
> **Why** — t2608182238's bump left the fixture at schema 4, so tests/test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote SKIPS. That test is the ONLY pin on stage_cmip6's cache-compatibility claim -- DEFAULT_BUFFER_CELLS' own comment names it as what catches a value diverging from the pipeline's -- and there is no non-fixture substitute, because a Snakefile is not importable. Until the refresh the claim is unchecked, not merely untested.
> **Effort** — small

## Progress

- [ ] <first step>

## Links

[[t2608182238]], the rename and schema bump that made the fixture stale.

## Closed 2026-08-19

Ran `snakemake all -c 3 -s analyze_projections.smk --configfile
test_case/snake_config_baseline.yml --keep-going` from the PRIMARY checkout.
24 jobs in 6:17 — all 9 raw slices re-fetched from `gs://cmip6` and all 9
series re-reduced, because the schema bump refused the schema-4 copies rather
than silently reusing them. That refusal is the feature working, observed.

Verified three ways, in increasing order of what they prove:

1. All 18 artifacts (9 raw + 9 scalar) now carry `cst_schema_version = '5'`
   and `cst_buffer_cells`; none carries the old attribute.
2. `pytest tests/test_stage_cmip6.py` is **12 passed, 0 skipped** — the
   digest-reproduction test RUNS again instead of skipping, so the staging
   tool's cache-compatibility claim is checked rather than asserted. That was
   the whole reason this item existed.
3. `check_baseline.py check` reports **OK — 7 target(s) match manifest**
   (3 with `--workflow analyze_projections`). This is the end-to-end proof of
   t2608182238's central claim: the rename moved the digest and no number.
   Had the footprint changed, the change-factor CSVs would have moved here.

Note the fetch cost was ~6 minutes for 9 slices, not the ~19 min PER SOURCE
that `dev/reference/workflows/wf2-climate-analysis-v2-design.md` records from
2026-07-30. The stage-A split is still right — a re-reduction should never pay
a network open — but that benchmark is stale as a planning number.
