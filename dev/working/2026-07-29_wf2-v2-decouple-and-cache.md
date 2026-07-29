# Task Brief — WF2 v2.0 steps 1 / 2a / 2b: model-free region + persistent series cache

> **Revised 2026-07-29** after the `wf2-climate-analysis-v2` design review closed
> at G2. The design this brief implements is
> `dev/workflows/wf2-climate-analysis-v2-design.md` (**ACCEPTED**); the audit
> trail is `…-design-review-record.md` beside it. Three review findings changed
> this brief materially — see *What the review changed* below. The superseded
> draft covered "steps 1–2"; the accepted plan splits that into **1, 2a, 2b**.

### Context

Canonical ruleset: `AGENTS.md`. **Authoritative spec: the accepted design** —
§5.3 (series store, identity, caching), §5.4 (region and reference window),
**D9** (region content identity), **D12** (store index), §8 (migration table
rows 1 / 2a / 2b), §9 (validation plan, cache tests (a)–(k)). Current-state map:
`dev/workflows/wf2_climate_projections_overview.md`.

This brief bounds the work and names the gates; **it does not restate the
design**. Where the two differ, the design wins — report the discrepancy rather
than choosing.

- WF2's only cross-workflow input today is
  `hydrology_model/staticgeoms/region.geojson`, used solely for
  `geom.geometry.bounds` + a 1° buffer. `snake_utils.climate_store_spec` already
  produces a model-free `store_region.geojson` and is declared identically in
  `Snakefile_model_creation` (1.10) and `Snakefile_climate_experiment` (3.02).
- **Measured 2026-07-29 on `test_case/test_local`:** both polygons bound
  `[9.658333, 0.35, 9.858333, 0.483333]` — identical. On this fixture the swap
  selects the same GCM cell set and cannot move a number.
- All three WF2 intermediate netCDF families are `temp()`, so a re-run with one
  changed horizon re-downloads the whole archive slice.
- `Snakefile_climate_projections:119` makes `monthly_stats_fut` depend on
  `monthly_stats_hist`'s output with the comment *"make sure starts with previous
  job"*. `get_stats_climate_proj.py` never opens that file. The likely real
  reason is the unguarded `os.mkdir(folder_out)` at lines 179–180.

### What the review changed (do not work from the superseded draft)

1. **`store_region.geojson` is a PLAIN input, not `ancient()`** (D9, finding
   `ext2-01`). The earlier draft said `ancient()`. Rationale: the polygon's
   *content* — not the region *specification* — is the cache identity, because a
   catalog or delineation change can rewrite the polygon while
   `shared.basin.region` is unchanged. With `ancient()` plus a
   specification-only digest, stale series were silently reusable.
2. **Step 2a is new** — the generator emits
   `config/catalogs/cmip6_store_index.json` (D12, finding `ext2-04`) pinning the
   observed `{grid_label}/{version}` per (entry, member, certified variable).
   Nothing reads it yet, so 2a is value-neutral on its own.
3. **The digest is much richer than the draft's** — entry + pinned physical
   paths + driver/adapter/metadata maps + polygon content fingerprint + module
   hash, plus the `cst_*` series attributes, the revalidating reduce-job entry
   check, read-time pin verification, and the stage-B backstop. §5.3 is
   normative.

### Goal

WF2 runs end-to-end with **no `hydrology_model/` on disk**, and a second run with
a changed `future_horizons` entry performs **zero network reads** — with every
manifested output byte-identical to the current baseline.

### Non-goals

- Retiring `historical_year_range` / moving the reference window to 1985–2014.
  That is step 5e, value-changing, and gated separately.
- Steps 3–7: rule collapse, the resolution ladder, `composition.csv`, weighting,
  calendar, the monthly table, the report, the gridded branch.
- Anything requiring a new third-party dependency.

### Allowed scope

**Permitted**
- `Snakefile_climate_projections`
- `blueearth_cst/projections/get_stats_climate_proj.py`
- `dev/scripts/generate_cmip6_catalog.py` (step 2a only)
- `tests/` — additive only

**Approval-gated** (pause and ask; name the reason)
- `blueearth_cst/shared/snake_utils.py` — `climate_store_spec` is co-owned by
  three DAGs. Prefer consuming it unchanged; if a change is unavoidable, stop at
  Gate 1.
- `config/catalogs/cmip6_data.yml` — **generated**; only ever via its generator,
  never by hand.
- `config/workflows/*.yml` — only if a new key is genuinely required.

**Forbidden**
- `Snakefile_model_creation`, `Snakefile_climate_experiment` — their
  `extract_climate_grid` declarations must not be edited (see Task constraints)
- `dev/baseline/manifest.json`, `pixi.lock`, anything under `project_dir`

### Required changes (checklist)

**Commit 1 — model-free region** (design §8 row 1)

1. Declare `extract_climate_grid` in `Snakefile_climate_projections` from
   `snake_utils.climate_store_spec`, with an input set **identical** to the other
   two declarations: exactly one entry, the catalog file, declared plain, never
   `ancient()`.
2. Point the stats rules at `store_region.geojson` as a **plain input** instead
   of `{basin_dir}/staticgeoms/region.geojson`.
3. Add a test asserting the three `climate_store_spec` declarations produce
   identical input sets.

**Commit 2a — store index** (design §8 row 2a, D12)

4. Extend `dev/scripts/generate_cmip6_catalog.py` to emit
   `config/catalogs/cmip6_store_index.json` from the **same crawl**, pinning the
   observed `{grid_label}/{version}` per (entry, member, certified variable).
   Catalog and index must carry an equal `crawled_on`.

**Commit 2b — persistent series** (design §8 row 2b, §5.3 normative)

5. Drop `temp()` from the two stats output families.
6. Implement the §5.3 digest and the `cst_*` series attributes.
7. Implement the revalidating reduce-job entry check, read-time pin
   verification, and the stage-B fingerprint/digest backstop.
8. Replace `os.mkdir(folder_out)` guarded by `os.path.exists` with
   `os.makedirs(folder_out, exist_ok=True)`.
9. Delete the `stats_time_nc_hist` input from `monthly_stats_fut` (the
   ordering-only edge).
10. Fix the acquisition-window contract per §5.3 (fixed span per experiment
    class; `future_horizons` excluded from the digest).

### Validation

Report each rung, per commit.

1. **Narrow** — `pytest tests/test_cli.py` (dry-runs all three Snakefiles).
2. **New behavioural tests** — the identical-input-set test (item 3); design §9
   cache tests **(a)–(k)**, including catalog-regeneration invariance, pin
   re-derivation, and the D9 revalidation cases; the series-schema test.
3. **DAG diff** — `snakemake -n` before and after each commit; report the job
   count and rule set against *that commit's* expected set (the design's
   end-state counts describe step 7, not these commits).
4. **Full gate** — `pytest tests/`; green and purely additive.
5. **Baseline** — `pixi run python dev/scripts/check_baseline.py check` against
   `test_case/test_local` after **each** commit, plus `semantic_tree_diff` over
   the WF2 output subtree. **Local gate — CI cannot run it.**
6. **Region re-check** — re-compare the bounds of `store_region.geojson` and
   `hydrology_model/staticgeoms/region.geojson` before relying on
   value-neutrality. If they differ, STOP (Gate 2).
7. **Decoupling proof** — move `hydrology_model/` aside, run WF2 to completion,
   restore it.

### Acceptance criteria

- All three commits pass `check_baseline.py check` with **zero** drift. Any drift
  means the change was not value-neutral → revert and report.
- WF2 completes with `hydrology_model/` absent.
- A second run after a `future_horizons` edit issues no network reads and
  schedules zero reduce jobs.
- A polygon rewritten with different content re-derives the affected series; a
  byte-identical rewrite does not re-download.
- `pytest tests/` green, additive only.
- No edits to the WF1/WF3 `extract_climate_grid` declarations.

**Rollback:** any baseline drift, or any WF1/WF3 DAG change visible in their
dry-runs, reverts the offending commit before continuing.

### Output requirements

- Three commits, scoped by explicit pathspec, in the order above.
- A note in `dev/working/` recording: measured region bounds at validation time,
  before/after job counts, the second-run network-read evidence, and each
  validation rung's outcome.
- **Results delta:** expected **empty**. If any manifested value changes, stop
  and report what changed and why rather than re-recording the baseline.

### Task constraints

- **`climate_store_spec` declarations stay symmetric across all three
  Snakefiles.** Its docstring: *"The input set is exactly one entry — the catalog
  — in both DAGs. An asymmetric input set re-creates the wf1↔wf3 re-extraction
  oscillation."* A third declaration inherits that verbatim.
- `--dry-run` before running and after editing any rule.
- `config/catalogs/cmip6_data.yml` is generated — never hand-edit it.
- Do not commit run outputs under `project_dir`; do not hand-edit `pixi.lock`.
- Follow `dev/conventions/naming.md` for any new identifier.

**Human gates**

- **Gate 1** — if `snake_utils.py` must change, PAUSE. A shared-helper edit
  affects three DAGs.
- **Gate 2** — if the two region polygons' bounds differ at validation time,
  PAUSE. Step 1 becomes value-changing and needs a re-record decision.
- **Gate 3** — after commit 1's baseline check, PAUSE and report before starting
  2a.
