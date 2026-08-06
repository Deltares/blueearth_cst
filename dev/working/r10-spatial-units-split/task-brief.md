# Task Brief — split `prepare_spatial_maps` so WF2/WF3 can consume basin boundaries

### Context

Canonical rules: `AGENTS.md`. Design: `dev/decisions/0003-one-shared-region-artifact.md`
§8–10 (**proposed**), tracked as `dev/followups.md` `[R10-6]`. Read §8, §8a, §8b
and validation items 1–5 and 7 before starting — they carry decisions this brief
only summarises.

- WF2 and WF3 declare `delineate_region` and **no other** spatial rule today, and
  neither workflow's scripts read a vector layer. This makes boundaries
  reachable; *using* them is a separate task (§10 leaves the consuming rules
  deliberately unnamed).
- All three gates are closed. §8a: the seam carries the hydrography grid, so the
  vector rule writes it as an intermediate. §8b: the vector rule's inputs/params
  must be a pure function of `project` + `shared.basin`. Validation 7: measured
  2026-08-06 — the split avoids ~71% of the incremental cost.
- **`delineate_region` is the pattern to copy**, not to reinvent:
  `snake_utils.region_spec` + three byte-identical declarations, enforced by
  `tests/test_region_spec.py`.
- Owner ruled WF2/WF3 need **no DEM or raster layer**, only boundaries.
- §11 and §12 of the ADR are **out of scope here** — they move outputs and land
  separately.

### Goal

`delineate_spatial_units` produces the vector layers + registry and is declared
identically in all three workflows; `prepare_spatial_maps` keeps
`spatial_maps.nc` and the thematic stack and stays WF1-only. Outputs
byte-identical to today.

### Non-goals

- No WF2/WF3 rule *consumes* the vectors yet (§10).
- No `max_per_basin` change (§11), no `wflow_id` renumbering (§12).
- No renaming or renumbering — R10 and `[R10-5]` land later, against this rule set.
- No WF0 preparation workflow (deferred; tripwire in the ADR).

### Allowed scope

**Permitted**
- `blueearth_cst/spatial/products.py`, `blueearth_cst/spatial/prepare_spatial_maps.py`
- new script module for the vector rule under `blueearth_cst/spatial/`
- `blueearth_cst/shared/snake_utils.py` — add `spatial_units_rule` beside `region_spec`
- `Snakefile_model_creation`, `Snakefile_climate_projections`, `Snakefile_climate_experiment`
- `tests/` — new `tests/test_spatial_units_rule.py`, extend existing spatial tests
- `dev/scripts/semantic_tree_diff.py` and `tests/test_r09_path_map.py` — the seam
  intermediate needs a path-map row

**Approval-gated** — pause and ask
- Any change to what `spatial_catalog.yml` contains or which rule writes it. The
  ADR rules it stays whole in the raster half; deviating needs a ruling.
- Any change to the six existing vector artifacts' schemas or paths.

**Forbidden**
- `dev/milestones/` (archive), `pixi.lock`, `Manifest.toml`, anything under a
  `project_dir`, and any vendored package under `.pixi/`.

### Required changes (checklist)

1. Split `prepare_spatial_products` into a vector half (through
   `_delineate_spatial_units`, plus the `rivers` catalog read) and a raster half
   (`_thematic_maps` onward, plus `spatial_catalog.yml` / `spatial_report.yml`).
2. Write the hydrography grid stack (`flow_direction`, `flow_accumulation`,
   `upstream_area`, `river_mask`, `basin_id`, `subbasin_id`) as the seam
   intermediate; the raster rule declares it as an input. It is a seam
   intermediate, **not** a product — keep it out of `spatial_catalog.yml`.
3. Add `snake_utils.spatial_units_rule(...)` returning a `SpatialUnitsRule`
   (`script`, `inputs`, `outputs`, `params`), mirroring `region_spec`.
4. Declare `delineate_spatial_units` in all three Snakefiles, splatted from that
   helper, differing only in `message` / `log` / `benchmark`.
5. Drop `config_snake` from the vector rule's inputs and narrow its params to
   `SpatialConfig` fields resolved from `shared.basin` alone (§8b). The
   deprecated `workflows.model_creation.output_locations` fallback cannot feed it
   — state that in the rule comment.
6. Add each workflow's new label to its `LOG_RULES`, in the position matching the
   existing order.
7. Add the seam intermediate to the R9 path map and `semantic_tree_diff.py`.

### Commit plan

Staged landing — each commit must leave the tree runnable.

| # | Subject | Paths | Invariant preserved |
|---|---|---|---|
| 1 | split `products.py` into two halves + seam intermediate, WF1 only | `spatial/`, `Snakefile_model_creation`, `tests/` | WF1's six vector artifacts and `spatial_maps.nc` stay byte-identical |
| 2 | add `spatial_units_rule` and declare it in WF2 + WF3 | `snake_utils.py`, two Snakefiles, `tests/test_spatial_units_rule.py` | the three declarations are byte-identical; WF2/WF3 gain no thematic read |
| 3 | path-map + tree-check rows for the seam intermediate | `dev/scripts/`, `tests/test_r09_path_map.py` | `pixi run tree-check` clean |

Commit 1 must not touch WF2/WF3; commit 2 must not change what WF1 produces.

### Validation

Ladder, with frequency:

1. **Narrow** (per edit) — `pytest tests/test_spatial_products.py
   tests/test_prepare_spatial_maps_rule.py tests/test_spatial_identity.py
   tests/test_spatial_delineation.py`. These run on synthetic fixtures and need
   no untracked tree — 29 passed at time of writing.
2. **New behavioural** (per commit) — `pytest tests/test_spatial_units_rule.py`,
   mirroring `tests/test_region_spec.py`: the helper's shape, and the three
   declarations differing only in `message`/`log`/`benchmark`.
3. **Contract** (per commit) — `pytest tests/test_log_rules_contract.py`. Fails
   if step 6 is missed; that is what it is for.
4. **Full gate** (before merge) — `pytest tests/test_cli.py`, then `pytest tests/`.
5. **Baseline** (once, before merge, primary checkout) —
   `python dev/scripts/check_baseline.py check`. §8–10 are behaviour-preserving,
   so this **must pass unchanged**. A diff here means the split changed
   behaviour and is a revert trigger, not a re-record.

**Falsifier for the property this task exists to buy.** §8 claims WF2 no longer
reads the thematic sources — an *absence*, which no unit test reaches. Run:

```
snakemake -n -s Snakefile_climate_projections --configfile tests/snake_config_model_test.yml
```

Expected: `delineate_spatial_units` appears; **no job's inputs mention `vito`,
`modis_lai` or `soilgrids`**. If any does, the split did not achieve its purpose
and is indistinguishable from the rejected unsplit alternative. Report the
scheduled-job list, not a pass/fail.

Report what each rung *caught*, not only that it passed.

### Acceptance criteria

- WF1's six vector artifacts, `spatial_maps.nc`, `spatial_catalog.yml` and
  `spatial_report.yml` are byte-identical to pre-split.
- The three `delineate_spatial_units` declarations differ only in
  `message`/`log`/`benchmark`, enforced by a test.
- The WF2 dry-run falsifier above passes.
- `check_baseline.py check` passes unchanged.
- **Rollback trigger:** any baseline diff, or a WF2 dry-run still scheduling a
  thematic read.

### Output requirements

- The commits above, each independently runnable.
- A short note appended to ADR 0003 recording the landed state, and §8–10's
  status moved from `proposed` toward `accepted`.
- Update `[R10-6]` and the landing-order table in `dev/followups.md`.
- **Results delta:** none expected. If any output moves, stop — that contradicts
  the design and is a rollback trigger, not a finding to write up.

### Task constraints

- Do not re-engineer how hydromt handles data (`AGENTS.md` hard constraint). The
  vector half must call the same hydromt APIs it calls today.
- Run the pipeline from the **primary checkout**, never a worktree. Dry-runs and
  `pytest` are fine in a worktree; `check_baseline.py check` is not.
- `AGENTS.md` validation ladder governs: unit tests while iterating, broader
  checks once at the commit.

**Human gates**

1. **After commit 1**, PAUSE. The seam intermediate is a new declared artifact in
   a tree R9 just settled — the owner confirms its path and that it stays out of
   `spatial_catalog.yml` before it reaches WF2/WF3.
2. **Before the baseline run**, PAUSE. It needs the primary checkout and is the
   gate that decides whether §8–10 are behaviour-preserving as claimed.
3. **If the WF2 falsifier fails**, STOP and report. Do not repair by widening
   scope — the design's premise is what failed.
