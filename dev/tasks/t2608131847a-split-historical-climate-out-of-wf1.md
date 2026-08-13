---
title: Split the historical-climate workflow out of WF1, with a forcing-selection evaluation layer
type: todo-item
status: backlog
effort: 2
area: wf1 / workflow split
origin: fao branch assessment (2026-08-13)
queue:
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — Carve a `Snakefile_climate_historical` out of `Snakefile_model_creation`, and give it the evaluation layer that makes it useful: run the model under several forcing datasets, compare each against station observations, and screen them in Budyko space.
> **Why** — The direction the owner asked for, and the capability behind it is the one an uncalibrated rapid assessment most lacks — *which forcing dataset should this basin use?* CST does no local calibration, so forcing choice is the dominant lever on the historical run and nothing currently supports the decision.
> **Effort** — Large. Needs a design pass before implementation: this is a workflow boundary and a new config surface, not a refactor.

## Do not scope this as a rewrite — WF1 is already separable

This is the finding that makes the item affordable, and it should be confirmed
before anything else is designed:

- Rule **1.04 `extract_historical_climate` is already the shared store
  producer** — WF1 and WF3 both consume it.
- Rule **1.05 `plot_climate_source`**'s subgraph builds with **neither**
  `models/hydrology/wflow/` **nor** `config/defaults/wflow_build_model.yml` on
  disk. Its docstring asserts this and `tests/test_plot_climate_source.py` pins
  it.

So the climate arm of WF1 is model-independent by construction. The split is a
Snakefile partition plus a new evaluation layer. The upstream `fao` branch's
`Snakefile_climate_historical` is a **target shape, not a migration path** — its
code is on hydromt 0.9/hydromt_wflow 0.6 and does not port.

The `dev/roadmap.md` "climate analysis subworkflow" entry is the same direction.
Its recorded tension against ADR 0002 is closed — 0002 was superseded by 0006 on
2026-08-09 — so nothing in that entry argues against this.

## The three pieces are one capability, not three

Assessment §5.1. Designing them separately produces three half-features.

1. **Multi-forcing historical runs.** `fao` uses a `forcing_options:` map — one
   entry per source, each naming `precip_fn` / `temp_pet_fn` / `pet_fn` /
   `dem_forcing_fn` / `pet_method` / a plot `color` — and runs the model once per
   entry. Mixing precipitation from one source with temperature from another is
   the case that matters (CHIRPS precip + ERA5 temperature), and it is exactly
   what `hydromt_wflow`'s `setup_precip_forcing` / `setup_temp_pet_forcing` split
   already supports.
2. **Station-observation climate evaluation.** A `climate_locations` +
   `climate_locations_timeseries` surface main does not have at all: sample each
   source at station points and over subregions, then compare against observed.
   `fao`'s `sample_climate_historical.py`, `plot_climate_location.py`,
   `plot_climate_basin.py` port (xarray/pandas, no model API).
3. **Budyko screening.** Runoff coefficient against aridity index per forcing
   dataset. Verified absent from main (`budyko`, `aridity` return nothing across
   `blueearth_cst/`, `config/`, `tests/`). `fao`'s `plot_budyko.py` is idea-only
   — it imports `hydromt.flw` and the v0 `WflowModel` — but the method is short.

Together they answer the forcing question with three independent lines of
evidence: precipitation volume against gauges, hydrograph fit, and long-term
water-balance plausibility. `fao`'s Piave example carries exactly that reasoning
across two notebooks and lands on "ERA5 overestimates volume but captures the
dynamics" — the shape of conclusion this layer exists to support.

## Cheap follow-ons, not separate items

Assessment harvest #4 and #6. Both are real gaps; neither justifies its own note
until this one has a home for them.

- **SPI, dry-day counts, heat-day counts, frost days.** Verified absent (`spi`
  matches only `spines`). `fao`'s `plot_scalar_climate.py` ports.
- **MODIS snow-cover validation** against modelled snow. Verified absent
  (`modis` in main is LAI only). Idea-only; needs a gridded output path.

## Out of scope — ruled

The `fao` branch's `Snakefile_future_hydrology_delta_change` is **not adopted**
(assessment §3, owner ruling 2026-08-13). CST stays strictly bottom-up. Do not
let the gridded change factors, the delta-change Julia driver, or `save_grids`
re-enter through this item — `blueearth_cst/projections/gridded_outputs.py`
rejects `save_grids: true` and that rejection stands.

## Progress

- [ ] Design pass. Workflow boundary, the `forcing_options` config surface, what
      1.04/1.05 keep vs. what moves, and rule numbering for the new workflow.
- [ ] Confirm the separability claim empirically — dry-run the carved workflow
      with no wflow model on disk, not just read the docstring.
- [ ] Split the Snakefile; keep `extract_historical_climate` shared.
- [ ] Multi-forcing runs (`forcing_options:`).
- [ ] Station/subregion sampling + observation comparison.
- [ ] Budyko screening.
- [ ] `scripts/run_workflows.py` ordering + `workflows.<name>.enabled` for the
      new workflow; its contract is pinned clause-by-clause by
      `tests/test_run_workflows.py`.
- [ ] Follow-ons: SPI/dry-day/heat-day indices; MODIS snow cover.

## Before starting

- **`lane/pipeline`.** Check `.lane-claim` first.
- **Grep the test suite for the old roots before moving anything.** `AGENTS.md`
  is explicit: the fixture-dependent layer cannot fail in CI or any worktree, so
  a stale path there survives every gate a branch can run — R9 left 22 such
  failures, three behind an `os.path.exists` guard that turned a wrong path into
  a silent skip.
- A new workflow means a new `--configfile` seed under `test_case/`. **Keep the
  `snake_config_` prefix** or it is silently untracked.
- `pixi run test-full` at the merge, not just at the push: this touches a
  Snakefile and `shared/`, which is the case that tier guards.

## Refs

- `dev/reviews/2026-08-13_fao-branch-assessment.md` §2.1, §5.1, §5.3, and the
  §3 ruling that bounds this.
- `upstream/fao:snakemake/Snakefile_climate_historical.smk`,
  `Snakefile_historical_hydrology.smk` — shape only.
- `dev/roadmap.md` — "climate analysis subworkflow"; `dev/decisions/0006`.
- Related: t2608131847 — **done** (closed 2026-08-13, `ca4c9df` + `1958747`).
  The three notebooks are the pattern a fourth follows: intro numbering what
  the Snakefile does, the shipped rapid config printed and narrated block by
  block, input-schema cells, a rule-by-rule table naming the config keys that
  tune each rule, then results read rather than displayed. **The fourth
  notebook is this item's obligation** — it was the one Progress line
  t2608131847 could not discharge, and it is deferred here rather than left
  open there. Keep the `rendered against <sha>` banner and the ordering links,
  and add the new notebook to `docs/notebooks/README.md`; the re-render
  counterweight is
  [[t2608132100-re-render-the-workflow-notebooks-when-their-banner-sha-falls-behind]].
