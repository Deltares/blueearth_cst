---
title: Split the historical-climate workflow out of WF1, with a forcing-selection evaluation layer
type: todo-item
status: active
effort: 2
area: wf1 / workflow split + workflow rename
origin: fao branch assessment (2026-08-13)
queue:
created: 2026-08-13
updated: 2026-08-14
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

## Scope grew on 2026-08-14 — this now carries a rename migration

The design pass took three owner rulings, and one of them widened the item.
**All four workflow entry points are renamed** to verb-first `.smk` files, with
the `workflows.<name>` config keys and every derived path following:

| was | becomes |
|---|---|
| *(new)* | `analyze_climate.smk` — wf**0**, sorts and runs first |
| `Snakefile_model_creation` | `build_model.smk` |
| `Snakefile_climate_projections` | `analyze_projections.smk` |
| `Snakefile_climate_experiment` | `run_stress_test.smk` |

It landed here rather than as its own note because adding a fourth workflow
already forces edits to `run_workflows.py`'s `WORKFLOW_ORDER` and its clause
(a)/(b) contract, `test_cli.py`, `plot_workflow_dag.py`'s digit map, the shared
producer symmetry tests, `check_baseline` and the config template — every one of
which the rename also touches. Two passes would pay that twice.

The other two rulings: the carve is **climate-only and model-free** (multi-forcing
model runs deferred to a follow-on), and the existing rule digits 1/2/3 stay put
— the new workflow takes **0**, so there is no renumber.

## Progress

- [x] Design pass — `dev/working/2026-08-14_climate-workflow-split/design.md`
      (2026-08-14). Workflow boundary, the `candidate_sources` config surface,
      rule numbering, the rename migration, a 10-commit plan in two landings, and
      six open items for the owner.
- [x] Confirm the separability claim empirically — **done, and it corrected the
      assessment.** A dry-run against an empty `project_dir`, with the build
      templates pointed at paths that do not exist, schedules exactly **four**
      jobs: 1.02 `delineate_region`, 1.03 `delineate_spatial_units`, 1.04
      `extract_historical_climate`, 1.05 `plot_climate_source`. Nothing
      model-side. Two corrections that changed the design: the subgraph needs
      1.02 and 1.03 as well — both shared producer contracts, so three symmetry
      tests go four-way — and rule 1.13 declares 1.05's `climate_levels.json` as
      a real input. Do not re-run this.
- [ ] Landing A — the rename. **Must run in the PRIMARY checkout**: the fixture
      trees carry the renamed paths as data, and the fixture-dependent layer
      skips rather than fails in a worktree, so a lane gate would prove nothing.
- [ ] Split the Snakefile — **additive, not a subtraction**. WF1 keeps 1.02–1.05
      exactly as they are; the new workflow declares the same shared rules
      generated per candidate source. Design §5.2 / §5.4.
- [ ] Multi-forcing model runs — **deferred out of this item** by the 2026-08-14
      ruling. Raise as its own note at Landing B's closure.
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
- ~~A new workflow means a new `--configfile` seed under `test_case/`.~~
  **Mostly wrong, corrected by the design (§9 O-5):** the fourth workflow is a
  fourth `workflows:` subsection in the existing seeds, not a new file. A new
  seed is warranted only for the evaluation-layer exercise at Landing B's last
  commit — and there the caution stands: **keep the `snake_config_` prefix** or
  it is silently untracked.
- **Landing A belongs in the primary checkout, Landing B in `lane/pipeline`,**
  and the docs half of both in `lane/devmeta` (design §6). The rename also
  invalidates `lane/pipeline`'s own claim glob `Snakefile_*`, which becomes
  `*.smk` in the same commit as the renames (§9 O-3b).
- `pixi run test-full` at the merge, not just at the push: this touches a
  Snakefile and `shared/`, which is the case that tier guards.

## Refs

- **`dev/working/2026-08-14_climate-workflow-split/design.md` — the design, and
  the source of record for everything above.** Read §5.2 (why the carve is
  additive), §5.4 (how N candidate sources are declared) and §9 (six open items
  needing an owner ruling) before starting either landing.
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
