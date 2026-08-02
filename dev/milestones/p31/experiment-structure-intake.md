# P3-1 — Project/experiment structure — design-run intake

**Status.** Scoping CONFIRMED 2026-07-23 (design-scoping session with the user;
all positions below explicitly confirmed, not inferred). This intake seeds a
`design-review-loop` run that produces the accepted design; implementation is a
later `task-brief` handoff. First milestone of Phase 3 (`dev/roadmap.md`
§ Phase 3).

## Problem

One `project_dir` = one basin project, but the experiment dimension is only
half-built: `workflows.climate_experiment.experiment_name` namespaces the wf3
realization/results tree (`climate_{experiment}/`) and the Wflow run dirs
(`hydrology_model/run_climate_{experiment}/`), while `config/` snapshots,
`logs/`, `benchmarks/`, `plots/`, and `climate_historical/` are project-global
and collide/overwrite across experiments. There is no self-describing record of
what settings produced an experiment, and the experiment's Wflow runs pollute
the wf1 model artifact directory. Users need multiple stress-test experiments
per basin project, trackable and reproducible, without re-building the model.

## Confirmed scoping decisions

1. **Experiment scope.** An experiment may vary: all wf3 stress-test settings
   (perturbation grid, `realizations_num`, `horizontime_climate`, `run_length`,
   `run_historical`, `Tlow`/`Tpeak`, `aggregate_rlz`) **plus** the climate
   window/source (`shared.historical_window`, `shared.clim_historical`).
   Project-level and shared by all experiments: the built model (wf1), the
   projections overlay (wf2), `shared.basin`, catalogs. Consequence: the wf1
   `run_default` historical run keeps the *project's* forcing window; an
   experiment overriding the window affects only its own wf3 extraction —
   documented divergence, not guarded.
2. **Tracking level: self-describing tree + provenance only.** Every experiment
   owns its config snapshot, logs, benchmarks, plots; the snapshot suffices to
   reproduce it. CUT (YAGNI, layer-on-later): machine-readable registry, CLI
   listing/status, cross-experiment comparison outputs.
3. **Approach A: one full config file per experiment + drift guard.** Config
   mechanics unchanged (single `--configfile`, `workflow.configfiles[0]`
   forwarding, `get_config` contract intact); `experiment_name` selects the
   subtree. New wf3 startup **drift guard**: the experiment config's `project`,
   `shared.basin`, `workflows.model_creation`, `workflows.climate_projections`
   sections must match the project-level snapshot (normalize-then-compare,
   reuse the R6 comparator pattern) — fail loud naming the diverging key; fail
   clearly when wf1 has never run (no snapshot). NOT guarded:
   `shared.historical_window`, `shared.clim_historical` (experiment-variable).
   Rejected: layered project+experiment configs (breaks the configfiles[0]
   contract to R; revisit only if duplication proves painful).
4. **Constraints confirmed.** (a) Baseline: handled as a documented,
   value-identical re-record at the milestone gate (paths move, values must
   not) — R3/R5 style, not R6's run-relative-only gate. (b) The current
   `project_dir` layout is NOT an external contract on this fork (upstream
   CST-API/GUI compatibility is not preserved); `MIGRATION.md` documents the
   old→new output-path map for any future upstreaming.

## Target tree (confirmed shape; minor names open)

```
<project_dir>/                  # one basin, outside the repo (R6 convention)
  config/                      # project-level snapshots (wf1 + wf2 provenance)
  hydrology_model/             # wf1 built model + run_default — pure wf1 artifact
  climate_projections/         # wf2 overlay, project-level, shared
  climate_historical/          # PROJECT-LEVEL dataset store (user amendment):
    <dataset>/                 #   e.g. era5/, chirps/, cru_ts/ — extracted once,
                               #   REFERENCED by experiments, never copied per-experiment
  logs/  benchmarks/  plots/   # wf1 + wf2 scoped only
  experiments/<name>/          # one dir per experiment (name = experiment_name)
    config/                    # this experiment's full config snapshot (provenance)
    realization_*/  stress_test/   # kept AS-IS structurally for now (see Deferred)
    model_runs/                # Wflow tomls+outputs (moved out of hydrology_model/;
    model_results/             #   model-neutral name — P3-2 forward-compat)
    plots/  logs/  benchmarks/
```

**User amendments folded in (2026-07-23):**

- `climate_historical/` is **project-level shared storage**, organized by
  dataset (ERA5, CHIRPS, CRU-TS, …). Experiments reference a stored dataset
  rather than owning a copy — one dataset serving N experiments is stored once.
  The experiment snapshot records which dataset it referenced.
- `realization_*/` and `stress_test/` internal file shapes/formats are
  acknowledged legacy ("artifacts from bad coding practice") but are **kept
  structurally as-is in P3-1**; efficiency redesign deferred (see Deferred).

## Open design questions for the loop

- **Dataset keying vs the window.** The historical extraction depends on both
  source dataset and `historical_window`. Same dataset + different windows:
  key the store by dataset+window (e.g. `era5_2000-2005/`), store full-period
  per dataset and subset downstream, or forbid window-divergent reuse? Decide
  with the reuse/staleness semantics (when does wf3 re-extract vs reuse?).
- **Run-dir move mechanics.** Moving Wflow runs to `experiments/<name>/model_runs/`
  changes toml-relative input paths (staticmaps, forcing, state) — enumerate
  the `wflow_sbm.toml` path fields affected (`dir_input`/absolute-vs-relative)
  per the wflow skill/docs before committing to the tree.
- **Guard mechanics.** Where the guard runs (rule vs Snakefile parse time),
  snapshot freshness (wf1 re-run after edits), and the exact normalized-compare
  key set.
- **Naming.** `model_runs/` vs alternatives; `experiments/` vs `runs/`; minor
  renames explicitly deferred by the user to the design/review discussion.
- **Baseline re-record procedure** consistent with the mixed-provenance
  manifest state (`baseline-manifest-coverage` memory).

## Non-goals

- No registry, no CLI listing/status, no cross-experiment comparison outputs.
- No layered/merged config mechanism.
- No change to wf1/wf2 computational paths; no model-swap abstraction beyond
  model-neutral directory names (P3-2 territory).
- No `realization_*`/`stress_test` file-format or data-structure redesign
  (deferred, below).
- No upstream CST-API/GUI layout compatibility work.

## Deferred (recorded, not lost)

- **Realization/stress-test data-structure efficiency** — file shapes/formats
  under `realization_*/` and `stress_test/` need a dedicated rethink (candidate
  P3-3 or its own item); user explicitly parked it 2026-07-23.
- Registry / CLI / comparison tooling — layer on the settled tree if needed.
- Layered configs — revisit only on demonstrated duplication pain.

## Success / exit criteria

- Two experiments coexist in one `project_dir` with zero file collisions
  across all seven collision classes (config snapshot, logs, benchmarks,
  plots, historical-climate reference, run dirs, results).
- A shared `climate_historical/<dataset>/` extraction is reused (not copied)
  by a second experiment referencing the same dataset.
- Each experiment dir is reproducible from its own config snapshot.
- Drift guard demonstrably fails loud on a mutated project-level section and
  on a missing project snapshot.
- `pytest tests/` green; full e2e green; baseline re-record value-identical
  modulo the documented path map; `MIGRATION.md` updated.

## Handoff

`design-review-loop` run `p31-experiment-structure` (this intake + the
scoping-confirmed positions are the G1 anchors) → accepted design lands at
`dev/milestones/p31/experiment-structure-design.md` (house pattern) → `task-brief` →
implementation per the Opus-default/Fable-gate-review pattern that ran R6.
