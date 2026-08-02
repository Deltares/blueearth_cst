# Changelog

Personal fork of `Deltares/blueearth_cst`. Release history follows
[Keep a Changelog](https://keepachangelog.com/) loosely.

Milestone-level detail (Phase 1 sealed M01–M02c, Phase 2 active R1–R6)
lives in `dev/roadmap.md` and `dev/milestones/phase-1/<milestone>/` /
`dev/milestones/r0N/<milestone>/` artifacts. This file captures release-level
summaries.

## [Unreleased] — workflow 2 v2.0 (milestone R8)

Workflow 2 restructured from a model/scenario fan-out into a monthly GCM
projections analysis. **Breaking config changes** — see
`docs/migration-r08-wf2.md`.

### Breaking

- `save_grids` renamed to `save_gridded`; the old key raises rather than being
  silently ignored.
- `variables` is a mapping declaring `canonical` and `change` per variable, not a
  list. Change semantics are no longer inferred from the literal name `"precip"`.
- A model absent from the generated catalog fails at DAG build. A model that does
  not publish a requested scenario or member is a recorded skip, not an error.

### Added

- `change_factors/annual.csv` and `change_factors/monthly.csv` — long-format
  tables, one row per (dataset, scenario, member, horizon, variable, statistic).
  The monthly table is new: it shows the seasonal shift an annual figure averages
  away.
- `summary/composition.csv` — every **requested** combination and how it resolved.
- `provenance.json` — sources with verified physical store paths, digests,
  windows and settings.
- `report.md` — with the disclaimer block: window clipping, alignment, weighting
  scheme and its approximation, the dry-month rule, catalog snapshot date, and
  unresolved combinations by status.
- A persistent series cache with content-addressed identity, and a fetch/reduce
  split so re-deriving from a formula change costs **zero network requests**.
- Optional `stats` and `relative_change` config keys.

### Changed — values move

- Spatial reduction is spherical cell-area weighted (D10); annual aggregates are
  weighted by month length in the model's own calendar; stage A no longer rounds
  to 2 decimals; the default statistic set is `mean, median, std` with tail
  quantiles opt-in and sample-size-labelled.
- **A reference-window off-by-one is fixed**: `[1990, 2010]` now yields 21
  complete hydrological years, not 20. The final complete year was discarded.
- Figures show **one trace per combination** — no multi-model median, no 5–95 %
  envelope. Each `(model, scenario, member)` is one data point.

### Fixed

- Series recorded `proleptic_gregorian` for every model, which is false for
  `noleap` and `360_day` ones. The true calendar is now read from the store.
- `kernel_hash` was not reproducible across processes for any function containing
  a closure, so a cache key moved on every invocation.
- Stage B had no hashed kernel, so an edit to imported change arithmetic left its
  outputs silently stale.

## [v0.2.0-alpha] — 2026-05-09

Foundation phase sealed; Phase 2 (workflow refactor) designed and
ready to execute. Substantial change since `v0.1.0-alpha` — minor
version bump reflects breaking library upgrades plus structural and
testing additions that make Phase 2 possible.

### Phase 1 (Foundation, sealed)

- **M01 — Replication baseline** (`m01-replication`): all three
  Snakemake workflows replicated end-to-end on the test config;
  baseline manifest at `dev/baseline/manifest.json`;
  `dev/scripts/check_baseline.py` with `record` / `check` subcommands.
- **M02 — Pixi env** (`m02-pixi`): pixi-driven Python + R + Julia env
  replacing conda + ad-hoc setup. Single declarative `pixi.toml`.
- **M02b — Library upgrades** (`m02b-upgrades`): hydromt 0.x → 1.3,
  hydromt_wflow 0.x → 1.0.2, Wflow.jl 0.7 → 1.0.2, Python stack caps
  lifted (numpy 2.x, xarray latest). Baseline re-recorded under the
  "intentional drift, document deltas" policy.
- **M02c — Test coverage** (`m02c-tests`): 32 new unit tests across 4
  `src/` modules plus 2 strict xfails for documented bugs (hydromt
  `to_yml` preprocess strip; `extract_climate_grid` silent truncation).
  Established the `sys.modules.setdefault` mocking pattern.

### Phase 2 (Refactor) — designed, not yet executed

- **R1 design** (`dev/milestones/r01/`): per-workflow modularity contracts.
  Sectioned config schema (`project` / `shared` / `workflows.<name>`),
  per-workflow contract docs, atomic migration plan including
  `src/` script updates and baseline-snapshot policy.
- **R2 design** (`dev/milestones/r02/`): prescriptive naming conventions style
  guide (snake_case, lowercase acronyms, `_path` canonical, suffix
  vocabulary split between paths and data objects, domain-identifier
  escape hatch, "do not rename without migration note" surfaces).

### Toolchain

- **Python 3.12 → 3.14.4**. The deprecated `Path(fn, resolve_path=True)`
  kwarg in `src/prepare_climate_data_catalog.py` was fixed (closes
  R3 followup early as prep for the cap bump).
- **Snakemake stays at 9.6.2**. Bump to 9.20 deferred — snakemake 9.7+
  requires `packaging<26` but the conda env pulls in `packaging==26.2`
  via another dep. Likely fixed in snakemake 9.21+.
- `datrie` pin removed (was redundant; modern snakemake doesn't need it
  as a hard conda dep).

### Repo structure

- Roadmap restructured into Phase 1 (sealed) / Phase 2 (active)
  sections with clear visual break.
- `dev/milestones/phase-1/{m01,m02,m02b,m02c}/` for sealed foundation milestones.
- `dev/milestones/r01/`, `dev/milestones/r02/` for active Phase 2 milestones.
- `dev/conventions/` reserved for R2 output.
- Stale forward-looking M3-M6 references renamed to R3-R6 across docs,
  tests, and code comments.
- New tag prefix convention: `r##-<topic>` for Phase 2 (e.g.
  `r03-model-builder`); `m##-<topic>` preserved for sealed Phase 1.

### Testing

- Suite total: 45 passed, 4 xfailed. Unchanged through the entire
  toolchain bump and repo restructure (validates that the moves and
  text updates don't affect runtime behavior).

### Documentation

- README updated to reflect fork status, Python 3.14, Julia 1.11.x via
  juliaup, deferred Docker, and pointers to `dev/install.md` and
  `dev/roadmap.md`.
- New `CHANGELOG.md` (this file).

### Known issues / followups

- Test `xfail`s tracked in `dev/followups.md`:
  - hydromt `to_yml` preprocess strip — upstream hydromt issue.
  - `extract_climate_grid` silent truncation when staged source < requested window — fix in R3.
- Snakemake bump to 9.20 deferred until upstream relaxes the
  `packaging<26` constraint.
- Linux / Docker validation deferred (see "Deferred: Linux replication"
  in `dev/roadmap.md`).

## [v0.1.0-alpha] — base

Upstream starting point. Forked from `Deltares/blueearth_cst` at this
version. Branch `base/v0.1.0-alpha` preserves the exact starting
commit. All Phase 1 work in this fork builds on top of this baseline.
