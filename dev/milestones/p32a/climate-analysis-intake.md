# P3-2a — Model-independent climate analysis — scoping intake

**Status.** AUTHORITATIVE scoping record, confirmed with the user 2026-07-24
via `design-scoping` (four decisions elicited and individually confirmed).
The design-review-loop run for this milestone starts from this file; the
"Confirmed scoping decisions" below are fixed anchors for that run, not
reopened by it. Loop start is **user-gated** (explicitly deferred at intake
confirmation — do not start the run without the user's go).

**Provenance.** Split out of roadmap § "P3-2 — Functional decomposition +
model-swap interfaces (not scoped)". P3-2's other half — the interchange
contracts (model-swap interfaces) — is **P3-2b**, a separate later scoping +
design cycle. Grounding: `dev/milestones/r06/structural-refactor-design.md` § "The
functional-decomposition scope fork" (DEFER position + forward-compatibility
inventory), the `modularization-climate-analysis-subworkflow` direction, and
ADR 0002 (`dev/decisions/0002-revive-subcatchment-climate-plots.md`).

## Confirmed scoping decisions (fixed anchors)

1. **P3-2 split.** P3-2a (this milestone: functional decomposition of
   climate analysis) proceeds first; P3-2b (model-swap interchange
   contracts) is deferred to its own cycle. Rationale: P3-2a settles where
   climate-analysis code lives before P3-2b pins contracts around that
   layout; the two halves touch different code and carry different risk
   classes (value-changing re-source vs contract pinning).
2. **Full re-source + lift.** The milestone delivers BOTH the structural
   lift (a `blueearth_cst/climate_analysis/` subpackage with strictly
   model-independent signatures) AND the data-path change: the wf1
   subcatchment climate plots re-source from **raw gridded climate**
   (region + catalog + window) instead of the built model's forcing
   (`mod.forcing.data`), unwinding the ADR-0002 coupling. Rejected
   alternatives: structural-lift-only (leaves "model-independent"
   aspirational); dual-source retention (more surface, no decoupling win).
3. **Subpackage now, entry point deferred.** No 4th Snakefile this
   milestone: the three entry points, `run_workflows.py` wrapper contract,
   config `workflows:` sections, and the GUI-facing platform surface are
   unchanged. A standalone climate-screening entry point becomes a cheap
   later addition once the subpackage exists. Rejected alternatives: 4th
   Snakefile now (platform-contract change stacked on a value-changing
   milestone); entry via an existing workflow's target subset (couples the
   capability to a config it does not need).
4. **Acceptance gate for the plot value change: visual QA + characterized
   diff.** Side-by-side old/new plots on the test fixture (and gabon when
   available) for user sign-off at the milestone gate, PLUS a quantitative
   characterization (e.g. per-subcatchment monthly mean deltas — expected
   regridding-scale) documented in the milestone's baseline record; the
   manifest then re-records the changed targets. Rejected: visual-only (no
   numeric record); a-priori threshold gate (arbitrary number risk).

## Repo-grounded inventory (for the design to verify, not re-litigate)

- **Lift candidates (R6 §8 "mechanical move" pair):**
  `blueearth_cst/experiment/extract_historical_climate.py` (backs
  `extract_climate_grid`; depends only on region.geojson + catalog) and
  `blueearth_cst/projections/prepare_climate_data_catalog.py`. Plotting
  primitives (`func_plot_signature.py` / `plot_clim`, `plot_map.py`) are
  already in `shared/` — zero-move.
- **Re-source site:** wf1 `plot_results.py` §4 + `climate_forcing.py`
  (ADR 0002 landed the `mod.forcing.data` coupling on `test/pre06`).
- **Already model-independent today:** `extract_climate_grid` (rule 3.02,
  `ancient(region.geojson)` input) and the wf2 `monthly_stats_*` rules.
- **P3-1 interaction:** the wf3 extraction now lives in the dataset+window
  keyed store (`climate_historical/<key>/`) with the key-level guard
  artifact (`.guard_ok`) — the lift must preserve the P3-1 keying + guard
  wiring exactly (`dev/milestones/p31/experiment-structure-design.md` §3d/§4).

## Constraints (standing; not new to this milestone)

- CST automation scope: hydromt/hydromt_wflow/wflow conventions verbatim; no
  upstream re-engineering (AGENTS.md Hard Constraints).
- No new dependencies without prior user approval.
- `blueearth_cst/weathergen/*.R` content untouched.
- Behavior discipline: everything EXCEPT the named wf1 plot re-source stays
  value-identical (semantic-tree-diff + manifest discipline, R3/R5 style);
  the re-source is the milestone's single sanctioned value change, accepted
  per decision 4.
- Naming per `dev/reference/naming.md`; commit prefix `p32a:` (registered
  in the roadmap prefix list at scoping).

## Success criteria

1. No function in `climate_analysis/` imports or receives the built Wflow
   model; signatures take region/catalog/window (model-independence is
   checkable, not aspirational).
2. wf1 subcatchment climate plots sourced from raw gridded climate; visual
   QA sign-off + characterized diff documented; changed targets re-recorded.
3. wf2/wf3 consume the lifted modules via mechanical rewires; the semantic
   tree diff is clean modulo the documented plot changes.
4. Full default suite green; three clean `--dry-run`s; full e2e via
   `run_workflows.py`; milestone branch + tag per the branch model.

## Cut (YAGNI) / deferred

- 4th Snakefile entry point — deferred (decision 3).
- P3-2b interchange contracts — separate cycle.
- Realization/stress-test file-format redesign — parked (P3-3 candidate).
- New plot types or analysis products — none; re-source and lift only.

## Handoff

Next step (user-gated): `design-review-loop` run `p32a-climate-analysis`
with this intake as scope authority, then task-brief → implementation under
the Opus-default / Fable-gate-review pattern.
