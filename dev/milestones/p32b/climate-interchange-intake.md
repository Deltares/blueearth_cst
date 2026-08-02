# P3-2b — Model-swap interchange contracts — scoping intake

**Status.** AUTHORITATIVE scoping record, confirmed with the user 2026-07-24
via `design-scoping` (four decisions elicited and individually confirmed).
The design cycle for this milestone starts from this file; the "Confirmed
scoping decisions" below are fixed anchors for that cycle, not reopened by
it. Design-cycle start is **user-gated** — do not start without the user's
go.

**Provenance.** Second half of the former P3-2, split at the P3-2a intake
(`dev/milestones/p32a/climate-analysis-intake.md`, decision 1): P3-2a settled where
climate-analysis code lives (sealed 2026-07-24, tag `p32a-climate-analysis`);
P3-2b now pins the interchange contracts around that settled layout.
Roadmap: `dev/roadmap.md` § P3-2b. Phase-3 driver: model flexibility
(R6-handoff expectations map).

## Confirmed scoping decisions (fixed anchors)

1. **Both seams, contracts-only.** P3-2b pins interchange contracts at BOTH
   substitution seams — the weather-generator seam and the
   hydrological-model seam — but wires **no** alternative implementation.
   A swap remains a future bounded exercise against the pinned contracts.
   Rationale: one risk class (contract pinning, value-identical), matching
   the split logic that separated P3-2a/P3-2b. Rejected alternatives:
   one-seam-only (leaves half the roadmap goal untouched for no risk win);
   contracts + proof-of-concept swap (two risk classes; if a PoC swap is
   wanted later it becomes its own cycle, e.g. P3-2c).
2. **Contract form: docs + validators-as-tests.** Per-seam contract
   documents (dims, coords, variables, units, dtypes, CRS, time
   axis/calendar, file naming, producer→consumer rules) PLUS hand-rolled
   validator functions exercised by pytest against real fixture artifacts.
   No pipeline-runtime enforcement in this milestone (validators stay
   reusable as future in-pipeline guards). Rejected alternatives: docs-only
   ("pinned" stays unverifiable — the aspirational-vs-checkable gap P3-2a
   closed for model-independence); in-pipeline enforcement now (touches
   Snakefile rule bodies, behavior-adjacent); third-party schema libraries
   (new dependency — requires prior approval, and hand-rolled validators
   suffice at this scale).
3. **Pure contracts — no deferred structural absorption.** The
   P3-2a-deferred items stay OUT: OQ-3 project-level climate store, OQ-8
   model-free subcatchment-zone source (+ deferred polygon-zonal
   aggregation), and the standalone climate-screening entry point. All are
   behavior-touching redesigns; they remain recorded candidates for later
   cycles (P3-3 or standalone).
4. **Success criteria confirmed as stated** (see § Success criteria):
   complete seam inventory, validator-per-contract green on fixture
   artifacts, zero behavior change, and a bounded-substitution walkthrough
   per seam. Milestone gate = user review of the contract docs + green
   validator suite.

## Repo-grounded seam inventory (for the design to verify, not re-litigate)

**Weather-generator seam (wf3 upstream; weathergenr today):**

- IN: `climate_historical/<key>/extract_historical.nc` (P3-1 keyed store;
  7 variables on the native catalog grid) + `weathergen_config.yml`
  template + the stress-test parameter grid (rule 3.03
  `climate_stress_parameters`).
- Generator surface: rules 3.04/3.06 (`prepare_weagen_config*`), 3.05
  `generate_weather_realization` (R, weathergenr), 3.07
  `generate_climate_stress_test` (R, impose_climate_change) →
  `realization_<n>/rlz_<n>_cst_<m>.nc` (temp()).
- OUT (consumed downstream): rule 3.09 `downscale_climate_realization` →
  `inmaps_rlz_<n>_cst_<m>.nc` (temp(); wflow forcing shape), with the
  orography/catalog side-channel via rule 3.08 `climate_data_catalog`
  (`prepare_clim_data_catalog`, now in `climate_analysis/`).

**Hydrological-model seam (Wflow today):**

- Build artifacts (wf1): `staticmaps.nc` (CSDMS names per
  `hydromt_wflow/naming.py`), `staticgeoms/*`, `wflow_sbm.toml`.
- Forcing: `inmaps_historical.nc` (wf1 rule 1.08) and
  `inmaps_rlz_*_cst_*.nc` (wf3) — precip/temp/pet on the model grid.
- Warm states: `instates`/`outstates_rlz_*_cst_*.nc` (wf3 per-cst run
  chaining; wflow `[state]` block).
- Results: `run_default/output.csv` (Q_outlets, gauges, `*_basavg`) →
  export reduction (rule 3.11) → `Qstats.csv`/`basin.csv`.

**Notes for the design (mechanism-level, not scope):** several seam
artifacts are `temp()`-deleted on completed runs — the design must pick the
validator data source (persisted fixture artifacts vs a `--notemp` capture
vs re-derivation); the warm-state contract depth must respect the CST
automation scope (the states schema is Wflow's — pin what OUR pipeline
relies on, not upstream internals); R-side weathergenr config/arg surfaces
are positional-args + YAML (P3-1/R5 conventions).

## Constraints (standing; not new to this milestone)

- CST automation scope: hydromt/hydromt_wflow/wflow conventions consumed
  verbatim; contracts describe OUR reliance on upstream formats, never
  re-specify or patch upstream (AGENTS.md Hard Constraints).
- No new dependencies without prior user approval (hand-rolled validators).
- `blueearth_cst/weathergen/*.R` content untouched.
- **Zero behavior change**: no Snakefile/pipeline edits, no output changes,
  no manifest re-record. (Contracts-only means the per-commit gates are
  cheap: suite + dry-runs.)
- Naming per `dev/reference/naming.md`; commit prefix `p32b:`
  (registered in the roadmap prefix list at scoping); tag
  `p32b-interchange-contracts`.

## Success criteria

1. **Complete seam inventory** — every interchange artifact at both seams
   has a contract doc: dims, coords, variables, units, dtypes, CRS, time
   axis/calendar, naming pattern, producer→consumer rules.
2. **Validator per contract** — each contract has a hand-rolled validator
   function, green under pytest against real fixture artifacts.
3. **Zero behavior change** — no Snakefile/pipeline edits; full suite +
   three dry-runs green; nothing re-recorded.
4. **Bounded-substitution walkthrough per seam** — each seam's contract doc
   names exactly which repo files a replacement weather generator /
   hydrological model would touch: the roadmap's "bounded substitution"
   made concrete and reviewable.

## Cut (YAGNI) / deferred

- Alternative generator/model implementation (PoC swap) — future cycle if
  wanted (P3-2c candidate).
- In-pipeline contract enforcement — future; validators are written to be
  liftable into guards.
- OQ-3 project-level climate store; OQ-8 model-free zone source +
  polygon-zonal aggregation; standalone climate-screening entry point — all
  recorded, all out (decision 3).

## Handoff

Next step (user-gated): design cycle for `p32b-interchange-contracts` with
this intake as scope authority — the design decides the contract-doc
format, validator placement, and the temp()-artifact validation mechanism —
then task-brief → implementation.
