# DEVLOG: feat/wf1-spatial-decoupling

**Timeline:**
- 2026-08-02 01:11 → Started

## Objective
Implement the master brief that splits Workflow 1 into an engine-neutral spatial
foundation and a Wflow-SBM build that consumes it.

## Pass Criteria
Phase 1 produces the documented spatial contract independently of Wflow, the
spatial-to-Wflow adapter proof passes Gate 1, Phase 2 consumes those products,
and all brief-specified tests, dry-runs, integration checks, and human gates are
completed before landing.

## Sessions

### 2026-08-02
**Done:**
- Created the canonical isolated worktree and feature branch from `main`.
- Confirmed the unmerged `feat/wf1-improvements` prerequisite overlaps the same
  Workflow 1 files and must be reconciled before implementation.
- Landed the validated prerequisite on `main` and rebased this branch onto it.
- Implemented the canonical `shared.basin.gauge_points` migration with explicit
  legacy conflict detection and validated spatial defaults.
- Implemented deterministic basin, subbasin, location-code, and `wflow_id`
  assignment with row-order invariance tests.
- Implemented model-neutral D8 hydrography/topography preparation, global-ceiling
  automatic fallback, gauge-controlled incremental subbasins, full contributing
  catchments, thematic-map resampling, the location registry, explicit vector
  products, and a portable HydroMT spatial catalog.
- Validated the spatial product with 43 focused tests and Ruff, including a
  write/read round trip through the generated catalog.

**Pending:**
- Build and wire the remaining P1 spatial products, then stop at the
  spatial-to-Wflow adapter gate.
- Execute P2 only after Gate 1 approval.

**Issues / Notes:**
- `feat/wf1-improvements` is clean and has one commit not present on `main`:
  `3b6bf59 feat(wf1): declare the observation files as inputs, not params`.
