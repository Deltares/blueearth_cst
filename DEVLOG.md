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
- Added the independently targetable `prepare_spatial_maps` rule with nine
  explicit outputs, tracked catalog/config/gauge inputs, log, and benchmark.
- Ran the spatial target against the local Gabon/Deltares sources. The written
  catalog and all entries reopen through HydroMT; topology, metadata, CRS,
  raster/vector/registry IDs, subbasin non-overlap, and catchment containment
  validate after write.
- Corrected three failures found by that integration run: decoded NetCDF nodata
  IDs, a one-cell analysis-grid border, and invalid self-touching polygons from
  raster vectorization.
- Completed the Gate 1 adapter proof using only P1 products and public
  HydroMT-Wflow APIs. The proof preserves the P1 grid and IDs and writes a
  reopenable Wflow model triplet without rerunning `setup_basemaps`.
- Passed Ruff, 45 focused spatial tests, 15 rule/DAG tests, the targeted
  one-job spatial dry-run, the 17-job Workflow 1 dry-run, and the full suite
  (1,004 passed, 31 skipped, 1 expected xfail).

**Pending:**
- Obtain owner approval at the spatial-to-Wflow adapter gate.
- Execute P2 only after that Gate 1 approval.

**Issues / Notes:**
- The prerequisite `feat/wf1-improvements` work was merged to `main` before
  this branch was rebased, avoiding an overlapping Snakefile implementation.
- Successful Windows runs can still print the repository's known benign empty
  `Error in sys.excepthook` cascade during GDAL/rasterio interpreter shutdown;
  the rule exits zero and all declared outputs validate.
- Improvement candidate captured for the `snakemake` skill: document that a
  Python file used by `script:` must not rely on a leading
  `from __future__ import ...`, because Snakemake prepends generated code before
  the file and turns that import into a syntax error.
