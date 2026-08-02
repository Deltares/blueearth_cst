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
- Received owner approval at Gate 1 and implemented the project-owned Wflow
  adapter. Rule 1.03 initializes the P1 grid/geometries/IDs, converts D8 to
  LDD, and runs only Wflow-owned parameter setup methods.
- Removed independent `setup_basemaps`/river-source configuration from the
  Wflow build template while retaining Wflow constants and soil pedotransfer.
- Rewired gauge/output and outlet-index steps to the resolved registry and
  added explicit pre-plot observation-header validation.
- Found and fixed automatic outlets below the Wflow river threshold by
  constraining fallback selection to the P1 river mask; the integration case
  now has five valid automatic outlets rather than twenty nominal cells.
- Completed a clean seven-rule P1→P2→Wflow run for 2000–2020. The primary
  discharge series has +0.015% mean bias, 1.48% mean-normalized RMSE, and
  0.99990 correlation against the pre-split run.
- Validated the actual Gabon observation header against a disposable
  latest-schema P1/P2 build: all four stations retain IDs 101–104.
- Extended the baseline discharge reader to select the primary outlet through
  `outlet_index.csv` when registry gauges add multiple Q columns.
- Passed the full suite (1,014 passed, 31 skipped, 1 expected xfail) and all
  three final DAG dry-runs (WF1 17 jobs, WF2 25, WF3 50).
- Confirmed the existing five-target Workflow 1 baseline still matches its
  manifest. The split discharge fails the strict per-timestep comparator on
  6,343/7,670 days, so Gate 2 is intentionally still open.

**Pending:**
- Obtain owner approval at the scientific-delta gate before recording a new
  baseline.
- Stop again at the landing gate before merging.

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
