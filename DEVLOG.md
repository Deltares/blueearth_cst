# DEVLOG: feat/wf2-cmip6-improvements

Historical task handoff. Retained results and qualifications now live in wf2-suite-integration.md
(private-checkout-only citation: `dev/records/validation/wf2-suite-integration.md`); the retirement map
(private-checkout-only citation: `dev/records/migrations/review-retirement-2026-10-20.md`) resolves old design
citations. The resume and landing instructions below describe the original task, not the current session.

## Resume here

**Objective:** Implement the G2-approved WF2 starter-suite integration, task `t2610071652`.
**State:** Technical gates passed; owner authorized local landing before visual review.
**Next action:** Land with candidate verification, preserve task records and close the board item.
**Blocker:** None; host access was required for the restored P-drive.

## Current contract and execution

- Accepted design: `dev/records/reviews/wf2-suite-integration/wf2-suite-integration-design.md`.
  Records move to that durable home during the authorized post-landing board closure.
  The primary checkout has uncommitted finalization artifacts; preserve them and coordinate with its writer.
- Execute design section 10 in order, with one verified commit per unit. Keep this branch; landing and pushing
  require separate user instructions.
- Unit 1 owns additive analysis-window, slicing, seasonal and detrended helpers, extracted series checks,
  and their focused tests. Preserve Stage B/reducer hashes, workflow inventories and aligned CSV bytes.
- Owner ruling 2026-10-07: the October fallback reference uses WY labels `1984–2013`, matching the lab.
  Use 30 complete historical water years ending at the last complete WY of the fallback calendar span;
  do not join scenario data into the reference. Preserve all other accepted numerical contracts.
- Prior P0 change-factor and lab snapshots exist in `.tmp/scratchpad/t2610071652-snapshots/`.
  Its rapid snapshot dates from before this branch's WF2 fixes and does not satisfy the before-image gate.
- Owner ruling 2026-10-07: proceed with cached CMIP6 inputs; leave fresh-run gates pending.
  Cached-input comparisons must be labelled as such and do not count as fresh end-to-end passes.
- Owner ruling 2026-10-07: seasonal precipitation percentages use rates consistently. Document the lab's
  total-based percentage calculation as a parity exception; precipitation CV still uses integrals.
- Unit 2 uses the approved 13-model family/country constants. Exact cited Zenodo 7920408 licence could
  not be verified; no CSV is redistributed or substituted from another record. Metadata matches the lab.
  Unlisted models use a flagged single-family institution match, otherwise their own family.
- Coverage accepts resolver COMBINATIONS directly to preserve exclusions. With resolved POINTS alone,
  requested counts remain unknown; absence is not mislabelled as an unrequested run.
- Scratch logs/configs for a fresh rapid run are in `.tmp/scratchpad/wf2-suite-evidence/`.
  The scratch project config preserves rapid settings and points to a fresh matching project root.
- A local data tree exists, but it lacks the current catalog's `rivers_ge30m.fgb` and MERIT VRTs.
  Substituting its older catalog would change the evidence basis and has not been authorized.
- Owner ruling 2026-10-07: P-drive restored; resume fresh-run gates. Host checks confirm both the mapped
  catalog root and its UNC target exist. Fresh rapid completed in 4m36s, with all 14 expected PNGs.
- Gate audit caught omitted retirement updates to baseline figure targets, console examples and the
  output guide. They now use suite paths; the stale baseline manifest remains unchanged. Baseline scope,
  provenance and activation checks passed 49 tests, then the corrected console example passed its recheck.
- Added the specified optional local parity test covering all design §9.1 table categories. Cached P0
  passed 11 tests; missing local inputs skip with named paths. Fresh P0 will use explicit input overrides.
- Fresh single-model WF2 passed after selecting `hydro_rivers_lin` in its P-drive seed config. Rapid and
  single-model availability, PNG and caption inventories are bijective (14 and 13 products).
- Ancillary final checks passed 51 tests (baseline scope and CLI). Fresh spatial parity passed all 19 tests,
  including both fixture checks, using `CST_SPATIAL_PARITY_PROJECT`; the standing baseline was not altered.
- First batch fast gate: 3963 passed, 12 failed, 17 skipped, 1 xfailed. Ten failures assumed pytest scratch
  was outside the checkout; one used the stale spatial fixture and one raced the console example edit.
  Test portability corrections and a frozen-tree rerun are pending; this result is not a passing gate.
- Corrected batch fast gate passed: 3975 passed, 17 skipped, 1 upstream xfailed (5m15s). The 11 local lab
  checks skipped in that generic invocation but passed separately against cached P0 with explicit paths.
  Six-file portability checks passed 454 tests; strengthened sweep checks passed another 19.
- Protected Stage B/reducer hashes and WF3/WF4 inventories remain identical to the before-image.
- P0 scheduler stalled after job 179 with three scalar outputs written and no CPU progress for over eight
  minutes. Preserved logs/output, stopped only that run and resumed with one core and `--rerun-incomplete`.
  The resume uses `--nolock` because other isolated fresh roots are active; clean stale locks afterward.
- `pixi run test-e2e` (task includes `--run-integration`) passed its one all-five-workflow test
  (4060 deselected, 47m39s); no integration skip. Fresh root is the scratch `e2e-pytest/.../project`.
- Synthetic dry-reference original/current Stage B and serializer comparison passed both raw-byte checks:
  annual 4 rows SHA256 `640f022494ed14f0e04f065073d5932ebd910161af8eb7073deb49c8471ae953`;
  monthly 48 rows SHA256 `a0a0f003af7e5fcd0ecc3af3a25c0a6788217df35e8f899af352d54d8132127d`.
  This deliberately synthetic fixture exercises aligned monthly `reference_below_threshold` flags; its
  disposable identities are not used by any fresh production run.
- Fresh rapid annual/monthly CSVs match the original Stage B oracle on the same validated fresh scalar
  inputs, raw byte for byte (12 and 144 rows). SHA256 annual
  `27c6916dd90204fbd5f433ded5e2dc1c5d4997c7872de476e54b8da66673ff2e`, monthly
  `5504fb8ecea0116535d0549011b6b028f4df42e845dd764f310574195c9a1b54`.
  This is an original-source oracle comparison, not a pre-edit fresh workflow execution.
- Fresh P0 completed after its serial recovery; all 125 raw inputs were validated and reused, all 125
  basin scalar series and downstream products rebuilt. This is a retained-raw fresh downstream run,
  not a claim of 125 new remote acquisitions. The original P0 archive was not changed.
- Fresh P0 lab parity passed all 11 tests, with the owner-approved seasonal precipitation rate exception.
  Its 600 annual / 7200 monthly aligned CSV rows match the original snapshot raw bytes: annual SHA256
  `360d2512fce5f62a06b13f272c7e001f7e914b2b55701eacc38d6ddb6e11eea8`, monthly
  `d41e232d6a5d80d8bfbce79b2f268d584ac9abaaf675effabc1fb432b97c0a9f`.
- P0 availability / written PNG / caption inventory is bijective: 15 products. Rule benchmarks are
  2.05b = 239.79 s, 2.06 = 18.78 s, combined 258.57 s (4m19s), below the 5-minute design budget.
  Fresh cloud and member-ratio figures were visually inspected; earlier cached P0 inspection covered all 15.
- Removed the two stale P0 Snakemake lock files only after the recovered run completed, verifying their
  output ownership first. All fresh-run gates are complete. The original-source P0 oracle also passed
  on the same validated fresh scalar inputs: both aligned CSVs match the original implementation bytes.
- First full gate: 4032 passed, 22 failed, 17 skipped, 1 upstream xfailed (24m28s). All 22 failures were
  console handler assertions after the new standalone-console inspection left `_RUN_HEADER` staged.
  Isolated that inspection's header, rule registries and token environment; the same file sequence then
  passed 384 tests. Lint/format passed and fix committed; full frozen-tree rerun is in progress.
- Final full gate passed: 4054 passed, 17 skipped, 1 upstream xfailed (13m54s). The 17 skips are five Linux
  tolerance controls on Windows, 11 default-path lab tests and the separately invoked integration test.
  All 11 lab tests passed with fresh P0 overrides; the all-five integration gate passed independently.
  Lint and format-check passed. All implementation and test corrections are committed on this branch.
- Final gate commands (each redirected to its named log under `wf2-suite-evidence`):

  ```powershell
  pixi run test-fast -p no:cacheprovider --basetemp .tmp/scratchpad/wf2-suite-evidence/fast-pytest-final
  pixi run test-full -p no:cacheprovider --basetemp .tmp/scratchpad/wf2-suite-evidence/full-pytest-final
  pixi run test-e2e -p no:cacheprovider --basetemp .tmp/scratchpad/wf2-suite-evidence/e2e-pytest
  ```

  Fast/full used `CST_SPATIAL_PARITY_PROJECT` pointing to the fresh all-five project. Their logs are
  `test-fast-final.log`, `test-full-final.log`; integration is `fresh-e2e-gate.log`. Numerical evidence is
  `fresh-rapid-byte-parity.log`, `fresh-p0-byte-parity.log`, `fresh-p0-inventory-snapshot-parity.log`,
  `fresh-p0-lab-parity.log`, `synthetic-dry-byte-parity.log` and `protected-identities-final.log`.
- Owner visual acceptance in accepted design §9.2 remains a human check. P0 captions and all 15 PNGs are
  at `.tmp/scratchpad/wf2-suite-evidence/p0-fresh/data/climate/projections/cmip6/plots/`.
  Landing and pushing are not authorized by completing these gates; the occupied lane remains pinned.
- Owner ruling: land locally first. Preserve accepted design/review history and implementation evidence
  under `dev/records/reviews/wf2-suite-integration/`, then close `t2610071652` on the canonical main board.
  Visual review remains separate; no push is authorized.

## Current validation

- `pixi run python scripts/run_workflow.py analyze_projections --config test_case/project_config_rapid.yml
  --project-dir test_case/test_rapid --cores 3 --dry-run` — passed with host access; 15 jobs.
- The equivalent execution against `test_rapid` failed before jobs: incompatible existing run archive.
- Fresh-root execution with a matching scratch config reached rule 2.01 and failed opening the P-drive catalog:
  HydroMT reported `None of the specified roots were found`. No fresh CSV before-image was produced.
- Unit 1 passed 42 focused tests, 20 CLI tests, and 11 tests after the review corrections. Repository lint,
  format-check, compilation and whitespace checks passed. Full batch gates run at approved landing.
- Unit 2 passed all 5 focused tests, repository lint/format-check, compilation and whitespace checks.
- Unit 3 passed 16 focused tests, scoped lint/format/compilation and unchanged protected inventories.
  Root repository lint and format-check passed after formatting the concurrent unit 4 edits.
- Unit 4 passed 167 producer/inventory tests, then 3 producer tests after benchmark metadata changes
  and 1 shared-window test after retaining both horizons' requested windows. CLI: 20 passed; rapid dry-run:
  16 jobs including 2.05b. Repository lint and format-check passed.
- Unit 5a wires 2.06 to the 14 analysis tables and record, with availability-driven owned folders,
  ordinary PNG caption keys and an explicit optional-captioned-copy bijection. The renderer never reads
  scalar NetCDFs or recomputes the analysis. Log-space model summaries are supplied in the record.
- Cached P0 final rendering passed: 15 ordinary PNGs and 15 manifest keys, one correctly unavailable
  combined-cloud product for the single analysis window. All ordinary products were visually inspected;
  synthetic rendering also produced 15 ordinary, 15 captioned and one combined-cloud PNG without warnings.
  Annual overviews retain absolute/integral and anomaly rows; precipitation cycles use percentage changes.
- Unit 5a artist checks passed (7 focused tests, then 2 optional-copy falsifiers); repository lint and
  format-check passed. CLI: 20 passed; updated rapid dry-run: 16 jobs. Fresh gates remain pending.
- Root final figure/producer batch passed all 12 tests. A sandbox-only run had four setup errors from
  the denied legacy pytest temporary root; the host-access rerun passed, with the known cache warning.
- Unit 5b removes predecessor plotting helpers, declarations and tests together. Rule 2.05 emits no
  figures; rule 2.06 and both gathers use the new caption manifest. The report links those captions and
  distinguishes aligned rates from suite annual integrals and percentages. Live docs and inventory
  fixtures use the new suite paths; sealed milestone records remain unchanged.
- Retirement checks passed 187 CLI/output-inventory tests and the 16-job rapid DAG dry-run. Current
  Stage B over isolated archived rapid and P0 values reproduces both before-images byte for byte.
  Stage B/reducer hashes and WF3/WF4 inventories are identical; surviving coordinate/merge ASTs match.
- Unit 5b's affected numerical/report/figure batch passed 101 tests; four final count/agreement
  falsifiers passed after root review. Availability and caption counts now use the artist's actual
  periods, variables, statistics and quantities. Repository lint and format-check passed.
- Final cached P0 render passed with host access: 15 ordinary PNGs and manifest keys, one unavailable
  combined cloud. A sandbox repeat exposed native Windows DLL exception `0xc06d007f` in Matplotlib;
  this was not a Python assertion failure. The successful host render is recorded separately and does
  not replace any deferred fresh-run gate.
- Deleted `blueearth_cst/projections/projection_plots.py`, `tests/test_projection_plots.py`, and
  `tests/test_plot_proj_timeseries.py`. Their cloud-pair regression lives in the new suite artist tests.
- The design's precautionary HydroMT import premise was stale: derive/summary have no raster-accessor
  callers. `get_stats_climate_proj.py` already imports HydroMT explicitly where it uses the accessor.
  Corrected the obsolete comment without adding an unused import or changing protected arithmetic.
- Cached P0 producer executed all 100 points into 14 CSV tables and a record: 18,700 change rows,
  30,200 annual member rows, 5,100 member ratios and 40 exact-mask panel benchmarks. It used isolated
  test identities on archived values, not current production-cache recipes or a fresh workflow run.
- Lab reduction parity passed for all unchanged fields of 2,444 model rows and 188 summary/agreement
  rows; no missing keys, value or count mismatches. Additional statistic rows are separately reported.
- Panel precipitation benchmarks use the median of model-mean eligible reference CVs. Every retained
  reference/future mask pair gets its own band; heterogeneous masks are not replaced by counts.
- Cached-value probing exposed a predecessor CSV defect: historical flags with no scenario dimension
  failed serialization. Companion broadcasting fixes it without changing the arithmetic kernels.
  The new regression failed before the fix; all 37 `test_change_factor_table.py` tests passed afterwards.
- Isolated archived-rapid before-image: 8 annual and 96 monthly rows, computed from committed Stage B source.
  Disposable copies use explicitly synthetic test identities because archived reducer metadata is stale;
  original cached NetCDFs are untouched. This tests numerical/serialization parity, not cache validity.
  Raw-byte SHA256 annual: `4984466e5f1ebd14f48e075f9573e83cf9567cc0d2c9f696e3c8ecff41a6f62a`;
  monthly: `706ab4593a9fd48ce4f793c51ba6147e1e6bdeb47abb7dc4289abcebab3f6864`.
  These replace the initial line-ending-normalized hashes produced by the environment hash helper.
- After extraction, the archived-input CSV hashes above match exactly. Stage B hash remains
  `a11fc4fbe4b92c37a623154a7a2f657c6f53790d14a8c1b577e153acc4a87fce`; reducer hash remains
  `942fa42db363742b5f3258cc386b7de7c7a1577973ec7a4a5936e8a6e3d7e71e`.
  WF3/WF4 repository inventories also match the pre-edit capture.
- Cached P0 comparison also passed: original and extracted Stage B reproduce the captured 600 annual
  and 7,200 monthly rows byte for byte. Raw SHA256 annual:
  `360d2512fce5f62a06b13f272c7e001f7e914b2b55701eacc38d6ddb6e11eea8`; monthly:
  `d41e232d6a5d80d8bfbce79b2f268d584ac9abaaf675effabc1fb432b97c0a9f`.
  Test-only copies have distinct per-series identities to keep reference caches isolated across models.
- Initial analysis-table comparison covers 11,900 absolute and 5,200 nonseasonal relative rows. Keys,
  counts and statuses match the lab; monthly mean/SD values exceed tolerance because archived scalar
  inputs are float32 while the lab reduces float64. Promote analysis slices only before kernel calls;
  keep the aligned arithmetic path unchanged. After correction all compared values match at the accepted
  tolerance (max absolute change difference 1.42e-14; relative difference 7.11e-14). The ruled seasonal
  precipitation percentage exception is excluded from this parity claim.
- Review found raw SD rows must use raw SD for zero-variability status: a linear trend has positive raw
  SD but zero detrended SD. Ratio rows use their own detrended rate/integral SD; a falsifier covers both.

## Integration plan state

Follow accepted design section 10. Numerical implementation uses one `python-engineer` thread, inherited
model/effort, with root responsible for reconciliation and scoped commits. Worker scope is one design unit
at a time; failed falsifiers or changed contracts reopen only the affected unit. No nested delegation.

| Unit | State | Evidence | Next action |
|---|---|---|---|
| 0 snapshots | Done | P0/lab snapshots; fresh-input original-source byte oracles | — |
| 1 numerical helpers/checks | Done | `1672686c`; focused checks and invariants above | — |
| 2 coverage/families | Done | `7683ddb6`; 5 focused tests | — |
| 3 reductions/variability | Done | `c606bcdd`; 16 tests, P0 parity, identities | — |
| 4 analysis rule/tables | Done | `5b5b219e`; producer/inventory, CLI, DAG and cached P0 | — |
| 5a figures/captions | Done | `a9d69d47`; 12 tests, 15 cached P0 figures, synthetic copies | — |
| 5b retirement | Done | Focused tests, CLI/inventory, DAG, byte parity, identities | — |
| 6 batch gates | Done | Fresh, parity, identities, timing, fast, full and e2e passed | Owner review |

## Historical branch record (retained)

The record below predates this branch's WF2 work.

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
  6,343/7,670 days.
- Received explicit owner approval for the documented material delta at Gate 2.
- Rebuilt the canonical relative Workflow 1 fixture from clean spatial/model
  outputs. A parallel build stalled while writing `staticmaps.nc`; after stopping
  only the identified Snakemake/build processes and unlocking the workdir, the
  serial `--rerun-incomplete` build completed all 17 jobs successfully.
- Corrected the stale baseline target from `forcing/plots/precip.png` to the
  actual `forcing/plots/forcing_precip_map.png` path and passed its focused
  validation batch (27 tests).
- Recorded only the five approved Workflow 1 baseline targets. The resulting
  manifest retains 14 total targets, and the post-record check confirms all five
  Workflow 1 targets match.
- Retired the renamed legacy `forcing/plots/precip.png` manifest entry that a
  scoped merge cannot infer is obsolete; the manifest now matches the 14-target
  table exactly.
- The unscoped checker is not runnable against this isolated WF1-only fixture:
  it correctly reports the nine absent WF2/WF3 outputs. Their retained manifest
  entries were not modified by the scoped record.
- Received explicit owner approval to land the branch at Gate 3.
- Rebased all six task commits onto the current local `main`. The only manual
  resolutions reconciled two moved `dev/records/milestones/` documentation paths; no
  behavioral code conflicted.
- Post-rebase validation passed the 94-test WF1/spatial/baseline landing batch
  and the five-target scoped baseline check. The broader suite passed 1,022
  tests; its 19 failures are fixture-dependent interchange checks for WF2/WF3
  files absent from this WF1-only worktree.

**Pending:**
- Apply the already-approved reusable Snakemake skill improvement only after the
  project task has landed.

**Issues / Notes:**
- The prerequisite `feat/wf1-improvements` work was merged to `main` before
  this branch was rebased, avoiding an overlapping Snakefile implementation.
- Successful Windows runs can still print the repository's known benign empty
  `Error in sys.excepthook` cascade during GDAL/rasterio interpreter shutdown;
  the rule exits zero and all declared outputs validate.
- The clean build is reliable at `-c 1`; concurrent NetCDF/HDF5 writes during
  model construction can contend on Windows, so the serial retry is the recorded
  baseline-generation path rather than evidence of a workflow dependency defect.
- Improvement candidate captured for the `snakemake` skill: document that a
  Python file used by `script:` must not rely on a leading
  `from __future__ import ...`, because Snakemake prepends generated code before
  the file and turns that import into a syntax error.
