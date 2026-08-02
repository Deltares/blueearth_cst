# Master Brief — WF1 spatial foundation and Wflow build split

### Goal

Split Workflow 1 into two explicit products: an engine-neutral spatial foundation that can run without Wflow, followed by a Wflow-SBM build that consumes that foundation. Preserve the existing downstream historical-forcing, run, and evaluation chain while making the spatial products independently targetable and reusable by future model adapters.

### Subsystem map

| Phase | Owner | Input | Expected output |
|---|---|---|---|
| P1 | `geospatial-data-analyst` or equivalent executor | `shared.basin`, gauge-point file, HydroMT catalog | Rule `prepare_spatial_maps` and the versioned `spatial/` product contract |
| P2 | `model-builder` or equivalent executor | P1 spatial products, Wflow build template, HydroMT catalog | Rule `build_wflow_model` and a runnable `hydrology_model/` Wflow-SBM bundle |

### Sequencing

P1 precedes P2 because P2 must consume the exact grid, domain, subbasin IDs, and location registry produced by P1. The phases must not implement against the same `Snakefile_model_creation` revision concurrently. Before P1 starts, land or explicitly reconcile the existing `feat/wf1-improvements` branch, which already changes observation-input handling in the same Snakefile and model scripts.

P2 starts with a bounded HydroMT-Wflow adapter proof. It may proceed only after Gate 1 confirms that the Wflow build consumes the P1 products rather than independently delineating the basin again.

### Shared constraints

- Read and follow repository-root `AGENTS.md`; use the pinned pixi environment and documented HydroMT/HydroMT-Wflow APIs.
- Keep upstream packages and vendored documentation read-only. Work around any upstream limitation only in `blueearth_cst/`, the Snakefile, or project configuration.
- The spatial foundation is model-neutral: no Wflow TOML, Wflow parameter constants, Wflow LDD naming, Feddes parameters, routing parameters, or pedotransfer-derived Wflow physics maps.
- Wflow-specific derivatives and `setup_constant_pars` remain in the Wflow phase; they are excluded only from spatial preparation, not removed from the Wflow model. Treat `setup_constant_pars` as Wflow configuration/TOML scalars, not as spatial-map outputs.
- Preserve the three root Snakefiles as entry points and keep `config_path = workflow.configfiles[0]` forwarding.
- Do not modify generated run outputs, `pixi.lock`, `Manifest.toml`, or a production `project_dir`. Integration runs use a clean, dedicated test project directory.
- Keep every cross-model identity explicit in attributes and the location registry. Never use a display name or file order as the sole join key.
- Preserve existing plot filenames for this task; use the registry for identity. Redesigning data-dependent plot enumeration is a separate concern.
- Update `dev/reference/workflows/model_creation.md` and the clean config template when the public configuration or output contract changes.

### Human gates

1. **Spatial-to-Wflow adapter gate — PAUSE before P2 implementation.** Present the generated spatial artifact schema and a minimal proof showing the pinned HydroMT-Wflow version can consume it through a generated catalog or a project-owned adapter. Re-running `setup_basemaps` from the original global hydrography is not acceptance evidence.
2. **Scientific-delta gate — PAUSE before recording any new baseline.** Present map-level differences and historical-discharge differences. Any change not explained solely by the new deterministic IDs, file layout, or documented resampling must receive owner approval.
3. **Landing gate — PAUSE before merging the implementation branch.** Present both rule-specific validation reports, three workflow dry-runs, the spatial-only independence proof, and the end-to-end Wflow smoke run.

### Cross-cutting validation

- Per Snakefile edit: `pixi run pytest tests/test_cli.py` and a Workflow 1 dry-run using the smallest documented test config.
- At each phase boundary: `pixi run pytest tests/`.
- Final DAG checks: dry-run all three Snakefiles with `tests/snake_config_model_test.yml` or the current documented fixture config.
- Final integration: build P1 alone in a clean project, then P2 and the remaining Workflow 1 chain. Report CRS, bounds, resolution, nodata, units, basin/subbasin IDs, generated Wflow triplet, and Wflow smoke-run result.
- Non-regression: run `python dev/scripts/check_baseline.py check --workflow model_creation`. Do not run `record` until Gate 2 authorizes an expected delta.
- Cross-phase falsifier: if `snakemake prepare_spatial_maps ...` schedules a Wflow build, creates `hydrology_model/`, imports `hydromt_wflow`, or writes a TOML, the split has failed.
- Consumption falsifier: if changing a P1 ID/source fixture does not propagate to the P2 Wflow artifact, or P2 can complete after its declared P1 products are removed, P2 is not consuming the spatial foundation.

### Phase brief index

- [P1 — engine-neutral spatial maps](phase-1-spatial-maps-task-brief.md) — complete; Gate 1 approved
- [P2 — Wflow-SBM model build](phase-2-wflow-build-task-brief.md) — implemented; Gate 2 review pending
