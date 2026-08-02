# Task Brief — P2 Wflow-SBM model build from spatial products

### Context

- Canonical ruleset: repository-root `AGENTS.md`. Program-level constraints and gates are in [the master brief](master-task-brief.md).
- P1 is a hard prerequisite. Its accepted `spatial_maps.nc`, vector layers, `location_registry.csv`, and `spatial_catalog.yml` are the authoritative grid/domain/identity inputs.
- The current rule 1.03 invokes `hydromt build wflow_sbm` from a template containing `setup_basemaps`, `setup_rivers`, land-use, LAI, soil, and `setup_constant_pars`. This phase must retain Wflow physics while removing independent domain preparation.
- In the pinned plugin, `setup_outlets()` reuses subcatchment-map values as outlet IDs when that map exists; otherwise it falls back to sequential `1..n`. The accepted design requires the inherited-ID path.

### Goal

Replace `create_model` with a rule named `build_wflow_model` that consumes P1 products and produces the standard runnable Wflow-SBM model bundle without independently delineating or sourcing a competing spatial domain.

### Non-goals

- No alternative hydrological engine implementation.
- No recalibration or change to the approved Wflow constant-parameter values.
- No moving Wflow-specific parameter maps into the neutral spatial contract.
- No data-dependent plot/checkpoint redesign; retain existing plot filenames and use the location registry for joins.
- No patch to HydroMT or HydroMT-Wflow internals.

### Allowed scope

**Permitted:** `Snakefile_model_creation`; `blueearth_cst/model/**` modules involved in build-config preparation, gauges/outlets, and outlet indexing; a project-owned adapter under `blueearth_cst/model/` or `blueearth_cst/spatial/`; `config/templates/wflow_build_model.yml` and new Wflow-adapter templates; relevant config examples; `tests/**`; `dev/reference/workflows/model_creation.md`; this brief directory.

**Approval-gated:** new dependencies; changes to Workflow 2/3; baseline manifest recording; deletion or renaming of existing `hydrology_model/` contract files; accepting a physics/discharge delta.

**Forbidden / generated:** upstream packages, `.pixi/**`, lock files, production run outputs, and manual edits to a generated Wflow model.

### Required changes (checklist)

1. Complete the Gate 1 adapter proof using this preference order:

   - Preferred: register P1 outputs through `spatial_catalog.yml` and point documented HydroMT-Wflow `setup_*` methods at those generated catalog entries.
   - Allowed fallback: a small project-owned adapter that initializes the Wflow model grid/geometries from P1 and then calls documented Wflow-specific setup methods.
   - Rejected: call `setup_basemaps` against the original global hydrography and merely compare its result afterward; that duplicates delineation instead of consuming P1.

   Record the selected route and evidence in the Workflow 1 contract before continuing. If neither allowed route works with the pinned API, stop at Gate 1 with the exact limitation.

2. Separate template responsibilities:

   - P1 owns domain, flow direction/accumulation, basin/subbasin IDs, rivers, and raw thematic data.
   - The Wflow template owns model parameterization: Wflow map naming/encoding, river-routing parameters, LULC parameter mapping, LAI conversion, soil pedotransfer outputs, Wflow configuration, and `setup_constant_pars`.
   - Remove only the Wflow template steps that would independently reproduce P1 work. Point retained Wflow-specific setup steps to P1 catalog entries where their APIs accept source data.

3. Add `build_wflow_model` to `Snakefile_model_creation`:

   - Declare the P1 catalog, raster, vector, and registry artifacts as inputs.
   - Produce `{project_dir}/hydrology_model/staticmaps.nc`, `wflow_sbm.toml`, required `staticgeoms/**`, and the existing safe rebuild sentinel/cascade contract.
   - Preserve logs, benchmarks, `run_logged`, and downstream ordering for waterbodies, gauges/outputs, runtime, forcing, Wflow execution, and plots.
   - Renumber rule banners/log names consistently if inserting the second rule changes definition-order numbering; update `LOG_RULES`, tests, and documentation atomically.
   - Ensure a P1 input/config change rebuilds P2 and its mutation cascade, while a Wflow-only template change does not rebuild P1.

4. Make location identity explicit in the Wflow adapter:

   - Generate the plugin-facing gauge table from `location_registry.csv`.
   - Call `setup_gauges(..., index_col="wflow_id", basename="locations")`; do not let file basename or row index determine model identities.
   - Ensure the Wflow subcatchment map is populated with P1 subbasin IDs before `setup_outlets()` so outlet IDs inherit them instead of falling back to `1..n`.
   - Require the primary control location's `wflow_id` to equal its `subbasin_id`; retain the reserved additional-location range from P1.
   - Replace or extend `outlet_index.csv` with a deterministic crosswalk to `basin_code`, `subbasin_code`, `location_code`, `station_name`, and `wflow_id`. Preserve existing positional plot names only as compatibility labels.
   - Validate observation-timeseries columns against the resolved registry before plotting; report missing, duplicate, or unexpected station IDs explicitly.

5. Preserve the produced HydroMT-Wflow handoff triplet and downstream contracts: `wflow_sbm.toml`, `staticmaps.nc`, and `staticgeoms/`, plus forcing/states when later rules add them. Keep the existing waterbody no-data behavior and close/write semantics.

6. Update config migration notes, observation templates, `dev/reference/workflows/model_creation.md`, rebuilding tests, interchange tests, map/ID tests, and any baseline target paths affected by the rule split. Do not record a baseline before Gate 2.

### Commit plan

| Subject | Paths | Invariant preserved |
|---|---|---|
| `docs(wf1): record the accepted spatial-to-wflow adapter` | contract/design note and focused proof tests | P2 has one evidenced input route before runtime code changes |
| `refactor(wflow): build model parameters from the spatial catalog` | adapter, Wflow template/config prep, unit tests | Wflow physics remains Wflow-owned while domain/IDs come from P1 |
| `refactor(wf1): split spatial preparation from wflow building` | Snakefile, cascade tests, logs/docs | Every commit has a valid DAG; P1 and P2 invalidation boundaries are explicit |
| `test(wf1): verify the split end to end` | integration/contract tests and approved baseline note only | Results evidence is attributable separately from implementation |

### Validation

1. **Per edit — narrow:** adapter/config tests, `setup_gauges_and_outputs` tests, outlet-ID/crosswalk tests, and `tests/test_model_rebuild_cascade.py`.
2. **Per Snakefile edit — DAG:** `pixi run pytest tests/test_cli.py`; Workflow 1 dry-run; confirm `prepare_spatial_maps` alone schedules no P2 rule.
3. **HydroMT validation — whenever the generated adapter workflow/catalog changes:** run the pinned `hydromt check <PLACEHOLDER>` command appropriate to the generated workflow and catalog; record unresolved sources.
4. **Integration — once after P2 stabilizes:** in a clean dedicated project, run P1, P2, then the remaining Workflow 1 rules. Inspect CRS, bounds, resolution, nodata, required Wflow maps, static geometries, TOML paths, forcing, and the historical smoke run.
5. **Identity behaviour:** assert every outlet ID equals a P1 subbasin ID; every configured gauge output joins exactly once to the registry and observation header; no output relies on file order or filename-derived basename.
6. **Non-regression — Gate 2:** compare current versus split `staticmaps.nc`, `wflow_sbm.toml`, and normalized `run_default/output.csv` discharge statistics/flow-duration quantiles. Run `python dev/scripts/check_baseline.py check --workflow model_creation`. Any tolerance for non-byte-identical numerical results is `<PLACEHOLDER>` until owner-approved.
7. **Full gate — phase end:** `pixi run pytest tests/` and dry-run all three Snakefiles.

Falsifiers:

- Consumption is disproved if P2 succeeds without declared P1 artifacts, reads the original global hydrography for domain delineation, or produces outlet IDs unrelated to P1 subbasin IDs.
- Invalidation separation is disproved if a Wflow-only constant/template change schedules P1, or a P1 map/ID change leaves P2 up to date.
- Identity inheritance is disproved by sequential `1..n` outlet IDs when a P1 subbasin map exists.
- Physics preservation is disproved by an unexplained Wflow map, TOML parameter, or discharge delta.

### Acceptance criteria

- The Workflow 1 DAG contains distinct `prepare_spatial_maps` and `build_wflow_model` rules with the required dependency edge.
- P2 consumes the P1 grid, basin/subbasin IDs, thematic sources, and registry through the Gate 1-approved route; it does not independently delineate the domain.
- The Wflow triplet is valid, `setup_outlets()` follows the inherited-ID path, and gauges/observations join through the registry.
- Wflow-specific constants and derived parameter maps remain in P2; none leak into P1.
- The downstream waterbody, forcing, run, plot, log, and benchmark chain remains functional.
- All relevant tests and dry-runs pass. Any intended result delta is approved at Gate 2 and recorded; otherwise unexplained deltas trigger rollback.

### Output requirements

- Code, Wflow-specific templates, tests, migration notes, and updated Workflow 1 contract.
- A phase report naming the selected adapter route, HydroMT validation result, produced triplet, ID crosswalk checks, commands run, and remaining risks.
- **Results delta:** map/TOML/discharge comparison against the pre-split build, with each accepted difference tied to a named design decision.

### Task constraints

- Use `WflowSbmModel` and `hydromt build/update wflow_sbm` terminology for the pinned v1 plugin; do not restore the pre-v1 `wflow` entry point.
- Run `setup_basemaps` first only if the Gate 1-approved adapter uses it to ingest P1 rather than to re-delineate from global data; otherwise initialize through the documented project-owned adapter route.
- Keep all plugin-touching steps before the produced Wflow triplet separate from Wflow.jl execution and model validation.
- Pair partial HydroMT updates that touch TOML state with the required config write/close behavior.
- Stop for the three master human gates; do not record baselines, accept physics changes, or merge without approval.
