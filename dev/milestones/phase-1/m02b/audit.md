# M2b — Library upgrade audit

Read at the start of M2b. Captures the upgrade landscape so the
implementation tasks (M2b.2 onward) can proceed against a known
target rather than discovering breakage as they go.

Audited 2026-05-07 from upstream release notes and our own grep of
`WflowModel`, `setup_*`, `hydromt build`, and friends.

## Target versions

| Library | Current | Target | Released |
|---|---|---|---|
| hydromt | `>=0.9.4` (pixi.toml) | **v1.3.1** | 2026-03-03 |
| hydromt_wflow | `>=0.4.1,<0.6` (pixi.toml) | **v1.0.2** | 2026-03-09 |
| Wflow.jl | `0.8.1` (Manifest.toml) | **v1.0.2** | 2026-03-05 |
| Python | `>=3.9,<3.12` (pixi.toml) | **>=3.12,<3.13** | — |

The three v1.x lines are coupled: `hydromt_wflow v1.0.0` explicitly
*"drops Wflow.jl < 1.0.0"*, so going to hydromt_wflow 1.x forces
Wflow.jl 1.x at the same time. Likewise hydromt 1.3.1 drops Python
3.11, which raises the minimum across the stack.

## Breaking changes that hit our code

Listed by upstream version where the break landed, with our
remediation.

### hydromt 1.x

- **v1.0.0** — Major rewrite. The `WflowModel` API surface changes
  significantly (see hydromt_wflow below for our specific usage).
- **v1.2.0** — Drops Python 3.9/3.10. Lifts our floor.
- **v1.3.0** —
  - `--opt` and `--components` flags removed from `hydromt build` and
    `hydromt update`. Config files are now mandatory. **Hits us:**
    `Snakefile_model_creation:75` uses
    `--opt setup_basemaps.res="{model_resolution}"`. We have to inline
    the resolution into the build config (or generate the build
    config from a template at run time).
  - `DataCatalog` signature changes (`kwargs` → `source_kwargs`,
    `time_tuple` → `time_range`). Need to grep our code for direct
    DataCatalog usage.
- **v1.3.1** — Drops Python 3.11. Lifts our floor again to **3.12**.

### hydromt_wflow 1.x

The big break. v1.0.0 affects every Python file in `src/` that
imports `WflowModel`.

- **v0.6.0** — `clip_staticmaps`, `read_staticmaps`,
  `write_staticmaps`, `read_staticgeoms`, `write_staticgeoms`
  deprecated. `setup_soilmaps` lost soil-texture parameter
  derivation; `InfiltCapSoil` no longer supported.
- **v1.0.0** —
  - **Class rename**: `WflowModel` → `WflowSbmModel`. Hits 7 files:
    `src/setup_reservoirs_lakes_glaciers.py`,
    `src/setup_gauges_and_outputs.py`,
    `src/setup_time_horizon.py`,
    `src/downscale_climate_forcing.py`,
    `src/plot_map.py`,
    `src/plot_map_forcing.py`,
    `src/plot_results.py`.
  - **CLI rename**: `hydromt build wflow` → `hydromt build wflow_sbm`,
    same for `update`. Hits Snakefile_model_creation:75 and :124.
  - **Config-file `setup_*` renames**:
    - `setup_reservoirs` → `setup_reservoirs_simple_control`
    - `setup_lakes` → `setup_reservoirs_no_control`
    - `setup_glaciers` — name unchanged per docs grep, verify
      against the v1.0.2 docs once we install.
    Hits `config/wflow_update_waterbodies.yml` (and the same file
    duplicated under `docs/config/`).
  - **Component access**: `model.grid` → `model.staticmaps`. Data
    access via `.data` property. Hits any code that reads
    `mod.grid` or related. Need to grep when fixing.
  - **Removed methods**: `get_*`, `set_*`, `read_*`, `write_*` on
    the model class are gone — use component equivalents instead.
    Likely hits us in `setup_gauges_and_outputs.py` (calls
    `mod.setup_gauges(...)`) and elsewhere.
  - **Config**: no longer a dict. Use `set_config()` instead of
    direct dict mutation. Hits any code that reads/writes
    `mod.config` directly.
  - **Workflow rename**: `workflows.waterbodies` → `workflows.reservoirs`.
  - **`setup_constant_pars`**: values now land in the TOML, not
    `staticmaps.nc`. Hits `config/wflow_build_model.yml:29`.
  - **PCRaster** support removed (we don't use it — non-issue).
  - Drops Wflow.jl < 1.0.0 (already handled below).
  - Python 3.11+ minimum.

### Wflow.jl 1.x

- **v1.0.0** —
  - **TOML structure refactored**: `[input.static]`, `[input.cyclic]`,
    `[input.forcing]` replace flat input lists. `[state.variables]`
    replaces older state syntax. Output sections cleaner.
    **Mitigation**: hydromt_wflow 1.x generates these TOMLs natively,
    so as long as we let hydromt_wflow drive the TOML, we don't have
    to hand-write the new format. The Snakefile's `julia ... Wflow.run()`
    just consumes whatever TOML hydromt_wflow produced.
  - **CSDMS standard names**: internal model variables/parameters now
    use CSDMS Standard Names. **Hits us**:
    `wflow_outvars: ['river discharge']` in the snake configs may
    need updating to the CSDMS-blessed name. Verify against the
    v1.0.2 docs / a successful build.
  - **Removed concepts**: HBV and FLEXTopo (we use SBM — non-issue).
  - **Reservoir/lake handling restructured** — pairs with the
    hydromt_wflow `setup_reservoirs_*` rename above.
  - Migration helper `upgrade_to_v1_wflow` exists for old TOMLs but
    we won't need it: hydromt_wflow 1.x produces fresh ones.

## Python stack caps to lift

Currently in `pixi.toml`, conservative caps that exist for known
reasons:

| Cap | Reason | Action |
|---|---|---|
| `python>=3.9,<3.12` | hydromt 0.x compat | Lift to `>=3.12,<3.13` (hydromt 1.3.1 floor). |
| `numpy<2` | older xarray/scipy compat | Likely safe to drop — verify by letting pixi resolve. |
| `xarray>=2023.1,<=2024.3.0` | imposed by hydromt_wflow 0.6.1 | Drop the upper cap. |
| `zarr<3` | xarray 2024.3 predates zarr 3 | Drop — newer xarray supports zarr 3. |
| `tabulate==0.8.10` | snakemake 0.9.0 bug | Verify against snakemake 9.6.2; likely droppable. |

## Concrete remediation list (input to M2b.4)

Bumps:
- `pixi.toml`: `python>=3.12,<3.13`, `hydromt=*`, `hydromt_wflow=*`,
  drop `numpy<2`, drop `xarray<=2024.3.0`, drop `zarr<3`, drop
  `tabulate==0.8.10` cap.
- `Project.toml`: `Wflow = "1"`, `julia = "1.12"` (no change).
- `Manifest.toml`: regenerated.

Code edits:
- `Snakefile_model_creation:75` — `hydromt build wflow` →
  `hydromt build wflow_sbm`; remove `--opt setup_basemaps.res=...`
  and bake the resolution into the build config (or generate a
  per-run config).
- `Snakefile_model_creation:124` — `hydromt update wflow` →
  `hydromt update wflow_sbm`.
- `src/setup_reservoirs_lakes_glaciers.py`,
  `src/setup_gauges_and_outputs.py`,
  `src/setup_time_horizon.py`,
  `src/downscale_climate_forcing.py`,
  `src/plot_map.py`,
  `src/plot_map_forcing.py`,
  `src/plot_results.py` — `WflowModel` → `WflowSbmModel`. Audit
  each file for `mod.grid`, `mod.config[…]`, `mod.get_*`, `mod.set_*`,
  `mod.read_*`, `mod.write_*` usage and rewrite per the new component
  API.
- `config/wflow_update_waterbodies.yml` — `setup_reservoirs` →
  `setup_reservoirs_simple_control`; `setup_lakes` →
  `setup_reservoirs_no_control`. Same edits to
  `docs/config/wflow_update_waterbodies.yml`.
- `config/wflow_build_model.yml` — re-evaluate `setup_constant_pars`
  block; the values now go to TOML. Likely just keep the block; the
  effect changes from "writing a static map of constant value" to
  "writing the value into the TOML." Worth verifying by inspecting
  staticmaps.nc and the wflow_sbm.toml after a build.
- `config/snake_config_*.yml` — `wflow_outvars: ['river discharge']`
  may need the CSDMS-standard name. Verify against a v1.0.2 build's
  available output names and update.

Tests:
- `tests/snake_config_model_test.yml` and `tests/wflow_build_model.yml`
  get the same setup_* renames as the production configs.

Verification:
- `pixi install` + `pixi run install` resolves cleanly.
- All three workflows complete end-to-end.
- `dev/baseline/manifest.json` is re-recorded; expected drift is
  large (CSDMS-rename, model behaviour changes, restructured
  reservoirs). Document per-target deltas in
  `dev/m02b_baseline_diffs.md`.

## Risk register

- **CSDMS rename for `wflow_outvars`** — if we miss the rename, the
  workflow runs but produces empty / wrong output. Validate by
  asserting the expected variables exist in the wflow run output
  before downstream rules consume them.
- **`--opt` removal** — easy to miss because it's a CLI flag, not a
  Python API. Snakemake error will be informative (`unrecognized
  arguments: --opt`).
- **Reservoir/lake renames** — model build will fail loudly with
  `setup_reservoirs is not a method of WflowSbmModel` or similar.
- **Numerical drift** — expected to be material (different model
  internals, possibly different reservoir physics, different default
  parameter values in Wflow.jl 1.x). Re-baseline aggressively per the
  M2b drift policy; no per-stat investigation required unless a
  variable disappears entirely.
- **Reservoir physics may change behaviorally**, not just numerically.
  Wflow.jl 1.x's reservoir restructure could change discharge in ways
  that look like real signal in stress-test outputs. Worth a sanity
  check (mean Q in roughly the right ballpark) before declaring
  victory.

## Sources

- [hydromt releases](https://github.com/Deltares/hydromt/releases)
- [hydromt_wflow releases](https://github.com/Deltares/hydromt_wflow/releases)
- [Wflow.jl releases](https://github.com/Deltares/Wflow.jl/releases)
- [hydromt CHANGELOG](https://raw.githubusercontent.com/Deltares/hydromt/main/docs/changelog.rst)
- [hydromt_wflow CHANGELOG](https://raw.githubusercontent.com/Deltares/hydromt_wflow/main/docs/changelog.rst)
