# M02c — Test coverage (pre-M3) — design

**Date.** 2026-05-08.

## Goal

Establish unit-test infrastructure and convert documented bugs into
regression coverage *before* M3-M5 begin refactoring workflow scripts.
Two reasons to do this now rather than fold it into M3-M5:

1. M3-M5 each refactor a workflow's scripts. Without unit tests, those
   refactors fly blind on edge cases. The baseline manifest catches
   numerical regressions on `rule all` outputs, but not malformed inputs,
   missing config keys, silent truncation, or other behavioral edges.
2. Doing it once now means M3-M5 inherit a working pattern instead of
   re-inventing fixtures, mocking, and test layout from scratch in each
   milestone.

Scope is deliberately narrow: five small, stable modules where bugs are
already documented and that M3-M5 are unlikely to substantially rewrite.
Plotting modules and workflow-script-heavy code stay out of scope until
their owning milestone.

## Approach: hybrid (#3 establish discipline + #2 catch latent bugs)

Build the testing infrastructure (#3) by writing tests for known bugs
(#2) in stable utility code. The infrastructure has lasting value; the
bug-driven tests have immediate value as regression catches.

Not-chosen alternatives, for the record:

- **Safety net for M3 alone.** Redundant with the baseline manifest at
  output level, and unit tests on code about to be refactored get
  rewritten alongside the code.
- **Catch latent bugs alone.** Produces a scattershot suite with no
  shared patterns; M3-M5 then re-invent fixtures.
- **Document existing behavior (golden master) alone.** Duplicates the
  baseline manifest with worse signal — per-function snapshots are
  brittle and don't tell you *why* a value matters.

## Scope: target modules

Four modules, ~480 lines of source code total. (Originally five; see
"Scope adjustments during planning" below.)

| Module                              | Lines | Reason to test now                                                                |
| ----------------------------------- | ----- | --------------------------------------------------------------------------------- |
| `prepare_climate_data_catalog.py`   | 132   | Regression test for M2b's `to_yml` workaround (`yaml.safe_dump` bypass).          |
| `extract_historical_climate.py`     | 181   | One unit-shaped bug in followups.md: silent truncation (M3 followup).             |
| `setup_time_horizon.py`             |  97   | Forcing-config YAML builder with chunksize branching. Pure logic + thin hydromt wrap. |
| `metrics_definition.py`             |  69   | Pure scoring functions. Edge cases (NaN, empty, single-element) are unit-shaped.  |

### Scope adjustments during planning

- **Dropped `prepare_build_config.py`.** Inspecting the source revealed
  it's a script (top-level code using `snakemake.input.template` etc.),
  not a function — not unit-testable without an extract-to-function
  refactor that lives in M3's territory. Defer.
- **Updated `setup_time_horizon.py` themes.** Module name suggests
  "date math" but the function `prep_hydromt_update_forcing_config` is
  actually a forcing-config YAML builder. Real testable logic is the
  chunksize branches (>1e6 → 1, >2.5e5 → 30, >1e5 → 100, else 365)
  and the precip-source branching (era5 vs eobs).
- **Dropped one of three xfails.** The "rule ignores `starttime`/
  `endtime` from config" issue (M5 followup) is a Snakefile-wiring bug,
  not a function bug — `prep_historical_climate` itself does honor its
  params. That belongs in an M5 integration test, not an M02c unit test.

**Out of scope** (deferred to owning milestone):

- Plotting modules (`func_plot_signature.py`, `plot_proj_timeseries.py`,
  `plot_results.py`, `plot_map.py`, `plot_map_forcing.py`).
- Workflow-orchestration scripts M3-M5 will substantially rewrite
  (`get_stats_climate_proj.py`, `get_change_climate_proj.py`,
  `export_wflow_results.py`, `downscale_climate_forcing.py`,
  `setup_gauges_and_outputs.py`, `setup_reservoirs_lakes_glaciers.py`,
  `prepare_cst_parameters.py`, `prepare_weagen_config.py`,
  `copy_config_files.py` beyond the existing test).
- R weathergenr testthat — parked at M5 per roadmap.
- Coverage-percentage targets. Coverage is measured by which modules
  have unit tests, not by line %.
- CI workflow setup. No GitHub Actions today; tracked separately under
  "Deferred: Linux replication" / "CI".

## Test pattern

Replicate `tests/test_stage_data.py`. Specifically:

- **One test file per source module**: `tests/test_<module_basename>.py`.
- **Stub heavy deps via `sys.modules.setdefault`** at top of each test
  file, e.g.:
  ```python
  import sys, types
  sys.modules.setdefault("hydromt", types.SimpleNamespace())
  sys.modules.setdefault("xarray", types.SimpleNamespace(Dataset=object))
  ```
  Keeps unit tests fast and free of network or staged-data dependencies.
- **Inline fixtures per file.** Promote to `conftest.py` only when reused
  in 3+ files.
- **No Snakemake `script:` injection.** Tests call functions directly.
  Existing `test_cli.py` covers Snakemake-level integration via
  `--dry-run`.
- **No real CMIP6 or staged data.** The baseline manifest already covers
  end-to-end numerical correctness; unit tests target a different
  failure mode.

Existing fixtures in `tests/conftest.py` (`config`, `project_dir`,
`data_sources`, `model_build_config`) stay as-is for integration tests
like `test_model_creation.py`. M02c does not modify them.

## Per-module test themes

Themes, not exact case enumeration — final case list lives in the
implementation plan.

| Module                              | Themes                                                                                                                |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `prepare_climate_data_catalog.py`   | preprocess-hook preserved on write (regression), driver/metadata structure, chirps adds orography, hydromt to_yml round-trip (xfail) |
| `extract_historical_climate.py`     | era5 path patches chunks under driver.options, driver-as-string conversion, chirps takes precip-only branch, params honored, truncation warning (xfail) |
| `setup_time_horizon.py`             | precip_source branching (era5/eobs), chunksize branches by staticmaps size, default chunksize without wflow_root, output YAML structure |
| `metrics_definition.py`             | Q7d_min / Q7d_maxyear on synthetic series, highpulse / lowpulse quantile thresholds, wetmonth_mean / drymonth_mean month-finding, BFI ratio |

Estimated total: **~30 tests across four new files, two strict xfails.**

## Bug-driven xfail discipline

Documented bugs in `dev/followups.md` that aren't fixed yet get a test
with `pytest.mark.xfail(strict=True, reason="...")`. When the bug is
fixed in M3 or M5, the test starts passing, `strict=True` flips it to a
CI failure, and whoever fixed it must remove the marker. This keeps
known bugs visible without letting the suite rot.

Two xfails introduced by M02c:

| Test                                                                          | xfail until fixed in   | Reference                |
| ----------------------------------------------------------------------------- | ---------------------- | ------------------------ |
| `prepare_climate_data_catalog`: `to_yml` round-trip preserves `preprocess`    | M3+ (upstream hydromt) | `dev/followups.md` M2b   |
| `extract_historical_climate`: warns when extracted window is shorter than ask | M3                     | `dev/followups.md` M3    |

The third originally-planned xfail (`extract_historical_climate` honors
`starttime`/`endtime` from config) was dropped during planning — it's
a Snakefile-wiring bug, not a function bug. The function already honors
its params; the rule definition hardcodes them. That regression test
belongs in an M5 integration test.

Pattern follows `tests/test_cli.py` precedent — that file has two strict
xfails for the same reason (Snakefile cleanups parked for M3).

## Exit criteria

- Four new test files in `tests/`, one per target module.
- `pytest tests/` passes, with the two xfails marked strict.
- All existing tests still pass (`test_cli.py`, `test_model_creation.py`,
  `test_stage_data.py`).
- The `test_stage_data.py` mocking pattern is documented in a short
  comment block at the top of one new test file so M3-M5 contributors
  see the convention immediately.
- `dev/roadmap.md` M02c section flipped to `**Status.** Sealed <date>.`
- Tag `m02c-tests` cut on the sealed commit.

## Risks and open questions

- **Mocking gets too brittle.** If `extract_historical_climate.py`'s
  coupling to hydromt makes the `sys.modules` stub impractical, fall
  back to a tiny real netCDF fixture (~1 KB) under `tests/data/`.
  Decision in passing during implementation.
- **xfail tests start passing accidentally.** `strict=True` catches
  this — they fail CI and force the marker removal. No silent decay.
- **M3 starts before M02c finishes.** Then M3 absorbs the remaining
  tests in its own milestone. Manageable.

## Tag

`m02c-tests`.
