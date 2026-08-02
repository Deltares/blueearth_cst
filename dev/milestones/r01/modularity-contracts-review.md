# Roadmap and R01 Review

**Date.** 2026-05-09.

**Scope.** Review of the cleanup roadmap in `dev/roadmap.md`, with
special attention to the high-level plan and current R01 modularity
contracts phase. This review is based on the current repository structure,
the three root Snakefiles, the `src/` scripts they call, the test suite,
and the R01 design / implementation plan.

## Executive summary

The roadmap is directionally strong. The sequence is risk-aware:
replicate and fingerprint first, migrate the environment second, upgrade
load-bearing libraries before refactoring, add unit tests before touching
workflow internals, then formalize config contracts before the workflow-by-
workflow cleanup in R3-R5. That is the right order for a scientific
workflow where output drift is expensive to diagnose after the fact.

The main adjustment I recommend is tightening R01 before execution. The
design says "contract only, no behavior change", but the implementation
plan currently changes the config shape without updating every code path
that reads the config file directly. In particular, scripts invoked by
Snakemake receive `config_path` and parse the YAML themselves; several of
those scripts still expect flat keys. If R01 migrates only the Snakefiles,
`climate_projections` and `climate_experiment` can parse successfully at
the Snakefile layer and still fail later inside the called scripts.

The second important correction is verification semantics. The baseline
checker includes copied snake-config snapshots as formal `rule all` targets.
Changing the config schema will change those YAML hashes even if scientific
outputs are bit-for-bit identical. A literal "zero baseline diff" is
therefore not a valid R01 exit criterion unless the checker or manifest
policy is adjusted.

## Current structure and behavior

The repository is still organized around three root-level Snakemake entry
points:

- `Snakefile_model_creation` builds a Wflow model with HydroMT, patches
  reservoirs / lakes / glaciers, adds gauges, prepares historical forcing,
  runs Wflow.jl, and produces model-performance plots.
- `Snakefile_climate_projections` computes monthly historical and future
  CMIP statistics, derives change factors by model / scenario / horizon,
  merges them, and produces summary CSV / netCDF / plots.
- `Snakefile_climate_experiment` generates weather realizations through
  R `weathergenr`, applies stress-test perturbations, downscales them for
  Wflow, runs Wflow for each realization / stress-test combination, and
  aggregates discharge outputs.

The current implementation is config-driven but flat. Each Snakefile has
its own copy of `get_config()`, reads the same global `config` namespace,
and re-parses `sys.argv` to recover `--configfile` so downstream scripts
can copy or parse the original YAML.

The important detail for R01 is that not all config reads live in the
Snakefiles. These scripts parse the snake config directly:

- `src/prepare_cst_parameters.py` reads `yml["temp"]` and `yml["precip"]`.
- `src/prepare_weagen_config.py` reads `yml_snake["realizations_num"]`,
  `yml_snake["temp"]`, and `yml_snake["precip"]`.
- `src/get_change_climate_proj.py` expects `time_horizon_hist` and
  `time_horizon_fut` to be comma-separated strings and calls `.split(", ")`.
- `src/copy_config_files.py` copies the raw config into project outputs,
  which makes config format part of the current baseline manifest.

Tests currently include fast unit coverage for selected stable utilities,
plus `tests/test_cli.py` dry-runs the three Snakefiles. Two Snakefile dry-
runs are strict xfails, intentionally deferred to R3.

## Roadmap review

The milestone decomposition is sound. The strongest parts are:

- Baseline-first discipline in M1, with output fingerprints rather than
  large reference artifacts.
- M2b inserted before refactoring, so R3-R5 do not target obsolete HydroMT /
  Wflow APIs.
- M2c inserted before workflow cleanup, giving R3-R5 a pattern for unit
  tests and strict xfails.
- R1 inserted before R3-R5, which prevents each workflow cleanup from
  inventing its own config ownership rules.
- Vertical-by-workflow cleanup in R3-R5, which limits blast radius and
  gives each milestone an owned baseline slice.

The main roadmap-level suggestion is to make R3 explicitly absorb the
R01 side effects that are already becoming cross-cutting:

- If R01 replaces the `sys.argv` configfile recovery with
  `workflow.configfiles[0]`, remove that from R3 deliverables or mark it
  already completed by R01.
- If R01 introduces `project/shared/workflows`, R3 should introduce a
  small config-access helper rather than letting three Snakefiles build
  independent local patterns again.
- R3 should include a dedicated "direct config-file readers" audit before
  further refactors. Otherwise the old flat schema can survive invisibly
  inside scripts.

## R01 design review

The contract-first approach is good. Keeping structure unchanged in R01 is
pragmatic because the repo does not yet have a fourth workflow, a module
registry, or an alternate hydrological backend. Formalizing ownership now
and deferring Snakemake `module:` composition keeps this phase small.

The proposed top-level sections also make sense:

- `project` for paths and external resources.
- `shared` for scientific knobs read by more than one workflow.
- `workflows.<name>` for owned workflow parameters.

However, the contract should be more explicit about keys that are currently
ambiguous:

- `shared.historical_window` is used by model creation and should eventually
  be used by climate experiment extraction, but projections has its own
  yearly historical range. That distinction is correct and should stay
  explicit in docs.
- `shared.wflow_outvars` is only read by model creation today, but it shapes
  the model output contract consumed downstream. Keeping it shared is
  defensible, but the reason should be stated.
- `project.data_sources_climate` is read only by climate projections, so it
  could live under `workflows.climate_projections`. Keeping it in `project`
  is reasonable if the intent is "external catalogs live under project";
  document that convention to avoid future bikeshedding.

## R01 implementation-plan issues

### 1. Direct config-file readers will break after sectioning

The implementation plan updates the three Snakefiles but does not update all
scripts that parse `config_path` directly. That creates parse-green,
runtime-red failures.

Concrete breakages:

- `src/prepare_cst_parameters.py` currently reads `yml["temp"]` and
  `yml["precip"]`. Under the proposed schema those live at
  `workflows.climate_experiment.stress_test.temp` and
  `.stress_test.precip`.
- `src/prepare_weagen_config.py` currently reads
  `yml_snake["realizations_num"]`, `yml_snake["temp"]`, and
  `yml_snake["precip"]`. Those move under
  `workflows.climate_experiment`.
- `src/get_change_climate_proj.py` currently expects historical and future
  horizons to be strings and calls `.split(", ")`. The plan intentionally
  changes these to lists. That requires either preserving string values in
  config or updating this script to accept lists.

Recommendation: add a R01 task before final verification: update the
direct config-file readers, or pass already-sectioned values from Snakemake
instead of giving scripts the whole config path. The lower-risk option for
R01 is a tiny compatibility helper local to the affected scripts:

- `experiment_cfg = yml["workflows"]["climate_experiment"]`
- `stress_cfg = experiment_cfg["stress_test"]`
- `projection_cfg = yml["workflows"]["climate_projections"]`
- normalize horizon values with a helper that accepts either `"1980, 2010"`
  or `[1980, 2010]` and returns two strings.

Do not rely on Snakefile dry-runs to catch this. Dry-run does not execute
the Python/R scripts.

### 2. "Zero baseline diff" is incompatible with copied config snapshots

`dev/scripts/check_baseline.py` includes these YAML targets:

- `{project_dir}/config/snake_config_model_creation.yml`
- `{project_dir}/config/snake_config_climate_projections.yml`
- `{project_dir}/config/snake_config_climate_experiment.yml`

`src/copy_config_files.py` copies the raw snake config text. After R01,
those copied YAML files will hash differently because the schema changed.
That is expected organizational drift, not scientific drift.

Recommendation: change the R01 exit criterion from "baseline check reports
zero diff" to one of these explicit policies:

- Preferred: scientific targets match exactly; config snapshot YAML diffs
  are documented in `dev/milestones/r01/baseline_diffs.md`, then the manifest is
  re-recorded as the new contract.
- Alternative: teach `check_baseline.py` an `--ignore-config-snapshots`
  mode for contract-only migrations, and run the normal full check after
  re-recording.

Without this correction, Task 7 is likely to fail for the right reason and
look like a regression.

### 3. Baseline project directory is inconsistent across the plan

`check_baseline.py` defaults to `examples/test_local`, while the R01 plan
asks for workflow runs against `tests/snake_config_model_test.yml` and
`config/snake_config_model_test.yml`, whose project dirs are
`tests/test_project` and `examples/test`.

Recommendation: every verification command should name the project dir
explicitly and use the same config/run outputs it intends to check, for
example:

```bash
pixi run python dev/scripts/check_baseline.py check --project-dir examples/test_local
```

or, if the milestone deliberately switches the canonical baseline project,
update the manifest policy and command text in the same commit. Avoid a
plan where Snakemake generates one project directory and the baseline
checker inspects another.

### 4. Pre-task clean-tree expectation conflicts with current state

The plan says `git status --short` should be empty, but
`dev/milestones/r01/modularity-contracts-plan.md` is currently untracked. That is not
a technical problem, but it will confuse an executing worker.

Recommendation: either commit the plan first, or change Step 0.1 to expect
that one untracked plan file until it is intentionally added.

### 5. Atomic migration is sensible, but add a mechanical key audit

The plan chooses an atomic migration for test config + all three Snakefiles.
That is reasonable because dual flat/sectioned fallback logic would add
temporary complexity.

What is missing is a mechanical audit after the migration. Add a step that
fails if old flat keys remain in Snakefiles or direct config readers.
Examples of keys to grep for:

- `project_dir`, `data_sources`, `data_sources_climate`
- `starttime`, `endtime`, `historical`
- `temp`, `precip`, `realizations_num`
- `future_horizons`, `models`, `scenarios`

The grep needs human interpretation because some names remain valid as
local variables, but it is still the fastest way to catch a missed flat read.

### 6. Contract docs should reflect actual produced outputs

The proposed contract docs are valuable, but a few output statements should
be checked against current `rule all` and intermediate contracts:

- `model_creation` `rule all` has three PNGs and the copied config as formal
  end targets. `performance_metrics.csv` may be important, but it is not
  currently a `rule all` target.
- `climate_projections` formal targets include summary netCDF, two CSVs,
  three plots, `gcm_timeseries.nc`, and copied config.
- `climate_experiment` formal targets include `Qstats.csv`, `basin.csv`,
  and copied config.

Recommendation: distinguish "formal end targets" from "important
intermediates" in the contract docs. The baseline manifest should continue
to track only formal end targets unless a representative fan-out sample is
intentionally added.

## Suggested R01 plan edits

1. Add a "Task 3a: update direct config-file readers" after the Snakefile
   migration. Include `prepare_cst_parameters.py`, `prepare_weagen_config.py`,
   and `get_change_climate_proj.py`.
2. Add a "Task 3b: old flat-key audit" using `rg` across `Snakefile_*`,
   `src/`, and `tests/`, with explicit review of remaining hits.
3. Change final verification to separate:
   - parse / dry-run verification,
   - unit tests,
   - scientific-output baseline check,
   - expected config-snapshot diff handling.
4. Add `dev/milestones/r01/baseline_diffs.md` as an expected artifact if the copied
   config YAML hashes change.
5. Make every baseline command pass `--project-dir` explicitly.
6. Update the roadmap R01 exit criteria so "zero diff" means zero
   scientific-output diff, not zero copied-config hash diff.
7. Adjust the R3 roadmap after R01 lands so already-completed configfile
   handling is not listed as future work.

## Recommended next step

Before executing R01, revise `dev/milestones/r01/modularity-contracts-plan.md` to
cover the direct config readers and the baseline-snapshot policy. Those are
small changes to the plan, but they prevent the milestone from producing a
schema that parses in Snakemake while failing inside scripts, or a baseline
check that fails for an expected documentation-level reason.

