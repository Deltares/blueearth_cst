---
title: "Running the workflows"
subtitle: "Run a whole assessment or one workflow at a time"
---

Run all enabled workflows with `scripts/run_workflows.py`. To run one workflow,
use `scripts/run_workflow.py <workflow_name>`. Scenario generation and simulation
delegate to their dedicated runners.

The runners hide routine Snakemake startup chatter during execution. Warnings,
errors and workflow diagnostics remain visible, and Snakemake's log files retain
the full record. Dry runs and `--verbose` keep native startup output.
Routine map-extent padding notes are omitted; map-overlap warnings remain visible.
WF4 reports checking, calculating and saving indicators by their selected set names.
Its final summary links each set's results, identifies reused results, and reports
missing or invalid values. If one set fails, it states whether earlier results
remain available. Internal request and publication identities remain in the
retained manifests and the execution's `wf4-indicators.log`.

| Workflow | What it does |
|---|---|---|
| Historical climate (`analyze_climate`) | Compare climate data sources before choosing the forcing for the model; optional |
| Model (`build_model`) | Build Wflow-SBM and run it with historical climate data |
| Projections (`analyze_projections`) | Calculate CMIP6 changes to compare with the stress-test range |
| Scenarios (`generate_scenarios`) | Generate climate series for the chosen perturbations |
| Simulation (`simulate_system`) | Run the model with those series and calculate indicators |

The runner uses this order: historical climate → model → projections → scenarios
→ simulation. Model building and scenario generation are both needed before
simulation. Historical climate analysis is optional. CMIP6 projections provide
context for the tested changes and never supply the generated scenarios.

## Create an external case

For a new external case, create its configuration set from this toolbox checkout's
`config/templates/example-basin/` starter. The cases repository needs an
`applications/` directory; it does not need a template copy. From the toolbox root:

```powershell
pixi run create-case my-basin --cases-root C:/path/to/cst-cases --longitude 12.5 --latitude -3 --purpose "First-order basin assessment" --dry-run
```

Earlier toolbox versions read `cst-cases/templates/example-basin/`. The starter
now lives at `blueearth_cst:config/templates/example-basin/`; update manual-copy
procedures to use that path. The current scaffolder ignores external template copies.

The starter defaults to `config/catalogs/deltares_data_pdrive.yml` and
data under `P:/wflow_global/hydromt`; mount the Deltares P-drive before
running. Its basin sources use `rivers: hydro_rivers_lin` and
`lulc: globcover`. The scaffold inherits the catalog and source settings
from the toolbox-owned starter, so new cases use the template shipped with the
toolbox version being run. Existing application configs are not updated automatically.

Remove `--dry-run` after reviewing the preview. The command creates
`applications/my-basin/`, sets portable input and output paths, replaces the
outlet coordinates, and starts fresh `configured; not run` case notes. It
preserves the template's scientific settings and starts with historical climate
analysis alone. The template's `basin.region` upstream-area threshold (`uparea`,
km²) is retained; review it for the outlet before running. Register the case in
the application inventory after the workflows complete. Use `--catalog` to override the
template's data catalog with a
`toolbox:` locator, case-config-relative path, or absolute path. Review catalog coverage and climate/scenario
settings before running. Existing applications are refused;
outputs under `runs/my-basin/` are neither created nor modified.

The scaffold uses `runs/my-basin/` as its output root. Choose another path in
`project.project_dir` or with `--project-dir` when retaining a separate assessment.
Existing paths, including `active/` folders, remain valid and are not moved or archived.

Each scaffolded YAML file starts with a commented title, filename, project name,
creation timestamp (local time with UTC offset), and controlling workflow ID and name.
The project file identifies shared settings and workflow enablement. Workflow labels
follow `workflows.*.config_path`, including custom filenames. All files and
`INSTRUCTIONS.md` share one creation timestamp; existing case files are not updated.

Parameters also receive short definitions and `Input:` or `Options:` hints, drawn from
the toolbox's maintained configuration templates. Existing explanations and settings
are preserved. Custom parameters without template help receive a basic YAML type hint.

The generated `INSTRUCTIONS.md` is a copy-paste quick reference for previewing, running,
rerunning incomplete jobs, unlocking, and recalculating indicators. Output paths follow the commands;
the configuration summary is last. After
selecting forcing, disable historical climate analysis before enabling downstream
workflows. The scaffold does not run workflows or commit files.

## Run everything enabled in the project file

```bash
pixi shell
python scripts/run_workflows.py --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3
```

The runner follows the `enabled:` choices in the project file and stops if a
workflow fails. It records the result of each attempt under
`<project_dir>/config/_engine/invocations/`, including checks and failed
runs. Set `enabled: false` for a workflow you want to skip. Check that any
earlier results used by a later workflow still match your current project
settings.

Historical climate, model building, and projections reuse completed steps when
their consumed settings and dependency contents are unchanged. You can rename
or move project/workflow configuration files, update their `config_path` links,
or enable other workflows without rebuilding those steps. Paths to scientific
data must still resolve to the same inputs. Each launch archives the original
configuration files and paths separately from the stable execution view. Identical
captures reuse their original configuration snapshot; each launch still records its
own command, timing and outcome. A changed capture creates an immutable snapshot,
so earlier attempt references remain valid even when the latest configuration changes.
Changing relevant settings or dependency contents still invalidates affected
work. Existing runs may rebuild once after upgrading to this execution-view
format; subsequent equivalent configurations reuse the staged inputs.

Inspect the latest WF0–WF2 settings in the exact captured files under the named workflow
folders; immutable snapshots and stable execution inputs live under `config/_engine/`.
WF3 and WF4 retain their existing collection/experiment configuration records.
Completed replacement directories and per-launch contexts are removed after their
contents or outcomes have been retained. These operational files do not provide run history.

For an existing case, inspect configuration maintenance without changing files:

```bash
pixi run python scripts/maintain_config_archives.py --project-dir /path/to/case-output
```

Add `--apply` to retain its available WF0–WF2 snapshots and remove completed
launch/replacement bookkeeping. Source configurations,
execution inputs, attempt history and scientific outputs are preserved. The command
reports unavailable historical references; it cannot reconstruct records already deleted.
It takes the project execution lock, so use it when the case is idle.

To preview the enabled workflows, add `-- --dry-run` to the command. A dry run
may show only part of a workflow until inputs from an earlier step exist.

## Regenerate figures after visual changes

When only styling, titles or captions change, target the plotting rule using the same project config
and output root as the completed run. From the toolbox root, preview first:

```powershell
pixi run python scripts/run_workflow.py analyze_projections `
  --config C:/path/to/project.yml --project-dir C:/path/to/existing-run `
  --target plot_climate_projections --dry-run -- --force --notemp
```

Check the scheduled jobs, then remove `--dry-run` to render. `--force` reruns the selected target;
`--notemp` preserves temporary inputs. Avoid `--forceall`, which also forces upstream jobs.
Missing or stale inputs can still schedule upstream work: inspect the dry-run before proceeding.

Substitute the workflow and target below. Repeat `--target` to select several rules in one workflow.
Replace `<source>` with its configured ID, for example `plot_climate_diagnostics_era5`.

| Figures | Workflow | Target |
|---|---|---|
| WF0 source diagnostics | `analyze_climate` | `plot_climate_diagnostics_<source>` |
| WF0 comparison | `analyze_climate` | `plot_climate_comparison` |
| WF1 spatial maps | `build_model` | `plot_basin_map` |
| WF1 forcing | `build_model` | `plot_model_forcing` |
| WF1 evaluation | `build_model` | `plot_model_evaluation` |
| WF2 projections | `analyze_projections` | `plot_climate_projections` |

Retain the diagnostic tables/map fields, projection-analysis tables, forcing and spatial inputs.
WF1 retains `<model>/run_default/output.csv` by default, so evaluation and table exports can reuse
the full-precision native responses. Older projects where this file was deleted require one historical
simulation to restore it; rounded exported tables cannot replace it.

Currently, WF0 comparison recomputes its statistics from retained daily series, WF1 forcing recomputes
plotting summaries, and WF1 evaluation recomputes indicators. These are combined calculation/rendering
steps. Other listed targets reuse prepared products. Title and caption inventories are refreshed with
the figures. Set project-level `plotting.annotation` to `none`, `caption` or `title` to choose
the annotation at the existing plot paths; changing this parameter schedules the plotting jobs.

## Run one workflow

From the toolbox root, after `pixi shell`, use the individual runner for any
workflow. Enable the selected workflow in the project config:

```bash
python scripts/run_workflow.py analyze_climate --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3
python scripts/run_workflow.py build_model --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3
python scripts/run_workflow.py analyze_projections --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3 --keep-going
python scripts/run_workflow.py generate_scenarios --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3
python scripts/run_workflow.py simulate_system --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --target all --cores 3
```

The individual runner launches only the named workflow; it does not run missing
prerequisites. WF3 delegates to the generation runner and supports only target
`all`. WF4 delegates to the simulation runner and needs a ready model and scenario
collection. The dedicated runners remain available.

The flags most users need are:

| Flag | Effect |
|---|---|
| `--config` | project file with basin and workflow settings |
| `--project-dir` | directory where this project writes its results |
| `--cores` | CPU budget for the run |
| `--target` | simulation outputs to produce; `all` is the usual choice |
| `--dry-run` | show planned steps without running them |

The all-workflow runner passes options after `--` to each enabled workflow.
For example, `-- --dry-run` checks each workflow. The individual runner accepts
`--dry-run` directly for all five workflows and forwards options after `--` to
the selected owned runner.

If Snakemake reports that the directory is locked after an interrupted run,
first make sure no other process is using the project. Then clear the lock for
the affected workflow, for example:

```bash
snakemake --unlock -s build_model.smk --configfile test_case/project_config_rapid.yml
```

If a run exhausts memory, lower `--cores` and rerun it. Larger basins and longer
scenario collections need more memory and disk space; timings and resource use
are recorded with the execution logs described below.

### Execution logs and benchmarks

Every supported launcher attempt gets a new directory, including dry runs and startup failures
when the project root is known:

```text
logs/2026-10-20_11-21-55Z__all-workflows/
  summary.md
  console.log
  wf0.log ... wf4.log
  wf0-benchmarks.md ... wf4-benchmarks.md
  wf0-benchmarks.tsv ... wf4-benchmarks.tsv
  wf4-indicators.log
  _parts/
```

Folder names use `YYYY-MM-DD_HH-MM-SSZ__<scope>`, with `Z` denoting UTC.
The scope describes the selected workflows: `all-workflows`, a single name such as
`analyze-climate`, or ordered workflow IDs such as `wf1-wf4` for a subset.
Dry runs append `__dry-run`. When no workflows are selected or configuration fails before
selection is known, the scope is `workflows`. A same-second name collision appends a short invocation ID.
Outcomes remain in the summary so paths stay stable. New attempts use this naming scheme;
existing folders and their recorded paths remain unchanged.

Open `summary.md` first. It links the canonical invocation records, workflow logs, benchmarks and
resource diagnostics. `console.log` retains launcher and child output in observed order, without
terminal colours or repeated progress redraws. Workflow logs retain the detailed rule/member view.
Only collected outputs appear: dry runs and reused steps need not produce rule logs or benchmarks.
Missing measurements are not zero costs, and unknown reuse remains labelled unknown.

Managed launches merge rule logs and benchmarks when the invocation finishes. These diagnostic files
are not required DAG targets, so a fresh execution folder does not schedule gather rules when all
scientific outputs are up to date. Direct Snakemake runs retain their gather targets.

WF1's output metadata refreshes when settings, referenced-file contents, environment files or upstream
products change. A toolbox commit or Git dirty-state change alone is recorded in the invocation archive
and journal without scheduling the metadata rule or relabelling reused outputs.

Initial runner preparation messages use two-space indentation without timestamps or module labels,
including elapsed-time updates such as
`  Preparing configuration and experiment checks... (0:00:15 elapsed)` and the final check duration.
The timer updates on one terminal row; captured output contains one row with the final duration.
Reused WF3 collections use the same green `Up to date` verdict
as the other workflows, without separate scenario-request or reuse rows.
WF2 shows reference-period warnings when executing an analysis, and omits them on up-to-date runs.

Change checks across WF0–WF4 use the canonical configuration, file-content, code-inventory and
environment helpers. Each workflow compares its consumed inputs; reusable observations are shared
within a checking phase, with no persistent check cache or alternate fingerprint format.

WF4 observes scientific inputs afresh on each invocation. A completed simulation reuses its frozen
native output layout after model, code, environment, preparation, settings and selected runs pass
their checks, avoiding Julia startup for layout reconstruction. Indicator sets share fresh code and
environment observations within each planning or publication phase; execution still checks independently.

WF4 retains an experiment's original simulation identity and provenance when only execution-diagnostic
code changes. Scientific input or simulation-code changes require a new experiment. A refusal
before workflow execution skips indicator publication; partial job failures still allow completed
indicator sets to publish independently.

Benchmark Markdown starts with the recorded workflow elapsed time and a table grouped by rule:
job/sample counts, summed job time, median, slowest job and peak individual-job RAM. Expand the
individual-job section for CPU and I/O measurements; the adjacent TSV retains precise samples.
Summed job durations can overlap and are not workflow elapsed time. Peak job RAM is not concurrent
workflow memory. WF4 reports available Julia load, compilation and later-member timings separately.

Each workflow's `_parts/` namespace belongs to its child invocation; WF3 source and generation
phases share that child. Successful gathers remove their merged parts; failures retain partial
logs and benchmark fragments. Retrying creates another execution directory and never merges old
parts. Previously written top-level workflow logs and `benchmarks/` reports remain historical;
they are not reassigned to new executions.

Configuration archives keep their existing locations under `config/`; resource traces remain under
`config/_engine/invocations/`. WF2 additionally copies the logs and benchmarks of the runs that produced
its current files into the product's `_engine/execution/`; see
[which run produced these files](outputs.md#which-run-produced-these-files). Removing an execution's diagnostic folder does not remove its
configuration provenance or scientific outputs.

The all-workflow console includes the selected experiment and its action in the
opening settings, describes subbasin outlets with their upstream-area threshold
in km², and distinguishes disabled workflows from pending ones. Long settings
and command paths wrap within the terminal width; redirected output uses 100 columns.

On color-capable terminals, launcher and workflow rows retain their colour scheme and animated
progress bars while console capture is active. Warning rows use orange without a `WARNING` column.
Redirected output stays plain; `NO_COLOR` disables colours while retaining terminal progress updates.
Saved transcripts omit terminal colours and repeated redraws; detailed workflow logs retain warning levels.
Coordinate adjustments share one heading, followed by each location's original and resolved coordinates.

The final summary totals the toolbox warning rows counted during this invocation
and lists the contributing workflows. WF3's total includes its source and generation
phases. The `commands` row points to a `.commands.txt` receipt beside the invocation
record. Copy its location-setting line and one workflow block to rerun through the
supported runner using the current configuration and experiment selection. The receipt
preserves complete arguments and uses PowerShell
continuations on Windows or shell continuations elsewhere. Credential-like values
are redacted; replace those placeholders before use. Hard-wrapped display paths
are for reading and must not be pasted as commands.

When a job fails, the console prints a prominent `FAILED` block naming its rule
(with the toolbox rule ID when available), Snakemake job ID, and individual log
paths alongside the original error details. Concurrent failures and
failed grouped jobs identify each affected job. The closing workflow summary
also points to the retained log parts. Errors before any job starts retain
Snakemake's original diagnosis.

::: {.callout-note}
## Run a project from one toolbox checkout

Use the same toolbox checkout for repeat runs of a project. Running one project
from two copies of the toolbox can give conflicting decisions about which
results need to be refreshed.
:::

## Rebuild outputs for testing

To recreate results in an existing project directory, use `--rebuild`:

```powershell
# Preview the erasure plan without changing files or starting workflows.
pixi run python scripts/run_workflows.py --config <config> --rebuild -- --dry-run

# Display the plan and ask Continue? [y/N]: before erasing anything.
pixi run python scripts/run_workflows.py --config <config> --rebuild

# Confirm explicitly for an automated test run.
pixi run python scripts/run_workflows.py --config <config> --rebuild --yes
```

This is a **project-wide, permanent reset**, including outputs of disabled workflows.
It erases `data/spatial/`, `data/climate/`, `models/hydrology/wflow/`, `scenarios/`,
and `experiments/`. It also removes the active WF0–WF2 configuration captures,
their pending replacement state, and generated execution configs. Archive transaction journals for
erased scenario collections and experiments are also removed, including stale journals whose archives
are already missing. No backup is made.
The source configuration and seeds are unchanged; only enabled workflows run afterward.
All experiments and their allocation state are removed, so automatic numbering starts again.

Source configuration, referenced source datasets, unrelated project paths, logs, invocation history,
and historical configuration snapshots are preserved. Project-local extracted climate stores are
generated outputs and are erased; external source datasets and acquisition caches are untouched.
Keep custom files outside the generated namespaces: their contents are treated as output. A configured
input overlapping one of these namespaces, a linked path, repository metadata, or a changed plan causes
refusal before erasure.
The project execution lock remains held throughout reset and workflow execution.

Press Enter or answer anything other than `y` or `yes` to cancel. Non-interactive invocations require
`--yes`; `--yes` alone is invalid. The rebuild dry-run only previews erasure, rather than building the
Snakemake DAG against the old results. It writes no run records. A failure during erasure or subsequent
execution stops the run without rollback; the invocation record identifies whether reset completed.

For WF4, enable WF1 and WF3 as well, because their retained inputs will be erased. Include simulations
in WF4's operations; indicators-only cannot rebuild deleted model responses. `--rebuild` cannot combine
with `--new-experiment`, `--recover-reservation`, forwarded config/directory overrides, or bookkeeping
modes such as `--unlock`. Use `--new-experiment` instead when previous experiments should remain available.

## Select, resume or create an experiment

With no `experiment_name`, the first run reserves `stress_test_01`. A plain rerun
selects the highest allocated number, whether complete or interrupted. Unfinished simulations
continue there if their scientific inputs still match; intact completed outputs are reused.

For another experiment, pass the flag to either execution runner:

```bash
python scripts/run_workflows.py --config <config> --new-experiment --note "revised perturbation grid"
python scripts/simulate_system.py --config <config> --new-experiment --target all
```

Set `experiment_name: drought` to use the `drought_01`, `drought_02` family.
An exact name such as `drought_02` pins that identity and cannot combine with
`--new-experiment`. The flag creates an experiment; it does not force regeneration
of an unchanged ready scenario collection.

If scientific inputs changed and the flag was omitted, the runner refuses before
producers start, for both complete and interrupted numbered experiments. Notes,
resource settings and indicator choices do not redefine those scientific inputs.
Deleted allocations remain consumed; numbers are never automatically recycled.

`scripts/suggest_experiment_name.py <config>` and dry-runs only preview selection.
They neither reserve directories nor edit YAML. For an interrupted reservation
or paired intent/archive publication, `--recover-reservation` finishes only a
verified retained recipe. It cannot repair partial WF3 generation or missing or
corrupt native responses; retain that evidence for diagnosis.

Completed batch checkpoints can be reused after an ordinary failed run. A hard
stop while writing a native CSV or restoring a batch may leave partial outputs
that need diagnosis; reservation recovery does not repair those files.

To skip unperturbed realizations in WF4, set `include_unperturbed: false` in its
`project_config_simulate_system.yml` simulation file. The default is `true`. WF4 then prepares
forcing and simulates only perturbed members; WF3 keeps the complete collection.
Existing run IDs stay unchanged. Changing this switch requires `--new-experiment`.
Indicators are calculated only for simulated members and their design-point groups.
Indicators needing unperturbed reference runs (such as wet/dry reference-month indicators)
require `true`; otherwise select a set without them or use `operations: [simulations]`.
Indicators-only runs always use the retained simulation's members.

## Calculate different indicators from existing simulations

You can calculate different indicators from completed simulations without rerunning the model.
The selected experiment must have a completed `_engine/simulation.json` record, even for a dry run.

Open your case's simulation settings file: `project_config_simulate_system.yml` in the starter.
If your case uses another filename, find it under `workflows.simulate_system.config_path` in
`project_config.yml`. The rapid example uses `project_config_rapid_simulate_system.yml`.

1. Keep the existing `experiment_name`.
2. Set `operations: [indicators]`.
3. In `indicator_sets`, list the sets you want to calculate.

For example, if your completed experiment is `stress_test_01`, the relevant settings are:

```yaml
# project_config_simulate_system.yml
experiment_name: stress_test_01
operations: [indicators]
indicator_sets: [default]
```

Then run:

```bash
python scripts/simulate_system.py --config <config> --cores 3
```

The runner reads the retained model responses and writes each selected set to its own
`results/indicator_sets/<set-id>/` folder.
You do not need `--target`; the runner infers it.
You can also use the all-workflow runner with the other four workflows disabled.
See [Reading the outputs](outputs.md) for the result files.

An experiment is tied to its scenario collection, model, and simulation settings.
Use `--new-experiment` when those simulation inputs change.
For a fresh experiment, use `operations: [simulations, indicators]` to run the model before
calculating indicators.
Enabling other workflows does not change `operations: [indicators]` into a simulation request.

## Next

- [Reading the outputs](outputs.md) — what the run wrote, and how to tell
  whether it is still current.
