# Toolbox architecture

Last updated: 2026-11-14

Use this reference when changing workflow responsibilities, shared configuration or retained outputs.
The [workflow implementation map](toolbox-workflow-map.md) locates rules and their consumers; the
[workflow guide](../../docs/site/toolbox-reference/workflows.qmd) covers supported execution.

## Workflow responsibilities

| Workflow | Responsibility | Entry point |
|---|---|---|
| WF0 | Optional historical climate analysis without a hydrological model | `analyze_climate.smk` |
| WF1 | Build Wflow-SBM and run historical forcing | `build_model.smk` |
| WF2 | Derive CMIP6 change factors for a plausibility overlay | `analyze_projections.smk` |
| WF3 | Generate durable, model-independent scenario collections | `generate_scenarios.smk` |
| WF4 | Simulate the system and publish indicators from retained responses | `scripts/simulate_system.py` |

WF1 and WF3 provide independent prerequisites for WF4. WF2 does not choose or drive the scenario-neutral
stress-test grid. WF0 shares region and climate producers with other workflows; it can run first or be
omitted. [`scripts/run_workflows.py`](../../scripts/run_workflows.py) selects enabled workflows in
WF0–WF4 order and checks a consumer after preceding producers.

WF3 collections do not depend on an experiment name. WF4 selects explicit run identities and checks
retained inputs before reuse. An indicators-only operation uses complete retained simulation responses;
it does not generate scenarios or rerun the model. The
[single-workflow runner](../../scripts/run_workflow.py) and all-workflow runner dispatch owned WF3/WF4
execution; WF4 also has a [direct runner](../../scripts/simulate_system.py). Their operation and
target validation is part of the supported execution path; bare Snakemake does not provide it.
The [retained handoffs](../../docs/site/toolbox-reference/retained-handoffs.qmd)
describe the artifact interface. The [generator](toolbox-weather-generator-interface.md) and
[hydrological-model](toolbox-hydrological-model-interface.md) references own replacement-engine
requirements.

## Configuration ownership

The project YAML owns settings shared across workflows; each workflow's own settings file owns the
rest. A cross-workflow value is never read from another workflow's file.
[`config_composition.py`](../../blueearth_cst/shared/config_composition.py) and
[`config_paths.py`](../../blueearth_cst/shared/config_paths.py) own the exact schema, merge and path
rules. The [configuration change procedure](dev-configuration-changes.md) owns where a key belongs and
how to add, rename or retire one.

## Retained products and execution records

WF3 stores identified scenario collections; WF4 stores identified simulations, complete response
inventories and indicator sets. Retained readers validate content and identity, not just path
existence. Indicator recalculation uses the retained responses and does not cause a new simulation.
The [retained-handoffs reference](../../docs/site/toolbox-reference/retained-handoffs.qmd) owns the
product contract and recovery behavior.

Store each file once: workflows, collections and experiments reference shared data, configuration and
results by path and content hash instead of copying them, and that single copy of data and user-facing
files lives outside `_engine/`, which holds machine records only. A new duplicate needs a stated reason
and owner approval in its design or task.

WF4 forcing elevation is project-owned ancillary data, shared by content identity rather than copied
into each WF3 collection. [`shared_orography.py`](../../blueearth_cst/experiment/shared_orography.py)
publishes and checks the bytes;
[`wf4_ancillary_descriptor.py`](../../blueearth_cst/experiment/wf4_ancillary_descriptor.py)
binds the checked dependency and its HydroMT reader. Preserving an experiment includes preserving
its referenced project dependencies.

Supported WF0–WF2 launchers capture original configuration bytes before Snakemake parses execution
documents. [`workflow_archive_launch.py`](../../blueearth_cst/shared/workflow_archive_launch.py)
handles capture and admission;
[`workflow_config_snapshot.py`](../../blueearth_cst/shared/workflow_config_snapshot.py)
publishes and validates the archive. These archives describe each workflow's latest captured
configuration, not a complete history. Generated execution views and parsed settings have different
purposes from original source bytes. A source checksum is not a scientific product identity, and an
archive alone proves neither a successful run nor a portable rerun.

The supported runners write separate invocation outcomes through
[`invocation_history.py`](../../blueearth_cst/shared/invocation_history.py). These records cover
supported launcher attempts; they do not establish resume or fencing guarantees for arbitrary raw
Snakemake launches. WF3 and WF4 retain configuration snapshots within their identified collection
or experiment rather than in one project-wide slot.

Supported launchers start Snakemake and other project children through `run_project_child` in
[`windows_job.py`](../../blueearth_cst/shared/windows_job.py). The parent holds the project lock. On
Windows the child is created suspended and assigned to a job object before it can start descendants, so
the whole process tree ends with the parent. On other platforms a writing child is refused because no
equivalent lifetime guarantee is verified; only non-writing launches, such as previews, run there.
Snakemake children run through `snakemake_console.py`, which applies `bounded_windows_socketpair` from
[`windows_socketpair.py`](../../blueearth_cst/shared/windows_socketpair.py) to bound Python 3.12's
fallback socket-pair setup without patching installed packages. Tests:
[`test_windows_job.py`](../../tests/test_windows_job.py) and
[`test_windows_socketpair.py`](../../tests/test_windows_socketpair.py).

## Implementation and verification

This repository is a Python/R/Julia Snakemake workflow, with no API or frontend. `blueearth_cst/`
contains importable and rule-invoked modules; `scripts/` contains user entry points. `pixi.toml`
defines the environment, while `pyproject.toml` configures tools and does not declare a package build.
Ordinary tooling changes must not add packaging tables or a build backend as a side effect.

Keep the toolbox lightweight: a change that adds a Python, R or Julia package needs the owner's approval,
recorded in its design or task, before it lands. Add Pixi dependencies through `pixi.toml`; never
hand-edit `pixi.lock` or `Manifest.toml`. Julia is installed with juliaup and stays outside Pixi because
conda-forge has no win-64 build. `runtime.julia_version` in `config/advanced_settings.yml` is the pin the
workflows read; it must agree with the `install-julia` task and `Manifest.toml`
([`test_julia_runtime.py`](../../tests/test_julia_runtime.py)). weathergenr is installed outside the lock by
`pixi run install-rdeps`. Installed package versions enter the stage environments that WF3, WF4 and
indicator identities hash, so an upgrade changes those identities; see the
[run identity reference](toolbox-run-identity.md#code-inventories). The
[installation guide](../../docs/site/setup/install.md) covers user setup.

The [spatial foundation](toolbox-spatial-foundation.md) owns model-neutral region and vector products.
The [WF1 reference](wf1-model-construction.md) owns Wflow construction and mutable model-root ordering.
Historical climate figures use WF0 and shared climate plotting; model evaluation consumes model
responses and optional observations rather than maintaining a second climate-diagnostic pipeline.

Use the [validation ladder](dev-validation.md) to choose checks. Relevant contract tests include
[`test_run_workflows.py`](../../tests/test_run_workflows.py),
[`test_workflow_archive_launch.py`](../../tests/test_workflow_archive_launch.py) and
[`test_invocation_history.py`](../../tests/test_invocation_history.py), plus affected retained-artifact
reader and runner tests. This is navigation, not a separate test gate.
