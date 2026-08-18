# Migration — the workflow entry points and their config keys (2026-08-14)

The three Snakemake entry points were renamed to verb-first `.smk` files, and
the `workflows.<name>` config keys and every derived path followed.

**If you have a project config, it will fail at parse time until you rename its
three `workflows:` subsections.** That is deliberate: the alternative — renaming
the files while freezing the keys — leaves a config surface that disagrees with
the thing it configures, which is the defect this repo already carries in
`workflows.model_creation.output_locations`.

## What to change in your config

```yaml
workflows:
  build_model:            # was: model_creation
    enabled: true
    ...
  analyze_projections:    # was: climate_projections
    enabled: true
    ...
  run_stress_test:        # was: climate_experiment
    enabled: true
    ...
```

Nothing inside those sections changed. `workflows.run_stress_test.stress_test`
reads a little oddly, and it is correct: the workflow is named for what it does,
the section for what it configures.

## What to change in your commands

| was | now |
|---|---|
| `snakemake ... -s Snakefile_model_creation` | `snakemake ... -s build_model.smk` |
| `snakemake ... -s Snakefile_climate_projections` | `snakemake ... -s analyze_projections.smk` |
| `snakemake ... -s Snakefile_climate_experiment` | `snakemake ... -s run_stress_test.smk` |

`pixi run run-workflows` and the `pixi run dag-wf*` tasks are unchanged — they
resolve the new names themselves.

## What moves inside an existing `project_dir`

A run under the new names writes to new paths; it does not migrate the old ones.
Either re-run the workflow, or rename these by hand:

| was | now |
|---|---|
| `logs/wf1_model_creation.log` | `logs/wf1_build_model.log` |
| `logs/wf2_climate_projections.log` | `logs/wf2_analyze_projections.log` |
| `logs/wf3_climate_experiment_<exp>.log` | `logs/wf3_run_stress_test_<exp>.log` |
| `config/runs/snake_config_model_creation.yml` | `config/runs/snake_config_build_model.yml` |
| `config/runs/snake_config_climate_projections.yml` | `config/runs/snake_config_analyze_projections.yml` |
| `config/runs/model_creation/run_record.yml` | `config/runs/build_model/run_record.yml` |
| `config/runs/climate_projections/run_record.yml` | `config/runs/analyze_projections/run_record.yml` |
| `<exp>/config/snake_config_climate_experiment.yml` | `<exp>/config/snake_config_run_stress_test.yml` |
| `<exp>/config/catalogs/data_catalog_climate_experiment.yml` | *(no longer produced — delete it)* |
| `<exp>/config/runs/climate_experiment/<digest>/` | `<exp>/config/runs/run_stress_test/<digest>/` |

`config/runs/journal.jsonl` records a `workflow` field per line; old lines keep
the old spelling, which is correct — they record runs that happened under it.

The generated climate catalog has no new name because it has no successor file:
rule 3.13 was removed on 2026-08-18 and rule 3.14 now writes a `temp()`
one-entry catalog per member beside that member's TOML. An existing project
carries the old file as a leftover; nothing reads it, and deleting it is safe.

## Why

The old names were nouns, and two were vague about what the workflow does:
`model_creation` also *runs* the model, and `climate_projections` computes change
factors that are a plausibility overlay, never a driver. The `Snakefile_` prefix
was also nonstandard and extensionless, which is why both `.editorconfig` and
`.zed/settings.json` carried explicit workarounds so the files would not open as
plain text.

The rename was done alongside the addition of a fourth workflow, because adding
one already forces edits to `run_workflows.py`'s `WORKFLOW_ORDER`, `test_cli.py`,
`plot_workflow_dag.py`'s digit map, the shared-producer symmetry tests,
`check_baseline` and the config template — every file the rename touches.

Design, alternatives and the owner rulings:
`dev/working/2026-08-14_climate-workflow-split/design.md` §4 and §5.8.

## What deliberately did NOT change

- **Rule numbers.** WF1/WF2/WF3 keep digits 1/2/3; the incoming fourth workflow
  takes `0`, so no rule reference in any existing document goes stale.
- **The sealed workflow reference docs.** `dev/reference/workflows/`'s
  `climate_projections.md` and `climate_experiment.md` are sealed records, kept
  because they are unedited; migrating a sealed document's paths makes a stale
  document read as current.
- **`dev/` records** — milestones, reviews, decisions, task notes, the pre-board
  archive — and `CHANGELOG.md` / `DEVLOG.md`. They record what was true when
  written.
- **Legacy project-tree directory names** inside `semantic_tree_diff.py`'s
  migration maps (`climate_experiment/`, `climate_projections/cmip6/`). Those
  name directories in *older* project trees, not workflows.
