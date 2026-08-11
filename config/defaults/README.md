# Rule-read configs

**Referenced from a project config and read by a rule. Changing one changes a
run.** These are not templates — do not copy them, point a config at them (or at
your own edited copy elsewhere).

| File | Consumer |
| --- | --- |
| `wflow_build_model.yml` | `Snakefile_model_creation` — default for `workflows.model_creation.model_build_config`; rule 1.06 `prepare_spatial_maps` and rule 1.07 `build_wflow_model` |
| `wflow_update_waterbodies.yml` | `Snakefile_model_creation` — default for `waterbodies_config`; rule 1.08 `add_reservoirs_lakes_glaciers` |
| `weathergen_config.yml` | `Snakefile_climate_experiment` — `default_config` for rule 3.10 `prepare_weathergen_config`, declared as BOTH an `input:` and a `params:` |

`weathergen_config.yml` carries the `input:` declaration for a reason: it was a
params-only read until 2026-08-05, so editing it changed nothing until something
else forced a rerun — rule 3.10 stayed satisfied, its generated config stayed
stale, and 3.11 kept generating realizations from superseded settings.

## Why this is a separate directory

Until 2026-08-11 these three sat in `config/templates/` alongside the scaffolds.
That name asserted one kind of file and the directory held two, so a rule input
read as something you were meant to copy and edit.

## The project tree keeps the name `templates/`

`<project_dir>/config/templates/` is unrelated to this split and did **not**
move. It is a generated provenance bin: rule 1.01 snapshots the shipped configs
a run actually used into it, so a finished project can state what it was
evaluated against (`blueearth_cst/model/copy_config_files.py`, which derives the
destination from `params.config_dir` and never from the source path). The word
means "verbatim snapshot of a shipped config" there, not "scaffold".

## Old paths

These files have moved twice: pre-R6 flat `config/`, then `config/templates/`,
now here. Both earlier spellings are normalized onto the current path by
`_COPIED_CONFIG_PATH_MAP` in `blueearth_cst/experiment/check_project_consistency.py`
and `COPIED_CONFIG_PATH_MAP` in `dev/scripts/semantic_tree_diff.py`, so a project
config or reference tree from either era still compares clean.
