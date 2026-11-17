---
title: "Changes and migration"
format: html
---

This reference covers moving existing cases and output directories to the current toolbox structure, and
the rules that apply when a trial is copied. The User Guide explains how to
[create a case and add a trial](../guide/new-case.md#create-the-case-files).

## Copying a trial

[Trial](../guide/glossary.md#trial) copying with `--from-trial` refuses inputs it cannot carry over safely.
Missing inputs, links, cross-trial inputs outside shared `data/`, retained-product references and schema-2
sources are refused with guidance. Migrate a schema-2 copy explicitly before using it as a source.
Review catalog-internal dataset paths and access separately; catalogs do not bundle their datasets.
Keep catalogs with relative dataset locators together with their datasets in shared case `data/`.
Copying a trial-local catalog with relative dataset dependencies is refused, rather than rewriting
HydroMT's catalog semantics or silently leaving datasets behind.

## The case README trial table

Creation adds one row to the case README's Trial/Purpose/Status table, initially `working`. No central
registry is updated. Change status manually to `retained` or `complete`; complete means the declared scope was
reviewed, not that every workflow ran. Preserve saved evidence for new
[experiments](../guide/glossary.md#experiment). Toolbox revision, source trial and run-note fields stay with
each trial.

## Copy and verify an existing case

Existing schema-2 and schema-3 split layouts remain runnable. The scaffold refuses a new case name
already present under `applications/` or `runs/` to avoid creating an ambiguous second case.
Migration is a separate, non-destructive operation:

1. Copy authored configurations and sidecars into a new case directory with a `default/` trial.
   Preserve the original configurations and results until verification is complete.
2. For schema 2, resolve each file path using its original base before explicitly changing the copy
   to schema 3. Review the [path reference](configuration.md#how-paths-are-read).
3. Set the copied trial's output root to `outputs`, and update copied YAML-relative input references.
   Keep shared inputs in case `data/`, with distinct inputs under the trial. Retain `toolbox:` locators.
4. Compose every workflow, inspect resolved input paths, and preview with the owned runner from a
   different invocation directory. Run a bounded workflow to check downstream execution paths.
5. Add the minimal README trial row and output ignore rule. Review and archive evidence separately;
   do not move or rewrite immutable product records as part of configuration migration.

An evidence archive preserves results, checks, configuration snapshots and provenance. A rerunnable
archive additionally requires the referenced datasets, catalogs and software environment. Record missing
external dependencies. Git ignore rules are not backups, and moving YAML alone does not relocate runtime
locators embedded in historical outputs.

## Output-directory layout versions

A new output directory records the [layout](../guide/glossary.md#layout) it was written with in
`config/_engine/layout.json` (layout 5 for this toolbox version); a directory without it uses layout 1. The
[run commands](../guide/glossary.md#run-command) refuse a directory written with a different layout: set
`project.project_dir` to a new folder and rerun. Existing directories are not converted.
