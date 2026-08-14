---
title: Collapsing the per-workflow config copies is blocked by WF3's ancient() input
type: watch-item
area: config snapshot / wf3 guard
origin: owner question on the runs bin (2026-08-13)
created: 2026-08-13
updated: 2026-08-14
---

> [!note] Overview
> **What** — Replace the two whole-config copies under config/runs/ with one honestly-named verbatim copy, moving the per-workflow role into <workflow>/run_record.yml.
> **Why** — The per-workflow filename promises a scope the content does not deliver (design P1). Examined 2026-08-13 and found to cost far more than it looks.
> **Trigger** — WF3's drift guard is being reworked for an independent reason AND the baseline is being re-recorded for its own sake -- i.e. both costs are already being paid.

## Why this is on the board at all

The owner opened `test_rapid/config/runs/`, saw two byte-identical 4851-byte
files under per-workflow names, and asked whether that was duplication. Design
v3 §7 item 4 had already dispositioned this as **P1, left standing
deliberately**, on the theory that `run_record.yml`'s `projection` field plus
`config/runs/README.md` disarm the confusion. **They did not** — the redesign
was live on that tree (README, journal, both run records all present) and the
question was asked anyway.

The mitigation was strengthened instead (`7902d38`, lane/pipeline): the README
now names the trap explicitly and `tests/test_copy_config_files.py` pins the
wording. This item exists so the *structural* fix is not re-proposed from the
same surface reading — which both the owner and the assistant made — without
the three costs below in view.

**Re-asked 2026-08-14, and the strengthened README did not prevent it either.**
The owner proposed two variants in one sitting: flatten `config/runs/` into
`config/`, and nest the flat copies under `config/runs/<workflow>/`. Both trip
the SAME wire as cost 1 — the first moves
`config/runs/snake_config_model_creation.yml` out of `runs/`, the second moves
it down into `<workflow>/` — so the item now covers three proposals, not one.
That is twice in two days from two different readings of the same directory,
which is the evidence that **prose mitigation has been tried and does not
work**; the next attempt should not be a third README revision. Answered on
screen from this note; nothing structural changed. The positive reason to keep
`runs/` was also recorded that day and belongs with the costs: the bin is the
only boundary in `config/` between *archived copies of what came in*
(`catalogs/`, `templates/`, `basin_data/`) and *records of what the run did*,
and it is what makes "everything here is written by the run" true of a whole
directory rather than of scattered files.

## What was measured (2026-08-13)

- `test_local/config/runs/snake_config_{model_creation,climate_projections}.yml`
  are **byte-identical**, 3155 bytes each. On `test_rapid`, 4851 each. Both
  files are the **whole** source config, not the workflow's section of it.
- The WF3 drift guard reads only its guarded sections from each snapshot:
  `project`, `shared.basin`, `workflows.model_creation` from WF1's copy and
  `workflows.climate_projections` from WF2's
  (`check_project_consistency.py:41-42`). **Whole-config content is therefore
  not a correctness requirement of the guard** — it is a cost argument, not a
  correctness one, and should not be defended as the latter.

## The three costs, in order of severity

1. **`Snakefile_climate_experiment:572` declares
   `wf1_snapshot = ancient(wf1_snapshot_path)` — a MANDATORY rule input.**
   Renaming or collapsing that path fails every existing project tree at DAG
   build with `MissingInputException` until WF1 is re-run. Per design v3 §7
   item 11 that failure class fires **no journal line and not even `onerror`**,
   because it is raised before the handler block is reached. A silent-breakage
   migration in exchange for a naming fix. This applies equally to the
   apparently-cheaper variant `config/runs/<workflow>/source_config.yml` — same
   rename of the same declared input.
2. **The capability lost is the one that just proved its worth.** The guard
   compares the live experiment config against *each workflow's own* snapshot,
   so "WF1 built under config A, WF2 ran under config B" is detectable only
   while two files exist. [[t2608131718]] exists precisely because WF1's copy
   refreshed and WF2's did not, which is how drift latent since 2026-08-12
   became visible. Collapsing to one file requires re-pointing the guard at
   `<workflow>/run_record.yml` — a guard rewrite, not a rename.
3. **Baseline re-record, forced.** `dev/baseline/manifest.json` fingerprints
   all three flat copies (lines 9, 13, 27). It must be re-recorded from the
   **primary checkout** with both WF1 and WF2 run against `test_local` — which
   [[t2608131718]] already defers, and which no worktree may do.

Plus a mechanical wart: `.snakemake/` metadata is per working directory and
shared by all three Snakefiles, so two rules writing one path would invalidate
each other's provenance and re-fire X.01 on every alternation.

## Doc surface, if this is ever done

`dev/reference/workflows/rule-index.md` (3 sites), `dev/roadmap.md:1177`,
`dev/reference/workflows/{model_creation,climate_projections,climate_experiment}.md`,
`dev/scripts/semantic_tree_diff.py`'s inventory, and the three manifest entries.
Sealed milestone records under `dev/milestones/r07/` keep their old spellings by
design — check `dev/reference/sealed-records.yml` before editing any of them.

**Separately (not part of this item):** several of those workflow reference docs
already carry **pre-R7 paths** — `climate_projections.md:98` gives
`{project_dir}/config/snake_config_climate_projections.yml`, missing the `runs/`
bin, and `climate_experiment.md:121` and `model_creation.md:137` want checking
too. That is a live-reference staleness defect under AGENTS.md's "keep
configuration references current", independent of whether this item ever moves.
