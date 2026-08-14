# Task Brief — P6: migration and tree inventory

### Context

`AGENTS.md`; design `design-v3.md` §6 (Migration). Depends on P4.

- Existing project trees — including `test_case/test_local`, the fixture the
  fixture-dependent test layer runs against — hold bundle, template and catalog
  copies that become **undeclared orphans** the moment the inventory changes.
- `dev/scripts/semantic_tree_diff.py` declares the tree contract;
  `tests/test_project_tree_inventory.py` pins bundle paths today.
- Ordering is load-bearing: **clean the fixture before rewriting the inventory
  tests**, or the rewritten tests bake the orphans in as expected state.
- Precedent: R9 P2 found stale files only an mtime sweep caught, because they
  sat under directories the path map routes wholesale.

### Goal

Update the tree contract and ship a one-shot cleanup that brings existing
projects to the new shape, without deleting anything the design keeps.

### Non-goals

Not a general-purpose pruner. Not a replacement for `prune_series_cache.py` or
`prune_climate_store.py`.

### Allowed scope

- **Permitted (`lane/devmeta`):** `dev/scripts/semantic_tree_diff.py`, a new
  `dev/scripts/prune_config_snapshots.py`.
- **Permitted (`lane/pipeline`):** `tests/test_project_tree_inventory.py`.
- **Approval-gated:** running the tool with `--delete` against any tree —
  released only by **Gate 2** in the master brief.
- **Forbidden:** deleting anything under `<exp_dir>/config/catalogs/`.

### Required changes (checklist)

1. Inventory: remove bundle / template / tracked-catalog snapshot paths; add
   `run_record.yml` (×3), the two values-used records, the two
   `run_metadata.json` sidecars, and `journal.jsonl` **whitelisted as an
   undeclared side effect** (it is written by a hook, not a rule).
2. `prune_config_snapshots.py`, **report-only by default, `--delete` explicit**
   — the house pattern of the two existing prune tools.
3. **Scope, exactly:** bundle directories
   `<project_dir>/config/runs/<workflow>/<12-hex>/` and
   `<exp_dir>/config/runs/climate_experiment/<12-hex>/`; and under
   `<project_dir>/config/templates/` and `<project_dir>/config/catalogs/`
   **only**, files byte-identical to a tracked toolbox file. Anything else is
   **reported, never deleted**.
4. **`<exp_dir>/config/catalogs/` is never touched** — it holds the generated
   experiment catalog the design keeps. A pattern match on `config/catalogs/`
   across the whole tree would delete it.
5. `config/basin_data/` copies are kept — outside-repo files the predicate
   copies by design.
6. Clean the fixture, **then** rewrite the inventory tests.

### Validation

- Rung 1: `pytest tests/test_project_tree_inventory.py`.
- Rung 2: a test that the tool's `--delete` scope excludes
  `<exp_dir>/config/catalogs/` and `config/basin_data/`.
- Rung 3: `pixi run tree-check --config test_case/snake_config_baseline.yml` —
  **red before this phase, green after**. That transition is the migration's own
  proof; record both results.
- Rung 5: `check_baseline.py check` — **runs once, here**, and is the program's
  expensive rung.

**Falsifier for the design's "no baseline re-record" claim:** a non-empty diff
from `check_baseline.py check`. If it fires, a fingerprinted target was modified
— stop and bring it to the owner rather than re-recording the manifest.

**Falsifier for "the cleanup deletes nothing it should keep":** run `--delete`
against a **copy** of the fixture and diff against the original; any missing
`<exp_dir>/config/catalogs/` or `config/basin_data/` file disproves it. Do
this before Gate 2, on a copy, never on the live fixture.

### Acceptance criteria

`tree-check` green; baseline clean; the cleanup's dry-run list reviewed and
approved at Gate 2 before any deletion.

### Task constraints

Run the baseline gate from the **primary checkout** against
`snake_config_baseline.yml` — never the rapid tree. WF1 runs feeding the
baseline need `--notemp` (`AGENTS.md`).
