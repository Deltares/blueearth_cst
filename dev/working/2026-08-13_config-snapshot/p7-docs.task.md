# Task Brief — P7: documentation

### Context

`AGENTS.md` ("Keep configuration references current" — a stale path in a
document someone reads to do their job is a defect, not a record); design
`design-v3.md` §5.10. Depends on P4.

- `README.md` ~170–186 documents the bundle, its 12-hex directory naming, and
  `referenced-files.json` — all of which cease to exist.
- Per-bin `README.md` files are the house pattern (`config/defaults/`,
  `config/templates/`, `config/basemap/` each have one).
- The one genuine trap in the project tree: `snake_config_model_creation.yml`
  sits under a directory called `config/`, looks exactly like a file you would
  edit and re-run, and editing it silently does nothing.

### Goal

Leave no document describing the removed bundle, and disarm the editability
trap where a reader meets it.

### Non-goals

No changes to `AGENTS.md`'s repo map beyond what the removal makes false. Not a
tutorial on provenance.

### Allowed scope

- **Permitted (`lane/devmeta`):** `README.md`, and a new
  `<project_dir>/config/runs/README.md` template shipped from the repo.
- **Forbidden:** `dev/` sealed records — check
  `dev/reference/sealed-records.yml` before editing anything under `dev/`; a
  document listed there is frozen and `tests/test_sealed_records.py` enforces it.

### Required changes (checklist)

1. Replace `README.md` ~170–186 with the new model: current-only
   `run_record.yml`, the journal, and the R4 copy rule.
2. New `config/runs/README.md`: "everything here is written by the run; edit the
   source config instead", naming `run_record.yml` as the projection-bearing
   artifact.
3. Grep the tree for surviving references to the removed names and fix every
   live one in the same commit:
   `grep -rn "referenced-files\|snapshot_bundle\|source\.yml\|effective\.yml"`
   across `README.md`, `docs/`, `AGENTS.md` and code comments.

### Validation

- Rung 1: `pytest tests/test_sealed_records.py` — proves no frozen record was
  edited.
- Rung 4: `pixi run test-fast`.

**Falsifier for "no live reference survives":** the grep in item 3 returning a
hit outside `dev/` milestone records and this task directory. Milestone records
under `dev/` are *deliberately* unedited history and are the expected hits —
distinguish them from live guidance rather than sweeping them.

### Acceptance criteria

Grep clean outside the historical-record exceptions; both READMEs accurate
against a freshly-run project tree.

### Task constraints

Do not update `dev/` milestone or review records to match the new design — they
are the baseline past work was checked against and are valuable *because* they
are unedited (`AGENTS.md`).
