# Task Brief — P1: provenance core

### Context

`AGENTS.md`; design `design-v3.md` §5.2, §5.3, §5.4, §5.7.

- `shared/provenance.py` is a contract surface with test consumers; it is
  imported by all three Snakefiles.
- Today `effective_config_document` digests the **whole** parsed config
  (`provenance.py:137-141`) and **does** include `advanced_settings` — the
  design keeps advanced settings inside `effective_config_sha256` to match.
- `scripts/run_workflows.py:424-433` already implements git metadata with
  swallowed failures; it becomes an importer, not a second definition.

### Goal

Land the pure-Python foundation — projection document, two digests, toolbox and
environment identity, journal append — with no Snakefile or writer changes yet.

### Non-goals

No Snakefile edits (P4). No `copy_config_files.py` edits (P2). Do not delete the
bundle helpers yet — their callers still exist until P2/P4.

### Allowed scope

- **Permitted:** `blueearth_cst/shared/provenance.py`,
  `scripts/run_workflows.py` (helper relocation only),
  `tests/test_shared_provenance.py`, `tests/test_run_workflows.py`.
- **Forbidden:** `Snakefile_*`, `copy_config_files.py`, anything under `dev/`.

### Required changes (checklist)

1. Projection-scoped `effective_config_document(config, advanced_settings, projection)`;
   `schema_version` → 2.
2. `effective_config_sha256` (projection + advanced settings) and
   `configuration_inputs_sha256` (adds toolbox identity, environment hashes,
   referenced-input sha256s). Document in the docstring that the latter
   **excludes scientific data identity** (design §5.4).
3. `toolbox_identity()` with the three-step resolution order (git →
   `.toolbox-commit` → nulls), returning `commit`, `commit_source`, `dirty`.
4. `environment_file_hashes()` over `pixi.lock` and `Manifest.toml`; missing
   file → `null`.
5. `append_journal_line()`: `O_APPEND`, one `write()` per line; reader tolerates
   a torn final line.
6. `run_workflows.py` imports 3 and 4 instead of defining them.

### Validation

- Rung 1: `pytest tests/test_shared_provenance.py tests/test_run_workflows.py`.
- Rung 2 (new behavioural tests, all required):
  - digest split — a change to a referenced-input hash moves
    `configuration_inputs_sha256` and **not** `effective_config_sha256`;
  - identity resolution order, including **git absent** (monkeypatch the command
    to fail) → falls through to `.toolbox-commit` → to nulls;
  - journal **accumulation across two appends**, and a torn final line parsed
    without raising.
- Rung 4: `pixi run test-fast` at phase merge. `pixi run lint`, `format-check`.

**Falsifier for "one definition":** `grep -rn "rev-parse" --include=*.py .`
returning more than one implementation disproves the relocation.

### Acceptance criteria

Every function importable and tested; `run_workflows.py` behaviour unchanged
(its own tests still pass); no Snakefile touched.

### Task constraints

Preserve `run_workflows.py`'s existing swallow-all-failures behaviour — a
provenance helper must never crash a run.
