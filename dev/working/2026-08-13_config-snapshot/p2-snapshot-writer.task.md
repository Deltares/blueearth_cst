# Task Brief — P2: snapshot writer

### Context

`AGENTS.md`; design `design-v3.md` §5.1, §5.2, §5.5. Depends on P1.

- `copy_config_files.py:81` copies with `destination_dir / source_path.name` —
  two configured files sharing a basename overwrite one another. The bundle
  avoided this with hash-prefixed names (`:144`); the flat bins never did.
- Referenced paths (`project.data_sources`, `model_build_config`,
  `waterbodies_config`, the two observation keys) are **arbitrary** — a project
  may name a site-specific catalog outside the toolbox.

### Goal

Stop writing the bundle; write `run_record.yml` atomically; decide per file
whether to copy, using recoverability rather than which bin it lands in.

### Non-goals

No journal code here — lifecycle hooks own it (P4). No Snakefile edits.

### Allowed scope

- **Permitted:** `blueearth_cst/model/copy_config_files.py`,
  `tests/test_copy_config_files.py`.
- **Forbidden:** `Snakefile_*`, `shared/provenance.py` (P1 owns it).

### Required changes (checklist)

1. Delete `_write_snapshot_bundle` and every call to it.
2. Write `run_record.yml` per the design §5.2 schema, atomically (temp file +
   `os.replace`).
3. Implement the §5.5 predicate per referenced file: inside the toolbox
   checkout **and** tracked at `toolbox.commit` **and** clean → record identity
   only, no copy; **otherwise copy**.
4. **Null-commit branch:** when `commit` is null or the tracking query is
   unevaluable, the predicate **falls back to copy** (design §5.5, open item 9).
5. Role-stable destination names (`output_locations.csv`,
   `observations_timeseries.csv`); **raise** on an unexpected destination
   collision rather than overwrite.
6. Every copy's origin → archive mapping recorded in `referenced_inputs`.

### Validation

- Rung 1: `pytest tests/test_copy_config_files.py`.
- Rung 2 (new behavioural tests, all required):
  - predicate: tracked-and-clean → not copied; outside-checkout → copied;
    locally-modified tracked file → copied;
  - **degraded mode**: git absent (monkeypatched) → everything copied, record
    carries `commit: null, commit_source: null`;
  - **collision**: two configured observation paths sharing a basename → raises,
    and neither file is silently lost;
  - record is written atomically — no partial file after a simulated crash
    mid-write.
- Rung 4: `pixi run test-fast`; `pixi run lint`, `format-check`.

**Falsifier for "no input is silently lost":** configure two same-basename
observation files and assert the run raises. A green run that leaves one file in
`config/observations/` disproves the property.

### Acceptance criteria

No bundle directory is created by any code path; `run_record.yml` matches the
design schema; the predicate is exercised in all four branches above.

### Task constraints

Do not change the three flat `snake_config_*.yml` copy paths or their contents —
they are baseline-fingerprinted and read by the WF3 drift guard.
