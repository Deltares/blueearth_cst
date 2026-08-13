# Master Brief — project config-snapshot redesign

Design: `dev/working/2026-08-13_config-snapshot/design-v3.md` (accepted
2026-08-13 after two external review rounds; review record in the same
directory). Canonical ruleset: `AGENTS.md`.

**Driver correction to design §6, binding on P4:** the design proposes rules
`1.16` and `3.17` for `write_run_metadata`. **Both numbers are already taken** —
WF1 `1.16 gather_benchmarks` / `1.17 gather_logs`, WF3 `3.17 gather_benchmarks` /
`3.18 gather_logs`. Per `dev/reference/naming.md` §8b ("DO NOT RENUMBER TO INSERT
A RULE. Use a letter suffix"), use **`1.15b`** and **`3.16b`**, placing each
before its workflow's gather rules so their log parts are still gathered.

### Goal

Replace the write-only content-addressed config bundle with a current-only,
correctly-scoped run record that stays fresh when the toolbox moves, records the
values hydromt was actually handed, and copies into the project only what the
toolbox repository cannot give back.

### Subsystem map

| Phase | Lane | Input | Expected output |
|---|---|---|---|
| P0 | devmeta | design §5.2 step 0 | Probe report: do `onstart`/`onsuccess`/`onerror` fire on normal, no-op, failed, and `--dry-run` invocations on the pinned Snakemake |
| P1 | pipeline | design §5.2–5.4, §5.7 | `shared/provenance.py`: projection document, two digests, `toolbox_identity()`, `environment_file_hashes()`, `append_journal_line()`; helpers moved out of `run_workflows.py` |
| P2 | pipeline | P1 | `copy_config_files.py`: bundle removed, `run_record.yml` written atomically, R4 per-file predicate, collision refusal |
| P3 | pipeline | design §5.6 | Post-normalization values-used records from rules 1.07 and 1.08 |
| P4 | pipeline | P0, P1, P2, P3 | Three Snakefiles wired: projections, parse-time digests, params threading, lifecycle hooks, rules `1.15b`/`3.16b`, bundle outputs dropped |
| P5 | **split** | P1 | `Dockerfile` `ARG TOOLBOX_COMMIT` → `.toolbox-commit` (pipeline); `.gitignore` entry (devmeta) |
| P6 | **split** | P4 | One-shot cleanup tool + tree inventory updated + fixture cleaned (`dev/scripts/**` devmeta; `tests/**` pipeline) |
| P7 | devmeta | P4 | `README.md` bundle section replaced; new `config/runs/README.md` |

### Sequencing

- **P0 first, blocking P4 only.** The journal contract is specified against
  observed handler behaviour; if the probe contradicts §5.2, P4's hook design
  changes and the design's open item 10 must be revisited before wiring.
  **It did** (2026-08-13, `p0-probe-result.md`): handlers do not fire on a
  "Nothing to be done" no-op, so hooks cover only invocations in which a job
  executed. P4 and §5.2/§5.7 await an owner decision; P1–P3 are unaffected.
  P0 ran in **devmeta**, not pipeline — its brief scopes the probe to `.tmp/`
  and its only committed artifact to `dev/working/`, both devmeta territory.
- **P1 blocks P2, P4, P5** — all three consume its function contracts.
- **P2 and P3 may run concurrently**: disjoint files (`copy_config_files.py` vs
  `build_wflow_model.py` + `setup_reservoirs_lakes_glaciers.py`).
- **P4 after P1+P2+P3**, because it declares their outputs as rule outputs; a
  Snakefile declaring an output no script writes fails at DAG build.
- **P6 after P4** — the cleanup's target set is defined by the final tree shape,
  and **the fixture must be cleaned before the inventory tests are rewritten**,
  or the rewritten tests bake the orphans in.
- **P5 and P7 may run any time after their input phase.**

### Shared constraints

- **Lane routing.** `blueearth_cst/**`, `tests/**`, `Snakefile_*`, `Dockerfile`
  → `lane/pipeline`. `dev/**`, `README.md`, `.gitignore` → `lane/devmeta`.
  P5 and P6 span both: split at the file boundary, pipeline leads.
  **`lane/pipeline` was claimed 2026-08-13T03:10 by another session — check
  `.lane-claim` before starting and report if still held.**
- Run the pipeline from the **primary checkout**, never a worktree
  (`AGENTS.md`, `.snakemake` divergence).
- Naming per `dev/reference/naming.md`: lowercase `snake_case` for generated
  outputs under `project_dir/`; no `_generated` suffix (design §5.10); letter
  suffix to insert a rule.
- Do not hand-edit `pixi.lock` / `Manifest.toml`; do not commit run outputs.
- No new dependencies without owner approval.

### Human gates

1. **Gate 1 — after P0, before P4.** If the probe contradicts design §5.2
   (handlers do not fire as specified), PAUSE: the journal mechanism needs an
   owner decision, not a workaround.
2. **Gate 2 — before P6 runs `--delete` against any tree.** Deletion is the one
   destructive action in this program. Report the dry-run target list and PAUSE
   for approval.
3. **Gate 3 — before pushing `main` to `origin`.** `pixi run test-full`, per the
   validation ladder in `AGENTS.md`.

### Cross-cutting validation

- **`pixi run test-fast` at each phase merge**; **`pixi run test-full` before
  any push** — this program touches Snakefiles, `script:` signatures and
  `shared/`, the three paths that tier guards.
- **`pytest tests/test_cli.py` after every phase that changes a Snakefile or a
  rule's declared input** — the only place a malformed rule surfaces.
- **`pixi run lint` + `pixi run format-check`** — CI gates, near-instant, and a
  pre-push hook enforces them.
- **Baseline (`check_baseline.py check`) once, at the end of P6.** The design
  claims **no baseline re-record**: all three fingerprinted flat copies and
  `q_indicators.csv` are untouched, and every added artifact is a new file.
  **Falsifier:** a non-empty diff from `check_baseline.py check` disproves it —
  and means either a fingerprinted target was modified or the manifest needs a
  deliberate re-record decision from the owner.
- **`pixi run tree-check --config test_case/snake_config_baseline.yml`** must be
  green after P6 and red before it (that redness is the migration's own proof).
- **End-to-end**: `pixi run python scripts/run_workflows.py --config
  test_case/snake_config_rapid.yml` after P4, from the primary checkout.

### Phase brief index

| Phase | Brief | State |
|---|---|---|
| P0 | `p0-probe.task.md` | **done — `p0-probe-result.md`. Gate 1 TRIPPED: handlers do not fire on a no-op.** |
| P1 | `p1-provenance-core.task.md` | not started |
| P2 | `p2-snapshot-writer.task.md` | not started |
| P3 | `p3-values-used.task.md` | not started |
| P4 | `p4-snakefile-wiring.task.md` | not started |
| P5 | `p5-deployment-identity.task.md` | not started |
| P6 | `p6-migration-inventory.task.md` | not started |
| P7 | `p7-docs.task.md` | not started |
