Task Brief — R9 P4: model fingerprint and experiment lifecycle

### Context

Canonical ruleset: `AGENTS.md`. Master brief:
`dev/milestones/r09/project-tree-task-brief.md`. Runs after P2's tree is settled.

- This is the only phase adding **new capability**. Everything else in R9 moves
  or renames.
- The fingerprint is **pointer-derived, not a fixed file list**
  (`project-tree-design.md`, *Model reproducibility contract*): hash
  `wflow_sbm.toml` plus every model-root file its path-valued keys resolve to,
  discovered rather than enumerated. A fixed triple would be correct only for the
  TOML shape the toolbox emits today — any hydromt `setup_*` that writes a
  TOML-referenced side file adds a runtime input outside it.
- Deliberately excluded, because Wflow.jl does not read them at run time:
  `staticgeoms/`, `hydromt.log`, `hydromt_data.yml`.
- Experiment IDs already slugify to lowercase with underscores. WF3 already has a
  drift guard (rule 3.00b) whose sentinel paths must stay experiment-invariant.

### Goal

Every experiment records which model state it used, and WF3 refuses to simulate
against a changed model. Experiment creation cannot silently reuse or overwrite,
and an experiment's configuration stops being editable once it has run.

### Non-goals

- **`config/project.yml` — explicitly out of scope** (master shared constraint).
  If this phase appears to need it, stop at Gate 2.
- No multi-model support, no `runs/` hierarchy, no execution-attempt manifests.
- No change to the existing 3.00b drift guard's sentinel paths.

### Allowed scope

**Permitted** — `Snakefile_climate_experiment`; a new module under
`blueearth_cst/shared/` for the digest; `blueearth_cst/experiment/**`; new tests.

**Approval-gated** — anything altering rule 3.00b's declared inputs or sentinel
paths, since those carry the incremental-execution constraint.

**Forbidden** — `config/project.yml`; the model build itself; `dev/baseline/**`.

### Required changes (checklist)

1. A digest function: resolve `wflow_sbm.toml`'s path-valued keys against the
   model root, sort by relative path, hash paths plus contents, and emit an
   explicit **absence marker** for an optional input that is not present.
2. Refuse to hash anything resolving **outside** the model root — a pointer that
   escapes is an error, not a silently widened digest.
3. Write `experiments/<id>/config/model_reference.yml` carrying the relative
   model path and the digest.
4. Recompute and compare **before WF3 performs simulation work**; fail loud on
   mismatch, naming the changed input.
5. ~~Experiment ID allocation~~ — **SUPERSEDED 2026-08-04** by
   [`phase-4-commit-4-task-brief.md`](phase-4-commit-4-task-brief.md). This item
   named a generated default `stress_test_<YYYYMMDD>` that appears nowhere in
   the codebase (the real one is `<project-basename>_<YYYYMMDD>`) and required
   atomic reservation where no reservation step exists.
6. ~~`experiment.yml` immutability~~ — **SUPERSEDED** by the same brief. Nothing
   writes `experiment.yml`; the file is named in design tree v10 and by nothing
   else, so this is new capability rather than a rule over an existing artifact.

### Commit plan

Staged so each commit is independently runnable and the guard cannot land before
the thing it guards.

| # | Subject | Paths | Invariant preserved |
|---|---|---|---|
| 1 | `r09: add the pointer-derived model digest` | new shared module, tests | pure function, no caller yet — testable in isolation |
| 2 | `r09: write model_reference.yml per experiment` | WF3 Snakefile, experiment scripts | the reference exists before anything reads it |
| 3 | `r09: fail WF3 on model drift before simulating` | WF3 Snakefile | the check lands only once every experiment has a reference to check |
| 4 | `r09: rule experiment id collisions and config immutability` | experiment creation path, tests | lifecycle rules are separately revertible from the fingerprint |

### Validation

**Named scope — run this and nothing else:** the new digest module's tests plus
the WF3 experiment tests. WF1 and WF2 suites are not in scope; this phase reads
the model root but changes nothing that builds it.

1. **Narrow** — the named scope and the falsifier set below (per edit).
2. **Integration** — `pixi run test-cli` after each Snakefile edit (commits 2–3).
3. **Phase gate** — `pixi run test-fast` once, at phase end. **Not** the full
   suite.

**Falsifiers.** The fingerprint's whole purpose is to detect an absence of
change, so tests that only confirm detection are half the job:

- **Pointer discovery** — adding a path-valued key to the TOML must bring a new
  file into the digest. Editing **that file's content alone**, TOML untouched,
  must change the digest. This is the property a fixed file list fails.
- **Exclusions hold** — modifying `staticgeoms/`, `hydromt.log` or
  `hydromt_data.yml` must **not** change the digest.
- **Optional state** — presence and absence of `instate/instates.nc` must give
  different digests, via the absence marker rather than by omission.
- **End to end** — an existing experiment fails *before simulation* after the
  live model changes; a newly created experiment succeeds.
- **Collisions** — a duplicate user-supplied name is rejected; a same-day
  default collides to `_v2` then `_v3`; resume of an existing ID allocates
  nothing.
- **Immutability** — `experiment.yml` is writable before the first successful run
  and refused after it.

### Acceptance criteria

- Every falsifier above passes, and the pointer-discovery one was shown to fail
  against a fixed-file-list implementation.
- Rule 3.00b's declared inputs and sentinel paths are unchanged.
- `pixi run test-fast` green; three clean `--dry-run`s. The full suite runs
  once for the program, in P5 — not here.
- Rollback: if the pre-simulation check cannot be made to fail loud *before*
  simulation work begins, revert commit 3 and report — a check that runs after
  the work is not a guard.

### Output requirements

A phase report listing each falsifier and its result, the digest's file-set for
the seed model, and the startup IO cost the check adds (dominated by hashing
`staticmaps.nc`).

### Task constraints

- The digest must be deterministic across platforms — sorted relative paths, no
  filesystem ordering, no absolute paths in the hashed material.
- Do not copy the model into the experiment; the reference is a path plus digest.
