Master Brief — R9 generated project tree

Revision: 2026-08-04, initial.

### Goal

Migrate every artifact under `project_dir` to the semantic tree accepted in
`dev/milestones/r09/project-tree-design.md` (v10), using the completed path map
in `dev/milestones/r09/migration_project-tree.md`, without changing any
scientific value. Five phases, each independently verifiable.

### Subsystem map

| Phase | Owner | Input | Expected output |
|---|---|---|---|
| P1 — Comparator and tooling | `python-engineer` | the path map | `build_r09_path_map` in `dev/scripts/semantic_tree_diff.py` + orphan-store reporting in the prune tooling; unit-tested, no `project_dir` touched |
| P2 — Tree migration | `python-engineer` | P1's comparator | all three workflows write the v10 tree; three clean `--dry-run`s; `semantic_tree_diff` clean against the R9 map |
| P3 — Result tables and rule 3.11 | `python-engineer` | P2's tree | `q_indicators.csv` / `basin_indicators.csv`, `RT_*.csv` gone, rule 3.11 renamed, `validate_hm7` fixed, baseline re-recorded exactly once |
| P4 — Fingerprint and experiment lifecycle | `python-engineer` | P2's tree | pointer-derived `model_reference.yml` + WF3 pre-simulation check; experiment ID collision and immutability rules |
| P5 — Conventions and docs | `technical-writer` | P1–P4 landed | `naming.md` §8 gains a generated-outputs rule and §7 is narrowed; `README.rst`, `AGENTS.md`, `docs/` updated |

### Sequencing

Strictly sequential. Every phase but P1 edits
`Snakefile_climate_experiment`, so concurrent phases would contend for one file;
they are not to be run in parallel worktrees.

- **P1 before P2** — the comparator is the only regression detector for the move.
  Migrating first leaves nothing to compare a pre/post tree against, which is how
  R7 nearly lost its gate.
- **P2 before P3** — the baseline re-record must happen once, against the settled
  tree. Re-recording before the paths stop moving spends the single allowed
  re-record on an intermediate state.
- **P2 before P4** — P4 writes `model_reference.yml` into the experiment's
  `config/`, whose location P2 establishes.
- **P3 and P4 both before P5** — documentation follows behaviour.

### Shared constraints

- **`config/project.yml` is out of scope for the whole program.** It is settled
  framing but **not built**; adopting it moves config ownership from the toolbox
  into the project. Any phase that finds itself needing it must stop at Gate 2.
- Do not change any scientific calculation. Every moved artifact must be
  value-equivalent; the only intended value change in the program is none.
- Follow design principle **P9**: where the design differs from what the code
  emits, the emitted structure wins unless the design states a reason. Four such
  divergences were already found and corrected; assume more exist.
- **Exactly one rule identifier changes in this whole program**: rule 3.11,
  `export_wflow_results` → `derive_wflow_indicators`. R9 renames that rule's
  outputs to `q_indicators.csv` / `basin_indicators.csv`, so both halves of its
  old name become wrong — a milestone renames what it falsifies. Every other rule
  identifier is **frozen**. The nine-rename sweep and its verb vocabulary are
  **R10** (`dev/milestones/r10/rule-naming-design.md`), a separate milestone, and
  must not be smuggled into any phase here.
- Every rule rename or log-path change updates `LOG_RULES` in the **same edit**.
  An unlisted label is not an error — `merge_logs` silently drops the section and
  strands its parts.
- Run the pipeline from the **primary checkout**, never a task worktree
  (`AGENTS.md`): two `.snakemake/` metadata stores over one `project_dir`
  disagree, and both hold locks.
- `pixi.toml` / `pixi.lock` and `Manifest.toml` are not to be edited.

### Human gates

1. **Comparator gate — PAUSE after P1.** Present the R9 path map applied to a
   materialized pre-migration tree, showing the intended post-migration paths.
   Do not begin moving files until the owner confirms the map reproduces the
   design's tree. A map that is wrong in the same direction as the migration is
   undetectable afterwards.
2. **Scientific-delta gate — PAUSE before recording any new baseline.** Present
   the discharge comparison and any map-level differences. The program's premise
   is that nothing changes value; a non-zero delta means either a defect or a
   design decision the owner has not made. Mirrors the gate the WF1 spatial work
   used.
3. **Landing gate — PAUSE before merging.** Present all five phase reports, three
   workflow dry-runs, a full three-workflow run, and the falsifier results named
   below.

### Cross-cutting validation

Whole-program checks no single phase can perform:

- **Pre/post tree comparison** — `dev/scripts/semantic_tree_diff.py` against the
  R9 path map, whole-tree, clean modulo a written MISSING/EXTRA allowlist.
- **Baseline** — `python dev/scripts/check_baseline.py check`. Red by
  construction from the first P2 commit until P3's re-record; that window must be
  named in the phase reports, not discovered. Re-record **exactly once**, after
  Gate 2.
- **Full suite** — `pytest tests/` once before merging, per the repository's
  validation ladder. Not per phase.
- **Full three-workflow run on the seed config** — the only check that exercises
  undeclared engine artifacts, which `--dry-run` cannot see.
- **Falsifiers for the two properties the program asserts.** Both assert an
  *absence* and neither is reachable by the existing suite:
  - *"No member's Wflow log is overwritten by another's."* Run two members
    concurrently in one batch; assert **content attribution** — each log
    describes its own `rlz_<r>_cst_<c>` and no other. A file count passes
    trivially and is not the test.
  - *"Sharing a dataset and window does not re-run shared work per experiment."*
    Run two experiments sharing both; assert the shared climate-store rule's
    input set is byte-identical for each, and that the second schedules zero
    store jobs.

### Phase brief index

- P1 — Comparator and tooling — `<PLACEHOLDER: phase-1 brief>` — not started
- P2 — Tree migration — `<PLACEHOLDER: phase-2 brief>` — not started
- P3 — Result tables and rule 3.11 — `<PLACEHOLDER: phase-3 brief>` — not started
- P4 — Fingerprint and experiment lifecycle — `<PLACEHOLDER: phase-4 brief>` — not started
- P5 — Conventions and docs — `<PLACEHOLDER: phase-5 brief>` — not started
