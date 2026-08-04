Master Brief — R9 generated project tree

Revision: 2026-08-04, initial; phase briefs linked same day; validation ladder
revised same day (owner, after the R8 retrospective on slow cycles).

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
- **Match the check to the blast radius** (`AGENTS.md` § Validation ladder). The
  full suite runs **once, before merging** — not per phase, not per commit. R8's
  slow cycles came from re-proving what the previous run already proved. Each
  phase brief names its own narrow scope; run that scope, not its neighbours'.
- **Trim regression runs, never falsifiers.** The two are different instruments.
  A regression run re-proves untouched behaviour and is the thing to batch; a
  falsifier is the only evidence that a claimed *absence* holds, and R9 has three
  failure modes no unit test reaches — a missed `LOG_RULES` entry (silent, not an
  error), undeclared engine artifacts (`--dry-run` structurally cannot see them),
  and the concurrent-log race (needs two members running at once). They cost
  seconds to minutes. Cutting them makes R9 faster and less safe at the same
  time.

### Human gates

1. **Comparator gate — PAUSE after P1.** Present the R9 path map applied to the
   two-tier inventory ruled in the map doc (*The inventory the map is validated
   against*), showing the intended post-migration paths: the **declared tier**
   from the Snakefiles' `output:` declarations, which P1 produces, and the
   **observed tier** — one clean three-workflow run from the primary checkout,
   snapshotted as a sorted path list. Both must show zero unmapped paths. P1 may
   complete with the observed tier unverified; **this gate may not close that
   way**, because undeclared engine artifacts appear in no declaration and
   `--dry-run` structurally cannot see them. Do not begin moving files until the
   owner confirms the map reproduces the design's tree. A map that is wrong in the
   same direction as the migration is undetectable afterwards.
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
- **Full suite** — `pixi run test-full` **once, before merging**. Not per phase.
  If it comes back red, bisect with `pixi run test-fast` (~3.3 min a probe, versus
  ~8 min) — which is the reason to keep the tiers distinct rather than collapse
  them.
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

### Validation ladder (binds every phase)

Measured 2026-08-03 and recorded in `pyproject.toml`: the suite is ~492 s, and 15
`slow` tests account for ~302 s of it — 61% of the cost in 3% of the tests. 13 of
those 15 sit inside `workflow_contract or process_isolation`, which `pixi.toml`
already tiers on. The tiers existed and were unused.

| Rung | Frequency | Command | Cost |
|---|---|---|---|
| 1 Narrow | per edit | the phase's **named scope** only | seconds |
| 2 Integration | per commit, **only if** a Snakefile or `script:` signature changed | `pixi run test-cli` | ~30 s |
| 3 Phase gate | end of each phase | `pixi run test-fast` | ~3.3 min |
| 4 Full gate | **once**, before merge | `pixi run test-full` | ~8 min |
| 5 Tree gates | landing gate; the re-record once in P3 | three-workflow run, `semantic_tree_diff`, `check_baseline` | expensive |

Rung 3 replaces the per-phase full suite. Rung 4 fires once for the program.

**Not in scope for R9:** `pyproject.toml` records an open decision on whether a
default deselect wires to the `slow` marker or to the
`workflow_contract`/`process_isolation` scheme — *"not both"*. Closing it would
change what `pytest tests/` means, which is a contract. R9 **uses** the existing
`pixi` tasks and leaves the default alone.

### Phase brief index

- P1 — Comparator and tooling — [`phase-1-comparator-task-brief.md`](phase-1-comparator-task-brief.md) — not started
- P2 — Tree migration — [`phase-2-tree-migration-task-brief.md`](phase-2-tree-migration-task-brief.md) — not started
- P3 — Result tables and rule 3.11 — [`phase-3-result-tables-task-brief.md`](phase-3-result-tables-task-brief.md) — not started
- P4 — Fingerprint and experiment lifecycle — [`phase-4-fingerprint-task-brief.md`](phase-4-fingerprint-task-brief.md) — not started
- P5 — Conventions and docs — [`phase-5-conventions-docs-task-brief.md`](phase-5-conventions-docs-task-brief.md) — not started
