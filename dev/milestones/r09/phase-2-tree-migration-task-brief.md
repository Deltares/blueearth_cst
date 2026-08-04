Task Brief — R9 P2: migrate the generated tree

### Context

Canonical ruleset: `AGENTS.md`. Master brief:
`dev/milestones/r09/project-tree-task-brief.md`. Released by master **Gate 1**.

- Target tree: `project-tree-design.md` v10. Row-by-row destinations:
  `migration_project-tree.md`. P1's `build_r09_path_map` is the comparator.
- Paths are **owned by the rules**, not by the scripts: `downscale_climate_forcing.py`
  derives `run_name` and `out_prefix` from the *declared* TOML path
  (`:41-55`), so the `rlz_<r>/` flattening needs no logic change there — only
  new templates in the Snakefile.
- The emitted Wflow TOML pointers (`input.path_static`, `state.path_input`,
  `input.path_forcing`) are written **absolute** and re-relativised by
  `hydromt_wflow` on write, so run-directory depth changes are absorbed
  automatically. Their *emitted strings* still change, which is why this phase
  needs a real run, not a dry-run.
- `check_baseline.py check` is **red by construction** for the whole of this
  phase. That is expected; P3 re-records once.

### Goal

All three workflows write the v10 tree, with `semantic_tree_diff` clean against
the R9 map and no change to any artifact's value.

### Non-goals

- No result-table renames, no `RT_*.csv` removal, no rule rename — that is P3.
- No fingerprint or experiment-lifecycle work — that is P4.
- No baseline re-record. Do not run `check_baseline.py record`.

### Allowed scope

**Permitted** — all three `Snakefile_*`; `blueearth_cst/**` path construction;
`scripts/plot_workflow_dag.py`; `tests/**` path expectations;
`dev/reference/contracts/*-seam.md`.

**Approval-gated** — anything implying `config/project.yml` (master Gate 2).

**Forbidden** — `dev/baseline/manifest.json`; `pixi.toml`/`pixi.lock`;
`Manifest.toml`; `build_r07_path_map`.

### Required changes (checklist)

1. `hydrology_model/` → `models/hydrology/wflow/`; `config/generated/*.yml` →
   that model's `config/`.
2. `spatial/` → `data/spatial/`; `climate_historical/` and
   `climate_projections/` → under `data/climate/`, **keeping** the store's
   `<source>_<window>` key and `cmip6/raw/` + `scalar/`.
3. Experiment: `weather_generator/` → `climate/weathergenr/` with `output/`
   retained as the directory name; `hydrology_runs/rlz_<r>/…` → flat
   `hydrology/wflow/{config,forcing,output}/rlz_<r>_cst_<c>.…`.
4. **In the same commit as 3:** set Wflow's `[logging] path_log` per member to
   `f"{out_prefix}{run_name}.log"` in `downscale_climate_forcing.py`.
5. DAG renders → `logs/dag/` and `experiments/<id>/logs/dag/`.
6. Update `LOG_RULES` and every `log:`/`benchmark:` prefix touched.

### Commit plan

Moves break every reference the instant they land, so each row is an atomic
move-plus-rewrite and must leave the tree runnable.

| # | Subject | Paths | Invariant preserved |
|---|---|---|---|
| 1 | `r09: move the wflow model under models/` | WF1 Snakefile, model scripts | model root resolves; WF3 still finds it |
| 2 | `r09: move spatial and climate data under data/` | WF1/WF2 Snakefiles, climate scripts | the store key stays experiment-invariant |
| 3 | `r09: restructure the experiment's engine subtrees` | WF3 Snakefile, `downscale_climate_forcing.py` | **`path_log` ships with the flattening** — otherwise concurrent members race on one log |
| 4 | `r09: route DAG renders out of config/` | `scripts/plot_workflow_dag.py` | generated artifacts leave the editable root |

Commit 3's invariant is the reason it cannot be split: the directory level and
the log path are one correctness unit.

### Validation

**Named scope by commit — do not run all four sets at every commit:**

| Commit | Narrow scope |
|---|---|
| 1 model → `models/` | WF1 model/build tests |
| 2 spatial + climate → `data/` | WF1 spatial + WF2 projection tests |
| 3 experiment subtrees | WF3 experiment tests |
| 4 DAG renders | plot/DAG tests |

1. **Narrow** — the row's scope only (per edit).
2. **Integration** — `pixi run test-cli` (per commit; every commit here edits a
   Snakefile, so this rung fires four times — it is ~30 s, not the suite).
3. **Phase gate** — `pixi run test-fast` once, at phase end. **Not** the full
   suite; that fires once for the program, before merge.
4. **Non-regression** — `semantic_tree_diff` against the R9 map, whole-tree,
   after a full three-workflow run. Once, at phase end.

**Falsifier — the concurrent-log race.** Claim: *no member's Wflow log is
overwritten by another's*. Run two members concurrently in one batch and assert
**content attribution**: each log describes its own `rlz_<r>_cst_<c>` and no
other. Counting files passes trivially once `path_log` is set and is **not** the
test. Run this deliberately with `path_log` unset first, to confirm the falsifier
can fail.

### Acceptance criteria

- Three clean `--dry-run`s; `pixi run test-fast` green. The **full** suite is
  not this phase's gate — it runs once for the program, in P5.
- A full three-workflow run completes on the seed config.
- `semantic_tree_diff` clean against the R9 map modulo a **written** allowlist.
- The concurrency falsifier passes, and was shown to fail before the fix.
- Rollback: if any moved artifact is not value-equivalent, revert the offending
  commit and report — do not adjust the expected value.

### Output requirements

A phase report with each ladder rung's result **and what it caught**, the
allowlist with a reason per entry, and a Results delta section stating
explicitly that no value changed — or, if one did, what and why, which is a
Gate 2 matter.

### Task constraints

- Run the pipeline from the **primary checkout**, never this worktree.
- `check_baseline` is red for this whole phase by design; report it, do not fix it.
- Exactly zero rule identifiers change in this phase.
