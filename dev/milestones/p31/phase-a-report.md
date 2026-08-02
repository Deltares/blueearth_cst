# Phase A report — pre-P31 baseline capture + commit 1

## Pre-P31 main SHA
`7d1be7130891e9fe7d5d8958af0b7b6aec675f94` (branch `main`)

`git status --short` at start:
```
 D dev/working/design-runs/p31-experiment-structure/intake.md   (pre-existing deletion — left as-is)
?? dev/milestones/p31/phase-a-report.md
```

## Baseline capture (config used, run outcomes, reference-tree location)

**Tracked test orchestration config:** `config/workflows/snake_config_model_test.yml`
(project_dir `examples/test_local`; confirmed via `check_baseline.py` PROJECT_DIR_DEFAULT
and `scripts/run_snake_test.cmd` CFG).

**Scratch capture config** (untracked, in scratchpad, NOT under `config/`):
`…/scratchpad/snake_config_pre_p31.yml`. Diff vs tracked (line-endings normalized):
```
12c12
<   project_dir: examples/test_local
---
>   project_dir: examples/test_local_pre_p31
```
Only the `project_dir` line differs, as required.

**Reference-tree location:** `examples/test_local_pre_p31/` (in-repo untracked dev
location, exempt from the outside-repo project_dir rule). Once frozen, nothing writes
to it again.

**Run outcomes:**
- wf1 (`Snakefile_model_creation`): PASS — exit 0, 14/14 steps. All 6 `rule all`
  terminal outputs present (3 PNGs, `config/snake_config_model_creation.yml`,
  `hydrology_model/staticgeoms/outlet_index.csv`, `benchmarks/wf1_benchmarks.md`).
- wf3 (`Snakefile_climate_experiment`): PASS — exit 0, 57/57 steps
  (2026-07-23; snakemake log `2026-07-23T193631.059711.snakemake.log`).
- terminal-output verification: all four wf3 `rule all` targets present —
  `climate_experiment/model_results/Qstats.csv`, `climate_experiment/model_results/basin.csv`,
  `config/snake_config_climate_experiment.yml`, `benchmarks/wf3_benchmarks.md`
  (plus all six wf1 targets, verified earlier).
- total file count of the frozen tree: **153 files**.

**FROZEN 2026-07-23 after wf3 completion** — nothing may write to
`examples/test_local_pre_p31/` again in this or any later phase. It is the
reference side of the commit-6 milestone diff.

## Branch
`task/p31-experiment-structure`, created off the `main` tip
(`7d1be713`) after the capture froze. All Phase-A edits and the commit live here.

**Commit 1:** `5e7bd05` —
`p31: add project-consistency drift guard (rule + root wiring + digest triggers + tests)`
(6 files: `Snakefile_climate_experiment`, `blueearth_cst/shared/snake_utils.py`,
`blueearth_cst/experiment/check_project_consistency.py`,
`tests/test_check_project_consistency.py`, `tests/test_guard_invalidation.py`,
`tests/test_cli.py`). Worktree after commit: only the pre-existing `intake.md`
deletion (unstaged, untouched) and this uncommitted report. Frozen tree
re-verified post-gates: 153 files, no writes after the capture.

## Commit 1 — what was implemented

Per design §10.1 + §3a–§3d (scope: drift guard only; `validate_experiment_name`,
`exp_dir` redefinition, run-dir move, store keying are later commits):

- **`blueearth_cst/shared/snake_utils.py`** — new `file_digest_or_absent(path) -> str`
  (peer of `get_config`): SHA-256 hex of file bytes when present; literal `"ABSENT"`
  when missing/unreadable; never raises (ext2-2).
- **`blueearth_cst/experiment/check_project_consistency.py`** — new guard script.
  Pure comparator `compare_project_consistency(live_cfg, wf1_path, wf2_path)`:
  section-scoped deep equality on `project`, `shared.basin`,
  `workflows.model_creation` (vs wf1 snapshot) and `workflows.climate_projections`
  (vs wf2 snapshot **only when present**, else unchecked+logged pass); symmetric
  defensive OLD→NEW path normalization (flat-vs-binned ⇒ pass, gate 2d); missing
  wf1 snapshot ⇒ friendly "run Snakefile_model_creation first" message; failure
  names the diverging key. On pass writes both artifacts; on failure writes
  neither. Harness reads the live config from `snakemake.config`; wired via
  `tee_to_log` like sibling scripts. (Note: no `from __future__` import — the
  Snakemake `script:` preamble would make it a SyntaxError.)
- **`Snakefile_climate_experiment`** — new rule **3.00b `check_project_consistency`**
  before `copy_config`:
  - `input:` `ancient(<project_dir>/config/snake_config_model_creation.yml)`
    (mandatory; absence ⇒ rule-level MissingInputException).
  - `params:` all experiment-invariant (ext2-1; `config_path` NOT present):
    the four guarded section names; SHA-256 of sorted-key-JSON of the four
    guarded live sections; wf1 snapshot digest via `file_digest_or_absent`
    (computed at parse); wf2 snapshot path + digest via `file_digest_or_absent`.
  - `output:` per-experiment sentinel `{exp_dir}/.project_consistency_ok`
    (currently `climate_<experiment>/`, moves with the commit-2 `exp_dir`
    redefinition) — small text file, not `temp()`; and the key-level guard
    artifact at `climate_historical/raw_data/.guard_ok` (§10.1 sequencing note —
    store un-keyed until commit 4).
  - `log:`/`benchmark:` per the house 3.NN pattern.
  - Root wirings exactly per §3a: sentinel as FRESH ordinary `input:` on
    `copy_config`, `climate_stress_parameters`, `prepare_weagen_config`,
    `prepare_weagen_config_st`; `.guard_ok` consumed with `ancient(...)` by
    `extract_climate_grid` ONLY. No other rule consumes `.guard_ok`.
- **`tests/test_check_project_consistency.py`** — gate 2 (a–h) pure comparator
  unit tests.
- **`tests/test_guard_invalidation.py`** — gate 2b (i–l) integration test
  (edit-then-`--dry-run` reading scheduling reasons, sentinel never deleted
  between mutations; one initial real guard execution seeds the `.snakemake`
  provenance metadata) + gate 2c fresh-project regression.
- **`tests/test_cli.py`** — fixture `config_with_staged_region` extended to also
  stage the wf1 config snapshot (the guard's new mandatory `ancient()`
  cross-workflow input, same class as the staged `region.geojson`).

## Gate results (2 a–h, 2b i–l, 2c, test_cli)

| Gate | Result |
| --- | --- |
| 2 (a–h) comparator unit tests | **PASS** — 8/8 (`tests/test_check_project_consistency.py`) |
| 2b (i–l) invalidation integration | **PASS** (`tests/test_guard_invalidation.py::test_guard_invalidation_i_to_l`) |
| 2c fresh-project regression | **PASS with one documented deviation on (ii)** — see Deviations (`::test_2c_fresh_project_missing_wf1_snapshot`) |
| `pytest tests/test_cli.py` | **PASS** — 4/4 |
| `--dry-run` wf1 (tracked test config) | **PASS** — exit 0, clean DAG, Snakefile untouched |
| `--dry-run` wf2 (tracked test config) | **PASS** — exit 0, clean DAG, Snakefile untouched |
| `--dry-run` wf3 (tracked test config) | **PASS** — exit 0; guard scheduled once; roots depend on it |
| full default suite `pytest tests/` | **PASS** — 240 passed, 3 skipped (integration), 1 xfailed |

## Dry-run trigger evidence (verbatim)

Captured on a staged scratch `project_dir` (scratchpad `guard_probe/`), guard
executed once for real, then edit-then-`--dry-run` targeting the sentinel with
the repo workflow profile disabled (`--workflow-profile none`; the profile's
`quiet: reason` suppresses the Reason lines):

Control (nothing changed):
```
Nothing to be done (all requested files are present and up to date).
```
(i) guarded live-config section mutated (`shared.basin.resolution` 0.00833 → 0.05):
```
Reason: Params have changed since last execution: Union of exclusive params before and now across all output: before: 'c9fc2c29eace38b057288ee07c4567e2624f8818f5d761a7ee62d6bd527c2772' now: '4032f3c8c95ac5881d1e0577a91a439ce926d0b6444f03f8b7ecc930c03504b5'
```
(restore live config ⇒ `Nothing to be done (all requested files are present and up to date).`)

(j) wf1 snapshot content mutated:
```
Reason: Params have changed since last execution: Union of exclusive params before and now across all output: before: <nothing exclusive> now: '0ed0a8df558058c93e443daafc2ed6ca98802b37a3c00be64da1f3f68998db30'
```
(k) wf2 snapshot content mutated:
```
Reason: Params have changed since last execution: Union of exclusive params before and now across all output: before: <nothing exclusive> now: '9c448fd8c46fedda596d70887039b271d01adc4d4b389f071d564042be1c8632'
```
(l) all comparands reverted to original bytes:
```
Nothing to be done (all requested files are present and up to date).
```

## Deviations & open questions

1. **Gate 2c(ii) — `--unlock` with the wf1 snapshot absent FAILS on the pinned
   Snakemake; design expectation unsatisfiable as written.** Design §7 gate
   2c(ii) expects `--unlock` to succeed with the snapshot absent. Observed and
   source-verified: Snakemake 9.6.2 `Workflow.unlock()`
   (`site-packages/snakemake/workflow.py:917`) calls `self._build_dag()` before
   `cleanup_locks()`, so `--unlock` fails with `MissingInputException` on ANY
   missing leaf input — including the **pre-existing**
   `ancient(region.geojson)` on `extract_climate_grid` (probe: with guard inputs
   present and region.geojson absent, `--unlock` fails identically). The
   design's own §3b mechanism (hard `ancient()` wf1 input ⇒ rule-level
   MissingInputException) therefore cannot coexist with 2c(ii) on this
   Snakemake. Resolution (minimal): the load-bearing mechanism is kept exactly
   as designed; `file_digest_or_absent` still delivers what ext2-2 demands at
   the parse layer (no parse-time traceback; the DAG builds; the failure is the
   clean rule-level exception). The test pins observed behavior: `--unlock`
   with snapshot absent fails ONLY with the rule-level MissingInputException
   (no traceback) — i.e. the guard does not degrade `--unlock` beyond the
   pre-existing baseline — and `--unlock` SUCCEEDS with every leaf input
   present (the real recoverable-lock scenario: a crashed run implies the
   snapshot existed). Gate-1 reviewer to confirm this reading.
2. **`tests/test_cli.py` edited (fixture only).** Not in the prompt's staging
   list, but required for its gate to pass: the guard's wf1-snapshot input is a
   new cross-workflow leaf the staged fixture must provide (exact precedent:
   the staged `region.geojson`). In the task brief's allowed scope (`tests/**`).
3. **Two new test modules, not three.** Gate 2c lives in
   `tests/test_guard_invalidation.py` rather than its own module — explicitly
   permitted by design §7 ("same file or its own").
4. **Integration marker not used.** `tests/test_guard_invalidation.py` is NOT
   `@pytest.mark.integration` (which auto-skips without `--run-integration`):
   commit 1's gates must run in the default suite, and the test needs no data
   mirror/Julia (guard rule only; ~35 s). Flag if the marker policy should
   apply anyway.
5. **Test invocations disable the repo workflow profile**
   (`--workflow-profile none`): `profiles/default/config.yaml` sets
   `quiet: reason`, which suppresses exactly the "Reason:" lines gate 2b
   asserts on.
6. **`json`/`hashlib` imports added to `Snakefile_climate_experiment`** for the
   parse-time guarded-sections digest (stdlib only, no new dependency).
