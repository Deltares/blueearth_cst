# Phase C report — commits 5–7 + milestone gate

## Starting state

- Branch `task/p31-experiment-structure`, HEAD `27a2a54` (commit 4) on top of `main` `7d1be71`.
- Phase A+B done: commits `5e7bd05` (guard), `3c77f81` (validate + exp_dir move),
  `6e2431c` (run-dir move + toml rewrite), `27a2a54` (keyed store + chirps + guard-artifact move).
- Full default suite at start of Phase C (per phase-b): 287 passed, 3 skipped, 1 xfailed.
- Frozen reference tree: `examples/test_local_pre_p31/` (153 files) — READ-ONLY, captured pre-change.
- Worktree: pre-existing unstaged deletion of `dev/working/design-runs/p31-experiment-structure/intake.md`
  (to be committed via `git rm` in commit 7); untracked phase-a/b/c reports.
- Pre-P31 main SHA: `7d1be7130891e9fe7d5d8958af0b7b6aec675f94`.

## Commit 5 — baseline machinery repoint + path-aware comparator

**Commit hash: `60fa2d5`** — `p31: repoint baseline manifest + semantic_tree_diff path map/allowlist/toml-comparator to new tree`
(3 files: `dev/scripts/check_baseline.py`, `dev/scripts/semantic_tree_diff.py`, `tests/test_semantic_tree_diff.py`).

**`check_baseline.py` — the two DISTINCT repoints (§1a C2, arch-6):**
- (i) `resolve()`: `exp_dir = f"{project_dir}/climate_{EXPERIMENT_NAME}"` →
  `f"{project_dir}/experiments/{EXPERIMENT_NAME}"` (`EXPERIMENT_NAME = "experiment"`
  unchanged — it is the name value). The two `{exp_dir}` results TARGETS
  (Qstats.csv, basin.csv) move with it.
- (ii) The wf3 config-snapshot TARGET template root changed
  `{project_dir}/config/snake_config_climate_experiment.yml` →
  `{exp_dir}/config/snake_config_climate_experiment.yml`.
- Existing `tests/test_check_baseline_scope.py`/`_discharge.py` resolve templates
  via `cb.resolve()` so they adapted with zero edits (verified: all pass).

**`semantic_tree_diff.py` — the §6a layer:**
- `build_p31_path_map(experiment_name, dataset_key)` — ORDERED directory-prefix
  rewrite rules on project-root-relative POSIX paths (exact-file rule for the
  config snapshot; prefix rules otherwise; first match wins; old→new direction):
  1. `config/snake_config_climate_experiment.yml → experiments/<name>/config/…` (exact)
  2. `hydrology_model/run_climate_<exp>/ → experiments/<name>/model_runs/`
  3. `climate_<exp>/ → experiments/<name>/`
  4. `climate_historical/raw_data/ → climate_historical/<dataset_key>/` (when key given)
- `diff_trees` gained `path_map`/`allowlist` params: ref relpaths are translated
  old→new before pairing, translated pairs are content-diffed; residual
  MISSING+EXTRA must be EMPTY modulo the allowlist (matched entries reported as
  ALLOWED); anything else fails the gate (risk-4). A path-map collision (two ref
  files → one target) raises.
- `build_p31_allowlist` — the two EXTRA-by-design presence exemptions:
  `experiments/<name>/.project_consistency_ok`, `climate_historical/<key>/.guard_ok`.
- Path-aware toml comparator (§6a step 3 verbatim, 4 steps): lexical
  normpath+join against the toml's own dir (never `.resolve()`; backslashes
  normalized — the ref tomls use `..\\staticmaps.nc`), strip that side's project
  root, apply the prefix map to the REF target, compare. Fields:
  `input.path_forcing`, `input.path_static`, `state.path_input`,
  `state.path_output`, `csv.path` (the design's three named fields + the two
  run-dir-resident outputs, which get the identical treatment and PASS via
  prefix rule 2). Mismatch ⇒ `failures` entry naming the field ("mis-repoint").
- CLI: `--experiment-name` (default `experiment`), `--dataset-key`,
  `--no-path-map`, `--allow` (repeatable). Map + built-in allowlist on by default.
- Backward compatible: `compare_toml(ref, cur)` without roots = raw compare;
  `diff_trees` without map = identical-relpath keying (all 17 pre-existing
  tests pass unchanged).

**Tests added (7):** the §6a matrix — `path_static` old-depth vs new-depth ⇒
PASS; `path_forcing` via the prefix rule with the target existing in NEITHER
tree (asserts the prefix-rewrite form) ⇒ PASS; `path_static` →
`staticmaps_WRONG.nc` ⇒ FAIL naming the field; a pure move pairs+content-diffs
CLEAN; a moved file with a value diff still FAILS (no masking); allowlist gate
contract (2 allowed sentinels pass, an unexplained extra fails); new-layout
self-diff with map active clean.

**Gates:**
| Gate | Result |
| --- | --- |
| comparator unit tests (`test_semantic_tree_diff.py`) | PASS — 24/24 (17 pre-existing + 7 new) |
| check_baseline tests (`test_check_baseline_{scope,discharge}.py`) | PASS (template-driven; adapted with zero edits) |
| self-diff smoke, frozen tree vs itself (`--no-path-map`) | CLEAN — 86 files compared, 0 failed/missing/extra |
| self-diff with map active | covered by unit test on a NEW-layout tree (the map is directional old→new, so an old-layout self-diff with the map is legitimately not clean; the real with-map run is the commit-6 milestone diff) |
| `pytest tests/test_cli.py` | PASS — 4/4 |
| full default suite | PASS — 294 passed, 3 skipped, 1 xfailed |

## Commit 5b — cross-root yml normalization + run-log exclusion (plan deviation, Fable-authored)

> Provenance note: the first Phase C agent died on a stream timeout after the
> wf3 regen + first semantic-diff run; the driver (Fable) took over the gate
> adjudication and authored 5b directly (the tiering rule's designed
> escalation), then drove the rest of commit 6 driver-owned.

**Commit hash: `576b6a6`** — `p31: extend semantic diff to cross-root yml paths + exclude run-log files (5b)`

**Why (gate-6 adjudication).** The first cross-root milestone diff run
(pre-5b, output preserved in `dev/milestones/p31/baseline_diffs.md`) reported
MISMATCH: 19 failures, 1 extra, 2 allowlisted. Fable adjudicated every entry
at parse level (deep leaf-diff of each failing yml pair):

- **17 yml pairs — ALL PATH-ONLY, 0 non-path leaf diffs**: the 2 config
  snapshots (`project.project_dir` records each tree's own root — differs by
  construction in a cross-root comparison), the 14 weathergen configs
  (root-prefixed `output.path` values moved with exp_dir), the experiment
  data catalog (absolute `uri`s under each root + layout move). Same
  behavior-neutral pointer-move class the design's ext1-3 toml comparator
  formalizes — in YAML files the design's §6a inventory missed.
- **3 log files** (`hydromt.log`, `model_runs/log.txt`, `run_default/log.txt`)
  — timestamp noise; run-log FILES outside the excluded `logs/` dirs.
- **1 EXTRA** `experiments/experiment/config/deltares_data.yml` — the
  de-collided per-experiment catalog copy (old layout: wf3 `copy_config`
  overwrote wf1's project-level copy — the exact collision P3-1 fixes).
  Verified byte-identical to both the cur and ref project-level copies.

**Mechanism** (mirrors ext1-3): `compare_yaml()` — reflexivity guard, R6
directional config-map layer (config-dir snapshots), then cross-root
normalization: a string leaf equal to / prefixed by that side's OWN project
root becomes `<PROJECT_ROOT>[/rest]`, with the old→new path map applied to
the REF remainder; non-path leaf drift still FAILs. Prefix-or-equality only —
no mid-string rewriting. Run-log files (`*.log`, `log.txt`) join the
excluded-logs class. 4 new tests (28/28 comparator suite); full default suite
298 passed / 3 skipped / 1 xfailed.

**Scope note for Gate 2:** 5b touches `dev/scripts/semantic_tree_diff.py` +
`tests/**` (both in the task-brief allowed scope) but is an 8th commit beyond
the design's 7-commit plan — presented for ratification at Gate 2.

## Commit 6 — post-change regen, semantic diff, re-record

Procedure in the mandated order:

1. **Rename-aside (no deletion):** `examples/test_local` →
   `examples/test_local_superseded_pre_p31` (preserved byte-for-byte; never
   written to again; user decides its fate post-milestone). DONE.
2. **Post-change regen** (wf1 then wf3 ONLY, matching the reference capture's
   §6a shape) into a fresh `examples/test_local`, tracked config.
   wf1: DONE (exit 0, terminal targets verified, 42 files, 21:17).
   wf3: DONE (exit 0, results written 22:41 — Qstats.csv, basin.csv, RT_Q_*.csv,
   snapshot, benchmarks all present; run supervised by the first Phase C agent).
3. **Semantic diff** ref=`test_local_pre_p31` cur=`test_local`, map+allowlist:
   **CLEAN** after 5b — verbatim below and in `baseline_diffs.md`.
4. **Wrapper e2e** (all three workflows): **exit 0.** wf1 up-to-date; wf2 ran
   fresh; the guard re-ran on the wf2-snapshot ABSENT→digest flip (job 7) and
   wf3 legitimately re-ran downstream (57/57) — the designed §3b behavior.
   Post-e2e re-diff vs the reference: **0 failed, 0 missing** (live
   determinism confirmation); 15 EXTRA = wf2 overlay artifacts only
   (ref deliberately wf1+wf3 per §6a).
5. **Re-record wf3 + check:** `record --workflow climate_experiment` → 3 rows
   at the new paths; `check --workflow climate_experiment` **OK 3/3**; wf1
   VALUE targets pass without re-record — the single wf1 fail is the config-
   snapshot sha row whose recording predates R6's config re-binning
   (`e705965`), the R6-adjudicated standing condition (same for the wf2 row;
   follow-up `chore:` re-record recommended, out of P3-1 scope). The known
   wf2 2-CSV nondeterminism did not fire this regen.

**Commit hash: `aaa993d`** — `p31: re-record wf3 baseline slice (value-identical) + baseline_diffs`
(`dev/baseline/manifest.json` + `dev/milestones/p31/baseline_diffs.md`).

## Commit 7 — migration note + prefix + docs

- `dev/milestones/p31/migration_experiment-structure.md`: old→new map (7 classes),
  final 3-entry allowlist with justifications + the run-specific `--allow`
  convention, the gate-2c(ii) design erratum (Gate-1 endorsed), baseline-
  machinery summary incl. 5b.
- `dev/roadmap.md`: `p31:` prefix registered (Phase 3 line).
- Stale-layout doc sweep (`climate_experiment/`, `run_climate_`,
  `climate_historical/raw_data` over README.rst / AGENTS.md / docs/ /
  dev/workflows/): README.rst and AGENTS.md were already layout-clean; the
  single stale surface was `dev/workflows/climate_experiment.md` (R5 contract
  doc) — three layout edits (exp_dir definition + validation note, config-
  snapshot target, logs/benchmarks side-effects), pre-existing stale `src/*`
  line-number references left as-is (R6-era, not layout).
- Phase reports a/b/c landed as the audit trail; the pruned run-dir
  `intake.md` deletion committed (`git rm`; intake lives at
  `dev/milestones/p31/experiment-structure-intake.md`).

## Full-gate results

| Gate | Result |
| --- | --- |
| `--dry-run` wf1 / wf2 / wf3 (tracked config) | all exit 0 |
| full default suite (final) | **298 passed, 3 skipped, 1 xfailed** |
| wrapper e2e `run_workflows.py` (all three) | exit 0 |
| milestone semantic diff (pre-e2e) | **CLEAN** — 83 compared, 0/0/0, 3 allowlisted |
| post-e2e re-diff | 0 failed, 0 missing (EXTRA = wf2 overlay only) |
| `check --workflow climate_experiment` (post-re-record) | OK 3/3 |
| wf1 value targets | pass, no re-record |
| gate-4/2b/2c/3/5/7/8/9 evidence | phase-a/b reports (Phases A+B) |

## Semantic-diff results (MISSING/EXTRA/failures, verbatim residuals)

Final run (post-5b), command:
`semantic_tree_diff.py --ref examples/test_local_pre_p31 --cur examples/test_local
--experiment-name experiment --dataset-key era5_20000101_20201231
--allow experiments/experiment/config/deltares_data.yml`

```
ALLOWED (EXTRA allowed: climate_historical/era5_20000101_20201231/.guard_ok)
ALLOWED (EXTRA allowed: experiments/experiment/.project_consistency_ok)
ALLOWED (EXTRA allowed: experiments/experiment/config/deltares_data.yml)
CLEAN: 83 files compared, 0 failed, 0 missing, 0 extra, 3 allowlisted
```

**Value-identity: ZERO value-level diffs.** Every `.nc` (element-wise), every
CSV (incl. Qstats/basin/RT and the run CSVs), every run toml (path-aware
comparator; 0 failures), every yml (cross-root normalization; non-path leaves
all equal). The §6b mixed-provenance residual did NOT appear — both trees were
regenerated on the restored wf1 model, as the milestone intended. Rollback
trigger NOT tripped.

## Deviations & open questions

1. **Commit 5b** (above) — an 8th commit beyond the §10 plan; gate-machinery
   only, no workflow-code change; for Gate-2 ratification.
2. **`--allow experiments/experiment/config/deltares_data.yml`** — run-specific
   allowlist entry via the CLI flag (code allowlist stays sentinel-only);
   justification above + migration note.
