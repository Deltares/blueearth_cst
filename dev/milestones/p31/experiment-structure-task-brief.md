# Task Brief — P3-1 Experiment structure (implementation)

> **Handoff from the ACCEPTED design.** The authoritative, load-bearing spec is
> `dev/milestones/p31/experiment-structure-design.md` (ACCEPTED 2026-07-23). Read it in
> full before touching code — this brief bounds and sequences the work; the
> design owns every path, mechanism, probe result, and gate definition. Where
> the two differ, the design wins. Audit trail:
> `dev/milestones/p31/experiment-structure-design-review-record.md`; scoping anchors:
> `dev/milestones/p31/experiment-structure-intake.md`.

### Context

- **Canonical ruleset:** `AGENTS.md` (repo root). Honor its Hard Constraints —
  CST automation scope: no upstream/vendored re-engineering; the R weathergen
  layer and `realization_*`/`stress_test` internals stay as-is (intake anchor).
- **Behavior-preserving layout milestone:** values identical, paths move. The
  proof mechanism is design §6/§6a (R3/R5-style value-identical re-record +
  the path-aware full-tree semantic diff), NOT R6's no-re-record stance —
  editing `dev/baseline/manifest.json` (commit 6) and `check_baseline.py` /
  `semantic_tree_diff.py` (commit 5) is **in scope by G1 amendment**.
- **Key mechanisms are probe-verified — implement them exactly:** params
  rerun-trigger + `file_digest_or_absent()` content digests for guard
  invalidation (§3b/§3c); per-experiment sentinel on the four per-experiment
  root rules + key-level `climate_historical/<key>/.guard_ok` consumed
  `ancient()` by `extract_climate_grid` only (§3d — do NOT swap these);
  `slugify_window` dataset+window store keys with the day-resolution assertion
  (§4/§4c); absolute `path_static` relying on hydromt re-relativization (§5a);
  project-root-relative toml comparator (§6a step 3).
- **Milestone mechanics (per `branch-model`):** short-lived branch off `main`
  tip; commit prefix `p31:`; merge to `main`; then `milestone/p31-experiments`
  branch + tag `p31-experiments` at the tip. Capture the pre-P31 `main` SHA
  before starting.

### Goal

Land the accepted P3-1 design: multiple non-colliding, self-describing
experiments per `project_dir` under `experiments/<name>/`, the drift-guard
rule with digest-based invalidation, the keyed shared historical-climate
store, the Wflow run-dir relocation, and the repointed baseline machinery —
with the wf3 baseline slice re-recorded value-identically.

### Non-goals

- No registry / CLI listing / cross-experiment comparison; no layered configs.
- No wf1/wf2 computational-path changes; no `project_dir` changes beyond the
  design's tree; no weathergen/realization file-format changes.
- No edits to root `MIGRATION.md` (R06-scoped; the P3-1 map goes to
  `dev/milestones/p31/migration_experiment-structure.md` per naming.md §7).
- No catalog-content staleness machinery (deferred; manual escape hatch only).

### Allowed scope

**Permitted:** `Snakefile_climate_experiment`; `blueearth_cst/shared/snake_utils.py`
(new helpers: `validate_experiment_name`, `file_digest_or_absent`,
`slugify_window`); the new guard rule script; `blueearth_cst/experiment/
{extract_historical_climate,downscale_climate_forcing}.py`;
`blueearth_cst/projections/prepare_climate_data_catalog.py` (orography lookup);
`tests/**`; `dev/scripts/check_baseline.py` + `dev/scripts/semantic_tree_diff.py`
(exactly the commit-5 edits); `dev/baseline/manifest.json` (commit 6 re-record
only); `dev/milestones/p31/**`, `dev/roadmap.md` (prefix registration), README/docs refs.

**Approval-gated:** anything touching `Snakefile_model_creation` /
`Snakefile_climate_projections` beyond zero expected edits — the design says
wf1/wf2 are untouched; if you find an unavoidable edit, PAUSE and raise it.

**Forbidden:** vendored upstream packages; `pixi.lock` / `Manifest.toml`;
`blueearth_cst/weathergen/*.R` content changes; `examples/test_local` by hand.

### Required changes (checklist)

The design §10 commit plan, verbatim — one `p31:` commit each; each of 1–4
leaves all three workflows runnable:

1. Drift guard: rule + script + root wirings + digest params +
   `file_digest_or_absent` + gates 2/2b/2c tests (§3, §10.1 incl. the
   `raw_data/.guard_ok` sequencing note).
2. `validate_experiment_name` at parse + `exp_dir` redefinition moving all wf3
   project-global outputs (incl. `gather_benchmarks` `parts_dir`) under
   `experiments/<name>/` (§2/§2a/§2b, §10.2).
3. Run-dir → `experiments/<name>/model_runs/` + toml-literal rewrite (§5,
   §10.3 — one interdependent commit).
4. Keyed store + chirps orography fix + guard-artifact move to the keyed dir
   (§4, §10.4).
5. Baseline-machinery repoint: `check_baseline.py` targets +
   `semantic_tree_diff.py` path map, allowlist assertion, path-aware toml
   comparator + tests (§6/§6a, §10.5).
6. Value-identical wf3 re-record + `dev/milestones/p31/baseline_diffs.md` (§6a/§6b,
   §10.6).
7. Migration note + `p31:` prefix registration + docs (§10.7).

### Validation

Follow the design §7 gate table exactly; a commit is not done until its named
gates pass. Ladder: (1) narrow — `pytest tests/test_cli.py` + the commit's
unit tests after every commit; (2) new behavioral tests — gates 2/2b/2c
(guard + invalidation + fresh-project), 9 (name validation), comparator
tests; (3) execution smokes for every dry-run-blind path — gates 3 (run-dir +
purity), 4 (A-runs/B-skips/C-re-extracts + alternation), 7 (gather), **8
(CHIRPS-path smoke — the era5-only escape class is closed, run it)**; (4)
full gate — `pytest tests/` green + clean `--dry-run` all three Snakefiles +
full three-workflow e2e via `scripts/run_workflows.py`; (5) baseline — §6a
procedure: pre-change tree capture, post-change full regen, path-aware
semantic diff (MISSING/EXTRA empty modulo the documented allowlist; **any
value diff = STOP**), then the commit-6 re-record.

### Acceptance criteria

- All 7 commits landed; two experiments coexist collision-free in one
  `project_dir`; second same-key experiment reuses the store extraction
  (gate-4 evidence); drift guard fails loud on each mutated comparand
  (gate-2b evidence); fresh project parses/`--dry-run`s/`--unlock`s (2c).
- Full gate green; baseline re-record value-identical modulo the documented
  path map; `baseline_diffs.md` + migration note landed.
- `milestone/p31-experiments` + tag `p31-experiments` at the merged tip.
- **Rollback trigger:** any value-level baseline diff, any guard false-fire
  in the alternation smoke, or a store re-extraction where gate 4 requires a
  skip → stop, do not tag, surface the evidence.

### Output requirements

- The commits on a short-lived branch, merged to `main` + branch/tag.
- A **Results delta**: confirm no computed value changed (path-map-only
  diffs), the gate-4/2b evidence, and the re-record summary.

### Task constraints

- Design §10 sequencing is binding (guard artifact placement depends on it).
- Preserve `workflow.configfiles[0]` forwarding, `get_config`, the `sys.path`
  shim, and per-rule `log:`/`benchmark:` house patterns.
- `--dry-run` is never sufficient for a runtime-only reference — every such
  path has a named execution smoke in §7; run them.
- Author spawns: put deliverable files on disk early; probes are dry-run/
  scratch-dir only until the gated full runs.

**Human gates** (otherwise drive commit-to-commit autonomously per the
standing preference):

- **Gate 1 — after commit 1 (guard + root wiring), PAUSE:** present the rule
  wiring diff, the gate-2/2b/2c results, and the dry-run trigger evidence
  before continuing (it edits existing rule `input:` blocks across wf3).
- **Gate 2 — milestone gate, before merge/tag, PAUSE:** present the full
  gate, the §6a semantic-diff + re-record results, gate-4/8 evidence, and
  `baseline_diffs.md` for sign-off.
- Raise immediately if wf1/wf2 Snakefiles need edits, or any baseline diff is
  a value change rather than a mapped path move.
