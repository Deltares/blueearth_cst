# Validation ladder — tiering, configs, and CI

The ladder table lives in `AGENTS.md` § Validation ladder. This file holds what does not fit there: how the tiers are drawn, which config a run takes, and how to read CI.

## Tiering

`test-fast` deselects the `workflow_contract` and `process_isolation` markers. Those tests each spawn a fresh Python process and build a Snakemake DAG, so they cost around fifty times what an ordinary test costs: they are roughly 2% of the suite and the large majority of its runtime. Count them with
`pytest tests/ -m "workflow_contract or process_isolation" --collect-only -q` rather than trusting a number written here — it drifts every time a rule gains a contract test.

Run the whole cheap tier rather than hand-picking relevant files: selecting by judgment saves little and misses cross-module regressions.

`test-fast` runs its workers in parallel (`-n auto --dist loadfile`). Keep `loadfile`: it puts every test in a file on one worker, which is what contains module-level state — a test that stubs a library process-globally stays confined to its own file. `--dist load` distributes per test and would break that. The floor is the per-worker import cost, so adding cores past a point buys nothing.

`test-contract` stays serial. Those tests share a `project_dir` and take `.snakemake` locks, so parallelising them needs each group proved independent first.

**`test-fast` is the local gate, including before a push.** CI runs the unmarked `pytest tests/` on ubuntu and windows from the push itself, so a local `test-full` re-proves on one platform what CI is about to check on two, for roughly ten times the wall-clock of the cheap tier. What it buys is knowing before the push rather than a quarter of an hour after it — worth paying only when a bad push is expensive to undo.

Reach for `test-full` locally when a change touched `blueearth_cst/shared/` or a `script:` signature, and before a milestone seal. Those are the cases where meeting a failure on two platforms at once costs more than the wait.

Between merges and the next push, a `workflow_contract` regression can sit on local `main` across several branches, so bisect across all of them. Reading the CI run after each push is what bounds that window.

State gate costs as orders of magnitude, never in seconds. The marker names and the test count are durable; the clock is not.

`.testing-policy.yml` pins `scope: rapid`, and `auto_push: false` keeps the push a deliberate decision.

## Which config to run

Default to `snake_config_rapid.yml`. Reach for `snake_config_baseline.yml` only when the run's numbers are the point.

| Config | `project_dir` | Run it for |
|---|---|---|
| `snake_config_rapid.yml` | `test_case/test_rapid` | anything you want to watch EXECUTE — a rule you edited, a DAG check, a WF3 smoke run, a figure render |
| `snake_config_baseline.yml` | `test_case/test_local` | recording or checking `dev/baseline/manifest.json`, `tree-check`, a milestone seal, any number you will quote |
| `snake_config_wf2_fast.yml` | `test_case/test_dev` | WF2 code iteration only — 2 series, and it drops `st_0` |

Rapid costs ~2.6× less wflow time (10 members × 9 forcing years, vs 14 × 17) and ~1.7× less weather generation (46 generated years, vs 78). To move the second number, change the horizon rather than `run_length`: `compute_nr_years` anchors the generated series at 2010, so it spans 2010 → `horizontime_climate` + `run_length`/2.

Rapid is cheap, not narrow. It keeps `run_historical: true`, since `st_0` is what the two class-C month indicators derive from — `false` silently drops 2 of 11 `q` metrics — and it keeps two CMIP6 models, since a one-model config never runs the ensemble reduction. A config that gives up coverage must say which, as `wf2_fast` does.

Record the baseline from `snake_config_baseline.yml` and nothing else; never point `check_baseline.py` at the rapid tree.

## Read the CI run after you push

Install the pre-push hook once per clone — cloning does not install it:

```bash
git config core.hooksPath .githooks     # runs the two ruff checks, ~2 s
```

Read the run regardless. The hook cannot see platform-specific failures, and a green local suite is no evidence about the ubuntu leg — the only place linux-64 is exercised.

`gh` resolves to `upstream` (`Deltares/blueearth_cst`) in this clone, not `origin` (`tanerumit/blueearth_cst`, where CI runs), so a bare `gh run list` queries Deltares, exits 0 and prints nothing. Point it at origin once per clone:

```bash
gh repo set-default tanerumit/blueearth_cst   # writes remote.origin.gh-resolved
```

In a script that must not depend on local config, pass `--repo tanerumit/blueearth_cst` explicitly, or use the API:

```bash
# the latest run, whatever branch it was on
gh api "repos/tanerumit/blueearth_cst/actions/runs?per_page=1" \
  --jq '.workflow_runs[0] | "\(.head_sha[0:7]) \(.status) \(.conclusion)"'

# which STEP failed, per leg — separates a lint failure from a suite failure
gh api "repos/tanerumit/blueearth_cst/actions/runs/<id>/jobs" \
  --jq '.jobs[] | "\(.name) -> \(.conclusion): " +
        ([.steps[] | select(.conclusion=="failure") | .name] | join(", "))'
```

Do not filter by `head_sha=<short sha>`: the parameter needs the full 40-character value and silently matches nothing otherwise.
