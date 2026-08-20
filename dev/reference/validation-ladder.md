# Validation ladder — measured costs and rationale

The ladder table itself lives in `AGENTS.md` § Validation ladder; this file is the long form it points to — why the fast/full split lands where it does, which config to run, and how to read a CI run. Lifted from `AGENTS.md` @ 46f9df2 during the 2026-08-20 slim-down, unedited except for this header.

**Why the split lands there.** `test-fast` deselects **55 tests — under 3% of the suite — and they cost the large majority of the runtime**. They are exactly the `workflow_contract` and `process_isolation` markers. Paying several times the wall-clock for that last 3% on every merge, when every task is a merge, is the cost that was actually being paid. Selecting "only the relevant test files" instead would save little over running the whole cheap tier, in exchange for a judgment call per task and the cross-module regressions that judgment misses — so run everything cheap, and tier the expensive part.

Times here are **orders of magnitude, deliberately**. Earlier revisions pinned them to the second (`1:29`, `7:07`, `427 s`); by 2026-08-12 every one was off by 2–3×, and a stale number is worse than a vague one precisely because someone uses it to decide whether a gate is affordable. Do not restore exact seconds — the test count and the marker names are the durable facts, the clock is not.

A push is the escalation trigger because it is where work leaves this machine: before it, `main` is local and revertible; after it, CI, the other platform leg, and anyone else's clone are downstream. The pin in `.testing-policy.yml` (`scope: rapid`) declares that posture, and `testing-policy`'s own `rapid` → `release` boundary is the same line.

**The residual risk, stated plainly:** a `workflow_contract` regression can now sit on local `main` across several merged branches until the next push, so bisecting it spans those branches rather than one. That is the deliberate trade for the speedup — and `auto_push: false` means the push is a decision you make, which is precisely when the extra minutes are worth spending. Run `test-full` at the merge anyway, not just at the push, when a branch touched a Snakefile, a `script:` signature, or `shared/`; those are the paths that tier exists to guard.

## Which config to run — rapid by default, comprehensive selectively

The ladder above governs `pytest`; this governs the pipeline itself. Default to `test_case/snake_config_rapid.yml` and reach for `snake_config_baseline.yml` only when the run's NUMBERS are the point.

| Config | `project_dir` | Run it for |
|---|---|---|
| `snake_config_rapid.yml` | `test_case/test_rapid` | anything you want to watch EXECUTE — a rule you edited, a DAG check, a WF3 smoke run, a figure render |
| `snake_config_baseline.yml` | `test_case/test_local` | recording or checking `dev/baseline/manifest.json`, `tree-check`, a milestone seal, any number you will quote |
| `snake_config_wf2_fast.yml` | `test_case/test_dev` | WF2 code iteration only — 2 series, and it drops `st_0` |

Rapid costs ~2.6× less wflow time (10 members × 9 forcing years, vs 14 × 17) and ~1.7× less weather generation (46 generated years, vs 78). **The horizon, not `run_length`, is what moves the second number**: `compute_nr_years` anchors the generated series at 2010, so it spans 2010 → `horizontime_climate` + `run_length`/2 and shortening the run alone barely touches it.

Rapid is CHEAP, not NARROW, and the difference is load-bearing. It keeps `run_historical: true`, because `st_0` is what the two class-C month indicators are derived *from* — `false` drops 2 of 11 `q` metrics with nothing reporting it (the R11 P3 case `interchange_contracts.py` was hardened against) — and it keeps two CMIP6 models, because a one-model config never runs the ensemble reduction. A config that gives up coverage must say which, as `wf2_fast` does.

The baseline is recorded from `snake_config_baseline.yml` and nothing else; never point `check_baseline.py` at the rapid tree.

## Read the CI run after you push

**CI was red on `main` for ten days and nobody noticed** (t2608071205): seven runs failing the ruff gate from 2026-07-30, then — once that was fixed — three more failing the ubuntu leg's unit suite, a *different* defect that had been hiding behind the first. Nothing alerted; the gate simply went unread.

Two things follow, and they are complementary rather than alternatives.

- A **pre-push hook** runs the two ruff checks (~2 s) so that class cannot leave the machine. Install it once per clone: `git config core.hooksPath .githooks`. It is not installed by cloning — check `git config core.hooksPath` if unsure.
- **Read the run anyway.** The hook is blind to everything platform-specific, which is exactly what the second outage was: four tests asserting Windows path spelling, green locally and red on ubuntu forever. A green local suite is not evidence about the other leg, and the ubuntu leg is the only place linux-64 is exercised at all (`dev/roadmap.md`, "Deferred: Linux replication").

**`gh` talks to the WRONG REPO here until you tell it otherwise, and fails silently when it does.** This clone has two remotes — `origin` (`tanerumit/blueearth_cst`, where CI runs) and `upstream` (`Deltares/blueearth_cst`) — and `gh` resolves to `upstream`. So a bare `gh run list` queries Deltares, exits 0 and prints nothing, with runs sitting on origin. Read that literally and you conclude CI has never run, which is worse than not looking. Diagnosed 2026-08-12; it is what the earlier note "`gh run list` does not work in this repo" was actually describing.

Fix it once per clone, next to `core.hooksPath`:

```bash
gh repo set-default tanerumit/blueearth_cst   # writes remote.origin.gh-resolved
```

After that every `gh` verb works bare. Until then — and in any script that must not depend on local config — pass `--repo tanerumit/blueearth_cst` explicitly, or go to the API:

```bash
# the latest run, whatever branch it was on
gh api "repos/tanerumit/blueearth_cst/actions/runs?per_page=1" \
  --jq '.workflow_runs[0] | "\(.head_sha[0:7]) \(.status) \(.conclusion)"'

# which STEP failed, per leg -- the only view that separates a lint failure
# from a suite failure from a platform-specific one
gh api "repos/tanerumit/blueearth_cst/actions/runs/<id>/jobs" \
  --jq '.jobs[] | "\(.name) -> \(.conclusion): " +
        ([.steps[] | select(.conclusion=="failure") | .name] | join(", "))'
```

Do **not** filter by `head_sha=<short sha>` — that parameter needs the full 40-character value and silently matches nothing otherwise, which reads as "the run has not started yet" and polls forever.
