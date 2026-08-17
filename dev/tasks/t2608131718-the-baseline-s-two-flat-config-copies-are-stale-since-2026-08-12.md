---
title: The baseline's two flat config copies are stale since 2026-08-12
type: watch-item
area: baseline / test fixtures
origin: config-snapshot P6 verification (2026-08-13)
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — `dev/baseline/manifest.json` records both flat config copies at the
> hash they had on 2026-08-12. Three commits have changed the source config
> since, so a re-run of WF1 or WF2 on `test_local` makes that target FAIL.
> **Why** — Not a defect and nothing numerical moved; but the next person to run
> the baseline gate will hit a red result whose cause is invisible from the
> output, and will have to re-derive this.
> **Effort** — small, but it needs an owner decision, not a fix.

## What was measured

Found 2026-08-13 while verifying that defect H's `river_upa` / `soil_fn`
coupling was value-neutral. WF1 was re-run on `test_local` with `--notemp`,
then `check_baseline.py check`:

- **6 of 7 targets match**, including `run_default/output.csv` (the wflow
  discharge, tolerance comparator) and `results/q_indicators.csv`. Defect H is
  therefore value-neutral **in fact**, not only in argument.
- **1 target differs**: `config/runs/snake_config_model_creation.yml`.

## Why it differs, and why it is not the redesign's doing

The manifest was recorded at `9285ae6` (2026-08-12). Three commits have since
changed `test_case/snake_config_baseline.yml`, which both flat copies are
verbatim copies OF:

- `07a994d` — spell factors moved into `stress_test`
- `086ba7b` — `shared.water_year_start`
- `56b7b56` — the seed configs renamed

The manifest stores the **same** hash for both copies, because both come from
that one source. The tell is which one broke: WF2's copy still matches the old
hash while WF1's does not, and WF1 is the only workflow that was re-run. So the
re-run refreshed WF1's copy to current content and **exposed drift that has been
latent since 2026-08-12** — unnoticed because nothing had re-run WF1 against
`test_local` in between.

## Why it was not re-recorded on the spot

Two reasons, and the second is the one that would have caused damage:

1. A non-empty diff is a stop-and-report by the P6 brief. Re-recording is how a
   real regression gets laundered into the new expected state.
2. **A re-record today would freeze an inconsistent pair.** WF1's copy is
   current and WF2's is not, so the manifest would capture one refreshed and one
   stale hash for two files that are by construction identical.

## Trigger

Re-record when someone next runs **both** WF1 and WF2 against `test_local`
deliberately — a milestone seal, or the next time the baseline is refreshed for
its own sake. Re-running WF2 only to fix this is not worth the time on its own;
nothing consumes the flat copies except the drift guard, which compares them
against each other rather than against the manifest.

Until then, a baseline FAIL naming **only** these two paths is this item and not
a regression. A FAIL naming `output.csv` or `q_indicators.csv` is not.
