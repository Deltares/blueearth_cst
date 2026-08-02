# R01 baseline diffs

**Date.** 2026-07-18 (sealing date).

> **Update 2026-07-18 (post-seal).** The baseline has since been **rebuilt**
> from a now-tracked, simplified config (`config/snake_config_model_test.yml`:
> project_dir `examples/test_local`, 3 models, single `far` horizon). All
> three workflows were re-run and `dev/baseline/manifest.json` re-recorded;
> `record` → `check` round-trips clean. The "stale M2b, left untouched"
> narrative below is now historical — it explains why R01 *itself* sealed on
> invariance-by-construction rather than re-recording. The untracked
> `*_local.yml` seed that caused the divergence is retired. See
> `dev/followups.md` → "Cross-cutting — baseline manifest integrity".

## Summary

R01 seals on **scientific invariance by construction**, not on a manifest
re-record. The `dev/baseline/manifest.json` contract is **left untouched**.

During Task 5 (the planned re-baseline against `examples/test_local`), a
fresh run of all three workflows against the migrated canonical config was
compared to the M2b manifest. The comparison surfaced far more than the
three expected config-snapshot diffs — including scientific targets. On
investigation, **none of that drift is caused by R01**: it is a
pre-existing mismatch between the M2b manifest and the current tracked
canonical config, which R01's Task 5 merely exposed (it is the first time
`check_baseline check` was run against the canonical model set). Recording
a new manifest from that run would have laundered the pre-existing drift
into the R01 scientific contract, so it was not done.

## Why the M2b manifest can't be reproduced

The M2b baseline manifest (`dev/baseline/manifest.json`, last recorded
2026-05-08, commit `159e197`) was recorded from an **untracked, 3-model
local config** — corroborated by `dev/followups.md` ("the full 3-model ×
2-scenario fetch") and by the manifest itself
(`annual_change_scalar_stats_summary.nc` records `model` dimension shape
`[3]`). The tracked canonical config (`config/snake_config_model_test.yml`)
has carried an **8-model** list since before the `pre-r01` checkpoint —
including at M2b. The two have been silently divergent since M2b; nobody
caught it because the workflow smoke tests only assert non-empty outputs
and never compare against the manifest.

Consequences observed when a fresh 8-model canonical run is checked
against the 3-model M2b manifest:

| Target(s)                                                         | Diff | Cause (not R01) |
| ----------------------------------------------------------------- | ---- | --------------- |
| `annual_change_scalar_stats_summary.{nc,csv}`, `_mean.csv`        | shape/values | 8 models vs 3 in the manifest |
| `plots/…/precipitation_anomaly_projections_abs.png`, `projected_climate_statistics.png` | size | 8 vs 3 models |
| `plots/wflow_model_performance/{basin_area,hydro_wflow_1,precip}.png` | size 19–32% | model-**independent** drift of undetermined origin (workflow 1 never reads CMIP6) |
| `climate_experiment/model_results/Qstats.csv`                     | sha  | downstream of workflow 1 |
| `config/snake_config_{model_creation,climate_projections,climate_experiment}.yml` | sha | the only genuinely R01-expected drift (sectioned schema → different bytes) |

The model-independent PNG drift (workflow 1 does not use the climate model
list) is the decisive evidence that the M2b local config differed from the
canonical in ways beyond the model count and **cannot be reconstructed** —
so "reproduce the baseline config and re-verify" is a dead end.

## Why R01 is scientifically neutral regardless

R01 changed the *shape* of the config, not any value or any computation:

- **Value-preservation gate passed** on every migrated leaf of both the
  tests config and the canonical config (`git show HEAD:<path>` vs the new
  sectioned path; see plan Step 2.15). Every scientific value is identical;
  only its location in the YAML changed.
- **The only algorithmic touches are identity-preserving.** The two
  list/string normalizations (`_to_str_tuple` in `get_change_climate_proj.py`
  and the inline guard in `get_change_climate_proj_summary.py`) yield the
  same result from a list `[2030, 2060]` as from the old comma-string
  `"2030, 2060"` — same tuple, same horizon midpoint year (2045). No number
  moves.
- **Suite green:** `51 passed, 3 skipped, 2 xfailed` (the pre-R01 47 plus 4
  focused R01 reader/normalization tests; no pre-existing test changed
  outcome).
- **All three Snakefiles dry-run** to their expected states (1 clean, 2
  known-failure ratchets), with no `KeyError` / `AttributeError`.

## Deferred: rebuild the baseline against current libraries

The M2b manifest is stale (3-model config; M2b library era). Rebuilding it
cleanly — a deliberately chosen model set, current libraries, a *tracked*
seed config so the baseline is reproducible — is tracked in
`dev/followups.md` and belongs to a dedicated task, not R01. Until then,
`dev/baseline/manifest.json` remains the M2b contract of record, with this
document as the note explaining why R01 did not re-record it.
