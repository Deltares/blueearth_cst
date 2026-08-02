# P3-1 baseline re-record — milestone diff evidence (commit 6)

Gate-6 record for the P3-1 experiment-structure milestone
(`dev/milestones/p31/experiment-structure-design.md` §6/§6a/§6b). Verdict up front:
**value-identity HOLDS — zero value-level diffs across two independent diff
runs (pre- and post-e2e); the rollback trigger never tripped.** All residuals
are presence exemptions or path-only pointer moves, each adjudicated below.

## Procedure

1. Pre-change reference tree `examples/test_local_pre_p31/` (153 files) —
   captured in Phase A on `main` `7d1be713` (wf1 then wf3, tracked test config
   with only `project_dir` swapped), FROZEN 2026-07-23.
2. Old `examples/test_local` renamed aside to
   `examples/test_local_superseded_pre_p31` (no deletion; regen lands clean —
   stale old-layout files in place would have matched the reference and masked
   the move).
3. Post-change regen (branch tip, commit 5): wf1 then wf3 ONLY (matching the
   reference capture's §6a shape), tracked config
   `config/workflows/snake_config_model_test.yml`, into a fresh
   `examples/test_local`. wf1 exit 0 (14/14), wf3 exit 0 (57/57).
4. Semantic diff (BEFORE any wf2 run): `semantic_tree_diff.py`
   ref=`examples/test_local_pre_p31` cur=`examples/test_local`, path map +
   allowlist + path-aware toml comparator active
   (`--experiment-name experiment --dataset-key era5_20000101_20201231
   --allow experiments/experiment/config/deltares_data.yml`).
5. Full three-workflow e2e via `scripts/run_workflows.py` (all three enabled),
   then a SECOND semantic diff on the final tree.
6. Re-record: `check_baseline.py record --workflow climate_experiment`; then
   scoped checks (wf3 green at new paths; wf1 value targets pass WITHOUT
   re-record).

## First diff run (pre-5b) — MISMATCH, fully adjudicated

The first run, with the commit-5 comparator, reported
`MISMATCH: 86 files compared, 19 failed, 0 missing, 1 extra, 2 allowlisted`.
Every entry was adjudicated by parse-level deep leaf-diff (driver-run probe,
2026-07-23) before any mechanism change:

- **17 yml pairs — ALL PATH-ONLY (0 non-path leaf diffs across all pairs):**
  - 2 config snapshots (`config/snake_config_model_creation.yml`, the wf3
    snapshot): sole differing leaf `project.project_dir` — each tree records
    its OWN root (`examples/test_local_pre_p31` vs `examples/test_local`), a
    cross-root capture artifact, not behavior.
  - 14 weathergen configs (`weathergen_config.yml` + 12 rlz/cst variants +
    catalog): sole differing leaf a root-prefixed output path also carrying
    the designed `climate_experiment/` → `experiments/experiment/` move.
  - the experiment data catalog (`data_catalog_climate_experiment.yml`):
    14 leaves, all absolute `uri`s under each root + the layout move.
- **3 run-log files** (`hydrology_model/hydromt.log`,
  `model_runs/log.txt`, `run_default/log.txt`): timestamp noise; run-log
  FILES outside the already-excluded `logs/` dirs.
- **1 EXTRA** `experiments/experiment/config/deltares_data.yml`: the
  de-collided per-experiment catalog copy — in the OLD layout wf3
  `copy_config` overwrote wf1's project-level copy (the exact collision P3-1
  fixes). Verified byte-identical to both the cur and ref project-level
  copies (Get-FileHash, all three equal).

These are the same behavior-neutral pointer-move class the design's ext1-3
toml comparator formalizes, in YAML files §6a's inventory missed. Commit 5b
(`576b6a6`) extends `semantic_tree_diff.py` with the mirrored mechanism
(cross-root `<PROJECT_ROOT>` normalization of path-prefixed string leaves,
ref-side path map; non-path leaf drift still FAILs) and excludes run-log
files (`*.log`, `log.txt`) as non-content-bearing.

## Final diff — CLEAN (pre-e2e, wf1+wf3 tree)

```
ALLOWED (EXTRA allowed: climate_historical/era5_20000101_20201231/.guard_ok)
ALLOWED (EXTRA allowed: experiments/experiment/.project_consistency_ok)
ALLOWED (EXTRA allowed: experiments/experiment/config/deltares_data.yml)
CLEAN: 83 files compared, 0 failed, 0 missing, 0 extra, 3 allowlisted
```

## Post-e2e diff — 0 failed, 0 missing (final tree, after wf3 re-ran)

The wrapper e2e ran wf2 fresh; its new snapshot flipped the guard's wf2
digest param ABSENT→digest, so the guard + the four per-experiment roots +
downstream wf3 legitimately re-ran (the designed §3b ABSENT→present
behavior, not thrash) — wf3 results were REWRITTEN at 23:37. A second diff
against the reference:

```
MISMATCH: 83 files compared, 0 failed, 0 missing, 15 extra, 3 allowlisted
```

**0 failed** = the re-run reproduced every reference value exactly (live
confirmation of the R5-verified wf3 determinism, seed 123). The 15 EXTRA are
ALL wf2 overlay artifacts (13 under `climate_projections/cmip6/` +
`config/cmip6_data.yml` + `config/snake_config_climate_projections.yml`) —
present only because the e2e ran wf2, which the §6a reference capture
deliberately omits (wf1-then-wf3 shape). Not part of the wf3 milestone-diff
scope; no value content compared against nothing.

## Allowlist entries + justifications

| Entry | Class | Justification |
| --- | --- | --- |
| `experiments/experiment/.project_consistency_ok` | EXTRA-by-design | per-experiment guard sentinel; new gate output, no pre-P3-1 counterpart, no scientific content |
| `climate_historical/era5_20000101_20201231/.guard_ok` | EXTRA-by-design | key-level guard artifact (ext2-1); same class |
| `experiments/experiment/config/deltares_data.yml` (via `--allow`) | EXTRA-by-design | de-collided per-experiment catalog copy; byte-identical to the project-level copy (verified); the old layout's overwrite WAS the collision P3-1 fixes |

Nothing is MISSING-by-design (no wf3 plots producer). Review rule honored:
every entry enumerated + justified here and in
`dev/milestones/p31/migration_experiment-structure.md`.

## Toml comparator outcomes

All run tomls paired via the path map and compared with the §6a step-3
project-root-relative comparator: **zero failures** across both diff runs.
`path_static` / `state.path_input` resolve to unmoved `hydrology_model/`
targets (no map entry needed); `path_forcing` translates via the
`climate_experiment/` → `experiments/experiment/` prefix rule (its `temp()`
target exists in neither tree — the prefix-rewrite form is what makes it
pass). No toml allowlist entries.

## §6b provenance note

Both trees are regenerated on the RESTORED wf1 model (the reference tree was
captured in Phase A, post-const-pars-restoration `main`), so the documented
immaterial wf1-propagated residual (max|dQ|/mean ~ 1.7e-4 surviving into wf3
rounding) was not expected to appear — and **did not appear**: the discharge
comparator and all wf3 CSVs compared byte-/value-clean in both runs. This
re-record also moves the wf3 manifest rows onto the restored model, closing
the mixed-provenance caveat (`baseline-manifest-coverage`) for the wf3 slice.

## Re-record summary (old vs new fingerprint paths; value identity)

`check_baseline.py record --workflow climate_experiment` (tracked config) —
3 targets re-recorded at the NEW paths, values identical to the old slice by
the diff evidence above:

| Old manifest path | New manifest path |
| --- | --- |
| `{project_dir}/climate_experiment/model_results/Qstats.csv` | `{project_dir}/experiments/experiment/model_results/Qstats.csv` |
| `{project_dir}/climate_experiment/model_results/basin.csv` | `{project_dir}/experiments/experiment/model_results/basin.csv` |
| `{project_dir}/config/snake_config_climate_experiment.yml` | `{project_dir}/experiments/experiment/config/snake_config_climate_experiment.yml` |

Post-re-record checks:

- `check --workflow climate_experiment`: **OK — 3/3 targets match.**
- `check --workflow model_creation`: all VALUE targets (discharge
  `output.csv`, csv, png) **pass without re-record** — no value regression.
  The single failing row is the wf1 config-snapshot sha, whose recorded
  hash predates R6's config three-bin split (`e705965` rewrote the tracked
  config's catalog paths) — the R6-adjudicated "copied-config snapshot"
  standing condition, P3-1-independent. Same for the wf2 snapshot row.
  **Recommended follow-up (out of P3-1 scope per the G1 amendment):** a
  `chore:` re-record of the two stale yaml snapshot rows.
- The known wf2 2-CSV serialization nondeterminism (dev/milestones/r04/baseline_diffs.md)
  did **not** fire on this regen (both summary CSVs matched the manifest).
