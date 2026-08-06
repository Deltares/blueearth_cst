# Migration — indicator-table axis columns (R9 followup)

Date: 2026-08-05
Status: **COMPLETE 2026-08-05** — code landed, suite green, baseline re-recorded.
Attribution evidence in §5.

Purpose: the old → new map mandated by `naming.md` §7 for a tier-2 table-label
rename. Sibling record to `migration_project-tree.md`, which renamed the two
files; this one renames two **columns inside** them. They are separate §7 events
because a consumer that survived the filename rename can still break on the
header.

## 1. The map

| Old | New | Table | Units | Semantics |
| --- | --- | --- | --- | --- |
| `tavg` | `temp_change` | `q_indicators.csv`, `basin_indicators.csv` | degC | **Absolute** shift, read straight from `cst_<m>.csv`'s `temp_mean` |
| `prcp` | `precip_change` | `q_indicators.csv`, `basin_indicators.csv` | % | **Relative** shift, `cst_<m>.csv`'s `precip_mean` factor as `× 100 − 100` |

`cst_0` is the unperturbed baseline, so both axes are `0` there by definition.

Nothing else in either header moves: `statistic`, the optional leading
`realization` (`aggregate_rlz: false`), the `Q_<mapid>` gauge columns, and the
`<var>_basavg` columns are all unchanged.

## 2. Why

`naming.md` §6 tier 2 declares `precip` / `temp` the cross-tool scientific
variable names. Every other producer already obeyed it — the WG-1 extraction
(`precip`/`temp`/`temp_min`/`temp_max`), the HM-2 and WG-6 wflow forcing
(`precip`/`pet`/`temp`), the config's `stress_test.temp` / `stress_test.precip`
blocks, the `cst_<m>.csv` perturbation files (`temp_mean`/`precip_mean`/
`precip_variance`), and the R weather-generator wrapper. The two indicator tables
were the **only** place in the repo that spelled the same two quantities `tavg`
and `prcp`, and they are the tables a user actually opens.

**Why not bare `temp` / `precip`.** Because these columns do not hold the
variable — they hold the **perturbation the member imposes**, and the two axes
are not even the same kind of quantity (one absolute, one relative). `temp` in a
response-surface table would read as a temperature. The `_change` suffix is the
vocabulary the repo already uses for exactly this distinction:
`stress_test.<var>.transient_change` in the config, and
`variables.<var>.change: absolute|relative` in `projections/variable_spec.py`.

**Why not `_delta`.** `delta` implies additive, but `precip_change` is a
percentage; and `temp_delta` is already weathergenr's argument name for a
*monthly vector*, which is a different thing from this scalar.

## 3. What changed

| File | Change |
| --- | --- |
| `blueearth_cst/experiment/export_wflow_results.py` | The producer: four `col_names` lists + the two local variables. A comment now states the units and the absolute-vs-relative split at the point the values are read |
| `blueearth_cst/shared/interchange_contracts.py` | `validate_hm7` + `validate_hm_gauge_column_identity`. The pair is now the module constant `_PERTURBATION_AXIS`, named once so the two validators cannot drift apart on a future rename — they stayed in step here only because one commit touched both |
| `tests/test_interchange_contracts.py` | Existing HM-7 and gauge-identity cases updated, **plus a new `test_hm7_rejects_the_pre_rename_axis_spelling`** |
| `dev/reference/contracts/hydrological-model-seam.md` | HM-7 pinned surface + the gauge-identity check-3 wording |
| `dev/reference/naming.md` | §6 tier 2 (canonical stems + the alias list that is *not* drift) and §7 (this record) |

**No consumer code changed, because there is none.** Rules 3.12 `gather_benchmarks`
and 3.13 `gather_logs` depend on the two files for DAG ordering only; no in-repo
module reads these columns. The declared consumer is the CST-API / GUI, which is
out of this repo's scope by standing decision.

## 4. The rejection test is the point

`validate_hm7` now **fails** on the old header rather than accepting either
spelling. A both-spellings-accepted validator would let a stale writer keep
emitting `tavg` / `prcp` undetected, which is the exact failure a migration
record exists to prevent.

## 5. Gate status

| Gate | Status |
| --- | --- |
| `pytest tests/test_interchange_contracts.py` (worktree) | **PASS** — 38 passed, 26 skipped |
| `pytest tests/test_export_wflow_results.py` | **PASS** |
| `pytest tests/` from the **primary checkout** | **PASS 2026-08-05** — 1356 passed, 8 skipped, 1 xfailed |
| `check_baseline.py check --workflow climate_experiment` | **PASS 2026-08-05** after re-record — 3 targets match |

### ATTRIBUTION — ADR 0001 step 7, settled conclusively

The step-7 immaterial branch asks that the movement be confirmed *consistent with
the recorded wf1 diff* before re-recording. It was possible to do better than
that here: **there is no numeric movement at all.**

**1. Only rule 3.11 re-ran.** Timestamps on the fixture: `results/*.csv` fresh
from the run, but the wflow run CSVs (`hydrology/wflow/output/rlz_*.csv`), the
stress-test parameter files and the wf1 discharge all older. So
`analyze_wflow_results` consumed byte-identical inputs; the only variable was the
code change.

**2. Reverting the header alone reproduces both recorded hashes.** Taking each
new table, replacing `temp_change`→`tavg` and `precip_change`→`prcp` **in the
header line only**, and re-hashing through `check_baseline.fingerprint_csv`'s own
normalisation:

| file | reverted-header sha256 | recorded sha256 | |
| --- | --- | --- | --- |
| `q_indicators.csv` | `b051ba53…a55d653` | `b051ba53…a55d653` | match |
| `basin_indicators.csv` | `6ece285f…47e2c9fd` | `6ece285f…47e2c9fd` | match |

Body bytes below the header compare equal in both files.

**3. Sizes moved by exactly +16 bytes each** — `len("temp_change") - len("tavg")`
= 7 plus `len("precip_change") - len("prcp")` = 9. The header delta and nothing
else. 3838→3854 and 71→87.

**4. Nothing else moved.** The full manifest diff is the two indicator entries
plus the recording provenance (`milestone/r09-project-tree@f054a771` → `main@03e546c`),
which also retires the cross-branch recording warning for this slice.

### This closes the documented residual too

`check_baseline.py`'s docstring warned that a wf3 regen might fail these
fingerprints if the sub-tolerance wf1 restoration delta (`max|dQ|/mean ≈ 1.7e-4`)
survived into them. It has now been tested: an earlier run the same day
regenerated the wf3 run CSVs **from the restored model**, and the resulting tables
still hashed to the pre-restoration recorded values — which is why `check` passed
before the rename landed. The delta does not propagate through the reduction. The
docstring has been updated from a warning to a result.

### Commands used

```
pixi run snakemake all -c 3 -s Snakefile_climate_experiment \
    --configfile config/workflows/snake_config_model_test.yml
pixi run pytest tests/
pixi run python dev/scripts/check_baseline.py check  --workflow climate_experiment
pixi run python dev/scripts/check_baseline.py record --workflow climate_experiment
```

`--workflow climate_experiment` merges into the existing manifest rather than
overwriting, so the wf1 and wf2 rows were preserved (14 targets total).

### What the gates looked like before the run (kept, because the prediction held)

Both failed for different reasons and were listed separately so a reader who
cleared only one would not think they were done:

| gate | predicted | actual |
| --- | --- | --- |
| `pytest tests/` | `test_hm7_integration` + 12 `test_gauge_identity_integration` fail, because `validate_hm7` is supposed to reject the old spelling (§4) | exactly those 13, then all green |
| `check_baseline check` | exactly two entries move; a third moving means something else changed | exactly two moved |

The live concern was that those two entries could move for **two reasons at
once** — the header rename *and* the pre-restoration wf3 provenance catching up
with the restored wf1 slice — so re-recording on sight would have baked in an
unattributed numeric change. The attribution above shows the second cause did not
materialise.

Kept forward: unrounded `float32` (CR-2's C14) will make numeric movement in
these tables MORE visible, not less, which is what Q8's tolerance comparator is
for.
