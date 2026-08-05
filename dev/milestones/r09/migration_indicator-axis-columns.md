# Migration — indicator-table axis columns (R9 followup)

Date: 2026-08-05
Status: **code COMPLETE**, **baseline re-record OUTSTANDING** (see §5)

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
| `pytest tests/test_interchange_contracts.py` (this worktree) | **PASS** — 38 passed, 26 skipped |
| `pytest tests/test_export_wflow_results.py` | **PASS** — covers `realization_from_run_csv` only; untouched by this change |
| `pytest tests/` from the **primary checkout** | **OUTSTANDING — expected to FAIL** (see below) |
| `check_baseline.py check` | **OUTSTANDING — expected to FAIL** (see below) |

**Both outstanding gates have the same trigger and one fix: re-run WF3 from the
primary checkout.** They are listed separately because they fail for different
reasons and a reader who clears only one is not done.

**Why `pytest tests/` fails there but passed here.** This worktree has no
`test_case/test_local`, so the 26 fixture-dependent cases skip. The primary
checkout HAS that tree, and its `q_indicators.csv` / `basin_indicators.csv`
still carry the pre-rename `tavg` / `prcp` headers. Two integration cases parse
them and will now fail — correctly, because `validate_hm7` is supposed to reject
the old spelling (§4):

| Test | Why it fails on a stale tree |
| --- | --- |
| `test_hm7_integration` (`test_interchange_contracts.py:640`) | Reads both tables; the axis columns are missing under their new names |
| `test_gauge_identity_integration` (`:653`, 12 parametrized cases) | Check 3 derives the gauge set as *header minus `statistic` minus `_PERTURBATION_AXIS`*; a stale header leaves `tavg`/`prcp` in that set, so the list-equality against `output_rlz` breaks |

This is **the branch's merge gate**, not a cosmetic one — `AGENTS.md`'s
validation ladder runs `pytest tests/` before merging.

**Why `check_baseline.py check` fails.** `dev/baseline/manifest.json`
fingerprints both tables **byte-exact** (`sha256`, `type: csv`) at
`test_case/test_local/experiments/experiment/results/`. A header change moves
both hashes. The re-record diff must be confined to those two `sha256` /
`size_bytes` pairs — any third entry moving means something other than this
rename also changed.

**DO NOT re-record on sight of those two moving — they can move for TWO reasons
at once.** `check_baseline.py`'s module docstring records a **mixed-provenance
baseline**: since the constant-parameter restoration the wf1 slice reflects the
RESTORED model, while the wf2/wf3 rows are still the PRE-restoration recording
(wf3 was deliberately not re-run, because the discharge move was immaterial —
0/7670 timesteps over tolerance). So this re-run is the first time that
sub-tolerance wf1 delta (`max|dQ|/mean ≈ 1.7e-4`) can reach these two tables,
and it lands in the same two entries as the header rename.

Follow **ADR 0001 step 7, the immaterial branch**: re-run wf3, confirm the
movement is consistent with the recorded wf1 diff
(`dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md`), then
re-record the wf3 slice **with a note**; else stop and investigate. Unrounded
`float32` values (CR-2's C14) will make this MORE visible in future, not less —
which is what Q8's tolerance comparator is for.

Commands, from the primary checkout:

```
pixi run snakemake all -c 3 -s Snakefile_climate_experiment \
    --configfile config/workflows/snake_config_model_test.yml
pixi run pytest tests/
pixi run python dev/scripts/check_baseline.py check  --workflow climate_experiment
# only after the ADR 0001 step-7 consistency check:
pixi run python dev/scripts/check_baseline.py record --workflow climate_experiment
```

`--workflow climate_experiment` merges into the existing manifest rather than
overwriting it, so the wf1 and wf2 rows are preserved.

Order: run WF3 from the primary checkout (the `.snakemake` metadata rule in
`AGENTS.md`), then `pytest tests/`, then the consistency check, then re-record.
