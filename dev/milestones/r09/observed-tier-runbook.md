# Runbook — producing the R9 observed-tier snapshot

Date: 2026-08-04. Closes master **Gate 1** item 3. Owner action: P1 cannot do
this, because the run must happen in the **primary checkout** and P1 works in a
task worktree.

The observed tier is *one clean three-workflow run, snapshotted as a sorted list
of project-relative paths*. Not a tree copy — the map is a function over paths,
so a path list is the whole input. It is the only tier that carries **undeclared
engine artifacts** (hydromt's, Wflow.jl's, weathergenr's), which appear in no
`output:` declaration and which `--dry-run` structurally cannot see.

---

## Before you start

| Requirement | Why |
| --- | --- |
| Run in `C:\Users\taner\workspace\blueearth_cst` (branch `main`) | Two `.snakemake/` stores over one `project_dir` disagree and both hold locks (`AGENTS.md`). Never a worktree. |
| Julia on `PATH` (juliaup) | Not in the pixi env; WF3 needs it. |
| Inside `pixi shell`, or prefix each command with `pixi run` | So `snakemake` resolves to the pixi env. |

The tree must be the **pre-migration** shape, which `main` still emits — P1
changed no runtime code and P2 has not started.

```powershell
cd C:\Users\taner\workspace\blueearth_cst
git status --short          # expect clean
git branch --show-current   # expect main
pixi shell
```

Two paths are used throughout:

```powershell
$P    = "test_case/test_local"                                        # project_dir
$CFG  = "config/workflows/snake_config_model_test.yml"                # the seed config
$WT   = "C:\Users\taner\workspace\.worktrees\blueearth_cst\r09-p1-comparator"
$SNAP = "$WT\dev\milestones\r09\observed_inventory.txt"
```

---

## Step 0 — find the orphans instead of guessing them

**`test_case/test_local` is a mixed-era tree.** A run into an existing
`project_dir` inherits whatever is already there, so pruning has to happen
*before* the snapshot or the snapshot bakes the orphans into the contract.

Rather than delete from a list written months ago, let the comparator name them.
Snapshot the tree **as it is now** and run the falsifier — every `UNMAPPED` line
is either a real map gap or a leftover:

```powershell
pixi run python "$WT\dev\scripts\snapshot_project_tree.py" --config $CFG --quiet
```

No `--out`, so **nothing is written** — this only inspects. The experiment name,
the historical store key and `clim_project` are derived from `$CFG`, so there is
no `--dataset-key` to mistype (a wrong key turns a mapped store into an unmapped
one, which reads as a map gap that is not there).

The script lives in the task worktree while the tree lives beside the primary
checkout; a relative `project_dir` resolves against the **current directory**,
as Snakemake resolves it, so run it from the primary checkout as above. Pass
`--project-dir` if you need to point it somewhere else.

Read the `UNMAPPED` lines. Each is one of:

- a **known orphan** → prune it in step 1;
- a **new map gap** → stop and report it; that is another F1-class ruling, not
  something to delete.

**Measured 2026-08-04** against the tree as it stands (240 files):

```
UNMAPPED PATHS: 240 paths, 119 moved, 97 identity (by rule),
                1 deleted-by-design, 23 unmapped
```

All 23 are orphans, **none is a map gap** — 22 numbered pre-merge log files
directly under `logs/`, plus `experiments/experiment/config/deltares_data.yml`.
So step 0 discriminates, which is the property that makes it worth running
before deleting anything. (The `1 deleted-by-design` is an `indicators/RT_*.csv`
left over in the tree, correctly classified rather than reported as a gap.)

This is a **runbook smoke test, not the observed tier.** `test_case/test_local`
is explicitly excluded as an inventory — it is a mixed-era tree whose orphans
are deliberately unmapped. The observed tier is the snapshot taken in step 3,
*after* the prune and the clean run.

## Step 1 — prune

Three classes, three mechanisms. **All three default to reporting**; run each
without `--delete` first and read the output.

```powershell
# 1a. WF2 series cache (orphaned + pre-4b key grammars)
pixi run python dev/scripts/prune_series_cache.py --config $CFG
pixi run python dev/scripts/prune_series_cache.py --config $CFG --delete

# 1b. Stale historical climate stores (a changed source or window strands one)
pixi run python dev/scripts/prune_climate_store.py --config $CFG
pixi run python dev/scripts/prune_climate_store.py --config $CFG --delete
```

**1c. By hand — no script covers these.** Verified present on 2026-08-04;
re-check against step 0's output before deleting, and delete nothing that step 0
did not name.

| Path | Why it is an orphan |
| --- | --- |
| `$P/logs/_parts/2.02_monthly_stats_hist/` | pre-step-4d WF2 rule name; the live labels are `2.01_fetch_gcm_raw`, `2.02_reduce_gcm_series`, `2.04_derive_change_factors`, `2.06_plot_climate_proj_timeseries`, `2.11_extract_climate_grid` (`Snakefile_climate_projections`, `LOG_RULES`) |
| `$P/logs/_parts/2.03_monthly_stats_fut/` | same |
| `$P/logs/_parts/2.04_monthly_change/` | same — note the live `2.04_derive_change_factors.log` beside it is **not** an orphan |
| `$P/experiments/experiment/logs/3.*.log` and `3.*/` **at the top level** | pre-`_parts/` shape. WF3 now writes parts to `experiments/<id>/logs/_parts/` and merges them into `wf3_climate_experiment.log` |
| `$P/logs/1.*.log`, `$P/logs/2.*.log` **at the top level** | the same pre-`_parts/` shape at project scope — 22 files, several from retired rule numbers (`1.10_extract_climate_grid_wf1`, and both `1.10_plot_results` and `1.11_plot_results`). Not in the map doc's fixture-orphan list; found by step 0 |
| `$P/experiments/experiment/config/deltares_data.yml` | superseded by `config/catalogs/` |

`logs/wf2_climate_projections.log` sits in the same directory and is **live** —
it is the merged product, not a part. The glob below is `[0-9]*.log`, which
matches only the numbered rule logs, precisely so a `*.log` sweep cannot take it.

```powershell
Remove-Item -Recurse -Force "$P/logs/_parts/2.02_monthly_stats_hist",
                            "$P/logs/_parts/2.03_monthly_stats_fut",
                            "$P/logs/_parts/2.04_monthly_change"
Remove-Item -Force          "$P/logs/[0-9]*.log"
Remove-Item -Recurse -Force "$P/experiments/experiment/logs/*"
Remove-Item -Force          "$P/experiments/experiment/config/deltares_data.yml"
```

`merge_logs` reads only the labels in `LOG_RULES`, so an orphan part dir is never
merged — it is a snapshot problem, not a correctness one. Clearing the WF3
`logs/` directory wholesale is safe because every file in it is regenerated by
the run in step 2.

## Step 2 — one clean three-workflow run, in order

```powershell
snakemake all -c 3 -s Snakefile_model_creation      --configfile $CFG
snakemake all -c 3 -s Snakefile_climate_projections --configfile $CFG --keep-going
snakemake all -c 3 -s Snakefile_climate_experiment  --configfile $CFG
```

If a run crashes and the workdir reports as locked:
`snakemake --unlock -s <Snakefile> --configfile $CFG`.

Order matters — `climate_experiment` consumes `model_creation` artifacts.

## Steps 3 and 4 — snapshot and check

Same command as step 0, now with `--out`. The snapshot is written with a
provenance header — both commits, the config, the derived experiment name and
store key — and the map check runs over the same path list, so the recorded file
and the checked list cannot disagree.

```powershell
pixi run python "$WT\dev\scripts\snapshot_project_tree.py" --config $CFG --out $SNAP
echo "exit=$LASTEXITCODE"
```

Keep the full table (no `--quiet`): the `MOVED` and `IDENTITY (rule)` rows *are*
Gate 1's evidence — the map applied to a pre-migration tree, showing the intended
post-migration paths.

**Exit 0 and `MAP CLEAN` closes Gate 1 item 3.** Then run it once more with
`--gap-rules`: the difference between the two runs is the answer to **F2** —
whether `hydrology_model/instate/` and a directory-wide `hydrology_model/plots/`
exist at all, which is the only thing still blocking those two rows from being
ruled.

```powershell
pixi run python "$WT\dev\scripts\snapshot_project_tree.py" --config $CFG --gap-rules --quiet
```

<details>
<summary>Doing it without the wrapper</summary>

The wrapper is `Get-ChildItem` plus `semantic_tree_diff --check-map`. If you
need the pieces separately — a different exclusion, a tree with no config
beside it — the equivalent is:

```powershell
$ROOT = (Resolve-Path $P).Path
Get-ChildItem -Path $P -Recurse -File -Force |
  ForEach-Object { $_.FullName.Substring($ROOT.Length + 1).Replace('\','/') } |
  Where-Object { $_ -notlike '.snakemake/*' } |
  Sort-Object -Unique |
  Set-Content -Encoding utf8 $SNAP

cd $WT
pixi run python dev/scripts/semantic_tree_diff.py `
    --check-map $SNAP --milestone r09 `
    --experiment-name experiment --dataset-key era5_20000101_20201231
```

`--dataset-key` is the one to get right by hand: a wrong key turns a mapped
store into an unmapped one, which reads as a map gap that is not there. That is
the argument the wrapper exists to derive.
</details>

---

## Reading the result

**Expected differences from the declared tier — neither is a defect:**

- **`temp()` artifacts are absent.** The per-realization forcing
  (`hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc`) is `temp()`-wrapped and
  deleted once consumed, so the observed list is *shorter* here than the
  declared one. Absence is expected; the map rows for them are covered by the
  row-driven unit test instead.
- **Undeclared artifacts are present.** `hydromt.log`, `hydromt_data.yml`,
  extra `staticgeoms/*`, `run_default/*` beyond `output.csv`, `evaluation/*`,
  weathergenr's `plots/*.png` and `output/{sim_dates,resampled_dates}.csv`,
  `config/generated/wflow_build_model_run.yml`, `store_region.geojson`, and
  Wflow's `log.txt`. **This is the point of the tier.** Each already has a map
  row; this run is what proves the rows match reality.

**A non-zero exit is not automatically a defect in the map.** Triage each
`UNMAPPED` line:

| Looks like | It is | Do |
| --- | --- | --- |
| a pre-merge log shape, a renamed rule's part dir, a superseded catalog | a missed orphan | prune it, re-snapshot; no map change |
| an engine artifact at a path no row covers | a real map gap | **stop and report** — an F1-class owner ruling, like F1a–F1c |
| a path under a directory the map routes wholesale | a row written too narrowly | same — report, do not widen the rule quietly |

## When it is done

Commit the snapshot under `dev/milestones/r09/` and hand back the `--check-map`
output. Then Gate 1 can close and P2 can begin.

Costs and caveats: the run is the expensive part (hydromt build, CMIP6 fetch,
`RLZ_NUM × ST_NUM` Wflow runs) — the prune in step 1 deliberately keeps the
CMIP6 `raw/` and `scalar/` caches and the historical store, so it is a warm run.
A cold run into a fresh empty `project_dir` would be orphan-free by construction
and is the higher-confidence option, at the cost of re-fetching everything; if
you would rather do that, skip step 1 entirely and point `project.project_dir`
at an empty directory outside the repository tree.
