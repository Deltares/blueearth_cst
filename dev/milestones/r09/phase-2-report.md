# R9 P2 report — migrate the generated tree

Date: 2026-08-04. Branch: `feat/r09-p2-tree-migration` (cut from
`milestone/r09-project-tree` at the P1 merge). Brief:
[`phase-2-tree-migration-task-brief.md`](phase-2-tree-migration-task-brief.md).

**Status: the four planned rows are landed and the tree is migrated. Two
decisions are open before the phase can be called complete**, both listed under
*Open decisions*. `check_baseline` is red by construction from commit 1 until
P3 re-records.

---

## The four rows

| # | Commit | Invariant preserved |
| --- | --- | --- |
| 1 | `r09: move the wflow model under models/` | model root resolves; WF3 still finds it |
| 2 | `r09: move spatial and climate data under data/` | store key stays experiment-invariant |
| 3 | `r09: restructure the experiment's engine subtrees` | `path_log` shipped WITH the flattening |
| 4 | `r09: route DAG renders out of config/` | generated artifacts leave the editable root |

Plus four unplanned commits, all for defects **no dry-run and no unit test could
reach**. That ratio is the phase's main lesson and is discussed under *What the
ladder caught*.

| Commit | Cause |
| --- | --- |
| `r09: repoint the three model-root sites commit 1 missed` | my error — a truncated survey |
| `fix(wf1): order plot_map after every writer of staticmaps.nc` | pre-existing since R7 |
| `r09: read the realization index from the file name...` | my error — checked the producer, not the consumer |
| `test(r09): add the P2 concurrency falsifier` | the brief's required falsifier |

## Ladder results, and what each rung caught

The brief asks for what each rung **caught**, not only that it ran.

| Rung | Result | What it caught |
| --- | --- | --- |
| 1 Narrow (per row) | green each time | test path expectations, nothing structural |
| 2 `pixi run test-cli` (per commit, ×5) | 12 passed each | **nothing** — every real defect was invisible to it |
| 3 `pixi run test-fast` (phase end) | 1216 passed | nothing |
| 4 Full three-workflow run | green after 3 fixes | **all three defects** |
| 5 `semantic_tree_diff` whole-tree | see below | a map gap + a possible P2 gap |

**Rungs 1–3 caught nothing that mattered, and that is the finding.** Every
defect this phase introduced or surfaced was a path built at RUNTIME inside a
script: the DAG resolves, the dry-run is clean, the unit tests pass, and the job
dies on execution. The brief's insistence that this phase needs a real run is
now evidence-backed rather than assumed.

## Whole-tree comparison — the P1 comparator, used for its purpose

Pre-migration reference: `test_case/test_local` (the P1 observed tier).
Post-migration: a cold three-workflow run into a dedicated `project_dir`.

```
python dev/scripts/semantic_tree_diff.py --milestone r09 \
    --ref  <primary>/test_case/test_local \
    --cur  <run>/r09_p2_post \
    --experiment-name experiment --dataset-key era5_20000101_20201231 \
    --ref-token test_case/test_local

MISMATCH: 160 files compared, 31 failed, 22 missing, 20 extra
```

Every residual was classified, not skimmed:

| Class | Count | Adjudication |
| --- | ---: | --- |
| **NUMERIC** — values, tolerance, NaN masks | **0** | — |
| **STRUCTURE** — dims, coords, variable sets | **0** | — |
| CMIP6 global attrs | 29 | allowlisted — not migration-caused |
| Path strings inside content | 2 | one is a real map gap (F3) |
| Digest bundles missing+extra | 17 + 17 | allowlisted — `project_dir` in the digest |
| Stale in the REFERENCE | 2 | allowlisted |
| Deferred to P3 | 2 + 2 | allowlisted — explicit P2 non-goal |
| Possible P2 gap | 1 + 1 | **open decision** |

**Zero numeric and zero structure failures is the program's premise holding.**
No moved artifact changed value.

`--ref-token test_case/test_local` is required and was missed on the first
invocation, costing 15 spurious failures: the reference tree RECORDS a relative
`project_dir` while being read from an absolute path, which is exactly the case
P1 built that flag for.

### The written allowlist, one reason per entry

| Entry | Count | Reason |
| --- | ---: | --- |
| `data/climate/projections/cmip6/{raw,scalar}/*.nc` global attrs | 29 | **Not migration-caused.** All values, coords and dims matched; only inherited CMIP6 provenance differs — `variable_id` reads `tas` on one side and `pr` on the other, with `tracking_id` and `status` likewise. The merged raw slice inherits its global attrs from whichever member won the fetch merge, so two independent fetches disagree regardless of any move. See F4. |
| `config/runs/<workflow>/<digest>/**` | 17 + 17 | The bundle name is a hash over the parsed config, which **includes `project_dir`** — measured in P1. Two trees under different roots necessarily carry different digest directories. |
| `data/climate/historical/<key>/store_region.geojson` | 1 | **Stale in the reference.** No current rule writes it; `climate_store_spec.outputs` is `climate_nc` alone (plus `oro_nc` on chirps). The reference's copy dates from an earlier era — consistent with P1's mtime sweep, which classified it as legitimately older than that run. |
| `models/hydrology/wflow/staticgeoms/meta_basins_highres.geojson` | 1 | Same class: present in the reference, written by no current build. |
| `experiments/<id>/indicators/{Qstats,basin}.csv` | 2 + 2 | **Explicit P2 non-goal.** "No result-table renames — that is P3." Reported as MISSING at the mapped destination and EXTRA at the source, which is exactly what an unimplemented-by-design row looks like. |

---

## Findings

### F1 — three model-root sites survived commit 1 — **my error**

Commit 1 asserted "every model-internal path is built from `basin_dir`,
verified". That was true for the **Snakefiles** and I extended it to the scripts
on a grep truncated with `head -20`. Three sites sorted past the cut:
`shared/plot_map.py`'s `MODEL_DIRNAME` (which failed the run),
`projections/get_stats_climate_proj.py` (which would have failed WF2 next), and
`dev/scripts/preview_basin_map.py`.

The correcting commit says so rather than patching quietly. The re-run of the
survey **without truncation** is what found the second and third.

### F2 — rule 3.11 read the realization index from the directory — **my error**

Commit 3 verified the **producer** — `downscale_climate_forcing.py` derives
everything from the declared TOML path, so the flattening cost it nothing — and
did not check the **consumer**. `export_wflow_results.py` read the index from
the grandparent directory, which R7 had put it in.

Caught at rule 3.11 after all twelve members had already run. The fix matches an
anchored full-stem regex rather than splitting on `_`, because the stem now
carries two indices and `split("_")[-1]` returns the **CST member number** — a
plausible integer, no exception, every result row mislabelled. Two tests pin
that specifically.

**The pattern in both F1 and F2 is the same: I verified one side of a contract
and generalised to the other.**

### F3 — the R9 path map has no rule for the bare `weather_generator/` directory

`weathergen_config.yml` carries `generateWeatherSeries.output.path` as the bare
directory `experiments/<id>/weather_generator/`. The map has prefix rules for
`output/`, `config/`, `_work/` and `plots/` beneath it, but none for the
directory itself, so the leaf falls through unmapped and reads as a content
regression.

**R7 hit this exact case and has a test for it** —
`test_r07_bare_realization_dir_maps_to_the_generator_output_dir`, whose comment
explains that `compare_yaml`'s cross-root leaf normalization feeds bare
directory strings through the map. R9's equivalent is missing. A genuine map
gap, found by using the comparator rather than by reasoning about it.

Not fixed here: `dev/scripts/semantic_tree_diff.py` is **not** in P2's permitted
scope, and amending the map is an owner decision. See *Open decisions*.

### F4 — WF2's raw fetch inherits global attrs nondeterministically

The 29 CMIP6 attr failures are not about R9. Two independent fetches of the same
slice produce datasets whose **global attrs come from different members of the
merge** — `variable_id` is `tas` on one side and `pr` on the other — while every
value, coordinate and dimension matches.

Pre-existing, unrelated to the tree move, and it makes any cross-tree comparison
of `cmip6/raw/` and `cmip6/scalar/` noisy for a reason that has nothing to do
with the change under test. Worth a follow-up: either normalise the merged
dataset's provenance attrs, or add them to `VOLATILE_NC_ATTRS`.

### F5 — `plot_map` raced with `staticmaps.nc`'s writers — pre-existing since R7

Rule 1.12 reads `staticmaps.nc` straight off disk but declared only the gauges
layer, so nothing ordered it after rule 1.04. Under `-c 1` they never overlap and
the ordering held by accident; a cold `-c 3` run schedules them together and
1.12 dies with **no Python traceback** — `HDF5_USE_FILE_LOCKING = "FALSE"` in the
pixi env means the concurrent read is unprotected and aborts below Python.

**My first fix was wrong and the record says so.** Declaring
`ancient(staticmaps.nc)` did not help, because **rule 1.04 writes that file as an
undeclared side effect** — its only declared output is
`reservoirs_lakes_glaciers.txt` — so Snakemake attributes staticmaps.nc to rule
1.03 and the new edge ordered 1.12 where it already sat. The working anchor is
`.outputs_configured`, rule 1.05's completion sentinel.

**Underlying defect, not fixed:** 1.04's undeclared write. Declaring it as an
output would order every reader correctly without per-rule sentinels, but it
changes what the rule claims to produce and belongs with a baseline re-record.

### F6 — `AGENTS.md` is wrong about the shared pixi environment

It states the env is shared and "a worktree resolves to the primary's copy
instead of building its own". **It does not.** Each worktree carries its own
tracked `pixi.toml`, so pixi creates a separate `.pixi/` beside it. The P2
worktree had its own env throughout, and WF3 failed there with
`there is no package called 'weathergenr'` because that package comes from
`pixi run install` (remotes), not from `pixi install`.

This matters beyond one session: the same passage tells a task that changes
`pixi.toml`/`pixi.lock` it must build its own env "rather than inherit" — advice
premised on an inheritance that does not happen. **P5** territory; flagged, not
edited.

It also corrects my own reasoning. I had justified running from the worktree by
arguing that the "run from the primary checkout" rule was about `.snakemake`
metadata and locks, which a separate `project_dir` satisfies. That was half the
rule. The environment is the other half — which `AGENTS.md`'s stated rationale
does not mention, but the rule was right and my reading was too narrow.

## Concurrency falsifier

`tests/test_wflow_log_attribution.py`, green against the post-migration run:
**12 member logs, 12 correctly attributed, 0 stray `log.txt`.**

Counting files is not the test and the module says so — twelve logs exist the
moment `path_log` is keyed per member, whatever is inside them. The
discriminator is content attribution: each log names its own `rlz_<r>_cst_<c>`
and no other. A third test guards the guard, since an **empty** log would pass
the attribution check vacuously.

**Outstanding:** the brief also requires showing the falsifier FAILS with
`path_log` unset. P1's observed tier is partial evidence — six members sharing
one log under the old layout — but the demonstration in the *flattened* layout
has not been run.

## Results delta

**No value changed.** Zero numeric and zero structure failures across 160
compared files. Every residual is a provenance string, a presence difference, or
work deferred to P3 by design — each allowlisted above with its reason.

## Validation

| Rung | Command | Result |
| --- | --- | --- |
| 1 Narrow | per-row scopes | green at each row |
| 2 Integration | `pixi run test-cli` | 12 passed, ×5 |
| 3 Phase gate | `pixi run test-fast` | **1216 passed**, 30 skipped, 42 deselected, 1 xfailed (37 s) |
| 4 Full run | three workflows, cold, dedicated `project_dir` | **all three green** |
| 5 Tree gate | `semantic_tree_diff` whole-tree | clean modulo the allowlist above |

`check_baseline` is **red from commit 1 until P3 re-records**, by design.
Reported, not fixed.

## Open decisions

1. **`data_catalog_climate_experiment.yml`** — the map routes it to
   `experiments/<id>/config/catalogs/`, and the tree still has it at the
   experiment root. It appears in no P2 checklist item. Implement in P2, or
   assign it?
2. **F3's map gap** — the bare `weather_generator/` directory needs a rule.
   Amending the map is an owner decision and `semantic_tree_diff.py` is outside
   P2's permitted scope.

Neither blocks the four landed rows; both block calling the tree gate clean
without an allowlist entry that says "known gap".
