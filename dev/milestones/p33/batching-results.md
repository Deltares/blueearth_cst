# P3-3 — batching results + GN-1..4 adjudication (commit 2)

> **Gate-2 material.** The measured before/after for the batching lever, the
> corrected cost decomposition, the value-identity evidence, and the §9 go/no-go
> adjudication. Spec: `performance-passes-design.md` §6.1 / §8 / §9 / §10.
> Companion: `performance-baseline.md` (commit 1 — **all its wall numbers are
> superseded**, see the supersession block at its head). Raw artifacts:
> `probes/_c2_batching/`, `probes/_batched_sweep_wall.txt`.
> **All walls are single-sample (n=1) on an otherwise-idle machine.**

## Headline

| | value |
|---|---|
| **before (B=1, per-process)** | **619.9 s** (58 jobs, exit 0) |
| **after (B=4, batched)** | **400.2 s** (49 jobs, exit 0) |
| **delta** | **−219.7 s = −35.4 %** |
| value identity | **byte-identical** — 102 files compared, 0 failed, tolerance **0** |
| go/no-go | **GN-1 ✅ GN-2 ✅ GN-3 ✅ GN-4 ✅ → batching stands; sysimage stays dormant** |

## Measurement conditions (the §5.6 contract, honored)

- Frozen triple: **`-c 3`, `--threads 4`**; **B is the only moved knob** (1 → 4).
- Identical invocation both runs, the commit-1 reproducibility anchor:
  `snakemake all -s Snakefile_climate_experiment --configfile
  config/workflows/snake_config_model_test.yml -c 3 --forceall`.
- Box: i5-1335U (10 physical / 12 logical), AC online, **`LoadPercentage` 1 % before
  the runs**, no sibling agent session active. Julia `+1.11.7` (juliaup), Snakemake
  9.6.2 with `profiles/default`.
- Timing: baseline 11:01:17→11:11:37, batched 11:14:31→11:21:11 (2026-07-25).
- **Gap between runs: ~3 min**, occupied by the light single-core dry-run
  demonstration. Stated plainly because it is a bias, not a neutral fact: the
  batched run went **second**, i.e. on the warmer box, which biases the comparison
  **against** batching. A −35.4 % win under an adverse ordering is a floor, not a
  ceiling.
- Job-count asymmetry (58 vs 49) is the construct itself: 12 `run_wflow` jobs
  collapse to 3 `run_wflow_batch_<b>` jobs. Everything else is identical.

**"B is the only moved knob" — measured, not asserted.** The two runs are minutes
apart on a machine whose cache state was changing (the era5 zarr was cold for the
baseline's 3.02 and warm for the batched one), so upstream parity is checked rather
than assumed. Summing every non-3.10 row of both runs' `wf3_benchmarks.md`:

| | non-3.10 rows | upstream total |
|---|---|---|
| baseline B=1 | 43 | 342.59 s |
| batched B=4 | 43 | 359.50 s |

Identical row counts (the same upstream work ran in both), and the batched run's
upstream was **+16.91 s / +4.9 % SLOWER**. The residual drift therefore works
**against** the lever: with identical upstream the batched sweep would have landed
near 383 s (−38.2 %), not 400.2 s. Combined with the adverse run ordering, −35.4 %
is a conservative figure on both counts. The directly-measured 3.10-stage numbers
(464 s → 228 s makespan) carry the claim independently of any upstream drift.

## Why the before-number had to be re-measured

The committed commit-1 baseline (2242.9 s) was contaminated by the concurrent
`stage_data` workstream and is superseded — full evidence in
`performance-baseline.md`'s supersession block. Using it would have reported a
**~5× win** for a lever that actually delivers **1.55×**. The corrected baseline
was measured on the same quiet box, minutes before the batched run.

**Input identity across the store rebuild was proven, not assumed.** The era5 store
was rebuilt between the two dates, so the 2026-07-24 output tree could only serve as
a reference if the rebuild was content-neutral. It is: a whole-tree
`semantic_tree_diff` of the pre-rebuild tree against the freshly re-measured
per-process tree is **CLEAN — 101 files, 0 failed** at tolerance 0 (the single
allowlisted EXTRA is `extract_historical.nc`, absent from the pre-rebuild snapshot
only because the failed 2026-07-24 23:35 run had deleted it).

## Decomposition, corrected (measured — supersedes §5.2)

The driver's per-cst `@elapsed` lines are the first direct measurement of the
warm-session discount the design could only carry as a design-time term:

| term | design-time (§5.2) | **measured 2026-07-25** | how |
|---|---|---|---|
| `F` per-process fixed (Julia start + `using Wflow`) | 135 s | **≈ 24 s** | batch wall − Σ member `@elapsed`: 228.14−202.7=25.4, 217.12−193.7=23.4, 220.65−197.6=23.0 |
| `S_cold` first run in a session | 208 s | **≈ 92 s** | driver lines: 90.1 / 90.4 / 95.0 s (one per batch) |
| `S_warm` runs 2..B in a session | 124 s (carried) | **≈ 35 s** | driver lines: 26.2–39.5 s (mean 34.9) |
| per-process job wall (`F + S_cold`) | 343 s | **115 s** | 3.10 benchmark rows 114.17–117.09; 24+92=116 ✓ |

**The decomposition closes**: `F + S_cold` = 116 s predicted vs 114–117 s measured.

**Where the win actually comes from — a correction worth stating.** It is **not**
Julia process startup. `F ≈ 24 s` is only ~21 % of a run, *less* than the design's
39 %. The win is **in-session warm-up of the Wflow run path**: 92 s → 35 s, a 0.38
warm/cold ratio versus the design's assumed 0.60. So batching amortizes more than
the design predicted, but through a different term than "startup/JIT per process"
suggests. Practical consequence: a PackageCompiler sysimage, which attacks `F`
only, can never capture most of this win (below).

## Predicted vs measured (the estimator, re-run at measured terms)

`dev/scripts/estimate_batch_makespan.py --table --k 12 --p 3 --f 24 --s-cold 92
--s-warm 35`:

```
# LPT makespan table  (K=12, p=3, F=24, S_cold=92, S_warm=35)
lever / B                             batches   makespan  vs today         Graham [lo, hi]
------------------------------------------------------------------------------------------
today (per-process, cold)        1+1+1+1+1+1+1+1+1+1+1+1        464        --              [464, 541]
sysimage (F->~2, always cold)    1+1+1+1+1+1+1+1+1+1+1+1        376      -19%              [376, 439]
batching B=2                      2+2+2+2+2+2        302      -35%              [302, 403]
batching B=3                          3+3+3+3        372      -20%              [248, 372]
batching B=4                            4+4+4        221      -52%              [221, 368]
batching B=6                              6+6        291      -37%              [291, 388]
```

| | predicted | measured | error |
|---|---|---|---|
| 3.10 stage makespan, B=4 | 221 s | 228.1 s (max of 228.14 / 217.12 / 220.65) | **+3.2 %** |
| full sweep, B=4 | ~377 s (221 + ~156 s upstream) | 400.2 s | **+6.2 %** |
| 3.10 stage makespan, B=1 | 464 s | ~460 s (4 waves × 115 s) | ~1 % |

- **B=4 is confirmed optimal** at measured terms, and it is what the config default
  `ceil(K / -c N)` picks unaided — the heuristic landed on the optimum here rather
  than needing hand-tuning.
- **The sysimage counterfactual got materially weaker.** At contaminated terms the
  table put sysimage at −39 % vs batching-B=4's −48 % (comparable). At measured
  terms it is **−19 % vs −52 %** — batching now dominates by ~2.7×, because the
  sysimage attacks `F ≈ 24 s` while the real cost is the per-session warm-up.
  This is independent evidence for keeping the sysimage dormant, beyond GN-1..4.

## GN-1 — throughput ✅ PASS

Measured **−35.4 %** against the **≥15 %** floor, both runs at the frozen triple.
Computed against the **measured** 619.9 s baseline, never against the estimator's
464 s model figure (the §5.5 denominator caution, honored).

## GN-2 — value identity ✅ PASS (the design's biggest unknown, now closed)

All four §8 gates green, **including the discriminating per-process-vs-batched
diff on identical inputs**:

| gate | command | result |
|---|---|---|
| 1. whole-tree semantic diff | `semantic_tree_diff.py --ref <per-process tree> --cur examples/test_local --no-path-map --tolerance 0` | **CLEAN — 102 files, 0 failed, 0 missing, 0 extra** |
| 2. manifested targets | `check_baseline.py check --workflow climate_experiment` | **OK — 3 targets match** |
| 3. P3-2b validators | `pytest tests/test_interchange_contracts.py -rs` | **53 passed, 3 skipped** (all three skips the pre-existing `temp()`-artifact class, split unchanged) |
| 4. suite + dry-runs | `pytest tests/` / `pytest tests/test_cli.py` | **397 passed, 6 skipped, 1 xfailed** / **4 passed** (three Snakefiles dry-run clean) |

**The untested assumption is now tested, positively.** Design §8 / OQ-3 flagged
**warm-session vs cold-process byte identity** as unproven, with commit-2 gate 1 as
its first evidence: runs 2..B of a batch reuse JIT-compiled method instances and
allocator/GC state a fresh process lacks, and the measured 57 s warm discount is
direct proof the execution path differs. It nonetheless produces **byte-identical
output at tolerance 0**. No batching-correlated diff exists, so the discriminating
rule never fires and the ADR-0001 immaterial branch is not invoked at all.

**Bonus — idempotence.** The post-GN-4 re-converged tree is also CLEAN (102 files,
0 failed) against the *pre-injection* batched tree: a failed-and-re-run batch
reproduces its outputs exactly.

## GN-3 — disk ceiling ✅ PASS (measured, at the cap)

Sampled every 2 s across the batched sweep (197 samples,
`probes/_c2_batching/gn3_disk_samples.csv`):

| quantity | value |
|---|---|
| peak forcing NCs resident | **12** (all of them), 119.67 MB |
| peak outstates NCs resident | **12** (all of them), 0.92 MB |
| **peak total `temp()` footprint** | **120.59 MB** |
| design cap `p × B × (forcing + state)` | 3 × 4 × (9.97 + 0.077) = **120.6 MB** |

**At the cap, not above it** — and the design *predicted exactly this*: at B=4 / p=3
there are only `ceil(12/4) = 3` batches for 3 workers, so all batches are resident
simultaneously and the cap degenerates to the whole sweep's temp footprint. That is
the honest reading: on this fixture the ceiling is not merely respected, it is
saturated by construction. At 120 MB it is harmless; on a production
`RLZ_NUM × ST_NUM` sweep this is the ceiling that binds first and forces `B` down,
exactly as §6.1 says.

**The default `B` is now clamped, because the fixture hid a scale hazard.** GN-3
passes here, but it passes on a sweep where the cap and the whole-sweep footprint
coincide *and* the footprint is 120 MB. The landed default was `ceil(K / -c N)`,
which implements only §6.1's **parallelism** ceiling — so `B`, and therefore peak
disk (`p × B × …`), scale **up** with sweep size, the opposite of what §6.1 asks
when it calls the disk ceiling "the BINDING constraint" that forces `B` small on
large runs. The fixture cannot surface this: `K=12` at `-c 3` gives `B=4` either
way. An overridable `batch_size_max` (default 8) now bounds it; a genuinely
disk-aware cap needs a headroom config key plus a per-run size estimate that is
not available at parse time (the forcing NCs are `temp()` and do not exist yet),
so it is recorded in `dev/followups.md` § Post-P3-3. **Every measurement in this
note is unaffected** — `min(ceil(12/3), 8) = 4`, verified — and the clamp only
binds from `K > 24` at `-c 3` (verified: `K=60` yields 8 batch rules, not 3).

**Outstates reclamation verified, not assumed** (the §6.1 verify-in-commit-2 item):
the sampler shows both `temp()` classes dropping in **B-sized groups** — 12 → 8 → 4
→ 0 at 11:20:39 / 11:20:49 / 11:21:03, ending at **0 MB**. Nothing leaks; deletion
moved to batch granularity as designed.

## GN-4 — failure injection ✅ PASS (with a mechanism deviation, stated)

**Injection method.** A temporary guard in `run_wflow_batch.jl` raised for one
member (`wflow_sbm_rlz_1_cst_1.toml`, a `batch_0` member) when
`CST_GN4_FAIL` matched it. The guard was **removed before commit** (driver restored
from a pre-injection copy and verified clean). A driver-level raise was chosen
because the obvious data-level injections are unreachable: the per-cst TOML and its
forcing NC are **both outputs of rule 3.09**, so any edit that would make Wflow fail
is regenerated by the very re-run it triggers.

**⚠ Deviation from the designed mechanism — adjudicated on observed consequences,
not on the design's wording.** Design §6.1/§9 describe the driver as *not*
rethrowing, so the batch exits 0 and the rule trips **`MissingOutputException`**.
The landed driver instead **exits nonzero** (`exit(exitcode)`), so Snakemake fails
the job directly on the exit status. GN-4 is therefore graded on the realized blast
radius. The nonzero exit is retained as the **safer** behavior: it fails the job on
the failure itself rather than on the downstream symptom of a missing file, and it
cannot be fooled by a partially-written CSV that would satisfy an output check.

| §9 criterion | observed | verdict |
|---|---|---|
| batch job fails | `Error in rule run_wflow_batch_0`, run exit 1 | ✅ (via nonzero exit, **not** `MissingOutputException`) |
| B−1 sibling CSVs deleted | Snakemake removed exactly `output_rlz_1_cst_{2,3,4}.csv` + `outstates_rlz_1_cst_{2,3,4}.nc` ("since they might be corrupted"). The failed member's CSV was never written. | ✅ blast radius = the batch, **exactly** `B` csts |
| no damage outside the batch | all 8 CSVs of `batch_1`/`batch_2` present and untouched | ✅ |
| 3.11 blocked only until re-run | `export_wflow_results` did **not** execute; `Qstats.csv` mtime stayed 11:21:08 (the prior good run) | ✅ |
| per-cst driver lines for every member incl. the failed one | `BATCH-RUN FAIL wflow_sbm_rlz_1_cst_1.toml …` + `BATCH-RUN OK` for all 11 others | ✅ compute isolation preserved |
| clean re-convergence passing GN-2 | fault removed → **exit 0 in 4 jobs** (only the failed batch + its downstream, not the whole sweep), then gate 1 **CLEAN** | ✅ |

> The driver rows quoted above are the ones observed at the time. They are
> spelled differently now — `HH:MM:SS - wflow - [k/N] rlz_<i>_st_<j>  <s> s`,
> with `FAILED` in place of `BATCH-RUN FAIL` — since the rows moved into the
> toolbox's house log format and gained a batch position. The observation
> stands; only its spelling moved. Emit site:
> `blueearth_cst/experiment/run_wflow_batch.jl`.

Realized blast radius is **equal to — never wider than — the documented C5 cost**.
C5 remains **DEGRADED by design** (blast radius `B`), documented and now measured
rather than asserted.

**Incidental finding (not a criterion).** Forcing one batch cascades wider than the
batch: `--forcerun run_wflow_batch_0` re-ran all three batches, because the batch's
`temp()` forcing inputs were already reclaimed, so 3.09 → 3.07 → 3.06 had to
regenerate, and 3.06 emits **all** realizations, invalidating every batch. Worth
knowing before anyone plans a surgical re-run of one batch. **Attribution is
REASONED, NOT MEASURED:** the same 3.09 → 3.07 → 3.06 chain and the same all-
realizations 3.06 output exist at B=1, so this should be a pre-existing `temp()`
cascade rather than a batching effect — but that was not tested here, and it is
flagged as reasoning so it does not read with the same authority as the measured
rows above.

## Tooling defects found and fixed (in `dev/scripts/estimate_batch_makespan.py`)

Both were latent in the commit-1 file and both had to be fixed to produce the
corrected table above, so they land with this commit:

1. **`--f` was silently ignored in `--table` mode.** `print_table` hardcoded
   `F_DEFAULT`, so `--table --f 24` reported the design-time fixed cost while
   *appearing* to honor the override — which would have published a wrong corrected
   table. Fixed by threading `f` through `print_table`; the sysimage row keeps its
   own `F→2` override (its whole point). The header now prints `F=` so the terms are
   visible in any captured table.
2. **`--help` crashed with `UnicodeEncodeError`** on the default Windows cp1252
   console: the module docstring carried `⌊⌋∤Σ·` glyphs. ASCII-ified.

Both covered by new regression tests (`test_table_honors_f`,
`test_help_text_is_console_encodable`); the file's suite is **28 passed** (26 design
assertions unchanged + 2 new).

## Adjudication (design §9 decision rule)

**All four criteria pass → batching stands; the PackageCompiler sysimage stays
dormant, no dependency ask is triggered.** The corrected estimator terms
independently weaken the sysimage case (−19 % vs batching's −52 %), so the dormancy
is now supported by measurement rather than only by the G1 gate ruling.

## Reproduction

```powershell
# value identity (the discriminating diff): needs a per-process reference tree
pixi run python dev/scripts/semantic_tree_diff.py --ref <per-process tree> --cur examples/test_local --no-path-map --tolerance 0
pixi run python dev/scripts/check_baseline.py check --workflow climate_experiment
pixi run pytest tests/test_interchange_contracts.py -rs
pixi run pytest tests/ tests/test_cli.py

# before/after (B via workflows.climate_experiment.batch_size; default ceil(K / -c N))
Measure-Command { pixi run snakemake all -s Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml -c 3 --forceall }

# the estimator at measured terms
pixi run python dev/scripts/estimate_batch_makespan.py --table --k 12 --p 3 --f 24 --s-cold 92 --s-warm 35
```

## Artifacts (`probes/_c2_batching/`, UNTRACKED working record)

These stay untracked, following the commit-1 / p31 convention for `_`-prefixed
probe output (`dev/milestones/p31/_semantic_diff.out` precedent): the durable claims live in
this note, the raw captures are prunable and local.


- `dryrun_demonstration.txt` — the arbitration-mandated demonstration, recorded
  **before** the sweep: all 12 per-cst targets dry-run individually, each resolving
  to exactly one `run_wflow_batch_<b>` rule (partition 4+4+4), plus the whole-DAG
  49-job `--forceall` table.
- `perprocess_sweep_B1.log` / `batched_sweep_B4.log` — the two timed sweeps.
- `gn3_disk_samples.csv` — the 2 s `temp()`-footprint sampler.
- `gn4_injection.log` / `gn4_reconverge.log` — failure injection and clean re-run.
- Progress-bar lines are stripped from the landed logs; all other content verbatim.

## Summary for Gate 2 (milestone sign-off — floor-free)

- **619.9 s → 400.2 s, −35.4 %**, at a frozen `(-c 3, --threads 4)` budget with `B`
  the only moved knob, batched run measured under an adverse (warmer, second)
  ordering.
- **No output value changed**: 102 files byte-identical at tolerance 0, including
  the per-cst `output_rlz_*_cst_*.csv`; manifested targets match; P3-2b validators
  green; full suite green. The design's flagged warm-vs-cold identity risk is
  resolved **positively** with first-time evidence.
- GN-1..4 all pass; C5 degradation realized exactly as documented, never wider.
- The commit-1 baseline was found contaminated and is superseded in place, with the
  contamination proven by byte-identical output at 6.4× the `cpu_time`.
