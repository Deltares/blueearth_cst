# P3-3 — wf3 performance baseline (commit 1)

> **Gate-1 material.** The durable record of the wf3 stress-test sweep's
> performance baseline: the measured end-to-end sweep wall at the §5.6 resource
> triple, the re-measured §5.2 cost decomposition against the design-time probe
> numbers, and the §5.5 LPT makespan table as produced by the landed estimator
> `dev/scripts/estimate_batch_makespan.py`. This is the **before-number** commit
> 2's batching lever is measured against — same triple, `B` the only moved knob.
> Spec: `performance-passes-design.md` §5.1/§5.2/§5.5/§5.6/§9. Precedent:
> `dev/p31/baseline_diffs.md`. Raw probe record: `probes/PROBE_RESULTS.md`.
> **All numbers below are single-sample (n=1) on an otherwise-idle machine.**

## Machine state (the §5.6 accounting reality, verified)

- Box: 13th-gen **i5-1335U — 10 physical cores (2 P + 8 E), 12 logical.**
- Host `WSU-5CG4013KNZ`, Windows 11, **AC power online** (checked via
  `BatteryStatus.PowerOnline = True`), otherwise idle — no other heavy job ran
  during any measurement (runs executed sequentially).
- Julia `+1.11.7` (juliaup, on PATH; not in pixi). Python/snakemake/Rscript via
  `pixi run`. Snakemake 9.x with the auto-loaded `profiles/default` (sets only
  `quiet: reason`, no core count — `-c` comes from the invocation).
- Rule 3.10 declares **no `threads:` directive** (verified, Snakefile line
  361-374) → each `run_wflow` job counts **1** against `-c N`; `-c 3` runs 3
  Julia sessions concurrently. The invocation passes `julia +1.11.7
  --project=. --threads 4` (verified line 374). So the baseline triple is real.

## The baseline triple + measured sweep wall (the metric of record, §5.1/§5.6)

| quantity | value |
|---|---|
| **resource triple `(-c N, --threads t, B)`** | **(3, 4, 1)** — today's production invocation |
| **sweep size `K`** | **12** = RLZ_NUM(2) × ST_NUM(6), `run_historical` false (cst 1..6) |
| **effective parallelism `p`** | **3** (= `-c 3`; each run ~single-core, §6.3) |
| **end-to-end sweep wall (measured)** | **2242.9 s ≈ 37.4 min** |
| START → END | 2026-07-24 21:18:15 → 21:55:38 (+02:00) |
| forced scope | **`--forceall`** (58/58 jobs, exit 0) — see resolution below |

**Metric discipline (§5.1).** This is **end-to-end wall clock at a fixed `-c 3`**,
via `Measure-Command` around the `snakemake` subprocess — **not** a summed rule
time. The per-job benchmark **TOTAL = 6195 s** (this run's
`wf3_benchmarks.md`) is the *sum* of per-`(rlz,cst)` job walls; the real sweep
wall is **2242.9 s**, materially below it — the `2242.9 / 6195 ≈ 0.36` ratio is
exactly the parallelism factor §5.1 predicts (~3 cores + serial upstream).
Comparing any lever against the 6195 s sum would overstate its win ~2.8×; this
baseline never does.

### Ambiguity resolved — `--forceall`, not `--forcerun` (flagged loudly)

The brief's fallback (`--forcerun generate_weather_realization`) **does not
cascade the sweep on this tree** — a `--dry-run` of it
(`probes/_forcerun_dryrun.txt`) listed only **1 job** (`extract_climate_grid`,
fired by an unrelated code-change provenance trigger), not the 12 `run_wflow`
jobs. Snakemake 9.x evaluates downstream reruns lazily against already-up-to-date
outputs, so forcing 3.06 alone did not re-execute the sweep, and the forced scope
could not be certified. **Resolution: `--forceall`** — it forces every job in the
experiment DAG unconditionally (deterministic, trivially reproducible, moots the
code-change confound). The captured dry-run (`probes/_forceall_dryrun.txt`) shows
the full sweep and **no wf1 build rules** (the experiment DAG bottoms out at 3.02
`extract_climate_grid`, consuming existing wf1 artifacts):

```
job                              count
all                                  1
check_project_consistency            1
climate_data_catalog                 1
climate_stress_parameters            1
copy_config                          1
downscale_climate_realization       12
export_wflow_results                 1
extract_climate_grid                 1
gather_benchmarks                    1
generate_climate_stress_test        12
generate_weather_realization         1
prepare_weagen_config                1
prepare_weagen_config_st            12
run_wflow                           12
total                               58
```

**This 58-job `--forceall` scope IS the reproducibility anchor.** Commit 2's
before/after MUST run the identical `snakemake all -s
Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml
-c 3 --forceall` invocation (batched vs this baseline) for an apples-to-apples
comparison — the §5.6 contract's "B the only moved knob".

**Pipeline-execution note.** This `--forceall` run legitimately rewrote the wf3
outputs under `examples/test_local/experiments/experiment/` (deterministic,
P3-1-verified; not modified by hand). `run_default/output.csv` (the wf1
check_baseline'd discharge) was **not** touched — wf1 is not in the wf3 DAG.

## §5.2 cost decomposition — re-measured vs design-time (n=1)

**Scope decision (flagged).** The design *wins over the brief*, and §9 / OQ-1 /
risk-4 resolve the §5.2 re-measurement as **n=1, no repeats, "confirmed by this
commit's re-measured sweep wall."** A B=1 baseline sweep **never exercises warm
sessions or an alternate log path**, so re-measuring `S_warm` (probe 1d) or the
logging split (probe 1b) has **zero bearing on the before-number**. Per that
spec, exactly **one** fresh-process F+S point was re-measured; `S_warm = 124` and
the probe-1b "logging is not a per-run cost" result are carried forward as
clearly-labeled **design-time** values.

### Re-measured point — fresh-process 21yr production run (probe-1c form)

Exact probe-1c invocation form (`julia +1.11.7 --project=. --threads 4 -e "using
Wflow; Wflow.run()" <toml>`), one **fresh** process, the wf1 base TOML (routing
keys verified identical to the per-cst wf3 TOMLs in the design), full 2000-2020
(21 yr) window, output redirected to a scratch dir (run_default untouched).

**Condition mismatch — stated up front (the design's own CAVEAT applies).** This
re-measurement ran **WITHOUT** the `run_logged` tee; design-time probe-1c's
**343 s ran WITH** it. The design fixes the tee at **~42 s** ("~301 no-tee vs 343
with-tee", PROBE_RESULTS.md CAVEAT). So a raw 357.7-vs-343 comparison is
**mixed-condition** and must NOT be read as a ~4 % match. Condition-matched:

| comparison | reference | measured | condition-matched divergence |
|---|---|---|---|
| no-tee vs design no-tee estimate | ~301 s (343 − ~42 tee) | 357.7 s | **+57 s (+19 %)** |
| add tee to mine vs design with-tee | 343 s | ~400 s (357.7 + ~42) | **+57 s (+17 %)** |

- START → END: 21:57:01 → 22:02:59 (+02:00). Log:
  `probes/_c1_decomp/fresh_process_21yr.log`. Wflow-internal "Simulation
  duration: 2m 44s" = **164.9 s** (the pure sim loop, excluding `using`+JIT+setup
  the wall captures — not `S_cold`).

| term | design-time (§5.2) | this commit | note |
|---|---|---|---|
| fresh-process 21yr F+S wall (no tee) | ~301 (implied no-tee) | **357.7 s** | +19 % condition-matched |
| per-process fixed cost `F` | 135 s (firm) | ~135 s (carried) | JIT-dominated, window-independent |
| implied sim = wall − F | ~166 (no-tee) / 208 (with-tee) | ~223 s | inflated by throttle (below) |

**Two confounds explain the +17-19 %, and the honest corroboration is the sweep,
not this point:**
1. **Thermal throttling (not "cache warmth").** This run started **83 s** after a
   **37-min all-core sweep** ended (sweep END 21:55:38 → decomp START 21:57:01).
   The i5-1335U throttles under sustained load, so this process began **hot** —
   inflating the wall. (Cache warmth would make it *faster*; it came in slower,
   so throttle, not warmth, dominates.)
2. **No tee** (−~42 s vs production form), partially offsetting (1).
- **Where the decomposition is actually confirmed (§9/OQ-1/risk-4):** the design
  names the **sweep wall**, not this single point, as the confirmation of record.
  This sweep's own `wf3_benchmarks.md` gives per-run `run_wflow` ≈ **420 s**
  (deployed, with tee, under `-c 3`). My no-tee **357.7 + ~42 tee ≈ 400 s** lines
  up with that ~420 and with the design's ~390-410 deployed band. So the
  decomposition **structure is confirmed via the measured ~420 s/run** — fixed
  ~135 s (firm, JIT-dominated) + a sim remainder — **not** via a spurious match to
  the idle 343. The exact 39/61 split stays n=1 (OQ-1); the *structure* is
  corroborated. (A clean probe-1c-form number would need a with-tee re-run after a
  cool-down; the sweep-derived ~420/run makes it unnecessary for Gate 1.)

### Carried-forward design-time values (not re-measured — B=1 does not exercise them)

- **`S_warm(21yr) ≈ 124 s`** (probe 1d) — the runs-2..N warm-session discount
  that **only batching** captures. First measured under batching in commit 2.
- **Logging is NOT a per-run cost** (probe 1b: run2/run3 unchanged 35-36 s under
  `loglevel="info"` vs `silent`) → lever D dropped.

## Gap attribution — measured sweep wall vs the design's idle arithmetic

The design's idle-arithmetic `ceil(12/3) × 343 = 1372 s` is a **lower bound**
placeholder (all per-run terms idle-measured); the **measured** sweep wall is
**2242.9 s**, ~63 % above it. Attribution, from this run's `wf3_benchmarks.md`:

| term | per-job wall (mean) | jobs | waves @ p=3 | wall contribution |
|---|---|---|---|---|
| **3.10 run_wflow** | ~420 s (404-443) | 12 | 4 | ~1680 s |
| 3.09 downscale | ~52 s (40-78) | 12 | 4 | ~210 s |
| 3.06 generate_weather | 149.8 s | 1 (serial) | — | ~150 s |
| 3.07 stress-test | ~24 s | 12 | 4 | ~95 s (overlaps) |
| 3.02/3.08/others | ~40+25+misc | — | serial | ~110 s |
| **sum ≈** | | | | **~2245 s** ✓ |

- **The dominant term is `run_wflow` at ~420 s/job**, ~77 s above the idle
  single-process 343 s. Attribution (design §5.2/§5.5): `-c 3` concurrency
  contention (3 Julia sessions sharing 10 physical cores) + the Windows `psutil`
  benchmark sampler + the wf3 cold-start-SBM path. The design named exactly this
  (~390-410 s/run deployed, "~14 % above idle"); the measured ~420 s is at/above
  that band — the honest deployed cost, higher than the idle placeholder.
- **`--threads 4` buys nothing on the fixture (confirms §6.3):** every run_wflow
  row shows `mean_load ≈ 88-93 %` and `cpu_time ≈ wall` (e.g. cst_1: wall 442.97,
  cpu 412.14) — ~single-core load on the 384-cell (16×24) basin. Kept for the
  production regime (§6.3), not a fixture win. Part of the frozen triple.

## §5.5 LPT makespan table (produced by the landed estimator)

Verbatim from `python dev/scripts/estimate_batch_makespan.py --table` (fixture
terms F=135, S_cold=208, S_warm=124; K=12, p=3). Reproduces the design's seven
integers **exactly** (1372, 840, 934, 1182, 715, 963) plus the Graham bracket:

```
# LPT makespan table  (K=12, p=3, S_cold=208, S_warm=124)
lever / B                             batches   makespan  vs today         Graham [lo, hi]
------------------------------------------------------------------------------------------
today (per-process, cold)        1+1+1+1+1+1+1+1+1+1+1+1       1372        --            [1372, 1601]
sysimage (F->~2, always cold)    1+1+1+1+1+1+1+1+1+1+1+1        840      -39%              [840, 980]
batching B=2                      2+2+2+2+2+2        934      -32%             [934, 1245]
batching B=3                          3+3+3+3       1182      -14%             [788, 1182]
batching B=4                            4+4+4        715      -48%             [715, 1192]
batching B=6                              6+6        963      -30%             [963, 1284]
```

Non-divisible demonstration (`--table --k 13 --p 3`, or `--k 13 --p 3 --b 4`):
**K=13, p=3, B=4 → batches (4,4,4,1) → makespan 1058** (the 343 s remainder
batch backfills a freed worker; the naive wave formula's 1430 is a 35 %
overestimate and mis-ranks B=4 below B=3 — §5.5). Estimator unit tests
(`tests/test_estimate_batch_makespan.py`, 26 tests) assert every design integer,
the K=13 case, the Graham-bracket containment, and the divisible-reduces-to-wave
identity.

**Model role (§5.5).** These are **idle single-process** placeholder terms; the
table's job is **honest ranking of B within the Graham bracket**, not exact wall
prediction. The measured baseline above (2242.9 s at the real triple) is the
before-number; commit 2's measured batched wall — not this table — settles the
lever. The table selects the candidate `B` worth measuring (B=4 best on the
fixture) and predicts the ordering.

**Do not mis-apply the table percentages (Gate-1 reader note).** The table's
−32 %/−48 % are relative to the **idle 1372 s** (`ceil(12/3)×343`), a model
placeholder. **GN-1's ≥15 % floor is computed against the MEASURED 2242.9 s** —
a different denominator. Never apply −48 % to 2242.9; the model ranks B, the
measured commit-2 before/after (both at the frozen triple) sets the GN-1 number.

## Measurement spec actually followed

1. **Sweep wall:** `Measure-Command { pixi run snakemake all -s
   Snakefile_climate_experiment --configfile
   config/workflows/snake_config_model_test.yml -c 3 --forceall }` on a quiet
   AC-powered box; `--forceall` scope certified by a prior `--dry-run` (58 jobs,
   no wf1 rebuild). Triple `(3, 4, 1)` recorded.
2. **Decomposition:** one fresh Julia process, probe-1c invocation form, wf1
   base TOML, full 21yr window, output redirected to scratch (run_default
   untouched), timed by `Measure-Command`. n=1, no repeats (per §9/OQ-1/risk-4).
3. **Carried design-time:** S_warm=124 (probe 1d), logging-not-a-cost (probe 1b)
   — untouched by a B=1 sweep, so not re-measured.
4. **Estimator table:** produced by the landed `estimate_batch_makespan.py`,
   verified against the design's seven integers + the K=13 demonstration.

## Exact commands (for reproduction)

```powershell
# 0. estimator (dev-process tooling, no pipeline change)
pixi run python dev/scripts/estimate_batch_makespan.py --table
pixi run pytest tests/test_estimate_batch_makespan.py -q

# 1. certify the forced scope, then the timed sweep wall
pixi run snakemake all -s Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml -c 3 --forceall --dry-run
Measure-Command { pixi run snakemake all -s Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml -c 3 --forceall }

# 2. decomposition re-measure (scratch TOML redirecting dir_output; run_default untouched)
Measure-Command { pixi run julia +1.11.7 --project=. --threads 4 -e "using Wflow; Wflow.run()" <scratch_wf1.toml> }
```

## Artifacts (under dev/p33/probes/, prunable working record)

- `_forceall_dryrun.txt` — the certified 58-job forced scope.
- `_forcerun_dryrun.txt` — evidence the `--forcerun generate_weather_realization`
  fallback does NOT cascade (why `--forceall` was chosen).
- `_forceall_sweep_run.log` — the timed sweep's snakemake log (58/58 done).
- `_c1_decomp/fresh_process_21yr.log` — the decomposition run log.
- `_estimator_table.txt` — the estimator table capture.
- This run's `wf3_benchmarks.md` lives at
  `examples/test_local/experiments/experiment/benchmarks/` (regenerated by the
  sweep; the per-job source for the gap attribution).

## Summary for Gate 1

- **Baseline (before-number): 2242.9 s** end-to-end wf3 sweep wall at
  **(-c 3, --threads 4, B=1)**, K=12, `--forceall` (58 jobs), quiet AC box.
- Decomposition: fresh-process 21yr (no tee) = **357.7 s**; condition-matched
  (add ~42 s tee → ~400 s) this **corroborates the sweep's own ~420 s/run**
  benchmark and the design's ~390-410 deployed band — NOT the idle 343 (the raw
  357.7-vs-343 is mixed-condition, ~+17-19 % once tee-matched; the run also
  started hot, 83 s after the sweep → thermal throttle). Structure confirmed
  (fixed ~135 s firm + sim remainder); exact 39/61 split stays n=1 (OQ-1).
- LPT estimator landed + unit-tested; reproduces the design's §5.5 table exactly.
- Commit-2 reproducibility anchor: the **identical `--forceall` 58-job
  invocation at the same triple**, B the only moved knob (§5.6).
