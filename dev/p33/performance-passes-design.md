# P3-3 — Performance passes: Design (ACCEPTED)

> **Status: ACCEPTED** — approved at human gate G2 (2026-07-24) under
> arbitration authority. The accepted revision (v4) closed the loop at the
> external round cap: internal lens panel (risk / architecture / repo-fit;
> 1 blocking / 7 major / 10 minor, all accepted in v2 — incl. the
> batching-first restructure per the G1 gate ruling that deferred the
> PackageCompiler sysimage to commit-2 evidence) → external GPT round 1
> (`revise`: remainder-aware makespan model, CPU resource contract,
> executable batching go/no-go; resolved in v3, Fable-escalated) → external
> GPT round 2 (`reject`: ext2-001 blocking — the v3 callable-`output:`
> construct is not expressible in Snakemake, probe-confirmed) → round-cap
> **user arbitration** (accepted, fix required) → stage-6a revision
> confined to ext2-001 (probe-verified loop-generated anonymous rules;
> driver scope-check clean). Ledger 22/22 accepted, none rejected or
> deferred; per-finding disposition in the consolidated review record
> (`performance-passes-design-review-record.md`); what changed per revision
> in §13. Probe evidence landed beside this doc under `probes/`.
> **Milestone:** P3-3 (wf3 stress-test sweep throughput, value-identical).
> **Genre:** decision-record (milestone design). **Commit prefix:** `p33:`.
> **Author role:** cst-architect. **Run:** `p33-performance`.
> **Scope authority:** `dev/p33/performance-passes-intake.md` (its three
> "Confirmed scoping decisions" are fixed anchors, not reopened here).
> Structure precedent: `dev/p32b/interchange-contracts-design.md` (ACCEPTED
> 2026-07-24). This doc is self-contained: a reviewer needs only this file plus
> the cited paths.

---

## 1. Problem statement

Production wf3 stress-test sweeps scale linearly in `RLZ_NUM × ST_NUM` Wflow
invocations; each `run_wflow` job pays full Julia process startup + package load
+ first-call JIT, then runs the simulation effectively single-core. P3-3 measures
that cost structure on the fixture, records a baseline + a production scaling
model, and lands the value-identical lever(s) that cut sweep wall time — without
touching outputs, entry points, or the wrapper contract.

## 2. Goals / Non-goals

### Goals
- G1 — recorded baseline + cost decomposition (startup vs `using Wflow` vs
  run-JIT vs simulation) + production scaling model.
- G2 — value-identical lever(s) landed, before/after measured on the fixture.
- G3 — strict value-identity proof (semantic diff clean, manifested targets
  unchanged, P3-2b validators green, suite + dry-runs green).
- G4 — honest scaling statement (fixture vs production).

### Non-goals
- File-format redesign (parked, I/O non-dominant).
- wf1/wf2 optimization; memory-headroom work; GUI/platform-surface change.
- Upstream re-engineering (Wflow.jl / hydromt / weathergenr internals).
- New dependencies without prior user approval (sysimage is approval-gated).

## 3. Constraints (standing; restated)
- CST automation scope — amortize OUR invocation overhead only.
- Value-identity discipline (R3/R5 style).
- Platform surface (three entry points, `rule all` targets, wrapper contract,
  config `workflows:` sections) unchanged.
- New-dependency approval gate.
- Naming per `dev/conventions/naming.md`; commit prefix `p33:`; tag at close.
- Fixture is the era5 test basin; production claims are a model, not measured.

## 4. Decision criteria
- C1 — measure the right term (wall clock at fixed cores, not summed rule time).
- C2 — value-identity is absolute.
- C3 — CST automation scope respected.
- C4 — anchors held (DAG may change; outputs, entry points, wrapper may not;
  HM-4/HM-5 byte-compatible; P3-2b relational validators stay green).
- C5 — failure isolation preserved (one bad cst must not corrupt/skip others).
- C6 — honest coverage / scaling (state what the fixture does and does not
  predict).

## 5. Measured cost decomposition (the probe results)

All probe artifacts (scratch TOMLs, drivers, raw output) live under
`dev/p33/probes/ (raw record: PROBE_RESULTS.md)`; the prunable run dir
is the record of record until commit 1 lands the durable baseline note.

### 5.1 The metric: wall clock at fixed cores, NOT summed rule time

The benchmark **TOTAL 5,668 s** in `wf3_benchmarks.md` is the **sum** of per-JOB
wall times, **not end-to-end wall clock.** `merge_benchmarks` derives the `rule`
column from each part file's relative path (`merge_benchmarks.py:66`), so
`run_wflow` already emits **one row per `(rlz,cst)` job** — 12 rows, not one
aggregated per-rule row — and the TOTAL sums over those per-job rows. wf3 runs at
`-c N` set by the **invocation** (`run_snake_test.cmd` / `run_workflows.py`
default is `-c 3`; the auto-loaded profile sets only `quiet: reason`, not a core
count), so the 12 `run_wflow` jobs already run ~3-at-a-time. Real sweep wall today
≈ `ceil(12 / eff_parallel) × per_run`, materially below the 5,668 s sum.

**Consequence pinned for the whole design:** every baseline and every
before/after number is **end-to-end wall clock at a fixed, stated `-c N`** —
never a summed rule time. Commit 1 records the real sweep wall at a known core
count (measured, not reconstructed from the sum). Comparing any lever against
the 5,668 s sum would overstate its win by the parallelism factor; the design
never does this.

### 5.2 Probe 1 — the startup/JIT/simulation split (DECISIVE, measured)

**Method.** `julia +1.11.7 --project=. --threads 4` invocations from the repo
root on Windows (the production invocation minus the `run_logged` tee). The
`temp()` wf3 forcing NCs are deleted on the completed fixture, so the pure-sim
term uses the **persisted wf1 forcing** (`inmaps_historical.nc`, same 16×24 grid,
same `precip`/`pet`/`temp` vars — a legitimate proxy because run-JIT is
forcing-content-independent), windowed to 6 years for probe budget and
run in one Julia session (`Wflow.run(t1); Wflow.run(t2); Wflow.run(t3)`).

| measurement | value | what it is |
|---|---|---|
| bare start `-e "1"` | **0.9 s** | process + project-env resolution |
| `using Wflow` (`--threads 4`) | **16.4 s** | package load + precompile-check + load-JIT |
| run 1 (JIT + 6yr sim) | **154.2 s** | first `Wflow.run` in the session |
| run 2 (6yr sim) | **35.2 s** | second run — JIT already paid |
| run 3 (6yr sim) | **37.5 s** | third run — confirms run2≈run3, no inter-run leak |

Derived:
- **First-call run-JIT ≈ 154.2 − 36.4 ≈ 118 s** — the dominant per-invocation
  fixed cost, ~7× `using Wflow` and ~130× bare start. This is the compilation of
  the full `Model(config)` → `run!` pipeline on first dispatch. It is
  **window-independent** (compilation, not simulation), so this term is firm.
- **Per-process fixed cost ≈ 0.9 + 16.4 + 118 ≈ 135 s (measured, firm).** Every
  fresh `run_wflow` process pays this once; batching pays it once per
  **session**, not once per run; a sysimage bakes most of it away.

**Probe 1c — the decisive ground-truth per-run (fresh process, 21yr, no
extrapolation).** The in-session pure-sim term (36 s/6yr) does NOT scale linearly
to the real per-run cost, so the reconstruction is replaced by a **direct
measurement**: one **fresh** Julia process, the exact production invocation form
(`run_logged` tee → `julia +1.11.7 --project=. --threads 4 -e "using Wflow;
Wflow.run()" <toml>`), the wf1 config (routing keys **verified identical** to the
per-cst wf3 TOMLs: `river_kinematic_wave__time_step=900`,
`land=3600`, `kinematic_wave__adaptive_time_step_flag=false`,
`timestepsecs=86400`), full 2000–2020 (21 yr) window:

> **Fresh-process 21yr production run = 343 s wall (measured).**

This is the ground truth — no proxy scaling. Decomposition against it:
- Fixed (measured, firm) = **135 s**.
- **Implied sim(21yr) = 343 − 135 ≈ 208 s** (the in-session 6yr→21yr linear
  extrapolation *underestimated* real sim by ~80 s — in-session runs 2/3 reuse
  warm allocations a fresh process lacks, and Wflow sim does not scale perfectly
  linearly in window length; the fresh-process number moots both).
- **Fixed fraction on the fixture = 135 / 343 ≈ 39 %.**
- Residual vs the benchmark's ~390–410 s per-run ≈ **~50 s**, attributable to the
  Windows benchmark `psutil` sampler overhead + the wf3 cold-start-SBM path — a
  small, honest gap, not the 130 s the proxy reconstruction implied.

**Probe 1d — the warm/cold sim split (batching's hidden advantage).** Batched
runs 2..N share a warm session; only run 1 is cold. Two **full-21yr** runs in one
session (in-session, no `run_logged` tee): run 1 = **283.6 s** (cold+JIT+sim),
run 2 = **124.0 s** (warm sim only). So:
- **cold sim(21yr) ≈ 208 s** (from the 343 s fresh-process, §probe 1c).
- **warm sim(21yr) ≈ 124 s** (measured) — an **84 s/run warm-cache discount**
  that **only batching captures** (runs 2..N of a session). Sysimage always
  launches **fresh cold** processes, so it pays cold sim every run and
  **structurally cannot** get this discount. This single fact inverts the naive
  arithmetic (§5.5, §6.5).

**Probe 1b — logging is NOT a per-run cost (rules out lever D).** The identical
3-run in-session probe re-run **without** `silent=true` (production
`loglevel="info"` from the TOML): run1=178.8, run2=**35.0**, run3=**36.0** s.
run2/run3 are essentially **unchanged** vs the silent variant (35.2, 37.5) — the
pure-sim term is log-level-independent. So **per-run logging I/O is not a cost**
on this basin, and **lever D (loglevel) is DROPPED** (§6.4): a `loglevel` change
would save nothing on the sweep wall. The ~25 s run-1 loud-vs-silent difference
is a one-time logging-setup cost folded into JIT, not a per-run tax.

**Net decomposition (all measured, fixture, 21yr, `-c` unset for the single
run):** fixed **135 s** (39 %) + sim **208 s** (61 %) = **343 s** per fresh
production run. The fixed term is what every fixed-cost lever attacks; the sim
term is untouchable except by threads on a threadable basin (§6.3). Commit 1
re-measures the **sweep wall at a fixed `-c N`** directly (not a single-run
number) to seat the before-baseline.

### 5.3 Probe 2 — downscale internals (~10 % of wall; note, don't fix)

Read of `blueearth_cst/experiment/downscale_climate_forcing.py`: rule 3.09
instantiates `WflowSbmModel(root=model_root, mode="r+", data_libs=…)` **per
cst** (line 43) — re-reading the full `staticmaps.nc` (44 data_vars) and
re-running `setup_precip_forcing` + `setup_temp_pet_forcing` regridding on every
`(rlz,cst)`. The staticmaps read and model instantiation are **invariant across
cst** (only the forcing NC changes), so a per-cst rework candidate exists: hoist
the model load out of the per-cst loop. **Recorded, not fixed:** it is inside
hydromt reliance (`WflowSbmModel`), captured only by the same batching
restructure as §6.1, and at ~10 % of wall it is a second-order lever behind the
run-JIT. Elevate only if probe-justified after the JIT lever lands.

### 5.4 Probe 3 — Snakemake scheduling overhead (negligible)

From the benchmark deltas: the non-wflow, non-downscale, non-weathergen rules
(3.00b/3.01/3.03/3.04/3.05/3.08/3.11) sum to **< 40 s across the whole sweep**
(most rows < 1 s). Snakemake scheduling/DAG overhead is **not** a lever; the
sweep is compute-bound in `run_wflow` (~84 %) with downscale (~10 %) and
weathergen (~6 %) the only other terms. Confirmed negligible.

### 5.5 The production scaling model

**Form.** Sweep wall as a function of the measured/structural terms —
per-process fixed cost `F` (start + `using Wflow` + run-JIT ≈ 135 s + gap-if-fixed),
per-run **cold** simulation `S_cold` (basin- and window-dependent; fixture 21yr
≈ 208 s, measured probe 1c) and per-run **warm** simulation `S_warm` (fixture 21yr
≈ 124 s, measured probe 1d), sweep size `K`, effective parallelism `p` (≈ `-c N`
capped by cores and by the single-core-per-run load), and batch size `B`.

**Sweep size `K`.** `K = RLZ_NUM × (ST_NUM + [run_historical])`: `run_wflow` fans
out over cst `{ST_START..ST_NUM}` with `ST_START = 0 if run_historical else 1`
(Snakefile lines 52-53), so the `cst_0` baseline run enters `K` **only when
`run_historical` is set**. The seed fixture has `run_historical` false → `K =
RLZ_NUM × ST_NUM = 2 × 6 = 12` (matching the 12 benchmark rows and the
`ceil(12/3)×343` arithmetic below); a production config that sets `run_historical`
adds `RLZ_NUM` runs.

```
wall_today    ≈ ceil(K / p) × (F + S_cold)                       # K uniform jobs across p cores
wall_sysimage ≈ ceil(K / p) × (F_img + S_cold),  F_img ≈ start only (~1–2 s), always cold

batched:  partition K into n = ⌈K/B⌉ batches of sizes b_1..b_n
          (⌊K/B⌋ full batches of B, plus one REMAINDER batch of r = K mod B when B ∤ K)
          per-batch duration  D(b_i) = F + S_cold + (b_i − 1)·S_warm     # run 1 cold, runs 2..b_i warm
wall_batched = LPT-simulated makespan of {D(b_1)..D(b_n)} across p workers   # estimator of record
          bracketed by  max(D_max, ΣD/p)  ≤  wall  ≤  ΣD/p + (1 − 1/p)·D_max  # Graham list-scheduling bounds
```

where the fixture values are the measured F = 135 s, S_cold = 208 s (probe 1c),
S_warm = 124 s (probe 1d). The batched form makes the warm discount explicit in
the per-batch cost — one cold sim + `(b_i − 1)` warm sims per session — and the
**makespan is estimated by scheduling the batch durations, not by a wave count**:
batches need not all hold `B` runs (the remainder batch is shorter), and the
makespan depends on which batch lands in which worker slot. The estimator of
record is a ~10-line greedy **LPT simulation** (longest batch first onto the
soonest-free of `p` workers), which the commit-1 baseline script runs as
`dev/scripts/estimate_batch_makespan.py`; Snakemake's actual scheduler is greedy
with unspecified job order, so the true makespan lies inside the Graham
list-scheduling bracket above — the model states the bracket alongside the LPT
value. **When `B | K` (every batch full) the LPT makespan reduces exactly to the
v2 wave formula `ceil(⌈K/B⌉/p) × D(B)`**, so the fixture table below is
unchanged (re-verified by simulation: B=2→934, B=3→1182, B=4→715, B=6→963;
today→1372; sysimage→840).

**What changes for non-divisible K/B (the case the wave formula got wrong).**
Worked demonstration, same fixture terms, K=13, p=3: at B=4 the batches are
(4,4,4,1) with durations (715,715,715,343); the wave formula claims
`ceil(4/3)×715 = 1430`, but the simulated makespan is **1058** (the 343 s
remainder batch backfills a freed worker — a 35 % overestimate). Worse, the wave
formula **mis-ranks**: it prefers B=3 (1182) over B=4 (1430), while the
simulation shows B=4 wins (1058 vs 1182). Remainder-aware scheduling is
therefore load-bearing for the batch-size choice, not a refinement.

**Precision claim (what the model is for).** The model's job is **honest ranking
of batch sizes at a given (K, p)** — not exact wall prediction. It claims
ordering fidelity within the Graham bracket (width `(1 − 1/p)·D_max`, i.e. up to
one near-full-batch duration); its per-run terms are idle single-process probe
values (mixed-condition caveat below), and the deployed wall under the tee and
`-c` concurrency will differ. Commit 2's measured before/after — not the model —
is the evidence of record; the model selects the candidate `B` values worth
measuring and predicts their ordering.

The two levers use **different** sim structure: sysimage always launches fresh
cold processes (every run pays `S_cold`), while batching pays `S_cold` on run 1
and `S_warm` on runs 2..N — the §probe-1d discount, which **only** batching
captures.

**The two regimes (the honest-scaling core, G4).**
- **Small basin (the fixture): F ≈ 39 % of per-run (measured).** Both fixed-cost
  levers help, but the ranking is a **trade, not an arithmetic sweep** (below).
- **Production basin (10× cells, 50-yr runs): S ≈ 3,000+ s, F still ~135 s → F ≈
  4 % of per-run.** The sweep is **simulation-bound**; sysimage saves only ~4 %,
  batching a bit more (warm discount on a bigger S) but both small. The remaining
  lever is **per-run sim speed** — Wflow's cell-parallel `--threads` (§6.3).

**Sweep-wall arithmetic on the fixture — ILLUSTRATIVE, mixed-condition, NOT a
firm ranking.** The per-run terms below come from **idle single-process** probes
under **mixed tee/no-tee** conditions (cold S=208 was measured *with* the
`run_logged` tee, probe 1c; warm S=124 *without* it, probe 1d — the tee is ~42 s,
so a consistently-sourced batch differs from the naive sum). Production rule 3.10
runs **every** job under the tee at **`-c 3` concurrency**. So these numbers show
the *shape* of the trade, not a reliable ~10-point ranking; **commit 2's
before/after settles it.**

| lever / B | sim per run | illustrative sweep | vs today* |
|---|---|---|---|
| today (per-process, cold) | 208 | 1,372 | — |
| **sysimage** (F→~2, always cold) | 208 | 840 | ~−39 % (monotone) |
| batching B=2 | 208 / 124 | 934 | ~−32 % |
| batching B=3 | 208 / 124 | 1,182 | ~−14 % (sched-quant) |
| batching B=4 | 208 / 124 | 715 | ~−48 % (best B) |
| batching B=6 | 208 / 124 | 963 | ~−30 % |

*\*"today" here is the idle-arithmetic `ceil(12/3)×343`; the **deployed** baseline
is the benchmark's ~390–410 s/run under actual `-c 3` (~14 % above idle). Commit 1
measures the real `-c N` sweep wall directly — all per-run terms above are
idle-measured placeholders.*

**Two structural findings the arithmetic surfaces (both robust to the number
noise).**
1. **Batching has a warm-cache advantage sysimage cannot capture.** Batched runs
   2..N pay warm sim (~124 s) not cold (~208 s) — an ~84 s/run discount; sysimage
   always launches fresh cold processes and pays cold sim every run (§probe 1d).
   This *may* put well-tuned batching ahead of sysimage on raw throughput, but the
   margin is within fixture noise (above) — commit 2's before/after decides. The
   confounds are asymmetric: tee-on-warm-runs and `-c 3` concurrency only shrink
   batching's edge, so the fixture illustration is a *lower* bound on how much
   batching's raw-throughput edge erodes under production conditions.
2. **Batching's win is batch-size-sensitive; sysimage's is not.** The realized
   win swings ~3× with how the `⌈K/B⌉` batch durations pack onto `p` workers
   (fixture: −48 % at B=4 vs −14 % at B=3), and remainder batches shift the
   packing again (the K=13 demonstration above) — batch size is load-bearing and
   must be chosen by running the §5.5 makespan simulation against the actual
   (K, p); a sysimage (monotone, K uniform jobs across p cores) has no such
   sensitivity. This scheduling-quantization fragility is a cost the batching
   lever owns.

**So the two levers trade throughput against safety (§6.5).** Batching is the
built-first, no-dependency lever (the G1 gate ruling); it *may* take a
raw-throughput margin when tuned (warm discount, up to ~−48 % at B=4) but owns
the batching-path costs — the C5 failure-isolation degradation (§6.1, §11), the
`p × B` disk peak (§6.1), the per-batch benchmark collapse (§7.3), the
scheduling-quantization tuning fragility, and the re-run blast radius (§11). A sysimage
would sidestep every one of those axes (zero DAG change, per-process isolation and
disk, per-cst benchmark rows, monotone), which is why the safety analysis is
**retained as the rationale for keeping sysimage alive as a deferred,
approval-gated follow-up** (§6.2, §6.5) — not as a reason to build it first. Both
levers win cleanly only on small basins; on large simulation-bound sweeps both are
marginal (regime 2).

**Crossover.** Fixed-cost levers dominate while `F ≳ S` per run; on the fixture
F/cold-S ≈ 0.65, collapsing toward 0 as cells×years grow. **P3-3's fixed-cost
levers are a small-basin optimization**, measured on the fixture and *modelled* —
never claimed as measured — for production. Commit 1's direct sweep-wall baseline
seats the fixture before-number; the production rows stay a model (OQ-4).

### 5.6 The CPU resource contract (fixed thread budget for every measurement)

The scaling model's `p`, Julia's `--threads`, and the batch concurrency are not
independent knobs — left jointly unfixed, a batched-vs-baseline comparison can
win or lose through **oversubscription or thread reallocation** rather than
fixed-cost amortization, and an operator cannot derive a safe `(B, -c,
--threads)` triple. This contract fixes them.

**Machine and accounting reality (verified against the tree).**
- Rule 3.10 declares **no `threads:` directive** (Snakefile line 361-374), so
  Snakemake counts each `run_wflow` job as **1** against `-c N` — `-c 3` runs 3
  Julia sessions concurrently regardless of their `--threads` setting. The
  batched rule keeps the same accounting (no `threads:` directive), so
  `p = min(N, #ready batches)` — identical job-counting before and after.
- The dev/measurement box is a 13th-gen i5-1335U: **10 physical cores (2 P + 8 E),
  12 logical.** Today's production invocation (`-c 3`, `julia --threads 4`)
  nominally claims 3 × 4 = 12 Julia threads — exactly the logical-core count —
  but the benchmark evidence shows each run at **~95 % single-core load, cpu ≈
  wall** (`wf3_benchmarks.md`; intake evidence table): on the 384-cell fixture
  `--threads 4` buys nothing (§6.3), so the *effective* load is ~3 busy cores.
  The nominal claim is at capacity; the effective one is far under it. Both
  numbers are stated so neither mode can hide oversubscription.

**The contract (binding on every measured mode — baseline, batched, deployed).**
1. **Every measurement states its triple.** A recorded wall number is meaningless
   without `(-c N, --threads t, B)` (B = 1 for per-process modes); commit 1's
   baseline note records the triple next to every number.
2. **No nominal oversubscription:** `N × t ≤ C_logical` (12 on this box). Today's
   default (3, 4) sits exactly at the cap.
3. **Before/after at the identical budget.** Commit 2's batched runs MUST use the
   **same `(N, t)` pair as the commit-1 baseline — `(-c 3, --threads 4)` on this
   box** — so `B` is the only moved knob. `p` (concurrent Julia sessions) is then
   identical in both modes, and any measured win or loss is attributable to
   fixed-cost amortization + the warm discount, not to changed thread allocation.
   A comparison run at differing `(N, t)` is invalid as commit-2 evidence.
4. **Deployment derivation order (how an operator gets a safe triple):**
   `t` first — from basin threadability (keep the config default unless the
   basin's measured per-run load shows multi-core scaling; the fixture's does
   not, §6.3); `N` second — from the machine: `N ≤ C_logical / t` as the nominal
   cap, relaxable toward `N ≤ C_logical` only where the per-run load is measured
   single-core (the fixture case: (3, 4) is nominally at cap but effectively ~3
   cores); `B` last — from the §5.5 makespan simulation over candidate values,
   capped by the §6.1 disk ceiling. The triple is derived `t → N → B`, never
   picked independently.

§6.1's batch-size selection, §6.3's threads decision, and §9's commit-1/commit-2
measurement specs all bind to this contract; the §9 go/no-go throughput criterion
is evaluated **only** at the fixed budget of contract item 3.

## 6. Selected approach — the lever set, mechanism-pinned

Four candidate levers attack **overlapping** slices of the per-invocation cost.
Their savings are NOT additive. Per the G1 gate ruling (status.md, 2026-07-24)
**batching (lever A) is built first** as the no-dependency lever; the sysimage
(lever B) is **deferred** to a conditional, approval-gated follow-up reached only
if batching fails the commit-2 go/no-go criteria (executable definition in §9;
§6.5).

```
 per run_wflow process (343 s) = [ start 0.9 ][ using 16.4 ][ run-JIT ~118 ][ ─ cold sim ~208 ─ ]
 sysimage (B) attacks:                       └──────── 16.4 + 118 ───────┘   (always cold sim)
 batching  (A) attacks:           └── 0.9+16.4+118 once/session ─┘  + warm sim 124 on runs 2..N
 threads   (C) attacks:                                                       └ sim (prod only) ┘
 (batching gets an 84 s/run warm-cache discount sysimage cannot — §probe 1d, §5.5)
 (lever D / loglevel dropped — probe 1b: logging is not a per-run cost, §6.4)
```

### 6.1 Lever A — batching N Wflow runs per Julia session

**Mechanism.** `Wflow.run(tomlpath)` (verified,
`Wflow/…/src/Wflow.jl:250`) takes the TOML path as a direct argument; the
production shell already reaches it via the no-arg `run()` reading `ARGS[1]`
(`:344`). A batched job is a **thin Julia driver** that loops over N TOML paths
in one session:

```julia
using Wflow
for t in ARGS
    try
        dt = @elapsed Wflow.run(t)          # production logging preserved
        # write dt + "OK" to this cst's own per-run log line
    catch e
        # write the exception to this cst's per-run log; DO NOT rethrow —
        # continue the batch (COMPUTE-level isolation only; persistence
        # isolation is DEGRADED under batching — see "Failure isolation (C5)")
    end
end
```

The driver is a new `blueearth_cst/experiment/run_wflow_batch.jl` (Julia is
juliaup-managed on PATH; the script is not a pixi entry point — mirrors the
existing inline `julia … Wflow.run()` invocation).

**Rule restructuring (DAG shape may change; outputs may not — anchor).** Rule
3.10 changes from a per-`(rlz,cst)` wildcard rule to **parse-time loop-generated
anonymous rules** — one rule per batch, named `run_wflow_batch_<b>`, each
declaring a **static** per-batch output list. **This construct has NO in-repo
precedent.** The existing aggregator `export_wflow_results` (Snakefile lines
377-395) is a **fixed-output** `expand`-input aggregator — two hardcoded outputs
(`Qstats.csv`, `basin.csv`), no wildcard, no generated rules — and a grep of all
three Snakefiles finds no loop-generated anonymous rule (`rule:` + `name:`)
anywhere, so this is genuinely new Snakemake surface for the batching
task-brief, not a copy of an existing rule.

**Expressibility (probe-verified — the ext2-001 arbitration evidence).** The v3
sketch declared the per-batch outputs via callable `output:` entries
(`csvs = lambda w: [...]`). Snakemake **rejects** that construct: the driver
probe
`dev/p33/probes/snakemake-output-expressibility/Snakefile_lambda`
fails on the pinned Snakemake 9.6.2 with `RuleException: Only input files can be
specified as functions` — rule outputs must be statically declared; only inputs
may be functions. The user-mandated replacement is probe-verified in the same
directory: `Snakefile_looprules` (parse-time loop-generated anonymous rules with
static per-batch output lists) **dry-runs clean** on the same pinned Snakemake,
with `rule all`'s per-cst targets each resolving to the generated batch rule
that statically declares them.

**Worked construct sketch (pin the shape; exact code deferred to the brief).** `K`
and `B` are **parse-time constants** (`K` from `RLZ_NUM`/`ST_NUM`/`run_historical`;
`B` a config knob), so the partition is fully determined at parse time and
**no checkpoint is needed** (Snakemake checkpoints are only required when outputs
depend on values unknown until a job runs; here they do not) — and because each
generated rule's member set is fixed at parse time, **no input function is needed
for member resolution** either:

```python
# parse-time, before the rules: a deterministic partition dict
members = [(r, c) for r in range(1, RLZ_NUM+1)
                   for c in range(ST_START, ST_NUM+1)]          # the K runs
batches = {bid: members[i:i+B] for bid, i in                    # batch id -> [(rlz,cst), ...]
           enumerate(range(0, len(members), B))}                # last slice = the remainder
                                                                # batch (< B members) when B does
                                                                # not divide K — first-class in
                                                                # the §5.5 makespan model

for _b, _members in batches.items():                            # one anonymous rule per batch,
    rule:                                                       # generated at parse time
        name:
            f"run_wflow_batch_{_b}"
        input:
            forcing = [f"{exp_dir}/realization_{r}/inmaps_rlz_{r}_cst_{c}.nc"
                       for (r, c) in _members],                 # STATIC lists — no input
            tomls   = [f"{exp_dir}/model_runs/wflow_sbm_rlz_{r}_cst_{c}.toml"
                       for (r, c) in _members]                  # function needed
        output:
            csvs   = [f"{exp_dir}/model_runs/output_rlz_{r}_cst_{c}.csv"
                      for (r, c) in _members],                  # explicit STATIC list
            states = [temp(f"{exp_dir}/model_runs/outstates_rlz_{r}_cst_{c}.nc")
                      for (r, c) in _members]                   # comprehension, NOT expand,
                                                                # NOT a callable
        params:
            members = _members                                  # (rlz,cst) pairs for the driver
# rule all / 3.11 still request the per-cst output_rlz_*_cst_*.csv paths;
# Snakemake routes each to the generated batch rule whose static output list
# contains it (Snakefile_looprules demonstrates exactly this target resolution).
```

Each generated rule's `output:` is an **explicit static Python list
comprehension over that batch's members** — not an `expand()`, not a callable —
so each batch rule declares exactly its own per-cst CSVs and `temp()` outstates
(paths unchanged from today), and the driver receives its members via `params:`
(no wildcard parsing). Follow-on facts of the construct, stated plainly:
**rule names become `run_wflow_batch_<b>`** — one named rule per batch in
`--dry-run` / `--list` output instead of a single wildcarded `run_wflow`; the
per-batch `log:`/`benchmark:` files follow the house pattern keyed by the batch
id (the `3.10_run_wflow` part directory retained so `merge_benchmarks`'s
path-derived rule attribution still works, filenames `batch_<b>.log` /
`batch_<b>.tsv` in place of `rlz_<n>_cst_<m>.*` — exact names in the brief), and
the per-cst **driver-written** log lines are unchanged (below). The downstream
`rule all` / 3.11 targets are unchanged: they still name the per-cst
`output_rlz_<n>_cst_<m>.csv` paths, and Snakemake resolves each to the generated
batch rule whose static output list contains it (probe-demonstrated, above).
Exact signatures are deferred to the task-brief; the **construct** — parse-time
partition dict, loop-generated anonymous rules `run_wflow_batch_<b>`, static
per-batch input/output lists, members via `params:`, no input function, no
checkpoint — is pinned here so the brief cannot pick an inexpressible shape.

**Outputs are byte-identical to today.** Each generated batch rule declares the
same per-cst `output_rlz_<n>_cst_<m>.csv` (persisted) + `temp()` `outstates_…nc`
at **identical paths and identical content** to today (HM-5 byte-compatible; the
P3-2b `validate_hm5` + `validate_hm_gauge_column_identity` stay green). A rule
declaring multiple static per-cst outputs is Snakemake-legal (probe-verified,
`Snakefile_looprules`); the driver writes each cst's CSV to its own declared
path exactly as `Wflow.run` does now.

**Batch partitioning — how batch size is chosen.** Three ceilings bound `B`:
1. **Parallelism ceiling.** Batching serializes within a session, so it must NOT
   destroy the across-cores parallelism it would otherwise consume. Partition the
   `K` runs into `ceil(K/B)` batches and let Snakemake schedule those batches
   across `-c N` — i.e. **`B` chosen so `ceil(K/B)` still saturates the cores**
   (roughly `B ≈ K / N`). All-K-in-one-session is rejected (§7.1): it is net
   **worse** than today's parallel wall.
2. **Scheduling-quantization (§5.5, measured, warm- and remainder-aware).** The
   realized win is dominated by how the batch durations pack onto `p` workers:
   on the fixture B=4 gives −48 % but B=3 only −14 % and B=6 −30 % (all net wins
   once the warm discount is counted, but the *magnitude swings ~3×* with B),
   and a remainder batch shifts the packing again (§5.5's K=13 demonstration,
   where the naive wave count mis-ranks B). Batch size is load-bearing and is
   chosen by running the §5.5 LPT makespan estimator
   (`dev/scripts/estimate_batch_makespan.py`) over candidate B values at the
   actual `(K, p)` under the §5.6 resource contract. This is a tuning-fragility
   cost the batching lever owns (a monotone sysimage, if ever built, would not
   have it — one axis of the deferred-sysimage rationale in §6.5).
3. **Disk ceiling (AGENTS.md `temp()` rule) — the BINDING constraint.** Rule 3.09
   (downscale) emits the per-cst forcing NC as a `temp()` output and rule 3.10
   emits the per-cst outstates NC as a second `temp()` output (Snakefile line 368),
   so **both** `temp()` classes are held per batch until the batch finishes. With
   `p` batch jobs running concurrently at `-c N`, peak disk ≈ **`p × B ×
   (forcing_size + state_size)`** — NOT `B × forcing` (all `p` concurrent batches
   are resident) and NOT forcing-only (the outstates temp is co-resident). At
   B=4/p=3 that is **all 12 forcing NCs + all 12 outstates NCs resident
   simultaneously — the entire sweep's temp footprint** — exactly the disk
   explosion the AGENTS.md `temp()` rule exists to prevent. So `B` is capped so
   `p × B × (forcing_size + state_size)` stays within a stated disk headroom; on
   large `RLZ_NUM×ST_NUM` production runs this ceiling binds well before the
   parallelism/scheduling-quantization ones and forces `B` small (recovering near-today per-run
   disk behavior). Both `temp()` classes' deletion moves to **batch** granularity —
   a documented disk-vs-throughput trade the batch size encodes. The batching
   task-brief must confirm the outstates reclamation timing under the batched rule
   (the outstates temp is consumed by nothing downstream, so it should be
   reclaimable at batch end, but that is verify-in-commit-2, not assumed).

`B` is a **config knob** (default chosen from `-c N`), not hardcoded, so a large
basin can shrink it toward 1 (recovering today's per-run isolation + disk
behavior) without a code change. The full `(B, -c N, --threads)` triple is
derived `t → N → B` per the §5.6 resource contract — `B` is never tuned in
isolation from the thread budget.

**Per-run logs + benchmarks (granularity preservation).** Snakemake emits **one**
`log:`/`benchmark:` per job. Today `run_wflow` already emits **one row per
`(rlz,cst)` job** — 12 rows, one per fan-out job (merge_benchmarks derives the
`rule` column from each part file's relative path, `merge_benchmarks.py:66`), not
one aggregated per-rule row. The generated `run_wflow_batch_<b>` rules collapse
those 12 per-job rows to `⌈K/B⌉` per-batch rows, one per generated rule (files
named by batch id under the retained `3.10_run_wflow` part directory — the
construct's follow-on naming fact, above). Preserve per-run visibility by having the driver write
each cst's `@elapsed` + status to a **per-cst log line** (the `run_logged` tee
still wraps the batch job); the coarse Snakemake benchmark row then covers the
batch, with the fine-grained per-run timing living in the driver-written logs.
This mirrors exactly what the probe driver does. This benchmark/log collapse is
**invisible to both value-identity gates by construction**: the `benchmarks/` and
`logs/` trees are excluded from `semantic_tree_diff` (`EXCLUDED_DIR_NAMES`) and
are absent from `check_baseline`'s manifested TARGETS — so the row-collapse cannot
threaten value-identity (§8, arch-6).

**Failure isolation (C5) — DEGRADED under batching, not preserved.** The per-TOML
`try/catch` above keeps *compute* isolation — one cst raising is logged and
skipped, the other csts in the batch still run to completion. But **persistence**
isolation degrades. The driver does not rethrow, so the batch process exits 0 with
the failing cst's output CSV **absent**; the rule then trips
MissingOutputException and is marked FAILED, and Snakemake's default behavior on a
failed job is to **remove that job's present output files** (absent
`--keep-incomplete`) — including the CSVs the driver **successfully produced for
the B−1 batch-mates**. So one bad cst in a batch of `B` causes Snakemake to
**delete the B−1 completed sibling CSVs and re-run the whole batch**. Downstream,
rule 3.11 `export_wflow_results` `expand`s over **all** rlz×cst CSVs, so one failed
batch blocks Qstats/basin for the **whole sweep** until that batch re-runs —
whereas today, each cst is its own job, so a failing cst deletes only its own
output and blocks only itself.

**This is a real degradation of decision criterion C5**, quantified: failure
**blast radius grows from 1 cst to `B` csts** (deleted + re-run), and rule 3.11 is
blocked for the sweep. **The design ACCEPTS and documents this degradation** rather
than baking in an unverified persistence mechanism (measure-first discipline): a
`--keep-incomplete` / split-output persistence scheme *might* narrow the blast
radius, but it interacts with `--keep-going` semantics in ways this design cannot
verify pre-commit. **If** commit 2's throughput win motivates recovering per-cst
persistence, the exact mechanism to probe is the `--keep-incomplete` ↔
`--keep-going` interaction (does `--keep-incomplete` preserve the successfully
written sibling CSVs across the failed batch job, and does the sweep still
re-run only the failed cst?) — with **accept-the-degradation as the explicit
fallback if the probe fails**. The stated posture, pending any such probe, is
**C5 = DEGRADED, blast radius `B`** — a cost the batching-first path honestly
owns (§5.5, §11); it is not softened by pointing at the deferred sysimage.

### 6.2 Lever B — PackageCompiler sysimage (DEFERRED, APPROVAL-GATED FOLLOW-UP)

**Status (G1 gate ruling).** The sysimage is **NOT** the built-first lever. Per
the recorded G1 ruling (status.md, 2026-07-24), it is **deferred to a conditional
follow-up** reached only if batching **fails the commit-2 go/no-go criteria
(GN-1..GN-4, §9)** — **and** it then needs a **fresh approval ask** for the new
dependency. This section specifies it fully so that (a) the safety analysis below
is on record as the rationale for keeping it alive as the deferred option, and (b)
if the go/no-go fails, the follow-up commit is already scoped. It is not a
commit-2 default and not an approval fork for commit 2 (§9).

**What it would attack.** A prebuilt sysimage bakes `using Wflow` (16.4 s) **and**,
if the precompile workload includes a full `Wflow.run`, the run-JIT (~118 s) — i.e.
~134 of the ~135 fixed seconds — **with ZERO DAG change.** The `run_wflow` rule
stays per-cst; only the `julia` invocation gains `--sysimage wflow.so`. This
preserves failure isolation, `temp()` per-run disk behavior, and the per-cst
benchmark rows — every anchor-risk axis batching perturbs. That anchor-clean
profile is precisely why the sysimage is worth keeping alive as the deferred
option: if the batching lever's measured C5/disk/re-run costs (§6.1, §11) prove
too painful in commit 2 — concretely, if any §9 go/no-go criterion fails — this
is the fallback that trades a new dependency for none of those degradations. **Estimated ceiling
(upper bound, NOT measured):** if the sysimage bakes the full 135 s fixed, per-run
343 → ~210 s ≈ a **~34–39 % sweep-wall reduction** (343-probe vs ~395-benchmark
per-run). The true figure depends on **how much of the 118 s run-JIT is bakeable
JIT vs unbakeable cold first-touch/I-O** — an open question (OQ-6) the
warm/cold gap (§5.2) shows is non-trivial and that only **commit 2's before/after
measures.** The claim is monotone (no scheduling-quantization) and anchor-clean; the
*magnitude* is an estimate pending commit 2.

**The ask (for the user decision — do NOT assume approval).**
- **New dependency:** `PackageCompiler.jl` in the Julia env (juliaup-managed,
  outside pixi — consistent with the existing Julia-not-in-pixi constraint).
  Gated by `new-dependency-requires-approval`.
- **Cost:** a one-time sysimage build (minutes) producing a platform-specific
  `.so`/`.dll` that is **not committable** (built per platform, like the Julia
  env itself; add to `.gitignore`, build in `pixi run install` or a documented
  step).
- **Staleness:** the sysimage must be **rebuilt on any change to the baked
  dependency closure**, not just a Wflow.jl version bump. Julia recompilation
  correctness depends on the **whole** project Manifest (Wflow's transitive deps
  included) **and** the Julia version (1.11.7) — a dep bump that changes a Wflow
  dependency without touching the Wflow Manifest line would leave a stale sysimage
  silently running mismatched compiled code. So the rebuild trigger must be keyed
  on a **hash of the full `Manifest.toml` plus the pinned Julia version** (or on
  the sysimage's own recorded build-Manifest hash), **not** on the single Wflow
  entry. A stale sysimage silently runs old code — a correctness risk that must be
  gated. (Spec-refinement note for the deferred sysimage commit; not needed while
  the lever is deferred — risk-6.)
- **Residual JIT caveat:** PackageCompiler bakes only what the precompile
  workload exercises; include a real `Wflow.run` over a tiny model so the
  simulation dispatch is captured. Residual first-call JIT for
  workload-uncovered paths is **possible, not zero** — claim reduction, not
  elimination.

**Deferral framing.** The sysimage's anchor-clean profile (dominates batching on
every anchor axis, §6.5) is the **rationale for keeping it as the deferred,
approval-gated follow-up** — not a reason to build it first. The design specifies
it fully so the follow-up is ready if the §9 go/no-go fails; the G1 ruling fixes
batching-first, so this section does **not** recommend building the sysimage in
commit 2.

### 6.3 Lever C — threads/cores tuning

The current `--threads 4` buys nothing **on the fixture**: `cpu_time ≈ wall` at
~95 % single-core load across all 12 `run_wflow` rows. This is a **384-cell
(16×24) fixture fact**, not a general one: Wflow's Polyester threading
parallelizes **over grid cells**, so a 384-cell basin has almost no thread
parallelism to exploit while a production basin (10⁴–10⁶ cells) does.

**Decision: keep `--threads 4` (or make it config-driven); do NOT remove it.**
Removal is output-identical and would trivially "simplify" the fixture, but it is
a likely **production regression** — on a simulation-bound large basin (§5.5
regime 2) cell-parallel threading is the *only* remaining lever. The design keeps
the flag and states plainly: **the fixture cannot measure its production
benefit** (same honest-scaling posture as the fixed-cost levers). Optionally
promote `--threads` to a config value so a deployment can tune it to its basin
without a Snakefile edit. Either way the value is governed by the §5.6 resource
contract: it is part of the stated `(N, t, B)` triple, held **identical** across
commit-2's before/after runs, and capped by `N × t ≤ C_logical` — a lever
comparison must never double as a thread-allocation change.

### 6.4 Lever D — reduce Wflow log verbosity (DROPPED — probe 1b)

**Dropped.** This lever was conditional on probe 1b attributing the 262-vs-400
gap to per-run logging I/O. Probe 1b (§5.2) shows run2/run3 are unchanged (35–36
s) under production `loglevel="info"` vs `silent=true` — **logging is not a
per-run cost on this basin**, so a `loglevel` change would save nothing on the
sweep wall. Recorded here as a considered-and-rejected lever, not carried into the
commit plan. (Re-openable only if a production basin's benchmark shows a
logging-bound per-run term, which the fixture does not.)

### 6.5 Lever ranking (batching-first per the G1 gate ruling)

The G1 gate ruling (status.md, 2026-07-24) fixes the sequence: **batching is built
first as the no-dependency lever; the sysimage is deferred**, reached only if
batching fails the commit-2 go/no-go (GN-1..GN-4, §9), and only after a fresh
approval ask. The ranking is therefore not an approval fork — it is a fixed
build order with a conditional follow-up:

| order | lever | status |
|---|---|---|
| **commit 2 (built first)** | **Batching (A)** | The no-dependency lever the gate fixes. Illustrative fixture win up to ~−48 % (B=4, warm-cache discount, §probe 1d); the win is batch-size-sensitive and the path **owns** the C5 degradation, `p × B × (forcing+state)` disk peak, per-batch benchmark collapse, and re-run blast radius (§6.1, §11). Commit 2 measures its real before/after wall. |
| **deferred follow-up (conditional)** | **Sysimage (B)** | Reached **only if** batching fails any commit-2 go/no-go criterion (GN-1..GN-4, §9), **and** the new PackageCompiler dependency is freshly approved (§6.2). Anchor-clean (zero DAG change, per-process isolation + disk, per-cst benchmarks, monotone) — that profile is why it is kept alive as the fallback, not a reason to build it first. |
| **retained** | **Threads (C)** | Kept for the production regime (§6.3); not a fixture win. Runs alongside whichever fixed-cost lever lands. |
| **dropped** | **Logging (D)** | Not a per-run cost (probe 1b, §6.4). |

**Why batching leads and sysimage is the deferred fallback (the retained safety
analysis).** On raw throughput the fixture *suggests* well-tuned batching may even
out-run a sysimage (illustrative ~−48 % B=4 vs ~−39 % sysimage, via the runs-2..N
warm-cache discount, §probe 1d) — but that margin is **inside the fixture's
mixed-condition sourcing error** (§5.5), so neither lever has a firm arithmetic
edge. The decisive fact is the **gate ruling**: batching carries **no new
dependency**, so it proceeds first and its commit-2 before/after is the evidence
that decides whether the sysimage follow-up is even needed. The sysimage's
countervailing merit — it would sidestep every axis batching degrades — is real
and is **why it is retained as the deferred, approval-gated fallback**, not
discarded:
- **Zero DAG change** — a sysimage keeps run_wflow per-cst; C4 anchor untouched.
- **Failure isolation** free (per-process, as today) — no try/catch driver, no C5
  degradation.
- **Per-run `temp()` disk** unchanged — no `p × B × (forcing+state)` peak.
- **Per-cst benchmark rows** preserved — no granularity collapse.
- **Monotone** — no scheduling-quantization trap; no batch-size to mis-tune.

Those five axes are the costs the batching-first path accepts (§6.1, §11). If
commit 2 fails the go/no-go — GN-1..GN-4 (§9) are the executable definition of
"the win does not justify the costs" — the sysimage follow-up recovers all five,
at the price of a new, approval-gated dependency. That is the exact deferral the
gate encodes.

**Overlapping-cost reasoning.** Sysimage and batching attack the **same** fixed
seconds, so combining them buys little beyond whichever lands — the design lands
**one** fixed-cost lever (batching first; sysimage only as the conditional
replacement, never both). Lever C (threads) attacks a **disjoint** term
(production sim), so it is retained alongside the fixed-cost lever. Lever D
(logging) is dropped (§6.4).

**Degenerate case (honest):** on a simulation-bound production basin (regime 2,
§5.5) the fixed cost is ~4 % of per-run, so **both A and B are near-worthless**
there — that is itself a headline finding, and the design records the scaling
model as a first-class deliverable, not a footnote. On the **fixture** the fixed
cost is ~39 % (measured), so a measurable fixture win is expected from the
built-first batching lever — up to ~48 % for well-tuned batching (warm discount)
at the cost of the anchor axes + tuning fragility it owns (§5.5/§6.5); the
deferred sysimage fallback would reach ~39 % monotone without those costs if the
§9 go/no-go fails.

## 7. Alternatives considered

### 7.1 Batching topology — all-in-one-session vs batch-per-core vs one-per-job
- **All K in one session.** Rejected: serializes the whole sweep. Fixture
  arithmetic (ground-truth F=135, S_cold=208) — `135 + 12×208 ≈ 2,631 s` **serial
  wall** vs today's ~4-wave parallel wall at `-c 3` (`ceil(12/3)×343 = 1,372 s`)
  — is nearly **2× worse**. It destroys the parallelism it aims to help.
- **One session per job (today).** The status quo: pays F once per run.
- **Batch-per-core (SELECTED for the batching lever, §6.1).** `ceil(K/B)` batches
  scheduled across `-c N`, `B` chosen by the §5.5 LPT makespan estimator (at the
  §5.6 budget) and capped by the disk ceiling — amortizes F while keeping cores
  saturated. The only topology that can win on **wall** (not per-run), and only
  for well-chosen B (§5.5 scheduling-quantization).

### 7.2 Batching first vs sysimage-deferred vs both
Covered in §6.5. The **G1 gate ruling decides the order**: batching is built first
because it carries **no new dependency**; the sysimage is deferred to a
conditional, freshly-approval-gated follow-up. **Not a clean arithmetic dominance
either way** — the fixture *suggests* well-tuned batching out-throughputs a
sysimage via the warm-cache discount (§probe 1d), but the margin is inside the
mixed-condition sourcing error (§5.5) and unreliable. The sysimage's structural
merit (zero DAG change, per-process isolation + disk, per-cst benchmarks,
monotone) is the **rationale for keeping it as the deferred fallback**, reached
only if batching fails the commit-2 go/no-go (§9) — not a reason to build it
first. Batching owns the anchor costs (C5 degradation, `p × B` disk, benchmark
collapse, re-run blast radius) as the price of leading. Combining buys little
(overlapping fixed cost). Not both.

### 7.3 Where per-run timing lives — Snakemake benchmark vs in-session @elapsed
Batching collapses Snakemake's per-job benchmark rows. Rejected: accepting coarse
per-batch rows only. Selected: the driver writes per-cst `@elapsed` to each cst's
log line (the probe driver's mechanism), preserving per-run visibility under a
batched rule. The Snakemake benchmark row then measures the batch; the fine
timing lives in logs — an honest, documented granularity shift, not a loss.

### 7.4 Baseline metric — summed rule time vs wall-at-cores
Rejected: the `wf3_benchmarks.md` summed TOTAL (5,668 s) as the baseline —
it is not wall clock and overstates every lever's win by the parallelism factor
(§5.1). Selected: end-to-end wall clock at a fixed `-c N`, measured directly in
commit 1.

### 7.5 Fix-first without the probe
Rejected (intake decision 2): risks optimizing the wrong term. The probe was
decisive — it revealed run-JIT (~118 s), not `using Wflow` (16.4 s), as the
dominant fixed cost, which reorders the levers (sysimage must bake `run`, not
just `using`; a `using`-only sysimage would miss the bulk).

## 8. Value-identity proof plan

The anchor is **strict value-identity** (R3/R5 discipline). Every lever's landing
commit must pass, before/after:

1. **Full wf3 semantic diff clean — the byte-identity gate on every persisted
   per-cst output.** `dev/scripts/semantic_tree_diff.py --ref <ref-tree> --cur
   <cur-tree>` over a full wf3 run pre/post the change (its actual CLI takes
   `--ref`/`--cur` directory args) — the DAG shape may change but the produced
   tree's content may not. This is the gate that asserts **per-cst
   `output_rlz_<n>_cst_<m>.csv` byte-identity** (they are NOT manifested, so gate 2
   does not cover them). The landed precedent is
   `dev/p31/baseline_diffs.md` + `dev/p31/_wf3_regen.log` (the durable p31
   determinism evidence); the untracked `dev/p31/_semantic_diff.out` is a working
   artifact, not the precedent of record.
2. **Manifested targets unchanged.** `python dev/scripts/check_baseline.py check
   --workflow climate_experiment` — this gate covers **only its manifested
   TARGETS**: `Qstats.csv`, `basin.csv`, and the wf3 config snapshot
   (`check_baseline.py:114-116`). It does **not** fingerprint the per-cst
   `output_rlz_*.csv` — those ride gate 1 (and gate 3). Note the
   **mixed-provenance caveat** (`check_baseline.py` docstring /
   `baseline-manifest-coverage` memory): wf3 rows are pre-restoration; a
   sub-tolerance wf1 discharge move *may* surface in a re-run Qstats — that is the
   documented ADR-0001 immaterial branch. **But see the discriminating rule below:
   that branch must not be used to launder a batching-induced drift.**
3. **P3-2b relational validators green.** `pixi run pytest
   tests/test_interchange_contracts.py -rs` — `validate_hm5`,
   `validate_hm_gauge_column_identity`, `validate_hm4` (the per-cst TOMLs), and
   `validate_hm7` must stay green: the batched rule must still emit per-cst
   `output_rlz_<n>_cst_<m>.csv` and per-cst `wflow_sbm_rlz_<n>_cst_<m>.toml` with
   identical content and the `<header>_<mapid>` gauge-column identity (HM-4→HM-5→
   HM-7). Together with gate 1, this is the byte-compatibility gate on the
   restructuring. (Gauge-column reduction is filename-keyed, not batch-order-keyed,
   so batch grouping cannot corrupt the identity — panel-verified positive.)
4. **Suite + three dry-runs green.** `pixi run pytest tests/test_cli.py` (dry-runs
   all three Snakefiles) after any Snakefile/signature edit; the full `pytest
   tests/` where the restructure touches Python.

**Gate coverage of the benchmark/log collapse (arch-6).** The per-batch
benchmark-row collapse (§6.1, §7.3) is **invisible to both value-identity gates by
construction**: `benchmarks/` and `logs/` are excluded from `semantic_tree_diff`
(`EXCLUDED_DIR_NAMES`) and are absent from `check_baseline`'s TARGETS. So the
granularity shift cannot threaten value-identity — it is provably outside gates 1
and 2, not merely "documented."

**Determinism handling — warm-vs-cold byte identity is an UNTESTED assumption.**
Value-identity presupposes the pipeline is deterministic given fixed seeds; the
weathergen seed is pinned (`weathergen_config.yml` `seed`; WG-3 contract). The
repo's landed determinism evidence (`dev/p31/baseline_diffs.md`: "0 failed …
R5-verified wf3 determinism, seed 123", via a whole-tree semantic diff) establishes
**per-process run-to-run** reproducibility **only**. Batching introduces a **new**
risk it does not cover: **warm-SESSION vs cold-PROCESS** byte identity. Runs 2..N
of a batch reuse in-session allocations, JIT-compiled method instances, and
GC/global state a fresh process lacks — and §probe-1d measures an 84 s/run
warm-cache effect, i.e. **direct proof the warm path is materially different
execution**. Whether that perturbs any floating-point reduction order (threaded
accumulation, allocation-dependent SIMD paths) to a sub-LSB CSV difference is
**untested; commit-2 gate-1 (the whole-tree semantic diff at exact tolerance) is
its first evidence** — not something the P3-1 per-process re-run already
discharged (OQ-3).

**Discriminating rule (arch-3 / risk-3) — the ADR-0001 branch must not launder a
batching drift.** Any `Qstats`/`basin`/per-cst diff that **correlates with
batching** — present in the batched tree, absent in a **per-process re-run of the
same inputs** — must be treated as a **lever regression and BLOCK the commit**.
The ADR-0001 immaterial branch is admissible **only after** a per-process
(non-lever) re-run **reproduces** the diff, proving it is the pre-existing wf1
move and not batching. Gate 1 is therefore run **per-process-vs-batched on
identical inputs** as the discriminating test: a gate-1 diff on a `run_wflow`
output is attributed to batching first and cleared only by ruling out the
warm/cold path. This keeps the strict value-identity anchor (C2, "absolute")
actually enforced for the one change class P3-3 introduces.

## 9. Commit plan

Probe/baseline first; each commit suite-green + dry-runs clean; the restructuring
commit carries its own before/after wall measurement.

The sealed three-commit plan lands **batching** as the fixed-cost lever (the G1
gate ruling). The sysimage is **not** a commit in this plan — it is a conditional
follow-up outside it (below).

1. **`p33: record wf3 performance baseline + cost decomposition`** — the durable
   baseline note under `dev/p33/` (à la the `baseline_diffs` precedent):
   end-to-end wall at the stated §5.6 triple (`-c N`, `--threads t`, B=1), the
   probe decomposition (§5.2), the scaling model (§5.5) with its per-B LPT
   makespan table produced by the new `dev/scripts/estimate_batch_makespan.py`
   (the ~10-line estimator of record, landed by this commit), the resolved gap
   attribution (probe 1b), and the **measurement spec** for the single-sample
   point estimates (§5.2 numbers are n=1 and confirmed by this commit's
   re-measured sweep wall — risk-4). No workflow-code change (the estimator is
   dev-process tooling under `dev/scripts/`, not shipped); suite + dry-runs
   green trivially. **User gate on the measured numbers before proceeding**
   (milestone gate is here — §10).
2. **`p33: batching N Wflow runs per Julia session`** — the built-first,
   no-dependency lever (§6.1): the `run_wflow_batch.jl` driver + the rule 3.10
   restructure to **parse-time loop-generated anonymous rules**
   (`run_wflow_batch_<b>`, static per-batch output lists, members via `params:`,
   no input function, no checkpoint — the ext2-001 probe-verified construct) +
   the batch-size config knob. **Commit-2 gate includes the arbitration-mandated
   minimal dry-run demonstration:** the expressibility probe pattern
   (`probes/snakemake-output-expressibility/Snakefile_looprules`) re-run against
   the **real** batched rules — a `--dry-run` of `Snakefile_climate_experiment`
   on the pinned Snakemake showing every per-cst `output_rlz_*_cst_*.csv` target
   resolving to its generated `run_wflow_batch_<b>` rule — recorded **before**
   the full sweep is executed. Carries its **own measured before/after wall** on
   the fixture, both runs at the **identical §5.6 resource-contract triple** —
   this is the commit-2 evidence the gate names, and it is evaluated against the
   **go/no-go criteria below**. Full value-identity gate (§8), including the
   per-process-vs-batched discriminating diff (warm/cold byte identity, first
   tested here), plus the GN-4 failure-injection run. Dry-runs clean,
   `test_cli.py` green, P3-2b validators green.
3. **`p33: <docs + roadmap seal>`** — update `dev/roadmap.md` P3-3 to sealed, the
   before/after headline, tag `p33-performance`.

Commit 1 is a hard prerequisite (the user gate); commit 2 is batching, the single
fixed-cost lever the gate fixes. The former conditional loglevel commit is dropped
(probe 1b, §6.4). If Lever C is promoted to a config knob, it folds into commit 2
(output-identical).

**Commit-2 gate — the batching go/no-go (the LEVER decision, NOT the milestone
floor).** "Batching disappoints" is not left to after-the-fact judgment; it is
defined as **failing any of the four criteria below**, all evaluated on commit
2's measured evidence at the fixed §5.6 resource budget:

- **GN-1 — throughput.** Measured end-to-end wf3 sweep-wall reduction on the
  fixture, batched vs the commit-1 baseline **at the identical `(N, t)`
  budget**, must be **≥ 15 %**. Derivation from the model (not a-priori): the
  most conservative feasible batch size (B=2) has a simulated win of **−32 %**
  (1,372 → 934 s, §5.5), and the documented confounds (tee on warm runs, `-c`
  concurrency) only *erode* batching's edge — so half the most-conservative
  prediction, ~16 %, rounded to 15 %, is the smallest measured win still clearly
  attributable to the amortization mechanism rather than noise, and the
  smallest that plausibly outweighs the C5 / disk / re-run costs the lever owns
  (§6.1, §11). The chosen B's own simulated prediction is also recorded
  (predicted vs measured), but the go/no-go floor is the fixed 15 %.
- **GN-2 — value-identity.** Every §8 gate green, **including** the
  discriminating per-process-vs-batched diff. Any batching-correlated diff is
  an automatic **no-go** — and blocks commit 2 outright regardless of the
  sysimage decision (C2 is absolute).
- **GN-3 — disk ceiling.** Measured peak temp-file footprint during the batched
  sweep stays within the stated `p × B × (forcing_size + state_size)` cap
  (§6.1), and the outstates `temp()` is observed reclaimed at batch end (the
  §6.1 verify-not-assume item). An unreclaimed temp class or a peak above the
  cap is a no-go.
- **GN-4 — failure injection.** One deliberately failing cst injected into one
  batch must realize **exactly the documented C5 cost and no more**: the batch
  job fails via MissingOutputException, the B−1 sibling CSVs are deleted and the
  batch re-runs, rule 3.11 is blocked only until that re-run, per-cst driver log
  lines are present for every batch member including the failed one, and a
  subsequent re-run (fault removed) converges to a clean sweep passing GN-2.
  Any realized blast radius **beyond** the documented one — silent corruption,
  damage outside the batch, a batch that cannot be cleanly re-run — is a no-go.

**Decision rule.** All four pass → batching stands; the sysimage stays dormant
(deferred indefinitely, no ask made). **Any one fails → the fresh PackageCompiler
approval ask is triggered** (the G1 ruling's condition is met), with the failed
criterion named as the evidence; batching's own disposition (keep, shrink `B`
toward 1, or revert) is decided at that ask. Adjudication: `cst-architect`
evaluates the criteria on `model-builder` / `model-validator` evidence at the
commit-2 gate; the dependency ask itself goes to the **user**.

**Anchor distinction (why GN-1 does not violate the no-a-priori-floor anchor).**
Intake decision 3's "no a-priori speedup floor" governs the **MILESTONE
acceptance gate** — the user signs off on the measured before/after numbers with
no threshold imposed on them. GN-1 is a different decision: an **internal
lever-routing criterion** that only decides whether the design's own deferred
fallback branch (the sysimage ask) is triggered. The two are independent: a
measured −12 % would fail GN-1 (triggering the sysimage ask) and could still be
accepted by the user at the milestone gate as the landed result; conversely no
GN number substitutes for the user's sign-off. The milestone gate remains
floor-free.

**Conditional follow-up (outside the sealed plan) — sysimage.** Reached **only
if** commit 2 fails at least one go/no-go criterion above, **and** only after a
**fresh approval ask** for the PackageCompiler dependency (§6.2). It is doubly
gated (go/no-go-failed AND dependency-approved) and is not part of the three
sealed commits above — the design scopes it (§6.2) so the follow-up is ready
without re-designing, but it is deliberately not a commit-2 default or an
approval fork.

## 10. Validation plan

| gate | who | checks |
|---|---|---|
| baseline correctness | `cst-architect` (this design) → **user** | commit-1 measured wall + decomposition + scaling model; **milestone gate = user sign-off on the numbers** |
| model runs | `model-builder` | executes the wf3 sweep before/after each lever **at the identical §5.6 resource-contract triple `(N, t)`** (a mismatched budget invalidates the comparison); produces the wall measurements + the full output tree for the diff; executes the GN-4 failure-injection run and the GN-3 disk-peak measurement |
| value-identity | `model-validator` | gate 1–3 of §8 on the before/after trees. **Acceptance criterion:** per-cst `output_rlz_*_cst_*.csv` byte-identity rides **gate 1 (semantic_tree_diff) + gate 3 (P3-2b `validate_hm5`)**, NOT check_baseline; check_baseline (gate 2) asserts only `Qstats.csv`/`basin.csv`/config-snapshot identity. **Discriminating rule (§8): any diff correlated with batching (present batched, absent in a per-process re-run on identical inputs) BLOCKS the commit; the ADR-0001 immaterial branch is admissible only after a per-process re-run reproduces the diff.** P3-2b suite green. |
| workflow mechanics | `python-engineer` (batching driver + rule) / delegated impl | `test_cli.py` + three dry-runs green; the rule restructure implements the §6.1 **loop-generated-rules construct** (`run_wflow_batch_<b>`, static per-batch outputs, members via `params:` — not callable outputs), and commit 2 records the **arbitration-mandated minimal dry-run demonstration** (§9): the `Snakefile_looprules` probe pattern re-run against the real rules, every per-cst target resolving to its generated batch rule, before the full sweep; failure-isolation try/catch present; C5 degradation documented (blast radius `B`), not claimed preserved |
| batching go/no-go (commit-2 gate) | `cst-architect` | evaluates GN-1..GN-4 (§9) on the `model-builder` / `model-validator` evidence at the fixed §5.6 budget; all-pass → sysimage stays dormant; any-fail → triggers the sysimage ask below, naming the failed criterion. **Internal lever-routing decision, not the milestone floor** (§9 anchor distinction) |
| sysimage approval (conditional) | **user** | approve/reject the PackageCompiler dependency (§6.2 ask) — a **fresh ask reached only if** commit 2 fails a §9 go/no-go criterion; does NOT gate commit 2, which is batching by the G1 ruling |

Delegation is by scoped brief (task-brief), never inline code: the batching
driver + rule restructure to `python-engineer` (Snakefile) + the Julia driver;
the sweep execution + before/after wall to `model-builder`; the value-identity
adjudication to `model-validator` with the acceptance criteria named above.

## 11. Consequences / risks

The batching-first path (the G1 gate ruling) owns the costs below; each is a cost
the deferred sysimage fallback would sidestep (§6.5), which is why the sysimage is
kept alive as the conditional follow-up — not a reason to build it first.

- **Batching shifts `temp()` deletion to batch granularity** → higher peak disk
  `p × B × (forcing_size + state_size)` (BOTH `temp()` classes — the 3.09 forcing
  NCs and the 3.10 outstates NCs are co-resident per batch; all `p` concurrent
  batches resident — at B=4/p=3 the whole sweep's temp footprint) on production
  basins; mitigated by the disk-ceiling batch-size cap (§6.1), which binds `B`
  small on large sweeps. The AGENTS.md `temp()` disk rule is the binding
  constraint.
- **Failure isolation (C5) DEGRADED, not preserved.** A batch's failed cst makes
  Snakemake delete the B−1 completed sibling CSVs and re-run the whole batch;
  rule 3.11 then blocks Qstats/basin for the whole sweep. Blast radius grows from
  1 cst to `B`. The design **accepts and documents** this (§6.1); a
  `--keep-incomplete` persistence mechanism is a commit-2 probe candidate with
  accept-the-degradation as the fallback, not an assumed fix.
- **Batch re-run granularity** → under batching the Snakemake job *is* the whole
  batch, so re-running one changed/failed cst re-runs its B−1 batch-mates. This is
  value-identical (idempotent, no anchor breach) but a real **re-run-cost
  regression** on a `temp()`-driven partial re-run — the same blast-radius `B`
  cost as the C5 degradation above.
- **Warm-vs-cold byte identity is untested** → batched runs 2..N execute a warm
  path materially different from a cold process (§probe 1d); commit-2 gate-1 is
  the first evidence it is byte-identical, guarded by the §8 discriminating rule.
- **Fixture over-promises.** The built-first batching lever caps at ~−48 % on the
  fixture, ~4 % on a simulation-bound production basin (§5.5). The scaling model +
  honest-scaling statement (G4) is the guard against a misread win.
- **Batching scheduling-quantization** (§5.5) — a poorly-chosen batch size is
  *slower* than today on small sweeps, and remainder batches shift the packing
  (the K=13 mis-ranking demonstration); the batch-size selection is fragile — a
  tuning cost the batching path owns, managed by choosing `B` with the §5.5 LPT
  estimator under the §5.6 resource contract.
- **Coarser benchmark rows** under batching → per-run timing moves to
  driver-written logs (§7.3); invisible to both value-identity gates (§8, arch-6).
- **Sysimage staleness (deferred-lever spec note)** → if the sysimage follow-up is
  ever built, a stale image silently runs old code; the rebuild trigger must be
  keyed on a hash of the **full `Manifest.toml` plus the pinned Julia version**
  (not the single Wflow entry — a transitive-dep bump would otherwise slip
  through), §6.2. Correctness risk if ungated; not active while the lever is
  deferred.

## 12. Open questions
- **OQ-1 (cost split).** RESOLVED by probes 1b + 1c (§5.2): logging is not a
  per-run cost (→ lever D dropped); the fresh-process 21yr run =
  **343 s ≈ 135 s fixed (39 %) + 208 s sim (61 %)**. These are **single-sample
  point estimates**, confirmed by commit 1 (which re-measures the **sweep wall at
  `-c N`**, not the single-run number, to seat the before-baseline). The
  decomposition's *structure* (JIT-dominant fixed cost, sim the rest) is settled;
  the exact split is n=1 pending commit 1's confirmation (risk-4).
- **OQ-2 (sysimage — deferred, not a commit-2 fork).** Per the G1 gate ruling,
  **commit 2 is batching** (the no-dependency lever); the sysimage is a **deferred
  conditional follow-up**. The PackageCompiler dependency approval is a **fresh ask
  reached only if** batching fails the commit-2 go/no-go (GN-1..GN-4, §9 — the
  executable definition of
  "disappoints") — it does **not** gate commit 2. (This resolves the v1 "sysimage
  or batching per OQ-2" fork, which inverted the ruling.)
- **OQ-3 (warm-vs-cold determinism — UNTESTED for batching).** The landed P3-1
  evidence establishes **per-process** determinism only; batching's new risk is
  **warm-session-vs-cold-process** byte identity, whose **first evidence is
  commit-2 gate-1** (§8). This is not something the P3-1 per-process re-run already
  discharged. Guarded by the §8 discriminating rule (a batching-correlated diff
  BLOCKS; ADR-0001 branch admissible only after a per-process re-run reproduces
  the diff).
- **OQ-4 (production basin run).** Should one production-sized basin be run to
  replace the modelled production numbers with measured ones? Out of the fixture
  budget; recorded as a candidate follow-up (the scaling model is otherwise
  extrapolation, per the intake).
- **OQ-5 (sysimage build wiring — deferred lever).** If the sysimage follow-up is
  ever reached, where does the build live — `pixi run install`, a standalone
  `dev/scripts/` builder, or a documented manual step — and how is the
  full-Manifest+Julia-version staleness trigger (§6.2, risk-6) implemented?
- **OQ-6 (bakeable JIT fraction — deferred lever).** The sysimage ceiling
  (~34–39 %) assumes it bakes the full 135 s fixed. How much of the 118 s run-JIT
  is bakeable compilation vs unbakeable cold first-touch/I-O is **not measured** —
  the large warm/cold sim gap (§probe 1d) shows cold-process effects are
  non-trivial. Only measured if the sysimage follow-up is reached; if the bakeable
  fraction is low, the sysimage's advantage over batching narrows further.

## 13. Revision log
- v1 — initial draft. Probe 1 (in-session startup/JIT/sim split): run-JIT ≈118 s
  the dominant fixed cost. Probe 1b (logging): not a per-run cost → lever D
  dropped. Probe 1c (DECISIVE, fresh-process 21yr production run): **343 s = 135 s
  fixed (39 %) + 208 s sim (61 %)** — replaces the proxy reconstruction, corrects
  the fixed fraction down from a mistaken ~52 %. Probe 1d (warm/cold sim split):
  warm sim(21yr) ≈124 s vs cold ≈208 s — batched runs 2..N capture an ~84 s/run
  warm-cache discount that sysimage's always-cold processes cannot, which *may*
  put well-tuned batching ahead of sysimage on throughput. That margin is
  **provisional** (fixture arithmetic mixes tee/no-tee + idle/concurrent
  conditions, §5.5; commit-2 settles it), and the confounds only shrink batching's
  edge — so sysimage is recommended on **safety** (zero DAG change, anchors
  intact, monotone), not arithmetic. Probes 2/3 from code + benchmark deltas.
  Skeleton-first, filled in place.
- v2 — internal-panel revision (1 blocking / 7 major / 10 minor, all dispositioned;
  the review record ledger). **Group A (blocking arch-1 + risk-2 + repo-1):** rewrote
  §5.5/§6.1/§6.2/§6.5/§7.2/§9/§11/OQ-2 **batching-first** per the recorded G1 gate
  ruling — batching is the built-first no-dependency commit-2 lever; the sysimage
  is a **deferred, doubly-gated follow-up** (batching-disappoints AND
  fresh-approval), its safety analysis retained as the deferral rationale, not a
  recommendation. The v1 "sysimage recommended default / commit-2 approval fork"
  framing is fully removed. **Group B (risk-1):** C5 stated **DEGRADED under
  batching** (blast radius `B`: Snakemake deletes the failed job's present outputs
  → B−1 completed batch-mates deleted + whole batch re-runs; rule 3.11 blocked for
  the sweep) — **accepted and documented**, with the `--keep-incomplete` ↔
  `--keep-going` interaction named as a commit-2 probe and accept-the-degradation
  the explicit fallback. **Group C (arch-3 + risk-3):** §8/OQ-3 reworded —
  warm-session-vs-cold-process byte identity is UNTESTED, first evidence =
  commit-2 gate-1; added the discriminating rule (a batching-correlated diff
  BLOCKS; ADR-0001 branch admissible only after a per-process re-run reproduces
  it). **Group D (arch-2):** replaced the boxed batched formula with the warm-aware
  `wall_batched ≈ ceil(⌈K/B⌉/p) × (F + S_cold + (B−1)·S_warm)` (verified to
  reproduce the table exactly: 934/1182/715/963); table re-derived from it as the
  single source of truth. **Group E (arch-4 + repo-3):** added the worked
  batched-rule construct sketch to §6.1 (parse-time partition dict, input function,
  explicit output-list comprehension, no checkpoint) and stated plainly there is NO
  in-repo precedent (export_wflow_results is a fixed-output expand aggregator only).
  **Group F (minors):** K = RLZ_NUM × (ST_NUM + [run_historical]) (arch-5);
  benchmark/log collapse invisible to both value-identity gates (arch-6); n=1
  language softened to single-sample-confirmed-by-commit-1 (risk-4, folded into the
  commit-1 measurement spec — no repeats run, not cheap read-only); disk ceiling =
  p × B × (forcing + state), both temp classes (risk-5); sysimage staleness keyed
  on full Manifest + Julia version (risk-6, deferred-lever note); per-cst CSV
  identity attributed to gates 1+3 not check_baseline (repo-2); §5.1 per-JOB row
  parenthetical fixed (repo-4); gate-1 cites dev/p31/baseline_diffs.md +
  --ref/--cur CLI (repo-5); -c N attributed to the invocation not the profile
  (repo-6). Panel-verified positives preserved.
- v3 — external-round-1 revision (3 major, all accepted; review-record
  ledger Round = external-r1). **ext1-001 (remainder batches + makespan; faulted the accepted
  arch-2 formula):** §5.5 model rebuilt remainder-aware — batches have sizes
  `b_i` (⌊K/B⌋ full + one remainder `r = K mod B` when `B ∤ K`), per-batch
  duration `D(b_i) = F + S_cold + (b_i−1)·S_warm`, makespan estimated by a
  ~10-line greedy **LPT simulation** (the estimator of record, landed by commit 1
  as `dev/scripts/estimate_batch_makespan.py`) bracketed by the Graham
  list-scheduling bounds `[max(D_max, ΣD/p), ΣD/p + (1−1/p)·D_max]`. For `B | K`
  the simulation reduces exactly to the v2 wave formula, so the fixture table is
  unchanged (re-verified by simulation: 934/1182/715/963; today 1372; sysimage
  840); for non-divisible K/B the wave formula both overestimates and
  **mis-ranks** (worked K=13, p=3 demonstration: wave claims B=3 1182 < B=4
  1430; simulation shows B=4 1058 < B=3 1182). Precision claim stated: honest
  ranking of B within the Graham bracket, not exact wall prediction.
  "Wave-quantization" renamed scheduling-quantization throughout. **ext1-002
  (CPU resource contract):** new §5.6 — every measured mode states its
  `(-c N, --threads t, B)` triple; nominal cap `N × t ≤ C_logical` (dev box
  i5-1335U: 10 physical / 12 logical cores; today's (3, 4) sits exactly at the
  nominal cap, ~3 busy cores effectively per the ~95 % single-core benchmark
  evidence); commit-2 before/after MUST run at the identical `(N, t)` as the
  commit-1 baseline so `B` is the only moved knob (any delta is attributable to
  amortization, not oversubscription or thread reallocation); deployment
  derivation order `t → N → B` (basin threadability → machine cap → §5.5
  simulation under the disk ceiling); rule 3.10 verified to declare no
  `threads:` directive (each job counts 1 against `-c N`), and the batched rule
  keeps that accounting. §6.1/§6.3/§9/§10 bound to the contract. **ext1-003
  (executable go/no-go):** §9 commit-2 gate defines "batching disappoints" as
  failing any of **GN-1** measured sweep-wall reduction ≥ 15 % at the fixed
  budget (derived, not a-priori: half the most-conservative feasible-B model
  win, B=2 → −32 %, confounds only erode batching's edge), **GN-2** all §8
  value-identity gates green incl. the discriminating diff (fail = no-go AND
  blocks the commit, C2 absolute), **GN-3** measured disk peak within the
  `p × B × (forcing+state)` cap + outstates reclaimed at batch end, **GN-4**
  failure injection realizes exactly the documented C5 blast radius (B−1
  siblings deleted, batch re-runs, 3.11 blocked only until the re-run, per-cst
  driver log lines present incl. the failure, clean re-convergence). Decision
  rule: all-pass → sysimage stays dormant; any-fail → the fresh PackageCompiler
  approval ask naming the failed criterion. Anchor distinction stated
  explicitly: the intake's no-a-priori-floor governs the MILESTONE gate (user
  sign-off on measured numbers, floor-free); GN-1 is an internal LEVER-routing
  threshold only — a GN-1 failure can coexist with user acceptance at the
  milestone gate. §6/§6.2/§6.5/§7.2/§10/§11/OQ-2 rewired from "disappoints" to
  the GN criteria. All 18 prior panel resolutions preserved.
- v4 — stage-6a ARBITRATION revision, confined to **ext2-001** (external-r2
  blocking; user ruling 2026-07-24: ACCEPTED, FIX REQUIRED; round cap reached,
  no further external rounds). The v3 §6.1 construct declared per-batch outputs
  as callables (`output: csvs = lambda w: [...]`), which Snakemake rejects —
  probe-verified: `probes/snakemake-output-expressibility/Snakefile_lambda`
  fails on the pinned Snakemake 9.6.2 with `RuleException: Only input files can
  be specified as functions`. §6.1 rebuilt around the user-mandated,
  probe-verified replacement (`Snakefile_looprules`, same dir, dry-runs clean):
  **parse-time loop-generated anonymous rules** — `for _b, _members in
  batches.items(): rule:` named `run_wflow_batch_<b>`, each with a **static**
  per-batch output-list comprehension (per-cst CSV + `temp()` outstates paths
  unchanged) and members passed via `params:`; no wildcard-dependent outputs, no
  input function, no checkpoint. Follow-on facts stated in §6.1: per-batch rule
  names in dry-run/list output; per-batch log/benchmark files keyed by batch id
  under the retained `3.10_run_wflow` part directory (per-cst driver-log
  preservation unchanged); `rule all`/3.11 target resolution unchanged
  (probe-demonstrated); the no-in-repo-precedent statement now covers
  loop-generated anonymous rules. §9 commit 2 + §10 workflow-mechanics gate name
  the loop-generated-rules mechanism and add the ruling-mandated minimal
  dry-run demonstration (the probe pattern re-run against the real rules before
  the full sweep). This supersedes the construct half of the accepted
  arch-4/repo-3 resolution; all other accepted resolutions (scaling model §5.5,
  resource contract §5.6, GN criteria, C5/§8 posture, lever ranking) untouched.
