# P3-3 — Performance passes — scoping intake

**Status.** AUTHORITATIVE scoping record, confirmed with the user 2026-07-24
via `design-scoping` (three decisions elicited and individually confirmed).
The design cycle for this milestone starts from this file; the "Confirmed
scoping decisions" below are fixed anchors for that cycle, not reopened by
it. Design-cycle start is **user-gated** — do not start without the user's
go.

**Provenance.** The last planned Phase-3 milestone (`dev/roadmap.md` §P3-3).
P3-1, P3-2a, P3-2b all sealed 2026-07-24. Candidate input at roadmap level —
the parked realization/stress-test file-format redesign (deferred from
P3-1) — is **excluded** by decision 1 (below), on evidence.

## Benchmark evidence (the scoping's factual basis)

Fixture wf3 run (2 rlz × 6 cst, 16×24 model grid, 20-year runs;
`examples/test_local/experiments/experiment/benchmarks/wf3_benchmarks.md`,
2026-07-24): total 5,668 s (1:34:28) of summed rule time, of which

- `run_wflow` (12 × ~390–410 s): **~84%** — each invocation
  `julia -e "using Wflow; Wflow.run()"` pays full package-load + JIT per
  TOML; `cpu_time ≈ wall` at ~95% single-core load despite `--threads 4`.
  On a basin this small a large fraction of each ~6.5-min run is plausibly
  startup, not hydrology — an our-side wrapper cost, inside CST scope.
- `downscale_climate_realization` (12 ×): ~10%.
- weathergenr + stress-test perturbation: ~6%. Everything else <1%.
- Per-run peak RSS ~930 MB (memory headroom noted, not this milestone's
  target).
- I/O is **not** dominant (84% is compute) — the basis for excluding the
  file-format redesign (decision 1).

## Confirmed scoping decisions (fixed anchors)

1. **Target: wf3 sweep throughput, value-identical only.** P3-3 optimizes
   the production stress-test sweep (`RLZ_NUM × ST_NUM` wflow runs) —
   where the benchmark puts ~84% of wall time — via value-identical
   speedups. The parked realization/stress-test **file-format redesign
   stays parked** (I/O is not the bottleneck per the evidence above; a
   format change is a second risk class). Rejected alternatives:
   whole-pipeline pass (wf1 is a one-off build, wf2 network-dominated —
   little recurring win); memory-headroom focus (real but not the pain);
   throughput+disk (adds the format risk class for a non-dominant cost).
2. **Approach: measure-first + structural latitude.** Commit 1 is a
   profiling probe decomposing the wflow-run cost (Julia package-load/JIT
   vs simulation; downscale internals). Then implement the probe-proven top
   levers — **including, if supported, restructuring how `run_wflow`
   executes** (e.g. batching N TOMLs per Julia session) — under strict
   value-identity: semantic diff clean, `rule all` targets / entry points /
   `run_workflows.py` wrapper contract unchanged. **DAG shape may change;
   outputs may not.** Rejected: non-structural-only (leaves the biggest
   lever unreachable if batching is what startup amortization needs);
   fix-first without the probe (risks optimizing the wrong term —
   production-sized basins may be simulation-dominated where the fixture is
   startup-dominated; the probe distinguishes them).
3. **Success criteria confirmed as stated** (see § Success criteria):
   probe-set expectations, **no a-priori speedup floor** (consistent with
   the repo's rejection of arbitrary thresholds — P3-2a intake decision 4
   precedent); the milestone gate is user sign-off on measured
   before/after + value-identity evidence.

## Constraints (standing; not new to this milestone)

- **CST automation scope.** No Wflow.jl / hydromt / weathergenr internal
  re-engineering. Amortizing OUR invocation overhead (wrappers, batching,
  session reuse, threads/cores) is in scope; patching upstream compute is
  not. `blueearth_cst/weathergen/*.R` untouched unless the probe elevates
  the R stage AND the user approves at design time.
- **New dependencies need explicit user approval** (e.g. a PackageCompiler
  sysimage is a candidate lever but is approval-gated —
  `new-dependency-requires-approval`).
- **Value-identity discipline (R3/R5 style):** wf3 outputs semantic-diff
  clean pre/post; manifested targets unchanged; suite + three dry-runs
  green per commit; platform surface (three entry points, wrapper contract,
  config `workflows:` sections) unchanged.
- Naming per `dev/reference/naming.md`; commit prefix `p33:` (registered
  at scoping); tag `p33-performance` at milestone close.
- Fixture is the era5 test basin; production scaling claims must be stated
  as a model (extrapolation), not measured claims, unless a production
  basin is actually run.

## Success criteria

1. **Recorded baseline + decomposition** — a performance baseline on the
   fixture with the wflow-run cost split (Julia startup/JIT vs simulation;
   downscale internals), plus a documented scaling model for
   production-sized sweeps (what an e.g. 10 × 20 sweep costs before/after).
2. **Strict value-identity** — wf3 semantic diff clean pre/post; manifested
   targets unchanged; suite + dry-runs green; entry points / `rule all`
   targets / wrapper contract untouched.
3. **Measured improvement** — end-to-end wf3 wall-time reduction on the
   fixture, before/after recorded; expected magnitude set by the probe (no
   a-priori threshold); **user sign-off on the measured numbers at the
   milestone gate**.
4. **Standing rules hold** — dependency approval gate; no upstream
   re-engineering; R internals untouched without probe-evidence + approval.

## Cut (YAGNI) / deferred

- Realization/stress-test file-format redesign — stays parked (decision 1;
  I/O non-dominant). Re-openable only with contrary profiling evidence.
- Memory-headroom work for large basins — recorded, not this milestone.
- wf1/wf2 optimization — out (one-off build; network-dominated).
- Any GUI/platform-surface change — out.

## Handoff

Next step (user-gated): design cycle for `p33-performance` with this intake
as scope authority — the design decides the probe design, the lever set and
their mechanisms (batching vs sysimage vs concurrency tuning), the
value-identity proof plan, and the commit sequence — then task-brief →
implementation. Full-loop vs lite is the user's call at design start
(structural DAG latitude argues for the full loop).
