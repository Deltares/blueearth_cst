# P3-3 probe results (raw record)

Julia 1.11.7, `--project=. --threads 4`, Windows, repo root. Fixture wf1 forcing
proxy (`inmaps_historical.nc`, 16x24 grid), windowed 2000-2005 (6yr, ~2190 steps),
3 runs in one session (JIT paid on run 1, amortized on runs 2/3).

- bare start `-e "1"`        : 0.9 s
- `using Wflow`              : 16.4 s
- silent=true : run1=154.2  run2=35.2  run3=37.5  s
- silent=false: run1=178.8  run2=35.0  run3=36.0  s   (loud; production loglevel=info)

Derived: pure sim(6yr)~36s -> sim(21yr wf3 window)~127s; first-call run-JIT~118s;
per-process fixed cost~135s (+ benchmark-implied residual to ~270s from the
Windows sampler + wf3 cold-start-SBM JIT). run2~=run3 under BOTH log modes ->
logging is NOT a per-run cost (lever D dropped). Drivers: batch_probe.jl,
batch_probe_loud.jl. Scratch TOMLs: wf1_{a,b,c}.toml.

## Probe 1c — DECISIVE (fresh process, 21yr, production form, no extrapolation)
Config diff verified: wf1 base TOML routing keys IDENTICAL to per-cst wf3 TOMLs
(river/land kinematic_wave time_step 900/3600, adaptive=false, timestepsecs=86400).
One fresh julia process, `run_logged` tee -> `using Wflow; Wflow.run()`, full
2000-2020 window: 18:51:00 -> 18:56:43 = **343 s wall**.
=> fixed 135 s (39%) + sim(21yr) 208 s (61%). In-session 6yr->21yr linear proxy
UNDERestimated sim by ~80s (warm-alloc reuse + non-linear sim scaling); the
fresh-process number is ground truth. Benchmark ~395s residual (~50s) = Windows
psutil sampler + wf3 cold-start-SBM. Fixed fraction 39%, NOT 52% (proxy-reconstruction
error, corrected). Batching wave-quantization: at K=12,p=3 some B are worse than
today (B=3 +11%, B=6 +1%); sysimage is a clean monotone 39% -> sysimage preferred.

## Probe 1d — warm/cold sim split (the ranking-inverting number)
Two full-21yr runs, one session (in-session, no run_logged tee):
  WARM run 1 (cold+JIT+sim) = 283.6 s
  WARM run 2 (warm sim only) = 124.0 s
=> warm sim(21yr) ~124 s vs cold sim ~208 s -> 84 s/run warm-cache discount that
ONLY batching (runs 2..N of a session) captures; sysimage always launches fresh
cold processes and cannot. Warm-aware sweep arithmetic (K=12,p=3, today 1372 s):
  sysimage (always cold, F->~2): 840 s (-39%, monotone)
  batching B=4 (cold run1 + 3 warm): 715 s (-48%, BEST) ; B=2 934(-32) B=3 1182(-14) B=6 963(-30)
CONCLUSION: batching out-throughputs sysimage when tuned; sysimage recommended on
SAFETY (zero DAG change, anchors intact, monotone), not arithmetic. Ranking is a
trade decided at OQ-2, not a clean sweep. (Corrected an earlier "sysimage
arithmetically dominates" that wrongly charged cold sim to every batched run.)

## CAVEAT on the batching-vs-sysimage arithmetic (mixed conditions)
The -48%/-39% comparison mixes measurement conditions and is NOT reliable to
~10-point precision: cold S=208 was measured WITH the run_logged tee (probe 1c,
343=135+208); warm S=124 was measured WITHOUT the tee (probe 1d, direct julia).
The tee is ~42 s (probe 1d in-session cold 283.6 + startup 17.3 ~= 301 no-tee vs
343 with-tee). Also "today"=ceil(12/3)x343 uses the IDLE single-process 343; the
DEPLOYED benchmark per-run under actual -c3 is ~390-410 (~14% higher). All per-run
terms are idle-measured placeholders; commit 1 (sweep wall @ -c N) and commit 2
(lever before/after) measure the real numbers. The confounds (tee on warm runs,
concurrency) only SHRINK batching's edge -> strengthens the safety-based sysimage
recommendation. Batching-beats-sysimage is PROVISIONAL, not a firm finding.
