"""LPT makespan estimator for the wf3 batching lever (P3-3 design §5.5).

The estimator of record for the batch-size choice. Given a sweep of `K` Wflow
runs partitioned into batches of size `B` (⌊K/B⌋ full batches plus one REMAINDER
batch of `r = K mod B` when `B` does not divide `K`), each batch runs in one
Julia session at per-batch duration::

    D(b_i) = F + S_cold + (b_i - 1) * S_warm

-- run 1 of the session pays the cold simulation `S_cold`, runs 2..b_i reuse the
warm session and pay `S_warm` (the probe-1d warm-cache discount that only batching
captures). The `n = ceil(K/B)` batch durations are packed onto `p` workers by a
greedy **LPT** simulation (longest batch first onto the soonest-free worker); the
makespan is the last worker to finish. Because Snakemake's real scheduler is
greedy with unspecified job order, the true makespan lies inside the Graham
list-scheduling bracket, reported alongside the LPT value::

    max(D_max, sum(D)/p)  <=  wall  <=  sum(D)/p + (1 - 1/p) * D_max

Two non-batching modes are just `B = 1` with a different fixed cost `F` (the
warm term is inert at B=1, so every run is cold by construction):

    today (per-process)  = (B=1, F=135)
    sysimage             = (B=1, F=2)   # bakes ~all of the 135 s fixed cost

Fixture terms are the measured probe values: F=135 s, S_cold=208 s (probe 1c),
S_warm=124 s (probe 1d). See dev/p33/performance-passes-design.md §5.5 and
dev/p33/probes/PROBE_RESULTS.md.

The `--table` mode reproduces the design's per-B fixture comparison table.

This is dev-process tooling under dev/scripts/ (not shipped, no snakemake
global). Runnable standalone.

Usage::

    python dev/scripts/estimate_batch_makespan.py --k 12 --p 3 --b 4
    python dev/scripts/estimate_batch_makespan.py --table
    python dev/scripts/estimate_batch_makespan.py --table --k 13 --p 3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class MakespanEstimate:
    """The estimator output for one (K, p, B, F, S_cold, S_warm) point."""

    batch_sizes: list[int]
    durations: list[float]
    makespan: float
    graham_lower: float
    graham_upper: float


def batch_sizes(k: int, b: int) -> list[int]:
    """Partition K runs into batches of B: ⌊K/B⌋ full + one remainder if B ∤ K.

    The remainder batch (size `K mod B`) is first-class -- it is shorter than a
    full batch and shifts the LPT packing (design §5.5, the K=13 demonstration).
    """
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    if b <= 0:
        raise ValueError(f"B must be positive, got {b}")
    full, rem = divmod(k, b)
    sizes = [b] * full
    if rem:
        sizes.append(rem)
    return sizes


def batch_duration(size: int, f: float, s_cold: float, s_warm: float) -> float:
    """D(b_i) = F + S_cold + (b_i - 1) * S_warm -- one cold sim, rest warm."""
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    return f + s_cold + (size - 1) * s_warm


def lpt_makespan(durations: list[float], p: int) -> float:
    """Greedy LPT: longest job first onto the soonest-free of p workers.

    Returns the makespan (last worker to finish). Makespan is independent of the
    tie-break order among equal-length jobs, so the result is deterministic.
    """
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")
    if not durations:
        return 0.0
    worker_free = [0.0] * p
    for d in sorted(durations, reverse=True):  # longest first (LPT)
        i = min(range(p), key=lambda w: worker_free[w])  # soonest-free worker
        worker_free[i] += d
    return max(worker_free)


def graham_bounds(durations: list[float], p: int) -> tuple[float, float]:
    """Graham list-scheduling bracket [max(D_max, ΣD/p), ΣD/p + (1-1/p)·D_max]."""
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")
    if not durations:
        return 0.0, 0.0
    total = sum(durations)
    d_max = max(durations)
    lower = max(d_max, total / p)
    upper = total / p + (1 - 1 / p) * d_max
    return lower, upper


def estimate(
    k: int,
    p: int,
    b: int,
    f: float,
    s_cold: float,
    s_warm: float,
) -> MakespanEstimate:
    """Full estimate for one (K, p, B, F, S_cold, S_warm) point."""
    sizes = batch_sizes(k, b)
    durations = [batch_duration(s, f, s_cold, s_warm) for s in sizes]
    makespan = lpt_makespan(durations, p)
    lower, upper = graham_bounds(durations, p)
    return MakespanEstimate(
        batch_sizes=sizes,
        durations=durations,
        makespan=makespan,
        graham_lower=lower,
        graham_upper=upper,
    )


# Design fixture defaults (measured probe values, §5.5 / PROBE_RESULTS.md).
F_DEFAULT = 135.0
S_COLD_DEFAULT = 208.0
S_WARM_DEFAULT = 124.0
K_DEFAULT = 12
P_DEFAULT = 3

# The §5.5 fixture comparison table rows: (label, B, F-override).
# "today" and "sysimage" are B=1 with different fixed cost; the rest are batching.
_TABLE_ROWS: list[tuple[str, int, float | None]] = [
    ("today (per-process, cold)", 1, F_DEFAULT),
    ("sysimage (F->~2, always cold)", 1, 2.0),
    ("batching B=2", 2, None),
    ("batching B=3", 3, None),
    ("batching B=4", 4, None),
    ("batching B=6", 6, None),
]


def _fmt_sizes(sizes: list[int]) -> str:
    return "+".join(str(s) for s in sizes)


def print_table(k: int, p: int, s_cold: float, s_warm: float) -> None:
    """Print the §5.5 per-B fixture comparison (LPT makespan + Graham bracket)."""
    today = estimate(k, p, 1, F_DEFAULT, s_cold, s_warm).makespan
    print(f"# LPT makespan table  (K={k}, p={p}, S_cold={s_cold:g}, S_warm={s_warm:g})")
    header = f"{'lever / B':<32} {'batches':>12} {'makespan':>10} {'vs today':>9}  {'Graham [lo, hi]':>22}"
    print(header)
    print("-" * len(header))
    for label, b, f_override in _TABLE_ROWS:
        f = F_DEFAULT if f_override is None else f_override
        est = estimate(k, p, b, f, s_cold, s_warm)
        vs = "--" if abs(est.makespan - today) < 1e-9 else f"{(est.makespan / today - 1) * 100:+.0f}%"
        bracket = f"[{est.graham_lower:.0f}, {est.graham_upper:.0f}]"
        print(f"{label:<32} {_fmt_sizes(est.batch_sizes):>12} {est.makespan:>10.0f} {vs:>9}  {bracket:>22}")


def _print_single(est: MakespanEstimate, k: int, p: int, b: int) -> None:
    print(f"K={k}  p={p}  B={b}")
    print(f"  batch sizes      : {_fmt_sizes(est.batch_sizes)}  (n={len(est.batch_sizes)})")
    print(f"  batch durations  : {[round(d, 1) for d in est.durations]}")
    print(f"  LPT makespan     : {est.makespan:.1f} s")
    print(f"  Graham bracket   : [{est.graham_lower:.1f}, {est.graham_upper:.1f}] s")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--k", type=int, default=K_DEFAULT,
                   help=f"Sweep size K = RLZ_NUM x (ST_NUM + [run_historical]) "
                        f"(default {K_DEFAULT}, the seed fixture)")
    p.add_argument("--p", type=int, default=P_DEFAULT,
                   help=f"Effective parallelism p (~= -c N) (default {P_DEFAULT})")
    p.add_argument("--b", type=int, default=4,
                   help="Batch size B (ignored in --table mode) (default 4)")
    p.add_argument("--f", type=float, default=F_DEFAULT,
                   help=f"Per-process fixed cost F, s (default {F_DEFAULT})")
    p.add_argument("--s-cold", type=float, default=S_COLD_DEFAULT,
                   help=f"Per-run cold simulation S_cold, s (default {S_COLD_DEFAULT})")
    p.add_argument("--s-warm", type=float, default=S_WARM_DEFAULT,
                   help=f"Per-run warm simulation S_warm, s (default {S_WARM_DEFAULT})")
    p.add_argument("--table", action="store_true",
                   help="Print the design §5.5 per-B fixture comparison table")
    args = p.parse_args()

    if args.table:
        print_table(args.k, args.p, args.s_cold, args.s_warm)
        return
    est = estimate(args.k, args.p, args.b, args.f, args.s_cold, args.s_warm)
    _print_single(est, args.k, args.p, args.b)


if __name__ == "__main__":
    main()
