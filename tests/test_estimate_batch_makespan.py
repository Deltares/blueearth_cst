"""Unit tests for dev/scripts/estimate_batch_makespan.py (P3-3 §5.5).

Pure unit tests -- no pipeline, no I/O. The load-bearing check is that the LPT
estimator reproduces the design's fixture table integers EXACTLY, and that every
row sits inside its Graham list-scheduling bracket.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# dev/scripts is not an importable package; load the module by path. Register it
# in sys.modules before exec so the frozen @dataclass can resolve annotations.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "dev" / "scripts" / "estimate_batch_makespan.py"
)
_spec = importlib.util.spec_from_file_location("estimate_batch_makespan", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
ebm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ebm
_spec.loader.exec_module(ebm)

# Fixture terms (design §5.5, measured probe values).
F, S_COLD, S_WARM = 135.0, 208.0, 124.0


def _makespan(k: int, p: int, b: int, f: float = F) -> float:
    return ebm.estimate(k, p, b, f, S_COLD, S_WARM).makespan


# --- The design's fixture table (K=12, p=3): the exact-integer targets. -------

@pytest.mark.parametrize(
    "b, f, expected",
    [
        (1, 135.0, 1372),   # today (per-process, cold)
        (1, 2.0, 840),      # sysimage (F->~2, always cold)
        (2, 135.0, 934),
        (3, 135.0, 1182),
        (4, 135.0, 715),    # best B on the fixture
        (6, 135.0, 963),
    ],
)
def test_design_table_exact(b: int, f: float, expected: int) -> None:
    assert _makespan(12, 3, b, f) == expected


def test_nondivisible_demonstration() -> None:
    """K=13, p=3, B=4 -> batches (4,4,4,1), remainder backfills -> 1058 (§5.5)."""
    est = ebm.estimate(13, 3, 4, F, S_COLD, S_WARM)
    assert est.batch_sizes == [4, 4, 4, 1]
    assert est.makespan == 1058


def test_nondivisible_wave_formula_misranks() -> None:
    """§5.5: LPT ranks B=4 (1058) BELOW B=3 (1182); the wave formula got this wrong."""
    assert _makespan(13, 3, 4) < _makespan(13, 3, 3)


# --- Structural invariants ----------------------------------------------------

@pytest.mark.parametrize(
    "k, p, b",
    [(12, 3, 1), (12, 3, 2), (12, 3, 3), (12, 3, 4), (12, 3, 6), (13, 3, 4),
     (7, 2, 3), (100, 8, 5), (1, 1, 1)],
)
def test_makespan_within_graham_bracket(k: int, p: int, b: int) -> None:
    est = ebm.estimate(k, p, b, F, S_COLD, S_WARM)
    assert est.graham_lower <= est.makespan <= est.graham_upper + 1e-9


def test_batch_sizes_partition_k() -> None:
    assert ebm.batch_sizes(12, 4) == [4, 4, 4]
    assert ebm.batch_sizes(13, 4) == [4, 4, 4, 1]
    assert ebm.batch_sizes(1, 5) == [1]
    assert sum(ebm.batch_sizes(37, 5)) == 37


def test_batch_duration_warm_discount() -> None:
    # B=1 is cold-only; the warm term is inert.
    assert ebm.batch_duration(1, F, S_COLD, S_WARM) == F + S_COLD
    # B=4: one cold + three warm.
    assert ebm.batch_duration(4, F, S_COLD, S_WARM) == F + S_COLD + 3 * S_WARM


def test_divisible_reduces_to_wave_formula() -> None:
    """When B | K, LPT makespan == ceil(ceil(K/B)/p) * D(B) (design §5.5)."""
    import math
    k, p, b = 24, 3, 4
    n = k // b
    d = ebm.batch_duration(b, F, S_COLD, S_WARM)
    wave = math.ceil(n / p) * d
    assert _makespan(k, p, b) == wave


# --- Validation / edge behavior -----------------------------------------------

@pytest.mark.parametrize("bad_k", [0, -1])
def test_bad_k_raises(bad_k: int) -> None:
    with pytest.raises(ValueError):
        ebm.batch_sizes(bad_k, 4)


@pytest.mark.parametrize("bad_b", [0, -1])
def test_bad_b_raises(bad_b: int) -> None:
    with pytest.raises(ValueError):
        ebm.batch_sizes(12, bad_b)


def test_bad_p_raises() -> None:
    with pytest.raises(ValueError):
        ebm.lpt_makespan([1.0, 2.0], 0)


def test_lpt_empty_is_zero() -> None:
    assert ebm.lpt_makespan([], 3) == 0.0
