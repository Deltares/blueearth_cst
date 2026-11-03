"""Unit tests for the shared ``stress_test_grid`` helper (R5 §3).

Pure arithmetic + strict validation; no heavy deps, no ``sys.modules``
pollution risk. Pins the strict contract both call sites (the Snakefile and
``prepare_cst_parameters.py``) now share.

Retyped by R14 `C-31` (P1b): the config declares ``n_levels``, the number of
LEVELS on an axis, where it declared ``step_num``, the number of intervals.
``n_levels`` = ``step_num`` + 1, so the helper no longer adds one and the
minimum legal value moved from 0 to 1.
"""

import re

import pytest

from blueearth_cst.shared.snake_utils import (
    index_width,
    member_index_regex,
    stress_test_grid,
)


def test_seed_config_grid():
    """The seed config yields (2, 3, 6).

    Under `C-31` the config declares the LEVEL counts (2, 3) that it used to
    declare as interval counts (1, 2). Same grid, same six members — this is
    the equivalence the retype turns on, pinned here on the shipped values.
    """
    cfg = {"temp": {"n_levels": 2}, "precip": {"n_levels": 3}}
    assert stress_test_grid(cfg) == (2, 3, 6)


def test_one_level_is_a_single_point_axis():
    """The degenerate axis is `n_levels: 1`, where it was `step_num: 0`.

    The FLOOR moves with the meaning, which is the part a bare rename would
    have got wrong: zero intervals was legal and describes one unperturbed
    level, so the minimum level count is one. `n_levels: 0` describes an axis
    with no points at all and is refused below.
    """
    cfg = {"temp": {"n_levels": 1}, "precip": {"n_levels": 1}}
    assert stress_test_grid(cfg) == (1, 1, 1)


@pytest.mark.parametrize(
    "cfg",
    [
        {"precip": {"n_levels": 3}},  # missing temp axis section
        {"temp": {"n_levels": 2}},  # missing precip axis section
        {"temp": {}, "precip": {"n_levels": 3}},  # missing temp.n_levels
        {"temp": {"n_levels": 2}, "precip": {}},  # missing precip.n_levels
    ],
)
def test_missing_n_levels_raises_keyerror(cfg):
    """A missing axis section or n_levels raises KeyError (no silent default)."""
    with pytest.raises(KeyError):
        stress_test_grid(cfg)


@pytest.mark.parametrize("bad", ["2", 1.5, True, False, None])
def test_non_integer_n_levels_raises_valueerror(bad):
    """A non-integer n_levels (incl. bool) raises ValueError."""
    cfg = {"temp": {"n_levels": bad}, "precip": {"n_levels": 3}}
    with pytest.raises(ValueError):
        stress_test_grid(cfg)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_level_count_below_one_raises_valueerror(bad):
    """`n_levels` below 1 raises. ZERO is the case the retype added.

    `step_num: 0` was legal and meant one level; `n_levels: 0` means an axis
    with no points, which is not a grid. A config that renamed the key without
    adding one lands exactly here, so this is where that mistake is caught
    rather than silently halving the grid.
    """
    cfg = {"temp": {"n_levels": bad}, "precip": {"n_levels": 3}}
    with pytest.raises(ValueError):
        stress_test_grid(cfg)


# ---------------------------------------------------------------------------
# R11 P2 — index_width / member_index_regex (C27)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "count,width", [(1, 1), (6, 1), (9, 1), (10, 2), (12, 2), (99, 2), (100, 3)]
)
def test_width_comes_from_the_count(count, width):
    """C27: derived, never fixed at three digits."""
    assert index_width(count) == width


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "6", None])
def test_a_width_needs_a_positive_int_count(bad):
    """Returning 1 for a broken count would paper the grid error over."""
    with pytest.raises(ValueError):
        index_width(bad)


def test_padding_makes_lexical_order_match_numeric():
    """The property, stated as the falsifier: unpadded ids sort wrong."""
    width = index_width(12)
    padded = [f"st_{m:0{width}d}" for m in range(1, 13)]
    unpadded = [f"st_{m}" for m in range(1, 13)]
    assert sorted(padded) == padded
    assert sorted(unpadded) != unpadded  # st_1, st_10, st_11, st_12, st_2, ...


def test_the_member_regex_bars_the_baseline_but_admits_padded_members():
    """Rule 3.12 must never become a second producer of the reserved st_0.

    Verified against Snakemake's own DAG too (a 12x12 dry-run schedules 144
    perturb jobs and no job resolves st_num=00); this pins the regex itself.
    """
    rx = re.compile(f"^{member_index_regex(2)}$")
    assert rx.match("01") and rx.match("10") and rx.match("12")
    assert not rx.match("00")  # the baseline
    assert not rx.match("1")  # UNPADDED -> MissingRuleException, not a silent route
    assert not rx.match("001")  # wrong width


def test_the_member_regex_at_width_one_is_just_the_nonzero_digits():
    """A 6-member grid pads nothing, so the constraint degenerates correctly."""
    rx = re.compile(f"^{member_index_regex(1)}$")
    assert all(rx.match(str(d)) for d in range(1, 10))
    assert not rx.match("0")
    assert not rx.match("01")


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "2"])
def test_the_member_regex_needs_a_positive_int_width(bad):
    with pytest.raises(ValueError):
        member_index_regex(bad)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_the_member_regex_holds_when_EMBEDDED_in_a_path(width):
    """Regression: the constraint must not rely on `$`.

    The obvious spelling, `(?!0+$)[0-9]{width}`, passes every anchored unit
    check above and is still WRONG. Snakemake embeds a wildcard constraint in
    the regex for the WHOLE path, so `$` binds to the end of that path, not the
    end of the wildcard; with `.nc` following, `0+$` never matches, the
    lookahead always succeeds, and the constraint degenerates to `[0-9]{width}`
    -- admitting the baseline and making rule 3.12 a second producer of it.

    That shipped once. It survived a 12x12 `--dry-run` (where the baseline is
    also reachable from its plural rule, so Snakemake prefers that one and the
    ambiguity stays hidden) and surfaced as a `CyclicGraphException` only in
    `test_cross_workflow_inputs` / `test_guard_invalidation`. This case pins the
    property directly, in the position that actually broke.
    """
    rx = member_index_regex(width)
    in_path = re.compile(f"rlz_1_st_({rx})[.]nc$")
    zeros = "0" * width
    assert not in_path.match(f"rlz_1_st_{zeros}.nc"), "the baseline must not match"
    assert in_path.match(f"rlz_1_st_{1:0{width}d}.nc")
    assert in_path.match(f"rlz_1_st_{width and 10**width - 1:0{width}d}.nc")
    if width > 1:
        assert not in_path.match("rlz_1_st_1.nc"), "an unpadded index must not match"
