"""Dry-month rule tests for step 6b (design §5.6, ruling A2). Falsifiers N1-N5.

The seed basin is equatorial and wet year-round, so **none** of this is reachable
through the fixture. Every behaviour below is unit-test territory by necessity,
which is also why the boundary cases are tested on both sides rather than one.
"""

import pytest

from blueearth_cst.projections.dry_month import (
    DEFAULT_MAX_FLAGGED_MONTHS,
    DEFAULT_MIN_REFERENCE,
    ThresholdError,
    combination_is_flagged,
    is_flagged,
    resolve_thresholds,
)
from blueearth_cst.projections.variable_spec import parse

SEED_SPEC = parse(
    {
        "precip": {"source": "precip", "canonical": "rate", "units": "mm/day",
                   "change": "relative"},
        "temp": {"source": "temp", "canonical": "state", "units": "degC",
                 "change": "absolute"},
    }
)


# --- N1: strict, on both sides ------------------------------------------------


def test_N1_below_the_threshold_is_flagged():
    assert is_flagged(0.099, 0.1) is True


def test_N1_exactly_at_the_threshold_is_NOT_flagged():
    """The design says strict and says the boundary is tested on both sides."""
    assert is_flagged(0.1, 0.1) is False


def test_N1_above_the_threshold_is_not_flagged():
    assert is_flagged(0.101, 0.1) is False


def test_N1_a_wet_month_is_nowhere_near_flagged():
    assert is_flagged(5.4, 0.1) is False


# --- N3: absolute variables are never flagged ---------------------------------


def test_N3_only_relative_variables_get_a_threshold():
    """A near-zero reference temperature is an ordinary 0 degC, not a defect."""
    thresholds = resolve_thresholds(SEED_SPEC)
    assert "precip" in thresholds and "temp" not in thresholds


def test_N3_the_default_is_the_A2_value():
    assert resolve_thresholds(SEED_SPEC)["precip"] == 0.1
    assert DEFAULT_MIN_REFERENCE == {"precip": 0.1}


# --- N4: an unknown relative variable must raise ------------------------------


RUNOFF = {"runoff": {"source": "runoff", "canonical": "rate", "units": "m3/s",
                     "change": "relative"}}


def test_N4_a_relative_variable_without_a_default_raises():
    with pytest.raises(ThresholdError, match="runoff"):
        resolve_thresholds(parse(RUNOFF))


def test_N4_the_error_refuses_to_borrow_the_precip_default():
    with pytest.raises(ThresholdError, match="unrelated quantity"):
        resolve_thresholds(parse(RUNOFF))


def test_N4_supplying_the_threshold_resolves_it():
    assert resolve_thresholds(parse(RUNOFF), {"runoff": 2.5})["runoff"] == 2.5


def test_N4_a_configured_value_overrides_the_default():
    assert resolve_thresholds(SEED_SPEC, {"precip": 0.05})["precip"] == 0.05


# --- N5: max_flagged_months, strict -------------------------------------------


def test_N5_exactly_max_does_not_flag_the_combination():
    """One season of flagged months is a dry basin's normal state, not a fault."""
    assert combination_is_flagged(3) is False
    assert DEFAULT_MAX_FLAGGED_MONTHS == 3


def test_N5_more_than_max_flags_it():
    assert combination_is_flagged(4) is True


def test_N5_the_bound_is_configurable():
    assert combination_is_flagged(3, max_flagged=2) is True


def test_N5_zero_flagged_months_is_not_flagged():
    assert combination_is_flagged(0) is False


# --- the shape Snakemake params actually carry --------------------------------


def test_thresholds_resolve_from_the_plain_field_lists_params_carry():
    """Params serialise the spec as lists, so the resolver must accept both."""
    params_shape = {
        "precip": ["precip", "precip", "rate", "mm/day", "relative"],
        "temp": ["temp", "temp", "state", "degC", "absolute"],
    }
    assert resolve_thresholds(params_shape) == {"precip": 0.1}
