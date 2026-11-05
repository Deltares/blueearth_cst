"""Dry-month rule tests for step 6b (design §5.6, ruling A2). Falsifiers N1-N5.

The seed basin is equatorial and wet year-round, so **none** of this is reachable
through the fixture. Every behaviour below is unit-test territory by necessity,
which is also why the boundary cases are tested on both sides rather than one.
"""

import pytest

from blueearth_cst.projections.dry_month import (
    DEFAULT_MAX_FLAGGED_MONTHS,
    ThresholdError,
    combination_is_flagged,
    is_flagged,
    resolve_thresholds,
)
from blueearth_cst.projections.variable_spec import parse
from blueearth_cst.shared.variable_registry import VARIABLES

#: Named so a test can vary ONE field of it. Since `C-66` a threshold is
#: declared inside the variable's own block rather than passed alongside the
#: spec, so the override cases below have to build a config rather than pass an
#: argument.
SEED = {
    "precip": {
        "source": "precip",
        "canonical": "rate",
        "units": "mm/day",
        "change": "relative",
    },
    "temp": {
        "source": "temp",
        "canonical": "state",
        "units": "degC",
        "change": "absolute",
    },
}

SEED_SPEC = parse(SEED)


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
    """`C-64` moved the shipped default; the VALUE is what N3 is about.

    It was `dry_month.DEFAULT_MIN_REFERENCE = {"precip": 0.1}` and is now
    `VARIABLES["precip"].projections.min_denominator`. Asserted through the
    resolver AND at its new home, so the move cannot be mistaken for the number
    changing.
    """
    assert resolve_thresholds(SEED_SPEC)["precip"] == 0.1
    assert VARIABLES["precip"].projections.min_denominator == 0.1


# --- N4: an unknown relative variable must raise ------------------------------


RUNOFF = {
    "runoff": {
        "source": "runoff",
        "canonical": "rate",
        "units": "m3/s",
        "change": "relative",
    }
}


def test_N4_a_relative_variable_without_a_default_raises():
    with pytest.raises(ThresholdError, match="runoff"):
        resolve_thresholds(parse(RUNOFF))


def test_N4_the_error_refuses_to_borrow_the_precip_default():
    with pytest.raises(ThresholdError, match="unrelated quantity"):
        resolve_thresholds(parse(RUNOFF))


def test_N4_supplying_the_threshold_resolves_it():
    """N4's claim, through the surface that exists after `C-66`.

    It used to pass the threshold as a second argument to `resolve_thresholds`,
    which carried `relative_change.min_reference` from the config. That section
    is dissolved and the argument with it, so the threshold is declared where
    the variable is — which is the whole point of `C-64`.
    """
    supplied = {"runoff": dict(RUNOFF["runoff"], min_denominator=2.5)}
    assert resolve_thresholds(parse(supplied))["runoff"] == 2.5


def test_N4_a_configured_value_overrides_the_default():
    """And it still beats the registry's shipped 0.1 for `precip`."""
    overridden = {
        "precip": dict(SEED["precip"], min_denominator=0.05),
        "temp": SEED["temp"],
    }
    assert resolve_thresholds(parse(overridden))["precip"] == 0.05


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
    """Params serialise the spec as lists, so the resolver must accept both.

    Six fields since `C-64`, and the threshold is the last one. Built from
    `VariableSpec` rather than hand-written, so this test cannot drift out of
    step with the shape `analyze_projections.smk` actually sends -- which is how
    it drifted into asserting the five-field shape in the first place.
    """
    params_shape = {name: list(spec) for name, spec in SEED_SPEC.items()}

    assert [len(fields) for fields in params_shape.values()] == [6, 6]
    assert resolve_thresholds(params_shape) == {"precip": 0.1}


def test_a_params_payload_predating_the_threshold_refuses_rather_than_guessing():
    """A five-field list carries no threshold, and the registry must not fill it.

    It cannot arise from a current run -- Snakemake rebuilds params from the
    Snakefile every time -- but the resolver has the variable NAME and could
    quietly look the threshold up. It must not: `_threshold` already applied the
    registry at parse time, so a value missing HERE means the config's own
    precedence decided against one, and re-deriving it would overrule the config
    from the far side of the params boundary.
    """
    legacy = {"precip": ["precip", "precip", "rate", "mm/day", "relative"]}
    with pytest.raises(ThresholdError):
        resolve_thresholds(legacy)
