"""The dry-month rule: near-zero reference denominators (design §5.6, ruling A2).

A relative change is undefined-in-practice when the reference is near zero. The
annual product largely avoids it — a year's total is rarely near zero — but the
**monthly** product added at 6a-ii walks straight into it on any basin with a dry
season, where a 0.2 mm/day reference January turns a trivial absolute change into
a four-figure percentage.

The rule (A2, closing OQ-9):

* flag when ``reference < min_denominator`` — **strictly** below; a reference
  exactly at the threshold is not flagged;
* a flagged month emits ``value = NaN`` and
  ``status = "reference_below_threshold"``, and keeps the **absolute** change in
  ``absolute_value``, because the ratio is meaningless while the difference still
  carries information. Dropping both would be a worse answer than the infinity it
  replaces;
* the default is ``precip: 0.1 mm/day`` (≈3 mm/month) — below which a reference
  month is hydrologically negligible for a stress test that perturbs
  precipitation by *percentage*. Deliberately conservative, and revisable by
  measurement without a design change;
* a ``change: relative`` variable outside the shipped set has **no default**: the
  config must supply a threshold and DAG build raises otherwise. Falling back to
  precipitation's 0.1 would apply a rainfall threshold to an unrelated quantity in
  unrelated units.
"""

from __future__ import annotations

from blueearth_cst.projections.variable_spec import VariableSpec

#: `C-64` moved the shipped thresholds into `shared/variable_registry.py`, per
#: variable, beside everything else that is true of that variable. What used to
#: be `DEFAULT_MIN_REFERENCE = {"precip": 0.1}` is now
#: `VARIABLES["precip"].projections.min_denominator`, and the VALUE is unchanged
#: -- only its home. A second table here would be a second record of one fact,
#: which is what D-10.6 and the registry exist to end.
#:
#: The positions below are derived rather than written down. Snakemake's params
#: carry plain data, so `analyze_projections.smk` flattens each `VariableSpec`
#: to a list and this module may receive either shape. Reading `fields[4]` for
#: `change` -- which is what it used to do -- is correct only for as long as
#: nobody adds a field, and a wrong index here does not raise: it reads some
#: other string, compares unequal to "relative", and SKIPS the threshold check
#: for every variable.
_CHANGE_AT = VariableSpec._fields.index("change")
_MIN_DENOMINATOR_AT = VariableSpec._fields.index("min_denominator")

#: A2. A basin with a genuine dry season produces about one season of
#: structurally flagged months as its NORMAL state; more than that means the
#: monthly relative product is undefined for over a quarter of the year, which a
#: reader should be told at combination level rather than by counting footnotes.
DEFAULT_MAX_FLAGGED_MONTHS = 3

#: The status a flagged month carries into the change-factor tables.
FLAGGED_STATUS = "reference_below_threshold"
OK_STATUS = "ok"


class ThresholdError(ValueError):
    """A relative variable whose near-zero threshold is unknown."""


def resolve_thresholds(variable_spec):
    """Threshold per RELATIVE variable, or raise naming the ones still unset.

    ``variable_spec`` maps name to a spec exposing ``.change``, or to the plain
    field list Snakemake params carry.

    **One source since `C-66`.** There used to be a second argument carrying
    ``relative_change.min_reference`` straight from the config, which made this
    function the place where config beat default. That section no longer exists,
    and the precedence moved with it: ``variable_spec._threshold`` resolves a
    declared ``variables.<name>.min_denominator`` over the registry's value
    before the spec is built, so by the time it arrives here the question has
    been answered. Keeping the parameter would have left a second, quieter way
    to override a threshold with no config path able to reach it.
    """
    thresholds, missing = {}, []
    for name, spec in dict(variable_spec).items():
        change = _field(spec, "change", _CHANGE_AT, default="absolute")
        if change != "relative":
            continue
        declared = _field(spec, "min_denominator", _MIN_DENOMINATOR_AT)
        if declared is not None:
            thresholds[name] = float(declared)
        else:
            missing.append(name)
    if missing:
        raise ThresholdError(
            f"no `min_denominator` for {sorted(missing)}: declared "
            "`change: relative` with no value in the variable registry and none "
            "declared. Set one in that variable's own canonical units, either "
            "in `variables.<name>.min_denominator` or in "
            "`blueearth_cst/shared/variable_registry.py`.\n"
            "  Refusing to borrow another variable's threshold: precipitation's "
            "0.1 mm/day applied to an unrelated quantity in unrelated units is "
            "a number, and it is not a meaningful one."
        )
    return thresholds


def _field(spec, name, index, default=None):
    """One field of a spec that may be a ``VariableSpec`` or a plain list.

    Snakemake's params carry plain data, so both shapes reach this module. The
    attribute is tried first and the position is the fallback; ``index`` is
    derived from ``VariableSpec._fields`` above rather than written down.
    """
    value = getattr(spec, name, None)
    if value is not None:
        return value
    if isinstance(spec, (list, tuple)):
        return spec[index] if len(spec) > index else default
    return default


def is_flagged(reference_value, threshold) -> bool:
    """``reference < threshold`` — STRICT, so exactly-at-threshold is not flagged.

    Its own function because the boundary is the part that goes wrong, and
    because strictness is a decision the design makes explicitly rather than an
    accident of whichever operator got typed.
    """
    if threshold is None:
        return False
    try:
        return bool(reference_value < threshold)
    except TypeError:
        return False


def combination_is_flagged(
    n_flagged_months, max_flagged=DEFAULT_MAX_FLAGGED_MONTHS
) -> bool:
    """``count > max`` — strict again; exactly ``max`` does not flag."""
    return bool(n_flagged_months > max_flagged)
