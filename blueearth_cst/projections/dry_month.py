"""The dry-month rule: near-zero reference denominators (design §5.6, ruling A2).

A relative change is undefined-in-practice when the reference is near zero. The
annual product largely avoids it — a year's total is rarely near zero — but the
**monthly** product added at 6a-ii walks straight into it on any basin with a dry
season, where a 0.2 mm/day reference January turns a trivial absolute change into
a four-figure percentage.

The rule (A2, closing OQ-9):

* flag when ``reference < min_reference`` — **strictly** below; a reference
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

#: A2, closing OQ-9. Keyed by the variable's canonical name, in its canonical
#: units. Only variables shipped in the default configs get a default.
DEFAULT_MIN_REFERENCE = {"precip": 0.1}

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


def resolve_thresholds(variable_spec, configured=None):
    """Threshold per RELATIVE variable, or raise naming the ones still unset.

    ``variable_spec`` maps name to a spec exposing ``.change``, or to the plain
    field list Snakemake params carry.
    """
    configured = dict(configured or {})
    thresholds, missing = {}, []
    for name, spec in dict(variable_spec).items():
        change = getattr(spec, "change", None)
        if change is None:
            fields = list(spec)
            change = fields[4] if len(fields) > 4 else "absolute"
        if change != "relative":
            continue
        if name in configured:
            thresholds[name] = float(configured[name])
        elif name in DEFAULT_MIN_REFERENCE:
            thresholds[name] = float(DEFAULT_MIN_REFERENCE[name])
        else:
            missing.append(name)
    if missing:
        raise ThresholdError(
            "relative_change.min_reference is required for "
            f"{sorted(missing)}: declared `change: relative` with no shipped "
            "default. Set a threshold in that variable's own canonical units. "
            f"Refusing to fall back to {DEFAULT_MIN_REFERENCE!r}, which would "
            "apply a precipitation threshold to an unrelated quantity."
        )
    return thresholds


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
