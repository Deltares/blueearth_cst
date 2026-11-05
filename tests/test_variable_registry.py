"""The one record per variable, and the two facts that look like one.

R14 `C-57`, Gate 1(b). The registry unified two tables that described the same
three variables in different vocabularies. These tests pin the parts of that
unification that a later reader would otherwise be tempted to "clean up".
"""

from __future__ import annotations

import pytest

from blueearth_cst.climate_analysis import climate_figures
from blueearth_cst.projections.dry_month import ThresholdError, resolve_thresholds
from blueearth_cst.projections.variable_spec import VariableSpec, parse
from blueearth_cst.shared.variable_registry import (
    CLIMATE_VARS,
    VARIABLES,
    Presentation,
    ProjectionSpec,
)

#: The long form, exactly as every shipped `analyze_projections` config spells
#: it. Written out rather than read from the registry, so the equality test
#: below compares the registry against an INDEPENDENT statement of the same
#: values instead of against itself.
LONG_FORM = {
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


def test_the_short_form_resolves_to_exactly_the_long_form():
    """Falsifier 1: assert the parsed STRUCTURE, not that parsing succeeded.

    A short form that silently resolved to some default would pass any smoke
    test — it produces a valid spec map, just not the right one. Comparing
    against the long form written out above is what makes this an assertion
    about values.
    """
    assert parse({"precip": None, "temp": None}) == parse(LONG_FORM)


def test_the_long_form_still_works_and_still_wins():
    """The registry is a default, not a wall.

    No `RETIRED_KEYS` row covers `variables:`, the v2 probe fixture uses the
    long form, and a variable the registry has never heard of has nowhere else
    to be declared. A registry that forced its own values would make the last
    case impossible.
    """
    override = {
        "precip": {
            "source": "precip",
            "canonical": "rate",
            "units": "m/day",  # deliberately NOT the registry's mm/day
            "change": "relative",
        }
    }
    assert parse(override)["precip"].units == "m/day"


def test_a_variable_outside_the_registry_can_still_be_declared():
    """The escape hatch that makes a Python registry acceptable at all.

    Gate 1(a) chose Python over YAML on the grounds that a project never needs
    to edit the registry, and that argument holds only while this passes.
    """
    spec = parse(
        {
            "runoff": {
                "source": "runoff",
                "canonical": "rate",
                "units": "mm/day",
                "change": "relative",
            }
        }
    )
    assert spec["runoff"].change == "relative"


def test_an_unknown_short_form_variable_refuses_with_both_remedies():
    with pytest.raises(ValueError) as excinfo:
        parse({"runoff": None})
    message = str(excinfo.value)
    assert "not in it" in message
    assert "variable_registry.py" in message
    assert "declare it in full" in message


def test_a_registered_variable_with_no_projection_spec_refuses_differently():
    """`pet` is IN the registry, so "add it to the registry" would be false.

    Two faults need two remedies. Telling a user to add a variable that is
    already there sends them to a file where they will find it and conclude the
    error is wrong about something else.
    """
    with pytest.raises(ValueError) as excinfo:
        parse({"pet": None})
    message = str(excinfo.value)
    assert "IS in the variable registry" in message
    assert "does not difference" in message
    assert "variable_registry.py" not in message


def test_the_figure_unit_and_the_canonical_units_are_not_the_same_fact():
    """The trap Gate 1 found: two spellings of one word, two different facts.

    `precip` is drawn as a yearly TOTAL in mm and stored as a monthly mean RATE
    in mm/day. Collapsing them — the obvious cleanup on seeing `unit` beside
    `units` — mislabels every figure axis or every change factor depending on
    which direction the collapse went, and nothing downstream raises either way.
    """
    entry = VARIABLES["precip"]
    assert entry.presentation.unit == "mm"
    assert entry.projections.units == "mm/day"
    assert entry.presentation.unit != entry.projections.units


def test_climate_vars_is_unchanged_in_value_and_in_order():
    """Existing consumers keep working, including their display order.

    `plot_map_forcing` selects the canonical climate set with
    `list(CLIMATE_VARS)`, so a dict that merely holds the same entries is not
    sufficient.
    """
    assert list(CLIMATE_VARS) == ["precip", "temp", "pet"]
    assert CLIMATE_VARS["precip"] == {
        "label": "precipitation",
        "unit": "mm",
        "how": "sum",
        "style": "precip",
    }
    assert CLIMATE_VARS["temp"]["how"] == "mean"
    assert CLIMATE_VARS["pet"]["style"] == "pet"


def test_the_figures_module_still_exports_the_name_it_always_did():
    """The re-export is what keeps the 22 sites out of this phase's diff."""
    assert climate_figures.CLIMATE_VARS is CLIMATE_VARS


@pytest.mark.parametrize("name", sorted(VARIABLES))
def test_every_entry_is_well_formed(name):
    """Both groups validated at the registry, not only where they are read.

    A malformed entry would otherwise surface as a config error blaming the
    project's `variables:` block for a value the project never wrote.
    """
    entry = VARIABLES[name]
    assert isinstance(entry.presentation, Presentation)
    assert entry.presentation.how in {"sum", "mean"}
    assert entry.presentation.label and entry.presentation.unit

    if entry.projections is None:
        return
    assert isinstance(entry.projections, ProjectionSpec)
    assert entry.projections.canonical in {"rate", "state"}
    assert entry.projections.change in {"relative", "absolute"}
    assert entry.projections.source and entry.projections.units


# ---------------------------------------------------------------------------
# `C-64` — the relative-change threshold is per-variable registry metadata
# ---------------------------------------------------------------------------


def test_the_shipped_threshold_moved_without_changing_value():
    """`dry_month.DEFAULT_MIN_REFERENCE = {"precip": 0.1}` lived in code.

    The row is a MOVE and the brief says a changed number is a defect rather
    than a row landing, so the value is asserted here and not merely its
    presence.
    """
    assert VARIABLES["precip"].projections.min_denominator == 0.1
    assert VARIABLES["temp"].projections.min_denominator is None


def test_every_shipped_config_still_gets_its_threshold():
    """The regression that would have broken all four projects at once.

    Every shipped config declares `precip` in the LONG form and none names a
    threshold. Without the registry fallback in `_threshold`, `C-64` would take
    `DEFAULT_MIN_REFERENCE` away and leave nothing behind, and WF2 would refuse
    to build a DAG for every existing project on the next run.
    """
    assert resolve_thresholds(parse(LONG_FORM)) == {"precip": 0.1}


def test_the_short_form_and_the_long_form_resolve_the_same_threshold():
    assert resolve_thresholds(parse({"precip": None, "temp": None})) == {"precip": 0.1}


def test_a_declared_threshold_beats_the_registry():
    """The precedence `C-64` introduces, and the quiet failure if reversed.

    Before this row there was no contest: the config was the only surface. Now
    both a registry value and a declared value can exist, and the wrong order
    hands a project the shipped number in place of the one it deliberately set
    — a plausible result rather than an error, which nothing downstream would
    question.

    Resolved at PARSE, in `variable_spec._threshold`, and asserted on the spec
    as well as through the resolver: since `C-66` there is no second place where
    a threshold could be overridden, so if this precedence is wrong it is wrong
    everywhere at once.
    """
    declared = dict(LONG_FORM["precip"], min_denominator=0.5)
    spec = parse({"precip": declared})

    assert spec["precip"].min_denominator == 0.5
    assert resolve_thresholds(spec) == {"precip": 0.5}


def test_a_relative_variable_outside_the_shipped_set_is_configurable():
    """Falsifier 2, at the level a unit test can reach it.

    This configuration was IMPOSSIBLE to express before P1c: `relative_change:`
    is refused wholesale by `RETIRED_KEYS`, so a v2 project could declare a
    non-`precip` relative variable and had nowhere to put its threshold. If this
    were still impossible after the phase, `C-66` would have removed a
    capability rather than moved it.
    """
    spec = parse(
        {
            "runoff": {
                "source": "runoff",
                "canonical": "rate",
                "units": "mm/day",
                "change": "relative",
                "min_denominator": 0.05,
            }
        }
    )
    assert resolve_thresholds(spec) == {"runoff": 0.05}


def test_a_relative_variable_with_no_threshold_anywhere_still_refuses():
    """The refusal must survive the move, or the move removed a guard.

    Borrowing precipitation's 0.1 mm/day for an unrelated quantity in unrelated
    units produces a number, and not a meaningful one — which is why this raises
    rather than defaulting.
    """
    spec = parse(
        {
            "runoff": {
                "source": "runoff",
                "canonical": "rate",
                "units": "mm/day",
                "change": "relative",
            }
        }
    )
    with pytest.raises(ThresholdError) as excinfo:
        resolve_thresholds(spec)
    message = str(excinfo.value)
    assert "runoff" in message
    # It must name a surface that EXISTS. The message used to say
    # `relative_change.min_reference`, which `RETIRED_KEYS` refuses and `C-66`
    # deletes — advice that sends a user to write a key the loader rejects.
    assert "relative_change" not in message
    assert "variables.<name>.min_denominator" in message


@pytest.mark.parametrize("bad", [0, -1, "wide", [0.1]])
def test_a_threshold_that_cannot_guard_a_denominator_refuses(bad):
    """Zero admits the division the guard exists to prevent."""
    with pytest.raises(ValueError, match="min_denominator"):
        parse({"precip": dict(LONG_FORM["precip"], min_denominator=bad)})


def test_the_threshold_survives_the_snakemake_params_boundary():
    """`analyze_projections.smk` flattens the spec to a list to cross it.

    `derive_change_factors` rebuilds it with `VariableSpec(*fields)` and
    `resolve_thresholds` may see either shape. A field inserted anywhere but the
    END shifts every value after it, and the failure is silent: `change` reads
    as some other string, compares unequal to "relative", and the threshold
    check is skipped for every variable rather than raising.
    """
    spec = parse(LONG_FORM)
    flattened = {name: list(value) for name, value in spec.items()}

    assert resolve_thresholds(flattened) == resolve_thresholds(spec)
    assert VariableSpec._fields[-1] == "min_denominator"
    assert VariableSpec._fields.index("change") == 4
