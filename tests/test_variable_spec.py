"""Variable-spec tests for step 5e-iii (design §5.5). Falsifier K7.

The point of the spec is that "nothing infers anything from a name". These tests
are where that claim is checkable, because the seed config's two variables happen
to be named exactly what the old name-based inference expected — so the fixture
cannot tell a working spec from a working guess.
"""

import pytest

from blueearth_cst.projections.variable_spec import (
    as_digest_component,
    change_kind,
    parse,
    source_names,
)

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


def test_K7_change_semantics_come_from_the_spec_not_the_name():
    """The falsifier: a relative variable NOT named precip must be relative.

    Under the old list form this was differenced as though it were a temperature,
    silently, and the fixture could never show it.
    """
    spec = parse(
        {
            "rainfall": {
                "source": "rainfall",
                "canonical": "rate",
                "units": "mm/day",
                "change": "relative",
            },
        }
    )
    assert change_kind(spec, "rainfall") == "relative"


def test_K7_a_variable_NAMED_precip_can_be_declared_absolute():
    """The inverse, which the name-based guess could never express."""
    spec = parse(
        {
            "precip": {
                "source": "precip",
                "canonical": "rate",
                "units": "mm/day",
                "change": "absolute",
            },
        }
    )
    assert change_kind(spec, "precip") == "absolute"


def test_the_seed_spec_parses_to_the_expected_semantics():
    spec = parse(SEED)
    assert change_kind(spec, "precip") == "relative"
    assert change_kind(spec, "temp") == "absolute"
    assert spec["precip"].canonical == "rate" and spec["temp"].canonical == "state"
    assert spec["precip"].units == "mm/day"


def test_the_old_list_form_raises_naming_the_new_shape():
    """Accepting it silently would leave the name-based guess in place for exactly
    the configs that had not been migrated."""
    with pytest.raises(ValueError, match="pre-5e shape"):
        parse(["precip", "temp"])


def test_the_error_shows_the_migration_for_the_variables_actually_configured():
    with pytest.raises(ValueError) as excinfo:
        parse(["precip", "temp"])
    assert "precip: {source: precip" in str(excinfo.value)


@pytest.mark.parametrize("field", ["source", "canonical", "units", "change"])
def test_every_field_is_required(field):
    body = dict(SEED["precip"])
    body.pop(field)
    with pytest.raises(ValueError, match="missing"):
        parse({"precip": body})


@pytest.mark.parametrize("field,bad", [("canonical", "flux"), ("change", "ratio")])
def test_enumerated_fields_reject_unknown_values(field, bad):
    body = dict(SEED["precip"])
    body[field] = bad
    with pytest.raises(ValueError, match=field):
        parse({"precip": body})


def test_source_names_are_sorted_for_a_stable_digest():
    """Unordered would make the cache key depend on mapping iteration order."""
    spec = parse(
        {
            "temp": SEED["temp"],
            "precip": SEED["precip"],
        }
    )
    assert source_names(spec) == ["precip", "temp"]


def test_digest_component_is_deterministic_across_key_order():
    a = as_digest_component(parse({"precip": SEED["precip"], "temp": SEED["temp"]}))
    b = as_digest_component(parse({"temp": SEED["temp"], "precip": SEED["precip"]}))
    assert a == b


def test_digest_component_moves_when_semantics_change():
    """The spec picks the arithmetic, so it must be part of the cache key."""
    base = parse(SEED)
    flipped = dict(SEED)
    flipped["precip"] = dict(SEED["precip"], change="absolute")
    assert as_digest_component(base) != as_digest_component(parse(flipped))
