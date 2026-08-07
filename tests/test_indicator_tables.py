# -*- coding: utf-8 -*-
"""The indicator-table set is derived from `wflow_outvars` (R11 CR-2).

What matters here is not the mapping's contents -- those are a published contract
-- but that the DERIVATION is total and loud: every configured variable yields
exactly one table, and a variable with no token stops the run rather than
silently producing a result set missing a variable the config asked for.
"""

import pytest

from blueearth_cst.shared.indicator_tables import (
    VARIABLE_TOKENS,
    UnknownOutputVariableError,
    indicator_table_filename,
    indicator_tables,
    variable_token,
)


ALL_SIX = list(VARIABLE_TOKENS)


def test_the_seed_config_yields_only_the_discharge_table():
    """`snake_config_model_test.yml` requests one variable, so one table."""
    assert indicator_tables(["river discharge"]) == {"q": "q_indicators.csv"}


def test_the_shipped_template_yields_two():
    assert list(indicator_tables(["river discharge", "actual evapotranspiration"])) == [
        "q",
        "aet",
    ]


@pytest.mark.parametrize("outvar", ALL_SIX)
def test_every_documented_variable_has_a_table(outvar):
    """Six entries, not five -- `precipitation` is one of them."""
    assert indicator_table_filename(outvar).endswith("_indicators.csv")


def test_the_set_follows_config_order_and_collapses_duplicates():
    """Order is stable for a given config; a repeated entry names one table."""
    tables = indicator_tables(["snow", "river discharge", "snow"])
    assert list(tables) == ["snow", "q"]


@pytest.mark.parametrize("empty", [None, []])
def test_no_configured_variables_yields_no_tables(empty):
    assert indicator_tables(empty) == {}


def test_an_unknown_variable_raises_rather_than_being_skipped():
    """The failure mode this prevents: a run whose results silently omit a
    variable the config requested, indistinguishable from never requesting it."""
    with pytest.raises(UnknownOutputVariableError) as excinfo:
        indicator_tables(["river discharge", "sediment yield"])
    message = str(excinfo.value)
    assert "sediment yield" in message
    # The error must say where to fix it -- both places, since the mapping is a
    # published contract and not only code.
    assert "indicator_tables.py" in message
    assert "hydrological-model-seam.md" in message


def test_tokens_are_distinct_so_two_variables_cannot_share_a_table():
    assert len(set(VARIABLE_TOKENS.values())) == len(VARIABLE_TOKENS)


@pytest.mark.parametrize(
    "outvar, token",
    [
        ("precipitation", "precip"),  # not `p` -- naming.md §6 tier 2
        ("actual evapotranspiration", "aet"),  # not `et` -- `pet` is canonical
        ("snow", "snow"),  # not `swe` -- CSDMS says snowpack liquid water
    ],
)
def test_the_three_tokens_that_were_deliberately_not_abbreviated(outvar, token):
    """Each of these had a shorter candidate rejected for a stated reason; a
    silent change here would reintroduce the ambiguity the ruling removed."""
    assert variable_token(outvar) == token
