# -*- coding: utf-8 -*-
"""The indicator-table set is derived from `wflow_outvars` (R11 CR-2).

What matters here is not the mapping's contents -- those are a published contract
-- but that the DERIVATION is total and loud: every configured variable yields
exactly one table, and a variable with no token stops the run rather than
silently producing a result set missing a variable the config asked for.
"""

import pytest

from blueearth_cst.shared.indicator_tables import (
    BASIN_METRIC_SUFFIXES,
    MIGRATION_NOTE,
    Q_METRIC_SUFFIXES,
    RETIRED_EXPERIMENT_KEYS,
    VARIABLE_TOKENS,
    RetiredConfigKeyError,
    UnknownOutputVariableError,
    basin_metric_name,
    basin_reduction,
    indicator_table_filename,
    indicator_tables,
    output_code,
    q_metric_name,
    refuse_retired_experiment_keys,
    variable_token,
)

ALL_SIX = list(VARIABLE_TOKENS)


def test_the_seed_config_yields_only_the_discharge_table():
    """A config requesting only `river discharge` yields the q table alone."""
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


# --- the metric vocabulary ----------------------------------------------------
# A published contract: these names leave the project tree inside result files,
# so a silent change to one breaks a consumer that cannot see this repo.


def test_the_return_levels_carry_their_return_period():
    """Tpeak/Tlow appeared in no column and no name before R11, so two runs with
    different settings produced identical-looking rows meaning different things."""
    assert q_metric_name("return_level_max") == "q_return_level_10yr_max"
    assert q_metric_name("return_level_7day_min") == "q_return_level_2yr_7day_min"


def test_q95_is_named_p95_because_the_conventional_name_means_the_opposite():
    """Ours is the mean annual 95th percentile, a HIGH flow. Conventional Q95 is
    the flow exceeded 95% of the time — a LOW-flow drought index."""
    assert q_metric_name("q95") == "q_mean_annual_p95"


@pytest.mark.parametrize(
    "statistic", [s for s, (_, cls) in Q_METRIC_SUFFIXES.items() if cls == "A"]
)
def test_class_a_metrics_are_the_ones_linear_in_years(statistic):
    """Class A is per-realization precisely because these average back to the
    pooled value exactly; nothing is lost by emitting the finer grain."""
    assert Q_METRIC_SUFFIXES[statistic][1] == "A"


def test_the_two_gev_fits_are_pooled_only():
    """A per-realization GEV fit over a short record is ill-conditioned."""
    for statistic in ("return_level_max", "return_level_7day_min"):
        assert Q_METRIC_SUFFIXES[statistic][1] == "B"


def test_the_month_selecting_metrics_are_pooled_only():
    """`idxmax()` picks ONE month, so different realizations can pick different
    ones; the month is chosen once from the pooled record."""
    for statistic in ("wetmonth_mean", "drymonth_mean"):
        assert Q_METRIC_SUFFIXES[statistic][1] == "C"


def test_every_metric_name_starts_with_its_variable_token():
    """The invariant `validate_hm7` asserts, since composing the variable into
    the metric is what normalisation would have given for free."""
    for statistic in Q_METRIC_SUFFIXES:
        assert q_metric_name(statistic).startswith("q_")
    for token in BASIN_METRIC_SUFFIXES:
        assert basin_metric_name(token).startswith(f"{token}_")


def test_overland_flow_reduces_with_a_mean_not_a_sum():
    """Q10: it is a volume flow rate, so summing daily values yields a quantity
    in no useful unit. The odd one out, and the defect that ruling fixed."""
    assert basin_reduction("overland_flow") == "mean"
    assert basin_metric_name("overland_flow") == "overland_flow_annual_mean"


@pytest.mark.parametrize("token", ["aet", "gwr", "precip"])
def test_fluxes_keep_their_annual_total_in_mm_per_year(token):
    """A daily sum of a mm Δt⁻¹ flux is a legitimate time-integral. Ruled
    2026-08-08 as scoped to overland flow, so these are deliberately NOT
    rescaled to per-timestep units."""
    assert basin_reduction(token) == "sum"
    assert basin_metric_name(token) == f"{token}_annual_total"


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


def test_groundwater_recharge_borrows_the_code_it_already_had():
    """The other half of the minting rule: where a canonical short name EXISTS,
    take it rather than mint a second one.

    `recharge` was the violation -- `gwr` is what `wflow_outputs.CODES` already
    writes into the TOML and therefore into every run csv header, so the token
    was a spelling the repo did not need. Renamed 2026-08-11; the table is
    `gwr_indicators.csv` and the metric `gwr_annual_total`.
    """
    assert variable_token("groundwater recharge") == "gwr"
    assert output_code("gwr") == "gwr"
    assert indicator_table_filename("groundwater recharge") == "gwr_indicators.csv"
    assert basin_metric_name("gwr") == "gwr_annual_total"


# --- retired config keys (Q7) -------------------------------------------------


def test_a_config_without_retired_keys_is_accepted():
    refuse_retired_experiment_keys({"realizations_num": 2, "stress_test": {}})


def test_a_stale_aggregate_rlz_is_refused_not_ignored():
    """Q7, ruled 2026-08-07. Workflow configs silently ignore keys nothing reads,
    so the alternative to refusing is a user believing a setting is in effect
    while it does nothing at all."""
    with pytest.raises(RetiredConfigKeyError) as excinfo:
        refuse_retired_experiment_keys({"aggregate_rlz": True, "realizations_num": 2})
    message = str(excinfo.value)
    assert "aggregate_rlz" in message
    # The error must state the migration, not merely refuse -- the
    # `variable_spec.parse` precedent.
    assert MIGRATION_NOTE in message
    assert "Delete the line" in message


def test_every_retired_key_is_refused_with_its_own_migration_note():
    """Retirements come from different milestones, so one shared pointer would
    send a reader to the wrong record.

    `Tpeak`/`Tlow` were removed from every config and from the reader on
    2026-08-12 WITHOUT an entry here, which for one commit gave a project
    declaring `Tpeak: 25` exactly the silent no-op this registry exists to
    prevent. Nothing catches that omission automatically -- the removal makes
    the key unread, which is indistinguishable from it never existing -- so the
    check is that every registered key refuses and names where to read about it.
    """
    for key, entry in RETIRED_EXPERIMENT_KEYS.items():
        with pytest.raises(RetiredConfigKeyError) as excinfo:
            refuse_retired_experiment_keys({key: 1, "realizations_num": 2})
        message = str(excinfo.value)
        assert key in message
        assert entry["note"] in message


def test_the_return_period_keys_name_the_constant_that_replaced_them():
    """A refusal that only says 'gone' leaves the user with no way forward. The
    replacement is a toolbox constant, so the error has to say which one."""
    with pytest.raises(RetiredConfigKeyError) as excinfo:
        refuse_retired_experiment_keys({"Tpeak": 25, "Tlow": 5})
    message = str(excinfo.value)
    assert "2 retired key(s)" in message
    assert "RETURN_PERIOD_PEAK_YR" in message
    assert "RETURN_PERIOD_LOW_YR" in message


def test_the_refusal_fires_on_the_value_being_present_not_truthy():
    """`aggregate_rlz: false` is just as stale as `true`; both mean the user
    thinks the flag still does something."""
    with pytest.raises(RetiredConfigKeyError):
        refuse_retired_experiment_keys({"aggregate_rlz": False})


@pytest.mark.parametrize("not_a_mapping", [None, [], "aggregate_rlz"])
def test_a_non_mapping_section_is_left_to_the_schema_check(not_a_mapping):
    """This guard has one job. A malformed section is someone else's error, and
    raising the wrong one here would misdirect."""
    refuse_retired_experiment_keys(not_a_mapping)
