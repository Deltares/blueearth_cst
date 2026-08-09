"""Resolving the names hydromt_wflow gave the user's gauges (shared/gauges.py).

The regression these pin, observed 2026-08-01 on a real basin: hydromt_wflow
renames the gauges basename ``output_locations`` -> ``output-locations``
(``wflow_base.py``, ``.replace("_", "-")``), three readers derived the name from
the FILENAME instead, and because every lookup was a membership test used as a
guard the user's four gauges vanished in silence — absent from the basin map, no
gauge hydrographs, no signature plots, and an EMPTY performance_metrics.csv,
while wflow's own output.csv carried all four correctly.
"""

import warnings

import pytest

from blueearth_cst.shared.gauges import (
    gauges_layer_name,
    gauges_variable_name,
    hydromt_basename,
    is_unset,
)

#: What the real model held, verbatim (C:/TESTS/CST/gabon_0108).
GEOMS = {"basins", "gauges_output-locations", "outlets", "rivers"}
RESULTS = {"Q_outlets", "Q_gauges_output-locations", "P_gauges_output-locations"}
CONFIGURED = "C:/TESTS/CST/observations/gabon/output_locations.csv"


# --- the unset sentinels ---------------------------------------------------


@pytest.mark.parametrize("unset", [None, "None"])
def test_both_unset_spellings_resolve_to_nothing(unset):
    """YAML null and the legacy 'None' string alike (R07 O-08)."""
    assert is_unset(unset)
    assert hydromt_basename(unset) is None
    assert gauges_layer_name(GEOMS, unset) is None
    assert gauges_variable_name(RESULTS, unset) is None


def test_an_unset_config_does_not_warn():
    """Not configuring gauges is normal, not a fault."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert gauges_layer_name(GEOMS, None) is None


def test_an_unset_config_never_adopts_the_outlets():
    """`outlets` is the model's own layer; a config without output_locations
    must not silently pick it up as if the user had supplied gauges."""
    assert gauges_layer_name({"outlets", "basins"}, None) is None


# --- the underscore/hyphen regression --------------------------------------


def test_hydromt_basename_mirrors_the_underscore_to_hyphen_rule():
    assert hydromt_basename("a/b/output_locations.csv") == "output-locations"
    assert hydromt_basename("gauges.geojson") == "gauges"


def test_the_layer_resolves_despite_the_rename():
    """The exact failure: filename says output_locations, model says
    output-locations."""
    assert gauges_layer_name(GEOMS, CONFIGURED) == "gauges_output-locations"


def test_the_output_variable_resolves_despite_the_rename():
    assert gauges_variable_name(RESULTS, CONFIGURED) == "Q_gauges_output-locations"
    assert gauges_variable_name(RESULTS, CONFIGURED, "P") == "P_gauges_output-locations"


def test_resolution_does_not_depend_on_the_rename_holding():
    """Discovery, not mirroring: setup_gauges accepts an explicit basename= and
    upstream could change the rule, so a layer that matches NEITHER derivation
    still resolves when it is the only gauge layer present."""
    geoms = {"basins", "outlets", "gauges_whatever-upstream-chose"}
    assert gauges_layer_name(geoms, CONFIGURED) == "gauges_whatever-upstream-chose"


def test_the_configured_layer_wins_when_several_exist():
    """Discovery must not pick arbitrarily when the exact name is available."""
    geoms = GEOMS | {"gauges_other-network"}
    assert gauges_layer_name(geoms, CONFIGURED) == "gauges_output-locations"


# --- the silence is the other half of the bug ------------------------------


def test_a_configured_but_missing_layer_warns_loudly():
    """The lesson was the silence, not the name. A skip here is what cost a
    whole run's evaluation outputs."""
    with pytest.warns(UserWarning, match="could not be resolved"):
        assert gauges_layer_name({"basins", "outlets"}, CONFIGURED) is None


def test_the_warning_names_what_it_looked_for_and_what_exists():
    """A warning a reader cannot act on is barely better than silence."""
    with pytest.warns(UserWarning) as caught:
        gauges_layer_name({"basins", "outlets"}, CONFIGURED)
    message = str(caught[0].message)
    assert "gauges_output-locations" in message
    assert "no gauge entries at all" in message
    assert "output_locations" in message


def test_an_ambiguous_set_warns_rather_than_guessing():
    with pytest.warns(UserWarning, match="could not be resolved"):
        assert (
            gauges_layer_name(
                {"gauges_one-network", "gauges_other-network"}, "a/b/absent.csv"
            )
            is None
        )


def test_a_missing_variable_warns_too():
    with pytest.warns(UserWarning, match="Q variable"):
        assert gauges_variable_name({"Q_outlets"}, CONFIGURED) is None


# --- the wflow_id convention -----------------------------------------------


def test_low_gauge_ids_warn_and_are_returned():
    """A CONVENTION, not a constraint -- so it warns and names the offenders
    rather than rejecting a dataset that works."""
    from blueearth_cst.shared.gauges import warn_if_low_gauge_ids

    with pytest.warns(UserWarning, match="below 100"):
        low = warn_if_low_gauge_ids([1, 2, 3, 4], "output_locations.csv")
    assert low == [1, 2, 3, 4]


def test_the_warning_says_what_renumbering_costs():
    """Renumbering without a rebuild silently desyncs the ids from the model,
    so the warning has to say so or it invites exactly that mistake."""
    from blueearth_cst.shared.gauges import warn_if_low_gauge_ids

    with pytest.warns(UserWarning) as caught:
        warn_if_low_gauge_ids([1], "obs/output_locations.csv")
    message = str(caught[0].message)
    assert "obs/output_locations.csv" in message
    assert "rule 1.05" in message
    assert "timeseries column" in message


def test_conforming_ids_are_silent():
    from blueearth_cst.shared.gauges import warn_if_low_gauge_ids

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_if_low_gauge_ids([100, 101, 102], "x.csv") == []


def test_non_integer_ids_are_ignored_rather_than_raising():
    """An advisory read of user data must not become a schema check."""
    from blueearth_cst.shared.gauges import warn_if_low_gauge_ids

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_if_low_gauge_ids([None, "", "abc", 101], "x.csv") == []
