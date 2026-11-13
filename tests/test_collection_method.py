"""One method helper for every collection consumer."""

import pytest

from blueearth_cst.experiment.collection_method import collection_method, is_delta


@pytest.mark.parametrize(
    ("marker", "method"),
    [
        ({"schema_version": "scenario-collection/3"}, "stochastic"),
        ({"schema_version": "delta-scenario-collection/1"}, "delta_change"),
        (
            {"schema_version": "anything/9", "scenario_type": "delta_change"},
            "delta_change",
        ),
        ({"schema_version": "anything/9", "scenario_type": "stochastic"}, "stochastic"),
    ],
)
def test_method_from_scenario_type_or_retained_schema(marker, method):
    assert collection_method(marker) == method
    assert is_delta(marker) == (method == "delta_change")


@pytest.mark.parametrize(
    "marker",
    [{"schema_version": "unknown/1"}, {"scenario_type": "other"}, [], {}],
)
def test_unknown_collections_are_refused(marker):
    with pytest.raises(ValueError):
        collection_method(marker)


def test_is_delta_treats_unrecognised_markers_as_not_delta():
    assert not is_delta({})
    assert not is_delta({"schema_version": "unknown/1"})
