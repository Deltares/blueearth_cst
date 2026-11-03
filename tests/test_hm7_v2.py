"""HM-7 independently verifies scoped indicator tables and explicit group members."""

from copy import deepcopy

import pandas as pd
import pytest

from blueearth_cst.shared.interchange_contracts import validate_hm7_v2


def contract():
    runs = ["run_01", "run_02", "run_03"]
    memberships = [
        ("group_01", "run_01"),
        ("group_01", "run_02"),
        ("group_02", "run_03"),
    ]
    lookup = pd.DataFrame(memberships, columns=["group_id", "run_id"])
    manifest = {
        "schema_version": "indicator-set/3",
        "run_groups": [
            {"run_group_id": g, "grain": "bundle", "run_id": r} for g, r in memberships
        ],
        "declarations": [
            {"name": "q_mean", "grain": "run", "required_responses": {"variable": "q"}},
            {
                "name": "q_rl",
                "grain": "bundle",
                "required_responses": {"variable": "q"},
            },
        ],
        "tables": [{"token": "q", "scope": scope} for scope in ("run", "group")],
    }
    tables = {
        "q_run": pd.DataFrame(
            [("q_mean", "101", r, 1.5) for r in runs],
            columns=["indicator", "location", "run_id", "value"],
        ),
        "q_group": pd.DataFrame(
            [
                (m, "101", g, 2.0)
                for m in ("q_mean", "q_rl")
                for g in ("group_01", "group_02")
            ],
            columns=["indicator", "location", "group_id", "value"],
        ),
    }
    series = [{"variable": "q", "location_id": "101", "run_id": r} for r in runs]
    return manifest, tables, lookup, series, runs


def check(values):
    manifest, tables, lookup, series, runs = values
    return validate_hm7_v2(
        manifest, tables, lookup=lookup, series=series, collection_run_ids=runs
    )


def test_scoped_tables_and_singletons_resolve_independently():
    assert check(contract()) == []
    values = contract()
    values[-1].append("run_04")
    assert check(values) == []
    manifest, tables, lookup, series, runs = contract()
    manifest["declarations"] = manifest["declarations"][1:]
    tables["q_run"] = tables["q_run"].iloc[:0]
    tables["q_group"] = tables["q_group"].query("indicator == 'q_rl'")
    assert check((manifest, tables, lookup, series, runs)) == []


@pytest.mark.parametrize(
    "defect",
    [
        "missing_indicator",
        "duplicate",
        "bad_member",
        "missing_member",
        "scope",
        "invalid_fit",
        "missing_scope",
        "legacy",
    ],
)
def test_corrupted_contract_is_refused(defect):
    values = deepcopy(contract())
    manifest, tables, lookup, series, runs = values
    if defect == "missing_indicator":
        tables["q_group"] = tables["q_group"].iloc[1:]
    elif defect == "duplicate":
        tables["q_run"] = pd.concat([tables["q_run"], tables["q_run"].iloc[:1]])
    elif defect == "bad_member":
        lookup.loc[0, "run_id"] = "run_99"
    elif defect == "missing_member":
        lookup.drop(index=0, inplace=True)
    elif defect == "scope":
        tables["q_group"].loc[0, "group_id"] = "run_01"
    elif defect == "invalid_fit":
        tables["q_group"].loc[tables["q_group"]["indicator"] == "q_rl", "value"] = (
            float("nan")
        )
    elif defect == "missing_scope":
        del tables["q_run"]
    else:
        manifest["schema_version"] = "indicator-set/2"
    assert check(values)
