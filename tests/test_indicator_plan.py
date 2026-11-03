"""Indicator requests and the v2 indicator-plan seams, tested without an experiment.

Planning, reduction, publication and reading over a real retained experiment
live in `test_indicator_plan_v2.py`; this module keeps the request-level rules and
the unit seams that need no experiment at all.
"""

import pytest

from blueearth_cst.experiment.content_identity import content_sha256, identity_segment
from blueearth_cst.experiment.indicator_plan import indicator_request
from blueearth_cst.experiment.indicator_sets import default_calculations
from tests._v2_experiment import FIXTURE_ENVIRONMENT

SIMULATION_ID = "a" * 64
GWR_RESPONSES = {"variables": [{"variable": "gwr", "locations": ["9"]}]}


def fixture_declaration():
    """The real shipped D4 declaration; new requests may carry nothing else."""
    from blueearth_cst.experiment import return_level_validation

    return return_level_validation.build_declaration()


def test_new_requests_refuse_the_legacy_validation_record():
    """D6: legacy acceptance is read compatibility, not a write escape hatch."""
    with pytest.raises(ValueError, match="readable, not writable"):
        indicator_request(
            SIMULATION_ID,
            default_calculations(["gwr"]),
            "YS-JAN",
            FIXTURE_ENVIRONMENT,
            {"status": "provisional_operational", "benchmark": "not assessed"},
            GWR_RESPONSES,
        )


def test_new_requests_refuse_an_unsupported_declaration():
    from blueearth_cst.experiment import return_level_validation

    altered = dict(fixture_declaration(), screening_policy_validated=True)
    with pytest.raises(return_level_validation.ReturnLevelValidationError):
        indicator_request(
            SIMULATION_ID,
            default_calculations(["gwr"]),
            "YS-JAN",
            FIXTURE_ENVIRONMENT,
            altered,
            GWR_RESPONSES,
        )


def test_indicator_environment_records_the_estimator_source():
    """D5: the resolver binds D7's verified source hashes, not just versions."""
    from blueearth_cst.experiment import gev_lmoments
    from blueearth_cst.experiment.indicator_plan import resolve_indicator_environment

    observed = resolve_indicator_environment()
    estimator = observed["return_level_estimator"]
    assert estimator["estimator_id"] == "gf15-lmoments-c/1"
    assert estimator["version"] == gev_lmoments.DEPENDENCY_VERSION
    assert estimator["source_sha256"] == gev_lmoments.SOURCE_SHA256
    assert "lmoments3" in observed["packages"] or any(
        "lmoments3" in key for key in observed["packages"]
    )


@pytest.mark.parametrize(
    "preexisting",
    [
        "absent",
        "empty-data",
        "unknown-dir",
        "partial-data",
        "engine-file",
        "data-file",
        "legacy-data",
        "aliased-data",
    ],
)
def test_indicator_set_v2_publishes_paired_paths_and_headers(
    tmp_path, monkeypatch, preexisting
):
    """The v2 marker and user-facing tables use one short-ID identity."""
    import csv
    from dataclasses import asdict

    from blueearth_cst.experiment import indicator_plan
    from blueearth_cst.experiment.indicator_registry import declarations
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    declaration = indicator_plan._plain(asdict(declarations(("gwr",))[0]))
    environment = {"packages": {"fixture": "1"}}
    simulation = {"schema_version": "simulation/3", "simulation_id": "a" * 64}
    inventory = {
        "response_inventory_sha256": "b" * 64,
        "response_identity_sha256": "c" * 64,
        "series": [{"run_id": "run_01", "variable": "gwr", "location_id": "9"}],
    }
    inventory_path = tmp_path / "_engine/response_inventory.json"
    inventory_path.parent.mkdir()
    indicator_plan.atomic_record(inventory_path, inventory)
    simulation["response_inventory"] = file_reference(
        inventory_path, "project_root", tmp_path.parent.parent
    )
    projection = {
        "simulation_id": simulation["simulation_id"],
        "response_identity_sha256": inventory["response_identity_sha256"],
        "indicator_definition_sha256": content_sha256(
            {
                "declarations": [declaration],
                "unavailable_calculations": [],
                "unavailable_results": [],
            }
        ),
        "indicator_environment_sha256": content_sha256(environment),
        "groups": {"": ["run_01"]},
        "run_groups": [
            {"run_group_id": "run_01", "grain": "run", "run_id": "run_01"},
            {"run_group_id": "group_01", "grain": "bundle", "run_id": "run_01"},
        ],
        "schema_version": "indicator-set-identity/5",
    }
    plan = {
        "schema_version": "indicator-request/5",
        "indicator_set_id": content_sha256(projection),
        "request": {
            "calculations": default_calculations(["gwr"]),
            "unavailable_calculations": [],
            "unavailable_results": [],
            "declarations": [declaration],
            "indicator_environment": environment,
            "return_level_validation": {"schema_version": "fixture/1"},
        },
        "identity_projection": projection,
        "indicator_definition": {
            "declarations": [declaration],
            "unavailable_calculations": [],
            "unavailable_results": [],
        },
        "indicator_definition_sha256": projection["indicator_definition_sha256"],
        "indicator_environment_sha256": projection["indicator_environment_sha256"],
        "response_inventory_sha256": inventory["response_inventory_sha256"],
        "groups": projection["groups"],
        "bundle_run_group_ids": {"": "group_01"},
        "units": projection["run_groups"],
        "expected_result_keys": [
            [declaration["name"], "9", unit] for unit in ("group_01", "run_01")
        ],
    }
    monkeypatch.setattr(
        indicator_plan, "check_live_indicator_environment", lambda _: None
    )
    monkeypatch.setattr(indicator_plan, "read_simulation_v2", lambda _: simulation)
    monkeypatch.setattr(
        indicator_plan, "read_response_inventory_v2", lambda _: inventory
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.simulation_record.read_simulation_v2",
        lambda _: simulation,
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.response_inventory.read_response_inventory_v2",
        lambda _: inventory,
    )
    monkeypatch.setattr(
        indicator_plan,
        "reduce_indicator_plan",
        lambda *_: (
            {
                "gwr": [
                    {
                        "indicator": declaration["name"],
                        "location": "9",
                        "run_group_id": unit,
                        "value": "1",
                        "status": "available",
                        "reason": "",
                    }
                    for unit in ("run_01", "group_01")
                ]
            },
            (),
            [],
        ),
    )
    monkeypatch.setattr(
        indicator_plan.return_level_validation,
        "load_report_bytes",
        lambda: b"fixture benchmark",
    )
    monkeypatch.setattr(
        indicator_plan.return_level_validation,
        "verify_declaration",
        lambda *_: None,
    )
    short = identity_segment(plan["indicator_set_id"], "indicator_set_id")
    engine = tmp_path / "_engine/indicator_sets" / short
    result = engine / "data"
    if preexisting == "empty-data":
        result.mkdir(parents=True)
    elif preexisting == "unknown-dir":
        (engine / "unknown").mkdir(parents=True)
    elif preexisting == "partial-data":
        result.mkdir(parents=True)
        (result / "group_members.csv").write_bytes(b"partial")
    elif preexisting == "engine-file":
        engine.parent.mkdir(parents=True)
        engine.write_bytes(b"foreign")
    elif preexisting == "data-file":
        engine.mkdir(parents=True)
        result.write_bytes(b"foreign")
    elif preexisting == "legacy-data":
        legacy = tmp_path / "results/indicator_sets" / short
        legacy.mkdir(parents=True)
        (legacy / "group_members.csv").write_bytes(b"partial")
    elif preexisting == "aliased-data":
        external = tmp_path / "external"
        external.mkdir()
        engine.mkdir(parents=True)
        result.symlink_to(external, target_is_directory=True)
    if preexisting not in {"absent", "empty-data"}:
        monkeypatch.setattr(
            indicator_plan,
            "reduce_indicator_plan",
            lambda *_: pytest.fail("refuse before reduction"),
        )
        with pytest.raises(
            indicator_plan.ImmutableIndicatorSetError, match="partial|aliased"
        ):
            indicator_plan._publish_indicator_set_v2(tmp_path, plan)
        assert not (engine / "indicators.json").exists()
        return
    marker = indicator_plan._publish_indicator_set_v2(tmp_path, plan)
    assert marker["schema_version"] == "indicator-set/6"
    assert engine.joinpath("indicators.json").is_file()
    assert result.joinpath("group_members.csv").is_file()
    assert marker["group_members"]["path"] == (
        f"_engine/indicator_sets/{short}/data/group_members.csv"
    )
    assert [item["file"]["path"] for item in marker["tables"]] == [
        f"_engine/indicator_sets/{short}/data/gwr_run_indicators.csv",
        f"_engine/indicator_sets/{short}/data/gwr_group_indicators.csv",
    ]
    with result.joinpath("group_members.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert csv.DictReader(handle).fieldnames == ["group_id", "run_id"]
    with result.joinpath("gwr_run_indicators.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert csv.DictReader(handle).fieldnames == [
            "indicator",
            "location",
            "run_id",
            "value",
            "status",
            "reason",
        ]
    assert (
        indicator_plan.read_indicator_set(tmp_path, engine / "indicators.json")
        == marker
    )


def test_v2_indicator_plan_reads_collection_from_intent(tmp_path, monkeypatch):
    from blueearth_cst.experiment import indicator_plan
    from blueearth_cst.experiment.content_identity import atomic_record
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    root = tmp_path / "experiments/experiment"
    (root / "_engine").mkdir(parents=True)
    (root / "_engine/simulation.json").write_bytes(b"{}\n")
    request = {
        "schema_version": "indicator-request/5",
        "simulation_schema_version": "simulation/3",
        "simulation_id": "a" * 64,
        "tokens": ["gwr"],
        "calculations": default_calculations(["gwr"]),
        "water_year_anchor": "YS-JAN",
        "indicator_environment": {},
        "return_level_validation": {},
    }
    collection = {
        "collection_id": "b" * 64,
        "collection_revision": "c" * 64,
        "manifest": {"path": "scenarios/_engine/collections/b/collection.json"},
    }
    intent = root / "_engine/simulation_intent.json"
    inventory = root / "_engine/response_inventory.json"
    atomic_record(
        intent,
        {"collection": collection, "documents": {"response_request": GWR_RESPONSES}},
    )
    atomic_record(inventory, {})
    monkeypatch.setattr(
        indicator_plan,
        "read_simulation_v2",
        lambda path: {
            "schema_version": "simulation/3",
            "simulation_id": "a" * 64,
            "intent": file_reference(intent, "project_root", tmp_path),
            "response_inventory": file_reference(inventory, "project_root", tmp_path),
        },
    )
    monkeypatch.setattr(
        indicator_plan, "indicator_request", lambda *args, **kwargs: dict(request)
    )
    seen = {}

    def stop_after_collection(simulation):
        seen.update(simulation)
        raise RuntimeError("stop after collection selection")

    monkeypatch.setattr(indicator_plan, "_collection", stop_after_collection)

    with pytest.raises(RuntimeError, match="stop after collection"):
        indicator_plan.build_indicator_plan(root, request)

    assert seen["collection"] == collection


def test_v2_scenario_lookup_preserves_member_ancestry():
    from blueearth_cst.experiment.indicator_plan import _v2_scenario_rows

    rows = _v2_scenario_rows(
        [
            {
                "run_id": "run_01",
                "evaluate": "true",
                "type": "stochastic",
                "rlz": "1",
                "st_id": "",
            },
            {
                "run_id": "run_02",
                "evaluate": "true",
                "type": "stochastic",
                "rlz": "1",
                "st_id": "1",
            },
        ]
    )

    assert [(row.run_id, row.derived_from, dict(row.payload)) for row in rows] == [
        ("run_01", "", {"rlz": "1", "st_id": ""}),
        ("run_02", "run_01", {"rlz": "1", "st_id": "1"}),
    ]


def test_v2_native_run_references_resolve_from_inventory(tmp_path):
    from blueearth_cst.experiment.indicator_plan import _native_runs_v2
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    csv_path = tmp_path / "output.csv"
    toml_path = tmp_path / "run.toml"
    csv_path.write_text("time,Q_1\n2046-01-02,1\n", encoding="utf-8")
    toml_path.write_text("[time]\n", encoding="utf-8")
    inventory = {
        "artifacts": [
            {
                "run_id": "run_01",
                "file": file_reference(csv_path, "experiment_root", tmp_path),
            }
        ],
        "series": [
            {
                "run_id": "run_01",
                "native_selector": {
                    "toml": file_reference(toml_path, "experiment_root", tmp_path)
                },
            }
        ],
    }

    native_runs = _native_runs_v2(tmp_path, inventory)

    assert native_runs["run_01"].csv_path == csv_path
    assert native_runs["run_01"].toml_path == toml_path
    assert native_runs["run_01"].temporal_path is None


def test_v2_run_groups_replace_legacy_unit_keys():
    from blueearth_cst.experiment.indicator_plan import _v2_run_groups

    assert _v2_run_groups(
        [{"unit_id": "run_01", "grain": "run", "member_run_id": "run_01"}]
    ) == [{"run_group_id": "run_01", "grain": "run", "run_id": "run_01"}]
