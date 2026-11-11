"""Compact response inventories retain complete evidence without native CSVs."""

import shutil
from dataclasses import replace

import pytest

from blueearth_cst.experiment import response_inventory
from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    content_sha256,
)
from blueearth_cst.experiment.response_inventory import (
    MissingResponseRequirement,
    ResponseReaderUnavailable,
    build_response_inventory_v2,
    publish_response_inventory_v2,
    read_response_inventory_v2,
)
from blueearth_cst.shared.response_tables import publish_response_tables
from tests._v2_experiment import (
    RUN_IDS,
    TEMPORAL,
    build_v2_experiment,
    copy_v2_experiment,
    default_responses,
    native_runs,
)


@pytest.fixture(scope="module")
def unpublished(tmp_path_factory):
    return build_v2_experiment(tmp_path_factory.mktemp("frozen"), publish=False)


@pytest.fixture
def v2(unpublished, tmp_path):
    root = copy_v2_experiment(unpublished, tmp_path).root
    native = native_runs(root)
    records = {
        run: publish_response_tables(
            item, item.csv_path.with_suffix(""), variables=("gwr",)
        )
        for run, item in native.items()
    }
    return root, native, records, dict(TEMPORAL)


def test_compact_publication_survives_transient_scalar_input_removal(v2):
    root, native, records, temporal = v2
    inventory = publish_response_inventory_v2(root, records, temporal)
    assert inventory["schema_version"] == "response-inventory/5"
    assert (
        inventory["identity_projection"]["schema_version"]
        == "response-inventory-identity/3"
    )
    assert [item["location_id"] for item in inventory["series"]] == ["9", "9"]
    assert all("native_selector" not in item for item in inventory["series"])
    for item in native.values():
        for path in (item.csv_path, item.toml_path, item.temporal_path):
            path.unlink()
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    assert read_response_inventory_v2(root) == inventory
    assert publish_response_inventory_v2(root, records, temporal) == inventory
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_record_relocation_changes_only_path_bearing_inventory(v2):
    root, _, records, temporal = v2
    first = build_response_inventory_v2(root, records, temporal)
    destination = root / "moved"
    shutil.copytree(records[RUN_IDS[0]].parent.parent, destination)
    moved = {**records, RUN_IDS[0]: destination / "_engine/response_tables.json"}
    second = build_response_inventory_v2(root, moved, temporal)
    assert first["response_identity_sha256"] == second["response_identity_sha256"]
    assert first["response_inventory_sha256"] != second["response_inventory_sha256"]


@pytest.mark.parametrize("target", ["record", "table"])
def test_changed_compact_evidence_refuses_reopening(v2, target):
    root, _, records, temporal = v2
    publish_response_inventory_v2(root, records, temporal)
    record = records[RUN_IDS[0]]
    path = record if target == "record" else record.parent.parent / "output_gwr.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(MissingResponseRequirement):
        read_response_inventory_v2(root)


def test_frozen_rounding_precision_is_required(v2):
    root, native, records, temporal = v2
    records[RUN_IDS[0]] = publish_response_tables(
        native[RUN_IDS[0]], root / "precision", variables=("gwr",), decimal_places=3
    )
    with pytest.raises(MissingResponseRequirement, match="precision"):
        publish_response_inventory_v2(root, records, temporal)
    assert not (root / "_engine/response_inventory.json").exists()


def test_complete_independent_variable_and_run_inventory_required(v2):
    root, native, records, temporal = v2
    with pytest.raises(MissingResponseRequirement, match="runs differ"):
        build_response_inventory_v2(root, {RUN_IDS[0]: records[RUN_IDS[0]]}, temporal)
    records[RUN_IDS[0]] = publish_response_tables(
        native[RUN_IDS[0]], root / "extra", variables=("q", "gwr")
    )
    with pytest.raises(MissingResponseRequirement, match="variables differ"):
        build_response_inventory_v2(root, records, temporal)


def test_old_native_inventory_is_refused_without_rewriting(v2):
    root, _, records, temporal = v2
    inventory = publish_response_inventory_v2(root, records, temporal)
    inventory["schema_version"] = "response-inventory/4"
    path = root / "_engine/response_inventory.json"
    before = canonical_json_bytes(inventory)
    path.write_bytes(before)
    with pytest.raises(MissingResponseRequirement, match="create a new experiment"):
        read_response_inventory_v2(root)
    assert path.read_bytes() == before


def test_native_artifacts_not_accepted_as_retained_inputs(v2):
    root, native, _, temporal = v2
    with pytest.raises(TypeError):
        build_response_inventory_v2(root, native, temporal)


def test_reader_source_change_requires_new_experiment(v2, monkeypatch):
    root, _, records, temporal = v2
    publish_response_inventory_v2(root, records, temporal)
    monkeypatch.setattr(
        response_inventory, "response_reader_revision", lambda: "0" * 64
    )
    with pytest.raises(ResponseReaderUnavailable, match="new experiment_name"):
        read_response_inventory_v2(root)


def test_corrupt_inventory_refuses_before_table_opening(v2, monkeypatch):
    root, _, records, temporal = v2
    inventory = publish_response_inventory_v2(root, records, temporal)
    inventory["series"][0]["units"] = "wrong"
    (root / "_engine/response_inventory.json").write_bytes(
        canonical_json_bytes(inventory)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("table reader called before inventory digest verification")

    monkeypatch.setattr(response_inventory, "open_table_responses", forbidden)
    with pytest.raises(MissingResponseRequirement, match="digest mismatch"):
        read_response_inventory_v2(root)


@pytest.mark.parametrize(
    "conversion", ["cftime_to_datetime64", "hydromt_reader_to_datetime64", None]
)
def test_noleap_conversion_must_be_declared(v2, conversion):
    root, _, records, temporal = v2
    temporal["source_calendar"] = "noleap"
    if conversion:
        temporal["operations"] = [conversion, *temporal["operations"]]
        publish_response_inventory_v2(root, records, temporal)
        assert read_response_inventory_v2(root)["temporal_preparation"] == temporal
    else:
        with pytest.raises(
            MissingResponseRequirement, match="temporal preparation chain"
        ):
            build_response_inventory_v2(root, records, temporal)


def test_clock_and_selector_tampering_refused(v2):
    root, _, records, temporal = v2
    bad = {**temporal, "response_start": temporal["prepared_start"]}
    with pytest.raises(MissingResponseRequirement, match="clock"):
        build_response_inventory_v2(root, records, bad)
    inventory = publish_response_inventory_v2(root, records, temporal)
    inventory["series"][0]["table_selector"]["variable"] = "q"
    inventory["response_inventory_sha256"] = content_sha256(
        {
            key: value
            for key, value in inventory.items()
            if key != "response_inventory_sha256"
        }
    )
    (root / "_engine/response_inventory.json").write_bytes(
        canonical_json_bytes(inventory)
    )
    with pytest.raises(MissingResponseRequirement, match="differs from compact"):
        read_response_inventory_v2(root)


def test_compact_inventory_keeps_trimmed_warmup_evidence(tmp_path):
    from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
    from blueearth_cst.shared.response_tables import open_table_responses

    original = default_responses()
    temporal = {
        **TEMPORAL,
        "operations": [
            "clip_to_configured_window",
            "prepend_warmup_years",
            "refresh_toml_endpoints",
        ],
        "prepared_end": "2046-01-04T00:00:00",
        "response_start": "2046-01-03 00:00:00",
        "response_end": "2046-01-04 00:00:00",
    }
    variable = {
        **original.variables[0],
        "start": temporal["response_start"],
        "end": temporal["response_end"],
    }
    responses = replace(
        original,
        csv={
            run: b"time,gwr_9\n2046-01-02,100\n2046-01-03,3\n2046-01-04,6\n"
            for run in RUN_IDS
        },
        toml=original.toml.replace("2046-01-03T00:00:00", "2046-01-04T00:00:00"),
        temporal=temporal,
        variables=[variable],
    )
    experiment = build_v2_experiment(tmp_path, responses)
    inventory = read_response_inventory_v2(experiment.root)
    assert inventory["temporal_preparation"] == temporal
    for artifact in inventory["artifacts"]:
        record = experiment.root / artifact["file"]["path"]
        series = open_table_responses(
            artifact["run_id"], record, ResponseRequest(("gwr",))
        )
        assert series[0].values.tolist() == [3.0, 6.0]
