"""Versioned native inventory binds science while retaining exact evidence.

Every case starts from a real v2 experiment frozen by the production producers
(`tests/_v2_experiment.py`) with its native runs retained and nothing yet
published -- the state rule 4.06 publishes from.
"""

import hashlib
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import pytest

from blueearth_cst.experiment import response_inventory
from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.response_inventory import (
    MissingResponseRequirement,
    ResponseReaderUnavailable,
    publish_response_inventory_v2,
    read_response_inventory_v2,
)
from tests._v2_experiment import (
    RUN_IDS,
    TEMPORAL,
    build_v2_experiment,
    copy_v2_experiment,
    native_runs,
)

FIRST = RUN_IDS[0]
#: The requested variable's Wflow parameter; the fixture's request asks for gwr.
RECHARGE = "soil_water_saturated_zone_top__net_recharge_volume_flux"


@pytest.fixture(scope="module")
def unpublished(tmp_path_factory):
    return build_v2_experiment(tmp_path_factory.mktemp("frozen"), publish=False)


@pytest.fixture
def v2(unpublished, tmp_path):
    root = copy_v2_experiment(unpublished, tmp_path).root
    assert not (root / "_engine/response_inventory.json").exists()
    return root, native_runs(root), dict(TEMPORAL)


@pytest.fixture(scope="module", params=["LF", "CRLF"])
def legacy_reader_experiment(tmp_path_factory, request):
    """Freeze the actual pre-rename reader contract, independently of its alias."""
    directory = Path(response_inventory.__file__).parent
    source = (directory / "response_series.py").read_bytes().replace(b"\r\n", b"\n")
    historical = source.replace(
        b"indicator implementations", b"metric implementations", 1
    )
    # Pinned from the pre-rename module, not imported from compatibility code.
    assert hashlib.sha256(historical).hexdigest() == (
        "148f4f51059adbaa902f0b53b7e4cbed65e9758384fa3c9f2bab67faab2788e9"
    )
    if request.param == "CRLF":
        historical = historical.replace(b"\n", b"\r\n")
    names = (
        "response_series.py",
        "wflow_response_reader.py",
        "../shared/native_locations.py",
    )
    revision = content_sha256(
        [
            {
                "path": name,
                "sha256": hashlib.sha256(
                    historical if name == names[0] else (directory / name).read_bytes()
                ).hexdigest(),
            }
            for name in names
        ]
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(response_inventory, "response_reader_revision", lambda: revision)
        return build_v2_experiment(tmp_path_factory.mktemp("legacy-reader"))


@pytest.fixture
def legacy_reader(legacy_reader_experiment, tmp_path):
    return copy_v2_experiment(legacy_reader_experiment, tmp_path).root


def test_vocabulary_only_reader_revision_reuses_retained_responses(legacy_reader):
    from blueearth_cst.experiment.simulation_record import read_simulation_v2

    root = legacy_reader
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    expected = read_canonical_json(root / "_engine/response_inventory.json")
    assert read_response_inventory_v2(root) == expected
    assert read_simulation_v2(root)["status"] == "ready"
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "changed", ["response_series.py", "wflow_response_reader.py", "native_locations.py"]
)
def test_legacy_reader_alias_refuses_unreviewed_source(
    legacy_reader, tmp_path, monkeypatch, changed
):
    directory = Path(response_inventory.__file__).parent
    copied = tmp_path / "reader/experiment"
    copied.mkdir(parents=True)
    (copied.parent / "shared").mkdir()
    for name in (
        "response_series.py",
        "wflow_response_reader.py",
        "../shared/native_locations.py",
    ):
        copyfile(directory / name, copied / name)
    # The copied, unchanged implementation must accept the same legacy contract.
    monkeypatch.setattr(
        response_inventory, "__file__", str(copied / "response_inventory.py")
    )
    read_response_inventory_v2(legacy_reader)
    path = (
        (copied.parent / "shared" / changed)
        if changed == "native_locations.py"
        else copied / changed
    )
    path.write_bytes(path.read_bytes() + b"\nUNREVIEWED_READER_CHANGE = True\n")
    with pytest.raises(ResponseReaderUnavailable):
        read_response_inventory_v2(legacy_reader)


def test_legacy_reader_alias_is_read_only(legacy_reader):
    with pytest.raises(ResponseReaderUnavailable):
        response_inventory.build_response_inventory_v2(
            legacy_reader, native_runs(legacy_reader), dict(TEMPORAL)
        )


def test_legacy_reader_alias_still_checks_native_bytes(legacy_reader):
    read_response_inventory_v2(legacy_reader)
    csv = native_runs(legacy_reader)[FIRST].csv_path
    csv.write_bytes(csv.read_bytes().replace(b",3\n", b",4\n"))
    with pytest.raises(MissingResponseRequirement):
        read_response_inventory_v2(legacy_reader)


def test_common_temporal_and_path_neutral_scientific_projection(v2):
    root, native_runs, temporal = v2
    inventory = publish_response_inventory_v2(root, native_runs, temporal)
    assert inventory["schema_version"] == "response-inventory/3"
    assert inventory["response_request"]["section"] == "/documents/response_request"
    assert inventory["series"][0]["native_selector"]["temporal"] == {
        "schema_version": "local-section-reference/1",
        "section": "/temporal_preparation",
        "section_sha256": inventory["temporal_preparation_sha256"],
    }
    assert inventory["identity_projection"]["series"][0]["native_selector"][
        "clock"
    ] == {
        "calendar": "standard",
        "starttime": "2046-01-01 00:00:00",
        "endtime": "2046-01-03 00:00:00",
        "timestepsecs": 86400,
    }
    native_runs[FIRST].temporal_path.unlink()
    assert read_response_inventory_v2(root) == inventory


def test_toml_path_change_preserves_response_identity_but_changes_full_evidence(v2):
    root, native_runs, temporal = v2
    first = publish_response_inventory_v2(root, native_runs, temporal)
    copyfile(native_runs[FIRST].toml_path, root / "moved.toml")
    moved = {
        **native_runs,
        FIRST: replace(native_runs[FIRST], toml_path=root / "moved.toml"),
    }
    from blueearth_cst.experiment.response_inventory import build_response_inventory_v2

    second = build_response_inventory_v2(root, moved, temporal)
    assert first["response_identity_sha256"] == second["response_identity_sha256"]
    assert first["response_inventory_sha256"] != second["response_inventory_sha256"]


def test_path_bearing_toml_change_keeps_scientific_identity(v2):
    root, native_runs, temporal = v2
    first = publish_response_inventory_v2(root, native_runs, temporal)
    toml = native_runs[FIRST].toml_path
    toml.write_text(toml.read_text() + '\n[input]\npath_forcing="elsewhere.nc"\n')
    from blueearth_cst.experiment.response_inventory import build_response_inventory_v2

    second = build_response_inventory_v2(root, native_runs, temporal)
    assert first["response_identity_sha256"] == second["response_identity_sha256"]
    assert first["response_inventory_sha256"] != second["response_inventory_sha256"]


def test_clock_change_refuses_native_reopening(v2):
    root, native_runs, temporal = v2
    publish_response_inventory_v2(root, native_runs, temporal)
    toml = native_runs[FIRST].toml_path
    toml.write_text(toml.read_text().replace("timestepsecs=86400", "timestepsecs=3600"))
    with pytest.raises(MissingResponseRequirement):
        read_response_inventory_v2(root)


def test_time_units_are_nonsemantic_native_clock_metadata(v2):
    root, native_runs, temporal = v2
    toml = native_runs[FIRST].toml_path
    toml.write_text(
        toml.read_text().replace(
            'calendar="standard"\n',
            'calendar="standard"\ntime_units="days since 1900-01-01 00:00:00"\n',
        )
    )

    inventory = publish_response_inventory_v2(root, native_runs, temporal)

    assert inventory["identity_projection"]["series"][0]["native_selector"][
        "clock"
    ] == {
        "calendar": "standard",
        "starttime": "2046-01-01 00:00:00",
        "endtime": "2046-01-03 00:00:00",
        "timestepsecs": 86400,
    }


def test_output_routing_metadata_is_not_response_selector_semantics(v2):
    root, native_runs, temporal = v2
    toml = native_runs[FIRST].toml_path
    toml.write_text(
        toml.read_text().replace(
            f'parameter="{RECHARGE}"\n',
            f'parameter="{RECHARGE}"\nmap="outlets"\nreducer="mean"\n',
        )
    )

    inventory = publish_response_inventory_v2(root, native_runs, temporal)

    assert inventory["identity_projection"]["series"][0]["native_selector"][
        "output_declarations"
    ] == [{"header": "gwr", "parameter": RECHARGE}]


def test_local_temporal_pointer_tamper_refuses_before_native_open(v2, monkeypatch):
    root, native_runs, temporal = v2
    inventory = publish_response_inventory_v2(root, native_runs, temporal)
    inventory["series"][0]["native_selector"]["temporal"]["section"] = "/other"
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

    def forbidden(*args, **kwargs):
        pytest.fail("native reader called before local section validation")

    monkeypatch.setattr(
        "blueearth_cst.experiment.response_inventory.open_responses", forbidden
    )
    with pytest.raises(MissingResponseRequirement, match="local temporal section"):
        read_response_inventory_v2(root)


def test_per_run_temporal_difference_refuses_publication(v2):
    root, native_runs, temporal = v2
    native_runs[FIRST].temporal_path.write_bytes(
        canonical_json_bytes({**temporal, "source_calendar": "standard"})
    )
    with pytest.raises(MissingResponseRequirement, match="inconsistent temporal"):
        publish_response_inventory_v2(root, native_runs, temporal)
    assert not (root / "_engine/response_inventory.json").exists()


# --- ported from the retired v1 inventory tests (t2610041227) ----------------


def _all_runs(native_runs, payload):
    for artifacts in native_runs.values():
        artifacts.temporal_path.write_bytes(canonical_json_bytes(payload))


def test_publication_is_self_contained_and_reuse_rewrites_nothing(v2):
    root, native_runs, temporal = v2
    inventory = publish_response_inventory_v2(root, native_runs, temporal)
    assert [item["location_id"] for item in inventory["series"]] == ["9", "9"]
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    assert read_response_inventory_v2(root) == inventory
    assert publish_response_inventory_v2(root, native_runs, temporal) == inventory
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "native",
    [
        b"time,Q_9,Q_2\n2046-01-02,1,2\n2046-01-03,4,\n",
        b"time,Q_9,Q_2,gwr_3\n2046-01-02,1,2,3\n2046-01-03,4,,6\n",
    ],
    ids=["missing", "unexpected"],
)
def test_a_missing_or_unexpected_series_refuses_publication(v2, native):
    root, native_runs, temporal = v2
    native_runs[FIRST].csv_path.write_bytes(native)
    # MissingResponseRequirement is itself a ValueError; the native reader
    # refuses a missing column before the inventory's own key check runs.
    with pytest.raises(ValueError):
        publish_response_inventory_v2(root, native_runs, temporal)
    assert not (root / "_engine/response_inventory.json").exists()


@pytest.mark.parametrize("artifact", ["csv_path", "toml_path"])
def test_a_changed_native_artifact_refuses_reopening(v2, artifact):
    root, native_runs, temporal = v2
    publish_response_inventory_v2(root, native_runs, temporal)
    path = getattr(native_runs[FIRST], artifact)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((MissingResponseRequirement, ValueError)):
        read_response_inventory_v2(root)


def test_a_temporal_chain_off_the_native_clock_refuses_publication(v2):
    root, native_runs, temporal = v2
    temporal["response_start"] = "2046-01-01 00:00:00"
    _all_runs(native_runs, temporal)
    with pytest.raises(MissingResponseRequirement, match="clock"):
        publish_response_inventory_v2(root, native_runs, temporal)


@pytest.mark.parametrize("conversion", ["cftime_to_datetime64", None])
def test_a_calendar_conversion_must_be_recorded(v2, conversion):
    root, native_runs, temporal = v2
    temporal["source_calendar"] = "noleap"
    if conversion is not None:
        temporal["operations"] = [conversion, *temporal["operations"]]
    _all_runs(native_runs, temporal)
    if conversion is None:
        with pytest.raises(
            MissingResponseRequirement, match="temporal preparation chain"
        ):
            publish_response_inventory_v2(root, native_runs, temporal)
    else:
        publish_response_inventory_v2(root, native_runs, temporal)
        assert read_response_inventory_v2(root)["temporal_preparation"] == temporal


def test_a_corrupt_inventory_digest_refuses_before_native_open(v2, monkeypatch):
    root, native_runs, temporal = v2
    inventory = publish_response_inventory_v2(root, native_runs, temporal)
    inventory["series"][0]["units"] = "wrong"
    (root / "_engine/response_inventory.json").write_bytes(
        canonical_json_bytes(inventory)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("native reader called before inventory digest validation")

    monkeypatch.setattr(
        "blueearth_cst.experiment.response_inventory.open_responses", forbidden
    )
    with pytest.raises(MissingResponseRequirement):
        read_response_inventory_v2(root)
