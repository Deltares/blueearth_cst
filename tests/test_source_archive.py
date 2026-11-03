"""P1 source archive: exact captured bytes and recoverable publication."""

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared import config_composition as cc
from blueearth_cst.shared import workflow_config_snapshot as snapshot
from blueearth_cst.shared.workflow_config_snapshot import (
    capture_rerun_predecessor,
    capture_sources,
    file_reference,
    publish_archive,
    read_archive,
    recover_archive,
    resolve_file_reference,
    resolve_rerun_execution,
    run_record_document,
    source_archive_paths,
    validate_archive,
)


def _archive(
    tmp_path: Path, invocation_id: str, data: bytes = b"# comment\r\nx: 1\r\n"
):
    project = tmp_path / "original" / "project_config.yml"
    project.parent.mkdir(exist_ok=True)
    project.write_bytes(data)
    captured = capture_sources([("project", "project_config", project)])
    project.write_bytes(b"# changed after capture\nx: 2\n")
    record, payloads = run_record_document(
        workflow="build_model",
        invocation_id=invocation_id,
        owner_kind="workflow",
        owner_id="build_model",
        loaded_config={"workflows": {"build_model": {"enabled": True}}},
        advanced_settings={},
        projection=["workflows.build_model"],
        toolbox={"commit": None},
        invocation={
            "entry_point": "run_workflow.py",
            "command": [],
            "targets": ["all"],
            "working_directory": str(tmp_path),
            "overrides": {},
        },
        sources=captured,
        environment={},
    )
    return record, payloads


def _legacy_archive(record, payloads):
    """Reproduce the retained v1 wire layout without changing its source bytes."""
    record = deepcopy(record)
    record["archive_schema_version"] = "source-archive/1"
    for entry in record["source_files"]:
        entry["archived_path"] = "sources/" + entry["archived_path"]
    identity = {
        key: record[key]
        for key in (
            "workflow",
            "invocation_id",
            "loaded_config",
            "advanced_settings",
            "configuration_projection",
            "referenced_inputs",
            "generated_inputs",
            "rerun",
        )
    }
    identity["schema_version"] = "source-archive/1"
    identity["source_files"] = [
        {
            key: entry[key]
            for key in ("id", "role", "archived_path", "sha256", "size_bytes")
        }
        for entry in record["source_files"]
    ]
    record["archive_id"] = snapshot.collection_sha256(identity)
    return record, {"sources/" + path: data for path, data in payloads.items()}


@pytest.mark.parametrize(
    "workflow,target,owner",
    [
        ("analyze_climate", "config/analyze_climate", "workflow"),
        ("build_model", "config/build_model", "workflow"),
        ("analyze_projections", "config/analyze_projections", "workflow"),
        (
            "generate_scenarios",
            "scenarios/_engine/data/123456abcdef/config",
            "scenario_collection",
        ),
        ("simulate_system", "experiments/test/config", "experiment"),
    ],
)
def test_all_workflows_publish_flat_captured_archives(
    tmp_path, workflow, target, owner
):
    sources = []
    for source_id, role, name in (
        ("project", "project_config", "project.yml"),
        ("workflow", f"workflow_config_{workflow}", f"{workflow}.yml"),
        ("catalog", "data_catalog", "catalog.yml"),
    ):
        path = tmp_path / name
        path.write_bytes(b"# retained\r\nx: 1\r\n")
        sources.append((source_id, role, path))
    record, payloads = run_record_document(
        workflow=workflow,
        invocation_id="flat",
        owner_kind=owner,
        owner_id=workflow,
        loaded_config={},
        advanced_settings={},
        projection=[],
        toolbox={},
        environment={},
        invocation={
            "entry_point": "test",
            "command": [],
            "targets": [],
            "working_directory": str(tmp_path),
            "overrides": {},
        },
        sources=capture_sources(sources),
    )
    project = tmp_path / "output"
    publish_archive(project, workflow, target, record, payloads)
    directory = project / target
    assert validate_archive(directory) == record
    assert record["archive_schema_version"] == "source-archive/2"
    assert {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    } == {
        "run_record.yml",
        "SOURCES.md",
        "project.yml",
        f"{workflow}.yml",
        "catalogs/catalog.yml",
    }
    assert not (directory / "sources").exists()


def test_reserved_and_equal_content_basenames_are_portably_disambiguated(tmp_path):
    captures = []
    names = [
        "run_record.yml",
        "SOURCES.md",
        "catalogs",
        "sources",
        "_engine",
        "predecessor",
        "same.yml",
        "SAME.yml",
        "same.yml",
    ]
    for index, name in enumerate(names):
        path = tmp_path / str(index) / name
        path.parent.mkdir()
        path.write_bytes(b"identical")
        captures.append((str(index), "input", path))
    sources = capture_sources(captures)
    paths = source_archive_paths(sources)
    assert paths == source_archive_paths(tuple(reversed(sources)))
    assert len({path.casefold() for path in paths.values()}) == len(names)
    assert not set(map(str.casefold, paths.values())) & snapshot._ARCHIVE_RESERVED_NAMES


@pytest.mark.parametrize(
    "name", ["run_record.yml", "SOURCES.md", "catalogs", "sources", "_engine"]
)
def test_flat_reader_refuses_reserved_source_paths(tmp_path, name):
    record, _ = _archive(tmp_path, "invalid")
    record["source_files"][0]["archived_path"] = name
    with pytest.raises(ValueError, match="wrong archive base"):
        snapshot._validate_run_record_shape(record)


def test_duplicate_capture_of_same_original_can_be_published(tmp_path):
    record, _ = _archive(tmp_path, "duplicate")
    source = tmp_path / "original/project_config.yml"
    entries, payloads = snapshot._source_entries(
        capture_sources(
            [("project", "project_config", source), ("shared", "input", source)]
        )
    )
    record["source_files"] = entries
    snapshot._validate_run_record_shape(record)
    assert entries[0]["archived_path"] == entries[1]["archived_path"]
    assert set(payloads) == {"project_config.yml", "SOURCES.md"}


def test_captured_bytes_survive_live_mutation_and_publish(tmp_path):
    record, payloads = _archive(tmp_path, "first")
    root = tmp_path / "output"
    publish_archive(root, "build_model", "config/build_model", record, payloads)
    loaded = read_archive(root, "build_model", "config/build_model")
    entry = loaded["source_files"][0]
    assert entry["archived_path"] == "project_config.yml"
    assert (
        root / "config/build_model" / entry["archived_path"]
    ).read_bytes() == b"# comment\r\nx: 1\r\n"
    assert loaded["schema_version"] == "run-record/2"
    assert not (root / "config/runs").exists()


@pytest.mark.parametrize("target", ["config/runs/build_model", "config/unknown"])
def test_archive_publication_refuses_noncanonical_workflow_directory(tmp_path, target):
    record, payloads = _archive(tmp_path, "first")
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="archive target"):
        publish_archive(root, "build_model", target, record, payloads)
    assert not root.exists()


def test_first_publication_replaces_empty_target_directory(tmp_path):
    record, payloads = _archive(tmp_path, "first")
    root = tmp_path / "output"
    (root / "config/build_model").mkdir(parents=True)

    publish_archive(root, "build_model", "config/build_model", record, payloads)

    assert (
        read_archive(root, "build_model", "config/build_model")["invocation_id"]
        == "first"
    )


def test_publication_retries_transient_windows_access_denied(tmp_path, monkeypatch):
    root = tmp_path / "output"
    first, first_payloads = _archive(tmp_path, "first")
    publish_archive(root, "build_model", "config/build_model", first, first_payloads)
    second, second_payloads = _archive(tmp_path, "second", b"second: true\n")
    real_replace = snapshot.os.replace
    attempts = 0

    def replace_with_contention(source, target):
        nonlocal attempts
        if Path(source).name == "build_model" and Path(target).suffix == ".previous":
            attempts += 1
            if attempts < 3:
                error = PermissionError(13, "Access is denied", str(source))
                error.winerror = 5
                raise error
        real_replace(source, target)

    monkeypatch.setattr(snapshot, "_WINDOWS_REPLACE_RETRY", True, raising=False)
    monkeypatch.setattr(snapshot.os, "replace", replace_with_contention)

    publish_archive(root, "build_model", "config/build_model", second, second_payloads)

    assert attempts == 3
    assert (
        read_archive(root, "build_model", "config/build_model")["invocation_id"]
        == "second"
    )


def test_replace_retry_budget_outlasts_the_old_10_attempt_cap(tmp_path, monkeypatch):
    """Regression for t2609231529: a live run's antivirus/indexer held a
    directory rename locked past the previous 10-attempt, flat-20ms budget
    (0.2s total). This asserts the budget now survives an 11th failure --
    something the prior `_replace` would have raised on instead of retrying.
    """
    monkeypatch.setattr(snapshot, "_WINDOWS_REPLACE_RETRY", True, raising=False)
    monkeypatch.setattr(snapshot.time, "sleep", lambda _seconds: None)
    real_replace = snapshot.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts <= 11:
            error = PermissionError(13, "Access is denied", str(source))
            error.winerror = 5
            raise error
        real_replace(source, target)

    monkeypatch.setattr(snapshot.os, "replace", flaky_replace)
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_text("data", encoding="utf-8")

    snapshot._replace(source, target)

    assert attempts == 12
    assert target.read_text(encoding="utf-8") == "data"


def test_experiment_owner_is_an_archive_owner_kind(tmp_path):
    project = tmp_path / "project.yml"
    project.write_text("project: {}\n", encoding="utf-8")
    record, _ = run_record_document(
        workflow="simulate_system",
        invocation_id="simulation",
        owner_kind="experiment",
        owner_id="a" * 64,
        loaded_config={"workflows": {"simulate_system": {"enabled": True}}},
        advanced_settings={},
        projection=["workflows.simulate_system"],
        toolbox={"commit": None},
        invocation={
            "entry_point": "simulate_system.py",
            "command": [],
            "targets": ["all"],
            "working_directory": str(tmp_path),
            "overrides": {},
        },
        sources=capture_sources([("project", "project_config", project)]),
        environment={},
    )
    assert record["owner"] == {"kind": "experiment", "id": "a" * 64}


@pytest.mark.parametrize(
    "failure_after,expected",
    [
        ("prepared", "rolled_back"),
        ("old_detached", "rolled_back"),
        ("new_installed", "committed"),
    ],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_recovery_and_next_writer(tmp_path, failure_after, expected, legacy):
    root = tmp_path / "output"
    first, first_payloads = _archive(tmp_path, "first")
    if legacy:
        first, first_payloads = _legacy_archive(first, first_payloads)
    publish_archive(root, "build_model", "config/build_model", first, first_payloads)
    second, second_payloads = _archive(tmp_path, "second", b"new: true\n")
    with pytest.raises(RuntimeError, match="injected failure"):
        publish_archive(
            root,
            "build_model",
            "config/build_model",
            second,
            second_payloads,
            failure_after=failure_after,
        )
    assert recover_archive(root, "build_model") == expected
    assert read_archive(root, "build_model", "config/build_model")["invocation_id"] == (
        "second" if expected == "committed" else "first"
    )
    assert recover_archive(root, "build_model") == expected
    third, third_payloads = _archive(tmp_path, "third", b"third: true\n")
    publish_archive(root, "build_model", "config/build_model", third, third_payloads)
    assert (
        read_archive(root, "build_model", "config/build_model")["invocation_id"]
        == "third"
    )
    assert list((root / "config/_engine/archive-transactions/completed").glob("*.json"))


def test_first_publication_failure_stays_unpublished(tmp_path):
    root = tmp_path / "output"
    first, payloads = _archive(tmp_path, "first")
    with pytest.raises(RuntimeError):
        publish_archive(
            root,
            "build_model",
            "config/build_model",
            first,
            payloads,
            failure_after="prepared",
        )
    assert recover_archive(root, "build_model") == "unpublished"
    assert not (root / "config/build_model").exists()
    publish_archive(root, "build_model", "config/build_model", first, payloads)
    assert (
        read_archive(root, "build_model", "config/build_model")["invocation_id"]
        == "first"
    )


@pytest.mark.parametrize(
    "failure_after,candidate",
    [
        ("prepared", "staged"),
        ("old_detached", "previous"),
        ("new_installed", "previous"),
    ],
)
def test_contradictory_candidate_refuses_recovery_reader_and_next_writer(
    tmp_path, failure_after, candidate
):
    root = tmp_path / "output"
    first, first_payloads = _archive(tmp_path, "first")
    publish_archive(root, "build_model", "config/build_model", first, first_payloads)
    second, second_payloads = _archive(tmp_path, "second", b"second: true\n")
    with pytest.raises(RuntimeError):
        publish_archive(
            root,
            "build_model",
            "config/build_model",
            second,
            second_payloads,
            failure_after=failure_after,
        )
    journal = root / "config/_engine/archive-transactions/build_model.json"
    transaction = json.loads(journal.read_text(encoding="utf-8"))
    path = root / transaction[candidate] / "run_record.yml"
    path.write_bytes(b"corrupt")
    for action in (
        lambda: recover_archive(root, "build_model"),
        lambda: read_archive(root, "build_model", "config/build_model"),
        lambda: publish_archive(
            root, "build_model", "config/build_model", second, second_payloads
        ),
    ):
        with pytest.raises(ValueError, match="contradicts"):
            action()
    assert path.read_bytes() == b"corrupt"


def test_duplicate_basenames_are_disambiguated_by_source_digest(tmp_path):
    """Distinct source paths with the same basename get deterministic suffixes."""
    first = tmp_path / "a" / "settings.yml"
    second = tmp_path / "b" / "settings.yml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one\n")
    second.write_bytes(b"two\n")
    sources = capture_sources(
        [("a", "catalog_a", first), ("b", "data_catalog", second)]
    )
    paths = source_archive_paths(sources)
    assert paths["a"] == "catalogs/settings.yml"
    digest = hashlib.sha256(str(second).encode() + b"\0two\n").hexdigest()[:12]
    assert paths["b"] == f"catalogs/settings-{digest}.yml"


def test_projection_store_index_travels_with_its_catalog(tmp_path):
    """`projection_store_index` (cmip6_store_index.json) has no "catalog" in its
    own role name, but ships beside `projection_catalog` and is bucketed with
    it for convenience rather than beside the record."""
    catalog = tmp_path / "cmip6_data.yml"
    index = tmp_path / "cmip6_store_index.json"
    catalog.write_bytes(b"meta: {}\n")
    index.write_bytes(b"{}\n")
    sources = capture_sources(
        [
            ("projection_catalog", "projection_catalog", catalog),
            ("projection_index", "projection_store_index", index),
        ]
    )
    paths = source_archive_paths(sources)
    assert paths == {
        "projection_catalog": "catalogs/cmip6_data.yml",
        "projection_index": "catalogs/cmip6_store_index.json",
    }


def test_a_shared_original_path_is_not_a_collision(tmp_path):
    """The same file captured under two ids shares one archive path, no suffix."""
    shared = tmp_path / "catalog.yml"
    shared.write_bytes(b"meta: {}\n")
    sources = capture_sources(
        [("a", "data_catalog", shared), ("b", "data_catalog", shared)]
    )
    paths = source_archive_paths(sources)
    assert paths == {
        "a": "catalogs/catalog.yml",
        "b": "catalogs/catalog.yml",
    }


def test_non_catalog_sources_are_flat_with_no_directory_structure(tmp_path):
    """A project config and a build config in different directories both land
    directly beside the record, with no mirrored subdirectory."""
    project = tmp_path / "outer" / "project_config.yml"
    build_config = tmp_path / "inner" / "wflow_build_model.yml"
    project.parent.mkdir()
    build_config.parent.mkdir()
    project.write_bytes(b"project: {}\n")
    build_config.write_bytes(b"setup_config: {}\n")
    sources = capture_sources(
        [
            ("project", "project_config", project),
            ("build_config", "build_config", build_config),
        ]
    )
    assert source_archive_paths(sources) == {
        "project": "project_config.yml",
        "build_config": "wflow_build_model.yml",
    }


def test_source_index_maps_flat_names_to_original_directories(tmp_path):
    """`SOURCES.md` mirrors what `run_record.yml` already records."""
    project = tmp_path / "outer" / "project_config.yml"
    catalog = tmp_path / "inner" / "catalog.yml"
    project.parent.mkdir()
    catalog.parent.mkdir()
    project.write_bytes(b"project: {}\n")
    catalog.write_bytes(b"meta: {}\n")
    sources = capture_sources(
        [
            ("project", "project_config", project),
            ("catalog", "data_catalog", catalog),
        ]
    )
    index = snapshot.write_source_index(sources)
    assert f"| `project_config.yml` | `{project.parent}` |" in index
    assert f"| `catalogs/catalog.yml` | `{catalog.parent}` |" in index


def test_file_reference_refuses_tampered_bytes_and_unbound_paths(tmp_path):
    path = tmp_path / "source.yml"
    path.write_bytes(b"one\n")
    reference = file_reference(path, "record_directory", tmp_path)
    assert resolve_file_reference(reference, {"record_directory": tmp_path}) == path
    path.write_bytes(b"two\n")
    with pytest.raises(ValueError, match="bytes differ"):
        resolve_file_reference(reference, {"record_directory": tmp_path})
    with pytest.raises(ValueError, match="unbound"):
        resolve_file_reference(reference, {"project_root": tmp_path})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update({"extra": 1}),
        lambda r: r["source_files"][0].update({"extra": 1}),
        lambda r: r.update({"source_config": "missing"}),
        lambda r: r["generated_inputs"].append(
            {
                "role": "input",
                "file": {
                    "schema_version": "artifact-reference/1",
                    "path_base": "arbitrary",
                    "path": "x",
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                },
            }
        ),
    ],
)
def test_closed_record_rejects_unknown_fields_and_bad_pointers(tmp_path, mutation):
    root = tmp_path / "output"
    record, payloads = _archive(tmp_path, "first")
    publish_archive(root, "build_model", "config/build_model", record, payloads)
    directory = root / "config/build_model"
    loaded = yaml.safe_load((directory / "run_record.yml").read_bytes())
    mutation(loaded)
    (directory / "run_record.yml").write_text(yaml.safe_dump(loaded), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_archive(directory)


def test_builder_refuses_unclassified_rerun_overlay(tmp_path):
    source = tmp_path / "project.yml"
    source.write_text("x: 1\n", encoding="utf-8")
    captured = capture_sources([("project", "project_config", source)])
    with pytest.raises(ValueError, match="resolver-overlay"):
        run_record_document(
            workflow="build_model",
            invocation_id="first",
            owner_kind="workflow",
            owner_id="build_model",
            loaded_config={"workflows": {"build_model": {}}},
            advanced_settings={},
            projection=["workflows.build_model"],
            toolbox={"commit": None},
            environment={},
            sources=captured,
            invocation={
                "entry_point": "x",
                "command": [],
                "targets": [],
                "working_directory": str(tmp_path),
                "overrides": {},
            },
            rerun_adjustments=[{"schema_version": "resolver-overlay/1", "extra": 1}],
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_rerun_retains_predecessor_after_original_is_removed(tmp_path, legacy):
    first_root = tmp_path / "A"
    first, payloads = _archive(tmp_path, "first")
    if legacy:
        first, payloads = _legacy_archive(first, payloads)
    publish_archive(first_root, "build_model", "config/build_model", first, payloads)
    predecessor = first_root / "config/build_model"
    second_root = tmp_path / "B"
    sources, retained, overlay = capture_rerun_predecessor(
        predecessor,
        predecessor_project_root=first_root,
        new_project_root=second_root,
        output_project_dir=str(second_root),
    )
    second, second_payloads = run_record_document(
        workflow="build_model",
        invocation_id="second",
        owner_kind="workflow",
        owner_id="build_model",
        loaded_config={"workflows": {"build_model": {"enabled": True}}},
        advanced_settings={},
        projection=["workflows.build_model"],
        toolbox={"commit": None},
        invocation={
            "entry_point": "run_workflow.py",
            "command": [],
            "targets": ["all"],
            "working_directory": str(tmp_path),
            "overrides": {},
        },
        sources=sources,
        rerun_adjustments=[overlay],
        environment={},
    )
    publish_archive(
        second_root,
        "build_model",
        "config/build_model",
        second,
        second_payloads | retained,
    )
    (tmp_path / "original" / "project_config.yml").unlink()
    moved = tmp_path / "C"
    second_root.rename(moved)
    assert (
        read_archive(moved, "build_model", "config/build_model")["invocation_id"]
        == "second"
    )


@pytest.mark.parametrize("schema_version", [2, 3])
def test_a_to_b_to_relocated_c_composes_without_a(tmp_path, schema_version):
    root_a = tmp_path / "A"
    source_dir = root_a / "inputs"
    source_dir.mkdir(parents=True)
    project_path = source_dir / "project_config.yml"
    workflow_path = source_dir / "project_config_build_model.yml"
    project_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": schema_version,
                "project": {"project_dir": str(root_a)},
                "basin": {"region": "{'subbasin': [1, 2]}"},
                "climate": {"sources": ["era5"], "selected": "era5"},
                "workflows": {
                    "build_model": {"enabled": True, "config_path": workflow_path.name}
                },
            }
        ),
        encoding="utf-8",
    )
    workflow_bytes = b"# retained\r\nengine:\r\n  build_config: old.yml\r\n"
    workflow_path.write_bytes(workflow_bytes)
    sources = cc.capture_configuration_sources(project_path, "build_model")
    composed, _ = cc.compose_captured_config(sources[0], sources, "build_model")
    invocation = {
        "entry_point": "run_workflow.py",
        "command": [],
        "targets": ["all"],
        "working_directory": str(tmp_path),
        "overrides": {},
    }
    first, payloads = run_record_document(
        workflow="build_model",
        invocation_id="first",
        owner_kind="workflow",
        owner_id="build_model",
        loaded_config=composed,
        advanced_settings={},
        projection=["workflows.build_model"],
        toolbox={"commit": None},
        invocation=invocation,
        sources=sources,
        environment={},
    )
    publish_archive(root_a, "build_model", "config/build_model", first, payloads)
    root_b = tmp_path / "B"
    captured, retained, overlay = capture_rerun_predecessor(
        root_a / "config/build_model",
        predecessor_project_root=root_a,
        new_project_root=root_b,
        output_project_dir=str(root_b),
    )
    second, second_payloads = run_record_document(
        workflow="build_model",
        invocation_id="second",
        owner_kind="workflow",
        owner_id="build_model",
        loaded_config=composed,
        advanced_settings={},
        projection=["workflows.build_model"],
        toolbox={"commit": None},
        invocation=invocation,
        sources=captured,
        rerun_adjustments=[overlay],
        environment={},
    )
    publish_archive(
        root_b,
        "build_model",
        "config/build_model",
        second,
        second_payloads | retained,
    )
    shutil.rmtree(root_a)
    root_c = tmp_path / "C"
    root_b.rename(root_c)
    resolved = resolve_rerun_execution(root_c, "build_model", "config/build_model")
    assert resolved.composed_config["workflows"]["build_model"]["engine"][
        "build_config"
    ] == ("old.yml" if schema_version == 2 else (source_dir / "old.yml").as_posix())
    assert resolved.composed_config["project"]["project_dir"] == str(root_c)
    assert (
        Path(resolved.workflow_config_paths["build_model"]).read_bytes()
        == workflow_bytes
    )
    assert resolved.resolve_path(project_path).is_file()
    assert (
        resolved.resolve_path(root_a / "data" / "input.nc")
        == root_c / "data" / "input.nc"
    )
    with pytest.raises(ValueError, match="unmapped external"):
        resolved.resolve_path(tmp_path / "unmapped.nc")
