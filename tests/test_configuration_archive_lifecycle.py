"""Configuration snapshots survive attempts without accumulating duplicate captures."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared import windows_job
from blueearth_cst.shared import workflow_archive_launch as launch
from blueearth_cst.shared.config_paths import resolve_project_paths
from blueearth_cst.shared.configuration_archive import (
    maintain_archives,
    maintenance_report,
)
from blueearth_cst.shared.workflow_config_snapshot import (
    CapturedSource,
    capture_rerun_predecessor,
    file_reference,
    publish_archive,
    read_archive,
    resolve_file_reference,
    run_record_document,
    validate_archive,
)
from scripts import maintain_config_archives, run_workflows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_bytes()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    output = tmp_path / "output"
    project["project"]["project_dir"] = str(output)
    catalog = tmp_path / "catalog.yml"
    catalog.write_text("meta:\n  roots: [C:/data]\nera5:\n  uri: data.nc\n")
    project["project"]["catalog"] = [str(catalog)]
    for stanza in project["workflows"].values():
        if "config_path" in stanza:
            stanza["config_path"] = str(ROOT / "test_case" / stanza["config_path"])
    config = tmp_path / "project.yml"
    config.write_text(yaml.safe_dump(project), encoding="utf-8")
    monkeypatch.setattr(
        launch,
        "toolbox_identity",
        lambda: {"commit": "a" * 40, "commit_source": "git", "dirty": False},
    )
    monkeypatch.setattr(launch, "environment_file_hashes", lambda: {})
    return config, output, catalog


def _prepare(case, workflow, invocation_id):
    config, output, _ = case
    return launch.prepare_workflow(
        workflow,
        config,
        output,
        command=["snakemake", "all"],
        targets=["all"],
        invocation_id=invocation_id,
    )


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("workflow", launch.WORKFLOWS)
def test_identical_attempt_reuses_capture_without_rewriting_payloads(case, workflow):
    _, output, _ = case
    execution, first_context = _prepare(case, workflow, "first")
    current = output / "config" / workflow
    before = _files(current)
    snapshots = output / "config/_engine/configuration-snapshots" / workflow
    snapshot_files = _files(snapshots)
    assert len(list(snapshots.glob("*/run_record.yml"))) == 1

    repeated_execution, second_context = _prepare(case, workflow, "second")

    assert repeated_execution == execution
    assert first_context != second_context
    assert _files(current) == before
    assert not (output / "config/SETTINGS.md").exists()
    assert _files(snapshots) == snapshot_files
    record = read_archive(output, workflow, f"config/{workflow}")
    assert record["invocation_id"] == "first"
    assert not list((output / "config").glob(".*.previous"))


def test_changed_capture_keeps_previous_attempt_reference_valid(case, monkeypatch):
    config, output, catalog = case
    contexts = []

    def child(command, **kwargs):
        context = Path(kwargs["env"][launch.CONTEXT_ENV])
        assert context.is_file()
        contexts.append(context)
        return 0

    monkeypatch.setattr(windows_job, "run_project_child", child)
    assert launch.run_workflow("analyze_climate", config, output) == 0
    invocations = output / "config/_engine/invocations"
    first_path = next(invocations.glob("*.json"))
    first = json.loads(first_path.read_text(encoding="utf-8"))
    first_reference = first["configuration"]["run_record"]
    first_record = resolve_file_reference(first_reference, {"project_root": output})
    assert first_record.is_relative_to(
        output / "config/_engine/configuration-snapshots/analyze_climate"
    )
    original = _files(first_record.parent)
    original_catalog = catalog.read_bytes()

    catalog.write_bytes(catalog.read_bytes() + b"\n# revised dependency\n")
    assert launch.run_workflow("analyze_climate", config, output) == 0

    second_path = next(
        path for path in invocations.glob("*.json") if path != first_path
    )
    second = json.loads(second_path.read_text(encoding="utf-8"))
    second_record = resolve_file_reference(
        second["configuration"]["run_record"], {"project_root": output}
    )
    assert second_record != first_record
    assert _files(first_record.parent) == original
    assert (
        resolve_file_reference(first_reference, {"project_root": output})
        == first_record
    )
    assert (
        validate_archive(first_record.parent)["archive_id"]
        != validate_archive(second_record.parent)["archive_id"]
    )
    assert not any(context.exists() for context in contexts)

    catalog.write_bytes(original_catalog)
    assert launch.run_workflow("analyze_climate", config, output) == 0
    paths = [
        path
        for path in invocations.glob("*.json")
        if path not in {first_path, second_path}
    ]
    assert len(paths) == 1
    restored = json.loads(paths[0].read_text(encoding="utf-8"))
    assert restored["configuration"]["run_record"] == first_reference
    assert _files(first_record.parent) == original
    assert len(list(first_record.parent.parent.glob("*/run_record.yml"))) == 2


@pytest.mark.parametrize("outcome", [0, 7, "exception"])
@pytest.mark.parametrize("runner", ["individual", "all"])
def test_launch_context_is_removed_after_child_completion(
    case, monkeypatch, outcome, runner
):
    config, output, _ = case
    contexts = []

    def child(command, **kwargs):
        context = Path(kwargs["env"][launch.CONTEXT_ENV])
        assert context.is_file()
        contexts.append(context)
        if outcome == "exception":
            raise OSError("child failed to start")
        return outcome

    if runner == "individual":
        monkeypatch.setattr(windows_job, "run_project_child", child)

        def run():
            return launch.run_workflow("analyze_climate", config, output)

    else:
        project = yaml.safe_load(config.read_bytes())
        for name, stanza in project["workflows"].items():
            stanza["enabled"] = name == "analyze_climate"
        config.write_text(yaml.safe_dump(project), encoding="utf-8")
        monkeypatch.setattr(run_workflows, "run_project_child", child)

        def run():
            return run_workflows.run(str(config), cores=1, extra=[])

    if outcome == "exception":
        with pytest.raises(OSError, match="child failed to start"):
            run()
    else:
        assert run() == outcome
    assert len(contexts) == 1
    assert not contexts[0].exists()


def test_committed_archive_remains_readable_after_previous_is_deleted(case):
    _, output, _ = case
    _prepare(case, "analyze_climate", "original")
    directory = output / "config/analyze_climate"
    record = read_archive(output, "analyze_climate", "config/analyze_climate")
    payloads = {
        item["archived_path"]: (directory / item["archived_path"]).read_bytes()
        for item in record["source_files"]
    }
    publish_archive(
        output, "analyze_climate", "config/analyze_climate", record, payloads
    )
    previous = list((output / "config").glob(".*.previous"))
    assert len(previous) == 1
    for path in previous:
        assert path.resolve().is_relative_to(output.resolve())
        shutil.rmtree(path)

    assert read_archive(output, "analyze_climate", "config/analyze_climate") == record


@pytest.fixture
def legacy_case(case):
    config, output, catalog = case
    _, context = _prepare(case, "analyze_climate", "seed")
    directory = output / "config/analyze_climate"
    seed = validate_archive(directory)
    sources = tuple(
        CapturedSource(
            item["id"],
            item["role"],
            Path(item["original_path"]),
            (directory / item["archived_path"]).read_bytes(),
            item["git_blob"],
        )
        for item in seed["source_files"]
    )
    for path in (
        directory,
        output / "config/_engine/configuration-snapshots",
        output / "config/_engine/archive-transactions",
    ):
        assert path.resolve().is_relative_to(output.resolve())
        shutil.rmtree(path)
    records = []
    histories = output / "config/_engine/invocations"
    histories.mkdir(parents=True, exist_ok=True)
    for number in range(3):
        record, payloads = run_record_document(
            workflow="analyze_climate",
            invocation_id=f"legacy-{number}",
            owner_kind="workflow",
            owner_id="analyze_climate",
            loaded_config=seed["loaded_config"],
            advanced_settings=seed["advanced_settings"],
            projection=seed["configuration_projection"]["paths"],
            toolbox=seed["toolbox"],
            environment=seed["environment"],
            invocation=seed["invocation"],
            sources=sources,
            referenced_inputs=seed["referenced_inputs"],
            generated_inputs=seed["generated_inputs"],
        )
        publish_archive(
            output, "analyze_climate", "config/analyze_climate", record, payloads
        )
        records.append((directory / "run_record.yml").read_bytes())
        (histories / f"legacy-{number}.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "run_record": file_reference(
                            directory / "run_record.yml", "project_root", output
                        )
                    }
                }
            ),
            encoding="utf-8",
        )
    assert len(list((output / "config").glob(".*.previous"))) == 2
    assert (
        len(
            list(
                (output / "config/_engine/archive-transactions/completed").glob(
                    "*.json"
                )
            )
        )
        == 2
    )
    assert context.exists()
    return config, output, catalog, records


def test_maintenance_reports_without_writes_and_migrates_legacy_history(
    legacy_case, capsys
):
    config, output, catalog, records = legacy_case
    before = _files(output)
    directories = {
        path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_dir()
    }
    expected = maintenance_report(output)
    assert expected["snapshots"] == 0
    assert expected["launch_contexts"] == 1
    assert expected["completed_transactions"] == 2
    assert expected["unavailable_history_references"] == 2
    capsys.readouterr()
    assert maintain_config_archives.main(["--project-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert _files(output) == before
    assert {
        path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_dir()
    } == directories

    execution = output / "config/_engine/execution-configs"
    inputs = {
        name: value
        for name, value in _files(execution).items()
        if not Path(name).name.startswith("launch-")
    }
    original_sources = {path: path.read_bytes() for path in (config, catalog)}
    history = _files(output / "config/_engine/invocations")
    assert maintain_config_archives.main(["--project-dir", str(output), "--apply"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["snapshots"] == 3
    assert report["launch_contexts"] == 0
    assert report["completed_transactions"] == 0
    assert report["unavailable_history_references"] == 2
    assert not list((output / "config").glob(".*.previous"))
    assert not list((output / "config/_engine/archive-transactions").rglob("*.json"))
    snapshots = output / "config/_engine/configuration-snapshots/analyze_climate"
    assert {path.read_bytes() for path in snapshots.glob("*/run_record.yml")} == set(
        records
    )
    for path in snapshots.iterdir():
        validate_archive(path)
    assert _files(execution) == inputs
    assert _files(output / "config/_engine/invocations") == history
    assert {path: path.read_bytes() for path in original_sources} == original_sources
    assert not (output / "config/SETTINGS.md").exists()
    assert "SOURCES.md" in (output / "config/README.md").read_text(encoding="utf-8")
    after = _files(output)
    assert maintain_archives(output) == report
    assert _files(output) == after


@pytest.mark.parametrize("runner", ["individual", "all"])
def test_setup_failure_after_capture_removes_launch_context(case, monkeypatch, runner):
    config, output, _ = case
    original_update = launch.invocation_history.update
    failed = False

    def update(path, record):
        nonlocal failed
        if not failed and record.get("configuration", {}).get("run_record") is not None:
            failed = True
            raise OSError("failed to bind configuration history")
        return original_update(path, record)

    monkeypatch.setattr(launch.invocation_history, "update", update)
    monkeypatch.setattr(
        windows_job,
        "run_project_child",
        lambda *args, **kwargs: pytest.fail("child launched after setup failure"),
    )
    monkeypatch.setattr(
        run_workflows,
        "run_project_child",
        lambda *args, **kwargs: pytest.fail("child launched after setup failure"),
    )
    if runner == "all":
        project = yaml.safe_load(config.read_bytes())
        for name, stanza in project["workflows"].items():
            stanza["enabled"] = name == "analyze_climate"
        config.write_text(yaml.safe_dump(project), encoding="utf-8")

        def run():
            return run_workflows.run(str(config), cores=1, extra=[])

    else:

        def run():
            return launch.run_workflow("analyze_climate", config, output)

    with pytest.raises(OSError, match="failed to bind configuration history"):
        run()
    assert failed
    assert not list(
        (output / "config/_engine/execution-configs").glob("*/*/launch-*.json")
    )


@pytest.mark.parametrize("journal_kind", ["current", "completed"])
def test_maintenance_refuses_unrelated_deletion_target(legacy_case, journal_kind):
    _, output, _, _ = legacy_case
    journals = output / "config/_engine/archive-transactions"
    journal = (
        journals / "analyze_climate.json"
        if journal_kind == "current"
        else next((journals / "completed").glob("*.json"))
    )
    transaction = json.loads(journal.read_text(encoding="utf-8"))
    unrelated = output / "data/must-keep"
    unrelated.mkdir(parents=True)
    sentinel = unrelated / "result.txt"
    sentinel.write_text("retained data", encoding="utf-8")
    transaction["previous"] = unrelated.relative_to(output).as_posix()
    journal.write_text(json.dumps(transaction), encoding="utf-8")
    with pytest.raises(ValueError):
        maintain_archives(output)
    assert sentinel.read_text(encoding="utf-8") == "retained data"


def test_maintenance_retains_rerun_predecessor_closure(legacy_case):
    _, output, _, _ = legacy_case
    predecessor = output / "config/analyze_climate"
    original = validate_archive(predecessor)
    rerun_root = output / "rerun"
    sources, retained, overlay = capture_rerun_predecessor(
        predecessor,
        predecessor_project_root=output,
        new_project_root=rerun_root,
        output_project_dir=str(rerun_root),
    )
    record, payloads = run_record_document(
        workflow="analyze_climate",
        invocation_id="rerun",
        owner_kind="workflow",
        owner_id="analyze_climate",
        loaded_config=original["loaded_config"],
        advanced_settings=original["advanced_settings"],
        projection=original["configuration_projection"]["paths"],
        toolbox=original["toolbox"],
        environment=original["environment"],
        invocation=original["invocation"],
        sources=sources,
        rerun_adjustments=[overlay],
    )
    publish_archive(
        rerun_root,
        "analyze_climate",
        "config/analyze_climate",
        record,
        payloads | retained,
    )
    maintain_archives(rerun_root)
    snapshot = next(
        (rerun_root / "config/_engine/configuration-snapshots/analyze_climate").glob(
            "*/run_record.yml"
        )
    ).parent
    assert validate_archive(snapshot) == record
    assert {name: (snapshot / name).read_bytes() for name in retained} == retained
    current = rerun_root / "config/analyze_climate"
    assert current.resolve().is_relative_to(rerun_root.resolve())
    shutil.rmtree(current)
    assert validate_archive(snapshot) == record


@pytest.mark.parametrize("exit_code", [0, 7])
def test_cleanup_permission_failure_preserves_child_exit_code(
    case, monkeypatch, capsys, exit_code
):
    config, output, _ = case
    original_unlink = Path.unlink
    contexts = []

    def unlink(path, *args, **kwargs):
        if path.name.startswith("launch-"):
            raise PermissionError("launch context is temporarily busy")
        return original_unlink(path, *args, **kwargs)

    def child(command, **kwargs):
        contexts.append(Path(kwargs["env"][launch.CONTEXT_ENV]))
        return exit_code

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(windows_job, "run_project_child", child)
    assert launch.run_workflow("analyze_climate", config, output) == exit_code
    assert len(contexts) == 1
    assert contexts[0].exists()
    assert (contexts[0].parent / "project.yml").is_file()
    assert "cleanup deferred" in capsys.readouterr().err
