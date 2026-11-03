"""Rebuild deletes only checked generated namespaces on disposable projects."""

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from blueearth_cst.shared import invocation_history
from blueearth_cst.shared.project_rebuild import (
    OUTPUT_ROOTS,
    erase_outputs,
    plan_rebuild,
)
from blueearth_cst.shared.workflow_config_snapshot import (
    archive_lock,
    capture_sources,
    publish_archive,
    read_archive,
    run_record_document,
)
from scripts import run_workflows as runner


def _write(path, text="generated"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    config = tmp_path / "project.yml"
    document = {
        "schema_version": 3,
        "project": {"project_dir": str(root)},
        "climate": {"sources": ["era5"], "selected": "era5"},
        "workflows": {
            name: {"enabled": name == "analyze_climate"}
            for name in runner.WORKFLOW_ORDER
        },
    }
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    return SimpleNamespace(root=root, config=config, document=document)


def _save(project):
    project.config.parent.mkdir(parents=True, exist_ok=True)
    project.config.write_text(yaml.safe_dump(project.document), encoding="utf-8")


def _files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_erasure_removes_readonly_generation_snapshot(project):
    snapshot = _write(
        project.root
        / "scenarios/_engine/generation_inputs/67d89e1b0ea7/weathergen_config.yml"
    )
    snapshot.chmod(0o444)
    try:
        plan = plan_rebuild(project.root, project.config)
        erase_outputs(plan, project.config)
        assert not (project.root / "scenarios").exists()
        assert project.config.is_file()
    finally:
        if snapshot.exists():
            snapshot.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize(
    "workflow,owner_kind,target",
    [
        (
            "generate_scenarios",
            "scenario_collection",
            "scenarios/_engine/data/123456abcdef/config",
        ),
        ("simulate_system", "experiment", "experiments/experiment_01/config"),
    ],
)
@pytest.mark.parametrize("missing_target", [False, True])
def test_rebuild_allows_republishing_artifact_archive(
    project, workflow, owner_kind, target, missing_target
):
    owner = "123456abcdef"
    record, payloads = run_record_document(
        workflow=workflow,
        invocation_id="before-rebuild",
        owner_kind=owner_kind,
        owner_id=owner,
        loaded_config={},
        advanced_settings={},
        projection=[],
        toolbox={},
        environment={},
        invocation={
            "entry_point": "test",
            "command": [],
            "targets": [],
            "working_directory": str(project.config.parent),
            "overrides": {},
        },
        sources=capture_sources([("project", "project_config", project.config)]),
    )
    with archive_lock(project.root, "project-execution"):
        publish_archive(project.root, owner, target, record, payloads)
        if missing_target:
            # Reproduce an earlier rebuild that erased outputs but left journals.
            journal = project.root / f"config/_engine/archive-transactions/{owner}.json"
            transaction = journal.read_bytes()
            erase_outputs(plan_rebuild(project.root, project.config), project.config)
            journal.write_bytes(transaction)
        erase_outputs(plan_rebuild(project.root, project.config), project.config)
        publish_archive(project.root, owner, target, record, payloads)
        assert read_archive(project.root, owner, target) == record


def test_erasure_preserves_sources_unrelated_paths_and_the_held_lock(project):
    for relative in OUTPUT_ROOTS:
        _write(project.root / relative / "nested/product.bin")
    staged = project.root / f"config/.build_model.{'a' * 32}.previous"
    _write(staged / "run_record.yml")
    _write(project.root / "config/_engine/archive-transactions/build_model.json")
    transactions = project.root / "config/_engine/archive-transactions"
    completed = _write(
        transactions / "completed/old-artifact.json",
        json.dumps({"target": "scenarios/_engine/data/old/config"}),
    )
    retained = [
        "data/output_locations.csv",
        "data/inputs/weather.nc",
        "models/custom/readme.txt",
        "config/source.yml",
        "config/_engine/invocations/previous.json",
        "config/_engine/configuration-snapshots/build_model/old/run_record.yml",
        "logs/previous/console.log",
        "notes.txt",
    ]
    for relative in retained:
        _write(project.root / relative, relative)
    for relative in ("other.json", "completed/other.json"):
        journal = _write(
            transactions / relative,
            json.dumps({"target": "config/retained/config"}),
        )
        retained.append(journal.relative_to(project.root).as_posix())
    before = {name: (project.root / name).read_bytes() for name in retained}
    with archive_lock(project.root, "project-execution"):
        lock = (
            project.root / "config/_engine/archive-transactions/project-execution.lock"
        )
        identity = lock.stat().st_ino
        plan = plan_rebuild(project.root, project.config)
        erase_outputs(plan, project.config)
        assert lock.stat().st_ino == identity
    assert all(not path.exists() for path in plan.targets)
    assert not completed.exists()
    assert {name: (project.root / name).read_bytes() for name in retained} == before
    assert project.config.is_file()


@pytest.mark.parametrize(
    "kind",
    ["config", "workflow", "dataset", "catalog-dataset", "region", "disabled-workflow"],
)
def test_source_overlap_refuses_before_erasing_any_product(project, kind):
    product = _write(project.root / "scenarios/series/run_01.nc", "keep")
    source = _write(project.root / "data/spatial/input.yml", "value: 1\n")
    if kind == "config":
        project.config = project.root / "config/build_model/user.yml"
    elif kind in {"workflow", "disabled-workflow"}:
        name = "analyze_climate" if kind == "workflow" else "build_model"
        project.document["workflows"][name]["config_path"] = str(source)
    elif kind == "dataset":
        project.document["basin"] = {"output_locations": str(source)}
    elif kind == "region":
        project.document["basin"] = {"region": str({"geom": str(source)})}
    else:
        catalog = _write(
            project.config.parent / "catalog.yml",
            yaml.safe_dump({"local": {"uri": str(source)}}),
        )
        project.document["project"]["catalog"] = str(catalog)
    _save(project)
    with pytest.raises(ValueError, match="source|input"):
        plan_rebuild(project.root, project.config)
    assert product.read_text(encoding="utf-8") == "keep"
    assert source.is_file()


@pytest.mark.parametrize("change", ["new-product", "changed-product", "changed-config"])
def test_changed_plan_is_refused_before_first_deletion(project, change):
    product = _write(project.root / "scenarios/series/run_01.nc", "keep")
    plan = plan_rebuild(project.root, project.config)
    if change == "new-product":
        _write(project.root / "models/hydrology/wflow/new.nc")
    elif change == "changed-product":
        product.write_text("changed", encoding="utf-8")
    else:
        project.config.write_text(
            project.config.read_text() + "# changed\n", encoding="utf-8"
        )
    before = _files(project.root)
    with pytest.raises(ValueError, match="changed"):
        erase_outputs(plan, project.config)
    assert _files(project.root) == before


@pytest.mark.parametrize(
    "relative", ["scenarios", "scenarios/nested", "config/_engine"]
)
def test_linked_targets_or_ancestors_are_refused(project, tmp_path, relative):
    outside = tmp_path / "outside"
    keep = _write(outside / "keep.txt", "source")
    link = project.root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="linked"):
        plan_rebuild(project.root, project.config)
    assert keep.read_text(encoding="utf-8") == "source"


@pytest.mark.parametrize("root_kind", ["drive", "home", "repository"])
def test_unsafe_root_is_refused(project, monkeypatch, root_kind):
    root = project.root
    if root_kind == "drive":
        root = Path(root.anchor)
    elif root_kind == "home":
        monkeypatch.setattr(Path, "home", lambda: root)
    else:
        _write(root / ".git", "gitdir: elsewhere")
    with pytest.raises(ValueError, match="root"):
        plan_rebuild(root, project.config)


@pytest.mark.parametrize("preview", ["--dry-run", "--dryrun", "-n"])
def test_runner_preview_has_no_writes_prompt_or_workflows(
    project, monkeypatch, capsys, preview
):
    product = _write(project.root / "experiments/old/results.csv")
    before = _files(project.root)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("preview prompted"))
    monkeypatch.setattr(
        runner, "_run_owned", lambda *_a, **_kw: pytest.fail("preview ran workflows")
    )
    assert (
        runner.main(["--config", str(project.config), "--rebuild", "--", preview]) == 0
    )
    output = capsys.readouterr().out
    assert "erase  experiments" in output and "Preview only" in output
    assert product.is_file() and _files(project.root) == before


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", None])
def test_runner_confirmation_defaults_to_no_without_writes(
    project, monkeypatch, answer
):
    _write(project.root / "scenarios/keep.nc")
    before = _files(project.root)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    def respond(prompt):
        assert "[y/N]" in prompt
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", respond)
    assert runner.run(str(project.config), 1, [], rebuild=True) == 0
    assert _files(project.root) == before


@pytest.mark.parametrize("confirmation", ["y", "YES", "--yes"])
def test_runner_erases_before_normal_admission_and_records_reset(
    project, monkeypatch, confirmation
):
    product = _write(project.root / "experiments/old/_engine/simulation.json")
    _write(project.root / "scenarios/_engine/data/old/series.nc")
    source = _write(project.root / "inputs/keep.csv", "source")
    monkeypatch.setattr(
        runner.sys, "stdin", SimpleNamespace(isatty=lambda: confirmation != "--yes")
    )
    monkeypatch.setattr("builtins.input", lambda *_: confirmation)
    calls = []

    def run_owned(*args, **kwargs):
        calls.append(kwargs)
        assert not product.exists()
        assert not (project.root / "scenarios").exists()
        assert source.read_text() == "source"
        path, record = kwargs["history_pair"]
        assert record["rebuild"]["status"] == "completed"
        invocation_history.finish(path, record, exit_code=0)
        return 0

    monkeypatch.setattr(runner, "_run_owned", run_owned)
    assert (
        runner.main(
            [
                "--config",
                str(project.config),
                "--rebuild",
                *(["--yes"] if confirmation == "--yes" else []),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    records = list((project.root / "config/_engine/invocations").glob("*.json"))
    assert json.loads(records[0].read_text())["rebuild"]["targets"] == [
        "scenarios",
        "experiments",
    ]


def test_noninteractive_rebuild_requires_yes(project, monkeypatch):
    _write(project.root / "scenarios/keep.nc")
    before = _files(project.root)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(ValueError, match="--yes"):
        runner.run(str(project.config), 1, [], rebuild=True)
    assert _files(project.root) == before


@pytest.mark.parametrize(
    "options",
    [
        {"yes": True},
        {"rebuild": True, "new_experiment": True},
        {"rebuild": True, "recover_reservation": True},
    ],
)
def test_incompatible_runner_flags_do_not_erase(project, options):
    product = _write(project.root / "scenarios/keep.nc")
    with pytest.raises(ValueError):
        runner.run(str(project.config), 1, [], **options)
    assert product.is_file()


@pytest.mark.parametrize(
    "extra",
    [
        ["--unlock"],
        ["--config", "project.project_dir=elsewhere"],
        ["--directory", "elsewhere"],
        ["--delete-all-output"],
        ["all"],
    ],
)
def test_forwarded_redirection_or_bookkeeping_is_refused_before_erasure(project, extra):
    product = _write(project.root / "scenarios/keep.nc")
    with pytest.raises(ValueError, match="does not support"):
        runner.run(str(project.config), 1, extra, rebuild=True, yes=True)
    assert product.is_file()


def test_deletion_failure_is_recorded_and_no_workflow_starts(project, monkeypatch):
    from blueearth_cst.shared import project_rebuild

    product = _write(project.root / "scenarios/keep.nc")

    def fail(*args):
        raise PermissionError("fixture denies erasure")

    monkeypatch.setattr(project_rebuild, "erase_outputs", fail)
    monkeypatch.setattr(
        runner,
        "_run_owned",
        lambda *_a, **_kw: pytest.fail("started after failed deletion"),
    )
    with pytest.raises(ValueError, match="stopped"):
        runner.run(str(project.config), 1, [], rebuild=True, yes=True)
    assert product.is_file()
    record = json.loads(
        next((project.root / "config/_engine/invocations").glob("*.json")).read_text()
    )
    assert record["rebuild"]["status"] == "failed"
    assert record["status"] == "failed"


@pytest.mark.parametrize("missing", ["build_model", "generate_scenarios"])
def test_rebuild_refuses_wf4_without_its_producers(project, missing):
    for name in ("build_model", "generate_scenarios", "simulate_system"):
        project.document["workflows"][name]["enabled"] = name != missing
    _save(project)
    product = _write(project.root / "experiments/old/results.csv")
    with pytest.raises(ValueError, match="requires enabled"):
        runner.run(str(project.config), 1, [], rebuild=True, yes=True)
    assert product.is_file()


def test_rebuild_refuses_indicators_only_before_erasure(project):
    settings = _write(
        project.config.parent / "simulation.yml", "operations: [indicators]\n"
    )
    for name in ("build_model", "generate_scenarios", "simulate_system"):
        project.document["workflows"][name]["enabled"] = True
    project.document["workflows"]["simulate_system"]["config_path"] = str(settings)
    _save(project)
    product = _write(project.root / "experiments/old/results.csv")
    with pytest.raises(ValueError, match="indicators-only"):
        runner.run(str(project.config), 1, [], rebuild=True, yes=True)
    assert product.is_file()


def test_catalog_dataset_locations_are_checked_without_opening_datasets(
    project, monkeypatch
):
    dataset = project.config.parent / "external" / "parameter_data"
    catalog = _write(
        project.config.parent / "catalog.yml",
        yaml.safe_dump({"source": {"uri": str(dataset)}}),
    )
    project.document["project"]["catalog"] = str(catalog)
    _save(project)
    original = Path.is_file

    def check(path):
        if path == dataset:
            pytest.fail("planner tried to access a catalog dataset")
        return original(path)

    monkeypatch.setattr(Path, "is_file", check)
    assert plan_rebuild(project.root, project.config).targets == ()


@pytest.mark.parametrize("locator", ["root-relative", "file-uri"])
def test_catalog_roots_and_encoded_file_uris_cannot_hide_an_input_overlap(
    project, locator
):
    product = _write(project.root / "scenarios/input data.nc", "keep")
    catalog_body = {"local": {"uri": product.as_uri()}}
    if locator == "root-relative":
        catalog_body = {
            "meta": {"root": str(project.root / "inputs")},
            "local": {"uri": "../scenarios/input data.nc"},
        }
    catalog = _write(
        project.config.parent / "catalog.yml", yaml.safe_dump(catalog_body)
    )
    project.document["project"]["catalog"] = str(catalog)
    _save(project)
    with pytest.raises(ValueError, match="input overlaps"):
        plan_rebuild(project.root, project.config)
    assert product.read_text() == "keep"


def test_runner_cannot_hide_a_linked_root_by_resolving_it(project, tmp_path):
    _write(project.root / "scenarios/keep.nc")
    alias = tmp_path / "linked-project"
    try:
        alias.symlink_to(project.root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    project.document["project"]["project_dir"] = str(alias)
    _save(project)
    with pytest.raises(ValueError, match="linked project root"):
        runner.run(str(project.config), 1, [], rebuild=True, yes=True)
    assert (project.root / "scenarios/keep.nc").is_file()


def test_nested_repository_is_ambiguous_ownership_and_is_preserved(project):
    marker = _write(project.root / "scenarios/custom/.git", "gitdir: elsewhere")
    product = _write(project.root / "data/spatial/keep.nc")
    with pytest.raises(ValueError, match="repository metadata"):
        plan_rebuild(project.root, project.config)
    assert marker.is_file() and product.is_file()


def test_legacy_inputs_are_also_checked_against_the_child_runtime_directory(
    project, tmp_path, monkeypatch
):
    from blueearth_cst.shared import project_rebuild

    runtime = tmp_path / "runtime"
    product = _write(project.root / "data/climate/source.nc", "keep")
    _write(runtime / "catalog.yml", yaml.safe_dump({"local": {"uri": str(product)}}))
    project.document["schema_version"] = 2
    project.document["project"]["catalog"] = "catalog.yml"
    _save(project)
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(project_rebuild, "TOOLBOX_ROOT", runtime)
    with pytest.raises(ValueError, match="input overlaps"):
        plan_rebuild(project.root, project.config)
    assert product.read_text() == "keep"


@pytest.mark.workflow_contract
@pytest.mark.usefixtures("isolated_snakemake")
@pytest.mark.skipif(
    os.name != "nt", reason="run_project_child refuses project writers off Windows"
)
def test_rebuild_executes_a_real_producer_at_the_existing_output_path(
    project, monkeypatch
):
    """Use a tiny real DAG to distinguish execution from retained-output reuse."""
    from blueearth_cst.shared import configuration_archive, workflow_archive_launch

    output = _write(project.root / "data/spatial/rebuilt.txt", "old output")
    keep = _write(project.root / "inputs/source.csv", "source data")
    snakefile = project.config.parent / "rebuild-smoke.smk"
    snakefile.write_text(
        "from pathlib import Path\n"
        f"target = {str(output)!r}\n"
        "rule all:\n"
        "    input: target\n"
        "rule produce:\n"
        "    output: target\n"
        "    run:\n"
        "        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)\n"
        "        Path(output[0]).write_text('fresh execution', encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(runner.SNAKEFILE, "analyze_climate", str(snakefile))
    # The toy DAG has no scientific configuration archive; preserve the actual
    # launcher, lock, history, Snakemake, producer, and child exit handling.
    monkeypatch.setattr(
        workflow_archive_launch,
        "prepare_workflow",
        lambda _name, config, _root, **_kw: (Path(config), Path(config)),
    )
    monkeypatch.setattr(runner, "_bind_workflow_archive", lambda *_a: None)
    monkeypatch.setattr(
        configuration_archive, "remove_launch_context", lambda *_a: None
    )
    assert runner.run(str(project.config), 1, [], rebuild=True, yes=True) == 0
    assert output.read_text(encoding="utf-8") == "fresh execution"
    assert keep.read_text(encoding="utf-8") == "source data"
    record = next(
        document
        for path in (project.root / "config/_engine/invocations").glob("*.json")
        if (document := json.loads(path.read_text())).get("rebuild")
    )
    assert record["rebuild"]["status"] == "completed"
    assert record["status"] == "succeeded"
