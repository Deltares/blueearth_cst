"""WF0--WF2 exact capture before Snakefile parsing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from blueearth_cst.shared import workflow_archive_launch as launch
from blueearth_cst.shared.config_paths import (
    resolve_project_paths,
    resolve_workflow_paths,
)
from blueearth_cst.shared.workflow_archive_launch import (
    CONTEXT_ENV,
    prepare_workflow,
    require_capture,
    split_config_overrides,
)
from blueearth_cst.shared.workflow_config_snapshot import (
    read_archive,
    resolve_file_reference,
)
from scripts import run_workflows

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workflow_name", launch.WORKFLOWS)
def test_execution_inputs_ignore_source_locations_and_other_enable_flags(
    tmp_path, workflow_name
):
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    project_root = tmp_path / "output"
    project["project"]["project_dir"] = str(project_root)
    catalog = tmp_path / "catalog.yml"
    catalog.write_bytes((ROOT / "config/catalogs/deltares_data.yml").read_bytes())
    project["project"]["catalog"] = [str(catalog)]
    stanza = project["workflows"][workflow_name]
    workflow_file = tmp_path / "workflow.yml"
    workflow_file.write_text(
        yaml.safe_dump(
            resolve_workflow_paths(
                yaml.safe_load(
                    (ROOT / "test_case" / stanza["config_path"]).read_bytes()
                ),
                ROOT / "test_case" / stanza["config_path"],
                workflow_name,
                project["schema_version"],
            )
        ),
        encoding="utf-8",
    )
    stanza["config_path"] = str(workflow_file)
    project_file = tmp_path / "project.yml"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")

    def prepare(path):
        return prepare_workflow(
            workflow_name,
            path,
            project_root,
            command=["snakemake", "all"],
            targets=["all"],
        )[0]

    original = prepare(project_file)
    assert not (project_root / "config/runs").exists()
    assert (
        "edit the source"
        in (project_root / "config/README.md").read_text(encoding="utf-8").lower()
    )
    original_bytes = original.read_bytes()
    original_mtime = original.stat().st_mtime_ns
    for name in ("generate_scenarios", "simulate_system"):
        project["workflows"][name]["enabled"] = not project["workflows"][name][
            "enabled"
        ]
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    assert prepare(project_file) == original

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    moved_workflow = relocated / "renamed-workflow.yml"
    moved_workflow.write_bytes(workflow_file.read_bytes() + b"\n# relocated config\n")
    moved_catalog = relocated / "renamed-catalog.yaml"
    moved_catalog.write_bytes(catalog.read_bytes())
    stanza["config_path"] = moved_workflow.name
    project["project"]["catalog"] = [str(moved_catalog)]
    moved_project = relocated / "renamed-project.yml"
    moved_project.write_text(yaml.safe_dump(project), encoding="utf-8")
    assert prepare(moved_project) == original
    assert original.read_bytes() == original_bytes
    assert original.stat().st_mtime_ns == original_mtime
    record = read_archive(project_root, workflow_name, f"config/{workflow_name}")
    archived_project = next(
        item for item in record["source_files"] if item["role"] == "project_config"
    )
    assert archived_project["original_path"] == str(moved_project)

    project["basin"]["resolution"] *= 2
    moved_project.write_text(yaml.safe_dump(project), encoding="utf-8")
    changed = prepare(moved_project)
    assert changed != original
    moved_catalog.write_bytes(moved_catalog.read_bytes() + b"\n# dependency changed\n")
    assert prepare(moved_project) != changed
    overridden_basin = dict(project["basin"], resolution=0.025)
    overridden, _ = prepare_workflow(
        workflow_name,
        moved_project,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
        overrides={"basin": overridden_basin},
    )
    assert overridden != prepare(moved_project)
    assert yaml.safe_load(overridden.read_bytes())["basin"]["resolution"] == 0.025


def test_preparse_capture_survives_source_edit(tmp_path, capsys):
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    source_workflow = ROOT / "test_case/project_config_rapid_analyze_climate.yml"
    workflow = tmp_path / source_workflow.name
    workflow.write_bytes(source_workflow.read_bytes())
    catalog = tmp_path / "catalog.yml"
    catalog.write_text("meta:\n  roots: [C:/data]\nera5:\n  uri: meteo/data.nc\n")
    project_root = tmp_path / "output"
    project["project"]["project_dir"] = str(project_root)
    project["project"]["catalog"] = [str(catalog)]
    project["workflows"]["analyze_climate"]["config_path"] = str(workflow)
    project_file = tmp_path / "project_config_test.yml"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")
    source_bytes = workflow.read_bytes()

    execution, context = prepare_workflow(
        "analyze_climate",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )
    workflow.write_bytes(source_bytes + b"\n# late comment\n")
    record = read_archive(project_root, "analyze_climate", "config/analyze_climate")
    assert record["schema_version"] == "run-record/2"
    archived = next(
        entry
        for entry in record["source_files"]
        if entry["role"] == "workflow_config_analyze_climate"
    )
    assert (
        project_root / "config/analyze_climate" / archived["archived_path"]
    ).read_bytes() == source_bytes
    execution_project = yaml.safe_load(execution.read_bytes())
    for name, stanza in execution_project["workflows"].items():
        if name != "analyze_climate":
            assert "config_path" not in stanza
    assert yaml.safe_load(
        Path(
            execution_project["workflows"]["analyze_climate"]["config_path"]
        ).read_bytes()
    ) == yaml.safe_load(source_bytes)
    assert json.loads(context.read_text())["archive_id"] == record["archive_id"]
    assert "[config_composition] skipped unreadable" not in capsys.readouterr().out


def test_shared_dependency_stages_once_across_workflows(tmp_path):
    """A dependency shared by two workflows resolves to one execution path.

    Regression for the bug where each workflow staged its own copy of the same
    data catalog under its own bundle digest: WF0/WF1/WF2 declare byte-identical
    shared-foundation rules (delineate_region and friends) over that catalog,
    so a per-workflow staging path made Snakemake see a changed input set --
    and rebuild the whole foundation -- every time a different workflow last
    ran, even with nothing to redo.
    """
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    project_root = tmp_path / "output"
    project["project"]["project_dir"] = str(project_root)
    project["project"]["catalog"] = [str((ROOT / "config/catalogs/deltares_data.yml"))]
    locations = tmp_path / "output_locations.csv"
    locations.write_text("id,x,y\noutlet,1,2\n", encoding="utf-8")
    project["basin"]["output_locations"] = str(locations)
    for name, stanza in project["workflows"].items():
        if "config_path" in stanza:
            stanza["config_path"] = str(
                (ROOT / "test_case" / stanza["config_path"]).resolve()
            )
    project_file = tmp_path / "project_config_test.yml"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")

    execution_a, _ = prepare_workflow(
        "analyze_climate",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )
    execution_b, _ = prepare_workflow(
        "build_model",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )
    execution_c, _ = prepare_workflow(
        "analyze_projections",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )
    catalog_a = yaml.safe_load(execution_a.read_bytes())["project"]["catalog"][0]
    catalog_b = yaml.safe_load(execution_b.read_bytes())["project"]["catalog"][0]
    catalog_c = yaml.safe_load(execution_c.read_bytes())["project"]["catalog"][0]
    assert catalog_a == catalog_b == catalog_c
    assert Path(catalog_a).is_relative_to(
        project_root / "config/_engine/execution-configs/_shared"
    )
    locations_a = yaml.safe_load(execution_a.read_bytes())["basin"]["output_locations"]
    locations_b = yaml.safe_load(execution_b.read_bytes())["basin"]["output_locations"]
    locations_c = yaml.safe_load(execution_c.read_bytes())["basin"]["output_locations"]
    assert locations_a == locations_b == locations_c
    assert Path(locations_a).is_relative_to(
        project_root / "config/_engine/execution-configs/_shared"
    )


def test_execution_config_schema_moves_immutable_bundle_directory(
    tmp_path, monkeypatch
):
    """A staging-format change gets a new bundle path for unchanged sources."""
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    project_root = tmp_path / "output"
    project["project"]["project_dir"] = str(project_root)
    project["project"]["catalog"] = [str((ROOT / "config/catalogs/deltares_data.yml"))]
    stanza = project["workflows"]["analyze_climate"]
    stanza["config_path"] = str((ROOT / "test_case" / stanza["config_path"]).resolve())
    project_file = tmp_path / "project_config_test.yml"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")

    monkeypatch.setattr(launch, "EXECUTION_CONFIG_SCHEMA_VERSION", "legacy")
    legacy, _ = prepare_workflow(
        "analyze_climate",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )
    monkeypatch.setattr(launch, "EXECUTION_CONFIG_SCHEMA_VERSION", "current")
    current, _ = prepare_workflow(
        "analyze_climate",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )

    assert legacy.parent != current.parent
    assert legacy.is_file()
    assert current.is_file()


def test_projection_catalog_and_store_index_stage_as_siblings(tmp_path):
    """`cmip6_data.yml` and `cmip6_store_index.json` must land in ONE shared
    subfolder even though their bytes (and therefore per-file content hashes)
    differ.

    Regression: staging keyed each dependency by its OWN content hash, so the
    catalog and its store index -- deliberately bucketed together under
    "projection_catalog" -- still landed in two DIFFERENT hash subfolders
    whenever their bytes differed (always, in practice: one is YAML, the
    other a generated JSON crawl). `analyze_projections.smk` finds the index
    by guessing "beside the catalog" (`STORE_INDEX = Path(DATA_SOURCES).parent
    / "cmip6_store_index.json"`, a documented D12 invariant, not a
    coincidence), so the guess silently missed and every series was staged
    with empty pins -- degrading not just physical-identity digesting but
    calendar detection, which refuses outright when it has no pin to read a
    store from.
    """
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    project_root = tmp_path / "output"
    project["project"]["project_dir"] = str(project_root)
    project["project"]["catalog"] = [str((ROOT / "config/catalogs/deltares_data.yml"))]
    stanza = project["workflows"]["analyze_projections"]
    stanza["config_path"] = str((ROOT / "test_case" / stanza["config_path"]).resolve())
    catalog = tmp_path / "cmip6_data.yml"
    index = tmp_path / "cmip6_store_index.json"
    catalog.write_text(
        (ROOT / "config/catalogs/cmip6_data.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # Deliberately different bytes from the catalog, as a real crawl output is.
    index.write_text('{"sources": {}}', encoding="utf-8")
    workflow_config = yaml.safe_load(
        (ROOT / "test_case" / stanza["config_path"]).read_text()
    )
    workflow_config["catalog"] = str(catalog)
    workflow_file = tmp_path / "project_config_rapid_analyze_projections.yml"
    workflow_file.write_text(yaml.safe_dump(workflow_config), encoding="utf-8")
    stanza["config_path"] = str(workflow_file)
    project_file = tmp_path / "project_config_test.yml"
    project_file.write_text(yaml.safe_dump(project), encoding="utf-8")

    execution, _ = prepare_workflow(
        "analyze_projections",
        project_file,
        project_root,
        command=["snakemake", "all"],
        targets=["all"],
    )
    execution_workflow = yaml.safe_load(
        Path(
            yaml.safe_load(execution.read_bytes())["workflows"]["analyze_projections"][
                "config_path"
            ]
        ).read_bytes()
    )
    staged_catalog = Path(execution_workflow["catalog"])
    staged_index = staged_catalog.parent / "cmip6_store_index.json"
    assert staged_index.is_file()
    assert staged_index.read_text(encoding="utf-8") == index.read_text(encoding="utf-8")


def test_raw_execution_refuses_without_capture(monkeypatch, tmp_path):
    monkeypatch.delenv(CONTEXT_ENV, raising=False)
    with pytest.raises(ValueError, match="pre-parse source capture"):
        require_capture("build_model", str(tmp_path / "project.yml"))
    require_capture("build_model", str(tmp_path / "project.yml"), dry_run=True)


def test_config_overrides_are_captured_before_forwarding():
    overrides, forwarded = split_config_overrides(
        ["--config", "debug=true", "--keep-going"]
    )
    assert overrides == {"debug": True}
    assert forwarded == ["--keep-going"]
    with pytest.raises(ValueError, match="--configfile"):
        split_config_overrides(["--configfile", "other.yml"])


def test_all_workflow_wrapper_routes_wf0_through_capture(tmp_path, monkeypatch):
    project = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    output = tmp_path / "output"
    project["project"]["project_dir"] = str(output)
    for name, stanza in project["workflows"].items():
        stanza["enabled"] = name == "analyze_climate"
        if "config_path" in stanza:
            stanza["config_path"] = str(
                (ROOT / "test_case" / stanza["config_path"]).resolve()
            )
    config = tmp_path / "project_config_test.yml"
    config.write_text(yaml.safe_dump(project), encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        run_workflows,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )
    assert run_workflows.run(str(config), cores=1, extra=[]) == 0
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[command.index("--configfile") + 1] != str(config)
    assert CONTEXT_ENV in kwargs["env"]
    record = read_archive(output, "analyze_climate", "config/analyze_climate")
    assert record["schema_version"] == "run-record/2"
    assert record["invocation"]["entry_point"] == "scripts/run_workflows.py"
    invocation_paths = (output / "config/_engine/invocations").glob("*.json")
    invocations = [
        json.loads(path.read_text(encoding="utf-8")) for path in invocation_paths
    ]
    parent = next(item for item in invocations if item["workflow"] is None)
    child = next(item for item in invocations if item["workflow"] == "analyze_climate")
    assert child["invocation_id"] == record["invocation_id"]
    assert child["parent_invocation_id"] == parent["invocation_id"]
    assert resolve_file_reference(
        child["configuration"]["run_record"], {"project_root": output}
    ).is_relative_to(output / "config/_engine/configuration-snapshots/analyze_climate")


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["snakemake", "all", "--dry-run"], True),
        (["snakemake", "all", "-n"], True),
        (["snakemake", "all", "--unlock"], True),
        (["snakemake", "all", "--unlock", "--dry-run"], True),
        (["snakemake", "all", "--touch"], True),
        (["snakemake", "all", "--delete-all-output"], True),
        (["snakemake", "--list-rules"], True),
        (["snakemake", "all", "-c", "3"], False),
        (["snakemake", "all", "--forceall"], False),
    ],
)
@pytest.mark.parametrize("workflow_name", launch.WORKFLOWS)
def test_non_producing_invocations_need_no_capture(
    argv, expected, workflow_name, monkeypatch
):
    from blueearth_cst.shared.workflow_archive_launch import non_producing_invocation

    assert non_producing_invocation(argv) is expected
    monkeypatch.delenv(CONTEXT_ENV, raising=False)
    monkeypatch.setattr("sys.argv", argv)
    if expected:
        assert require_capture(workflow_name, "project.yml") is None
    else:
        with pytest.raises(ValueError, match="requires scripts/run_workflow.py"):
            require_capture(workflow_name, "project.yml")
