"""Case scaffolding checks using isolated, representative case repositories."""

import shutil
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.config_composition import WORKFLOW_NAMES, load_composed_config
from scripts.create_case import (
    _WORKFLOW_LABELS,
    _annotate_config,
    _template_comments,
    _yaml_parameters,
    main,
)


@pytest.fixture(autouse=True)
def template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    toolbox = tmp_path / "toolbox"
    monkeypatch.setattr("scripts.create_case.TOOLBOX_ROOT", toolbox)
    template = toolbox / "config/templates/example-basin"
    (template / "data").mkdir(parents=True)
    (template / "project_config.yml").write_text(
        "# Keep scientific comments\nschema_version: 2\n"
        "project:\n  project_dir: old-output\n  catalog: config/catalogs/data.yml\n"
        "basin:\n  region: \"{'subbasin': [1, 2], 'uparea': 100}\"\n"
        "  output_locations: old-outlet\n  resolution: 0.00833\n"
        "climate:\n  selected: era5\n"
        "workflows:\n  analyze_climate:\n    enabled: true\n"
        "  build_model:\n    enabled: false\n"
        "    config_path: project_config_build_model.yml\n",
        encoding="utf-8",
    )
    (template / "project_config_build_model.yml").write_text(
        "observations:\n  river discharge: null\n"
        "engine:\n  build_config: config/defaults/wflow_build_model.yml\n"
        "simulation_window: {start: 2010, end: 2016}\n",
        encoding="utf-8",
    )
    (template / "data" / "output_locations.csv").write_text(
        "station_name,x,y\noutlet,1,2\n", encoding="utf-8"
    )
    (template / "README.md").write_text("Old example\n", encoding="utf-8")
    (template / "notes.md").write_text("Old success\n", encoding="utf-8")
    return template


@pytest.fixture
def cases(tmp_path: Path) -> Path:
    cases = tmp_path / "cases"
    (cases / "applications").mkdir(parents=True)
    (cases / "cst-applications.md").write_bytes(
        b"Inventory\r\n\r\n"
        b"| application | project | date-created | cst-version | purpose |\r\n"
        b"|---|---|---|---|---|\r\n"
        b"| [old](applications/old/notes.md) | | 2020-01-01 | | Old |\r\n"
    )
    return cases


def args(cases: Path) -> list[str]:
    return [
        "new-basin",
        "--cases-root",
        str(cases),
        "--longitude",
        "12.5",
        "--latitude",
        "-3",
        "--purpose",
        "Test | basin",
    ]


@pytest.mark.parametrize("dry_run", [False, True])
def test_create_preserves_template_and_leaves_inventory(
    cases: Path, template: Path, dry_run: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    before = (cases / "cst-applications.md").read_bytes()
    assert main(args(cases) + (["--dry-run"] if dry_run else [])) == 0
    assert f"Outputs: {cases / 'runs/new-basin'}\n" in capsys.readouterr().out
    application = cases / "applications" / "new-basin"
    assert not (cases / "runs").exists()
    if dry_run:
        assert not application.exists()
        assert (cases / "cst-applications.md").read_bytes() == before
        return
    config_text = (application / "project_config.yml").read_text()
    config = yaml.safe_load(config_text)
    assert config["project"]["project_dir"] == (cases / "runs/new-basin").as_posix()
    assert (
        config["basin"]["output_locations"]
        == (application / "data/output_locations.csv").as_posix()
    )
    assert config["basin"]["region"] == "{'subbasin': [12.5, -3.0], 'uparea': 100}"
    assert config["climate"]["selected"] == "era5"
    assert config["workflows"]["build_model"]["enabled"] is False
    assert "# Keep scientific comments" in config_text
    model_text = (application / "project_config_build_model.yml").read_text(
        encoding="utf-8"
    )
    assert (
        "# Configure initial model construction. Input: HydroMT YAML file path.\n"
        "  build_config: config/defaults/wflow_build_model.yml"
    ) in model_text
    assert (
        "# start: First simulation water year. Input: integer year, <= end."
        in model_text
    )
    assert yaml.safe_load(model_text)["simulation_window"] == {
        "start": 2010,
        "end": 2016,
    }
    assert "outlet,12.5,-3.0" in (application / "data/output_locations.csv").read_text()
    assert "configured; not run" in (application / "notes.md").read_text()
    assert not (application / "README.md").exists()
    instructions = (application / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    created = next(line for line in instructions.splitlines() if "| Created |" in line)
    timestamp = created.split("|")[2].strip()
    assert datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S %z").utcoffset() is not None
    for name, title, workflow in (
        (
            "project_config.yml",
            "Project configuration",
            "All workflows — shared settings and workflow enablement",
        ),
        (
            "project_config_build_model.yml",
            "Model building configuration",
            "WF1 — Model building (build_model)",
        ),
    ):
        text = (application / name).read_text(encoding="utf-8")
        assert text.splitlines()[1:5] == [
            f"# {title} — {name}",
            f"# Created: {timestamp}",
            "# Project: new-basin",
            f"# Workflow: {workflow}",
        ]
    assert "| Status |" not in instructions
    assert f'$projectDir = "{cases.as_posix()}/runs/new-basin"' in instructions
    assert instructions.rsplit("\n## ", 1)[1].startswith("Configuration")
    assert "--cores 3 -- --rerun-incomplete" in instructions
    assert "--cores 3 -- --unlock" in instructions
    assert "Set-Location" not in instructions
    assert "Toolbox version at creation: `" in instructions
    assert "Input datasets are external" in instructions
    assert "The starter enables WF0 alone" in instructions
    assert (
        "completed `_engine/simulation.json` record, even for a dry run" in instructions
    )
    for chapter in ("new-case", "configuration", "running", "indicators", "outputs"):
        assert (
            f"<{template.parents[2].as_posix()}/docs/guide/{chapter}.md>"
            in instructions
        )
    notes = (application / "notes.md").read_text(encoding="utf-8")
    assert "Purpose: Test | basin" in notes
    assert "CST version at creation: " in notes
    # Cases are registered by hand only after every workflow completes.
    assert (cases / "cst-applications.md").read_bytes() == before
    assert main(args(cases)) == 2


@pytest.mark.parametrize("dry_run", [False, True])
def test_create_preserves_existing_output_tree(cases: Path, dry_run: bool) -> None:
    output_root = cases / "runs/new-basin"
    old_output = output_root / "active" / "retained-response.csv"
    old_output.parent.mkdir(parents=True)
    old_output.write_bytes(b"retained native response\n")
    before = {
        item.relative_to(output_root): item.read_bytes()
        for item in output_root.rglob("*")
        if item.is_file()
    }

    assert main(args(cases) + (["--dry-run"] if dry_run else [])) == 0

    after = {
        item.relative_to(output_root): item.read_bytes()
        for item in output_root.rglob("*")
        if item.is_file()
    }
    assert after == before


@pytest.mark.parametrize("workflow", _WORKFLOW_LABELS)
def test_headers_follow_custom_paths_and_replace_inherited_metadata(
    cases: Path, template: Path, workflow: str
) -> None:
    project = template / "project_config.yml"
    config = yaml.safe_load(project.read_text(encoding="utf-8"))
    config["workflows"] = {
        workflow: {
            "enabled": workflow == "analyze_climate",
            "config_path": "custom.yaml",
        }
    }
    project.write_text(yaml.safe_dump(config), encoding="utf-8")
    border = "# " + "-" * 77
    body = "# Keep this explanation\nsettings: {}\n"
    (template / "custom.yaml").write_text(
        f"{border}\n# Old configuration — custom.yaml\n"
        "# Created: 2020-01-01 00:00:00 +00:00\n# Project: old-basin\n"
        f"# Workflow: old\n{border}\n\n{body}",
        encoding="utf-8",
    )

    assert main(args(cases)) == 0

    text = (cases / "applications/new-basin/custom.yaml").read_text(encoding="utf-8")
    code, label = _WORKFLOW_LABELS[workflow]
    assert f"# Workflow: {code} — {label} ({workflow})\n" in text
    assert text.count("# Created:") == 1
    assert "old-basin" not in text
    assert "# Keep this explanation\n" in text
    assert text.endswith("settings: {}\n")
    assert yaml.safe_load(text) == {"settings": {}}


def test_parameter_comments_preserve_flow_yaml_aliases_and_existing_help():
    text = (
        "# Existing explanation\n"
        "window: &period {start: 2010, end: 2016}\n"
        "copy: *period\n"
        "description: |\n  Keep this multiline value.\n"
    )
    comments = {
        ("window",): "# Select inclusive simulation years. Input: year mapping.",
        ("window", "start"): "# First simulation year. Input: integer year.",
        ("window", "end"): "# Last simulation year. Input: integer year.",
    }
    annotated = _annotate_config(text, comments)
    assert yaml.safe_load(annotated) == yaml.safe_load(text)
    assert "# Existing explanation\n" in annotated
    assert "# start: First simulation year. Input: integer year.\n" in annotated
    assert "# end: Last simulation year. Input: integer year.\n" in annotated
    assert _annotate_config(annotated, comments) == annotated


@pytest.mark.parametrize(
    "workflow,text,expected",
    [
        (
            "analyze_projections",
            "variables:\n  custom_temperature:\n    source: temp\n",
            "# Select the catalog's renamed variable. Input: post-rename variable name, such as temp.",
        ),
        (
            "simulate_system",
            "indicator_set_definitions:\n  hydrology:\n    config_path: hydrology.yml\n",
            "# Locate the indicator set definition. Input: YAML path relative to this file or absolute.",
        ),
    ],
)
def test_parameter_comments_support_user_named_entries(workflow, text, expected):
    template = (
        Path(__file__).resolve().parents[1]
        / "config/templates/example-basin"
        / f"project_config_{workflow}.yml"
    )
    annotated = _annotate_config(text, _template_comments(template))
    assert expected in annotated
    assert yaml.safe_load(annotated) == yaml.safe_load(text)


@pytest.mark.parametrize(
    "template",
    sorted(
        (Path(__file__).resolve().parents[1] / "config/templates/example-basin").glob(
            "project_config*.yml"
        )
    )
    + [
        Path(__file__).resolve().parents[1]
        / "config/templates/indicator_set.template.yml"
    ],
    ids=lambda path: path.name,
)
def test_template_parameter_help_is_complete_and_concise(template: Path):
    text = template.read_text(encoding="utf-8")
    comments = _template_comments(template)
    assert comments
    annotated = _annotate_config(text, comments)
    assert annotated == text, "Every active parameter must already carry template help"
    for comment in comments.values():
        definition = comment.split(". Input:")[0].split(". Options:")[0]
        assert len(definition.removeprefix("# ").split()) <= 10, comment
    assert list(_yaml_parameters(yaml.compose(text)))


@pytest.mark.parametrize("relative", ["runs", "runs/new-basin"])
@pytest.mark.parametrize("link_check", ["is_symlink", "is_junction"])
def test_output_root_link_is_refused(
    cases: Path,
    relative: str,
    link_check: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    linked = cases / relative
    original = getattr(Path, link_check)

    def is_link(path: Path) -> bool:
        return path == linked or original(path)

    monkeypatch.setattr(Path, link_check, is_link)

    assert main(args(cases)) == 2
    assert (
        "Destination must not be a symbolic link or junction" in capsys.readouterr().err
    )
    assert not (cases / "applications/new-basin").exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("slug", "../bad"),
        ("slug", "con"),
        ("--latitude", "nan"),
        ("--longitude", "181"),
        ("--purpose", " "),
    ],
)
def test_invalid_arguments_write_nothing(cases: Path, option: str, value: str) -> None:
    command = args(cases)
    command[0 if option == "slug" else command.index(option) + 1] = value
    before = (cases / "cst-applications.md").read_bytes()
    assert main(command) == 2
    assert not (cases / "applications/new-basin").exists()
    assert (cases / "cst-applications.md").read_bytes() == before


@pytest.mark.parametrize("fault", ["config", "observations", "write"])
def test_preflight_failure_leaves_no_partial_case(
    cases: Path, template: Path, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if fault == "config":
        (template / "project_config.yml").write_text("project: [")
    elif fault == "observations":
        (template / "project_config_build_model.yml").write_text(
            "observations:\n  river discharge: /external/observations.csv\n"
        )
    else:

        def fail_move(*args: object) -> None:
            raise PermissionError("Case write rejected")

        monkeypatch.setattr("scripts.create_case.shutil.move", fail_move)
    before = (cases / "cst-applications.md").read_bytes()
    assert main(args(cases)) == 2
    assert not (cases / "applications/new-basin").exists()
    assert (cases / "cst-applications.md").read_bytes() == before


@pytest.mark.parametrize("external_template", [False, True])
def test_shipped_template_composes_for_every_workflow(
    cases: Path, external_template: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the maintained starter against the current composition contract."""
    monkeypatch.setattr(
        "scripts.create_case.TOOLBOX_ROOT", Path(__file__).resolve().parents[1]
    )
    assert not (cases / "templates").exists()
    if external_template:
        stale = cases / "templates/example-basin"
        stale.mkdir(parents=True)
        (stale / "project_config.yml").write_text("obsolete: true\n", encoding="utf-8")
    assert main(args(cases)) == 0
    config_path = cases / "applications/new-basin/project_config.yml"
    config = load_composed_config(config_path)
    assert set(config["workflows"]) == set(WORKFLOW_NAMES)
    for workflow in WORKFLOW_NAMES:
        composed = load_composed_config(config_path, workflow)
        assert composed["workflows"][workflow] == config["workflows"][workflow]
    assert config["project"]["project_dir"] == (cases / "runs/new-basin").as_posix()
    assert Path(config["basin"]["output_locations"]).is_file()
    declared = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert declared["schema_version"] == 3
    assert declared["project"]["project_dir"] == "../../runs/new-basin"
    assert declared["basin"]["output_locations"] == "data/output_locations.csv"
    moved = cases.parent / "relocated-cases"
    shutil.copytree(cases, moved)
    relocated = load_composed_config(
        moved / "applications/new-basin/project_config.yml"
    )
    assert Path(relocated["project"]["project_dir"]) == moved / "runs/new-basin"
    assert Path(relocated["basin"]["output_locations"]).is_file()
