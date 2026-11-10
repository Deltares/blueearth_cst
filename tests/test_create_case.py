"""Named trial creation, preservation, portability and failure contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.config_composition import WORKFLOW_NAMES, load_composed_config
from scripts.create_case import (
    _annotate_config,
    _template_comments,
    _yaml_parameters,
    main,
)


@pytest.fixture
def cases(tmp_path: Path) -> Path:
    root = tmp_path / "cases"
    root.mkdir()
    return root


def args(cases: Path, trial: str = "default") -> list[str]:
    return [
        "new-basin",
        "--cases-root",
        str(cases),
        "--trial",
        trial,
        "--purpose",
        "Test | basin",
    ]


def create(cases: Path, trial: str = "default") -> None:
    assert main(args(cases, trial) + ["--longitude", "12.5", "--latitude", "-3"]) == 0


def snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("trial", ["default", "v2", "2026-11-08"])
def test_new_case_and_additional_trials(cases, trial):
    create(cases, trial)
    case = cases / "new-basin"
    project = case / trial / "project_config.yml"
    config = yaml.safe_load(project.read_text())
    assert config["project"]["project_dir"] == "outputs"
    assert config["basin"]["output_locations"] == "../data/output_locations.csv"
    assert "outlet,12.5,-3.0" in (case / "data/output_locations.csv").read_text()
    assert not (case / trial / "outputs").exists()
    assert f"[{trial}]({trial}/project_config.yml)" in (case / "README.md").read_text()
    assert "| Test &#124; basin | working |" in (case / "README.md").read_text()
    assert f"/{trial}/outputs/" in (case / ".gitignore").read_text()
    if trial != "default":
        assert main(args(cases, "next")) == 2
        assert main(args(cases, "next") + ["--from-trial", trial]) == 0
    else:
        before = snapshot(case / trial)
        assert main(args(cases, "next")) == 0
        assert snapshot(case / trial) == before


@pytest.mark.parametrize("mode", ["new", "starter", "copy"])
def test_preview_writes_nothing(cases, mode, capsys):
    if mode != "new":
        create(cases)
    before = snapshot(cases)
    command = args(cases, "v2")
    command += ["--longitude", "12.5", "--latitude", "-3"] if mode == "new" else []
    command += ["--from-trial", "default"] if mode == "copy" else []
    assert main(command + ["--dry-run"]) == 0
    assert snapshot(cases) == before
    assert not list(cases.glob(".create-case-*"))
    output = capsys.readouterr().out
    assert "Source:" in output and "Create:" in output and "-- --dry-run" in output


def test_copy_preserves_science_and_required_sidecars(cases, monkeypatch, tmp_path):
    create(cases)
    case = cases / "new-basin"
    source = case / "default"
    (source / "custom").mkdir()
    model = source / "project_config_build_model.yml"
    model.rename(source / "custom/model.yml")
    (source / "data").mkdir()
    (source / "data/observed.csv").write_text("time,value\n2000,1\n")
    (case / "data/shared.csv").write_text("time,value\n2000,2\n")
    text = (source / "custom/model.yml").read_text()
    text = text.replace(
        '"river discharge": null', '"river discharge": ../data/observed.csv'
    )
    # Use parsed YAML to avoid coupling this behavioural fixture to template quoting.
    body = yaml.safe_load(text)
    body["observations"] = {
        "river discharge": "../data/observed.csv",
        "precipitation": "../../data/shared.csv",
    }
    (source / "custom/model.yml").write_text(yaml.safe_dump(body))
    project = yaml.safe_load((source / "project_config.yml").read_text())
    project["workflows"]["build_model"]["config_path"] = "custom/model.yml"
    project["workflows"]["build_model"]["enabled"] = True
    (source / "project_config.yml").write_text(yaml.safe_dump(project))
    (source / "outputs/logs").mkdir(parents=True)
    (source / "outputs/logs/success.txt").write_text("old output")
    (source / "notes.md").write_text("Complete: old result")
    readme = case / "README.md"
    readme.write_bytes(readme.read_bytes() + b"\nUser-authored finding.\n")
    before = snapshot(case)
    assert main(args(cases, "v2") + ["--from-trial", "default"]) == 0
    after = snapshot(case)
    assert all(
        after[name] == content
        for name, content in before.items()
        if name not in {"README.md", ".gitignore"}
    )
    assert "User-authored finding." in readme.read_text()
    assert "Complete: old result" not in (case / "v2/notes.md").read_text()
    assert not (case / "v2/outputs").exists()
    copied = load_composed_config(case / "v2/project_config.yml", "build_model")
    assert copied["workflows"]["build_model"]["enabled"] is True
    assert (
        Path(copied["workflows"]["build_model"]["observations"]["river discharge"])
        == case / "v2/data/observed.csv"
    )
    moved = tmp_path / "relocated"
    shutil.copytree(case, moved)
    monkeypatch.chdir(tmp_path)
    for workflow in WORKFLOW_NAMES:
        composed = load_composed_config(moved / "v2/project_config.yml", workflow)
        assert Path(composed["project"]["project_dir"]) == moved / "v2/outputs"
        assert (
            Path(composed["basin"]["output_locations"])
            == moved / "data/output_locations.csv"
        )
    observations = load_composed_config(moved / "v2/project_config.yml", "build_model")[
        "workflows"
    ]["build_model"]["observations"]
    assert Path(observations["river discharge"]).read_text() == "time,value\n2000,1\n"
    assert Path(observations["precipitation"]) == moved / "data/shared.csv"


@pytest.mark.parametrize(
    "option,value",
    [
        ("slug", "../bad"),
        ("slug", "con"),
        ("--trial", "data"),
        ("--trial", "../bad"),
        ("--longitude", "181"),
        ("--latitude", "nan"),
        ("--purpose", " "),
        ("--from-trial", "../default"),
        ("--from-trial", "missing"),
    ],
)
def test_invalid_arguments_leave_no_files(cases, option, value):
    command = args(cases) + ["--longitude", "12.5", "--latitude", "-3"]
    if option == "slug":
        command[0] = value
    elif option in command:
        command[command.index(option) + 1] = value
    else:
        command += [option, value]
    assert main(command) == 2
    assert snapshot(cases) == {}
    assert list(cases.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    ["collision", "identity", "missing", "schema2", "retained", "move", "record"],
)
def test_existing_case_preserved_on_failure(cases, monkeypatch, failure):
    create(cases)
    case = cases / "new-basin"
    command = args(cases, "v2") + ["--from-trial", "default"]
    if failure == "collision":
        command = args(cases)
    elif failure == "identity":
        command += ["--longitude", "13", "--latitude", "-3"]
    elif failure == "schema2":
        path = case / "default/project_config.yml"
        path.write_text(
            path.read_text().replace("schema_version: 3", "schema_version: 2")
        )
    elif failure in {"missing", "retained"}:
        path = case / "default/project_config_simulate_system.yml"
        manifest = case / "default/outputs/collection.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        path.write_text(
            path.read_text()
            + "\nscenario_collection:\n  manifest_path: outputs/collection.json\n"
        )
        if failure == "missing":
            manifest.unlink()
    elif failure == "move":

        def fail(*args, **kwargs):
            raise PermissionError("injected move failure")

        monkeypatch.setattr("scripts.create_case.shutil.move", fail)
    elif failure == "record":
        import os

        original = os.replace

        def fail(source, target):
            if Path(source).name == ".gitignore":
                raise PermissionError("injected record failure")
            return original(source, target)

        monkeypatch.setattr("scripts.create_case.os.replace", fail)
    before = snapshot(case)
    assert main(command) == 2
    assert snapshot(case) == before
    assert not list(cases.glob(".create-case-*"))
    assert not (case / ".create-trial.lock").exists()


def test_legacy_layout_refused_without_modification(cases):
    (cases / "applications/new-basin").mkdir(parents=True)
    (cases / "applications/new-basin/project_config.yml").write_text(
        "schema_version: 2\n"
    )
    before = snapshot(cases)
    assert main(args(cases) + ["--longitude", "12.5", "--latitude", "-3"]) == 2
    assert snapshot(cases) == before


def test_absolute_local_input_becomes_portable(cases):
    create(cases)
    case = cases / "new-basin"
    path = case / "default/project_config.yml"
    config = yaml.safe_load(path.read_text())
    config["basin"]["output_locations"] = (
        case / "data/output_locations.csv"
    ).as_posix()
    path.write_text(yaml.safe_dump(config))
    assert main(args(cases, "v2") + ["--from-trial", "default"]) == 0
    copied = yaml.safe_load((case / "v2/project_config.yml").read_text())
    assert copied["basin"]["output_locations"] == "../data/output_locations.csv"


def test_catalog_override_resolves_beside_destination(cases, tmp_path, capsys):
    create(cases)
    catalog = cases / "new-basin/data/catalog.yml"
    catalog.write_text("meta: {version: 1}\n")
    assert main(args(cases, "v2") + ["--catalog", "../data/catalog.yml"]) == 0
    project = cases / "new-basin/v2/project_config.yml"
    assert load_composed_config(project)["project"]["catalog"] == catalog.as_posix()
    before = snapshot(cases)
    assert main(args(cases, "bad") + ["--catalog", "missing.yml"]) == 2
    assert snapshot(cases) == before
    assert "Missing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "relative", ["new-basin", "new-basin/default", "new-basin/data"]
)
@pytest.mark.parametrize("method", ["is_symlink", "is_junction"])
def test_links_are_refused(cases, monkeypatch, relative, method):
    original = getattr(Path, method)
    monkeypatch.setattr(
        Path, method, lambda path: path == cases / relative or original(path)
    )
    assert main(args(cases) + ["--longitude", "12.5", "--latitude", "-3"]) == 2
    assert snapshot(cases) == {}


def test_git_ignores_only_outputs(cases):
    create(cases)
    subprocess.run(["git", "init", str(cases)], check=True, capture_output=True)
    output = subprocess.run(
        [
            "git",
            "-C",
            str(cases),
            "check-ignore",
            "new-basin/default/outputs/result.nc",
            "new-basin/default/project_config.yml",
            "new-basin/data/output_locations.csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert output.stdout.splitlines() == ["new-basin/default/outputs/result.nc"]


def test_default_trial_and_failed_new_install(cases, monkeypatch):
    command = args(cases)
    del command[command.index("--trial") : command.index("--trial") + 2]
    command += ["--longitude", "12.5", "--latitude", "-3"]
    original = shutil.move

    def fail(*args, **kwargs):
        raise PermissionError("injected installation failure")

    monkeypatch.setattr("scripts.create_case.shutil.move", fail)
    assert main(command) == 2
    assert list(cases.iterdir()) == []
    monkeypatch.setattr("scripts.create_case.shutil.move", original)
    assert main(command) == 0
    assert (cases / "new-basin/default/project_config.yml").is_file()


def test_source_generator_recipe_is_copied(cases):
    create(cases)
    source = cases / "new-basin/default"
    recipe = source / "data/weather.yml"
    recipe.parent.mkdir()
    recipe.write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "config/defaults/weathergen_config.yml"
        ).read_bytes()
    )
    workflow = source / "project_config_generate_scenarios.yml"
    workflow.write_text(
        workflow.read_text() + "\n  weathergen_config: data/weather.yml\n"
    )
    assert main(args(cases, "v2") + ["--from-trial", "default"]) == 0
    copied = cases / "new-basin/v2/data/weather.yml"
    assert copied.read_bytes() == recipe.read_bytes()
    config = load_composed_config(
        copied.parent.parent / "project_config.yml", "generate_scenarios"
    )
    assert (
        Path(config["workflows"]["generate_scenarios"]["weathergen_config"]) == copied
    )


def test_trial_local_catalog_requires_portable_dataset_dependencies(cases, capsys):
    create(cases)
    source = cases / "new-basin/default"
    (source / "catalog.yml").write_text("station: {uri: data/stations.csv}\n")
    project = source / "project_config.yml"
    config = yaml.safe_load(project.read_text())
    config["project"]["catalog"] = "catalog.yml"
    project.write_text(yaml.safe_dump(config))
    before = snapshot(cases)
    assert main(args(cases, "v2") + ["--from-trial", "default"]) == 2
    assert "relative dataset dependencies" in capsys.readouterr().err
    assert snapshot(cases) == before


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
