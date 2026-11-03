"""Legacy slug helpers and the shipped read-only experiment preview entry point."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config
from blueearth_cst.shared.snake_utils import (
    suggest_experiment_name,
    validate_experiment_name,
)

SNAKEDIR = Path(__file__).resolve().parents[1]
CLI = SNAKEDIR / "scripts" / "suggest_experiment_name.py"


@pytest.mark.parametrize(
    "project_dir, expected",
    [
        ("examples/Gabon", "gabon_20260728"),
        ("test_case/test_local", "test_local_20260728"),
        ("/mnt/data/My Basin-2024", "my_basin_2024_20260728"),
        (r"C:\runs\Rhine--Upper", "rhine_upper_20260728"),
        ("trailing/slash/", "slash_20260728"),
        ("__leading_junk", "leading_junk_20260728"),
    ],
)
def test_slugification(project_dir, expected):
    assert suggest_experiment_name(project_dir, "20260728") == expected


def test_suggestion_always_satisfies_the_validator():
    """The retained pure helper validates its own legacy slug proposal."""
    for project_dir in ("examples/Gabon", "/mnt/x/A B C", "UPPER", "9lives"):
        name = suggest_experiment_name(project_dir, "20260728")
        assert validate_experiment_name(name, "/tmp/proj") == name


def test_truncation_keeps_the_date_and_stays_valid():
    name = suggest_experiment_name("x/" + "a" * 200, "20260728")
    assert len(name) <= 64
    assert name.endswith("_20260728")
    assert validate_experiment_name(name, "/tmp/proj") == name


def test_basename_without_alphanumerics_raises():
    with pytest.raises(ValueError, match="no alphanumeric"):
        suggest_experiment_name("/data/---", "20260728")


def _run(config, *options):
    """Exercise argument parsing and the actual shipped subprocess entry point."""
    return subprocess.run(
        [sys.executable, str(CLI), str(config), *options],
        cwd=SNAKEDIR,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _inventory(root):
    return {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else path.read_bytes()
        )
        for path in root.rglob("*")
    }


def _annotated_config(base, body):
    """Config bytes and comments must survive; the output root starts absent."""
    project = base / "Gabon"
    settings = base / "simulate_system.yml"
    settings.write_text(body, encoding="utf-8")
    config = base / "project.yml"
    config.write_text(
        "# project banner\n"
        "project:\n"
        f"  project_dir: {project.as_posix()}  # output destination\n"
        "\n"
        "workflows:\n"
        "  simulate_system:\n"
        "    enabled: true\n"
        f"    config_path: {settings.name}\n",
        encoding="utf-8",
    )
    return config, project


def test_subprocess_previews_configured_family_and_override_without_writing(tmp_path):
    config, project = _annotated_config(
        tmp_path,
        "# fixed family retained in user YAML\n"
        "experiment_name: basin_stress  # keep this annotation\n"
        "operation: simulate-and-indicators\n"
        "compute:\n"
        "  julia_threads: 2  # unrelated scheduling choice\n",
    )
    before = _inventory(tmp_path)
    configured = _run(config)
    assert configured.returncode == 0, configured.stderr
    assert configured.stdout.strip() == "basin_stress_01"
    assert "Advisory only" in configured.stderr
    override = _run(config, "--name", "drought")
    assert override.returncode == 0, override.stderr
    assert override.stdout.strip() == "drought_01"
    assert _inventory(tmp_path) == before
    assert not project.exists()
    assert not (project / "experiments").exists()
    assert not (project / "experiments/_engine/allocation.json").exists()


def test_subprocess_previews_default_with_bare_key_and_preserves_comments(tmp_path):
    config, project = _annotated_config(
        tmp_path,
        "# default numbered experiment is chosen at execution\n"
        "experiment_name:   # intentionally unset\n"
        "operation: simulate-and-indicators\n",
    )
    before = _inventory(tmp_path)
    result = _run(config)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stress_test_01"
    assert _inventory(tmp_path) == before
    assert not project.exists()


@pytest.mark.parametrize("missing", ["project file", "config_path", "workflow file"])
def test_subprocess_missing_config_refuses_without_any_writes(tmp_path, missing):
    config, project = _annotated_config(
        tmp_path, "operation: simulate-and-indicators\n"
    )
    if missing == "project file":
        config.unlink()
        expected_error = config.name
    elif missing == "config_path":
        config.write_text(
            "project:\n"
            f"  project_dir: {project.as_posix()}\n"
            "workflows:\n"
            "  simulate_system:\n"
            "    enabled: true\n",
            encoding="utf-8",
        )
        expected_error = "config_path"
    else:
        (tmp_path / "simulate_system.yml").unlink()
        expected_error = "simulate_system.yml"
    before = _inventory(tmp_path)
    result = _run(config)
    assert result.returncode == 2
    assert expected_error in result.stderr
    assert result.stdout == ""
    assert _inventory(tmp_path) == before
    assert not project.exists()


def test_template_leaves_experiment_name_for_numbered_execution_selection():
    template = load_composed_config(
        SNAKEDIR / "config/templates/example-basin/project_config.yml"
    )
    assert "experiment_name" not in template["workflows"]["simulate_system"]


def test_test_fixtures_deliberately_keep_a_fixed_name():
    """Numerical fixture roots remain fixed while the user template uses the default."""
    for relative in (
        "test_case/project_config_baseline.yml",
        "test_case/project_config_wf2_fast.yml",
    ):
        config = load_composed_config(SNAKEDIR / relative)
        assert config["workflows"]["simulate_system"]["experiment_name"]


def test_shipped_template_set_previews_without_modifying_any_copied_file(tmp_path):
    templates = SNAKEDIR / "config/templates/example-basin"
    source = templates / "project_config.yml"
    config = tmp_path / source.name
    shutil.copyfile(source, config)
    for sibling in templates.glob("project_config_*.yml"):
        shutil.copyfile(sibling, tmp_path / sibling.name)
    original = load_composed_config(config)["project"]["project_dir"]
    project = tmp_path / "Gabon"
    config.write_text(
        config.read_text(encoding="utf-8").replace(original, project.as_posix(), 1),
        encoding="utf-8",
    )
    before = _inventory(tmp_path)
    result = _run(config)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stress_test_01"
    assert "Advisory only" in result.stderr
    assert _inventory(tmp_path) == before
    assert not project.exists()
