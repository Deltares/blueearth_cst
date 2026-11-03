"""The suggestion and compatibility helpers cannot claim experiment identities."""

import pytest
import yaml

from blueearth_cst.experiment.allocate import (
    ExperimentCollisionError,
    allocate_experiment_name,
    next_available_name,
    reserve_experiment,
    resolve_default_experiment_name,
)
from scripts import suggest_experiment_name


@pytest.fixture
def config(tmp_path):
    settings = tmp_path / "simulation.yml"
    settings.write_text("operation: simulate-and-indicators\n", encoding="utf-8")
    path = tmp_path / "project.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"project_dir": str(tmp_path / "outputs")},
                "workflows": {
                    "simulate_system": {"enabled": True, "config_path": settings.name}
                },
            }
        ),
        encoding="utf-8",
    )
    return path, settings


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], "stress_test_01"),
        (["--dry-run"], "stress_test_01"),
        (["--name", "drought"], "drought_01"),
        (["--name", "drought_37"], "drought_37"),
        (["--new-experiment"], "stress_test_01"),
    ],
)
def test_suggestion_always_previews_without_mutation(config, capsys, options, expected):
    path, settings = config
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    assert suggest_experiment_name.main([str(path), *options]) == 0
    assert capsys.readouterr().out.strip() == expected
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
    assert not (path.parent / "outputs").exists()


def test_default_and_next_previews_do_not_create(tmp_path):
    assert (
        resolve_default_experiment_name(tmp_path, "old_slug", "20261010")
        == "stress_test_01"
    )
    assert next_available_name(tmp_path, "drought") == "drought_01"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "function,args",
    [
        (reserve_experiment, ("name",)),
        (allocate_experiment_name, ("name", True)),
    ],
)
def test_retired_mutating_helpers_refuse_intact(tmp_path, function, args):
    with pytest.raises(ExperimentCollisionError, match="execution runner"):
        function(tmp_path, *args)
    assert list(tmp_path.iterdir()) == []


def test_explicit_number_and_new_flag_conflict_preserves_yaml(config, capsys):
    path, settings = config
    before = settings.read_bytes()
    assert (
        suggest_experiment_name.main(
            [str(path), "--name", "drought_02", "--new-experiment"]
        )
        == 2
    )
    assert "numbered pin" in capsys.readouterr().err
    assert settings.read_bytes() == before
