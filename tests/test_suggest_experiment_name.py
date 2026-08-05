"""R07 B8: experiment_name is SUGGESTED once, never generated at run time.

The helper slugifies because ``basename(project_dir)`` is not guaranteed to
satisfy the grammar the workflow enforces (repo-7): ``examples/Gabon`` was live
in six shipped configs, and production project_dir values routinely carry
uppercase, hyphens or spaces. The design's original evidence ("both
gabon260725 and gabon_20260726 already satisfy the grammar") only tested names
that already conformed.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.snake_utils import (  # noqa: E402
    suggest_experiment_name,
    validate_experiment_name,
)

SNAKEDIR = Path(__file__).resolve().parents[1]
CLI = SNAKEDIR / "scripts" / "suggest_experiment_name.py"


@pytest.mark.parametrize(
    "project_dir, expected",
    [
        ("examples/Gabon", "gabon_20260728"),          # the live counterexample
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
    """The proposal is passed back through validate_experiment_name, so the
    suggester and the workflow's own gate can never disagree."""
    for pd in ("examples/Gabon", "/mnt/x/A B C", "UPPER", "9lives"):
        name = suggest_experiment_name(pd, "20260728")
        assert validate_experiment_name(name, "/tmp/proj") == name


def test_truncation_keeps_the_date_and_stays_valid():
    name = suggest_experiment_name("x/" + "a" * 200, "20260728")
    assert len(name) <= 64
    assert name.endswith("_20260728")
    assert validate_experiment_name(name, "/tmp/proj") == name


def test_basename_without_alphanumerics_raises():
    with pytest.raises(ValueError, match="no alphanumeric"):
        suggest_experiment_name("/data/---", "20260728")


def _run(cfg, *extra):
    return subprocess.run(
        [sys.executable, str(CLI), str(cfg), "--date", "20260728", *extra],
        capture_output=True, text=True,
    )


def _cfg(tmp_path, experiment_name=None):
    # project_dir must be under tmp_path. Since R9 P4 the command RESERVES the
    # name by creating experiments/<id>/, so a repo-relative project_dir here
    # would write real directories into the working tree on every test run --
    # it resurrected `examples/`, retired at R7, before this was caught. The
    # basename still slugifies to `gabon`, which is what these cases are about.
    project_dir = tmp_path / "Gabon"
    project_dir.mkdir(exist_ok=True)
    doc = {
        "project": {"project_dir": str(project_dir).replace("\\", "/")},
        "workflows": {"climate_experiment": {"enabled": True}},
    }
    if experiment_name is not None:
        doc["workflows"]["climate_experiment"]["experiment_name"] = experiment_name
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_cli_writes_when_absent(tmp_path):
    cfg = _cfg(tmp_path)
    res = _run(cfg)
    assert res.returncode == 0, res.stderr
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert doc["workflows"]["climate_experiment"]["experiment_name"] == "gabon_20260728"


def test_cli_refuses_to_overwrite(tmp_path):
    """The experiment name is the directory every wf3 artifact hangs off;
    silently changing it would strand a completed experiment's outputs."""
    cfg = _cfg(tmp_path, experiment_name="already_here")
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg)
    assert res.returncode != 0
    assert "already_here" in res.stderr
    assert cfg.read_text(encoding="utf-8") == before, "config must be untouched"


def test_cli_dry_run_leaves_the_config_alone(tmp_path):
    cfg = _cfg(tmp_path)
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "gabon_20260728"
    assert cfg.read_text(encoding="utf-8") == before


def test_cli_preserves_other_config_content(tmp_path):
    """Round-tripping the YAML must not drop or reorder unrelated keys."""
    cfg = _cfg(tmp_path)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["shared"] = {"clim_historical": "era5"}
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert _run(cfg).returncode == 0
    out = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert out["shared"] == {"clim_historical": "era5"}
    # Compare against the value actually written rather than a literal: the
    # fixture is tmp_path-based since P4 made the command reserve.
    assert out["project"]["project_dir"] == doc["project"]["project_dir"]


def test_cli_dry_run_reports_even_when_a_value_is_set(tmp_path):
    """The point of inspecting first is to SEE what would be proposed; a
    dry-run that only says "refusing" tells you nothing you did not know."""
    cfg = _cfg(tmp_path, experiment_name="already_here")
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "gabon_20260728"
    assert cfg.read_text(encoding="utf-8") == before


def test_cli_refusal_names_what_it_would_have_suggested(tmp_path):
    cfg = _cfg(tmp_path, experiment_name="already_here")
    res = _run(cfg)
    assert res.returncode != 0
    assert "already_here" in res.stderr and "gabon_20260728" in res.stderr
