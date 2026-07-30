"""OQ-12: `save_grids` was renamed to `save_gridded` at step 5e.

The design's §8 gate for 5e names an "old-key-raises test". This is it.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def _dry_run(config_path):
    return subprocess.run(
        [
            sys.executable, "-m", "snakemake", "all", "-n",
            "-s", str(REPO / "Snakefile_climate_projections"),
            "--configfile", str(config_path),
        ],
        cwd=REPO, capture_output=True, text=True,
    )


@pytest.fixture
def seed_config():
    return yaml.safe_load(
        (REPO / "config/workflows/snake_config_dev_fast.yml").read_text(encoding="utf-8")
    )


def test_K1_the_old_key_raises_naming_the_new_one(tmp_path, seed_config):
    """Ignoring `save_grids` is worse than crashing on it.

    A user who set `save_grids: true` would silently get `false` behaviour, with
    no signal that their config had stopped being read.
    """
    cfg = seed_config
    proj = cfg["workflows"]["climate_projections"]
    proj.pop("save_gridded", None)
    proj["save_grids"] = True
    path = tmp_path / "old_key.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = _dry_run(path)
    assert result.returncode != 0, "a renamed key must not be silently ignored"
    combined = result.stdout + result.stderr
    assert "save_grids" in combined and "save_gridded" in combined, (
        "the error must name BOTH the dead key and its replacement"
    )


def test_K1_it_fails_at_DAG_build_not_at_run_time(tmp_path, seed_config):
    """A config error should not require scheduling jobs to discover."""
    cfg = seed_config
    cfg["workflows"]["climate_projections"]["save_grids"] = False
    path = tmp_path / "old_key_false.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = _dry_run(path)
    # -n never executes a job, so a nonzero exit here is necessarily parse-time.
    assert result.returncode != 0
    assert "save_grids" in result.stdout + result.stderr


def test_K2_shipped_configs_all_use_the_new_key():
    """The rename must have reached the configs, or K1 guards nothing real."""
    for cfg_path in (REPO / "config/workflows").glob("*.yml"):
        text = cfg_path.read_text(encoding="utf-8")
        assert "save_grids" not in text, f"{cfg_path.name} still carries the dead key"
