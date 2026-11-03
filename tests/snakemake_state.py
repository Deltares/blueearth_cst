"""Test-owned working directories for real Snakemake invocations."""

import shutil
from pathlib import Path

import yaml


def prepare_workdir(workdir: Path) -> Path:
    """Copy small relative-path inputs without sharing outputs or engine state."""
    repo = Path(__file__).resolve().parents[1]
    workdir.mkdir(parents=True, exist_ok=True)
    # Spatial rules declare their package sources as relative DAG inputs.
    for relative in ("blueearth_cst", "config", "tests/data", "test_case/test_data"):
        shutil.copytree(
            repo / relative,
            workdir / relative,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return workdir


def write_profile(workdir: Path) -> Path:
    """Select an isolated directory through Snakemake's supported profile API."""
    profile = workdir / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        yaml.safe_dump({"directory": workdir.as_posix()}), encoding="utf-8"
    )
    return profile
