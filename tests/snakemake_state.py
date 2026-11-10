"""Test-owned working directories for real Snakemake invocations."""

import json
import logging
import shutil
import sys
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


def _capture_cli_job_stats() -> None:
    """Observe native RUN_INFO statistics without changing CLI/DAG behavior."""
    from snakemake.cli import main
    from snakemake.logging import logger

    destination = Path(sys.argv[1])
    sys.argv = ["snakemake", *sys.argv[2:]]

    class JobStats(logging.Filter):
        def filter(self, record):
            if (
                getattr(record, "event", None) == "run_info"
                and not destination.exists()
            ):
                destination.write_text(json.dumps(record.stats), encoding="utf-8")
            if getattr(record, "event", None) == "job_info":
                with destination.with_suffix(".reasons.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps({"rule": record.rule_name, "reason": record.reason})
                        + "\n"
                    )
            return True

    logger.addFilter(JobStats())
    main()


if __name__ == "__main__":
    _capture_cli_job_stats()
