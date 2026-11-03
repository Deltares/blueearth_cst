"""A failed batch's finished members survive Snakemake's real cleanup.

The claim of t2608071217 is about Snakemake, not about the staging module:
when a batch job fails, Snakemake deletes every declared output of that job.
Only a real run shows that staged CSVs are out of its reach and that the retry
runs just the failed member, so this drives a batch-shaped rule -- `update()`
outputs, the same `run_staged_batch` call as rule 4.05 -- with a stub driver
that behaves like `run_wflow_batch.jl` and fails one chosen member.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

STUB = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    staging = Path(os.environ["CST_BATCH_STAGING"])
    fail = os.environ.get("STUB_FAIL", "")
    args = sys.argv[1:]
    with open(os.environ["STUB_CALLS"], "a") as calls:
        calls.write(",".join(args[1::3]) + "\\n")
    code = 0
    for i in range(1, len(args), 3):
        run, out = args[i], Path(args[i + 2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"csv {run}")
        if run == fail:
            code = 1
            continue
        os.replace(out, staging / f"{run}.csv")
        os.replace(staging / f"{run}.expect", staging / f"{run}.ok")
    sys.exit(code)
    """
)

SNAKEFILE = textwrap.dedent(
    """
    import sys
    from blueearth_cst.experiment.batch_staging import run_staged_batch
    RUNS = ["run_01", "run_02", "run_03"]
    rule all:
        input: [f"output/{r}.csv" for r in RUNS]
    rule batch:
        input:
            simulation="simulation.json",
            tomls=[f"settings/{r}.toml" for r in RUNS],
        output:
            csvs=[update(f"output/{r}.csv") for r in RUNS],
        params:
            records=["0", *[v for r in RUNS for v in (r, f"settings/{r}.toml", f"output/{r}.csv")]],
        run:
            run_staged_batch(params.records, input.simulation, "_engine/batch_staging",
                [sys.executable, "stub.py"])
    """
)


def _snakemake(root, **env):
    return subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", "Snakefile", "-c", "1", "--quiet"],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "STUB_CALLS": str(root / "calls.txt"),
            **env,
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.workflow_contract
def test_a_retry_after_one_failure_runs_only_that_member(tmp_path):
    (tmp_path / "Snakefile").write_text(SNAKEFILE, encoding="utf-8")
    (tmp_path / "stub.py").write_text(STUB, encoding="utf-8")
    (tmp_path / "simulation.json").write_text("{}", encoding="utf-8")
    for run in ("run_01", "run_02", "run_03"):
        toml = tmp_path / "settings" / f"{run}.toml"
        toml.parent.mkdir(parents=True, exist_ok=True)
        toml.write_text(f"run = {run}\n", encoding="utf-8")
    staging = tmp_path / "_engine" / "batch_staging"

    failed = _snakemake(tmp_path, STUB_FAIL="run_02")
    assert failed.returncode != 0, failed.stdout + failed.stderr
    # Snakemake's cleanup removed every declared output of the failed job ...
    assert not list((tmp_path / "output").glob("run_*.csv"))
    # ... but the finished siblings were already out of its reach.
    assert sorted(p.name for p in staging.glob("run_*.ok")) == [
        "run_01.ok",
        "run_03.ok",
    ]

    retried = _snakemake(tmp_path)
    assert retried.returncode == 0, retried.stdout + retried.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == ["run_01,run_02,run_03", "run_02"]
    for run in ("run_01", "run_02", "run_03"):
        assert (tmp_path / "output" / f"{run}.csv").read_text() == f"csv {run}"
    assert not staging.exists()
