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
def test_native_temp_survives_failed_conversion_and_retires_after_verified_tables(
    tmp_path,
):
    from blueearth_cst.experiment.content_identity import canonical_json_bytes
    from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
    from blueearth_cst.shared.response_tables import open_table_responses
    from tests._v2_experiment import NATIVE_CSV, NATIVE_TOML, TEMPORAL

    # The same temp(update(...)) boundary as the live batch rule, with real
    # conversion rather than a stubbed retained-response writer.
    snakefile = """
from pathlib import Path
import shutil
from blueearth_cst.shared.response_tables import publish_response_tables
from blueearth_cst.experiment.wflow_response_reader import NativeRunArtifacts
rule all:
    input: "output/run_1/_engine/response_tables.json"
rule native:
    input: "seed.csv"
    output: temp(update("output/run_1.csv"))
    run:
        shutil.copyfile(input[0], output[0])
rule convert:
    input:
        csv="output/run_1.csv", toml="run.toml", temporal="temporal.json"
    output: "output/run_1/_engine/response_tables.json"
    run:
        publish_response_tables(
            NativeRunArtifacts(Path(input.csv), Path(input.toml), Path(input.temporal)),
            Path("output/run_1"), variables=("gwr",), decimal_places=5)
"""
    (tmp_path / "Snakefile").write_text(textwrap.dedent(snakefile), encoding="utf-8")
    (tmp_path / "seed.csv").write_text(NATIVE_CSV, encoding="utf-8")
    toml = tmp_path / "run.toml"
    toml.write_text(
        NATIVE_TOML.replace(
            "soil_water_saturated_zone_top__net_recharge_volume_flux", "wrong_parameter"
        ),
        encoding="utf-8",
    )
    (tmp_path / "temporal.json").write_bytes(canonical_json_bytes(TEMPORAL))
    failed = _snakemake(tmp_path)
    assert failed.returncode != 0, failed.stdout + failed.stderr
    raw = tmp_path / "output/run_1.csv"
    assert raw.read_text(encoding="utf-8") == NATIVE_CSV
    record = tmp_path / "output/run_1/_engine/response_tables.json"
    assert not record.exists()
    toml.write_text(NATIVE_TOML, encoding="utf-8")
    succeeded = _snakemake(tmp_path)
    assert succeeded.returncode == 0, succeeded.stdout + succeeded.stderr
    assert not raw.exists()
    responses = open_table_responses("run_1", record, ResponseRequest(("gwr",)))
    assert responses[0].values.tolist() == [3.0, 6.0]


@pytest.mark.workflow_contract
def test_a_retry_after_one_failure_runs_only_that_member(tmp_path):
    (tmp_path / "Snakefile").write_text(SNAKEFILE, encoding="utf-8")
    (tmp_path / "stub.py").write_text(STUB, encoding="utf-8")
    (tmp_path / "simulation.json").write_text("{}", encoding="utf-8")
    for run in ("run_01", "run_02", "run_03"):
        toml = tmp_path / "settings" / f"{run}.toml"
        toml.parent.mkdir(parents=True, exist_ok=True)
        toml.write_text(f'run = "{run}"\n', encoding="utf-8")
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
