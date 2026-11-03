"""The production indicator checkpoint under real Snakemake, over a retained simulation.

`simulate_system.smk` plans indicators through `checkpoint prepare_indicator_plan`
and publishes an immutable set. These cases pin what a Snakemake upgrade or a
rule edit could quietly change, and what no unit test can see: a fresh dry-run
writes nothing, `--forceall` reuses the published set byte for byte, and a
byte-stale retained response is refused before anything is scheduled.
Indicators-only needs neither Julia nor a model build, so this runs on a bare
checkout; the publishing half is Windows-only because the harness refuses
project writers elsewhere. It replaces the R12 prototype checkpoint gates (t2610041227).
"""

import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tests._v2_experiment import RUN_IDS, build_v2_experiment, default_responses

pytestmark = [
    pytest.mark.workflow_contract,
    pytest.mark.usefixtures("isolated_snakemake"),
]

REPO = Path(__file__).resolve().parents[1]


def _simulate(experiment, *extra, target="indicators"):
    environment = os.environ.copy()
    environment.pop("SNAKEMAKE_WORKFLOW_PROFILE", None)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/simulate_system.py"),
            "--config",
            str(experiment.config),
            *(["--target", target] if target else []),
            "--cores",
            "1",
            *extra,
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode, result.stdout + result.stderr


def _published(root):
    files = sorted(
        path
        for directory in ("_engine/indicator_sets", "results/indicator_sets")
        for path in (root / directory).rglob("*")
        if path.is_file()
    )
    return {path: path.read_bytes() for path in files}


def _tamper_keeping_timestamps(root):
    """Change a retained response in place without moving its mtime."""
    native = root / f"hydrology/wflow/output/{RUN_IDS[0]}.csv"
    stat = native.stat()
    native.write_bytes(native.read_bytes().replace(b",3\n", b",4\n"))
    os.utime(native, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_a_dry_run_writes_nothing_and_a_stale_response_is_refused(tmp_path):
    """Both happen before any project writer is launched, so on every platform."""
    responses = default_responses()
    discharge = {
        **responses.variables[0],
        "variable": "q",
        "locations": ["9", "2"],
        "units": "m3 s-1",
    }
    experiment = build_v2_experiment(
        tmp_path,
        replace(responses, variables=[*responses.variables, discharge]),
        capacity=4,
    )
    root = experiment.root
    settings = experiment.config.parent / "simulate_system.yml"
    settings.write_text(
        yaml.safe_dump(
            {
                "experiment_name": root.name,
                "operations": ["indicators"],
                "indicator_sets": ["default", "flow_regime"],
            }
        ),
        encoding="utf-8",
    )

    code, output = _simulate(experiment, "--dry-run", target=None)
    assert code == 0, output
    assert re.search(r"prepare_indicator_plan\s+2", output), output
    assert not (root / "_engine/indicator_requests").exists()
    assert not (root / "_engine/indicator_sets").exists()
    assert not (root / "results/indicator_sets").exists()

    # Refused before any job is scheduled. A corrupt publication's refusal is
    # pinned in-process by test_indicator_plan_v2, at a fraction of the cost.
    _tamper_keeping_timestamps(root)
    code, output = _simulate(experiment, target=None)
    assert code != 0, output
    assert "bytes differ" in output, output
    assert "Job stats:" not in output, output
    assert not (root / "_engine/indicator_sets").exists()


@pytest.mark.skipif(
    os.name != "nt",
    reason="run_project_child refuses project writers off Windows: no verified "
    "lifetime guarantee for their descendants",
)
def test_a_published_set_is_reused_byte_for_byte_under_forceall(tmp_path):
    experiment = build_v2_experiment(tmp_path)
    root = experiment.root

    code, output = _simulate(experiment)
    assert code == 0, output
    assert output.count("DONE all") == 1, output
    assert output.index("results saved") < output.index(
        "wf4 simulate_system completed"
    ), output
    assert output.rstrip().endswith("- DONE all"), output
    marker = next((root / "_engine/indicator_sets").glob("*/indicators.json"))
    assert (marker.parent / "data").is_dir()
    assert not (root / "results/indicator_sets" / marker.parent.name).exists()
    published = _published(root)
    assert len(list((root / "_engine/indicator_sets").glob("*/indicators.json"))) == 1
    assert {
        "gwr_run_indicators.csv",
        "gwr_group_indicators.csv",
        "group_members.csv",
        "group_metadata.csv",
    } <= {path.name for path in published}

    tracking = root / "_engine/indicator_sets/names/default/current.json"
    assert tracking.is_file()
    stable = root / "results/indicator_sets/default/gwr_run_indicators.csv"
    original = stable.read_bytes()
    stable.unlink()
    # No immutable rule output is missing: successful no-op reuse repairs the export.
    marker = next((root / "_engine/indicator_sets").glob("*/indicators.json"))
    code, output = _simulate(experiment, "--target", str(marker))
    assert code == 0, output
    assert stable.read_bytes() == original
    assert _published(root) == published

    code, output = _simulate(experiment, "--forceall")
    assert code == 0, output
    assert _published(root) == published

    _tamper_keeping_timestamps(root)
    code, output = _simulate(experiment)
    assert code != 0, output
    assert _published(root) == published


@pytest.mark.skipif(
    os.name != "nt",
    reason="run_project_child refuses project writers off Windows",
)
def test_a_forced_run_refuses_a_damaged_retained_set_before_launch(tmp_path):
    experiment = build_v2_experiment(tmp_path)
    root = experiment.root
    code, output = _simulate(experiment)
    assert code == 0, output
    table = next(
        (root / "_engine/indicator_sets").glob("*/data/gwr_run_indicators.csv")
    )
    table.write_bytes(table.read_bytes() + b"tampered\n")
    before = _published(root)
    code, output = _simulate(experiment, "--forceall")
    assert code != 0, output
    assert "indicator table differs" in output, output
    assert "Job stats:" not in output, output
    assert _published(root) == before
