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


def _results(root):
    files = sorted(
        path for path in (root / "results/indicator_sets").rglob("*") if path.is_file()
    )
    return {path: path.read_bytes() for path in files}


def _tamper_keeping_timestamps(root):
    """Change a retained response in place without moving its mtime."""
    native = root / f"simulations/{RUN_IDS[0]}/output_gwr.csv"
    stat = native.stat()
    native.write_bytes(native.read_bytes() + b"\n")
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
    assert "wf4 simulate_system" in output
    assert "Snakemake command" in output and "DRY RUN" in output
    assert "Work plan" in output and "Planning complete" in output
    assert "Execution log" not in output and " - RUN " not in output
    assert "None" not in output.splitlines()
    assert re.search(
        r"run\s+4\.07\s+Check simulation outputs for indicators\s+2\b", output
    ), output
    assert re.search(r"pending\s+4\.08\s+Calculate indicators\s+2 planned", output), (
        output
    )
    assert not (root / "_engine/indicator_requests").exists()
    assert not (root / "_engine/indicator_sets").exists()
    assert not (root / "results/indicator_sets").exists()

    # Refused before any job is scheduled. A corrupt publication's refusal is
    # pinned in-process by test_indicator_plan_v2, at a fraction of the cost.
    _tamper_keeping_timestamps(root)
    code, output = _simulate(experiment, target=None)
    assert code != 0, output
    assert "checksum mismatch" in output, output
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
    # One copy of every table (R1, ruling D20): results/ only.
    assert not (marker.parent / "data").exists()
    assert len(list((root / "_engine/indicator_sets").glob("*/indicators.json"))) == 1
    results = _results(root)
    assert {
        "gwr_run_indicators.csv",
        "gwr_group_indicators.csv",
        "group_members.csv",
        "group_metadata.csv",
    } <= {path.name for path in results}

    tracking = root / "_engine/indicator_sets/names/default/current.json"
    assert tracking.is_file()
    stable = root / "results/indicator_sets/default/gwr_run_indicators.csv"
    original = stable.read_bytes()
    stable.unlink()
    # Its only copy is gone, so the set is recomputed to the same bytes.
    code, output = _simulate(experiment)
    assert code == 0, output
    assert stable.read_bytes() == original
    assert _results(root) == results

    code, output = _simulate(experiment, "--forceall")
    assert code == 0, output
    assert _results(root) == results

    _tamper_keeping_timestamps(root)
    code, output = _simulate(experiment)
    assert code != 0, output
    assert _results(root) == results


@pytest.mark.skipif(
    os.name != "nt",
    reason="run_project_child refuses project writers off Windows",
)
def test_an_edited_published_set_is_recomputed(tmp_path):
    experiment = build_v2_experiment(tmp_path)
    root = experiment.root
    code, output = _simulate(experiment)
    assert code == 0, output
    table = root / "results/indicator_sets/default/gwr_run_indicators.csv"
    original = table.read_bytes()
    table.write_bytes(original + b"tampered\n")
    # Its only copy changed (R1, D20): recompute, never refuse or keep the edit.
    code, output = _simulate(experiment, "--forceall")
    assert code == 0, output
    assert table.read_bytes() == original
