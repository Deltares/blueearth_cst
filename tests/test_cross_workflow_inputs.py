"""Consumer-local leaf contract after the R12 workflow extraction.

WF2 and WF3 build real DAGs without WF1 artifacts. WF4's parsed model-reference
rule declares precisely the shared model leaf set, and no WF4 producer can
rebuild those inputs. The collection boundary is substituted by the shared
fixture; retained collection content is covered by its typed-reader tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.workflow_contract,
    pytest.mark.usefixtures("isolated_snakemake"),
]

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "project_config_fixture.yml"

sys.path.insert(0, str(SNAKEDIR / "dev" / "scripts"))
import cross_workflow_inputs as cwi  # noqa: E402

from blueearth_cst.shared.config_composition import load_composed_config  # noqa: E402
from tests.conftest import phase_env, write_config  # noqa: E402

#: The workflows that consume wf1 artifacts. WF1 produces them, so it is not here.
DOWNSTREAM = {
    "wf2": "analyze_projections.smk",
    "wf3": "generate_scenarios.smk",
}

#: Every leaf currently belongs to WF3 (rules 3.00b and 3.01c); WF2 has consumed
#: nothing from wf1 since ADR 0003. `test_wf2_needs_no_wf1_artifact` pins that,
#: so if it ever changes this constant is wrong loudly rather than quietly.
MINIMALITY_WORKFLOW = "wf3"


def _staged_config(tmp_path: Path, leaves) -> Path:
    """Stage ``leaves`` into a scratch project and return its config path."""
    base = load_composed_config(CONFIG_FN)
    project_dir = tmp_path / "proj"
    base["project"]["project_dir"] = str(project_dir).replace("\\", "/")
    config_text = yaml.safe_dump(base)

    project_dir.mkdir(parents=True, exist_ok=True)
    cwi.stage(project_dir, config_text, leaves=leaves)

    # The --configfile target is written SPLIT; `config_text` above stays
    # whole because it is staged as the wf1 project SNAPSHOT, which the
    # drift guard compares against the composed live config.
    return write_config(tmp_path, base, stem="project_config_staged")


def _dry_run(snakefile: str, config_path: Path) -> subprocess.CompletedProcess:
    """Build the DAG only. Returns the completed process; stdout+stderr captured.

    Runs on BOTH CI legs against the Windows-flavoured `tests/` config, and that
    is deliberate. AGENTS.md tells a *user* to pick `*_linux.yml` on Linux
    because data-catalog paths differ — but a dry-run resolves the catalog
    file's existence, not its interior data paths, so the distinction does not
    reach here. `test_cli.py` proves it: its dry-run tests use this same config
    unconditionally, and its `linux_config_fn` case is a separate "the Linux
    config still parses" assertion that also runs on both legs rather than a
    platform switch. No skip guard, therefore, and none wanted.
    """
    return subprocess.run(
        [
            "snakemake",
            "all",
            "-c",
            "1",
            "-s",
            str(SNAKEDIR / snakefile),
            "--configfile",
            str(config_path),
            "--dry-run",
        ],
        cwd=SNAKEDIR,
        env=phase_env(snakefile),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_leaf_set_is_not_empty():
    """Guard on the guards: an empty LEAVES would make minimality vacuous.

    With no leaves, `test_each_leaf_is_required` gets zero parameters and the
    completeness test degenerates into "the DAG builds from nothing".
    """
    assert cwi.LEAVES, "LEAVES is empty; the tests below would prove nothing"


def test_extras_are_not_leaves():
    """The EXTRA_* constants must stay out of the required set.

    They are staged by some callers for reasons that are real but not DAG
    requirements. Folding one into LEAVES is how a file nothing reads survives
    a sweep — which is exactly what happened to the region.
    """
    for extra in (cwi.EXTRA_REGION, cwi.EXTRA_WF2_SNAPSHOT):
        assert extra not in cwi.LEAVES, (
            f"{extra} is declared as a deliberate non-leaf but is in LEAVES"
        )


@pytest.mark.parametrize("name,snakefile", sorted(DOWNSTREAM.items()))
def test_leaves_are_complete(tmp_path, name, snakefile):
    """Staging exactly LEAVES — no extras — resolves the DAG.

    If this fails with a MissingInputException, a rule has started declaring a
    cross-workflow input that LEAVES does not carry. Add it there, not in the
    fixture that happened to go red.
    """
    config_path = _staged_config(tmp_path, cwi.LEAVES)
    proc = _dry_run(snakefile, config_path)
    assert proc.returncode == 0, (
        f"{name}: DAG did not build with exactly LEAVES staged.\n"
        f"{(proc.stdout + proc.stderr)[-2000:]}"
    )


def test_wf2_needs_no_wf1_artifact(tmp_path):
    """WF2 builds its DAG against an EMPTY project_dir.

    Since ADR 0003 the extent is model-free and WF2 delineates its own region,
    declaring it as an output; R07 B1 had already retired the extraction's
    region input. So WF2 has no cross-workflow leaf at all. Pinned because two
    stagers still hand it a region, and because `MINIMALITY_WORKFLOW` above
    assumes it.
    """
    config_path = _staged_config(tmp_path, ())
    proc = _dry_run(DOWNSTREAM["wf2"], config_path)
    assert proc.returncode == 0, (
        "wf2 no longer builds from an empty project_dir, so it has acquired a "
        "cross-workflow input. Add it to LEAVES and revisit MINIMALITY_WORKFLOW.\n"
        f"{(proc.stdout + proc.stderr)[-2000:]}"
    )


@pytest.mark.parametrize("dropped", cwi.LEAVES)
def test_each_leaf_is_required(tmp_path, monkeypatch, dropped):
    """Every static model leaf is required by WF4 and has no local producer."""
    from tests.simulation_workflow_fixture import parse_simulation_workflow

    workflow = parse_simulation_workflow(tmp_path, monkeypatch)
    project = tmp_path / "project"
    declared = {
        Path(str(path)).relative_to(project).as_posix()
        for path in workflow.get_rule("write_model_fingerprint").input
    }
    assert dropped in declared
    assert declared == set(cwi.LEAVES)
    # They are external leaves: no successor rule may rebuild the model.
    produced = {str(path) for rule in workflow.rules for path in rule.output}
    assert not {str(project / leaf) for leaf in cwi.LEAVES} & produced
