"""The shared cross-workflow leaf set is COMPLETE and MINIMAL (R9 P5 F3).

`dev/scripts/cross_workflow_inputs.LEAVES` replaced three hand-kept copies of
the same staging logic. That alone does not stop it going stale — it makes it go
stale in one place instead of three. This module is what stops it, by checking
the list against the real DAG rather than against another list:

* **complete** — staging exactly `LEAVES` lets both downstream workflows build
  their DAG. A rule that starts declaring a new cross-workflow input turns this
  red on the next run. That is precisely the escape R9 P4 shipped: rule 3.01c
  `write_model_reference` added two model leaves, one of the three stagers was
  not updated, and the failure surfaced later as a red assertion that read like
  a defect in the guard it was testing.
* **minimal** — dropping any single leaf breaks WF3's DAG, and the error names
  the file that was dropped. This is what keeps vestigial entries out. Both test
  fixtures still stage a region that nothing has read since ADR 0003; had it
  been folded into `LEAVES` rather than passed as an explicit extra, this test
  would fail and say so.

Deliberately NOT parametrized over a second list of expected paths: a test that
compared `LEAVES` to a copy of `LEAVES` would pass forever and prove nothing.
Snakemake is the authority here, which is why every assertion below runs it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.workflow_contract

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_model_test.yml"

sys.path.insert(0, str(SNAKEDIR / "dev" / "scripts"))
import cross_workflow_inputs as cwi  # noqa: E402

#: The workflows that consume wf1 artifacts. WF1 produces them, so it is not here.
DOWNSTREAM = {
    "wf2": "Snakefile_climate_projections",
    "wf3": "Snakefile_climate_experiment",
}

#: Every leaf currently belongs to WF3 (rules 3.00b and 3.01c); WF2 has consumed
#: nothing from wf1 since ADR 0003. `test_wf2_needs_no_wf1_artifact` pins that,
#: so if it ever changes this constant is wrong loudly rather than quietly.
MINIMALITY_WORKFLOW = "wf3"


def _staged_config(tmp_path: Path, leaves) -> Path:
    """Stage ``leaves`` into a scratch project and return its config path."""
    base = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    project_dir = tmp_path / "proj"
    base["project"]["project_dir"] = str(project_dir).replace("\\", "/")
    config_text = yaml.safe_dump(base)

    project_dir.mkdir(parents=True, exist_ok=True)
    cwi.stage(project_dir, config_text, leaves=leaves)

    config_path = tmp_path / "snake_config_staged.yml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


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
            "snakemake", "all", "-c", "1",
            "-s", str(SNAKEDIR / snakefile),
            "--configfile", str(config_path),
            "--dry-run",
        ],
        cwd=SNAKEDIR,
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
def test_each_leaf_is_required(tmp_path, dropped):
    """Drop one leaf and WF3's DAG fails, naming the file that went.

    Minimality is the half that keeps the list honest. A leaf nothing needs
    would pass the completeness test above forever.
    """
    kept = tuple(leaf for leaf in cwi.LEAVES if leaf != dropped)
    config_path = _staged_config(tmp_path, kept)
    proc = _dry_run(DOWNSTREAM[MINIMALITY_WORKFLOW], config_path)

    assert proc.returncode != 0, (
        f"{dropped} is in LEAVES but the DAG builds without it. Either a rule "
        f"stopped declaring it — drop it from LEAVES — or it is a deliberate "
        f"non-leaf and belongs beside EXTRA_REGION instead."
    )
    combined = proc.stdout + proc.stderr
    assert Path(dropped).name in combined, (
        f"{MINIMALITY_WORKFLOW} failed without {dropped}, but its error does not "
        f"name that file, so the failure may be unrelated.\n{combined[-2000:]}"
    )
