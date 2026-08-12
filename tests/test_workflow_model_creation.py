"""End-to-end smoke test for the model-creation workflow (workflow 1).

Unlike ``test_cli.py`` (which only ``--dry-run``s the DAG), this actually runs
``snakemake all`` on the small test basin: it builds the Wflow model with
hydromt, adds forcing, and runs Wflow.jl to completion, then checks that the
discharge output was produced. This is the check that would have caught the
data-catalog and Julia failures that a dry-run cannot see.

It is opt-in and slow. It needs:
  * the local Deltares data mirror referenced by ``config/deltares_data.yml``
  * juliaup-managed Julia 1.11.7 with the project instantiated (``pixi run install``)

Run it with::

    pixi run pytest tests/test_workflow_model_creation.py --run-integration

Skipped by default, and self-skips if the data mirror or Julia is absent.
"""

import os
import shutil
import subprocess
from os.path import dirname, exists, getsize, join, realpath

import pytest
import yaml

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

# The config that actually runs end-to-end (project_dir: test_case/test_local,
# data_sources: the full config/deltares_data.yml mirror).
#
# All three of these were stale and none of them could fail a gate: this module
# is `pytest.mark.integration`, so it only runs under --run-integration, which
# no CI leg passes. R7 moved the config under `config/workflows/` and R9 moved
# the model to `models/hydrology/wflow`. The config path was the worse of the
# two -- `_catalog_root()` opens it, so the module ERRORED on a missing file
# rather than reaching its own data-mirror skip. Corrected 2026-08-09 against
# the paths on disk and the baseline manifest's own wf1 discharge target.
CONFIG = "test_case/snake_config_baseline.yml"
# The DERIVED table, not wflow's `output.csv`: since 2026-08-10 rule 1.14
# declares the raw csv as temp(), so this rule-`all` run consumes it in 1.14b
# and 1.15 and then drops it. Asserting the raw file here would fail on a
# correct run. This is the better target anyway -- it is the deliverable, and
# it is non-empty only if the wflow run that fed it produced rows.
OUTPUT_CSV = join(
    SNAKEDIR,
    "test_case",
    "test_local",
    "models",
    "hydrology",
    "wflow",
    "run_default",
    "output_q.csv",
)

pytestmark = pytest.mark.integration


def _catalog_root():
    """First data root declared by the config's data catalog, or None.

    Read lazily (inside the test), never at import time: the R1 config-schema
    migration must not be able to break collection of the whole suite through a
    module-level read here.
    """
    with open(join(SNAKEDIR, CONFIG)) as f:
        cfg = yaml.safe_load(f)
    # R01 sectioned schema: data_sources lives under project.
    with open(join(SNAKEDIR, cfg["project"]["data_sources"])) as f:
        cat = yaml.safe_load(f)
    meta = cat.get("meta", {}) or {}
    roots = meta.get("roots") or ([meta["root"]] if "root" in meta else [])
    return roots[0] if roots else None


def test_model_creation_end_to_end():
    """Force a full rebuild of workflow 1 and assert Wflow output is produced."""
    root = _catalog_root()
    if root is None or not exists(root):
        pytest.skip(f"data mirror not found (catalog root: {root})")
    if shutil.which("julia") is None:
        pytest.skip("julia not on PATH (juliaup-managed Julia 1.11.7 required)")

    os.chdir(SNAKEDIR)
    cmd = (
        f"snakemake all -c 1 -s Snakefile_model_creation "
        f"--configfile {CONFIG} --forceall"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"snakemake exited {result.returncode}\n"
        f"--- stderr (tail) ---\n{(result.stderr or '')[-4000:]}"
    )
    assert exists(OUTPUT_CSV), f"expected output not created: {OUTPUT_CSV}"
    assert getsize(OUTPUT_CSV) > 0, f"output is empty: {OUTPUT_CSV}"
