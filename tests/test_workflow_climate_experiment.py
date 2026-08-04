"""End-to-end smoke test for the climate-experiment workflow (workflow 3).

Like the workflow 1 and 2 smoke tests, this actually runs ``snakemake all`` to
completion — here on ``Snakefile_climate_experiment``: it extracts historical
climate for the basin, generates stochastic weather realizations (weathergenr,
R), perturbs them across the stress-test grid, runs Wflow.jl for every
realization x stress-test combination, and reduces the runs to the Qstats/basin
indicator tables. On the test config that is 2 realizations x 6 stress-test
cells = 12 Wflow runs, so expect a long wall-clock time (tens of minutes).

It needs:
  * the local Deltares data mirror referenced by ``config/deltares_data.yml``
  * workflow 1 artifacts under ``test_case/test/hydrology_model`` (run the
    model-creation workflow or its smoke test first)
  * juliaup-managed Julia 1.11.7 with the project instantiated
  * the weathergenr R package in the pixi R library (``pixi run install``)

Run it with::

    pixi run pytest tests/test_workflow_climate_experiment.py --run-integration

Skipped by default, and self-skips if any prerequisite is absent. Config is
read lazily inside the test (never at import time) so a config-schema change
cannot break suite collection.

Note: the known ``CyclicGraphException`` xfail in ``test_cli.py`` does not
apply here — it only triggers on the ``tests/`` config whose project_dir has
no workflow 1 outputs. On this config the DAG resolves (verified: 56 jobs
under ``--forceall``). Tracked in ``dev/followups.md`` under R3.
"""

import os
import shutil
import subprocess
from os.path import join, dirname, realpath, exists, getsize

import yaml
import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

CONFIG = "config/snake_config_model_test.yml"

pytestmark = pytest.mark.integration


def _load_cfg():
    with open(join(SNAKEDIR, CONFIG)) as f:
        return yaml.safe_load(f)


def _catalog_root(cfg):
    """First data root declared by the config's data catalog, or None."""
    # R01 sectioned schema: data_sources lives under project.
    with open(join(SNAKEDIR, cfg["project"]["data_sources"])) as f:
        cat = yaml.safe_load(f)
    meta = cat.get("meta", {}) or {}
    roots = meta.get("roots") or ([meta["root"]] if "root" in meta else [])
    return roots[0] if roots else None


def _weathergenr_available():
    """True if Rscript can load the weathergenr package."""
    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=as.integer(!requireNamespace('weathergenr', quietly=TRUE)))",
        ],
        capture_output=True,
    )
    return result.returncode == 0


def test_climate_experiment_end_to_end():
    """Force a full rebuild of workflow 3 and assert the indicator tables are produced."""
    cfg = _load_cfg()
    # R01 sectioned schema: project_dir under project, experiment_name under
    # workflows.climate_experiment.
    project_dir = cfg["project"]["project_dir"]
    experiment = cfg["workflows"]["climate_experiment"]["experiment_name"]

    root = _catalog_root(cfg)
    if root is None or not exists(root):
        pytest.skip(f"data mirror not found (catalog root: {root})")

    region = join(
        SNAKEDIR, project_dir, "hydrology_model", "staticgeoms", "region.geojson"
    )
    if not exists(region):
        pytest.skip(f"missing {region}; run the model-creation workflow first")

    if shutil.which("julia") is None:
        pytest.skip("julia not on PATH (juliaup-managed Julia 1.11.7 required)")

    if shutil.which("Rscript") is None or not _weathergenr_available():
        pytest.skip("weathergenr not loadable via Rscript (run `pixi run install`)")

    os.chdir(SNAKEDIR)
    cmd = (
        f"snakemake all -c 1 -s Snakefile_climate_experiment "
        f"--configfile {CONFIG} --forceall"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"snakemake exited {result.returncode}\n"
        f"--- stderr (tail) ---\n{(result.stderr or '')[-4000:]}"
    )
    for name in ("q_indicators.csv", "basin_indicators.csv"):
        out = join(
            SNAKEDIR, project_dir, "experiments", experiment, "results", name
        )
        assert exists(out), f"expected output not created: {out}"
        assert getsize(out) > 0, f"output is empty: {out}"
