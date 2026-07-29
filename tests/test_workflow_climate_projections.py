"""End-to-end smoke test for the climate-projections workflow (workflow 2).

Like ``test_workflow_model_creation.py`` but for workflow 2: it runs
``snakemake all`` on ``Snakefile_climate_projections`` to completion and checks
that the CMIP6 change-factor summary is produced.

Workflow 2 reads CMIP6 monthly data from Google Cloud Storage (``gs://cmip6/...``
via gcsfs), so this needs internet access — it does not use the local data mirror
or Julia. It also depends on ``region.geojson`` from workflow 1 (an
``ancient(...)`` cross-Snakefile input that Snakemake will not build itself), so
run the model-creation workflow (or its smoke test) first.

Run it with::

    pixi run pytest tests/test_workflow_climate_projections.py --run-integration

Skipped by default, and self-skips if ``region.geojson`` is missing or GCS is
unreachable.
"""

import os
import socket
import subprocess
from os.path import join, dirname, realpath, exists, getsize

import yaml
import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

CONFIG = "config/snake_config_model_test.yml"

pytestmark = pytest.mark.integration


# R01 sectioned schema: project_dir lives under `project`, clim_project under
# `workflows.climate_projections`. Read lazily (inside the test), never at
# import time, so the migration cannot break collection of the whole suite.
_CFG_PATHS = {
    "project_dir": ("project", "project_dir"),
    "clim_project": ("workflows", "climate_projections", "clim_project"),
}


def _cfg(key):
    with open(join(SNAKEDIR, CONFIG)) as f:
        cfg = yaml.safe_load(f)
    node = cfg
    for part in _CFG_PATHS[key]:
        node = node[part]
    return node


def _gcs_reachable():
    try:
        socket.create_connection(("storage.googleapis.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


def test_climate_projections_end_to_end():
    """Force a full rebuild of workflow 2 and assert the change summary is produced."""
    project_dir = _cfg("project_dir")
    clim_project = _cfg("clim_project")
    region = join(
        SNAKEDIR, project_dir, "hydrology_model", "staticgeoms", "region.geojson"
    )
    summary_csv = join(
        SNAKEDIR,
        project_dir,
        "climate_projections",
        clim_project,
        "summary",
        "annual_change_scalar_stats_summary.csv",
    )
    if not exists(region):
        pytest.skip(f"missing {region}; run the model-creation workflow first")
    if not _gcs_reachable():
        pytest.skip(
            "Google Cloud Storage (storage.googleapis.com:443) not reachable"
        )

    os.chdir(SNAKEDIR)
    cmd = (
        f"snakemake all -c 1 -s Snakefile_climate_projections "
        f"--configfile {CONFIG} --forceall"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"snakemake exited {result.returncode}\n"
        f"--- stderr (tail) ---\n{(result.stderr or '')[-4000:]}"
    )
    assert exists(summary_csv), f"expected output not created: {summary_csv}"
    assert getsize(summary_csv) > 0, f"output is empty: {summary_csv}"
