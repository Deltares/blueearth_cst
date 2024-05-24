"""Test some snake command line interface (CLI) for validity of snakefiles."""

import os
from os.path import join, dirname, realpath
import subprocess
import pytest

TESTDIR = dirname(realpath(__file__))
MAINDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")
fao_config_fn = join(TESTDIR, "snake_config_fao_test.yml")

_snakefiles = {
    "model_creation": {
        "file": "Snakefile_historical_hydrology.smk",
        "config": config_fn,
    },
    "climate_projections": {
        "file": "Snakefile_climate_projections.smk",
        "config": config_fn,
    },
    "climate_experiment": {
        "file": "Snakefile_climate_experiment.smk",
        "config": config_fn,
    },
    "climate_historical": {
        "file": "Snakefile_climate_historical.smk",
        "config": fao_config_fn,
    },
}


@pytest.mark.parametrize("snakefile", list(_snakefiles.keys()))
def test_snakefile_cli(snakefile):
    # Test if snake command line runs successfully
    # snakemake all -c 1 -s Snakefile_model_creation --configfile tests/snake_config_model_test.yml --dry-run
    # move to SNAKEDIR
    os.chdir(MAINDIR)
    snakefile_path = f"snakemake/{_snakefiles[snakefile]['file']}"
    configfile = _snakefiles[snakefile]["config"]
    cmd = f"snakemake all -c 1 -s {snakefile_path} --configfile {configfile} --dry-run"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # Check the output of the subprocess command
    assert result.returncode == 0
