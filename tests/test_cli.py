"""Test some snake command line interface (CLI) for validity of snakefiles."""

import os
from os.path import join, dirname, realpath
import subprocess
import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")

_snakefiles = [
    "Snakefile_model_creation",
    "Snakefile_climate_projections",
    "Snakefile_climate_experiment",
]


@pytest.mark.parametrize("snakefile", _snakefiles)
def test_snakefile_cli(snakefile):
    # Test if snake command line runs successfully
    # snakemake all -c 1 -s Snakefile_model_creation --configfile tests/snake_config_model_test.yml --dry-run
    # move to SNAKEDIR
    os.chdir(SNAKEDIR)
    cmd = f"snakemake all -c 1 -s {snakefile} --configfile {config_fn} --dry-run"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # Check the output of the subprocess command
    assert result.returncode == 0
    # assert result.stdout == b'Hello, world!\n'
