"""Test functions from the model creation workflow."""

import os
from os.path import join, dirname, realpath
import pytest

from ..src import copy_config_files

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")


def test_copy_config(project_dir, data_sources, model_build_config):
    """Test if config files are copied to project_dir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fn,
        output_dir=join(project_dir, "config"),
        config_out_name="snake_config_model_creation.yml",
        other_config_files=[data_sources, model_build_config],
    )

    # Check if config files are copied to project_dir/config folder
    assert os.path.exists(f"{project_dir}/config/snake_config_model_creation.yml")
    assert os.path.exists(f"{project_dir}/config/wflow_build_model.yml")
    assert os.path.exists(f"{project_dir}/config/tests_data_catalog.yml")
