"""Test functions from the model creation workflow."""

import os
from os.path import join, dirname, realpath
import pytest

from blueearth_cst.model import copy_config_files

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")


def test_copy_config(project_dir, data_sources, model_build_config):
    """Config files are snapshotted into the bin their KIND belongs in.

    R07 B9 replaced the single derived ``output_dir`` with explicit per-file
    routing -- a signature change, because one directory cannot serve
    runs/catalogs/templates/generated.
    """
    cfg = join(project_dir, "config")
    copy_config_files.copy_config_files(
        config=config_fn,
        config_out_path=join(cfg, "runs", "snake_config_model_creation.yml"),
        other_config_files={
            data_sources: join(cfg, "catalogs"),
            model_build_config: join(cfg, "templates"),
        },
    )

    assert os.path.exists(f"{project_dir}/config/runs/snake_config_model_creation.yml")
    assert os.path.exists(f"{project_dir}/config/templates/wflow_build_model.yml")
    assert os.path.exists(f"{project_dir}/config/catalogs/tests_data_catalog.yml")
