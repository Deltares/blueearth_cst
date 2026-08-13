"""Test functions from the model creation workflow."""

import os
from os.path import dirname, join, realpath

from blueearth_cst.model import copy_config_files

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_fixture.yml")


def test_copy_config(tmp_path, data_sources, model_build_config):
    """A referenced file is copied ONLY when the repo cannot give it back.

    R07 B9 replaced the single derived ``output_dir`` with explicit per-file
    routing. The config-snapshot redesign (2026-08-13) then made the COPY
    itself conditional: a file inside the checkout, tracked and clean, is
    recorded by its git blob id instead, because duplicating what version
    control already holds serves nobody.

    Hermetic on ``tmp_path``, not the shared ``project_dir`` fixture. The
    previous version asserted both tracked files WERE copied and passed locally
    for a reason unrelated to the code: ``tests/test_project/`` is untracked and
    persists between runs, so copies made before the predicate existed were
    still lying there. CI builds a fresh tree and the test failed there and only
    there -- the exact "green locally, red on the other leg" shape this repo has
    been bitten by before.
    """
    cfg = tmp_path / "config"
    outside = tmp_path / "site_specific_catalog.yml"
    outside.write_text("meta: {}\n", encoding="utf-8")

    copy_config_files.copy_config_files(
        config=config_fn,
        config_out_path=join(cfg, "runs", "snake_config_model_creation.yml"),
        other_config_files={
            data_sources: join(cfg, "catalogs"),
            model_build_config: join(cfg, "templates"),
            str(outside): join(cfg, "catalogs"),
        },
    )

    # The flat copy is unconditional -- the drift guard reads it, and it is
    # baseline-fingerprinted in a real project.
    assert os.path.exists(join(cfg, "runs", "snake_config_model_creation.yml"))

    # Both fixture inputs are TRACKED toolbox files, so neither is copied.
    assert not os.path.exists(join(cfg, "templates", "wflow_build_model.yml"))
    assert not os.path.exists(join(cfg, "catalogs", "tests_data_catalog.yml"))

    # A file the repository has never held is copied, which is the whole point
    # of the predicate being per-FILE rather than per-bin.
    assert os.path.exists(join(cfg, "catalogs", "site_specific_catalog.yml"))
