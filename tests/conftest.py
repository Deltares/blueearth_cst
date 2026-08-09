"""Global test attributes and fixtures"""

from os.path import dirname, join, realpath

import pytest
import yaml

# The repo root is on sys.path via `pythonpath = ["."]` in pyproject.toml
# [tool.pytest.ini_options] (O-14 decision 1), applied before conftest is
# imported -- which is why this module-level import resolves with no
# sys.path.insert shim. 34 such inserts were removed from tests/ once the
# declarative setting replaced them. The remaining inserts in this directory
# point at dev/scripts/ and scripts/, which are NOT packages and are not
# shipped; those stay.
from blueearth_cst.shared.snake_utils import get_config  # shared helper (R3 §3)

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run slow end-to-end workflow tests (need the data mirror + Julia)",
    )


# The `integration` marker is DECLARED in pyproject.toml [tool.pytest.ini_options]
# (O-14 decision 1), not registered programmatically here — one source of truth,
# and visible to `pytest --markers` and to anyone reading the repo config.
# Registering it in both places produced a duplicate entry in --markers output.
# The --run-integration option and the skip logic below stay here: they are
# behaviour, not configuration.


def pytest_collection_modifyitems(config, items):
    """Skip integration-marked tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture()
def config():
    """Return config dictionary"""
    with open(config_fn, "rb") as f:
        cfdict = yaml.safe_load(f)
    return cfdict


@pytest.fixture()
def project_dir(config):
    """Return project directory"""
    project_dir = get_config(config["project"], "project_dir", optional=False)
    project_dir = join(SNAKEDIR, project_dir)
    return project_dir


@pytest.fixture()
def data_sources(config):
    """Return data sources"""
    data_sources = get_config(config["project"], "data_sources", optional=False)
    data_sources = join(SNAKEDIR, data_sources)
    return data_sources


@pytest.fixture()
def model_build_config(config):
    """Return model build config"""
    model_build_config = get_config(
        config["workflows"]["model_creation"], "model_build_config", optional=False
    )
    model_build_config = join(SNAKEDIR, model_build_config)
    return model_build_config
