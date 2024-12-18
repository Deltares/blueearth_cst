"""Global test attributes and fixtures"""

import os
from os.path import join, dirname, realpath
import yaml
import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")


# Function to get argument from config file and return default value if not found
def get_config(config, arg, default=None, optional=True):
    """
    Function to get argument from config file and return default value if not found

    Parameters
    ----------
    config : dict
        config file
    arg : str
        argument to get from config file
    default : str/int/float/list, optional
        default value if argument not found, by default None
    optional : bool, optional
        if True, argument is optional, by default True
    """
    if arg in config:
        return config[arg]
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config file")


@pytest.fixture()
def config():
    """Return config dictionary"""
    with open(config_fn, "rb") as f:
        cfdict = yaml.safe_load(f)
    return cfdict


@pytest.fixture()
def project_dir(config):
    """Return project directory"""
    project_dir = get_config(config, "project_dir", optional=False)
    project_dir = join(SNAKEDIR, project_dir)
    return project_dir


@pytest.fixture()
def data_sources(config):
    """Return data sources"""
    data_sources = get_config(config, "data_sources", optional=False)
    data_sources = join(SNAKEDIR, data_sources)
    return data_sources


@pytest.fixture()
def model_build_config(config):
    """Return model build config"""
    model_build_config = get_config(config, "model_build_config", optional=False)
    model_build_config = join(SNAKEDIR, model_build_config)
    return model_build_config
