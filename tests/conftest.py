"""Global test attributes and fixtures"""

from os.path import join, dirname, realpath
from pathlib import Path
from typing import Dict, Union
import yaml
import pytest

TESTDIR = dirname(realpath(__file__))
MAINDIR = join(TESTDIR, "..")
SAMPLE_PROJECTDIR = join(TESTDIR, "test_project_sample")

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
def config() -> Dict:
    """Return config dictionary"""
    with open(config_fn, "rb") as f:
        cfdict = yaml.safe_load(f)
    return cfdict


@pytest.fixture()
def project_dir(config) -> Path:
    """Return project directory"""
    project_dir = get_config(config, "project_dir", optional=False)
    project_dir = join(MAINDIR, project_dir)
    return project_dir


@pytest.fixture()
def data_sources(config) -> Union[str, Path]:
    """Return data sources"""
    data_sources = get_config(config, "data_sources", optional=False)
    data_sources = join(MAINDIR, data_sources)
    return data_sources


@pytest.fixture()
def model_build_config(config) -> Union[str, Path]:
    """Return model build config"""
    model_build_config = get_config(config, "model_build_config", optional=False)
    model_build_config = join(MAINDIR, model_build_config)
    return model_build_config


@pytest.fixture()
def waterbodies_config(config) -> Union[str, Path]:
    """Return waterbodies config"""
    waterbodies_config = get_config(config, "waterbodies_config", optional=False)
    waterbodies_config = join(MAINDIR, waterbodies_config)
    return waterbodies_config
