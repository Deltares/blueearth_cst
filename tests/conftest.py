"""Global test attributes and fixtures"""

from os.path import join, dirname, realpath, splitext
from pathlib import Path
import numpy as np
from typing import Dict, Union, List
import yaml
import pytest

from ..snakemake.get_config import get_config

TESTDIR = dirname(realpath(__file__))
MAINDIR = join(TESTDIR, "..")
SAMPLE_PROJECTDIR = join(TESTDIR, "test_project_sample")

config_fn = join(TESTDIR, "snake_config_model_test.yml")
config_fao_fn = join(TESTDIR, "snake_config_fao_test.yml")


@pytest.fixture()
def config() -> Dict:
    """Return config dictionary"""
    with open(config_fn, "rb") as f:
        cfdict = yaml.safe_load(f)
    return cfdict


@pytest.fixture()
def config_fao() -> Dict:
    """Return config dictionary"""
    with open(config_fao_fn, "rb") as f:
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
    """Return data sources from fao config"""
    data_libs = np.atleast_1d(config["data_sources"]).tolist()
    data_sources = []
    for source in data_libs:
        ext = splitext(source)[-1]
        if not len(ext) == 0:
            data_sources.append(join(MAINDIR, source))
        else:
            # predefined catalogs if no file extension
            data_sources.append(source)
    return data_sources


@pytest.fixture()
def data_sources_climate(config) -> Union[str, Path]:
    """Return data sources climate"""
    data_sources_climate = get_config(config, "data_sources_climate", optional=False)
    data_sources_climate = join(MAINDIR, data_sources_climate)
    return data_sources_climate


@pytest.fixture()
def data_libs_fao(config_fao) -> List:
    """Return data sources from fao config"""
    data_libs = np.atleast_1d(config_fao["data_sources"]).tolist()
    data_libs_fao = []
    for source in data_libs:
        ext = splitext(source)[-1]
        if not len(ext) == 0:
            data_libs_fao.append(join(MAINDIR, source))
        else:
            # predefined catalogs if no file extension
            data_libs_fao.append(source)
    return data_libs_fao


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
