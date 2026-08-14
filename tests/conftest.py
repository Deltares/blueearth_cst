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

config_fn = join(TESTDIR, "snake_config_fixture.yml")


@pytest.fixture(autouse=True)
def _no_ambient_path_tokens(monkeypatch):
    """Start every test with NO declared key folders.

    ``declare_path_tokens`` writes a process-wide env var, which is correct at
    runtime — a rule's output is written by a child of the process that parsed
    the Snakefile. In the suite it is leakage: four tests build rules through
    ``snakemake.api``, which parses a real Snakefile IN-PROCESS and therefore
    declares that project's folders for every test that runs afterwards. Three
    header tests then failed in the full suite while passing in isolation
    (measured 2026-08-14), which is the failure mode that costs the most to
    diagnose.

    Autouse and unconditional: no test should be reading a declaration it did
    not make, and a test that wants one makes it (see ``declare_folders`` in
    tests/test_snake_utils.py).
    """
    monkeypatch.setenv("CST_PATH_TOKENS", "")


#: Screen resolution, for tests that SAVE a figure to assert something about it.
#: Low enough to be cheap, high enough that a render still exercises the same
#: draw path; nothing here asserts on pixel counts.
TEST_FIGURE_DPI = 100


@pytest.fixture
def fast_figure_dpi(monkeypatch):
    """Write saved figures at screen resolution for the duration of one test.

    A test that asserts a figure's STRUCTURE — which panels exist, what they are
    called, which files were written — is answered identically at 100 dpi and at
    the 600 dpi export default, and 600 dpi costs seconds per sheet because the
    whole figure is rasterised at 4251 px wide. The resolution is a property of
    the shipped artifact, not of the assertion.

    Patched at EVERY binding site, because the export default is imported into
    four module namespaces rather than read through one: ``plot_evaluation``
    reads ``plot_style.RASTER_DPI`` at call time, while ``plot_map``,
    ``plot_spatial_maps`` and ``climate_figures`` bound their own copies at
    import. Patching one of them leaves the others at 600 and the test looks
    like it got faster without having got faster. ``hasattr`` guards the loop so
    a module that stops importing the name does not turn this into an error.

    Opt in per module with ``pytestmark = pytest.mark.usefixtures(...)``, not
    autouse: a test that DOES care about the export resolution should get the
    real one, and an autouse fixture would take it away invisibly.

    Opt in only where it MEASURES faster. ``test_plot_climate_source`` was
    marked and then unmarked: its slow test costs 10 s in climate preparation,
    not in rasterisation, and the marker bought nothing. ``climate_figures``
    stays in the list below regardless — the list is what makes the patch
    correct wherever it is used, not a record of who uses it.
    """
    from blueearth_cst.climate_analysis import climate_figures
    from blueearth_cst.shared import plot_map, plot_spatial_maps, plot_style

    for module in (plot_style, plot_map, plot_spatial_maps, climate_figures):
        if hasattr(module, "RASTER_DPI"):
            monkeypatch.setattr(module, "RASTER_DPI", TEST_FIGURE_DPI)
    return TEST_FIGURE_DPI


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
