"""Global test attributes and fixtures"""

from copy import deepcopy
from os.path import dirname, join, realpath
from pathlib import Path

import pytest
import yaml

# The repo root is on sys.path via `pythonpath = ["."]` in pyproject.toml
# [tool.pytest.ini_options] (O-14 decision 1), applied before conftest is
# imported -- which is why this module-level import resolves with no
# sys.path.insert shim. 34 such inserts were removed from tests/ once the
# declarative setting replaced them. The remaining inserts in this directory
# point at dev/scripts/ and scripts/, which are NOT packages and are not
# shipped; those stay.
from blueearth_cst.shared.config_composition import (  # R13 loader (D-12.0, D-12.6)
    load_composed_config,
)
from blueearth_cst.shared.snake_utils import get_config  # shared helper (R3 §3)

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "project_config_fixture.yml")

pytest_plugins = ["dev.scripts.pytest_temp"]


@pytest.fixture
def isolated_snakemake(tmp_path, monkeypatch):
    """Keep real CLI calls, including launcher children, out of repo state."""
    from tests.snakemake_state import prepare_workdir, write_profile

    workdir = prepare_workdir(tmp_path / "snakemake")
    monkeypatch.setenv("SNAKEMAKE_PROFILE", str(write_profile(workdir)))
    monkeypatch.delenv("SNAKEMAKE_WORKFLOW_PROFILE", raising=False)
    return workdir


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
    monkeypatch.setenv("CST_CONSOLE_PATH_TOKENS", "")


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
    more than one module namespace rather than read through one:
    ``plot_evaluation`` reads ``plot_style.RASTER_DPI`` at call time, while
    ``plot_spatial_maps`` bound its own copy at import. Patching one of them leaves the others at 600 and the test looks
    like it got faster without having got faster. ``hasattr`` guards the loop so
    a module that stops importing the name does not turn this into an error.

    Opt in per module with ``pytestmark = pytest.mark.usefixtures(...)``, not
    autouse: a test that DOES care about the export resolution should get the
    real one, and an autouse fixture would take it away invisibly.

    Opt in only where it MEASURES faster. ``test_plot_climate_source`` was
    marked and then unmarked: its slow test costs 10 s in climate preparation,
    not in rasterisation, and the marker bought nothing. The list below is what
    makes the patch correct wherever it is used, not a record of who uses it.
    """
    from blueearth_cst.shared import plot_spatial_maps, plot_style

    for module in (plot_style, plot_spatial_maps):
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
    """Return the fixture config as one whole mapping, composed from T1 + T2.

    Composed rather than raw-loaded so that consumers see the same shape a run
    sees: since R13 the file at ``config_fn`` is the project file only, and a
    raw load would hand every consumer two-key workflow stanzas.
    """
    return load_composed_config(config_fn)


@pytest.fixture()
def project_dir(config):
    """Return project directory"""
    project_dir = get_config(config["project"], "project_dir", optional=False)
    project_dir = join(SNAKEDIR, project_dir)
    return project_dir


@pytest.fixture()
def data_sources(config):
    """Return data sources"""
    # `C-40`: `project.data_sources` is `project.catalog`.
    data_sources = get_config(config["project"], "catalog", optional=False)
    data_sources = join(SNAKEDIR, data_sources)
    return data_sources


@pytest.fixture()
def model_build_config(config):
    """Return model build config, read through the composed document.

    Composes rather than indexing a raw load: since R13 the settings live in
    ``tests/project_config_fixture_build_model.yml`` and the project file carries
    only ``{enabled, config_path}``, so a raw index finds nothing. This fixture
    sits in the layer that cannot fail in CI or in any worktree, which is why it
    goes through the same loader a run does rather than a second reader.
    """
    engine = get_config(config["workflows"]["build_model"], "engine", optional=False)
    model_build_config = get_config(engine, "build_config", optional=False)
    return join(SNAKEDIR, model_build_config)


def write_config(tmp_path, cfg, stem: str = "project_config") -> Path:
    """Split a whole-config mapping into T1 + T2 files under ``tmp_path``.

    Returns the T1 path, ready to hand to ``--configfile``. The inverse of
    ``compose_config``, for fixtures — and the reason the suite's dominant
    config idiom did not need rewriting when the config layout split:

        cfg = load_composed_config(CONFIG_FN)   # one whole mapping, as before
        cfg["workflows"]["build_model"]["x"] = ...       # mutate, as before
        cfg_path = write_config(tmp_path, cfg)  # was: safe_dump to one file

    A workflow section that is empty once ``enabled`` is removed gets no file
    and no ``config_path``, matching what the migration splitter emits and what
    the loader treats as "this workflow has no settings". Anything outside
    ``project`` / ``shared`` / ``workflows`` is hoisted into its owning
    workflow's file, so a test can keep writing ``reporting:`` at the top level
    of the mapping it mutates.

    Correctness is gated by ``compose_config(write_config(cfg)) == cfg`` in
    tests/test_config_composition.py — without it this helper would be a second,
    unchecked way to build a config.
    """
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(dict(cfg))
    workflows = dict(cfg.pop("workflows", {}) or {})

    # No hoist step: `HOISTED_SECTIONS` retired in R14 P1 (D-10.1), so a T2
    # file's top-level sections are its own and nothing is lifted out of `cfg`.
    stanzas: dict[str, dict] = {}
    for name, section in workflows.items():
        section = dict(section or {})
        stanza = {}
        if "enabled" in section:
            stanza["enabled"] = section.pop("enabled")
        if section:
            t2 = tmp_path / f"{stem}_{name}.yml"
            t2.write_text(yaml.safe_dump(section, sort_keys=False), encoding="utf-8")
            stanza["config_path"] = t2.name
        stanzas[name] = stanza

    cfg["workflows"] = stanzas
    t1 = tmp_path / f"{stem}.yml"
    t1.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return t1


def parse_workflow(snakefile: str, config_path):
    """Parse a Snakefile in-process, as a non-producing invocation, and return it.

    Rules are built exactly as a real invocation builds them. ``--dry-run`` is on
    ``sys.argv`` for the duration, because WF0-WF2 refuse a producing parse
    without the launcher's capture context (``require_capture``); WF3 also
    needs ``CST_GENERATION_PHASE``, and parses as its source phase.
    ``wf_api._workflow`` is private on the pinned Snakemake; there is no public
    accessor for the parsed workflow.
    """
    import os
    import sys

    import snakemake.api as api

    argv = sys.argv[:]
    phase = os.environ.get("CST_GENERATION_PHASE")
    try:
        sys.argv.append("--dry-run")
        if Path(snakefile).name == "generate_scenarios.smk" and phase is None:
            os.environ["CST_GENERATION_PHASE"] = "source"
        with api.SnakemakeApi() as sa:
            wf_api = sa.workflow(
                resource_settings=api.ResourceSettings(cores=1),
                config_settings=api.ConfigSettings(configfiles=[Path(config_path)]),
                storage_settings=api.StorageSettings(),
                workflow_settings=api.WorkflowSettings(),
                snakefile=Path(SNAKEDIR) / snakefile,
                workdir=Path(SNAKEDIR),
            )
            workflow = wf_api._workflow
            workflow.include(workflow.main_snakefile, overwrite_default_target=True)
            return workflow
    finally:
        sys.argv[:] = argv
        if phase is None:
            os.environ.pop("CST_GENERATION_PHASE", None)
        else:
            os.environ["CST_GENERATION_PHASE"] = phase


def phase_env(snakefile) -> dict:
    """Subprocess environment for a direct Snakemake call on ``snakefile``.

    WF3 refuses to parse without the launcher's phase; a direct test call runs
    it as the source phase, the one that needs no frozen plan.
    """
    import os

    env = dict(os.environ)
    if Path(str(snakefile)).name == "generate_scenarios.smk":
        env.setdefault("CST_GENERATION_PHASE", "source")
    return env
