"""R07 O-24 / O-08: wf1's figure outputs are declared, and the sentinel is read.

* ``test_delete_all_output_removes_the_declared_plot_outputs`` — O-24. Before
  R07, rule 1.13 wrote three PNGs and declared one, and rule 1.11 wrote
  ``clim_wflow_1_{month,year}.png`` + ``performance_metrics.csv`` and declared
  none of them; undeclared outputs survive ``--delete-all-output`` and are
  invisible to the baseline. The claim is scoped to configs without extra
  gauges: ``signatures_{station}.png`` and the per-station
  ``clim_{station}_{period}.png`` stay undeclared because their COUNT is the
  model's outlet/subcatchment count — a rule-1.03 product, unknown at parse
  time — so this asserts the config-invariant subset and nothing wider.

* ``test_delete_all_output_removes_a_basavg_figure`` — the half of O-24 that
  IS derivable. ``plot_basavg``'s PNGs are a pure function of
  ``wflow_outvars``, so rule 1.11 declares them (2026-08-01); this proves the
  derivation reaches ``--delete-all-output`` for a config that has one, and the
  seed config (``wflow_outvars: ["river discharge"]``) still declares none.

* ``test_gauges_layer_name_*`` — O-08. An unquoted ``output_locations: None``
  is YAML for the Python **string** ``"None"``. The pre-R07 guard tested only
  ``is not None`` and derived the layer name ``gauges_None``, which can never
  exist in ``geoms`` — so the gauges were dropped silently rather than
  deliberately. The shipped configs moved to a real ``null`` in 2026-08, but the
  string stays tolerated for project configs still carrying it, so both
  spellings must keep resolving to "unset".
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from blueearth_cst.climate_analysis.climate_figures import figure_names as _figure_names

#: A model's staticgeoms layers, spelled as hydromt_wflow actually writes them
#: (`output_locations.csv` -> `gauges_output-locations`, note the HYPHEN).
_GAUGES_LAYER = "gauges_output-locations"
_GEOMS = {"basins", "rivers", "outlets", _GAUGES_LAYER}

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_model_test.yml"


#: The config-invariant subset O-24 declares, project-root-relative. The
#: forcing entries come from climate_figures rather than being restated, so a
#: change to the canonical set cannot leave this list quietly behind.
DECLARED_PLOT_OUTPUTS = (
    "models/hydrology/wflow/evaluation/plots/hydro_wflow_1.png",
    "models/hydrology/wflow/evaluation/plots/clim_wflow_1_month.png",
    "models/hydrology/wflow/evaluation/plots/clim_wflow_1_year.png",
    "models/hydrology/wflow/evaluation/performance_metrics.csv",
    # Rule 1.12 renders once and writes both: PDF for publication, PNG preview.
    "models/hydrology/wflow/plots/basin_area.pdf",
    "models/hydrology/wflow/plots/basin_area.png",
) + tuple(
    f"models/hydrology/wflow/forcing/plots/{name}"
    for name in _figure_names("forcing")
)


# --------------------------------------------------------------------------- #
# O-24 — the declarations are real
# --------------------------------------------------------------------------- #


@pytest.fixture()
def fabricated_project(tmp_path):
    """A project_dir pre-filled with every declared wf1 figure output."""
    from blueearth_cst.shared.snake_utils import climate_store_spec

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    cfg["project"]["project_dir"] = project_dir.as_posix()
    # Repo-relative leaves (templates, catalog) keep working: snakemake runs
    # with cwd=SNAKEDIR.
    cfg_path = tmp_path / "snake_config_fabricated.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    spec = climate_store_spec(
        project_dir=project_dir.as_posix(),
        model_region=cfg["shared"]["basin"]["region"],
        clim_source=cfg["shared"]["clim_historical"],
        historical_window=cfg["shared"]["historical_window"],
        data_sources=cfg["project"]["data_sources"],
    )
    store_plots = Path(spec.store_dir, "plots")
    expected = [project_dir / rel for rel in DECLARED_PLOT_OUTPUTS]
    expected += [store_plots / name for name in _figure_names("source")]
    # Knowingly UNDECLARED (config-dependent): it must survive, which is what
    # makes the assertion below a discriminating check rather than a tautology
    # about an emptied directory.
    undeclared = project_dir / "models/hydrology/wflow/evaluation/plots/signatures_wflow_1.png"
    for path in [*expected, undeclared]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    return cfg_path, expected, undeclared


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not on PATH")
@pytest.mark.workflow_contract
def test_delete_all_output_removes_the_declared_plot_outputs(fabricated_project):
    cfg_path, expected, undeclared = fabricated_project
    assert all(p.is_file() for p in expected)

    result = subprocess.run(
        "snakemake all --delete-all-output --workflow-profile none -c 1 "
        f'-s Snakefile_model_creation --configfile "{cfg_path}"',
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(SNAKEDIR),
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-4000:]

    still_there = [p.as_posix() for p in expected if p.exists()]
    assert not still_there, (
        "these declared outputs survived --delete-all-output:\n"
        + "\n".join(still_there)
    )
    assert undeclared.is_file(), (
        "the knowingly-undeclared control file was removed too — the assertion "
        "above no longer discriminates declared from undeclared outputs"
    )


#: What rule 1.11 derives from wflow_outvars: the CSV column name verbatim,
#: spaces and all (func_plot_signature.plot_basavg writes f"{dvar}.png").
_BASAVG_REL = (
    "models/hydrology/wflow/evaluation/plots/actual evapotranspiration_basavg.png"
)


@pytest.fixture()
def project_with_basavg_outvar(tmp_path):
    """A project whose wflow_outvars asks for a basin-average output.

    The seed config is discharge-only, so nothing in the suite would otherwise
    exercise the derivation — or the fact that the derived filename contains a
    SPACE, which every consumer of a declared output has to survive.
    """
    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    cfg["project"]["project_dir"] = project_dir.as_posix()
    cfg["workflows"]["model_creation"]["wflow_outvars"] = [
        "river discharge",
        "actual evapotranspiration",
    ]
    cfg_path = tmp_path / "snake_config_basavg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    basavg = project_dir / _BASAVG_REL
    basavg.parent.mkdir(parents=True, exist_ok=True)
    basavg.write_bytes(b"placeholder")
    # Same control as the fixture above: an undeclared sibling must survive.
    undeclared = project_dir / "models/hydrology/wflow/evaluation/plots/signatures_wflow_1.png"
    undeclared.write_bytes(b"placeholder")
    return cfg_path, basavg, undeclared


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not on PATH")
@pytest.mark.workflow_contract
def test_delete_all_output_removes_a_basavg_figure(project_with_basavg_outvar):
    cfg_path, basavg, undeclared = project_with_basavg_outvar

    result = subprocess.run(
        "snakemake all --delete-all-output --workflow-profile none -c 1 "
        f'-s Snakefile_model_creation --configfile "{cfg_path}"',
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(SNAKEDIR),
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-4000:]
    assert not basavg.exists(), (
        f"{basavg.name} survived --delete-all-output, so rule 1.11 is not "
        "really declaring the wflow_outvars-derived figures"
    )
    assert undeclared.is_file(), (
        "the undeclared control file was removed too — the assertion above no "
        "longer discriminates declared from undeclared outputs"
    )


def test_river_discharge_alone_derives_no_basavg_figure():
    """The exclusions are load-bearing, so pin them without a snakemake run.

    'river discharge' never gets a basavg column (rule 1.05 filters it out of
    the basin-average setup), and 'precipitation' gets one that plot_results
    drops before plotting — so neither may contribute a declared output.
    """
    source = (SNAKEDIR / "Snakefile_model_creation").read_text(encoding="utf-8")
    # Read the derivation out of the Snakefile rather than restating it here,
    # so a change to the exclusion tuple is caught instead of duplicated.
    namespace = {}
    for line in source.splitlines():
        if line.startswith("_WFLOW_OUTVARS_WITHOUT_BASAVG_PLOT"):
            exec(line, namespace)  # noqa: S102 - a literal tuple from our own tree
            break
    excluded = namespace["_WFLOW_OUTVARS_WITHOUT_BASAVG_PLOT"]
    assert set(excluded) == {"river discharge", "precipitation"}
    assert [v for v in ["river discharge"] if v not in excluded] == []
    assert [
        v for v in ["river discharge", "actual evapotranspiration"] if v not in excluded
    ] == ["actual evapotranspiration"]


# --------------------------------------------------------------------------- #
# O-08 — the "None" sentinel
# --------------------------------------------------------------------------- #


def test_gauges_layer_name_rejects_both_unset_spellings():
    from blueearth_cst.shared.gauges import gauges_layer_name

    assert gauges_layer_name(_GEOMS, None) is None
    assert gauges_layer_name(_GEOMS, "None") is None


def test_gauges_layer_name_resolves_a_real_layer():
    """Resolved against the model, NOT derived from the filename.

    This asserted ``gauges_output_locations`` until 2026-08-01 — the underscore
    spelling hydromt never produces. The test agreed with the code and both were
    wrong, which is why the real basin found the bug and the suite did not.
    ``tests/test_gauges.py`` owns the resolution rules; this keeps the O-08
    entry point covered from here.
    """
    from blueearth_cst.shared.gauges import gauges_layer_name

    assert gauges_layer_name(_GEOMS, "d/output_locations.csv") == _GAUGES_LAYER
    assert gauges_layer_name(_GEOMS, Path("d/output_locations.csv")) == _GAUGES_LAYER


def test_the_shipped_sentinel_yields_no_layer():
    """Whatever spelling the shipped config uses, it must resolve to "unset".

    Was pinned to the STRING "None" until 2026-08; the shipped configs now use a
    real YAML null and the string survives only as a tolerated legacy spelling
    (tests/test_cli.py owns both halves of that). What matters HERE is narrower
    and does not care which spelling won: the value the config actually carries
    must not produce a layer name. If someone reverts the O-08 guard, this fires
    for the string; if a real null ever stopped short-circuiting, it fires for
    that.
    """
    from blueearth_cst.shared.gauges import gauges_layer_name

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    sentinel = cfg["shared"]["basin"]["gauge_points"]
    assert sentinel in (None, "None"), (
        f"unexpected gauge_points sentinel {sentinel!r} — if the config now "
        f"names a real file this test needs rethinking, not relaxing"
    )
    assert gauges_layer_name(_GEOMS, sentinel) is None
