"""R07 O-24 / O-08: wf1's figure outputs are declared, and the sentinel is read.

* ``test_delete_all_output_removes_the_declared_plot_outputs`` — O-24. Before
  R07, rule 1.13 wrote three PNGs and declared one, and rule 1.11 wrote
  ``clim_wflow_1_{month,year}.png`` + ``performance_metrics.csv`` and declared
  none of them; undeclared outputs survive ``--delete-all-output`` and are
  invisible to the baseline. The claim is scoped to the **seed-config class**:
  ``plot_basavg``'s per-``wflow_outvars`` PNGs, ``signatures_{station}.png`` and
  the per-station ``clim_{station}_{period}.png`` stay knowingly undeclared
  (design § "O-24 scope, stated"), so this asserts the config-invariant subset
  and nothing wider.

* ``test_gauges_layer_name_*`` — O-08. ``output_locations: None`` in the shipped
  configs is unquoted YAML, i.e. the Python **string** ``"None"``. The pre-R07
  guard tested only ``is not None`` and derived the layer name ``gauges_None``,
  which can never exist in ``geoms`` — so the gauges were dropped silently
  rather than deliberately.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_model_test.yml"

sys.path.insert(0, str(SNAKEDIR))

#: The config-invariant subset O-24 declares, project-root-relative.
DECLARED_PLOT_OUTPUTS = (
    "hydrology_model/evaluation/plots/hydro_wflow_1.png",
    "hydrology_model/evaluation/plots/clim_wflow_1_month.png",
    "hydrology_model/evaluation/plots/clim_wflow_1_year.png",
    "hydrology_model/evaluation/performance_metrics.csv",
    "hydrology_model/plots/basin_area.png",
    "hydrology_model/forcing/plots/precip.png",
    "hydrology_model/forcing/plots/temp.png",
    "hydrology_model/forcing/plots/pet.png",
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
    expected += [store_plots / f"source_{v}.png" for v in ("precip", "temp", "pet")]
    # Knowingly UNDECLARED (config-dependent): it must survive, which is what
    # makes the assertion below a discriminating check rather than a tautology
    # about an emptied directory.
    undeclared = project_dir / "hydrology_model/evaluation/plots/signatures_wflow_1.png"
    for path in [*expected, undeclared]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    return cfg_path, expected, undeclared


@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not on PATH")
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


# --------------------------------------------------------------------------- #
# O-08 — the "None" sentinel
# --------------------------------------------------------------------------- #


def test_gauges_layer_name_rejects_both_unset_spellings():
    from blueearth_cst.shared.plot_map import gauges_layer_name

    assert gauges_layer_name(None) is None
    assert gauges_layer_name("None") is None


def test_gauges_layer_name_derives_a_real_layer():
    from blueearth_cst.shared.plot_map import gauges_layer_name

    assert gauges_layer_name("d/output_locations.csv") == "gauges_output_locations"
    assert gauges_layer_name(Path("d/output_locations.csv")) == "gauges_output_locations"


def test_shipped_sentinel_is_the_string_none_and_yields_no_layer():
    """The sentinel's on-disk spelling is unchanged; only the reader learned it.

    Guards both halves at once: if someone "fixes" the config to YAML ``null``
    the first assertion fires, and if someone reverts the guard the second does.
    """
    from blueearth_cst.shared.plot_map import gauges_layer_name

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    sentinel = cfg["workflows"]["model_creation"]["output_locations"]
    assert sentinel == "None" and isinstance(sentinel, str)
    assert gauges_layer_name(sentinel) is None
