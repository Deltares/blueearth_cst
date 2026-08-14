"""R07 B4 / P4: the climate figures come from the store, model-free.

Two properties, deliberately separate:

* ``test_source_figures_build_without_a_model`` — **the P4 assertion**, and a
  stated exit criterion of the milestone: a real Snakemake dry-run schedules
  only the source-figure/store subgraph, with neither the model-build rule nor
  its deliberately absent template in the DAG.

* ``test_source_grid_pet_*`` — the source-PET unit tests: the transform stays on
  the extraction grid, applies no lapse shift (``dem_model == dem_forcing``),
  and genuinely consumes the source orography.

Everything is hermetic: the fixtures are written by this module, so the suite
needs no data mirror and no network (the figures are plain matplotlib — no
cartopy basemap tiles).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_fixture.yml"


# The synthetic domain: a small era5-like grid around the test basin.
_XS = np.arange(9.00, 10.51, 0.25)
_YS = np.arange(0.00, 1.26, 0.25)
# >= snake_utils.MIN_HISTORICAL_YEARS (16), or WF1 rejects the generated config
# at parse time and the --touch step below never runs. Was 2018..2019 until
# 2026-08-01; only the store KEY derives from these dates (the synthetic grid
# carries a single timestamp), so widening the window is inert here.
_START, _END = "2000-01-01", "2019-12-31"


def _set_crs(obj):
    """Attach EPSG:4326 through hydromt's raster accessor (writes spatial_ref)."""
    import hydromt  # noqa: F401  registers the .raster accessor

    obj.raster.set_crs(4326)
    return obj


def _orography(xs=None, ys=None, offset: float = 0.0) -> xr.DataArray:
    """A smooth synthetic DEM named ``elevtn``, in metres.

    It carries a **scalar ``time`` coordinate** on purpose: the shipped
    ``era5_orography`` source is a single-timestep field, so a squeezed fetch
    leaves exactly that behind, and propagating it through ``reproject_like``
    destroys the climate array's time axis. That defect survived a fixture
    without the coordinate and only surfaced on a real run.
    """
    xs = _XS if xs is None else xs
    ys = _YS if ys is None else ys
    elev = (
        200.0
        + 300.0
        * np.sin(np.linspace(0, np.pi, ys.size))[:, None]
        * np.cos(np.linspace(0, np.pi, xs.size))[None, :]
    )
    da = xr.DataArray(
        (elev + offset).astype("float32"),
        dims=("y", "x"),
        coords={"y": ys, "x": xs, "time": pd.Timestamp("2018-01-01")},
        name="elevtn",
    )
    return _set_crs(da)


def _extraction() -> xr.Dataset:
    """A synthetic ``extract_historical.nc`` carrying the PET workflow's inputs."""
    time = pd.date_range(_START, _END, freq="D")
    doy = time.dayofyear.values.astype("float32")
    season = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    shape = (time.size, _YS.size, _XS.size)
    ones = np.ones(shape, dtype="float32")
    grid = np.linspace(0, 1, _XS.size, dtype="float32")[None, None, :] * ones

    def _var(values):
        return (("time", "y", "x"), values.astype("float32"))

    ds = xr.Dataset(
        {
            "precip": _var(
                np.clip(4.0 + 3.0 * season[:, None, None] + 2.0 * grid, 0, None)
            ),
            "temp": _var(24.0 + 3.0 * season[:, None, None] + grid),
            "temp_min": _var(20.0 + 3.0 * season[:, None, None] + grid),
            "temp_max": _var(29.0 + 3.0 * season[:, None, None] + grid),
            "press_msl": _var(101300.0 + 200.0 * season[:, None, None] * ones),
            "kin": _var(190.0 + 40.0 * season[:, None, None] * ones + 10.0 * grid),
            "kout": _var(410.0 + 20.0 * season[:, None, None] * ones),
        },
        coords={"time": time, "y": _YS, "x": _XS},
    )
    return _set_crs(ds)


# --------------------------------------------------------------------------- #
# Unit tests — source-grid PET
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def source_climate():
    from blueearth_cst.climate_analysis.plot_climate_source import source_grid_climate

    ds_raw = _extraction()
    return ds_raw, source_grid_climate(ds_raw, _orography()).compute()


def test_source_grid_pet_is_produced_on_the_extraction_grid(source_climate):
    """PET exists, is finite and positive, and nothing was regridded."""
    ds_raw, ds_src = source_climate

    assert set(ds_src.data_vars) == {"precip", "temp", "pet"}
    assert np.array_equal(ds_src["pet"]["x"].values, ds_raw["x"].values)
    assert np.array_equal(ds_src["pet"]["y"].values, ds_raw["y"].values)

    pet = ds_src["pet"].values
    assert np.isfinite(pet).all()
    assert (pet > 0).all(), "de Bruin PET should be positive over this domain"
    # Sanity band for a humid tropical grid cell, in mm day-1.
    assert 0.5 < float(np.nanmean(pet)) < 15.0


def test_source_grid_temp_carries_no_lapse_shift(source_climate):
    """``dem_model == dem_forcing`` ⇒ the lapse correction is a no-op.

    This is the property that makes the parity machinery reusable here: the
    extraction's ``temp`` is already stated at the source orography's
    elevations, so re-correcting it against the same DEM must not move it.
    """
    ds_raw, ds_src = source_climate
    got = ds_src["temp"].values
    want = ds_raw["temp"].transpose("time", "y", "x").values
    assert np.allclose(got, want, atol=1e-4)


def test_source_grid_pet_uses_the_source_orography():
    """Raising the DEM changes PET — the orography is genuinely consumed.

    Without this, ``load_source_orography`` could return anything (or the
    branch could silently fall through) and the figures would still look fine.
    """
    from blueearth_cst.climate_analysis.plot_climate_source import source_grid_climate

    ds_raw = _extraction()
    low = source_grid_climate(ds_raw, _orography()).compute()["pet"]
    high = source_grid_climate(ds_raw, _orography(offset=2000.0)).compute()["pet"]
    assert not np.allclose(low.values, high.values)


def test_missing_parity_variable_raises(tmp_path):
    """A short extraction fails loud, not with an opaque MissingOutputException."""
    from blueearth_cst.climate_analysis.plot_climate_source import plot_climate_source

    ds = _extraction().drop_vars("kout")
    nc = tmp_path / "extract_historical.nc"
    ds.to_netcdf(nc)
    with pytest.raises(ValueError, match="kout"):
        plot_climate_source(nc, tmp_path / "plots", data_sources=None)


# --------------------------------------------------------------------------- #
# The P4 assertion — a real Snakemake DAG dry-run
# --------------------------------------------------------------------------- #


@pytest.fixture()
def modelfree_project(tmp_path):
    """A model-free project config plus a deliberately absent build template.

    Returns ``(config_path, project_dir, store_dir, absent_template)``.
    """
    from blueearth_cst.shared.snake_utils import climate_store_rule

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    # The store producer declares the catalog as its sole input. The dry-run
    # never opens it, so a minimal valid YAML mapping is sufficient.
    catalog = tmp_path / "catalog.yml"
    catalog.write_text("{}\n", encoding="utf-8")

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    cfg["project"]["project_dir"] = project_dir.as_posix()
    cfg["project"]["data_sources"] = catalog.as_posix()
    cfg["shared"]["historical_window"] = {
        "starttime": f"{_START}T00:00:00",
        "endtime": f"{_END}T00:00:00",
    }
    # P4, made provable rather than asserted: the build template and the
    # waterbodies template are pointed at paths that DO NOT EXIST. Any edge from
    # the figure targets to the model-build side would now be a
    # MissingInputException, not a silent dependency.
    absent_template = tmp_path / "absent" / "wflow_build_model.yml"
    cfg["workflows"]["build_model"]["model_build_config"] = absent_template.as_posix()
    cfg["workflows"]["build_model"]["waterbodies_config"] = (
        tmp_path / "absent" / "wflow_update_waterbodies.yml"
    ).as_posix()
    cfg_path = tmp_path / "snake_config_modelfree.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    spec = climate_store_rule(
        project_dir=project_dir.as_posix(),
        model_region=cfg["shared"]["basin"]["region"],
        clim_source=cfg["shared"]["clim_historical"],
        historical_window=cfg["shared"]["historical_window"],
        data_sources=catalog.as_posix(),
    )
    store = Path(spec.store_dir)
    return cfg_path, project_dir, store, absent_template


def _snakemake(args, cfg_path):
    """Invoke snakemake on wf1 with the repo workflow profile disabled."""
    cmd = (
        f"snakemake {args} --workflow-profile none -c 1 "
        f'-s build_model.smk --configfile "{cfg_path}"'
    )
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=str(SNAKEDIR)
    )


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake not on PATH")
@pytest.mark.workflow_contract
def test_source_figures_build_without_a_model(modelfree_project):
    """P4: source figures schedule without the model-build subgraph."""
    cfg_path, project_dir, store, absent_template = modelfree_project
    from blueearth_cst.climate_analysis.climate_figures import figure_names

    targets = [store / "plots" / name for name in figure_names("source")]
    quoted = " ".join(f'"{t.as_posix()}"' for t in targets)
    result = _snakemake(f"--dry-run {quoted}", cfg_path)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-4000:]

    assert "extract_historical_climate" in combined
    assert "plot_climate_source" in combined
    assert "prepare_spatial_maps" not in combined
    assert "build_wflow_model" not in combined
    assert absent_template.as_posix() not in combined.replace("\\", "/")
    assert not (project_dir / "hydrology_model").exists()
