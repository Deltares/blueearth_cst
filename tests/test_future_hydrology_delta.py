import os
from os.path import join, dirname
import numpy as np
from tomli import load as load_toml
import subprocess
import glob

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, config_fao_fn

from ..snakemake.get_config import get_config

from ..src import copy_config_files
from ..src import downscale_delta_change
from ..src import setup_config_future
from ..src import plot_results_delta


def test_copy_config(tmpdir):
    """Test if config files are copied to tmpdir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fao_fn,
        output_dir=join(tmpdir, "config"),
        config_out_name="snake_config_future_hydrology_delta_change.yml",
    )

    # Check if config files are copied to tmpdir/config folder
    assert os.path.exists(
        f"{tmpdir}/config/snake_config_future_hydrology_delta_change.yml"
    )


def test_downscale_monthly_delta_change_grids(tmpdir):
    """Test downscaling and unit conversion of the cmip6 change grids."""
    delta_change_fn = join(
        SAMPLE_PROJECTDIR,
        "climate_projections",
        "cmip6",
        "monthly_change_grid",
        "NOAA-GFDL_GFDL-ESM4_ssp245_near.nc",
    )
    delta_change_downscaled_fn = join(
        tmpdir,
        "climate_projections",
        "cmip6",
        "monthly_change_grid",
        "NOAA-GFDL_GFDL-ESM4_ssp245_near_downscaled.nc",
    )
    wflow_grid = join(SAMPLE_PROJECTDIR, "hydrology_model", "staticmaps.nc")

    # Call the downscale function
    ds_dwn = downscale_delta_change.downscale_delta_change(
        delta_change_grid_fn=delta_change_fn,
        dst_grid_fn=wflow_grid,
        path_output=dirname(delta_change_downscaled_fn),
    )

    # Check if the output file is created
    assert os.path.exists(delta_change_downscaled_fn)

    # Check the content
    assert len(ds_dwn["time"]) == 12
    assert len(ds_dwn["latitude"]) == 18
    assert len(ds_dwn["longitude"]) == 34
    assert np.isclose(ds_dwn["precip"].mean().values, 1.15, atol=0.01)
    assert np.isclose(ds_dwn["temp"].mean().values, 1.17, atol=0.01)
    assert np.isclose(ds_dwn["pet"].mean().values, 1.05, atol=0.01)


def test_setup_toml_delta_change_hydrology(tmpdir, config_fao):
    """Test updating the wflow toml for delta change run."""
    wflow_config_template = join(
        SAMPLE_PROJECTDIR,
        "hydrology_model",
        get_config(
            config_fao, "config_model_historical", default="wflow_sbm_era5.toml"
        ),
    )
    delta_change_fn = join(
        SAMPLE_PROJECTDIR,
        "climate_projections",
        "cmip6",
        "monthly_change_grid",
        "NOAA-GFDL_GFDL-ESM4_ssp245_near_downscaled.nc",
    )

    # Call the setup config function
    setup_config_future.update_config_run_future(
        config_model_historical_fn=wflow_config_template,
        delta_change_fn=delta_change_fn,
        model="NOAA-GFDL_GFDL-ESM4",
        scenario="ssp245",
        horizon="near",
        config_root=join(tmpdir, "hydrology_model"),
        ref_time=("2000", "2005"),
    )

    # Check the output file
    fn_yml = (
        os.path.basename(wflow_config_template).split(".")[0]
        + "_delta_NOAA-GFDL_GFDL-ESM4_ssp245_near.toml"
    )
    fn_yml = join(tmpdir, "hydrology_model", fn_yml)
    assert os.path.exists(fn_yml)
    # Check the content of the output file
    with open(fn_yml, "rb") as f:
        content = load_toml(f)

    assert content["model"]["reinit"] == False
    assert content["csv"]["path"] == "output_delta_NOAA-GFDL_GFDL-ESM4_ssp245_near.csv"
    assert (
        content["state"]["path_output"]
        == "outstate/outstates_NOAA-GFDL_GFDL-ESM4_ssp245_near.nc"
    )
    assert "path_forcing_scale" in content["input"]
    assert content["endtime"] == "2005-12-31T00:00:00"

    # Run wflow with this config file
    wflow_julia_src = join(MAINDIR, "src", "wflow", "run_wflow_change_factors.jl")
    cmd = f""" julia "{wflow_julia_src}" "{fn_yml}" """
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # Check the output of the subprocess command
    assert result.returncode == 0
    # Check if the output files are created
    assert os.path.exists(
        join(
            tmpdir,
            "hydrology_model",
            "output_delta_NOAA-GFDL_GFDL-ESM4_ssp245_near.csv",
        )
    )


def test_plot_results_delta(tmpdir, config_fao):
    """Test if results from the delta runs are correctly made"""
    # Call the results from the delta runs function

    wflow_delta_runs_config = glob.glob(
        join(
            SAMPLE_PROJECTDIR,
            "hydrology_model",
            "run_delta_change",
            "wflow_sbm_*_delta_*.toml",
        )
    )

    wflow_historical_config = join(
        SAMPLE_PROJECTDIR,
        "hydrology_model",
        get_config(
            config_fao, "config_model_historical", default="wflow_sbm_era5.toml"
        ),
    )

    plot_dir = f"{tmpdir}/plots"

    gauges_locs = get_config(config_fao, "output_locations", default=None)
    gauges_locs = join(MAINDIR, gauges_locs)

    future_horizons = get_config(config_fao, "future_horizons", default=None)
    near_horizon = future_horizons["near"].replace(", ", "-")
    far_horizon = future_horizons["far"].replace(", ", "-")

    plot_results_delta.analyse_wflow_delta(
        wflow_hist_run_config=wflow_historical_config,
        wflow_delta_runs_config=wflow_delta_runs_config,
        gauges_locs=gauges_locs,
        plot_dir=join(plot_dir, "model_delta_runs"),
        near_legend=f"Horizon {near_horizon}",
        far_legend=f"Horizon {far_horizon}",
    )

    # Check if plots exist
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/cumsum_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/mean_monthly_Q_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/nm7q_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/max_annual_q_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/mean_annual_q_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/qhydro_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/rel_mean_annual_q_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/rel_max_annual_q_1.png")
    assert os.path.exists(f"{plot_dir}/model_delta_runs/flow/1/rel_nm7q_annual_q_1.png")
    assert os.path.exists(
        f"{plot_dir}/model_delta_runs/flow/6349400/boxplot_q_abs_6349400.png"
    )
    assert os.path.exists(
        f"{plot_dir}/model_delta_runs/flow/6349400/boxplot_q_rel_6349400.png"
    )
    assert os.path.exists(f"{plot_dir}/model_delta_runs/other/boxplot_snow_rel.png")
    assert os.path.exists(
        f"{plot_dir}/model_delta_runs/other/mean_monthly_overland flow_basavg.png"
    )
    assert os.path.exists(
        f"{plot_dir}/model_delta_runs/other/mean_monthly_groundwater recharge_basavg.png"
    )
