import os
from os.path import join, dirname, basename
import numpy as np
import yaml

from .conftest import SAMPLE_PROJECTDIR, config_fao_fn

from ..snakemake.get_config import get_config

from ..src import copy_config_files
from ..src import downscale_delta_change
from ..src import setup_config_future


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
        get_config(config_fao, "config_model_historical", "wflow_sbm_era5.toml"),
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
        config_root=None,  # join(tmpdir, "hydrology_model"),
    )

    # # Check the output file
    # fn_yml = (
    #     os.path.basename(wflow_config_template).split(".")[0]
    #     + "_delta_NOAA-GFDL_GFDL-ESM4_ssp245_near.toml"
    # )
    # fn_yml = join(tmpdir, "hydrology_model", fn_yml)
    # assert os.path.exists(fn_yml)
    # # Check the content of the output file
    # with open(fn_yml, "rb") as f:
    #     content = yaml.safe_load(f)
