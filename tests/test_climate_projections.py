import os
from os.path import join, isfile
import glob

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, TESTDIR

from ..snakemake.get_config import get_config

from ..src import copy_config_files
from ..src import derive_region
from ..src import get_stats_climate_proj
from ..src import get_change_climate_proj
from ..src import get_change_climate_proj_summary
from ..src import plot_proj_timeseries

config_fn = join(TESTDIR, "snake_config_model_test.yml")


def test_copy_config(tmpdir, data_sources_climate):
    """Test if config files are copied to tmpdir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fn,
        output_dir=join(tmpdir, "config"),
        config_out_name="snake_config_climate_projections.yml",
        other_config_files=[data_sources_climate],
    )

    # Check if config files are copied to tmpdir/config folder
    assert os.path.exists(f"{tmpdir}/config/snake_config_climate_projections.yml")
    assert os.path.exists(f"{tmpdir}/config/cmip6_data.yml")


def test_select_region(tmpdir):
    """Test if region is created properly."""
    # Check if derived region is the same as wflow model (default setting)
    basin_fn = join(
        SAMPLE_PROJECTDIR, "hydrology_model", "staticgeoms", "region.geojson"
    )
    region = {"geom": basin_fn}
    # Try to derive the region from config file
    derive_region.derive_region(
        region=region,
        path_output=join(tmpdir, "region"),
        buffer=None,
        data_catalog=None,
        hydrography_fn=None,
    )

    assert os.path.exists(f"{tmpdir}/region/region.geojson")

    # Check if region is the same as expected
    basin = gpd.read_file(basin_fn)
    region = gpd.read_file(f"{tmpdir}/region/region.geojson")
    assert_geodataframe_equal(region, basin, check_like=True, check_less_precise=True)


def test_get_climate_historical_statistics(tmpdir, data_sources_climate):
    """Test if historical statistics are calculated properly."""
    region_fn = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    # Compute historical statistics for a specific climate model
    get_stats_climate_proj.extract_climate_projections_statistics(
        region_fn=region_fn,
        data_catalog=data_sources_climate,
        path_output=join(tmpdir, "climate_projections", "cmip6"),
        clim_source="cmip6",
        scenario="historical",
        members=["r1i1p1f1"],
        model="NOAA-GFDL/GFDL-ESM4",
        variables=["precip", "temp"],
        save_grids=True,
        time_horizon={"historical": ("2000", "2010")},
    )

    # Check if the files were produced
    fn_out = f"{tmpdir}/climate_projections/cmip6/historical_stats_time_NOAA-GFDL/GFDL-ESM4.nc"
    assert os.path.exists(fn_out)
    fn_out_grid = (
        f"{tmpdir}/climate_projections/cmip6/historical_stats_NOAA-GFDL/GFDL-ESM4.nc"
    )
    assert os.path.exists(fn_out_grid)

    # Check the content of the basin averaged file
    ds = xr.open_dataset(fn_out)
    ds_dims = [name for name in ds.dims]
    assert ds_dims == ["time", "clim_project", "model", "scenario", "member"]
    assert ds.model.values[0] == "NOAA-GFDL/GFDL-ESM4"
    assert ds.scenario.values[0] == "historical"
    # Check that the len of time (for cmip6 historical) is 780 months
    assert len(ds.time) == 780
    # Check that precip and temp are in the data_vars
    assert "precip" in ds.data_vars
    assert "temp" in ds.data_vars

    # Check the content of the grid file
    ds_grid = xr.open_dataset(fn_out_grid)
    ds_grid_dims = [name for name in ds_grid.dims]
    assert ds_grid_dims == [
        "horizon",
        "lat",
        "lon",
        "month",
        "clim_project",
        "model",
        "scenario",
        "member",
    ]
    # Check that the month dimension is 12
    # gridded output is cyclic to avoid saving too much data
    assert len(ds_grid.month) == 12


def test_get_climate_future_statistics(tmpdir, data_sources_climate, config):
    """Test if historical statistics are calculated properly."""
    region_fn = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    time_horizon = get_config(config, "future_horizons", optional=False)
    for key, value in time_horizon.items():
        time_horizon[key] = tuple(map(str, value.split(", ")))

    # Compute historical statistics for a specific climate model
    get_stats_climate_proj.extract_climate_projections_statistics(
        region_fn=region_fn,
        data_catalog=data_sources_climate,
        path_output=join(tmpdir, "climate_projections", "cmip6"),
        clim_source="cmip6",
        scenario="ssp245",
        members=["r1i1p1f1"],
        model="NOAA-GFDL/GFDL-ESM4",
        variables=["precip", "temp"],
        save_grids=True,
        time_horizon=time_horizon,
    )

    # Check if the files were produced
    fn_out = (
        f"{tmpdir}/climate_projections/cmip6/stats_time-NOAA-GFDL/GFDL-ESM4_ssp245.nc"
    )
    assert os.path.exists(fn_out)
    fn_out_grid = (
        f"{tmpdir}/climate_projections/cmip6/stats-NOAA-GFDL/GFDL-ESM4_ssp245.nc"
    )
    assert os.path.exists(fn_out_grid)

    # Check the content of the basin averaged file
    ds = xr.open_dataset(fn_out)
    # Check that the len of time (for cmip6 future) is 1032 months
    assert len(ds.time) == 1032
    assert ds.scenario.values[0] == "ssp245"

    # Check the content of the grid file
    ds_grid = xr.open_dataset(fn_out_grid)
    assert np.all(ds_grid["horizon"].values == ["far", "near"])

    # Try for a different model that does not have data for a specific ssp
    get_stats_climate_proj.extract_climate_projections_statistics(
        region_fn=region_fn,
        data_catalog=data_sources_climate,
        path_output=join(tmpdir, "climate_projections", "cmip6"),
        clim_source="cmip6",
        scenario="ssp119",
        members=["r1i1p1f1"],
        model="INM/INM-CM5-0",
        variables=["precip", "temp"],
        save_grids=True,
        time_horizon=get_config(config, "future_horizons", optional=False),
    )

    # Check if the files were produced
    fn_out = f"{tmpdir}/climate_projections/cmip6/stats_time-INM/INM-CM5-0_ssp119.nc"
    assert os.path.exists(fn_out)
    # Read to check that the dataset is empty
    ds = xr.open_dataset(fn_out)
    assert len(ds.data_vars) == 0


def test_monthly_change(tmpdir):
    """Test the calculation of monthly change."""

    # Scalar change
    get_change_climate_proj.get_expected_change_scalar(
        nc_historical=join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "historical_stats_time_NOAA-GFDL/GFDL-ESM4.nc",
        ),
        nc_future=join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "stats_time-NOAA-GFDL/GFDL-ESM4_ssp245.nc",
        ),
        path_output=join(tmpdir, "climate_projections", "cmip6"),
        time_tuple_historical=("2000", "2010"),
        time_tuple_future=("2050", "2060"),
        start_month_hyd_year="Oct",
        name_horizon="near",
        name_model="NOAA-GFDL/GFDL-ESM4",
        name_scenario="ssp245",
    )

    # Check if the file was produced
    fn_out = f"{tmpdir}/climate_projections/cmip6/annual_change_scalar_stats-NOAA-GFDL/GFDL-ESM4_ssp245_near.nc"
    assert os.path.exists(fn_out)
    # Check the content of the basin averaged file
    ds = xr.open_dataset(fn_out)
    assert "stats" in ds.dims
    assert "time" not in ds.dims
    assert ds.horizon.values[0] == "near"

    # Grid change
    get_change_climate_proj.get_expected_change_grid(
        nc_historical=join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "historical_stats_NOAA-GFDL/GFDL-ESM4.nc",
        ),
        nc_future=join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "stats-NOAA-GFDL/GFDL-ESM4_ssp245.nc",
        ),
        path_output=join(tmpdir, "climate_projections", "cmip6"),
        name_horizon="near",
        name_model="NOAA-GFDL/GFDL-ESM4",
        name_scenario="ssp245",
    )
    # Check if the file was produced
    fn_out = f"{tmpdir}/climate_projections/cmip6/monthly_change_mean_grid-NOAA-GFDL/GFDL-ESM4_ssp245_near.nc"
    assert os.path.exists(fn_out)
    # Check the content of the gridded file
    ds = xr.open_dataset(fn_out)
    assert ds.horizon.values[0] == "near"
    assert np.round(ds["precip"].mean().values, 2) == 14.01
    assert np.round(ds["temp"].mean().values, 2) == 1.11


def test_monthly_change_scalar_merge(tmpdir, config):
    """Test merging and outputs of the scalar change files."""
    clim_dir = join(tmpdir, "climate_projections", "cmip6")
    # Scalar files
    clim_folders = glob.glob(
        join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "annual_change_scalar_stats*",
        )
    )
    clim_files = []
    for folder in clim_folders:
        clim_files.extend(glob.glob(join(folder, "*.nc")))

    get_change_climate_proj_summary.summary_climate_proj(
        clim_dir=clim_dir,
        clim_files=clim_files,
        horizons=get_config(config, "future_horizons", optional=False),
    )

    # Check if the different files were added
    assert isfile(join(clim_dir, "annual_change_scalar_stats_summary.nc"))
    assert isfile(join(clim_dir, "annual_change_scalar_stats_summary.csv"))
    assert isfile(join(clim_dir, "annual_change_scalar_stats_summary_mean.csv"))
    assert isfile(join(clim_dir, "plots", "projected_climate_statistics.png"))

    # Open and check the summary mean csv file (CST file)
    df = pd.read_csv(join(clim_dir, "annual_change_scalar_stats_summary_mean.csv"))
    assert len(clim_files) == len(df) == 12
    assert "temp" in df.columns
    assert "precip" in df.columns
    assert np.round(max(df["precip"]), 2) == 11.83
    assert np.round(min(df["temp"]), 2) == 1.19


def test_plot_climate_projections(tmpdir):
    """Test plotting the climate projections scalar and gridded."""
    # Historical files
    clim_folders = glob.glob(
        join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "historical_stats_time_*",
        )
    )
    nc_historical = []
    for folder in clim_folders:
        nc_historical.extend(glob.glob(join(folder, "*.nc")))

    # Future files
    clim_folders = glob.glob(
        join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "stats_time-*",
        )
    )
    nc_future = []
    for folder in clim_folders:
        nc_future.extend(glob.glob(join(folder, "*.nc")))

    # Grid files
    clim_folders_grid = glob.glob(
        join(
            SAMPLE_PROJECTDIR,
            "climate_projections",
            "cmip6",
            "monthly_change_mean_grid*",
        )
    )
    grid_files = []
    for folder in clim_folders_grid:
        grid_files.extend(glob.glob(join(folder, "*.nc")))

    path_output = join(tmpdir, "climate_projections", "cmip6")
    plot_proj_timeseries.plot_climate_projections(
        nc_historical=nc_historical,
        nc_future=nc_future,
        path_output=path_output,
        scenarios=["ssp245", "ssp585"],
        horizons=["near", "far"],
        nc_grid_projections=grid_files,
    )

    # Check if the files were created
    assert isfile(
        join(path_output, "plots", "precipitation_anomaly_projections_abs.png")
    )
    assert isfile(join(path_output, "plots", "temperature_anomaly_projections_abs.png"))
    assert isfile(
        join(path_output, "plots", "precipitation_anomaly_projections_anom.png")
    )
    assert isfile(
        join(path_output, "plots", "temperature_anomaly_projections_anom.png")
    )
    assert isfile(
        join(path_output, "plots", "precipitation_monthly_projections_abs.png")
    )
    assert isfile(join(path_output, "plots", "temperature_monthly_projections_abs.png"))
    assert isfile(
        join(path_output, "plots", "precipitation_monthly_projections_anom.png")
    )
    assert isfile(
        join(path_output, "plots", "temperature_monthly_projections_anom.png")
    )

    # Check for a couple of grid plots
    assert isfile(
        join(
            path_output,
            "plots",
            "gridded_monthly_precipitation_change_ssp245_far-future-horizon.png",
        )
    )
    assert isfile(
        join(
            path_output,
            "plots",
            "gridded_precipitation_change_ssp245_far-future-horizon.png",
        )
    )

    # Merged files
    assert isfile(join(path_output, "gcm_timeseries.nc"))
    assert isfile(join(path_output, "gcm_grid_factors_025.nc"))
