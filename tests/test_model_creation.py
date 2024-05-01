"""Test functions from the model creation workflow."""

import os
from os.path import join, basename
import subprocess
import yaml
import pytest

import xarray as xr
import pandas as pd
import numpy as np
from hydromt_wflow import WflowModel

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, config_fn
from .conftest import get_config

from ..src import copy_config_files
from ..src import setup_reservoirs_lakes_glaciers
from ..src import setup_gauges_and_outputs
from ..src import setup_time_horizon
from ..src import plot_map
from ..src import plot_map_forcing
from ..src import plot_results


def test_copy_config(tmpdir, data_sources, model_build_config):
    """Test if config files are copied to tmpdir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fn,
        output_dir=join(tmpdir, "config"),
        config_out_name="snake_config_model_creation.yml",
        other_config_files=[data_sources, model_build_config],
    )

    # Check if config files are copied to tmpdir/config folder
    assert os.path.exists(f"{tmpdir}/config/snake_config_model_creation.yml")
    assert os.path.exists(f"{tmpdir}/config/wflow_build_model.yml")
    assert os.path.exists(f"{tmpdir}/config/tests_data_catalog.yml")


def test_create_model_full(
    tmpdir, config, model_build_config, waterbodies_config, data_sources
):
    """Test building the wflow model including all steps."""

    ### 1. Create model base ###
    basin_dir = f"{tmpdir}/hydrology_model"
    model_region = get_config(config, "model_region", optional=False)
    model_resolution = get_config(config, "model_resolution", 0.00833333)
    hydromt_ini = model_build_config

    # Run hydromt build command similarly as in snakemake workflow
    #np.atleast_1d to convert string to list (if there is only one element in a list)
    data_sources = np.atleast_1d(data_sources).tolist()
    data_catalogs = [f"-d {cat} " for cat in data_sources]
    #-d {data_sources} should be called similarly as in the snakefile -  ".join(data_catalogs) combine list elements to string
    cmd = f"""hydromt build wflow {basin_dir} --region "{model_region}" --opt setup_basemaps.res={model_resolution} -i {hydromt_ini} {" ".join(data_catalogs)} --fo -vv"""
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # Check the output of the subprocess command
    assert result.returncode == 0

    # Check the output files
    assert os.path.exists(f"{tmpdir}/hydrology_model/staticmaps.nc")
    assert os.path.exists(f"{tmpdir}/hydrology_model/wflow_sbm.toml")

    # Compare to the wflow_dem sample output file as not all variables are built yet
    # gauges, waterbodies, glaciers etc are not yet added
    ds_project = xr.open_dataset(f"{tmpdir}/hydrology_model/staticmaps.nc")
    ds_sample = xr.open_dataset(f"{SAMPLE_PROJECTDIR}/hydrology_model/staticmaps.nc")

    xr.testing.assert_allclose(ds_project["wflow_dem"], ds_sample["wflow_dem"])

    # Close and release the datasets
    ds_project.close()
    ds_sample.close()

    ### 2. Add reservoirs, lakes and glaciers to the built model ###
    setup_reservoirs_lakes_glaciers.update_wflow_waterbodies_glaciers(
        wflow_root=basin_dir,
        data_catalog=data_sources,
        config_fn=waterbodies_config,
    )

    # Check the output files
    # In this case we should get glaciers and reservoirs added but no lakes
    assert os.path.exists(
        f"{tmpdir}/hydrology_model/staticgeoms/reservoirs_lakes_glaciers.txt"
    )
    assert os.path.exists(f"{tmpdir}/hydrology_model/staticgeoms/glaciers.geojson")
    assert os.path.exists(f"{tmpdir}/hydrology_model/staticgeoms/reservoirs.geojson")
    assert not os.path.exists(f"{tmpdir}/hydrology_model/staticgeoms/lakes.geojson")

    ### 3. Add gauges and select outputs ###
    gauges_fn = get_config(config, "output_locations", None)
    gauges_fn = join(MAINDIR, gauges_fn)
    outputs = get_config(config, "wflow_outvars")

    setup_gauges_and_outputs.update_wflow_gauges_outputs(
        wflow_root=basin_dir,
        data_catalog=data_sources,
        gauges_fn=gauges_fn,
        outputs=outputs,
    )

    # Check the output files
    assert os.path.exists(f"{tmpdir}/hydrology_model/staticgeoms/gauges.geojson")
    assert os.path.exists(
        f"{tmpdir}/hydrology_model/staticgeoms/gauges_discharge-locations-grdc.geojson"
    )

    # Check if our output vars are in the wflow_sbm.toml file
    wflow = WflowModel(root=basin_dir, mode="r")

    vars = [v["header"] for v in wflow.config["csv"]["column"]]
    for output in outputs:
        assert f"{output}_basavg" in vars


def test_setup_runtime(tmpdir, config):
    """Test preparing the forcing config file."""
    starttime = get_config(config, "starttime", optional=False)
    endtime = get_config(config, "endtime", optional=False)
    fn_yml = f"{tmpdir}/config/wflow_build_forcing_historical.yml"
    precip_source = get_config(config, "clim_historical", optional=False)

    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime=starttime,
        endtime=endtime,
        fn_yml=fn_yml,
        precip_source=precip_source,
    )

    # Check the output file
    assert os.path.exists(fn_yml)
    # Check the content of the output file
    with open(fn_yml, "rb") as f:
        content = yaml.safe_load(f)
    assert "setup_config" in content
    assert "setup_precip_forcing" in content
    assert "setup_temp_pet_forcing" in content

    assert content["setup_config"]["starttime"] == starttime
    assert content["setup_precip_forcing"]["precip_fn"] == precip_source

def test_setup_runtime_suffix(tmpdir, config):
    """Test preparing the forcing config file when there are multiple data sources and suffix is required."""
    starttime = get_config(config, "starttime", optional=False)
    endtime = get_config(config, "endtime", optional=False)
    precip_source = get_config(config, "clim_historical", optional=False)
    fn_yml = f"{tmpdir}/config/wflow_build_forcing_historical_{precip_source}.yml"

    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime=starttime,
        endtime=endtime,
        fn_yml=fn_yml,
        precip_source=precip_source,
        suffix=True,
    )

    # Check the output file
    assert os.path.exists(fn_yml)
    # Check the content of the output file
    with open(fn_yml, "rb") as f:
        content = yaml.safe_load(f)
    assert "setup_config" in content
    assert "write_config" in content

    assert content["setup_config"]["dir_output"] == f"run_default_{precip_source}"
    assert content["setup_config"]["input.path_forcing"] == f"../climate_historical/wflow_data/inmaps_historical_{precip_source}.nc"
    assert content["write_config"]["config_name"] == f"wflow_sbm_{precip_source}.toml"

@pytest.mark.timeout(120)  # max 2 min
def test_add_forcing(tmpdir, data_sources, config):
    """Test adding forcing to the updated model."""
    # Settings
    forcing_ini = f"{SAMPLE_PROJECTDIR}/config/wflow_build_forcing_historical.yml"
    # Compared to build model, use the SAMPLE model for updating
    sample_model = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    output_model = f"{tmpdir}/hydrology_model"

    cmd = f"""hydromt update wflow {sample_model} -o {output_model} -i {forcing_ini} -d {data_sources} --fo -vv"""
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # Check the output of the subprocess command
    assert result.returncode == 0

    # Check the output file
    forcing_fn = f"{tmpdir}/climate_historical/wflow_data/inmaps_historical.nc"
    assert os.path.exists(forcing_fn)

    ds = xr.open_dataset(forcing_fn)
    assert "precip" in ds
    assert "temp" in ds
    assert "pet" in ds

    starttime = get_config(config, "starttime", optional=False)
    endtime = get_config(config, "endtime", optional=False)
    assert pd.to_datetime(ds.time[0].item()) == pd.to_datetime(starttime)
    assert pd.to_datetime(ds.time[-1].item()) == pd.to_datetime(endtime)


def test_plot_map(tmpdir, config):
    """Test plotting the model map."""
    wflow_root = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    plot_dir = f"{tmpdir}/plots/wflow_model_performance"
    gauges_fn = get_config(config, "output_locations")
    gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'

    # Try without gauges
    plot_map.plot_wflow_map(
        wflow_root=wflow_root,
        plot_dir=plot_dir,
        gauges_name=None,
    )

    # Try with gauges
    plot_map.plot_wflow_map(
        wflow_root=wflow_root,
        plot_dir=plot_dir,
        gauges_name=gauges_name,
    )

    # Check output
    assert os.path.exists(f"{plot_dir}/basin_area.png")


def test_plot_forcing(tmpdir, config):
    """Test plotting the forcing maps."""
    wflow_root = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    plot_dir = f"{tmpdir}/plots/wflow_model_performance"
    gauges_fn = get_config(config, "output_locations")
    gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'

    # Test first without gauges
    plot_map_forcing.plot_forcing(
        wflow_root=wflow_root,
        plot_dir=plot_dir,
        gauges_name=None,
    )

    # Test with gauges
    plot_map_forcing.plot_forcing(
        wflow_root=wflow_root,
        plot_dir=plot_dir,
        gauges_name=gauges_name,
    )

    # Check output
    assert os.path.exists(f"{plot_dir}/precip.png")
    assert os.path.exists(f"{plot_dir}/temp.png")
    assert os.path.exists(f"{plot_dir}/pet.png")


def test_plot_results(tmpdir, config):
    """Test plotting the model results."""
    wflow_root = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    plot_dir = f"{tmpdir}/plots"
    gauges_locs = get_config(config, "output_locations")
    gauges_locs = join(MAINDIR, gauges_locs)
    observations_fn = get_config(config, "observations_timeseries")
    observations_fn = join(MAINDIR, observations_fn)
    climate_source = get_config(config, "clim_historical")
    climate_source = np.atleast_1d(climate_source).tolist()

    # 1. Plot all results
    plot_results.analyse_wflow_historical(
        wflow_root=wflow_root,
        plot_dir=join(plot_dir, "long_run"),
        observations_fn=observations_fn,
        gauges_locs=gauges_locs,
        wflow_config_fn="wflow_sbm.toml",
        climate_sources=climate_source, #option to add name of climate source in plots
    )

    # Check monthly and yearly clim plots are there
    assert os.path.exists(f"{plot_dir}/long_run/clim_wflow_1_month.png")
    assert os.path.exists(f"{plot_dir}/long_run/clim_wflow_1_year.png")
    # Check that the other basin average plots are there
    assert os.path.exists(f"{plot_dir}/long_run/snow_basavg.png")
    # Check that the gauges plots are there
    assert os.path.exists(f"{plot_dir}/long_run/hydro_wflow_1.png")
    assert os.path.exists(f"{plot_dir}/long_run/hydro_PONTE DELLA IASTA.png")
    # Check the signature plot for the gauges_locs was also created
    assert os.path.exists(f"{plot_dir}/long_run/signatures_PONTE DELLA IASTA.png")
    # Check the performance metrics table is not empty
    perf = pd.read_csv(f"{plot_dir}/long_run/performance_metrics.csv")
    assert not perf.empty
    assert "PONTE DELLA IASTA" in perf.columns
    metrics = ["KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE"]
    assert np.all([c in np.unique(perf["metrics"]) for c in metrics])
    assert np.array_equal(np.unique(perf["time_type"]), ["daily", "monthly"])
    # Check if trend plots are present
    assert os.path.exists(f"{plot_dir}/long_run/timeseries_anomalies_Q_era5.png")
    assert os.path.exists(f"{plot_dir}/long_run/timeseries_anomalies_Q_obs.png")
    # Check budyko plot 
    assert os.path.exists(f"{plot_dir}/long_run/budyko_qobs.png")

    # 2. Plot medium length and no observations timeseries
    plot_results.analyse_wflow_historical(
        wflow_root=wflow_root,
        plot_dir=join(plot_dir, "medium_run"),
        observations_fn=None,
        gauges_locs=gauges_locs,
        wflow_config_fn="wflow_sbm_medium.toml",
    )
    # Check monthly and yearly clim plots are there
    assert os.path.exists(f"{plot_dir}/medium_run/clim_wflow_1_month.png")
    assert os.path.exists(f"{plot_dir}/medium_run/clim_wflow_1_year.png")
    # Check that the other basin average plots are there
    assert os.path.exists(f"{plot_dir}/medium_run/snow_basavg.png")
    # Check that the gauges plots are there
    assert os.path.exists(f"{plot_dir}/medium_run/hydro_wflow_1.png")
    assert os.path.exists(f"{plot_dir}/medium_run/hydro_PONTE DELLA IASTA.png")
    # Check the signature plot for the gauges_locs was not created
    assert not os.path.exists(f"{plot_dir}/medium_run/signatures_PONTE DELLA IASTA.png")
    # Check the performance metrics table is empty
    perf = pd.read_csv(f"{plot_dir}/medium_run/performance_metrics.csv")
    assert perf.empty

    # 3. Plot medium length and no observations locs and timeseries
    plot_results.analyse_wflow_historical(
        wflow_root=wflow_root,
        plot_dir=join(plot_dir, "medium_run_no_obs"),
        observations_fn=None,
        gauges_locs=None,
        wflow_config_fn="wflow_sbm_medium.toml",
    )
    # Check plot for obs locs is not there
    assert not os.path.exists(
        f"{plot_dir}/medium_run_no_obs/hydro_PONTE DELLA IASTA.png"
    )

    # 4. Plot short run with observations
    plot_results.analyse_wflow_historical(
        wflow_root=wflow_root,
        plot_dir=join(plot_dir, "short_run"),
        observations_fn=observations_fn,
        gauges_locs=gauges_locs,
        wflow_config_fn="wflow_sbm_short.toml",
    )
    # No clim plot should be there
    assert not os.path.exists(f"{plot_dir}/short_run/clim_wflow_1_month.png")
    assert not os.path.exists(f"{plot_dir}/short_run/clim_wflow_1_year.png")
    # Check that the other basin average plots are there
    assert os.path.exists(f"{plot_dir}/short_run/snow_basavg.png")
    # Check that the gauges plots are there
    assert os.path.exists(f"{plot_dir}/short_run/hydro_wflow_1.png")
    assert os.path.exists(f"{plot_dir}/short_run/hydro_PONTE DELLA IASTA.png")
    # Check the signature plot was not created
    assert not os.path.exists(f"{plot_dir}/short_run/signatures_PONTE DELLA IASTA.png")
    # Check the performance metrics table is empty
    perf = pd.read_csv(f"{plot_dir}/short_run/performance_metrics.csv")
    assert perf.empty
    #check budyko plot
    assert os.path.exists(f"{plot_dir}/short_run/budyko_qobs.png")
