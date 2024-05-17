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

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, TESTDIR, config_fao_fn, data_libs_fao
from .conftest import get_config

from ..src import copy_config_files
from ..src import setup_time_horizon
from ..src import plot_map
from ..src import plot_map_forcing
from ..src import plot_results


def test_copy_config(tmpdir, data_libs_fao, model_build_config):
    """Test if config files are copied to tmpdir/config folder"""
    # Call the copy file function
    data_libs_fao = np.atleast_1d(data_libs_fao).tolist()
    copy_config_files.copy_config_files(
        config=config_fao_fn,
        output_dir=join(tmpdir, "config"),
        config_out_name="snake_config_model_creation.yml",
        other_config_files=data_libs_fao + [model_build_config],
    )

    # Check if config files are copied to tmpdir/config folder
    assert os.path.exists(f"{tmpdir}/config/snake_config_model_creation.yml")
    assert os.path.exists(f"{tmpdir}/config/wflow_build_model.yml")
    assert os.path.exists(f"{tmpdir}/config/tests_data_catalog.yml")


# model creation and model running not tested -- same as in CST test workflow


def test_setup_runtime_suffix(tmpdir, config_fao):
    """Test preparing the forcing config file when there are multiple data sources and suffix is required."""
    starttime = get_config(config_fao, "starttime", optional=False)
    endtime = get_config(config_fao, "endtime", optional=False)
    precip_sources = get_config(config_fao, "clim_historical", optional=False)
    precip_sources = np.atleast_1d(precip_sources).tolist()

    for precip_source in precip_sources:
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
        assert (
            content["setup_config"]["input.path_forcing"]
            == f"../climate_historical/wflow_data/inmaps_historical_{precip_source}.nc"
        )
        assert (
            content["write_config"]["config_name"] == f"wflow_sbm_{precip_source}.toml"
        )


@pytest.mark.timeout(120)  # max 2 min
def test_add_forcing(tmpdir, data_libs_fao, config_fao):
    """Test adding forcing to the updated model."""
    # Compared to build model, use the SAMPLE model for updating
    sample_model = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    output_model = f"{tmpdir}/hydrology_model"
    precip_sources = get_config(config_fao, "clim_historical", optional=False)
    precip_sources = np.atleast_1d(precip_sources).tolist()
    # data catalogs
    data_catalogs = np.atleast_1d(data_libs_fao).tolist()
    data_catalogs = [f"-d {cat} " for cat in data_catalogs]

    for precip_source in precip_sources:
        # Settings
        forcing_ini = f"{SAMPLE_PROJECTDIR}/config/wflow_build_forcing_historical_{precip_source}.yml"

        cmd = f"""hydromt update wflow {sample_model} -o {output_model} -i {forcing_ini} {" ".join(data_catalogs)} --fo -vv"""
        result = subprocess.run(cmd, shell=True, capture_output=True)
        # Check the output of the subprocess command
        assert result.returncode == 0

        # Check the output file
        forcing_fn = f"{tmpdir}/climate_historical/wflow_data/inmaps_historical_{precip_source}.nc"
        assert os.path.exists(forcing_fn)

        ds = xr.open_dataset(forcing_fn)
        assert "precip" in ds
        assert "temp" in ds
        assert "pet" in ds

def test_plot_forcing(tmpdir, config_fao):
    """Test plotting the forcing maps."""
    wflow_root = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    gauges_fn = get_config(config_fao, "output_locations")
    gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'
    precip_sources = get_config(config_fao, "clim_historical", optional=False)
    precip_sources = np.atleast_1d(precip_sources).tolist()

    for precip_source in precip_sources:
        plot_dir = f"{tmpdir}/plots/wflow_model_performance/{precip_source}"

        # Test first without gauges
        plot_map_forcing.plot_forcing(
            wflow_root=wflow_root,
            plot_dir=plot_dir,
            gauges_name=None,
            config_fn=f"wflow_sbm_{precip_source}.toml",
        )

        # Test with gauges
        plot_map_forcing.plot_forcing(
            wflow_root=wflow_root,
            plot_dir=plot_dir,
            gauges_name=gauges_name,
            config_fn=f"wflow_sbm_{precip_source}.toml",
        )

        # Check output
        assert os.path.exists(f"{plot_dir}/precip.png")
        assert os.path.exists(f"{plot_dir}/temp.png")
        assert os.path.exists(f"{plot_dir}/pet.png")


# todo!
# @pytest.mark.parametrize("model", list(_supported_models.keys())) check -- https://github.com/Deltares/hydromt_wflow/blob/main/tests/test_model_class.py def test_model_build
# _supported_models = {"wflow": WflowModel, "wflow_sediment": WflowSedimentModel} to have same tests for two config files -- make a list !
def test_plot_results(tmpdir, config_fao):
    """Test plotting the model results."""
    wflow_root = f"{SAMPLE_PROJECTDIR}/hydrology_model"
    plot_dir = f"{tmpdir}/plots"
    gauges_locs = get_config(config_fao, "output_locations")
    gauges_locs = join(MAINDIR, gauges_locs)
    observations_fn = get_config(config_fao, "observations_timeseries")
    observations_fn = join(MAINDIR, observations_fn)
    precip_sources = get_config(config_fao, "clim_historical", optional=False)
    precip_sources = np.atleast_1d(precip_sources).tolist()
    clim_historical_colors = get_config(
        config_fao, "clim_historical_colors", optional=False
    )

    # 1. Plot all results
    plot_results.analyse_wflow_historical(
        wflow_root=wflow_root,
        plot_dir=join(plot_dir, "long_run"),
        observations_fn=observations_fn,
        gauges_locs=gauges_locs,
        wflow_config_fn_prefix="wflow_sbm", #prefix name toml file instead? "wflow_sbm"
        climate_sources=precip_sources,
        climate_sources_colors=clim_historical_colors,
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
    assert os.path.exists(
        f"{plot_dir}/long_run/timeseries_anomalies_Q_chirps_global.png"
    )
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
        wflow_config_fn_prefix="wflow_sbm_medium",
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
        wflow_config_fn_prefix="wflow_sbm_medium",
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
        wflow_config_fn_prefix="wflow_sbm_short",
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
    # check budyko plot
    assert os.path.exists(f"{plot_dir}/short_run/budyko_qobs.png")
