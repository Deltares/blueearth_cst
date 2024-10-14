import xarray as xr
import numpy as np
import os
from os.path import join, basename, dirname
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import hydromt
from hydromt_wflow import WflowModel
import seaborn as sns

from typing import Union, List

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from wflow.wflow_utils import get_wflow_results
    from calibration.plot_utils import plot_hydro_all_timeseries, plot_hydro_all_per_year
    from calibration.plot_utils import plot_hydro_all_month, plot_snow_glacier
else:
    from .wflow.wflow_utils import get_wflow_results
    from .calibration.plot_utils import plot_hydro_all_timeseries, plot_hydro_all_per_year
    from .calibration.plot_utils import plot_hydro_all_month, plot_snow_glacier

def plot_results_calib_runs(
    wflow_tomls: List[Union[str, Path]],
    plot_folder: Union[str, Path],
    observations_locations: Union[str, Path],
    observations_timeseries: Union[str, Path],
    uncalibrated_run: int = 0,
    calibration_runs_selection: List[int] = [],

):
    """Compare results of the calibration runs."""
        
    # 1. Read wflow results
    qsim_uncal, ds_clim, ds_basin_uncal = get_wflow_results(
        wflow_root=dirname(wflow_tomls[uncalibrated_run]),
        config_fn=basename(wflow_tomls[uncalibrated_run]),
        gauges_locs=observations_locations,
        remove_warmup=True,
    )
    if len(calibration_runs_selection) == 0:
        calibration_runs_selection = range(len(wflow_tomls))
    # Remove uncalibrated run
    calibration_runs_selection = [run for run in calibration_runs_selection if run != uncalibrated_run]
    qsim_cal = []
    ds_basin_cal = []
    name_cal = []
    for calib_run in calibration_runs_selection:
        name_cal.append(basename(wflow_tomls[calib_run]).rstrip(".toml").lstrip("wflow_sbm_"))
        qsim_cal_run, _, ds_basin_cal_run = get_wflow_results(
            wflow_root=dirname(wflow_tomls[calib_run]),
            config_fn=basename(wflow_tomls[calib_run]),
            gauges_locs=observations_locations,
            remove_warmup=True,
        )
        qsim_cal.append(qsim_cal_run)
        ds_basin_cal.append(ds_basin_cal_run)

    # 2. Read observations
    # Get wflow basins to clip observations
    wflow = WflowModel(root=dirname(wflow_tomls[uncalibrated_run]), config_fn=basename(wflow_tomls[uncalibrated_run]), mode="r")

    # Read
    gdf_obs = hydromt.io.open_vector(
        observations_locations, crs=4326, sep=",", geom=wflow.basins
    )
    da_ts_obs = hydromt.io.open_timeseries_from_table(
        observations_timeseries, name="Q", index_dim="wflow_id", sep=";"
    )
    ds_obs = hydromt.vector.GeoDataset.from_gdf(
        gdf_obs, da_ts_obs, merge_index="inner"
    )
    # Rename wflow_id to index
    ds_obs = ds_obs.rename({"wflow_id": "index"})
    qobs = ds_obs["Q"].load()
    # Sel qobs to the time in qsim
    qobs = qobs.sel(time=slice(qsim_uncal.time[0], qsim_uncal.time[-1]))

    # 3. Plot results
    # a) Monthly timeseries
    plot_hydro_all_timeseries(
        qsim_uncalibrated=qsim_uncal,
        qobs=qobs,
        qsim_cals=qsim_cal,
        qsim_cals_names=name_cal,
        Folder_out=plot_folder,
    )

    # b) Annual hydrographs
    plot_hydro_all_per_year(
        qsim_uncalibrated=qsim_uncal,
        qobs=qobs,
        qsim_cals=qsim_cal,
        qsim_cals_names=name_cal,
        Folder_out=plot_folder,
    )

    # c) Discharge per month
    plot_hydro_all_month(
        qsim_uncalibrated=qsim_uncal,
        qobs=qobs,
        qsim_cals=qsim_cal,
        qsim_cals_names=name_cal,
        Folder_out=plot_folder,
    )

    # d) Snow and glacier plot
    if "snow_basavg" in ds_basin_uncal:
        print("Plot snow (and glacier)")
        plot_snow_glacier(
            ds_basin_uncal,
            ds_basin_cal,
            names_cal=name_cal,
            Folder_out=plot_folder,
        )

if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        text_out = sm.output.output_txt

        plot_results_calib_runs(
            wflow_tomls = sm.input.toml_files,
            plot_folder = dirname(text_out),
            observations_locations = sm.params.observations_locations,
            observations_timeseries = sm.params.observations_timeseries,
            uncalibrated_run = sm.params.uncalibrated_run,
            calibration_runs_selection = sm.params.calibration_runs_selection,
        )
        # Write a file for snakemake tracking
        if not os.path.exists(dirname(text_out)):
            os.makedirs(dirname(text_out))
        with open(text_out, "w") as f:
            f.write(f"Plotted combined wflow results.\n")
    else:
        print("This script should be run from snakemake.")