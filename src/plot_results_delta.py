# -*- coding: utf-8 -*-
"""
Plot wflow results and compare to observations if any
"""

import xarray as xr
import numpy as np
import os
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import hydromt
from hydromt_wflow import WflowModel
import seaborn as sns

from typing import Union, List, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from func_plot_signature import (
        plot_signatures,
        plot_hydro,
        compute_metrics,
        plot_clim,
        plot_basavg,
    )
    from plot_utils.plot_anomalies import plot_timeseries_anomalies
    from plot_utils.plot_change_delta_runs import (
        plot_near_far_abs,
    )
else:
    from .func_plot_signature import (
        plot_signatures,
        plot_hydro,
        compute_metrics,
        plot_clim,
        plot_basavg,
    )
    from .plot_utils.plot_anomalies import plot_timeseries_anomalies
    from .plot_utils.plot_change_delta_runs import (
        plot_near_far_abs,
    )



def get_wflow_results(
    wflow_root: Union[str, Path],
    config_fn: str = "wflow_sbm.toml", 
    gauges_locs: Union[Path, str] = None,
):
    """
    Get wflow results as xarray.Dataset for simulated discharges, simulated flux/states as basin averages.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    wflow_config_fn : str, optional
        Name of the wflow configuration file, by default "wflow_sbm.toml". Used to read
        the right results files from the wflow model.
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .

    Returns
    ----------
    qsim: xr.Dataset
        simulated discharge at wflow basin locations and at observation locations 
    ds_basin: xr.Dataset
        basin average flux and state variables 

    """
    mod = WflowModel(
    root=wflow_root,
    mode="r",
    config_fn=config_fn,
    )

    #Q at wflow locations 
    qsim = mod.results["Q_gauges"].rename("Q")
    qsim = qsim.assign_coords(
    station_name=(
        "index",
        ["wflow_" + x for x in list(qsim["index"].values.astype(str))],
    )
    )      

    # Discharge at the gauges_locs if present
    if gauges_locs is not None and os.path.exists(gauges_locs):
        # Get name of gauges dataset from the gauges locations file
        gauges_output_name = os.path.basename(gauges_locs).split(".")[0]
        if f"Q_gauges_{gauges_output_name}" in mod.results:
            qsim_gauges = mod.results[f"Q_gauges_{gauges_output_name}"].rename("Q")
            # Add station_name
            gdf_gauges = (
                mod.geoms[f"gauges_{gauges_output_name}"]
                .rename(columns={"wflow_id": "index"})
                .set_index("index")
            )
            qsim_gauges = qsim_gauges.assign_coords(
                station_name=(
                    "index",
                    list(gdf_gauges["station_name"][qsim_gauges.index.values].values),
                )
            )
    else:
        qsim_gauges = None

    #merge qsim and qsim_gauges
    qsim = xr.concat([qsim, qsim_gauges], dim="index")

    # Other catchment average outputs
    ds_basin = xr.merge(
        [mod.results[dvar] for dvar in mod.results if "_basavg" in dvar]
    )
    ds_basin = ds_basin.squeeze(drop=True)
    # If precipitation, skip as this will be plotted with the other climate data
    if "precipitation_basavg" in ds_basin:
        ds_basin = ds_basin.drop_vars("precipitation_basavg")

    return qsim, ds_basin

def analyse_wflow_delta(
    wflow_hist_run_config: Path,
    wflow_delta_runs_config: List[Path],
    models: List[str],
    scenarios: List[str],
    gauges_locs: Union[Path, str] = None,
    plot_dir: Union[str, Path] = None,


):
    """
    Analyse and plot wflow model performance for delta change runs.

    Model results should include the following keys: Q_gauges,
    Q_gauges_{basename(gauges_locs)}, P_subcatchment, T_subcatchment, EP_subcatchment.


    Outputs:
    TODO !! 

    - plot of hydrographs at the outlet(s) and gauges_locs if provided. If wflow run is
      three years or less, only the daily hydrograph will be plotted. If wflow run is
      longer than three years, plots will also include the yearly hydrograph, the
      monthly average and hydrographs for the wettest and driest years. If observations
      are available, they are added as well.
    - plot of signature plots if wflow run is longer than a year and if observations
      are available.
    - plot of climate data per year and per month at the subcatchment level if wflow run
      is longer than a year.
    - plot of basin average outputs (e.g. soil moisture, snow, etc.). The variables to
      include should have the postfix _basavg in the wflow output file.
    - compute performance metrics (daily and monthly KGE, NSE, NSElog, RMSE, MSE, Pbias,
      VE) if observations are available and if wflow run is longer than a year. Metrics
      are saved to a csv file.
    - plot of annual trends in streamflow for each climate source and for observations.
    - optional: plot of runoff coefficient as a function of aridity index (Budyko
      framework) for each discharge observation station.

    Parameters
    ----------
    wflow_hist_run_config : Union[str, Path]
        Path to the wflow model config file of the historical run  
    wflow_runs_toml : List[Path]
        List of paths of config files for the delta change runs. 
    plot_dir : Union[str, Path], optional
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_hist_run_config folder.
    observations_fn : Union[Path, str], optional
        Path to observations timeseries file, by default None
        Required columns: time, wflow_id IDs of the locations as in ``gauges_locs``.
        Separator is , and decimal is .
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    """
    ### 1. Prepare output and plotting options ###

    # If plotting dir is None, create
    if plot_dir is None:
        wflow_root = os.path.dirname(wflow_hist_run_config)
        plot_dir = os.path.join(wflow_root, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plotting options
    fs = 7
    lw = 0.8

    # Other plot options
    linestyle = "-"
    marker = "o"

    #read model results for historical 
    root = os.path.dirname(wflow_hist_run_config)
    config_fn =  os.path.basename(wflow_hist_run_config)
    qsim_hist, ds_basin_hist = get_wflow_results(root, config_fn, gauges_locs)
   

    #read the model results and merge to single netcdf
    qsim_delta = []
    ds_basin_delta = []
    for delta_config in wflow_delta_runs_config:
        model = os.path.basename(delta_config).split(".")[0].split("_")[-3]
        scenario = os.path.basename(delta_config).split(".")[0].split("_")[-2]
        horizon = os.path.basename(delta_config).split(".")[0].split("_")[-1]
        root = os.path.dirname(delta_config)
        config_fn =  os.path.basename(delta_config)
        qsim_delta_run, ds_basin_delta_run = get_wflow_results(root, config_fn, gauges_locs)
        qsim_delta_run = qsim_delta_run.assign_coords({"horizon":horizon, "model":model, "scenario":scenario}).expand_dims(["horizon", "model", "scenario"])
        ds_basin_delta_run = ds_basin_delta_run.assign_coords({"horizon":horizon, "model":model, "scenario":scenario}).expand_dims(["horizon", "model", "scenario"])
        qsim_delta.append(qsim_delta_run)
        ds_basin_delta.append(ds_basin_delta_run)
    qsim_delta = xr.merge(qsim_delta) 
    ds_basin_delta = xr.merge(ds_basin_delta)

    cmap = sns.color_palette("Set2", len(np.atleast_1d(scenarios).tolist()))

    #plot cum flows
    for index in qsim_delta.index.values: 

        #plot cumsum
        plot_near_far_abs(
            qsim_delta["Q"].cumsum("time"), 
            qsim_hist.cumsum("time"),
            index=index,
            plot_dir=plot_dir,
            ylabel="Q",
            figname_prefix="cumsum",
            cmap=cmap,
            fs=fs
        )

        #plot mean monthly flow
        plot_near_far_abs(
            qsim_delta["Q"].groupby("time.month").mean("time"), 
            qsim_hist.groupby("time.month").mean("time"),
            index=index,
            plot_dir=plot_dir,
            ylabel="Q (m$^3$s$^{-1}$)",
            figname_prefix="mean_monthly_Q",
            cmap=cmap,
            fs=fs
        )



        #plot nm7q
        #plot maxq



    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir
        Folder_plots = f"{project_dir}/plots/wflow_model_performance"
        root = f"{project_dir}/hydrology_model"

        analyse_wflow_delta(
            #TODO!!
            wflow_hist_run_config=root,
            wflow_delta_runs_config=Folder_plots,
            models=sm.params.observations_file,
            scenarios=sm.params.gauges_output_fid,
            gauges_locs=sm.params.climate_sources,
            plot_dir=sm.params.climate_sources_colors,
        )
    else:
        print("run with snakemake please")