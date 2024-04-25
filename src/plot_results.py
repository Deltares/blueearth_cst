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

from typing import Union, List

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
else:
    from .func_plot_signature import (
        plot_signatures,
        plot_hydro,
        compute_metrics,
        plot_clim,
        plot_basavg,
    )

def get_wflow_results(
    wflow_root: Union[str, Path],
    wflow_config_fn: str = "wflow_sbm.toml",
    climate_source: List[str] = None, # climate source name used as suffix in config file 
    gauges_locs: Union[Path, str] = None,
):
    """
    TODO!
    """
    if climate_source is None:
        mod = WflowModel(root=wflow_root, mode="r", config_fn=wflow_config_fn) 
        qsim = mod.results["Q_gauges"].rename("Q") 
    else:
        mod = WflowModel(root=wflow_root, mode="r", config_fn=wflow_config_fn.replace(".toml", f"_{climate_source}.toml"))   
        qsim = mod.results["Q_gauges"].rename("Q") 
        #add climate data source 
        qsim = qsim.assign_coords(climate_source=(f"{climate_source}")).expand_dims(["climate_source"])
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

        # Climate data P, EP, T for wflow_subcatch
    ds_clim = xr.merge(
        [
            mod.results["P_subcatchment"],
            mod.results["T_subcatchment"],
            mod.results["EP_subcatchment"],
        ]
    )
    if climate_source is not None:
        #add climate data source 
        ds_clim = ds_clim.assign_coords(climate_source=(f"{climate_source}")).expand_dims(["climate_source"])

    # Other catchment average outputs
    ds_basin = xr.merge(
        [mod.results[dvar] for dvar in mod.results if "_basavg" in dvar]
    )
    ds_basin = ds_basin.squeeze(drop=True)
    # If precipitation, skip as this will be plotted with the other climate data
    if "precipitation_basavg" in ds_basin:
        ds_basin = ds_basin.drop_vars("precipitation_basavg")
    if climate_source is not None:
        #add climate data source 
        ds_basin = ds_basin.assign_coords(climate_source=(f"{climate_source}")).expand_dims(["climate_source"])


    return qsim, qsim_gauges, ds_clim, ds_basin


def analyse_wflow_historical(
    wflow_root: Union[str, Path],
    plot_dir: Union[str, Path] = None,
    observations_fn: Union[Path, str] = None,
    gauges_locs: Union[Path, str] = None,
    wflow_config_fn: str = "wflow_sbm.toml",
    climate_sources: List[str] = None,
    climate_sources_colors: List[str] = None,
):
    """
    Analyse and plot wflow model performance for historical run.

    Model results should include the following keys: Q_gauges,
    Q_gauges_{basename(gauges_locs)}, P_subcatchment, T_subcatchment, EP_subcatchment.


    Outputs:

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

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    plot_dir : Union[str, Path], optional
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_root folder.
    observations_fn : Union[Path, str], optional
        Path to observations timeseries file, by default None
        Required columns: time, wflow_id IDs of the locations as in ``gauges_locs``.
        Separator is , and decimal is .
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    wflow_config_fn : str, optional
        Name of the wflow configuration file, by default "wflow_sbm.toml". Used to read
        the right results files from the wflow model.
    climate_sources: List[str], optional 
        List of climate datasets used to run wflow. 
    climate_sources_colors: List[str], optional 
        List of colors to use for the different climate sources. Default is None. 
    """
    ### 1. Prepare output and plotting options ###

    # If plotting dir is None, create
    if plot_dir is None:
        plot_dir = os.path.join(wflow_root, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plotting options
    fs = 7
    lw = 0.8

    # Other plot options
    label = "simulated"  # "observed"
    color = "steelblue"  # "red"
    linestyle = "-"
    marker = "o"

    # Prepare colors dict
    if climate_sources_colors is not None and climate_sources is not None:
        color = {k: v for k, v in zip(climate_sources, climate_sources_colors)}
    elif climate_sources is not None:
        color = {k: None for k in climate_sources}
    else:
        color = {"climate_source":None}

    ### 2. Read the observations ###
    # check if user provided observations
    has_observations = False

    if observations_fn is not None and os.path.exists(observations_fn):
        has_observations = True

        # Read
        gdf_obs = hydromt.io.open_vector(gauges_locs, crs=4326, sep=",")
        da_ts_obs = hydromt.io.open_timeseries_from_table(
            observations_fn, name="Q", index_dim="wflow_id", sep=";"
        )
        ds_obs = hydromt.vector.GeoDataset.from_gdf(
            gdf_obs, da_ts_obs, merge_index="inner"
        )
        # Rename wflow_id to index
        ds_obs = ds_obs.rename({"wflow_id": "index"})
        qobs = ds_obs["Q"].load()

    ### 3. Read the wflow model and results ###
    # Instantiate wflow model
    #read wflow runs from several climate sources if more than one
    if len(np.atleast_1d(climate_sources).tolist())>1:  
        qsim = []
        qsim_gauges = []
        ds_clim = []
        ds_basin = []
        for climate_source in climate_sources:
            qsim_source, qsim_gauges_source, ds_clim_source, ds_basin_source = get_wflow_results(wflow_root=wflow_root, wflow_config_fn=wflow_config_fn, climate_source=climate_source, gauges_locs=gauges_locs)
            qsim.append(qsim_source)
            qsim_gauges.append(qsim_gauges_source)
            ds_clim.append(ds_clim_source)
            ds_basin.append(ds_basin_source)
        qsim = xr.merge(qsim)
        ds_clim = xr.merge(ds_clim)
        ds_basin = xr.merge(ds_basin)
        if qsim_gauges[0] is not None:
            qsim_gauges = xr.merge(qsim_gauges, compat="override")
            #merge with qsim
            qsim = xr.merge([qsim, qsim_gauges])["Q"]
    
    else:
        qsim, qsim_gauges, ds_clim, ds_basin = get_wflow_results(wflow_root=wflow_root, wflow_config_fn=wflow_config_fn, climate_source=None, gauges_locs=gauges_locs)

        # Merge with qsim
        if qsim_gauges is not None:
            qsim = xr.merge([qsim, qsim_gauges])["Q"]

    #make sure qsim, ds_clim and ds_basin have coord climate_source (even if only one climate source is used in the default case)
    if "climate_source" not in qsim.coords:
        if climate_sources is not None:
            qsim = qsim.assign_coords(climate_source=(f"{climate_sources[0]}")).expand_dims(["climate_source"])
            ds_clim = ds_clim.assign_coords(climate_source=(f"{climate_sources[0]}")).expand_dims(["climate_source"])
            ds_basin = ds_basin.assign_coords(climate_source=(f"{climate_sources[0]}")).expand_dims(["climate_source"])
        else:
            qsim = qsim.assign_coords(climate_source=("climate_source")).expand_dims(["climate_source"])
        # if "climate_source" not in ds_clim.coords:
            ds_clim = ds_clim.assign_coords(climate_source=("climate_source")).expand_dims(["climate_source"])
        # if "climate_source" not in ds_basin.coords:
            ds_basin = ds_basin.assign_coords(climate_source=("climate_source")).expand_dims(["climate_source"])


    ### 4. Plot climate data ###
    # No plots of climate data if wflow run is less than a year
    if len(ds_clim.time) <= 366:
        print("less than 1 year of data is available " "no yearly clim plots are made.")
    else:
        for index in ds_clim.index.values:
            print(f"Plot climatic data at wflow basin {index}")
            ds_clim_i = ds_clim.sel(index=index)
            # Plot per year
            plot_clim(ds_clim_i, plot_dir, f"wflow_{index}", "year", color)
            plt.close()
            # Plot per month
            plot_clim(ds_clim_i, plot_dir, f"wflow_{index}", "month", color)
            plt.close()

    ### 5. Plot other basin average outputs ###
    print("Plot basin average wflow outputs")
    plot_basavg(ds_basin, plot_dir, color)
    plt.close()

    ### 6. Plot hydrographs and compute performance metrics ###
    # Initialise the output performance table
    df_perf_all = pd.DataFrame()
    # Flag for plot signatures
    # (True if wflow run is longer than a year and observations are available)
    do_signatures = False

    # If possible, skip the first year of the wflow run (warm-up period)
    if len(qsim.time) > 365:
        print("Skipping the first year of the wflow run (warm-up period)")
        qsim = qsim.sel(
            time=slice(
                f"{qsim['time.year'][0].values+1}-{qsim['time.month'][0].values}-{qsim['time.day'][0].values}",
                None,
            )
        )
        if has_observations:
            do_signatures = True
    else:
        print("Simulation is less than a year so model warm-up period will be plotted.")
    # Sel qsim and qobs so that they have the same time period
    if has_observations:
        start = max(qsim.time.values[0], qobs.time.values[0])
        end = min(qsim.time.values[-1], qobs.time.values[-1])
        # make sure obs and sim have period in common
        if start < end:
            qsim = qsim.sel(time=slice(start, end))
            qobs = qobs.sel(time=slice(start, end))
        else:
            has_observations = False
            print("No common period between observations and simulation.")

    # Loop over the stations
    for station_id, station_name in zip(qsim.index.values, qsim.station_name.values):
        # Select the station
        qsim_i = qsim.sel(index=station_id)
        qobs_i = None
        if has_observations:
            if station_id in qobs.index.values:
                qobs_i = qobs.sel(index=station_id)

        # a) Plot hydrographs
        print(f"Plot hydrographs at wflow station {station_name}")
        plot_hydro(
            qsim=qsim_i,
            qobs=qobs_i,
            Folder_out=plot_dir,
            station_name=station_name,
            label=label,
            color=color,
            lw=lw,
            fs=fs,
        )
        plt.close()
        # b) Signature plot and performance metrics
        if do_signatures and qobs_i is not None:
            print("observed timeseries are available - making signature plots.")
            # Plot signatures
            plot_signatures(
                qsim=qsim_i,
                qobs=qobs_i,
                Folder_out=plot_dir,
                station_name=station_name,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                fs=fs,
                lw=lw,
            )
            plt.close()
            # Compute performance metrics
            df_perf = compute_metrics(
                qsim=qsim_i,
                qobs=qobs_i,
                station_name=station_name,
            )
            # Join with other stations
            if df_perf_all.empty:
                df_perf_all = df_perf
            else:
                df_perf_all = df_perf_all.join(df_perf)
        else:
            print(
                "observed timeseries are not available " "no signature plots are made."
            )

    # Save performance metrics to csv
    df_perf_all.to_csv(os.path.join(plot_dir, "performance_metrics.csv"))

    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir
        Folder_plots = f"{project_dir}/plots/wflow_model_performance"
        root = f"{project_dir}/hydrology_model"

        analyse_wflow_historical(
            wflow_root=root,
            plot_dir=Folder_plots,
            observations_fn=sm.params.observations_file,
            gauges_locs=sm.params.gauges_output_fid,
            climate_sources=sm.params.climate_sources,
        )
    else:
        analyse_wflow_historical(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model"),
            plot_dir=None,
            observations_fn=None,
            gauges_locs=None,
            climate_sources=None,
        )
