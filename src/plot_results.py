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
import seaborn as sns

from typing import Union, List, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.func_plot_signature import (
        plot_signatures,
        plot_hydro,
        compute_metrics,
        plot_clim,
        plot_basavg,
    )
    from plot_utils.plot_anomalies import plot_timeseries_anomalies
    from plot_utils.plot_budyko import (
        get_upstream_clim_basin,
        determine_budyko_curve_terms,
        plot_budyko,
    )
    from wflow.wflow_utils import get_wflow_results
else:
    from .plot_utils.func_plot_signature import (
        plot_signatures,
        plot_hydro,
        compute_metrics,
        plot_clim,
        plot_basavg,
    )
    from .plot_utils.plot_anomalies import plot_timeseries_anomalies
    from .plot_utils.plot_budyko import (
        get_upstream_clim_basin,
        determine_budyko_curve_terms,
        plot_budyko,
    )
    from .wflow.wflow_utils import get_wflow_results


def analyse_wflow_historical(
    wflow_root: Union[str, Path],
    climate_sources: List[str],
    plot_dir: Optional[Union[str, Path]] = None,
    observations_fn: Optional[Union[Path, str]] = None,
    gauges_locs: Optional[Union[Path, str]] = None,
    wflow_config_fn_prefix: str = "wflow_sbm",
    climate_sources_colors: Optional[List[str]] = None,
    split_year: Optional[int] = None,
    add_budyko_plot: bool = True,
    max_nan_year: int = 60,
    max_nan_month: int = 5,
    skip_temp_pet_sources: List[str] = [],
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
    - plot of annual trends in streamflow for each climate source and for observations.
    - optional: plot of runoff coefficient as a function of aridity index (Budyko
      framework) for each discharge observation station.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    climate_sources: List[str]
        List of climate datasets used to run wflow.
    plot_dir : Union[str, Path], optional
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_root folder.
    observations_fn : Union[Path, str], optional
        Path to observations timeseries file, by default None
        Required columns: time, wflow_id IDs of the locations as in ``gauges_locs``.
        Separator is ; and decimal is .
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    wflow_config_fn_prefix : str, optional
        Prefix name of the wflow configuration file, by default "wflow_sbm". Used to
        read the right results files from the wflow model.
    climate_sources_colors: List[str], optional
        List of colors to use for the different climate sources. Default is None.
    split_year : int, optional
        Derive additional trends for years before and after this year.
    add_budyko_plot : bool, optional
        If True, plot the budyko framework. Default is True.
    max_nan_year : int, optional
        Maximum number of missing days per year in the observations data to consider
        the year for the discharge analysis. By default 60.
    max_nan_month : int, optional
        Maximum number of missing days per month in the observations data to consider
        the month for the discharge analysis. By default 5.
    skip_temp_pet_sources : List[str]
        List of climate sources for which to skip the plotting temperature and
        potential evapotranspiration.
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
    linestyle = "-"
    marker = "o"

    # Prepare colors dict
    if climate_sources_colors is not None:
        cmap = np.atleast_1d(climate_sources_colors).tolist()
    else:
        cmap = sns.color_palette("Set2", len(np.atleast_1d(climate_sources).tolist()))
    color = {k: v for k, v in zip(np.atleast_1d(climate_sources).tolist(), cmap)}

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
    # read wflow runs for different climate sources
    qsim = []
    ds_clim = []
    ds_basin = []
    for climate_source in np.atleast_1d(climate_sources).tolist():
        wflow_config_fn = wflow_config_fn_prefix + f"_{climate_source}.toml"
        qsim_source, ds_clim_source, ds_basin_source = get_wflow_results(
            wflow_root=wflow_root,
            config_fn=wflow_config_fn,
            gauges_locs=gauges_locs,
            remove_warmup=False,
        )
        # Add climate source to dimension
        qsim_source = qsim_source.assign_coords(
            climate_source=(f"{climate_source}")
        ).expand_dims(["climate_source"])
        ds_clim_source = ds_clim_source.assign_coords(
            climate_source=(f"{climate_source}")
        ).expand_dims(["climate_source"])
        ds_basin_source = ds_basin_source.assign_coords(
            climate_source=(f"{climate_source}")
        ).expand_dims(["climate_source"])
        qsim.append(qsim_source)
        ds_clim.append(ds_clim_source)
        ds_basin.append(ds_basin_source)
    qsim = xr.concat(qsim, dim="climate_source")
    ds_clim = xr.concat(ds_clim, dim="climate_source")
    ds_basin = xr.concat(ds_basin, dim="climate_source")

    ### make sure the time period of all climate sources is the same for the subsequent plots.
    qsim = qsim.dropna("time")
    ds_clim = ds_clim.dropna("time")
    ds_basin = ds_basin.dropna("time")

    ### 4. Plot climate data ###
    # No plots of climate data if wflow run is less than a year
    if len(ds_clim.time) <= 366:
        print("less than 1 year of data is available " "no yearly clim plots are made.")
    else:
        for index in ds_clim.index.values:
            print(f"Plot climatic data at wflow basin {index}")
            ds_clim_i = ds_clim.sel(index=index)
            # Plot per year
            plot_clim(
                ds_clim_i,
                Folder_out=plot_dir,
                station_name=f"wflow_{index}",
                period="year",
                color=color,
                skip_temp_pet_sources=skip_temp_pet_sources,
            )
            plt.close()
            # Plot per month
            plot_clim(
                ds_clim_i,
                Folder_out=plot_dir,
                station_name=f"wflow_{index}",
                period="month",
                color=color,
                skip_temp_pet_sources=skip_temp_pet_sources,
            )
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
            color=color,
            lw=lw,
            fs=fs,
            max_nan_year=max_nan_year,
            max_nan_month=max_nan_month,
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
                color=color,
                linestyle=linestyle,
                marker=marker,
                fs=fs,
                lw=lw,
            )
            plt.close()
            # Compute performance metrics
            df_perf = pd.DataFrame()
            for climate_source in qsim.climate_source.values:
                df_perf_source = compute_metrics(
                    qsim=qsim_i.sel(climate_source=climate_source),
                    qobs=qobs_i,
                    station_name=station_name,
                    climate_source=climate_source,
                )
                df_perf = pd.concat([df_perf, df_perf_source])
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

    ### 7. Plot trends in mean annual streamflow ###
    # Derive the anomalies and trends for each climate source
    for climate_source in qsim.climate_source.values:
        # Filter the dataset
        ds_source = qsim.sel(climate_source=climate_source).to_dataset()

        # Plot the anomalies if there is more than 3 years of data
        nb_years = np.unique(qsim["time.year"].values).size
        if nb_years >= 3:
            plot_timeseries_anomalies(
                ds=ds_source,
                path_output=plot_dir,
                split_year=split_year,
                suffix=climate_source,
            )

    # if there are observations also plot anomalies and trends of observations
    if has_observations:
        # Plot the anomalies if there is more than 3 years of data
        nb_years = np.unique(qobs["time.year"].values).size
        if nb_years >= 3:
            plot_timeseries_anomalies(
                ds=qobs.to_dataset(),
                path_output=plot_dir,
                split_year=split_year,
                suffix="obs",
            )

    ### 8. Plot budyko framework if there are observations ###
    if has_observations and add_budyko_plot:
        ds_clim_sub_annual = []
        for climate_source in np.atleast_1d(climate_sources).tolist():
            config_fn = wflow_config_fn_prefix + f"_{climate_source}.toml"
            # get mean annual precip, pet and q_specific upstream of observation locations for a specific source
            ds_clim_sub_source, ds_clim_sub_annual_source = get_upstream_clim_basin(
                qobs, wflow_root, config_fn
            )
            # calculate budyko terms
            ds_clim_sub_annual_source = determine_budyko_curve_terms(
                ds_clim_sub_annual_source
            )
            ds_clim_sub_annual_source = ds_clim_sub_annual_source.assign_coords(
                climate_source=(f"{climate_source}")
            ).expand_dims(["climate_source"])
            ds_clim_sub_annual.append(ds_clim_sub_annual_source)
        ds_clim_sub_annual = xr.merge(ds_clim_sub_annual)

        # plot budyko with different sources.
        plot_budyko(ds_clim_sub_annual, plot_dir, color=color)

    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir
        Folder_plots = f"{project_dir}/plots/wflow_model_performance"
        root = f"{project_dir}/hydrology_model/run_default"

        analyse_wflow_historical(
            wflow_root=root,
            plot_dir=Folder_plots,
            observations_fn=sm.params.observations_file,
            gauges_locs=sm.params.gauges_output_fid,
            climate_sources=sm.params.climate_sources,
            climate_sources_colors=sm.params.climate_sources_colors,
            add_budyko_plot=sm.params.add_budyko_plot,
            max_nan_year=sm.params.max_nan_year,
            max_nan_month=sm.params.max_nan_month,
            skip_temp_pet_sources=sm.params.skip_temp_pet_sources,
        )
    else:
        analyse_wflow_historical(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model"),
            plot_dir=None,
            observations_fn=None,
            gauges_locs=None,
            climate_sources=None,
        )
