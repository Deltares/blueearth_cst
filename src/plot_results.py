# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:18:38 2021

@author: bouaziz
"""


import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import hydromt
import os
from hydromt_wflow import WflowModel

from func_plot_signature import (
    plot_signatures,
    plot_hydro,
    plot_hydro_1y,
    plot_clim,
    plot_basavg,
)


fs = 8
lw = 0.8

# Snakemake options
project_dir = snakemake.params.project_dir
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime

observations_timeseries = snakemake.params.observations_file
gauges_output_name = snakemake.params.gauges_output_fid
gauges_output_name = os.path.basename(gauges_output_name).split(".")[0]

outputs = snakemake.params.outputs

project = gauges_output_name.split("-")[-1]

Folder_plots = f"{project_dir}/plots/wflow_model_performance"
Folder_run = f"{project_dir}/hydrology_model"

if not os.path.isdir(Folder_plots):
    os.mkdir(Folder_plots)

# Other options
labels = ["Mod."]
colors = ["orange"]
linestyles = ["-"]
markers = ["o"]

#%%
# Instantiate wflow model
mod = WflowModel(root=Folder_run, mode="r")

# read gauges staticgeoms
gdf_gauges = mod.staticgeoms["gauges"]

# read outputlocs staticgeoms
if f"gauges_{gauges_output_name}" in mod.staticgeoms:
    gdf_outlocs = mod.staticgeoms[f"gauges_{gauges_output_name}"]
    stationID = "wflow_id"  # column name in staticgeoms containing the stations IDs
    gdf_outlocs.index = gdf_outlocs[stationID]

# read model output at gauges locations from model setup
qsim_gauges = mod.results["Q_gauges"]
# add station_name
qsim_gauges = qsim_gauges.assign_coords(
    station_name=(
        "index",
        ["wflow_" + x for x in list(qsim_gauges["index"].values.astype(str))],
    )
)


# read model output at output locations provided by user
if f"gauges_{gauges_output_name}" in mod.staticgeoms:
    qsim_outloc = mod.results[f"Q_gauges_{gauges_output_name}"]
    # add station_name
    qsim_outloc = qsim_outloc.assign_coords(
        station_name=(
            "index",
            list(gdf_outlocs["station_name"][qsim_outloc.index.values].values),
        )
    )
    # rename run to label and rename var to Q
    ds_sim_outlocs = (
        qsim_outloc.to_dataset()
        .assign_coords({"runs": labels[0]})
        .expand_dims("runs")
        .rename({f"Q_gauges_{gauges_output_name}": "Q"})
    )

# P, EP and T for wflow_subcatch
ds_clim = xr.merge(
    [
        mod.results["P_subcatchment"],
        mod.results["T_subcatchment"],
        mod.results["EP_subcatchment"],
    ]
)

# Other wflow outputs for wflow basin
ds_basin = xr.merge([mod.results[dvar] for dvar in mod.results if "_basavg" in dvar])
ds_basin = ds_basin.squeeze(drop=True)
# If precipitation, skip as this will be plotted with the other climate data
if "precipitation_basavg" in ds_basin:
    ds_basin = ds_basin.drop_vars("precipitation_basavg")

#%% Read the observations data
# read timeseries data and match with existing gdf
has_observations = False
if observations_timeseries is not None:
    if os.path.exists(observations_timeseries):
        has_observations = True

# Discharge data
# make sure the user provided a observation file and ouput locations
if (f"gauges_{gauges_output_name}" in mod.staticgeoms) & (
    has_observations
):
    name = f"gauges_{gauges_output_name}"  # gauges locations in staticgeoms
    da_ts = hydromt.io.open_timeseries_from_table(
        observations_timeseries, name=name, sep=";"
    )
    da = hydromt.vector.GeoDataArray.from_gdf(gdf_outlocs, da_ts, index_dim="index")
    qobs_outloc = da
    # rename run to Obs. and rename var to Q
    ds_obs = (
        qobs_outloc.to_dataset()
        .assign_coords({"runs": "Obs."})
        .expand_dims("runs")
        .rename({f"gauges_{gauges_output_name }": "Q"})
    )

#%% make plots - first loop over output locations

# combine sim and obs at outputloc in one dataset if timeseries observations exist
if (f"gauges_{gauges_output_name}" in mod.staticgeoms) & (
    has_observations
):
    ds_outlocs = ds_obs.combine_first(ds_sim_outlocs)
    # combine_first now seems to somehow drop the coordinate station_name?
    ds_outlocs["station_name"] = ds_obs["station_name"]
    ds_outlocs = ds_outlocs.set_coords("station_name")

# rename Q in modeled dataset at gauges locations
ds_sim_gauges = (
    qsim_gauges.to_dataset()
    .assign_coords({"runs": labels[0]})
    .expand_dims("runs")
    .rename({"Q_gauges": "Q"})
)

# select dataset based on gauges or/and outputloc locations
# if no user output and observations are provided:
if ((f"{gauges_output_name}" in mod.staticgeoms) == False) & (
    not has_observations
):
    ds_list = [ds_sim_gauges]
# if user output locs are available but no observations timeseries:
elif ((f"{gauges_output_name}" in mod.staticgeoms) == True) & (
    not has_observations
):
    ds_list = [ds_sim_gauges, ds_sim_outlocs]
# if output locs and observations are available - make hydro plots for gauges and make hydro and signature plots for outputlocs
else:
    ds_list = [ds_sim_gauges, ds_outlocs]

# plot and loop over datasets with outlocs and gauges locations
df_perf_all = pd.DataFrame()
for ds in ds_list:
    for station_id, station_name in zip(ds.index.values, ds.station_name.values):
        print(station_id, station_name)
        # skip first year for hydro -- warm up period
        if len(ds.time) > 365:
            dsq = ds.sel(index=station_id).sel(
                time=slice(
                    f"{ds['time.year'][0].values+1}-{ds['time.month'][0].values}-{ds['time.day'][0].values}",
                    None,
                )
            )  # .dropna(dim='time')
        else:
            dsq = ds.sel(index=station_id)
        # plot hydro
        if len(np.unique(dsq["time.year"])) >= 3:
            year_min = pd.to_datetime(dsq["Q"].sel(runs="Mod.").idxmin().values).year
            year_max = pd.to_datetime(dsq["Q"].sel(runs="Mod.").idxmax().values).year
            # if min and max occur during the same year, select 2nd year with modeled min flow:
            if year_min == year_max:
                year_min = (
                    dsq.resample(time="A").min("time").isel(time=1)["time.year"].values
                )
            plot_hydro(
                dsq,
                dsq.time[0],
                dsq.time[-1],
                str(year_max),
                str(year_min),
                labels,
                colors,
                Folder_plots,
                station_name,
            )
            plt.close()
        else:
            plot_hydro_1y(
                dsq,
                dsq.time[0],
                dsq.time[-1],
                labels,
                colors,
                Folder_plots,
                station_name,
            )
            plt.close()

        # dropna time for signature calculations.
        # skip first year for signatures -- warm up period -- if model did not run for a full year - skip signature plots
        if len(ds.time) > 365:
            dsq = (
                ds["Q"]
                .sel(index=station_id)
                .sel(
                    time=slice(
                        f"{ds['time.year'][0].values+1}-{ds['time.month'][0].values}-{ds['time.day'][0].values}",
                        None,
                    )
                )
                .to_dataset()
                .dropna(dim="time")
            )
            if (len(dsq["Q"].time) > 0) & (
                "Obs." in dsq["runs"]
            ):  # only plot signatures if observations timeseries are present
                print("observed timeseries are available - making signature plots.")
                df_perf = plot_signatures(
                    dsq, labels, colors, linestyles, markers, Folder_plots, station_name
                )
                if df_perf_all.empty:
                    df_perf_all = df_perf
                else:
                    df_perf_all = df_perf_all.join(df_perf)
            else:
                print(
                    "observed timeseries are not available - no signature plots are made."
                )
            plt.close()
        else:
            print(
                "less than 1 year of data is available - no signature plots are made."
            )

# save performance metrics to csv
df_perf_all.to_csv(os.path.join(Folder_plots, "performance_metrics.csv"))

for index in ds_clim.index.values:
    print(f"Plot climatic data at wflow basin {index}")
    ds_clim_i = ds_clim[["P_subcatchment", "EP_subcatchment", "T_subcatchment"]].sel(
        index=index
    )
    plot_clim(ds_clim_i, Folder_plots, f"wflow_{index}")
    plt.close()

# Plots for other wflow outputs
print("Plot basin average wflow outputs")
plot_basavg(ds_basin, Folder_plots)
plt.close()

# TODO add summary maps mean prec and temp spatially?
