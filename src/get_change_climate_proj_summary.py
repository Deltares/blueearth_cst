# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

import hydromt
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr
import numpy as np

#%%

# Snakemake options
clim_project_dir = snakemake.params.clim_project_dir
clim_project = os.path.basename(clim_project_dir)
list_files = snakemake.input.stats_nc_change
horizons = snakemake.params.horizons

# merge summary maps across models, scnearios and horizons.
# prefixes = ["monthly_change_mean_grid", "monthly_change_mean_scalar", "annual_change_scalar_stats"]
prefix = "annual_change_scalar_stats"
# for prefix in prefixes:
print(f"merging netcdf files {prefix}")
# open annual scalar summary and merge
list_files_not_empty = []
# list_files = glob.glob(os.path.join(clim_project_dir, f"{prefix}-*.nc"))
for file in list_files:
    ds_f = xr.open_dataset(file)
    # don't read in the dummy datasets
    if len(ds_f) > 0:
        list_files_not_empty.append(file)
# ds = xr.open_mfdataset(os.path.join(clim_project_dir, f"{prefix}-*.nc"))
ds = xr.open_mfdataset(list_files_not_empty)
dvars = ds.raster.vars
name_nc_out = f"{prefix}_summary.nc"
ds.to_netcdf(
    os.path.join(clim_project_dir, name_nc_out),
    encoding={k: {"zlib": True} for k in dvars},
)


# make csv summary for annual scalar values:
annual_summary = xr.open_dataset(
    os.path.join(clim_project_dir, "annual_change_scalar_stats_summary.nc")
)
# write to csv
annual_summary.to_dataframe().to_csv(
    os.path.join(clim_project_dir, "annual_change_scalar_stats_summary.csv")
)

# just keep mean for temp and precip for response surface plots
df = annual_summary.sel(stats="mean").to_dataframe()
df.to_csv(os.path.join(clim_project_dir, "annual_change_scalar_stats_summary_mean.csv"))

# plot change
if not os.path.exists(os.path.join(clim_project_dir, "plots")):
    os.mkdir(os.path.join(clim_project_dir, "plots"))

# Rename horizon names to the middle year of the period
hz_list = df.index.levels[df.index.names.index("horizon")].tolist()
for hz in horizons:
    # Get start and end year
    period = horizons[hz].split(",")
    period = [int(i) for i in period]
    horizon_year = int((period[0] + period[1]) / 2)
    # Replace hz values by horizon_year in hz_list
    hz_list = [horizon_year if h == hz else h for h in hz_list]

# Set new values in multiindex dataframe
df.index = df.index.set_levels(hz_list, level="horizon")

scenarios = np.unique(df.index.get_level_values('scenario'))
clrs = []
for s in scenarios: 
    if s == 'ssp126':
        clrs.append('#003466')
    if s == 'ssp245':
        clrs.append('#f69320')
    if s == 'ssp370':
        clrs.append('#df0000')
    elif s == 'ssp585':
        clrs.append('#980002')
g = sns.JointGrid(
    data=df,
    x="precip",
    y="temp",
    hue="scenario",
)
g.plot_joint(sns.scatterplot, s=100, alpha=0.5, data=df, style="horizon", palette=clrs)
g.plot_marginals(sns.kdeplot, palette=clrs)
g.set_axis_labels(xlabel="Change in mean precipitation (%)", ylabel="Change in mean temperature (degC)")
g.ax_joint.grid()
g.ax_joint.legend(loc="right", bbox_to_anchor=(1.5, 0.5))
g.savefig(os.path.join(clim_project_dir, "plots", "projected_climate_statistics.png"))
