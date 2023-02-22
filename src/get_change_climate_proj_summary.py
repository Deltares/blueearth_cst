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

#%%

# Snakemake options
clim_project_dir = snakemake.params.clim_project_dir
clim_project = os.path.basename(clim_project_dir)
list_files = snakemake.input.stats_nc_change

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

g = sns.JointGrid(
    data=df,
    x="precip",
    y="temp",
    hue="scenario",
)
g.plot_joint(sns.scatterplot, s=100, alpha=0.5, data=df, style="horizon")
g.plot_marginals(sns.kdeplot)
g.set_axis_labels(xlabel="Mean precipitation (%)", ylabel="Mean temperature (degC)")
g.ax_joint.grid()
g.ax_joint.legend(loc="upper right", bbox_to_anchor=(1.3, 1.4))
g.savefig(os.path.join(clim_project_dir, "plots", "projected_climate_statistics.png"))
