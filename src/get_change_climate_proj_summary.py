# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

import hydromt
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

#%% outside snake
# clim_project_dir = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\isimip3"

#%%

# Snakemake options
clim_project_dir = snakemake.params.clim_project_dir
clim_project = os.path.basename(clim_project_dir)

#merge summary maps across models, scnearios and horizons. 
prefixes = ["monthly_change_mean_grid", "monthly_change_mean_scalar", "annual_change_scalar_stats"]
for prefix in prefixes:
    print(f"merging netcdf files {prefix}")
    #open annual scalar summary and merge
    ds = xr.open_mfdataset(os.path.join(clim_project_dir, f"{prefix}-*.nc"))
    dvars = ds.raster.vars        
    name_nc_out = f"{prefix}_summary.nc"
    ds.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})


#make csv summary for annual scalar values:
annual_summary = xr.open_dataset(os.path.join(clim_project_dir, "annual_change_scalar_stats_summary.nc"))
#write to csv
annual_summary.to_dataframe().to_csv(os.path.join(clim_project_dir, "annual_change_scalar_stats_summary.csv"))
