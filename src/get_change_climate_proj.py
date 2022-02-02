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
# stats_nc_hist = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\historical_stats_gfdl.nc"
# stats_nc = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\stats-gfdl_ssp370_far.nc"

#%%

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")

# Snakemake options
project_dir = snakemake.params.project_dir
stats_nc_hist = snakemake.input.stats_nc_hist
stats_nc = snakemake.input.stats_nc
stats_time_nc_hist = snakemake.input.stats_time_nc_hist
stats_time_nc = snakemake.input.stats_time_nc

#additional folder structure info
folder_out = os.path.join(project_dir, "climate_projections")

ds_hist = xr.open_dataset(stats_nc_hist)
ds_clim = xr.open_dataset(stats_nc)

#get lat lon name of data
for dim in XDIMS:
    if dim in ds_hist.coords:
        x_dim = dim
for dim in YDIMS:
    if dim in ds_hist.coords:
        y_dim = dim

def get_change_clim_projections(ds_hist, ds_clim):   
    """
    Parameters
    ----------
    ds_hist : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for historical climate simulation.
    ds_clim : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for projected climate data.

    Returns
    -------
    Writes netcdf files with mean monthly (12 maps) and annual (1 map) change for the grid.
    Also writes scalar mean monthly and annual values averaged over the grid. 
    
    Returns
    -------
    mean_change : xarray dataset
        mean monthly change over the grid.
    mean_change_annual : xarray dataset
        mean annual change over the grid.
    mean_change_scalar : xarray dataset
        mean monthly change averaged over the grid.
    mean_change_annual_scalar : xarray dataset
        mean annual change averaged over the grid.

    """
    ds = []
    for var in ds_hist.data_vars:
        if var == "precip":
            #multiplicative for precip
            change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0])) / ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]) * 100
        else: #for temp
            #additive for temp
            change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]))
        ds.append(change.to_dataset())
    
    mean_change = xr.merge(ds)
    mean_change_annual = mean_change.mean("month")
    
    #scalar 
    mean_change_scalar = mean_change.mean([x_dim, y_dim])
    mean_change_annual_scalar = mean_change_annual.mean([x_dim, y_dim])
    return mean_change, mean_change_annual, mean_change_scalar, mean_change_annual_scalar

#calculate change
mean_change, mean_change_annual, mean_change_scalar, mean_change_annual_scalar = get_change_clim_projections(ds_hist, ds_clim)

#write to netcdf files
strings = ["mean_change", "mean_change_annual", "mean_change_scalar", "mean_change_annual_scalar"]
for i, ds in enumerate([mean_change, mean_change_annual, mean_change_scalar, mean_change_annual_scalar]):
    print(f"writing netcdf files {strings[i]}")
    dvars = ds.raster.vars        
    name_model = ds.model.values[0]
    name_scenario = ds.scenario.values[0]
    name_horizon = ds.horizon.values[0]
    name_nc_out = f"{strings[i]}-{name_model}_{name_scenario}_{name_horizon}.nc"
    ds.to_netcdf(os.path.join(folder_out, name_nc_out), encoding={k: {"zlib": True} for k in dvars})


#todo annual statistics uit de tijdseries halen. 
#resample AS-Month_hyd_start 



