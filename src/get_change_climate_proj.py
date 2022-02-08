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
# stats_nc_hist = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\isimip3\historical_stats_gfdl.nc"
# stats_nc = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\isimip3\stats-gfdl_ssp370_far.nc"

# stats_time_nc_hist = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\isimip3\historical_stats_time_gfdl.nc"
# stats_time_nc = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\isimip3\stats_time-gfdl_ssp370_far.nc"

# start_month_hyd_year = "Jan"
# 
#%%

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")

# Snakemake options
clim_project_dir = snakemake.params.clim_project_dir
stats_nc_hist = snakemake.input.stats_nc_hist
stats_nc = snakemake.input.stats_nc
stats_time_nc_hist = snakemake.input.stats_time_nc_hist
stats_time_nc = snakemake.input.stats_time_nc
start_month_hyd_year = snakemake.params.start_month_hyd_year


ds_hist = xr.open_dataset(stats_nc_hist)
ds_clim = xr.open_dataset(stats_nc)

ds_hist_time = xr.open_dataset(stats_time_nc_hist)
ds_clim_time = xr.open_dataset(stats_time_nc)

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
    Writes netcdf files with mean monthly (12 maps) change for the grid.
    Also writes scalar mean monthly values averaged over the grid. 
    
    Returns
    -------
    monthly_change_mean_grid : xarray dataset
        mean monthly change over the grid.
    monthly_change_mean_scalar : xarray dataset
        mean monthly change averaged over the grid.

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
    
    monthly_change_mean_grid = xr.merge(ds)
    
    #scalar 
    monthly_change_mean_scalar = monthly_change_mean_grid.mean([x_dim, y_dim])
    return monthly_change_mean_grid, monthly_change_mean_scalar

#calculate change
monthly_change_mean_grid, monthly_change_mean_scalar = get_change_clim_projections(ds_hist, ds_clim)

#write to netcdf files
strings = ["monthly_change_mean_grid", "monthly_change_mean_scalar"]
for i, ds in enumerate([monthly_change_mean_grid, monthly_change_mean_scalar]):
    print(f"writing netcdf files {strings[i]}")
    dvars = ds.raster.vars        
    name_model = ds.model.values[0]
    name_scenario = ds.scenario.values[0]
    name_horizon = ds.horizon.values[0]
    name_nc_out = f"{strings[i]}-{name_model}_{name_scenario}_{name_horizon}.nc"
    ds.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})



#%% get annual statistics from time series of monthly variables


# stats = ["mean", "std", "median", "q_90", "q_75", "q_10", "q_25"]

def get_change_annual_clim_proj(ds_hist_time, ds_clim_time, stats = ["mean", "std", "var", "median", "q_90", "q_75", "q_10", "q_25"], start_month_hyd_year = "Jan"):
    """

    Parameters
    ----------
    ds_hist_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over the grid (historical).
    ds_clim_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over the grid (projection).
    stats : list of strings of statistics 
        quantiles should be provided as q_xx. The default is ["mean", "std", "var", "median", "q_90", "q_75", "q_10", "q_25"]
    start_month_hyd_year : str, optional
        Month start of hydrological year. The default is "Jan".

    Returns
    -------
    stats_annual_change : xarray dataset
        annual statistics per each models/scenario/horizon.

    """
    ds = []
    for var in ds_hist_time.data_vars:
        if var == "precip":
            #multiplicative for precip
            hist = ds_hist_time[var].resample(time = f"AS-{start_month_hyd_year}").sum("time").sel(horizon = ds_hist_time.horizon.values[0], scenario = ds_hist_time.scenario.values[0])
            clim = ds_clim_time[var].resample(time = f"AS-{start_month_hyd_year}").sum("time")
            # change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0])) / ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]) * 100
        else: #for temp
            #additive for temp
            hist = ds_hist_time[var].resample(time = f"AS-{start_month_hyd_year}").mean("time").sel(horizon = ds_hist_time.horizon.values[0], scenario = ds_hist_time.scenario.values[0])
            clim = ds_clim_time[var].resample(time = f"AS-{start_month_hyd_year}").mean("time")
        
        #calc statistics
        for stat_name in stats: #, stat_props in stats_dic.items():
            if "q_" in stat_name:
                qvalue = int(stat_name.split("_")[1])/100
                hist_stat = getattr(hist, "quantile")(qvalue, "time")
                clim_stat = getattr(clim, "quantile")(qvalue, "time")
            else:
                hist_stat = getattr(hist, stat_name)("time")
                clim_stat = getattr(clim, stat_name)("time")
            
            if var == "precip":
                change = (clim_stat - hist_stat) / hist_stat * 100
            else:
                change = (clim_stat - hist_stat)
            change = change.assign_coords({"stats": stat_name}).expand_dims("stats")
            
            if "quantile" in change.coords:
                change = change.drop("quantile")
            ds.append(change.to_dataset())
                
    stats_annual_change = xr.merge(ds)
    return stats_annual_change

#calculate statistics (mean, std, 0.1 0.25 0.50 0.75 0.90 quantiles of annual precip sum and mean temp)    
stats_annual_change = get_change_annual_clim_proj(ds_hist_time, ds_clim_time)

#write to netcdf files

dvars = stats_annual_change.raster.vars        
name_model = stats_annual_change.model.values[0]
name_scenario = stats_annual_change.scenario.values[0]
name_horizon = stats_annual_change.horizon.values[0]
name_nc_out = f"annual_change_scalar_stats-{name_model}_{name_scenario}_{name_horizon}.nc"
stats_annual_change.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})


