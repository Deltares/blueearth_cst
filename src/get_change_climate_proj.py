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

#%%

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

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
    for var in intersection(ds_hist.data_vars, ds_clim.data_vars):
        if var == "precip":
            #multiplicative for precip
            change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0])) / ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]) * 100
        else: #for temp
            #additive for temp
            change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]))
        ds.append(change.to_dataset())
    
    monthly_change_mean_grid = xr.merge(ds)
    
    return monthly_change_mean_grid


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
    for var in intersection(ds_hist_time.data_vars, ds_clim_time.data_vars):
        #only keep full hydrological years
        start_hyd_year_hist = pd.to_datetime(f"{ds_hist_time['time.year'][0].values}-{start_month_hyd_year}")
        end_hyd_year_hist = pd.to_datetime(f"{ds_hist_time['time.year'][-1].values}-{start_month_hyd_year}") - pd.DateOffset(months=1)
        #same for clim
        start_hyd_year_clim = pd.to_datetime(f"{ds_clim_time['time.year'][0].values}-{start_month_hyd_year}")
        end_hyd_year_clim = pd.to_datetime(f"{ds_clim_time['time.year'][-1].values}-{start_month_hyd_year}") - pd.DateOffset(months=1)
        
        if var == "precip":
            #multiplicative for precip
            hist = ds_hist_time[var].sel(time = slice(start_hyd_year_hist, end_hyd_year_hist)).resample(time = f"AS-{start_month_hyd_year}").sum("time").sel(horizon = ds_hist_time.horizon.values[0], scenario = ds_hist_time.scenario.values[0])
            clim = ds_clim_time[var].sel(time = slice(start_hyd_year_clim, end_hyd_year_clim)).resample(time = f"AS-{start_month_hyd_year}").sum("time")
            # change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0])) / ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]) * 100
        else: #for temp
            #additive for temp
            hist = ds_hist_time[var].sel(time = slice(start_hyd_year_hist, end_hyd_year_hist)).resample(time = f"AS-{start_month_hyd_year}").mean("time").sel(horizon = ds_hist_time.horizon.values[0], scenario = ds_hist_time.scenario.values[0])
            clim = ds_clim_time[var].sel(time = slice(start_hyd_year_clim, end_hyd_year_clim)).resample(time = f"AS-{start_month_hyd_year}").mean("time")
        
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

#%%


# Snakemake options
clim_project_dir = snakemake.params.clim_project_dir
stats_nc_hist = snakemake.params.stats_nc_hist
stats_nc = snakemake.params.stats_nc
stats_time_nc_hist = snakemake.input.stats_time_nc_hist
stats_time_nc = snakemake.input.stats_time_nc
start_month_hyd_year = snakemake.params.start_month_hyd_year
name_horizon = snakemake.params.name_horizon
name_scenario = snakemake.params.name_scenario
name_model = snakemake.params.name_model
save_grids = snakemake.params.save_grids

# open datasets
ds_hist_time = xr.open_dataset(stats_time_nc_hist)
ds_clim_time = xr.open_dataset(stats_time_nc)

# Get names of grids if save_grids
if save_grids:
    #open datasets
    ds_hist = xr.open_dataset(stats_nc_hist)
    ds_clim = xr.open_dataset(stats_nc)

#get lat lon name of data
XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")
for dim in XDIMS:
    if dim in ds_hist_time.coords:
        x_dim = dim
for dim in YDIMS:
    if dim in ds_hist_time.coords:
        y_dim = dim


# #only calc statistics if netcdf is filled (for snake all the files are made, even dummy when no data)
# if len(ds_clim_time) > 0:

#     #if time format is CFTimeIndex, convert to DatetimeIndex 
#     #(NB: This may lead to subtle errors in operations that depend on the length of time between dates. However needed if we want to have the possibility to resample over hydrological years with "AS-Mon") 
#     if ds_hist_time.indexes['time'].dtype == "O":
#         datetimeindex_hist = ds_hist_time.indexes['time'].to_datetimeindex()
#         ds_hist_time['time'] = datetimeindex_hist
#         #same for clim
#         datetimeindex_clim = ds_clim_time.indexes['time'].to_datetimeindex()
#         ds_clim_time['time'] = datetimeindex_clim
   
#     #calculate change
#     monthly_change_mean_grid, monthly_change_mean_scalar = get_change_clim_projections(ds_hist, ds_clim)
    
#     #write to netcdf files
#     strings = ["monthly_change_mean_grid", "monthly_change_mean_scalar"]
#     for i, ds in enumerate([monthly_change_mean_grid, monthly_change_mean_scalar]):
#         print(f"writing netcdf files {strings[i]}")
#         dvars = ds.raster.vars        
#         name_model = ds.model.values[0]
#         name_scenario = ds.scenario.values[0]
#         name_horizon = ds.horizon.values[0]
#         name_nc_out = f"{strings[i]}-{name_model}_{name_scenario}_{name_horizon}.nc"
#         ds.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})



#%% get annual statistics from time series of monthly variables

#only calc statistics if netcdf is filled (for snake all the files are made, even dummy when no data)
#create dummy netcdf otherwise as this is the file snake is checking:

if len(ds_clim_time) > 0:

    #calculate statistics (mean, std, 0.1 0.25 0.50 0.75 0.90 quantiles of annual precip sum and mean temp)    
    stats_annual_change = get_change_annual_clim_proj(ds_hist_time, ds_clim_time)
    
    #write to netcdf files
    dvars = stats_annual_change.raster.vars        
    name_model = stats_annual_change.model.values[0]
    name_scenario = stats_annual_change.scenario.values[0]
    name_horizon = stats_annual_change.horizon.values[0]
    name_nc_out = f"annual_change_scalar_stats-{name_model}_{name_scenario}_{name_horizon}.nc"
    stats_annual_change.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})

else: #create a dummy netcdf
    name_nc_out = f"annual_change_scalar_stats-{name_model}_{name_scenario}_{name_horizon}.nc"
    ds_dummy = xr.Dataset()
    ds_dummy.to_netcdf(os.path.join(clim_project_dir, name_nc_out))

if save_grids:
    if len(ds_clim) > 0:
        #calculate change
        monthly_change_mean_grid = get_change_clim_projections(ds_hist, ds_clim)
        
        #write to netcdf files
        print(f"writing netcdf files monthly_change_mean_grid")
        dvars = monthly_change_mean_grid.raster.vars        
        name_model = monthly_change_mean_grid.model.values[0]
        name_scenario = monthly_change_mean_grid.scenario.values[0]
        name_horizon = monthly_change_mean_grid.horizon.values[0]
        name_nc_out = f"monthly_change_mean_grid-{name_model}_{name_scenario}_{name_horizon}.nc"
        monthly_change_mean_grid.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})
    else: #create a dummy netcdf
        name_nc_out = f"monthly_change_mean_grid-{name_model}_{name_scenario}_{name_horizon}.nc"
        ds_dummy = xr.Dataset()
        ds_dummy.to_netcdf(os.path.join(clim_project_dir, name_nc_out))

