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

from dask.diagnostics import ProgressBar

#%%

# Snakemake options
project_dir = snakemake.params.project_dir
path_yml = snakemake.params.yml_fid
time_tuple = snakemake.params.time_horizon
time_tuple = tuple(map(str, time_tuple.split(', ')))
name_horizon = snakemake.params.name_horizon
name_scenario = snakemake.params.name_scenario
name_members = snakemake.params.name_members
name_model = snakemake.params.name_model
name_clim_project = snakemake.params.name_clim_project
variables = snakemake.params.variables

#additional folder structure info
folder_model = os.path.join(project_dir, "hydrology_model")
folder_out = os.path.join(project_dir, "climate_projections", name_clim_project)

if not os.path.exists(folder_out):
    os.mkdir(folder_out)

# initialize model and region properties
mod = hydromt.WflowModel(root=folder_model, mode='r')
bbox = list(mod.region.geometry.bounds.values[0])
geom = mod.region.geometry
buffer = 1

#initialize data_catalog from yml file
data_catalog = hydromt.DataCatalog(data_libs=path_yml)

def get_stats_clim_projections(data, name_clim_project, name_model, name_scenario, name_member, time_tuple):
    """
    Parameters
    ----------
    data: dataset
        dataset for all available variables after opening data catalog
    name_clim_project : str
        name of the climate project (e.g. cmip5, cmip6, isimip3).
        should link to the name in the yml catalog. 
    name_model : str
        model name of the climate model (e.g. ipsl, gfdl).
    name_scenario : str
        scenario name of the climate model (e.g. rcp4.5, rcp8.5).
    name_member : str
        member name of the climate model (e.g. r1i1p1f1).
    time_tuple : tuple
        time period over which to calculate statistics.

    Returns
    -------
    Writes a netcdf file with mean monthly precipitation and temperature regime (12 maps) over the geom.
    todo: Writes a csv file with mean monthly timeseries of precip and temp statistics (mean) over the geom

    """
    
    #get lat lon name of data
    XDIMS = ("x", "longitude", "lon", "long")
    YDIMS = ("y", "latitude", "lat")
    for dim in XDIMS:
        if dim in data.coords:
            x_dim = dim
    for dim in YDIMS:
        if dim in data.coords:
            y_dim = dim

    #ds = []
    ds_scalar = []
    #filter variables for precip and temp 
    # data_vars = list(data.data_vars)
    # var_list = [str for str in data_vars if any(sub in str for sub in variables)]
    for var in data.data_vars: #var_list: 
        if var == "precip":
            var_m = data[var].resample(time='MS').sum('time')
        else: #for temp
        # elif "temp" in var: #for temp
            var_m = data[var].resample(time='MS').mean('time')

        #get scalar average over grid for each month
        var_m = var_m.sel(time = slice(*time_tuple)) #needed for cmip5 cftime.Datetime360Day which is not picked up before.
        var_m_scalar = var_m.mean([x_dim, y_dim])
        ds_scalar.append(var_m_scalar.to_dataset())
        
        #get grid average over time for each month
        #var_mm = var_m.groupby('time.month').mean('time')
        #ds.append(var_mm.to_dataset())
    #mean stats over grid and time
    #mean_stats = xr.merge(ds)
    mean_stats_time = xr.merge(ds_scalar)
    
    # #add coordinate on project, model, scenario, realization to later merge all files
    # mean_stats = mean_stats.assign_coords({"clim_project":f"{name_clim_project}",
    #                           "model":f"{name_model}",
    #                           "scenario":f"{name_scenario}",
    #                           "horizon":f"{name_horizon}",
    #                           "member":f"{name_member}",
    #                           }).expand_dims(["clim_project", "model", "scenario", "horizon", "member"])
    
    #same for time scalar dataset
    mean_stats_time = mean_stats_time.assign_coords({"clim_project":f"{name_clim_project}",
                              "model":f"{name_model}",
                              "scenario":f"{name_scenario}",
                              "horizon":f"{name_horizon}",
                              "member":f"{name_member}",
                              }).expand_dims(["clim_project", "model", "scenario", "horizon", "member"])
    
    #return mean_stats, mean_stats_time
    return mean_stats_time


    
#check if model really exists from data catalog entry - else skip and provide empty ds?? 

#ds_members_mean_stats = []
ds_members_mean_stats_time = []

for name_member in name_members:
    print(name_member)
    entry = f"{name_clim_project}_{name_model}_{name_scenario}_{name_member}"
    if entry in data_catalog:
            
        try: #todo can this be replaced by if statement?
            data = data_catalog.get_rasterdataset(entry, bbox=bbox, buffer=buffer, time_tuple=time_tuple) 
        except:
            #if it is not possible to open all variables at once, loop over each one, remove duplicates and then merge:
            ds_list = []
            for var in variables:
                try:
                    data_ = data_catalog.get_rasterdataset(entry, bbox=bbox, buffer=buffer, time_tuple=time_tuple, variables = [var])
                    #drop duplicates if any
                    data_ = data_.drop_duplicates(dim = 'time', keep = 'first')
                    ds_list.append(data_)    
                except:
                    print(f"{name_scenario}", f"{name_model}", f"{var} not found")
            #merge all variables back to data
            data = xr.merge(ds_list)               
        
        #calculate statistics
        #mean_stats, mean_stats_time = get_stats_clim_projections(data, name_clim_project, name_model, name_scenario, name_member, time_tuple)
        mean_stats_time = get_stats_clim_projections(data, name_clim_project, name_model, name_scenario, name_member, time_tuple)

    else:
        #mean_stats = xr.Dataset()
        mean_stats_time = xr.Dataset()
        
    #merge members results
    #ds_members_mean_stats.append(mean_stats)
    ds_members_mean_stats_time.append(mean_stats_time)
      
        
#nc_mean_stats =xr.merge(ds_members_mean_stats)
nc_mean_stats_time = xr.merge(ds_members_mean_stats_time)
        
    

#write netcdf:
    
#use hydromt function instead to write to netcdf?
dvars = nc_mean_stats_time.raster.vars

if name_scenario == "historical":
    #name_nc_out = f"historical_stats_{name_model}.nc"
    name_nc_out_time = f"historical_stats_time_{name_model}.nc"
else:
    #name_nc_out = f"stats-{name_model}_{name_scenario}_{name_horizon}.nc"
    name_nc_out_time = f"stats_time-{name_model}_{name_scenario}_{name_horizon}.nc"

#print('writing stats over grid to nc')        
#delayed_obj = nc_mean_stats.to_netcdf(os.path.join(folder_out, name_nc_out), encoding={k: {"zlib": True} for k in dvars}, compute=False)
#with ProgressBar():
#    delayed_obj.compute()

print('writing stats over time to nc')
delayed_obj = nc_mean_stats_time.to_netcdf(os.path.join(folder_out, name_nc_out_time), encoding={k: {"zlib": True} for k in dvars}, compute=False)
with ProgressBar():
    delayed_obj.compute()



#%%
# def drop_duplicates(ds):
#     ds = ds.sel(time=~ds.indexes['time'].duplicated())
#     return ds