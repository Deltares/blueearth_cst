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
# folder_model = r"d:\repos\blueearth_cst\examples\Gabon\hydrology_model"
# folder_out = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections\isimip3"
# path_yml = r"d:\repos\blueearth_cst\config\deltares_data_climate_projections.yml"
# name_scenario = "historical"
# name_realisation = "r1i1p1f1"
# name_clim_project = "isimip3"
# name_model = "ipsl"
# time_tuple = ('1991-01-01', '2014-12-31')
# time_tuple_fut = ('2071-01-01', '2100-12-31')

#%%

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")

# Snakemake options
project_dir = snakemake.params.project_dir
path_yml = snakemake.params.yml_fid
time_tuple = snakemake.params.time_horizon
time_tuple = tuple(map(str, time_tuple.split(', ')))
name_horizon = snakemake.params.name_horizon
name_scenario = snakemake.params.name_scenario
name_realisation = snakemake.params.name_realisation
name_model = snakemake.params.name_model
name_clim_project = snakemake.params.name_clim_project

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

def get_stats_clim_projections(name_clim_project, name_model, name_scenario, name_realisation, geom, time_tuple, buffer):
    """
    Parameters
    ----------
    name_clim_project : str
        name of the climate project (e.g. cmip5, cmip6, isimip3).
        should link to the name in the yml catalog. 
    name_model : str
        model name of the climate model (e.g. ipsl, gfdl).
    name_scenario : str
        scenario name of the climate model (e.g. rcp4.5, rcp8.5).
    name_realisation : str
        realisation name of the climate model (e.g. r1i1p1f1).
    geom : polygon
        model region file.
    time_tuple : tuple
        time period over which to calculate statistics.
    buffer : int
        buffer around area (in pixels).

    Returns
    -------
    Writes a netcdf file with mean monthly precipitation and temperature regime (12 maps) over the geom.
    todo: Writes a csv file with mean monthly timeseries of precip and temp statistics (mean) over the geom

    """
    data = data_catalog.get_rasterdataset(f"{name_clim_project}_{name_model}_{name_scenario}_{name_realisation}", bbox=bbox, buffer=buffer, time_tuple=time_tuple)
    
    #get lat lon name of data
    for dim in XDIMS:
        if dim in data.coords:
            x_dim = dim
    for dim in YDIMS:
        if dim in data.coords:
            y_dim = dim
    
    ds = []
    ds_scalar = []
    for var in data.data_vars:
        if var == "precip":
            var_m = data[var].resample(time='M').sum('time')
        else: #for temp
            var_m = data[var].resample(time='M').mean('time')

        #get scalar average over grid for each month
        var_m_scalar = var_m.mean([x_dim, y_dim])
        ds_scalar.append(var_m_scalar.to_dataset())
        
        #get grid average over time for each month
        var_mm = var_m.groupby('time.month').mean('time')
        ds.append(var_mm.to_dataset())
    #mean stats over grid and time
    mean_stats = xr.merge(ds)
    mean_stats_time = xr.merge(ds_scalar)
    
    #add coordinate on project, model, scenario, realization to later merge all files
    mean_stats = mean_stats.assign_coords({"clim_project":f"{name_clim_project}",
                              "model":f"{name_model}",
                              "scenario":f"{name_scenario}",
                              "horizon":f"{name_horizon}",
                              "realisation":f"{name_realisation}",
                              }).expand_dims(["clim_project", "model", "scenario", "horizon", "realisation"])
    
    #same for time scalar dataset
    mean_stats_time = mean_stats_time.assign_coords({"clim_project":f"{name_clim_project}",
                              "model":f"{name_model}",
                              "scenario":f"{name_scenario}",
                              "horizon":f"{name_horizon}",
                              "realisation":f"{name_realisation}",
                              }).expand_dims(["clim_project", "model", "scenario", "horizon", "realisation"])
    
    #use hydromt function instead to write to netcdf?
    dvars = mean_stats.raster.vars

    if name_scenario == "historical":
        name_nc_out = f"historical_stats_{name_model}.nc"
        name_nc_out_time = f"historical_stats_time_{name_model}.nc"
    else:
        name_nc_out = f"stats-{name_model}_{name_scenario}_{name_horizon}.nc"
        name_nc_out_time = f"stats_time-{name_model}_{name_scenario}_{name_horizon}.nc"

    print('writing stats over grid to nc')        
    mean_stats.to_netcdf(os.path.join(folder_out, name_nc_out), encoding={k: {"zlib": True} for k in dvars})
    print('writing stats over time to nc')
    mean_stats_time.to_netcdf(os.path.join(folder_out, name_nc_out_time), encoding={k: {"zlib": True} for k in dvars})
    

get_stats_clim_projections(name_clim_project, name_model, name_scenario, name_realisation, bbox, time_tuple, buffer)


