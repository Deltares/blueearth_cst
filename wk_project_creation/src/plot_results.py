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
import geopandas as gpd

from func_plot_signature import plot_signatures
from func_plot_signature import plot_hydro
from func_plot_signature import plot_hydro_1y
from func_plot_signature import plot_clim


fs = 8
lw=0.8


project = snakemake.params.project
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime

output_locations = snakemake.params.output_locs
observations_timeseries = snakemake.params.observations_file

Folder_plots = f"../examples/{project}/plots"
Folder_run = f"../examples/{project}/hydrology_model" 

case = "run_default" #do we want at some point to show multiple cases? 
labels = ['Mod.']
colors = ['orange']
linestyles = ['-']
markers =   ['o']


#%%
#test outside snake - to be removed !
#starttime = "2000-01-01T00:00:00"
#endtime = "2003-12-31T00:00:00"
#output_locations = r"d:\repos\blueearth_cst\wk_project_creation\config\Gabon\output_locations_Gabon.csv"
#observations_timeseries = r"d:\repos\blueearth_cst\examples\Gabon\observations\observations_timeseries_Gabon.csv"
#run01 = pd.read_csv(r"d:\repos\blueearth_cst\examples\Gabon\hydrology_model\run_default\output.csv", index_col=0, header=0, parse_dates =True)
#project = "Gabon"
#gauges = gpd.read_file(r"d:\repos\blueearth_cst\examples\Gabon\hydrology_model\staticgeoms\gauges.geojson")

#%%

#read output_locations.csv (should include id, name, x and y coordinate)
if os.path.exists(observations_timeseries):
    df_output_locs = pd.read_csv(output_locations, header = 0)
    #make a dict of station name and station id 
    stations_dic = dict(df_output_locs[["wflow_id", "station_name"]].values)
else:
    stations_dic = dict()

#add wflow locations already present in wflow_gauges (check if id's are not duplicated from output_locations file)
gauges = gpd.read_file(os.path.join(Folder_run, "staticgeoms", "gauges.geojson"))
for gauge_id in gauges.index.values+1: #TODO check ids in gauges geojson!!
    print(gauge_id)
    if gauge_id in stations_dic:
        print("wflow id and output location id overlap")
    else:
        stations_dic[gauge_id] = f"wflow_{gauge_id}"
#stations_dic[1] = project #TODO check!
stations = list(stations_dic.keys()) #+ [1] #TODO check how to deal with id 1 of gauges from setup 

#read observation timeseries TODO check format 
#rows with variable and unit are skipped
#TODO: sep , or ; and dayfirst or not 
if os.path.exists(observations_timeseries):
    df_obs = pd.read_csv(observations_timeseries, header = 0, parse_dates = True, index_col=0, sep = ';', skiprows = [1,2])  
else:
    df_obs = pd.DataFrame()


## read csv modeled timeseries
run01 = pd.read_csv(os.path.join(Folder_run, case, "output.csv"), index_col=0, header=0, parse_dates =True)


#make dataset to combine results from several runs
variables = ['Q', 'P', 'EP', 'T']
runs = ['Obs.'] + labels
rng = pd.date_range(starttime, endtime)

S = np.zeros((len(rng), len(stations), len(runs)))
v = (('time', 'stations', 'runs'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h, 
        coords={'time': rng,
                'stations': stations,
                'runs': runs})
ds = ds * np.nan


#TODO: id 1 from gauges also in output !! 
#fill dataset with model and observed data
for id_obs_station in df_obs.columns:
    start_up = max(df_obs.index[0], rng[0])
    end_up = min(df_obs.index[-1], rng[-1])
    ds['Q'].loc[dict(runs = 'Obs.', stations = int(id_obs_station), time = df_obs.loc[start_up:end_up].index)] = df_obs[str(id_obs_station)].loc[start_up:end_up]
#add model results for all output locations
for label in labels: # not really needed as only one run, but list is needed for the plot functions. 
    ds['Q'].loc[dict(runs = label)] = run01[['Q_' + sub for sub in list(map(str,stations))]].loc[starttime:endtime]

#add P and PET for basin outlet - TODO for each id of subcatch
for gauge_id in gauges.index.values+1: #TODO check ids in gauges geojson!!
    ds['P'].loc[dict(stations = gauge_id, runs = labels[0])] = run01[f'P_{gauge_id}'].loc[starttime:endtime]
    ds['EP'].loc[dict(stations = gauge_id, runs = labels[0])] = run01[f'EP_{gauge_id}'].loc[starttime:endtime]
    ds['T'].loc[dict(stations = gauge_id, runs = labels[0])] = run01[f'T_{gauge_id}'].loc[starttime:endtime]
    
for station_id, station_name in stations_dic.items():
    print( station_id)
    #skip first year for hydro -- warm up period
    if len(ds.time) > 365:
        dsq = ds.sel(stations = station_id).sel(time = slice(f'{rng[0].year+1}-{rng[0].month}-{rng[0].day}', None))#.dropna(dim='time')
    else:
        dsq = ds.sel(stations = station_id)
    #plot hydro
    if len(np.unique(dsq['time.year'])) >= 3:
        year_min = pd.to_datetime(dsq['Q'].sel(runs = 'Mod.').idxmin().values).year
        year_max = pd.to_datetime(dsq['Q'].sel(runs = 'Mod.').idxmax().values).year
        plot_hydro(dsq, dsq.time[0], dsq.time[-1], year_max, year_min, labels, colors, Folder_plots, station_name)
        plt.close()
    else:
        plot_hydro_1y(dsq, dsq.time[0], dsq.time[-1], labels, colors, Folder_plots, station_name)
        plt.close()
    #make plot using function
    #dropna for signature calculations. 
    #skip first year for signatures -- warm up period -- if model did not run for a full year - skip signature plots 
    try:
        ds.sel(time = f'{rng[0].year+1}-{rng[0].month}-{rng[0].day}')
        dsq = ds['Q'].sel(stations = station_id).sel(time = slice(f'{rng[0].year+1}-{rng[0].month}-{rng[0].day}', None)).to_dataset().dropna(dim='time')
        if len(dsq['Q'])>0: #only plot signatures if observations are present
            plot_signatures(dsq, labels, colors, linestyles, markers, Folder_plots, station_name)
        plt.close()
    except:
        print("less than 1 year of data is available - no signature plots are made.")

for gauge_id in gauges.index.values+1:
    ds_clim = ds[['P', 'EP', 'T']].sel(stations = gauge_id, runs = labels[0])
    plot_clim(ds_clim, Folder_plots, f"wflow_{gauge_id}")
    plt.close()


#TODO add summary maps mean prec and temp spatially?

