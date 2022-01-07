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


from func_plot_signature import plot_signatures
from func_plot_signature import plot_hydro

fs = 8
lw=0.8

import pdb; pdb.set_trace()

project = snakemake.params.project
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime

output_locations = snakemake.input.output_locs
observations_timeseries = snakemake.input.observations_file

Folder_plots = f"../examples/{project}/plots"
Folder_run = f"../examples/{project}/hydrology_model" 

case = "run_default" #do we want at some point to show multiple cases? 
label = 'Mod.'
color = 'orange'

#%%
#test outside snake - to be removed !
#starttime = "2000-01-01T00:00:00"
#endtime = "2000-01-31T00:00:00"
#output_locations = r"d:\repos\blueearth_cst\wk_project_creation\config\Gabon\output_locations_Gabon.csv"
#observations_timeseries = r"d:\repos\blueearth_cst\examples\Gabon\observations\observations_timeseries_Gabon.csv"
#run01 = pd.read_csv(r"d:\repos\blueearth_cst\examples\Gabon\hydrology_model\run_default\output.csv", index_col=0, header=0, parse_dates =True)

#%%

#read output_locations.csv (should include id, name, x and y coordinate)
df_output_locs = pd.read_csv(output_locations, header = 0)
#make a dict of station name and station id 
stations_dic = dict(df_output_locs[["wflow_id", "station_name"]].values)
stations = list(stations_dic.keys()) + [1] #TODO check how to deal with id 1 of gauges from setup 

#read observation timeseries TODO check format 
#rows with variable and unit are skipped
#TODO: sep , or ; and dayfirst or not 
df_obs = pd.read_csv(observations_timeseries, header = 0, parse_dates = True, index_col=0, sep = ';', skiprows = [1,2])  


## read csv modeled timeseries
run01 = pd.read_csv(os.path.join(Folder_run, case, "output.csv"), index_col=0, header=0, parse_dates =True)


#make dataset to combine results from several runs
variables = ['Q'] # TODO check if we also would like to plot forcing 'P', 'EP']
runs = ['Obs.'] + [label]
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

start_up = max(df_obs.index[0], rng[0])
end_up = min(df_obs.index[-1], rng[-1])

#TODO: id 1 from gauges also in output !! 
#fill dataset with model and observed data
for id_obs_station in df_obs.columns:
     ds['Q'].loc[dict(runs = 'Obs.', stations = int(id_obs_station), time = df_obs.loc[start_up:end_up].index)] = df_obs[id_obs_station].loc[start_up:end_up]
#add model results for all output locations
ds['Q'].loc[dict(runs = label)] = run01[['Q_' + sub for sub in list(map(str,stations))]].loc[starttime:endtime]

#TODO adapt functions to make plots per output location 



