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
from matplotlib import cm, colors
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np

#%%

# Snakemake options
clim_project_dir = snakemake.params.clim_project_dir
stats_time_nc_hist = snakemake.input.stats_time_nc_hist
stats_time_nc = snakemake.input.stats_time_nc
rcps = snakemake.params.scenarios
horizons = snakemake.params.horizons
save_grids = snakemake.params.save_grids
change_grids_nc = snakemake.params.change_grids


#%% Historical
print("Opening historical gcm timeseries")
# open historical datasets
def todatetimeindex_dropvars(ds):
    if "time" in ds.coords:
        if ds.indexes["time"].dtype == "O":
            ds["time"] = ds.indexes["time"].to_datetimeindex()
    if 'spatial_ref' in ds.coords:
        ds = ds.drop_vars('spatial_ref')
    return ds

fns_hist = stats_time_nc_hist.copy()
for fn in stats_time_nc_hist:
    ds = xr.open_dataset(fn)
    if len(ds) == 0 or ds is None:
        fns_hist.remove(fn)
ds_hist = xr.open_mfdataset(fns_hist, preprocess=todatetimeindex_dropvars)

# convert to df and compute anomalies
print("Computing historical gcm timeseries anomalies")
#precip
gcm_pr = ds_hist['precip'].squeeze(drop=True).transpose().to_pandas()
gcm_pr_annmn = gcm_pr.resample('A').mean()
gcm_pr_ref = gcm_pr_annmn.mean()
gcm_pr_anom = (gcm_pr_annmn-gcm_pr_ref)/gcm_pr_ref*100
q_pr_anom = gcm_pr_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose()

# temp
gcm_tas = ds_hist['temp'].squeeze(drop=True).transpose().to_pandas()
gcm_tas_annmn = gcm_tas.resample('A').mean()
gcm_tas_ref = gcm_tas_annmn.mean()
gcm_tas_anom = (gcm_tas_annmn-gcm_tas_ref)/gcm_tas_ref*100
q_tas_anom = gcm_tas_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose()


#%% Future
# remove files containing empty dataset
fns_future = stats_time_nc.copy()
for fn in stats_time_nc:
    ds = xr.open_dataset(fn)
    if len(ds) == 0 or ds is None:
        fns_future.remove(fn)

# Initialise list of future df per rcp
pr_fut = []
tas_fut = []
anom_pr_fut = []
anom_tas_fut = []
qpr_fut = []
qtas_fut = []
for i in range(len(rcps)):
    pr_fut.append([])
    tas_fut.append([])
    anom_pr_fut.append([])
    anom_tas_fut.append([])
    qpr_fut.append([])
    qtas_fut.append([])
# read files
for i in range(len(rcps)):
    print(f"Opening future gcm timeseries for rcp {rcps[i]}")
    fns_rcp = [fn for fn in fns_future if rcps[i] in fn]
    ds_rcp = xr.open_mfdataset(fns_rcp, preprocess=todatetimeindex_dropvars)
    ds_rcp_pr = ds_rcp['precip'].squeeze(drop=True)
    ds_rcp_tas = ds_rcp['temp'].squeeze(drop=True)
    if len(ds_rcp.horizon) > 1:
        hz = ds_rcp.horizon
        ds_rcp_pr = xr.merge([ds_rcp_pr.sel({"horizon":hz[0]}, drop=True), ds_rcp_pr.sel({"horizon":hz[1]}, drop=True)])
        ds_rcp_pr = ds_rcp_pr["precip"]
        ds_rcp_tas = xr.merge([ds_rcp_tas.sel({"horizon":hz[0]}, drop=True), ds_rcp_tas.sel({"horizon":hz[1]}, drop=True)])
        ds_rcp_tas = ds_rcp_tas["temp"]
    # to dataframe
    pr_fut[i] = ds_rcp_pr.transpose().to_pandas()
    tas_fut[i] = ds_rcp_tas.transpose().to_pandas()

#compute anomalies
print("Computing future gcm timeseries anomalies")
fut_pr_ref = gcm_pr_annmn.mean()
fut_tas_ref = gcm_tas_annmn.mean()

for i in range(len(anom_pr_fut)): 
    anom_pr_fut[i] = (pr_fut[i].resample('A').mean()-fut_pr_ref)/fut_pr_ref*100
    qpr_fut[i] = anom_pr_fut[i].quantile([0.05, 0.5, 0.95], axis=1).transpose()
    
    anom_tas_fut[i] = (tas_fut[i].resample('A').mean()-fut_tas_ref)/fut_tas_ref*100
    qtas_fut[i] = anom_tas_fut[i].quantile([0.05, 0.5, 0.95], axis=1).transpose()

#%% Plots
if not os.path.exists(os.path.join(clim_project_dir, "plots")):
    os.mkdir(os.path.join(clim_project_dir, "plots"))
# precip anomaly
plt.figure(figsize=(8,6))
plt.title('Average annual precipitation anomaly')
plt.fill_between(x=q_pr_anom.index, y1=q_pr_anom[0.95], y2=q_pr_anom[0.05], color='lightgrey', alpha=0.5);
plt.plot(q_pr_anom[0.5].index, q_pr_anom[0.5], color='darkgrey', label='multi-model median')
for i in range(len(qpr_fut)): 
    plt.fill_between(x=qpr_fut[i].index, y1=qpr_fut[i].iloc[:,2], y2=qpr_fut[i].iloc[:,0], alpha=0.5);
    plt.plot(qpr_fut[i].index, qpr_fut[i].iloc[:,1], label=rcps[i]+' multi-model median');
plt.ylabel('Anomaly (%)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(clim_project_dir, "plots", "precipitation_anomaly_projections"), dpi=300, bbox_inches="tight")

# temp anomaly
plt.figure(figsize=(8,6))
plt.title('Average annual temperature anomaly')
plt.fill_between(x=q_tas_anom.index, y1=q_tas_anom[0.95], y2=q_tas_anom[0.05], color='lightgrey', alpha=0.5);
plt.plot(q_tas_anom[0.5].index, q_tas_anom[0.5], color='darkgrey', label='multi-model median')
for i in range(len(qtas_fut)): 
    plt.fill_between(x=qtas_fut[i].index, y1=qtas_fut[i].iloc[:,2], y2=qtas_fut[i].iloc[:,0], alpha=0.5);
    plt.plot(qtas_fut[i].index, qtas_fut[i].iloc[:,1], label=rcps[i]+' multi-model median');
plt.ylabel('Anomaly (%)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(clim_project_dir, "plots", "temperature_anomaly_projections.png"), dpi=300, bbox_inches="tight")

#%%
# Map plots of gridded change per scenario / horizon
if save_grids:
    fns_grids = change_grids_nc.copy()
    for fn in change_grids_nc:
        ds = xr.open_dataset(fn)
        if len(ds) == 0 or ds is None:
            fns_grids.remove(fn)
    
    # Loop over rcp and horizon
    for rcp in rcps:
        for hz in horizons:
            print(f"Preparing change map plots for {rcp} and horizon {hz}")
            fns_rcp_hz = [fn for fn in fns_grids if rcp in fn and hz in fn]
            ds_rcp_hz = []
            for fn in fns_rcp_hz:
                ds = xr.open_dataset(fn)
                if "time" in ds.coords:
                    if ds.indexes["time"].dtype == "O":
                        ds["time"] = ds.indexes["time"].to_datetimeindex()
                ds_rcp_hz.append(ds)
            ds_rcp_hz = xr.merge(ds_rcp_hz)           
            ds_rcp_hz_med = ds_rcp_hz.median(dim='model').squeeze(drop=True)

            # Facetplots
            # precip
            plt.figure(0)
            pr = ds_rcp_hz_med["precip"]
            pr.attrs.update(long_name='Precipitation Change (median over GCMs)', units='%')
            g = pr.plot(x="lon", y="lat", col="month", col_wrap=3)
            g.set_axis_labels("longitude [degree east]", "latitude [degree north]")
            plt.savefig(os.path.join(clim_project_dir, "plots",f"gridded_monthly_precipitation_change_{rcp}_{hz}-future-horizon.png"))
            #temp
            plt.figure(1)
            tas = ds_rcp_hz_med["temp"]
            tas.attrs.update(long_name='Temperature Change (median over GCMs)', units='degC')
            g = tas.plot(x="lon", y="lat", col="month", col_wrap=3)
            g.set_axis_labels("longitude [degree east]", "latitude [degree north]")
            plt.savefig(os.path.join(clim_project_dir, "plots",f"gridded_monthly_temperature_change_{rcp}_{hz}-future-horizon.png"))

            # Average maps
            grids = ds_rcp_hz_med.mean(dim='month')
            plt.style.use("seaborn-whitegrid")  # set nice style
            # we assume the model maps are in the geographic CRS EPSG:4326
            proj = ccrs.PlateCarree()
            # adjust zoomlevel and figure size to your basis size & aspect
            zoom_level = 8
            figsize = (10, 8)
            
            # precip
            pr = grids['precip']
            #minmax = max(abs(np.amin(pr.values)), np.amax(pr.values))
            #divnorm=colors.TwoSlopeNorm(vmin=-minmax, vcenter=0., vmax=minmax)

            # initialize image with geoaxes
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection=proj)
            extent = np.array(pr.raster.box.buffer(0.5).total_bounds)[[0, 2, 1, 3]]
            ax.set_extent(extent, crs=proj)
            # add sat background image
            ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

            #plot da variables. 
            pr.plot(
                transform=proj, ax=ax, zorder=1, cbar_kwargs=dict(aspect=30, shrink=0.8, label='Precipitation Change (median over GCMs) [%]'), cmap="bwr",)# norm=divnorm) # **kwargs)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            ax.set_ylabel(f"latitude [degree north]")
            ax.set_xlabel(f"longitude [degree east]")
            _ = ax.set_title(f"Annual mean precipitation change for {rcp} and time horizon {hz}")
            plt.savefig(os.path.join(clim_project_dir, "plots",f"gridded_precipitation_change_{rcp}_{hz}-future-horizon.png"), dpi=300, bbox_inches="tight")

            # temp
            tas = grids['temp']
            minmax = max(abs(np.amin(tas.values)), np.amax(tas.values))
            divnorm=colors.TwoSlopeNorm(vmin=-minmax, vcenter=0., vmax=minmax)

            # initialize image with geoaxes
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection=proj)
            extent = np.array(tas.raster.box.buffer(0.5).total_bounds)[[0, 2, 1, 3]]
            ax.set_extent(extent, crs=proj)
            # add sat background image
            ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

            #plot da variables. 
            tas.plot(
                transform=proj, ax=ax, zorder=1, cbar_kwargs=dict(aspect=30, shrink=0.8, label='Temperature Change (median over GCMs) [degC]'), cmap="bwr", norm=divnorm) # **kwargs)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            ax.set_ylabel(f"latitude [degree north]")
            ax.set_xlabel(f"longitude [degree east]")
            _ = ax.set_title(f"Annual mean temperature change for {rcp} and time horizon {hz}")
            plt.savefig(os.path.join(clim_project_dir, "plots",f"gridded_temperature_change_{rcp}_{hz}-future-horizon.png"), dpi=300, bbox_inches="tight")

