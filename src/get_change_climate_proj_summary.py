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
# folder_out = r"d:\repos\blueearth_cst\examples\Gabon\climate_projections"

#%%

# Snakemake options
project_dir = snakemake.params.project_dir

#additional folder structure info
folder_out = os.path.join(project_dir, "climate_projections")

#merge summary maps across models, scnearios and horizons. 
prefixes = ["mean_change", "mean_change_annual", "mean_change_scalar", "mean_change_annual_scalar"]
for prefix in prefixes:
    print(f"merging netcdf files {prefix}")
    #open annual scalar summary and merge
    ds = xr.open_mfdataset(os.path.join(folder_out, f"{prefix}-*.nc"))
    dvars = ds.raster.vars        
    name_nc_out = f"{prefix}_summary.nc"
    ds.to_netcdf(os.path.join(folder_out, name_nc_out), encoding={k: {"zlib": True} for k in dvars})
