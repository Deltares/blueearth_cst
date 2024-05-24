from pathlib import Path
import os
import numpy as np
import hydromt
from os.path import join, dirname, basename
import xarray as xr
from typing import List, Tuple, Union

def downscale_delta_change(
        delta_change_grid_fn: Union[str, Path],
        dst_grid_fn: Union[str, Path],
        path_output: Union[str, Path] = None, 
        method: str = "nearest"
        ):
    """
    Converts the delta change percentages into fractions.
    Output is a netcdf file with the delta change factors downscaled to model resolution grid (dst_grid_fn).

    Parameters
    ----------
    delta_change_grid_fn : Union[str, Path]
        Path to the monthly percentage change (precip and pet) and absolute change (temp) 
    dst_grid_fn : Union[str, Path]
        Path to a dataset with destination resolution to resample to
    path_output : Union[str, Path]
        Path to the output directory. Default is None, which means the netcdf is saved 
        in the same directory as the delta_change_grid_fn with "_downscaled" as suffix
    method: str
        method for the resampling. Default is nearest. 

    """

    # Prepare the output filename and directory
    name_nc_out = os.path.basename(delta_change_grid_fn).split(".")[0] + "_downscaled.nc"

    if path_output is None:
        path_output = dirname(delta_change_grid_fn)

    # Create output dir (model name can contain subfolders)
    dir_output = dirname(join(path_output, name_nc_out))
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # open datasets and slice times
    delta_change_grid = xr.open_dataset(delta_change_grid_fn, lock=False)
    dst_grid = xr.open_dataset(dst_grid_fn, lock=False)

    # squeeze
    delta_change_grid = delta_change_grid.squeeze()

    # convert from percentage to fraction for pet and precip
    delta_change_grid["precip"] = 1 + delta_change_grid["precip"]/100
    delta_change_grid["pet"] = 1 + delta_change_grid["pet"]/100

    delta_change_grid_downscaled = delta_change_grid.raster.reproject_like(dst_grid, method=method)

    # write netcdf
    delta_change_grid_downscaled.to_netcdf(
        join(path_output, name_nc_out),
        encoding={k: {"zlib": True} for k in delta_change_grid_downscaled.data_vars},
    )

    return delta_change_grid_downscaled


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        downscale_delta_change(
            delta_change_grid_fn=sm.input.monthly_change_mean_grid,
            dst_grid_fn=sm.input.staticmaps_fid,
        )
    else:
        downscale_delta_change(
            delta_change_grid_fn="delta_change.nc",
            dst_grid_fn="staticmaps.nc",
        )


