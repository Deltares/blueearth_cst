from pathlib import Path
import os
import hydromt
from os.path import join, dirname
import xarray as xr
from typing import Union


def downscale_delta_change(
    delta_change_grid_fn: Union[str, Path],
    dst_grid_fn: Union[str, Path],
    path_output: Union[str, Path] = None,
    method: str = "nearest",
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
    name_nc_out = (
        os.path.basename(delta_change_grid_fn).split(".")[0] + "_downscaled.nc"
    )

    if path_output is None:
        path_output = dirname(delta_change_grid_fn)

    # Create output dir (model name can contain subfolders)
    dir_output = dirname(join(path_output, name_nc_out))
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # open datasets
    delta_change_grid = xr.open_dataset(delta_change_grid_fn, lock=False)
    # Open dst_grid similar to WflowModel.read_grid
    dst_grid = xr.open_dataset(
        dst_grid_fn, mask_and_scale=False, decode_coords="all"
    ).load()
    dst_grid.close()
    # make sure maps are always North -> South oriented for hydromt
    if dst_grid.raster.res[1] > 0:
        dst_grid = dst_grid.raster.flipud()

    # squeeze
    delta_change_grid = delta_change_grid.squeeze()

    # convert from percentage to fraction for variables that are not temperature
    for var in delta_change_grid.data_vars:
        if not var.startswith("temp"):
            delta_change_grid[var] = 1 + delta_change_grid[var] / 100

    delta_change_grid_downscaled = delta_change_grid.raster.reproject_like(
        dst_grid, method=method
    )

    # rename from month to time for wflow
    if "month" in delta_change_grid_downscaled.coords:
        delta_change_grid_downscaled = delta_change_grid_downscaled.rename(
            {"month": "time"}
        )

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
