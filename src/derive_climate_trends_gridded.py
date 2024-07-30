"""Plot gridded historical anomalies to see if there is a trend."""

from os.path import join
from pathlib import Path
from typing import Union, List, Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt.nodata import NoDataStrategy
from hydromt import DataCatalog

import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_anomalies import plot_gridded_anomalies
    from plot_utils.plot_gridded_climate import plot_gridded_precip
else:
    from .plot_utils.plot_anomalies import plot_gridded_anomalies
    from .plot_utils.plot_gridded_climate import plot_gridded_precip


def derive_gridded_trends(
    climate_filenames: List[Union[str, Path]],
    path_output: Union[str, Path],
    region_filename: Optional[Union[str, Path]] = None,
):
    """
    Plot gridded historical anomalies of precip and temp for a specific region.

    Outputs:
    * **gridded_trends.txt**: a file to indicate that the plots were created.
    * **trends**: plots of the gridded historical anomalies for each source and per
      climate variable.


    Parameters
    ----------
    climate_filenames : List of str or Path
        Path to the gridded files extracted for a specific domain. They
        should contain the climate ``source`` in the coords or dims.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    region_filename : str or Path, optional
        Path to the region vector file. If provided, it will be added to the plots.
    """
    # Read the region file
    if region_filename is not None:
        region = gpd.read_file(region_filename)
    else:
        region = None

    # Initialize gridded precip and temp dict
    precip_dict = dict()
    temp_dict = dict()

    # Open the climate data and plot anomalies
    for file in climate_filenames:
        # Open the climate data
        ds = xr.open_dataset(file, mask_and_scale=False)
        climate_source = ds["source"].values.item()

        # Clip to the region
        if region is not None:
            ds = ds.raster.clip_geom(region, buffer=2, mask=False)
            ds = ds.assign_coords(
                mask=ds.raster.geometry_mask(region, all_touched=True)
            )
            ds = ds.raster.mask(ds.coords["mask"])

        # Check the number of days in the first year in ds_clim.time
        # and remove the year if not complete
        if len(ds.sel(time=ds.time.dt.year.isin(ds.time.dt.year[0]))) < 365:
            ds = ds.sel(time=~ds.time.dt.year.isin(ds.time.dt.year[0]))
        # Same for the last year
        if len(ds.sel(time=ds.time.dt.year.isin(ds.time.dt.year[-1]))) < 365:
            ds = ds.sel(time=~ds.time.dt.year.isin(ds.time.dt.year[-1]))

        # Add to dict
        if "precip" in ds:
            precip_dict[climate_source] = ds["precip"]
        if "temp" in ds:
            temp_dict[climate_source] = ds["temp"]

    # Plot the anomalies
    plot_gridded_anomalies(
        clim_dict=precip_dict,
        path_output=join(path_output, "trends"),
        gdf_region=region,
    )
    plot_gridded_anomalies(
        clim_dict=temp_dict,
        path_output=join(path_output, "trends"),
        gdf_region=region,
    )

    # Plot the gridded median yearly precipitation
    plot_gridded_precip(
        precip_dict=precip_dict,
        path_output=join(path_output, "grid"),
        gdf_region=region,
    )

    if "snakemake" in globals():
        # Write a file when everything is done for snakemake tracking
        text_out = join(path_output, "trends", "gridded_trends.txt")
        with open(text_out, "w") as f:
            f.write("Gridded anomalies plots were made.\n")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir

        derive_gridded_trends(
            climate_filenames=sm.input.grid,
            path_output=join(project_dir, "plots", "climate_historical"),
            region_filename=sm.input.region_fn,
        )

    else:
        print("This script should be run from a snakemake environment")
