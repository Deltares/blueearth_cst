"""Plot gridded historical anomalies to see if there is a trend."""

from os.path import join
from pathlib import Path
from typing import Union, List

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
    region_filename: Union[str, Path],
    path_output: Union[str, Path],
    climate_catalog: Union[str, Path, List],
    climate_sources: Union[str, List[str]],
    climate_variables: List[str] = ["precip", "temp"],
    buffer: int = 2,
):
    """
    Plot gridded historical anomalies for a specific region.

    Outputs:
    * **gridded_trends.txt**: a file to indicate that the plots were created.
    * **trends**: plots of the gridded historical anomalies for each source and per
      climate variable.


    Parameters
    ----------
    region_filename : str or Path
        Path to the region boundary vector file.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    climate_catalog : str or list of str
        Path to the climate data catalog(s) to use.
    climate_sources : str or list of str
        Name of the climate data sources to plot. Should be available in
        climate_catalog.
    climate_variables : list of str, optional
        List of climate variables to plot. Default is ["precip", "temp"].
    buffer : int, optional
        Buffer to add to the region boundary to clip the climate data. Default is 2.
    """
    # Read the region file
    region = gpd.read_file(region_filename)

    # Initialize the data catalog
    data_catalog = DataCatalog(climate_catalog)

    # Initiliaze gridded precipi dict
    precip_dict = dict()

    # Open the climate data and plot anomalies
    for climate_source in climate_sources:
        ds_clim = []
        for var in climate_variables:
            # Open the climate data
            ds = data_catalog.get_rasterdataset(
                climate_source,
                bbox=region.total_bounds,
                buffer=buffer,
                handle_nodata=NoDataStrategy.IGNORE,
                variables=var,
                single_var_as_array=False,
            )

            # Check if any data intersects with region
            if ds is None or var not in ds.data_vars:
                print(f"Skipping {climate_source} as it does not intersect the region")
                continue

            # Append to the list
            ds_clim.append(ds)
            if var == "precip":
                precip_dict[climate_source] = ds[var]

        ds_clim = xr.merge(ds_clim)
        # Try clipping to region
        try:
            ds_clip = ds_clim.raster.clip_geom(region, mask=True)
            ds_clim = ds_clip
        except ValueError:
            if np.any(np.asarray(ds_clim.raster.shape) == 1):
                print(
                    f"Skipping {climate_source} as it does not contain enough cells in the "
                    "region (at least 2*2). Try increasing the buffer."
                )
                continue

        # Check the number of days in the first year in ds_clim.time
        # and remove the year if not complete
        if (
            len(ds_clim.sel(time=ds_clim.time.dt.year.isin(ds_clim.time.dt.year[0])))
            < 365
        ):
            ds_clim = ds_clim.sel(
                time=~ds_clim.time.dt.year.isin(ds_clim.time.dt.year[0])
            )
        # Same for the last year
        if (
            len(ds_clim.sel(time=ds_clim.time.dt.year.isin(ds_clim.time.dt.year[-1])))
            < 365
        ):
            ds_clim = ds_clim.sel(
                time=~ds_clim.time.dt.year.isin(ds_clim.time.dt.year[-1])
            )

        # Plot the anomalies
        plot_gridded_anomalies(
            ds=ds_clim,
            path_output=join(path_output, "trends"),
            suffix=climate_source,
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
            region_filename=sm.input.region_file,
            path_output=join(project_dir, "climate_historical", "plots"),
            climate_catalog=sm.params.data_catalog,
            climate_sources=sm.params.climate_sources,
        )

    else:
        print("This script should be run from a snakemake environment")
