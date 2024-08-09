"""Plot gridded historical anomalies to see if there is a trend."""

from os.path import join
from pathlib import Path
from typing import Union, List, Optional

import geopandas as gpd
import numpy as np
import xarray as xr
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
    data_catalog: List[Union[str, Path]] = [],
    region_filename: Optional[Union[str, Path]] = None,
    river_filename: Optional[Union[str, Path]] = None,
    year_per_line: int = 5,
    line_height_yearly_plot: int = 6,
    line_height_mean_precip: int = 6,
    fs_yearly_plot: int = 8,
    fs_mean_precip: int = 8,
    y_title_yearly_plot: float = 1.0,
):
    """
    Plot gridded historical anomalies of precip and temp for a specific region.

    If provided the region and river files will be added to the plots.

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
    data_catalog : List of str or Path, optional
        List of paths to the data catalogs to use for the plotting. Needed if the
        river filename are data catalog entries.
    region_filename : str or Path, optional
        Path to the region vector file. If provided, it will be added to the plots.
    river_filename : str or Path, optional
        Path or data catalog entry to the river vector file. If provided, it will be
        added to the plots.
    year_per_line : int, optional
        Number of years per line in the gridded anomalies plot. Default is 5.
    line_height_yearly_plot : int, optional
        Height of a tile in the yearly climate plot in cm. Default is 6.
    line_height_mean_precip : int, optional
        Height of a tile in the average annual precipitation plot in cm. Default is 6.
    fs_yearly_plot : int, optional
        Font size of the yearly climate plot. Default is 8.
    fs_mean_plot : int, optional
        Font size of the average annual precipitation plot. Default is 8.
    y_title_yearly_plot: float, optional
        Y position of the title in the subplot (between 0 and 1). Default is 1.
    """
    # Start a data catalog
    data_catalog = DataCatalog(data_catalog)

    # Read the region file
    if region_filename is not None:
        region = gpd.read_file(region_filename)
    else:
        region = None

    # Read the river file
    if river_filename is not None:
        rivers = data_catalog.get_geodataframe(river_filename, geom=region)
    else:
        rivers = None

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
        if len(ds.sel(time=ds.time.dt.year.isin(ds.time.dt.year[0]))) < 364:
            ds = ds.sel(time=~ds.time.dt.year.isin(ds.time.dt.year[0]))
        # Same for the last year
        if len(ds.sel(time=ds.time.dt.year.isin(ds.time.dt.year[-1]))) < 364:
            ds = ds.sel(time=~ds.time.dt.year.isin(ds.time.dt.year[-1]))

        # Add to dict
        if "precip" in ds:
            precip_dict[climate_source] = ds["precip"]
        if "temp" in ds:
            temp_dict[climate_source] = ds["temp"]

    # Plot the anomalies
    if len(precip_dict) > 0:
        plot_gridded_anomalies(
            clim_dict=precip_dict,
            path_output=join(path_output, "trends"),
            gdf_region=region,
            year_per_line=year_per_line,
            line_height=line_height_yearly_plot,
            fs=fs_yearly_plot,
            y_title=y_title_yearly_plot,
        )
    if len(temp_dict) > 0:
        plot_gridded_anomalies(
            clim_dict=temp_dict,
            path_output=join(path_output, "trends"),
            gdf_region=region,
            year_per_line=year_per_line,
            line_height=line_height_yearly_plot,
            fs=fs_yearly_plot,
            y_title=y_title_yearly_plot,
        )

    # Plot the gridded median yearly precipitation
    if len(precip_dict) > 0:
        plot_gridded_precip(
            precip_dict=precip_dict,
            path_output=join(path_output, "grid"),
            gdf_region=region,
            gdf_river=rivers,
            line_height=line_height_mean_precip,
            fs=fs_mean_precip,
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
            data_catalog=sm.params.data_catalog,
            region_filename=sm.input.region_fn,
            river_filename=sm.params.river_fn,
            year_per_line=sm.params.year_per_line,
            line_height_yearly_plot=sm.params.line_height_yearly_plot,
            line_height_mean_precip=sm.params.line_height_mean_precip,
            fs_yearly_plot=sm.params.fs_yearly_plot,
            fs_mean_precip=sm.params.fs_mean_precip,
            y_title_yearly_plot=sm.params.y_title_yearly_plot,
        )

    else:
        print("This script should be run from a snakemake environment")
