"""Plot gridded historical anomalies to see if there is a trend."""

from os.path import join
from pathlib import Path
from typing import Union, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
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

def get_combined_bounds(subregions: Optional[dict], region: Optional[gpd.GeoDataFrame], buffer_km: float = 2.0):
    """
    Calculate the combined bounding box from subregions and region with a buffer.
    
    Parameters
    ----------
    subregions : dict or None
        Dictionary of GeoDataFrames with subregion boundaries
    region : gpd.GeoDataFrame or None
        GeoDataFrame with region boundary
    buffer_km : float
        Buffer distance in kilometers to add around the bounds
        
    Returns
    -------
    tuple or None
        Bounding box as (minx, miny, maxx, maxy) or None if no geometries found
    """
    all_bounds = []
    
    # Collect bounds from subregions
    if subregions is not None:
        for gdf in subregions.values():
            if len(gdf) > 0:
                all_bounds.append(gdf.total_bounds)
    
    # Collect bounds from region
    if region is not None and len(region) > 0:
        all_bounds.append(region.total_bounds)
    
    if len(all_bounds) == 0:
        return None
    
    # Calculate overall bounds
    all_bounds = np.array(all_bounds)
    minx = np.min(all_bounds[:, 0])
    miny = np.min(all_bounds[:, 1])
    maxx = np.max(all_bounds[:, 2])
    maxy = np.max(all_bounds[:, 3])
    
    # Add buffer
    # Convert buffer from km to degrees (approximate: 1 degree ≈ 111 km)
    buffer_deg = buffer_km / 111.0
    minx -= buffer_deg
    miny -= buffer_deg
    maxx += buffer_deg
    maxy += buffer_deg
    
    return (minx, miny, maxx, maxy)

def derive_gridded_trends(
    climate_filenames: List[Union[str, Path]],
    path_output: Union[str, Path],
    data_catalog: List[Union[str, Path]] = [],
    region_filename: Optional[Union[str, Path]] = None,
    river_filename: Optional[Union[str, Path]] = None,
    year_per_line: int = 5,
    fs_yearly_plot: int = 8,
    fs_mean_precip: int = 8,
    subregion_filename: Optional[Union[str, Path]] = None,
    bounds_to_subregions: bool = False,
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
    fs_yearly_plot : int, optional
        Font size of the yearly climate plot. Default is 8.
    fs_mean_plot : int, optional
        Font size of the average annual precipitation plot. Default is 8.
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
    
    if subregion_filename is not None:
        if isinstance(subregion_filename, list):
            subregions = {f"{subregion}": data_catalog.get_geodataframe(subregion) for subregion in subregion_filename}
        else:
            subregions = {f"{subregion_filename}": data_catalog.get_geodataframe(subregion_filename)}
    else:
        subregions = None
    
    # Calculate plot bounds if requested (only affects plot extent, not data clipping)
    plot_bounds = None
    if bounds_to_subregions and subregions is not None:
        plot_bounds = get_combined_bounds(subregions, region, buffer_km=2)
        if plot_bounds is not None:
            print(f"Plot bounds set to subregions and region bounds with 2 km buffer (data not clipped)")

    # Initialize gridded precip and temp dict
    precip_dict = dict()
    temp_dict = dict()

    # Open the climate data and plot anomalies
    for file in climate_filenames:
        # Open the climate data
        ds = xr.open_dataset(file, mask_and_scale=False)
        climate_source = ds["source"].values.item()

        # Clip to the region (always use full region for data clipping)
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
            plot_bounds=plot_bounds,
            year_per_line=year_per_line,
            fs=fs_yearly_plot,
        )
    if len(temp_dict) > 0:
        plot_gridded_anomalies(
            clim_dict=temp_dict,
            path_output=join(path_output, "trends"),
            gdf_region=region,
            plot_bounds=plot_bounds,
            year_per_line=year_per_line,
            fs=fs_yearly_plot,
        )

    # Plot the gridded median yearly precipitation
    if len(precip_dict) > 0:
        plot_gridded_precip(
            precip_dict=precip_dict,
            path_output=join(path_output, "grid"),
            gdf_region=region,
            gdf_river=rivers,
            plot_bounds=plot_bounds,
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
            subregion_filename=sm.params.subregion_fn,
            bounds_to_subregions=sm.params.bounds_to_subregions,
            region_filename=sm.input.region_fn,
            river_filename=sm.params.river_fn,
            year_per_line=sm.params.year_per_line,
            fs_yearly_plot=sm.params.fs_yearly_plot,
            fs_mean_precip=sm.params.fs_mean_precip,
        )

    else:
        print("This script should be run from a snakemake environment")
