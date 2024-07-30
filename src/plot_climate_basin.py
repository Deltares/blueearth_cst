"""Plot historical climate for a region and optionally subregions"""

from os.path import join, dirname
from pathlib import Path
from typing import Union, Optional, List

import geopandas as gpd
import xarray as xr

import hydromt

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_scalar_climate import plot_scalar_climate_statistics
else:
    from .plot_utils.plot_scalar_climate import plot_scalar_climate_statistics


def plot_historical_climate_region(
    climate_filenames: List[Union[str, Path]],
    path_output: Union[str, Path],
    climate_sources: Union[str, List[str]],
    climate_sources_colors: Optional[Union[str, List[str]]] = None,
    climate_variables: List[str] = ["precip", "temp"],
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25,
    add_inset_map: bool = False,
    region_filename: Optional[Union[str, Path]] = None,
):
    """Plot historical climate for a region and optionally subregions.

    Outputs:
    * plots of the historical climate for the region.

    The function plots the following statistics for each location in `climate_filename`:

    - Precipitation:
        1. Precipitation per year.
        2. Cumulative precipitation.
        3. Monthly mean precipitation.
        4. Rainfall peaks > ``precip_peak_treshold`` mm/day.
        5. Number of dry days (dailyP < ``dry_days_threshold``mm) per year.
        6. Weekly plot for the wettest year.
    - Temperature:
        1. Monthly mean temperature.
        2. Long-term average temperature per month.
        3. Number of frost days per year.
        4. Number of heat days per year (temp > ``heat_threshold`` $\degree$C).

    Parameters
    ----------
    climate_filenames : List of str or Path
        Path to the timeseries geodataset files extracted for specific regions. They
        should contain the climate ``source`` in the coords or dims.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    climate_sources : str or list of str
        Name of the climate data sources to plot. Should be available in
        the coords or dims of the climate_filenames.
    climate_sources_colors : str or list of str, optional
        Color to use for each ``climate_sources``. If None, unique color per source is
        not assured.
    climate_variables : list of str, optional
        List of climate variables to sample/plot.
    precip_peak_threshold : float, optional
        Threshold for the peak precipitation in mm/day to define the rainfall peaks.
        By default 40 mm/day.
    dry_days_threshold : float, optional
        Threshold for the precipitation in mm/day to define a dry day. By default 0.2
        mm/day.
    heat_threshold : float, optional
        Threshold for the daily mean temperature in degrees Celsius to define a
        heatwave. By default 25 degrees Celsius.
    add_inset_map : bool, optional
        Add an inset map to the plots. By default True.
    region_filename : str or Path, optional
        Path to a region vector file for the inset map.
    """

    # Read the different geodataset file and merge them
    geods_list = []
    for climate_file in climate_filenames:
        geods = hydromt.vector.GeoDataset.from_netcdf(climate_file)
        geods_list.append(geods)

    geods_region = xr.concat(geods_list, dim="source")

    # Prepare colors dict
    if climate_sources_colors is not None:
        colors = {k: v for k, v in zip(climate_sources, climate_sources_colors)}
    else:
        colors = None

    # Read region if provided
    if region_filename is not None:
        region = gpd.read_file(region_filename)
    else:
        region = None

    # Plot historical climate for regions in climate_filename
    geods_region = plot_scalar_climate_statistics(
        geods=geods_region,
        path_output=path_output,
        climate_variables=climate_variables,
        colors=colors,
        precip_peak_threshold=precip_peak_threshold,
        dry_days_threshold=dry_days_threshold,
        heat_threshold=heat_threshold,
        gdf_region=region,
        add_map=add_inset_map,
    )

    if "snakemake" in globals():
        # Write a file when everything is done for snakemake tracking
        text_out = join(path_output, "basin_climate.txt")
        with open(text_out, "w") as f:
            f.write("Timeseries plots per region were made.\n")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir

        plot_historical_climate_region(
            climate_filename=sm.input.basin_climate,
            path_output=dirname(sm.output.basin_plot_done),
            climate_sources=sm.params.climate_sources,
            climate_sources_colors=sm.params.climate_sources_colors,
            precip_peak_threshold=sm.params.precip_peak,
            dry_days_threshold=sm.params.precip_dry,
            heat_threshold=sm.params.temp_heat,
        )

    else:
        print("This script should be run from a snakemake environment")
