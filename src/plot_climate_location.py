"""Plot historical climate for specific locations"""

import os
from os.path import join, dirname, isfile
from pathlib import Path
from typing import Union, Optional, List

import geopandas as gpd
import numpy as np
import xarray as xr

import hydromt
from hydromt import DataCatalog

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_scalar_climate import plot_scalar_climate_statistics
else:
    from .plot_utils.plot_scalar_climate import plot_scalar_climate_statistics


def plot_historical_climate_point(
    climate_filenames: List[Union[str, Path]],
    path_output: Union[str, Path],
    climate_sources: Union[str, List[str]],
    climate_sources_colors: Optional[Union[str, List[str]]] = None,
    data_catalog: Union[str, Path, List] = [],
    locations_filename: Union[str, Path] = None,
    precip_observations_filename: Optional[Union[str, Path]] = None,
    climate_variables: List[str] = ["precip", "temp"],
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25,
    add_inset_map: bool = False,
    region_filename: Optional[Union[str, Path]] = None,
    export_observations: bool = True,
):
    """Plot historical climate for specific point locations.

    If observations are available, they will also be plotted.

    Outputs:
    * plots of the historical climate for the point locations.
    * netcdf geodataset of the observed data if provided. The file will be saved in the
      same directory as the climate_filenames with name "point_observed.nc".

    The function plots the following statistics for each location in
    `climate_filename`:

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
        Path to the timeseries geodataset files extracted for specific locations. They
        should contain the climate ``source`` in the coords or dims.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    climate_sources : str or list of str
        Name of the climate data sources to plot. Should be available in
        the coords or dims of the climate_filenames.
    climate_sources_colors : str or list of str, optional
        Color to use for each ``climate_sources``. If None, unique color per source is
        not assured.
    data_catalog : str or list of str
        Path to the data catalog(s) to use. Useful mainly to read observations if any.
    locations_filename : str or Path
        Path or data catalog source of the point locations file. If the observed
        timeseries data are provided in ``precip_observations_filename``, the index
        should match the column names in the timeseries file.
        Optional variables: "name" for the location name and "elevtn" for the elevation
        of the location.
    precip_observations_filename: str or Path, optional
        Path or data catalog source to the observed precipitation timeseries data file
        for the locations in ``locations_filename``.
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
    export_observations : bool, optional
        Export the observed data to a netcdf file with name "point_observed.nc" in the
        same folder as climate_filenames. By default True.
    """
    # Create dirs
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Small function to set the index of the geodataset
    def _update_geods_index(geods, prefix="location", legend_column="name"):
        # Update the index
        index_dim = geods.vector.index_dim
        if legend_column in geods:
            new_indexes = geods[legend_column].values
        else:
            new_indexes = [f"{prefix}_{i}" for i in geods[index_dim].values]
        geods[index_dim] = new_indexes
        geods = geods.rename({index_dim: "index"})

        return geods

    # Read the different geodataset file and merge them
    geods_list = []
    for climate_file in climate_filenames:
        geods = hydromt.vector.GeoDataset.from_netcdf(climate_file, lock=False)
        geods_list.append(geods)

    geods_locs = xr.concat(geods_list, dim="source")

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

    # Load the climate data catalog
    data_catalog = DataCatalog(data_catalog)

    # location vector + timeseries dataset
    # If the locs are a direct file without a crs property, assume 4326
    if isfile(locations_filename):
        crs = 4326
    else:
        crs = None
    locations = data_catalog.get_geodataframe(
        locations_filename,
        crs=crs,
        bbox=geods_locs.vector.bounds,
        buffer=1000,
    )

    if precip_observations_filename is not None:
        # Load the timeseries data
        if isfile(precip_observations_filename):
            # Direct dataframe file
            ds_obs = hydromt.io.open_timeseries_from_table(
                precip_observations_filename,
                name="precip",
                index_dim=locations.index.name,
            )
            # Sel dates
            ds_obs = ds_obs.sel(time=slice(geods_locs.time[0], geods_locs.time[-1]))
        else:  # dataset data catalog entry
            ds_obs = data_catalog.get_dataset(
                precip_observations_filename,
                time_tuple=(geods_locs.time[0], geods_locs.time[-1]),
            )
        # Convert to Geodataset
        geods_obs = hydromt.vector.GeoDataset.from_gdf(
            locations,
            data_vars=ds_obs,
            index_dim=locations.index.name,
            merge_index="inner",
        )
        geods_obs = _update_geods_index(
            geods_obs, prefix="location", legend_column="name"
        )
    else:
        geods_obs = None

    # Plot historical climate for region and optionally subregions
    geods_locs = plot_scalar_climate_statistics(
        geods=geods_locs,
        path_output=join(path_output),
        geods_obs=geods_obs,
        climate_variables=climate_variables,
        colors=colors,
        precip_peak_threshold=precip_peak_threshold,
        dry_days_threshold=dry_days_threshold,
        heat_threshold=heat_threshold,
        gdf_region=region,
        add_map=add_inset_map,
    )

    # Save the observed data
    if geods_obs is not None and export_observations:
        geods_obs = geods_obs.expand_dims(dim={"source": np.array(["observed"])})
        dir_out = dirname(climate_filenames[0])
        geods_obs.vector.to_netcdf(join(dir_out, "point_observed.nc"))

    if "snakemake" in globals():
        # Write a file when everything is done for snakemake tracking
        text_out = join(path_output, "point_climate.txt")
        with open(text_out, "w") as f:
            f.write("Timeseries plots per location were made.\n")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        plot_historical_climate_point(
            climate_filenames=sm.input.point_climate,
            path_output=dirname(sm.output.point_plot_done),
            climate_sources=sm.params.climate_sources,
            climate_sources_colors=sm.params.climate_sources_colors,
            data_catalog=sm.params.data_catalog,
            locations_filename=sm.params.location_file,
            precip_observations_filename=sm.params.location_timeseries_precip,
            precip_peak_threshold=sm.params.precip_peak,
            dry_days_threshold=sm.params.precip_dry,
            heat_threshold=sm.params.temp_heat,
        )

    else:
        print("This script should be run from a snakemake environment")
