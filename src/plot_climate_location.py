"""Plot historical climate for specific locations"""

import os
from os.path import join, dirname, isfile
from pathlib import Path
from typing import Union, Optional, List

import geopandas as gpd
import pandas as pd
import xarray as xr

import hydromt
from hydromt.nodata import NoDataStrategy
from hydromt import DataCatalog

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_scalar_climate import plot_scalar_climate_statistics
else:
    from .plot_utils.plot_scalar_climate import plot_scalar_climate_statistics


def plot_historical_climate_point(
    locations_filename: Union[str, Path],
    region_filename: Union[str, Path],
    path_output: Union[str, Path],
    data_catalog: Union[str, Path, List],
    climate_sources: Union[str, List[str]],
    climate_sources_colors: Optional[Union[str, List[str]]] = None,
    observations_timeseries_filename: Optional[Union[str, Path]] = None,
    climate_variables: List[str] = ["precip", "temp"],
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25,
    region_buffer: Optional[float] = 2,
    add_inset_map: bool = True,
):
    """Plot historical climate for a region and optionally subregions.

    Outputs:
    * **basin_climate.nc**: sampled timeseries plots over the region (and subregions).
    * **plots/point/**: plots of the historical climate for the point locations.

    The function plots the following statistics for each location in
    `locations_filename`:

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
    locations_filename : str or Path
        Path or data catalog source of the point locations file. If the observed
        timeseries data is not present in this file, they can be provided in
        ``observations_timeseries_filename``.
    region_filename : str or Path
        Path to the region boundary vector file to select the locations within the
        region of interest.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    data_catalog : str or list of str
        Path to the data catalog(s) to use.
    climate_sources : str or list of str
        Name of the climate data sources to plot. Should be available in
        climate_catalog.
    climate_sources_colors : str or list of str, optional
        Color to use for each ``climate_sources``. If None, unique color per source is
        not assured.
    observations_timeseries_filename: str or Path, optional
        Path or data catalog source to the observed timeseries data file for the
        locations in ``locations_filename``.
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
    region_buffer : float, optional
        Buffer around the region boundary to select the locations. By default 2.0 km.
    add_inset_map : bool, optional
        Add an inset map to the plots. By default True.
    """

    # Small function to set the index of the geodataframe
    def _update_geods_index(geods, prefix="location"):
        # Update the index
        index_dim = geods.vector.index_dim
        new_indexes = [f"{prefix}_{i}" for i in geods[index_dim].values]
        geods[index_dim] = new_indexes
        geods = geods.rename({index_dim: "index"})

        return geods

    # Load the region boundary
    region = gpd.read_file(region_filename)

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
        geom=region,
        buffer=region_buffer * 1000,
    )

    if observations_timeseries_filename is not None:
        # Load the timeseries data
        if isfile(observations_timeseries_filename):
            # Direct dataframe file
            ds_obs = hydromt.io.open_timeseries_from_table(
                observations_timeseries_filename,
                name="precip",
                index_dim=locations.index.name,
            )
        else:  # dataset data catalog entry
            ds_obs = data_catalog.get_dataset(observations_timeseries_filename)
        # Convert to Geodataset
        geods_obs = hydromt.vector.GeoDataset.from_gdf(
            locations,
            data_vars=ds_obs,
            index_dim=locations.index.name,
            merge_index="inner",
        )
        geods_obs = _update_geods_index(geods_obs, prefix="location")
    else:
        geods_obs = None

    # Update the index of the locations
    locations.index = f"location_" + locations.index.astype(str)
    locations.index.name = "index"

    # Load the climate data sources
    ds_clim_locs = []
    for climate_source in climate_sources:
        # Load per variable in case not all variables are present in the dataset
        for var in climate_variables:
            ds_clim = data_catalog.get_rasterdataset(
                climate_source,
                bbox=region.total_bounds,
                buffer=region_buffer * 1000,
                handle_nodata=NoDataStrategy.IGNORE,
                variables=var,
                single_var_as_array=False,  # HydroMT rename bug
            )
            # HydroMT rename bug... after bugfix check will be on None
            if var not in ds_clim.data_vars:
                continue

            # Sample climate data for point locations
            ds_locs = ds_clim.raster.sample(locations, wdw=0)
            # Add climate_source as an extra dim
            ds_locs = ds_locs.expand_dims(source=[climate_source])
            ds_locs = ds_locs.drop_vars(
                ["spatial_ref", ds_clim.raster.x_dim, ds_clim.raster.y_dim]
            )
            # Append to list
            ds_clim_locs.append(ds_locs)

    # Concatenate all climate sources
    ds_clim_locs = xr.merge(ds_clim_locs)

    # Convert to Geodataset
    geods_locs = hydromt.vector.GeoDataset.from_gdf(locations, data_vars=ds_clim_locs)

    # Prepare colors dict
    if climate_sources_colors is not None:
        colors = {k: v for k, v in zip(climate_sources, climate_sources_colors)}
    else:
        colors = None

    # Plot historical climate for region and optionally subregions
    geods_locs = plot_scalar_climate_statistics(
        geods=geods_locs,
        path_output=join(path_output, "plots", "point"),
        geods_obs=geods_obs,
        climate_variables=climate_variables,
        colors=colors,
        precip_peak_threshold=precip_peak_threshold,
        dry_days_threshold=dry_days_threshold,
        heat_threshold=heat_threshold,
        gdf_region=region,
        add_map=add_inset_map,
    )

    # Save the sampled timeseries
    geods_filename = join(path_output, "statistics", "point_climate.nc")
    if not os.path.exists(dirname(geods_filename)):
        os.makedirs(dirname(geods_filename))
    geods_locs.vector.to_netcdf(geods_filename)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir

        plot_historical_climate_point(
            locations_filename=sm.params.location_file,
            region_filename=sm.input.region_file,
            path_output=join(project_dir, "climate_historical"),
            data_catalog=sm.params.data_catalog,
            climate_sources=sm.params.climate_sources,
            climate_sources_colors=sm.params.climate_sources_colors,
            observations_timeseries_filename=sm.params.location_timeseries,
            precip_peak_threshold=sm.params.precip_peak,
            dry_days_threshold=sm.params.precip_dry,
            heat_threshold=sm.params.temp_heat,
            region_buffer=sm.params.region_buffer,
        )

    else:
        print("This script should be run from a snakemake environment")
