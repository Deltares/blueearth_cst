"""Plot historical climate for a region and optionally subregions"""

import os
from os.path import join, dirname
from pathlib import Path
from typing import Union, Optional, List

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np

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


def plot_historical_climate_region(
    region_filename: Union[str, Path],
    path_output: Union[str, Path],
    climate_catalog: Union[str, Path, List],
    climate_sources: Union[str, List[str]],
    climate_sources_colors: Optional[Union[str, List[str]]] = None,
    subregions_filename: Optional[Union[str, Path]] = None,
    climate_variables: List[str] = ["precip", "temp"],
    legend_column: str = "value",
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25,
):
    """Plot historical climate for a region and optionally subregions.

    Outputs:
    * **basin_climate.nc**: sampled timeseries plots over the region (and subregions).


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
    climate_sources_colors : str or list of str, optional
        Color to use for each ``climate_sources``. If None, unique color per source is
        not assured.
    subregions_filename : str or Path, optional
        Path to the subregions boundary vector file to produce similar plots for
        subregions.
    climate_variables : list of str, optional
        List of climate variables to sample/plot.
    legend_column : str, optional
        Column name of region/subregions to use for the legend in the plots. By default
        'value' is used. Else the index will be used.
    precip_peak_threshold : float, optional
        Threshold for the peak precipitation in mm/day to define the rainfall peaks.
        By default 40 mm/day.
    dry_days_threshold : float, optional
        Threshold for the precipitation in mm/day to define a dry day. By default 0.2
        mm/day.
    heat_threshold : float, optional
        Threshold for the daily mean temperature in degrees Celsius to define a
        heatwave. By default 25 degrees Celsius.
    """

    # Small function to set the index of the geodataframe
    def _update_gdf_index(gdf, prefix="region", legend_column="value"):
        if legend_column in gdf.columns:
            if gdf[legend_column].dtype == float:
                gdf[legend_column] = gdf[legend_column].astype(int)
            gdf.index = f"{prefix}_" + gdf[legend_column].astype(str)
            gdf.index.name = "index"
        else:
            gdf.index = f"{prefix}_" + gdf.index.astype(str)

        return gdf

    # Load region boundary
    region = gpd.read_file(region_filename)
    region = _update_gdf_index(region, prefix="region", legend_column=legend_column)

    # Load climate data catalog
    data_catalog = DataCatalog(climate_catalog)

    # Load subregions boundary if provided
    if subregions_filename:
        subregions = gpd.read_file(subregions_filename)
        subregions = _update_gdf_index(
            subregions, prefix="subregion", legend_column=legend_column
        )
        ds_clim_subregions = []
    else:
        subregions = None

    # Load climate data sources
    ds_clim_region = []
    for climate_source in climate_sources:
        # Load per variable in case not all variables are present in the dataset
        for var in climate_variables:
            ds_clim = data_catalog.get_rasterdataset(
                climate_source,
                bbox=region.total_bounds,
                buffer=2,
                handle_nodata=NoDataStrategy.IGNORE,
                variables=var,
                single_var_as_array=False,  # HydroMT rename bug
            )
            # HydroMT rename bug... after bugfix check will be on None
            if var not in ds_clim.data_vars:
                continue

            # Sample climate data for region
            ds_region = ds_clim.raster.zonal_stats(region, stats="mean")
            # Add climate_source as an extra dim
            ds_region = ds_region.expand_dims(source=[climate_source])
            # Append to list
            ds_clim_region.append(ds_region)

            # Same for subregions if provided
            if subregions is not None:
                ds_subregions = ds_clim.raster.zonal_stats(subregions, stats="mean")
                # Zonal stats does not work well if polygons
                # are within less than 2*2 cells
                # If this happens use the centroid of the polygon to sample
                if np.all(ds_subregions[f"{var}_mean"].isnull().values):
                    ds_sample = ds_clim.raster.sample(
                        subregions.representative_point(), wdw=0
                    )
                    ds_subregions[f"{var}_mean"] = (
                        ("time", "index"),
                        ds_sample[var].values,
                    )

                ds_subregions = ds_subregions.expand_dims(source=[climate_source])
                ds_clim_subregions.append(ds_subregions)

    # Concatenate all climate sources
    ds_clim_region = xr.merge(ds_clim_region)
    if subregions is not None:
        ds_clim_subregions = xr.merge(ds_clim_subregions)
        # Merge region and subregions datasets
        ds_clim_region = xr.merge([ds_clim_region, ds_clim_subregions])
        # Merge the gdf
        region_all = gpd.GeoDataFrame(
            pd.concat([region, subregions], ignore_index=False)
        )
    else:
        region_all = region

    # Rename the variables
    ds_clim_region = ds_clim_region.rename_vars(
        {v: v.replace("_mean", "") for v in ds_clim_region.data_vars}
    )

    # Convert to Geodataset
    geods_region = hydromt.vector.GeoDataset.from_gdf(
        region_all, data_vars=ds_clim_region
    )

    # Prepare colors dict
    if climate_sources_colors is not None:
        colors = {k: v for k, v in zip(climate_sources, climate_sources_colors)}
    else:
        colors = None

    # Plot historical climate for region and optionally subregions
    plot_scalar_climate_statistics(
        geods=geods_region,
        path_output=join(path_output, "plots", "region"),
        climate_variables=climate_variables,
        colors=colors,
        precip_peak_threshold=precip_peak_threshold,
        dry_days_threshold=dry_days_threshold,
        heat_threshold=heat_threshold,
        gdf_region=region,
    )

    # Save the sampled timeseries
    geods_filename = join(path_output, "statistics", "basin_climate.nc")
    if not os.path.exists(dirname(geods_filename)):
        os.makedirs(dirname(geods_filename))
    geods_region.vector.to_netcdf(geods_filename)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir

        plot_historical_climate_region(
            region_filename=sm.input.region_file,
            path_output=join(project_dir, "climate_historical"),
            climate_catalog=sm.params.data_catalog,
            climate_sources=sm.params.climate_sources,
            climate_sources_colors=sm.params.climate_sources_colors,
            subregions_filename=sm.params.subregion_file,
            precip_peak_threshold=sm.params.precip_peak,
            dry_days_threshold=sm.params.precip_dry,
            heat_threshold=sm.params.temp_heat,
        )

    else:
        print("This script should be run from a snakemake environment")
