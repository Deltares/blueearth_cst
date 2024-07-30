import os
from os.path import join, dirname, isfile
from pathlib import Path
from typing import Union, Optional, List

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np

import hydromt


def sample_climate_historical(
    clim_filename: Union[str, Path],
    region_filename: Union[str, Path],
    path_output: Union[str, Path],
    climate_catalog: Union[str, Path, List] = [],
    clim_source: Optional[str] = None,
    climate_variables: List[str] = ["precip", "temp"],
    subregions_filename: Optional[Union[str, Path]] = None,
    locations_filename: Optional[Union[str, Path]] = None,
    buffer: Optional[float] = 2.0,
):
    """
    Extract timeseries at specific locations from a climate dataset.

    Specific locations can be polygons or points. At minimum, region polygons should be
    provided to extract timeseries for the entire region.
    Additional polygons or points locations can be provided.

    Outputs:
    * **basin_{clim_source}.nc**: sampled timeseries plots over the region
        (and subregions).
    * **point_{clim_source}.nc**: sampled timeseries plots at specific point locations.

    Parameters
    ----------
    clim_filename : str, Path
        Path to the climate dataset file.
    region_filename : str, Path
        Path to the region vector file.
    path_output : str, Path
        Path to the output directory.
    climate_catalog : str, Path, List
        Path to the data catalogs yaml file or pre-defined catalogs. Catalogs can help
        read the subregions and locations files.
    clim_source : str
        Name of the climate source. It is used for the output filename and to add a
        source coordinate to the output dataset if provided.
    climate_variables : List
        List of climate variables to extract. By default ["precip", "temp"].
    subregions_filename : str, Path
        Path to the subregions vector file or data catalog entry.
        Optional variables: "name" for the subregion name.
    locations_filename : str, Path
        Path to the locations vector file or data catalog entry.
        Optional variables: "name" for the location name and "elevtn" for the elevation
        of the location.
    buffer : float
        Buffer in km around the region to extract the data.
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

    # Create data catalog
    data_catalog = hydromt.DataCatalog(data_libs=climate_catalog)

    # Read region
    region = gpd.read_file(region_filename)
    region = _update_gdf_index(region, prefix="region", legend_column="value")

    # Read subregions
    if subregions_filename is not None:
        subregions = data_catalog.get_geodataframe(subregions_filename)
        if "name" in subregions.columns:
            subregions.index = subregions["name"]
            subregions.index.name = "index"
        else:
            subregions = _update_gdf_index(
                subregions,
                prefix="subregion",
            )

    # Read locations
    if locations_filename is not None:
        # If the locs are a direct file without a crs property, assume 4326
        if isfile(locations_filename):
            crs = 4326
        else:
            crs = None
        locations = data_catalog.get_geodataframe(
            locations_filename,
            crs=crs,
            geom=region,
            buffer=buffer * 1000,
        )
        if "name" in locations.columns:
            locations.index = locations["name"]
            locations.index.name = "index"
        else:
            locations = _update_gdf_index(
                locations,
                prefix="location",
            )

    # Read climate dataset
    ds_clim = data_catalog.get_rasterdataset(clim_filename)
    # Select variables
    variables = [v for v in climate_variables if v in ds_clim.data_vars]
    ds_clim = ds_clim[variables]

    # Sample climate data for region
    print("Sampling climate data for region")
    ds_region = ds_clim.raster.zonal_stats(region, stats="mean")
    # Add climate_source as an extra dim
    ds_region = ds_region.expand_dims(source=[clim_source])
    # Rename the variables
    ds_region = ds_region.rename_vars(
        {v: v.replace("_mean", "") for v in ds_region.data_vars}
    )

    # Sample for subregions
    if subregions_filename is not None:
        print("Sampling climate data for subregions")
        ds_subregions = ds_clim.raster.zonal_stats(subregions, stats="mean")
        # Zonal stats does not work well if polygons
        # are within less than 2*2 cells
        # If this happens use the centroid of the polygon to sample
        if np.all(ds_subregions.isnull().values):
            ds_subregions = ds_clim.raster.sample(
                subregions.representative_point(), wdw=0
            )
        else:
            # Rename the variables
            ds_subregions = ds_subregions.rename_vars(
                {v: v.replace("_mean", "") for v in ds_region.data_vars}
            )

        ds_subregions = ds_subregions.expand_dims(source=[clim_source])

        # Merge region and subregions datasets
        ds_region = xr.merge([ds_region, ds_subregions])
        # Merge the gdf
        region_all = gpd.GeoDataFrame(
            pd.concat([region, subregions], ignore_index=False)
        )
    else:
        region_all = region

    # Convert to Geodataset
    geods_region = hydromt.vector.GeoDataset.from_gdf(region_all, data_vars=ds_region)

    # Save the sampled timeseries
    print("Saving the sampled timeseries per region (and subregions)")
    fn_out = "basin_climate.nc" if clim_source is None else f"basin_{clim_source}.nc"
    geods_filename = join(path_output, fn_out)
    if not os.path.exists(dirname(geods_filename)):
        os.makedirs(dirname(geods_filename))
    geods_region.vector.to_netcdf(geods_filename)

    # Sample for locations
    if locations_filename is not None:
        print("Sampling climate data for locations")
        # Sample climate data for point locations
        ds_locs = ds_clim.raster.sample(locations, wdw=0)
        # Add climate_source as an extra dim
        ds_locs = ds_locs.expand_dims(source=[clim_source])
        ds_locs = ds_locs.drop_vars(
            ["spatial_ref", ds_clim.raster.x_dim, ds_clim.raster.y_dim]
        )
        # Convert to Geodataset
        geods_locs = hydromt.vector.GeoDataset.from_gdf(locations, data_vars=ds_locs)

        # Save the sampled timeseries
        print("Saving the sampled timeseries per locations")
        fn_out = (
            "point_climate.nc" if clim_source is None else f"point_{clim_source}.nc"
        )
        geods_filename = join(path_output, fn_out)
        if not os.path.exists(dirname(geods_filename)):
            os.makedirs(dirname(geods_filename))
        geods_locs.vector.to_netcdf(geods_filename)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        # Extract the climate data
        sample_climate_historical(
            clim_filename=sm.input.grid_fn,
            region_filename=sm.input.region_fn,
            path_output=dirname(sm.output.basin),
            climate_catalog=sm.params.data_catalog,
            clim_source=sm.params.clim_source,
            climate_variables=sm.params.climate_variables,
            subregions_filename=sm.params.subregion_fn,
            locations_filename=sm.params.location_fn,
            buffer=sm.params.buffer_km,
        )

    else:
        print("This script should be run from a snakemake environment")
