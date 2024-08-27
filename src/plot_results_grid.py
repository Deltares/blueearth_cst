"""Plot gridded wflow results and compare to observations if any"""

import xarray as xr
import numpy as np
from os.path import join
from pathlib import Path
import hydromt

from typing import Union, List, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_gridded_hydrology import plot_gridded_snow_cover
else:
    from .plot_utils.plot_gridded_hydrology import plot_gridded_snow_cover


def plot_grid_wflow_historical(
    wflow_output_filenames: List[Union[str, Path]],
    climate_sources: List[str],
    plot_dir: Union[str, Path],
    observations_snow: Optional[Union[str, Path]] = None,
    data_catalog: List[Union[str, Path]] = [],
    basin_filename: Optional[Union[str, Path]] = None,
    river_filename: Optional[Union[str, Path]] = None,
    fontsize=8,
):
    """
    Analyse and plot gridded wflow results for historical runs.

    Supported variables: ['snow']

    Snow can be compared to observations of snow cover [%].

    Outputs:
    - plot of number of days with snow cover

    Parameters
    ----------
    wflow_output_filenames : List of Union[str, Path]
        Path to wflow gridded output files. Climate source should be included in the
        filename.
    climate_sources : List[str]
        List of climate datasets used to run wflow.
    plot_dir : Optional[Union[str, Path]], optional
        Path to the output folder for plots.
    observations_snow : Optional[Union[str, Path]], optional
        Path or data source in catalog to observations of snow cover.
        Required variables: ['snow_cover']
    data_catalog : List[Union[str, Path]], optional
        List of paths to data catalogs to search for the observations_snow data source.
    """
    # Create dictionary with snow cover for different datasets
    snow_cover = dict()
    # Initialize data_catalog
    data_catalog = hydromt.DataCatalog(data_catalog)

    # Read the observations if available for now just to get the start and end
    if observations_snow is not None:
        obs_snow = data_catalog.get_rasterdataset(observations_snow)
        # Check the start and end year-month of the observations
        start_obs_year = obs_snow.time[0].dt.year.values
        start_obs_month = obs_snow.time[0].dt.month.values
        end_obs_year = obs_snow.time[-1].dt.year.values
        end_obs_month = obs_snow.time[-1].dt.month.values
        obs_snow.close()

    # Read wflow results
    for climate_source in climate_sources:
        # Read wflow results
        fn = [fn for fn in wflow_output_filenames if climate_source in str(fn)][0]
        ds = xr.open_dataset(fn)
        if "snow" not in ds:
            continue

        # Get the start and end time
        start_sim = ds.time[0].values
        end_sim = ds.time[-1].values
        # Create a catchment mask based on the first timestep
        mask = ds["snow"].isel(time=0).notnull()
        # Assume snow is present if snow depth greater than 5.0 mm
        snow = ds["snow"] > 5.0
        # Slice in time to match observations
        if observations_snow is not None:
            snow = snow.sel(
                time=slice(
                    f"{start_obs_year}-{start_obs_month}-01",
                    f"{end_obs_year}-{end_obs_month}-31",
                )
            )
        # Calculate the percentage of time the snow depths exceeds 5.0mm
        snow = snow.mean(dim="time") * 100
        # Mask
        snow = snow.where(mask)
        # Load to memory
        snow = snow.compute()
        # Add to dictionary
        snow_cover[climate_source] = snow
        ds.close()

    # Add the observations if available
    if len(snow_cover) > 0 and observations_snow is not None:
        # Get the name of the first climate_source in snow_cover
        source = list(snow_cover.keys())[0]
        # This time read the observations for the model domain
        obs_snow = data_catalog.get_rasterdataset(
            observations_snow,
            time_tuple=(start_sim, end_sim),
            bbox=snow_cover[source].raster.bounds,
            buffer=0.2,
        )
        # Only use values where snow cover is between 10% and 100%
        obs_snow = obs_snow.where((obs_snow >= 10) & (obs_snow <= 100))
        # Percent of time snow cover is observed
        obs_snow = obs_snow.mean(dim="time")
        # Add to dictionary
        snow_cover["observed"] = obs_snow

    # Plot the snow cover
    if len(snow_cover) > 0:
        source = list(snow_cover.keys())[0]
        source_crs = snow_cover[source].raster.crs
        # Read the river and basin files
        if basin_filename is not None:
            gdf_basin = data_catalog.get_geodataframe(
                basin_filename, bbox=snow_cover[source].raster.bounds
            )
            if source_crs is not None and gdf_basin.crs != source_crs:
                gdf_basin = gdf_basin.to_crs(snow_cover[source].raster.crs)
            # Mask the observations with gdf_basin
            if "observed" in snow_cover:
                snow_cover["observed"] = snow_cover["observed"].assign_coords(
                    mask=snow_cover["observed"].raster.geometry_mask(
                        gdf_basin, all_touched=True
                    )
                )
                if snow_cover["observed"].raster.nodata is None:
                    snow_cover["observed"].raster.set_nodata(np.nan)
                snow_cover["observed"] = snow_cover["observed"].raster.mask(
                    snow_cover["observed"].coords["mask"]
                )
        else:
            gdf_basin = None
        if river_filename is not None:
            gdf_river = data_catalog.get_geodataframe(
                river_filename, bbox=snow_cover[source].raster.bounds
            )
            if source_crs is not None and gdf_river.crs != source_crs:
                gdf_river = gdf_river.to_crs(snow_cover[source].raster.crs)
        # Do the plot
        plot_gridded_snow_cover(
            snow_cover,
            plot_dir,
            gdf_basin=gdf_basin,
            gdf_river=gdf_river,
            fs=fontsize,
        )

    return


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        text_out = sm.output.output_txt
        plot_dir = join(sm.params.project_dir, "plots", "wflow_model_performance")
        wflow_root = join(sm.params.project_dir, "hydrology_model")

        # Check if gridded outputs exist
        wflow_output_filenames = sm.input.nc_files

        if len(wflow_output_filenames) == 0:
            print("No wflow output files found, skip plotting.")
            # Write a file for snakemake tracking
            with open(text_out, "w") as f:
                f.write(f"No gridded outputs for wflow. No plots were made.\n")

        else:
            plot_grid_wflow_historical(
                wflow_output_filenames=wflow_output_filenames,
                climate_sources=sm.params.climate_sources,
                plot_dir=plot_dir,
                observations_snow=sm.params.observations_snow,
                data_catalog=sm.params.data_catalog,
                basin_filename=join(wflow_root, "staticgeoms", "basins.geojson"),
                river_filename=join(wflow_root, "staticgeoms", "rivers.geojson"),
            )
            # Write a file for snakemake tracking
            with open(text_out, "w") as f:
                f.write(f"Plotted gridded wflow results.\n")
    else:
        print("This script should be run from a snakemake script.")
