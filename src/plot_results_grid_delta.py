"""Plot gridded wflow results future and compare to historical"""

import xarray as xr
import numpy as np
import os
from os.path import join, dirname, isfile
from pathlib import Path
import hydromt
from hydromt_wflow import WflowModel

from typing import Union, List, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_gridded_hydrology import (
        plot_gridded_snow_cover,
        plot_gridded_snow_cover_relative,
    )
else:
    from .plot_utils.plot_gridded_hydrology import (
        plot_gridded_snow_cover,
        plot_gridded_snow_cover_relative,
    )


def percent_of_time_snow(da):
    # Create a catchment mask based on the first timestep
    mask = da.isel(time=0).notnull()
    # Assume snow is present if snow depth greater than 5.0 mm
    da = da > 5.0
    # Calculate the percentage of time the snow depths exceeds 5.0mm
    da = da.mean(dim="time") * 100
    # Mask
    da = da.where(mask)
    # Load to memory
    da = da.compute()

    return da


def plot_grid_wflow_future(
    wflow_output_filenames: List[Union[str, Path]],
    scenarios: List[str],
    horizons: List[str],
    models: List[str],
    plot_dir: Union[str, Path],
    wflow_root: Optional[Union[str, Path]] = None,
    config_historical: Union[str, Path] = "wflow_sbm.toml",
    fontsize=8,
):
    """
    Analyse and plot gridded wflow results for future runs.

    Supported variables: ['snow']

    Outputs:
    - plot of number of days with snow cover

    Parameters
    ----------
    wflow_output_filenames : List of Union[str, Path]
        Path to wflow gridded output files. Climate scenario, model and horizon should
        be included in the filename.
    scenarios : List[str]
        List of climate scenarios used to run wflow.
    horizons : List[str]
        List of time horizons used to run wflow.
    models : List[str]
        List of climate models used to run wflow.
    plot_dir : Optional[Union[str, Path]], optional
        Path to the output folder for plots.
    wflow_root : Optional[Union[str, Path]], optional
        Path to the wflow model root folder. Used to read the gridded outputs
        of the historical run as well as the basins and rivers.
    config_historical : Optional[Union[str, Path]], optional
        Path to the historical configuration file. Default is 'wflow_sbm.toml'.
    """
    # Create dictionary with snow cover for different datasets
    snow_cover = dict()

    # Load historical data
    if wflow_root is not None:
        wflow = WflowModel(
            wflow_root,
            config_fn=os.path.relpath(config_historical, wflow_root),
            mode="r",
        )
        # Read basins and rivers
        gdf_basin = wflow.basins
        gdf_river = wflow.rivers
        # Read historical output
        output_dir = wflow.get_config("dir_output", fallback="")
        nc_fn = wflow.get_config("output.path", abs_path=False)
        if nc_fn is not None:
            nc_fn = join(
                wflow_root,
                dirname(os.path.relpath(config_historical, wflow_root)),
                output_dir,
                nc_fn,
            )
        if nc_fn is not None and isfile(nc_fn):
            ds = xr.open_dataset(nc_fn)
            if "snow" in ds:
                snow = percent_of_time_snow(ds["snow"])
                # Add to dictionary
                snow_cover["Historical"] = snow
                ds.close()
    else:
        gdf_basin = None
        gdf_river = None

    # Loop over the scenarios and horizons
    for scenario in scenarios:
        for horizon in horizons:
            snow_cover_scenario = snow_cover.copy()
            # Read wflow results
            filenames_scenario = [
                fn
                for fn in wflow_output_filenames
                if scenario in str(fn) and horizon in str(fn)
            ]
            for model in models:
                fn_model = [fn for fn in filenames_scenario if model in str(fn)]
                if isfile(fn_model[0]):
                    ds = xr.open_dataset(fn_model[0])
                    if "snow" in ds:
                        snow = percent_of_time_snow(ds["snow"])
                        # Add to dictionary
                        snow_cover_scenario[f"{model}"] = snow
                        ds.close()
            # Plot
            if len(snow_cover_scenario) > 0:
                plot_gridded_snow_cover(
                    snow_cover=snow_cover_scenario,
                    plot_dir=plot_dir,
                    plot_filename=f"snow_cover_{scenario}_{horizon}.png",
                    gdf_basin=gdf_basin,
                    gdf_river=gdf_river,
                    fs=fontsize,
                )
                if "Historical" in snow_cover_scenario:
                    plot_gridded_snow_cover_relative(
                        snow_cover=snow_cover_scenario,
                        reference_snow_cover="Historical",
                        plot_dir=plot_dir,
                        plot_filename=f"snow_cover_{scenario}_{horizon}_rel.png",
                        gdf_basin=gdf_basin,
                        gdf_river=gdf_river,
                        fs=fontsize,
                    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        text_out = sm.output.output_txt
        plot_dir = dirname(text_out)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        wflow_root = join(sm.params.project_dir, "hydrology_model")

        # Check if gridded outputs exist
        wflow_output_filenames = sm.input.nc_file_near
        wflow_output_filenames.extend(sm.input.nc_file_far)

        if len(wflow_output_filenames) == 0:
            print("No wflow output files found, skip plotting.")
            # Write a file for snakemake tracking
            with open(text_out, "w") as f:
                f.write(f"No gridded outputs for wflow. No plots were made.\n")

        else:
            plot_grid_wflow_future(
                wflow_output_filenames=sm.input.nc_file_near,
                scenarios=sm.params.scenarios,
                horizons=sm.params.future_horizons,
                models=sm.params.gcms,
                plot_dir=plot_dir,
                wflow_root=wflow_root,
                config_historical=sm.params.config_historical,
            )
            # Write a file for snakemake tracking
            with open(text_out, "w") as f:
                f.write(f"Plotted gridded wflow results.\n")
    else:
        print("This script should be run from a snakemake script.")
