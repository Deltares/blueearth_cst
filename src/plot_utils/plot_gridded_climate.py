"""Utility plot functions for gridded climate data."""

import os
from os.path import join
from pathlib import Path
from typing import Union, Dict, Optional

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

__all__ = ["plot_gridded_precip"]


def plot_gridded_precip(
    precip_dict: Dict[str, xr.DataArray],
    path_output: Union[str, Path],
    gdf_region: Optional[gpd.GeoDataFrame] = None,
):
    """
    Plot the median annual precipitation for multiple climate sources.

    Parameters
    ----------
    precip_dict : dict
        Dictionary with the precipitation data for each climate source.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    gdf_region : gpd.GeoDataFrame, optional
        The total region of the project to add to the inset map if provided.
    """

    # Find the common time period between sources
    time_start = max([v.time.values[0] for v in precip_dict.values()])
    time_end = min([v.time.values[-1] for v in precip_dict.values()])
    # Check if time start is January 1st and time end is December 31st
    # Else remove the first and last year
    time_start = pd.to_datetime(time_start)
    time_end = pd.to_datetime(time_end)
    if time_start.month != 1 or time_start.day != 1:
        time_start = time_start.replace(year=time_start.year + 1, month=1, day=1)
    if time_end.month != 12 or time_end.day != 31:
        time_end = time_end.replace(year=time_end.year - 1, month=12, day=31)
    # Sel each source
    precip_dict = {
        k: v.sel(time=slice(time_start, time_end)) for k, v in precip_dict.items()
    }

    # Compute the sum of precipitation per year
    precip_dict = {k: v.resample(time="YE").sum() for k, v in precip_dict.items()}
    # Compute the median of the annual precipitation
    precip_dict = {k: v.median("time") for k, v in precip_dict.items()}

    # Find the max and min over each source
    max_precip = max([v.max().values for v in precip_dict.values()])
    min_precip = min([v.min().values for v in precip_dict.values()])

    # Plot the precipitation in one figure
    fig, ax = plt.subplots(
        1, len(precip_dict), figsize=(16 / 2.54, 8 / 2.54), sharex=True, sharey=True
    )
    fs = 8
    for i in range(len(precip_dict)):
        k = list(precip_dict.keys())[i]
        v = precip_dict[k]
        v.plot(ax=ax[i], label=k, vmin=min_precip, vmax=max_precip, cmap="viridis")
        # add outline basin
        if gdf_region is not None:
            gdf_region.plot(ax=ax[i], facecolor="None")
        ax[i].set_title(k, fontsize=fs)
        ax[i].set_xlabel("Longitude", fontsize=fs)
        ax[i].set_ylabel("Latitude", fontsize=fs)
        ax[i].tick_params(axis="both", labelsize=fs)
        # Change the colorbar title and fonctsize
        cbar = ax[i].collections[0].colorbar
        cbar.set_label("precipitation [mm/year]", fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    fig.suptitle("Median annual precipitation", fontsize=fs + 2)

    fig.tight_layout()
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fig.savefig(join(path_output, "median_annual_precipitation.png"))
