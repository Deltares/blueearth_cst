"""Utility plot functions for scalar climate data."""

import os
from os.path import join
from pathlib import Path
from typing import Union, Optional, List

import xarray as xr
import geopandas as gpd
import numpy as np
import hydromt

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

__all__ = ["plot_scalar_climate_statistics"]


def plot_scalar_climate_statistics(
    geods: xr.Dataset,
    path_output: Union[str, Path],
    geods_obs: Optional[xr.Dataset] = None,
    climate_variables: Optional[List] = ["precip", "temp"],
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25,
):
    """
    Plot scalar climate statistics from a xarray dataset.

    Parameters
    ----------
    geods : xr.Dataset
        HydroMT GeoDataset with climate data to plot.
    path_output : str or Path
        Path to the output directory where the plots are stored.
    geods_obs : xr.Dataset, optional
        HydroMT GeoDataset with observed climate data to plot.
    climate_variables : list of str, optional
        List of climate variables to plot. By default ["precip", "temp"].
        Allowed values are "precip" for precipitation [mm] or "temp" for temperature
        [oC].
    precip_peak_threshold : float, optional
        Threshold for the precipitation peaks [mm/day] to highlight in the plot.
    dry_days_threshold : float, optional
        Threshold for the number of dry days [mm/day] to highlight in the plot.
    heat_threshold : float, optional
        Threshold for the number of heat days [oC] to highlight in the plot.
    """

    # Check if the output directory exists
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # No data inspection in the observed data
    if geods_obs is not None:
        # Find common period between obs and geods for fair comparison
        time_start = max(geods.time.min(), geods_obs.time.min())
        time_end = min(geods.time.max(), geods_obs.time.max())
        geods = geods.sel(time=slice(time_start, time_end))
        geods_obs = geods_obs.sel(time=slice(time_start, time_end))
    # Check the number of days in the first year in geods.time
    # and remove the year if not complete
    if len(geods.sel(time=geods.time.dt.year.isin(geods.time.dt.year[0]))) < 365:
        geods = geods.sel(time=~geods.time.dt.year.isin(geods.time.dt.year[0]))
        if geods_obs is not None:
            geods_obs = geods_obs.sel(
                time=~geods.time.dt.year.isin(geods.time.dt.year[0])
            )
    # Same for the last year
    if len(geods.sel(time=geods.time.dt.year.isin(geods.time.dt.year[-1]))) < 365:
        geods = geods.sel(time=~geods.time.dt.year.isin(geods.time.dt.year[-1]))
        if geods_obs is not None:
            geods_obs = geods_obs.sel(
                time=~geods.time.dt.year.isin(geods.time.dt.year[-1])
            )

    # Precipitation plots
    if "precip" in climate_variables and "precip" in geods.data_vars:
        # Loop over the index of geods
        for st in geods.index:
            print(f"Plotting precipitation for {st}")
            prec_st = geods.precip.sel(index=st)

            prec_st_obs = None
            if geods_obs is not None:
                if st in geods_obs.index and "precip" in geods_obs.data_vars:
                    prec_st_obs = geods_obs.precip.sel(index=st)

            # Plot the precipitation per location
            plot_precipitation_per_location(
                geoda=prec_st,
                path_output=path_output,
                geoda_obs=prec_st_obs,
                peak_threshold=precip_peak_threshold,
                dry_days_threshold=dry_days_threshold,
            )

    # Temperature plots
    if "temp" in climate_variables and "temp" in geods.data_vars:
        # Loop over the index of geods
        for st in geods.index:
            print(f"Plotting temperature for {st}")
            temp_st = geods.temp.sel(index=st)

            temp_st_obs = None
            if geods_obs is not None:
                if st in geods_obs.index and "temp" in geods_obs.data_vars:
                    temp_st_obs = geods_obs.temp.sel(index=st)

            # Plot the temperature per location
            plot_temperature_per_location(
                geoda=temp_st,
                path_output=path_output,
                geoda_obs=temp_st_obs,
                heat_threshold=heat_threshold,
            )

    return


def plot_precipitation_per_location(
    geoda: xr.DataArray,
    path_output: Union[str, Path],
    geoda_obs: Optional[xr.DataArray] = None,
    peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
):
    """Plot the precipitation per location."""
    # Get index value of geoda
    st = geoda["index"].values
    # Remove source for which there is no data
    geoda = geoda.dropna(dim="source", how="all")

    # Start the subplots for precipitation
    fig, axes = plt.subplots(
        3, 2, figsize=(18, 15)
    )  # , subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    fig.suptitle(f"Rainfall analysis for location {st}", fontsize="20")

    # 1. Plot per year
    for source in geoda.source.values:
        geoda.sel(source=source).resample(time="Y").sum().plot.line(
            ax=axes[0], label=source
        )
    if geoda_obs is not None:
        geoda_obs.resample(time="Y").sum().plot.line(
            ax=axes[0], label="Observed", linestyle="--", color="black"
        )

    axes[0].set_title("Rainfall amounts per year")
    axes[0].set_ylabel("Rainfall [mm/yr]")
    axes[0].set_xlabel("Year")
    axes[0].legend()

    # 2. Plot the cumulative
    for source in geoda.source.values:
        geoda.sel(source=source).cumsum().plot.line(ax=axes[1], label=source)
    if geoda_obs is not None:
        geoda_obs.cumsum().plot.line(
            ax=axes[1], label="Observed", linestyle="--", color="black"
        )

    axes[1].set_title("Cumulative rainfall")
    axes[1].set_ylabel("Cumulative rainfall [mm]")
    axes[1].set_xlabel("Year")
    axes[1].legend()

    # 3. Plot the monthly mean
    for source in geoda.source.values:
        geoda.sel(source=source).resample(time="M").sum().groupby(
            "time.month"
        ).mean().plot.line(ax=axes[2], label=source)
    if geoda_obs is not None:
        geoda_obs.resample(time="M").sum().groupby("time.month").mean().plot.line(
            ax=axes[2], label="Observed", linestyle="--", color="black"
        )

    axes[2].set_title("Long-term average rainfall per month")
    axes[2].set_ylabel("Rainfall [mm/month]")
    axes[2].set_xlabel("Month")
    axes[2].legend()

    # 4. Rainfall peaks > 40 mm/day
    for source in geoda.source.values:
        peak = (
            geoda.sel(source=source)
            .where(geoda.sel(source=source) > peak_threshold)
            .dropna(dim="time")
        )
        peak.plot.line(
            ax=axes[3], marker="o", linestyle=":", label=f"{source}: {len(peak)} peaks"
        )
    if geoda_obs is not None:
        peak_obs = geoda_obs.where(geoda_obs > peak_threshold).dropna(dim="time")
        peak_obs.plot.line(
            ax=axes[3],
            marker="o",
            label=f"Observed: {len(peak_obs)} peaks",
            linestyle="--",
            color="black",
        )

    axes[3].set_title(f"Rainfall Event > {peak_threshold}mm")
    axes[3].set_ylabel("Rainfall [mm/day]")
    axes[3].set_xlabel("Time")
    axes[3].grid(True)
    axes[3].legend()

    # 5. Number of dry days (dailyP < 0.2mm) per year
    for source in geoda.source.values:
        dry = geoda.sel(source=source)
        # Find if the data frequency is less than daily
        # if dry.time.freqstr != "D":
        #    dry = dry.resample(time="D").sum()
        dry = dry.where(dry < dry_days_threshold)
        dry.resample(time="Y").count().plot.line(
            ax=axes[4], marker="o", linestyle=":", label=f"{source}"
        )
        # # Find longest consecutive dry days spell per year
        # TODO

        # dry_spell.plot.line(
        #     ax=axes[4],
        #     marker="o",
        #     linestyle=":",
        #     label=f"{source} dry days",
        #     secondary_y=True,
        # )
    if geoda_obs is not None:
        dry_obs = geoda_obs
        # Find if the data frequency is less than daily
        # if dry_obs.time.freqstr != "D":
        #    dry_obs = dry_obs.resample(time="D").sum()
        dry_obs = dry_obs.where(dry_obs < dry_days_threshold)
        dry_obs.resample(time="Y").count().plot.line(
            ax=axes[4],
            marker="o",
            linestyle=":",
            label="Observed",
            color="black",
        )
        # # Find longest consecutive dry days spell per year
        # TODO

    axes[4].set_title(f"Number of dry days per year (P < {dry_days_threshold}mm)")
    axes[4].set_ylabel("Number of dry days")
    axes[4].set_xlabel("Year")
    axes[4].legend()

    # 6. Location plot
    # Add a background map
    # proj = ccrs.PlateCarree()
    # axes[5] = plt.axes(projection=proj)
    # axes[5].add_image(cimgt.OSM(), 10, alpha=0.8)
    # Plot the geometry of the location
    gdf = gpd.GeoSeries(
        data=geoda.geometry.values,
        index=np.atleast_1d(geoda.index.values),
        crs=geoda.vector.crs,
    )
    gdf.plot(ax=axes[5], color="white", edgecolor="black", alpha=0.5)

    axes[5].set_title("Location")
    axes[5].set_xlabel("Longitude")
    axes[5].set_ylabel("Latitude")

    # Save the figure
    fig.savefig(join(path_output, f"precipitation_{st}.png"))


def plot_temperature_per_location(
    geoda: xr.DataArray,
    path_output: Union[str, Path],
    geoda_obs: Optional[xr.DataArray] = None,
    heat_threshold: float = 25,
):
    """Plot the temperature per location."""
    # Get index value of geoda
    st = geoda["index"].values
    # Remove source for which there is no data
    geoda = geoda.dropna(dim="source", how="all")

    # Start the subplots for temperature
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle(f"Temperature analysis for location {st}", fontsize="20")

    # 1. Plot with mean, min, max per month filled in between
    for source in geoda.source.values:
        temp = geoda.sel(source=source)
        temp_mean = temp.resample(time="M").mean()
        temp_min = temp.resample(time="M").min()
        temp_max = temp.resample(time="M").max()
        axes[0].fill_between(
            temp_mean.time, temp_min, temp_max, alpha=0.2, label=source
        )
        temp_mean.plot.line(ax=axes[0], label=source)
    if geoda_obs is not None:
        temp_obs = geoda_obs.resample(time="M").mean()
        temp_obs_min = geoda_obs.resample(time="M").min()
        temp_obs_max = geoda_obs.resample(time="M").max()
        axes[0].fill_between(
            temp_obs.time,
            temp_obs_min,
            temp_obs_max,
            alpha=0.2,
            label="Observed",
            color="black",
        )
        temp_obs.plot.line(ax=axes[0], label="Observed", color="black", linestyle="--")

    axes[0].set_title("Monthly mean temperature (including minimum/maximum)")
    axes[0].set_ylabel("Temperature [oC]")
    axes[0].set_xlabel("Time")
    axes[0].legend()

    # 2. Plot with mean, min, max per year filled in between long term average
    for source in geoda.source.values:
        temp = geoda.sel(source=source).resample(time="M").mean()
        temp_mean = temp.groupby("time.month").mean()
        temp_min = temp.groupby("time.month").min()
        temp_max = temp.groupby("time.month").max()
        axes[1].fill_between(
            temp_mean.month, temp_min, temp_max, alpha=0.2, label=source
        )
        temp_mean.plot.line(ax=axes[1], label=source)
    if geoda_obs is not None:
        temp_obs = geoda_obs.resample(time="M").mean()
        temp_obs_mean = temp_obs.groupby("time.month").mean()
        temp_obs_min = temp_obs.groupby("time.month").min()
        temp_obs_max = temp_obs.groupby("time.month").max()
        axes[1].fill_between(
            temp_obs_mean.month,
            temp_obs_min,
            temp_obs_max,
            alpha=0.2,
            label="Observed",
            color="black",
        )
        temp_obs_mean.plot.line(
            ax=axes[1], label="Observed", color="black", linestyle="--"
        )

    axes[1].set_title("Long-term average temperature per month")
    axes[1].set_ylabel("Temperature [oC]")
    axes[1].set_xlabel("Month")
    axes[1].legend()

    # 3. Number of frost days per year
    for source in geoda.source.values:
        frost = (
            geoda.sel(source=source)
            .where(geoda.sel(source=source) < 0)
            .resample(time="Y")
            .count()
        )
        frost.plot.line(ax=axes[2], marker="o", linestyle=":", label=f"{source}")
    if geoda_obs is not None:
        frost_obs = geoda_obs.where(geoda_obs < 0).resample(time="Y").count()
        frost_obs.plot.line(
            ax=axes[2], marker="o", linestyle=":", label="Observed", color="black"
        )

    axes[2].set_title("Number of frost days per year (T < 0oC)")
    axes[2].set_ylabel("Number of frost days")
    axes[2].set_xlabel("Year")
    axes[2].legend()

    # 4. Number of heat days per year
    for source in geoda.source.values:
        heat = (
            geoda.sel(source=source)
            .where(geoda.sel(source=source) > heat_threshold)
            .resample(time="Y")
            .count()
        )
        heat.plot.line(ax=axes[3], marker="o", linestyle=":", label=f"{source}")
    if geoda_obs is not None:
        heat_obs = (
            geoda_obs.where(geoda_obs > heat_threshold).resample(time="Y").count()
        )
        heat_obs.plot.line(
            ax=axes[3], marker="o", linestyle=":", label="Observed", color="black"
        )

    axes[3].set_title(f"Number of heat days per year (T > {heat_threshold}oC)")
    axes[3].set_ylabel("Number of heat days")
    axes[3].set_xlabel("Year")
    axes[3].legend()

    # Save the figure
    fig.savefig(join(path_output, f"temperature_{st}.png"))
