"""Utility plot functions for scalar climate data."""

import os
from os.path import join
from pathlib import Path
from typing import Union, Optional, List

import xarray as xr
import xclim
import numpy as np
import geopandas as gpd
import hydromt

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import seaborn as sns

__all__ = ["plot_scalar_climate_statistics"]


def plot_scalar_climate_statistics(
    geods: xr.Dataset,
    path_output: Union[str, Path],
    geods_obs: Optional[xr.Dataset] = None,
    climate_variables: Optional[List] = ["precip", "temp"],
    colors: Optional[List] = None,
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25,
    gdf_region: Optional[gpd.GeoDataFrame] = None,
) -> xr.Dataset:
    """
    Plot scalar climate statistics from a xarray dataset.

    The function plots the following statistics for each location in the geods:

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
    geods : xr.Dataset
        HydroMT GeoDataset with climate data to plot (organised along a source
        dimension).

        - Supported variables: ["precip", "temp"].
        - Required dimensions: ["time", "index"].
    path_output : str or Path
        Path to the output directory where the plots are stored.
    geods_obs : xr.Dataset, optional
        HydroMT GeoDataset with observed climate data to plot.
    climate_variables : list of str, optional
        List of climate variables to plot. By default ["precip", "temp"].
        Allowed values are "precip" for precipitation [mm] or "temp" for temperature
        [$\degree$C].
    colors: List, optional
        List of colors to use for each source. If None, unique color per source is
        not assured.
    precip_peak_threshold : float, optional
        Threshold for the precipitation peaks [mm/day] to highlight in the plot.
    dry_days_threshold : float, optional
        Threshold for the number of dry days [mm/day] to highlight in the plot.
    heat_threshold : float, optional
        Threshold for the number of heat days [$\degree$C] to highlight in the plot.
    gdf_region : gpd.GeoDataFrame, optional
        The total region of the project to add to the inset map if provided.

    Returns
    -------
    geods : xr.Dataset
        The input geods with an additional source dimension for the observed data and
        filtered over the common time period between geods and geods_obs.
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
    # Add a source dimension to the observed data
    if geods_obs is not None:
        geods_obs = geods_obs.expand_dims(dim={"source": np.array(["observed"])})
        geods = xr.merge([geods, geods_obs])
        if colors is not None:
            colors["observed"] = "black"
    # Check the number of days in the first year in geods.time
    # and remove the year if not complete
    if len(geods.sel(time=geods.time.dt.year.isin(geods.time.dt.year[0]))) < 365:
        geods = geods.sel(time=~geods.time.dt.year.isin(geods.time.dt.year[0]))
    # Same for the last year
    if len(geods.sel(time=geods.time.dt.year.isin(geods.time.dt.year[-1]))) < 365:
        geods = geods.sel(time=~geods.time.dt.year.isin(geods.time.dt.year[-1]))

    # Precipitation plots
    if "precip" in climate_variables and "precip" in geods.data_vars:
        # Loop over the index of geods
        for st in geods.index.values:
            print(f"Plotting precipitation for {st}")
            prec_st = geods.precip.sel(index=[st])

            # Plot the precipitation per location
            plot_precipitation_per_location(
                geoda=prec_st,
                path_output=path_output,
                colors=colors,
                peak_threshold=precip_peak_threshold,
                dry_days_threshold=dry_days_threshold,
                gdf_region=gdf_region,
            )

        # And plot for all locations (if more than 1)
        prec = geods.precip.copy()
        prec = prec.dropna(dim="index", how="all")
        if len(prec.index.values) > 1:
            print("Plotting precipitation boxplot for all locations")
            # Plot the precipitation per location
            boxplot_clim(
                da=prec,
                path_output=path_output,
                colors=colors,
            )

    # Temperature plots
    if "temp" in climate_variables and "temp" in geods.data_vars:
        # Loop over the index of geods
        for st in geods.index.values:
            print(f"Plotting temperature for {st}")
            temp_st = geods.temp.sel(index=[st])

            # Plot the temperature per location
            plot_temperature_per_location(
                geoda=temp_st,
                path_output=path_output,
                colors=colors,
                heat_threshold=heat_threshold,
                gdf_region=gdf_region,
            )

    return geods


def plot_precipitation_per_location(
    geoda: xr.DataArray,
    path_output: Union[str, Path],
    colors: Optional[List] = None,
    peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    gdf_region: Optional[gpd.GeoDataFrame] = None,
):
    """Plot the precipitation per location."""
    # Get index value of geoda
    st = geoda["index"].values[0]
    # Remove source for which there is no data
    geoda = geoda.dropna(dim="source", how="all")
    # Get the geoda geometry now before reducing st and loading
    gdf = geoda.vector.geometry
    geoda = geoda.sel(index=st).load()

    # Get the wettest year from Observed if available
    wettest_source = (
        "observed" if "observed" in geoda.source.values else geoda.source.values[0]
    )
    year_sum = geoda.sel(source=wettest_source).resample(time="Y").sum()
    year_sum["time"] = year_sum["time.year"]
    wet_year = str(year_sum.isel(time=year_sum.argmax().values).time.values)

    # Start the subplots for precipitation
    fig, axes = plt.subplots(
        3, 2, figsize=(16 / 2.54, 21 / 2.54)
    )  # , subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    ax_twin = axes[4].twinx()
    fig.suptitle(f"Precipitation analysis for location {st}", fontsize="10")

    ### Do the plots ###
    for source in geoda.source.values:
        # Styling
        ls = "-" if source != "observed" else "--"
        fs = 8
        if source == "observed":
            c = "black"
        elif colors is not None:
            c = colors[source]
        else:
            c = None
        # Select current source
        prec = geoda.sel(source=source).copy()

        # 1. Plot per year
        prec_yr = prec.resample(time="Y").sum()
        prec_yr["time"] = prec_yr["time.year"]
        prec_yr.plot.line(
            ax=axes[0],
            label=source,
            linestyle=ls,
            color=c,
        )

        # 2. Plot the cumulative
        prec.cumsum().plot.line(ax=axes[1], label=source, linestyle=ls, color=c)

        # 3. Plot the monthly mean
        prec_m = prec.resample(time="M").sum().groupby("time.month").mean()
        # Add the 25-75 quantile range
        prec_m_25 = prec.resample(time="M").sum().groupby("time.month").quantile(0.25)
        prec_m_75 = prec.resample(time="M").sum().groupby("time.month").quantile(0.75)
        prec_month = np.array(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        axes[2].fill_between(
            prec_month,
            prec_m_25,
            prec_m_75,
            alpha=0.2,
            label=f"{source} 25%-75%",
            color=c,
        )
        prec_m["month"] = prec_month
        prec_m.plot.line(ax=axes[2], label=source, linestyle=ls, color=c)

        # 4. Rainfall peaks > 40 mm/day
        peak = prec.where(geoda.sel(source=source) > peak_threshold).dropna(dim="time")
        peak.plot.line(
            ax=axes[3],
            marker="o",
            linestyle=":",
            color=c,
            label=f"{source}: {len(peak)} peaks",
        )

        # 5. Number of dry days (dailyP < 0.2mm) per year
        dry = prec.where(prec < dry_days_threshold).resample(time="Y").count()
        dry["time"] = dry["time.year"]
        dry.plot.line(ax=axes[4], marker="o", linestyle=":", color=c, label=f"{source}")
        # Find longest consecutive dry days spell per year - xclim
        prec.attrs.update({"units": "mm/day"})
        dry_spell = xclim.indices.maximum_consecutive_dry_days(
            prec,
            thresh=f"{dry_days_threshold*4} mm/day",
            freq="YS",
        )
        dry_spell["time"] = dry_spell["time.year"]
        ax_twin.plot(
            dry_spell.time.values,
            dry_spell.values,
            marker="+",
            linestyle="",
            color=c,
            label=source,
        )

        # 6. Weekly plot for the wettest year
        wettest_year = (
            prec.sel(time=slice(f"{wet_year}-01-01", f"{wet_year}-12-31"))
            .resample(time="W")
            .sum()
        )
        wettest_year.plot.step(ax=axes[5], label=source, color=c, ls=ls)

    ### Add legends ###
    # 1. Plot per year
    axes[0].set_title("Precipitation per year", fontsize=fs)
    axes[0].set_ylabel("Precipitation [mm/yr]", fontsize=fs)
    axes[0].set_xlabel("", fontsize=fs)
    axes[0].legend(fontsize=fs)

    # 2. Plot the cumulative
    axes[1].set_title("Cumulative precipitation", fontsize=fs)
    axes[1].set_ylabel("Cumulative precipitation [mm]", fontsize=fs)
    axes[1].set_xlabel("", fontsize=fs)
    axes[1].legend(fontsize=fs)

    # 3. Plot the monthly mean
    axes[2].set_title("Long-term average precipitation per month", fontsize=fs)
    axes[2].set_ylabel("Precipitation [mm/month]", fontsize=fs)
    axes[2].set_xlabel("", fontsize=fs)
    axes[2].legend(fontsize=fs)

    # 4. Rainfall peaks > 40 mm/day
    axes[3].set_title(f"Precipitation Events > {peak_threshold} mm/day", fontsize=fs)
    axes[3].set_ylabel("Precipitation [mm/day]", fontsize=fs)
    axes[3].set_xlabel("", fontsize=fs)
    axes[3].grid(True)
    axes[3].legend(fontsize=fs)

    # 5. Number of dry days (dailyP < 0.2mm) per year
    axes[4].set_title(
        f"Number of dry days per year (P < {dry_days_threshold}mm)", fontsize=fs
    )
    axes[4].set_ylabel("Number of dry days", fontsize=fs)
    ax_twin.set_ylabel(
        f"Longest dry spell (P < {dry_days_threshold*4}mm) [days]", fontsize=fs
    )
    ax_twin.grid(True)
    ax_twin.legend(title="Longest dry spell", fontsize=fs)
    axes[4].set_xlabel("", fontsize=fs)
    axes[4].legend(title="Number of dry days", fontsize=fs)

    # 6. Select the wettest year and do daily plot
    axes[5].set_title(
        f"Weekly precipitation for the wettest year ({wet_year} according to {wettest_source})",
        fontsize=fs,
    )
    axes[5].set_xlabel("", fontsize=fs)
    axes[5].set_ylabel("Precipitation [mm/week]", fontsize=fs)
    axes[5].legend(fontsize=fs)

    # A. Location plot
    # Add an inset map on the top right corner of the figure
    proj = ccrs.PlateCarree()
    ax_inset = fig.add_axes([0.84, 0.89, 0.15, 0.1], projection=proj)
    if gdf_region is not None:
        bbox = gdf_region.to_crs(3857).buffer(20e3).to_crs(gdf_region.crs).total_bounds
    else:
        bbox = gdf.to_crs(3857).buffer(20e3).to_crs(gdf.crs).total_bounds
    extent = np.array(bbox)[[0, 2, 1, 3]]
    ax_inset.set_extent(extent, crs=proj)
    # Add background image
    ax_inset.add_image(cimgt.GoogleTiles(style="street"), 8, alpha=0.8)
    # Add the total region if provided
    if gdf_region is not None:
        gdf_region.plot(ax=ax_inset, color="white", edgecolor="black", alpha=0.5)
    # Plot the geometry of the location
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        gdf.plot(ax=ax_inset, color="white", edgecolor="red", alpha=0.5)
    else:
        gdf.plot(ax=ax_inset, color="red", edgecolor="black", alpha=0.5)

    # Save the figure
    plt.tight_layout()
    fig.savefig(join(path_output, f"precipitation_{st}.png"))


def plot_temperature_per_location(
    geoda: xr.DataArray,
    path_output: Union[str, Path],
    colors: Optional[List] = None,
    heat_threshold: float = 25,
    gdf_region: Optional[gpd.GeoDataFrame] = None,
):
    """Plot the temperature per location."""
    # Get index value of geoda
    st = geoda["index"].values[0]
    # Remove source for which there is no data
    geoda = geoda.dropna(dim="source", how="all")
    # Get the geoda geometry now before reducing st and loading
    gdf = geoda.vector.geometry
    geoda = geoda.sel(index=st).load()

    # Start the subplots for temperature
    fig, axes = plt.subplots(2, 2, figsize=(16 / 2.54, 14 / 2.54))
    axes = axes.flatten()
    fig.suptitle(f"Temperature analysis for location {st}", fontsize="10")

    ### Do the plots ###
    for source in geoda.source.values:
        # Styling
        ls = "-" if source != "observed" else "--"
        fs = 8
        if source == "observed":
            c = "black"
        elif colors is not None:
            c = colors[source]
        else:
            c = None
        # Select current source
        temp = geoda.sel(source=source).copy()

        # 1. Plot with mean, min, max per month filled in between
        temp_mean = temp.resample(time="M").mean()
        temp_min = temp.resample(time="M").min()
        temp_max = temp.resample(time="M").max()
        axes[0].fill_between(
            temp_mean.time,
            temp_min,
            temp_max,
            alpha=0.2,
            label=f"{source} min-max",
            color=c,
        )
        temp_mean.plot.line(ax=axes[0], label=source, color=c)

        # 2. Plot with mean, min, max per year filled in between long term average
        temp_m = temp.resample(time="M").mean()
        temp_mean = temp_m.groupby("time.month").mean()
        temp_min = temp_m.groupby("time.month").quantile(0.25)
        temp_max = temp_m.groupby("time.month").quantile(0.75)
        month_names = np.array(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        axes[1].fill_between(
            month_names,
            temp_min,
            temp_max,
            alpha=0.2,
            label=f"{source} 25%-75%",
            color=c,
        )
        temp_mean["month"] = month_names
        temp_mean.plot.line(ax=axes[1], label=source, linestyle=ls, color=c)

        # 3. Number of frost days per year
        frost = temp.where(temp < 0).resample(time="Y").count()
        frost["time"] = frost["time.year"]
        frost.plot.line(
            ax=axes[2], marker="o", linestyle=":", label=f"{source}", color=c
        )

        # 4. Number of heat days per year
        heat = temp.where(temp > heat_threshold).resample(time="Y").count()
        heat["time"] = heat["time.year"]
        heat.plot.line(
            ax=axes[3], marker="o", linestyle=":", label=f"{source}", color=c
        )

    ### Add legends ###
    # 1. Plot with mean, min, max per month filled in between
    axes[0].set_title("Monthly mean temperature", fontsize=fs)
    axes[0].set_ylabel("Temperature [$\degree$C]", fontsize=fs)
    axes[0].set_xlabel("", fontsize=fs)
    axes[0].legend(fontsize=fs)

    # 2. Plot with mean, min, max per year filled in between long term average
    axes[1].set_title("Long-term average temperature per month", fontsize=fs)
    axes[1].set_ylabel("Temperature [$\degree$C]", fontsize=fs)
    axes[1].set_xlabel("", fontsize=fs)
    axes[1].legend(fontsize=fs)

    # 3. Number of frost days per year
    axes[2].set_title("Number of frost days per year (T < 0$\degree$C)", fontsize=fs)
    axes[2].set_ylabel("Number of frost days", fontsize=fs)
    axes[2].set_xlabel("", fontsize=fs)
    axes[2].legend(fontsize=fs)

    # 4. Number of heat days per year
    axes[3].set_title(
        f"Number of heat days per year (T > {heat_threshold}$\degree$C)",
        fontsize=fs,
    )
    axes[3].set_ylabel("Number of heat days", fontsize=fs)
    axes[3].set_xlabel("", fontsize=fs)
    axes[3].legend(fontsize=fs)

    # A. Location plot
    # Add an inset map on the top right corner of the figure
    proj = ccrs.PlateCarree()
    ax_inset = fig.add_axes([0.84, 0.89, 0.15, 0.1], projection=proj)
    if gdf_region is not None:
        bbox = gdf_region.to_crs(3857).buffer(20e3).to_crs(gdf_region.crs).total_bounds
    else:
        bbox = gdf.to_crs(3857).buffer(20e3).to_crs(gdf.crs).total_bounds
    extent = np.array(bbox)[[0, 2, 1, 3]]
    ax_inset.set_extent(extent, crs=proj)
    # Add background image
    ax_inset.add_image(cimgt.GoogleTiles(style="street"), 8, alpha=0.8)
    # Add the total region if provided
    if gdf_region is not None:
        gdf_region.plot(ax=ax_inset, color="white", edgecolor="black", alpha=0.5)
    # Plot the geometry of the location
    if gdf.geometry.geom_type.iloc[0] == "Polygon":
        gdf.plot(ax=ax_inset, color="white", edgecolor="red", alpha=0.5)
    else:
        gdf.plot(ax=ax_inset, color="red", edgecolor="black", alpha=0.5)

    # Save the figure
    fig.savefig(join(path_output, f"temperature_{st}.png"))


def boxplot_clim(
    da: xr.DataArray,
    path_output: Union[str, Path],
    colors: Optional[List] = None,
):
    """Boxplot of annual precipitation data at different locations."""
    fs = 8
    # First convert to a dataframe
    # Resample to year
    da = da.resample(time="Y").sum(min_count=1)
    dfa = da.to_dataframe()[["precip"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        ax=ax,
        data=dfa,
        x="index",
        hue="source",
        y="precip",
        # order = stations_z,
        palette=colors,
        width=0.8,
        fliersize=0.5,
        linewidth=0.5,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    ax.set_xlabel("", fontsize=fs)
    if da.name == "precip":
        ax.set_ylabel("Precipitation [mm/year]", fontsize=fs)
    ax.legend(fontsize=fs)

    # Save the figure
    plt.tight_layout()
    plt.savefig(join(path_output, f"precipitation_boxplot.png"), dpi=300)
