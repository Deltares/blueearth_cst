"""Utilities plot functions for gridded and scalar anomalies."""

import os
from os.path import join
from pathlib import Path
from typing import Union, Optional, Dict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

import xarray as xr
import numpy as np
import scipy.stats as stats
import geopandas as gpd
import cartopy.crs as ccrs

__all__ = ["plot_gridded_anomalies", "plot_timeseries_anomalies"]


def linealg(x, a, b):
    """Linear function for curvefit."""
    return a * x + b


def compute_anomalies_year(
    da: xr.DataArray, no_data_limit: Optional[int] = None
) -> xr.DataArray:
    """Resample timeseries to year and compute anomalies for a given DataArray.

    Supported variables are "precip" and "temp" and "Q".

    Parameters
    ----------
    da : xr.DataArray
        DataArray with the climate and hydrology data.
    no_data_limit : int, optional
        Maximum number of nodata values allowed per year. If exceeded, the year is
        considered as nodata.

    Returns
    -------
    da_yr_anom: xr.DataArray
        DataArray with the anomalies.
    """
    if no_data_limit is not None:
        # Compute the number of nodata per year
        da_nodata = da.isnull().resample(time="YS").sum("time")
    if da.name == "precip":
        # Calculate the yearly sum
        da_yr = da.resample(time="YS").sum("time")
        # Filter out if too many nodata
        if no_data_limit is not None:
            da_yr = da_yr.where(da_nodata < no_data_limit, np.nan)
        da_yr["time"] = da_yr.time.dt.year
        # Derive the median
        da_yr_median = da_yr.median("time")
        # Derive the anomalies
        da_yr_anom = (da_yr - da_yr_median) / da_yr_median * 100
        da_yr_anom.attrs.update(
            units="%", long_name=f"Anomalies in precipitation relative to the median"
        )
    elif da.name == "temp":
        # Calculate the yearly mean
        da_yr = da.resample(time="YS").mean("time")
        # Filter out if too many nodata
        if no_data_limit is not None:
            da_yr = da_yr.where(da_nodata < no_data_limit, np.nan)
        da_yr["time"] = da_yr.time.dt.year
        # Derive the median
        da_yr_median = da_yr.median("time")
        # Derive the anomalies
        da_yr_anom = da_yr - da_yr_median
        da_yr_anom.attrs.update(
            units="°C", long_name=f"Anomalies in temperature relative to the median"
        )
    elif da.name == "Q":
        # Calculate the yearly mean
        da_yr = da.resample(time="YS").mean("time")
        # Filter out if too many nodata
        if no_data_limit is not None:
            da_yr = da_yr.where(da_nodata < no_data_limit, np.nan)
        da_yr["time"] = da_yr.time.dt.year
        # Derive the median
        da_yr_median = da_yr.median("time")
        # Derive the anomalies
        da_yr_anom = (da_yr - da_yr_median) / da_yr_median * 100
        da_yr_anom.attrs.update(
            units="%", long_name=f"Anomalies in streamflow relative to the median"
        )

    return da_yr_anom


def plot_gridded_anomalies(
    clim_dict: Dict[str, xr.DataArray],
    path_output: Union[str, Path],
    gdf_region: Optional[gpd.GeoDataFrame] = None,
    year_per_line: int = 5,
    fs: float = 6,
):
    """Plot gridded historical annual anomalies for a specific region.

    Parameters
    ----------
    clim_dict : dict(str, xr.DataArray)
        Dictionary for a specific climate variables containing one DataArray for each
        climate source. The climate source name will be used in the output filename of
        the plot.

        Supported climate variables: ["precip", "temp"]
    path_output : str or Path
        Path to the output directory where the plots are stored.
    gdf_region : gpd.GeoDataFrame, optional
        The total region of the project to add to the inset map if provided.
    year_per_line : int, optional
        Number of years to plot per line. Default is 5.
    fs : int, optional
        Font size for the plot. Default is 10.
    """
    clim_dict_anom = dict()

    # Compute the anomalies per year for each climate source
    for source, da in clim_dict.items():
        # Mask nodata
        da_yr_anom = compute_anomalies_year(da.raster.mask_nodata())
        clim_dict_anom[source] = da_yr_anom

    # Find the common time period between sources
    time_start = max([v.time.values[0] for v in clim_dict_anom.values()])
    time_end = min([v.time.values[-1] for v in clim_dict_anom.values()])
    # Sel each source
    clim_dict_anom = {
        k: v.sel(time=slice(time_start, time_end)) for k, v in clim_dict_anom.items()
    }

    # Find the min and max over all sources
    clim_max = max([v.max().values for v in clim_dict_anom.values()])
    clim_min = min([v.min().values for v in clim_dict_anom.values()])
    # Create corresponding color scale
    minmax = max(abs(clim_min), clim_max)
    divnorm = colors.TwoSlopeNorm(vmin=-minmax, vcenter=0.0, vmax=minmax)

    # Proj, extent
    proj = ccrs.PlateCarree()
    if gdf_region is not None:
        extent = np.array(gdf_region.buffer(0.01).total_bounds)[[0, 2, 1, 3]]
    else:
        extent = None

    # Plot the anomalies
    for source, da_yr_anom in clim_dict_anom.items():
        da_yr_anom = da_yr_anom.rename({"time": "year"})
        # Just loop...
        nb_years = len(da_yr_anom.year)
        fig_width = 8 if nb_years == 1 else 16
        nb_cols = year_per_line if nb_years > 1 else 1
        fig_height = np.ceil(nb_years / year_per_line) * 8
        nb_rows = int(np.ceil(nb_years / year_per_line))

        fig, ax = plt.subplots(
            nb_rows,
            nb_cols,
            figsize=(fig_width / 2.54, fig_height / 2.54),
            sharex=True,
            sharey=True,
            layout="compressed",
            subplot_kw={"projection": proj},
        )
        # Reduce white space between subplots
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        ax = [ax] if (nb_rows * nb_cols) == 1 else ax.flatten()

        for i in range(len(ax)):
            ax[i].axis("off")
            if i >= nb_years:
                continue
            im = da_yr_anom.isel(year=i).plot(
                ax=ax[i],
                cmap="bwr" if da_yr_anom.name == "temp" else "bwr_r",
                norm=divnorm,
                add_colorbar=False,
            )
            # add outline basin
            if gdf_region is not None:
                gdf_region.plot(ax=ax[i], facecolor="None")
            ax[i].set_title(
                da_yr_anom.year[i].values.item(),
                fontsize=fs + 1,
                fontweight="bold",
            )
            ax[i].set_extent(extent, crs=proj)
            # Add title in caps and bold font
            ax[i].xaxis.set_visible(True)
            ax[i].yaxis.set_visible(True)
            ax[i].set_xlabel("longitude [degree east]", fontsize=fs)
            ax[i].set_ylabel("latitude [degree north]", fontsize=fs)
            ax[i].tick_params(axis="both", labelsize=fs)

        # Add common colorbar
        cbar = fig.colorbar(
            im,
            ax=ax,
            label=f"{da_yr_anom.attrs['long_name']} [{da_yr_anom.attrs['units']}]",
            shrink=0.95,
            aspect=30,
        )
        # Change the fontsize of the colorbar label
        cbar.ax.yaxis.label.set_fontsize(fs + 1)
        # Set label wrap
        cbar.ax.yaxis.label.set_wrap(True)
        # Change the fontsize of the colorbar ticks
        cbar.ax.tick_params(labelsize=fs)

        # Save the plots
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        plt.savefig(
            join(path_output, f"gridded_anomalies_{da_yr_anom.name}_{source}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        # Close the figure
        plt.close(fig)


def plot_timeseries_anomalies(
    ds: xr.Dataset,
    path_output: Union[str, Path],
    split_year: Optional[int] = None,
    suffix: Optional[str] = None,
):
    """
    Plot the historical anomalies for a set of locations.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with the climate data.

        Supported variables: ["precip", "temp"]
    path_output : str or Path
        Path to the output directory where the plots are stored.
    split_year : int, optional
        Derive additional trends for years before and after this year.
    suffix : str, optional
        Suffix to add to the output filename.
    """
    for var in ds.data_vars:
        # Check if the data is all NaN
        if ds[var].isnull().all():
            continue
        da_yr_anom = compute_anomalies_year(ds[var], no_data_limit=100)

        # Start the plot
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot a line for each locations in different shades of grey
        da_yr_anom.plot.line(
            x="time", hue="index", ax=ax, add_legend=False, alpha=0.5, color="black"
        )

        # Add straight line for the 90th and 10th percentile
        p90 = da_yr_anom.quantile(0.9, dim="index").max("time").values
        p10 = da_yr_anom.quantile(0.1, dim="index").min("time").values

        # add a line using a constant value
        ax.axhline(p90, color="lightgrey", linestyle="--", label="90% percentile")
        ax.axhline(p10, color="lightgrey", linestyle="--", label="10% percentile")

        # also add a  line at 0% anomaly
        ax.axhline(0, color="darkgrey", linestyle="--")

        # Derive linear trend using xarray curvefit
        start_year = ds.time.dt.year.min().values
        end_year = ds.time.dt.year.max().values
        if split_year is not None:
            # Check that the split year is within the data range
            if split_year < start_year or split_year > end_year:
                print(
                    f"Split year {split_year} is not within the data range "
                    f"[{start_year}-{end_year}]. It will be ignored."
                )
                split_year = None
        if split_year is not None:
            starts = [start_year, start_year, split_year]
            ends = [end_year, split_year, end_year]
            colors = ["red", "yellow", "orange"]
        else:
            starts = [start_year]
            ends = [end_year]
            colors = ["red"]

        for start, end, color in zip(starts, ends, colors):
            da_yr_trend = da_yr_anom.sel(time=slice(start, end))
            # Remove NaN values along time dimension
            da_yr_trend = da_yr_trend.dropna("time", how="all")
            trend = da_yr_trend.curvefit(
                coords="time",
                func=linealg,
                reduce_dims="index",
                skipna=True,
            )
            # Construct the trend line and plot it
            a = trend.curvefit_coefficients.values[0]
            b = trend.curvefit_coefficients.values[1]
            trend_line = b + a * da_yr_trend.time

            # also get r_value and p_value
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                da_yr_trend.time, da_yr_trend.mean("index", skipna=True)
            )

            trend_line.plot.line(
                ax=ax,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Trend {start}-{end} (slope={a:.2f}, R$^2$={r_value**2:.2f}, p={p_value:.3f})",
            )

        # Add the legend in a white box on the top right corner
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.7)

        if var == "precip":
            legend = "precipitation anomalies [%]"
            variable = "precipitation"
        elif var == "temp":
            legend = "temperature anomalies [°C]"
            variable = "temperature"
        elif var == "Q":
            legend = "streamflow anomalies [%]"
            variable = "streamflow"
        ax.set_ylabel(legend)
        ax.set_title(
            f"Anomalies in {variable} for different locations using {suffix} data"
        )

        # Save the plot
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        plt.savefig(
            join(path_output, f"timeseries_anomalies_{var}_{suffix}.png"),
            bbox_inches="tight",
            dpi=300,
        )

        # Close the figure
        plt.close(fig)
