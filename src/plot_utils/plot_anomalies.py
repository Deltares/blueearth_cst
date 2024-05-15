"""Utilities plot functions for gridded and scalar anomalies."""

import os
from os.path import join
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

import xarray as xr
import numpy as np
import scipy.stats as stats
import geopandas as gpd

__all__ = ["plot_gridded_anomalies", "plot_timeseries_anomalies"]


def linealg(x, a, b):
    """Linear function for curvefit."""
    return a * x + b


def compute_anomalies_year(da: xr.DataArray) -> xr.DataArray:
    """Resample timeseries to year and compute anomalies for a given DataArray.

    Supported variables are "precip" and "temp".

    Parameters
    ----------
    da : xr.DataArray
        DataArray with the climate data.

    Returns
    -------
    da_yr_anom: xr.DataArray
        DataArray with the anomalies.
    """
    if da.name == "precip":
        # Calculate the yearly sum
        da_yr = da.resample(time="YS").sum("time")
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
        da_yr["time"] = da_yr.time.dt.year
        # Derive the median
        da_yr_median = da_yr.median("time")
        # Derive the anomalies
        da_yr_anom = da_yr - da_yr_median
        da_yr_anom.attrs.update(
            units="°C", long_name=f"Anomalies in temperature relative to the median"
        )

    return da_yr_anom


def plot_gridded_anomalies(
    ds: xr.Dataset,
    path_output: Union[str, Path],
    suffix: Optional[str] = None,
    gdf_region: Optional[gpd.GeoDataFrame] = None,
):
    """Plot gridded historical annual anomalies for a specific region.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with the gridded climate data.

        Supported variables: ["precip", "temp"]
    path_output : str or Path
        Path to the output directory where the plots are stored.
    suffix : str, optional
        Suffix to add to the output filename.
    gdf_region : gpd.GeoDataFrame, optional
        The total region of the project to add to the inset map if provided.
    """

    # Mask nodata
    ds = ds.raster.mask_nodata()
    for var in ds.data_vars:
        da_yr_anom = compute_anomalies_year(ds[var])

        # Plot the anomalies
        plt.style.use("seaborn-v0_8-whitegrid")  # set nice style
        fig = plt.figure()
        da_yr_anom = da_yr_anom.rename({"time": "year"})
        minmax = max(abs(np.nanmin(da_yr_anom.values)), np.nanmax(da_yr_anom.values))
        divnorm = colors.TwoSlopeNorm(vmin=-minmax, vcenter=0.0, vmax=minmax)

        p = da_yr_anom.plot(
            x=da_yr_anom.raster.x_dim,
            y=da_yr_anom.raster.y_dim,
            col="year",
            col_wrap=5,
            cmap="bwr",
            norm=divnorm,
        )
        p.set_axis_labels("longitude [degree east]", "latitude [degree north]")

        if gdf_region is not None:
            for ax in p.axes.flatten():
                gdf_region.plot(ax=ax, facecolor ="None")

        # Save the plots
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        plt.savefig(
            join(path_output, f"gridded_anomalies_{var}_{suffix}.png"),
            bbox_inches="tight",
            dpi=300,
        )


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
        da_yr_anom = compute_anomalies_year(ds[var])

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
        ax.axhline(p90, color="grey", linestyle="--", label="90% percentile")
        ax.axhline(p10, color="grey", linestyle="--", label="10% percentile")

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
            trend = da_yr_trend.curvefit(
                coords="time",
                func=linealg,
                reduce_dims="index",
            )
            # Construct the trend line and plot it
            a = trend.curvefit_coefficients.values[0]
            b = trend.curvefit_coefficients.values[1]
            trend_line = b + a * da_yr_trend.time

            # also get r_value and p_value
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                da_yr_trend.time, da_yr_trend.mean("index")
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
