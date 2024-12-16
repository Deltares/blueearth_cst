"""Plot functions for scalar and gridded future climate projections."""

import os
from os.path import join, exists, dirname
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from typing import List, Union, Optional

__all__ = ["plot_scalar_anomaly", "plot_gridded_anomaly"]

COLORS = {
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp585": "#980002",
}


def plot_scalar_anomaly(
    data_hist: pd.DataFrame,
    data_fut: List[pd.DataFrame],
    scenario_names: List[str],
    title: str,
    y_label: str,
    monthly: bool = False,
    figure_filename: Optional[Union[str, Path]] = None,
):
    """
    Plot anomaly or absolute change series for a scalar variable.

    Parameters
    ----------
    data_hist : pd.DataFrame
        Historical data (absolute or anomaly).
        Should contain the columns 0.5, 0.05, and 0.95 for mean and 5% and 95%
        quantiles.
    data_fut : List[pd.DataFrame]
        List of future data (absolute or anomaly) for each scenario.
        Should contain the columns 0.5, 0.05, and 0.95 for median and 5% and 95%
        quantiles.
    scenario_names : List[str]
        List of scenario names. Same order as the data_fut list. Used for color
        selection. Supported scenarios are 'ssp126', 'ssp245', 'ssp370', 'ssp585'.
    title : str
        Title of the plot.
    y_label : str
        Label for the y-axis. Eg. mm/year for absolute precipitation, or anomaly (%) for
        precipitation anomaly etc.
    monthly : bool, optional
        If True, the x-axis will be labelled with month names. If False, the x-axis will
        be labelled with years.
    figure_filename : Union[str, Path], optional
        If provided, the figure will be saved to this file. If not provided, the figure
        will only be displayed (e.g. in the current Jupyter notebook).
    """
    # Fontsize
    fs = 8

    # Create figure
    fig, ax = plt.subplots(figsize=(16 / 2.54, 12 / 2.54))
    ax.set_title(title, fontsize=fs + 2)

    # Historical data
    ax.fill_between(
        x=data_hist.index,
        y1=data_hist[0.95],
        y2=data_hist[0.05],
        color="lightgrey",
        alpha=0.5,
    )
    ax.plot(
        data_hist.index,
        data_hist[0.5],
        color="darkgray",
        label="historical multi-model median",
    )

    # Future data
    for i, data in enumerate(data_fut):
        ax.fill_between(
            x=data.index,
            y1=data[0.95],
            y2=data[0.05],
            color=COLORS[scenario_names[i]],
            alpha=0.5,
        )
        ax.plot(
            data.index,
            data[0.5],
            color=COLORS[scenario_names[i]],
            label=f"{scenario_names[i]} multi-model median",
        )

    # Labels and legend
    ax.set_xlabel("", fontsize=fs)
    if monthly:
        ax.set_xticks(data_hist.index)
        ax.set_xticklabels(
            np.array(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        )
    ax.set_ylabel(y_label, fontsize=fs)
    ax.legend(fontsize=fs)
    ax.grid(True)

    # Save figure
    if figure_filename is not None:
        # Create the directory if it does not exist
        if not exists(dirname(figure_filename)):
            os.makedirs(dirname(figure_filename))
        fig.savefig(figure_filename, dpi=300, bbox_inches="tight")

    plt.close()


def plot_gridded_anomaly_month(
    da: xr.DataArray,
    title: str,
    unit: str,
    y_label: str = "latitude [degree north]",
    x_label: str = "longitude [degree east]",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "bwr",
    use_diverging_cmap: bool = True,
    figure_filename: Optional[Union[str, Path]] = None,
):
    """
    Plot absolute monthly gridded anomaly for a specific scenario and time horizon.

    Parameters
    ----------
    da : xr.DataArray
        Gridded data array with the dimensions "month", "lat", "lon".
    title : str
        Title of the plot.
    unit : str
        Unit of the data. Eg. mm/year for precipitation, or degree Celsius for temperature.
    y_label : str, optional
        Label for the y-axis. Default is "latitude [degree north]".
    x_label : str, optional
        Label for the x-axis. Default is "longitude [degree east]".
    vmin : float, optional
        Minimum value for the colorbar. If not provided, the minimum value of the data
        array will be used.
    vmax : float, optional
        Maximum value for the colorbar. If not provided, the maximum value of the data
        array will be used.
    cmap : str, optional
        Colormap to use. Default is "bwr".
    use_diverging_cmap : bool, optional
        If True, a diverging colormap centered around zero will be used.
        Default is True.
    figure_filename : Union[str, Path], optional
        If provided, the figure will be saved to this file. If not provided, the figure
        will only be displayed (e.g. in the current Jupyter notebook).
    """

    # Fontsize
    fs = 8

    # Create figure
    plt.figure(figsize=(16 / 2.54, 18 / 2.54))

    # Create a colorbar using TwoSLopeNorm if use_diverging_cmap is True
    vmin = vmin if vmin is not None else np.min(da.values)
    vmax = vmax if vmax is not None else np.max(da.values)
    if use_diverging_cmap:
        if vmin > 0:
            vmin = -vmax
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vcenter=0,
            vmax=vmax,
        )
    else:
        norm = colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        )

    # Update the array attributes for correct labels/legend
    da.attrs.update(
        long_name=title,
        units=unit,
    )

    # Facetplots
    g = da.plot(
        x="lon",
        y="lat",
        col="month",
        col_wrap=3,
        cmap=cmap,
        norm=norm,
    )

    # Labels
    g.set_xlabels(x_label, fontsize=fs)
    g.set_ylabels(y_label, fontsize=fs)

    # Save figure
    if figure_filename is not None:
        # Create the directory if it does not exist
        if not exists(dirname(figure_filename)):
            os.makedirs(dirname(figure_filename))
        plt.savefig(figure_filename, dpi=300, bbox_inches="tight")

    plt.close()


def plot_gridded_anomaly(
    da: xr.DataArray,
    title: str,
    legend: str,
    y_label: str = "latitude [degree north]",
    x_label: str = "longitude [degree east]",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "bwr",
    use_diverging_cmap: bool = True,
    add_background_image: bool = True,
    figure_filename: Optional[Union[str, Path]] = None,
):
    """
    Plot the absolute gridded anomaly for a specific scenario and time horizon.

    Parameters
    ----------
    da : xr.DataArray
        Gridded data array with the dimensions "lat", "lon".
    title : str
        Title of the plot.
    legend : str
        Legend of the plot cbar.
    y_label : str, optional
        Label for the y-axis. Default is "latitude [degree north]".
    x_label : str, optional
        Label for the x-axis. Default is "longitude [degree east]".
    vmin : float, optional
        Minimum value for the colorbar. If not provided, the minimum value of the data
        array will be used.
    vmax : float, optional
        Maximum value for the colorbar. If not provided, the maximum value of the data
        array will be used.
    cmap : str, optional
        Colormap to use. Default is "bwr".
    use_diverging_cmap : bool, optional
        If True, a diverging colormap centered around zero will be used.
        Default is True.
    add_background_image : bool, optional
        If True, a background satellite image will be added. Default is True.
    figure_filename : Union[str, Path], optional
        If provided, the figure will be saved to this file. If not provided, the figure
        will only be displayed (e.g. in the current Jupyter notebook).
    """

    plt.style.use("seaborn-v0_8-whitegrid")  # set nice style
    # we assume the model maps are in the geographic CRS EPSG:4326
    proj = ccrs.PlateCarree()
    # adjust zoomlevel and figure size to your basis size & aspect
    zoom_level = 8
    fs = 8
    figsize = (16 / 2.54, 12 / 2.54)

    # initialize image with geoaxes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=proj)
    extent = np.array(da.raster.box.buffer(0.5).total_bounds)[[0, 2, 1, 3]]
    ax.set_extent(extent, crs=proj)
    # add sat background image
    if add_background_image:
        ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

    # Create a colorbar using TwoSLopeNorm if use_diverging_cmap is True
    vmin = vmin if vmin is not None else np.min(da.values)
    vmax = vmax if vmax is not None else np.max(da.values)
    if use_diverging_cmap:
        if vmin > 0:
            vmin = -vmax
        norm = colors.TwoSlopeNorm(
            vmin=vmin,
            vcenter=0,
            vmax=vmax,
        )
    else:
        norm = colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        )

    da.plot(
        transform=proj,
        ax=ax,
        zorder=1,
        cbar_kwargs=dict(
            aspect=30,
            shrink=0.8,
            label=legend,
        ),
        cmap=cmap,
        norm=norm,
    )

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.set_xlabel(x_label, fontsize=fs)
    _ = ax.set_title(title, fontsize=fs + 2)

    # Save figure
    if figure_filename is not None:
        # Create the directory if it does not exist
        if not exists(dirname(figure_filename)):
            os.makedirs(dirname(figure_filename))
        fig.savefig(figure_filename, dpi=300, bbox_inches="tight")

    plt.close()
