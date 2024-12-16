"""Utility plot functions for gridded climate data."""

import os
from os.path import join
from pathlib import Path
from typing import Union, Dict, Optional

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import hydromt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

__all__ = ["plot_gridded_precip"]


def create_grid_from_geom(
    geom: gpd.GeoDataFrame, res: float, align: bool = True
) -> xr.DataArray:
    """
    Create regular grid from a geometry.

    Taken from hydromt.GridModel.setup_grid. Should be available in hydromt version 1.0
    when this is moved to a workflow.

    """
    xmin, ymin, xmax, ymax = geom.total_bounds
    res = abs(res)
    if align:
        xmin = round(xmin / res) * res
        ymin = round(ymin / res) * res
        xmax = round(xmax / res) * res
        ymax = round(ymax / res) * res
    xcoords = np.linspace(
        xmin + res / 2,
        xmax - res / 2,
        num=round((xmax - xmin) / res),
        endpoint=True,
    )
    ycoords = np.flip(
        np.linspace(
            ymin + res / 2,
            ymax - res / 2,
            num=round((ymax - ymin) / res),
            endpoint=True,
        )
    )

    coords = {"y": ycoords, "x": xcoords}
    grid = hydromt.raster.full(
        coords=coords,
        nodata=1,
        dtype=np.uint8,
        name="mask",
        attrs={},
        crs=geom.crs,
        lazy=False,
    )

    grid = grid.raster.geometry_mask(geom, all_touched=True)
    grid.name = "mask"

    return grid


def plot_gridded_precip(
    precip_dict: Dict[str, xr.DataArray],
    path_output: Union[str, Path],
    gdf_region: Optional[gpd.GeoDataFrame] = None,
    gdf_river: Optional[gpd.GeoDataFrame] = None,
    fs: int = 8,
    colorbar_shrink: float = 0.9,
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
    gdf_river : gpd.GeoDataFrame, optional
        The river network of the project to add to the inset map if provided.
        Optional variable for styling: `strord`.
    fs : int, optional
        Font size for the labels. Default is 8.
    colorbar_shrink : float, optional
        Shrink the colorbar size. Default is 0.9.
    """
    # Mask no data values
    for k, v in precip_dict.items():
        precip_dict[k] = precip_dict[k].raster.mask_nodata()

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
    precip_dict = {
        k: v.resample(time="YE").sum(skipna=False) for k, v in precip_dict.items()
    }
    # Compute the mean of the annual precipitation
    precip_dict = {k: v.mean("time") for k, v in precip_dict.items()}

    # Reproject to the same grid
    if gdf_region is not None:
        grid = create_grid_from_geom(gdf_region, res=0.008333, align=False)
        for k, v in precip_dict.items():
            v_out = v.raster.reproject_like(grid, method="nearest")
            # Mask values outside of the grid mask
            v_out = v_out.where(grid, v_out.raster.nodata)
            precip_dict[k] = v_out

    # Find the max and min over each source
    max_precip = max([v.max().values for v in precip_dict.values()])
    min_precip = min([v.min().values for v in precip_dict.values()])

    # Proj, extent and zoom level
    # we assume the model maps are in the geographic CRS EPSG:4326
    proj = ccrs.PlateCarree()
    # adjust zoomlevel and figure size to your basis size & aspect
    # zoom_level = 10
    if gdf_region is not None:
        extent = np.array(gdf_region.buffer(0.02).total_bounds)[[0, 2, 1, 3]]
    else:
        extent = None

    # Plot the precipitation in one figure
    fig_width = 8 if len(precip_dict) == 1 else 16
    nb_cols = 2 if len(precip_dict) > 1 else 1
    fig_height = np.ceil(len(precip_dict) / 2) * 8
    nb_rows = int(np.ceil(len(precip_dict) / 2))

    fig, ax = plt.subplots(
        nb_rows,
        nb_cols,
        figsize=(fig_width / 2.54, fig_height / 2.54),
        sharex=True,
        sharey=True,
        layout="compressed",
        subplot_kw={"projection": proj},
    )
    ax = [ax] if (nb_rows * nb_cols) == 1 else ax.flatten()

    for i in range(len(ax)):
        if i >= len(precip_dict):
            ax[i].axis("off")
            continue
        k = list(precip_dict.keys())[i]
        v = precip_dict[k]
        if extent is not None:
            ax[i].set_extent(extent, crs=proj)
            # add sat background image
            # ax[i].add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)
        im = v.plot(
            ax=ax[i],
            label=k,
            vmin=min_precip,
            vmax=max_precip,
            cmap="viridis",
            add_colorbar=False,
        )
        # add rivers
        if gdf_river is not None:
            gdf_river.plot(
                ax=ax[i], linewidth=gdf_river["strord"] / 2, color="blue", label="river"
            )
        # add outline basin
        if gdf_region is not None:
            gdf_region.plot(ax=ax[i], facecolor="None")
        # Add title in caps and bold font
        ax[i].set_title(k.upper(), fontsize=fs + 2, fontweight="bold")
        ax[i].xaxis.set_visible(True)
        ax[i].yaxis.set_visible(True)
        ax[i].set_xlabel("Longitude", fontsize=fs)
        ax[i].set_ylabel("Latitude", fontsize=fs)
        ax[i].tick_params(axis="both", labelsize=fs)

    # Add common colorbar
    cbar = fig.colorbar(
        im,
        ax=ax,
        label="precipitation [mm/year]",
        shrink=colorbar_shrink,
        aspect=30,
    )
    # Change the fontsize of the colorbar label
    cbar.ax.yaxis.label.set_fontsize(fs + 1)
    # Change the fontsize of the colorbar ticks
    cbar.ax.tick_params(labelsize=fs)

    # fig.tight_layout()
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fig.savefig(
        join(path_output, "mean_annual_precipitation.png"),
        bbox_inches="tight",
        dpi=300,
    )
