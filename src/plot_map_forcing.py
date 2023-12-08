# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:23:11 2022

@author: bouaziz
"""

# plot map

import xarray as xr
import numpy as np
from os.path import basename, join
import os
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import colors

# plot maps dependencies
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

# import descartes  # required to plot polygons
import cartopy.io.img_tiles as cimgt

from hydromt_wflow import WflowModel

def plot_map_model(mod, da, figname, gauges_name):
    # read/derive river geometries
    gdf_riv = mod.rivers
    # read/derive model basin boundary
    gdf_bas = mod.basins
    plt.style.use("seaborn-whitegrid")  # set nice style
    # we assume the model maps are in the geographic CRS EPSG:4326
    proj = ccrs.PlateCarree()
    # adjust zoomlevel and figure size to your basis size & aspect
    zoom_level = 10
    figsize = (10, 8)
    shaded = (
        False  # shaded elevation (looks nicer with more pixels (e.g.: larger basins))!
    )

    # initialize image with geoaxes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=proj)
    extent = np.array(da.raster.box.buffer(0.02).total_bounds)[[0, 2, 1, 3]]
    ax.set_extent(extent, crs=proj)

    # add sat background image
    ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

    # plot da variables.
    da.plot(
        transform=proj,
        ax=ax,
        zorder=1,
        cbar_kwargs=dict(aspect=30, shrink=0.8),
    )  # **kwargs)
    # plot elevation with shades
    if shaded:
        ls = colors.LightSource(azdeg=315, altdeg=45)
        dx, dy = da.raster.res
        _rgb = ls.shade(
            da.fillna(0).values,
            norm=kwargs["norm"],
            cmap=kwargs["cmap"],
            blend_mode="soft",
            dx=dx,
            dy=dy,
            vert_exag=200,
        )
        rgb = xr.DataArray(dims=("y", "x", "rgb"), data=_rgb, coords=da.raster.coords)
        rgb = xr.where(np.isnan(da), np.nan, rgb)
        rgb.plot.imshow(transform=proj, ax=ax, zorder=2)

    # plot rivers with increasing width with stream order
    gdf_riv.plot(
        ax=ax, linewidth=gdf_riv["strord"] / 2, color="blue", zorder=3, label="river"
    )
    # plot the basin boundary
    gdf_bas.boundary.plot(ax=ax, color="k", linewidth=0.3)
    # plot various vector layers if present
    if "gauges" in mod.geoms:
        mod.geoms["gauges"].plot(
            ax=ax, marker="d", markersize=25, facecolor="k", zorder=5, label="gauges"
        )
    if gauges_name in mod.geoms:
        mod.geoms[gauges_name].plot(
            ax=ax,
            marker="d",
            markersize=25,
            facecolor="blue",
            zorder=5,
            label="output locs",
        )
    patches = (
        []
    )  # manual patches for legend, see https://github.com/geopandas/geopandas/issues/660
    if "lakes" in mod.geoms:
        kwargs = dict(
            facecolor="lightblue", edgecolor="black", linewidth=1, label="lakes"
        )
        mod.geoms["lakes"].plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))
    if "reservoirs" in mod.geoms:
        kwargs = dict(
            facecolor="blue", edgecolor="black", linewidth=1, label="reservoirs"
        )
        mod.geoms["reservoirs"].plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))
    if "glaciers" in mod.geoms:
        kwargs = dict(facecolor="grey", edgecolor="grey", linewidth=1, label="glaciers")
        mod.geoms["glaciers"].plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel(f"latitude [degree north]")
    ax.set_xlabel(f"longitude [degree east]")
    _ = ax.set_title(f"wflow base map")
    legend = ax.legend(
        handles=[*ax.get_legend_handles_labels()[0], *patches],
        title="Legend",
        loc="lower right",
        frameon=True,
        framealpha=0.7,
        edgecolor="k",
        facecolor="white",
    )

    # save figure
    plt.savefig(
        os.path.join(Folder_plots, f"{figname}.png"), dpi=300, bbox_inches="tight"
    )


def plot_forcing(
    wflow_root: Union[str, Path],
    plot_dir = None,
    gauges_name: str = None,
):
    """
    Plot the wflow forcing in separate maps.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder
    plot_dir : str, optional
        Path to the output folder. If None (default), create a folder "plots" 
        in the wflow_root folder.
    gauges_name : str, optional
        Name of the gauges to plot. If None (default), no gauges are plot.
    """
    mod = WflowModel(wflow_root, mode="r")

    # If plotting dir is None, create
    if plot_dir is None:
        plot_dir = os.path.join(wflow_root, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Forcing variables to plot
    forcing_vars = {
        "precip": {"long_name": "precipitation", "unit": "mm y$^{-1}$"},
        "pet": {"long_name": "potential evap.", "unit": "mm y$^{-1}$"},
        "temp": {"long_name": "temperature", "unit": "degC"},
    }

    # plot mean annual precip temp and potential evap.
    for forcing_var, forcing_char in forcing_vars.items():
        print(forcing_var, forcing_char)
        if forcing_var == "temp":
            da = mod.forcing[forcing_var].resample(time="A").mean("time").mean("time")
        else:
            da = mod.forcing[forcing_var].resample(time="A").sum("time").mean("time")
            da = da.where(da>0)
        da = da.where(mod.grid["wflow_subcatch"] >= 0)
        da.attrs.update(long_name=forcing_char["long_name"], units=forcing_char["unit"])
        figname = f"{forcing_var}"
        plot_map_model(mod, da, figname, gauges_name)

if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        # Parse snake options
        project_dir = sm.params.project_dir
        gauges_fn = sm.params.gauges_fid
        gauges_name = basename(gauges_fn).split(".")[0]

        Folder_plots = f"{project_dir}/plots/wflow_model_performance"
        root = f"{project_dir}/hydrology_model"

        plot_forcing(
            wflow_root = root,
            plot_dir = Folder_plots,
            gauges_name = gauges_name,
        )
    else:
        plot_forcing(
            wflow_root = join(os.getcwd(), "examples", "my_project", "hydrology_model")
        )
