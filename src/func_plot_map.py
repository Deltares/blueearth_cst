"""Utility function to plot a wflow model map."""

import xarray as xr
import numpy as np
import os
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe

# plot maps dependencies
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

# import descartes  # required to plot polygons
import cartopy.io.img_tiles as cimgt

from hydromt_wflow import WflowModel


__all__ = ["plot_map_model"]


def plot_map_model(
    mod: WflowModel,
    da: xr.DataArray,
    figname: str,
    plot_dir: Union[str, Path] = None,
    gauges_name: str = None,
    **kwargs,
):
    """
    Plot wflow model forcing map for one variable.

    Output map will be saved in plot_dir/figname.png.

    Parameters
    ----------
    mod : WflowModel
        wflow model instance used to also plot basins, rivers, etc.
    da : xr.DataArray
        Forcing DataArray to plot. The annual mean will be plotted.
    figname : str
        Name of the output figure file.
    plot_dir : Path
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_root folder.
    gauges_name : str, optional
        Name of the gauges in model to plot. If None (default), no gauges are plot.
    kwargs : dict
        Additional keyword arguments to pass to da.plot()
    """
    # If plotting dir is None, create
    if plot_dir is None:
        plot_dir = os.path.join(mod.root, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # read/derive river geometries
    gdf_riv = mod.rivers
    # read/derive model basin boundary
    gdf_bas = mod.basins
    plt.style.use("seaborn-v0_8-whitegrid")  # set nice style
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
        **kwargs,
    )
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
        if "station_name" in mod.geoms[gauges_name].columns:
            mod.geoms[gauges_name].apply(
                lambda x: ax.annotate(
                    text=x["station_name"],
                    xy=x.geometry.coords[0],
                    xytext=(2.0, 2.0),
                    textcoords="offset points",
                    # ha='left',
                    # va = 'top',
                    fontsize=5,
                    fontweight="bold",
                    color="black",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                ),
                axis=1,
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
    plt.savefig(os.path.join(plot_dir, f"{figname}.png"), dpi=300, bbox_inches="tight")
