"""Utility function to plot a wflow model map."""

import xarray as xr
import numpy as np
import geopandas as gpd
import os
from pathlib import Path
from typing import Union, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe

# plot maps dependencies
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

# import descartes  # required to plot polygons
import cartopy.io.img_tiles as cimgt

from hydromt.gis_utils import utm_crs
from hydromt_wflow import WflowModel


__all__ = ["plot_map_model", "plot_map"]


def plot_map(
    figname: str,
    plot_dir: Union[str, Path],
    basins: gpd.GeoDataFrame,
    da: Optional[xr.DataArray] = None,
    rivers: Optional[gpd.GeoDataFrame] = None,
    subregions: Dict[str, gpd.GeoDataFrame] = {},
    # outlets: Optional[gpd.GeoDataFrame] = None,
    gauges: Dict[str, gpd.GeoDataFrame] = {},
    lakes: Optional[gpd.GeoDataFrame] = None,
    reservoirs: Optional[gpd.GeoDataFrame] = None,
    glaciers: Optional[gpd.GeoDataFrame] = None,
    buffer_km: Optional[float] = 2.0,
    annotate_regions: bool = False,
    **kwargs,
):
    """
    Plot wflow model map for one variable.

    Output map will be saved in plot_dir/figname.png.

    Parameters
    ----------
    figname : str
        Name of the output figure file.
    plot_dir : Path
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_root folder.
    basins : gpd.GeoDataFrame
        GeoDataFrame with the basin boundaries.
    da : xr.DataArray
        Forcing DataArray to plot. The annual mean will be plotted.
    rivers : gpd.GeoDataFrame
        GeoDataFrame with the river network. The stream order should be in the
        "strord" column.
    subregions : dict of gpd.GeoDataFrame
        Dict of GeoDataFrame with different subregion boundaries. The keys are used in
        the legend. If `name` is in the columns, this will be used to annotate the regions.
    gauges : dict of gpd.GeoDataFrame
        Dict of GeoDataFrame with the different gauges/point locations. The keys are
        used in the legend. If `name` is in the columns and/or 'elevtn', this will be
        used to annotate the locations.
    lakes : gpd.GeoDataFrame
        GeoDataFrame with the lake polygons.
    reservoirs : gpd.GeoDataFrame
        GeoDataFrame with the reservoir polygons.
    glaciers : gpd.GeoDataFrame
        GeoDataFrame with the glacier polygons.
    annotate_regions : bool
        If True, annotate the basins and subregions with the region name.
    kwargs : dict
        Additional keyword arguments to pass to da.plot()
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.style.use("seaborn-v0_8-whitegrid")  # set nice style
    # we assume the model maps are in the geographic CRS EPSG:4326
    proj = ccrs.PlateCarree()
    # adjust zoomlevel and figure size to your basis size & aspect
    zoom_level = 10
    figsize = (10, 8)
    shaded = (
        True  # shaded elevation (looks nicer with more pixels (e.g.: larger basins))!
    )

    # initialize image with geoaxes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=proj)
    extent = da.raster.box
    extent = (
        extent.to_crs(utm_crs(da.raster.bounds))
        .buffer(buffer_km * 1000)
        .to_crs(da.raster.crs)
    )
    extent = np.array(extent.total_bounds)[[0, 2, 1, 3]]
    ax.set_extent(extent, crs=proj)

    # add sat background image
    ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.4)

    # plot da variables.
    if da is not None:
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
            rgb = xr.DataArray(
                dims=("y", "x", "rgb"), data=_rgb, coords=da.raster.coords
            )
            rgb = xr.where(np.isnan(da), np.nan, rgb)
            rgb.plot.imshow(transform=proj, ax=ax, zorder=2)

    # plot rivers with increasing width with stream order
    if rivers is not None:
        rivers.plot(
            ax=ax, linewidth=rivers["strord"] / 2, color="blue", zorder=3, label="river"
        )

    # plot the basin boundary
    basins.boundary.plot(ax=ax, color="k", linewidth=0.3, zorder=4)
    # if several basins annotate
    if annotate_regions:
        basins.apply(
            lambda x: ax.annotate(
                text=f"basin {x.name}",
                xy=x.geometry.representative_point().coords[0],
                xytext=(2.0, 2.0),
                textcoords="offset points",
                # ha='left',
                # va = 'top',
                fontsize=7,
                fontweight="bold",
                color="black",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            ),
            axis=1,
        )

    # plot various vector layers if present
    # subregions
    if len(subregions) > 0:
        for key, subregion in subregions.items():
            subregion.boundary.plot(
                ax=ax, color="r", linewidth=0.5, zorder=5, label=key
            )
            if annotate_regions:
                subregion.apply(
                    lambda x: ax.annotate(
                        text=f"{key} {x.name}",
                        xy=x.geometry.representative_point().coords[0],
                        xytext=(2.0, 2.0),
                        textcoords="offset points",
                        # ha='left',
                        # va = 'top',
                        fontsize=7,
                        fontweight="bold",
                        color="r",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    ),
                    axis=1,
                )

    # Gauges
    if len(gauges) > 0:

        def _annotation_text(x):
            text = str(x.name)
            if "name" in x:
                text = text + f" - {x['name']}"
            if "elevtn" in x:
                text = text + f" - {x['elevtn']}m"
            return text

        for k, v in gauges.items():
            v.plot(
                ax=ax,
                marker="d",
                markersize=25,
                # facecolor="k",
                zorder=6,
                label=k,
            )
            v.apply(
                lambda x: ax.annotate(
                    text=_annotation_text(x),
                    xy=x.geometry.coords[0],
                    xytext=(2.0, 2.0),
                    textcoords="offset points",
                    # ha='left',
                    # va = 'top',
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                ),
                axis=1,
            )
    patches = (
        []
    )  # manual patches for legend, see https://github.com/geopandas/geopandas/issues/660
    if lakes is not None:
        kwargs = dict(
            facecolor="lightblue", edgecolor="black", linewidth=1, label="lakes"
        )
        lakes.plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))
    if reservoirs is not None:
        kwargs = dict(
            facecolor="blue", edgecolor="black", linewidth=1, label="reservoirs"
        )
        reservoirs.plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))
    if glaciers is not None:
        kwargs = dict(facecolor="grey", edgecolor="grey", linewidth=1, label="glaciers")
        glaciers.plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel(f"latitude [degree north]")
    ax.set_xlabel(f"longitude [degree east]")
    _ = ax.set_title(f"")
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
    gauges = {}

    model_gauges = mod.geoms.get("gauges", None)
    if model_gauges is not None:
        gauges["gauges"] = model_gauges

    if gauges_name is not None:
        output_locs = mod.geoms.get(gauges_name, None)
        if output_locs is not None:
            if "station_name" in output_locs.columns:
                output_locs = output_locs.rename(columns={"station_name": "name"})
            gauges["output locs"] = output_locs

    plot_map(
        da=da,
        figname=figname,
        plot_dir=join(mod.root, "plots") if plot_dir is None else plot_dir,
        basins=mod.basins,
        rivers=mod.rivers,
        gauges=gauges,
        lakes=mod.geoms.get("lakes", None),
        reservoirs=mod.geoms.get("reservoirs", None),
        glaciers=mod.geoms.get("glaciers", None),
        **kwargs,
    )
