# -*- coding: utf-8 -*-
"""Plot the wflow basin, rivers, gauges/outlets and DEM on a map.

Created 2022-01-13 (@author: bouaziz); refactored in R3 into a guarded
function so the rule's output can be tee'd to its log and the module stays
importable.
"""

import os
from os.path import basename

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from blueearth_cst.shared.snake_utils import save_figure


def gauges_layer_name(gauges_fn):
    """Staticgeoms layer name for ``output_locations``, or None when unset.

    R07 O-08. ``output_locations`` is "unset" in two spellings: YAML ``null``
    (Python ``None``) and the repo's unquoted ``None`` sentinel, which
    ``yaml.safe_load`` parses to the **string** ``"None"``. The pre-R07 guard
    tested only the former and then derived ``gauges_None`` from the latter — a
    layer name that can never exist in ``geoms``, so the gauges were silently
    dropped from the map instead of being deliberately omitted. Both spellings
    now return None. The sentinel's on-disk spelling is unchanged; only this
    reader learns to recognise it.
    """
    if gauges_fn is None or str(gauges_fn) == "None":
        return None
    return f'gauges_{basename(str(gauges_fn)).split(".")[0]}'


def plot_basin_map(project_dir, gauges_fn, plot_dir=None):
    """Render basin_area.png (DEM + rivers + basin + outlets/gauges/waterbodies)."""
    from hydromt_wflow import WflowSbmModel

    gauges_name = gauges_layer_name(gauges_fn)

    if plot_dir is None:
        # R07 B10: basin_area.png depicts the MODEL, not its evaluation, so it
        # sits at the model root's plots/ — not under evaluation/ (P1).
        plot_dir = f"{project_dir}/hydrology_model/plots"
    root = f"{project_dir}/hydrology_model"

    mod = WflowSbmModel(root, mode="r")

    # read and mask the model elevation
    da = mod.staticmaps.data["land_elevation"].raster.mask_nodata()
    da.attrs.update(long_name="elevation", units="m")
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
    shaded = False  # shaded elevation (looks nicer with more pixels)

    # initialize image with geoaxes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=proj)
    extent = np.array(da.raster.box.buffer(0.02).total_bounds)[[0, 2, 1, 3]]
    ax.set_extent(extent, crs=proj)

    # add sat background image
    ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

    ## plot elevation
    # create nice colormap
    vmin, vmax = da.quantile([0.0, 0.98]).compute()
    c_dem = plt.cm.terrain(np.linspace(0.25, 1, 256))
    cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    kwargs = dict(cmap=cmap, norm=norm)
    # plot 'normal' elevation
    da.plot(
        transform=proj, ax=ax, zorder=1, cbar_kwargs=dict(aspect=30, shrink=0.8), **kwargs
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
    geoms = mod.geoms.data
    if "outlets" in geoms:
        geoms["outlets"].plot(
            ax=ax, marker="d", markersize=25, facecolor="k", zorder=5, label="outlets"
        )
    if gauges_name is not None and gauges_name in geoms:
        geoms[gauges_name].plot(
            ax=ax,
            marker="d",
            markersize=25,
            facecolor="blue",
            zorder=5,
            label="output locs",
        )
        if "station_name" in geoms[gauges_name].columns:
            geoms[gauges_name].apply(
                lambda x: ax.annotate(
                    text=x["station_name"],
                    xy=x.geometry.coords[0],
                    xytext=(2.0, 2.0),
                    textcoords="offset points",
                    fontsize=5,
                    fontweight="bold",
                    color="black",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                ),
                axis=1,
            )

    # manual patches for legend (geopandas/geopandas#660)
    patches = []
    if "lakes" in geoms:
        kwargs = dict(facecolor="lightblue", edgecolor="black", linewidth=1, label="lakes")
        geoms["lakes"].plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))
    if "reservoirs" in geoms:
        kwargs = dict(facecolor="blue", edgecolor="black", linewidth=1, label="reservoirs")
        geoms["reservoirs"].plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))
    if "glaciers" in geoms:
        kwargs = dict(facecolor="grey", edgecolor="grey", linewidth=1, label="glaciers")
        geoms["glaciers"].plot(ax=ax, zorder=4, **kwargs)
        patches.append(mpatches.Patch(**kwargs))

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_ylabel("latitude [degree north]")
    ax.set_xlabel("longitude [degree east]")
    _ = ax.set_title("")
    ax.legend(
        handles=[*ax.get_legend_handles_labels()[0], *patches],
        title="Legend",
        loc="lower right",
        frameon=True,
        framealpha=0.7,
        edgecolor="k",
        facecolor="white",
    )

    # save figure
    save_figure(
        os.path.join(plot_dir, "basin_area.png"), dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            plot_basin_map(
                project_dir=sm.params.project_dir,
                gauges_fn=sm.params.output_locations,
            )
