# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:23:11 2022

@author: bouaziz
"""

#%% plot map

import xarray as xr
import numpy as np
from os.path import basename
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe

# plot maps dependencies
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

# import descartes  # required to plot polygons
import cartopy.io.img_tiles as cimgt

from hydromt_wflow import WflowModel

project_dir = snakemake.params.project_dir
gauges_fn = snakemake.params.output_locations
gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'

Folder_plots = f"{project_dir}/plots/wflow_model_performance"
root = f"{project_dir}/hydrology_model"


mod = WflowModel(root, mode="r")

# read and mask the model elevation
da = mod.grid["wflow_dem"].raster.mask_nodata()
da.attrs.update(long_name="elevation", units="m")
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
shaded = False  # shaded elevation (looks nicer with more pixels (e.g.: larger basins))!


# initialize image with geoaxes
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(projection=proj)
extent = np.array(da.raster.box.buffer(0.02).total_bounds)[[0, 2, 1, 3]]
ax.set_extent(extent, crs=proj)

# add sat background image
ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

## plot elevation\
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
    if 'station_name' in mod.geoms[gauges_name].columns:
        mod.geoms[gauges_name].apply(
            lambda x: ax.annotate(
                text=x['station_name'], 
                xy=x.geometry.coords[0],
                xytext=(2.0, 2.0), 
                textcoords='offset points', 
                #ha='left',
                #va = 'top',
                fontsize=5,
                fontweight='bold',
                color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            ), 
            axis=1
        )

patches = (
    []
)  # manual patches for legend, see https://github.com/geopandas/geopandas/issues/660
if "lakes" in mod.geoms:
    kwargs = dict(facecolor="lightblue", edgecolor="black", linewidth=1, label="lakes")
    mod.geoms["lakes"].plot(ax=ax, zorder=4, **kwargs)
    patches.append(mpatches.Patch(**kwargs))
if "reservoirs" in mod.geoms:
    kwargs = dict(facecolor="blue", edgecolor="black", linewidth=1, label="reservoirs")
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
# NOTE create figs folder in model root if it does not exist
# fn_out = join(mod.root, "figs", "basemap.png")
plt.savefig(os.path.join(Folder_plots, "basin_area.png"), dpi=300, bbox_inches="tight")
