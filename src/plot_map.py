# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:23:11 2022

@author: bouaziz
"""

# plot map

import numpy as np
from os.path import basename, join, isfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Union, Optional, List

from hydromt_wflow import WflowModel
from hydromt import DataCatalog

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.func_plot_map import plot_map_model
else:
    from .plot_utils.func_plot_map import plot_map_model


def plot_wflow_map(
    wflow_root: Union[str, Path],
    plot_dir: Union[str, Path] = None,
    gauges_name: str = None,
    gauges_name_legend: str = "output locations",
    data_catalog: List[Union[str, Path]] = [],
    meteo_locations: Optional[Union[str, Path]] = None,
    buffer_km: float = 2.0,
):
    """
    Plot the wflow model map with rivers, basins, gauges, etc.

    Output file will be saved in plot_dir/basin_area.png.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    plot_dir : Union[str, Path], optional
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_root folder.
    gauges_name : str, optional
        Name of the gauges in model to plot. If None (default), no gauges are plot.
    gauges_name_legend : str, optional
        Name of the gauges in the legend.
    data_catalog : List of str, Path
        Path to the data catalogs yaml file or pre-defined catalogs to read the meteo
        locations from if needed.
    meteo_locations : gpd.GeoDataFrame, optional
        Path or data catalog entry for the meteorological stations to add to the plot.
        The index should be the ID of the station. Optional variables for the label:
        "name" for the station name and "elevtn" for the elevation of the station.
    buffer_km : float, optional
        Buffer in km around the region to extract the data and do the plot.
        Default is 2 km.
    """
    mod = WflowModel(wflow_root, mode="r")

    # read and mask the model elevation
    da = mod.grid["wflow_dem"].raster.mask_nodata()
    da.attrs.update(long_name="elevation", units="m")

    # create nice colormap for elevation
    vmin, vmax = da.quantile([0.0, 0.98]).compute()
    c_dem = plt.cm.terrain(np.linspace(0.25, 1, 256))
    cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    kwargs = dict(cmap=cmap, norm=norm)

    # Read the meteo locations
    if meteo_locations is not None:
        data_catalog = DataCatalog(data_catalog)
        # If the locs are a direct file without a crs property, assume 4326
        if isfile(meteo_locations):
            crs = 4326
        else:
            crs = None
        locations = data_catalog.get_geodataframe(
            meteo_locations,
            crs=crs,
            geom=mod.basins,
            buffer=buffer_km * 1000,
        )
        locations.index.name = "index"
    else:
        locations = None

    # Plot the basin map
    plot_map_model(
        mod=mod,
        da=da,
        figname="basin_area",
        plot_dir=plot_dir,
        gauges_name=gauges_name,
        gauges_name_legend=gauges_name_legend,
        meteo_locations=locations,
        buffer_km=buffer_km,
        shaded=True,
        **kwargs,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        # Parse snake options
        project_dir = sm.params.project_dir
        gauges_fn = sm.params.output_locations
        if gauges_fn is not None:
            gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'
        else:
            gauges_name = None

        Folder_plots = f"{project_dir}/plots/wflow_model_performance"
        root = f"{project_dir}/hydrology_model"

        plot_wflow_map(
            wflow_root=root,
            plot_dir=Folder_plots,
            gauges_name=gauges_name,
            gauges_name_legend=sm.params.output_locations_legend,
            data_catalog=sm.params.data_catalog,
            meteo_locations=sm.params.meteo_locations,
            buffer_km=sm.params.buffer_km,
        )
    else:
        print("This script should be run from a snakemake environment")
