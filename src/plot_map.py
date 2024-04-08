# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:23:11 2022

@author: bouaziz
"""

# plot map

import numpy as np
from os.path import basename, join
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Union

from hydromt_wflow import WflowModel

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from func_plot_map import plot_map_model
else:
    from .func_plot_map import plot_map_model


def plot_wflow_map(
    wflow_root: Union[str, Path],
    plot_dir: Union[str, Path] = None,
    gauges_name: str = None,
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

    # Plot the basin map
    plot_map_model(
        mod=mod,
        da=da,
        figname="basin_area",
        plot_dir=plot_dir,
        gauges_name=gauges_name,
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
        )
    else:
        plot_wflow_map(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model")
        )
