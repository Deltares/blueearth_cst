# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:23:11 2022

@author: bouaziz
"""

# plot map

from os.path import basename, join
import os
from pathlib import Path
from typing import Union

from hydromt_wflow import WflowModel

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from func_plot_map import plot_map_model
else:
    from .func_plot_map import plot_map_model

def plot_forcing(
    wflow_root: Union[str, Path],
    plot_dir: Union[str, Path] = None,
    gauges_name: str = None,
    config_fn: str = "wflow_sbm.toml",
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
    config_fn : str, optional
        name of the config file, default is wflow_sbm.toml
    """
    mod = WflowModel(wflow_root, mode="r", config_fn=config_fn)

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
            da = da.where(da > 0)
        da = da.where(mod.grid["wflow_subcatch"] >= 0)
        da.attrs.update(long_name=forcing_char["long_name"], units=forcing_char["unit"])
        figname = f"{forcing_var}"
        plot_map_model(mod, da, figname, plot_dir, gauges_name)


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
            wflow_root=root,
            plot_dir=Folder_plots,
            gauges_name=gauges_name,
        )
    else:
        plot_forcing(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model")
        )
