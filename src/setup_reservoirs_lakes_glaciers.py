"""Function to update a wflow model and add gauges and outputs"""
from hydromt_wflow import WflowModel
from hydromt.exceptions import NoDataException
from hydromt.cli.cli_utils import parse_config

import os
from os.path import join
from pathlib import Path
import numpy as np
from typing import Union


def update_wflow_waterbodies_glaciers(
    wflow_root: Union[str, Path],
    config_fn: Union[str, Path],
    data_catalog: Union[str, Path] = "deltares_data",
):
    """
    Update wflow model with reservoirs, lakes and glaciers.

    Write a file when everything is done for snakemake tracking.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder
    config_fn : Union[str, Path]
        Path to the config file for setup of reservoirs, lakes and glaciers
    data_catalog : str
        Name of the data catalog to use
    """
    # Read the config file
    config = parse_config(config_fn)

    # Check if "global" in config and pass to model init instead
    kwargs = config.pop("global", {})
    # parse data catalog options from global section in config and cli options
    data_libs = np.atleast_1d(kwargs.pop("data_libs", [])).tolist()  # from global
    data_libs += list(np.atleast_1d(data_catalog))  # add data catalogs from cli

    # Instantiate wflow model
    mod = WflowModel(wflow_root, mode="r+", data_libs=data_libs, **kwargs)

    # List of methods that ran successfully
    successful_methods = []
    # List of methods that failed
    failed_methods = []
    reasons = []

    # Loop over reservoirs, lakes and glaciers methods
    for method in config:
        kwargs = {} if config[method] is None else config[method]
        try:
            mod._run_log_method(method, **kwargs)
            successful_methods.append(method)
        except (NoDataException, FileNotFoundError) as error:
            failed_methods.append(method)
            reasons.append(error)

    # Write model if there is any data to add
    if len(successful_methods) > 0:
        mod.write()

    # Write a file when everything is done for snakemake tracking
    text_out = join(wflow_root, "staticgeoms", "reservoirs_lakes_glaciers.txt")
    with open(text_out, "w") as f:
        f.write(f"Successful methods: {successful_methods}\n")
        f.write(f"Failed methods: {failed_methods}\n")
        f.write(f"Reasons: {reasons}\n")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        update_wflow_waterbodies_glaciers(
            wflow_root=os.path.dirname(sm.input.basin_nc),
            data_catalog=sm.params.data_catalog,
            config_fn=sm.params.config,
        )
    else:
        update_wflow_waterbodies_glaciers(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            config_fn=join(os.getcwd(), "config", "wflow_update_waterbodies.yml"),
        )
