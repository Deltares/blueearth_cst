"""Prepare a hydromt config file to be able to add forcing to a wflow model"""

import os
from os.path import dirname, basename, relpath, join
from hydromt_wflow import WflowModel
from pathlib import Path
from typing import Union, Optional


def update_config_run_future(
    config_model_historical_fn: Union[str, Path],
    delta_change_fn: Union[str, Path],
    model: str,
    scenario: str,
    horizon: str,
    config_root: Optional[Union[str, Path]] = None,
):
    """Create a config file to run the model for a delta change scenario.

    Updated config file will have the name of the historical config file and suffix:
    "_delta_{model}_{scenario}_{horizon}"

    Updated fields in config file are: dir_output, instate, reinit, path_forcing_scale.

    Parameters
    ----------
    config_model_historical_fn : str
        Path of the config file
    delta_change_fn : str
        Path of the downscaled monthly gridded delta change grid
    model: str
        name of the model for the output config filename
    scenario: str
        name of the scenario for the output config filename
    horizon: str
        name of the horizon for the output config filename. Only far and near are
        supported.
    config_root : str, optional
        Folder where the output config file will be written to. If None (default), will
        be written to the same folder as the historical config file.

    """
    # Get the wflow root from the config file
    mod = WflowModel(
        root=dirname(config_model_historical_fn),
        config_fn=basename(config_model_historical_fn),
        mode="r+",
    )
    wflow_root = mod.root
    if "dir_input" in mod.config:
        wflow_root_input = join(wflow_root, mod.config["dir_input"])
    else:
        wflow_root_input = wflow_root

    # Set the output config root
    if config_root is None:
        config_root = wflow_root
    else:
        # Update the dir_input folder in case config_root is different than wflow root
        try:
            dir_input = relpath(wflow_root_input, config_root)
        except ValueError:
            dir_input = wflow_root_input
        mod.set_config("dir_input", dir_input)

    # update dir_output, state,
    if "dir_output" in mod.config:
        outstates_path_hist_run = join(
            wflow_root, mod.config["dir_output"], mod.config["state"]["path_output"]
        )
    else:
        outstates_path_hist_run = join(wflow_root, mod.config["state"]["path_output"])
    # get the relative path
    try:
        outstates_path_hist_run = relpath(outstates_path_hist_run, wflow_root_input)
    except ValueError:
        pass

    mod.set_config("state.path_input", outstates_path_hist_run)
    mod.set_config("dir_output", "")
    mod.set_config("model.reinit", False)
    mod.set_config("csv.path", f"output_delta_{model}_{scenario}_{horizon}.csv")
    mod.set_config(
        "state.path_output", f"outstate/outstates_{model}_{scenario}_{horizon}.nc"
    )
    mod.set_config("output.path", f"output_delta_{model}_{scenario}_{horizon}.nc")

    # Add the (relative) path to the downscaled delta change grids
    try:
        delta_change_fn = relpath(delta_change_fn, wflow_root_input)
    except ValueError:
        pass
    mod.set_config("input.path_forcing_scale", delta_change_fn)

    if ("near" in config_model_historical_fn) and (horizon == "far"):
        config_delta_change_fn = basename(
            config_model_historical_fn.replace("near", horizon)
        )
    else:
        config_delta_change_fn = (
            basename(config_model_historical_fn).split(".")[0]
            + f"_delta_{model}_{scenario}_{horizon}.toml"
        )

    # Create output dir if needed
    if not os.path.exists(config_root):
        os.makedirs(config_root)
    mod.write_config(config_name=config_delta_change_fn, config_root=config_root)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        config_root = dirname(sm.output.config_model_out_fn)

        update_config_run_future(
            config_model_historical_fn=sm.input.config_model_historical_fn,
            delta_change_fn=sm.input.monthly_change_mean_grid,
            model=sm.params.model_name,
            scenario=sm.params.scenario_name,
            horizon=sm.params.horizon,
            config_root=config_root,
        )
    else:
        print("Please run this script using snakemake.")
