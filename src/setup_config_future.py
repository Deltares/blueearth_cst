"""Prepare a hydromt config file to be able to add forcing to a wflow model"""

import os
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
        name of the horizon for the output config filename
    config_root : str, optional
        Folder where the output config file will be written to. If None (default), will
        be written to the same folder as the historical config file.

    """
    # Check if wflow_root is provided and adjust the forcing computation chunksizes
    wflow_root = os.path.dirname(config_model_historical_fn)
    mod = WflowModel(
        root=wflow_root,
        config_fn=os.path.basename(config_model_historical_fn),
        mode="r+",
    )

    # update dir_output, state,
    outstates_path_hist_run = os.path.join(
        wflow_root, mod.config["dir_output"], mod.config["state"]["path_output"]
    )
    # get the relative path
    outstates_path_hist_run = os.path.relpath(outstates_path_hist_run, wflow_root)

    mod.set_config("state.path_input", outstates_path_hist_run)
    mod.set_config("dir_output", f"run_delta_{model}_{scenario}_{horizon}")
    mod.set_config("model.reinit", False)
    mod.set_config("csv.path", f"output_delta_{model}_{scenario}_{horizon}.csv")

    # Add the (relative) path to the downscaled delta change grids
    config_root = config_root or wflow_root
    delta_change_fn = os.path.relpath(delta_change_fn, config_root)
    mod.set_config("input.path_forcing_scale", delta_change_fn)

    if ("near" in config_model_historical_fn) and (horizon == "far"):
        config_delta_change_fn = os.path.basename(
            config_model_historical_fn.replace("near", horizon)
        )
    else:
        config_delta_change_fn = (
            os.path.basename(config_model_historical_fn).split(".")[0]
            + f"_delta_{model}_{scenario}_{horizon}.toml"
        )

    mod.write_config(config_name=config_delta_change_fn, config_root=config_root)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        update_config_run_future(
            config_model_historical_fn=sm.input.config_model_historical_fn,
            delta_change_fn=sm.input.monthly_change_mean_grid,
            model=sm.params.model_name,
            scenario=sm.params.scenario_name,
            horizon=sm.params.horizon,
            config_root=None,
        )
    else:
        print("Please run this script using snakemake.")
