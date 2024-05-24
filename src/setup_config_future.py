"""Prepare a hydromt config file to be able to add forcing to a wflow model"""

import hydromt
import os
from hydromt_wflow import WflowModel
from pathlib import Path
from typing import Union, Optional


def update_config_run_future(
    config_model_historical_fn: str,
    model: str,
    scenario: str,
    horizon: str,
    
):
    """Create a config file to run the model for a delta change scenario 
    with the name of the historical config file and suffix: "_delta_{model}_{scenario}_{horizon}"
    Updated fields in config file are: dir_output, instate, reinit

    Parameters
    ----------
    config_model_historical_fn : str
        Path of the config file 
    model: str
        name of the model
    scenario: str
        name of the scenario
    horizon: str
        name of the horizon

    """

    # Check if wflow_root is provided and adjust the forcing computation chunksizes
    wflow_root = os.path.dirname(config_model_historical_fn)
    mod = WflowModel(root=wflow_root, config_fn=os.path.basename(config_model_historical_fn), mode="r+")
    
    #update dir_output, state, 
    outstates_path_hist_run = os.path.join(wflow_root, mod.config["dir_output"], mod.config["state"]["path_output"])
    
    mod.set_config("state.path_input",outstates_path_hist_run)
    mod.set_config("dir_output", f"run_delta_{model}_{scenario}_{horizon}")
    mod.set_config("model.reinit",False)

    #TODO fix better!
    if "/" in model:
        model = model.replace("/", "_")

    config_delta_change_fn = os.path.basename(config_model_historical_fn).split(".")[0] + f"_delta_{model}_{scenario}_{horizon}.toml"

    mod.write_config(config_name=config_delta_change_fn, config_root=wflow_root)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        update_config_run_future(
            config_model_historical_fn=sm.input.config_model_historical_fn,
            model=sm.params.model_name,
            scenario=sm.params.scenario_name,
            horizon=sm.params.horizon,
        )
    else:
        update_config_run_future(
            config_model_historical_fn="hydrology_model/wflow_sbm_era5.toml",
            model="model",
            scenario="scenario",
            horizon="horizon",
        )
