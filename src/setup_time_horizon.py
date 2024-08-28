"""Prepare a hydromt config file to be able to add forcing to a wflow model"""

import hydromt
import os
from hydromt_wflow import WflowModel
from pathlib import Path
from typing import Union, Optional, Dict


def prep_hydromt_update_forcing_config(
    starttime: str,
    endtime: str,
    fn_yml: Union[str, Path] = "wflow_build_forcing_historical.yml",
    precip_source: str = "era5",
    wflow_root: Optional[Union[str, Path]] = None,
    forcing_options: Dict = {},
):
    """Prepare a hydromt config file to be able to add forcing to a wflow model

    Output config will be saved in wflow_root/run_default/wflow_sbm_{precip_source}.toml

    Parameters
    ----------
    starttime : str
        Start time of the forcing, format YYYY-MM-DDTHH:MM:SS
    endtime : str
        End time of the forcing, format YYYY-MM-DDTHH:MM:SS
    fn_yml : str, Path
        Path to the output hydromt config file
    precip_source : str
        Name of the precipitation source to use
    wflow_root : str, Path
        Path to the wflow model root directory, if provided reads the model
        and adjust the forcing computation chunksizes depending on model size.
    forcing_options : dict
        Dictionary with additional forcing options to set. If not provided, era5 with
        debruin will be used for all precip_source. Available options are:

        - temp_pet_fn : source of temp and pet if different from precip_source
        - pet_fn: source for direct pet (ie no computation needed)
        - dem_forcing_fn: source for the dem forcing (orography) data
        - pet_method: method to compute pet
    """
    # Check if forcing options are provided
    clim_source = forcing_options.get("temp_pet_fn", "era5")
    pet_fn = forcing_options.get("pet_fn", None)
    oro_source = forcing_options.get("dem_forcing_fn", "era5_orography")
    pet_method = forcing_options.get("pet_method", "debruin")

    # Check if direct pet forcing is provided
    if pet_fn is not None:
        skip_pet = True
    else:
        skip_pet = False

    path_forcing = (
        f"../climate_historical/wflow_data/inmaps_historical_{precip_source}.nc"
    )
    config_name = f"wflow_sbm_{precip_source}.toml"

    # Check if wflow_root is provided and adjust the forcing computation chunksizes
    if wflow_root is not None:
        mod = WflowModel(root=wflow_root, mode="r")
        size = mod.grid.raster.size
        if size > 1e6:
            chunksize = 1
        elif size > 2.5e5:
            chunksize = 30
        elif size > 1e5:
            chunksize = 100
        else:
            chunksize = 365
        config_root = os.path.join(wflow_root, "run_default")
    else:
        chunksize = 30
        config_root = "run_default"

    setup_config = {
        "starttime": starttime,
        "endtime": endtime,
        "timestepsecs": 86400,
        "dir_input": "..",
        "input.path_forcing": path_forcing,
        "output.path": f"output_{precip_source}.nc",
        "output.compressionlevel": 1,
        "csv.path": f"output_{precip_source}.csv",
        "state.path_output": f"outstate/outstates_{precip_source}.nc",
    }
    setup_precip_forcing = {
        "precip_fn": precip_source,
        "chunksize": chunksize,
    }
    setup_temp_pet_forcing = {
        "temp_pet_fn": clim_source,
        "press_correction": True,
        "temp_correction": True,
        "dem_forcing_fn": oro_source,
        "pet_method": pet_method,
        "skip_pet": skip_pet,
        "chunksize": chunksize,
    }
    write_config = {
        "config_name": config_name,
        "config_root": config_root,
    }

    # Order matters
    if skip_pet:
        forcing_config = {
            "setup_config": setup_config,
            "setup_precip_forcing": setup_precip_forcing,
            "setup_temp_pet_forcing": setup_temp_pet_forcing,
            "setup_pet_forcing": {
                "pet_fn": pet_fn,
                "chunksize": chunksize,
            },
            "write_config": write_config,
            "write_forcing": {},
        }
    else:
        forcing_config = {
            "setup_config": setup_config,
            "setup_precip_forcing": setup_precip_forcing,
            "setup_temp_pet_forcing": setup_temp_pet_forcing,
            "write_config": write_config,
            "write_forcing": {},
        }

    # Create output dir if it does not exist
    if not os.path.exists(os.path.dirname(fn_yml)):
        os.makedirs(os.path.dirname(fn_yml))
    # Save it to a hydroMT ini file
    hydromt.config.configwrite(fn_yml, forcing_config)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        prep_hydromt_update_forcing_config(
            starttime=sm.params.starttime,
            endtime=sm.params.endtime,
            fn_yml=sm.output.forcing_yml,
            precip_source=sm.params.clim_source,
            wflow_root=sm.params.basin_dir,
            forcing_options=sm.params.forcing_options,
        )
    else:
        prep_hydromt_update_forcing_config(
            starttime="2010-01-01T00:00:00",
            endtime="2010-12-31T00:00:00",
            fn_yml="wflow_build_forcing_historical.yml",
            precip_source="era5",
        )
