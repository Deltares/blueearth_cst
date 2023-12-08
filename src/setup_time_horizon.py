"""Prepare a hydromt config file to be able to add forcing to a wflow model"""
import hydromt
from pathlib import Path
from typing import Union

def prep_hydromt_update_forcing_config(
    starttime: str,
    endtime: str,
    fn_yml: Union[str, Path] = "wflow_build_forcing_historical.yml",
    precip_source: str = "era5",
):
    """Prepare a hydromt config file to be able to add forcing to a wflow model

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
    """
    # Check precip source and set options accordingly
    if precip_source == "eobs":
        clim_source = "eobs"
        oro_source = "eobs_orography"
        pet_method = "makkink"
    else:  # (chirps is precip only)
        clim_source = "era5"
        oro_source = "era5_orography"
        pet_method = "debruin"


    forcing_options = {
        "setup_config": {
            "starttime": starttime,
            "endtime": endtime,
            "timestepsecs": 86400,
            "input.path_forcing": "../climate_historical/wflow_data/inmaps_historical.nc",
        },
        "setup_precip_forcing": {
            "precip_fn": precip_source,
        },
        "setup_temp_pet_forcing": {
            "temp_pet_fn": clim_source,
            "press_correction": True,
            "temp_correction": True,
            "dem_forcing_fn": oro_source,
            "pet_method": pet_method,
            "skip_pet": False,
        },
        "write_config": {},
        "write_forcing": {},
    }

    # Save it to a hydroMT ini file
    hydromt.config.configwrite(fn_yml, forcing_options)

if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        prep_hydromt_update_forcing_config(
            starttime = sm.params.starttime,
            endtime = sm.params.endtime,
            fn_yml = sm.output.forcing_yml,
            precip_source = sm.params.clim_source,
        )
    else:
        prep_hydromt_update_forcing_config(
            starttime = "2010-01-01T00:00:00",
            endtime = "2010-12-31T00:00:00",
            fn_yml = "wflow_build_forcing_historical.yml",
            precip_source = "era5",
        )