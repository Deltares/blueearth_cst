import hydromt
import os

starttime = snakemake.params.starttime
endtime = snakemake.params.endtime
fn_ini = snakemake.output.forcing_ini


forcing_options = {
    "setup_config": {
        "starttime": starttime,
        "endtime": endtime,
        "timestepsecs": 86400,
        "input.path_forcing": "../climate_historical/wflow_data/inmaps_era5.nc",
    },
    "setup_precip_forcing": {
        "precip_fn": "era5",
        "precip_clim_fn": "None",
    },
    "setup_temp_pet_forcing": {
        "temp_pet_fn": "era5",
        "press_correction": True,
        "temp_correction": True,
        "dem_forcing_fn": "era5_orography",
        "pet_method": "debruin",
        "skip_pet ": False,
    },
    "write_config":{},
    "write_forcing":{},
}

# Save it to a hydroMT ini file
hydromt.config.configwrite(fn_ini, forcing_options)
