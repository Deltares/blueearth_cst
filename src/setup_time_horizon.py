import hydromt
import os

starttime = snakemake.params.starttime
endtime = snakemake.params.endtime
fn_ini = snakemake.output.forcing_ini
precip_source = snakemake.params.clim_source

if precip_source == 'eobs':
    clim_source = 'eobs'
    oro_source = 'eobs_orography'
    pet_method = 'makkink'
else: # (chirps is precip only)
    clim_source = 'era5'
    oro_source = 'era5_orography'
    pet_method = 'debruin'


forcing_options = {
    "setup_config": {
        "starttime": starttime,
        "endtime": endtime,
        "timestepsecs": 86400,
        "input.path_forcing": "../climate_historical/wflow_data/inmaps_historical.nc",
    },
    "setup_precip_forcing": {
        "precip_fn": precip_source,
        "precip_clim_fn": "None",
    },
    "setup_temp_pet_forcing": {
        "temp_pet_fn": clim_source,
        "press_correction": True,
        "temp_correction": True,
        "dem_forcing_fn": oro_source,
        "pet_method": pet_method,
        "skip_pet ": False,
    },
    "write_config": {},
    "write_forcing": {},
}

# Save it to a hydroMT ini file
hydromt.config.configwrite(fn_ini, forcing_options)
