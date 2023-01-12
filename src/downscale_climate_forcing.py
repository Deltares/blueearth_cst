import hydromt
from hydromt_wflow import WflowModel
from pathlib import Path
import os
import xarray as xr

# # FIXME: temp from snakefile

# starttime = "2000-01-01T00:00:00"

# endtime = "2020-12-31T00:00:00"

# repo_root = str(Path(__file__).parent.parent.absolute())

# DATA_SOURCES = f"{repo_root}/config/deltares_data_linux.yml"
# experiment = "experiment_02"
# exp_dir = f"{repo_root}/examples/Gabon2/climate_{experiment}"
# basin_dir = f"{repo_root}/examples/Gabon2/hydrology_model"
# rlz_num = 1
# st_num2 = 0
# precip_source="era5"
# model_root = basin_dir
# # input
# fn_in = f"{exp_dir}/realization_{rlz_num}/rlz_{rlz_num}_cst_{st_num2}.nc"
# data_libs = [f"{exp_dir}/data_catalog_climate_experiment.yml", DATA_SOURCES]
# # output
# fn_out = f"{exp_dir}/realization_"+"{rlz_num}"+"/inmaps_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".nc",
# config_out_fn = f"{basin_dir}/run_climate_{experiment}/wflow_sbm_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".toml"

# Snakemake parameters
config_out_fn = snakemake.output.toml
fn_out = snakemake.output.nc
fn_in = snakemake.input.nc
data_libs = snakemake.input.data_sources
model_root = snakemake.params.model_dir
precip_source = snakemake.params.clim_source

oro_source = f"{precip_source}_orography"
if precip_source == "eobs":
    pet_method = "makkink"
else:  # (chirps is precip only so combined with era5)
    pet_method = "debruin"

# Get name of climate scenario (rlz_*_cst_*)
fn_in_path = Path(fn_in, resolve_path=True)
climate_name = os.path.basename(fn_in_path).split(".")[0]

# Get start and endtime from fn_in
ds_in = xr.open_dataset(fn_in_path)
starttime = ds_in.time.values[0].strftime(format="%Y-%m-%dT%H:%M:%S")
endtime = ds_in.time.values[-1].strftime(format="%Y-%m-%dT%H:%M:%S")

# Get options for toml file name
config_out_fn = Path(config_out_fn)
config_out_root = os.path.dirname(config_out_fn)
config_out_name = os.path.basename(config_out_fn)

# Hydromt ini dictionnaries for update options
update_options = {
    "setup_config": {
        "calendar": "noleap",
        "starttime": starttime,
        "endtime": endtime,
        "timestepsecs": 86400,
        "state.path_input": "../instate/instates.nc",
        "state.path_output": f"outstates_{climate_name}.nc",
        "input.path_static": "../staticmaps.nc",
        "input.path_forcing": f"../../../../{fn_out}",
        "csv.path": f"output_{climate_name}.csv",
    },
    "set_root": {
        "root": config_out_root,
        "mode": "r+",
    },
    "setup_precip_forcing": {
        "precip_fn": climate_name,
        "precip_clim_fn": None,
    },
    "setup_temp_pet_forcing": {
        "temp_pet_fn": climate_name,
        "press_correction": True,
        "temp_correction": True,
        "dem_forcing_fn": oro_source,
        "pet_method": pet_method,
    },
    "write_forcing": {},
    "write_config": {
        "config_name": config_out_name,
        "config_root": config_out_root,
    },
}

### Run Hydromt update using update_options dict ###
# Instantiate model
mod = WflowModel(root=model_root, mode="r+", data_libs=data_libs)

# Update
mod.update(opt=update_options)
