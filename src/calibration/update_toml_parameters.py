import os
import hydromt_wflow

#%% Read the data. SnakeMake arguments are automatically passed

paramset = snakemake.params.calibration_parameters
paramsetname = snakemake.params.calibration_pattern
toml_default_fn = snakemake.input.toml_default
toml_out = snakemake.output.toml_fid
wflow_dir_input = snakemake.params.wflow_dir_input
calib_folder = snakemake.params.calib_folder
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime

# Instantiate wflow model and read/update config
root = os.path.dirname(toml_default_fn)
mod = hydromt_wflow.WflowModel(root, config_fn=os.path.basename(toml_default_fn), mode="r+")

# Parameters for which value is updated instead of a scale
parameters_value = ["tt", "ttm", "cfmax", "water_holding_capacity", "g_cfmax", "g_tt", "g_sifrac"]

def update_scale_toml(mod, param_name, param_value):
    # get param name in staticmaps netcdf
    param_netcdf = mod.config["input"]["vertical"][f"{param_name}"]
    if not isinstance(param_netcdf, str):
        if "netcdf" in param_netcdf:
            param_netcdf = param_netcdf["netcdf"]["variable"]["name"]
    mod.set_config(f"input.vertical.{param_name}.netcdf.variable.name", param_netcdf)
    mod.set_config(f"input.vertical.{param_name}.scale", float(param_value))

# Update parameter value in toml file
for param in paramset:
    if param == "ksathorfrac":
        if (paramset[param] == "ksathorfrac_RF_250") | (paramset[param] == "ksathorfrac_BRT_250"):
            mod.set_config("input.lateral.subsurface.ksathorfrac", paramset[param])
        else: #value           
            mod.set_config("input.lateral.subsurface.ksathorfrac", {"value": float(paramset['ksathorfrac'])})
    elif param == "f":
        if paramset[param] == "f_":
            mod.set_config(f"input.vertical.{param}", paramset[param])
        else: #scale
            update_scale_toml(mod, "f", paramset[param])
    elif param == "kv_0":
        # if paramset[param].startswith("KSatVer_"):
        if (paramset[param]== "KvB"): # | (paramset[param]== "KsatVer_Brakensiek_Bonetti") | (paramset[param]== "KsatVer_Cosby_Bonetti"):
            mod.set_config(f"input.vertical.{param}", paramset[param])
        else: #scale
            update_scale_toml(mod, "kv_0", paramset[param])
    elif param in parameters_value:
        mod.set_config(f"input.vertical.{param}", {"value": float(paramset[param])})
    else:
        update_scale_toml(mod, param, paramset[param])

# Find the realtive path of toml_out to the wflow_dir_input
rel_path = os.path.relpath(wflow_dir_input, os.path.dirname(toml_out) )

# Update parameters for saving outputs
setting_toml = {
    "starttime": starttime,
    "endtime": endtime,
    "state.path_output": f"outstates_{paramsetname}.nc",
    "csv.path": f"output_{paramsetname}.csv",
    "dir_input": rel_path,
    "dir_output": "",
}

for option in setting_toml:
    mod.set_config(option, setting_toml[option])

# Remove the output section from the config dict
mod._config.pop("output", None)



# Write new toml file
# toml_root = os.path.join(os.path.dirname(toml_default_fn))
toml_root = os.path.join(wflow_dir_input, calib_folder)
toml_name = f"wflow_sbm_{paramsetname}.toml"
mod.write_config(config_name=toml_name, config_root=toml_root)