import os
import yaml

# Snakemake config file
yml_snake = snakemake.params.snake_config
weagen_config = snakemake.output.weagen_config
cftype = snakemake.params.cftype


def read_yml(yml_fn):
    """ "Read yml file and return a dictionnary"""
    with open(yml_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)
    return yml


print(f"Preparing and writting the weather generator config file {weagen_config}")

# Read existing config file
yml_snake = read_yml(yml_snake)

if cftype == "generate":
    # arguments from the default weagen config file
    yml_dict = read_yml(snakemake.params.default_config)
    # add new arguments from snakemake and yml_snake
    yml_add = {
        "output.path": snakemake.params.output_path,
        "sim.year.start": snakemake.params.start_year,
        "sim.year.num": snakemake.params.sim_years,
        "nc.file.prefix": snakemake.params.nc_file_prefix,
        "realizations_num": yml_snake["realizations_num"]
    }
    for k,v in yml_add.items():
        yml_dict["generateWeatherSeries"][k] = v

else:  # stress test
    # new arguments
    yml_dict = {
        "imposeClimateChanges": {
            "output.path": snakemake.params.output_path,
            "nc.file.prefix": snakemake.params.nc_file_prefix,
            "nc.file.suffix": snakemake.params.nc_file_suffix,
        }
    }
    # arguments from yml_snake
    yml_dict["temp"] = yml_snake["temp"]
    yml_dict["precip"] = yml_snake["precip"]

# Write the new weagen config
if not os.path.isdir(os.path.dirname(weagen_config)):
    os.makedirs(os.path.dirname(weagen_config))
with open(weagen_config, "w") as f:
    yaml.dump(yml_dict, f, default_flow_style=False, sort_keys=False)
