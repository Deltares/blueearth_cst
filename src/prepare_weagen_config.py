import os
import yaml

# Snakemake config file
yml_default = snakemake.params.default_config
weagen_config = snakemake.output.weagen_config
cftype = snakemake.params.cftype


def read_yml(yml_fn):
    """ "Read yml file and return a dictionnary"""
    with open(yml_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)
    return yml


print(f"Preparing and writting the weather generator config file {weagen_config}")

# Read existing config file
yml_snake = read_yml(yml_default)

if cftype == "generate":
    # new arguments
    yml_dict = {
        "general": {"variables": ["precip", "temp", "temp_min", "temp_max"]},
        "generateWeatherSeries": {
            "output.path": snakemake.params.output_path,
            "sim.year.start": snakemake.params.start_year,
            "sim.year.num": snakemake.params.sim_years,
            "nc.file.prefix": snakemake.params.nc_file_prefix,
            "warm.variable": "precip",
            "warm.subset.criteria": None,
            "evaluate.model": True,
            "evaluate.grid.num": 20,
            "compute.parallel": False,
            "num.cores": None,
        },
    }
    # arguments from yml_snake
    yml_dict["generateWeatherSeries"]["month.start"] = yml_snake["month.start"]
    yml_dict["generateWeatherSeries"]["warm.sample.num"] = yml_snake["warm.sample.num"]
    yml_dict["generateWeatherSeries"]["mc.wet.quantile"] = yml_snake["mc.wet.quantile"]
    yml_dict["generateWeatherSeries"]["mc.extreme.quantile"] = yml_snake["mc.extreme.quantile"]    
    yml_dict["generateWeatherSeries"]["seed"] = yml_snake["seed"]
    yml_dict["generateWeatherSeries"]["realizations_num"] = yml_snake["realizations_num"]
    yml_dict["generateWeatherSeries"]["warm.signif.level"] = yml_snake["warm.signif.level"]
    yml_dict["generateWeatherSeries"]["warm.sample.num"] = yml_snake["warm.sample.num"]
    yml_dict["generateWeatherSeries"]["knn.sample.num"] = yml_snake["knn.sample.num"]
    yml_dict["generateWeatherSeries"]["knn.sample.num"] = yml_snake["knn.sample.num"]
    yml_dict["generateWeatherSeries"]["dry.spell.change"] = yml_snake["dry.spell.change"]
    yml_dict["generateWeatherSeries"]["wet.spell.change"] = yml_snake["wet.spell.change"]
    
    


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
