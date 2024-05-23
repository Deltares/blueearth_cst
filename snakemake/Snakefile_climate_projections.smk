import itertools
import numpy as np
import sys

from get_config import get_config

# read path of the config file to give to the weagen scripts
args = sys.argv
config_path = args[args.index("--configfile") + 1]

# Parsing the Snakemake config file (options for basins to build, data catalog, model output directory)
project_dir = get_config(config, 'project_dir', optional=False)
# Data catalogs
DATA_SOURCES = get_config(config, "data_sources", [])
DATA_SOURCES = np.atleast_1d(DATA_SOURCES).tolist() #make sure DATA_SOURCES is a list format (even if only one DATA_SOURCE)
DATA_SOURCES_CLIMATE = np.atleast_1d(get_config(config, "data_sources_climate", optional=False)).tolist()
data_catalogs = []
data_catalogs.extend(DATA_SOURCES)
data_catalogs.extend(DATA_SOURCES_CLIMATE)

clim_project = get_config(config, "clim_project", optional=False)
models = get_config(config, "models", optional=False)
scenarios = get_config(config, "scenarios", optional=False)
members = get_config(config, "members", optional=False)
variables = get_config(config, "variables", optional=False)
pet_method = get_config(config, "pet_method", "makkink")
future_horizons = get_config(config, "future_horizons", optional=False)

save_grids = get_config(config, "save_grids", False)

basin_dir = f"{project_dir}/hydrology_model"
clim_project_dir = f"{project_dir}/climate_projections/{clim_project}"
region_default = {"geom": f"{basin_dir}/staticgeoms/region.geojson"}

### Dictionary elements from the config based on wildcards
def get_horizon(wildcards):
    return config["future_horizons"][wildcards.horizon]

# Master rule: end with csv with change factors for each scenario and model
rule all:
    input:
        #ancient(expand((clim_project_dir + "/annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc"), model = config["models"], scenario = config["scenarios"], horizon = config["future_horizons"])),
        (clim_project_dir + "/annual_change_scalar_stats_summary.nc"),
        (clim_project_dir + "/annual_change_scalar_stats_summary.csv"),
        (clim_project_dir + "/annual_change_scalar_stats_summary_mean.csv"),
        stats_change_plt = (clim_project_dir + "/plots/projected_climate_statistics.png"),
        precip_plt = (clim_project_dir + "/plots/precipitation_anomaly_projections_abs.png"),
        temp_plt = (clim_project_dir + "/plots/temperature_anomaly_projections_abs.png"),
        snake_config = f"{project_dir}/config/snake_config_climate_projections.yml",

ruleorder: monthly_stats_hist > monthly_stats_fut > monthly_change > monthly_change_scalar_merge

# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_snake = config_path,
    params:
        data_catalogs = data_catalogs,
        workflow_name = "climate_projections",
    output:
        config_snake_out = f"{project_dir}/config/snake_config_climate_projections.yml",
    script:
        "../src/copy_config_files.py"

# Rule to derive the region of interest from the hydromt region dictionary
rule select_region:
    params:
        hydromt_region = get_config(config, "model_region", optional=region_default),
        buffer_km = get_config(config, "region_buffer", 10),
        data_catalog = DATA_SOURCES,
        hydrography_fn = get_config(config, "hydrography_fn", "merit_hydro"),
        basin_index_fn = get_config(config, "basin_index_fn", "merit_hydro_index"),
    output:
        region_file = f"{project_dir}/region/region.geojson",
    script:
        "../src/derive_region.py"

# Rule to calculate mean monthly statistics for historical and future scenarios - grid saved to netcdf
# also calculate monthly time series averaged over the grid.
rule monthly_stats_hist:
    input:
        region_fid = ancient(f"{project_dir}/region/region.geojson"),
    output:
        stats_time_nc_hist = temp(clim_project_dir + "/historical_stats_time_{model}.nc"),
        stats_grid_nc_hist = temp(clim_project_dir + "/historical_stats_{model}.nc") if save_grids else [],
    params:
        yml_fid = DATA_SOURCES_CLIMATE,
        project_dir = f"{project_dir}",
        name_scenario = "historical",
        name_members = members,
        name_model = "{model}",
        name_clim_project = clim_project,
        variables = variables,
        pet_method = pet_method,
        save_grids = save_grids,
        time_horizon = {"historical": get_config(config, "historical", optional=False)},
    script: "../src/get_stats_climate_proj.py"

# Rule to calculate mean monthly statistics for historical and future scenarios - grid saved to netcdf
# also calculate monthly time series averaged over the grid.
rule monthly_stats_fut:
    input:
        region_fid = ancient(f"{project_dir}/region/region.geojson"),
        stats_time_nc_hist = (clim_project_dir + "/historical_stats_time_{model}.nc"), #make sure starts with previous job
        stats_grid_nc_hist = (clim_project_dir + "/historical_stats_{model}.nc") if save_grids else [],
    output:
        stats_time_nc = temp(clim_project_dir + "/stats_time-{model}_{scenario}.nc"),
        stats_grid_nc = temp(clim_project_dir + "/stats-{model}_{scenario}.nc") if save_grids else [],
    params:
        yml_fid = DATA_SOURCES_CLIMATE,
        project_dir = f"{project_dir}",
        name_scenario = "{scenario}",
        name_members = members,
        name_model = "{model}",
        name_clim_project = clim_project,
        variables = variables,
        pet_method = pet_method,
        save_grids = save_grids,
        time_horizon = get_config(config, "future_horizons", optional=False),
    script: "../src/get_stats_climate_proj.py"

# Rule to calculate change stats over the grid
rule monthly_change:
    input:
        stats_time_nc_hist = ancient(clim_project_dir + "/historical_stats_time_{model}.nc"),
        stats_time_nc = ancient(clim_project_dir + "/stats_time-{model}_{scenario}.nc"),
        stats_grid_nc_hist = ancient(clim_project_dir + "/historical_stats_{model}.nc") if save_grids else [],
        stats_grid_nc = ancient(clim_project_dir + "/stats-{model}_{scenario}.nc") if save_grids else [],
    output:
        stats_nc_change = temp(clim_project_dir + "/annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc"),
        monthly_change_mean_grid = (clim_project_dir + "/monthly_change_mean_grid-{model}_{scenario}_{horizon}.nc") if save_grids else [],
    params:
        clim_project_dir = f"{clim_project_dir}",
        start_month_hyd_year = get_config(config, "start_month_hyd_year", "Jan"), 
        name_model = "{model}",
        name_scenario = "{scenario}",
        name_horizon = "{horizon}",
        time_horizon_hist = get_config(config, "historical", optional=False),
        time_horizon_fut = get_horizon,
        save_grids = save_grids,
        #stats_nc_hist = (clim_project_dir + "/historical_stats_{model}.nc"),
        #stats_nc = (clim_project_dir + "/stats-{model}_{scenario}.nc"),
    script: "../src/get_change_climate_proj.py"

#rule to merge results in one netcdf / todo: add plotting
rule monthly_change_scalar_merge:
    input:
        stats_nc_change = ancient(expand((clim_project_dir + "/annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc"), model = models, scenario = scenarios, horizon = future_horizons)),
    output:
        stats_change_summary = (clim_project_dir + "/annual_change_scalar_stats_summary.nc"),
        stats_change_summary_csv = (clim_project_dir + "/annual_change_scalar_stats_summary.csv"),
        stats_change_summary_csv_mean = (clim_project_dir + "/annual_change_scalar_stats_summary_mean.csv"),
        stats_change_plt = (clim_project_dir + "/plots/projected_climate_statistics.png"),
    params:
        clim_project_dir = f"{clim_project_dir}",
        horizons = future_horizons,
        save_grids = save_grids,
    script: "../src/get_change_climate_proj_summary.py"

#rule to plot timeseries
rule plot_climate_proj_timeseries:
    input:
        stats_change_summary = (clim_project_dir + "/annual_change_scalar_stats_summary.nc"),
        stats_time_nc_hist =[(clim_project_dir + f"/historical_stats_time_{mod}.nc") for mod in models],
        stats_time_nc = expand((clim_project_dir + "/stats_time-{model}_{scenario}.nc"), model = models, scenario = scenarios),
        monthly_change_mean_grid = expand((clim_project_dir + "/monthly_change_mean_grid-{model}_{scenario}_{horizon}.nc"), model = models, scenario = scenarios, horizon = future_horizons) if save_grids else [],
    params:
        clim_project_dir = f"{clim_project_dir}",
        scenarios = scenarios,
        horizons = future_horizons,
        #save_grids = save_grids,
        #change_grids = [(clim_project_dir + f"/monthly_change_mean_grid-{mod}_{sc}_{hz}.nc") for mod,sc,hz in list(itertools.product(models,scenarios,future_horizons))],
    output:
        precip_plt = (clim_project_dir + "/plots/precipitation_anomaly_projections_abs.png"),
        temp_plt = (clim_project_dir + "/plots/temperature_anomaly_projections_abs.png"),
        timeseries_nc = (clim_project_dir + "/gcm_timeseries.nc"),
    script: "../src/plot_proj_timeseries.py"
