import sys

# Get the snake_config file from the command line
args = sys.argv
config_path = args[args.index("--configfile") + 1]

# Function to get argument from config file and return default value if not found
def get_config(config, arg, default=None, optional=True):
    """
    Function to get argument from config file and return default value if not found

    Parameters
    ----------
    config : dict
        config file
    arg : str   
        argument to get from config file
    default : str/int/float/list, optional
        default value if argument not found, by default None
    optional : bool, optional
        if True, argument is optional, by default True
    """
    if arg in config:
        value = config[arg]
        if value == "None":
            value = None
        return value
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config file")

# Config settings
project_dir = get_config(config, 'project_dir', optional=False)
climate_sources = get_config(config, "clim_historical", optional=False)
data_catalog = get_config(config, "data_sources", optional=False)

rule all:
    input:
        f"{project_dir}/config/snake_config_historical_climate.yaml",
        f"{project_dir}/plots/climate_historical/precipitation_basin.png",
        f"{project_dir}/plots/climate_historical/temperature_basin.png",
        f"{project_dir}/plots/climate_historical/precipitation_location.png",
        f"{project_dir}/climate_historical/trends/precipitation_trend_basin.png",
        f"{project_dir}/climate_historical/trends/precipitation_trend_point.png",
        f"{project_dir}/climate_historical/trends/temperature_trend_basin.png",
        f"{project_dir}/climate_historical/trends/temperature_trend_point.png"

# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_snake = config_path,
    params:
        workflow_name = "historical_climate",
    output:
        config_snake_out = f"{project_dir}/config/snake_config_historical_climate.yaml",
    script:
        "../src/copy_config_files.py"

# Rule to derive the region of interest from the hydromt region dictionary
rule select_region:
    params:
        hydromt_region = get_config(config, "model_region", optional=False),
        buffer_km = get_config(config, "region_buffer", 10),
    output:
        region_file = f"{project_dir}/region/region.shp",
        region_buffer_file = f"{project_dir}/region/region_buffer.shp",
    script:
        "../src/derive_region.py"

# Region/basin wide plots
rule plot_basin_climate:
    input:
        region_file = f"{project_dir}/region/region.shp",
        region_buffer_file = f"{project_dir}/region/region_buffer.shp",
    params:
        subbasin_file = get_config(config, "climate_subregions", None),
        climate_sources = climate_sources,
        data_catalog = data_catalog,
    output:
        geods_basin = f"{project_dir}/climate_historical/statistics/basin_climate.nc",
        precip_basin_plot = f"{project_dir}/plots/climate_historical/precipitation_basin.png",
        temp_basin_plot = f"{project_dir}/plots/climate_historical/temperature_basin.png",
    script:
        "../src/plot_climate_basin.py"

# Location specific plots
# Output an empty geods if no location is provided
rule plot_location_climate:
    input:
        region_buffer_file = f"{project_dir}/region/region_buffer.shp",
    params:
        location_file = get_config(config, "climate_locations"),
        location_timeseries = get_config(config, "climate_locations_timeseries"),
        climate_sources = climate_sources,
        data_catalog = data_catalog,
    output:
        geods_point = f"{project_dir}/climate_historical/statistics/point_climate.nc",
        precip_location_plot = f"{project_dir}/plots/climate_historical/precipitation_location.png",
    script:
        "../src/plot_climate_location.py"

# Rule to derive trends in the historical data
rule derive_trends:
    input:
        geods_basin = f"{project_dir}/climate_historical/statistics/basin_climate.nc",
        geods_point = f"{project_dir}/climate_historical/statistics/point_climate.nc",
    output:
        precip_trends_basin = f"{project_dir}/climate_historical/trends/precipitation_trend_basin.png",
        precip_trends_point = f"{project_dir}/climate_historical/trends/precipitation_trend_point.png",
        temp_trends_basin = f"{project_dir}/climate_historical/trends/temperature_trend_basin.png",
        temp_trends_point = f"{project_dir}/climate_historical/trends/temperature_trend_point.png",
    script:
        "../src/derive_climate_trends.py"