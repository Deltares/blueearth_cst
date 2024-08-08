import sys

from get_config import get_config

# Get the snake_config file from the command line
args = sys.argv
config_path = args[args.index("--configfile") + 1]

# Config settings
project_dir = get_config(config, 'project_dir', optional=False)
climate_sources = get_config(config, "clim_historical", optional=False)
data_catalog = get_config(config, "data_sources", optional=False)

rule all:
    input:
        f"{project_dir}/config/snake_config_climate_historical.yaml",
        #f"{project_dir}/region/region.geojson",
        f"{project_dir}/plots/climate_historical/region_plot.png",
        expand((f"{project_dir}/climate_historical/raw_data/" + "extract_{source}.nc"), source=climate_sources),
        expand((f"{project_dir}/climate_historical/statistics/" + "basin_{source}.nc"), source=climate_sources),
        expand((f"{project_dir}/climate_historical/statistics/" + "point_{source}.nc"), source=climate_sources),
        f"{project_dir}/plots/climate_historical/region/basin_climate.txt",
        f"{project_dir}/plots/climate_historical/point/point_climate.txt",
        f"{project_dir}/plots/climate_historical/trends/gridded_trends.txt",
        f"{project_dir}/plots/climate_historical/trends/timeseries_trends.txt",

# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_snake = config_path,
    params:
        workflow_name = "climate_historical",
    output:
        config_snake_out = f"{project_dir}/config/snake_config_climate_historical.yaml",
    script:
        "../src/copy_config_files.py"

# Rule to derive the region of interest from the hydromt region dictionary
rule select_region:
    params:
        hydromt_region = get_config(config, "model_region", optional=False),
        buffer_km = get_config(config, "region_buffer", default=10),
        data_catalog = data_catalog,
        hydrography_fn = get_config(config, "hydrography_fn", default="merit_hydro"),
        basin_index_fn = get_config(config, "basin_index_fn", default="merit_hydro_index"),
    output:
        region_file = f"{project_dir}/region/region.geojson",
        region_buffer_file = f"{project_dir}/region/region_buffer.geojson",
    script:
        "../src/derive_region.py"

# Plot the reagion and subbasin/locations of interest
rule plot_region_and_location:
    input:
        region_file = f"{project_dir}/region/region.geojson",
    params:
        data_catalog = data_catalog,
        subregion_file = get_config(config, "climate_subregions", default=None),
        location_file = get_config(config, "climate_locations", default=None),
        river_fn = get_config(config, "river_geom_fn", default=None),
        hydrography_fn = get_config(config, "hydrography_fn", default="merit_hydro"),
        buffer_km = get_config(config, "region_buffer", default=10),
    output:
        region_plot = f"{project_dir}/plots/climate_historical/region_plot.png",
    script:
        "../src/plot_region_and_location.py"

# Rule to extract the historical climate data for the region of interest
rule extract_climate_historical_grid:
    input:
        region_fn = ancient(f"{project_dir}/region/region.geojson"),
    params:
        clim_source = "{source}",
        buffer_km = get_config(config, "region_buffer", default=10),
        data_sources = data_catalog,
        starttime = get_config(config, "starttime", optional=False),
        endtime = get_config(config, "endtime", optional=False),
        climate_variables = ["precip", "temp"],
        combine_with_era5 = False,
    output:
        climate_nc = f"{project_dir}/climate_historical/raw_data/" + "extract_{source}.nc",
    script:
        "../src/extract_historical_climate.py"

# Rule to sample historical climate at specific locations
rule sample_historical_climate:
    input:
        grid_fn = f"{project_dir}/climate_historical/raw_data/" + "extract_{source}.nc",
        region_fn = ancient(f"{project_dir}/region/region.geojson"),
    params:
        clim_source = "{source}",
        climate_variables = ["precip", "temp"],
        buffer_km = get_config(config, "region_buffer", default=10),
        subregion_fn = get_config(config, "climate_subregions", default=None),
        location_fn = get_config(config, "climate_locations", optional=False),
        data_catalog = data_catalog,
    output:
        basin = f"{project_dir}/climate_historical/statistics/" + "basin_{source}.nc",
        point = f"{project_dir}/climate_historical/statistics/" + "point_{source}.nc",
    script:
        "../src/sample_climate_historical.py"

# Region/basin wide plots
rule plot_basin_climate:
    input:
        basin_climate = expand((f"{project_dir}/climate_historical/statistics/"+"basin_{source}.nc"), source=climate_sources),
    params:
        climate_sources = climate_sources,
        climate_sources_colors = get_config(config, "clim_historical_colors", default=None),
        precip_peak = get_config(config, "climate_thresholds.precip.peak", default=40),
        precip_dry = get_config(config, "climate_thresholds.precip.dry", default=0.2),
        temp_heat = get_config(config, "climate_thresholds.temp.heat", default=25),
    output:
        basin_plot_done = f"{project_dir}/plots/climate_historical/region/basin_climate.txt",
    script:
        "../src/plot_climate_basin.py"

# Location specific plots
rule plot_location_climate:
    input:
        point_climate = expand((f"{project_dir}/climate_historical/statistics/"+"point_{source}.nc"), source=climate_sources),
    params:
        location_file = get_config(config, "climate_locations", optional=False),
        location_timeseries_precip = get_config(config, "climate_locations_timeseries", default=None),
        #location_timeseries_temp = get_config(config, "climate_locations_timeseries_temp", None),
        climate_sources = climate_sources,
        climate_sources_colors = get_config(config, "clim_historical_colors", default=None),
        data_catalog = data_catalog,
        precip_peak = get_config(config, "climate_thresholds.precip.peak", default=40),
        precip_dry = get_config(config, "climate_thresholds.precip.dry", default=0.2),
        temp_heat = get_config(config, "climate_thresholds.temp.heat", default=25),
    output:
        point_plot_done = f"{project_dir}/plots/climate_historical/point/point_climate.txt",
    script:
        "../src/plot_climate_location.py"

# Rule to derive trends in the historical data
rule derive_trends_timeseries:
    input:
        point_climate = expand((f"{project_dir}/climate_historical/statistics/"+"point_{source}.nc"), source=climate_sources),
        point_plot_done = f"{project_dir}/plots/climate_historical/point/point_climate.txt",
    params:
        split_year = get_config(config, "historical_climate_plots.timeseries_trends.split_year", default=None),
        point_observed = f"{project_dir}/climate_historical/statistics/point_observed.nc"
    output:
        trends_timeseries_done = f"{project_dir}/plots/climate_historical/trends/timeseries_trends.txt",
    script:
        "../src/derive_climate_trends.py"

rule derive_trends_gridded:
    input:
        grid = expand((f"{project_dir}/climate_historical/raw_data/" + "extract_{source}.nc"), source=climate_sources),
        region_fn = ancient(f"{project_dir}/region/region.geojson"),
    params:
        project_dir = project_dir,
        data_catalog = data_catalog,
        river_fn = get_config(config, "river_geom_fn", default=None) if get_config(config, "historical_climate_plots.mean_precipitation.add_rivers", default=False) else None,
        year_per_line = get_config(config, "historical_climate_plots.climate_per_year.year_per_line", default=8),
        line_height_yearly_plot = get_config(config, "historical_climate_plots.climate_per_year.line_height", default=8),
        line_height_mean_precip = get_config(config, "historical_climate_plots.mean_precipitation.line_height", default=8),
        fs_yearly_plot = get_config(config, "historical_climate_plots.climate_per_year.fontsize", default=8),
        fs_mean_precip = get_config(config, "historical_climate_plots.mean_precipitation.fontsize", default=8),
    output:
        trends_gridded_done = f"{project_dir}/plots/climate_historical/trends/gridded_trends.txt",
    script:
        "../src/derive_climate_trends_gridded.py"