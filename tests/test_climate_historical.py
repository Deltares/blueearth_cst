import os
from os.path import join, isfile
import glob

import geopandas as gpd
import xarray as xr
from geopandas.testing import assert_geodataframe_equal

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, TESTDIR
from .conftest import get_config

from ..src import copy_config_files
from ..src import derive_region
from ..src import plot_climate_basin
from ..src import plot_climate_location
from ..src import derive_climate_trends_gridded
from ..src import derive_climate_trends
from ..src import extract_historical_climate
from ..src import sample_climate_historical
from ..src import plot_region_and_location

config_fn = join(TESTDIR, "snake_config_fao_test.yml")


def test_copy_config(tmpdir):
    """Test if config files are copied to tmpdir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fn,
        output_dir=join(tmpdir, "config"),
        config_out_name="snake_config_climate_historical.yml",
    )

    # Check if config files are copied to tmpdir/config folder
    assert os.path.exists(f"{tmpdir}/config/snake_config_climate_historical.yml")


def test_select_region(tmpdir, config_fao, data_libs_fao):
    """Test if region is created properly."""
    # Try to derive the region from config file
    derive_region.derive_region(
        region=config_fao["model_region"],
        path_output=join(tmpdir, "region"),
        buffer=config_fao["region_buffer"],
        data_catalog=data_libs_fao,
        hydrography_fn=config_fao["hydrography_fn"],
    )

    assert os.path.exists(f"{tmpdir}/region/region.geojson")
    assert os.path.exists(f"{tmpdir}/region/region_buffer.geojson")

    # Check if region is the same as expected
    region_sample = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    region_sample = gpd.read_file(region_sample)
    region = gpd.read_file(f"{tmpdir}/region/region.geojson")
    assert_geodataframe_equal(
        region, region_sample, check_like=True, check_less_precise=True
    )

    # Try from bbox without buffer and hydrography data
    bounds = region.total_bounds
    derive_region.derive_region(
        region={"bbox": bounds},
        path_output=join(tmpdir, "region_bbox"),
        buffer=None,
        data_catalog=None,
        hydrography_fn=None,
        basin_index_fn=None,
    )
    assert os.path.exists(f"{tmpdir}/region_bbox/region.geojson")
    assert os.path.exists(f"{tmpdir}/region_bbox/region_buffer.geojson")

    # Without a buffer region and region_buffer should be equal
    region = gpd.read_file(f"{tmpdir}/region_bbox/region.geojson")
    region_buffer = gpd.read_file(f"{tmpdir}/region_bbox/region_buffer.geojson")
    assert_geodataframe_equal(region, region_buffer)


def test_plot_region_and_location(tmpdir, config_fao, data_libs_fao):
    """Test the plotting of the region and locations."""
    region_filename = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    subregions_filename = join(
        SAMPLE_PROJECTDIR,
        "hydrology_model",
        "staticgeoms",
        "subcatch_discharge-locations-grdc.geojson",
    )
    obs_fn = join(MAINDIR, get_config(config_fao, "climate_locations"))
    fn_out = join(tmpdir, "plots", "climate_historical", "region_plot.png")

    plot_region_and_location.plot_region_and_location(
        region_fn=region_filename,
        fn_out=fn_out,
        data_catalog=data_libs_fao,
        subregions_fn=subregions_filename,
        locations_fn=obs_fn,
        hydrography_fn=config_fao["hydrography_fn"],
        rivers_fn=config_fao["river_geom_fn"],
        buffer_km=config_fao["region_buffer"],
    )

    # Check if the output files are created
    assert isfile(fn_out)


def test_extract_climate_historical_grid(tmpdir, config_fao, data_libs_fao):
    """Test the extraction of historical climate data."""
    region_filename = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    fn_out = join(tmpdir, "climate_historical", "raw_data", "extract_chirps_global.nc")

    extract_historical_climate.prep_historical_climate(
        region_fn=region_filename,
        fn_out=fn_out,
        data_libs=data_libs_fao,
        clim_source="chirps_global",
        climate_variables=["precip", "temp"],
        starttime=config_fao["starttime"],
        endtime=config_fao["endtime"],
        buffer=config_fao["region_buffer"],
        combine_with_era5=False,
    )

    # Check if the output files are created
    assert isfile(fn_out)

    # Read the output file
    ds = xr.open_dataset(fn_out)
    assert "precip" in ds.data_vars
    assert "temp" not in ds.data_vars
    assert ds["source"].values.item() == "chirps_global"


def test_sample_historical_climate(tmpdir, config_fao, data_libs_fao):
    """Test extracting the climate data for specific regions and locations."""
    region_filename = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    subregions_filename = join(
        SAMPLE_PROJECTDIR,
        "hydrology_model",
        "staticgeoms",
        "subcatch_discharge-locations-grdc.geojson",
    )
    obs_fn = join(MAINDIR, get_config(config_fao, "climate_locations"))
    clim_filename = join(TESTDIR, "data", "chirps_global.nc")
    path_output = join(tmpdir, "climate_historical", "statistics")

    sample_climate_historical.sample_climate_historical(
        clim_filename=clim_filename,
        region_filename=region_filename,
        path_output=path_output,
        climate_catalog=data_libs_fao,
        clim_source="chirps_global",
        climate_variables=["precip", "temp"],
        subregions_filename=subregions_filename,
        locations_filename=obs_fn,
        buffer=config_fao["region_buffer"],
    )

    # Check if the output files are created
    assert isfile(f"{path_output}/basin_chirps_global.nc")
    assert isfile(f"{path_output}/point_chirps_global.nc")


def test_plot_basin_climate(tmpdir, config_fao):
    """Test the basin averaged plots."""
    grids = glob.glob(
        join(SAMPLE_PROJECTDIR, "climate_historical", "statistics", "basin_*.nc")
    )

    # Call the plot function
    plot_climate_basin.plot_historical_climate_region(
        climate_filenames=grids,
        path_output=join(tmpdir, "plots", "climate_historical", "region"),
        climate_sources=config_fao["clim_historical"],
        climate_sources_colors=config_fao["clim_historical_colors"],
        heat_threshold=15,
    )

    # Check if the output files are created
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/region/precipitation_region_1.png"
    )
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/region/temperature_region_1.png"
    )
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/region/precipitation_boxplot.png"
    )


def test_plot_climate_point(tmpdir, config_fao, data_libs_fao):
    """Test the point location plots."""
    locs = glob.glob(
        join(SAMPLE_PROJECTDIR, "climate_historical", "statistics", "point_*.nc")
    )

    obs_fn = join(MAINDIR, get_config(config_fao, "climate_locations"))
    timeseries_fn = join(
        MAINDIR, get_config(config_fao, "climate_locations_timeseries")
    )

    # Call the plot function
    plot_climate_location.plot_historical_climate_point(
        climate_filenames=locs,
        path_output=join(tmpdir, "plots", "climate_historical", "point"),
        climate_sources=config_fao["clim_historical"],
        climate_sources_colors=config_fao["clim_historical_colors"],
        data_catalog=data_libs_fao,
        locations_filename=obs_fn,
        precip_observations_filename=timeseries_fn,
        heat_threshold=15,
        export_observations=False,
    )

    # Check if the output files are created
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/point/precipitation_SILLIAN.png"
    )
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/point/temperature_SILLIAN.png"
    )


def test_timeseries_historical_trends(tmpdir):
    """Test the timeseries trends plots."""
    grids = glob.glob(
        join(SAMPLE_PROJECTDIR, "climate_historical", "statistics", "basin_*.nc")
    )

    derive_climate_trends.derive_timeseries_trends(
        clim_filenames=grids,
        path_output=join(tmpdir, "plots", "climate_historical", "trends"),
        split_year=2005,
    )

    # Check if the output files are created
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/trends/timeseries_anomalies_precip_era5.png"
    )
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/trends/timeseries_anomalies_temp_era5.png"
    )


def test_gridded_historical_trends(tmpdir, data_libs_fao):
    """Test the gridded trends plots."""
    region_filename = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    grids = glob.glob(
        join(SAMPLE_PROJECTDIR, "climate_historical", "raw_data", "extract_*.nc")
    )

    derive_climate_trends_gridded.derive_gridded_trends(
        climate_filenames=grids,
        path_output=join(tmpdir, "plots", "climate_historical"),
        data_catalog=data_libs_fao,
        region_filename=region_filename,
        river_filename="river_atlas",
        year_per_line=5,
        line_height_yearly_plot=6,
        line_height_mean_precip=6,
        fs_yearly_plot=10,
        fs_mean_precip=8,
    )

    # Check if the output files are created
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/trends/gridded_anomalies_precip_chirps_global.png"
    )
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/trends/gridded_anomalies_temp_era5.png"
    )
    assert not os.path.exists(
        f"{tmpdir}/plots/climate_historical/trends/gridded_trends.txt"
    )
    assert os.path.exists(
        f"{tmpdir}/plots/climate_historical/grid/mean_annual_precipitation.png"
    )
