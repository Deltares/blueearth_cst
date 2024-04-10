import os
from os.path import join

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, TESTDIR
from .conftest import get_config

from ..src import copy_config_files
from ..src import derive_region
from ..src import plot_climate_basin

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


def test_plot_climate_region(tmpdir, config_fao, data_libs_fao):
    """Test the basin averaged plots."""
    region_filename = join(SAMPLE_PROJECTDIR, "region", "region.geojson")
    subregions_filename = join(
        SAMPLE_PROJECTDIR,
        "hydrology_model",
        "staticgeoms",
        "subcatch_discharge-locations-grdc.geojson",
    )

    # Call the plot function
    plot_climate_basin.plot_historical_climate_region(
        region_filename=region_filename,
        path_output=join(tmpdir, "climate_historical"),
        climate_sources=config_fao["clim_historical"],
        climate_catalog=data_libs_fao,
        subregions_filename=subregions_filename,
        heat_threshold=15,
    )

    # Check if the output files are created
    assert os.path.exists(f"{tmpdir}/climate_historical/plots")
    assert os.path.exists(f"{tmpdir}/climate_historical/statistics/basin_climate.nc")
