import os
from os.path import join

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from .conftest import MAINDIR, SAMPLE_PROJECTDIR, TESTDIR

from ..src import copy_config_files
from ..src import derive_region

config_fn = join(TESTDIR, "snake_config_model_test.yml")


def test_copy_config(tmpdir, data_sources_climate):
    """Test if config files are copied to tmpdir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fn,
        output_dir=join(tmpdir, "config"),
        config_out_name="snake_config_climate_projections.yml",
        other_config_files=[data_sources_climate],
    )

    # Check if config files are copied to tmpdir/config folder
    assert os.path.exists(f"{tmpdir}/config/snake_config_climate_projections.yml")
    assert os.path.exists(f"{tmpdir}/config/cmip6_data.yml")


def test_select_region(tmpdir):
    """Test if region is created properly."""
    # Check if derived region is the same as wflow model (default setting)
    basin_fn = join(
        SAMPLE_PROJECTDIR, "hydrology_model", "staticgeoms", "region.geojson"
    )
    region = {"geom": basin_fn}
    # Try to derive the region from config file
    derive_region.derive_region(
        region=region,
        path_output=join(tmpdir, "region"),
        buffer=None,
        data_catalog=None,
        hydrography_fn=None,
    )

    assert os.path.exists(f"{tmpdir}/region/region.geojson")

    # Check if region is the same as expected
    basin = gpd.read_file(basin_fn)
    region = gpd.read_file(f"{tmpdir}/region/region.geojson")
    assert_geodataframe_equal(region, basin, check_like=True, check_less_precise=True)
