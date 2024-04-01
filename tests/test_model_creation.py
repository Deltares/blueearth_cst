"""Test functions from the model creation workflow."""

import os
from os.path import join
import subprocess
import pytest

import xarray as xr
from hydromt_wflow import WflowModel

from .conftest import SNAKEDIR, SAMPLE_PROJECTDIR, config_fn
from .conftest import get_config

from ..src import copy_config_files
from ..src import setup_reservoirs_lakes_glaciers
from ..src import setup_gauges_and_outputs


def test_copy_config(project_dir, data_sources, model_build_config):
    """Test if config files are copied to project_dir/config folder"""
    # Call the copy file function
    copy_config_files.copy_config_files(
        config=config_fn,
        output_dir=join(project_dir, "config"),
        config_out_name="snake_config_model_creation.yml",
        other_config_files=[data_sources, model_build_config],
    )

    # Check if config files are copied to project_dir/config folder
    assert os.path.exists(f"{project_dir}/config/snake_config_model_creation.yml")
    assert os.path.exists(f"{project_dir}/config/wflow_build_model.yml")
    assert os.path.exists(f"{project_dir}/config/tests_data_catalog.yml")


def test_create_model_full(
    project_dir, config, model_build_config, waterbodies_config, data_sources
):
    """Test building the wflow model including all steps."""

    ### 1. Create model base ###
    basin_dir = f"{project_dir}/hydrology_model"
    model_region = get_config(config, "model_region", optional=False)
    model_resolution = get_config(config, "model_resolution", 0.00833333)
    hydromt_ini = model_build_config

    # Run hydromt build command similarly as in snakemake workflow
    cmd = f"""hydromt build wflow {basin_dir} --region "{model_region}" --opt setup_basemaps.res={model_resolution} -i {hydromt_ini} -d {data_sources} --fo -vv"""
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # Check the output of the subprocess command
    assert result.returncode == 0

    # Check the output files
    assert os.path.exists(f"{project_dir}/hydrology_model/staticmaps.nc")
    assert os.path.exists(f"{project_dir}/hydrology_model/wflow_sbm.toml")

    # Compare to the wflow_dem sample output file as not all variables are built yet
    # gauges, waterbodies, glaciers etc are not yet added
    ds_project = xr.open_dataset(f"{project_dir}/hydrology_model/staticmaps.nc")
    ds_sample = xr.open_dataset(f"{SAMPLE_PROJECTDIR}/hydrology_model/staticmaps.nc")

    xr.testing.assert_allclose(ds_project["wflow_dem"], ds_sample["wflow_dem"])

    # Close and release the datasets
    ds_project.close()
    ds_sample.close()

    ### 2. Add reservoirs, lakes and glaciers to the built model ###
    setup_reservoirs_lakes_glaciers.update_wflow_waterbodies_glaciers(
        wflow_root=basin_dir,
        data_catalog=data_sources,
        config_fn=waterbodies_config,
    )

    # Check the output files
    # In this case we should get glaciers and reservoirs added but no lakes
    assert os.path.exists(
        f"{project_dir}/hydrology_model/staticgeoms/reservoirs_lakes_glaciers.txt"
    )
    assert os.path.exists(f"{project_dir}/hydrology_model/staticgeoms/glaciers.geojson")
    assert os.path.exists(
        f"{project_dir}/hydrology_model/staticgeoms/reservoirs.geojson"
    )
    assert not os.path.exists(
        f"{project_dir}/hydrology_model/staticgeoms/lakes.geojson"
    )

    ### 3. Add gauges and select outputs ###
    gauges_fn = get_config(config, "output_locations", None)
    gauges_fn = join(SNAKEDIR, gauges_fn)
    outputs = get_config(config, "wflow_outvars")

    setup_gauges_and_outputs.update_wflow_gauges_outputs(
        wflow_root=basin_dir,
        data_catalog=data_sources,
        gauges_fn=gauges_fn,
        outputs=outputs,
    )

    # Check the output files
    assert os.path.exists(f"{project_dir}/hydrology_model/staticgeoms/gauges.geojson")
    assert os.path.exists(
        f"{project_dir}/hydrology_model/staticgeoms/gauges_discharge-locations-grdc.geojson"
    )

    # Check if our output vars are in the wflow_sbm.toml file
    wflow = WflowModel(root=basin_dir, mode="r")

    vars = [v["header"] for v in wflow.config["csv"]["column"]]
    for output in outputs:
        assert f"{output}_basavg" in vars
