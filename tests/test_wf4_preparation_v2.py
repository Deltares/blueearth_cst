"""WF4 binds shared elevation with a checked relocatable HydroMT catalog."""

import numpy as np
import pytest
import xarray as xr
import yaml

from blueearth_cst.climate_analysis import prepare_climate_data_catalog
from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
from blueearth_cst.experiment.wf4_ancillary_descriptor import (
    preparation_identity_v2,
    reader_identity_v2,
    resolve_wf4_preparation,
)
from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference


def test_wf4_elevation_binding_is_shared_and_relocatable(tmp_path, monkeypatch):
    source = tmp_path / "external/era5_orography_2018.nc"
    source.parent.mkdir()
    xr.Dataset(
        {"z": (("latitude", "longitude"), np.ones((2, 2)))},
        coords={"latitude": [1.0, 2.0], "longitude": [3.0, 4.0]},
    ).to_netcdf(source)
    interpretation = UnitInterpretation("fixture/1", "evidence", (("temp", "degC"),))
    source_entry = {
        "data_type": "RasterDataset",
        "uri": "unused.nc",
        "driver": "raster_xarray",
        "metadata": {"crs": 4326},
    }
    elevation_entry = {
        "data_type": "RasterDataset",
        "uri": str(source),
        "driver": "raster_xarray",
        "data_adapter": {"rename": {"z": "elevtn"}},
        "metadata": {"crs": 4326},
    }
    planned = prepare_climate_data_catalog.plan_preparation_payloads(
        source_entry, "era5", elevation_entry, source, interpretation
    )
    monkeypatch.setattr(
        prepare_climate_data_catalog,
        "resolve_preparation_payloads",
        lambda *_args: planned,
    )
    project = tmp_path / "project"
    experiment = project / "experiments/fixture"
    preparation = resolve_wf4_preparation(
        project,
        experiment,
        climate_source="era5",
        catalogs=[tmp_path / "unused.yml"],
        unit_interpretation=interpretation,
    )
    assert preparation["schema_version"] == "forcing-preparation/2"
    identity = preparation_identity_v2(
        preparation, project_root=project, experiment_root=experiment
    )
    assert identity["schema_version"] == "forcing-preparation-identity/2"
    elevation = preparation["ancillary"][0]["file"]
    assert elevation["path"].startswith("data/climate/ancillary/era5/")
    catalog_path = resolve_file_reference(
        preparation["catalog"], {"project_root": project}
    )
    uri = yaml.safe_load(catalog_path.read_text())["elevation"]["uri"]
    assert uri.startswith("../../../../../data/climate/ancillary/era5/")
    assert (
        resolve_wf4_preparation(
            project,
            experiment,
            climate_source="era5",
            catalogs=[tmp_path / "unused.yml"],
            unit_interpretation=interpretation,
        )
        == preparation
    )
    original_catalog = catalog_path.read_bytes()
    catalog_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="catalog differs"):
        resolve_wf4_preparation(
            project,
            experiment,
            climate_source="era5",
            catalogs=[tmp_path / "unused.yml"],
            unit_interpretation=interpretation,
        )
    catalog_path.write_bytes(original_catalog)
    shared = resolve_file_reference(elevation, {"project_root": project})
    shared.write_bytes(b"changed")
    with pytest.raises(ValueError, match="target differs"):
        resolve_wf4_preparation(
            project,
            experiment,
            climate_source="era5",
            catalogs=[tmp_path / "unused.yml"],
            unit_interpretation=interpretation,
        )


def test_reader_identity_classifies_extent_and_citations():
    entry = {
        "data_type": "RasterDataset",
        "driver": "raster_xarray",
        "metadata": {
            "crs": 4326,
            "license": "first citation",
            "extent": {"bbox": {"West": -1, "South": -2, "East": 3, "North": 4}},
            "cst_unit_interpretation": {
                "revision": "fixture/1",
                "evidence": "first evidence",
                "variables": [["temp", "degC"]],
            },
        },
    }
    first = reader_identity_v2(entry, elevation=False)
    entry["metadata"]["license"] = "second citation"
    entry["metadata"]["source_version"] = "ERA5 daily data on pressure levels"
    entry["metadata"]["cst_unit_interpretation"]["evidence"] = "second evidence"
    assert reader_identity_v2(entry, elevation=False) == first
    entry["metadata"]["extent"]["bbox"]["East"] = 5
    assert reader_identity_v2(entry, elevation=False) != first
    entry["metadata"]["unknown"] = "unclassified"
    with pytest.raises(ValueError, match="unclassified"):
        reader_identity_v2(entry, elevation=False)
