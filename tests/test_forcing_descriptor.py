"""Persisted physical descriptors observe real NetCDF bytes and explicit units."""

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from blueearth_cst.experiment.content_identity import canonical_json_bytes
from blueearth_cst.experiment.forcing_descriptor import (
    collection_forcing_descriptor,
)
from blueearth_cst.experiment.wf4_ancillary_descriptor import describe_ancillary


@pytest.fixture
def physical(tmp_path):
    path = tmp_path / "forcing.nc"
    ds = xr.Dataset(
        {"temp": (("time", "latitude", "longitude"), np.ones((3, 2, 2)))},
        coords={
            "time": np.arange("2001-01-01", "2001-01-04", dtype="datetime64[D]"),
            "latitude": [1.0, 2.0],
            "longitude": [3.0, 4.0],
        },
    )
    ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": CRS.from_epsg(4326).to_wkt()})
    ds.temp.attrs["units"] = "K"
    ds.to_netcdf(path)
    reader = {
        "metadata": {
            "cst_unit_interpretation": {
                "revision": "fixture/1",
                "evidence": "converted upstream; labels retained",
                "variables": [["temp", "degC"]],
            }
        }
    }
    return path, ds, reader


def test_collection_descriptor_preserves_effective_units_and_raw_fill(physical):
    path, _, reader = physical
    before = path.read_bytes()
    result = collection_forcing_descriptor(path, reader)
    assert result["source_calendar"] == "proleptic_gregorian"
    assert result["timestep"] == "P1D"
    assert result["start"] == "2001-01-01 00:00:00"
    assert result["variables"] == [
        {
            "name": "temp",
            "units": "degC",
            "missing_value": {"_FillValue": {"nonfinite": "nan"}},
        }
    ]
    canonical_json_bytes(result)
    assert path.read_bytes() == before


def test_grid_identity_tracks_coordinates_but_not_forcing_values(physical):
    path, ds, reader = physical
    original = collection_forcing_descriptor(path, reader)
    ds.temp.values[:] = 42
    ds.to_netcdf(path, mode="w")
    assert collection_forcing_descriptor(path, reader) == original
    ds = ds.assign_coords(longitude=[3.0, 4.1])
    ds.to_netcdf(path, mode="w")
    assert (
        collection_forcing_descriptor(path, reader)["spatial_representation_sha256"]
        != original["spatial_representation_sha256"]
    )


def test_missing_interpretation_refuses_native_label_fallback(physical):
    path, _, _ = physical
    with pytest.raises(ValueError, match="unit interpretation"):
        collection_forcing_descriptor(path, {"metadata": {}})


def test_ancillary_descriptor_observes_grid_units_and_missingness(physical):
    path, ds, _ = physical
    ds = ds.isel(time=0, drop=True).rename({"temp": "elevtn"})
    ds.elevtn.attrs["units"] = "m"
    ds.elevtn.values[0, 0] = np.nan
    ds.to_netcdf(path, mode="w")
    result = describe_ancillary(path)
    assert result["crs"] == "EPSG:4326"
    assert result["variables"][0]["name"] == "elevtn"
    assert result["variables"][0]["missing_count"] == 1
    assert result["variables"][0]["units"] == "m"
    canonical_json_bytes(result)


@pytest.mark.parametrize(
    "source,pet",
    [
        ("era5", "debruin"),
        ("chirps", "debruin"),
        ("chirps_global", "debruin"),
        ("eobs", "makkink"),
    ],
)
def test_preparation_plan_retains_reader_and_elevation_bytes(physical, source, pet):
    import yaml

    from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
        plan_preparation_payloads,
    )
    from blueearth_cst.experiment.forcing_descriptor import reader_unit_interpretation

    path, ds, reader = physical
    ds = ds.isel(time=0, drop=True).drop_vars("spatial_ref")
    ds.to_netcdf(path, mode="w")
    source_entry = {
        "data_type": "RasterDataset",
        "uri": "unavailable.nc",
        "driver": "netcdf",
        "data_adapter": {"unit_mult": {"temp": 99}},
        "metadata": {"crs": 4326},
    }
    elevation = {
        "data_type": "RasterDataset",
        "uri": "original.nc",
        "driver": {
            "name": "raster_xarray",
            "options": {"preprocess": "harmonise_dims"},
        },
        "data_adapter": {
            "rename": {"z": "elevtn"},
            "unit_mult": {"elevtn": 0.10197162129779283},
        },
        "metadata": {"crs": 4326},
    }
    before = path.read_bytes()
    context, catalog, payloads = plan_preparation_payloads(
        source_entry, source, elevation, path, reader_unit_interpretation(reader)
    )
    assert context["pet_method"] == pet
    assert "data_adapter" not in context["generated_forcing_reader"]
    retained = yaml.safe_load(catalog)["elevation"]
    assert retained["driver"] == elevation["driver"]
    assert retained["data_adapter"] == elevation["data_adapter"]
    assert retained["metadata"] == elevation["metadata"]
    assert payloads[retained["uri"]].read_bytes() == before
    assert context["ancillary"][0]["descriptor"]["crs"] is None
    canonical_json_bytes(context)
