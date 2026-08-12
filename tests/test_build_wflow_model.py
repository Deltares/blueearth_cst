"""Focused tests for the P1-to-Wflow public adapter boundary."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import Point

from blueearth_cst.model.build_wflow_model import (
    _BASE_CONFIG,
    _apply_parameter_steps,
    _validate_registry,
    arcgis_d8_to_wflow_ldd,
    read_parameter_steps,
)


def _template(path: Path, steps: list[dict]) -> Path:
    path.write_text(
        yaml.safe_dump({"modeltype": "wflow_sbm", "steps": steps}),
        encoding="utf-8",
    )
    return path


def test_parameter_template_rejects_setup_basemaps(tmp_path):
    path = _template(tmp_path / "build.yml", [{"setup_basemaps": {}}])

    with pytest.raises(ValueError, match="forbidden after P1"):
        read_parameter_steps(path)


def test_direct_base_maps_have_required_toml_mapping():
    assert _BASE_CONFIG["input"]["static"]["land_surface__slope"] == "land_slope"


def test_shipped_template_retains_only_wflow_parameter_steps():
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "wflow_build_model.yml"
    )

    steps = read_parameter_steps(path)

    assert [name for name, _ in steps] == [
        "setup_rivers",
        "setup_lulcmaps",
        "setup_laimaps",
        "setup_soilmaps",
        "setup_constant_pars",
    ]


def test_arcgis_d8_conversion_preserves_grid_and_active_domain():
    flow = xr.DataArray(
        np.array([[1, 4], [64, 0]], dtype="uint8"),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
    )
    basin = xr.DataArray(
        np.array([[1, 1], [1, 0]], dtype="int32"),
        coords=flow.coords,
        dims=flow.dims,
    )
    flow.raster.set_crs(4326)
    basin.raster.set_crs(4326)

    ldd = arcgis_d8_to_wflow_ldd(flow, basin)

    assert ldd.raster.shape == flow.raster.shape
    assert ldd.raster.bounds == flow.raster.bounds
    assert ldd.raster.crs.to_epsg() == 4326
    assert int(ldd.values[1, 1]) == 255
    assert set(np.unique(ldd.values[:1])).issubset(set(range(1, 10)))


def test_registry_requires_primary_ids_to_end_in_zero():
    registry = pd.DataFrame(
        {
            "subbasin_id": [101],
            "wflow_id": [999],
            "location_id": [1],
            "location_code": ["B001-S01-L01"],
            "location_role": ["control"],
        }
    )
    locations = gpd.GeoDataFrame(registry.copy(), geometry=[Point(1, 1)], crs=4326)

    with pytest.raises(ValueError, match="must end in 0"):
        _validate_registry(registry, locations)


def test_parameter_steps_use_p1_objects_and_keep_soil_catalog_owned():
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def setup_rivers(self, **kwargs):
            calls.append(("setup_rivers", kwargs))

        def setup_lulcmaps(self, **kwargs):
            calls.append(("setup_lulcmaps", kwargs))

        def setup_laimaps(self, **kwargs):
            calls.append(("setup_laimaps", kwargs))

        def setup_soilmaps(self, **kwargs):
            calls.append(("setup_soilmaps", kwargs))

        def setup_constant_pars(self, **kwargs):
            calls.append(("setup_constant_pars", kwargs))

    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    shape = (2, 2)
    maps = xr.Dataset(
        {
            "flow_direction": (("y", "x"), np.ones(shape, dtype="uint8")),
            "basin_id": (("y", "x"), np.ones(shape, dtype="int32")),
            "upstream_area": (("y", "x"), np.ones(shape)),
            "elevation": (("y", "x"), np.ones(shape)),
            "land_cover": (("y", "x"), np.ones(shape, dtype="int16")),
            "leaf_area_index": (
                ("month", "y", "x"),
                np.ones((12, *shape), dtype="float32"),
            ),
        },
        coords={**coords, "month": range(1, 13)},
        attrs={"lulc_source": "vito"},
    )
    maps.raster.set_crs(4326)
    rivers = gpd.GeoDataFrame(geometry=[], crs=4326)
    steps = [
        ("setup_rivers", {"hydrography_fn": "global", "river_upa": 32}),
        ("setup_lulcmaps", {"lulc_fn": "vito"}),
        ("setup_laimaps", {"lai_fn": "modis_lai"}),
        ("setup_soilmaps", {"soil_fn": "soilgrids"}),
        ("setup_constant_pars", {"snowpack__degree_day_coefficient": 3.7}),
    ]

    _apply_parameter_steps(FakeModel(), steps, maps, rivers)

    by_name = dict(calls)
    assert isinstance(by_name["setup_rivers"]["hydrography_fn"], xr.Dataset)
    assert by_name["setup_rivers"]["hydrography_fn"]["flwdir"].dtype == np.uint8
    assert by_name["setup_rivers"]["hydrography_fn"]["flwdir"].raster.nodata == 247
    assert by_name["setup_rivers"]["river_geom_fn"] is rivers
    assert isinstance(by_name["setup_lulcmaps"]["lulc_fn"], xr.DataArray)
    assert by_name["setup_lulcmaps"]["lulc_mapping_fn"] == "vito_mapping_default"
    assert isinstance(by_name["setup_laimaps"]["lai_fn"], xr.DataArray)
    assert by_name["setup_soilmaps"] == {"soil_fn": "soilgrids"}
