"""Tests for blueearth_cst/model/write_outlet_index.py (R3 sections 4, 8).

Round-trips a tiny outlets.geojson through real geopandas — no hydromt or model
build. Sibling tests (test_extract_historical_climate, test_stage_data) stub
geopandas via sys.modules.setdefault, which would shadow the real package here
(the M02c pollution hazard, dev/tasks/ R3+). The `woi` fixture removes
that stub for this test only (monkeypatch restores it) and imports the module
fresh against the real geopandas.
"""

import importlib
import sys

import pytest


@pytest.fixture()
def woi(monkeypatch):
    """blueearth_cst.model.write_outlet_index imported against the real (un-stubbed) geopandas."""
    for name in [
        m for m in list(sys.modules) if m == "geopandas" or m.startswith("geopandas.")
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("geopandas")  # bind the real package in sys.modules
    monkeypatch.delitem(
        sys.modules, "blueearth_cst.model.write_outlet_index", raising=False
    )
    return importlib.import_module("blueearth_cst.model.write_outlet_index")


def _tiny_outlets(path):
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"fid": [130000086, 5]},
        geometry=[Point(9.6625, 0.44583), Point(10.0, 1.0)],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")


def _registry(path):
    import pandas as pd

    pd.DataFrame(
        {
            "basin_code": ["B001", "B001"],
            "subbasin_id": [130000086, 5],
            "subbasin_code": ["B001-S01", "B001-S02"],
            "location_id": [1, 1],
            "location_code": ["B001-S01-L01", "B001-S02-L01"],
            "station_name": ["outlet_a", "outlet_b"],
            "wflow_id": [130000086, 5],
        }
    ).to_csv(path, index=False)


def test_build_outlet_index_positional_naming(tmp_path, woi):
    outlets = tmp_path / "outlets.geojson"
    _tiny_outlets(outlets)
    registry = tmp_path / "registry.csv"
    _registry(registry)
    df = woi.build_outlet_index(outlets, registry)
    assert list(df["compat_station_name"]) == ["wflow_1", "wflow_2"]
    assert list(df["subcatchment_id"]) == [130000086, 5]
    assert list(df["station_name"]) == ["outlet_a", "outlet_b"]
    assert list(df["wflow_id"]) == [130000086, 5]
    assert df["x"].tolist() == pytest.approx([9.6625, 10.0])
    assert df["y"].tolist() == pytest.approx([0.44583, 1.0])


def test_write_outlet_index_creates_csv(tmp_path, woi):
    outlets = tmp_path / "outlets.geojson"
    _tiny_outlets(outlets)
    registry = tmp_path / "registry.csv"
    _registry(registry)
    out = tmp_path / "sub" / "outlet_index.csv"  # parent does not exist yet
    woi.write_outlet_index(outlets, registry, out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "compat_station_name,subcatchment_id,x,y,basin_code,subbasin_id,"
        "subbasin_code,location_code,station_name,wflow_id"
    )
