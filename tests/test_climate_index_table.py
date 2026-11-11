"""Direct climate-index tables preserve spatial and aggregation semantics."""

import json
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401 -- registers the raster accessor
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, Point, box

from blueearth_cst.climate_analysis.climate_index_table import (
    export_climate_index_table,
)
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings
from blueearth_cst.indicators.climate.definitions import INDEX_METADATA
from blueearth_cst.indicators.climate.evaluate import evaluate
from blueearth_cst.indicators.climate.periods import annual_periods, seasonal_periods
from blueearth_cst.indicators.climate.reference import fit_reference
from blueearth_cst.shared.provenance import file_sha256


def _read(path):
    return pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            key: str
            for key in (
                "spatial_id",
                "subbasin_id",
                "wflow_id",
                "spatial_location_id",
                "period",
            )
        },
    ).assign(value=lambda frame: pd.to_numeric(frame.value, errors="coerce"))


@pytest.fixture(scope="module", params=["era5", "chirps"])
def exported(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"climate-index-table-{request.param}")
    dates = pd.date_range("2000-01-01", "2001-12-31")
    shape = (len(dates), 2, 2)
    precipitation = np.zeros(shape)
    precipitation[::2, 0, 0] = 20.0
    precipitation[1::2, 1, 1] = 20.0
    data = xr.Dataset(
        coords={"time": dates, "latitude": [1.0, 2.0], "longitude": [3.0, 4.0]}
    )
    for name, values, units in (
        ("precip", precipitation, "mm/day"),
        ("temp", np.full(shape, 20.0), "degC"),
        ("temp_min", np.full(shape, 15.0), "degC"),
        ("temp_max", np.full(shape, 25.0), "degC"),
    ):
        data[name] = xr.DataArray(
            values, dims=("time", "latitude", "longitude"), attrs={"units": units}
        )
    data.raster.set_crs(4326)
    climate, cells = root / "climate.nc", root / "cells.csv"
    data.to_netcdf(climate)
    pd.DataFrame({"latitude": [1.0, 2.0], "longitude": [3.0, 4.0]}).to_csv(
        cells, index=False
    )
    settings = parse_settings(
        {"climate_indices": {"enabled": True, "seasons": {"JJA": [6, 7, 8]}}},
        {"start": 2000, "end": 2001},
        [request.param],
    )
    subbasins = root / "subbasins.geojson"
    diagonal = MultiPolygon([box(2.8, 0.8, 3.2, 1.2), box(3.8, 1.8, 4.2, 2.2)])
    gpd.GeoDataFrame(
        {
            "subbasin_id": ["001", "002", "099", "100"],
            "name": ["Diagonal", "First cell", "Outside", "Missing"],
        },
        geometry=[diagonal, box(2.8, 0.8, 3.2, 1.2), box(30, 30, 31, 31), None],
        crs=4326,
    ).to_file(subbasins, driver="GeoJSON")
    locations = root / "locations.geojson"
    gpd.GeoDataFrame(
        {
            "wflow_id": ["01001", "01002", "01003", "01004"],
            "original_x": [3.1, 50.0, 4.0, None],
            "original_y": [1.1, 50.0, 1.0, None],
            "station_name": [
                "Original point",
                "Outside grid",
                "Outside basin",
                "Missing coordinates",
            ],
            "location_id": ["A", "B", "C", "D"],
            "subbasin_id": ["002", "", "", ""],
        },
        # Snapped elsewhere: only the original coordinates may sample climate.
        geometry=[Point(4, 2)] * 4,
        crs=4326,
    ).to_file(locations, driver="GeoJSON")
    store = root / "store"
    args = (climate, cells, subbasins, locations, store / "tables/climate_indices.csv")
    options = {
        "source": request.param,
        "settings": settings,
        "start_month": 1,
        "record_path": store / "_engine/climate_indices.json",
        "spell_path": store / "_engine/wet_spell_exceedance.csv",
    }
    assert export_climate_index_table(*args, **options) == args[-1]
    return args, options, _read(args[-1]), data


def test_all_supports_and_periods_keep_every_registered_index(exported):
    _, _, table, _ = exported
    assert set(table.spatial_scope) == {"basin", "subbasin", "location"}
    assert set(table.loc[table.spatial_scope.eq("subbasin"), "spatial_id"]) == {
        "001",
        "002",
        "099",
        "100",
    }
    assert set(table.loc[table.spatial_scope.eq("location"), "spatial_id"]) == {
        "01001",
        "01002",
        "01003",
        "01004",
    }
    assert set(table.period) == {"2000", "2001", "JJA:2000", "JJA:2001"}
    for _, group in table.groupby(["spatial_scope", "spatial_id", "period"]):
        assert set(group["index"]) == set(INDEX_METADATA)
    assert len(table) == 9 * 4 * len(INDEX_METADATA)
    assert not table.duplicated(
        ["spatial_scope", "spatial_id", "period", "index"]
    ).any()


def test_basin_and_original_point_match_public_kernels_without_retained_grid(exported):
    _, options, table, raw = exported
    data = raw if options["source"] == "era5" else raw[["precip"]]
    point = data.sel(latitude=1.0, longitude=3.0)
    # The two declared diagonal basin cells are averaged daily, not their indices.
    basin = (point + data.sel(latitude=2.0, longitude=4.0)) / 2
    for variable in basin:
        basin[variable].attrs = dict(data[variable].attrs)
    periods = annual_periods(2000, 2001) + seasonal_periods(
        2000, 2001, name="JJA", months=(6, 7, 8)
    )
    for scope, identifier, series in (
        ("basin", "basin", basin),
        ("location", "01001", point),
    ):
        reference = fit_reference(series, 2000, 2001)
        expected = evaluate(series, reference, periods)[0]
        rows = table[table.spatial_scope.eq(scope) & table.spatial_id.eq(identifier)]
        for row in rows.itertuples():
            if row.index in expected:
                np.testing.assert_allclose(
                    row.value,
                    expected[row.index].sel(period=row.period).item(),
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                )
            else:
                assert pd.isna(row.value) and row.status == "unavailable"


def test_daily_aggregate_then_threshold_index_differs_from_mean_cellwise_indices(
    exported,
):
    _, _, table, _ = exported
    selected = table[table.period.eq("2000") & table["index"].eq("r10mm")]
    assert selected.loc[selected.spatial_scope.eq("basin"), "value"].item() == 366
    assert (
        selected.loc[
            selected.spatial_scope.eq("subbasin") & selected.spatial_id.eq("001"),
            "value",
        ].item()
        == 366
    )
    # Each native cell alternates 20/0 mm; its threshold count is half the year.
    assert (
        selected.loc[
            selected.spatial_scope.eq("location") & selected.spatial_id.eq("01001"),
            "value",
        ].item()
        == 183
    )
    assert (
        selected.loc[
            selected.spatial_scope.eq("subbasin") & selected.spatial_id.eq("002"),
            "value",
        ].item()
        == 183
    )


def test_unavailable_variables_and_spatial_supports_are_explicit(exported):
    _, options, table, _ = exported
    for scope, identifier, reason in (
        ("subbasin", "099", "subbasin intersects no native climate cells"),
        ("subbasin", "100", "subbasin intersects no native climate cells"),
        ("location", "01002", "outside native climate grid"),
        ("location", "01003", "outside_basin"),
        ("location", "01004", "invalid original location coordinates"),
    ):
        missing = table[table.spatial_scope.eq(scope) & table.spatial_id.eq(identifier)]
        assert missing.value.isna().all()
        assert (
            missing.status.eq("unavailable").all() and missing.reason.eq(reason).all()
        )
    if options["source"] == "chirps":
        missing = table[table.spatial_scope.eq("basin") & table["index"].eq("txx")]
        assert missing.value.isna().all()
        assert missing.reason.eq("not available from this source").all()


def test_only_csv_record_and_basin_spell_product_are_retained(exported):
    args, options, _, _ = exported
    root = options["record_path"].parent.parent
    assert {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    } == {
        "tables/climate_indices.csv",
        "_engine/climate_indices.json",
        "_engine/wet_spell_exceedance.csv",
    }
    record = json.loads(options["record_path"].read_text())
    assert record["schema_version"] == "climate-index-results/1"
    assert set(record["products"]) == {"table", "spells"}
    for product in record["products"].values():
        assert file_sha256(root / product["path"]) == product["sha256"]
    assert record["inputs"]["climate"] == file_sha256(args[0])
    assert record["inputs"]["cells"] == file_sha256(args[1])
    # Basin rain is continuously wet; both open-ended events are censored.
    assert pd.read_csv(options["spell_path"]).empty


def _existing_targets(tmp_path, options):
    root = tmp_path / "store"
    replacement = options | {
        "record_path": root / "_engine/climate_indices.json",
        "spell_path": root / "_engine/wet_spell_exceedance.csv",
    }
    output = root / "tables/climate_indices.csv"
    targets = [output, replacement["record_path"], replacement["spell_path"]]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing product")
    return output, replacement, targets


def test_invalid_basin_cells_preserve_existing_products(exported, tmp_path):
    args, options, _, _ = exported
    output, options, targets = _existing_targets(tmp_path, options)
    cells = tmp_path / "invalid_cells.csv"
    cells.write_text("latitude,longitude\n30,30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Declared basin cells"):
        export_climate_index_table(args[0], cells, args[2], args[3], output, **options)
    assert all(path.read_bytes() == b"existing product" for path in targets)


@pytest.mark.parametrize("drift", ["input", "implementation"])
def test_calculation_drift_preserves_all_existing_products(
    exported, monkeypatch, tmp_path, drift
):
    from blueearth_cst.climate_analysis import climate_index_table as module

    args, options, _, _ = exported
    output, options, targets = _existing_targets(tmp_path, options)
    locations = tmp_path / "locations.geojson"
    locations.write_bytes(Path(args[3]).read_bytes())
    evaluate_support = module._evaluate_support
    changed = False

    def change_input(*values):
        nonlocal changed
        result = evaluate_support(*values)
        if not changed:
            if drift == "input":
                locations.write_bytes(locations.read_bytes() + b" ")
            else:
                monkeypatch.setattr(
                    module, "implementation_digest", lambda: "different"
                )
            changed = True
        return result

    monkeypatch.setattr(module, "_evaluate_support", change_input)
    with pytest.raises(ValueError, match="changed during calculation"):
        export_climate_index_table(*args[:3], locations, output, **options)
    assert all(path.read_bytes() == b"existing product" for path in targets)
