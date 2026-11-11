"""Independent native-cell, matched-period and two-view comparison contracts."""

import calendar
import json
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

from blueearth_cst.climate_analysis.diagnostic_tables import verified_temperature
from blueearth_cst.climate_analysis.location_comparison import (
    evaluate,
    monthly_agreement,
    temperature_tables,
)
from blueearth_cst.climate_analysis.plot_location_comparison import (
    agreement_figure,
    render,
)
from blueearth_cst.climate_analysis.rain_gauges import parse_rain_gauges


@pytest.mark.parametrize("with_gauges", [False, True])
def test_location_views_sample_native_points_and_retain_observations(
    tmp_path, with_gauges
):
    region = tmp_path / "region.geojson"
    subbasins = tmp_path / "subbasins.geojson"
    gpd.GeoDataFrame(
        {"subbasin_id": [101]}, geometry=[box(-1, -1, 4, 4)], crs=4326
    ).to_file(subbasins)
    gpd.GeoDataFrame(geometry=[box(-1, -1, 4, 4)], crs=4326).to_file(region)
    locations = tmp_path / "locations.geojson"
    gpd.GeoDataFrame(
        {
            "wflow_id": [1010, 1011],
            "station_name": ["Outlet", "Control"],
            "location_id": [1, 2],
            "location_code": ["B001-S01-L01", "B001-S01-L02"],
            "location_role": ["automatic_outlet", "control"],
            "subbasin_id": [101, 101],
            "original_x": [0.2, 0.3],
            "original_y": [0.2, 0.3],
        },
        geometry=[Point(0.2, 0.2), Point(0.3, 0.3)],
        crs=4326,
    ).to_file(locations)
    axis = pd.date_range("2000-01-02", "2001-01-01")
    month = (axis - pd.Timedelta(days=1)).month.to_numpy()
    sources = {}
    for source, multiplier in (("era5", 1), ("chirps", 2)):
        values = (
            month[:, None, None]
            * np.array([[3.0, 4.0], [1.0, 2.0]])[None, :, :]
            * multiplier
        )
        ds = xr.Dataset(
            {"precip": (("time", "latitude", "longitude"), values)},
            coords={"time": axis, "latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
        )
        ds.precip.attrs["units"] = "mm/day"
        if source == "era5":
            ds["temp"] = xr.full_like(ds.precip, 0) + xr.DataArray(
                month - 6.0, dims="time", coords={"time": axis}
            )
            ds.temp.attrs["units"] = "degC"
            if with_gauges:
                ds.temp.attrs["units"] = "K"
                ds.attrs["region_geojson_sha256"] = "test-extraction-provenance"
        ds.raster.set_crs(4326)
        sources[source] = str(tmp_path / f"{source}.nc")
        ds.to_netcdf(sources[source])
    gauges = None
    if with_gauges:
        stations = tmp_path / "stations.csv"
        stations.write_text(
            "id,name,x,y\n001,Outlet gauge,0.2,0.2\n002,Hill gauge,1.2,1.2\n003,Outside,5,5\n004,NoGrid,3,3\n"
        )
        precip = tmp_path / "precip.csv"
        temp = tmp_path / "temp.csv"
        pd.DataFrame(
            {key: month for key in ("001", "002", "003", "004")}, index=axis
        ).to_csv(precip, index_label="time")
        pd.DataFrame({"001": month - 6.0}, index=axis).to_csv(temp, index_label="time")
        gauges = {
            "stations_path": str(stations),
            "precip_path": str(precip),
            "temp_path": str(temp),
            "time_label": "interval_end",
        }
        gauges = parse_rain_gauges(gauges)
    catalog = tmp_path / "catalog.yml"
    catalog.write_text(
        "alternate:\n  data_adapter:\n    unit_add:\n      temp: -273.15\n"
    )
    forcing = (
        {"era5": {"temp_pet_source": "alternate", "orography_source": "alternate_dem"}}
        if with_gauges
        else None
    )
    root = tmp_path / "comparison"
    evaluate(
        root,
        sources,
        str(locations),
        str(subbasins),
        str(region),
        {"start": 2000, "end": 2000},
        {"start": 2000, "end": 2000},
        1,
        gauges,
        [str(catalog)],
        forcing,
    )
    registry = pd.read_csv(
        root / "locations.csv",
        dtype={"id": str, "gauge_id": str},
        keep_default_na=False,
    ).set_index("id")
    assert set(registry.index) == {
        "location_1010",
        "location_1011",
        *({"station_002", "station_003", "station_004"} if with_gauges else set()),
    }
    assert registry.loc["location_1010", "gauge_id"] == ("001" if with_gauges else "")
    assert registry.loc["location_1010", "location_code"] == "B001-S01-L01"
    table_root = root / "tables/locations/location_1010"
    monthly = pd.read_csv(table_root / "monthly.csv")
    february = monthly[monthly.month == "2000-02"].set_index("source")
    assert february.loc["era5", "total_mm"] == 58
    assert february.loc["chirps", "total_mm"] == 116
    annual = pd.read_csv(table_root / "annual.csv").set_index("source")
    expected = sum(calendar.monthrange(2000, m)[1] * m for m in range(1, 13))
    assert annual.loc["era5", "total_mm"] == expected
    assert annual.loc["chirps", "total_mm"] == expected * 2
    temperatures = pd.read_csv(table_root / "temp_monthly.csv")
    january = temperatures[temperatures.period == "2000-01"].set_index("source")
    assert january.loc["era5", "mean_degC"] == -5
    if with_gauges:
        assert february.loc["observed", "total_mm"] == 58
        assert january.loc["observed", "mean_degC"] == -5
        metrics = pd.read_csv(table_root / "monthly_agreement.csv").set_index("source")
        assert metrics.loc["era5", "paired_months"] == 12
        assert metrics.loc["era5", "pearson_r"] == pytest.approx(1)
        assert metrics.loc["era5", "bias"] == 0
        assert metrics.loc["chirps", "bias"] == pytest.approx(expected / 12)
        excluded = pd.read_csv(root / "_engine/locations/exclusions.csv")
        assert {"outside_basin", "outside_source_grid", "no_observation_column"} <= set(
            excluded.reason
        )
    before = {p: p.read_bytes() for p in table_root.glob("*.csv")}
    records = render(root)
    assert records
    assert all((root / key).is_file() for key in records)
    assert not (root / "plots/subbasins").exists()
    assert not (root / "stations").exists()
    assert all(Path(key).parent.parent == Path("plots/locations") for key in records)
    assert not any("daily_agreement" in key for key in records)
    assert any("monthly_agreement" in key for key in records) == with_gauges
    assert any("temp_monthly_agreement" in key for key in records) == with_gauges
    assert (root / "plots/locations/location_1010/figure_titles.md").is_file()
    assert (
        json.loads((root / "_engine/plots/locations/figure_captions.json").read_text())[
            "figures"
        ].keys()
        == records.keys()
    )
    assert all(p.read_bytes() == content for p, content in before.items())
    if with_gauges:
        temperature_record = next(
            item for item in records.values() if item["variable"] == "temp"
        )
        assert "ALTERNATE (on ERA5 grid)" in temperature_record["caption"]
        cells = pd.read_csv(table_root / "native_cells.csv")
        assert cells.loc[cells.variable == "temp", "data_source"].eq("alternate").all()


@pytest.mark.parametrize("values", [[20.0, np.inf], [20.0, -np.inf], [np.nan, np.inf]])
def test_verified_temperature_rejects_infinite_values(values):
    verified, _ = verified_temperature(pd.Series(values), {"units": "degC"}, {}, "era5")
    assert not verified


def test_temperature_gaps_and_negative_values_remain_explicit():
    axis = pd.date_range("1999-10-02", "2000-10-01")
    frame = pd.DataFrame({"observed": -4.0, "era5": -3.0}, index=axis)
    frame.loc["2000-02-10", "observed"] = np.nan
    tables = temperature_tables(frame, 10)
    feb = tables["monthly"].query("period == '2000-02'")
    assert feb.expected_days.eq(29).all()
    assert feb.paired_days.eq(28).all()
    assert feb.mean_degC.isna().all()
    assert tables["annual"].mean_degC.isna().all()
    january = tables["monthly"].query("period == '2000-01'").set_index("source")
    assert january.loc["observed", "mean_degC"] == -4


def test_monthly_panels_keep_pair_counts_and_identical_axes():
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared.plot_preset import CST_PRESET

    table = pd.DataFrame(
        [
            {"period": month, "source": source, "value": value}
            for source, values in (
                ("observed", [1, 2, 3]),
                ("era5", [2, 4, 6]),
                ("chirps", [np.nan, np.nan, np.nan]),
            )
            for month, value in zip(
                ["2000-01", "2000-02", "2000-03"], values, strict=True
            )
        ]
    )
    metrics = monthly_agreement(table, "value")
    assert metrics.set_index("source").loc["era5", "rmse"] == pytest.approx(
        np.sqrt(14 / 3)
    )
    assert metrics.set_index("source").loc["chirps", "paired_months"] == 0
    with use_preset(CST_PRESET):
        fig = agreement_figure(
            table,
            "value",
            {"era5": ("blue", "o"), "chirps": ("red", "s")},
            {"era5": "ERA5", "chirps": "CHIRPS"},
            "mm/month",
            "precip",
            metrics,
        )
        assert len(fig.axes) == 2
        assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
        assert any("n = 3 months" in text.get_text() for text in fig.axes[0].texts)
        assert any(
            "No complete paired months" in text.get_text() for text in fig.axes[1].texts
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
