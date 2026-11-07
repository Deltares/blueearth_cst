"""Rain-gauge identities, interval attribution, sampling and independent metrics."""

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.climate_analysis.rain_gauges import (
    matched_tables,
    parse_rain_gauges,
    read_precip,
    read_stations,
)


def test_station_identity_and_validation(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text("id,name,x,y\n001,Same,1,2\n002,Same,2,3\na/b,Other,3,4\n")
    stations = read_stations(path)
    assert list(stations.index) == ["001", "002", "a/b"]
    assert stations.token.is_unique
    assert stations.loc["001", "token"] == "001"
    path.write_text("id,x,y\n001,1,2\n001,2,3\n")
    with pytest.raises(ValueError, match="unique"):
        read_stations(path)


@pytest.mark.parametrize(
    "body",
    [
        "time,001,001\n2000-01-01,1,2\n",
        "time,001\n2000-01-01,-1\n",
        "time,001\n2000-01-01,inf\n",
        "time,001\n2000-01-01,invalid\n",
        "time,001\n2000-01-01,1\n2000-01-01,2\n",
        "time,001\n2000-01-01T09:00,1\n",
    ],
)
def test_invalid_observations(tmp_path, body):
    path = tmp_path / "obs.csv"
    path.write_text(body)
    with pytest.raises(ValueError):
        read_precip(path, "interval_end")


def test_labels_missing_rows_and_dry_days(tmp_path):
    path = tmp_path / "obs.csv"
    path.write_text("time,001\n2000-01-01,0\n2000-01-03,\n")
    frame = read_precip(path, "interval_start").reindex(
        pd.date_range("2000-01-02", "2000-01-04")
    )
    assert frame.iloc[0, 0] == 0
    assert frame.iloc[1:, 0].isna().all()


def test_independent_metrics_and_unequal_source_gaps():
    daily = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 4.0, 100.0],
            "a": [2.0, 4.0, 3.0, 99.0],
            "b": [1.0, 2.0, 4.0, np.nan],
        },
        index=pd.date_range("2000-01-02", periods=4),
    )
    result = matched_tables(daily, 1)
    row = result["agreement"].set_index("source").loc["a"]
    assert row.paired_days == 3
    assert row.bias_mm == pytest.approx(2 / 3)
    assert row.mae_mm == pytest.approx(4 / 3)
    assert row.rmse_mm == pytest.approx(np.sqrt(2))
    assert row.percentage_bias == pytest.approx(200 / 7)


def test_leap_month_and_water_year_attribution():
    # Ends March 1 belong to leap February; ends October 1 belong to prior water year.
    ends = pd.date_range("1999-10-02", "2000-10-01")
    daily = pd.DataFrame({"observed": 1.0, "a": 2.0}, index=ends)
    daily.loc["2000-03-01", "a"] = 12
    result = matched_tables(daily, 10)
    feb = result["monthly"].query("month == '2000-02' and source == 'a'").iloc[0]
    assert feb.total_mm == 68
    assert feb.expected_days == 29
    annual = result["annual"].query("source == 'a'").iloc[0]
    assert annual.reporting_year == 1999
    assert annual.total_mm == 742
    incomplete = matched_tables(daily.iloc[:-1], 10)["annual"]
    assert incomplete.total_mm.isna().all()


@pytest.mark.parametrize(
    "observed,source",
    [
        ([0.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [2.0, 2.0]),
        ([np.nan, np.nan], [2.0, 3.0]),
    ],
)
def test_undefined_metrics(observed, source):
    result = matched_tables(
        pd.DataFrame(
            {"observed": observed, "a": source},
            index=pd.date_range("2000-01-02", periods=2),
        ),
        1,
    )["agreement"].iloc[0]
    assert np.isnan(result.pearson_r)
    if np.nansum(observed) == 0:
        assert np.isnan(result.percentage_bias)


@pytest.mark.parametrize(
    "block",
    [
        {},
        False,
        {"stations_file": "x"},
        {"stations_path": "x", "precip_path": "y", "time_label": "local"},
    ],
)
def test_settings_rejections(block):
    with pytest.raises(ValueError):
        parse_rain_gauges(block)


def test_disabled():
    assert parse_rain_gauges(None) is None


@pytest.mark.parametrize("case", ["matched", "no_common", "all_excluded", "nodata"])
def test_actual_sampling_and_render(tmp_path, case):
    import geopandas as gpd
    import hydromt  # noqa: F401
    import xarray as xr
    from shapely.geometry import box

    from blueearth_cst.climate_analysis.evaluate_rain_gauges import evaluate
    from blueearth_cst.climate_analysis.plot_rain_gauges import render

    stations = tmp_path / "stations.csv"
    stations.write_text(
        "id,name,x,y\n001,Same,0.2,0.2\nNA,Same,1.2,1.2\n003,Outside,5,5\n004,NoGrid,3,3\n005,NoObs,0,0\n"
    )
    precip = tmp_path / "precip.csv"
    obs = pd.DataFrame(
        1.0,
        index=pd.date_range("1999-10-02", "2000-10-01"),
        columns=["001", "NA", "003", "004", "unknown"],
    )
    obs.to_csv(precip, index_label="time")
    if case == "no_common":
        obs.loc[:, :] = np.nan
        obs.to_csv(precip, index_label="time")
    if case == "all_excluded":
        stations.write_text("id,name,x,y\n003,Outside,5,5\n")
    region = tmp_path / "region.geojson"
    gpd.GeoDataFrame(geometry=[box(-1, -1, 4, 4)], crs=4326).to_file(region)
    field = xr.DataArray(
        np.tile([[3.0, 4.0], [1.0, 2.0]], (366, 1, 1)),
        dims=("time", "latitude", "longitude"),
        coords={"time": obs.index, "latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
        name="precip",
        attrs={"units": "mm"},
    )
    from blueearth_cst.climate_analysis.extract_historical_climate import (
        _normalize_precip_units,
    )

    field.raster.set_crs(4326)
    if case == "nodata":
        field.loc[dict(latitude=0.0, longitude=0.0)] = np.nan
    source = tmp_path / "source.nc"
    produced = _normalize_precip_units(field.to_dataset(), "chirps")
    assert produced.precip.attrs["units"] == "mm d**-1"
    produced.to_netcdf(source)
    before = source.read_bytes()
    settings = {
        "stations_path": str(stations),
        "precip_path": str(precip),
        "time_label": "interval_end",
    }
    root = tmp_path / "rain_gauges"
    evaluate(
        settings,
        {"era5": str(source)},
        str(region),
        str(root),
        {"start": 1999, "end": 2000},
        {"start": 1999, "end": 1999},
        10,
    )
    table = pd.read_csv(
        root / "tables/agreement.csv",
        dtype={"id": str},
        keep_default_na=False,
        na_values={"bias_mm": [""]},
    ).set_index("id")
    if case == "matched":
        assert table.loc["001", "bias_mm"] == 0
        assert table.loc["NA", "bias_mm"] == 3
    elif case == "no_common":
        assert table.bias_mm.isna().all()
    elif case == "nodata":
        assert np.isnan(table.loc["001", "bias_mm"])
        assert table.loc["NA", "bias_mm"] == 3
    else:
        assert table.empty
    excluded = pd.read_csv(root / "tables/exclusions.csv")
    period_columns = [
        "expected_days",
        "paired_days",
        "coverage",
        "first_interval_end",
        "last_interval_end",
    ]
    expected_headers = {
        "coverage": ["id", "source", "valid_days", *period_columns],
        "agreement": [
            "id",
            "source",
            *period_columns,
            "bias_mm",
            "mae_mm",
            "rmse_mm",
            "percentage_bias",
            "pearson_r",
            "reason",
            "correlation_reason",
            "percentage_bias_reason",
        ],
        "monthly": [
            "id",
            "month",
            "calendar_month",
            "source",
            "expected_days",
            "paired_days",
            "total_mm",
            "reason",
            "climatology_n",
        ],
        "annual": [
            "id",
            "reporting_year",
            "source",
            "expected_days",
            "paired_days",
            "total_mm",
            "reason",
        ],
    }
    for name, header in expected_headers.items():
        assert list(pd.read_csv(root / f"tables/{name}.csv").columns) == header
    expected_exclusions = (
        {
            "unknown_station_id",
            "no_observation_column",
            "outside_basin",
            "outside_source_grid",
        }
        if case != "all_excluded"
        else {"outside_basin", "unknown_station_id"}
    )
    assert expected_exclusions <= set(excluded.reason)
    if case == "nodata":
        assert "sampled_cell_nodata_in_period" in set(excluded.reason)
    assert source.read_bytes() == before
    import json

    inventory = json.loads((root / "_engine/evaluation.json").read_text())["outputs"]
    assert "_engine/evaluation.json" in inventory
    figure_count = 4 if case == "all_excluded" else 20
    assert len([path for path in inventory if path.endswith(".png")]) == figure_count
    render(root)
    assert len(list((root / "plots").glob("*.png"))) == figure_count
    # Obsolete files must never enter the declared inventory on reevaluation.
    (root / "plots/obsolete_station.png").write_bytes(b"obsolete")
    evaluate(
        settings,
        {"era5": str(source)},
        str(region),
        str(root),
        {"start": 1999, "end": 2000},
        {"start": 1999, "end": 1999},
        10,
    )
    assert (
        json.loads((root / "_engine/evaluation.json").read_text())["outputs"]
        == inventory
    )
    print("RAIN_GAUGE_RENDER_ROOT", root)
