"""Independent scalar and calendar checks for observed water balance."""

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.model.budyko import (
    budyko_settings,
    discharge_depth,
    matched_water_balance,
    read_daily_discharge,
)


@pytest.mark.parametrize("value", [None, False])
def test_disabled(value):
    assert budyko_settings(value, None) is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        {},
        "false",
        0,
        {"time_label": "interval_end"},
        {"discharge_statistic": "instantaneous", "time_label": "interval_end"},
        {"discharge_statistic": "daily_mean", "time_label": "local_day"},
        {"discharge_statistic": "daily_mean", "time_label": "interval_end", "extra": 1},
    ],
)
def test_invalid_settings(value):
    with pytest.raises(ValueError, match="discharge_statistic: daily_mean"):
        budyko_settings(value, "observations.csv")


def test_enabled_requires_observations():
    with pytest.raises(ValueError, match="observations.river discharge"):
        budyko_settings(
            {"discharge_statistic": "daily_mean", "time_label": "interval_end"}, None
        )


def test_start_end_totals_across_water_year_and_leap_day(tmp_path):
    starts = pd.date_range("1999-10-01", "2000-09-30")
    results = []
    for label, shift in [("interval_start", 0), ("interval_end", 1)]:
        path = tmp_path / f"{label}.csv"
        pd.DataFrame({"time": starts + pd.Timedelta(days=shift), "1010": 1.0}).to_csv(
            path, sep=";", index=False
        )
        q = read_daily_discharge(path, label)["1010"]
        frame = pd.DataFrame(
            {"precip": 2.0, "pet": 3.0, "runoff_mm": discharge_depth(q, 86400000)}
        )
        results.append(matched_water_balance(frame, "1999-10-01", "2000-10-01", 10))
    pd.testing.assert_frame_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    assert results[0][1]["precip_mm"] == 732
    assert results[0][1]["runoff_mm"] == 366


def test_start_end_equivalence_and_preservation(tmp_path):
    start = tmp_path / "start.csv"
    end = tmp_path / "end.csv"
    start.write_text("time;101\n2000-02-28;1\n2000-02-29;2\n2000-03-02;3\n")
    end.write_text(
        "time;101\n2000-02-29T01:00:00+01:00;1\n2000-03-01;2\n2000-03-03;3\n"
    )
    before = start.read_bytes(), end.read_bytes()
    pd.testing.assert_frame_equal(
        read_daily_discharge(start, "interval_start"),
        read_daily_discharge(end, "interval_end"),
    )
    assert before == (start.read_bytes(), end.read_bytes())


@pytest.mark.parametrize(
    "labels", ["2000-01-01;1\n2000-01-01;2", "2000-01-01T00:00:00+01:00;1", "bad;1"]
)
def test_invalid_observation_labels(tmp_path, labels):
    path = tmp_path / "observed.csv"
    path.write_text("time;101\n" + labels + "\n")
    with pytest.raises(ValueError):
        read_daily_discharge(path, "interval_end")


def test_discharge_depth():
    result = discharge_depth(pd.Series([1.0, 0.0, np.nan]), 86400000)
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.0
    assert np.isnan(result.iloc[2])


@pytest.mark.parametrize("area", [0, -1, np.inf, np.nan])
def test_invalid_area(area):
    with pytest.raises(ValueError, match="area"):
        discharge_depth(pd.Series([1.0]), area)


def test_totals_and_leap_water_year():
    index = pd.date_range("1999-10-02", "2001-10-01", freq="D")
    data = pd.DataFrame(0.0, index=index, columns=["precip", "pet", "runoff_mm"])
    data.loc["1999-10-02"] = [100, 200, 50]
    data.loc["2000-10-02"] = [1000, 500, 100]
    annual, summary = matched_water_balance(data, "1999-10-01", "2001-10-01", 10)
    assert annual.expected_days.tolist() == [366, 365]
    assert annual.eligible.tolist() == [True, True]
    assert summary["retained_years"] == [1999, 2000]
    assert summary["aridity_index"] == pytest.approx(700 / 1100)
    assert summary["runoff_coefficient"] == pytest.approx(150 / 1100)
    assert summary["runoff_coefficient"] != pytest.approx(0.3)


@pytest.mark.parametrize("column", ["precip", "pet", "runoff_mm"])
def test_missing_terminal_interval_excludes_year(column):
    index = pd.date_range("2000-01-02", "2001-01-01", freq="D")
    data = pd.DataFrame(1.0, index=index, columns=["precip", "pet", "runoff_mm"])
    data.loc[index[-1], column] = np.nan
    annual, summary = matched_water_balance(data, "2000-01-01", "2001-01-01", 1)
    assert annual.iloc[0].matched_days == 365
    assert annual.iloc[0].reason == "missing_days"
    assert summary["reason"] == "no_complete_years"


def test_partial_year_zero_precip_and_unusual_ratio():
    index = pd.date_range("2000-01-02", "2001-01-02", freq="D")
    data = pd.DataFrame(1.0, index=index, columns=["precip", "pet", "runoff_mm"])
    data["runoff_mm"] = 2.0
    annual, summary = matched_water_balance(data, "2000-01-01", "2001-01-02", 1)
    assert annual.reason.tolist() == ["", "partial_window"]
    assert summary["runoff_coefficient"] == 2.0
    data["precip"] = 0
    _, summary = matched_water_balance(data, "2000-01-01", "2001-01-01", 1)
    assert summary["reason"] == "zero_precipitation"
    assert np.isnan(summary["aridity_index"])


@pytest.mark.parametrize(
    "headers",
    [("1010", "1011"), ("01010", "1011"), (" 1010 ", "1011"), ("01010", " 01011 ")],
)
def test_real_upstream_masks_and_unequal_area_weighting(tmp_path, headers):
    import hydromt  # noqa: F401
    import xarray as xr

    from blueearth_cst.model.evaluate_budyko import evaluate

    coords = {"y": [1500.0, 500.0], "x": [500.0, 1500.0]}
    maps = xr.Dataset(
        {
            "flow_direction": (("y", "x"), np.array([[2, 4], [1, 0]], dtype="uint8")),
            "cell_area": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
        },
        coords=coords,
    )
    maps.raster.set_crs(3857)
    maps.flow_direction.raster.set_nodata(247)
    maps.to_netcdf(tmp_path / "hydro.nc")
    time = pd.date_range("2000-01-02", "2001-01-01")
    values = np.broadcast_to([[10.0, 20.0], [30.0, 40.0]], (366, 2, 2))
    forcing = xr.Dataset(
        {
            "precip": (("time", "y", "x"), values),
            "pet": (("time", "y", "x"), values * 2),
        },
        coords={"time": time, **coords},
    )
    forcing.raster.set_crs(3857)
    for variable in forcing:
        forcing[variable].attrs["unit"] = "mm"
    forcing.to_netcdf(tmp_path / "forcing.nc")
    pd.DataFrame(
        {
            "wflow_id": [1010, 1011],
            "location_role": ["control", "observation"],
            "station_name": ["outlet", "extra"],
            "snapped_row": [1, 0],
            "snapped_col": [1, 0],
        }
    ).to_csv(tmp_path / "registry.csv", index=False)
    pd.DataFrame({"time": time, headers[0]: 1.0, headers[1]: 1.0}).to_csv(
        tmp_path / "q.csv", sep=";", index=False
    )
    original_csv = (tmp_path / "q.csv").read_bytes()
    (tmp_path / "recipe.yml").write_text("synthetic controlled forcing\n")
    root = tmp_path / "result"
    evaluate(
        str(tmp_path / "hydro.nc"),
        str(tmp_path / "registry.csv"),
        str(tmp_path / "forcing.nc"),
        str(tmp_path / "recipe.yml"),
        str(tmp_path / "q.csv"),
        str(root),
        {"discharge_statistic": "daily_mean", "time_label": "interval_end"},
        "2000-01-01",
        "2001-01-01",
        1,
    )
    stations = pd.read_csv(root / "tables/stations.csv")
    assert stations.wflow_id.tolist() == [1010, 1011]
    assert original_csv == (tmp_path / "q.csv").read_bytes()
    assert stations.cell_count.tolist() == [4, 1]
    assert stations.area_m2.tolist() == [10e6, 1e6]
    summary = pd.read_csv(root / "tables/summary.csv")
    assert summary.aridity_index.tolist() == pytest.approx([2, 2])
    assert summary.precip_mm.tolist() == pytest.approx([30 * 366, 10 * 366])
    assert summary.runoff_coefficient.tolist() == pytest.approx([8.64 / 30, 86.4 / 10])


def test_duplicate_observation_aliases_rejected(tmp_path):
    from blueearth_cst.model.observation_validation import (
        validate_observation_station_ids,
    )

    observations = tmp_path / "q.csv"
    observations.write_text("time;1010; 01010 \n2000-01-02;1;2\n")
    original = observations.read_bytes()
    registry = tmp_path / "registry.csv"
    registry.write_text("wflow_id,location_role\n1010,control\n")
    with pytest.raises(ValueError, match=r"duplicate=\[1010\]"):
        validate_observation_station_ids(observations, registry)
    with pytest.raises(ValueError, match="duplicate"):
        read_daily_discharge(observations, "interval_end")
    assert observations.read_bytes() == original
