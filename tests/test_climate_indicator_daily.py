"""Daily input contract independent of workflow acquisition and spatial layout."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.indicators.climate.daily import daily_array


def test_gap_is_preserved_as_unknown_and_grid_coordinates_survive():
    source = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("time", "location"),
        coords={
            "time": pd.to_datetime(["2000-02-28", "2000-03-01"]),
            "location": ["a", "b"],
        },
        attrs={"units": "mm/day"},
    )
    result = daily_array(source, units="mm/day")
    np.testing.assert_equal(result.values, [[1, 2], [np.nan, np.nan], [3, 4]])
    assert list(result.location.values) == ["a", "b"]
    result.values[0, 0] = 99
    assert source.values[0, 0] == 1


@pytest.mark.parametrize(
    "dates",
    [
        ["2001-01-02", "2001-01-01"],
        ["2001-01-01", "2001-01-01"],
        ["2001-01-01T12:00", "2001-01-02T12:00"],
    ],
)
def test_non_daily_order_is_rejected(dates):
    data = xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": pd.to_datetime(dates)},
        attrs={"units": "degC"},
    )
    with pytest.raises(ValueError, match="daily dates"):
        daily_array(data, units="degC")


def test_units_are_not_guessed_and_infinity_is_not_missing():
    data = xr.DataArray(
        [np.inf],
        dims="time",
        coords={"time": pd.date_range("2001-01-01", periods=1)},
        attrs={"units": "K"},
    )
    with pytest.raises(ValueError, match="normalized units"):
        daily_array(data, units="degC")
    with pytest.raises(ValueError, match="Infinite"):
        daily_array(data, units="K")
