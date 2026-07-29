"""Tests for get_stats_climate_proj.get_stats_clim_projections (workflow-2).

The function resamples daily/sub-monthly fields to monthly stats:
- ``precip``  -> monthly SUM  (conserves monthly water volume).
- every other var (``temp``) -> monthly MEAN.

It also auto-detects the x/y dims from XDIMS/YDIMS and averages over the grid
to produce the scalar (``mean_stats_time``) series. Synthetic constant grids
make every expected value self-evidently correct.
"""
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import xarray as xr

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

from blueearth_cst.projections.get_stats_climate_proj import get_stats_clim_projections  # noqa: E402


def _make_daily_grid(precip_day, temp_day, periods, xname="lon", yname="lat"):
    """A 2x2 constant grid of `periods` daily steps starting 2001-01-01.

    2001 is NOT a leap year -> Jan has 31 days, Feb has 28.
    """
    times = pd.date_range("2001-01-01", periods=periods, freq="D")
    shape = (periods, 2, 2)
    ds = xr.Dataset(
        {
            "precip": (("time", yname, xname), np.full(shape, precip_day, float)),
            "temp": (("time", yname, xname), np.full(shape, temp_day, float)),
        },
        coords={"time": times, yname: [0, 1], xname: [0, 1]},
    )
    return ds


def test_stats_precip_sum_temp_mean_monthly():
    """precip -> monthly SUM; temp -> monthly MEAN, on constant grids.

    31 days in Jan 2001 -> precip sum 31*2 = 62; Feb (28) -> 28*2 = 56.
    temp mean is 5.0 in every month regardless of month length.
    31 + 28 = 59 days spans exactly Jan+Feb (no partial March month).
    """
    ds = _make_daily_grid(precip_day=2.0, temp_day=5.0, periods=59)  # Jan+Feb
    mean_stats, mean_stats_time = get_stats_clim_projections(
        ds, "cmip6", "MODEL", "ssp245", "r1i1p1f1", save_grids=False
    )

    precip = mean_stats_time["precip"].values.ravel()
    temp = mean_stats_time["temp"].values.ravel()
    # two months retained (Jan, Feb)
    assert precip.tolist() == [62.0, 56.0]
    assert temp.tolist() == [5.0, 5.0]


def test_stats_scalar_carries_traceability_coords():
    """The scalar output expands the project/model/scenario/member coords."""
    ds = _make_daily_grid(precip_day=1.0, temp_day=3.0, periods=31)  # Jan only
    _, mean_stats_time = get_stats_clim_projections(
        ds, "cmip6", "GFDL", "ssp585", "r1i1p1f1", save_grids=False
    )
    assert str(mean_stats_time["clim_project"].values[0]) == "cmip6"
    assert str(mean_stats_time["model"].values[0]) == "GFDL"
    assert str(mean_stats_time["scenario"].values[0]) == "ssp585"
    assert str(mean_stats_time["member"].values[0]) == "r1i1p1f1"


def test_stats_mean_stats_empty_when_no_grids():
    """save_grids=False -> gridded stats dataset is empty."""
    ds = _make_daily_grid(precip_day=1.0, temp_day=3.0, periods=31)
    mean_stats, _ = get_stats_clim_projections(
        ds, "cmip6", "M", "ssp245", "r1i1p1f1", save_grids=False
    )
    assert len(mean_stats) == 0


def test_stats_dim_detection_alternate_names():
    """XDIMS/YDIMS detection: x/y and longitude/latitude both resolve.

    A constant grid means the spatial mean equals the constant, so a correct
    x/y detection reproduces the same scalar regardless of the coord names.
    """
    for xname, yname in [("x", "y"), ("longitude", "latitude"), ("lon", "lat")]:
        ds = _make_daily_grid(
            precip_day=2.0, temp_day=5.0, periods=31, xname=xname, yname=yname
        )
        _, mean_stats_time = get_stats_clim_projections(
            ds, "cmip6", "M", "ssp245", "r1i1p1f1", save_grids=False
        )
        assert mean_stats_time["precip"].values.ravel().tolist() == [62.0], xname
        assert mean_stats_time["temp"].values.ravel().tolist() == [5.0], yname


def test_stats_save_grids_true_returns_monthly_climatology():
    """save_grids=True -> gridded monthly climatology is populated."""
    ds = _make_daily_grid(precip_day=2.0, temp_day=5.0, periods=31)
    mean_stats, _ = get_stats_clim_projections(
        ds, "cmip6", "M", "ssp245", "r1i1p1f1", save_grids=True
    )
    assert len(mean_stats) > 0
    # constant field -> gridded monthly-mean precip total for Jan = 62 everywhere
    assert float(np.unique(mean_stats["precip"].values)[0]) == 62.0
