"""Tests for get_change_climate_proj.py (workflow-2 change helpers).

Realizes the R4 §7 audit-evidence matrix for ``get_change_annual_clim_proj``
(rows U, C1-C3, H, V, P) plus unit tests for ``_to_str_tuple`` and
the annual and monthly change paths.

Design contract: `dev/milestones/r04/climate-projections-design.md` §7.

Analytically-trivial synthetic inputs (constant fields) make every expected
value self-evident: a constant precip 10 -> 12 gives a multiplicative change
of (12-10)/10*100 = 20.0; a constant temp 5 -> 7 gives an additive change of
7-5 = 2.0, *regardless* of windowing, calendar, or hydrological-year start.

CALENDAR DEFECT (rows C1/C2): against today's code the cftime rows do not
merely mis-window — they RAISE ``TypeError``, because
``get_change_annual_clim_proj`` builds its slice bounds with
``pd.to_datetime(...)`` (pandas Timestamps) and applies them to a cftime index
(``.sel(time=slice(...))``), which xarray/pandas refuses to compare across
calendars. Every CMIP6 source is cftime-backed (see the ``cmip6
cftime.Datetime360Day`` note in get_stats_climate_proj.py), so this is a real
correctness defect on the production path, not a synthetic artefact. The rows
are wired strict-xfail and routed to a split task; the norm they assert is the
correct (calendar-invariant) 20.0 / 2.0 answer.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections.get_change_climate_proj import (  # noqa: E402
    _to_str_tuple,
    get_change_annual_clim_proj,
)


# --------------------------------------------------------------------------- #
# Synthetic-dataset builders
# --------------------------------------------------------------------------- #
def _monthly_time(years, calendar=None):
    """A monthly ('MS') time index spanning `len(years)` full calendar years."""
    start = f"{years[0]}-01-01"
    n = len(years) * 12
    if calendar is None:
        return pd.date_range(start=start, periods=n, freq="MS")
    return xr.date_range(
        start=start, periods=n, freq="MS", use_cftime=True, calendar=calendar
    )


def _make_time_ds(
    precip,
    temp,
    years,
    calendar=None,
    scenario="historical",
    members=("r1i1p1f1",),
    with_temp=True,
):
    """Constant-field hist/fut dataset mirroring stats_time-*.nc structure.

    Dims: time (monthly), scenario, member. Vars: precip (+ temp).
    """
    times = _monthly_time(years, calendar=calendar)
    members = list(members)
    shape = (len(times), 1, len(members))
    data_vars = {
        "precip": (("time", "scenario", "member"), np.full(shape, precip, float)),
    }
    if with_temp:
        data_vars["temp"] = (
            ("time", "scenario", "member"),
            np.full(shape, temp, float),
        )
    return xr.Dataset(
        data_vars,
        coords={"time": times, "scenario": [scenario], "member": members},
    )


# --------------------------------------------------------------------------- #
# Row U (PASS): core change formula on trivial constant fields
# --------------------------------------------------------------------------- #
def test_row_U_change_precip_multiplicative_temp_additive():
    """Row U: precip multiplicative (c-h)/h*100; temp additive c-h.

    hist precip==10, fut precip==12 -> (12-10)/10*100 = 20.0
    hist temp==5,    fut temp==7    -> 7-5           = 2.0
    Constant fields -> the annual stat over any windowing is exactly the
    constant, so the change is calendar/window invariant.
    """
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992])
    fut = _make_time_ds(12.0, 7.0, [2050, 2051, 2052])
    res = get_change_annual_clim_proj(hist, fut, stats=["mean"])

    assert float(res["precip"].values.ravel()[0]) == 20.0
    assert float(res["temp"].values.ravel()[0]) == 2.0
    # both variables survive when hist/fut share {precip, temp}. S8-04 added the
    # `__level` companions -- COLUMNS of their variable, never variables of their
    # own, which is exactly what the base-variable filter asserts.
    base = {v for v in res.data_vars if "__" not in str(v)}
    assert base == {"precip", "temp"}
    # and the level is the FUTURE level, in the variable's own units
    assert float(res["temp__level"].values.ravel()[0]) == 7.0


# --------------------------------------------------------------------------- #
# Rows C1-C3: calendar classes.
#   C3 (datetime64, proleptic Gregorian) PASSES today.
#   C1 (360_day) and C2 (noleap) RAISE TypeError today -> genuine calendar
#   defect -> strict-xfail, split to t260720c.
# --------------------------------------------------------------------------- #
def test_row_C3_gregorian_datetime64_change_invariant():
    """Row C3 (PASS): proleptic-Gregorian datetime64 index.

    Constant fields -> the calendar cannot change the answer: 20.0 / 2.0.
    Three full calendar years -> three hydrological years retained (Jan start).
    """
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992], calendar=None)
    fut = _make_time_ds(12.0, 7.0, [2050, 2051, 2052], calendar=None)
    res = get_change_annual_clim_proj(hist, fut, stats=["mean"])
    assert float(res["precip"].values.ravel()[0]) == 20.0
    assert float(res["temp"].values.ravel()[0]) == 2.0


def test_row_C1_cftime_360day_change_invariant():
    """Row C1 (t260720c, fixed): cftime.Datetime360Day (CMIP6-native) index.

    NORM: constant fields yield the calendar-invariant 20.0 / 2.0. The cftime
    index is converted to a DatetimeIndex before slicing, so no TypeError.
    """
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992], calendar="360_day")
    fut = _make_time_ds(12.0, 7.0, [2050, 2051, 2052], calendar="360_day")
    res = get_change_annual_clim_proj(hist, fut, stats=["mean"])
    assert float(res["precip"].values.ravel()[0]) == 20.0
    assert float(res["temp"].values.ravel()[0]) == 2.0


def test_row_C2_cftime_noleap_change_invariant():
    """Row C2 (t260720c, fixed): cftime.DatetimeNoLeap index.

    NORM: constant fields yield the calendar-invariant 20.0 / 2.0. The cftime
    index is converted to a DatetimeIndex before slicing, so no TypeError.
    """
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992], calendar="noleap")
    fut = _make_time_ds(12.0, 7.0, [2050, 2051, 2052], calendar="noleap")
    res = get_change_annual_clim_proj(hist, fut, stats=["mean"])
    assert float(res["precip"].values.ravel()[0]) == 20.0
    assert float(res["temp"].values.ravel()[0]) == 2.0


# --------------------------------------------------------------------------- #
# Row H (PASS): non-January hydrological-year boundary
# --------------------------------------------------------------------------- #
def test_row_H_october_hydro_year_window_and_change():
    """Row H (PASS): start_month_hyd_year='Oct'.

    Four calendar years of monthly data -> three full Oct->Sep hydrological
    years are retained (1990-10..1991-09, 1991-10..1992-09, 1992-10..1993-09).
    Constant fields -> change is still 20.0 / 2.0.
    """
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992, 1993])
    fut = _make_time_ds(12.0, 7.0, [2050, 2051, 2052, 2053])
    res = get_change_annual_clim_proj(
        hist, fut, stats=["mean"], start_month_hyd_year="Oct"
    )
    assert float(res["precip"].values.ravel()[0]) == 20.0
    assert float(res["temp"].values.ravel()[0]) == 2.0


def test_row_H_january_control_matches_row_U():
    """Row H control: explicit 'Jan' reproduces the default (Row U) answer."""
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992, 1993])
    fut = _make_time_ds(12.0, 7.0, [2050, 2051, 2052, 2053])
    res = get_change_annual_clim_proj(
        hist, fut, stats=["mean"], start_month_hyd_year="Jan"
    )
    assert float(res["precip"].values.ravel()[0]) == 20.0
    assert float(res["temp"].values.ravel()[0]) == 2.0


# --------------------------------------------------------------------------- #
# Row V (t260720d, fixed): asymmetric variables (hist has temp, clim does not)
#   must fail loud with a ValueError naming the missing variable.
# --------------------------------------------------------------------------- #
def test_row_V_asymmetric_variables_should_raise():
    """Row V (ext3-1): asymmetric variable sets raise ValueError naming the
    missing variable(s). Pre-t260720d intersection() dropped them silently."""
    hist = _make_time_ds(10.0, 5.0, [1990, 1991, 1992], with_temp=True)
    clim = _make_time_ds(12.0, 7.0, [2050, 2051, 2052], with_temp=False)
    with pytest.raises(ValueError, match=r"asymmetric.*variables.*temp"):
        get_change_annual_clim_proj(hist, clim, stats=["mean"])


# --------------------------------------------------------------------------- #
# Row P (t260720d, fixed): asymmetric members (hist has r1+r2, clim has r1)
#   must fail loud with a ValueError naming the unshared member.
# --------------------------------------------------------------------------- #
def test_row_P_asymmetric_members_should_raise():
    """Row P (ext3-1): an unshared member raises ValueError naming the dropped
    member. Pre-t260720d xarray's inner-join dropped it silently."""
    hist = _make_time_ds(
        10.0, 5.0, [1990, 1991, 1992], members=("r1i1p1f1", "r2i1p1f1")
    )
    clim = _make_time_ds(12.0, 7.0, [2050, 2051, 2052], members=("r1i1p1f1",))
    with pytest.raises(ValueError, match=r"asymmetric.*members.*r2i1p1f1"):
        get_change_annual_clim_proj(hist, clim, stats=["mean"])


# --------------------------------------------------------------------------- #
# Unit: _to_str_tuple (R01 list-vs-comma-string horizon parser)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        ([1980, 2010], ("1980", "2010")),  # R01 list form
        ("1980, 2010", ("1980", "2010")),  # legacy comma-separated string
        ([2030, 2060], ("2030", "2060")),
        (["1980", "2010"], ("1980", "2010")),  # already-str list
        ((1980, 2010), ("1980", "2010")),  # tuple input
    ],
)
def test_to_str_tuple_list_and_string(value, expected):
    assert _to_str_tuple(value) == expected


def test_to_str_tuple_empty_list():
    """Falsey/empty list -> empty tuple (no crash)."""
    assert _to_str_tuple([]) == ()


# --------------------------------------------------------------------------- #
# `test_get_change_clim_projections_grid_change` stood here. S8-08(c) removed
# the cellwise grid change it exercised, together with the gridded branch.
