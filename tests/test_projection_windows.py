"""Analysis-window placement, joins, seasons and OLS known answers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import get_change_climate_proj as change


def _series(start, end, calendar="standard", scenario="ssp245"):
    time = pd.date_range(start, end, freq="MS")
    values = np.where(time.month == 12, 1.0, np.where(time.month == 2, 3.0, 2.0))
    return xr.Dataset(
        {"temp": ("time", values), "precip": ("time", values)},
        coords={"time": time, "scenario": scenario, "member": "r1"},
        attrs={"cst_calendar": calendar},
    )


@pytest.mark.parametrize(
    "month,horizon,used",
    [
        ("Jan", (2041, 2060), (2041, 2060)),
        ("Jan", (2081, 2100), (2081, 2100)),
        ("Oct", (2081, 2100), (2081, 2099)),
        ("Jan", (2015, 2030), (2015, 2030)),
        ("Jan", (2050, 2050), (2050, 2050)),
    ],
)
def test_analysis_windows_keep_complete_configured_years(month, horizon, used):
    record = change.analysis_windows({"future": horizon}, month, (1985, 2014))["future"]
    assert record["used"] == used
    assert record["window_id"] == f"{used[0]}-{used[1]}"
    assert record["requested"] == used
    assert record["shifted"] is False
    assert record["joined"] is False
    assert record["available"] is True
    if month == "Jan":
        assert record["reference"] == (1985, 2014)
    else:
        assert record["reference"] == (1985, 2013)
        assert record["effective"]["reference"] == {
            "start": "1985-10-01",
            "end": "2014-09-30",
            "n_water_years": 29,
        }
        assert change.analysis_windows({"f": horizon}, "Oct", (1970, 2014))["f"][
            "reference"
        ] == (1970, 2013)
    long_reference = change.analysis_windows({}, "Jan", (1970, 2014))
    assert long_reference == {}
    assert change.analysis_windows({"f": horizon}, "Jan", (1970, 2014))["f"][
        "reference"
    ] == (1970, 2014)
    assert change.analysis_windows({"f": horizon}, "Jan", (2000, 2014))["f"][
        "reference"
    ] == (2000, 2014)


@pytest.mark.parametrize(
    "future,reference,month,reason",
    [
        ((2041, 2060), (2014, 2014), "Oct", "no complete"),
        ((2050, 2050), (1985, 2014), "Oct", "no complete"),
        ((2010, 2030), (1985, 2014), "Jan", "outside acquisition"),
        ((2090, 2110), (1985, 2014), "Jan", "outside acquisition"),
        ((2041, 2060), (1985, 2015), "Jan", "outside acquisition"),
        ((2060, 2041), (1985, 2014), "Jan", "start must not exceed"),
    ],
)
def test_analysis_windows_refuse_invalid_bounds(future, reference, month, reason):
    with pytest.raises(ValueError, match=reason):
        change.analysis_windows({"future": future}, month, reference)


def test_twenty_year_window_excludes_outside_values_and_scales_season_threshold():
    record = change.analysis_windows({"future": (2041, 2060)})["future"]
    historical = _series("1985-01-01", "2014-12-01")
    future = _series("2036-01-01", "2065-12-01")
    historical["temp"][:] = 10
    future["temp"][:] = 999
    future["temp"].loc[{"time": slice("2041", "2060")}] = 12
    reference = change.analysis_slice(historical, record["reference"])
    selected = change.analysis_slice(future, record["used"])
    assert selected.sizes["time"] == 240
    assert record["effective"]["future"] == {
        "start": "2041-01-01",
        "end": "2060-12-31",
        "n_water_years": 20,
    }
    annual = change.get_change_annual_clim_proj(
        reference.expand_dims("scenario"),
        selected.expand_dims("scenario"),
        stats=["mean"],
    )
    assert annual.temp.item() == pytest.approx(2)
    seasons = change.get_change_seasonal_clim_proj(reference, selected)
    assert {row["status"] for row in seasons} == {"ok"}
    assert {
        row["absolute"]
        for row in seasons
        if row["variable"] == "temp" and row["statistic"] == "mean"
    } == {2}
    rejected = change.get_change_seasonal_clim_proj(
        reference, selected, min_season_years=27
    )
    assert {row["status"] for row in rejected} == {"insufficient_observations"}


def test_analysis_slice_joins_same_member_and_refuses_unavailable_span():
    historical = _series(
        "2000-01-01", "2014-12-01", scenario="historical"
    ).assign_attrs(cst_series_digest="historical-digest")
    future = _series("2015-01-01", "2040-12-01").assign_attrs(
        cst_series_digest="future-digest"
    )
    joined = change.analysis_slice(future, (2008, 2037), historical=historical)
    assert joined.sizes["time"] == 360
    assert joined.scenario.item() == "ssp245"
    assert joined.member.item() == "r1"
    assert joined.attrs == future.attrs
    with pytest.raises(change.AnalysisWindowUnavailable, match="historical"):
        change.analysis_slice(future, (2008, 2037))
    with pytest.raises(change.AnalysisWindowUnavailable, match="cover"):
        change.analysis_slice(future, (2030, 2059))
    with pytest.raises(ValueError, match="different calendars"):
        change.analysis_slice(
            future,
            (2008, 2037),
            historical=historical.assign_attrs(cst_calendar="360_day"),
        )
    with pytest.raises(ValueError, match="member") as corrupt:
        change.analysis_slice(
            future, (2008, 2037), historical=historical.assign_coords(member="r2")
        )
    assert not isinstance(corrupt.value, change.AnalysisWindowUnavailable)
    with pytest.raises(ValueError, match="asymmetric"):
        change.get_change_seasonal_clim_proj(historical.drop_vars("temp"), future)


@pytest.mark.parametrize("calendar", ["standard", "noleap", "360_day"])
@pytest.mark.parametrize("month", ["Jan", "Oct"])
def test_unequal_season_windows_keep_calendar_rates_and_boundary_ownership(
    calendar, month
):
    record = change.analysis_windows({"f": (2041, 2060)}, month)["f"]
    historical = _series("1985-01-01", "2014-12-01", calendar)
    future = _series("2041-01-01", "2060-12-01", calendar)
    historical["temp"] = xr.full_like(historical.temp, 10)
    historical["precip"] = xr.full_like(historical.precip, 2)
    future["temp"] = xr.full_like(future.temp, 12)
    future["precip"] = xr.full_like(future.precip, 3)
    rows = change.get_change_seasonal_clim_proj(
        change.analysis_slice(historical, record["reference"], month),
        change.analysis_slice(future, record["used"], month),
        stats=["mean"],
    )
    assert {row["status"] for row in rows} == {"ok"}
    for row in rows:
        assert row["absolute"] == pytest.approx(2 if row["variable"] == "temp" else 1)
        if row["variable"] == "precip":
            assert row["relative"] == pytest.approx(50)
        expected = (29, 19) if month == "Oct" or row["season"] == "DJF" else (30, 20)
        if month == "Oct" and row["season"] == "SON":
            expected = (28, 18)
        assert (row["n_reference"], row["n_future"]) == expected


@pytest.mark.parametrize(
    "side,missing,eligible",
    [
        ("reference", 3, True),
        ("reference", 4, False),
        ("future", 2, True),
        ("future", 3, False),
    ],
)
def test_missing_seasons_do_not_shrink_required_window_coverage(
    side, missing, eligible
):
    historical = _series("1985-01-01", "2014-12-01")
    future = _series("2041-01-01", "2060-12-01")
    target = historical if side == "reference" else future
    first = 1985 if side == "reference" else 2041
    # Remove internal months entirely, rather than merely using NaN values.
    target = target.drop_sel(time=[f"{first + i}-04-01" for i in range(missing)])
    if side == "reference":
        historical = target
    else:
        future = target
    rows = change.get_change_seasonal_clim_proj(historical, future, stats=["mean"])
    mam = [row for row in rows if row["season"] == "MAM"]
    assert {(row["status"] == "ok") for row in mam} == {eligible}
    assert all(np.isfinite(row["change"]) == eligible for row in mam)


def test_default_season_eligibility_requires_at_least_three_observations():
    short = _series("2000-01-01", "2001-12-01")
    rows = change.get_change_seasonal_clim_proj(short, short)
    assert {row["status"] for row in rows} == {"insufficient_observations"}


@pytest.mark.parametrize(
    "month,window,calendar,djf_years,son_years,expected",
    [
        ("Jan", (2001, 2003), "standard", [2002, 2003], [2001, 2002, 2003], 177 / 90),
        ("Oct", (2001, 2003), "standard", [2002, 2003, 2004], [2002, 2003], 177 / 90),
        ("Jan", (2001, 2003), "360_day", [2002, 2003], [2001, 2002, 2003], 2.0),
        ("Jan", (2001, 2003), "noleap", [2002, 2003], [2001, 2002, 2003], 177 / 90),
    ],
)
def test_seasons_own_only_complete_in_window_months(
    month, window, calendar, djf_years, son_years, expected
):
    source = _series("2000-01-01", "2005-12-01", calendar)
    sliced = change.analysis_slice(source, window, month)
    rows = change.get_change_seasonal_clim_proj(sliced, sliced, min_season_years=1)
    djf = next(
        row
        for row in rows
        if row["variable"] == "temp"
        and row["season"] == "DJF"
        and row["statistic"] == "mean"
    )
    son = next(
        row
        for row in rows
        if row["variable"] == "temp"
        and row["season"] == "SON"
        and row["statistic"] == "mean"
    )
    assert djf["reference_years"].tolist() == djf_years
    assert son["reference_years"].tolist() == son_years
    assert djf["reference_values"][0] == pytest.approx(expected)
    if month == "Oct" and calendar == "standard":
        assert djf["reference_values"][-1] == pytest.approx(180 / 91)
    assert djf["n_reference"] == len(djf_years)
    assert djf["absolute"] == 0
    # Equal rates in nonleap vs leap DJF: rate change is zero, while the lab's
    # integral percentage changes by 100/90 %. Owner chose the rate convention.
    reference = _series("2002-12-01", "2003-02-01", calendar)
    future = _series("2003-12-01", "2004-02-01", calendar)
    reference["precip"][:] = 2
    future["precip"][:] = 2
    rate_rows = change.get_change_seasonal_clim_proj(
        reference, future, min_season_years=1
    )
    rate = next(
        row
        for row in rate_rows
        if row["variable"] == "precip"
        and row["season"] == "DJF"
        and row["statistic"] == "mean"
    )
    assert rate["relative"] == 0
    if calendar == "standard":
        assert (
            rate["future_integrals"][0] / rate["reference_integrals"][0] - 1
        ) * 100 == pytest.approx(100 / 90)
    # A missing in-window month drops its whole instance, not just that month.
    missing = sliced.drop_sel(time="2002-01-01")
    dropped = change.get_change_seasonal_clim_proj(missing, sliced, min_season_years=1)
    row = next(
        row
        for row in dropped
        if row["variable"] == "temp"
        and row["season"] == "DJF"
        and row["statistic"] == "mean"
    )
    assert row["reference_years"].tolist() == [
        year for year in djf_years if year != 2002
    ]
    # Altering months beyond the owned slice cannot affect any season value.
    source["temp"] = source.temp.where(source.time.isin(sliced.time), 100000)
    outside = change.get_change_seasonal_clim_proj(
        change.analysis_slice(source, window, month), sliced, min_season_years=1
    )
    assert (
        next(
            row
            for row in outside
            if row["variable"] == "temp"
            and row["season"] == "DJF"
            and row["statistic"] == "mean"
        )["reference"]
        == djf["reference"]
    )


def test_detrended_std_uses_retained_years_and_population_denominator():
    # Residuals are orthogonal to both a constant and these uneven year coordinates.
    years = np.array([2000, 2001, 2003, 2006])
    residual = np.array([2.0, -3.0, 1.0, 0.0])
    values = 10 + 4 * (years - 2000) + residual
    assert change.detrended_std(values, years) == pytest.approx(np.sqrt(14 / 4))
    assert not np.isclose(values.std(), np.sqrt(14 / 4))
    assert np.isnan(change.detrended_std([1, 2], [2000, 2001]))
    with pytest.raises(ValueError, match="same"):
        change.detrended_std([1, 2, 3], [2000, 2001])
