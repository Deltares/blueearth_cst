# -*- coding: utf-8 -*-
"""The WF2 figure inputs — `blueearth_cst/projections/plot_proj_timeseries.py`.

The subject here is WHERE each number comes from, which is the defect this
layer exists to close. The shipped figures recomputed the monthly change from
the series with two departures from the authoritative table -- a historical
ANNUAL reference instead of the corresponding calendar month, and the full
2015-2100 future series instead of the horizon -- so the picture and the table
described different quantities under one name.

The fix is structural rather than arithmetic: the monthly figure now READS
`{clim_project}_change_factors_monthly.csv`. So the load-bearing test is not
"does the formula agree" but "is there a formula at all" -- see
`test_values_come_through_untouched`, which passes a table no recomputation
could reproduce.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import plot_proj_timeseries as ppt


def _monthly_table(rows):
    """A change-factor table with the columns the reader actually keys on."""
    return pd.DataFrame(rows)


def _row(
    model="GFDL-ESM4",
    scenario="ssp245",
    member="r1i1p1f1",
    horizon="far",
    month=1,
    variable="precip",
    statistic="mean",
    relative_value=-6.436,
    status="ok",
):
    return {
        "model": model,
        "scenario": scenario,
        "member": member,
        "horizon": horizon,
        "month": month,
        "variable": variable,
        "statistic": statistic,
        "relative_value": relative_value,
        "status": status,
    }


class TestNormaliseModel:
    def test_drops_the_institute_prefix(self):
        """`scalar/*.nc` says NOAA-GFDL/GFDL-ESM4; the tables say GFDL-ESM4."""
        assert ppt.normalise_model("NOAA-GFDL/GFDL-ESM4") == "GFDL-ESM4"

    def test_leaves_a_bare_model_name_alone(self):
        assert ppt.normalise_model("INM-CM4-8") == "INM-CM4-8"


class TestParseWindow:
    def test_reads_the_inclusive_years_off_the_artifact(self):
        assert ppt.parse_window("2000-01-01 / 2014-12-01") == (2000, 2014)

    def test_rejects_a_malformed_window(self):
        with pytest.raises(ValueError, match="malformed window"):
            ppt.parse_window("2000-2014")


class TestMonthlyChangeFromTable:
    def test_values_come_through_untouched(self):
        """The falsifier for a re-introduced recomputation.

        These numbers are not the change of anything -- no series and no
        arithmetic produces 999.0 and -42.0. If they arrive at the figure, the
        producer is reading the table. If someone later replaces the read with
        a formula, this is the test that fails.
        """
        table = _monthly_table(
            [
                _row(variable="precip", relative_value=999.0),
                _row(variable="temp", relative_value=-42.0),
            ]
        )
        out = ppt.monthly_change_from_table(table, horizon="far")
        assert out.loc[0, "precip_change"] == 999.0
        assert out.loc[0, "temp_change"] == -42.0

    def test_selects_one_statistic_so_traces_are_not_multiplied(self):
        """The table carries mean, median and std; an unfiltered read triples
        every trace and still looks like a plausible figure."""
        table = _monthly_table(
            [
                _row(statistic="mean", relative_value=1.0),
                _row(statistic="median", relative_value=2.0),
                _row(statistic="std", relative_value=3.0),
            ]
        )
        out = ppt.monthly_change_from_table(table, horizon="far", statistic="mean")
        assert len(out) == 1
        assert out.loc[0, "precip_change"] == 1.0

    def test_median_is_reachable_but_not_the_default(self):
        table = _monthly_table(
            [
                _row(statistic="mean", relative_value=1.0),
                _row(statistic="median", relative_value=2.0),
            ]
        )
        out = ppt.monthly_change_from_table(table, horizon="far", statistic="median")
        assert out.loc[0, "precip_change"] == 2.0

    def test_drops_flagged_rows_rather_than_drawing_them(self):
        """A flagged month is one whose near-zero reference made the ratio
        meaningless. The table declines to publish it; so does the figure."""
        table = _monthly_table(
            [
                _row(month=1, relative_value=5.0, status="ok"),
                _row(month=2, relative_value=1e6, status="reference_below_minimum"),
            ]
        )
        out = ppt.monthly_change_from_table(table, horizon="far")
        assert out["month"].tolist() == [1]

    def test_keeps_the_horizons_apart(self):
        table = _monthly_table(
            [
                _row(horizon="near", relative_value=1.0),
                _row(horizon="far", relative_value=9.0),
            ]
        )
        assert (
            ppt.monthly_change_from_table(table, horizon="near").loc[0, "precip_change"]
            == 1.0
        )

    def test_raises_on_a_horizon_the_table_does_not_carry(self):
        """Louder than an empty figure, which reads as "no change" rather than
        as "the table and the config disagree about the horizons"."""
        table = _monthly_table([_row(horizon="far")])
        with pytest.raises(ValueError, match="no 'mean' rows for horizon 'near'"):
            ppt.monthly_change_from_table(table, horizon="near")

    def test_one_row_per_combination_and_month(self):
        table = _monthly_table(
            [
                _row(month=m, variable=v, relative_value=float(m))
                for m in (1, 2)
                for v in ("precip", "temp")
            ]
        )
        out = ppt.monthly_change_from_table(table, horizon="far")
        assert len(out) == 2
        assert set(out.columns) >= {
            "model",
            "scenario",
            "member",
            "month",
            "precip_change",
            "temp_change",
        }


def _series(years, scenarios=("historical",), precip=1.0, temp=10.0, model="M"):
    rows = []
    for scenario in scenarios:
        for year in years:
            for month in range(1, 13):
                rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "member": "r1i1p1f1",
                        "year": year,
                        "month": month,
                        # A seasonal cycle, so a per-month reference and an
                        # annual one are not the same number.
                        "precip": precip * (1.0 + 0.5 * np.cos(month / 12 * 2 * np.pi)),
                        "temp": temp + month,
                    }
                )
    return pd.DataFrame(rows)


class TestAnnualSeries:
    def test_anomaly_is_measured_against_the_reference_window_only(self):
        """Not against the whole historical run, which is what the shipped
        figures used (`gcm_pr_annmn.mean()` over every historical year)."""
        inside = _series([2000, 2001], precip=1.0)
        outside = _series([1980], precip=100.0)
        series = pd.concat([outside, inside], ignore_index=True)

        out = ppt.annual_series(series, (2000, 2001))
        in_window = out[out["year"].isin([2000, 2001])]
        # Flat inside the window, so every in-window anomaly is zero. Had the
        # 1980 value entered the reference, none of them would be.
        assert np.allclose(in_window["precip_anomaly"], 0.0)

    def test_each_model_is_differenced_against_its_own_historical_run(self):
        """A cross-model reference would offset a future trace from the
        historical one it continues."""
        a = _series([2000], model="A", precip=1.0)
        b = _series([2000], model="B", precip=50.0)
        out = ppt.annual_series(pd.concat([a, b], ignore_index=True), (2000, 2000))
        assert np.allclose(out["precip_anomaly"], 0.0)
        assert np.allclose(out["temp_anomaly"], 0.0)

    def test_raises_when_the_window_contains_no_historical_years(self):
        series = _series([2000])
        with pytest.raises(ValueError, match="no historical years"):
            ppt.annual_series(series, (2050, 2060))

    def test_carries_both_absolute_and_anomaly_columns(self):
        """The overview draws a) absolute and b) anomaly from one frame."""
        out = ppt.annual_series(_series([2000, 2001]), (2000, 2001))
        assert {"precip", "temp", "precip_anomaly", "temp_anomaly"} <= set(out.columns)


class TestLoadScalarSeries:
    def _write(self, path, model, scenario, member, years):
        times = pd.date_range(f"{years[0]}-01-01", periods=12 * len(years), freq="MS")
        ds = xr.Dataset(
            {
                "precip": ("time", np.arange(len(times), dtype="float32")),
                "temp": ("time", np.arange(len(times), dtype="float32") + 100.0),
            },
            coords={
                "time": times,
                "model": model,
                "scenario": scenario,
                "member": member,
            },
        )
        ds.to_netcdf(path)

    def test_reads_year_and_month_and_the_scalar_coords(self, tmp_path):
        path = tmp_path / "one.nc"
        self._write(path, "NOAA-GFDL/GFDL-ESM4", "historical", "r1i1p1f1", [2000])
        out = ppt.load_scalar_series([path])
        assert len(out) == 12
        assert out["model"].unique().tolist() == ["GFDL-ESM4"]
        assert out["month"].tolist() == list(range(1, 13))
        assert out["year"].unique().tolist() == [2000]

    def test_concatenates_every_file_into_one_frame(self, tmp_path):
        self._write(tmp_path / "a.nc", "A", "historical", "r1i1p1f1", [2000])
        self._write(tmp_path / "b.nc", "A", "ssp245", "r1i1p1f1", [2000])
        out = ppt.load_scalar_series([tmp_path / "a.nc", tmp_path / "b.nc"])
        assert sorted(out["scenario"].unique()) == ["historical", "ssp245"]
        assert len(out) == 24

    def test_raises_rather_than_returning_an_empty_frame(self):
        """An empty frame would draw a blank figure and exit 0."""
        with pytest.raises(ValueError, match="no scalar series"):
            ppt.load_scalar_series([])
