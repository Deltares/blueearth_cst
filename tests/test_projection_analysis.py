"""Known answers for analysis-only changes, eligibility and ensemble reductions."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import projection_analysis as analysis

IDENTITY = {"model": "TEST/M1", "scenario": "ssp245", "member": "r1"}


def _series(start, precip, temp, scenario, calendar="360_day"):
    time = pd.date_range(f"{start}-01-01", periods=360, freq="MS")
    values = {}
    for name, data, units in (("precip", precip, "mm/day"), ("temp", temp, "degC")):
        data = np.broadcast_to(np.asarray(data), (30,))
        values[name] = ("time", np.repeat(data, 12), {"units": units})
    return xr.Dataset(
        values, coords={"time": time}, attrs={"cst_calendar": calendar}
    ).expand_dims(
        model=["TEST/M1"], scenario=[scenario], member=["r1"], clim_project=["cmip6"]
    )


def _changes(hist, future, **options):
    return analysis.analysis_change_table(
        hist,
        future,
        identity=IDENTITY,
        window_id="2036-2065",
        reference_window=(1985, 2014),
        future_window=(2036, 2065),
        **options,
    )


def _row(table, variable, period="annual", statistic="mean", quantity="change"):
    return table.loc[
        (table.variable == variable)
        & (table.period == period)
        & (table.statistic == statistic)
        & (table.quantity == quantity)
    ].iloc[0]


def test_float32_analysis_reductions_match_promoted_source_values():
    years = np.arange(30)
    hist = _series(
        1985, 0.53 + years * 0.007, 10.1 + years * 0.013, "historical"
    ).astype("float32")
    future = _series(2036, 0.71 + years * 0.009, 12.2 + years * 0.017, "ssp245").astype(
        "float32"
    )
    actual = _changes(hist, future)
    expected = _changes(hist.astype("float64"), future.astype("float64"))
    np.testing.assert_allclose(
        actual.value, expected.value, rtol=0, atol=0, equal_nan=True
    )
    assert hist.temp.dtype == np.float32


@pytest.mark.parametrize(
    "missing_side,missing,eligible",
    [
        ("future", 0, True),
        ("future", 2, True),
        ("future", 3, False),
        ("reference", 4, False),
    ],
)
def test_seasonal_changes_and_variability_use_separate_owned_window_thresholds(
    missing_side, missing, eligible
):
    alternating = (-1.0) ** np.arange(30)
    hist = _series(1985, 2 + alternating / 10, 10 + alternating, "historical")
    future = _series(2041, 3 + alternating / 5, 12 + 2 * alternating, "ssp245")
    target = hist if missing_side == "reference" else future
    start = 1985 if missing_side == "reference" else 2041
    # Remove leading complete years while retaining original acquisition coverage.
    # The configured window, not the shortened finite data, still owns the threshold.
    for variable in ("temp", "precip"):
        target[variable] = target[variable].where(
            target.time.dt.year >= start + missing
        )
    options = dict(
        identity=IDENTITY,
        window_id="2041-2060",
        reference_window=(1985, 2014),
        future_window=(2041, 2060),
    )
    changes = analysis.analysis_change_table(hist, future, **options)
    mam = changes.loc[changes.period == "MAM"]
    assert set(mam.mean_change_eligible) == {eligible}
    assert set(mam.status) == {"ok" if eligible else "insufficient_observations"}
    ratios = analysis.member_ratios(changes)
    assert set(ratios.loc[ratios.period == "MAM", "variability_eligible"]) == {eligible}
    if missing == 0:
        assert set(changes.loc[changes.period == "DJF", "n_reference"]) == {29}
        assert set(changes.loc[changes.period == "DJF", "n_future"]) == {19}
        assert _row(changes, "temp", "MAM").value == pytest.approx(2)
        assert _row(
            changes, "precip", "MAM", quantity="change_pct"
        ).value == pytest.approx(50)
        strict = analysis.analysis_change_table(
            hist, future, min_season_years=27, **options
        )
        assert set(strict.loc[strict.period == "MAM", "status"]) == {
            "insufficient_observations"
        }


def test_mean_keyed_dry_rule_restores_wet_low_sd_and_keeps_dry_absolute():
    alternating = (-1.0) ** np.arange(30)
    hist = _series(1985, 0.5 + 0.05 * alternating, 10 + alternating, "historical")
    future = _series(2036, 0.6 + 0.1 * alternating, 12 + alternating, "ssp245")
    table = _changes(hist, future, min_reference={"precip": 0.1})
    std = _row(table, "precip", "Jan", "std", "change_pct")
    assert std.status == "ok"
    assert std.value == pytest.approx(100)
    assert std.mean_change_eligible
    dry = _changes(
        hist.assign(precip=hist.precip / 10), future, min_reference={"precip": 0.1}
    )
    absolute, relative = _row(dry, "precip"), _row(dry, "precip", quantity="change_pct")
    assert absolute.status == relative.status == "dry"
    assert absolute.value == pytest.approx(0.55)
    assert absolute.mean_change_eligible
    assert not relative.mean_change_eligible
    assert np.isnan(relative.value)
    models = analysis.model_changes(dry)
    assert _row(models, "precip").n_members == 1
    assert _row(models, "precip", quantity="change_pct").n_members == 0


def test_constant_reference_mean_changes_survive_but_ratios_do_not():
    hist = _series(1985, 2, 10, "historical")
    future = _series(2036, 3, 12, "ssp245")
    table = _changes(hist, future)
    temperature = _row(table, "temp")
    assert temperature.status == "zero_reference_variability"
    assert temperature.value == 2
    assert temperature.mean_change_eligible
    ratios = analysis.member_ratios(table)
    assert not ratios.variability_eligible.any()
    assert not np.isinf(ratios.ratio.astype(float)).any()
    models = analysis.model_changes(table)
    assert _row(models, "temp").value == 2
    agreement = analysis.sign_agreement(models)
    assert _row(agreement, "temp").k == 1
    # Gregorian integral variability includes duration even at constant rates.
    gregorian = _changes(
        hist.assign_attrs(cst_calendar="standard"),
        future.assign_attrs(cst_calendar="standard"),
    )
    cv = _row(
        analysis.member_ratios(gregorian),
        "precip",
        statistic="cv_detrended",
        quantity="ratio",
    )
    assert cv.reference_value > 0
    assert cv.variability_eligible
    # A pure trend has positive rawSD but zero residualSD: do not erase raw std%.
    trend = np.arange(30, dtype=float) + 1
    linear = _changes(
        _series(1985, trend, trend, "historical"),
        _series(2036, 2 * trend, 2 * trend, "ssp245"),
    )
    raw = _row(linear, "precip", statistic="std", quantity="change_pct")
    detrended = _row(
        analysis.member_ratios(linear),
        "temp",
        statistic="std_detrended",
        quantity="ratio",
    )
    assert raw.status == "ok"
    assert raw.value == pytest.approx(100)
    assert raw.mean_change_eligible
    assert detrended.status == "zero_reference_variability"
    assert not detrended.variability_eligible


def test_model_weights_family_votes_and_zero_future_exclusion():
    hist = _series(
        1985, 2 + (-1.0) ** np.arange(30), 10 + (-1.0) ** np.arange(30), "historical"
    )
    future = _series(2036, 3, 12, "ssp245")
    zero = analysis.member_ratios(_changes(hist, future))
    assert (zero.status == "zero_future_variability").all()
    assert (zero.ratio == 0).all()
    assert not zero.variability_eligible.any()
    base = _row(zero, "temp", statistic="std_detrended", quantity="ratio").to_dict()
    rows = [
        dict(
            base,
            model="TEST/A",
            source_id="A",
            member=f"r{i}",
            ratio=2.0,
            status="ok",
            variability_eligible=i < 3,
        )
        for i in range(5)
    ]
    rows.append(
        dict(
            base,
            model="TEST/B",
            source_id="B",
            member="r0",
            ratio=0.5,
            status="ok",
            variability_eligible=True,
        )
    )
    ratios = pd.DataFrame(rows)
    summary = analysis.ratio_summary(ratios)
    assert summary.iloc[0]["median"] == pytest.approx(1)
    assert summary.iloc[0]["median"] != pytest.approx(np.median([2, 2, 2, 0.5]))
    assert summary.iloc[0].n_eligible == 4
    assert summary.iloc[0].n_excluded == 2
    eligible = analysis.member_weights(ratios)
    assert eligible.loc[eligible.model == "TEST/A", "weight"].tolist() == [1 / 3] * 3
    mean = _row(analysis.model_changes(_changes(hist, future)), "temp").to_dict()
    models = pd.DataFrame(
        [
            dict(mean, model=f"TEST/{i}", family="A" if i < 2 else "B", value=value)
            for i, value in enumerate([2, 4, -1, 1])
        ]
    )
    agreed = analysis.sign_agreement(models).iloc[0]
    # FamilyB's median of[-1,1] is neutral, so only familyA votes with the median.
    assert (agreed.k, agreed.n, agreed.k_fam, agreed.n_fam) == (3, 4, 1, 2)
    assert not agreed.small_change
    assert not analysis.ensemble_summary(models).iloc[0].iqr_shown


@pytest.mark.parametrize(
    "period,variable,statistic",
    [("Jan", "temp", "std_detrended"), ("annual", "precip", "mean")],
)
def test_availability_requires_plotted_period_and_statistic(
    period, variable, statistic
):
    base = dict(
        window_id="2036-2065",
        scenario="ssp245",
        period=period,
        variable=variable,
        statistic=statistic,
        quantity="change",
        value=2.0,
        ratio=2.0,
        variability_eligible=True,
    )
    members = pd.DataFrame(
        [dict(base, model="TEST/A", member=f"r{i}") for i in range(2)]
    )
    models = members.drop_duplicates("model")
    result = analysis.product_availability(
        {"models": models, "model_ratios": models, "member_ratios": members},
        ["2036-2065"],
    ).set_index("filename")
    assert not result.loc[
        "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png", "available"
    ]
    if period == "Jan":
        assert not result.loc["cmip6_temp_seasonal_sd_box_basin_avg.png", "available"]
        means = models.assign(statistic="mean")
        result = analysis.product_availability(
            {"models": means}, ["2036-2065"]
        ).set_index("filename")
        assert not result.loc[
            "cmip6_temp_annual_seasonal_change_box_basin_avg.png", "available"
        ]
    assert members.variability_eligible.all()


def test_precipitation_sd_members_enable_member_ratio_comparison():
    members = pd.DataFrame(
        [
            dict(
                model="TEST/A",
                scenario="ssp245",
                member=f"r{i}",
                window_id="2036-2065",
                period="annual",
                variable="precip",
                statistic="std_detrended",
                ratio=2.0,
                variability_eligible=True,
            )
            for i in range(2)
        ]
    )
    result = analysis.product_availability(
        {"member_ratios": members}, ["2036-2065"]
    ).set_index("filename")
    assert result.loc[
        "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png", "available"
    ]
