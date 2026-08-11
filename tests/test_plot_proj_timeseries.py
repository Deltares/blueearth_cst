"""WF2 projection-figure contracts."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blueearth_cst.projections.plot_proj_timeseries import (
    figure_relative_paths,
    monthly_change_series,
    plot_annual_projection,
    plot_monthly_change_factors,
)


def _legend_labels(fig):
    return [text.get_text() for legend in fig.legends for text in legend.get_texts()]


def _monthly_table(horizons=("near", "far")):
    rows = []
    for horizon in horizons:
        for model, scenario, member in (
            ("Model-A", "ssp245", "r1i1p1f1"),
            ("Model-B", "ssp585", "r2i1p1f1"),
        ):
            for month in range(1, 13):
                for variable, value in (
                    ("precip", month * (1 if horizon == "near" else 100)),
                    ("temp", month * (0.1 if horizon == "near" else 10)),
                ):
                    rows.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "member": member,
                            "horizon": horizon,
                            "month": month,
                            "variable": variable,
                            "statistic": "mean",
                            "relative_value": value,
                        }
                    )
    return pd.DataFrame(rows)


def test_multi_window_paths_include_sanitized_names_and_years():
    paths = figure_relative_paths({"Near term": [2030, 2060], "far/late": "2070, 2090"})

    assert paths == [
        "overview/annual-precipitation.png",
        "overview/annual-temperature.png",
        "overview/change-factor-cloud.png",
        "windows/near-term-2030-2060/monthly-change-factors.png",
        "windows/far-late-2070-2090/monthly-change-factors.png",
    ]


def test_monthly_changes_use_matching_month_and_only_selected_horizon():
    """The old annual-baseline/full-series calculation gives neither result.

    Historical Jan/Feb are 10/20 and the selected future Jan/Feb are 20/10,
    hence +100%/-50%. The far horizon carries extreme values so a full-future
    climatology would also fail this hand calculation.
    """
    table = pd.DataFrame(
        [
            {
                "model": "Model-A",
                "scenario": "ssp245",
                "member": "r1i1p1f1",
                "horizon": "near",
                "month": 1,
                "variable": "precip",
                "statistic": "mean",
                "reference_value": 10.0,
                "absolute_value": 20.0,
                "relative_value": 100.0,
            },
            {
                "model": "Model-A",
                "scenario": "ssp245",
                "member": "r1i1p1f1",
                "horizon": "near",
                "month": 2,
                "variable": "precip",
                "statistic": "mean",
                "reference_value": 20.0,
                "absolute_value": 10.0,
                "relative_value": -50.0,
            },
            {
                "model": "Model-A",
                "scenario": "ssp245",
                "member": "r1i1p1f1",
                "horizon": "far",
                "month": 1,
                "variable": "precip",
                "statistic": "mean",
                "reference_value": 10.0,
                "absolute_value": 10_000.0,
                "relative_value": 99_900.0,
            },
        ]
    )

    traces = monthly_change_series(table, horizon="near", variable="precip")

    assert list(traces) == [("Model-A", "ssp245", "r1i1p1f1")]
    assert traces[("Model-A", "ssp245", "r1i1p1f1")].loc[[1, 2]].tolist() == [
        100.0,
        -50.0,
    ]


def test_monthly_figure_draws_each_combination_with_scenario_only_legend():
    fig = plot_monthly_change_factors(
        _monthly_table(),
        horizon="near",
        period=[2030, 2060],
        scenarios=["ssp245", "ssp585"],
        reference_window=("1990-01-01", "2010-12-31"),
    )
    try:
        traces = [
            line
            for ax in fig.axes
            for line in ax.lines
            if line.get_gid() == "combination-trace"
        ]
        assert len(traces) == 4  # two combinations x two variable panels
        assert _legend_labels(fig) == ["SSP2-4.5", "SSP5-8.5"]
        assert all(text not in _legend_labels(fig) for text in ("Model-A", "r1i1p1f1"))
        assert [tick.get_text() for tick in fig.axes[1].get_xticklabels()] == [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        assert any(
            text.get_text() == "Reference: 1990-01-01 to 2010-12-31"
            for text in fig.texts
        )
    finally:
        plt.close(fig)


def test_annual_figure_draws_all_traces_with_historical_and_scenario_legend():
    hist_index = pd.date_range("1990-01-01", periods=24, freq="MS")
    future_index = pd.date_range("2015-01-01", periods=24, freq="MS")
    historical = {
        (model, "historical", "r1i1p1f1"): pd.DataFrame(
            {"precip": np.arange(24, dtype=float) + offset}, index=hist_index
        )
        for model, offset in (("Model-A", 10.0), ("Model-B", 20.0))
    }
    future = {
        (model, scenario, "r1i1p1f1"): pd.DataFrame(
            {"precip": np.arange(24, dtype=float) + offset}, index=future_index
        )
        for model, scenario, offset in (
            ("Model-A", "ssp245", 30.0),
            ("Model-B", "ssp245", 40.0),
            ("Model-A", "ssp585", 50.0),
            ("Model-B", "ssp585", 60.0),
        )
    }

    fig = plot_annual_projection(
        historical,
        future,
        variable="precip",
        reference_window=("1990-01-01", "1991-12-31"),
        scenarios=["ssp245", "ssp585"],
    )
    try:
        traces = [
            line
            for ax in fig.axes
            for line in ax.lines
            if line.get_gid() == "combination-trace"
        ]
        assert len(traces) == 12  # (2 historical + 4 future) x two panels
        assert _legend_labels(fig) == ["Historical", "SSP2-4.5", "SSP5-8.5"]
        assert all(text not in _legend_labels(fig) for text in ("Model-A", "r1i1p1f1"))
        assert "1990-01-01 to 1991-12-31" in fig.axes[1].get_title()
    finally:
        plt.close(fig)
