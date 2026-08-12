# -*- coding: utf-8 -*-
"""Unit tests for ``blueearth_cst/shared/func_plot_signature.py``.

The module had NO direct coverage until 2026-08-11
(`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §4), while being a
shared helper with three consumers — ``model/plot_results.py``,
``shared/plot_evaluation.py`` and ``shared/wflow_outputs.py``. That is the
profile AGENTS.md names as *"a contract surface with other callers"*: only its
rule wiring was pinned, never its arithmetic.

Two halves, and they are tested differently.

* ``compute_metrics`` writes ``performance_metrics.csv``, which the performance
  sheet RENDERS rather than recomputes — so the numbers here reach a deliverable
  unchecked by anything downstream. Tested as arithmetic: the perfect-fit
  identities, the sign convention of ``Pbias``, and the daily/monthly split
  actually discriminating (the whole reason the table has two rows per metric).
* ``plot_basavg`` is a figure, and a figure is verified by rendering it and
  looking at it (AGENTS.md, "Figures are terminal artifacts"). What is tested
  here is therefore STRUCTURE, not appearance — how many panels exist, in what
  order, under which resample rule. Every case below stands for the 2026-08-10
  defect recorded in the function's own docstring: the per-subcatchment series
  were drawn as one object, which was correct only by accident while a basin had
  exactly one subcatchment, and dispatched to ``pcolormesh`` at four.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from blueearth_cst.shared import func_plot_signature as fps
from blueearth_cst.shared.func_plot_signature import (
    compute_metrics,
    plot_basavg,
    rsquared,
)

#: Every figure test here asserts STRUCTURE, never the export resolution.
pytestmark = pytest.mark.usefixtures("fast_figure_dpi")

#: A CONSTANT observation is the only way to state a bias or an error exactly,
#: and it is degenerate for the variance-normalised metrics: ``skills`` divides
#: by ``nanstd(qobs) == 0`` and reports NSE/KGE as inf/nan. The tests that use a
#: flat observation assert on RMSE, Pbias and VE, never on those — so the
#: warning is a property of the fixture, not a finding. Scoped per test rather
#: than module-wide, so a real divide-by-zero elsewhere still surfaces.
_FLAT_OBS_DIVIDE = pytest.mark.filterwarnings(
    "ignore:(divide by zero|invalid value) encountered in scalar divide:RuntimeWarning"
)


def _daily(values, start="2001-01-01"):
    time = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(
        np.asarray(values, dtype="float64"), dims="time", coords={"time": time}
    )


# ---------------------------------------------------------------------------
# rsquared
# ---------------------------------------------------------------------------


def test_a_perfect_line_scores_one():
    x = np.arange(50, dtype="float64")
    assert rsquared(x, 3.0 * x + 7.0) == pytest.approx(1.0)


def test_the_score_is_the_SQUARE_so_an_inverse_relation_also_scores_one():
    """The trap: R^2 cannot tell a perfect fit from a perfect ANTI-fit.

    Anything using this to decide "is the simulation good" needs the sign from
    elsewhere. Pinned so the property is visible rather than surprising.
    """
    x = np.arange(50, dtype="float64")
    assert rsquared(x, -3.0 * x + 7.0) == pytest.approx(1.0)


def test_pure_noise_against_a_ramp_scores_near_zero():
    rng = np.random.default_rng(0)
    x = np.arange(500, dtype="float64")
    assert rsquared(x, rng.normal(size=500)) < 0.05


# ---------------------------------------------------------------------------
# compute_metrics — the table shape
# ---------------------------------------------------------------------------


@pytest.fixture
def two_years():
    """Two years of daily flow with a seasonal cycle."""
    return _daily(10.0 + 5.0 * np.sin(np.arange(730) / 30.0))


def test_the_table_is_seven_metrics_by_two_time_types(two_years):
    df = compute_metrics(two_years, two_years, "S1")

    assert df.index.names == ["metrics", "time_type"]
    assert list(df.index.get_level_values("metrics").unique()) == [
        "KGE",
        "NSE",
        "NSElog",
        "RMSE",
        "MSE",
        "Pbias",
        "VE",
    ]
    assert list(df.index.get_level_values("time_type").unique()) == [
        "daily",
        "monthly",
    ]
    assert df.shape == (14, 1)


def test_the_single_column_is_named_after_the_station(two_years):
    """The column name is how ``performance_metrics.csv`` identifies its station.

    A default of ``"station"`` would collide the moment a basin has more than
    one, so the caller's name must reach the frame unmodified.
    """
    assert compute_metrics(two_years, two_years, "outlet_101").columns.tolist() == [
        "outlet_101"
    ]
    assert compute_metrics(two_years, two_years).columns.tolist() == ["station"]


# ---------------------------------------------------------------------------
# compute_metrics — the arithmetic
# ---------------------------------------------------------------------------


def test_a_simulation_identical_to_the_observation_scores_perfectly(two_years):
    """The identities every one of the seven metrics must satisfy at qsim == qobs.

    Efficiencies go to 1, errors to 0. This is the single check that catches an
    argument-order swap in any of the fourteen ``skills`` calls, since every
    metric here is symmetric ONLY at a perfect fit.
    """
    df = compute_metrics(two_years, two_years, "S")[["S"]]

    for metric in ("KGE", "NSE", "NSElog", "VE"):
        assert df.loc[metric, "S"].tolist() == pytest.approx([1.0, 1.0])
    for metric in ("RMSE", "MSE", "Pbias"):
        assert df.loc[metric, "S"].tolist() == pytest.approx([0.0, 0.0])


@_FLAT_OBS_DIVIDE
def test_pbias_is_positive_when_the_simulation_over_predicts():
    """Sign convention, and it is the easy one to invert.

    A simulation 10% above the observation reports ``+10``. Reported the other
    way round, a wet bias would read as a dry one in every evaluation sheet.
    """
    obs = _daily(np.full(730, 10.0))
    sim = _daily(np.full(730, 11.0))

    df = compute_metrics(sim, obs, "S")
    assert df.loc[("Pbias", "daily"), "S"] == pytest.approx(10.0)
    assert df.loc[("Pbias", "monthly"), "S"] == pytest.approx(10.0)


@_FLAT_OBS_DIVIDE
def test_the_monthly_row_forgives_what_the_daily_row_punishes():
    """Why the table carries two time types rather than one.

    A simulation that alternates +5/-5 about a flat observation has the right
    monthly mean and the wrong daily value everywhere. If the monthly row were
    computed from the same series as the daily one — a resample dropped or
    applied to only one operand — the two would agree and the distinction the
    table exists to draw would be gone.
    """
    obs = _daily(np.full(730, 10.0))
    sim = _daily(10.0 + 5.0 * (-1.0) ** np.arange(730))

    df = compute_metrics(sim, obs, "S")
    daily, monthly = df.loc[("RMSE", "daily"), "S"], df.loc[("RMSE", "monthly"), "S"]

    assert daily == pytest.approx(5.0)
    assert monthly < 1.0
    assert df.loc[("VE", "daily"), "S"] < df.loc[("VE", "monthly"), "S"]


@_FLAT_OBS_DIVIDE
def test_both_operands_are_resampled_not_just_the_simulation():
    """A one-sided resample is invisible at a perfect fit and wrong everywhere else.

    Observation constant, simulation seasonal: if ``qobs`` were compared at daily
    resolution against a monthly ``qsim`` (or vice versa), xarray would align on
    time and the monthly row would collapse to a handful of points. Pinned by
    the count the monthly row is built from — 24 months over two years.
    """
    obs = _daily(np.full(730, 10.0))
    sim = _daily(10.0 + 5.0 * np.sin(np.arange(730) / 30.0))

    monthly_rmse = compute_metrics(sim, obs, "S").loc[("RMSE", "monthly"), "S"]
    expected = float(
        np.sqrt(
            ((sim.resample(time="ME").mean("time") - 10.0) ** 2).mean("time").values
        )
    )
    assert monthly_rmse == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# plot_basavg — structure
# ---------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch):
    """Record ``(path, figure)`` for every save, without asserting on pixels.

    ``plot_basavg`` writes through the module-level ``save_figure`` binding and
    returns nothing, so the figure is only reachable at save time. Patching the
    binding is also what keeps these tests off the disk.
    """
    saves: list[tuple[str, plt.Figure]] = []

    def _fake_save_figure(path, *args, **kwargs):
        saves.append((path, kwargs.get("fig") or plt.gcf()))

    monkeypatch.setattr(fps, "save_figure", _fake_save_figure)
    yield saves
    plt.close("all")


def _basavg(codes, indices=None, years=2, value=1.0):
    """A basin-average dataset in the shape hydromt writes it.

    ``<code>_subcatchment`` is hydromt's ``<header>_<mapname>`` naming for a csv
    output reduced over the ``subcatchment`` map, which is what
    ``code_from_variable`` recovers the PLOT_META key from.
    """
    time = pd.date_range("2001-01-01", periods=365 * years, freq="D")
    data = {}
    for code in codes:
        if indices is None:
            data[f"{code}_subcatchment"] = xr.DataArray(
                np.full(len(time), value), dims="time", coords={"time": time}
            )
        else:
            data[f"{code}_subcatchment"] = xr.DataArray(
                np.full((len(indices), len(time)), value),
                dims=("index", "time"),
                coords={"index": list(indices), "time": time},
            )
    return xr.Dataset(data)


def test_one_file_is_written_per_variable_named_after_it(captured, tmp_path):
    """``_basavg_pngs`` declares one output per variable; the writer must match.

    A file per SUBCATCHMENT could not be declared, because the subcatchment
    count is unknown at DAG-parse time — hence faceting.
    """
    plot_basavg(_basavg(["gwr", "aet", "p"], indices=[101, 102]), str(tmp_path))

    assert [path for path, _ in captured] == [
        str(tmp_path / "gwr_subcatchment.png"),
        str(tmp_path / "aet_subcatchment.png"),
        str(tmp_path / "p_subcatchment.png"),
    ]


def test_every_subcatchment_gets_its_own_panel(captured, tmp_path):
    """The 2026-08-10 defect, directly.

    Four subcatchments must yield four axes. Drawn as one object the array
    arrives 2-D, ``.plot()`` dispatches to ``pcolormesh``, and ``fill_between``
    would break next because it assumes exactly 12 values.
    """
    plot_basavg(_basavg(["gwr"], indices=[103, 101, 104, 102]), str(tmp_path))

    ((_, fig),) = captured
    assert len(fig.axes) == 4


def test_the_panels_read_in_sorted_subcatchment_order(captured, tmp_path):
    """Not wflow's internal order, which was measured as [103, 101, 104, 102].

    The titles are the only place the ordering is observable, and a reader who
    cannot trust them has to cross-reference every panel against the map.
    """
    plot_basavg(_basavg(["gwr"], indices=[103, 101, 104, 102]), str(tmp_path))

    ((_, fig),) = captured
    assert [ax.get_title() for ax in fig.axes] == [
        "subcatchment 101",
        "subcatchment 102",
        "subcatchment 103",
        "subcatchment 104",
    ]


def test_a_series_without_an_index_dimension_still_draws_exactly_one_panel(
    captured, tmp_path
):
    """The single-subcatchment path: ``plot_results`` squeezes a size-1 index away.

    That squeeze is what made the old one-object implementation look correct, so
    the fix must not have broken the case it accidentally worked for. The lone
    panel carries no subcatchment title, because there is no id to name.
    """
    plot_basavg(_basavg(["gwr"], indices=None), str(tmp_path))

    ((_, fig),) = captured
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == ""


def test_a_single_subcatchment_keeps_its_id_in_the_title(captured, tmp_path):
    """A size-1 index that survived the squeeze is still identified."""
    plot_basavg(_basavg(["gwr"], indices=[101]), str(tmp_path))

    ((_, fig),) = captured
    assert [ax.get_title() for ax in fig.axes] == ["subcatchment 101"]


# ---------------------------------------------------------------------------
# plot_basavg — the resample rule
# ---------------------------------------------------------------------------


def test_a_flux_variable_is_SUMMED_within_the_month(captured, tmp_path):
    """``gwr`` is mm month-1: a constant 1 mm/day must plot as ~30, not as 1.

    This is the arithmetic PLOT_META's ``resample`` key exists to select, and it
    is invisible in a rendered figure unless you read the axis.
    """
    plot_basavg(_basavg(["gwr"], indices=[101], value=1.0), str(tmp_path))

    ((_, fig),) = captured
    plotted = fig.axes[0].lines[0].get_ydata()
    assert len(plotted) == 12
    assert plotted.min() >= 28.0
    assert plotted.max() <= 31.0


def test_a_rate_variable_is_AVERAGED_within_the_month(captured, tmp_path):
    """``qof`` is m3 s-1: a constant 1 must stay 1, not become ~30."""
    plot_basavg(_basavg(["qof"], indices=[101], value=1.0), str(tmp_path))

    ((_, fig),) = captured
    assert fig.axes[0].lines[0].get_ydata() == pytest.approx(np.full(12, 1.0))


def test_the_y_label_is_the_declared_legend_for_the_variable(captured, tmp_path):
    """One table (``PLOT_META``) keys both the resample rule and the axis text."""
    plot_basavg(_basavg(["aet"], indices=[101, 102]), str(tmp_path))

    ((_, fig),) = captured
    assert {ax.get_ylabel() for ax in fig.axes} == {
        "Actual Evapotranspiration (mm month$^{-1}$)"
    }


def test_the_x_axis_is_twelve_months_labelled_by_initial(captured, tmp_path):
    """A climatology panel, so the axis is months — never the raw time index."""
    plot_basavg(_basavg(["gwr"], indices=[101]), str(tmp_path))

    ((_, fig),) = captured
    bottom = fig.axes[-1]
    assert list(bottom.get_xticks()) == list(range(1, 13))
    assert [t.get_text() for t in bottom.get_xticklabels()] == list("JFMAMJJASOND")


def test_an_unsupported_variable_fails_loud_rather_than_drawing_a_blank_panel(
    captured, tmp_path
):
    """``PLOT_META`` is the closed set of plottable outputs.

    A variable outside it has no resample rule and no axis legend, so guessing
    one would produce a figure that looks finished and means nothing.
    """
    with pytest.raises(KeyError):
        plot_basavg(_basavg(["not_a_wflow_output"], indices=[101]), str(tmp_path))
