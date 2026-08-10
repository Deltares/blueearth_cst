# -*- coding: utf-8 -*-
"""Unit tests for the four evaluation sheets and the station identity they use.

What is worth testing here is not how the figures look — a figure is verified by
rendering it and looking at it (AGENTS.md, "Figures are terminal artifacts") —
but the arithmetic and the identity that decide WHAT is drawn and what it is
called. Each of the invariants below stands for a defect the previous
implementation had:

* the outlet's ``wflow_id`` is resolved through the location registry, because
  its series is keyed by a SUBCATCHMENT id and observations are keyed by wflow_id
  — so an observation at the basin outlet used to match nothing;
* an outlet and a gauge on one cell resolve to one station, because they used to
  produce two sets of figures under two names;
* the wettest and driest years are ranked on a MEAN, not a sum, so a year with
  missing days is not ranked by how much data it has.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.model.plot_results import resolve_stations
from blueearth_cst.shared import plot_evaluation
from blueearth_cst.shared.plot_evaluation import (
    Station,
    annual_extremes,
    r_squared,
    wettest_and_driest_year,
)

#: Every test in this module that saves a figure asserts its STRUCTURE — which
#: sheets exist, what they are named, which panels they carry. None asserts on
#: the export resolution, and 600 dpi costs seconds per sheet. See the fixture.
pytestmark = pytest.mark.usefixtures("fast_figure_dpi")


def _series(values, start="2001-01-01"):
    time = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(
        np.asarray(values, dtype="float64"), dims="time", coords={"time": time}
    )


def _indexed(indices, names):
    return xr.DataArray(
        np.zeros(len(indices)),
        dims="index",
        coords={"index": list(indices), "station_name": ("index", list(names))},
    )


@pytest.fixture
def registry(tmp_path):
    """A two-subbasin location registry, in the shape ``spatial.products`` writes."""
    path = tmp_path / "location_registry.csv"
    pd.DataFrame(
        {
            "subbasin_id": [101, 102],
            "wflow_id": [1010, 1020],
            "station_name": ["auto_01", "auto_02"],
            "is_primary": [True, True],
        }
    ).to_csv(path, index=False)
    return str(path)


# --- station identity ---------------------------------------------------------


def test_an_outlet_series_resolves_through_its_subcatchment_id(registry):
    """``Q_101`` is the outlet; its wflow_id is 1010 and nothing else knows 101."""
    stations = resolve_stations(
        _indexed([101], ["wflow_1"]), registry, log=lambda m: None
    )
    assert stations[101].wflow_id == 1010
    assert stations[101].subbasin_id == 101


def test_a_gauge_series_is_already_keyed_by_wflow_id(registry):
    stations = resolve_stations(
        _indexed([1020], ["auto_02"]), registry, log=lambda m: None
    )
    assert stations[1020].wflow_id == 1020


def test_the_registrys_name_wins_over_the_synthetic_counter(registry):
    """``wflow_1`` is a 1..N counter invented in plot_results; it names nothing."""
    stations = resolve_stations(
        _indexed([101], ["wflow_1"]), registry, log=lambda m: None
    )
    assert stations[101].station_name == "auto_01"


def test_an_outlet_and_a_gauge_on_one_cell_are_plotted_once(registry):
    """101 and 1010 are the same model cell with different index values, which
    ``merge_outlet_and_gauge_series`` cannot see — it dedupes on the raw index."""
    messages = []
    stations = resolve_stations(
        _indexed([101, 1010, 1020], ["wflow_1", "auto_01", "auto_02"]),
        registry,
        log=messages.append,
    )
    assert sorted(s.wflow_id for s in stations.values()) == [1010, 1020]
    assert any("same model cell" in message for message in messages)


def test_an_unresolvable_series_keeps_its_index_and_says_so(registry):
    messages = []
    stations = resolve_stations(
        _indexed([999], ["mystery"]), registry, log=messages.append
    )
    assert stations[999].wflow_id == 999
    assert any("999" in message for message in messages)


def test_without_a_registry_every_series_keeps_its_own_index():
    """WF3 and ad-hoc callers have no registry; the figures still get names."""
    stations = resolve_stations(_indexed([7], ["x"]), None, log=lambda m: None)
    assert stations[7].wflow_id == 7


def test_a_registry_without_is_primary_still_resolves(tmp_path):
    """`DataFrame.get(col, True)` returns the scalar default, which has no
    `.astype` — the fallback has to be explicit or this raises."""
    path = tmp_path / "registry.csv"
    pd.DataFrame(
        {"subbasin_id": [101], "wflow_id": [1010], "station_name": ["a"]}
    ).to_csv(path, index=False)
    stations = resolve_stations(
        _indexed([101], ["wflow_1"]), str(path), log=lambda m: None
    )
    assert stations[101].wflow_id == 1010


# --- the caption --------------------------------------------------------------


def test_the_caption_leads_with_the_wflow_id():
    assert Station(1010, 101, "auto_01").caption.startswith("wflow_id 1010")


def test_the_caption_survives_a_station_with_no_context():
    assert Station(1010).caption == "wflow_id 1010"


# --- what gets drawn ----------------------------------------------------------


def test_the_wettest_year_is_ranked_on_a_mean_not_a_sum():
    """A sum ranks a year by how much data it has as much as by how wet it was.

    2001 here is uniformly wetter; 2002 has more days of record.
    """
    wet = _series([10.0] * 200, start="2001-01-01")
    dry = _series([1.0] * 365, start="2002-01-01")
    series = xr.concat([wet, dry], dim="time")
    assert wettest_and_driest_year(series) == (2001, 2002)


def test_annual_maxima_use_a_september_water_year():
    """One flood season must not be split across two 'years'."""
    values = np.full(365 * 3, 1.0)
    series = _series(values, start="2001-01-01")
    sim_max, obs_max, sim_nm7q, obs_nm7q = annual_extremes(series, None)
    assert obs_max is None and obs_nm7q is None
    assert str(sim_max["time"].dt.month.values[0]) == "9"


def test_annual_extremes_returns_observed_when_it_is_given_one():
    series = _series(np.arange(400.0))
    sim_max, obs_max, sim_nm7q, obs_nm7q = annual_extremes(series, series)
    assert obs_max is not None and obs_nm7q is not None


def test_r_squared_ignores_non_finite_pairs():
    """NM7Q on a basin that dries out produces them; one NaN would blank it."""
    left = np.array([1.0, 2.0, 3.0, np.nan])
    right = np.array([1.0, 2.0, 3.0, 5.0])
    assert r_squared(left, right) == pytest.approx(1.0)


def test_r_squared_is_nan_when_there_is_nothing_to_regress():
    assert np.isnan(r_squared(np.array([np.nan]), np.array([1.0])))


# --- the sheets ---------------------------------------------------------------


def _plottable(days=800, seed=3):
    rng = np.random.default_rng(seed)
    values = np.abs(rng.normal(10.0, 4.0, days)) + np.sin(np.arange(days) / 58.0) * 3.0
    return _series(np.clip(values, 0.0, None))


def test_the_hydrograph_is_drawn_without_observations(tmp_path):
    """It is the only sheet that must never depend on them."""
    png = plot_evaluation.plot_hydrograph(_plottable(), Station(1010), str(tmp_path))
    assert (tmp_path / "hydrograph_1010.png").is_file()
    assert png.endswith("hydrograph_1010.png")
    # PNG only since 2026-08-10; the vector copy went unread.
    assert not list(tmp_path.glob("*.pdf"))


def test_every_sheet_is_named_after_the_wflow_id(tmp_path):
    simulated = _plottable()
    observed = simulated * 1.1
    metrics = pd.DataFrame(
        {"daily": [0.9] * 7, "monthly": [0.8] * 7},
        index=["KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE"],
    )
    plot_evaluation.plot_station_evaluation(
        simulated=simulated,
        observed=observed,
        station=Station(1010, 101, "auto_01"),
        plot_dir=str(tmp_path),
        metrics=metrics,
        signatures=True,
        log=lambda m: None,
    )
    written = {path.name for path in tmp_path.glob("*.png")}
    assert written == {
        "hydrograph_1010.png",
        "signatures_peaks_1010.png",
        "signatures_lows_1010.png",
        "performance_1010.png",
    }


def test_without_observations_only_the_hydrograph_is_drawn_and_it_is_reported(tmp_path):
    """ "There are no observations" and "the figure failed" look identical in an
    empty folder, so the skip is announced."""
    messages = []
    plot_evaluation.plot_station_evaluation(
        _plottable(), Station(1010), str(tmp_path), log=messages.append
    )
    assert {path.name for path in tmp_path.glob("*.png")} == {"hydrograph_1010.png"}
    assert any("no observations" in message for message in messages)


def test_a_short_record_reports_why_it_has_no_extremes_sheets(tmp_path):
    messages = []
    simulated = _plottable(days=200)
    plot_evaluation.plot_station_evaluation(
        simulated,
        Station(1010),
        str(tmp_path),
        observed=simulated * 1.1,
        signatures=False,
        log=messages.append,
    )
    assert "signatures_peaks_1010.png" not in {p.name for p in tmp_path.glob("*.png")}
    assert any("too short" in message for message in messages)


def test_an_unknown_extremes_kind_is_refused(tmp_path):
    simulated = _plottable()
    with pytest.raises(ValueError, match="expected 'peaks' or 'lows'"):
        plot_evaluation.plot_extremes(
            simulated, simulated * 1.1, Station(1010), str(tmp_path), "middling"
        )


def test_observed_is_blue_and_simulated_is_red():
    """Owner's call 2026-08-10. The hexes are the Okabe-Ito pair, so the two
    stay separable under all three dichromacies and differ in lightness."""
    assert plot_evaluation.COLOR_OBSERVED == "#0072B2"
    assert plot_evaluation.COLOR_SIMULATED == "#D55E00"


def test_every_metric_the_table_can_show_declares_an_ideal_and_a_format():
    """A reader should not have to remember whether Pbias wants 0 or 1."""
    grouped = {name for _, names in plot_evaluation.METRIC_GROUPS for name in names}
    assert grouped == set(plot_evaluation.METRIC_FORMAT)
