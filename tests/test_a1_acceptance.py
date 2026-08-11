"""A1 acceptance test for the OQ-4 30-year reference window (step 5f).

The design's §8 gate for 5f: "effective values asserted (`n_hyd_years`, effective
bounds, per-end dropped months) for January and non-January
`start_month_hyd_year`, plus the no-clip/no-short-window check".

This test exists because A1's claim is *arithmetic about a specific window*, and
that arithmetic was wrong until 2026-07-30 — the January case returned 29 where
A1 says 30 (`dev/milestones/r08/2026-07-30_wf2-5f-hydyear-offbyone.md`). Written after
the fix, deliberately, so it asserts the ruling rather than the behaviour.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections.get_change_climate_proj import hydrological_year_bounds
from blueearth_cst.projections.reference_window import (
    HISTORICAL_END_YEAR,
    SHORT_WINDOW_YEARS,
    clip_reference_window,
    dropped_months,
    window_warnings,
)

#: OQ-4, closed by the owner 2026-07-29.
OQ4_WINDOW = [1985, 2014]


@pytest.fixture
def series():
    """Monthly series covering the OQ-4 window exactly: 30 calendar years."""
    t = pd.date_range("1985-01-01", "2014-12-01", freq="MS")
    assert len(t) == 360
    return xr.DataArray(
        np.arange(len(t), dtype="float64"), dims="time", coords={"time": t}
    ).to_dataset(name="v")


# --- the no-clip / no-short-window check --------------------------------------


def test_the_recommended_window_sits_inside_the_historical_experiment():
    """A1: "that window sits entirely inside the historical experiment, so no clip
    warning fires and it clears the 20-year short-window floor"."""
    w = clip_reference_window(OQ4_WINDOW)
    assert w.clipped is False
    assert w.effective == (1985, 2014)


def test_the_recommended_window_emits_no_warnings_at_all():
    assert window_warnings(clip_reference_window(OQ4_WINDOW)) == []


def test_it_clears_the_short_window_floor_with_room():
    w = clip_reference_window(OQ4_WINDOW)
    assert w.n_years >= SHORT_WINDOW_YEARS


# --- effective values, January vs non-January ---------------------------------


def test_A1_january_start_yields_30_complete_hydrological_years(series):
    """The claim the implementation contradicted until the 2026-07-30 fix."""
    start, end, n = hydrological_year_bounds(series, "Jan")
    assert n == 30
    assert (str(start.date()), str(end.date())) == ("1985-01-01", "2014-12-01")


def test_A1_january_start_drops_no_months_at_either_end(series):
    start, end, _ = hydrological_year_bounds(series, "Jan")
    leading, trailing = dropped_months(
        series["time"].values[0], series["time"].values[-1], start, end
    )
    assert (leading, trailing) == (0, 0)


@pytest.mark.parametrize("month", ["Feb", "Apr", "Jul", "Oct", "Dec"])
def test_A1_any_other_start_month_yields_29(series, month):
    """ "for any other start month the window contains 29" — all of them."""
    _, _, n = hydrological_year_bounds(series, month)
    assert n == 29


def test_A1_october_start_drops_nine_months_leading_and_three_trailing(series):
    """The dropped months must account for exactly the missing year.

    9 + 3 = 12: 30 calendar years minus 29 hydrological years is one year, and
    A1 requires that be visible per end rather than inferred from a count.
    """
    start, end, _ = hydrological_year_bounds(series, "Oct")
    leading, trailing = dropped_months(
        series["time"].values[0], series["time"].values[-1], start, end
    )
    assert (leading, trailing) == (9, 3)
    assert leading + trailing == 12


@pytest.mark.parametrize(
    "month,leading,trailing",
    [("Feb", 1, 11), ("Apr", 3, 9), ("Jul", 6, 6), ("Oct", 9, 3), ("Dec", 11, 1)],
)
def test_A1_dropped_months_always_account_for_exactly_one_year(
    series, month, leading, trailing
):
    """Every non-January start loses exactly 12 months, split by its offset."""
    start, end, n = hydrological_year_bounds(series, month)
    assert dropped_months(
        series["time"].values[0], series["time"].values[-1], start, end
    ) == (leading, trailing)
    assert leading + trailing == 12 and n == 29


# --- the template must actually carry the recommendation ----------------------


def test_the_template_config_recommends_the_30_year_window():
    """5f puts OQ-4 in the TEMPLATE only; the seed fixture keeps [1990, 2010]."""
    import re
    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[1]
        / "config/templates/snake_config.template.yml"
    ).read_text(encoding="utf-8")
    m = re.search(r"historical_year_range:\s*\[\s*(\d{4})\s*,\s*(\d{4})\s*\]", template)
    assert m, "template must declare historical_year_range"
    assert [int(m.group(1)), int(m.group(2))] == OQ4_WINDOW


def test_the_seed_fixture_keeps_its_own_shorter_window():
    """OQ-4 is the TEMPLATE's ruling; the seed runs a deliberately smaller window.

    This assertion used to freeze the seed at ``[1990, 2010]`` to prove step 5f
    "moved no number". 5f is closed, and the seed was deliberately shortened on
    2026-08-10 to speed the fixture up, so that freeze is a fossil rather than a
    live invariant.

    What still matters, and what this now checks, is the property the seed has to
    hold whatever its numbers are: it must not END AFTER the CMIP6 historical
    experiment. A later end is silently clipped to 2014, and if the clipped range
    had equalled ``shared.historical_window`` the run also warns that the two no
    longer align — which is exactly the trap a "make the windows consistent" edit
    walks into.
    """
    import re
    from pathlib import Path

    seed = (
        Path(__file__).resolve().parents[1] / "test_case/snake_config_model_test.yml"
    ).read_text(encoding="utf-8")
    m = re.search(r"historical_year_range:\s*\[\s*(\d{4})\s*,\s*(\d{4})\s*\]", seed)
    assert m, "seed must declare historical_year_range"
    start, end = int(m.group(1)), int(m.group(2))

    assert end <= HISTORICAL_END_YEAR, (
        f"seed historical_year_range ends {end}, after the CMIP6 historical "
        f"experiment ({HISTORICAL_END_YEAR}); it would be clipped silently"
    )
    assert [start, end] != OQ4_WINDOW, (
        "the seed is the small/fast fixture and should not adopt the template's "
        "30-year OQ-4 window"
    )
    assert not clip_reference_window([start, end]).clipped
