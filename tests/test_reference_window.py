"""Reference-window clip and warning tests for step 5e (design §5.4 D1).

The seed config exercises **none** of the warning paths — `[1990, 2010]` needs no
clip and includes 21 calendar years. Observed climate windows are independent.
The warning conditions live here, not in the tree
diff. Falsifiers K3–K6 of `dev/records/milestones/r08/2026-07-30_wf2-5e-falsifier.md`.
"""

import pytest

from blueearth_cst.projections.reference_window import (
    HISTORICAL_END_YEAR,
    alignment_record,
    clip_reference_window,
    window_warnings,
)

SEED_REQUESTED = [1990, 2010]
SEED_SHARED = ["2000-01-01T00:00:00", "2020-12-31T00:00:00"]


# --- K3: the clip must clip, and say so ---------------------------------------


def test_K3_window_overrunning_the_historical_end_is_clipped():
    w = clip_reference_window([1990, 2030])
    assert w.effective == (1990, HISTORICAL_END_YEAR)
    assert w.clipped is True


def test_K3_the_warning_names_requested_AND_effective():
    """ "Clipped" alone does not let a reader judge the damage."""
    lines = window_warnings(clip_reference_window([1990, 2030]))
    clip_line = next(line for line in lines if "shortened" in line)
    assert "1990-2030" in clip_line and f"1990-{HISTORICAL_END_YEAR}" in clip_line


def test_K3_an_unclipped_window_is_left_alone_and_silent():
    w = clip_reference_window(SEED_REQUESTED)
    assert w.effective == (1990, 2010)
    assert w.clipped is False
    assert window_warnings(w) == []


# --- K4: entirely after the historical end must RAISE -------------------------


def test_K4_window_entirely_after_the_historical_end_raises():
    """The one exception to "never raises": there is nothing to clip TO.

    Clipping to a zero-length window would propagate an empty denominator into
    every relative change factor.
    """
    with pytest.raises(ValueError, match="entirely after"):
        clip_reference_window([2020, 2040])


def test_K4_a_window_ending_exactly_at_the_boundary_is_not_an_error():
    w = clip_reference_window([2000, HISTORICAL_END_YEAR])
    assert w.clipped is False and w.n_years == HISTORICAL_END_YEAR - 2000 + 1


# --- K5: alignment is silent by default, promoted only when the clip broke it --


def test_K5_alignment_difference_alone_does_NOT_warn():
    """The seed differs on 100% of runs; warning would train readers to filter."""
    w = clip_reference_window(SEED_REQUESTED)
    assert window_warnings(w, shared_window=SEED_SHARED) == []


def test_observed_window_is_not_recorded_as_a_gcm_reference_constraint():
    """Legacy callers cannot reintroduce the observed-window comparison."""
    record = alignment_record(clip_reference_window(SEED_REQUESTED), SEED_SHARED)
    assert "reference_alignment" not in record
    assert "shared_historical_window" not in record
    assert record["reference_window_effective"] == "1990-2010"


def test_clipping_never_warns_about_observed_window_alignment():
    """Clipping is a GCM coverage issue, independent of observed climate."""
    shared = [2000, 2030]
    lines = window_warnings(clip_reference_window([2000, 2030]), shared_window=shared)
    assert not any("align" in line for line in lines)


def test_K5_matching_windows_record_as_matching():
    record = alignment_record(clip_reference_window([2000, 2010]), [2000, 2010])
    assert "reference_alignment" not in record


# --- K6: the short-window boundary the seed sits exactly on -------------------


def test_K6_the_seed_window_includes_both_calendar_endpoints():
    """An off-by-one here is invisible on the fixture and wrong everywhere else."""
    w = clip_reference_window(SEED_REQUESTED)
    assert w.n_years == 21
    assert not any("Reference period is short" in line for line in window_warnings(w))


def test_K6_nineteen_years_does_warn():
    w = clip_reference_window([1992, 2010])
    assert w.n_years == 19
    assert any("Reference period is short" in line for line in window_warnings(w))


def test_user_reference_window_has_twenty_inclusive_years():
    from blueearth_cst.projections.reference_window import console_rows

    window = clip_reference_window([1985, 2004])
    assert window.n_years == 20
    record = alignment_record(window, [1951, 2020])
    assert console_rows(record) == ["Reference period: 1985-2004 (20 years)"]
    assert window_warnings(window, shared_window=[1951, 2020]) == []


def test_K6_a_clipped_window_can_trigger_BOTH_warnings():
    """Conditions are independent; D1 says per condition, not lumped."""
    lines = window_warnings(clip_reference_window([2000, 2030]))
    assert any("shortened" in line for line in lines)
    assert any("Reference period is short" in line for line in lines)
    assert len(lines) == 2


# --- input shapes --------------------------------------------------------------


@pytest.mark.parametrize(
    "requested",
    [[1990, 2010], ("1990", "2010"), "1990, 2010", ["1990-01-01", "2010-12-31"]],
)
def test_every_configured_window_shape_normalises_to_the_same_years(requested):
    """`historical_year_range` has appeared as a list, a string and dates."""
    assert clip_reference_window(requested).effective == (1990, 2010)


def test_console_rows_state_the_period_then_its_alignment():
    from blueearth_cst.projections.reference_window import console_rows

    record = {
        "reference_window_requested": "1985-2014",
        "reference_window_effective": "1985-2014",
        "reference_window_clipped": False,
        "reference_window_years": 29,
        "shared_historical_window": "2000-2016",
        "reference_alignment": "differs",
    }
    assert console_rows(record) == [
        "Reference period: 1985-2014 (29 years)",
    ]


def test_console_rows_name_a_clip_and_omit_a_missing_comparison():
    from blueearth_cst.projections.reference_window import console_rows

    record = {
        "reference_window_requested": "1990-2020",
        "reference_window_effective": "1990-2014",
        "reference_window_clipped": True,
        "reference_window_years": 25,
    }
    assert console_rows(record) == [
        "Reference period: 1990-2014 (25 years); shortened from "
        "1990-2020 because CMIP6 historical data end in 2014"
    ]
