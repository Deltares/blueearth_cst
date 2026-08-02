"""Reference-window clip and warning tests for step 5e (design §5.4 D1).

The seed config exercises **none** of the warning paths — `[1990, 2010]` needs no
clip, is exactly 20 years, and differs from `shared.historical_window` in the way
that must stay off stderr. So every condition D1 names lives here, not in the tree
diff. Falsifiers K3–K6 of `dev/milestones/r08/2026-07-30_wf2-5e-falsifier.md`.
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
    """"Clipped" alone does not let a reader judge the damage."""
    lines = window_warnings(clip_reference_window([1990, 2030]))
    clip_line = next(line for line in lines if "clipped" in line)
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
    assert w.clipped is False and w.n_years == HISTORICAL_END_YEAR - 2000


# --- K5: alignment is silent by default, promoted only when the clip broke it --


def test_K5_alignment_difference_alone_does_NOT_warn():
    """The seed differs on 100% of runs; warning would train readers to filter."""
    w = clip_reference_window(SEED_REQUESTED)
    assert window_warnings(w, shared_window=SEED_SHARED) == []


def test_K5_but_the_difference_IS_recorded_durably():
    """Silence and absence are different failures."""
    record = alignment_record(clip_reference_window(SEED_REQUESTED), SEED_SHARED)
    assert record["reference_alignment"] == "differs"
    assert record["shared_historical_window"] == "2000-2020"
    assert record["reference_window_effective"] == "1990-2010"


def test_K5_alignment_IS_promoted_when_the_clip_is_what_broke_it():
    """The user plausibly intended alignment and did not get it."""
    shared = [2000, 2030]
    lines = window_warnings(clip_reference_window([2000, 2030]), shared_window=shared)
    assert any("no longer aligns" in line for line in lines)


def test_K5_matching_windows_record_as_matching():
    record = alignment_record(clip_reference_window([2000, 2010]), [2000, 2010])
    assert record["reference_alignment"] == "matches"


# --- K6: the short-window boundary the seed sits exactly on -------------------


def test_K6_the_seed_window_of_exactly_20_years_does_NOT_warn():
    """An off-by-one here is invisible on the fixture and wrong everywhere else."""
    w = clip_reference_window(SEED_REQUESTED)
    assert w.n_years == 20
    assert not any("short reference window" in line for line in window_warnings(w))


def test_K6_nineteen_years_does_warn():
    w = clip_reference_window([1991, 2010])
    assert w.n_years == 19
    assert any("short reference window" in line for line in window_warnings(w))


def test_K6_a_clipped_window_can_trigger_BOTH_warnings():
    """Conditions are independent; D1 says per condition, not lumped."""
    lines = window_warnings(clip_reference_window([2000, 2030]))
    assert any("clipped" in line for line in lines)
    assert any("short reference window" in line for line in lines)
    assert len(lines) == 2


# --- input shapes --------------------------------------------------------------


@pytest.mark.parametrize(
    "requested", [[1990, 2010], ("1990", "2010"), "1990, 2010", ["1990-01-01", "2010-12-31"]]
)
def test_every_configured_window_shape_normalises_to_the_same_years(requested):
    """`historical_year_range` has appeared as a list, a string and dates."""
    assert clip_reference_window(requested).effective == (1990, 2010)
