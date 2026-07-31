"""report.md tests for step 7-ii (design §5.9). Falsifiers P3 and P4."""

import pytest

from blueearth_cst.projections.report import build, disclaimer_block

CLEAN_RUN = {
    "clim_project": "cmip6",
    "reference_window": {
        "reference_window_requested": "1990-2010",
        "reference_window_clipped": False,
        "reference_alignment": "differs",
        "shared_historical_window": "2000-2020",
    },
    "sources": [{"reference_window_effective": "1990-01-01 / 2010-12-01",
                 "n_hyd_years_reference": 21}],
    "composition": {"requested": 6, "resolved": 6, "models": 3,
                    "unresolved_by_status": {}},
    "flagged_months": [],
    "weighting_scheme": "spherical_cell_area_midpoint_edges",
    "catalog_crawled_on": "2026-07-29",
}


def _lines(doc, **kw):
    return "\n".join(disclaimer_block(doc, thresholds={"precip": 0.1}, **kw))


# --- P3: every required disclaimer element ------------------------------------


@pytest.mark.parametrize(
    "needle",
    ["Reference window", "Effective window", "Alignment", "Window length",
     "Spatial weighting", "Approximation", "Dry-month rule", "Catalog snapshot",
     "Composition"],
)
def test_P3_every_required_element_is_present(needle):
    assert needle in _lines(CLEAN_RUN)


def test_P3_the_weighting_approximation_is_named_not_just_the_scheme():
    """The design asks for the scheme AND its approximation label -- a scheme name
    alone tells a reader nothing about what it cannot do."""
    text = _lines(CLEAN_RUN)
    assert "spherical_cell_area_midpoint_edges" in text
    assert "midpoints" in text and "bounds variables" in text


# --- P4: absence must be STATED, not implied ----------------------------------


def test_P4_no_clip_is_stated_explicitly():
    assert "no clip" in _lines(CLEAN_RUN)


def test_P4_no_flagged_months_is_stated_with_the_threshold():
    """"Nothing to report" is the correct output here and also what a dead code
    path emits, so the negative must name the rule that found nothing."""
    text = _lines(CLEAN_RUN)
    assert "no months flagged" in text and "0.1" in text


def test_P4_a_fully_resolved_run_says_none_skipped():
    assert "none skipped" in _lines(CLEAN_RUN)


def test_P4_the_short_window_floor_is_stated_when_NOT_breached():
    assert "at or above the 20-year floor" in _lines(CLEAN_RUN)


# --- the positive cases, so the negatives are not the only path tested --------


def test_a_clipped_window_says_so():
    doc = dict(CLEAN_RUN)
    doc["reference_window"] = dict(CLEAN_RUN["reference_window"],
                                   reference_window_clipped=True)
    assert "**clipped**" in _lines(doc)


def test_a_short_window_is_called_out():
    doc = dict(CLEAN_RUN)
    doc["sources"] = [{"reference_window_effective": "1996-01-01 / 2010-12-01",
                       "n_hyd_years_reference": 15}]
    assert "below the 20-year floor" in _lines(doc)


def test_flagged_combinations_are_counted_and_the_excess_named():
    doc = dict(CLEAN_RUN)
    doc["flagged_months"] = [
        {"n_flagged_months": 5, "exceeds_max": True},
        {"n_flagged_months": 2, "exceeds_max": False},
    ]
    text = _lines(doc, max_flagged_months=3)
    assert "2 combination(s)" in text and "1 exceed" in text


def test_unresolved_combinations_are_broken_down_by_status():
    doc = dict(CLEAN_RUN)
    doc["composition"] = {"requested": 92, "resolved": 62, "models": 16,
                          "unresolved_by_status": {"scenario_not_published": 30}}
    text = _lines(doc)
    assert "62 of 92" in text and "30 scenario_not_published" in text


def test_matching_alignment_says_matches():
    doc = dict(CLEAN_RUN)
    doc["reference_window"] = dict(CLEAN_RUN["reference_window"],
                                   reference_alignment="matches")
    assert "matches `shared.historical_window`" in _lines(doc)


# --- whole report --------------------------------------------------------------


def test_the_report_states_the_R3_rule_up_front():
    """A reader must not mistake these for ensemble statistics."""
    text = build(CLEAN_RUN, thresholds={"precip": 0.1}, figures=["a.png"])
    assert "nothing here is averaged across models" in text


def test_figures_and_tables_are_listed():
    text = build(CLEAN_RUN, thresholds={"precip": 0.1}, figures=["a.png", "b.png"])
    assert "`plots/a.png`" in text and "change_factors/monthly.csv" in text
