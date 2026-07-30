"""Unit tests for ``blueearth_cst/projections/resolution.py``.

WF2 v2.0 migration step 4a. The ladder's job is to separate **absence** from
**failure** (design §4 criterion 7): under ruling R3′ a model that publishes no
ssp370, or one member where another publishes three, is the expected shape of a
correct run, and only two conditions are configuration errors.

The cases below are drawn from the real store where possible, so they test the
shapes that actually occur rather than invented ones:

* ``no_historical_entry`` — `DKRZ/MPI-ESM1-2-HR` publishes SSP members and zero
  historical members (verified in the generated catalog).
* ``reference_member_unpublished`` — D7's strict same-member pairing, which
  replaces the run-time ``asymmetric hist/clim members`` raise.
* distinct references — the non-obvious half of the job arithmetic.

All offline.
"""
from __future__ import annotations

import pytest

from blueearth_cst.projections import resolution as res

CP = "cmip6"


def _catalog(spec, crawled_on="2026-07-29"):
    """Build a minimal generated-catalog mapping from {(model, exp): [members]}."""
    catalog = {"meta": {"crawled_on": crawled_on}}
    for (model, experiment), members in spec.items():
        catalog[res.entry_key(CP, model, experiment)] = {
            "uri": f"gs://cmip6/.../{model}/{experiment}/{{member}}/Amon/{{variable}}/*/*",
            "placeholders": {"member": list(members)},
        }
    return catalog


FULL = _catalog(
    {
        ("AAA/M1", "historical"): ["r1i1p1f1", "r2i1p1f1"],
        ("AAA/M1", "ssp245"): ["r1i1p1f1", "r2i1p1f1"],
        ("AAA/M1", "ssp585"): ["r1i1p1f1"],
    }
)


def _statuses(combos):
    return {(c.dataset, c.scenario, c.member): c.status for c in combos}


# --------------------------------------------------------------------------
# the happy path and the union semantics
# --------------------------------------------------------------------------

def test_fully_published_combination_resolves():
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.RESOLVED]


def test_requested_members_are_intersected_not_required():
    """R3′: `members:` is a requested SET; the resolved set is the intersection.

    ssp585 publishes only r1i1p1f1 here, so requesting both members yields one
    resolved point and one normal skip -- not an error.
    """
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp585"],
        members=["r1i1p1f1", "r2i1p1f1"],
    )
    got = _statuses(combos)
    assert got[("AAA/M1", "ssp585", "r1i1p1f1")] == res.RESOLVED
    assert got[("AAA/M1", "ssp585", "r2i1p1f1")] == res.MEMBER_NOT_PUBLISHED


def test_one_row_per_REQUESTED_combination_not_per_resolved():
    """The skips are the point -- they are what makes composition.csv auditable."""
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1", "r2i1p1f1"],
    )
    assert len(combos) == 4  # 1 model x 2 scenarios x 2 members
    assert sum(c.resolved for c in combos) == 3


# --------------------------------------------------------------------------
# the normal skips
# --------------------------------------------------------------------------

def test_scenario_not_published_is_a_normal_skip():
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp370"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.SCENARIO_NOT_PUBLISHED]
    assert not res.unknown_models(combos)  # NOT a config error


def test_no_historical_entry_is_its_own_status():
    """The DKRZ/MPI-ESM1-2-HR shape: SSP members, zero historical members.

    Neither an unknown model nor a missing scenario -- design-v2's ladder had no
    row for it, which is why revision 4 added one.
    """
    catalog = _catalog({("BBB/M2", "ssp585"): ["r1i1p1f1"]})
    combos = res.resolve(
        catalog, clim_project=CP, models=["BBB/M2"], scenarios=["ssp585"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.NO_HISTORICAL_ENTRY]
    assert not res.unknown_models(combos)


def test_reference_member_unpublished_implements_D7_pairing():
    """D7: strict same-member pairing, no substitution.

    The scenario publishes r1i1p1f2 but historical publishes only r1i1p1f1.
    Pairing them would difference two runs differing in FORCING VARIANT as well
    as scenario, so the combination is skipped rather than cross-paired.
    """
    catalog = _catalog(
        {
            ("CCC/M3", "historical"): ["r1i1p1f1"],
            ("CCC/M3", "ssp245"): ["r1i1p1f2"],
        }
    )
    combos = res.resolve(
        catalog, clim_project=CP, models=["CCC/M3"], scenarios=["ssp245"],
        members=["r1i1p1f2"],
    )
    assert [c.status for c in combos] == [res.REFERENCE_MEMBER_UNPUBLISHED]


# --------------------------------------------------------------------------
# the two config errors
# --------------------------------------------------------------------------

def test_unknown_model_is_a_config_error():
    """The only MODEL-level error, justified by C7: absent from the generated
    catalog means absent from the store -- a typo or a stale config."""
    combos = res.resolve(
        FULL, clim_project=CP, models=["ZZZ/NOPE"], scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.MODEL_NOT_IN_CATALOG]
    assert res.unknown_models(combos) == ["ZZZ/NOPE"]


def test_unknown_model_is_distinguished_from_a_thin_one():
    """A model present but lacking the scenario must NOT be a config error."""
    catalog = _catalog({("DDD/M4", "historical"): ["r1i1p1f1"]})
    combos = res.resolve(
        catalog, clim_project=CP, models=["DDD/M4"], scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.SCENARIO_NOT_PUBLISHED]
    assert res.unknown_models(combos) == []


# --------------------------------------------------------------------------
# job arithmetic
# --------------------------------------------------------------------------

def test_references_are_DISTINCT_across_scenarios():
    """A reference is reduced once however many scenarios share it.

    This is why the seed config is 6 + 3 = 9 reduce jobs, not 12.
    """
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1"],
    )
    assert sum(c.resolved for c in combos) == 2
    assert res.references(combos) == [("AAA/M1", "r1i1p1f1")]  # ONE, not two


def test_references_split_by_member():
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245"],
        members=["r1i1p1f1", "r2i1p1f1"],
    )
    assert res.references(combos) == [
        ("AAA/M1", "r1i1p1f1"),
        ("AAA/M1", "r2i1p1f1"),
    ]


# --------------------------------------------------------------------------
# D12 — one crawl
# --------------------------------------------------------------------------

def test_index_from_a_different_crawl_raises():
    """R14: two artifacts from separate crawls could disagree undetectably."""
    with pytest.raises(RuntimeError, match="different crawls"):
        res.assert_index_matches_catalog(FULL, {"crawled_on": "2026-08-01"})


def test_matching_crawl_passes_and_missing_index_is_tolerated():
    res.assert_index_matches_catalog(FULL, {"crawled_on": "2026-07-29"})
    res.assert_index_matches_catalog(FULL, None)
    res.assert_index_matches_catalog(FULL, {})


def test_ambiguous_pin_is_reported():
    """D8/D12: ~6% of pinned stores really do match more than one grid/version."""
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    index = {
        "crawled_on": "2026-07-29",
        "sources": {
            res.entry_key(CP, "AAA/M1", "ssp245"): {
                "r1i1p1f1": {"pr": ["gn/v1", "gn/v2"], "tas": ["gn/v1"]}
            }
        },
    }
    problems = res.ambiguous_pins(index, combos, CP)
    assert len(problems) == 1
    assert "2 stores match" in problems[0]
    assert "pr" in problems[0]


def test_unambiguous_pins_report_nothing():
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    index = {
        "crawled_on": "2026-07-29",
        "sources": {
            res.entry_key(CP, "AAA/M1", "ssp245"): {
                "r1i1p1f1": {"pr": ["gn/v1"], "tas": ["gn/v1"]}
            }
        },
    }
    assert res.ambiguous_pins(index, combos, CP) == []


# --------------------------------------------------------------------------
# A3 — the certified/best-effort tier
# --------------------------------------------------------------------------

def test_best_effort_variables_are_identified():
    rename = {"pr": "precip", "tas": "temp", "rsds": "kin", "psl": "press_msl"}
    assert res.best_effort_variables(["precip", "temp"], rename) == []
    assert res.best_effort_variables(["precip", "temp", "kin"], rename) == ["kin"]
    assert res.best_effort_variables(["kin", "press_msl"], rename) == [
        "kin",
        "press_msl",
    ]


# --------------------------------------------------------------------------
# the stderr summary
# --------------------------------------------------------------------------

def test_status_report_is_empty_when_everything_resolves():
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert res.format_status_report(combos) == ""


def test_status_report_names_each_skip_and_its_reason():
    combos = res.resolve(
        FULL, clim_project=CP, models=["AAA/M1"], scenarios=["ssp370"],
        members=["r1i1p1f1"],
    )
    report = res.format_status_report(combos)
    assert "0 of 1 requested combinations resolved" in report
    assert res.SCENARIO_NOT_PUBLISHED in report
    assert "AAA/M1" in report


def test_combination_splits_institution_and_source():
    """composition.csv carries both, so downstream de-duplication (N9) needs no
    re-parsing."""
    c = res.Combination("NOAA-GFDL/GFDL-ESM4", "ssp245", "r1i1p1f1", res.RESOLVED)
    assert c.institution == "NOAA-GFDL"
    assert c.source_id == "GFDL-ESM4"


def test_crawl_dates_compare_across_yaml_date_and_json_string():
    """The generator writes crawled_on unquoted, so YAML gives datetime.date
    while the JSON index gives str. Same crawl, two types -- comparing them raw
    made this guard fire on every correct run."""
    import datetime

    catalog = {"meta": {"crawled_on": datetime.date(2026, 7, 29)}}
    res.assert_index_matches_catalog(catalog, {"crawled_on": "2026-07-29"})  # no raise

    with pytest.raises(RuntimeError, match="different crawls"):
        res.assert_index_matches_catalog(catalog, {"crawled_on": "2026-08-01"})
