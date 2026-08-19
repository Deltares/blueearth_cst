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
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.RESOLVED]


def test_requested_members_are_intersected_not_required():
    """R3′: `members:` is a requested SET; the resolved set is the intersection.

    ssp585 publishes only r1i1p1f1 here, so requesting both members yields one
    resolved point and one normal skip -- not an error.

    Names `selection="all"` explicitly since t2608192107: R3′ is no longer the
    default, and this test is about R3′ rather than about whatever the default
    happens to be.
    """
    combos = res.resolve(
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp585"],
        members=["r1i1p1f1", "r2i1p1f1"],
        selection=res.ALL_MEMBERS,
    )
    got = _statuses(combos)
    assert got[("AAA/M1", "ssp585", "r1i1p1f1")] == res.RESOLVED
    assert got[("AAA/M1", "ssp585", "r2i1p1f1")] == res.MEMBER_NOT_PUBLISHED


def test_one_row_per_REQUESTED_combination_not_per_resolved():
    """The skips are the point -- they are what makes composition.csv auditable.

    Asserted under BOTH policies since t2608192107, because that is exactly what
    the emission contract claims: the row count follows the REQUEST, and only
    the statuses follow the policy. `first_available` resolves one fewer here --
    r2i1p1f1 is complete for ssp245 and would have resolved under `all` -- and
    it is recorded as superseded rather than dropped.
    """
    kwargs = dict(
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1", "r2i1p1f1"],
    )
    every = res.resolve(FULL, selection=res.ALL_MEMBERS, **kwargs)
    first = res.resolve(FULL, selection=res.FIRST_AVAILABLE, **kwargs)

    assert len(every) == len(first) == 4  # 1 model x 2 scenarios x 2 members
    assert sum(c.resolved for c in every) == 3
    assert sum(c.resolved for c in first) == 2
    assert sum(c.status == res.MEMBER_SUPERSEDED for c in first) == 1


# --------------------------------------------------------------------------
# the normal skips
# --------------------------------------------------------------------------


def test_scenario_not_published_is_a_normal_skip():
    combos = res.resolve(
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp370"],
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
        catalog,
        clim_project=CP,
        models=["BBB/M2"],
        scenarios=["ssp585"],
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
        catalog,
        clim_project=CP,
        models=["CCC/M3"],
        scenarios=["ssp245"],
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
        FULL,
        clim_project=CP,
        models=["ZZZ/NOPE"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert [c.status for c in combos] == [res.MODEL_NOT_IN_CATALOG]
    assert res.unknown_models(combos) == ["ZZZ/NOPE"]


def test_unknown_model_is_distinguished_from_a_thin_one():
    """A model present but lacking the scenario must NOT be a config error."""
    catalog = _catalog({("DDD/M4", "historical"): ["r1i1p1f1"]})
    combos = res.resolve(
        catalog,
        clim_project=CP,
        models=["DDD/M4"],
        scenarios=["ssp245"],
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
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1"],
    )
    assert sum(c.resolved for c in combos) == 2
    assert res.references(combos) == [("AAA/M1", "r1i1p1f1")]  # ONE, not two


def test_references_split_by_member():
    """Under `all`, two members of one model are two distinct references.

    Names the policy explicitly since t2608192107: this is the multi-member
    ensemble case, and `first_available` deliberately produces ONE reference per
    model -- see `test_a_model_settles_on_ONE_member_across_every_scenario`,
    which is the property the job arithmetic depends on.
    """
    combos = res.resolve(
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1", "r2i1p1f1"],
        selection=res.ALL_MEMBERS,
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
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
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
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
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
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert res.format_status_report(combos) == ""


def test_status_report_names_each_skip_and_its_reason():
    combos = res.resolve(
        FULL,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp370"],
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


# --- member selection: first_available vs all (t2608192107) -------------------

#: `AAA/M1` is complete at BOTH forcing variants -- the shape that makes
#: `members: [r1i1p1f1, r1i1p1f2]` double-count. Measured on the real store for
#: CAMS-CSM1-0, EC-Earth3 and NorESM2-LM.
BOTH_VARIANTS = _catalog(
    {
        ("AAA/M1", "historical"): ["r1i1p1f1", "r1i1p1f2"],
        ("AAA/M1", "ssp245"): ["r1i1p1f1", "r1i1p1f2"],
        ("AAA/M1", "ssp585"): ["r1i1p1f1", "r1i1p1f2"],
    }
)

#: `BBB/M2` publishes ONLY the f2 variant -- the eight models a config asking
#: for f1 alone cannot reach at all (CNRM-CM6-1, UKESM1-0-LL, MIROC-ES2L, ...).
ONLY_F2 = _catalog(
    {
        ("BBB/M2", "historical"): ["r1i1p1f2"],
        ("BBB/M2", "ssp245"): ["r1i1p1f2"],
        ("BBB/M2", "ssp585"): ["r1i1p1f2"],
    }
)


def _resolved(combos):
    return {(c.dataset, c.scenario, c.member) for c in combos if c.resolved}


def test_a_model_complete_at_both_variants_resolves_ONCE():
    """The defect this policy closes, not the feature it adds.

    CAMS-CSM1-0, EC-Earth3 and NorESM2-LM are complete at f1 AND f2, so under
    the old union rule a config reaching for f2 made those three contribute two
    data points where every other model contributes one --
    `get_change_climate_proj_summary.py` merges across models and reduces with
    `stats="mean"`, so they were weighted double in the ensemble.
    """
    combos = res.resolve(
        BOTH_VARIANTS,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1", "r1i1p1f2"],
    )
    assert _resolved(combos) == {("AAA/M1", "ssp245", "r1i1p1f1")}


def test_the_passed_over_member_is_RECORDED_not_dropped():
    """One Combination per REQUESTED triple, which is what makes the record
    auditable -- the report has to be able to say why f2 was not used."""
    combos = res.resolve(
        BOTH_VARIANTS,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1", "r1i1p1f2"],
    )
    assert len(combos) == 2
    superseded = [c for c in combos if c.status == res.MEMBER_SUPERSEDED]
    assert [c.member for c in superseded] == ["r1i1p1f2"]
    assert "superseded by r1i1p1f1" in superseded[0].detail
    assert res.format_status_report(combos).count("r1i1p1f2") == 1


def test_all_keeps_todays_union_behaviour():
    """R3′ is not deleted, it is demoted to an opt-in for a deliberate
    multi-member ensemble."""
    combos = res.resolve(
        BOTH_VARIANTS,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1", "r1i1p1f2"],
        selection=res.ALL_MEMBERS,
    )
    assert _resolved(combos) == {
        ("AAA/M1", "ssp245", "r1i1p1f1"),
        ("AAA/M1", "ssp245", "r1i1p1f2"),
    }


def test_a_model_published_only_at_f2_is_reached_by_the_preference_list():
    """The feature half: eight models are unreachable while `members` is a flat
    list applied to every model."""
    combos = res.resolve(
        ONLY_F2,
        clim_project=CP,
        models=["BBB/M2"],
        scenarios=["ssp245"],
        members=["r1i1p1f1", "r1i1p1f2"],
    )
    assert _resolved(combos) == {("BBB/M2", "ssp245", "r1i1p1f2")}


def test_a_member_missing_from_historical_falls_through_to_the_next():
    """D7 pairs a scenario point with the SAME member's historical.

    So a member the scenario publishes but historical does not must not win --
    it would resolve here and fail at reduce time, which is the run-time raise
    D7 exists to replace.
    """
    catalog = _catalog(
        {
            ("CCC/M3", "historical"): ["r1i1p1f2"],
            ("CCC/M3", "ssp245"): ["r1i1p1f1", "r1i1p1f2"],
        }
    )
    combos = res.resolve(
        catalog,
        clim_project=CP,
        models=["CCC/M3"],
        scenarios=["ssp245"],
        members=["r1i1p1f1", "r1i1p1f2"],
    )
    assert _resolved(combos) == {("CCC/M3", "ssp245", "r1i1p1f2")}
    losers = {c.member: c.status for c in combos if not c.resolved}
    assert losers == {"r1i1p1f1": res.REFERENCE_MEMBER_UNPUBLISHED}


def test_a_model_settles_on_ONE_member_across_every_scenario():
    """The constraint that makes this per-MODEL rather than per-(model, scenario).

    Here f1 covers ssp245 only and f2 covers both. Choosing per scenario would
    give ssp245->f1 and ssp585->f2, each individually D7-valid -- and
    `analyze_projections.smk` builds its historical need set as
    `{(dataset, "historical", member)}`, so the model would acquire TWO
    historical baselines and difference its scenarios against different
    references.
    """
    catalog = _catalog(
        {
            ("DDD/M4", "historical"): ["r1i1p1f1", "r1i1p1f2"],
            ("DDD/M4", "ssp245"): ["r1i1p1f1", "r1i1p1f2"],
            ("DDD/M4", "ssp585"): ["r1i1p1f2"],
        }
    )
    combos = res.resolve(
        catalog,
        clim_project=CP,
        models=["DDD/M4"],
        scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1", "r1i1p1f2"],
    )
    assert _resolved(combos) == {
        ("DDD/M4", "ssp245", "r1i1p1f2"),
        ("DDD/M4", "ssp585", "r1i1p1f2"),
    }
    # and the job arithmetic downstream still gets ONE reference for the model
    assert res.references(combos) == [("DDD/M4", "r1i1p1f2")]


def test_a_model_with_no_complete_member_resolves_nowhere():
    """f1 covers only ssp245, f2 only ssp585: neither clears the whole set.

    Recorded, not dropped -- and the detail says nothing superseded them.
    """
    catalog = _catalog(
        {
            ("EEE/M5", "historical"): ["r1i1p1f1", "r1i1p1f2"],
            ("EEE/M5", "ssp245"): ["r1i1p1f1"],
            ("EEE/M5", "ssp585"): ["r1i1p1f2"],
        }
    )
    combos = res.resolve(
        catalog,
        clim_project=CP,
        models=["EEE/M5"],
        scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1", "r1i1p1f2"],
    )
    assert _resolved(combos) == set()
    superseded = [c for c in combos if c.status == res.MEMBER_SUPERSEDED]
    assert len(superseded) == 2
    assert all("no requested member resolves" in c.detail for c in superseded)


def test_a_single_member_list_resolves_identically_under_both_policies():
    """Why the default can change at all: every tracked config is one member,
    so flipping it invalidates no cached slice."""
    kwargs = dict(
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245", "ssp585"],
        members=["r1i1p1f1"],
    )
    first = res.resolve(BOTH_VARIANTS, selection=res.FIRST_AVAILABLE, **kwargs)
    every = res.resolve(BOTH_VARIANTS, selection=res.ALL_MEMBERS, **kwargs)
    assert first == every


# --- per-model overrides ------------------------------------------------------


def test_an_override_replaces_the_global_preference_for_that_model():
    """REPLACES rather than prepends: naming a realisation is an assertion about
    which one, and a fall-back to the global list would defeat it."""
    catalog = _catalog(
        {
            ("FFF/M6", "historical"): ["r1i1p1f1", "r13i1p1f2"],
            ("FFF/M6", "ssp245"): ["r1i1p1f1", "r13i1p1f2"],
        }
    )
    combos = res.resolve(
        catalog,
        clim_project=CP,
        models=["FFF/M6"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
        overrides={"FFF/M6": ["r13i1p1f2"]},
    )
    assert _resolved(combos) == {("FFF/M6", "ssp245", "r13i1p1f2")}
    assert {c.member for c in combos} == {"r13i1p1f2"}, "the global list is not tried"


def test_an_override_that_resolves_nothing_is_reported_for_the_caller_to_raise():
    """`resolve` records; the Snakefile decides what is fatal -- the same split
    as `unknown_models` and the nothing-resolved check."""
    catalog = _catalog(
        {
            ("GGG/M7", "historical"): ["r1i1p1f1"],
            ("GGG/M7", "ssp245"): ["r1i1p1f1"],
        }
    )
    overrides = {"GGG/M7": ["r9i9p9f9"]}
    combos = res.resolve(
        catalog,
        clim_project=CP,
        models=["GGG/M7"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
        overrides=overrides,
    )
    assert _resolved(combos) == set()
    assert res.unresolved_overrides(combos, overrides) == ["GGG/M7"]


def test_an_override_naming_a_model_the_run_does_not_request_is_reported():
    """A typo in the model key would otherwise be a silent no-op: nothing else
    sees it, because `unknown_models` only looks at models that ARE requested."""
    combos = res.resolve(
        BOTH_VARIANTS,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
        overrides={"AAA/M1-typo": ["r1i1p1f1"]},
    )
    assert res.unresolved_overrides(combos, {"AAA/M1-typo": ["r1i1p1f1"]}) == [
        "AAA/M1-typo"
    ]


def test_no_overrides_reports_nothing():
    combos = res.resolve(
        BOTH_VARIANTS,
        clim_project=CP,
        models=["AAA/M1"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )
    assert res.unresolved_overrides(combos, None) == []
    assert res.unresolved_overrides(combos, {}) == []


def test_an_unknown_policy_is_refused_at_the_call():
    """A typo in `member_selection` must not silently fall back to a policy."""
    with pytest.raises(ValueError, match="unknown member_selection"):
        res.resolve(
            BOTH_VARIANTS,
            clim_project=CP,
            models=["AAA/M1"],
            scenarios=["ssp245"],
            members=["r1i1p1f1"],
            selection="firstavailable",
        )
