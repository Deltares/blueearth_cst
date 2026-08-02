"""provenance.json tests for step 6a-iii (design §5.9). Falsifier M6.

M6 is not "are the fields present" but "are they the same values the other
artifacts used" — because this milestone has watched one quantity acquire two
definitions three separate times.
"""

import json

import pytest

from blueearth_cst.projections.provenance import PROVENANCE_SCHEMA_VERSION, build, write

SERIES_ATTRS = {
    "cmip6_INM_INM-CM4-8_historical_r1i1p1f1": {
        "cst_catalog_entry": "cmip6_INM/INM-CM4-8_historical_{member}",
        "cst_source_paths": '{"r1i1p1f1": {"pr": ["gr1/v20190530"], "tas": ["gr1/v20190530"]}}',
        "cst_series_digest": "abc123",
        "cst_raw_digest": "def456",
        "cst_acquisition_window": "1950-01-01 / 2014-12-31",
        "cst_time_first": "1950-01-16", "cst_time_last": "2014-12-16",
        "cst_calendar": "noleap",
        "cst_geometry_check": "1d_strictly_monotonic; lat=2 lon=2",
        "cst_weighting_scheme": "spherical_cell_area_midpoint_edges",
        "cst_reducer_module_hash": "hash789",
        "cst_members": "r1i1p1f1", "cst_crs": "4326",
    }
}

COMPOSITION = [
    {"status": "resolved", "dataset": "INM/INM-CM4-8", "institution": "INM",
     "source_id": "INM-CM4-8", "member": "r1i1p1f1"},
    {"status": "scenario_not_published", "dataset": "SNU/SAM0-UNICON",
     "institution": "SNU", "source_id": "SAM0-UNICON", "member": "r1i1p1f1"},
]


def _doc(**over):
    kwargs = dict(
        clim_project="cmip6",
        reference_record={"reference_window_requested": "1990-2010",
                          "reference_window_clipped": False,
                          "reference_alignment": "differs"},
        variable_spec={"precip": ["precip", "precip", "rate", "mm/day", "relative"]},
        composition_rows=COMPOSITION,
        series_attrs=SERIES_ATTRS,
        catalog_crawled_on="2026-07-29",
        reducer_module_hash="hash789",
        effective_config_sha256="effective123",
        region_fingerprint="fingerprint000",
        horizons={"far": "2070 / 2090"},
        weighting_scheme="spherical_cell_area_midpoint_edges",
    )
    kwargs.update(over)
    return build(**kwargs)


# --- M6: every required fact, and reconstructibility --------------------------


@pytest.mark.parametrize(
    "key",
    ["reference_window", "region_fingerprint", "reducer_module_hash",
     "effective_config_sha256", "variable_spec",
     "catalog_crawled_on", "sources", "composition", "weighting_scheme",
     "horizon_windows", "flagged_months", "schema_version"],
)
def test_M6_every_required_top_level_fact_is_present(key):
    assert key in _doc()


def test_M6_store_paths_are_structure_not_a_string():
    """D12's verified physical paths must be queryable, not a JSON blob to re-parse."""
    src = _doc()["sources"][0]
    assert src["store_paths"]["r1i1p1f1"]["tas"] == ["gr1/v20190530"]


def test_M6_effective_config_digest_is_the_supplied_workflow_identity():
    """Result provenance points to the exact merged configuration snapshot."""
    assert _doc(effective_config_sha256="abc123")["effective_config_sha256"] == (
        "abc123"
    )


def test_M6_the_effective_window_is_PER_SOURCE_not_run_level():
    """It depends on the data each combination has, so a run-level value would be
    a third definition of a quantity that already has two homes."""
    doc = _doc(effective_windows={
        "cmip6_INM_INM-CM4-8_historical_r1i1p1f1": {
            "effective": "1990-01-01 / 2010-12-01", "n_years": 21},
    })
    src = doc["sources"][0]
    assert src["reference_window_effective"] == "1990-01-01 / 2010-12-01"
    assert src["n_hyd_years_reference"] == 21
    # ...and the run-level block carries the REQUESTED window, not the effective one
    assert doc["reference_window"]["reference_window_requested"] == "1990-2010"


def test_M6_calendar_and_geometry_come_from_the_series_not_recomputed():
    src = _doc()["sources"][0]
    assert src["calendar"] == "noleap"
    assert src["geometry_check"].startswith("1d_strictly_monotonic")


# --- composition counts --------------------------------------------------------


def test_unresolved_are_counted_by_status():
    comp = _doc()["composition"]
    assert comp["requested"] == 2 and comp["resolved"] == 1
    assert comp["unresolved_by_status"] == {"scenario_not_published": 1}


def test_resolved_only_counts_feed_models_and_institutions():
    """A skipped combination must not inflate the model count."""
    comp = _doc()["composition"]
    assert comp["models"] == 1
    assert comp["institutions"] == {"INM": 1}


def test_flagged_months_exists_as_a_stated_zero():
    """6b fills it; its presence now makes absence a claim rather than an omission."""
    assert _doc()["flagged_months"] == []


# --- serialisation -------------------------------------------------------------


def test_written_json_is_sorted_and_round_trips(tmp_path):
    out = tmp_path / "provenance.json"
    write(str(out), _doc())
    text = out.read_text(encoding="utf-8")
    loaded = json.loads(text)
    assert loaded["schema_version"] == PROVENANCE_SCHEMA_VERSION
    # sort_keys makes the diff reviewable rather than order-dependent
    assert text.index('"catalog_crawled_on"') < text.index('"region_fingerprint"')
