"""Tests for derive_change_factors.py — the composition record (design §5.7).

The change arithmetic is not retested here: step 4d imports it unchanged from
``get_change_climate_proj`` / ``get_change_climate_proj_summary``, which keep their
own suites. What is new in 4d, and therefore what is pinned here, is the
composition record — one row per **requested** combination, with the resolved-only
columns populated only when the point actually resolved.
"""

import csv

from blueearth_cst.projections.derive_change_factors import (
    COMPOSITION_CSV_COLUMNS,
    COMPOSITION_FIELDS,
    composition_rows,
    write_composition,
)


def _combo(dataset, scenario, member, status, detail=""):
    institution, _, source_id = dataset.partition("/")
    return {
        "point_key": f"{dataset.replace('/', '_')}_{scenario}_{member}",
        "dataset": dataset,
        "institution": institution,
        "source_id": source_id or dataset,
        "scenario": scenario,
        "member": member,
        "status": status,
        "detail": detail,
        "catalog_entry": f"cmip6_{dataset}_{scenario}_{{member}}",
    }


RESOLVED_FACTS = {
    "INM_INM-CM4-8_ssp245_r1i1p1f1": {
        "series_key": "cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1",
        "reference_series_key": "cmip6_INM_INM-CM4-8_historical_r1i1p1f1",
        "tier": "certified",
        "reference_window_effective": "1970-01-01 / 2009-12-01",
        "n_hyd_years_reference": 40,
    }
}


def _rows():
    combinations = [
        _combo("INM/INM-CM4-8", "ssp245", "r1i1p1f1", "resolved"),
        _combo(
            "SNU/SAM0-UNICON",
            "ssp245",
            "r1i1p1f1",
            "scenario_not_published",
            "no entry cmip6_SNU/SAM0-UNICON_ssp245_{member}",
        ),
    ]
    return composition_rows(
        combinations,
        RESOLVED_FACTS,
        catalog_crawled_on="2026-07-29",
        window_nominal="1970 / 2010",
    )


def test_one_row_per_requested_combination_not_per_resolved_one():
    """The skips are the point: a record of only resolved rows is not auditable."""
    rows = _rows()
    assert len(rows) == 2
    assert [r["status"] for r in rows] == ["resolved", "scenario_not_published"]


def test_resolved_row_carries_the_series_keys_tier_and_both_windows():
    resolved = _rows()[0]
    assert resolved["series_key"] == "cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1"
    assert resolved["reference_series_key"] == "cmip6_INM_INM-CM4-8_historical_r1i1p1f1"
    assert resolved["tier"] == "certified"
    # nominal is what the config asked for; effective is what the hydrological-year
    # windowing actually used. They differ, and the record must show both.
    assert resolved["reference_window_nominal"] == "1970 / 2010"
    assert resolved["reference_window_effective"] == "1970-01-01 / 2009-12-01"
    assert resolved["n_hyd_years_reference"] == 40


def test_skip_row_states_why_and_leaves_resolved_only_columns_empty():
    """A skip must be legible without cross-referencing, and must not fake data."""
    skip = _rows()[1]
    assert skip["status"] == "scenario_not_published"
    assert "SNU/SAM0-UNICON" in skip["reason"]
    # It still names the entry consulted and the snapshot -- that is what makes the
    # skip checkable against the store later.
    assert skip["catalog_entry"].startswith("cmip6_SNU/SAM0-UNICON")
    assert skip["catalog_crawled_on"] == "2026-07-29"
    for column in (
        "series_key",
        "reference_series_key",
        "tier",
        "reference_window_nominal",
        "reference_window_effective",
        "n_hyd_years_reference",
    ):
        assert skip[column] == "", f"{column} must stay empty on a non-resolved row"


def test_written_csv_has_the_design_columns_in_order(tmp_path):
    out = tmp_path / "summary" / "composition.csv"
    write_composition(str(out), _rows())

    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)

    assert header == [name for name, _ in COMPOSITION_CSV_COLUMNS]
    assert len(body) == 2
    # Created its parent: stage B writes into summary/ which need not exist yet.
    assert out.parent.is_dir()


# --- S8-05: the FILE is leaner than the in-memory record ----------------------


def test_the_csv_is_a_projection_of_the_wider_record(tmp_path):
    """`provenance.py` builds its institution roll-up from the full rows, so the
    record keeps fields the CSV drops. Projected on write, not at construction."""
    for field in ("institution", "source_id", "dataset", "catalog_crawled_on"):
        assert field in COMPOSITION_FIELDS
    written = {name for name, _ in COMPOSITION_CSV_COLUMNS}
    for dropped in (
        "institution",
        "dataset",
        "catalog_crawled_on",
        "reference_window_nominal",
        "reference_window_effective",
    ):
        assert dropped not in written


def test_the_csv_reports_the_source_id_under_the_name_model(tmp_path):
    out = tmp_path / "composition.csv"
    write_composition(str(out), _rows())
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["model"] == "INM-CM4-8"
    assert rows[0]["n_reference_years"] == "40"


def test_a_skip_row_stays_empty_in_the_projected_csv(tmp_path):
    out = tmp_path / "composition.csv"
    write_composition(str(out), _rows())
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    skip = rows[1]
    assert skip["status"] == "scenario_not_published"
    for column in ("series_key", "reference_series_key", "tier", "n_reference_years"):
        assert skip[column] == ""
