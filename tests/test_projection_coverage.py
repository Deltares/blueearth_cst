"""Explicit ensemble coverage and conservative genealogy-family fallback."""

import pandas as pd
import pytest

from blueearth_cst.projections import projection_coverage as coverage


@pytest.mark.parametrize(
    "source,institution,expected",
    [
        ("MRI-ESM2-0", "MRI", ("UCLA GCM", "listed", "Japan")),
        ("MRI-NEW", "MRI", ("UCLA GCM", "provisional", "Japan")),
        ("FOO-1", "XYZ", ("FOO-1", "own", "")),
    ],
)
def test_family_lookup_marks_known_provisional_and_own(source, institution, expected):
    family = coverage.family_metadata(source, institution)
    assert (family["family"], family["family_source"], family["country"]) == expected
    listed = coverage.model_families()
    assert len(listed) == 13
    assert listed.source_id.is_unique
    ambiguous = pd.DataFrame(
        [("ONE", "A", "MIX", "USA"), ("TWO", "B", "MIX", "USA")],
        columns=["source_id", "family", "institution", "country"],
    )
    assert coverage.family_metadata("THREE", "MIX", ambiguous) == {
        "family": "THREE",
        "family_source": "own",
        "country": "USA",
    }


def _point(model, scenario, member):
    return {
        "model": model,
        "scenario": scenario,
        "member": member,
        "series_key": f"cmip6_{model.replace('/', '_')}_{scenario}_{member}",
        "reference_series_key": f"cmip6_{model.replace('/', '_')}_historical_{member}",
    }


def test_points_coverage_keeps_rows_and_deduplicates_only_member_counts():
    points = [
        _point("MRI/MRI-ESM2-0", "ssp245", "r1"),
        _point("MRI/MRI-ESM2-0", "ssp245", "r2"),
        _point("MRI/MRI-ESM2-0", "ssp585", "r1"),
        _point("NCAR/NEW-CESM", "ssp245", "r1"),
    ]
    table = coverage.coverage_table(points)
    assert len(table) == len(points)
    assert table.available.all()
    assert not table.requested_known.any()
    new = table.loc[table.source_id == "NEW-CESM"].iloc[0]
    assert (new.institution, new.family, new.family_source, new.country) == (
        "NCAR",
        "CCM",
        "provisional",
        "USA",
    )
    available = coverage.availability_table(table, ["ssp245", "ssp585", "ssp370"])
    mri = available.loc[available.source_id == "MRI-ESM2-0"].iloc[0]
    assert (mri.historical, mri.ssp245, mri.ssp585, mri.ssp370) == (2, 2, 1, 0)
    matrix = coverage.coverage_matrix(table, ["ssp245", "ssp585", "ssp370"])
    missing = matrix.loc[
        (matrix.source_id == "NEW-CESM") & (matrix.scenario == "ssp585")
    ].iloc[0]
    assert missing.state == "no_resolved_member"
    assert missing.n_resolved == 0
    assert pd.isna(missing.n_requested)
    with pytest.raises(ValueError, match="duplicate"):
        coverage.coverage_table(points + [points[0]])
    assert coverage.coverage_table([]).empty
    assert coverage.availability_table(coverage.coverage_table([])).empty


def test_requested_combinations_preserve_exclusions_and_distinguish_not_requested():
    points = [_point("MRI/MRI-ESM2-0", "ssp245", "r1")]
    combinations = [
        {
            "dataset": "MRI/MRI-ESM2-0",
            "scenario": "ssp245",
            "member": "r1",
            "status": "resolved",
        },
        {
            "dataset": "MRI/MRI-ESM2-0",
            "scenario": "ssp245",
            "member": "r2",
            "status": "member_not_published",
            "detail": "not in store",
        },
        {
            "dataset": "XYZ/FOO-1",
            "scenario": "ssp245",
            "member": "r1",
            "status": "no_historical_entry",
            "detail": "no reference",
        },
    ]
    table = coverage.coverage_table(points, combinations=combinations)
    assert len(table) == 3
    assert table.requested_known.all()
    assert table.available.sum() == 1
    assert table.loc[table.source_id == "FOO-1", "reason"].item() == "no reference"
    matrix = coverage.coverage_matrix(table, ["ssp245", "ssp585"])
    by_key = matrix.set_index(["source_id", "scenario"])
    assert by_key.loc[("MRI-ESM2-0", "ssp245"), "state"] == "partial"
    assert by_key.loc[("MRI-ESM2-0", "ssp245"), "n_requested"] == 2
    assert by_key.loc[("FOO-1", "ssp245"), "state"] == "excluded"
    assert by_key.loc[("FOO-1", "ssp585"), "state"] == "not_requested"
    assert "not in store" in by_key.loc[("MRI-ESM2-0", "ssp245"), "reason"]
    with pytest.raises(ValueError, match="resolved.*point"):
        coverage.coverage_table([], combinations=combinations)
