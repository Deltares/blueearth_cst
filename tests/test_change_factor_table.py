"""Tidy change-factor table tests for step 6a-i (design §5.9). Falsifiers M1-M4."""

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.projections.change_factor_table import (
    TABLE_COLUMNS,
    tidy_rows,
    write_table,
)


@pytest.fixture
def wide():
    """The shape stage B produces: variables as data_vars over a coord grid."""
    coords = {
        "clim_project": ["cmip6"],
        "model": ["INM/INM-CM4-8", "NOAA-GFDL/GFDL-ESM4"],
        "scenario": ["ssp245", "ssp585"],
        "horizon": ["far"],
        "member": ["r1i1p1f1"],
        "stats": ["mean", "median", "std"],
    }
    shape = tuple(len(v) for v in coords.values())
    rng = np.arange(np.prod(shape), dtype="float64").reshape(shape)
    return xr.Dataset(
        {
            "precip": (tuple(coords), rng),
            "temp": (tuple(coords), rng * -1.5),
            # the CRS coordinate that leaks into today's CSV
            "spatial_ref": ((), 0),
        },
        coords=coords,
    )


# --- M1: long format, one row per full key ------------------------------------


def test_M1_row_count_is_the_full_cross_product(wide):
    """2 models x 2 scenarios x 1 horizon x 3 statistics x 2 variables = 24."""
    rows = tidy_rows(wide)
    assert len(rows) == 24


def test_M1_variables_become_rows_not_columns(wide):
    rows = tidy_rows(wide)
    assert {r["variable"] for r in rows} == {"precip", "temp"}
    assert "precip" not in TABLE_COLUMNS


def test_M1_the_key_fields_are_all_present(wide):
    row = tidy_rows(wide)[0]
    for field in ("dataset", "institution", "scenario", "member", "horizon",
                  "period", "variable", "statistic"):
        assert row[field] != "", f"{field} must be populated"


def test_M1_institution_is_split_from_the_dataset(wide):
    row = next(r for r in tidy_rows(wide) if r["dataset"] == "NOAA-GFDL/GFDL-ESM4")
    assert row["institution"] == "NOAA-GFDL"
    assert row["source_id"] == "GFDL-ESM4"


# --- M2: spatial_ref must not survive -----------------------------------------


def test_M2_the_crs_coordinate_is_not_a_change_factor(wide):
    """It is in today's CSV only because to_dataframe() dumps every coordinate."""
    rows = tidy_rows(wide)
    assert "spatial_ref" not in {r["variable"] for r in rows}


# --- M3: values identical to the wide table -----------------------------------


def test_M3_every_value_matches_the_wide_dataset(wide):
    """6a reshapes; it does not recompute. Joined, not counted."""
    for row in tidy_rows(wide):
        expected = wide[row["variable"]].sel(
            clim_project="cmip6",
            model=row["dataset"],
            scenario=row["scenario"],
            horizon=row["horizon"],
            member=row["member"],
            stats=row["statistic"],
        ).values.item()
        assert row["value"] == expected


def test_M3_no_row_is_dropped_or_duplicated(wide):
    """A reshape that loses rows still produces a plausible-looking file."""
    rows = tidy_rows(wide)
    keys = [
        (r["dataset"], r["scenario"], r["horizon"], r["member"], r["variable"], r["statistic"])
        for r in rows
    ]
    assert len(keys) == len(set(keys))


# --- M4: provenance columns carry the EFFECTIVE window ------------------------


def test_M4_n_years_reports_the_effective_count_not_the_nominal(wide):
    """Untestable on the seed, where the two coincide -- hence a window here
    where they differ."""
    facts = {
        "reference_window_nominal": "1985-2014",
        "reference_window_effective": "1986-01-01 / 2014-09-01",
        "n_years": 29,          # effective, after dropping partial years
        "n_years_dropped": 12,
    }
    row = tidy_rows(wide, window_facts=facts)[0]
    assert row["n_years"] == 29
    assert row["n_years_dropped"] == 12
    assert row["reference_window_nominal"] != row["reference_window_effective"]


def test_M4_status_column_exists_and_defaults_to_ok(wide):
    """6b's dry-month rule needs somewhere to land."""
    assert all(r["status"] == "ok" for r in tidy_rows(wide))
    assert "status" in TABLE_COLUMNS


# --- determinism / schema ------------------------------------------------------


def test_row_order_is_deterministic(wide):
    """The CSV is fingerprinted by sha256; an unstable order makes it
    unreproducible for no reason."""
    assert tidy_rows(wide) == tidy_rows(wide)


def test_written_csv_has_the_design_columns_in_order(wide, tmp_path):
    import csv

    out = tmp_path / "change_factors" / "annual.csv"
    write_table(str(out), tidy_rows(wide))
    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == TABLE_COLUMNS
        assert len(list(reader)) == 24


def test_the_period_column_distinguishes_annual_from_monthly(wide):
    assert {r["period"] for r in tidy_rows(wide)} == {"annual"}
    assert {r["period"] for r in tidy_rows(wide, period="3")} == {"3"}
