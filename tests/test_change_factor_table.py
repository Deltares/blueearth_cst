"""Tidy change-factor table tests for step 6a-i (design §5.9). Falsifiers M1-M4.

Reshaped at S8-04: two values per row (`absolute_value` = the future level,
`relative_value` = the change against the reference), each with its own units
column, and nothing inferred from a variable name.
"""

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.projections.change_factor_table import (
    TABLE_COLUMNS_ANNUAL,
    TABLE_COLUMNS_MONTHLY,
    csv_value,
    tidy_rows,
    write_table,
)
from blueearth_cst.projections.variable_spec import VariableSpec

SPEC = {
    "precip": VariableSpec("precip", "precip", "rate", "mm/day", "relative"),
    "temp": VariableSpec("temp", "temp", "state", "degC", "absolute"),
}


@pytest.fixture
def wide():
    """The shape stage B produces: variables as data_vars over a coord grid,
    plus the `__level` companion carrying the future level."""
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
            "precip__level": (tuple(coords), rng + 100.0),
            "temp__level": (tuple(coords), rng + 200.0),
            "precip__reference": (tuple(coords), rng + 10.0),
            "temp__reference": (tuple(coords), rng + 20.0),
            # the CRS coordinate that leaks into today's CSV
            "spatial_ref": ((), 0),
        },
        coords=coords,
    )


def _rows(ds, **kw):
    return tidy_rows(ds, variable_spec=SPEC, **kw)


# --- M1: long format, one row per full key ------------------------------------


def test_M1_row_count_is_the_full_cross_product(wide):
    """2 models x 2 scenarios x 1 horizon x 3 statistics x 2 variables = 24."""
    assert len(_rows(wide)) == 24


def test_M1_variables_become_rows_not_columns(wide):
    rows = _rows(wide)
    assert {r["variable"] for r in rows} == {"precip", "temp"}
    assert "precip" not in TABLE_COLUMNS_ANNUAL


def test_M1_companions_are_columns_never_rows(wide):
    """`__level` qualifies its variable; it is not a variable of its own."""
    variables = {r["variable"] for r in _rows(wide)}
    assert "precip__level" not in variables
    assert "precip__reference" not in variables


def test_M1_the_key_fields_are_all_present(wide):
    row = _rows(wide)[0]
    for field in ("model", "scenario", "member", "horizon", "variable", "statistic"):
        assert row[field] != "", f"{field} must be populated"


def test_M1_model_is_the_source_id_with_the_institution_collapsed(wide):
    """S8-05: `dataset`/`institution`/`source_id` encoded two facts in three
    columns. `source_id` is unique in the CMIP6 controlled vocabulary."""
    models = {r["model"] for r in _rows(wide)}
    assert models == {"INM-CM4-8", "GFDL-ESM4"}
    assert "institution" not in TABLE_COLUMNS_ANNUAL
    assert "dataset" not in TABLE_COLUMNS_ANNUAL


# --- M2: spatial_ref must not survive -----------------------------------------


def test_M2_the_crs_coordinate_is_not_a_change_factor(wide):
    """It is in today's CSV only because to_dataframe() dumps every coordinate."""
    assert "spatial_ref" not in {r["variable"] for r in _rows(wide)}


# --- M3: values identical to the wide table -----------------------------------


def test_M3_every_value_matches_the_wide_dataset(wide):
    """6a reshapes; it does not recompute. Joined, not counted."""
    by_source_id = {"INM-CM4-8": "INM/INM-CM4-8", "GFDL-ESM4": "NOAA-GFDL/GFDL-ESM4"}
    for row in _rows(wide):
        sel = dict(
            clim_project="cmip6",
            model=by_source_id[row["model"]],
            scenario=row["scenario"],
            horizon=row["horizon"],
            member=row["member"],
            stats=row["statistic"],
        )
        assert row["relative_value"] == wide[row["variable"]].sel(**sel).values.item()
        assert (
            row["absolute_value"]
            == wide[f"{row['variable']}__level"].sel(**sel).values.item()
        )
        assert (
            row["reference_value"]
            == wide[f"{row['variable']}__reference"].sel(**sel).values.item()
        )


def test_M3_no_row_is_dropped_or_duplicated(wide):
    """A reshape that loses rows still produces a plausible-looking file."""
    keys = [
        (
            r["model"],
            r["scenario"],
            r["horizon"],
            r["member"],
            r["variable"],
            r["statistic"],
        )
        for r in _rows(wide)
    ]
    assert len(keys) == len(set(keys))


# --- S8-04: units come from the SPEC, never from the variable name -------------


def test_units_describe_absolute_value_and_relative_units_describe_the_change(wide):
    precip = next(r for r in _rows(wide) if r["variable"] == "precip")
    temp = next(r for r in _rows(wide) if r["variable"] == "temp")
    # `absolute_value` is a level, always in the variable's own units
    assert precip["units"] == "mm/day"
    assert temp["units"] == "degC"
    # `relative_value` is a percent for a relative variable, a difference for an
    # absolute one. This is the column that used to say `mm/day` over a percent.
    assert precip["relative_units"] == "%"
    assert temp["relative_units"] == "degC"


def test_a_relative_variable_NOT_called_precip_is_still_a_percent(wide):
    """Falsifier K7, one layer up. The old `units` came from the spec but the
    ARITHMETIC came from the name, so the two could disagree; now one source
    decides both."""
    spec = {
        "rainfall": VariableSpec("rainfall", "rainfall", "rate", "mm/day", "relative"),
    }
    ds = wide.rename({"precip": "rainfall", "precip__level": "rainfall__level"})
    rows = [r for r in tidy_rows(ds, variable_spec=spec) if r["variable"] == "rainfall"]
    assert rows
    assert all(r["relative_units"] == "%" for r in rows)


def test_an_unknown_variable_falls_back_without_claiming_units(wide):
    rows = [r for r in tidy_rows(wide) if r["variable"] == "precip"]
    assert all(r["units"] == "" for r in rows)


# --- M4: the window column carries the EFFECTIVE window -----------------------


def test_M4_run_level_facts_fill_both_window_columns(wide):
    """The run-level fallback: what a row carries when nothing resolved it."""
    facts = {
        "reference_window": "1986-01-01 / 2014-09-01",
        "horizon_window": {"far": "2070-2090"},
    }
    row = _rows(wide, window_facts=facts)[0]
    assert row["reference_window"] == "1986-01-01 / 2014-09-01"
    assert row["horizon_window"] == "2070-2090"


def test_M4_a_per_row_window_overrides_the_run_level_one(wide):
    """The effective bounds are a property of a SERIES, not of the run — a model
    with a short record reports its own."""
    facts = {"reference_window": "1990-01-01 / 2010-12-01", "horizon_window": {}}
    row_facts = {
        ("NOAA-GFDL/GFDL-ESM4", "ssp245", "r1i1p1f1", "far"): {
            "reference_window": "1995-01-01 / 2010-12-01",
        }
    }
    rows = _rows(wide, window_facts=facts, row_facts=row_facts)
    overridden = [
        r for r in rows if r["model"] == "GFDL-ESM4" and r["scenario"] == "ssp245"
    ]
    assert overridden
    assert all(r["reference_window"] == "1995-01-01 / 2010-12-01" for r in overridden)
    others = [r for r in rows if r["model"] == "INM-CM4-8"]
    assert all(r["reference_window"] == "1990-01-01 / 2010-12-01" for r in others)


def test_M4_both_windows_read_in_the_same_effective_form(wide):
    """`horizon_window` used to be the config's nominal years (`2070-2090`) while
    `reference_window` was the effective span — two meanings, two formats, in
    adjacent columns. Both are now the effective bounds from
    `hydrological_year_bounds`, in one `%Y-%m-%d / %Y-%m-%d` form."""
    facts = {
        "reference_window": "nominal-fallback",
        "horizon_window": {"far": "2070-2090"},
    }
    row_facts = {
        (model, scenario, "r1i1p1f1", "far"): {
            "reference_window": "1990-01-01 / 2010-12-01",
            "horizon_window": "2070-01-01 / 2090-12-01",
        }
        for model in ("INM/INM-CM4-8", "NOAA-GFDL/GFDL-ESM4")
        for scenario in ("ssp245", "ssp585")
    }
    rows = _rows(wide, window_facts=facts, row_facts=row_facts)
    assert rows
    for r in rows:
        assert r["reference_window"] == "1990-01-01 / 2010-12-01"
        assert r["horizon_window"] == "2070-01-01 / 2090-12-01"
        # the property the user asked for: one shape, both columns
        for column in ("reference_window", "horizon_window"):
            start, sep, end = r[column].partition(" / ")
            assert sep, f"{column} is not a two-sided window: {r[column]!r}"
            for side in (start, end):
                assert len(side) == 10 and side[4] == side[7] == "-", side


def test_M4_the_horizon_window_varies_with_the_horizon(wide):
    """Unlike the reference window, it depends on the horizon — which is why the
    override key carries one."""
    two = wide.reindex(horizon=["far"]).copy()
    facts = {"reference_window": "", "horizon_window": {"far": "2070-2090"}}
    row_facts = {
        ("INM/INM-CM4-8", "ssp245", "r1i1p1f1", "far"): {
            "horizon_window": "2070-01-01 / 2090-12-01",
        }
    }
    rows = _rows(two, window_facts=facts, row_facts=row_facts)
    hit = [r for r in rows if r["model"] == "INM-CM4-8" and r["scenario"] == "ssp245"]
    miss = [r for r in rows if r["scenario"] == "ssp585"]
    assert all(r["horizon_window"] == "2070-01-01 / 2090-12-01" for r in hit)
    assert all(r["horizon_window"] == "2070-2090" for r in miss)


def test_M4_status_column_exists_and_defaults_to_ok(wide):
    """6b's dry-month rule needs somewhere to land."""
    assert all(r["status"] == "ok" for r in _rows(wide))
    assert "status" in TABLE_COLUMNS_ANNUAL


def test_a_flagged_month_keeps_its_level_and_says_why(wide):
    """The two-value layout expresses 6b natively."""
    from blueearth_cst.projections.dry_month import FLAGGED_STATUS

    ds = wide.copy()
    ds["precip__flagged"] = xr.ones_like(ds["precip"]).astype("int8")
    rows = [r for r in _rows(ds) if r["variable"] == "precip"]
    assert rows
    assert all(r["status"] == FLAGGED_STATUS for r in rows)
    # S8-08: 6b drops the meaningless ratio and KEEPS the informative difference.
    # Both levels survive, so the difference is exactly recoverable from the row.
    assert all(r["absolute_value"] != "" for r in rows)
    assert all(r["reference_value"] != "" for r in rows)
    assert all(
        r["absolute_value"] - r["reference_value"] == pytest.approx(90.0) for r in rows
    )
    # temp carries no companion, so it is untouched
    assert all(r["status"] == "ok" for r in _rows(ds) if r["variable"] == "temp")


# --- determinism / schema ------------------------------------------------------


def test_row_order_is_deterministic(wide):
    """The CSV is fingerprinted by sha256; an unstable order makes it
    unreproducible for no reason."""
    assert _rows(wide) == _rows(wide)


def test_written_csv_has_the_design_columns_in_order(wide, tmp_path):
    import csv

    out = tmp_path / "summary" / "cmip6_change_factors_annual.csv"
    write_table(str(out), _rows(wide), columns=TABLE_COLUMNS_ANNUAL)
    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == TABLE_COLUMNS_ANNUAL
        assert len(list(reader)) == 24
    assert "reference_value" in TABLE_COLUMNS_ANNUAL


def test_the_month_column_exists_only_in_the_monthly_table(wide):
    """S8-04 split the schemas. `period` was constant in one table and the key in
    the other, and no consumer stacked them."""
    assert "month" not in TABLE_COLUMNS_ANNUAL
    assert "month" in TABLE_COLUMNS_MONTHLY
    assert all("month" not in r for r in _rows(wide))
    assert {r["month"] for r in _rows(wide, month=3)} == {3}


def test_dead_columns_are_gone(wide):
    """`horizon_window_effective` and `n_years_dropped` were hardcoded empty."""
    for dead in (
        "period",
        "units_of_value",
        "horizon_window_effective",
        "n_years_dropped",
        "n_years",
        "reference_series_key",
    ):
        assert dead not in TABLE_COLUMNS_ANNUAL
        assert dead not in TABLE_COLUMNS_MONTHLY


# --- CSV number formatting ----------------------------------------------------
# The written cell is fixed to CSV_DECIMALS places. Two goals, one change: Excel
# stops prompting to convert (a full float64 repr runs to 17 significant digits),
# and the tables carry the 3 dp the results are meaningful to.


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (9.331056256303583, "9.331"),
        (-1.8660387707268327, "-1.866"),
        (1.9014461250814094, "1.901"),
        (0.0, "0.000"),
        (2.0, "2.000"),
        # never an exponent: str() would render these 1e-06 / 1.5e-07, the form
        # Excel converts most eagerly
        (0.000001, "0.000"),
        (1.5e-07, "0.000"),
        # a negative value rounded away must not keep its sign
        (-0.0001, "0.000"),
        (-1e-09, "0.000"),
    ],
)
def test_floats_are_written_fixed_to_three_places(value, expected):
    assert csv_value(value) == expected


def test_non_floats_pass_through_untouched():
    """Integer columns stay integral and text stays text."""
    for value in (21, 1, "ok", "mm/day", "", "1990-01-01 / 2010-12-01", True):
        assert csv_value(value) is value


def test_nan_is_unchanged():
    """Whether a flagged ratio should be an empty cell is a separate question
    about missing-vs-undefined; this is number formatting."""
    nan = float("nan")
    assert csv_value(nan) != csv_value(nan) or csv_value(nan) == "nan"
    assert str(csv_value(nan)) == "nan"


def test_written_cells_carry_no_long_float(wide, tmp_path):
    """The end-to-end property: no cell in a written table exceeds 3 decimals,
    and none is in scientific notation."""
    import csv as _csv
    import re

    out = tmp_path / "cmip6_change_factors_monthly.csv"
    write_table(str(out), _rows(wide, month=3), columns=TABLE_COLUMNS_MONTHLY)
    with out.open(encoding="utf-8", newline="") as fh:
        cells = [c for row in _csv.DictReader(fh) for c in row.values()]
    assert cells, "fixture wrote no rows"
    for cell in cells:
        assert "e" not in cell.lower() or not re.match(r"^[+-]?[\d.]+e", cell, re.I)
        if re.match(r"^[+-]?\d+\.\d+$", cell):
            assert len(cell.split(".")[1]) <= 3, cell
