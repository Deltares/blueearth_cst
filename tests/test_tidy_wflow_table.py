"""Tests for the derived per-variable wflow tables (blueearth_cst/shared/tidy_wflow_table.py).

The raw ``output.csv`` is an interface with hydromt's reader, so these tables are
DERIVED. What matters is that the derivation is lossless where it claims to be
(station identity, row count, ordering) and lossy only where it intends to be
(five decimal places).
"""

import pandas as pd
import pytest

from blueearth_cst.shared.tidy_wflow_table import (
    _format_value,
    slugify,
    split_columns,
    tidy_tables,
    write_tidy_tables,
)


def _frame():
    """A frame shaped like wflow's csv: unsorted ids, two variables, e-notation."""
    return pd.DataFrame(
        {
            "time": ["2000-01-02T00:00:00", "2000-01-03T00:00:00"],
            "Q_101": [0.007244979034877556, 9.865960861963168e-5],
            "Q_1040": [0.029751873471128395, 0.0016064011491603307],
            "Q_1010": [0.0035748876837983497, 0.00090277317850291],
            "P_1040": [16.010000228881836, 0.17000000178813934],
        }
    )


# --- decimal places, with a deliberate floor --------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.00343887581965451, "0.00344"),
        (9.865960861963168e-5, "0.0001"),  # 5 dp, trailing zeros trimmed
        (16.010000228881836, "16.01"),
        (0.0, "0"),
        (123456.789, "123456.789"),
    ],
)
def test_format_value_is_plain_decimal_at_five_places(value, expected):
    assert _format_value(value, 5) == expected


def test_no_scientific_notation_anywhere():
    """The failure this exists to prevent: Excel showing 9.87E-05.

    `f"{v:.5g}"` -- the obvious implementation -- fails exactly here.

    "Anywhere" holds because the cap is on DECIMAL PLACES: a value too small to
    write out in five places is zero, not an exponent, so there is no magnitude
    at which the decimal form becomes unreadable.
    """
    for value in (9.865960861963168e-5, 0.007245, 1.0, 38.55, 123456.789):
        assert "e" not in _format_value(value, 5).lower()


def test_values_below_the_decimal_floor_read_as_zero():
    """Five decimal places puts a deliberate floor under floating-point noise.

    wflow's recharge spins up through ~1e-39. Under a significant-digit cap that
    rendered as `0.000000000000000000000000000000000000001054` -- 44 characters
    in a table whose purpose is readability. Owner ruling 2026-08-10: a value
    that small IS zero at any hydrological scale, so it prints as `0`.
    """
    for value in (1.054e-39, 3.2491e-42, 1e-7, -1e-9):
        assert _format_value(value, 5) == "0"


def test_the_floor_is_a_stated_trade_and_is_raisable():
    """The cost is real: 9.87e-5 rounds to 0.0001 rather than 0.00009866.

    Acceptable on a basin whose discharge runs 1e-5..1e-1, and recoverable by
    raising `decimals` -- which is why it is a parameter and not a constant.
    """
    assert _format_value(9.865960861963168e-5, 5) == "0.0001"
    assert _format_value(9.865960861963168e-5, 8) == "0.00009866"


# --- column grammar ----------------------------------------------------------


def test_columns_group_by_variable_and_sort_stations_numerically():
    """As TEXT '1010' sorts before '101'; station ids are integers."""
    grouped = split_columns(["time", "Q_101", "Q_1040", "Q_1010", "P_1040"])
    assert grouped == {"P": ["P_1040"], "Q": ["Q_101", "Q_1010", "Q_1040"]}


def test_non_station_columns_are_excluded():
    """`<var>_basavg` is per-subcatchment: it has no station id to key on."""
    assert split_columns(["time", "snow_basavg", "Q_101"]) == {"Q": ["Q_101"]}


# --- the derived tables ------------------------------------------------------


def test_one_table_per_variable_with_bare_sorted_ids():
    tables = tidy_tables(_frame())
    assert sorted(tables) == ["P", "Q"]
    assert list(tables["Q"].columns) == ["time", "101", "1010", "1040"]
    assert list(tables["P"].columns) == ["time", "1040"]


def test_stamps_are_date_only():
    """Date-only, so no locale can re-render it.

    `2000-01-02T00:00:00` imports as TEXT (the `T`), and the space-separated
    form this replaced imports as a DATETIME, which Excel re-renders per locale
    -- the same file read as `02-01-2000 00:00` elsewhere.
    """
    assert tidy_tables(_frame())["Q"]["time"].tolist() == [
        "2000-01-02",
        "2000-01-03",
    ]


def test_sub_daily_stamps_raise_rather_than_collapse():
    """Date-only would silently merge 24 hourly rows onto one date."""
    frame = pd.DataFrame(
        {
            "time": ["2000-01-02T00:00:00", "2000-01-02T06:00:00"],
            "Q_101": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="sub-daily"):
        tidy_tables(frame)


def test_no_row_is_dropped():
    frame = _frame()
    assert len(tidy_tables(frame)["Q"]) == len(frame)


def test_every_station_survives_the_split():
    """Losing a station silently would be the worst failure mode here."""
    frame = _frame()
    raw = {c.split("_")[1] for c in frame.columns if c.startswith("Q_")}
    assert set(tidy_tables(frame)["Q"].columns) - {"time"} == raw


# --- writing -----------------------------------------------------------------


def test_write_emits_one_file_per_variable(tmp_path):
    src = tmp_path / "output.csv"
    _frame().to_csv(src, index=False)

    written = write_tidy_tables(src, tmp_path / "out")
    assert [p.name for p in written] == ["output_p.csv", "output_q.csv"]

    q = pd.read_csv(written[1], dtype=str)
    assert list(q.columns) == ["time", "101", "1010", "1040"]
    assert q["101"].tolist() == ["0.00724", "0.0001"]


def test_write_leaves_the_source_untouched(tmp_path):
    """The raw csv is hydromt's to read; deriving must not perturb it."""
    src = tmp_path / "output.csv"
    _frame().to_csv(src, index=False)
    before = src.read_bytes()

    write_tidy_tables(src, tmp_path / "out")
    assert src.read_bytes() == before


def test_a_csv_without_station_columns_writes_nothing(tmp_path):
    src = tmp_path / "output.csv"
    pd.DataFrame({"time": ["2000-01-02T00:00:00"], "snow_basavg": [1.0]}).to_csv(
        src, index=False
    )
    assert write_tidy_tables(src, tmp_path / "out") == []


def test_a_variable_with_spaces_becomes_a_safe_filename(tmp_path):
    """`groundwater recharge` is a real wflow_outvars value; its columns arrive
    as `groundwater recharge_basavg_101`, which must not become a filename with
    a space in it."""
    src = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "time": ["2000-01-02T00:00:00"],
            "Q_101": [1.0],
            "groundwater recharge_basavg_101": [0.5],
        }
    ).to_csv(src, index=False)

    names = [p.name for p in write_tidy_tables(src, tmp_path / "out")]
    assert names == ["output_groundwater_recharge_basavg.csv", "output_q.csv"]
    assert all(" " not in n for n in names)


def test_slugify_collapses_runs_and_trims():
    assert slugify(
        "groundwater recharge_basavg"
    ) == "groundwater_recharge_basavg".replace(" ", "_")
    assert slugify("Q") == "q"


def test_no_cell_is_ever_absurdly_wide():
    """The concrete regression: a 48-character cell in a readability table."""
    for value in (1.054e-39, 3.2491e-42, 5.3913e-36, 1e-300):
        assert len(_format_value(value, 5)) <= 8


def test_ordinary_hydrological_values_stay_decimal():
    """The floor must not swallow the range these files actually carry.

    Measured on a real run: discharge ~1e-3..1e-1, recharge -5.1..38.6.
    """
    for value in (0.007245, 38.55, -5.098, 0.029752):
        rendered = _format_value(value, 5)
        assert "e" not in rendered.lower()
        assert rendered != "0"


def test_stale_tables_are_removed(tmp_path):
    """The set on disk is a function of the CURRENT config, not cumulative."""
    src = tmp_path / "output.csv"
    out = tmp_path / "out"
    _frame().to_csv(src, index=False)
    write_tidy_tables(src, out)

    orphan = out / "output_retired_name.csv"
    orphan.write_text("time,101\n2000-01-02 00:00:00,1\n", encoding="utf-8")
    write_tidy_tables(src, out)
    assert not orphan.exists(), "a table for a dropped variable must not survive"
    assert (out / "output_q.csv").exists()


def test_the_raw_wflow_csv_is_never_removed_as_stale(tmp_path):
    """`output.csv` is wflow's own and hydromt reads it; the glob must miss it."""
    out = tmp_path / "out"
    out.mkdir()
    src = out / "output.csv"
    _frame().to_csv(src, index=False)

    write_tidy_tables(src, out)
    assert src.exists(), "the raw csv was deleted as a stale derived table"
