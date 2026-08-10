"""Tests for the derived per-variable wflow tables (blueearth_cst/shared/tidy_wflow_table.py).

The raw ``output.csv`` is an interface with hydromt's reader, so these tables are
DERIVED. What matters is that the derivation is lossless where it claims to be
(station identity, row count, ordering) and lossy only where it intends to be
(significant digits).
"""

import pandas as pd
import pytest

from blueearth_cst.shared.tidy_wflow_table import (
    _format_value,
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


# --- significant digits, not decimal places ---------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.00343887581965451, "0.0034389"),
        # 5 significant digits, then trailing zeros dropped: the zero in
        # 0.000098660 carries no information and only widens the column.
        (9.865960861963168e-5, "0.00009866"),  # never e-notation
        (16.010000228881836, "16.01"),
        (0.0, "0"),
        (123456.789, "123460"),
    ],
)
def test_format_value_is_plain_decimal_at_five_significant_digits(value, expected):
    assert _format_value(value, 5) == expected


def test_format_value_never_emits_scientific_notation():
    """The failure this exists to prevent: Excel showing 9.87E-05.

    `f"{v:.5g}"` -- the obvious implementation -- fails exactly here, which is
    why the module formats through a Decimal instead.
    """
    for value in (1e-12, 9.865960861963168e-5, 1e20):
        assert "e" not in _format_value(value, 5).lower()


def test_small_values_survive_rounding():
    """A decimal-places round would destroy these; a significant-digit cap must not."""
    assert float(_format_value(9.865960861963168e-5, 5)) == pytest.approx(
        9.866e-5, rel=1e-4
    )


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


def test_timestamps_are_space_separated_for_excel():
    """Excel imports '2000-01-02T00:00:00' as TEXT because of the T."""
    assert tidy_tables(_frame())["Q"]["time"].tolist() == [
        "2000-01-02 00:00:00",
        "2000-01-03 00:00:00",
    ]


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
    assert q["101"].tolist() == ["0.007245", "0.00009866"]


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
