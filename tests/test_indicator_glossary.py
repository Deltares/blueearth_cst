# -*- coding: utf-8 -*-
"""`dev/reference/indicator-glossary.md` is checked against the code, not trusted.

The glossary is a fifth hand-maintained copy of the variable vocabulary, and this
repo already knows what happens to those: 8bd51de changed the csv header scheme
while the reducer's matcher kept the retired spelling, and two indicator tables
were written empty with every rule green. A prose table with no test drifts the
same way, only more quietly -- nothing runs it.

So the glossary declares itself DERIVED, and this module is what makes that true:
it parses the markdown tables and asserts they agree, row for row, with the dicts
that own each column. A failure here means the GLOSSARY is stale -- fix the
document, not the code.

Deliberately parsing the rendered markdown rather than a data file the document
is generated from: a generated glossary would be one more artifact to keep in the
tree, and the document is short enough that a table parser is cheaper than a
build step. The parser is strict about the pipe/backtick shape for that reason --
a row it cannot read fails loudly rather than being skipped.
"""

import re
from pathlib import Path

import pytest

from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS
from blueearth_cst.shared.indicator_tables import (
    BASIN_METRIC_SUFFIXES,
    Q_METRIC_SUFFIXES,
    VARIABLE_TOKENS,
    basin_metric_name,
    indicator_table_filename,
)
from blueearth_cst.shared.wflow_outputs import CODES, PLOT_META

GLOSSARY = (
    Path(__file__).resolve().parents[1] / "dev" / "reference" / "indicator-glossary.md"
)


def _text() -> str:
    return GLOSSARY.read_text(encoding="utf-8")


def _rows(text: str, first_cell: str) -> list[list[str]]:
    """Every row of the markdown table whose first data cell is ``first_cell``.

    Located by content rather than by heading or ordinal, so inserting a section
    above a table does not silently start checking a different one.
    """
    table: list[list[str]] = []
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if found:
                break
            table = []
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):  # the header separator
            continue
        table.append(cells)
        if cells[0] == first_cell:
            found = True
    assert found, f"no glossary table row starts with {first_cell!r}"
    return table


def _code(cell: str) -> str:
    """The content of a single-backtick cell, refusing anything else."""
    match = re.fullmatch(r"`([^`]+)`", cell)
    assert match, f"glossary cell {cell!r} is not a single `code` span"
    return match.group(1)


# --- §1, the variable table ---------------------------------------------------


def _variable_rows() -> dict[str, list[str]]:
    rows = _rows(_text(), "`river discharge`")
    header, *body = rows
    assert header[0] == "`wflow_outvars` label"
    assert header[1:5] == ["CSDMS name", "csv code", "token", "table"]
    return {_code(r[0]): r for r in body}


def test_the_variable_table_lists_exactly_the_configurable_variables():
    """Neither a variable the code cannot emit nor one it can but the glossary
    omits. The omission is the dangerous direction: a reader concludes the
    vocabulary is closed when it is not."""
    assert set(_variable_rows()) == set(WFLOW_VARS) == set(VARIABLE_TOKENS)


@pytest.mark.parametrize("label", sorted(WFLOW_VARS))
def test_each_row_agrees_with_the_dict_that_owns_each_column(label):
    row = _variable_rows()[label]
    assert _code(row[1]) == WFLOW_VARS[label], "CSDMS name (setup_gauges_and_outputs)"
    assert _code(row[3]) == VARIABLE_TOKENS[label], "token (indicator_tables)"
    assert _code(row[4]) == indicator_table_filename(label), "table filename"


@pytest.mark.parametrize("label", sorted(CODES))
def test_the_csv_code_column_agrees_with_wflow_outputs(label):
    """Discharge is excluded: it carries the fixed `Q` header and is deliberately
    absent from `CODES`, which the glossary says in prose."""
    assert _code(_variable_rows()[label][2]) == CODES[label]


def test_discharge_is_the_one_row_whose_code_is_not_in_codes():
    assert "river discharge" not in CODES
    assert _code(_variable_rows()["river discharge"][2]) == "Q"


@pytest.mark.parametrize("label", sorted(set(WFLOW_VARS) - {"river discharge"}))
def test_the_basin_scalar_metric_column_is_the_composed_name(label):
    token = VARIABLE_TOKENS[label]
    assert _code(_variable_rows()[label][5]) == basin_metric_name(token)


# --- §1's figure table, keyed by CODE not by label ----------------------------


def test_the_legend_table_agrees_with_plot_meta():
    """Keyed by code, because a figure sees only what the csv carried. Only the
    resample rule is compared -- the legend strings carry LaTeX in the code and
    Unicode in the document, and pinning that spelling would make the glossary
    unreadable to buy nothing."""
    rows = _rows(_text(), "`p`")
    header, *body = rows
    assert header == ["code", "resample", "legend"]
    documented = {_code(r[0]): _code(r[1]) for r in body}
    assert documented == {code: meta["resample"] for code, meta in PLOT_META.items()}


# --- §2, the metric vocabulary ------------------------------------------------


def test_every_discharge_metric_is_documented_with_its_grain():
    """Including the two whose names interpolate `Tpeak`/`Tlow` -- the glossary
    carries the braced form, which is what a reader needs in order to recognise
    `q_return_level_50yr_max` as the same metric."""
    rows = _rows(_text(), "`q_annual_mean`")
    header, *body = rows
    assert header == ["metric", "statistic", "grain"]
    documented = {_code(r[0]): r[2] for r in body}
    expected = {
        f"q_{suffix}": ("per-realization" if cls == "A" else "pooled (`rlz_id = 0`)")
        for suffix, cls in Q_METRIC_SUFFIXES.values()
    }
    assert documented == expected


def test_the_metric_column_of_the_variable_table_covers_every_basin_scalar():
    """The §1 metric column and `BASIN_METRIC_SUFFIXES` are the same set, so a
    new basin-scalar variable cannot land with no documented metric."""
    documented = {
        _code(row[5])
        for label, row in _variable_rows().items()
        if label != "river discharge"
    }
    assert documented == {basin_metric_name(t) for t in BASIN_METRIC_SUFFIXES}


# --- the claims the glossary makes about itself -------------------------------


def test_the_glossary_names_the_dicts_it_is_derived_from():
    """It calls itself derived; the pointers must resolve, or a reader chasing a
    disagreement has nowhere to go."""
    text = _text()
    for owner in (
        "setup_gauges_and_outputs.py::WFLOW_VARS",
        "wflow_outputs.py::CODES",
        "indicator_tables.py::VARIABLE_TOKENS",
        "indicator_tables.py::Q_METRIC_SUFFIXES",
        "indicator_tables.py::BASIN_METRIC_SUFFIXES",
        "indicator_tables.py::INDICATOR_COLUMNS",
        "wflow_outputs.py::PLOT_META",
    ):
        assert owner in text, f"glossary does not name {owner}"
    assert "tests/test_indicator_glossary.py" in text
