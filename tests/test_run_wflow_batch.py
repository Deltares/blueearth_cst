"""The batched Wflow driver's console contract (rule 3.15).

``blueearth_cst/experiment/run_wflow_batch.jl`` cannot be unit-tested the way a
Python module can — running it needs Julia, an instantiated project and Wflow
itself, which is what ``--run-integration`` is for. What CAN be checked cheaply
is the part the console depends on, and the part that silently rots: the SHAPE
of the rows it prints.

That shape is load-bearing rather than cosmetic. Rule 3.15 is the toolbox's
longest step, and Wflow is silent on the terminal there (``[logging] silent =
true``, set so Julia's box-drawing records do not swamp the console), so these
rows are the only thing a batch emits while it works. They are also the only
rows in the toolbox written in Julia, so nothing else in the suite would notice
them drifting out of the house format.
"""

from __future__ import annotations

import re
from pathlib import Path

from blueearth_cst.shared.snake_utils import _ANSI_FAIL, _severity_code

DRIVER = (
    Path(__file__).resolve().parents[1]
    / "blueearth_cst"
    / "experiment"
    / "run_wflow_batch.jl"
)
SOURCE = DRIVER.read_text(encoding="utf-8")

#: The rows the driver prints, as they reach the tee. Kept beside the assertions
#: rather than derived from the source: a test that re-derived the format from
#: the file it is checking would pass on any format.
OK_ROW = "08:03:16 - wflow - [1/3] rlz_1_st_0  0.2 s"
FAIL_ROW = "08:03:16 - wflow - FAILED [2/3] rlz_1_st_2  boom: forcing not found"


def _code_lines():
    """The driver's source without comment-only lines."""
    return [line for line in SOURCE.splitlines() if not line.lstrip().startswith("#")]


def test_a_failure_row_is_one_the_console_paints_red():
    """The one coupling worth pinning: the Julia spelling and the Python matcher.

    ``_SEVERITY_PATTERNS`` decides a row's colour by reading its text, so the
    driver's failure word has to be one that matcher knows. It was ``FAIL``
    before the matcher existed, and a batch failure scrolling past in body-tier
    grey is exactly the row that must not.
    """
    assert _severity_code(FAIL_ROW) == _ANSI_FAIL


def test_a_success_row_is_not_painted_as_a_failure():
    """`rlz_1_st_0` and a duration must not trip the severity matcher."""
    assert _severity_code(OK_ROW) is None


def test_both_rows_are_written_in_the_house_format():
    """``HH:MM:SS - <module> - <message>``, the shape every other rule emits.

    Checked on the ``row`` helper's definition, which is the single place the
    format is spelled.
    """
    assert re.search(
        r'row\(body\)\s*=\s*println\(\s*"\$\(Dates\.format\(now\(\),\s*"HH:MM:SS"\)\)'
        r'\s*-\s*wflow\s*-\s*\$\(body\)"\s*\)',
        SOURCE,
    ), SOURCE


def test_status_rows_go_through_the_one_helper():
    """So the OK and FAIL rows cannot drift apart in timestamp or counter.

    Any ``println`` outside the helper's own definition is a second spelling of
    the row format, which is how the two rows came to differ in the first place.
    """
    stray = [
        line
        for line in _code_lines()
        if "println(" in line and not line.lstrip().startswith("row(body)")
    ]
    assert stray == [], stray


def test_every_status_row_carries_its_position_in_the_batch():
    """A row saying a member finished, without saying how much is left, is the
    gap this driver had: the batch is opaque until its last member returns."""
    rows = [line for line in _code_lines() if "row(" in line and "=" not in line]
    assert rows, SOURCE
    for line in rows:
        assert "[$(k)/$(total)]" in line, line


def test_the_total_is_this_batch_and_not_the_experiment():
    """Rule 3.15 runs several batches concurrently, each its own process, so a
    run-wide denominator is not knowable here and would be a lie if printed."""
    assert re.search(r"^total\s*=\s*length\(ARGS\)\s*$", SOURCE, re.MULTILINE)


def test_the_member_tag_is_the_toml_stem():
    """`<exp>/hydrology/wflow/config/rlz_<i>_st_<j>.toml` -> `rlz_<i>_st_<j>`.

    The tag used to prepend the toml's grandparent directory, which that layout
    made meaningful and the current one resolves to the constant ``wflow`` on
    every row of every batch.
    """
    assert "tag = first(splitext(basename(t)))" in SOURCE
    assert "dirname(dirname(t))" not in SOURCE
