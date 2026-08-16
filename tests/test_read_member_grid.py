"""V17 — the R side refuses a malformed member slice, loudly and BY MEMBER.

These are the falsifier for D29's postcondition, and they exist as their own
file for a reason worth stating: `tests/test_r_scripts.py` declares itself
syntax-only ("`Rscript -e parse(...)` — syntax only, no evaluation, no side
effects"), so quietly evaluating R inside it would make its own scope clause
false.

They are cheap because `read_member_grid.R` sources NOTHING — not even
`global.R` — so each case is one `Rscript -e 'source(...)'` needing neither
`weathergenr` nor a netCDF. Before the extraction (D34) this guard had no
reachable falsifier at all: the only proposed check was a WF3 run on a VALID
config, which is green whether the guard exists or not.

cwd is pinned to the repo root because this repo's R scripts are sourced by
repo-relative path.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "blueearth_cst" / "weathergen" / "read_member_grid.R"

_NO_RSCRIPT = "Rscript not on PATH (run inside `pixi shell`)"

HEADER = "st_id,month,temp_change,precip_change,precip_variance_change"


def _rows(st_id, months):
    return [f"{st_id},{m},1.5,-30.0,0.0" for m in months]


def _lookup(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return path


def _run(lookup_path, token):
    """Source the script, call it, and print the frame's shape."""
    r = (
        f'source("{SCRIPT.as_posix()}"); '
        f'g <- read_member_grid("{lookup_path.as_posix()}", "{token}"); '
        f'cat(nrow(g), paste(g$month, collapse="-"), g$precip_change[[1]])'
    )
    return subprocess.run(
        ["Rscript", "--vanilla", "-e", r],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )


def _four_member_lookup(tmp_path, name="lookup.csv"):
    lines = []
    for st_id in ("1", "2", "3", "4"):
        lines += _rows(st_id, range(1, 13))
    return _lookup(tmp_path, name, lines)


# ---------------------------------------------------------------------------
# The three negatives — each must exit NONZERO and name the member
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_missing_month_refused(tmp_path):
    """Eleven rows is a SHORT vector, which R would recycle into a wrong answer
    rather than an error — the precise hazard the postcondition exists for."""
    lines = _rows("1", [m for m in range(1, 13) if m != 7]) + _rows("2", range(1, 13))
    path = _lookup(tmp_path, "missing_month.csv", lines)

    result = _run(path, "1")
    assert result.returncode != 0, result.stdout
    assert "'1'" in result.stderr, result.stderr


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_duplicated_month_refused(tmp_path):
    """Thirteen rows is the other side of the same failure: a join that matched
    too much. `nrow == 12` alone would not catch it if a month were also
    missing, which is why the month VECTOR is compared, not just the count."""
    lines = _rows("1", list(range(1, 13)) + [7])
    path = _lookup(tmp_path, "duplicate_month.csv", lines)

    result = _run(path, "1")
    assert result.returncode != 0, result.stdout
    assert "'1'" in result.stderr, result.stderr


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_token_matching_no_row_refused(tmp_path):
    """A token outside the grid yields a ZERO-length vector, which is the
    failure the migration newly makes possible: before the lookup, Snakemake
    guaranteed the member file's existence with a structural
    MissingInputException."""
    path = _four_member_lookup(tmp_path)

    result = _run(path, "9")
    assert result.returncode != 0, result.stdout
    assert "'9'" in result.stderr, result.stderr
    assert "9" in result.stderr and "0 row" in result.stderr, result.stderr


# ---------------------------------------------------------------------------
# The two positives — a well-formed slice, and an UNORDERED one that must
# normalise rather than raise
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_well_formed_member_returns_twelve_ordered_months(tmp_path):
    path = _four_member_lookup(tmp_path)

    result = _run(path, "3")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("12 1-2-3-4-5-6-7-8-9-10-11-12"), result.stdout


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_unordered_months_are_normalised_not_rejected(tmp_path):
    """D21 sorts BEFORE it asserts, so unordered input is normalised.

    A distinct file from the positive above, deliberately: the two claims are
    "a good member is accepted" and "ordering is this function's job, not the
    caller's", and collapsing them would let a sort-then-assert regression hide
    behind the first.
    """
    shuffled = [1, 5, 12, 3, 7, 2, 11, 4, 9, 6, 10, 8]
    lines = _rows("2", shuffled) + _rows("1", range(1, 13))
    path = _lookup(tmp_path, "shuffled.csv", lines)

    result = _run(path, "2")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("12 1-2-3-4-5-6-7-8-9-10-11-12"), result.stdout


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_zero_padded_token_is_matched_as_text(tmp_path):
    """WG-2's dtype clause, from the consuming side: `01` must not become `1`.

    Read with inferred types, `st_id` comes back numeric and the comparison
    against the padded token matches nothing — which under this encoding
    presents as a missing member rather than as a type error.
    """
    lines = []
    for st_id in (f"{m:02d}" for m in range(1, 13)):
        lines += _rows(st_id, range(1, 13))
    path = _lookup(tmp_path, "padded.csv", lines)

    result = _run(path, "01")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("12 "), result.stdout
