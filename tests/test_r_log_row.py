"""The R side emits the toolbox's log grammar, not a spelling of its own.

`blueearth_cst/weathergen/global.R` carries the R counterpart of
`snake_utils.log_row`, and the two must agree on one shape --
`HH:MM:SS - <module> - <message>`, level shown only when it is not INFO --
because the merged rule log is read as four fixed fields. A drift here is
invisible to every other gate: `tests/test_r_scripts.py` is syntax-only by
declaration, no rule consumes a log line, and a WF3 run is green whatever the
rows say.

Cheap for the same reason `test_read_member_grid.py` is: `global.R` sources
nothing and needs neither `weathergenr` nor a netCDF, so each case is one
`Rscript -e`. Rows go to stderr (the function uses `message()`, like the calls
it replaced), so that is where they are read from.

cwd is pinned to the repo root because this repo's R scripts are sourced by
repo-relative path.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "blueearth_cst" / "weathergen" / "global.R"

_NO_RSCRIPT = "Rscript not on PATH (r-base is in the pixi env; run inside it)"

#: One row, anchored: stamp, module, then the message -- no level field.
INFO_ROW = re.compile(r"^\d\d:\d\d:\d\d - weathergen - (?P<message>.*)$")

#: A line that IS one of our rows. `message()` shares stderr with everything
#: else R writes there -- on a leg with an unset locale it opens with
#: `Setting LC_CTYPE failed` / `During startup - Warning message:` -- so the
#: cases below count rows, not lines. Filtering on blankness alone would make
#: every `len(rows) == n` assertion fail on one CI leg only, which is exactly
#: the class AGENTS.md records as ten days of red ubuntu behind a green local
#: suite. A genuinely malformed row is still caught: it fails the unpack or the
#: `fullmatch`, rather than being silently tolerated.
ROW = re.compile(r"^\d\d:\d\d:\d\d - ")


def _emit(calls, env=None):
    """Source `global.R`, run `calls`, and return the rows it wrote to stderr."""
    child_env = dict(os.environ)
    child_env.pop("CST_LOG_LEVEL", None)
    child_env.update(env or {})
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", f'source("{SCRIPT.as_posix()}"); {calls}'],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env=child_env,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stderr.splitlines() if ROW.match(line)]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_info_row_carries_no_level_field():
    """The INFO case is 259 of 272 rows on a WF1 build; the field is omitted."""
    (row,) = _emit('log_row("Reading weather netcdf: x.nc")')
    match = INFO_ROW.match(row)
    assert match, row
    assert match.group("message") == "Reading weather netcdf: x.nc"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_positional_parts_concatenate_and_cannot_bind_module():
    """`...` comes FIRST so a multi-part call cannot silently mangle a row.

    With `log_row(message, module, level)` the second positional argument would
    bind to `module` -- no error, just `12:00:00 - 2 - ...`. R matches arguments
    after `...` by exact name only, which is what forecloses it, so the shape of
    the signature is the guard and this is its falsifier.
    """
    (row,) = _emit('log_row("Resampling on ", 2, " basin cell(s) of ", 20)')
    match = INFO_ROW.match(row)
    assert match, row
    assert match.group("message") == "Resampling on 2 basin cell(s) of 20"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_non_info_level_is_shown_uppercased():
    (row,) = _emit('log_row("state file not found", level = "warning")')
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d - weathergen - WARNING - state file not found", row
    ), row


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_module_column_is_overridable():
    (row,) = _emit('log_row("perturbing", module = "change")')
    assert re.fullmatch(r"\d\d:\d\d:\d\d - change - perturbing", row), row


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_unset_level_emits_everything():
    """Quiet mode is opt-in: nothing changes until `CST_LOG_LEVEL` is set."""
    rows = _emit('log_row("a"); log_row("b", level = "DEBUG")')
    assert len(rows) == 2, rows


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_floor_drops_rows_below_it():
    """`CST_LOG_LEVEL=WARNING` must silence the R rows too.

    Honouring the floor on one side only would make quiet mode partial -- the
    Python rows gone, the R ones talking -- with nothing reporting it.
    """
    rows = _emit(
        'log_row("chatter"); log_row("state file not found", level = "WARNING")',
        env={"CST_LOG_LEVEL": "WARNING"},
    )
    assert len(rows) == 1, rows
    assert rows[0].endswith("WARNING - state file not found"), rows


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
def test_unrecognized_level_is_never_suppressed():
    """A level the rank table does not know keeps printing, floor or no floor.

    A filter that swallowed an unfamiliar level would hide exactly the unusual
    row worth seeing -- the same property `snake_utils.log_row` documents.
    """
    rows = _emit(
        'log_row("odd one", level = "NOTICE")', env={"CST_LOG_LEVEL": "CRITICAL"}
    )
    assert len(rows) == 1, rows
    assert rows[0].endswith("NOTICE - odd one"), rows
