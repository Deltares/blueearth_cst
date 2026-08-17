"""The shipped R is syntax-checked, because nothing else checked it at all.

Three R files are executed by Snakemake `shell:` bodies
(`blueearth_cst/weathergen/`), and until 2026-08-13 **no test touched them**.
Three R-changing commits landed that day past a green 2105-test suite; a
syntax error in any of them would have surfaced only when a rule ran, minutes
into WF3, as a bare non-zero exit.

**What this does and does not buy.** `parse()` catches syntax — an unbalanced
brace, a mangled string, a stray edit. It does NOT execute anything, so it
cannot catch a wrong argument, a mis-set variable, or the grid/data mismatch
that broke rule 3.11. Behavioural coverage needs `weathergenr`, which CI
deliberately does not install (`.github/workflows/ci.yml`: no
`pixi run install`), so it cannot live here. That gap is tracked on the board,
not papered over with a test that implies more than it checks.

The parse is worth having anyway: it is the cheapest possible gate over the
only code in the repository with zero automated coverage, it costs under a
second per file, and it runs on BOTH CI legs because `r-base` is in the pixi
environment itself.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The R Snakemake actually runs. `dev/scripts/*.R` is deliberately excluded:
#: it is never part of a run, so a repo-meta change there should not red a
#: pipeline gate.
R_SCRIPTS = sorted((REPO / "blueearth_cst").rglob("*.R"))

_NO_RSCRIPT = "Rscript not on PATH (r-base is in the pixi env; run inside it)"


def test_the_shipped_r_set_is_not_empty():
    """A glob that silently matches nothing would make every case below vacuous.

    This is the failure mode the R files are most exposed to: they moved once
    already (`src/weathergen/` -> `blueearth_cst/weathergen/`), and a stale
    glob after the next move would leave the suite green over an unchecked
    directory.
    """
    assert R_SCRIPTS, f"no .R files found under {REPO / 'blueearth_cst'}"
    names = {p.name for p in R_SCRIPTS}
    assert {"generate_weather.R", "impose_climate_change.R"} <= names, names


@pytest.mark.skipif(shutil.which("Rscript") is None, reason=_NO_RSCRIPT)
@pytest.mark.parametrize("script", R_SCRIPTS, ids=lambda p: p.name)
def test_r_script_parses(script: Path):
    """`Rscript -e parse(...)` — syntax only, no evaluation, no side effects."""
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", f'invisible(parse("{script.as_posix()}"))'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script.relative_to(REPO).as_posix()} does not parse:\n{result.stderr}"
    )
