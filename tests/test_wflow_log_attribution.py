"""R9 P2 falsifier: no member's Wflow log is overwritten by another's.

The claim this asserts is an ABSENCE, and the master brief lists it among the
three R9 failure modes the ordinary suite cannot reach. It needs a real run,
because the property is about what several concurrently-batched Wflow processes
do to one another's files.

**Counting files is not the test.** Twelve logs exist the moment `path_log` is
keyed per member, whatever ends up inside them; a count passes trivially and
proves nothing about interleaving. What discriminates is CONTENT ATTRIBUTION:
each log must describe its own ``rlz_<r>_cst_<c>`` and no other. A log that had
been written by two members would carry both names.

**The measured failure mode is OVERWRITING, not interleaving** (R9 landing gate,
2026-08-05). With ``path_log`` removed and WF3 re-run at ``-c 3``, twelve
members produced ZERO per-member logs and one shared ``log.txt`` naming exactly
ONE member -- whichever finished last. Each wflow process opens the default path
and truncates it, so eleven members' logs were not merged, they were destroyed.
Attribution still catches it, but through absence rather than through a file
carrying two names; ``_member_logs`` below is what makes that absence a failure
instead of a skip.

Why the race is real rather than theoretical: Wflow's ``[logging] path_log``
defaults to ``log.txt`` beside the TOML. Before R9 each realization owned a run
directory, so that default was already ONE SHARED LOG PER REALIZATION -- the P1
observed tier measured exactly two logs for twelve members, six writers each.
R9 P2 dissolves the ``rlz_<r>/`` level, which would put all twelve on one path,
and rule 3.10 batches members concurrently. The directory flattening and the
per-member ``path_log`` are therefore one correctness unit and ship together.

Skips unless a materialized post-migration run is present. Point
``$R09_P2_RUN_DIR`` at one, or the default location the phase used.
"""

import os
import re
from pathlib import Path

import pytest

MEMBER = re.compile(r"rlz_(\d+)_cst_(\d+)")

DEFAULT_RUN_DIR = Path("C:/Users/taner/workspace/.cst_runs/r09_p2_post")


def _output_dir() -> Path:
    root = Path(os.environ.get("R09_P2_RUN_DIR", DEFAULT_RUN_DIR))
    return root / "experiments" / "experiment" / "hydrology" / "wflow" / "output"


def _member_logs() -> list[Path]:
    """The member logs, or a skip if no run is present — but NEVER a skip when a
    run IS present and its logs are missing.

    That distinction was missing until the R9 landing gate, and it made this
    module skip itself in the one condition it exists to catch. Demonstrated:
    `path_log` was removed and WF3 re-run, and the two attribution tests
    SKIPPED — twelve members had collapsed onto one `log.txt`, so there were no
    per-member logs to attribute, so `if not logs: skip` fired. Only the
    module's own self-described "weakest" assertion caught the defect. Delete
    that one and the module would have gone green over an empty set.

    A run is present when the member CSVs are: they are declared outputs of rule
    3.10 and are not temp(), so they persist. If they exist and the logs do not,
    every member's log was overwritten — which IS the finding, not a reason to
    skip.
    """
    out = _output_dir()
    if not out.is_dir():
        pytest.skip(f"no post-migration run at {out}; set $R09_P2_RUN_DIR")
    logs = sorted(out.glob("rlz_*_cst_*.log"))
    if not logs:
        members = sorted(out.glob("rlz_*_cst_*.csv"))
        if not members:
            pytest.skip(f"no post-migration run at {out}; set $R09_P2_RUN_DIR")
        raise AssertionError(
            f"{len(members)} member(s) ran and produced NO per-member log under "
            f"{out}. Every member's log was written to wflow's default shared "
            f"path and overwritten by the next writer -- the exact condition "
            f"`logging.path_log` exists to prevent."
        )
    return logs


@pytest.mark.slow
def test_every_member_log_describes_only_its_own_member():
    """THE falsifier. Each log names its own member and no other.

    Reads the member from the FILENAME and requires every `rlz_<r>_cst_<c>`
    token in the body to match it. Two members sharing a file would leave the
    other's name behind, which is precisely what this catches and what a file
    count does not.
    """
    offenders = {}
    for log in _member_logs():
        own = MEMBER.search(log.stem)
        assert own, f"unexpected log name: {log.name}"
        found = {
            m.group(0)
            for m in MEMBER.finditer(log.read_text(encoding="utf-8", errors="replace"))
        }
        foreign = found - {own.group(0)}
        if foreign:
            offenders[log.name] = sorted(foreign)
    assert not offenders, (
        "a member's log names another member -- concurrent members are sharing "
        f"a log file: {offenders}"
    )


@pytest.mark.slow
def test_each_member_log_actually_identifies_itself():
    """Guard on the falsifier: an EMPTY log would pass the test above vacuously.

    If `path_log` were keyed per member but Wflow wrote nothing to it, the set
    of foreign names would be empty and the check would go green while proving
    nothing. So each log must positively name its own member at least once.
    """
    silent = [
        log.name
        for log in _member_logs()
        if MEMBER.search(log.stem).group(0)
        not in log.read_text(encoding="utf-8", errors="replace")
    ]
    assert not silent, (
        f"logs that never name their own member, so attribution is untestable "
        f"for them: {silent}"
    )


@pytest.mark.slow
def test_the_wflow_default_shared_log_is_gone():
    """The weakest of the three, kept because its ABSENCE is the visible symptom.

    `log.txt` beside the TOML is what the pre-fix layout produced. It must not
    exist anywhere under the experiment -- but on its own this proves little,
    which is why it is listed last and why the two tests above exist.
    """
    root = _output_dir().parents[2]
    strays = sorted(p.as_posix() for p in root.rglob("log.txt"))
    assert not strays, f"wflow's default shared log reappeared: {strays}"
