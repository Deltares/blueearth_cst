"""`dev/scripts/sweep_common.py` — the contract the repository sweeps share.

`sweep_unread_config_keys` imports `Allowance`, `classify`, `Floor` and
`check_floors` from here, and
`dev/scripts/` libraries are contract surfaces with test consumers: an import
error or a behaviour change here fails CI on a bare checkout, on both legs.

Each sweep's own tests exercise ONE path through this file. The boundaries —
where a floor starts failing, what happens to a path outside the repository —
are only ever hit incidentally there, which is how a shared primitive drifts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dev.scripts.sweep_common import (
    EXCLUDED_DIRS,
    REPO_ROOT,
    Allowance,
    Floor,
    check_floors,
    classify,
    iter_lines,
)

FLOOR = Floor(
    name="thing",
    minimum=10,
    count=40,
    measured="a test",
    why="because the test says so",
)


# --- check_floors, at its boundary -------------------------------------------


@pytest.mark.parametrize(
    "seen,expected_failure", [(11, False), (10, False), (9, True), (0, True)]
)
def test_a_floor_fails_below_its_minimum_and_not_at_it(seen, expected_failure):
    """`minimum` is inclusive. Off by one here and every sweep's floor is one
    louder or one quieter than its comment claims."""
    assert bool(check_floors([FLOOR], {"thing": seen})) is expected_failure


def test_a_missing_observation_counts_as_zero_not_as_absent():
    """A form that stopped running yields no key at all, and treating that as
    "nothing to check" is precisely the fail-open shape these floors exist to
    prevent."""
    failures = check_floors([FLOOR], {})
    assert len(failures) == 1
    assert "saw 0" in failures[0]


def test_a_failure_message_carries_the_number_the_floor_was_measured_at():
    """A floor that starts failing has to be tellable from a floor that was
    never true, which takes the count and when it was taken."""
    message = check_floors([FLOOR], {"thing": 1})[0]
    assert "40 when measured a test" in message
    assert "because the test says so" in message


# --- Allowance.covers --------------------------------------------------------


def test_a_path_outside_the_repository_matches_no_path_allowance():
    """Every sweep is run against throwaway trees by its tests and, in one
    case, by its own specimen check. Before this fallback existed those raised
    `ValueError` out of `relative_to` — and the reason a REPOSITORY file is
    excused must not travel to a fixture that merely shares its name."""
    allowance = Allowance(name="x", reason="y" * 50, paths=("dev/scripts/tool.py",))
    outside = REPO_ROOT.parent / "sweep-external-fixture" / "dev/scripts/tool.py"
    assert not outside.is_relative_to(REPO_ROOT)
    assert allowance.covers(REPO_ROOT / "dev/scripts/tool.py", "any line")
    assert not allowance.covers(outside, "any line")


def test_a_line_pattern_still_applies_to_an_outside_path():
    """The fallback narrows PATH matching only. A line-shaped reason is a
    property of the text, so it holds wherever the text is."""
    allowance = Allowance(name="x", reason="y" * 50, line_patterns=(r"^ok$",))
    outside = REPO_ROOT.parent / "sweep-external-fixture" / "anything.py"
    assert not outside.is_relative_to(REPO_ROOT)
    assert allowance.covers(outside, "ok")


def test_a_glob_allowance_matches_by_repository_relative_path():
    allowance = Allowance(name="x", reason="y" * 50, path_globs=("dev/scripts/*.py",))
    assert allowance.covers(REPO_ROOT / "dev/scripts/tool.py", "line")
    assert not allowance.covers(REPO_ROOT / "tests/tool.py", "line")


# --- classify ----------------------------------------------------------------


def test_the_first_covering_allowance_is_the_reason_recorded():
    """A sweep's ALLOWANCES order is part of its explanation, so the ordering
    has to be load-bearing rather than incidental."""
    first = Allowance(name="first", reason="a" * 50, line_patterns=(r"x",))
    second = Allowance(name="second", reason="b" * 50, line_patterns=(r"x",))
    _defects, allowed = classify([(REPO_ROOT / "f.py", 1, "x")], (first, second))
    assert [a[3] for a in allowed] == ["first"]


def test_an_uncovered_hit_is_a_defect_never_a_warning():
    defects, allowed = classify([(REPO_ROOT / "f.py", 1, "x")], ())
    assert allowed == []
    assert [d[3] for d in defects] == ["UNCLASSIFIED"]


# --- iter_lines --------------------------------------------------------------


def test_a_file_matched_by_two_globs_is_visited_once(tmp_path, monkeypatch):
    """A sweep's SEARCHED tuple overlaps on purpose — `config/migrations/*.yml`
    sits inside `config/**/*.yml` — and double-reporting one line as two
    findings would be a defect count nobody can act on."""
    (tmp_path / "a.py").write_text("one\ntwo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    lines = list(iter_lines(("*.py", "a.py"), Path(".")))
    assert [text for _p, _n, text in lines] == ["one", "two"]


@pytest.mark.parametrize("excluded", EXCLUDED_DIRS)
def test_the_excluded_directories_are_never_walked(tmp_path, excluded, monkeypatch):
    """Generated output, the environment and scratch. `.tmp` is the one that
    bites: a fixture written there is invisible to every sweep."""
    target = tmp_path / excluded / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("x\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("visible\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert list(iter_lines(("**/*.py",), Path("."))) == [
        (Path("visible.py"), 1, "visible")
    ]


def test_an_unreadable_file_is_skipped_rather_than_ending_the_walk(
    tmp_path, monkeypatch
):
    """A sweep that stopped at the first undecodable byte would go quiet on
    everything after it, which is a smaller version of failing open."""
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00binary")
    (tmp_path / "good.py").write_text("ok\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    seen = [Path(p).name for p, _n, _t in iter_lines(("*.py",), Path("."))]
    assert seen == ["good.py"]
