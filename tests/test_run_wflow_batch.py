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

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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
OK_ROW = "08:03:16 - wflow - Run run_01  ━━━━━━━━━━  100.0%  0:00:01 elapsed  [1/3]"
FAIL_ROW = (
    "08:03:16 - wflow - FAILED [2/3] Run run_02 batch=b1  boom: forcing not found"
)


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
    rows = [line for line in _code_lines() if 'row("' in line]
    assert rows, SOURCE
    for line in rows:
        assert "[$(k)/$(total)]" in line, line


def test_the_total_is_this_batch_and_not_the_experiment():
    """Rule 3.15 runs several batches concurrently, each its own process, so a
    run-wide denominator is not knowable here and would be a lie if printed."""
    assert re.search(r"^\s*total\s*=\s*length\(members\)\s*$", SOURCE, re.MULTILINE)


def test_the_member_tag_is_explicit_and_failure_names_member_and_batch():
    """Run identity is supplied by the batch, never parsed from a filename.

    A failure costs only its own member since staging (batch_staging.py), so
    the row names that member and its batch, not every batch-mate.
    """
    assert "tag = member.run_id" in SOURCE
    assert "basename(" not in SOURCE and "splitext(" not in SOURCE
    assert "FAILED [$(k)/$(total)] Run $(tag) batch=$(batch_id)" in SOURCE


def test_a_member_sends_its_position_on_the_bar_instead_of_a_success_row():
    """The finished bar row is the member's record, so success prints no row."""
    assert 'position="[$(k)/$(total)]"' in SOURCE
    assert not [line for line in _code_lines() if 'row("[' in line]


def test_a_finished_member_is_staged_before_its_marker():
    """The CSV moves first and the marker second, so an `.ok` implies its CSV."""
    code = chr(10).join(_code_lines())
    move_csv = code.index(
        'mv(member.native_output_path, joinpath(staging, "$(tag).csv")'
    )
    promote = code.index('joinpath(staging, "$(tag).ok")')
    assert move_csv < promote
    assert 'get(ENV, "CST_BATCH_STAGING", "")' in code


def test_explicit_batch_parser_in_julia(tmp_path):
    """The real stdlib-only parser preserves records and rejects bad ordering."""
    julia = shutil.which("julia")
    if julia is None:
        pytest.skip("Julia is not available for the standalone batch parser check")
    code = """
include(ARGS[1])
using Test
batch, members = parse_batch(["b4", "run_007", "z.toml", "a.csv", "run_012", "q.toml", "b.csv"])
@test batch == "b4"
@test members[1] == (run_id="run_007", toml_path="z.toml", native_output_path="a.csv")
@test members[2].run_id == "run_012"
@test_throws ErrorException parse_batch(["b4", "run_012", "z.toml", "a.csv", "run_007", "q.toml", "b.csv"])
@test_throws ErrorException parse_batch(["b4", "run_007", "z.toml"])
@test_throws ErrorException parse_batch(["b4", "run_007", "z.toml", "a.csv", "run_007", "q.toml", "b.csv"])
@test_throws ErrorException parse_batch(["b4", "007", "z.toml", "a.csv"])
include(joinpath(dirname(ARGS[1]), "..", "shared", "resource_timing.jl"))
ENV["CST_RESOURCE_DIRECTORY"] = ARGS[2]
@test with_resource_timing(() -> sum(1:100), "member", "b4", "run_007") == 5050
@test_throws ErrorException with_resource_timing(() -> error("expected"), "member", "b4", "run_012")
rows = readlines(joinpath(ARGS[2], "julia-$(getpid()).tsv"))
@test length(rows) == 3
@test split(rows[1], '\\t')[6:10] == ["wall_s", "gc_s", "allocated_bytes", "compile_s", "recompile_s"]
@test split(rows[2], '\\t')[3:5] == ["b4", "run_007", "ok"]
@test parse(Float64, split(rows[2], '\\t')[6]) >= 0
@test split(rows[3], '\\t')[5] == "failed"
"""
    result = subprocess.run(
        [julia, "--startup-file=no", "-e", code, str(DRIVER), str(tmp_path)],
        capture_output=True,
        text=True,
        # Budget for PRECOMPILATION, not for the parser. `using Test` costs
        # under a second against a warm depot -- which is every developer
        # machine, and why 60s looked ample -- but a CI runner starts with an
        # empty one and precompiles the stdlib from scratch. CI installs Julia
        # per run and caches no depot (Julia is juliaup-managed and absent from
        # pixi by design), so every run pays the cold cost, and 60s flaked on
        # BOTH platforms in run 35009891318 after passing minutes earlier on the
        # same tree.
        #
        # This is a hang guard, not a performance assertion: the test checks
        # what the parser returns, and nothing here is measuring how fast Julia
        # starts. The generous ceiling is the point -- a timeout tight enough to
        # be occasionally wrong reports a red suite for a reason that has
        # nothing to do with the code under test.
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_instrumented_driver_preserves_member_outputs_and_failure(tmp_path):
    """Execute the real driver against a tiny simulator, including package loading."""
    julia = shutil.which("julia")
    if julia is None:
        pytest.skip("Julia is not available for the instrumented driver check")
    package = tmp_path / "packages/Wflow/src"
    package.mkdir(parents=True)
    (package / "Wflow.jl").write_text(
        'module Wflow\nusing Logging\nconst VERSION = "test"\n'
        "Config(path) = (logging=(silent=true,), fews_run__flag=false, path=path)\n"
        "init_logger(config; silent) = (NullLogger(), IOBuffer())\n"
        "function run(config)\n"
        '  endswith(config.path, "fail.toml") && error("expected failure")\n'
        '  write(read(config.path, String), "native response\\n")\nend\nend\n',
        encoding="utf-8",
    )
    output = tmp_path / "run_01.csv"
    settings = tmp_path / "ok.toml"
    settings.write_text(str(output), encoding="utf-8")
    diagnostics = tmp_path / "diagnostics"
    env = {
        **os.environ,
        "JULIA_LOAD_PATH": os.pathsep.join([str(tmp_path / "packages"), "@stdlib"]),
    }
    env.pop("CST_BATCH_STAGING", None)
    for enabled in (False, True):
        if enabled:
            env["CST_RESOURCE_DIRECTORY"] = str(diagnostics)
        else:
            env.pop("CST_RESOURCE_DIRECTORY", None)
        result = subprocess.run(
            [
                julia,
                "--startup-file=no",
                "--compiled-modules=no",
                str(DRIVER),
                "0",
                "run_01",
                str(settings),
                str(output),
                "run_02",
                str(tmp_path / "fail.toml"),
                str(tmp_path / "run_02.csv"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "expected failure" in result.stdout + result.stderr
        assert output.read_bytes() == b"native response\n"
        assert not (tmp_path / "run_02.csv").exists()
    (trace,) = diagnostics.glob("julia-*.tsv")
    with trace.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["phase"], row["run_id"], row["status"]) for row in rows] == [
        ("wflow_load", "", "ok"),
        ("member", "run_01", "ok"),
        ("member", "run_02", "failed"),
    ]
