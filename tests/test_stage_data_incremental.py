"""Integration test: incremental netcdf_glob staging is value-identical.

Drives `tests/_stage_equiv_harness.py` in a subprocess so it runs against the
real xarray (the pure unit tests in `test_stage_data.py` install a lightweight
xarray mock at import time, which would otherwise leak into this process).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.process_isolation

HARNESS = Path(__file__).resolve().parent / "_stage_equiv_harness.py"

# The child normally finishes in ~30 s locally and a few minutes on a CI
# runner, so this is a HANG DETECTOR, not a performance budget -- it must never
# fire on a slow-but-working machine.
#
# Why it exists: on 2026-08-09 this test hung the windows CI leg. The suite
# reached 89 % at 18:56:55, printed nothing for 24 minutes, and the job was
# killed by `timeout-minutes: 30`. An unbounded `subprocess.run` cannot do
# anything else -- the parent waits forever, pytest never reports, and the log
# ends mid-progress-bar with no indication of which test stalled or why.
#
# That is also why the flake hunt on the same day found nothing in 15 runs
# (t2608071208): it counted failures, and a hang never produces one. Bounding
# the child converts the next occurrence from a 30-minute kill with no
# information into a normal test failure carrying the child's partial output.
# It does NOT fix the underlying stall -- see that item.
#
# 600 -> 240 on 2026-08-21 (owner ruling, t2608071208). The bound governs how
# long an OCCURRENCE costs, and ten minutes of waiting is most of what
# investigating one costs. 240 s keeps the headroom the paragraph above demands
# -- roughly 15x the 12-16 s passing runs, and still above the "few minutes" a
# CI runner takes -- while capturing the same evidence: the harness arms
# `faulthandler.dump_traceback_later(120, repeat=True)`, so a stalled child
# dumps every thread's stack at 120 s and again at 240 s. The three further
# dumps 600 s bought never told anyone anything the first two had not.
HARNESS_TIMEOUT_S = 240


def test_netcdf_glob_widening_is_incremental_and_value_identical(tmp_path) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        result = subprocess.run(
            [sys.executable, str(HARNESS), str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",  # child emits UTF-8 banners; avoid cp1252 decode
            errors="replace",
            env=env,
            timeout=HARNESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        # Report what the child managed to emit before it stalled. `capture_output`
        # means TimeoutExpired carries the partial streams, and they are the only
        # evidence of WHERE it stopped -- exactly what the CI kill destroyed.
        #
        # STDERR IN FULL. It carries the child's faulthandler dump, and a tail
        # cap cut the first thread's stack mid-frame on the 2026-09-18 ubuntu
        # occurrence -- the one capture a once-a-month stall owes us. Stdout is
        # progress banners, so its tail is enough. TimeoutExpired can carry
        # bytes even under text=True, so both are decoded here.
        def _text(stream):
            if isinstance(stream, bytes):
                return stream.decode("utf-8", errors="replace")
            return stream or ""

        pytest.fail(
            f"the staging harness did not finish within {HARNESS_TIMEOUT_S}s -- "
            f"treat this as the t2608071208 stall, not as a slow machine.\n"
            f"PARTIAL STDOUT (tail):\n{_text(exc.stdout)[-4000:]}\n"
            f"PARTIAL STDERR (full):\n{_text(exc.stderr)}"
        )
    assert result.returncode == 0, (
        f"equivalence harness failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "PASS" in result.stdout
