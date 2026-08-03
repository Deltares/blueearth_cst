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


@pytest.mark.slow
def test_netcdf_glob_widening_is_incremental_and_value_identical(tmp_path) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, str(HARNESS), str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",       # child emits UTF-8 banners; avoid cp1252 decode
        errors="replace",
        env=env,
    )
    assert result.returncode == 0, (
        f"equivalence harness failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "PASS" in result.stdout
