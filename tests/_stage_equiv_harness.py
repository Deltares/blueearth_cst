"""Equivalence harness for incremental ``netcdf_glob`` staging (Phase 1).

Run as a script inside the real env (needs xarray/pandas) — the leading
underscore keeps pytest from collecting it. `tests/test_stage_data_incremental.py`
invokes it in a subprocess so the real xarray is used, not the lightweight mock
that `tests/test_stage_data.py` installs for the pure unit tests.

It proves two things about widening a glob's ``time_range``:
  1. incremental correctness — the widened run's per-year outputs are byte-for-
     value identical to staging the wide range from scratch, and
  2. incrementality — the widened run does NOT rewrite the unchanged interior
     year files (their mtimes are untouched; the report shows them as EXISTS).

    python tests/_stage_equiv_harness.py <workdir>
Exit 0 = pass; 1 = failure (with a diagnostic on stdout).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import stage_data as sd  # noqa: E402

YEARS = [1998, 1999, 2000, 2001, 2002, 2003]
INTERIOR = [2000, 2001, 2002]          # staged by the narrow window
NEW = [1998, 1999, 2003]               # added only by the widened window


def _make_year_file(path: Path, year: int) -> None:
    """Write a small deterministic per-year netcdf (4 weekly steps, 3x3 grid)."""
    times = pd.date_range(f"{year}-01-01", periods=4, freq="7D")
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([0.0, 1.0, 2.0])
    # Deterministic in (year, step, lat, lon) so incremental and from-scratch
    # outputs must match exactly if the clip is correct.
    step = np.arange(len(times))[:, None, None] * 100.0
    data = year * 1000.0 + step + lat[None, :, None] * 10.0 + lon[None, None, :]
    ds = xr.Dataset(
        {"precip": (("time", "lat", "lon"), data.astype("float32"))},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _cfg(source_root: Path, target_root: Path, time_range) -> dict:
    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "datasets": [
            {
                "name": "glob",
                "type": "netcdf_glob",
                "path": "meteo/glob_test",
                "pattern": "y_*.nc",
                "time_range": list(time_range),
            }
        ],
    }


def main(workdir: str) -> None:
    work = Path(workdir)
    src = work / "src"
    for y in YEARS:
        _make_year_file(src / "meteo" / "glob_test" / f"y_{y}.nc", y)

    a, b = work / "A", work / "B"

    def out(root: Path, y: int) -> Path:
        return root / "meteo" / "glob_test" / f"y_{y}.nc"

    # 1) Narrow stage into A.
    sd.stage(_cfg(src, a, ["2000-01-01", "2002-12-31"]))
    for y in INTERIOR:
        assert out(a, y).exists(), f"narrow: {y} not staged"
    for y in NEW:
        assert not out(a, y).exists(), f"narrow: {y} unexpectedly staged"
    mtimes = {y: out(a, y).stat().st_mtime_ns for y in INTERIOR}

    # 2) Widen the window into A — must be incremental.
    report = sd.stage(_cfg(src, a, ["1998-01-01", "2003-12-31"]))
    for y in INTERIOR:
        assert out(a, y).stat().st_mtime_ns == mtimes[y], (
            f"widen: interior year {y} was rewritten (not incremental)"
        )
    for y in NEW:
        assert out(a, y).exists(), f"widen: new year {y} not staged"
    counts = report.counts()
    assert counts[sd.WRITTEN] == len(NEW) and counts[sd.EXISTS] == len(INTERIOR), (
        f"widen: expected {len(NEW)} written + {len(INTERIOR)} existing, got {counts}"
    )

    # 3) Wide stage from scratch into B.
    sd.stage(_cfg(src, b, ["1998-01-01", "2003-12-31"]))

    # 4) Value-identity: incremental A == from-scratch B, per year.
    for y in YEARS:
        da = xr.open_dataset(out(a, y))
        db = xr.open_dataset(out(b, y))
        try:
            xr.testing.assert_identical(da, db)
            times = da["time"].values
            assert (times == np.sort(times)).all(), f"{y}: time axis not sorted"
        finally:
            da.close()
            db.close()

    print("PASS")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
