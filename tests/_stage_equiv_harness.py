"""Equivalence harness for incremental staging (Phases 1 & 2).

Run as a script inside the real env (needs xarray/pandas) — the leading
underscore keeps pytest from collecting it. `tests/test_stage_data_incremental.py`
invokes it in a subprocess so the real xarray is used, not the lightweight mock
that `tests/test_stage_data.py` installs for the pure unit tests.

Scenarios (each exits the process non-zero on failure):
  - netcdf_glob widening (Phase 1): only newly-in-range year files are staged;
    interior files are not rewritten; result == wide-from-scratch.
  - zarr / single-netcdf rebuild (Phase 2): widening the time_range (incl. a
    PREPEND) or adding a variable reuses the local store and reads only the
    delta from the source. Proven two ways:
      * value-identity — incremental result == wide-from-scratch, and
      * provenance — after staging narrow, the SOURCE overlap is mutated; the
        widened output must still show the ORIGINAL values in the overlap
        (reused from local) and the mutated/new values only in the delta.

    python tests/_stage_equiv_harness.py <workdir>
Exit 0 = all pass; 1 = failure (with a diagnostic on stdout).
"""

from __future__ import annotations

import faulthandler
import os
import shutil
import sys
from pathlib import Path

# --- the stall's self-diagnosis (t2608071208) --------------------------------
#
# The hang was captured for the first time on 2026-08-18 and localised only as
# far as its `STEP` marker: `STEP glob`, the widen stage, 0 of 6 files. What it
# is BLOCKED ON is the open question, and the answer is a stack trace of the
# stalled child -- which nothing could take, because the child is a subprocess
# that never returns and py-spy is not a dependency of this repo.
#
# `faulthandler` is stdlib and needs no dependency, no signal, and no second
# process: it dumps EVERY thread's stack from inside this process on a timer,
# to stderr, which `test_stage_data_incremental` already captures and prints
# under PARTIAL STDERR when the 600 s bound fires. A Python frame blocked in a
# C call still shows the frame that entered it, which is the whole question
# here (netCDF/HDF5 open, a lock, a rename).
#
# 120 s, repeating: every passing run takes 12-14 s, so a dump means a stall
# rather than a slow machine, and repeating turns one occurrence into a series
# that distinguishes "stuck in one place" from "livelocked between two".
# CST_HARNESS_DUMP_SECS overrides it; 0 disables.
_DUMP_SECS = float(os.environ.get("CST_HARNESS_DUMP_SECS", "120"))
if _DUMP_SECS > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(_DUMP_SECS, repeat=True, exit=False)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import dask  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import stage_data as sd  # noqa: E402
import xarray as xr  # noqa: E402


def _stage(cfg):
    # Force the synchronous dask scheduler so this end-to-end test is fast and
    # deterministic: the fixtures are tiny, and it sidesteps the concurrent
    # zarr-v3 metadata renames that are flaky on Windows. This test targets the
    # incremental logic (scheduler-independent); the production write path's
    # retry against that flakiness is covered by a dedicated unit test.
    with dask.config.set(scheduler="synchronous"):
        return sd.stage(cfg)


BBOX = [0.0, 0.0, 2.0, 2.0]
LAT = np.array([0.0, 1.0, 2.0])
LON = np.array([0.0, 1.0, 2.0])


def _year_times(year: int, n: int = 4) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", periods=n, freq="7D")


def _year_ds(year: int, *, with_temp: bool, precip_bump: float) -> xr.Dataset:
    times = _year_times(year)
    step = np.arange(len(times))[:, None, None] * 100.0
    base = year * 1000.0 + step + LAT[None, :, None] * 10.0 + LON[None, None, :]
    data = {"precip": (("time", "lat", "lon"), (base + precip_bump).astype("float32"))}
    if with_temp:
        # temp is independent of the precip bump, so provenance tests can mutate
        # precip without touching temp.
        data["temp"] = (("time", "lat", "lon"), (base + 0.5).astype("float32"))
    return xr.Dataset(data, coords={"time": times, "lat": LAT, "lon": LON})


def _source_ds(years, *, with_temp=False, bump_year=None, bump=0.0) -> xr.Dataset:
    frames = [
        _year_ds(y, with_temp=with_temp, precip_bump=(bump if y == bump_year else 0.0))
        for y in years
    ]
    return xr.concat(frames, dim="time")


def _write_source(path: Path, ds: xr.Dataset, *, as_zarr: bool) -> None:
    """Write one generation of a source dataset, replacing any previous one.

    The zarr leg goes through the PRODUCT's writer rather than a bare
    ``to_zarr``. `stage_data._write_zarr` already absorbs the zarr-v3 metadata
    rename that intermittently raises ``PermissionError [WinError 5]`` on
    Windows -- it clears the partial store and retries, three attempts with a
    backoff -- and this fixture write was the one zarr write in the harness that
    had no such protection.

    That is not hypothetical: it is what failed on 2026-08-18 (t2608071208).
    The traceback landed on `_atomic_write`'s `tmp_path.replace(path)` for
    `zarr.json`, in THIS function, called the second time
    `_rebuild_provenance` writes its source -- the delete-then-recreate of the
    same store path is what gives the OS a delete-pending target to deny.

    Reusing the product's helper rather than adding a second retry here is
    deliberate: a private import from a module this harness already imports
    beats two implementations of one workaround drifting apart. `serial=True`
    because these fixtures are tiny, matching `_stage`'s own reason for pinning
    the synchronous scheduler.

    Writing each generation to a FRESH path was the other candidate and is
    rejected: the scenario's whole point is that the source at the CONFIGURED
    path changes between stages, so moving it would alter the property under
    test rather than the flake.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    if as_zarr:
        sd._write_zarr(ds, path, {}, serial=True)
    else:
        ds.to_netcdf(path)


def _cfg(
    source_root, target_root, rel, typ, time_range, *, variables=None, pattern=None
):
    entry = {"name": "x", "type": typ, "path": rel, "time_range": list(time_range)}
    if variables is not None:
        entry["variables"] = list(variables)
    if pattern is not None:
        entry["pattern"] = pattern
    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "bbox": list(BBOX),
        "datasets": [entry],
    }


def _open(path: Path, *, as_zarr: bool) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True) if as_zarr else xr.open_dataset(path)


# --- Phase 1: netcdf_glob ----------------------------------------------------


def scenario_glob(work: Path) -> None:
    src = work / "src"
    years = [1998, 1999, 2000, 2001, 2002, 2003]
    for y in years:
        _write_source(
            src / "meteo" / "glob" / f"y_{y}.nc", _source_ds([y]), as_zarr=False
        )

    a, b = work / "gA", work / "gB"

    def out(root, y):
        return root / "meteo" / "glob" / f"y_{y}.nc"

    _stage(
        _cfg(
            src,
            a,
            "meteo/glob",
            "netcdf_glob",
            ["2000-01-01", "2002-12-31"],
            pattern="y_*.nc",
        )
    )
    interior, new = [2000, 2001, 2002], [1998, 1999, 2003]
    for y in new:
        assert not out(a, y).exists(), f"glob narrow: {y} unexpectedly staged"
    mtimes = {y: out(a, y).stat().st_mtime_ns for y in interior}

    report = _stage(
        _cfg(
            src,
            a,
            "meteo/glob",
            "netcdf_glob",
            ["1998-01-01", "2003-12-31"],
            pattern="y_*.nc",
        )
    )
    for y in interior:
        assert out(a, y).stat().st_mtime_ns == mtimes[y], f"glob widen: {y} rewritten"
    for y in new:
        assert out(a, y).exists(), f"glob widen: {y} not staged"
    c = report.counts()
    assert c[sd.WRITTEN] == 3 and c[sd.EXISTS] == 3, f"glob widen counts {c}"

    _stage(
        _cfg(
            src,
            b,
            "meteo/glob",
            "netcdf_glob",
            ["1998-01-01", "2003-12-31"],
            pattern="y_*.nc",
        )
    )
    for y in years:
        da, db = xr.open_dataset(out(a, y)), xr.open_dataset(out(b, y))
        try:
            xr.testing.assert_identical(da, db)
        finally:
            da.close()
            db.close()


# --- Phase 2: single-store rebuild (zarr + netcdf) ---------------------------


def _rebuild_value_identity(work: Path, typ: str, as_zarr: bool) -> None:
    """Widening (a prepend) incrementally == wide-from-scratch."""
    src = work / "src"
    rel = f"meteo/store_{typ}.{'zarr' if as_zarr else 'nc'}"
    _write_source(src / rel, _source_ds(range(1998, 2004)), as_zarr=as_zarr)
    a, b = work / "A", work / "B"

    _stage(_cfg(src, a, rel, typ, ["2001-01-01", "2002-12-31"]))  # narrow
    _stage(_cfg(src, a, rel, typ, ["1999-01-01", "2002-12-31"]))  # widen (prepend)
    _stage(_cfg(src, b, rel, typ, ["1999-01-01", "2002-12-31"]))  # from scratch

    da, db = _open(a / rel, as_zarr=as_zarr), _open(b / rel, as_zarr=as_zarr)
    try:
        xr.testing.assert_identical(da, db)
        t = da["time"].values
        assert (t == np.sort(t)).all(), f"{typ}: time axis not sorted after prepend"
    finally:
        da.close()
        db.close()


def _rebuild_provenance(work: Path, typ: str, as_zarr: bool) -> None:
    """Reused overlap reflects LOCAL data, not a source mutated between stages."""
    src = work / "src"
    rel = f"meteo/prov_{typ}.{'zarr' if as_zarr else 'nc'}"
    _write_source(src / rel, _source_ds(range(1998, 2004)), as_zarr=as_zarr)
    a = work / "P"

    _stage(_cfg(src, a, rel, typ, ["2001-01-01", "2002-12-31"]))  # narrow
    ds0 = _open(a / rel, as_zarr=as_zarr)
    overlap_2001 = ds0["precip"].sel(time=_year_times(2001)).values.copy()
    ds0.close()

    # Mutate the source's 2001 overlap, then widen: 2001 must NOT be re-read.
    _write_source(
        src / rel,
        _source_ds(range(1998, 2004), bump_year=2001, bump=9999.0),
        as_zarr=as_zarr,
    )
    _stage(_cfg(src, a, rel, typ, ["1999-01-01", "2002-12-31"]))  # widen

    ds1 = _open(a / rel, as_zarr=as_zarr)
    try:
        got_2001 = ds1["precip"].sel(time=_year_times(2001)).values
        assert np.allclose(got_2001, overlap_2001), (
            f"{typ}: overlap year 2001 was re-read from the (mutated) source, not reused"
        )
        # The prepended delta (1999) is present and unmutated.
        got_1999 = ds1["precip"].sel(time=_year_times(1999)).values
        expected_1999 = _year_ds(1999, with_temp=False, precip_bump=0.0)[
            "precip"
        ].values
        assert np.allclose(got_1999, expected_1999), (
            f"{typ}: prepended 1999 delta wrong"
        )
    finally:
        ds1.close()


def _rebuild_add_variable(work: Path) -> None:
    """Adding a variable pulls only the new var from source; precip is reused."""
    src = work / "src"
    rel = "meteo/addvar.zarr"
    _write_source(
        src / rel, _source_ds(range(2001, 2003), with_temp=True), as_zarr=True
    )
    a = work / "V"

    _stage(
        _cfg(src, a, rel, "zarr", ["2001-01-01", "2002-12-31"], variables=["precip"])
    )
    ds0 = xr.open_zarr(a / rel, consolidated=True)
    assert "temp" not in ds0.data_vars, "addvar narrow: temp should not be staged yet"
    precip0 = ds0["precip"].values.copy()
    ds0.close()

    # Mutate source precip; add temp to the request. precip must be reused.
    _write_source(
        src / rel,
        _source_ds(range(2001, 2003), with_temp=True, bump_year=2001, bump=9999.0),
        as_zarr=True,
    )
    _stage(
        _cfg(
            src,
            a,
            rel,
            "zarr",
            ["2001-01-01", "2002-12-31"],
            variables=["precip", "temp"],
        )
    )

    ds1 = xr.open_zarr(a / rel, consolidated=True)
    try:
        assert "temp" in ds1.data_vars, "addvar widen: temp not added"
        assert np.allclose(ds1["precip"].values, precip0), (
            "addvar: precip was re-read from the mutated source, not reused"
        )
        expected_temp = _source_ds(range(2001, 2003), with_temp=True)["temp"].values
        assert np.allclose(ds1["temp"].values, expected_temp), (
            "addvar: temp values wrong"
        )
    finally:
        ds1.close()


def _rebuild_preserves_packing(work: Path) -> None:
    """A widened int16-packed source stays int16-packed after the rebuild."""
    src = work / "src"
    rel = "meteo/packed.nc"
    ds = _source_ds(range(1999, 2003))
    ds["precip"].encoding = {
        "dtype": "int16",
        "scale_factor": 0.1,
        "_FillValue": -9999,
        "zlib": True,
    }
    _write_source(src / rel, ds, as_zarr=False)
    a, b = work / "A", work / "B"

    _stage(_cfg(src, a, rel, "netcdf", ["2001-01-01", "2002-12-31"]))  # narrow
    _stage(_cfg(src, a, rel, "netcdf", ["1999-01-01", "2002-12-31"]))  # widen (rebuild)
    _stage(_cfg(src, b, rel, "netcdf", ["1999-01-01", "2002-12-31"]))  # from scratch

    da, db = xr.open_dataset(a / rel), xr.open_dataset(b / rel)
    try:
        xr.testing.assert_identical(da, db)  # decoded values match
        assert da["precip"].encoding.get("dtype") == np.dtype("int16"), (
            "rebuild lost int16 packing (stored as float)"
        )
        assert db["precip"].encoding.get("dtype") == np.dtype("int16")
    finally:
        da.close()
        db.close()


def _chunked_download_equiv(work: Path) -> None:
    """The block-wise download path == a single load, and preserves packing."""
    src = work / "src"
    rel = "meteo/chunked.nc"
    ds = _source_ds(range(1999, 2003))  # 16 timesteps
    ds["precip"].encoding = {
        "dtype": "int16",
        "scale_factor": 0.1,
        "_FillValue": -9999,
        "zlib": True,
    }
    _write_source(src / rel, ds, as_zarr=False)
    a, b = work / "A", work / "B"

    orig = sd.DOWNLOAD_BLOCK_STEPS
    try:
        sd.DOWNLOAD_BLOCK_STEPS = 10_000  # single-load reference
        _stage(_cfg(src, b, rel, "netcdf", ["1999-01-01", "2002-12-31"]))
        sd.DOWNLOAD_BLOCK_STEPS = 3  # force a multi-block chunked download
        _stage(_cfg(src, a, rel, "netcdf", ["1999-01-01", "2002-12-31"]))
    finally:
        sd.DOWNLOAD_BLOCK_STEPS = orig

    da, db = xr.open_dataset(a / rel), xr.open_dataset(b / rel)
    try:
        xr.testing.assert_identical(da, db)
        assert da["precip"].encoding.get("dtype") == np.dtype("int16"), (
            "chunked download lost int16 packing"
        )
    finally:
        da.close()
        db.close()


def _step(name: str) -> None:
    """Announce a scenario, FLUSHED, so a stall can be localized.

    The caller runs this harness with ``capture_output=True``, i.e. through a
    pipe, where stdout is block-buffered -- so without an explicit flush every
    line is still sitting in the buffer when a hung child is killed, and the
    captured output arrives empty. That is exactly what happened on 2026-08-09:
    the windows CI leg stalled here for 24 minutes and the kill produced no
    indication of which scenario was running (t2608071208).
    """
    print(f"STEP {name}", flush=True)


def main(workdir: str) -> None:
    work = Path(workdir)
    _step("glob")
    scenario_glob(work / "glob")
    for typ, as_zarr in (("zarr", True), ("netcdf", False)):
        _step(f"value_identity:{typ}")
        _rebuild_value_identity(work / f"vi_{typ}", typ, as_zarr)
        _step(f"provenance:{typ}")
        _rebuild_provenance(work / f"pv_{typ}", typ, as_zarr)
    _step("add_variable")
    _rebuild_add_variable(work / "addvar")
    _step("preserves_packing")
    _rebuild_preserves_packing(work / "packed")
    _step("chunked_download")
    _chunked_download_equiv(work / "chunked")
    print("PASS", flush=True)


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
