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

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import dask  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import stage_data as sd  # noqa: E402


def _stage(cfg):
    # Force the synchronous dask scheduler: the fixtures are tiny, so serial is
    # instant, and it avoids the concurrent zarr-v3 metadata renames that are
    # flaky on Windows (a pre-existing property of the from-scratch write path,
    # independent of the incremental logic under test).
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    if as_zarr:
        ds.to_zarr(path, consolidated=True, mode="w")
    else:
        ds.to_netcdf(path)


def _cfg(source_root, target_root, rel, typ, time_range, *, variables=None, pattern=None):
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
        _write_source(src / "meteo" / "glob" / f"y_{y}.nc", _source_ds([y]), as_zarr=False)

    a, b = work / "gA", work / "gB"

    def out(root, y):
        return root / "meteo" / "glob" / f"y_{y}.nc"

    _stage(_cfg(src, a, "meteo/glob", "netcdf_glob", ["2000-01-01", "2002-12-31"], pattern="y_*.nc"))
    interior, new = [2000, 2001, 2002], [1998, 1999, 2003]
    for y in new:
        assert not out(a, y).exists(), f"glob narrow: {y} unexpectedly staged"
    mtimes = {y: out(a, y).stat().st_mtime_ns for y in interior}

    report = _stage(_cfg(src, a, "meteo/glob", "netcdf_glob", ["1998-01-01", "2003-12-31"], pattern="y_*.nc"))
    for y in interior:
        assert out(a, y).stat().st_mtime_ns == mtimes[y], f"glob widen: {y} rewritten"
    for y in new:
        assert out(a, y).exists(), f"glob widen: {y} not staged"
    c = report.counts()
    assert c[sd.WRITTEN] == 3 and c[sd.EXISTS] == 3, f"glob widen counts {c}"

    _stage(_cfg(src, b, "meteo/glob", "netcdf_glob", ["1998-01-01", "2003-12-31"], pattern="y_*.nc"))
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

    _stage(_cfg(src, a, rel, typ, ["2001-01-01", "2002-12-31"]))   # narrow
    _stage(_cfg(src, a, rel, typ, ["1999-01-01", "2002-12-31"]))   # widen (prepend)
    _stage(_cfg(src, b, rel, typ, ["1999-01-01", "2002-12-31"]))   # from scratch

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

    _stage(_cfg(src, a, rel, typ, ["2001-01-01", "2002-12-31"]))   # narrow
    ds0 = _open(a / rel, as_zarr=as_zarr)
    overlap_2001 = ds0["precip"].sel(time=_year_times(2001)).values.copy()
    ds0.close()

    # Mutate the source's 2001 overlap, then widen: 2001 must NOT be re-read.
    _write_source(src / rel, _source_ds(range(1998, 2004), bump_year=2001, bump=9999.0), as_zarr=as_zarr)
    _stage(_cfg(src, a, rel, typ, ["1999-01-01", "2002-12-31"]))   # widen

    ds1 = _open(a / rel, as_zarr=as_zarr)
    try:
        got_2001 = ds1["precip"].sel(time=_year_times(2001)).values
        assert np.allclose(got_2001, overlap_2001), (
            f"{typ}: overlap year 2001 was re-read from the (mutated) source, not reused"
        )
        # The prepended delta (1999) is present and unmutated.
        got_1999 = ds1["precip"].sel(time=_year_times(1999)).values
        expected_1999 = _year_ds(1999, with_temp=False, precip_bump=0.0)["precip"].values
        assert np.allclose(got_1999, expected_1999), f"{typ}: prepended 1999 delta wrong"
    finally:
        ds1.close()


def _rebuild_add_variable(work: Path) -> None:
    """Adding a variable pulls only the new var from source; precip is reused."""
    src = work / "src"
    rel = "meteo/addvar.zarr"
    _write_source(src / rel, _source_ds(range(2001, 2003), with_temp=True), as_zarr=True)
    a = work / "V"

    _stage(_cfg(src, a, rel, "zarr", ["2001-01-01", "2002-12-31"], variables=["precip"]))
    ds0 = xr.open_zarr(a / rel, consolidated=True)
    assert "temp" not in ds0.data_vars, "addvar narrow: temp should not be staged yet"
    precip0 = ds0["precip"].values.copy()
    ds0.close()

    # Mutate source precip; add temp to the request. precip must be reused.
    _write_source(src / rel, _source_ds(range(2001, 2003), with_temp=True, bump_year=2001, bump=9999.0), as_zarr=True)
    _stage(_cfg(src, a, rel, "zarr", ["2001-01-01", "2002-12-31"], variables=["precip", "temp"]))

    ds1 = xr.open_zarr(a / rel, consolidated=True)
    try:
        assert "temp" in ds1.data_vars, "addvar widen: temp not added"
        assert np.allclose(ds1["precip"].values, precip0), (
            "addvar: precip was re-read from the mutated source, not reused"
        )
        expected_temp = _source_ds(range(2001, 2003), with_temp=True)["temp"].values
        assert np.allclose(ds1["temp"].values, expected_temp), "addvar: temp values wrong"
    finally:
        ds1.close()


def _rebuild_preserves_packing(work: Path) -> None:
    """A widened int16-packed source stays int16-packed after the rebuild."""
    src = work / "src"
    rel = "meteo/packed.nc"
    ds = _source_ds(range(1999, 2003))
    ds["precip"].encoding = {
        "dtype": "int16", "scale_factor": 0.1, "_FillValue": -9999, "zlib": True,
    }
    _write_source(src / rel, ds, as_zarr=False)
    a, b = work / "A", work / "B"

    _stage(_cfg(src, a, rel, "netcdf", ["2001-01-01", "2002-12-31"]))   # narrow
    _stage(_cfg(src, a, rel, "netcdf", ["1999-01-01", "2002-12-31"]))   # widen (rebuild)
    _stage(_cfg(src, b, rel, "netcdf", ["1999-01-01", "2002-12-31"]))   # from scratch

    da, db = xr.open_dataset(a / rel), xr.open_dataset(b / rel)
    try:
        xr.testing.assert_identical(da, db)                     # decoded values match
        assert da["precip"].encoding.get("dtype") == np.dtype("int16"), (
            "rebuild lost int16 packing (stored as float)"
        )
        assert db["precip"].encoding.get("dtype") == np.dtype("int16")
    finally:
        da.close()
        db.close()


def main(workdir: str) -> None:
    work = Path(workdir)
    scenario_glob(work / "glob")
    for typ, as_zarr in (("zarr", True), ("netcdf", False)):
        _rebuild_value_identity(work / f"vi_{typ}", typ, as_zarr)
        _rebuild_provenance(work / f"pv_{typ}", typ, as_zarr)
    _rebuild_add_variable(work / "addvar")
    _rebuild_preserves_packing(work / "packed")
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
