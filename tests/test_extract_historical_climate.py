"""Unit tests for blueearth_cst/climate_analysis/extract_historical_climate.py.

This module is heavily coupled to hydromt I/O; we test the function's
configuration logic (driver options, variable lists, clim_source
branching) and skip the deeper reprojection paths.

Layer B -- what the staged source ACTUALLY delivers against what the config
requested -- is the second half of the file, below the fakes.
"""

from __future__ import annotations

import re
import types

import dask
import numpy as np
import pytest

from blueearth_cst.shared.snake_utils import MIN_HISTORICAL_YEARS

# --- Stubs for heavy deps (set up BEFORE importing the source module) ---


class _FakeRasterAccessor:
    """Mimics ds.raster on an xarray-like dataset."""

    def __init__(self, vars_, box=None):
        self.vars = list(vars_)
        self.box = box if box is not None else object()

    def reproject_like(self, *_args, **_kwargs):
        return _FakeDataset(["dummy"])


class _FakeDataArrayRaster:
    """ds_clim[var].raster used in the chirps branch for reproject_like."""

    def reproject_like(self, *_args, **_kwargs):
        return _FakeDataArray("reprojected")


class _FakeDataArray:
    def __init__(self, name, dtype="float32"):
        self.name = name
        self.raster = _FakeDataArrayRaster()
        # t2608161450: the producer coerces the WG-1 variables to float32 before
        # writing, and asks each one what it already is. Conforming by default,
        # so the cast is a no-op against this double -- the coercion's own
        # falsifier runs against real xarray objects in
        # tests/test_climate_store_wg1_conformance.py, and asserting it twice
        # here (once against a double that cannot be wrong) would be worse than
        # not asserting it at all.
        self.dtype = dtype
        self.attrs = {}


class _FakeDataset:
    """Quacks enough like an xarray Dataset for prep_historical_climate."""

    #: Scheduler in force at each ``to_netcdf(...).compute()``, newest last.
    #: Class-level because the catalog fake builds its datasets internally and
    #: hands the caller no reference to the one that actually gets written.
    _COMPUTE_SCHEDULERS = []

    def __init__(self, vars_, time_size=None, time_start="1980-01-01"):
        self._vars = list(vars_)
        self.raster = _FakeRasterAccessor(vars_)
        # A real (yearly) time axis from `time_start` so the coverage report has
        # a span to compare. size drives the span: default 100 -> ~1980 to 2079
        # (covers any test request); a narrow catalog (size 10) -> only
        # ~1980-1989, shorter than a 2000-2020 request. `time_start` moves the
        # whole axis, which is how a chirps/era5 coverage MISMATCH is built.
        n = time_size or 100
        self.time = types.SimpleNamespace(
            size=n,
            values=np.datetime64(time_start) + np.arange(n) * np.timedelta64(365, "D"),
        )
        self._tonetcdf_calls = []
        # ADR 0003: the producer stamps the extent provenance on the extraction
        # (region_bbox / region_geojson_sha256 / region_source), so the fake has
        # to carry an attrs mapping like a real Dataset.
        self.attrs = {}
        # t2608161450: the producer also coerces the WG-1 coords and variables
        # to float32 before writing. `_coerce_store_dtypes` asks a Dataset what
        # it holds, so the fake has to answer -- EMPTY coords and the declared
        # variables, which makes every cast a no-op here. That is deliberate:
        # these tests are about the branch logic and the coverage report, and
        # the dtype coercion has its own falsifier against real xarray objects
        # in tests/test_climate_store_wg1_conformance.py. A fake that pretended
        # to have float64 coords would be asserting the cast twice, once against
        # a double that cannot be wrong.
        self.coords = {}

    @property
    def data_vars(self):
        return {name: _FakeDataArray(name) for name in self._vars}

    def __getitem__(self, key):
        return _FakeDataArray(key)

    def __setitem__(self, key, value):
        if key not in self._vars:
            self._vars.append(key)
            self.raster.vars.append(key)

    def to_dataset(self):
        return self

    def sel(self, time=None, **_kwargs):
        """The chirps branch clips both reads to their overlapping window.

        Enough of `.sel(time=slice(a, b))` to keep the fake's axis honest: the
        returned fake carries only the timestamps inside the slice, so a test
        can assert on what the clip actually produced rather than on the call.
        """
        if time is None:
            return self
        values = self.time.values
        keep = values[
            (values >= np.datetime64(time.start)) & (values <= np.datetime64(time.stop))
        ]
        clipped = _FakeDataset(self._vars)
        clipped.time = types.SimpleNamespace(size=keep.size, values=keep)
        clipped.attrs = dict(self.attrs)
        return clipped

    def close(self):
        """Real xr.Datasets always have this; the fake did not.

        prep_historical_climate closes the store deterministically after
        writing, which broke all seven tests that drive this fake through it --
        a gap in the double, not in the product. Recorded because the fake will
        keep drifting from xarray unless each addition says why it exists.
        """
        self.closed = True

    def squeeze(self):
        return self

    def to_netcdf(self, fn, **kwargs):
        self._tonetcdf_calls.append((fn, kwargs))

        class _Delayed:
            def compute(self_inner):
                # Recorded AT COMPUTE TIME, which is the only moment the
                # scheduler is actually in force -- the producer sets it with a
                # `dask.config.set` context manager around this call.
                _FakeDataset._COMPUTE_SCHEDULERS.append(
                    dask.config.get("scheduler", None)
                )
                return None

        return _Delayed()


class _RecordingDataCatalog:
    """Records calls so tests can assert what was requested from hydromt."""

    _CATALOG = {}
    _LAST_INSTANCE = None

    def __init__(self, data_libs=None):
        self.data_libs = data_libs
        self.get_rasterdataset_calls = []
        type(self)._LAST_INSTANCE = self

    def to_dict(self):
        return {k: dict(v) for k, v in type(self)._CATALOG.items()}

    def from_dict(self, d):
        type(self)._CATALOG = d
        return self

    def get_rasterdataset(self, source, **kwargs):
        self.get_rasterdataset_calls.append({"source": source, **kwargs})
        # Return a dataset shaped to satisfy the function body.
        vars_ = kwargs.get("variables", ["precip"])
        return _FakeDataset(vars_)


def _fake_temp(*_args, **_kwargs):
    return _FakeDataArray("temp_corrected")


# Heavy-dep stubbing (P3-2a hardening). The pre-P3-2a version installed these
# stubs via sys.modules.setdefault(...) at import time, which silently no-ops
# whenever ANY earlier-collected test module has already imported the real
# geopandas/hydromt — an import-order landmine (the P3-2a parity tests, which
# exercise real hydromt, tripped it). Instead, an autouse fixture below
# monkeypatches the bindings on the source module object itself
# (ehc.gpd / ehc.hydromt / ehc.temp), which is order-independent and reverts
# per test. The fake classes and every test assertion are unchanged.

_geopandas_stub = types.SimpleNamespace(
    read_file=lambda fn: types.SimpleNamespace(
        geometry=types.SimpleNamespace(total_bounds=(0.0, 0.0, 1.0, 1.0)),
    ),
)


# Note: dask is NOT stubbed because pandas does a lazy `import dask` and
# accesses dask.__spec__ during type checks. A SimpleNamespace stub there
# breaks unrelated test files that import pandas during collection. dask
# is in the env (pixi-installed), and `DaskProgress` — which replaced
# `dask.diagnostics.ProgressBar` here — is a cheap callback that draws
# nothing when no compute runs, so let the real one run.

from blueearth_cst.climate_analysis import (  # noqa: E402
    extract_historical_climate as ehc,
)


@pytest.fixture(autouse=True)
def _stub_heavy_deps(monkeypatch):
    """Rebind the source module's heavy deps to the fakes, per test.

    Patching ehc's own attribute bindings (not sys.modules) works whether or
    not the real packages are already imported elsewhere in the session, and
    monkeypatch reverts them after each test. Individual tests may layer
    further patches on top (e.g. a narrower DataCatalog).
    """
    monkeypatch.setattr(ehc, "gpd", _geopandas_stub)
    monkeypatch.setattr(
        ehc, "hydromt", types.SimpleNamespace(DataCatalog=_RecordingDataCatalog)
    )
    monkeypatch.setattr(ehc, "temp", _fake_temp)


@pytest.fixture
def fake_era5_catalog():
    _RecordingDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/data/era5.nc",
            "driver": {"name": "netcdf", "options": {"chunks": "default"}},
        }
    }
    yield
    _RecordingDataCatalog._CATALOG = {}


@pytest.fixture
def fake_era5_string_driver_catalog():
    """Source where 'driver' is a bare string (older catalog format)."""
    _RecordingDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/data/era5.nc",
            "driver": "netcdf",
        }
    }
    yield
    _RecordingDataCatalog._CATALOG = {}


@pytest.fixture
def fake_chirps_catalog():
    _RecordingDataCatalog._CATALOG = {
        "chirps_global": {
            "data_type": "RasterDataset",
            "uri": "/data/chirps.nc",
            "driver": {"name": "netcdf"},
        },
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/data/era5.nc",
            "driver": {"name": "netcdf"},
        },
        # The DEFAULT hydrography entry, which is what the branch reads when the
        # caller passes no `hydrography`. It was `merit_hydro` until 2026-08-16,
        # a source nothing else in the toolbox names.
        "merit_hydro_ihu": {
            "data_type": "RasterDataset",
            "uri": "/data/merit_ihu.nc",
            "driver": {"name": "netcdf"},
        },
        "era5_orography": {
            "data_type": "RasterDataset",
            "uri": "/data/era5_oro.nc",
            "driver": {"name": "netcdf"},
        },
    }
    yield
    _RecordingDataCatalog._CATALOG = {}


def _last_catalog():
    return _RecordingDataCatalog._LAST_INSTANCE


def test_era5_path_requests_full_seven_variable_stack(tmp_path, fake_era5_catalog):
    region = tmp_path / "region.geojson"
    region.write_text("{}")  # contents irrelevant; geopandas stub ignores it
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    calls = _last_catalog().get_rasterdataset_calls
    assert len(calls) == 1
    assert calls[0]["source"] == "era5"
    assert sorted(calls[0]["variables"]) == sorted(
        ["precip", "temp", "temp_min", "temp_max", "kin", "kout", "press_msl"]
    )


@pytest.mark.parametrize(
    "requested,expected",
    [
        (("1950-01-01", "2020-12-31"), ("1950-01-02", "2020-12-31")),
        (("2000-01-01", "2023-12-31"), ("2000-01-01", "2023-02-01")),
        (("1940-01-01", "2024-12-31"), ("1950-01-02", "2023-02-01")),
    ],
)
def test_era5_request_uses_catalog_overlap_and_warns(
    tmp_path, capsys, requested, expected
):
    """Late first days and early ends narrow the read, never extend the request."""
    _RecordingDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/data/era5.zarr",
            "driver": {"name": "raster_xarray", "options": {}},
            "metadata": {
                "extent": {
                    "time_range": {
                        "start": "1950-01-02",
                        "end": "2023-02-01",
                    }
                }
            },
        }
    }
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="era5",
        starttime=requested[0],
        endtime=requested[1],
    )
    assert _last_catalog().get_rasterdataset_calls[0]["time_range"] == expected
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert f"requested {requested[0]}..{requested[1]}" in output
    assert f"extracting {expected[0]}–{expected[1]}" in output


def test_catalog_window_with_no_overlap_fails_before_read(fake_era5_catalog):
    _RecordingDataCatalog._CATALOG["era5"]["metadata"] = {
        "extent": {"time_range": {"start": "1980-01-01", "end": "2020-12-31"}}
    }
    catalog = _RecordingDataCatalog()
    with pytest.raises(ValueError, match="era5.*no overlap.*1950-01-01..1970-12-31"):
        ehc._read_source(
            catalog,
            "era5",
            requested=("1950-01-01", "1970-12-31"),
            time_range=("1950-01-01", "1970-12-31"),
        )
    assert catalog.get_rasterdataset_calls == []


@pytest.mark.parametrize(
    "bounds",
    [
        {"start": "invalid", "end": "2020-12-31"},
        {"start": "NaT", "end": "2020-12-31"},
        {"start": "2020-12-31", "end": "1980-01-01"},
    ],
)
def test_invalid_catalog_window_is_not_treated_as_missing_coverage(
    fake_era5_catalog, bounds
):
    _RecordingDataCatalog._CATALOG["era5"]["metadata"] = {
        "extent": {"time_range": bounds}
    }
    catalog = _RecordingDataCatalog()
    with pytest.raises(ValueError, match="invalid catalog metadata.extent.time_range"):
        ehc._read_source(catalog, "era5", requested=("2000-01-01", "2020-12-31"))
    assert catalog.get_rasterdataset_calls == []


def test_catalog_covering_the_window_preserves_the_read_and_stays_quiet(
    fake_era5_catalog, capsys
):
    _RecordingDataCatalog._CATALOG["era5"]["metadata"] = {
        "extent": {"time_range": {"start": "1980-01-01", "end": "2020-12-31"}}
    }
    catalog = _RecordingDataCatalog()
    requested = ("2000-01-01T00:00:00", "2016-12-31T00:00:00")
    ehc._read_source(catalog, "era5", requested=requested, time_range=requested)
    assert catalog.get_rasterdataset_calls[0]["time_range"] == requested
    assert "WARNING" not in capsys.readouterr().out


def test_era5_path_aligns_driver_options_chunks_to_the_store(
    tmp_path, fake_era5_catalog
):
    """The read defers to the store's ENCODED chunking, spelled `{}`.

    Not a style preference over `"auto"`: dask's auto sizing targets a byte
    budget and so merges several on-disk chunks into one, which over a network
    store drags every merged chunk across the wire to slice one basin out of it.
    Measured 2026-09-18 on `deltares_data_pdrive.yml`, that was 4x the bytes and
    46 min for a 76 KB store. `{}` also beats naming sizes here, because the
    catalog's own spec can be misaligned with the file (era5's `longitude: 240`
    splits a stored 480-wide chunk) and that catalog is vendored upstream-
    verbatim and hash-pinned, so it cannot be corrected in place.
    """
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    # The function calls from_dict on a patched catalog. Inspect what was set.
    patched = _RecordingDataCatalog._CATALOG["era5"]
    assert patched["driver"]["options"]["chunks"] == {}
    # An empty dict is falsy, and the value has to SURVIVE as one: hydromt dumps
    # driver options with `exclude_unset=True`, so `{}` reaches `open_mfdataset`
    # only because it was set explicitly. A truthiness test here would pass just
    # as well against the key having been dropped.
    assert "chunks" in patched["driver"]["options"]


def test_era5_path_normalizes_string_driver_to_dict(
    tmp_path, fake_era5_string_driver_catalog
):
    """When the source's 'driver' is a bare string, the function must wrap it
    in {'name': <str>} before adding options.chunks. Regression for hydromt
    1.x catalog format support."""
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    patched = _RecordingDataCatalog._CATALOG["era5"]
    assert isinstance(patched["driver"], dict)
    assert patched["driver"]["name"] == "netcdf"
    assert patched["driver"]["options"]["chunks"] == {}


def test_chirps_global_branch_requests_precip_only_from_chirps(
    tmp_path, fake_chirps_catalog
):
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    calls = _last_catalog().get_rasterdataset_calls
    chirps_calls = [c for c in calls if c["source"] == "chirps_global"]
    era5_calls = [c for c in calls if c["source"] == "era5"]

    assert len(chirps_calls) == 1
    assert chirps_calls[0]["variables"] == ["precip"]
    # era5 fallback fetches the rest, but NOT precip.
    assert len(era5_calls) == 1
    assert "precip" not in era5_calls[0]["variables"]
    assert "temp" in era5_calls[0]["variables"]


def test_chirps_comparison_candidate_does_not_read_era5_or_orography(
    tmp_path, fake_chirps_catalog
):
    """A wf0-only candidate reads only the variable the source provides."""
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    oro_out = tmp_path / "orography.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        oro_out=oro_out,
        forcing_required=False,
    )

    calls = _last_catalog().get_rasterdataset_calls
    assert [(call["source"], call["variables"]) for call in calls] == [
        ("chirps_global", ["precip"])
    ]
    assert not oro_out.exists()


def _fake_netcdf4(monkeypatch, default=(67108864, 1000, 0.75)):
    """Record every `set_chunk_cache` the extraction performs.

    The decorator resolves `netCDF4` from the module globals at CALL time, so
    rebinding `ehc.netCDF4` is enough and no real HDF5 is involved.
    """
    state = {"cache": default}
    seen = []

    def set_chunk_cache(size, nelems, preemption):
        state["cache"] = (size, nelems, preemption)
        seen.append(state["cache"])

    monkeypatch.setattr(
        ehc,
        "netCDF4",
        types.SimpleNamespace(
            get_chunk_cache=lambda: state["cache"],
            set_chunk_cache=set_chunk_cache,
        ),
    )
    return state, seen


def test_chirps_branch_aligns_the_era5_chunks_to_the_store_too(
    tmp_path, fake_chirps_catalog
):
    """The chunk override covers BOTH arms, not just the era5 one.

    It lived inside the `else` arm until 2026-09-18, so the chirps branch read
    era5 for its other six variables at whatever the catalog declared --
    `longitude: 240` under `deltares_data_pdrive.yml`, which splits the stored
    480-wide chunk and costs a second decompression for any basin straddling
    the split.
    """
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    for name in ("chirps_global", "era5"):
        options = _RecordingDataCatalog._CATALOG[name]["driver"]["options"]
        assert options["chunks"] == {}
        # `{}` is falsy and has to SURVIVE as one; see the era5 test above.
        assert "chunks" in options


def test_aligning_chunks_skips_a_source_the_catalog_does_not_carry(
    tmp_path, fake_era5_catalog
):
    """A catalog with no era5 is not given one.

    The caller names every source the extraction MIGHT read, and only the
    chirps arm reads era5 -- so the era5 arm passes a name its own catalog is
    free to be the only entry for.
    """
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    assert set(_RecordingDataCatalog._CATALOG) == {"era5"}


def test_the_extraction_bounds_the_hdf5_chunk_cache_and_restores_it(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """netcdf-c's 64 MiB per-variable cache is pure cost for a one-touch read.

    Every chunk is read once, sliced to the basin and dropped, so the cache
    retains bytes nothing asks for again -- once per (file, variable), which
    against yearly files makes resident memory grow with the WINDOW. Measured
    2026-09-18: 2.50 GB held over a six-year era5 read, 394 MB bounded.
    """
    state, seen = _fake_netcdf4(monkeypatch)
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    assert seen[0] == (ehc._HDF5_CHUNK_CACHE_BYTES, 1000, 0.75)
    # Restored, so a process that calls this for one extraction does not
    # inherit the bound for every other netCDF it touches.
    assert state["cache"] == (67108864, 1000, 0.75)


def test_an_already_smaller_chunk_cache_is_left_alone(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """The bound is a ceiling, not a setting -- an operator's smaller choice wins."""
    smaller = (65536, 1000, 0.75)
    _state, seen = _fake_netcdf4(monkeypatch, default=smaller)
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    assert seen[0] == smaller


def test_the_chunk_cache_is_restored_when_the_extraction_raises(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """A failed extraction must not leave the bound behind on the process."""
    state, _seen = _fake_netcdf4(monkeypatch)

    with pytest.raises(ValueError):
        ehc.prep_historical_climate(
            region_fn=tmp_path / "region.geojson",
            bbox=(0.0, 0.0, 1.0, 1.0),  # both given: the guard raises immediately
            fn_out=tmp_path / "out.nc",
            data_libs="dummy.yml",
            clim_source="era5",
            starttime="2010-01-01T00:00:00",
            endtime="2010-12-31T00:00:00",
        )

    assert state["cache"] == (67108864, 1000, 0.75)


def _grid_ds(y_name, x_name):
    """A minimal 2-D grid dataset spelled with the given coord names."""
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {"precip": ((y_name, x_name), np.zeros((2, 3), dtype="float32"))},
        coords={y_name: [1.0, 2.0], x_name: [1.0, 2.0, 3.0]},
    )


@pytest.mark.parametrize(
    "y_name,x_name",
    [("lat", "lon"), ("y", "x"), ("latitude", "longitude")],
)
def test_grid_names_normalize_to_the_store_spelling(y_name, x_name):
    """Every source spelling lands on the store's `latitude`/`longitude`.

    WG-1 pins the store's dims and `basin_cells.csv` writes the same two names,
    so a source that spells its grid differently (CHIRPS uses `lat`/`lon`) must
    be renamed at the read rather than handled by each consumer.
    """
    out = ehc._normalize_grid_names(_grid_ds(y_name, x_name))

    assert "latitude" in out.dims and "longitude" in out.dims
    assert y_name not in out.dims or y_name == "latitude"
    assert x_name not in out.dims or x_name == "longitude"
    # Values ride along unchanged -- this is a rename, not a reindex.
    assert [float(v) for v in out["latitude"].values] == [1.0, 2.0]
    assert [float(v) for v in out["longitude"].values] == [1.0, 2.0, 3.0]


def test_grid_names_normalization_preserves_spatial_ref_and_attrs():
    """The CRS coord and global attrs must survive the rename.

    A rename that dropped `spatial_ref` or `crs` would still read fine and would
    fail the WG-1 seam validator -- the failure mode this test exists for.
    """
    ds = _grid_ds("lat", "lon")
    ds = ds.assign_coords(spatial_ref=0)
    ds.attrs["crs"] = 4326

    out = ehc._normalize_grid_names(ds)

    assert "spatial_ref" in out.coords
    assert out.attrs["crs"] == 4326


def test_chirps_branch_reads_its_dem_from_the_default_hydrography(
    tmp_path, fake_chirps_catalog
):
    """No `hydrography` argument -> the toolbox default, not a branch-local name.

    Pins the 2026-08-16 change away from a hardcoded `merit_hydro`: that entry
    is named nowhere else in the toolbox, so the branch demanded a staged
    dataset a working project need not have.
    """
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    sources = [c["source"] for c in _last_catalog().get_rasterdataset_calls]
    assert ehc.DEFAULT_HYDROGRAPHY in sources
    assert "merit_hydro" not in sources


def test_chirps_branch_dem_follows_the_configured_hydrography(
    tmp_path, fake_chirps_catalog
):
    """The rule passes `shared.basin.hydrography`; the DEM read must follow it.

    Otherwise a project that delineates its basin on one elevation source would
    lapse-correct its temperature against another.
    """
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        hydrography="merit_hydro_1k",
    )

    dem_calls = [
        c
        for c in _last_catalog().get_rasterdataset_calls
        if c["source"] == "merit_hydro_1k"
    ]
    assert len(dem_calls) == 1
    assert dem_calls[0]["variables"] == ["elevtn"]


def test_era5_branch_reads_no_dem_at_all(tmp_path, fake_era5_catalog):
    """`hydrography` is chirps-branch-only; era5 extracts without any DEM read.

    Guards the docstring's "Ignored outside the chirps branch" against a future
    edit that hoists the DEM read out of the branch.
    """
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        hydrography="merit_hydro_1k",
    )

    sources = [c["source"] for c in _last_catalog().get_rasterdataset_calls]
    assert sources == ["era5"]


def test_starttime_and_endtime_passed_to_get_rasterdataset(tmp_path, fake_era5_catalog):
    """The function MUST pass its starttime/endtime params through to hydromt.
    Note: this tests the FUNCTION's behavior, not the Snakefile rule that
    invokes it. The rule-level bug (run_stress_test.smk hardcoding
    dates) is separately tracked in dev/tasks/ R5 and belongs to an
    integration test, not this unit."""
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="1995-06-15T00:00:00",
        endtime="2005-06-15T00:00:00",
    )

    calls = _last_catalog().get_rasterdataset_calls
    assert calls[0]["time_range"] == (
        "1995-06-15T00:00:00",
        "2005-06-15T00:00:00",
    )


# --- Layer B: what the staged source ACTUALLY delivers -----------------------
# The parse-time half (what the config REQUESTS) is
# tests/test_validate_historical_window.py; these cover what arrived, which is
# knowable only here.
#
# `shared.historical_window` is a CEILING, not a demand (2026-08-16): a source
# that cannot fill it is extracted over the widest span it holds inside it and
# the narrowing is REPORTED. Reporting goes through `log_row` -> stdout -> the
# rule's log part, so these read capsys rather than the warnings filter.


def _run_with_span(
    monkeypatch,
    tmp_path,
    time_size,
    catalog_cls,
    *,
    enforce_min_years=True,
    time_start="1980-01-01",
):
    """Drive prep_historical_climate against a fake catalog of ``time_size``
    YEARLY steps from ``time_start``."""

    class _SpanDataCatalog(catalog_cls):
        def get_rasterdataset(self, source, **kwargs):
            self.get_rasterdataset_calls.append({"source": source, **kwargs})
            return _FakeDataset(
                kwargs.get("variables", ["precip"]),
                time_size=time_size,
                time_start=time_start,
            )

    monkeypatch.setattr(ehc.hydromt, "DataCatalog", _SpanDataCatalog)
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2000-01-01T00:00:00",
        endtime="2020-12-31T00:00:00",
        enforce_min_years=enforce_min_years,
    )


def test_a_narrowed_window_is_reported_not_raised(
    tmp_path, fake_era5_catalog, monkeypatch, capsys
):
    """20 yearly steps from 1980 end in ~1999, missing most of a 2000..2020 ask.

    It still clears the 16-year floor, so this is purely the narrowing case: the
    extraction proceeds and says what it actually got.
    """
    _run_with_span(monkeypatch, tmp_path, 20, _RecordingDataCatalog)
    out = capsys.readouterr().out
    assert "era5: requested 2000-01-01..2020-12-31" in out
    assert "staged data cover only part of climate.window" in out
    assert "widest available span" in out


def test_a_covered_window_reports_the_span_without_the_narrowing_line(
    tmp_path, fake_era5_catalog, monkeypatch, capsys
):
    """The delivered span is ALWAYS logged; only the narrowing line is
    conditional, so its presence stays informative rather than background."""
    _run_with_span(monkeypatch, tmp_path, 100, _RecordingDataCatalog)
    out = capsys.readouterr().out
    assert "era5: requested 2000-01-01..2020-12-31; extracted period" in out
    assert "does not cover the full" not in out


def test_short_extraction_raises_naming_the_unified_floor(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """Ten yearly steps = ~9 years, under the 16-year floor.

    The floor survives the 2026-08-16 relaxation for the source that FEEDS the
    pipeline. Before the unified floor (owner ruling 2026-08-01) this failed
    either at rule 1.11 with MissingOutputException or a whole workflow away
    inside weathergenr.
    """
    with pytest.raises(ValueError) as excinfo:
        _run_with_span(monkeypatch, tmp_path, 10, _RecordingDataCatalog)
    message = str(excinfo.value)
    assert f"{MIN_HISTORICAL_YEARS}-year forcing requirement" in message
    assert "climate.window" in message
    assert "era5" in message
    assert "weathergenr" in message


def test_a_single_timestep_raises_too(tmp_path, fake_era5_catalog, monkeypatch):
    """The degenerate end of the same check -- no separate code path."""
    with pytest.raises(
        ValueError, match=f"{MIN_HISTORICAL_YEARS}-year forcing requirement"
    ):
        _run_with_span(monkeypatch, tmp_path, 1, _RecordingDataCatalog)


def test_a_relaxed_candidate_below_the_floor_warns_instead_of_raising(
    tmp_path, fake_era5_catalog, monkeypatch, capsys
):
    """wf0's extra candidate_sources end at a comparison figure.

    The floor exists for weathergenr, which never sees these -- so the same
    record that is fatal above is a logged warning here, and the message says
    what the consequence would be rather than what to fix.
    """
    _run_with_span(
        monkeypatch, tmp_path, 10, _RecordingDataCatalog, enforce_min_years=False
    )
    out = capsys.readouterr().out
    assert f"{MIN_HISTORICAL_YEARS}-year forcing requirement" in out
    assert "comparison candidate only" in out
    assert "WARNING" in out


def test_relaxing_the_floor_still_writes_the_store(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """The point of relaxing: a short candidate produces figures rather than
    stopping the workflow."""
    _run_with_span(
        monkeypatch, tmp_path, 10, _RecordingDataCatalog, enforce_min_years=False
    )
    assert (tmp_path / "out.nc").parent.exists()
    # The fake records to_netcdf calls rather than writing; the absence of an
    # exception plus the recorded call is what "the store was written" means here.


def test_the_store_write_runs_under_the_synchronous_scheduler(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """The store write must never run on dask's THREAD POOL.

    netCDF4/HDF5 takes a process-global lock and xarray funnels every netCDF
    touch through it, so a graph that both reads netCDF and writes netCDF
    deadlocks: writers and readers each hold one sub-lock of a `CombinedLock`
    and wait for the other. Measured 2026-08-18, when wf0 parked on the chirps
    store at 94.2% with every worker in `xarray/backends/locks.py` and the
    process burning 0.08 s of CPU per 20 s of wall clock.

    This is a UNIT guard for a defect only integration could otherwise catch,
    and it is worth having because the deadlock is source-shaped: era5 reads
    zarr, which takes no such lock, so an all-era5 run is green and proves
    nothing about the netCDF-backed sources that can actually reach it.
    """
    _FakeDataset._COMPUTE_SCHEDULERS.clear()

    _run_with_span(monkeypatch, tmp_path, 40, _RecordingDataCatalog)

    assert _FakeDataset._COMPUTE_SCHEDULERS == ["synchronous"]
    # And scoped, not pinned process-wide: everything downstream of this rule
    # still gets the threaded scheduler it expects.
    assert dask.config.get("scheduler", None) != "synchronous"


def test_zero_overlap_names_the_source_and_the_window(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """The ONE shortfall no widest-possible-range can rescue.

    hydromt's own NoDataException names neither the source nor the window that
    missed, which reads as a code defect rather than a config one.
    """

    class _EmptyDataCatalog(_RecordingDataCatalog):
        def get_rasterdataset(self, source, **kwargs):
            raise ehc.NoDataException("No data left after temporal slicing.")

    monkeypatch.setattr(ehc.hydromt, "DataCatalog", _EmptyDataCatalog)
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    with pytest.raises(ValueError) as excinfo:
        ehc.prep_historical_climate(
            region_fn=region,
            fn_out=tmp_path / "out.nc",
            data_libs="dummy.yml",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",
        )
    message = str(excinfo.value)
    assert "'era5'" in message
    assert "2000-01-01..2020-12-31" in message
    assert "overlaps it nowhere" in message


# --- the chirps branch assembles ONE store from TWO sources ------------------


def _chirps_run(monkeypatch, tmp_path, chirps_start, era5_start, span=40):
    """Drive the chirps branch with the two sources starting in different years."""

    class _MismatchedCatalog(_RecordingDataCatalog):
        def get_rasterdataset(self, source, **kwargs):
            self.get_rasterdataset_calls.append({"source": source, **kwargs})
            start = chirps_start if source == "chirps_global" else era5_start
            return _FakeDataset(
                kwargs.get("variables", ["precip"]),
                time_size=span,
                time_start=start,
            )

    monkeypatch.setattr(ehc.hydromt, "DataCatalog", _MismatchedCatalog)
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="1980-01-01T00:00:00",
        endtime="2019-12-31T00:00:00",
        oro_out=tmp_path / "orography.nc",
    )


def test_chirps_branch_clips_both_sources_to_their_overlap(
    tmp_path, fake_chirps_catalog, monkeypatch, capsys
):
    """The store's window is what BOTH sources cover.

    Without the clip, `ds[var] = ds_clim[var]` REINDEXES era5 onto the longer
    chirps axis and NaN-fills the non-overlap -- a store carrying real
    precipitation beside all-NaN temperature, which passes WG-1 and reaches
    weathergenr's area average twenty rules later.
    """
    _chirps_run(
        monkeypatch, tmp_path, chirps_start="1981-01-01", era5_start="1990-01-01"
    )
    out = capsys.readouterr().out
    assert "the store takes their overlap 1990-01-01" in out
    # And the coverage line reports the CLIPPED record, not chirps' own longer
    # one. The fake steps 365 days at a time, so the first surviving chirps
    # timestamp lands near but not on the overlap boundary -- the year is the
    # honest assertion, the exact date would only pin the fake's arithmetic.
    delivered = re.search(r"extracted period (\d{4})-\d\d-\d\d", out)
    assert delivered is not None, out
    assert int(delivered.group(1)) >= 1990


def test_chirps_branch_stays_quiet_when_the_two_sources_agree(
    tmp_path, fake_chirps_catalog, monkeypatch, capsys
):
    """No overlap line when there is nothing to reconcile."""
    _chirps_run(
        monkeypatch, tmp_path, chirps_start="1981-01-01", era5_start="1981-01-01"
    )
    assert "takes their overlap" not in capsys.readouterr().out


def test_chirps_branch_refuses_a_pair_that_never_overlaps(
    tmp_path, fake_chirps_catalog, monkeypatch
):
    """chirps supplies precipitation only; era5 supplies everything else.

    Two records that miss each other entirely cannot be assembled into one
    store, and saying so beats writing seven variables of which six are NaN.
    """
    with pytest.raises(ValueError, match="do not overlap"):
        _chirps_run(
            monkeypatch,
            tmp_path,
            chirps_start="1981-01-01",
            era5_start="2030-01-01",
            span=20,
        )


@pytest.mark.parametrize("source", ["era5", "chirps_global"])
def test_store_save_row_identifies_the_source(
    tmp_path, fake_chirps_catalog, capsys, source
):
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=tmp_path / "out.nc",
        data_libs="dummy.yml",
        clim_source=source,
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        forcing_required=False,
    )
    rows = capsys.readouterr().out.splitlines()
    saves = [
        row for row in rows if "writing extracted climate grid to netcdf" in row.lower()
    ]
    assert len(saves) == 1
    assert saves[0].endswith(
        f" - extract - {source}: writing extracted climate grid to netCDF"
    )


def test_region_source_is_stamped_relative_to_the_output(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """The same extraction in another project folder must be byte-identical."""
    region = tmp_path / "data/spatial/geoms/region.geojson"
    region.parent.mkdir(parents=True)
    region.write_text("{}")
    out_nc = tmp_path / "data/climate/historical/era5/out.nc"
    out_nc.parent.mkdir(parents=True)
    written = []
    original = _FakeDataset.to_netcdf

    def recording(self, fn, **kwargs):
        written.append(dict(self.attrs))
        return original(self, fn, **kwargs)

    monkeypatch.setattr(_FakeDataset, "to_netcdf", recording)
    ehc.prep_historical_climate(
        region_fn=None,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        bbox=(0.0, 0.0, 1.0, 1.0),
        region_source=region,
    )
    assert written[-1]["region_source"] == "../../../spatial/geoms/region.geojson"
