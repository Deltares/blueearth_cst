"""Unit tests for blueearth_cst/climate_analysis/extract_historical_climate.py.

This module is heavily coupled to hydromt I/O; we test the function's
configuration logic (driver options, variable lists, clim_source
branching) and skip the deeper reprojection paths. The truncation
warning xfail captures the R3 followup bug from dev/tasks/.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

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
    def __init__(self, name):
        self.name = name
        self.raster = _FakeDataArrayRaster()


class _FakeDataset:
    """Quacks enough like an xarray Dataset for prep_historical_climate."""

    def __init__(self, vars_, time_size=None):
        self._vars = list(vars_)
        self.raster = _FakeRasterAccessor(vars_)
        # A real (yearly) time axis from 1980 so the truncation check has a
        # coverage span to compare. size drives the span: default 100 -> ~1980
        # to 2079 (covers any test request); a narrow catalog (size 10) -> only
        # ~1980-1989, shorter than a 2000-2020 request.
        n = time_size or 100
        self.time = types.SimpleNamespace(
            size=n,
            values=np.datetime64("1980-01-01")
            + np.arange(n) * np.timedelta64(365, "D"),
        )
        self._tonetcdf_calls = []
        # ADR 0003: the producer stamps the extent provenance on the extraction
        # (region_bbox / region_geojson_sha256 / region_source), so the fake has
        # to carry an attrs mapping like a real Dataset.
        self.attrs = {}

    def __getitem__(self, key):
        return _FakeDataArray(key)

    def __setitem__(self, key, value):
        if key not in self._vars:
            self._vars.append(key)
            self.raster.vars.append(key)

    def to_dataset(self):
        return self

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
# is in the env (pixi-installed), and dask.diagnostics.ProgressBar is a
# cheap context manager — let the real one run.

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


def test_era5_path_patches_driver_options_chunks_auto(tmp_path, fake_era5_catalog):
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
    assert patched["driver"]["options"]["chunks"] == "auto"


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
    assert patched["driver"]["options"]["chunks"] == "auto"


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


def test_warns_when_extracted_window_is_shorter_than_requested(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """Drive a fake DataCatalog whose datasets have a narrow time span.
    prep_historical_climate emits a truncation warning that this test verifies.
    monkeypatch updates ehc.hydromt.DataCatalog directly because `import hydromt`
    in the source module binds at import time — rewriting sys.modules['hydromt']
    later does not affect that binding."""

    class _NarrowDataCatalog(_RecordingDataCatalog):
        def get_rasterdataset(self, source, **kwargs):
            self.get_rasterdataset_calls.append({"source": source, **kwargs})
            # 20 yearly steps from 1980: ends ~1999, so it falls short of the
            # 2000..2020 request (the advisory) while still clearing the
            # 16-year floor (which would otherwise raise before the warning).
            return _FakeDataset(kwargs.get("variables", ["precip"]), time_size=20)

    monkeypatch.setattr(ehc.hydromt, "DataCatalog", _NarrowDataCatalog)

    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ehc.prep_historical_climate(
            region_fn=region,
            fn_out=out_nc,
            data_libs="dummy.yml",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",  # 21 years requested, ds returns 10 timesteps
        )

    assert any(
        "truncat" in str(w.message).lower() or "shorter" in str(w.message).lower()
        for w in caught
    ), "Expected a warning about time-window truncation; got none."


# --- Layer B: the hard floor and the weathergenr advisory --------------------
# The parse-time half (what the config REQUESTS) is
# tests/test_validate_historical_window.py; these cover what the staged source
# ACTUALLY delivers, which is knowable only here.


def _run_with_span(monkeypatch, tmp_path, time_size, catalog_cls):
    """Drive prep_historical_climate against a fake catalog of ``time_size``
    YEARLY steps, returning the warnings it raised."""

    class _SpanDataCatalog(catalog_cls):
        def get_rasterdataset(self, source, **kwargs):
            self.get_rasterdataset_calls.append({"source": source, **kwargs})
            return _FakeDataset(
                kwargs.get("variables", ["precip"]), time_size=time_size
            )

    monkeypatch.setattr(ehc.hydromt, "DataCatalog", _SpanDataCatalog)
    region = tmp_path / "region.geojson"
    region.write_text("{}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ehc.prep_historical_climate(
            region_fn=region,
            fn_out=tmp_path / "out.nc",
            data_libs="dummy.yml",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",
        )
    return caught


def test_short_extraction_raises_naming_the_unified_floor(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """Ten yearly steps = ~9 years, under the 16-year floor.

    The UNIFIED floor (owner ruling 2026-08-01) makes this fatal in the
    producer. Before, it warned here and then failed either at rule 1.11 with
    MissingOutputException or a whole workflow away inside weathergenr.
    """
    with pytest.raises(ValueError) as excinfo:
        _run_with_span(monkeypatch, tmp_path, 10, _RecordingDataCatalog)
    message = str(excinfo.value)
    assert f"{ehc.MIN_HISTORICAL_YEARS}-year minimum" in message
    assert "historical_window" in message
    assert "era5" in message
    assert "weathergenr" in message


def test_a_single_timestep_raises_too(tmp_path, fake_era5_catalog, monkeypatch):
    """The degenerate end of the same check -- no separate code path."""
    with pytest.raises(ValueError, match=f"{ehc.MIN_HISTORICAL_YEARS}-year minimum"):
        _run_with_span(monkeypatch, tmp_path, 1, _RecordingDataCatalog)


def test_long_enough_extraction_is_silent(tmp_path, fake_era5_catalog, monkeypatch):
    """The default 100-year fake covers the request: no coverage warning at all,
    so the shortfall advisory stays meaningful rather than background noise."""
    caught = _run_with_span(monkeypatch, tmp_path, 100, _RecordingDataCatalog)
    noisy = [w for w in caught if "shorter than the requested" in str(w.message)]
    assert not noisy, [str(w.message) for w in noisy]
