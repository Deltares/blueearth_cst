from __future__ import annotations

import sys
import types
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
from shapely.geometry import Point

# NO `sys.modules.setdefault` STUBS HERE, and none may be re-added. This module
# used to install fake `geopandas`, `rasterio`, `rasterio.windows` and `xarray`
# at import time; `setdefault` mutates `sys.modules` for the whole pytest
# PROCESS and nothing restored them, so whichever module imported first decided
# what every later one saw. `pytest tests/test_stage_cmip6.py` alone was green
# and the full suite was green, but running the two staging modules together
# made three of test_stage_cmip6's tests fail against the fakes -- a false
# failure in exactly the subset run someone iterating on staging makes.
# stage_data imports fine without them (t2608191420).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import stage_data  # noqa: E402


class _FakeArray:
    def __init__(self, dims, sizes, encoding=None):
        self.dims = tuple(dims)
        self.sizes = dict(sizes)
        self.encoding = dict(encoding or {})
        self.chunks = None


class _FakeDataset:
    def __init__(self):
        self.data_vars = {
            "tp": _FakeArray(
                ("time", "lat", "lon"),
                {"time": 800, "lat": 5, "lon": 7},
                {
                    "chunks": (1, 721, 1440),
                    "compressor": "source-compressor",
                    "_FillValue": np.float32(-9999),
                },
            ),
            "station_id": _FakeArray(("station",), {"station": 3}),
        }

    def __getitem__(self, name):
        return self.data_vars[name]

    @property
    def variables(self):
        # `_zarr_subset_write_plan` -> `_strip_source_codecs` iterates
        # `ds.variables.values()` to drop inherited source codecs.
        return self.data_vars

    def chunk(self, chunks):
        for array in self.data_vars.values():
            array.chunks = tuple(
                tuple(
                    chunks.get(dim, array.sizes[dim])
                    for _ in range(
                        array.sizes[dim] // chunks.get(dim, array.sizes[dim])
                    )
                )
                + (
                    (array.sizes[dim] % chunks.get(dim, array.sizes[dim]),)
                    if array.sizes[dim] % chunks.get(dim, array.sizes[dim])
                    else ()
                )
                for dim in array.dims
            )
        return self


def test_zarr_subset_write_plan_rechunks_daily_meteo_subset() -> None:
    rechunked, encoding = stage_data._zarr_subset_write_plan(_FakeDataset())

    # Daily time dim (800) is split into ZARR_TIME_CHUNK-sized pieces; the two
    # spatial dims stay whole. Encoding chunks mirror the rechunking.
    assert rechunked["tp"].chunks == ((365, 365, 70), (5,), (7,))
    assert encoding["tp"]["chunks"] == (365, 5, 7)
    # The source zarr-v2 `compressor` codec is stripped (not carried into the
    # write encoding) so zarr 3.x can apply its own default codec — see
    # `_strip_source_codecs`. Only CF packing keys (ZARR_ENCODING_KEYS) survive.
    assert "compressor" not in encoding["tp"]
    assert encoding["tp"]["_FillValue"] == np.float32(-9999)
    assert "station_id" not in encoding


def test_raster_output_profile_uses_tiling_for_larger_geotiffs() -> None:
    profile = {
        "driver": "GTiff",
        "height": 2000,
        "width": 3000,
        "transform": "old",
        "blockxsize": 512,
        "blockysize": 512,
    }

    out = stage_data._raster_output_profile(
        profile,
        height=600,
        width=700,
        transform="new",
    )

    assert out["height"] == 600
    assert out["width"] == 700
    assert out["transform"] == "new"
    assert out["compress"] == "deflate"
    assert out["tiled"] is True
    assert out["blockxsize"] == 256
    assert out["blockysize"] == 256


def test_raster_output_profile_keeps_tiny_geotiffs_striped() -> None:
    out = stage_data._raster_output_profile(
        {"driver": "GTiff"},
        height=12,
        width=20,
        transform="new",
    )

    assert out["tiled"] is False
    assert "blockxsize" not in out
    assert "blockysize" not in out


def test_validate_lonlat_crs_rejects_projected_crs() -> None:
    crs = types.SimpleNamespace(is_geographic=False, to_string=lambda: "EPSG:3857")

    try:
        stage_data._validate_lonlat_crs(crs, "raster", Path("source.tif"))
    except ValueError as exc:
        assert "bbox is lon/lat" in str(exc)
        assert "EPSG:3857" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_lonlat_crs_accepts_epsg_4326_string() -> None:
    stage_data._validate_lonlat_crs("EPSG:4326", "vector", Path("source.gpkg"))


def test_vector_read_kwargs_include_optional_columns() -> None:
    assert stage_data._vector_read_kwargs((1, 2, 3, 4), ["geometry", "id"]) == {
        "bbox": (1, 2, 3, 4),
        "columns": ["geometry", "id"],
    }


@pytest.mark.parametrize("suffix,driver", [(".gpkg", "GPKG"), (".fgb", "FlatGeobuf")])
def test_vector_staging_preserves_format_and_values(tmp_path, suffix, driver):
    source = tmp_path / f"source{suffix}"
    target = tmp_path / "local" / source.name
    frame = gpd.GeoDataFrame(
        {"id": [1, 2]}, geometry=[Point(0, 0), Point(5, 5)], crs=4326
    )
    frame.to_file(source, driver=driver)

    assert (
        stage_data.subset_vector(source, target, (-1, -1, 1, 1))[0]
        == stage_data.WRITTEN
    )
    assert pyogrio.read_info(target)["driver"] == driver
    result = gpd.read_file(target)
    assert result["id"].tolist() == [1]
    assert result.geometry.iloc[0].equals(Point(0, 0))
    assert result.crs == frame.crs


@pytest.mark.parametrize("suffix,driver", [(".gpkg", "GPKG"), (".fgb", "FlatGeobuf")])
def test_vector_staging_can_retain_an_empty_source(tmp_path, suffix, driver):
    source = tmp_path / f"source{suffix}"
    target = tmp_path / "local" / source.name
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(5, 5)], crs=4326).to_file(
        source, driver=driver
    )
    bbox = (-1, -1, 1, 1)
    assert stage_data.subset_vector(source, target, bbox)[0] == stage_data.SKIPPED
    assert not target.exists()
    assert stage_data.subset_vector(source, target, bbox, allow_empty=True) == (
        stage_data.WRITTEN,
        "0 features",
    )
    result = gpd.read_file(target)
    assert result.empty
    assert "id" in result.columns
    assert result.crs.to_epsg() == 4326
    assert pyogrio.read_info(target)["driver"] == driver
    assert (
        stage_data.subset_vector(source, target, bbox, allow_empty=True)[0]
        == stage_data.EXISTS
    )
    # An older fake-FGB manifest must not certify the new writer's output.
    if suffix == ".fgb":
        manifest = stage_data._read_manifest(target)
        manifest.pop("vector_format")
        stage_data._write_manifest(target, manifest)
        assert (
            stage_data.subset_vector(source, target, bbox, allow_empty=True)[0]
            == stage_data.WRITTEN
        )


def test_raster_glob_workers_are_bounded_and_configurable() -> None:
    assert stage_data._raster_glob_workers({"workers": 2}, file_count=10) == 2
    assert stage_data._raster_glob_workers({}, file_count=10) == 4
    assert stage_data._raster_glob_workers({}, file_count=2) == 1


def test_netcdf_glob_type_dispatches_to_netcdf_glob_stager(monkeypatch) -> None:
    # A `netcdf_glob` entry must route to `_stage_netcdf_glob` (the per-file
    # xarray path) and NOT fall through to the SUBSETTERS unknown-type failure.
    calls = []
    monkeypatch.setattr(
        stage_data,
        "_stage_netcdf_glob",
        lambda entry, name, src, dst, bbox, report: calls.append(
            (name, str(src), str(dst))
        ),
    )
    report = stage_data.RunReport()
    entry = {
        "name": "chirps",
        "type": "netcdf_glob",
        "path": "meteo/chirps_africa_daily_v2.0",
    }
    stage_data._stage_dataset(entry, Path("/src"), Path("/dst"), (0, 0, 1, 1), report)

    assert len(calls) == 1
    name, src, dst = calls[0]
    assert name == "chirps"
    assert src.replace("\\", "/").endswith("meteo/chirps_africa_daily_v2.0")
    assert dst.replace("\\", "/").endswith("meteo/chirps_africa_daily_v2.0")
    assert all(status != stage_data.FAILED for status, *_ in report.results)


def test_clip_window_intersects_request_with_natural_span() -> None:
    cw = stage_data._clip_window
    # Interior full year: a wider request leaves the window unchanged (-> skip).
    assert cw(["2005-01-01", "2005-12-31"], ["1990-01-01", "2020-12-31"]) == [
        "2005-01-01",
        "2005-12-31",
    ]
    # Left-boundary partial year: request start clips into the file.
    assert cw(["2000-01-01", "2000-12-31"], ["2000-06-01", "2020-12-31"]) == [
        "2000-06-01",
        "2000-12-31",
    ]
    # Out-of-range file: inverted window (start > end) -> "no time overlap".
    lo, hi = cw(["1995-01-01", "1995-12-31"], ["2000-01-01", "2020-12-31"])
    assert lo > hi
    # No request window: the file's full natural span.
    assert cw(["2005-01-01", "2005-12-31"], None) == ["2005-01-01", "2005-12-31"]


def test_time_cover_fresh_skips_interior_but_restages_on_real_change(tmp_path) -> None:
    dst = tmp_path / "CHIRPS_rainfall_2005.nc"
    dst.write_bytes(b"x")  # a time_cover output must exist to be considered fresh
    natural = ["2005-01-01", "2005-12-31"]
    fp0 = stage_data._fingerprint(
        src=Path("s/2005.nc"),
        bbox=(0, 0, 2, 2),
        time_range=["2000-01-01", "2010-12-31"],
        variables=None,
    )
    stage_data._write_manifest(
        dst,
        {
            **fp0,
            "natural_time": natural,
            "clip_time": stage_data._clip_window(natural, fp0["time_range"]),
        },
    )

    def fp(**over):
        base = dict(
            src=Path("s/2005.nc"),
            bbox=(0, 0, 2, 2),
            time_range=["1990-01-01", "2020-12-31"],
            variables=None,
        )
        return stage_data._fingerprint(**{**base, **over})

    # Widened request, interior year unchanged -> fresh (skip, no rewrite).
    assert stage_data._time_cover_fresh(dst, fp()) is True
    # Request clips into the file's own span -> stale (must restage exactly).
    assert (
        stage_data._time_cover_fresh(dst, fp(time_range=["2005-06-01", "2020-12-31"]))
        is False
    )
    # Different bbox or variables -> stale regardless of time.
    assert stage_data._time_cover_fresh(dst, fp(bbox=(0, 0, 3, 3))) is False
    assert stage_data._time_cover_fresh(dst, fp(variables=["precip"])) is False


def test_time_cover_fresh_requires_existing_output(tmp_path) -> None:
    # No output file on disk -> never fresh even with a matching manifest.
    dst = tmp_path / "missing.nc"
    assert (
        stage_data._time_cover_fresh(
            dst,
            stage_data._fingerprint(
                src=Path("s.nc"),
                bbox=(0, 0, 1, 1),
                time_range=None,
                variables=None,
            ),
        )
        is False
    )


def test_reusable_requires_matching_src_and_bbox(tmp_path) -> None:
    dst = tmp_path / "store.nc"
    dst.write_bytes(b"x")
    fp = stage_data._fingerprint(
        src=Path("s/f.nc"),
        bbox=(0, 0, 2, 2),
        time_range=["2000-01-01", "2010-12-31"],
        variables=["a"],
    )
    stage_data._write_manifest(dst, fp)

    def fp2(**over):
        base = dict(
            src=Path("s/f.nc"),
            bbox=(0, 0, 2, 2),
            time_range=["1990-01-01", "2020-12-31"],
            variables=["a", "b"],
        )
        return stage_data._fingerprint(**{**base, **over})

    # Same src+bbox but different time/vars -> reusable (those are the deltas).
    assert stage_data._reusable(dst, fp2()) is True
    # Changed bbox or src -> not reusable (must full-restage).
    assert stage_data._reusable(dst, fp2(bbox=(0, 0, 3, 3))) is False
    assert stage_data._reusable(dst, fp2(src=Path("s/other.nc"))) is False


def test_reusable_false_without_output_or_manifest(tmp_path) -> None:
    fp = stage_data._fingerprint(src=Path("s.nc"), bbox=(0, 0, 1, 1))
    assert stage_data._reusable(tmp_path / "missing.nc", fp) is False
    # Output present but no manifest -> not reusable.
    (tmp_path / "orphan.nc").write_bytes(b"x")
    assert stage_data._reusable(tmp_path / "orphan.nc", fp) is False


def test_write_zarr_retries_transient_permission_error(tmp_path, monkeypatch) -> None:
    # zarr-v3's atomic metadata rename intermittently raises PermissionError on
    # Windows; `_write_zarr` must clear the partial store and retry, succeeding
    # once the transient lock clears.
    calls = {"to_zarr": 0, "remove": 0, "sleep": 0}

    class _FakeResult:
        def to_zarr(self, dst, **kwargs):
            calls["to_zarr"] += 1
            if calls["to_zarr"] < 3:  # fail twice, then succeed
                raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(
        stage_data,
        "_remove",
        lambda p: calls.__setitem__("remove", calls["remove"] + 1),
    )
    monkeypatch.setattr(
        stage_data, "sleep", lambda s: calls.__setitem__("sleep", calls["sleep"] + 1)
    )

    stage_data._write_zarr(_FakeResult(), tmp_path / "s.zarr", {}, serial=True)
    assert calls["to_zarr"] == 3  # two failures + one success
    assert calls["remove"] == 2  # partial store cleared before each retry
    assert calls["sleep"] == 2


def test_write_zarr_reraises_after_exhausting_retries(tmp_path, monkeypatch) -> None:
    class _AlwaysFails:
        def to_zarr(self, dst, **kwargs):
            raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(stage_data, "_remove", lambda p: None)
    monkeypatch.setattr(stage_data, "sleep", lambda s: None)
    try:
        stage_data._write_zarr(_AlwaysFails(), tmp_path / "s.zarr", {}, serial=True)
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError after exhausting retries")


def test_year_from_name_extracts_single_plausible_year() -> None:
    y = stage_data._year_from_name
    assert y("CHIRPS_rainfall_1981.nc") == 1981
    assert y("chirps-v2.0.2001.days_p05.nc") == 2001
    # No year, a 2-digit version, or two distinct years -> None (keep + open).
    assert y("rr_ens_mean_0.1deg_reg_v22.0e.nc") is None
    assert y("merged_1990_2000.nc") is None
    assert y("elev_ens_0.1deg.nc") is None


def test_filter_glob_years_drops_out_of_range_years() -> None:
    files = [Path(f"CHIRPS_rainfall_{yr}.nc") for yr in (1988, 1990, 2005, 2020, 2023)]
    kept, dropped = stage_data._filter_glob_years(files, ["1990-01-01", "2020-12-31"])
    assert [f.name for f in kept] == [
        "CHIRPS_rainfall_1990.nc",
        "CHIRPS_rainfall_2005.nc",
        "CHIRPS_rainfall_2020.nc",
    ]
    assert dropped == 2  # 1988 (before) and 2023 (after)
    # No time_range -> keep everything.
    kept_all, dropped_none = stage_data._filter_glob_years(files, None)
    assert len(kept_all) == 5 and dropped_none == 0
    # A yearless file is always kept (falls through to open-then-check).
    mixed = [Path("rr_v22.0e.nc"), Path("CHIRPS_rainfall_1985.nc")]
    kept_mixed, _ = stage_data._filter_glob_years(mixed, ["2000-01-01", "2010-12-31"])
    assert [f.name for f in kept_mixed] == ["rr_v22.0e.nc"]


def test_print_filters_shows_only_applied_selection(capsys) -> None:
    stage_data._print_filters(["1970-01-01", "2020-12-31"], ["tp", "t2m"])
    out = capsys.readouterr().out
    assert "time_range filter: 1970-01-01 -> 2020-12-31" in out
    assert "variables filter:" in out and "tp, t2m" in out


def test_print_filters_omits_absent_filters(capsys) -> None:
    # time_range only -> no variables line; nothing at all when both absent.
    stage_data._print_filters(["1990-01-01", "2020-12-31"], None)
    out = capsys.readouterr().out
    assert "time_range filter:" in out and "variables filter:" not in out
    stage_data._print_filters(None, None)
    assert capsys.readouterr().out == ""


def test_unpack_clip_result_accepts_two_or_three_tuples() -> None:
    assert stage_data._unpack_clip_result(("written", "d")) == ("written", "d", {})
    assert stage_data._unpack_clip_result(
        ("written", "d", {"natural_time": ["a", "b"]})
    ) == ("written", "d", {"natural_time": ["a", "b"]})


def test_completion_detail_appends_elapsed_time() -> None:
    # The wall-clock `completed:` stamp was dropped; the signature is now
    # (detail, elapsed, *, status). A WRITTEN entry always appends elapsed.
    assert (
        stage_data._completion_detail("12.3 MB", 1.24, status=stage_data.WRITTEN)
        == "12.3 MB; elapsed: 1.2s"
    )


def test_completion_detail_handles_empty_detail() -> None:
    assert (
        stage_data._completion_detail("", 61.0, status=stage_data.WRITTEN)
        == "elapsed: 1m01s"
    )


def test_format_bytes_uses_readable_units() -> None:
    assert stage_data._format_bytes(0) == "0 B"
    assert stage_data._format_bytes(1_500) == "1.5 KB"
    assert stage_data._format_bytes(2_500_000) == "2.5 MB"
    assert stage_data._format_bytes(3_500_000_000) == "3.5 GB"


def test_total_output_bytes_sums_written_and_existing_results() -> None:
    # `_total_output_bytes` was folded into `RunReport.total_output_bytes()`;
    # it still sums only WRITTEN + EXISTS (skipped/failed produce no bytes).
    report = stage_data.RunReport()
    report.record("written", "a", "detail", 1_000)
    report.record("exists", "b", "detail", 2_000)
    report.record("skipped", "c", "detail", 4_000)
    report.record("failed", "d", "detail", 8_000)

    assert report.total_output_bytes() == 3_000


# --- t2608071208: one thread in netCDF at a time --------------------------------


def test_netcdf_glob_units_never_overlap(tmp_path, monkeypatch):
    """The glob's per-file unit -- open, clip, write, close -- runs one at a time.

    The 2026-09-25 stall dump showed the WRITER holding the old write-only lock
    while two readers entered xarray's netCDF locks. Green runs cannot show a
    once-a-month hang is gone, so the property that excludes it is pinned
    instead: however many workers `_run_glob` starts, peak occupancy is 1.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    active, peak, guard = [0], [0], threading.Lock()

    def unit(src, dst, bbox, time_range, variables):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        dst.write_bytes(b"clipped")
        with guard:
            active[0] -= 1
        return stage_data.WRITTEN, "", {}

    monkeypatch.setattr(stage_data, "_subset_netcdf_file_locked", unit)
    sources = []
    for i in range(8):
        src = tmp_path / f"src_{i}.nc"
        src.write_bytes(b"source")
        sources.append(src)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda src: stage_data.subset_netcdf_file(
                    src, tmp_path / "out" / src.name, (0, 0, 1, 1)
                ),
                sources,
            )
        )
    assert [status for status, _ in results] == [stage_data.WRITTEN] * 8
    assert peak[0] == 1
