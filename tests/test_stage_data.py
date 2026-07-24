from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.modules.setdefault("geopandas", types.SimpleNamespace())
sys.modules.setdefault("rasterio", types.SimpleNamespace())
sys.modules.setdefault("rasterio.windows", types.SimpleNamespace())
sys.modules.setdefault("xarray", types.SimpleNamespace(Dataset=object))
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
    stage_data._stage_dataset(
        entry, Path("/src"), Path("/dst"), (0, 0, 1, 1), report
    )

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
        "2005-01-01", "2005-12-31",
    ]
    # Left-boundary partial year: request start clips into the file.
    assert cw(["2000-01-01", "2000-12-31"], ["2000-06-01", "2020-12-31"]) == [
        "2000-06-01", "2000-12-31",
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
        src=Path("s/2005.nc"), bbox=(0, 0, 2, 2),
        time_range=["2000-01-01", "2010-12-31"], variables=None,
    )
    stage_data._write_manifest(
        dst,
        {**fp0, "natural_time": natural,
         "clip_time": stage_data._clip_window(natural, fp0["time_range"])},
    )

    def fp(**over):
        base = dict(src=Path("s/2005.nc"), bbox=(0, 0, 2, 2),
                    time_range=["1990-01-01", "2020-12-31"], variables=None)
        return stage_data._fingerprint(**{**base, **over})

    # Widened request, interior year unchanged -> fresh (skip, no rewrite).
    assert stage_data._time_cover_fresh(dst, fp()) is True
    # Request clips into the file's own span -> stale (must restage exactly).
    assert stage_data._time_cover_fresh(
        dst, fp(time_range=["2005-06-01", "2020-12-31"])
    ) is False
    # Different bbox or variables -> stale regardless of time.
    assert stage_data._time_cover_fresh(dst, fp(bbox=(0, 0, 3, 3))) is False
    assert stage_data._time_cover_fresh(dst, fp(variables=["precip"])) is False


def test_time_cover_fresh_requires_existing_output(tmp_path) -> None:
    # No output file on disk -> never fresh even with a matching manifest.
    dst = tmp_path / "missing.nc"
    assert stage_data._time_cover_fresh(dst, stage_data._fingerprint(
        src=Path("s.nc"), bbox=(0, 0, 1, 1), time_range=None, variables=None,
    )) is False


def test_unpack_clip_result_accepts_two_or_three_tuples() -> None:
    assert stage_data._unpack_clip_result(("written", "d")) == ("written", "d", {})
    assert stage_data._unpack_clip_result(
        ("written", "d", {"natural_time": ["a", "b"]})
    ) == ("written", "d", {"natural_time": ["a", "b"]})


def test_completion_detail_appends_elapsed_time() -> None:
    # The wall-clock `completed:` stamp was dropped; the signature is now
    # (detail, elapsed, *, status). A WRITTEN entry always appends elapsed.
    assert stage_data._completion_detail(
        "12.3 MB", 1.24, status=stage_data.WRITTEN
    ) == "12.3 MB; elapsed: 1.2s"


def test_completion_detail_handles_empty_detail() -> None:
    assert stage_data._completion_detail(
        "", 61.0, status=stage_data.WRITTEN
    ) == "elapsed: 1m01s"


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
