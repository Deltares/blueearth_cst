"""Stage a bbox subset of a remote data root to a local SSD.

What it does
------------
Mirrors the source directory tree under a local root, clipping each dataset
to a bbox so only a fraction of bytes lands locally. The matching data
catalog yaml then just needs its ``meta.root`` swapped to the local path.

Configuration
-------------
All knobs live in a YAML file (default: ``dev/scripts/stage_data.yml``):

    source_root: P:/wflow_global/hydromt
    target_root: C:/data/wflow_global/hydromt
    bbox: [8.5, -0.5, 11.0, 1.5]   # west, south, east, north
    datasets:
      - {name: vito,  type: raster,      path: landuse/vito/.../foo.tif}
      - {name: tiles, type: raster_glob, path: topography/.../30sec, pattern: "*.tif", workers: 4}
      - {name: idx,   type: vector,      path: topography/.../basin_index.gpkg,
         columns: [geometry, id]}
      - {name: era5,  type: zarr,        path: meteo/era5_daily.zarr}
      - {name: orog,  type: netcdf,      path: meteo/.../era5_orography_2018.nc}
      - {name: chirps, type: netcdf_glob, path: meteo/chirps_africa_daily_v2.0,
         pattern: "CHIRPS_rainfall_*.nc", time_range: [...], variables: [...]}

Usage
-----
    python dev/scripts/stage_data.py
    python dev/scripts/stage_data.py --config dev/scripts/stage_data.yml
    python dev/scripts/stage_data.py --bbox 8 -1 12 2     # CLI overrides
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

# GDAL/SMB performance tuning. Set as defaults so a user-set env wins.
# Must be set BEFORE geopandas/rasterio import — GDAL reads these once at
# library init.
# - GDAL_CACHEMAX: raster block cache size (MB).
# - VSI_CACHE / VSI_CACHE_SIZE: read-side cache for any VSI handler.
# - GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR: skip the per-open directory scan
#   GDAL does looking for sidecars (.aux.xml, .ovr, etc.). Each scan is
#   another SMB round-trip; on a network drive this is the single biggest
#   per-file overhead reduction.
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "10000000")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import rasterio
import rasterio.windows
import xarray as xr
import yaml

try:
    from dask.diagnostics import ProgressBar  # type: ignore[import-untyped]
except ImportError:
    ProgressBar = None  # type: ignore[assignment]

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:
    tqdm = None  # type: ignore[assignment]

# Make `console.py` importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from console import banner, bold, cyan, dim, fmt_path, green, pad, red, yellow  # noqa: E402

CONFIG_DEFAULT = Path(__file__).resolve().parent / "stage_data.yml"

# Outcome status names.
WRITTEN, EXISTS, SKIPPED, FAILED = "written", "exists", "skipped", "failed"


class RunReport:
    """Collects per-output outcomes and renders the run header/recap.

    Replaces the former module-level `_results`/`_run_started` globals so the
    module is import-safe and testable: create one in `main()` and thread it
    through the staging call chain.  Holds `(status, name, detail, size_bytes)`
    tuples plus the run start time, and owns the counting/printing logic.
    """

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str, int]] = []
        self.started: float = perf_counter()

    def record(
        self,
        status: str,
        name: str,
        detail: str = "",
        size_bytes: int = 0,
    ) -> None:
        """Record an outcome without printing (for tqdm-driven quiet loops)."""
        self.results.append((status, name, detail, size_bytes))

    def print_entry(
        self,
        status: str,
        name: str,
        detail: str = "",
        *,
        size_bytes: int = 0,
    ) -> None:
        """Print a glyph-prefixed entry line and record it for the TOTAL recap."""
        glyph_color = {
            WRITTEN: ("+", green),
            EXISTS:  ("=", dim),
            SKIPPED: ("-", yellow),
            FAILED:  ("x", red),
        }[status]
        glyph, color = glyph_color
        detail = _entry_detail(detail, size_bytes, status)
        print(f"    {color(glyph)} {name}")
        if detail:
            print(f"      {dim(detail)}")
        self.results.append((status, name, detail, size_bytes))

    def counts(self) -> dict[str, int]:
        counts = {WRITTEN: 0, EXISTS: 0, SKIPPED: 0, FAILED: 0}
        for status, *_rest in self.results:
            counts[status] = counts.get(status, 0) + 1
        return counts

    def total_output_bytes(self) -> int:
        """Return bytes for outputs that were written or already fresh."""
        return sum(
            size for status, *_rest, size in self.results
            if status in (WRITTEN, EXISTS)
        )

    def elapsed(self) -> float:
        return perf_counter() - self.started


def _remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def _path_size(path: Path) -> int:
    """Return file or directory size in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@contextmanager
def _cleanup_on_error(dst: Path):
    """On failure, delete partial output so a re-run actually retries."""
    try:
        yield
    except Exception:
        _remove(dst)
        raise


def _zarr_complete(dst: Path) -> bool:
    if not dst.exists():
        return False
    # `.zmetadata`/`.zgroup` are zarr v2 markers; `zarr.json` is the v3 root
    # marker. zarr-python 3.x `to_zarr` writes v3 stores, so without checking
    # `zarr.json` a v3 output is never seen as complete and gets re-staged
    # (an expensive full re-download over SMB) on every run.
    return any((dst / f).exists() for f in (".zmetadata", ".zgroup", "zarr.json"))


# --- Per-output manifest -----------------------------------------------------
#
# Each staged output gets a sidecar JSON noting the parameters it was produced
# with (bbox, time_range, variables, source path).  Subsequent runs compare
# the current parameters to the stored manifest; if they differ, the cached
# output is treated as stale and re-staged.  This is what makes "skip if
# exists" actually safe across YAML edits.

MANIFEST_VERSION = 1
DEFAULT_RASTER_GLOB_WORKERS = 4
RASTER_TILE_SIZE = 256
RASTER_TILE_MIN_SIZE = 16


def _manifest_path(dst: Path) -> Path:
    # Zarr stores are directories; keep the manifest inside.  File outputs get
    # a sidecar next to them.
    if dst.suffix.lower() == ".zarr" or (dst.exists() and dst.is_dir()):
        return dst / ".stage.json"
    return dst.with_name(dst.name + ".stage.json")


def _read_manifest(dst: Path) -> dict | None:
    p = _manifest_path(dst)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_manifest(dst: Path, fingerprint: dict) -> None:
    p = _manifest_path(dst)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_manifest_version": MANIFEST_VERSION, **fingerprint}
    p.write_text(json.dumps(payload, indent=2, default=str))


def _is_fresh(dst: Path, fingerprint: dict, *, is_zarr: bool = False) -> bool:
    """Output exists, looks complete, and its manifest matches the fingerprint."""
    exists = _zarr_complete(dst) if is_zarr else dst.exists()
    if not exists:
        return False
    m = _read_manifest(dst)
    if m is None:
        return False
    return all(m.get(k) == v for k, v in fingerprint.items())


def _fingerprint(
    *,
    src: Path,
    bbox,
    time_range=None,
    variables=None,
    columns=None,
) -> dict:
    return {
        "src": str(src).replace("\\", "/"),
        "bbox": list(bbox),
        "time_range": list(time_range) if time_range else None,
        "variables": list(variables) if variables else None,
        "columns": list(columns) if columns else None,
    }


@contextmanager
def _dask_progress():
    """Context manager that shows a dask progress bar if dask is available."""
    if ProgressBar is None:
        yield
        return
    with ProgressBar():
        yield


def _format_elapsed(seconds: float) -> str:
    """Return a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining:02d}s"
    return f"{minutes}m{remaining:02d}s"


def _format_bytes(size_bytes: int) -> str:
    """Return a compact human-readable byte size."""
    if size_bytes < 1_000:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1_000
        if value < 1_000 or unit == "TB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


def _completion_detail(detail: str, elapsed: float, *, status: str) -> str:
    """Append elapsed time to a detail string when it is worth showing.

    A fully cached re-run should not print dozens of `elapsed: 0.0s` lines, so
    only append `elapsed:` when the entry was WRITTEN or the step actually took
    more than ~1s.  The wall-clock `completed:` stamp is dropped entirely.
    """
    if status != WRITTEN and elapsed <= 1.0:
        return detail
    suffix = f"elapsed: {_format_elapsed(elapsed)}"
    return f"{detail}; {suffix}" if detail else suffix


def _entry_detail(detail: str, size_bytes: int, status: str) -> str:
    """Prepend a uniform size string for written/existing outputs.

    Sizes are rendered here from `size_bytes` (computed once in `_worker_result`)
    rather than hand-formatted inside each subsetter, so every WRITTEN/EXISTS
    line reports size the same way.  Type-specific detail (vector feature count,
    zarr dims) is preserved after the size.
    """
    if status not in (WRITTEN, EXISTS) or size_bytes <= 0:
        return detail
    size = _format_bytes(size_bytes)
    return f"{size}; {detail}" if detail else size


# --- Worker functions: each returns (status, detail). ---
#
# The four `subset_*` functions share the same staging ritual: build a
# fingerprint, early-return EXISTS if the cached output is fresh, drop a stale
# output plus its sidecar manifest, mkdir the parent, run the clip under
# `_cleanup_on_error`, then write the manifest.  `@staged` owns that ritual so
# each subsetter is just its clip logic.  A subsetter returns:
#   - (WRITTEN, detail) on success  -> wrapper writes the manifest,
#   - (SKIPPED, detail) for e.g. no overlap  -> wrapper writes no manifest.


def staged(*, fingerprint_keys: tuple[str, ...] = (), is_zarr: bool = False):
    """Wrap a clip function with the shared freshness/cleanup/manifest ritual.

    `fingerprint_keys` names the optional keyword args (e.g. ``time_range``,
    ``variables``, ``columns``) that participate in the cached-output
    fingerprint alongside ``src`` and ``bbox``.
    """
    def decorator(clip):
        def wrapper(src: Path, dst: Path, bbox, **kwargs) -> tuple[str, str]:
            fp = _fingerprint(
                src=src,
                bbox=bbox,
                **{k: kwargs.get(k) for k in fingerprint_keys},
            )
            if _is_fresh(dst, fp, is_zarr=is_zarr):
                return EXISTS, ""
            if dst.exists():
                _remove(dst)
                _remove(_manifest_path(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            with _cleanup_on_error(dst):
                status, detail = clip(src, dst, bbox, **kwargs)
            if status == WRITTEN:
                _write_manifest(dst, fp)
            return status, detail
        wrapper.__name__ = clip.__name__
        wrapper.__doc__ = clip.__doc__
        return wrapper
    return decorator


def _validate_lonlat_crs(crs, kind: str, src: Path) -> None:
    """Raise if a lon/lat bbox is being applied to a projected dataset."""
    if crs is None:
        return
    if isinstance(crs, str):
        try:
            parsed_crs = rasterio.crs.CRS.from_user_input(crs)
        except Exception:
            parsed_crs = None
        if parsed_crs is not None:
            crs = parsed_crs
        elif crs.upper() in ("EPSG:4326", "OGC:CRS84", "WGS84", "WGS 84"):
            return
    is_geographic = getattr(crs, "is_geographic", None)
    if is_geographic:
        return
    crs_name = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    raise ValueError(
        f"{kind} source {src} has CRS {crs_name}; bbox is lon/lat. "
        "Reproject the bbox or stage from a geographic source."
    )


def _raster_tile_size(size: int) -> int | None:
    """Return a GeoTIFF tile size for a clipped dimension."""
    if size < RASTER_TILE_MIN_SIZE:
        return None
    return min(
        RASTER_TILE_SIZE,
        (size // RASTER_TILE_MIN_SIZE) * RASTER_TILE_MIN_SIZE,
    )


def _raster_output_profile(profile: dict, *, height: int, width: int, transform) -> dict:
    """Return a clipped raster output profile with efficient GeoTIFF tiling."""
    out = profile.copy()
    out.pop("blockxsize", None)
    out.pop("blockysize", None)
    out.update(
        height=height,
        width=width,
        transform=transform,
        compress=out.get("compress") or "deflate",
    )
    if out.get("driver", "GTiff").lower() == "gtiff":
        blockxsize = _raster_tile_size(width)
        blockysize = _raster_tile_size(height)
        if blockxsize is not None and blockysize is not None:
            out.update(tiled=True, blockxsize=blockxsize, blockysize=blockysize)
        else:
            out.update(tiled=False)
    return out


# --- Filename-based tile pre-filter (raster_glob) ---
#
# Many global tiled rasters embed lat/lon corner info in the filename:
# MERIT/GMTED-style "n00e009_30sec.tif" or MODIS sinusoidal "h21v08.tif".
# When a known pattern matches we can skip files whose nominal bounds do
# not overlap the bbox, avoiding a slow SMB open per non-overlapping tile.
# Filenames that don't match any known pattern fall through to the slow
# (open-then-check) path safely.
#
# The n/e/w/s tiles are labelled by their SW corner but the filename does not
# carry the tile span.  Getting the span wrong in the over-inclusive direction
# is safe (a too-wide tile bbox just falls through to open-then-check); getting
# it wrong in the over-exclusive direction silently drops overlapping tiles and
# loses data.  So the assumed span is conservative: 5° (MERIT-family), not 1°.
# A dataset with a different tiling can override it with a `tile_span` YAML key.

DEFAULT_TILE_SPAN = 5.0

_GEO_TILE_RE = re.compile(
    r"(?:^|[^a-zA-Z0-9])([ns])(\d{1,3})([ew])(\d{1,3})(?=[^a-zA-Z0-9]|$)",
    re.IGNORECASE,
)

_MODIS_RE = re.compile(
    r"(?:^|[^a-zA-Z0-9])h(\d{2})v(\d{2})(?=[^a-zA-Z0-9]|$)",
    re.IGNORECASE,
)


def _tile_bounds_from_name(
    name: str, *, span: float = DEFAULT_TILE_SPAN,
) -> tuple[float, float, float, float] | None:
    """Return (W, S, E, N) lon/lat bounds for a recognised tile filename.

    Returning None means "unknown pattern, fall back to opening the file".
    `span` is the assumed n/e/w/s tile size in degrees (default conservative 5°;
    over-inclusive is safe, over-exclusive loses data).  The MODIS pattern uses
    fixed 10° latitude bands, so `span` does not apply there.
    """
    m = _GEO_TILE_RE.search(name)
    if m:
        lat_hem, lat_deg, lon_hem, lon_deg = m.groups()
        lat_n_val, lon_n_val = int(lat_deg), int(lon_deg)
        if 0 <= lat_n_val <= 90 and 0 <= lon_n_val <= 180:
            lat = lat_n_val * (-1 if lat_hem.lower() == "s" else 1)
            lon = lon_n_val * (-1 if lon_hem.lower() == "w" else 1)
            # Tile labelled by SW corner; assumed `span`° wide.
            return float(lon), float(lat), float(lon + span), float(lat + span)

    m = _MODIS_RE.search(name)
    if m:
        v = int(m.group(2))
        if 0 <= v <= 17:
            lat_n = 90.0 - v * 10.0
            lat_s = lat_n - 10.0
            # Sinusoidal lon span varies with lat; use full lon for safety
            # (lat-only filter — still cuts most non-overlapping tiles).
            return -180.0, lat_s, 180.0, lat_n

    return None


def _bbox_overlap(a, b) -> bool:
    """True if two (W, S, E, N) bboxes overlap (touching edges = no overlap)."""
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return aw < be and ae > bw and as_ < bn and an > bs


def _vector_read_kwargs(bbox, columns=None) -> dict:
    """Return geopandas read_file kwargs for a clipped vector subset."""
    kwargs = {"bbox": bbox}
    if columns:
        kwargs["columns"] = list(columns)
    return kwargs


def _raster_glob_workers(entry: dict, *, file_count: int) -> int:
    """Return bounded worker count for independent raster-glob staging."""
    if file_count <= 5:
        return 1
    requested = entry.get("workers", DEFAULT_RASTER_GLOB_WORKERS)
    return max(1, min(int(requested), file_count))


@staged()
def subset_raster(src: Path, dst: Path, bbox) -> tuple[str, str]:
    with rasterio.open(src) as ds:
        _validate_lonlat_crs(ds.crs, "raster", src)
        win = rasterio.windows.from_bounds(*bbox, transform=ds.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        if win.width <= 0 or win.height <= 0:
            return SKIPPED, "no overlap"
        data = ds.read(window=win)
        profile = _raster_output_profile(
            ds.profile,
            height=int(win.height),
            width=int(win.width),
            transform=ds.window_transform(win),
        )
        with rasterio.open(dst, "w", **profile) as out:
            out.write(data)
    return WRITTEN, ""


@staged(fingerprint_keys=("columns",))
def subset_vector(src: Path, dst: Path, bbox, *, columns=None) -> tuple[str, str]:
    # Single open: bbox read first, validate CRS from the result. If the
    # CRS turns out projected, the bbox (lon/lat scale) will match no
    # features and the validation below raises before any write — so the
    # cost of the wasted query is bounded and no bad output is produced.
    gdf = gpd.read_file(src, **_vector_read_kwargs(bbox, columns))
    _validate_lonlat_crs(gdf.crs, "vector", src)
    if len(gdf) == 0:
        return SKIPPED, "no overlap"
    gdf.to_file(dst, driver="GPKG")
    return WRITTEN, f"{len(gdf)} features"


def _spatial_dim(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for c in ds.coords:
        if c.lower() in candidates:
            return c
    raise KeyError(f"none of {candidates} found in coords {list(ds.coords)}")


def _spatial_slices(ds: xr.Dataset, bbox):
    lat = _spatial_dim(ds, ("lat", "latitude", "y"))
    lon = _spatial_dim(ds, ("lon", "longitude", "x"))
    w, s, e, n = bbox
    lat_desc = ds[lat].values[0] > ds[lat].values[-1]
    return lat, lon, (slice(n, s) if lat_desc else slice(s, n)), slice(w, e)


def _apply_time_range(ds: xr.Dataset, time_range) -> xr.Dataset:
    if not time_range:
        return ds
    tdim = next((d for d in ds.dims if d.lower() in ("time", "t")), None)
    if tdim is None:
        return ds
    start, end = time_range
    return ds.sel({tdim: slice(str(start), str(end))})


def _apply_variables(ds: xr.Dataset, variables) -> xr.Dataset:
    if not variables:
        return ds
    keep = [v for v in variables if v in ds.data_vars]
    return ds[keep] if keep else ds


ZARR_TIME_CHUNK = 365
# Only CF packing keys are carried from the source encoding; the source zarr
# codec objects are stripped separately (see `_strip_source_codecs`).
ZARR_ENCODING_KEYS = {
    "_FillValue",
    "add_offset",
    "dtype",
    "scale_factor",
}


def _zarr_subset_chunks(ds: xr.Dataset) -> dict[str, int]:
    """Return output chunk sizes suited to a clipped daily meteo subset."""
    chunks = {}
    for array in ds.data_vars.values():
        for dim in array.dims:
            dim_lower = dim.lower()
            if dim_lower in ("time", "t"):
                chunks[dim] = min(array.sizes[dim], ZARR_TIME_CHUNK)
            elif dim_lower in ("lat", "latitude", "y", "lon", "longitude", "x"):
                chunks[dim] = array.sizes[dim]
    return chunks


def _zarr_subset_encoding(ds: xr.Dataset, chunks: dict[str, int]) -> dict:
    """Build zarr encoding that preserves codecs but replaces source chunks."""
    encoding = {}
    for name, array in ds.data_vars.items():
        var_encoding = {
            key: value
            for key, value in array.encoding.items()
            if key in ZARR_ENCODING_KEYS
        }
        var_chunks = tuple(chunks[dim] for dim in array.dims if dim in chunks)
        if len(var_chunks) == len(array.dims) and var_chunks:
            var_encoding["chunks"] = var_chunks
        if var_encoding:
            encoding[name] = var_encoding
    return encoding


def _strip_source_codecs(ds: xr.Dataset) -> None:
    """Drop v2 numcodecs codec objects inherited from the source encoding.

    `xr.open_zarr` attaches the source `.encoding` (including a `numcodecs`
    `compressor`/`filters` for a zarr v2 store) to every variable *and*
    coordinate. On write, `to_zarr` reuses that per-variable encoding for any
    variable we do not override, so the v2 codec reaches the zarr 3.x writer and
    it raises "Expected a BytesBytesCodec. Got numcodecs.blosc.Blosc instead."
    Clearing these lets zarr 3 apply its own default compressor. Mutates in place.
    Both the v2 (`compressor`) and v3 (`compressors`, plural tuple) key spellings
    are dropped, along with `filters`, since either can carry legacy codecs.
    """
    for variable in ds.variables.values():
        for key in ("compressor", "compressors", "filters"):
            variable.encoding.pop(key, None)


def _zarr_subset_write_plan(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """Return a rechunked subset and matching zarr write encoding."""
    chunks = _zarr_subset_chunks(ds)
    rechunked = ds if not chunks else ds.chunk(chunks)
    _strip_source_codecs(rechunked)
    return rechunked, _zarr_subset_encoding(rechunked, chunks)


@staged(fingerprint_keys=("time_range", "variables"), is_zarr=True)
def subset_zarr(
    src: Path,
    dst: Path,
    bbox,
    *,
    ds: xr.Dataset,
    time_range=None,
    variables=None,
) -> tuple[str, str]:
    # `ds` is opened once by the dispatcher (shared with the describe block) and
    # closed there; this function must not close it.
    ds = _apply_variables(ds, variables)
    lat, lon, lat_slice, lon_slice = _spatial_slices(ds, bbox)
    sub = ds.sel({lat: lat_slice, lon: lon_slice})
    if sub.sizes.get(lat, 0) == 0 or sub.sizes.get(lon, 0) == 0:
        return SKIPPED, "no overlap"
    sub = _apply_time_range(sub, time_range)
    sub, encoding = _zarr_subset_write_plan(sub)
    with _dask_progress():
        sub.to_zarr(dst, mode="w", consolidated=True, encoding=encoding)
    return WRITTEN, "x".join(f"{sub.sizes[d]}" for d in sub.sizes)


def _clip_netcdf_to_file(
    ds: xr.Dataset,
    dst: Path,
    bbox,
    *,
    time_range=None,
    variables=None,
    show_progress: bool = True,
) -> tuple[str, str]:
    """Clip an open dataset spatially/temporally and write it as a netcdf.

    Shared clip body for the single-file (`subset_netcdf`) and per-file-glob
    (`subset_netcdf_file`) paths.  `show_progress` is suppressed on the glob
    path, where many small per-file writes would each spawn a dask bar.
    """
    ds = _apply_variables(ds, variables)
    lat, lon, lat_slice, lon_slice = _spatial_slices(ds, bbox)
    sub = ds.sel({lat: lat_slice, lon: lon_slice})
    if sub.sizes.get(lat, 0) == 0 or sub.sizes.get(lon, 0) == 0:
        # bbox misses the dataset extent (e.g. Europe-only E-OBS vs an African
        # basin). Skip cleanly rather than letting an empty write raise.
        return SKIPPED, "no overlap"
    sub = _apply_time_range(sub, time_range)
    if show_progress:
        with _dask_progress():
            sub.to_netcdf(dst)
    else:
        sub.to_netcdf(dst)
    return WRITTEN, ""


@staged(fingerprint_keys=("time_range", "variables"))
def subset_netcdf(
    src: Path,
    dst: Path,
    bbox,
    *,
    ds: xr.Dataset,
    time_range=None,
    variables=None,
) -> tuple[str, str]:
    # `ds` is opened once by the dispatcher with `chunks="auto"` (dask-backed,
    # so the `_dask_progress` bar around `to_netcdf` is meaningful) and closed
    # there; this function must not close it.
    return _clip_netcdf_to_file(
        ds, dst, bbox, time_range=time_range, variables=variables
    )


@staged(fingerprint_keys=("time_range", "variables"))
def subset_netcdf_file(
    src: Path,
    dst: Path,
    bbox,
    *,
    time_range=None,
    variables=None,
) -> tuple[str, str]:
    """Clip one netcdf file, opening it here (for the ``netcdf_glob`` path).

    Unlike `subset_netcdf` (whose `ds` is hoisted once by the dispatcher), each
    glob member is opened lazily *after* the `@staged` freshness check, so a
    fully cached re-run never pays an SMB open per file.  The per-file dask bar
    is suppressed — the file-level tqdm bar in `_stage_netcdf_glob` carries the
    progress signal instead.
    """
    with xr.open_dataset(src, chunks="auto") as ds:
        return _clip_netcdf_to_file(
            ds, dst, bbox,
            time_range=time_range, variables=variables, show_progress=False,
        )


SUBSETTERS = {
    "raster": subset_raster,
    "vector": subset_vector,
    "zarr": subset_zarr,
    "netcdf": subset_netcdf,
}


def _worker_result(
    label: str,
    fn,
    src: Path,
    dst: Path,
    bbox,
    **kwargs,
) -> tuple[str, str, str, int]:
    """Run one staging worker and return a printable result tuple.

    Pure and thread-safe: touches no shared state, so it is safe to call from a
    `ThreadPoolExecutor`.  Only the main-thread `_record_worker_result` mutates
    the report.
    """
    started = perf_counter()
    try:
        status, detail = fn(src, dst, bbox, **kwargs)
    except Exception as exc:
        status, detail = FAILED, str(exc).splitlines()[0][:80]
        sys.stderr.write(traceback.format_exc())
    detail = _completion_detail(detail, perf_counter() - started, status=status)
    size_bytes = _path_size(dst) if status in (WRITTEN, EXISTS) else 0
    return status, label, detail, size_bytes


def _record_worker_result(
    report: RunReport,
    status: str,
    label: str,
    detail: str,
    size_bytes: int,
    *,
    _verbose=True,
    _counts=None,
) -> None:
    """Record and optionally print one worker result."""
    if _counts is not None:
        _counts[status] = _counts.get(status, 0) + 1
    # In quiet mode (tqdm-driven loops) only break out of the bar for
    # failures and skipped entries — those are the lines a user needs to
    # see. Successful writes / existing files are summarised after the bar.
    if _verbose or status in (FAILED, SKIPPED):
        report.print_entry(status, label, detail, size_bytes=size_bytes)
    else:
        # Still record for the TOTAL recap, just don't print per-line.
        report.record(status, label, detail, size_bytes)


def _run_worker(
    report: RunReport,
    label: str,
    fn,
    src: Path,
    dst: Path,
    bbox,
    *,
    _verbose=True,
    _counts=None,
    **kwargs,
) -> None:
    status, result_label, detail, size_bytes = _worker_result(
        label, fn, src, dst, bbox, **kwargs
    )
    _record_worker_result(
        report,
        status,
        result_label,
        detail,
        size_bytes,
        _verbose=_verbose,
        _counts=_counts,
    )


def _print_metadata(lines) -> None:
    """Print zero or more dim-styled metadata lines indented under a dataset header."""
    for line in lines:
        if line:
            print(f"      {dim(line)}")


def _fmt_bbox(w, s, e, n) -> str:
    return f"{w:.3f}..{e:.3f} lon, {s:.3f}..{n:.3f} lat"


def _describe_raster(src: Path) -> list[str]:
    try:
        with rasterio.open(src) as ds:
            res_x = abs(ds.transform.a)
            res_y = abs(ds.transform.e)
            crs = ds.crs.to_string() if ds.crs else "<none>"
            return [
                f"crs: {crs}   res: {res_x:.6g}° x {res_y:.6g}°   "
                f"size: {ds.width}x{ds.height}",
            ]
    except Exception as exc:
        return [f"(could not read raster metadata: {exc})"]


def _describe_vector(src: Path) -> list[str]:
    try:
        import pyogrio  # type: ignore[import-untyped]
        info = pyogrio.read_info(src)
        crs = info.get("crs") or "<none>"
        n = info.get("features")
        geom = info.get("geometry_type") or "?"
        bbox = info.get("total_bounds")
        bbox_str = _fmt_bbox(*bbox) if bbox is not None and len(bbox) == 4 else "?"
        return [
            f"crs: {crs}   geometry: {geom}   features: {n}",
            f"bounds: {bbox_str}",
        ]
    except Exception as exc:
        return [f"(could not read vector metadata: {exc})"]


def _describe_xarray(ds: xr.Dataset, *, time_range=None) -> list[str]:
    out = []
    try:
        lat_name = next((c for c in ds.coords if c.lower() in ("lat", "latitude", "y")), None)
        lon_name = next((c for c in ds.coords if c.lower() in ("lon", "longitude", "x")), None)
        time_name = next((c for c in ds.coords if c.lower() in ("time", "t")), None)

        spatial_bits = []
        if lat_name and lon_name:
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            if lat.size > 1 and lon.size > 1:
                res_lat = abs(float(lat[1] - lat[0]))
                res_lon = abs(float(lon[1] - lon[0]))
                spatial_bits.append(
                    f"grid: {ds.sizes[lat_name]}x{ds.sizes[lon_name]} "
                    f"({lat_name}x{lon_name})   res: {res_lon:.4g}° x {res_lat:.4g}°"
                )
                spatial_bits.append(
                    f"extent: {float(min(lon)):.3f}..{float(max(lon)):.3f} lon, "
                    f"{float(min(lat)):.3f}..{float(max(lat)):.3f} lat"
                )
        out.extend(spatial_bits)

        if time_name and ds.sizes.get(time_name, 0) > 0:
            t = ds[time_name].values
            n = len(t)
            t0, t1 = str(t[0])[:10], str(t[-1])[:10]
            try:
                freq = xr.infer_freq(ds[time_name][:50]) or "?"
            except Exception:
                freq = "?"
            out.append(f"time: {t0} -> {t1}   ({n} steps, freq: {freq})")
            if time_range:
                out.append(f"time_range filter: {time_range[0]} -> {time_range[1]}")

        vars_ = list(ds.data_vars)
        sample = ", ".join(vars_[:8]) + (f", ... ({len(vars_)} total)" if len(vars_) > 8 else "")
        out.append(f"variables: {sample}")
    except Exception as exc:
        out.append(f"(could not read metadata: {exc})")
    return out


# Describe function per dataset type.  raster/vector take a source path; the
# xarray family (zarr/netcdf) takes the already-open dataset the dispatcher
# hoists (so the source is opened once, not once for describe + once for the
# subset — the big SMB win).
DESCRIBERS = {
    "raster": _describe_raster,
    "vector": _describe_vector,
    "zarr": _describe_xarray,
    "netcdf": _describe_xarray,
}

# Optional YAML keys forwarded to each subsetter as keyword args.
EXTRA_KEYS = {
    "zarr": ("time_range", "variables"),
    "netcdf": ("time_range", "variables"),
    "vector": ("columns",),
}


def _open_xarray(kind: str, src: Path) -> xr.Dataset:
    """Open a zarr/netcdf source once for both describe and subset."""
    if kind == "zarr":
        return xr.open_zarr(src, consolidated=True)
    # chunks="auto" makes the arrays dask-backed so the write streams and the
    # `_dask_progress` bar around `to_netcdf` is meaningful.
    return xr.open_dataset(src, chunks="auto")


def _stage_raster_glob(
    entry: dict,
    name: str,
    src: Path,
    dst: Path,
    bbox,
    report: RunReport,
) -> None:
    """Stage a directory of tiled rasters, one clipped GeoTIFF per source tile."""
    dataset_started = perf_counter()
    pattern = entry.get("pattern", "*.tif")
    if not src.exists():
        report.print_entry(FAILED, name, f"source dir missing: {fmt_path(src)}")
        return
    all_files = sorted(src.glob(pattern))
    if not all_files:
        report.print_entry(SKIPPED, name, f"no files match {pattern}")
        return

    # Filename-based pre-filter: drop tiles whose name-encoded bounds
    # cannot overlap the bbox without paying a per-file SMB open.
    # Filenames matching no known tile pattern fall through to the
    # slow path (open-then-check inside subset_raster).
    tile_span = float(entry.get("tile_span", DEFAULT_TILE_SPAN))
    files = []
    prefiltered = 0
    for f in all_files:
        tb = _tile_bounds_from_name(f.name, span=tile_span)
        if tb is not None and not _bbox_overlap(tb, bbox):
            prefiltered += 1
            continue
        files.append(f)
    if not files:
        report.print_entry(
            SKIPPED, name, f"all {len(all_files)} tiles filtered out by bbox"
        )
        return

    suffix = f"   ({prefiltered} pre-filtered)" if prefiltered else ""
    _print_metadata([
        f"{len(all_files)} files matching {pattern}{suffix}   "
        f"sample: {files[0].name}"
    ])
    _print_metadata(_describe_raster(files[0]))
    workers = _raster_glob_workers(entry, file_count=len(files))
    _run_glob(
        name, files, subset_raster, dst, bbox, report,
        workers=workers, dataset_started=dataset_started,
    )


def _run_glob(
    name: str,
    files: list[Path],
    fn,
    dst: Path,
    bbox,
    report: RunReport,
    *,
    workers: int,
    dataset_started: float,
    extra: dict | None = None,
) -> None:
    """Run a clip `fn` over each glob member with bounded workers + a tqdm bar.

    Shared executor/summary tail for `raster_glob` and `netcdf_glob`.  Emits
    per-file glyphs only on the slow path (verbose, i.e. single-worker with few
    files); otherwise tqdm carries the progress signal and only failures/skips
    break out.  `extra` holds per-type keyword args (e.g. time_range/variables)
    forwarded to `fn`.
    """
    extra = extra or {}
    verbose = workers == 1 and (tqdm is None or len(files) <= 5)
    progress_items = files if workers == 1 else range(len(files))
    bar = (
        tqdm(progress_items, desc=f"    {name}", unit="file", leave=False)
        if (tqdm and not verbose)
        else progress_items
    )
    counts = {WRITTEN: 0, EXISTS: 0, SKIPPED: 0, FAILED: 0}
    if workers == 1:
        for f in bar:
            _run_worker(
                report, f.name, fn, f, dst / f.name, bbox,
                _verbose=verbose, _counts=counts, **extra,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _worker_result, f.name, fn, f, dst / f.name, bbox, **extra,
                )
                for f in files
            ]
            for future, _ in zip(as_completed(futures), bar):
                status, result_label, detail, size_bytes = future.result()
                _record_worker_result(
                    report,
                    status,
                    result_label,
                    detail,
                    size_bytes,
                    _verbose=False,
                    _counts=counts,
                )
    if not verbose:
        summary = (
            f"{counts[WRITTEN]} written, {counts[EXISTS]} existing, "
            f"{counts[SKIPPED]} skipped, {counts[FAILED]} failed"
        )
        if workers > 1:
            summary = f"{summary}; workers: {workers}"
        summary = (
            f"{summary}; elapsed: "
            f"{_format_elapsed(perf_counter() - dataset_started)}"
        )
        print()
        print(f"    {green('+')} {name}")
        print(f"      {dim(summary)}")


def _stage_netcdf_glob(
    entry: dict,
    name: str,
    src: Path,
    dst: Path,
    bbox,
    report: RunReport,
) -> None:
    """Stage a directory of per-file netcdfs, one clipped .nc per source file.

    The xarray analogue of `_stage_raster_glob`: used for meteo datasets whose
    catalog URI templates a `{year}` (CHIRPS) or `{variable}` axis across many
    single files, which the single-file `netcdf` type cannot open.  Each source
    file is mirrored under `dst` so the catalog's templated URI still resolves
    against the local copy.  Supports optional `time_range`/`variables` clips.
    """
    dataset_started = perf_counter()
    pattern = entry.get("pattern", "*.nc")
    if not src.exists():
        report.print_entry(FAILED, name, f"source dir missing: {fmt_path(src)}")
        return
    files = sorted(src.glob(pattern))
    if not files:
        report.print_entry(SKIPPED, name, f"no files match {pattern}")
        return

    _print_metadata([
        f"{len(files)} files matching {pattern}   sample: {files[0].name}"
    ])
    # Describe the first file once so the run log shows grid/time/vars; a
    # describe-only failure must not abort staging.
    try:
        with _open_xarray("netcdf", files[0]) as ds0:
            _print_metadata(
                _describe_xarray(ds0, time_range=entry.get("time_range"))
            )
    except Exception as exc:
        _print_metadata([f"(could not read metadata: {str(exc).splitlines()[0][:80]})"])

    extra = {k: entry[k] for k in ("time_range", "variables") if k in entry}
    workers = _raster_glob_workers(entry, file_count=len(files))
    _run_glob(
        name, files, subset_netcdf_file, dst, bbox, report,
        workers=workers, dataset_started=dataset_started, extra=extra,
    )


def _stage_dataset(
    entry: dict,
    source_root: Path,
    target_root: Path,
    bbox,
    report: RunReport,
) -> None:
    try:
        name = entry["name"]
        kind = entry["type"]
        rel = Path(entry["path"])
    except KeyError as exc:
        print(f"  {red('x')} <invalid entry>  missing key {exc}: {entry!r}")
        report.record(FAILED, str(entry), f"missing key {exc}", 0)
        return

    src, dst = source_root / rel, target_root / rel
    print(f"  {cyan('▸')} {bold(name)}")

    if kind == "raster_glob":
        _stage_raster_glob(entry, name, src, dst, bbox, report)
        return

    if kind == "netcdf_glob":
        _stage_netcdf_glob(entry, name, src, dst, bbox, report)
        return

    fn = SUBSETTERS.get(kind)
    if fn is None:
        report.print_entry(FAILED, name, f"unknown type {kind!r}")
        return
    extra = {k: entry[k] for k in EXTRA_KEYS.get(kind, ()) if k in entry}

    # zarr/netcdf: open the source once, reuse it for the describe block and
    # the subset, then close it.  An open failure is fatal for this dataset
    # (the subset can't run either), so record a FAILED entry and return; a
    # describe-only error is swallowed inside `_describe_xarray` and must not
    # abort staging.
    if kind in ("zarr", "netcdf"):
        try:
            ds = _open_xarray(kind, src)
        except Exception as exc:
            report.print_entry(
                FAILED, name, f"could not open {kind}: {str(exc).splitlines()[0][:80]}"
            )
            return
        try:
            _print_metadata(
                DESCRIBERS[kind](ds, time_range=extra.get("time_range"))
            )
            _run_worker(report, rel.name, fn, src, dst, bbox, ds=ds, **extra)
        finally:
            ds.close()
        return

    # raster/vector: describe reads the source path directly (cheap).
    describe = DESCRIBERS.get(kind)
    if describe is not None:
        _print_metadata(describe(src))
    _run_worker(report, rel.name, fn, src, dst, bbox, **extra)


def stage(cfg: dict, report: RunReport | None = None) -> RunReport:
    report = report if report is not None else RunReport()
    source_root = Path(cfg["source_root"])
    target_root = Path(cfg["target_root"])
    bbox = tuple(cfg["bbox"])
    datasets = cfg.get("datasets", [])
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values [W S E N], got {bbox}")

    print(banner("STAGE"))
    print(f"  {len(datasets)} dataset(s)")
    print()
    for entry in datasets:
        _stage_dataset(entry, source_root, target_root, bbox, report)
        print()
    return report


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("source_root", "target_root", "bbox", "datasets"):
        if key not in cfg:
            raise ValueError(f"config {path} missing required key '{key}'")
    return cfg


def _print_description() -> None:
    print(banner("DESCRIPTION"))
    print(
        "Stage a bbox-clipped subset of a remote data root onto local storage, "
        "mirroring the source tree so an existing data catalog can point to "
        "the local copy. Re-runs use per-output manifests to skip fresh "
        "outputs and restage stale ones."
    )
    print()


def _print_parameters(cfg: dict, config_path: Path) -> None:
    print(banner("PARAMETERS"))

    print(bold("inputs:"))
    rows = [
        ("config",      fmt_path(config_path)),
        ("source_root", fmt_path(cfg["source_root"])),
        ("target_root", fmt_path(cfg["target_root"])),
    ]
    for label, value in rows:
        print(f"  {pad(label, 12, dim)}  {value}")
    print()

    datasets = cfg.get("datasets", [])
    name_width = max((len(d.get("name", "?")) for d in datasets), default=0) + 2
    print(bold(f"datasets ({len(datasets)}):"))
    for d in datasets:
        name = d.get("name", "?")
        kind = d.get("type", "?")
        print(f"  {pad(name, name_width, cyan)}  {dim(kind)}")
    print()

    print(bold("flags:"))
    w, s, e, n = cfg["bbox"]
    print(
        f"  {pad('bbox', 12, dim)} "
        f"{pad(f'{w} {s} {e} {n}', 22, cyan)} "
        f"{dim('west south east north (lon/lat)')}"
    )
    print()


def _print_total(report: RunReport) -> None:
    counts = report.counts()

    print(banner("TOTAL"))
    pill = (
        f"{green(f'written: {counts[WRITTEN]}')}"
        f" {dim('·')} "
        f"{dim(f'existing: {counts[EXISTS]}')}"
        f" {dim('·')} "
        f"{yellow(f'skipped: {counts[SKIPPED]}')}"
        f" {dim('·')} "
        f"{red(f'failed: {counts[FAILED]}')}"
    )
    print(pill)
    print(
        f"{dim('elapsed:')} {_format_elapsed(report.elapsed())}"
        f" {dim('·')} "
        f"{dim('size:')} {_format_bytes(report.total_output_bytes())}"
    )

    # Results are per-file for raster_glob, so these are outputs, not datasets.
    total_ok = counts[WRITTEN] + counts[EXISTS]
    if counts[FAILED] == 0 and total_ok > 0:
        print()
        print(green(bold(f"OK — all {total_ok} output(s) staged successfully.")))
    elif counts[FAILED] == 0 and total_ok == 0:
        print()
        print(yellow(bold("nothing to do — no datasets matched.")))
    else:
        print()
        print(red(bold(f"FAILED — {counts[FAILED]} output(s) did not stage; see recap below.")))

    failures = [(n, d) for s, n, d, _size in report.results if s == FAILED]
    if failures:
        print()
        print(f"{bold('failed')} ({len(failures)}):")
        for name, detail in failures:
            print(f"  {red('x')} {name}  {dim(detail)}")

    skips = [(n, d) for s, n, d, _size in report.results if s == SKIPPED]
    if skips:
        print()
        print(f"{bold('skipped')} ({len(skips)}):")
        for name, detail in skips:
            print(f"  {yellow('-')} {name}  {dim(detail)}")


def main() -> None:
    report = RunReport()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=CONFIG_DEFAULT,
                   help=f"YAML config (default: {CONFIG_DEFAULT})")
    p.add_argument("--src", type=Path, help="override source_root from the config")
    p.add_argument("--dst", type=Path, help="override target_root from the config")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="override bbox from the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.src is not None:
        cfg["source_root"] = str(args.src)
    if args.dst is not None:
        cfg["target_root"] = str(args.dst)
    if args.bbox is not None:
        cfg["bbox"] = list(args.bbox)

    _print_description()
    _print_parameters(cfg, args.config)
    stage(cfg, report)
    _print_total(report)

    if any(s == FAILED for s, *_ in report.results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
