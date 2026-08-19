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
import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, sleep

# zarr-python 3.x warns that consolidated metadata is not part of the Zarr v3
# spec on every open/write.  We rely on it (fast open of the big era5 source;
# consolidated output for the freshness check) and the caveat does not apply
# here, so silence just this one message to keep the run log clean.
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3",
)

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
from console import (
    banner,
    bold,
    cyan,
    dim,
    fmt_path,
    glyph,
    green,
    pad,
    red,
    rule,
    section_banner,
    yellow,
)

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
        # `state_glyph`, never `glyph`: the latter is the imported
        # `console.glyph()` fallback, and binding it here would shadow the
        # function for this whole method.
        state_glyph, color = {
            WRITTEN: ("+", green),
            EXISTS: ("=", dim),
            SKIPPED: ("-", yellow),
            FAILED: ("x", red),
        }[status]
        detail = _entry_detail(detail, size_bytes, status)
        print(f"    {color(state_glyph)} {name}")
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
            size for status, *_rest, size in self.results if status in (WRITTEN, EXISTS)
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
#
# Two freshness strategies (see `staged`):
#   - "exact"      : every fingerprint key must match (raster/vector/zarr/netcdf).
#   - "time_cover" : bbox + variables must match exactly, but TIME is coverage-
#                    based — used for per-file `netcdf_glob` members so that
#                    WIDENING a time_range only stages the newly-in-range files
#                    and leaves unchanged years untouched.  Each such manifest
#                    also records the source file's natural time span
#                    (`natural_time`) and the effective window written
#                    (`clip_time`).  A file is fresh iff, recomputing the clip
#                    window from the stored natural span and the NEW request, it
#                    equals the stored `clip_time` — an exact test (no false
#                    skips on partial boundary years) that needs no source read.

MANIFEST_VERSION = 2
DEFAULT_RASTER_GLOB_WORKERS = 4

#: Serialises every netCDF WRITE this module makes (`_clip_netcdf_to_file`).
#:
#: `_run_glob` stages a glob across a `ThreadPoolExecutor`, and
#: `_raster_glob_workers` returns 4 for any glob holding more than five files --
#: a policy written for rasterio, where threads are safe, and inherited by the
#: netcdf path, where they are not. HDF5 is not thread-safe and netCDF4-python
#: funnels through a global lock, so four workers all ending in
#: `sub.to_netcdf(dst)` open the library concurrently.
#:
#: On 2026-08-18 one of them stopped coming back: a worker blocked forever in
#: `xarray/backends/netCDF4_.py NetCDF4DataStore.open`, and the main thread sat
#: in `as_completed` behind it, which is the t2608071208 stall that had taken
#: the Windows CI leg down for 30 minutes. Same family as the CHIRPS deadlock
#: `b03d965` fixed by making that write synchronous.
#:
#: Deliberately narrow: only the WRITE serialises. Reading, clipping and the
#: whole `raster_glob` path keep their workers, so the parallel staging added
#: for CMIP6 is untouched. For `netcdf_glob` specifically the write is most of
#: the work, so that stage is now effectively serial -- which is the trade, and
#: a slower stage beats one that hangs without a timeout.
#:
#: HONEST LIMIT: this is reasoned from the captured stacks, not demonstrated.
#: The hang would not reproduce on demand -- 180 staged widen passes across
#: three deliberate attempts, including four concurrent processes, all clean --
#: so nothing here can be shown to fix it, only to remove the contention the
#: stacks show. If it recurs, the harness now dumps its own threads and the
#: next stack will say whether this lock was held.
_NETCDF_WRITE_LOCK = threading.Lock()
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


def _as_day(value) -> str:
    """Normalise any ISO date/datetime (or numpy datetime64 str) to YYYY-MM-DD.

    Daily meteo data only needs day granularity, and truncated ISO dates compare
    lexicographically, so `min`/`max` over these strings is correct without
    parsing.
    """
    return str(value)[:10]


def _clip_window(natural, time_range) -> list[str]:
    """Return the effective [start, end] day-window a request clips out of a file.

    `natural` is the source file's own [min, max] time span.  The window is the
    intersection with the requested `time_range` (both clamped to the file's
    span).  If the request misses the file entirely the result is inverted
    (start > end), which callers treat as "no time overlap".
    """
    nmin, nmax = _as_day(natural[0]), _as_day(natural[1])
    if not time_range:
        return [nmin, nmax]
    return [max(nmin, _as_day(time_range[0])), min(nmax, _as_day(time_range[1]))]


def _time_cover_fresh(dst: Path, fingerprint: dict, *, is_zarr: bool = False) -> bool:
    """Coverage-based freshness for per-file `netcdf_glob` members.

    Fresh iff the output exists and, for a manifest matching src/bbox/variables
    exactly, the clip window recomputed from the stored natural span and the NEW
    request equals the stored `clip_time`.  Falls back to exact `time_range`
    equality when the source has no time axis (or a pre-v2 manifest without a
    recorded natural span), forcing a one-time restage that upgrades it.
    """
    if not dst.exists():
        return False
    m = _read_manifest(dst)
    if m is None:
        return False
    if (m.get("src"), m.get("bbox"), m.get("variables")) != (
        fingerprint["src"],
        fingerprint["bbox"],
        fingerprint["variables"],
    ):
        return False
    natural = m.get("natural_time")
    if natural is None:
        return m.get("time_range") == fingerprint["time_range"]
    return m.get("clip_time") == _clip_window(natural, fingerprint["time_range"])


FRESHNESS = {"exact": _is_fresh, "time_cover": _time_cover_fresh}


def _unpack_clip_result(result) -> tuple[str, str, dict]:
    """Normalise a clip return to (status, detail, manifest_extras).

    Clips may return (status, detail) or (status, detail, extras); the latter
    lets a subsetter contribute extra manifest fields (e.g. the natural time
    span for `time_cover` freshness).
    """
    if len(result) == 3:
        return result
    status, detail = result
    return status, detail, {}


def _natural_time(ds: xr.Dataset) -> list[str] | None:
    """Return the source dataset's [min, max] day span, or None if no time axis."""
    tdim = next((d for d in ds.dims if d.lower() in ("time", "t")), None)
    if tdim is None or ds.sizes.get(tdim, 0) == 0:
        return None
    t = ds[tdim].values
    return [_as_day(t.min()), _as_day(t.max())]


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


@contextmanager
def _serial_dask():
    """Force the synchronous dask scheduler for the enclosed writes.

    A rebuilt output is small and already in memory; writing it single-threaded
    avoids concurrent zarr metadata renames that are flaky on Windows.
    """
    try:
        import dask
    except ImportError:
        yield
        return
    with dask.config.set(scheduler="synchronous"):
        yield


ZARR_WRITE_ATTEMPTS = 3


def _write_zarr(result: xr.Dataset, dst: Path, encoding: dict, *, serial: bool) -> None:
    """Write a zarr store, retrying transient Windows metadata-rename failures.

    zarr-v3's LocalStore commits metadata with an atomic ``.partial -> zarr.json``
    rename that intermittently raises ``PermissionError [WinError 5]`` on Windows
    (an antivirus/indexer briefly locking the file).  A bounded retry — clearing
    the partial store first — absorbs it in our code, without patching zarr or
    serialising large from-scratch writes.  `serial` writes the (small, already
    materialised) rebuilt result single-threaded to avoid the race outright.
    """
    ctx = _serial_dask if serial else _dask_progress
    for attempt in range(ZARR_WRITE_ATTEMPTS):
        try:
            with ctx():
                result.to_zarr(dst, mode="w", consolidated=True, encoding=encoding)
            return
        except (PermissionError, OSError):
            if attempt == ZARR_WRITE_ATTEMPTS - 1:
                raise
            _remove(dst)  # drop the partial store before retrying
            sleep(0.25 * (attempt + 1))


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


def staged(
    *,
    fingerprint_keys: tuple[str, ...] = (),
    is_zarr: bool = False,
    freshness: str = "exact",
):
    """Wrap a clip function with the shared freshness/cleanup/manifest ritual.

    `fingerprint_keys` names the optional keyword args (e.g. ``time_range``,
    ``variables``, ``columns``) that participate in the cached-output
    fingerprint alongside ``src`` and ``bbox``.  `freshness` selects the
    cached-output test: ``"exact"`` (default) requires every fingerprint key to
    match; ``"time_cover"`` treats time as coverage-based (see the manifest
    notes).  A clip may return ``(status, detail)`` or
    ``(status, detail, extras)`` to add fields to the manifest it writes.
    """
    fresh_fn = FRESHNESS[freshness]

    def decorator(clip):
        def wrapper(src: Path, dst: Path, bbox, **kwargs) -> tuple[str, str]:
            fp = _fingerprint(
                src=src,
                bbox=bbox,
                **{k: kwargs.get(k) for k in fingerprint_keys},
            )
            if fresh_fn(dst, fp, is_zarr=is_zarr):
                return EXISTS, ""
            if dst.exists():
                _remove(dst)
                _remove(_manifest_path(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            with _cleanup_on_error(dst):
                status, detail, extras = _unpack_clip_result(
                    clip(src, dst, bbox, **kwargs)
                )
            if status == WRITTEN:
                _write_manifest(dst, {**fp, **extras})
            return status, detail

        wrapper.__name__ = clip.__name__
        wrapper.__doc__ = clip.__doc__
        return wrapper

    return decorator


def _reusable(dst: Path, fingerprint: dict, *, is_zarr: bool = False) -> bool:
    """True if an existing output can seed an incremental rebuild.

    Requires a complete output whose manifest was produced from the same source
    and bbox.  Time-range / variable differences are fine — those are exactly
    the deltas the rebuild resolves; only a bbox (or source) change forces a
    full re-stage.
    """
    exists = _zarr_complete(dst) if is_zarr else dst.exists()
    if not exists:
        return False
    m = _read_manifest(dst)
    if m is None:
        return False
    return m.get("src") == fingerprint["src"] and m.get("bbox") == fingerprint["bbox"]


def staged_rebuild(*, fingerprint_keys: tuple[str, ...] = (), is_zarr: bool = False):
    """Like `staged` (exact freshness) but rebuilds an expanded output in place.

    When the request differs from a valid existing output only by a wider (or
    narrower) time_range and/or a changed variable set at the same bbox, the
    wrapped clip is handed that output's path (``existing=``) so it can reuse the
    already-staged LOCAL data and read only the missing pieces from the source.
    The clip materialises the reused data (closing the old output), removes it,
    and writes the fresh one — non-atomic, like the exact `staged` path, but a
    failed rebuild simply restages from scratch next run (the source is intact).
    The clip receives the source dataset as ``ds`` (opened once by the
    dispatcher) and returns ``(status, detail[, extras])``.
    """

    def decorator(clip):
        def wrapper(src: Path, dst: Path, bbox, *, ds, **kwargs) -> tuple[str, str]:
            fp = _fingerprint(
                src=src,
                bbox=bbox,
                **{k: kwargs.get(k) for k in fingerprint_keys},
            )
            if _is_fresh(dst, fp, is_zarr=is_zarr):
                return EXISTS, ""
            existing = dst if _reusable(dst, fp, is_zarr=is_zarr) else None
            dst.parent.mkdir(parents=True, exist_ok=True)
            if existing is None and dst.exists():
                # Stale and not reusable (e.g. bbox changed): clear before write.
                _remove(dst)
                _remove(_manifest_path(dst))
            with _cleanup_on_error(dst):
                status, detail, extras = _unpack_clip_result(
                    clip(src, dst, bbox, ds=ds, existing=existing, **kwargs)
                )
            if status == WRITTEN:
                _write_manifest(dst, {**fp, **extras})
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


def _raster_output_profile(
    profile: dict, *, height: int, width: int, transform
) -> dict:
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
    name: str,
    *,
    span: float = DEFAULT_TILE_SPAN,
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
        # `dim_name`, not `dim`: the module imports console's dim() styler at the
        # top, and a loop variable named `dim` shadows it for the rest of the
        # function (F402) -- anyone adding a dim(...) call inside would get a
        # TypeError with no obvious cause.
        for dim_name in array.dims:
            dim_lower = dim_name.lower()
            if dim_lower in ("time", "t"):
                chunks[dim_name] = min(array.sizes[dim_name], ZARR_TIME_CHUNK)
            elif dim_lower in ("lat", "latitude", "y", "lon", "longitude", "x"):
                chunks[dim_name] = array.sizes[dim_name]
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
        var_chunks = tuple(
            chunks[dim_name] for dim_name in array.dims if dim_name in chunks
        )
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


# --- Incremental single-store rebuild (zarr / single netcdf) -----------------
#
# When a single-store output already exists at the same bbox and the request
# only widens the time_range and/or changes the variable set, the result is
# assembled from the already-staged LOCAL output plus only the missing
# (variable, time) pieces read from the source.  The combined dataset is
# value-identical to a from-scratch clip: reused cells equal the source (the
# local output was itself a correct clip), and the final reindex to the source's
# request-time axis fixes ordering (so a prepend lands in the right place).


def _time_dim(ds: xr.Dataset):
    return next((d for d in ds.dims if d.lower() in ("time", "t")), None)


def _align_spatial(existing: xr.Dataset, want: xr.Dataset) -> xr.Dataset:
    """Snap the existing output's spatial coords onto the source's.

    Same bbox + same source grid means identical coordinates, but a float
    round-trip through the local store could drift them by an ULP and break the
    concat/merge alignment.  Assigning the source coords (guarded on equal size)
    makes the join exact.
    """
    spatial = ("lat", "latitude", "y", "lon", "longitude", "x")
    # `dim_name`, not `dim` -- see _zarr_subset_chunks: `dim` is console's styler.
    for dim_name in list(existing.dims):
        if (
            dim_name.lower() in spatial
            and dim_name in want.dims
            and existing.sizes[dim_name] == want.sizes[dim_name]
        ):
            existing = existing.assign_coords({dim_name: want[dim_name].values})
    return existing


# CF packing / compression keys carried from the source onto a rebuilt result
# so an incrementally-staged output is stored as compactly as a from-scratch one
# (concat/merge drop `.encoding`).  Shape-dependent keys (chunksizes) are NOT
# carried — the rebuilt time length differs from the source.
PACK_ENCODING_KEYS = {
    "dtype",
    "scale_factor",
    "add_offset",
    "_FillValue",
    "zlib",
    "complevel",
}


def _carry_packing(result: xr.Dataset, source: xr.Dataset) -> xr.Dataset:
    """Reattach the source's CF packing/compression encoding to result vars."""
    for name in result.data_vars:
        if name in source.data_vars:
            result[name].encoding = {
                k: v
                for k, v in source[name].encoding.items()
                if k in PACK_ENCODING_KEYS
            }
    return result


def _combine_reuse(want: xr.Dataset, existing: xr.Dataset, tdim: str) -> xr.Dataset:
    """Assemble `want` (source over the request) reusing `existing` local data.

    Source is read only for variables absent from `existing` and for time steps
    `existing` does not already hold; everything else comes from the local copy.
    """
    want_times = want[tdim]
    have_vars = [v for v in want.data_vars if v in existing.data_vars]
    add_vars = [v for v in want.data_vars if v not in existing.data_vars]
    existing = _align_spatial(existing, want)
    is_have = want_times.isin(existing[tdim])
    reuse_times = want_times.where(is_have, drop=True)
    missing_times = want_times.where(~is_have, drop=True)

    pieces = []
    if have_vars:
        kept = existing[have_vars].sel({tdim: reuse_times})
        if missing_times.size:
            delta = want[have_vars].sel({tdim: missing_times})  # source: delta only
            kept = xr.concat([kept, delta], dim=tdim)
        pieces.append(kept)
    if add_vars:
        pieces.append(want[add_vars])  # source: new vars

    result = xr.merge(pieces) if len(pieces) > 1 else pieces[0]
    result = result.sortby(tdim).sel({tdim: want_times})
    # concat/merge dropped encoding; restore packing so the rebuilt store is as
    # compact as a from-scratch stage (value-identical either way).
    return _carry_packing(result, want)


# Timesteps per download block.  A block ~= one era5 store chunk (365 days), so
# the read is chunk-aligned and the tqdm bar advances (with an ETA) per block.
DOWNLOAD_BLOCK_STEPS = 365


def _download(ds: xr.Dataset, tdim) -> xr.Dataset:
    """Materialise `ds` (the slow SMB read), showing per-block download progress.

    dask's own ProgressBar is unreliable for this — graph fusion and parallel
    chunk reads collapse it into a single 0->100 jump — so for a long time series
    we load in `DOWNLOAD_BLOCK_STEPS` blocks under a tqdm bar that advances once
    per block (reads still parallelise *within* a block).  Short series, no time
    axis, or no tqdm fall back to a plain dask-progress load.  The block concat
    drops encoding, so packing is re-attached from the source.
    """
    n = 0 if tdim is None else ds.sizes.get(tdim, 0)
    if tdim is None or tqdm is None or n <= DOWNLOAD_BLOCK_STEPS:
        # No time axis (e.g. orography) or a short series: a quick load, no bar.
        return ds.load()
    blocks = [
        ds.isel({tdim: slice(i, i + DOWNLOAD_BLOCK_STEPS)}).load()
        for i in tqdm(
            range(0, n, DOWNLOAD_BLOCK_STEPS),
            desc="    downloading",
            unit="blk",
            leave=False,
        )
    ]
    return _carry_packing(xr.concat(blocks, dim=tdim), ds)


def _resolve_incremental(
    src_spatial: xr.Dataset,
    existing_path,
    time_range,
    variables,
    *,
    is_zarr: bool,
):
    """Return ``(dataset_to_write, reused)``, reusing an existing output if able.

    `src_spatial` is the source already clipped to bbox and the requested
    variables.  The dataset is None when the request has no time overlap with
    the source (caller emits SKIPPED).  `reused` is True only when existing
    local data was folded in.  With no existing output (or a source without a
    time axis) the result is the plain source clip — identical to a full stage —
    and `reused` is False.

    In every case the result is **materialised under a dask progress bar**: this
    is where the (slow, SMB) download happens, so the bar advances per source
    chunk read — the meaningful "downloading" signal for large grids like era5.
    The result is small (a bbox clip) and is written from memory by the caller.
    """
    tdim = _time_dim(src_spatial)
    want = _apply_time_range(src_spatial, time_range)
    if tdim is not None and want.sizes.get(tdim, 0) == 0:
        return None, False
    if existing_path is None or tdim is None:
        return _download(want, tdim), False
    existing = (
        xr.open_zarr(existing_path, consolidated=True)
        if is_zarr
        else xr.open_dataset(existing_path)
    )
    try:
        # Materialise (download the delta) before closing the reused source.
        return _download(_combine_reuse(want, existing, tdim), tdim), True
    finally:
        existing.close()


@staged_rebuild(fingerprint_keys=("time_range", "variables"), is_zarr=True)
def subset_zarr(
    src: Path,
    dst: Path,
    bbox,
    *,
    ds: xr.Dataset,
    existing=None,
    time_range=None,
    variables=None,
) -> tuple[str, str]:
    # `ds` is opened once by the dispatcher (shared with the describe block) and
    # closed there; this function must not close it.  `existing` (or None) is the
    # prior output to reuse.  `_resolve_incremental` downloads (materialises) the
    # result under the progress bar, so the write below is from memory.
    ds = _apply_variables(ds, variables)
    lat, lon, lat_slice, lon_slice = _spatial_slices(ds, bbox)
    sub = ds.sel({lat: lat_slice, lon: lon_slice})
    if sub.sizes.get(lat, 0) == 0 or sub.sizes.get(lon, 0) == 0:
        return SKIPPED, "no overlap"
    result, reused = _resolve_incremental(
        sub, existing, time_range, variables, is_zarr=True
    )
    if result is None:
        return SKIPPED, "no time overlap"
    if reused:
        _remove(dst)  # the reused store is closed and in memory; clear before write
    result, encoding = _zarr_subset_write_plan(result)
    _write_zarr(result, dst, encoding, serial=True)  # from memory (already downloaded)
    return WRITTEN, "x".join(f"{result.sizes[d]}" for d in result.sizes)


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
    tdim = next((d for d in sub.dims if d.lower() in ("time", "t")), None)
    if tdim is not None and sub.sizes.get(tdim, 0) == 0:
        # time_range misses this file's span (e.g. an out-of-range year in a
        # netcdf_glob). Skip rather than writing an empty time axis.
        return SKIPPED, "no time overlap"
    # One writer at a time -- see `_NETCDF_WRITE_LOCK`. Held across the whole
    # write rather than around the open, because the hang was observed inside
    # `NetCDF4DataStore.open` and a lock that the opener does not hold to
    # completion leaves the same window.
    with _NETCDF_WRITE_LOCK:
        if show_progress:
            with _dask_progress():
                sub.to_netcdf(dst)
        else:
            sub.to_netcdf(dst)
    return WRITTEN, ""


@staged_rebuild(fingerprint_keys=("time_range", "variables"))
def subset_netcdf(
    src: Path,
    dst: Path,
    bbox,
    *,
    ds: xr.Dataset,
    existing=None,
    time_range=None,
    variables=None,
) -> tuple[str, str]:
    # `ds` is opened once by the dispatcher with `chunks="auto"` and closed
    # there; this function must not close it.  `existing` (or None) is the prior
    # output to reuse.  `_resolve_incremental` downloads (materialises) the result
    # under the progress bar, so the write below is from memory.
    ds = _apply_variables(ds, variables)
    lat, lon, lat_slice, lon_slice = _spatial_slices(ds, bbox)
    sub = ds.sel({lat: lat_slice, lon: lon_slice})
    if sub.sizes.get(lat, 0) == 0 or sub.sizes.get(lon, 0) == 0:
        return SKIPPED, "no overlap"
    result, reused = _resolve_incremental(
        sub, existing, time_range, variables, is_zarr=False
    )
    if result is None:
        return SKIPPED, "no time overlap"
    if reused:
        _remove(dst)  # the reused file is closed and in memory; clear before write
    with _serial_dask():
        result.to_netcdf(dst)  # from memory (already downloaded)
    return WRITTEN, ""


@staged(fingerprint_keys=("time_range", "variables"), freshness="time_cover")
def subset_netcdf_file(
    src: Path,
    dst: Path,
    bbox,
    *,
    time_range=None,
    variables=None,
) -> tuple[str, str] | tuple[str, str, dict]:
    """Clip one netcdf file, opening it here (for the ``netcdf_glob`` path).

    Unlike `subset_netcdf` (whose `ds` is hoisted once by the dispatcher), each
    glob member is opened lazily *after* the `@staged` freshness check, so a
    fully cached re-run never pays an SMB open per file.  The per-file dask bar
    is suppressed — the file-level tqdm bar in `_stage_netcdf_glob` carries the
    progress signal instead.

    Uses ``time_cover`` freshness: the manifest records this file's natural time
    span and the effective clip window, so widening a glob's `time_range` only
    stages the newly-in-range files and leaves unchanged years untouched.
    """
    with xr.open_dataset(src, chunks="auto") as ds:
        natural = _natural_time(ds)
        status, detail = _clip_netcdf_to_file(
            ds,
            dst,
            bbox,
            time_range=time_range,
            variables=variables,
            show_progress=False,
        )
        if status != WRITTEN:
            return status, detail
        clip_time = _clip_window(natural, time_range) if natural else None
        return status, detail, {"natural_time": natural, "clip_time": clip_time}


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


def _print_filters(time_range=None, variables=None) -> None:
    """Print the applied selection, emphasised below the dim source features.

    Only the filters actually in effect are shown (an omitted time_range or
    variables keeps everything, so nothing is printed for it).  Rendered in
    bold cyan with a `->` marker so the kept subset stands out from the plain
    dataset features above it.
    """
    rows = []
    if time_range:
        rows.append(("time_range filter", f"{time_range[0]} -> {time_range[1]}"))
    if variables:
        rows.append(("variables filter", ", ".join(str(v) for v in variables)))
    for label, value in rows:
        head = f"-> {label}:".ljust(
            22
        )  # pad plain text before colouring (aligns values)
        print(f"      {cyan(head)}{bold(cyan(value))}")


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


def _describe_xarray(ds: xr.Dataset) -> list[str]:
    """Describe a source dataset's own features (grid, extent, time, variables).

    The applied selection (time_range / variables) is printed separately and
    emphasised by `_print_filters`, so source features and what we keep from
    them read as two distinct blocks.
    """
    out = []
    try:
        lat_name = next(
            (c for c in ds.coords if c.lower() in ("lat", "latitude", "y")), None
        )
        lon_name = next(
            (c for c in ds.coords if c.lower() in ("lon", "longitude", "x")), None
        )
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

        vars_ = list(ds.data_vars)
        sample = ", ".join(vars_[:8]) + (
            f", ... ({len(vars_)} total)" if len(vars_) > 8 else ""
        )
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
    _print_metadata(
        [f"{len(all_files)} files matching {pattern}{suffix}   sample: {files[0].name}"]
    )
    _print_metadata(_describe_raster(files[0]))
    workers = _raster_glob_workers(entry, file_count=len(files))
    _run_glob(
        name,
        files,
        subset_raster,
        dst,
        bbox,
        report,
        workers=workers,
        dataset_started=dataset_started,
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
                report,
                f.name,
                fn,
                f,
                dst / f.name,
                bbox,
                _verbose=verbose,
                _counts=counts,
                **extra,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _worker_result,
                    f.name,
                    fn,
                    f,
                    dst / f.name,
                    bbox,
                    **extra,
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
            f"{summary}; elapsed: {_format_elapsed(perf_counter() - dataset_started)}"
        )
        print()
        print(f"    {green('+')} {name}")
        print(f"      {dim(summary)}")


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _year_from_name(name: str) -> int | None:
    """Return the single plausible year (19xx/20xx) in a per-year filename, else None.

    Lets a `netcdf_glob` skip files whose year is entirely outside the requested
    window without an SMB open each.  Only an unambiguous single year counts —
    multiple distinct years (or none) return None so the file is kept and opened
    (over-inclusive is safe; over-exclusive would silently lose data).
    """
    years = {int(m.group(1)) for m in _YEAR_RE.finditer(name)}
    return years.pop() if len(years) == 1 else None


def _filter_glob_years(files, time_range):
    """Drop per-year files outside the requested window; return (kept, dropped)."""
    if not time_range:
        return files, 0
    start_y, end_y = int(str(time_range[0])[:4]), int(str(time_range[1])[:4])
    kept, dropped = [], 0
    for f in files:
        year = _year_from_name(f.name)
        if year is not None and not (start_y <= year <= end_y):
            dropped += 1
        else:
            kept.append(f)
    return kept, dropped


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

    # Drop per-year files whose filename year is entirely outside the window,
    # without opening them, so the progress bar counts only the files that will
    # actually stage (and out-of-range years don't clutter the run).
    time_range = entry.get("time_range")
    files, dropped = _filter_glob_years(files, time_range)
    if not files:
        report.print_entry(SKIPPED, name, f"all files outside time_range {time_range}")
        return

    suffix = f"   ({dropped} outside time_range)" if dropped else ""
    _print_metadata(
        [f"{len(files)} files matching {pattern}{suffix}   sample: {files[0].name}"]
    )
    # Describe the first file once so the run log shows grid/time/vars; a
    # describe-only failure must not abort staging.
    try:
        with _open_xarray("netcdf", files[0]) as ds0:
            _print_metadata(_describe_xarray(ds0))
    except Exception as exc:
        _print_metadata([f"(could not read metadata: {str(exc).splitlines()[0][:80]})"])
    _print_filters(entry.get("time_range"), entry.get("variables"))

    extra = {k: entry[k] for k in ("time_range", "variables") if k in entry}
    workers = _raster_glob_workers(entry, file_count=len(files))
    _run_glob(
        name,
        files,
        subset_netcdf_file,
        dst,
        bbox,
        report,
        workers=workers,
        dataset_started=dataset_started,
        extra=extra,
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
    print(f"  {cyan(glyph('▸'))} {bold(name)}")

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
            _print_metadata(DESCRIBERS[kind](ds))
            _print_filters(extra.get("time_range"), extra.get("variables"))
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

    print(banner("Stage"))
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
    print(banner("Description"))
    print(
        "Stage a bbox-clipped subset of a remote data root onto local storage, "
        "mirroring the source tree so an existing data catalog can point to "
        "the local copy. Re-runs use per-output manifests to skip fresh "
        "outputs and restage stale ones."
    )
    print()


def _print_parameters(cfg: dict, config_path: Path) -> None:
    print(banner("Parameters"))

    print(bold("inputs:"))
    rows = [
        ("config", fmt_path(config_path)),
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

    print(rule())
    print(section_banner("total"))
    pill = (
        f"{green(f'written: {counts[WRITTEN]}')}"
        f" {dim(glyph('·'))} "
        f"{dim(f'existing: {counts[EXISTS]}')}"
        f" {dim(glyph('·'))} "
        f"{yellow(f'skipped: {counts[SKIPPED]}')}"
        f" {dim(glyph('·'))} "
        f"{red(f'failed: {counts[FAILED]}')}"
    )
    print(pill)
    print(
        f"{dim('elapsed:')} {_format_elapsed(report.elapsed())}"
        f" {dim(glyph('·'))} "
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
        print(
            red(
                bold(
                    f"FAILED — {counts[FAILED]} output(s) did not stage; see recap below."
                )
            )
        )

    # Only failures get a detailed recap — they need action. Skipped outputs
    # (out-of-range files, no spatial overlap) are expected and only clutter the
    # tail; their count stays in the pill above.
    failures = [(n, d) for s, n, d, _size in report.results if s == FAILED]
    if failures:
        print()
        print(f"{bold('failed')} ({len(failures)}):")
        for name, detail in failures:
            print(f"  {red('x')} {name}  {dim(detail)}")


def main() -> None:
    report = RunReport()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help=f"YAML config (default: {CONFIG_DEFAULT})",
    )
    p.add_argument("--src", type=Path, help="override source_root from the config")
    p.add_argument("--dst", type=Path, help="override target_root from the config")
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="override bbox from the config",
    )
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
