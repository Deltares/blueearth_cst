# Current guidance: dev/reference/toolbox-spatial-foundation.md
"""Extract historical climate data for a given region and time period.

The SINGLE producer of the shared ``data/climate/historical/<key>/`` store.
Declared identically as ``extract_historical_climate`` in ``build_model.smk``
(1.04) and ``generate_scenarios.smk`` (3.08), and generated per candidate source
as ``extract_historical_climate_<source>`` by ``analyze_climate.smk`` (0.04) —
all from ``snake_utils.climate_store_rule`` (R07 B1). The extraction extent stays
**model-free**: it comes from ``data/spatial/geoms/region.geojson``, the one
project region artifact delineated from ``shared.basin`` + the catalog by rule
``delineate_region``, so nothing here reads a built model.

Until the shared-region refactor this script delineated that polygon itself and wrote a
per-store-key copy (``store_region.geojson``). The copy is gone; the store's
extent provenance now travels *in* ``extract_historical.nc`` as the
``region_bbox`` / ``region_geojson_sha256`` / ``region_source`` attributes,
which cannot be separated from the data they describe.
"""

import os
from collections.abc import Mapping
from functools import wraps
from os.path import join
from pathlib import Path
from typing import Optional, Union

import dask
import geopandas as gpd
import hydromt
import netCDF4
import pandas as pd
from hydromt.error import NoDataException
from hydromt.model.processes.meteo import temp

from blueearth_cst.shared.climate_window import (
    intersect_bounds,
    report_coverage,
    resolve_coverage,
    time_axis_bounds,
)
from blueearth_cst.shared.progress import DaskProgress
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.snake_utils import (
    DEFAULT_HYDROGRAPHY,
    log_row,
)
from blueearth_cst.spatial.delineate_region import delineate_region, read_region

#: Grid cells kept AROUND the basin bbox when reading a source.
#:
#: One cell on EACH side preserves cells touching the basin beyond HydroMT's
#: rounded bbox and provides a margin for the current nearest-index forcing
#: reprojection. HydroMT-Wflow requests two cells for precipitation, but that
#: read margin does not require the store to retain two rings. Changing the
#: reprojection method requires rechecking the necessary source coverage.
#:
#: The ring is NOT free downstream: weathergenr averages every cell in the store
#: (``compute_area_averages``, an unweighted mean over ``n_grids``), so including
#: the ring would dilute the series driving the stress test. That is why the
#: extraction also writes ``basin_cells.csv`` -- see ``write_basin_cell_mask``.
BUFFER_CELLS = 1

#: The grid coordinate spelling EVERY store carries, whatever its source calls
#: them. WG-1 pins the store's dims as ``(time, latitude, longitude)``, and
#: ``basin_cells.csv`` writes the same two names for its consumer to match on.
#:
#: era5 arrives spelled that way already; CHIRPS arrives as ``lat``/``lon``.
#: Normalising at the READ, rather than teaching each consumer both spellings,
#: is what keeps ONE store contract across sources -- the seam validator checks
#: the store, so a name-agnostic consumer would leave the artifact itself
#: non-conforming while looking fixed.
_Y_DIM_ALIASES = ("latitude", "lat", "y")
_X_DIM_ALIASES = ("longitude", "lon", "x")


def _normalize_grid_names(ds):
    """Rename a source's y/x grid coords onto the store's canonical spelling.

    First alias present wins, and a dataset already using the canonical names is
    returned untouched (``rename`` with an empty mapping is a no-op, but the
    explicit return keeps that obvious). Renaming here -- immediately after the
    source read, before the era5 variables are reprojected onto this grid and
    before the DEM is built with ``reproject_like(ds)`` -- means everything
    downstream inherits the canonical names instead of needing its own rename.
    """
    rename = {}
    for aliases, target in (
        (_Y_DIM_ALIASES, "latitude"),
        (_X_DIM_ALIASES, "longitude"),
    ):
        for name in aliases:
            if name in getattr(ds, "dims", ()) or name in getattr(ds, "coords", ()):
                if name != target:
                    rename[name] = target
                break
    return ds.rename(rename) if rename else ds


def write_basin_cell_mask(climate_nc, region_gdf, out_csv):
    """List the store cells the basin TOUCHES, for weathergenr to average over.

    weathergenr resamples on a spatial mean of every cell handed to it
    (``compute_area_averages``: ``daily_mat / n_grids``, no mask, no weights).
    The store is a bbox read plus ``BUFFER_CELLS``, so most of those cells can
    lie outside the basin, and the wider buffer the forcing path needs makes
    that worse. Measured on gabon_1008: the basin spans 0.80 x 0.53 ERA5 cells,
    the ``buffer=1`` store held 6, and the basin touches 2 -- so two thirds of
    the series driving that stress test was neighbouring climate.

    Selection is INTERSECTS, and the reason is the same basin: a
    centre-in-polygon test picks **zero** cells there, because the basin is
    smaller than one 0.25-degree cell and contains no cell centre. Centre tests
    are the obvious implementation and they fail exactly on the small basins CST
    exists for.

    Every selected cell counts EQUALLY (owner ruling 2026-08-10). No fractional
    area weighting: the cell either touches the basin or it does not, which
    avoids area arithmetic entirely and leaves weathergenr's own unweighted mean
    CORRECT for the subset it is given.

    Writes ``latitude,longitude`` for the kept cells. The consumer matches on
    those coordinates rather than on index order, so it cannot be silently
    broken by a change in how either side enumerates the grid.
    """
    import xarray as xr
    from shapely.geometry import box

    with xr.open_dataset(climate_nc) as ds:
        lats = [float(v) for v in ds["latitude"].values]
        lons = [float(v) for v in ds["longitude"].values]

    geom = (
        region_gdf.union_all()
        if hasattr(region_gdf, "union_all")
        else region_gdf.unary_union
    )
    half_lat = abs(lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.0
    half_lon = abs(lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.0

    kept = [
        (la, lo)
        for la in lats
        for lo in lons
        if geom.intersects(
            box(lo - half_lon, la - half_lat, lo + half_lon, la + half_lat)
        )
    ]
    if not kept:
        # Unreachable for a region inside its own bbox read, but a store whose
        # grid degenerated to a single cell has zero half-width above and the
        # boxes collapse to points. Falling back to the nearest centre keeps the
        # contract non-empty; an empty list would hand weathergenr no data at
        # all, twenty rules from anything that could explain it.
        centre = geom.centroid
        kept = [
            min(
                ((la, lo) for la in lats for lo in lons),
                key=lambda c: abs(c[0] - centre.y) + abs(c[1] - centre.x),
            )
        ]

    frame = pd.DataFrame(kept, columns=["latitude", "longitude"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    log_row(
        f"Basin intersects {len(frame)} of {len(lats) * len(lons)} extracted grid cells "
        f"-> {os.path.basename(str(out_csv))}",
        module="extract",
    )
    return frame


#: What the below-the-floor message tells an ENFORCING caller to do. The
#: reporting caller gets ``_FLOOR_ADVISORY`` instead: for a candidate that only
#: feeds a comparison figure there is nothing to fix, only a consequence to know.
_FLOOR_REMEDY = (
    "Stage a sufficiently long record and set climate.window within its coverage."
)

_FLOOR_ADVISORY = (
    "This source is a comparison candidate only, so the extraction proceeds -- "
    "but it cannot be selected as climate.selected over this window: "
    "the weather generator requires a longer record"
)


def _check_window_coverage(ds, starttime, endtime, clim_source, enforce_min_years=True):
    """Report what the extraction ACTUALLY covers, and enforce the floor if asked.

    The parse-time guard (``snake_utils.validate_historical_window``) can only
    check what the config REQUESTS. What the staged source holds is knowable
    only here, and a silently-truncated record is the original defect: a config
    asking 1980..2010 against an era5 staging that starts in 2000 yielded 11
    years with no signal, and WF3 then died on weathergenr's wavelet minimum
    twenty rules away (dev/tasks/ R3, observed 2026-05-07).

    Since 2026-08-16 the requested window is a CEILING rather than a demand
    (``shared/climate_window.py``): a source that cannot fill it is extracted
    over the widest span it holds inside it, and the narrowing is logged. Only
    ``enforce_min_years`` still raises, and only for the source that feeds the
    pipeline -- ``shared.clim_historical``. wf0's extra
    ``candidate_sources`` pass ``False``: they end at a comparison figure, so
    weathergenr's minimum is not their constraint.

    That relaxation cannot leak into WF1/WF3 by way of a shared store. Those two
    declare the store ONLY for ``shared.clim_historical``, without the flag, so
    a candidate promoted to primary re-extracts (the params differ, which is
    Snakemake's rerun trigger) and meets the floor here. Rule 1.09 checks the
    store it consumes as well -- see ``model/add_climate_forcing.py`` -- so the
    guarantee does not rest on that trigger alone.
    """
    return report_coverage(
        resolve_coverage(time_axis_bounds(ds), starttime, endtime, clim_source),
        enforce_min_years=enforce_min_years,
        where=_FLOOR_REMEDY if enforce_min_years else _FLOOR_ADVISORY,
    )


def _resolve_source_window(data_catalog, source, starttime, endtime):
    """Return the widest catalog-advertised overlap, warning when it narrows.

    Keep the original request when metadata is absent; the delivered time axis
    still determines actual coverage and the selected forcing's length floor.
    No overlap and invalid catalog bounds remain errors.
    """
    source_spec = data_catalog.to_dict().get(source)
    if not isinstance(source_spec, Mapping):
        return starttime, endtime
    metadata = source_spec.get("metadata")
    extent = metadata.get("extent") if isinstance(metadata, Mapping) else None
    time_range = extent.get("time_range") if isinstance(extent, Mapping) else None
    if not isinstance(time_range, Mapping):
        return starttime, endtime
    available_start = time_range.get("start")
    available_end = time_range.get("end")
    if available_start is None or available_end is None:
        return starttime, endtime

    try:
        requested = (pd.Timestamp(starttime).date(), pd.Timestamp(endtime).date())
        available = (
            pd.Timestamp(available_start).date(),
            pd.Timestamp(available_end).date(),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source!r} has an invalid catalog metadata.extent.time_range: "
            f"{available_start!r}..{available_end!r}"
        ) from exc

    if any(pd.isna(bound) for bound in available) or available[0] > available[1]:
        raise ValueError(
            f"{source!r} has an invalid catalog metadata.extent.time_range: "
            f"{available_start!r}..{available_end!r}"
        )
    overlap = max(requested[0], available[0]), min(requested[1], available[1])
    if overlap[0] > overlap[1]:
        raise ValueError(
            f"{source!r} has no overlap with requested "
            f"{requested[0]}..{requested[1]}; the catalog advertises "
            f"{available[0]}..{available[1]}"
        )
    if overlap != requested:
        log_row(
            f"{source}: extracting {overlap[0]}–{overlap[1]}; "
            f"requested {requested[0]}–{requested[1]}, but catalog coverage is "
            f"{available[0]}–{available[1]}",
            module="extract",
            level="WARNING",
        )
        return overlap[0].isoformat(), overlap[1].isoformat()
    return starttime, endtime


def _read_source(data_catalog, source, *, requested, **kwargs):
    """Read the catalog overlap; name the requested window when no data exists.

    Missing year files and a shorter actual time axis can narrow coverage
    further. HydroMT returns their overlap; ``NoDataException`` means no usable
    data, whose native message names neither the source nor the requested dates.
    """
    kwargs["time_range"] = _resolve_source_window(
        data_catalog, source, *kwargs.get("time_range", requested)
    )
    try:
        return data_catalog.get_rasterdataset(source, **kwargs)
    except NoDataException as exc:
        raise ValueError(
            f"{source!r} has no data at all inside "
            f"{pd.Timestamp(requested[0]).date()}.."
            f"{pd.Timestamp(requested[1]).date()}. A source that merely falls "
            f"SHORT of shared.historical_window is extracted over what it does "
            f"hold; this one overlaps it nowhere, so there is nothing to "
            f"extract. Check that the source is staged for this basin, and that "
            f"shared.historical_window names years it covers. ({exc})"
        ) from exc


def _align_chunks_to_store(data_catalog, sources):
    """Return a catalog whose ``sources`` read at the store's ENCODED chunking.

    ``{}`` asks xarray for the encoded chunking, whatever the backend, rather
    than naming sizes this function cannot know. It reaches ``open_mfdataset``
    / ``open_zarr`` as an extra driver kwarg (hydromt's ``DriverOptions`` dumps
    with ``exclude_unset=True``, so an empty dict set explicitly survives).

    The era5 read used to ask for ``"auto"``, reasoning that "we can afford
    larger chunks as we only extract and save". That holds on local disk, where
    over-read is free, and INVERTS over a network store: a dask chunk spanning
    several on-disk chunks drags every one of them across the wire so a basin
    covering one of them can be sliced out.

    Measured 2026-09-18 against ``deltares_data_pdrive.yml`` (era5 as yearly
    global netCDFs on ``p:/``, HDF5 chunks 30x250x480). ``"auto"`` chunked
    60x500x960 -- 2 latitude bands x 2 longitude bands. One year of the seven
    variables over a Gabon bbox, through ``get_rasterdataset``::

        "auto"  2.14 GB  101.8 s        {}  0.55 GB  28.8 s

    3.85x the bytes for the same 5x5 grid. At 17 years that is the 37.5 GB and
    46 min a real extraction spent writing a 76 KB store.

    MEASURE THIS THROUGH ``get_rasterdataset``, NOT ``open_mfdataset`` +
    ``.sel`` -- a probe written the direct way shows the two specs within 4% of
    each other and reads as evidence that the chunking is irrelevant. It is
    not: slicing an opened dataset lets dask fuse the slice into the backend
    read, so only the touched disk chunks move whatever the dask chunk says.
    hydromt applies this source's ``rename`` and ``unit_mult`` between the read
    and the clip, and that elementwise layer is what blocks the fusion -- which
    is precisely why the dask chunk spec governs the transfer on the real path
    and not on a naive one.

    Naming the catalog's own sizes would not be a cure either. era5's
    ``longitude: 240`` SPLITS the stored 480-wide chunk, so two dask tasks each
    want half of one compressed chunk and whether the second one costs a second
    decompression is a question about the HDF5 chunk cache -- which
    ``_bounded_hdf5_chunk_cache`` below deliberately shrinks to 1 MiB, far
    under one 14.4 MB chunk. So under this function's own settings a split spec
    WOULD pay twice, for a basin that straddles the split. Ntoum does not:
    measured 2026-09-18, ``240`` and the encoded ``480`` read the same 0.48 GB.
    ``{}`` makes the question unaskable rather than answered -- the dask chunk
    is the disk chunk by construction -- and it needs no edit to a catalog that
    is vendored upstream-verbatim and hash-pinned.

    An entry the catalog does not carry is skipped rather than invented: the
    caller names every source the extraction MIGHT read, and a chirps-only
    catalog legitimately has no era5.

    In hydromt 1.x the source schema changed: chunks lives under
    ``driver.options`` instead of the old top-level ``driver_kwargs``.
    """
    as_dict = data_catalog.to_dict()
    for name in dict.fromkeys(sources):
        source = as_dict.get(name)
        if source is None:
            continue
        driver = source.setdefault("driver", {})
        if isinstance(driver, str):
            driver = {"name": driver}
            source["driver"] = driver
        driver.setdefault("options", {})["chunks"] = {}
    return hydromt.DataCatalog().from_dict(as_dict)


#: Bytes of HDF5 chunk cache each netCDF variable this extraction opens may keep.
#:
#: netcdf-c gives EVERY VARIABLE of EVERY open file a 64 MiB chunk cache
#: (``netCDF4.get_chunk_cache()`` -> 67108864, libnetcdf 4.10.1). That cache
#: pays for itself when chunks are revisited. Here none are: ``chunks={}``
#: makes the dask chunk the disk chunk, so each chunk is read once, sliced to
#: the basin and dropped. The cache therefore retains bytes nothing will ask
#: for again, once per (file, variable) -- so against a source stored as yearly
#: files, resident memory grows with the LENGTH OF THE WINDOW rather than with
#: the size of the basin or of the store.
#:
#: Measured 2026-09-18 through this function, Ntoum, era5 on ``p:/``: 21
#: (file, variable) pairs over 2000..2002 held 1.26 GB and 42 pairs over
#: 2000..2005 held 2.50 GB -- 60 MB each, the four 30x250x480 chunks that fit
#: under 64 MiB. Over the real 17-year window that projects to ~7 GB resident
#: to write a 2.3 MB store, and the run measuring it was reaped under
#: system-wide memory pressure before it could finish.
#:
#: IT IS THE READ, NOT THE WRITE. The streaming ``to_netcdf(compute=False)``
#: below was the first suspect and is innocent: an eager read and a read with
#: no write at all grow identically (1596 / 1570 MB peak against the shipped
#: path's 1578 over 2000..2002), so the scheduler note further down is right
#: that synchronous streaming leaves peak memory unchanged -- the growth was
#: never in that graph.
#:
#: 1 MiB is below one chunk, so HDF5 bypasses the cache for these reads
#: outright. Same 3.34 GB over the wire, peak RSS 2811 -> 394 MB, and the read
#: came out marginally FASTER (2.90 min against 3.02) -- the cache was pure
#: cost. A source whose chunks fit under 1 MiB is still cached normally.
_HDF5_CHUNK_CACHE_BYTES = 1 << 20


def _bounded_hdf5_chunk_cache(func):
    """Run ``func`` with netCDF's default chunk cache bounded, then restore it.

    ``nc_set_chunk_cache`` is process-global and applies to files opened
    AFTERWARDS, so the bound has to be in force around the reads themselves --
    setting it once at import would miss a caller that opens a store before
    importing this module, and would also never be undone. Restoring keeps a
    process that calls this for one extraction from inheriting the bound for
    every other netCDF it touches, which is the property that makes bounding
    the cache safe to apply to the whole function rather than to one branch.

    ``min`` rather than a bare set, so an operator who has already chosen a
    SMALLER cache keeps it.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        previous = netCDF4.get_chunk_cache()
        size, nelems, preemption = previous
        netCDF4.set_chunk_cache(min(size, _HDF5_CHUNK_CACHE_BYTES), nelems, preemption)
        try:
            return func(*args, **kwargs)
        finally:
            netCDF4.set_chunk_cache(*previous)

    return wrapper


#: Global attributes WG-1 pins, with the values it pins them to. Both are
#: constants of the contract rather than of a source: every climate store this
#: toolbox writes is EPSG:4326 meteorological data by construction.
_WG1_GLOBAL_ATTRS = {"crs": 4326, "category": "meteo"}

#: Store variables and coordinates WG-1 requires as ``float32``.
_WG1_FLOAT32_COORDS = ("latitude", "longitude")
_WG1_FLOAT32_VARS = (
    "precip",
    "temp",
    "temp_min",
    "temp_max",
    "kin",
    "kout",
    "press_msl",
)


def _stamp_catalog_metadata(ds, data_catalog, source):
    """Put the catalog entry's ``metadata:`` block back onto the store.

    hydromt attaches a source's metadata to the Dataset it returns, so the era5
    path gets it for free. A branch that fetches one variable and calls
    ``.to_dataset()`` does not: the DataArray never carried the Dataset-level
    attrs, so they are simply gone by the time the store is written.

    Existing attributes WIN. This runs after the read and before the run-level
    provenance below, so a source that already carried its metadata keeps
    exactly what it read -- the function fills gaps rather than overwriting an
    answer it did not compute.

    A catalog that cannot be interrogated for the entry is not an error here:
    the two WG-1 attrs are contract constants and are stamped regardless, and a
    missing citation block degrades the run record rather than the store's
    conformance.
    """
    for key, value in _WG1_GLOBAL_ATTRS.items():
        ds.attrs.setdefault(key, value)

    try:
        entry = data_catalog.get_source(source)
        metadata = getattr(entry, "metadata", None)
        items = dict(metadata) if metadata else {}
    except Exception:  # noqa: BLE001 -- see the docstring: this must not fail a run
        items = {}

    for key, value in items.items():
        if value is None or key in ds.attrs:
            continue
        # netCDF attributes are scalars or arrays of them; anything structured
        # is rendered rather than dropped, so a nested `metadata:` block still
        # reaches a reader instead of vanishing on `to_netcdf`.
        ds.attrs[key] = value if isinstance(value, (str, int, float)) else str(value)
    return ds


def _normalize_precip_units(ds, source):
    """Label verified CHIRPS daily depths without rescaling their values."""
    if (
        source in {"chirps", "chirps_global"}
        and "precip" in ds.data_vars
        and ds["precip"].attrs.get("units") == "mm"
    ):
        ds["precip"].attrs["units"] = "mm d**-1"
    return ds


def _coerce_store_dtypes(ds):
    """Cast the WG-1 coordinates and variables to ``float32``.

    The contract pins ``float32`` for the lat/lon coordinates and all seven
    variables. era5 arrives that way from hydromt's read; a hand-assembled
    dataset does not, and a ``float64`` store fails ``validate_wg1`` on one row
    per coordinate and per variable.

    This is a narrowing cast, and it is the one the era5 path already performs
    implicitly -- so applying it to every store makes the two branches agree
    rather than introducing a precision choice. Variables absent from a store
    are skipped rather than created: a precipitation-only source has no
    ``temp``, and inventing one is the failure mode
    ``skip-outputs-for-missing-variables`` exists to prevent.
    """
    for coord in _WG1_FLOAT32_COORDS:
        if coord in ds.coords and str(ds[coord].dtype) != "float32":
            ds = ds.assign_coords({coord: ds[coord].astype("float32")})
    for var in _WG1_FLOAT32_VARS:
        if var in ds.data_vars and str(ds[var].dtype) != "float32":
            ds[var] = ds[var].astype("float32")
    return ds


@_bounded_hdf5_chunk_cache
def prep_historical_climate(
    region_fn: Optional[Union[str, Path]],
    fn_out: Union[str, Path],
    data_libs: Union[str, Path] = "deltares_data",
    clim_source: str = "era5",
    *,
    starttime: str,
    endtime: str,
    bbox=None,
    oro_out: Optional[Union[str, Path]] = None,
    hydrography: str = DEFAULT_HYDROGRAPHY,
    region_sha256: Optional[str] = None,
    region_source: Optional[Union[str, Path]] = None,
    enforce_min_years: bool = True,
    forcing_required: bool = True,
):
    """
    Extract historical climate data for a given region and time period.

    If ``clim_source`` is CHIRPS, its precipitation is combined with ERA5
    forcing variables only when ``forcing_required`` is true.

    Parameters
    ----------
    region_fn : str, Path, optional
        Path to the region geojson file. Exactly one of ``region_fn`` and
        ``bbox`` must be provided.
    fn_out : str, Path
        Path to the output netcdf file
    data_libs : str, Path
        Path to the data catalogs yaml file or pre-defined catalogs
    clim_source : str
        Name of the climate source to use
    starttime : str
        Start time of the forcing, format YYYY-MM-DDTHH:MM:SS
    endtime : str
        End time of the forcing, format YYYY-MM-DDTHH:MM:SS
    bbox : tuple of float, optional
        Extraction bounds (xmin, ymin, xmax, ymax) used instead of the region
        file's total bounds. The rule passes
        ``read_region(...).total_bounds``; ``region_fn`` remains for
        standalone/unit use.
    oro_out : str, Path, optional
        Destination for the chirps/chirps_global orography sidecar. The rule
        passes its declared ``oro_nc`` output (``<store>/orography.nc``) so the
        DAG edge, not a filename convention, carries the DEM/climate
        co-provenance contract. Defaults to the historical
        ``<dirname(fn_out)>/<clim_source>_orography.nc`` when omitted. Ignored
        outside the chirps branch.
    hydrography : str, optional
        Catalog ENTRY NAME of the elevation source the chirps branch reads
        ``elevtn`` from, to reproject onto the precipitation grid and to lapse-
        correct temperature against ``era5_orography``. Defaults to
        ``DEFAULT_HYDROGRAPHY``; the rule passes ``shared.basin.hydrography``,
        which is the same entry the delineation and the model build read.

        It was the hardcoded ``merit_hydro`` until 2026-08-16. That was a source
        NOTHING else in the toolbox names -- ``DEFAULT_HYDROGRAPHY`` is
        ``merit_hydro_ihu`` and the shipped ``setup_basemaps`` agrees -- so the
        chirps branch demanded a staged dataset a working project need not have,
        and failed inside hydromt with a catalog-resolution error that reads as
        a code defect. Reading the CONFIGURED entry also means the DEM behind
        the store is the DEM behind the basin, which is the property that was
        actually wanted. Ignored outside the chirps branch.
    region_sha256 : str, optional
        Content digest of the region artifact the ``bbox`` came from, stamped
        on the extraction as ``region_geojson_sha256``.
    region_source : str, Path, optional
        Path of that artifact, stamped as ``region_source`` relative to the
        directory of ``fn_out``.
    enforce_min_years : bool, optional
        Whether a delivered record below ``MIN_HISTORICAL_YEARS`` is an ERROR
        (default) or a logged warning. ``False`` only for wf0's extra
        ``candidate_sources``, which end at a comparison figure; see
        ``_check_window_coverage``.
    forcing_required : bool, optional
        Whether a precipitation-only source must be enriched with ERA5 and
        orography for WF1/WF3. ``False`` for wf0-only comparison candidates.
    """
    if (region_fn is None) == (bbox is None):
        raise ValueError(
            "prep_historical_climate: exactly one of region_fn or bbox must be provided"
        )
    if bbox is None:
        # Read region
        region = gpd.read_file(region_fn)
        bbox = region.geometry.total_bounds
    # Read data catalog
    data_catalog = hydromt.DataCatalog(data_libs=data_libs)

    # ALIGN THE DASK CHUNKS TO THE STORE'S OWN CHUNKS, for EVERY source this
    # function may read -- see `_align_chunks_to_store` for the measurement that
    # motivates it.
    #
    # Hoisted above the branch on 2026-09-18. The override used to live inside
    # the `else` arm only, so the chirps branch read era5 for its other six
    # variables at whatever the catalog happened to declare. That is not a
    # hypothetical second-class path: `deltares_data_pdrive.yml` declares era5
    # at `longitude: 240`, which splits the stored 480-wide chunk, so the chirps
    # branch was the one arm still exposed to the straddle trap the docstring
    # describes. Ntoum sits inside one half and reads 0.48 GB either way, so
    # this buys that basin no bytes -- it removes a basin-dependent cliff, and
    # it stops the two arms from disagreeing about how to read the same source.
    is_precip_only = clim_source in {"chirps", "chirps_global"}
    sources = [clim_source]
    if is_precip_only and forcing_required:
        sources.append("era5")
    data_catalog = _align_chunks_to_store(data_catalog, sources)
    # Extract climate data
    log_row(f"Extracting historical climate grid for {clim_source}", module="extract")
    if is_precip_only:
        log_row(
            (
                f"{clim_source} provides precipitation only; "
                + (
                    "adding ERA5 companion climate fields for selected-source forcing"
                    if forcing_required
                    else "retaining native precipitation for source comparison"
                )
            ),
            module="extract",
        )
        # Get precip first
        ds = _read_source(
            data_catalog,
            clim_source,
            requested=(starttime, endtime),
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=["precip"],
        ).to_dataset()
        # CHIRPS spells its grid `lat`/`lon`; the store contract is
        # `latitude`/`longitude`. Done HERE so the era5 reprojection below and
        # the DEM's `reproject_like(ds)` both inherit the canonical names.
        ds = _normalize_grid_names(ds)
        if forcing_required:
            # Get clim
            ds_clim = _read_source(
                data_catalog,
                "era5",
                requested=(starttime, endtime),
                bbox=bbox,
                time_range=(starttime, endtime),
                buffer=BUFFER_CELLS,
                variables=[
                    "temp",
                    "temp_min",
                    "temp_max",
                    "kin",
                    "kout",
                    "press_msl",
                ],
            )
            # THE STORE'S WINDOW IS WHAT BOTH SOURCES COVER, and clipping to it
            # here is load-bearing rather than tidy. The six era5-derived
            # variables are assigned into `ds` below, and `ds[var] = da`
            # REINDEXES `da` onto `ds`'s time axis -- so a chirps record longer
            # than the era5 one would leave real precipitation beside all-NaN
            # temperature and radiation over the non-overlap. That store passes
            # WG-1 and hands the NaNs to weathergenr's area average twenty rules
            # later. Intersecting first turns silent corruption into a shorter
            # window.
            _chirps_bounds = time_axis_bounds(ds)
            _era5_bounds = time_axis_bounds(ds_clim)
            _shared_bounds = intersect_bounds(_chirps_bounds, _era5_bounds)
            if _shared_bounds is None and None not in (_chirps_bounds, _era5_bounds):
                raise ValueError(
                    f"{clim_source} covers "
                    f"{_chirps_bounds[0].date()}..{_chirps_bounds[1].date()} and era5 "
                    f"covers {_era5_bounds[0].date()}..{_era5_bounds[1].date()} inside "
                    f"the requested window; the two do not overlap, so no store can "
                    f"be assembled -- {clim_source} supplies precipitation only and "
                    f"era5 supplies every other variable"
                )
            if _shared_bounds is not None:
                if _chirps_bounds != _shared_bounds or _era5_bounds != _shared_bounds:
                    log_row(
                        f"{clim_source} covers "
                        f"{_chirps_bounds[0].date()}..{_chirps_bounds[1].date()}, era5 "
                        f"covers {_era5_bounds[0].date()}..{_era5_bounds[1].date()}; "
                        f"the store takes their overlap "
                        f"{_shared_bounds[0].date()}..{_shared_bounds[1].date()}",
                        module="extract",
                        level="WARNING",
                    )
                ds = ds.sel(time=slice(*_shared_bounds))
                ds_clim = ds_clim.sel(time=slice(*_shared_bounds))
            # Prepare orography from the configured hydrography DEM for
            # downscaling -- the same entry delineation and model build read.
            log_row(
                f"Using {hydrography} orography to downscale ERA5 companion fields for {clim_source}",
                module="extract",
            )
            dem = data_catalog.get_rasterdataset(
                hydrography,
                bbox=bbox,
                time_range=(starttime, endtime),
                buffer=BUFFER_CELLS,
                variables=["elevtn"],
            )
            dem = dem.raster.reproject_like(ds, method="average")
            # Resample the remaining variables onto the precipitation grid.
            log_row(
                f"Downscaling ERA5 companion fields to the {clim_source} grid",
                module="extract",
            )
            for var in ["press_msl", "kin", "kout"]:
                ds[var] = ds_clim[var].raster.reproject_like(ds, method="nearest_index")

            # Read ERA5 orography for temperature downscaling.
            dem_era5 = data_catalog.get_rasterdataset(
                "era5_orography",
                geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
                buffer=2,
                variables=["elevtn"],
            ).squeeze()
            for var in ["temp", "temp_min", "temp_max"]:
                ds[var] = temp(
                    ds_clim[var],
                    dem,
                    dem_forcing=dem_era5,
                    lapse_correction=True,
                    freq=None,
                    reproj_method="nearest_index",
                    lapse_rate=-0.0065,
                )
            # Save the DEM at the caller's declared output.
            fn_dem = (
                os.fspath(oro_out)
                if oro_out is not None
                else os.path.join(
                    os.path.dirname(fn_out), f"{clim_source}_orography.nc"
                )
            )
            dem.to_netcdf(fn_dem, mode="w")

    else:
        ds = _read_source(
            data_catalog,
            clim_source,
            requested=(starttime, endtime),
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=[
                "precip",
                "temp",
                "temp_min",
                "temp_max",
                "kin",
                "kout",
                "press_msl",
            ],
        )

    _check_window_coverage(
        ds, starttime, endtime, clim_source, enforce_min_years=enforce_min_years
    )

    # The store's extent provenance is in the extraction rather than beside it.
    # `store_region.geojson` used to sit in the store directory as
    # the record of where the bbox came from; the region is now one shared
    # project artifact, so a copy per store key would be a second source of
    # truth that can drift. Attributes cannot be separated from the data they
    # describe, and the sha256 says WHICH polygon, not merely which numbers.
    # WG-1 conformance, applied HERE rather than inside a branch.
    #
    # The chirps branch fetches ONE variable and calls `.to_dataset()` on the
    # resulting DataArray, and the catalog entry's `metadata:` block does not
    # survive that: measured 2026-08-16, a chirps store carried exactly one
    # attribute (`region_bbox`) against era5's eight, and `validate_wg1` failed
    # it on eight counts. `crs` and `category` were never two isolated
    # omissions -- they are the two rows the validator happens to check out of
    # eight that go missing for the same reason. Its dtypes drifted to float64
    # for the same underlying cause: era5 inherits both from hydromt's read,
    # the chirps branch assembles its dataset by hand.
    #
    # Stamping at the single write path rather than in the branch that failed is
    # deliberate. A per-branch fix is correct for chirps and silently absent for
    # the next source someone adds to `_SUPPORTED_SOURCES` -- which is exactly
    # how `chirps_global` came to carry the identical defect behind a second
    # name. These operations are idempotent, so the era5 path is unaffected.
    ds = _stamp_catalog_metadata(ds, data_catalog, clim_source)
    ds = _coerce_store_dtypes(ds)
    ds = _normalize_precip_units(ds, clim_source)

    ds.attrs["region_bbox"] = [float(value) for value in bbox]
    if region_sha256 is not None:
        ds.attrs["region_geojson_sha256"] = region_sha256
    if region_source is not None:
        # Relative to the written file, so the same extraction in another
        # project folder is byte-identical: this file's hash feeds the WF3
        # automatic seed and collection id. region_geojson_sha256 pins content.
        ds.attrs["region_source"] = Path(
            os.path.relpath(region_source, Path(fn_out).parent)
        ).as_posix()

    dvars = ds.raster.vars
    encoding = {k: {"zlib": True} for k in dvars}

    log_row(
        f"{clim_source}: writing extracted climate grid to netCDF", module="extract"
    )
    delayed_obj = ds.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
    # Labelled with the SOURCE, because a multi-source project runs this rule
    # once per source and the console would otherwise show identical bars.
    #
    # SYNCHRONOUS, and that is the whole fix for a hard deadlock -- measured
    # 2026-08-18, when wf0 parked forever on the chirps store at 94.2%.
    #
    # netCDF4/HDF5 takes a PROCESS-GLOBAL lock and xarray serializes every
    # netCDF touch through it. Under the threaded scheduler this graph asks for
    # that lock from both directions at once: the `to_netcdf` write tasks
    # (`netCDF4_.py::__setitem__`) and the source's own read tasks
    # (`netCDF4_.py::_getitem`) run in the SAME pool, so writers and readers
    # each hold one sub-lock of a `CombinedLock` while waiting for the other and
    # nobody is left making progress. A py-spy dump shows every worker parked at
    # `xarray/backends/locks.py:66` with the process burning 0.08 s of CPU per
    # 20 s of wall clock -- a deadlock, not a slow write.
    #
    # It is SOURCE-SHAPED, which is why it stayed hidden: under
    # `deltares_data.yml` era5 reads from `era5_daily.zarr`, zarr takes no such
    # lock, and that store writes in seconds. chirps reads 17 yearly
    # `CHIRPS_rainfall_{year}.nc` plus `era5_orography_2018.nc` for the lapse
    # correction -- netCDF on both sides of one graph, so under THAT catalog the
    # deadlock is reachable only there.
    #
    # BEING SOURCE-SHAPED IS NOT THE SAME AS BEING CHIRPS-SHAPED, and reading
    # this as "only chirps needs the synchronous scheduler" is the mistake the
    # paragraph above invites. `deltares_data_pdrive.yml` -- hydromt's own
    # catalog against the Deltares network store, which this repo vendors --
    # points era5 at yearly global netCDFs instead (`meteo/era5_daily/nc_merged/
    # era5_{year}_daily.nc`, confirmed 2026-09-18). That is netCDF on both sides
    # again, so under that catalog the era5 path would be deadlock-REACHABLE the
    # moment anyone reintroduced a threaded scheduler here.
    #
    # It has never deadlocked, and nothing below is fixing a live fault: the
    # deadlock needs read and write tasks contending over `CombinedLock` in one
    # pool, and `scheduler="synchronous"` leaves no pool to contend in. The
    # point is only that "era5 is safe because it reads zarr" is a catalog's
    # property, not this function's to assume -- so the scheduler stays
    # synchronous for EVERY source.
    #
    # The same failure is diagnosed in bf1f4a5 and worked around at three WF2
    # call sites by materializing the READ eagerly (`.load()`) before writing.
    # That fix would be wrong here: those sites load small per-member summaries,
    # whereas this is the climate store itself, whose size is set by the basin
    # and the historical window and is bounded by nothing in this function.
    # Eager would trade a deadlock for an OOM on a large basin. Going
    # synchronous removes the concurrency the lock contends over while still
    # streaming chunk by chunk, so peak memory is unchanged.
    #
    # THAT LAST CLAUSE WAS CHALLENGED AND HELD, 2026-09-18. A 2026-09-18
    # extraction showed resident memory tracking bytes read almost 1:1, which
    # read as evidence that this graph accumulates and that an eager read with
    # a size guard would be both faster and safer. It is not: over 2000..2002
    # the shipped path peaked at 1578 MB, an eager read at 1596 and a read with
    # no write at all at 1570. The growth was never in this graph -- it was the
    # per-variable HDF5 chunk cache on the READ side, now bounded by
    # `_bounded_hdf5_chunk_cache` above. Streaming stays, and the size argument
    # against eager stands untouched.
    #
    # Cost is one core on a zlib-bound write. `DaskProgress` is unaffected:
    # `dask.local.get_sync` drives the same callback machinery, so the bar
    # still renders (verified, not assumed).
    #
    # Set through `dask.config` rather than passed as `compute(scheduler=...)`
    # so the call signature stays what every caller and fake already expects,
    # and so any compute nested inside `to_netcdf` inherits it too.
    with DaskProgress(f"{clim_source} store"), dask.config.set(scheduler="synchronous"):
        delayed_obj.compute()
    # Release the store handles deterministically rather than leaving them to
    # the garbage collector. Good practice on its own terms.
    #
    # It does NOT silence the "Error in sys.excepthook: / Original exception
    # was:" cascade that follows this rule under Snakemake -- measured, 14 lines
    # before and after. Recorded so nobody retries it: the cascade reproduces
    # ONLY under Snakemake's `script:` execution (0 lines standalone, same data
    # and same tee), and a probe excepthook installed to capture the original
    # exception never fired -- which is itself the diagnosis. By the time these
    # fire, CPython finalization has already torn module globals down, so any
    # excepthook fails and the interpreter prints the bare marker pair. It is
    # post-success noise from interpreter shutdown, not from this workflow.
    ds.close()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            # Two declared inputs. The catalog is the store's
            # freshness boundary (R07 ext2-01), so an in-place catalog edit
            # mtime-triggers exactly one re-extraction; the region is the shared
            # project artifact, delineated once by WF0 rule 0.01, WF1 rule 1.02,
            # WF2 rule 2.02 or WF3 rule 3.03 rather than re-derived here per key.
            catalog = sm.input.catalog
            region_fn = sm.input.region_geojson
            gdf = read_region(region_fn)
            prep_historical_climate(
                region_fn=None,
                fn_out=sm.output.climate_nc,
                data_libs=catalog,
                clim_source=sm.params.clim_source,
                starttime=sm.params.starttime,
                endtime=sm.params.endtime,
                bbox=tuple(gdf.total_bounds),
                # Absent outside the chirps/chirps_global branch, where the spec
                # declares no oro_nc output.
                oro_out=getattr(sm.output, "oro_nc", None),
                # Already in the store spec's params for the delineation edge;
                # the chirps branch reads its DEM from the same entry.
                hydrography=sm.params.hydrography,
                region_sha256=file_sha256(region_fn),
                region_source=region_fn,
                # Absent everywhere except wf0's extra `candidate_sources`.
                # `climate_store_rule` omits the param entirely when the floor
                # is enforced, so every pre-existing declaration -- and every
                # store already on disk -- keeps its params byte-identical.
                enforce_min_years=getattr(sm.params, "enforce_min_years", True),
                forcing_required=getattr(sm.params, "forcing_required", True),
            )
            # Which of the extracted cells the basin actually touches. Written
            # HERE rather than derived by the consumer because this is the only
            # place holding both the grid and the region polygon, and because R
            # has no geometry library in this env (no sf/terra/sp) -- so a
            # consumer-side mask would need a new dependency to do what one CSV
            # answers.
            write_basin_cell_mask(sm.output.climate_nc, gdf, sm.output.basin_cells)
    else:
        # Standalone demo (no Snakemake). Point the paths and the region at your
        # own project before running; the shape mirrors the rule above.
        demo_dir = join(os.getcwd(), "_climate_store_demo")
        demo_gdf = delineate_region(
            "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
            "deltares_data",
            region_out=join(demo_dir, "region.geojson"),
        )
        prep_historical_climate(
            region_fn=None,
            fn_out=join(demo_dir, "extract_historical.nc"),
            data_libs="deltares_data",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",
            bbox=tuple(demo_gdf.total_bounds),
        )
