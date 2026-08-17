"""Extract historical climate data for a given region and time period.

The SINGLE producer of the shared ``data/climate/historical/<key>/`` store.
Declared identically as ``extract_historical_climate`` in ``build_model.smk``
(1.04) and ``run_stress_test.smk`` (3.08), and generated per candidate source
as ``extract_historical_climate_<source>`` by ``analyze_climate.smk`` (0.04) —
all from ``snake_utils.climate_store_rule`` (R07 B1). The extraction extent stays
**model-free**: it comes from ``data/spatial/geoms/region.geojson``, the one
project region artifact delineated from ``shared.basin`` + the catalog by rule
``delineate_region`` (ADR 0003), so nothing here reads a built model.

Until ADR 0003 this script delineated that polygon itself and wrote a
per-store-key copy (``store_region.geojson``). The copy is gone; the store's
extent provenance now travels *in* ``extract_historical.nc`` as the
``region_bbox`` / ``region_geojson_sha256`` / ``region_source`` attributes,
which cannot be separated from the data they describe.
"""

import os
from os.path import join
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import hydromt
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
#: Two, not one, since 2026-08-10. The store is becoming the forcing source for
#: rule 1.10 instead of that rule re-reading the global dataset from the
#: catalog, and hydromt reads precipitation for a model region with
#: ``buffer=2`` (``hydromt_wflow/wflow_sbm.py:3288``,
#: ``setup_precip_forcing``). A store built at ``buffer=1`` is one ring short of
#: what that reader sees, so the regrid onto the model grid would differ at the
#: basin edge -- silently, and worst on the small basins this toolbox targets.
#:
#: The ring is NOT free downstream: weathergenr averages every cell in the store
#: (``compute_area_averages``, an unweighted mean over ``n_grids``), so widening
#: alone would dilute the series driving the stress test. That is why the
#: extraction also writes ``basin_cells.csv`` -- see ``write_basin_cell_mask``.
BUFFER_CELLS = 2

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
        f"Basin touches {len(frame)} of {len(lats) * len(lons)} store cells "
        f"-> {os.path.basename(str(out_csv))}",
        module="extract",
    )
    return frame


#: What the below-the-floor message tells an ENFORCING caller to do. The
#: reporting caller gets ``_FLOOR_ADVISORY`` instead: for a candidate that only
#: feeds a comparison figure there is nothing to fix, only a consequence to know.
_FLOOR_REMEDY = (
    "The staged source does not cover the configured historical_window. Either "
    "stage data for that period, or move shared.historical_window onto the "
    "years the source actually holds"
)

_FLOOR_ADVISORY = (
    "This source is a comparison candidate only, so the extraction proceeds -- "
    "but it cannot be promoted to shared.clim_historical over this window: "
    "weathergenr would reject the record and WF3 would fail at rule 3.11"
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
    Snakemake's rerun trigger) and meets the floor here. Rule 1.10 checks the
    store it consumes as well -- see ``model/add_climate_forcing.py`` -- so the
    guarantee does not rest on that trigger alone.
    """
    return report_coverage(
        resolve_coverage(time_axis_bounds(ds), starttime, endtime, clim_source),
        enforce_min_years=enforce_min_years,
        where=_FLOOR_REMEDY if enforce_min_years else _FLOOR_ADVISORY,
    )


def _read_source(data_catalog, source, *, requested, **kwargs):
    """``get_rasterdataset`` with a NoDataException that names the window.

    Partial coverage never reaches here: the convention resolver globs one URI
    per year and simply skips the years a source has no file for, and hydromt's
    temporal slice returns the overlap. ``NoDataException`` therefore means ZERO
    overlap -- the one shortfall no widest-possible-range can rescue -- and
    hydromt's own message names neither the source nor the window that missed.
    """
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
):
    """
    Extract historical climate data for a given region and time period.

    If clim_source is chirps or chirps_global, then only precip is extracted and will be
    combined with other climate data from era5.

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
        on the extraction as ``region_geojson_sha256`` (ADR 0003).
    region_source : str, Path, optional
        Path of that artifact, stamped as ``region_source``.
    enforce_min_years : bool, optional
        Whether a delivered record below ``MIN_HISTORICAL_YEARS`` is an ERROR
        (default) or a logged warning. ``False`` only for wf0's extra
        ``candidate_sources``, which end at a comparison figure; see
        ``_check_window_coverage``.
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

    # Extract climate data
    log_row("Extracting historical climate grid", module="extract")
    if clim_source == "chirps" or clim_source == "chirps_global":  # precip only
        log_row(
            f"{clim_source} only contains precipitation data. Combining with climate data from era5",
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
        # Get clim
        ds_clim = _read_source(
            data_catalog,
            "era5",
            requested=(starttime, endtime),
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=["temp", "temp_min", "temp_max", "kin", "kout", "press_msl"],
        )
        # THE STORE'S WINDOW IS WHAT BOTH SOURCES COVER, and clipping to it here
        # is load-bearing rather than tidy. The six era5-derived variables are
        # assigned into `ds` below, and `ds[var] = da` REINDEXES `da` onto `ds`'s
        # time axis -- so a chirps record longer than the era5 one would leave
        # real precipitation beside all-NaN temperature and radiation over the
        # non-overlap. That store passes WG-1 (all seven variables present, right
        # dtypes) and hands the NaNs to weathergenr's area average twenty rules
        # later. Intersecting first turns a silent corruption into a window that
        # is merely shorter, which is the whole stance of this change.
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
        # Prepare orography data corresponding to chirps from the CONFIGURED
        # hydrography DEM (needed for downscaling of climate variables) -- the
        # same catalog entry the delineation and the model build read, not a
        # second elevation source only this branch names.
        log_row(
            f"Preparing orography data for {clim_source} from {hydrography} "
            "to downscale climate variables.",
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
        # Resample other variables and add to ds_precip
        log_row(
            f"Downscaling era5 variables to the resolution of {clim_source}",
            module="extract",
        )
        for var in ["press_msl", "kin", "kout"]:
            ds[var] = ds_clim[var].raster.reproject_like(ds, method="nearest_index")

        # Read era5 dem for temp downscaling
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
        # Save dem grid to netcdf, at the caller's declared output when given.
        fn_dem = (
            os.fspath(oro_out)
            if oro_out is not None
            else os.path.join(os.path.dirname(fn_out), f"{clim_source}_orography.nc")
        )
        dem.to_netcdf(fn_dem, mode="w")

    else:
        # Here we can afford larger chunks as we only extract and save.
        # In hydromt 1.x the source schema changed: chunks lives under
        # driver.options instead of the old top-level driver_kwargs.
        data_catalog_temp = data_catalog.to_dict()
        source = data_catalog_temp[clim_source]
        driver = source.setdefault("driver", {})
        if isinstance(driver, str):
            driver = {"name": driver}
            source["driver"] = driver
        driver.setdefault("options", {})["chunks"] = "auto"
        data_catalog = hydromt.DataCatalog().from_dict(data_catalog_temp)

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

    # The store's extent provenance, IN the extraction rather than beside it
    # (ADR 0003). `store_region.geojson` used to sit in the store directory as
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
    # name. Both calls are idempotent, so the era5 path is unaffected.
    ds = _stamp_catalog_metadata(ds, data_catalog, clim_source)
    ds = _coerce_store_dtypes(ds)

    ds.attrs["region_bbox"] = [float(value) for value in bbox]
    if region_sha256 is not None:
        ds.attrs["region_geojson_sha256"] = region_sha256
    if region_source is not None:
        ds.attrs["region_source"] = os.fspath(region_source)

    dvars = ds.raster.vars
    encoding = {k: {"zlib": True} for k in dvars}

    log_row("Saving to netcdf", module="extract")
    delayed_obj = ds.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
    # Labelled with the SOURCE, because a multi-source project runs this rule
    # once per source and the console would otherwise show identical bars.
    with DaskProgress(f"{clim_source} store"):
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
            # Two declared inputs (ADR 0003). The catalog is the store's
            # freshness boundary (R07 ext2-01), so an in-place catalog edit
            # mtime-triggers exactly one re-extraction; the region is the shared
            # project artifact, delineated once by rule 1.01b/2.03b/3.01b rather
            # than re-derived here per store key.
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
