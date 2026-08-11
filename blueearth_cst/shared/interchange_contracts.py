"""Interchange-contract validators for the two CST substitution seams.

This module pins — as machine-checkable, pure functions — the contract
surfaces at the two points where a CST component could be swapped for an
alternative implementation:

- the **weather-generator seam** (``weathergenr`` today): validators
  ``validate_wg1``..``validate_wg6`` + the relational
  ``validate_wg5_catalog_grid``;
- the **hydrological-model seam** (Wflow-SBM built by hydromt today):
  validators ``validate_hm1``..``validate_hm7`` (no ``validate_hm6a`` — its
  contract surface is pinned transitively by HM-4) + the relational
  ``validate_hm_gauge_column_identity``.

Source of record: ``dev/milestones/p32b/interchange-contracts-design.md`` (ACCEPTED
2026-07-24, §5.5) and the two seam docs ``dev/reference/contracts/*-seam.md``.

Design invariants this module obeys (do not relax without a design change):

1. **Pure functions over PARSED objects.** Every validator takes an already
   parsed object — ``xarray.Dataset`` / ``pandas.DataFrame`` /
   ``dict``-from-yaml / ``dict``-from-``tomllib`` / ``geopandas.GeoDataFrame``
   — never a path. The caller owns all file I/O. This is what lets the same
   function serve a synthetic in-memory unit test, a real-fixture integration
   test, and a future in-pipeline guard with no move (design C5).
2. **``-> list[str]`` divergence report; empty list ⇒ pass.** Mirrors the
   house drift-guard ``compare_project_consistency``
   (``blueearth_cst/experiment/check_project_consistency.py``), not the
   ``ValueError``-raising ``validate_experiment_name``. Every violation is
   surfaced at once (better for a swapper diagnosing a candidate artifact).
3. **No ``assert`` / ``AssertionError`` in validator bodies.** ``assert`` is
   stripped under ``python -O`` / ``PYTHONOPTIMIZE``, so a future optimized
   in-pipeline guard lifting these functions would silently no-op — it would
   fail *open* on exactly the path this module is built for (design §6.5).
   A returned report never vanishes.
4. **Asserted-if-present semantics** where the design records a property but
   does not pin it as a hard contract surface — chiefly the HM-2 forcing
   units (wflow is name-keyed via the TOML ``[input.forcing]`` block, so no
   consumer reads the unit attr): the validator appends a message only when
   the attr is *present and wrong*; an absent attr never blocks (design §5.5).
5. **CST automation scope (C3).** Validators pin OUR consumed/rewritten
   subset of the upstream hydromt / wflow / weathergenr formats; they never
   assert an upstream-owned internal (the full staticmaps schema, wflow
   physics blocks, the outlets-map id derivation). See the per-validator
   docstrings for the pinned-vs-unpinned boundary.

Stdlib + xarray + pandas + geopandas + pyyaml + tomllib only — no new
dependency. This module is imported by ``tests/test_interchange_contracts.py``
and by **no** Snakefile rule: it is DAG-invisible and changes no pipeline
behavior (design C2).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Shared internal helpers (small, factored; not part of the public surface).
# All accept parsed objects and return list[str] fragments.
# ---------------------------------------------------------------------------


def _check_dims(ds: Any, expected: Sequence[str], label: str) -> list[str]:
    """Report any expected dimension name absent from ``ds.sizes``."""
    have = set(getattr(ds, "sizes", {}))
    return [
        f"{label}: expected dimension {d!r} absent (have {sorted(have)})"
        for d in expected
        if d not in have
    ]


def _check_coords(ds: Any, expected: Sequence[str], label: str) -> list[str]:
    """Report any expected coordinate name absent from ``ds.coords``."""
    have = set(getattr(ds, "coords", {}))
    return [
        f"{label}: expected coordinate {c!r} absent (have {sorted(have)})"
        for c in expected
        if c not in have
    ]


def _check_data_vars(ds: Any, expected: Sequence[str], label: str) -> list[str]:
    """Report any expected data variable absent from ``ds.data_vars``."""
    have = set(getattr(ds, "data_vars", {}))
    return [
        f"{label}: expected data variable {v!r} absent (have {sorted(have)})"
        for v in expected
        if v not in have
    ]


def _check_var_dtype(ds: Any, var: str, dtype: str, label: str) -> list[str]:
    """Report a present variable whose dtype kind differs from ``dtype``.

    Compares by numpy dtype *name* (e.g. ``float32``). Skips silently if the
    variable is absent — that omission is a ``_check_data_vars`` finding, not
    a dtype finding, so it is not double-reported here.
    """
    if var not in getattr(ds, "data_vars", {}):
        return []
    actual = str(ds[var].dtype)
    if actual != dtype:
        return [f"{label}: variable {var!r} dtype {actual} != expected {dtype}"]
    return []


def _check_crs_4326(ds: Any, label: str) -> list[str]:
    """Report a missing ``spatial_ref`` coord/variable (the EPSG:4326 marker).

    We pin the *presence* of the CRS descriptor a downstream regrid/co-registration
    relies on, not the full WKT string (upstream-owned, brittle to version bumps).
    """
    names = set(getattr(ds, "coords", {})) | set(getattr(ds, "variables", {}))
    if "spatial_ref" not in names:
        return [f"{label}: expected 'spatial_ref' CRS coordinate absent"]
    return []


def _check_global_attr(ds: Any, key: str, value: Any, label: str) -> list[str]:
    """Report a global attr that is absent or unequal to ``value`` (coerced str)."""
    attrs = getattr(ds, "attrs", {})
    if key not in attrs:
        return [f"{label}: expected global attr {key!r}={value!r} absent"]
    if str(attrs[key]) != str(value):
        return [
            f"{label}: global attr {key!r}={attrs[key]!r} != expected {value!r}"
        ]
    return []


def _check_global_attr_if_present(
    ds: Any, key: str, value: Any, label: str
) -> list[str]:
    """Report a global attr only when it is PRESENT and unequal to ``value``.

    The asserted-if-present form (the units precedent, design §5.2): absence is
    not a violation because the value's authority lives elsewhere, but a present
    contradictory value still is. Use this where a sibling contract already pins
    the field on the surface that actually carries it.
    """
    attrs = getattr(ds, "attrs", {})
    if key not in attrs:
        return []
    if str(attrs[key]) != str(value):
        return [
            f"{label}: global attr {key!r}={attrs[key]!r} != expected {value!r}"
        ]
    return []


def _columns(df: Any) -> list[str]:
    """Return a DataFrame's column labels as a plain list of str."""
    return [str(c) for c in getattr(df, "columns", [])]


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Float agreement for C28's cached-copy check.

    A tolerance rather than equality because the two sides are computed by the
    same function but travel through a CSV round-trip: the design table is
    written and re-read as text, so the results value and the design value can
    differ in the last bit without anything being wrong. The tolerance is tight
    enough that a genuine drift -- a different member, a unit confusion, a stale
    table -- is orders of magnitude outside it.
    """
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


# ---------------------------------------------------------------------------
# Weather-generator seam — WG-1, WG-2, WG-3, WG-5 (persisted).
# WG-4 / WG-6 (temp()) live in the temp-validator section below.
# ---------------------------------------------------------------------------

#: WG-1 extraction variables and their fixture-verified units (under the
#: ``units`` *plural* attr key — contrast HM-2's ``unit`` singular).
_WG1_VARS_UNITS = {
    "precip": "mm d**-1",
    "temp": "K",
    "temp_min": "K",
    "temp_max": "K",
    "kin": "J m**-2",
    "kout": "J m**-2",
    "press_msl": "Pa",
}


def validate_wg1(ds: Any) -> list[str]:
    """WG-1 — historical climate extraction (``extract_historical.nc``).

    Pinned surface (design §5.2, era5 branch): dims ``(time, latitude,
    longitude)``; ``float32`` lat/lon coords + a ``spatial_ref`` CRS; the seven
    ``float32`` variables ``precip``/``temp``/``temp_min``/``temp_max``/``kin``/
    ``kout``/``press_msl``; global attrs ``crs=4326`` / ``category=meteo``.
    WG-1 units live under the ``units`` (plural) key — asserted only where
    present (the extraction always writes them, but a swap need not).

    chirps-branch facts (precip-only + the orography sidecar) are NOT checked
    here — no chirps fixture exists (design R2); this validator is era5-grounded.
    """
    label = "WG-1"
    diffs: list[str] = []
    diffs += _check_dims(ds, ("time", "latitude", "longitude"), label)
    diffs += _check_coords(ds, ("time", "latitude", "longitude"), label)
    # Coord dtypes (float32 lat/lon is the pinned surface).
    for coord, dtype in (("latitude", "float32"), ("longitude", "float32")):
        if coord in getattr(ds, "coords", {}) and str(ds[coord].dtype) != dtype:
            diffs.append(
                f"{label}: coord {coord!r} dtype {ds[coord].dtype} != {dtype}"
            )
    diffs += _check_data_vars(ds, tuple(_WG1_VARS_UNITS), label)
    for var, units in _WG1_VARS_UNITS.items():
        diffs += _check_var_dtype(ds, var, "float32", label)
        # Units asserted-if-present (under the ``units`` plural key).
        if var in getattr(ds, "data_vars", {}):
            attrs = ds[var].attrs
            if "units" in attrs and str(attrs["units"]) != units:
                diffs.append(
                    f"{label}: {var!r} units {attrs['units']!r} != expected "
                    f"{units!r} (asserted-if-present)"
                )
    diffs += _check_crs_4326(ds, label)
    diffs += _check_global_attr(ds, "crs", 4326, label)
    diffs += _check_global_attr(ds, "category", "meteo", label)
    return diffs


#: WG-2 stress-test CSV header, exact and ordered.
_WG2_HEADER = ("month", "temp_mean", "precip_mean", "precip_variance")


def validate_wg2(df: Any) -> list[str]:
    """WG-2 — stress-test perturbation grid (``st_<m>.csv``, m>=1).

    Pinned surface (design §5.2): header exactly ``month,temp_mean,precip_mean,
    precip_variance``; 12 rows with ``month`` domain ``1..12``. Column
    *semantics* (additive vs multiplicative) are documented, not machine-checked
    — the values change per stress-test point.
    """
    label = "WG-2"
    diffs: list[str] = []
    cols = _columns(df)
    if tuple(cols) != _WG2_HEADER:
        diffs.append(
            f"{label}: header {cols} != expected {list(_WG2_HEADER)}"
        )
    n = int(getattr(df, "shape", (0,))[0])
    if n != 12:
        diffs.append(f"{label}: expected 12 rows (month 1..12), got {n}")
    if "month" in cols:
        months = sorted(int(m) for m in df["month"].tolist())
        if months != list(range(1, 13)):
            diffs.append(f"{label}: month domain {months} != 1..12")
    return diffs


#: WG-3 config surface — the key set the R side reads (design §5.2, read-only
#: from weathergen/{global.R,generate_weather.R}). Upstream-spelled dot.case
#: keys preserved verbatim (naming.md §2 — YAML under an upstream schema).
_WG3_GWS_KEYS = (
    "knn.sample.num",
    "month.start",
    "warm.variable",
    "seed",
    "dry.spell.change",
    "wet.spell.change",
    "output.path",
    "sim.year.start",
    "sim.year.num",
    "nc.file.prefix",
    "realizations_num",
    # C34 (R11 P2): surfaced weathergenr arguments. Pinned for the same reason
    # the transient flags are -- the R reads them, and an omission would reach
    # the generator as NULL and silently take whatever upstream defaults to.
    "save.plots",
    "pet.method",
)


def validate_wg3(cfg: Any) -> list[str]:
    """WG-3 — weathergenr config surface (``weathergen_config.yml``).

    Pinned surface (design §5.2, OQ-6): the *key set* the R side reads —
    top-level ``general.variables`` (a list), the ``generateWeatherSeries``
    key set, and the two ``transient_change`` flags — NOT weathergenr's config
    *semantics* or value ranges. A replacement generator may define its own
    config surface entirely; this pins the *current* generator's contract.

    **ONE file since C29 (2026-08-05).** This used to cover a second, per-member
    ``weathergen_config_rlz_<n>_cst_<m>.yml`` as well. That file carried nothing
    that varied except its own output filename, so it was retired with rule 3.05
    and its two ``transient_change`` flags moved here — which is why they are
    pinned. ``impose_climate_change.R`` now reads THIS file.
    """
    label = "WG-3"
    diffs: list[str] = []
    if not isinstance(cfg, Mapping):
        return [f"{label}: config is not a mapping ({type(cfg).__name__})"]
    general = cfg.get("general")
    if not isinstance(general, Mapping) or "variables" not in general:
        diffs.append(f"{label}: 'general.variables' section absent")
    elif not isinstance(general["variables"], list):
        diffs.append(
            f"{label}: 'general.variables' must be a list, got "
            f"{type(general['variables']).__name__}"
        )
    gws = cfg.get("generateWeatherSeries")
    if not isinstance(gws, Mapping):
        diffs.append(f"{label}: 'generateWeatherSeries' section absent")
    else:
        for key in _WG3_GWS_KEYS:
            if key not in gws:
                diffs.append(
                    f"{label}: 'generateWeatherSeries.{key}' absent"
                )
    # The perturbation-step flags (C29). Read by impose_climate_change.R; absent,
    # the R hands NULL to apply_climate_perturbations and the ramp-vs-step
    # behaviour is whatever weathergenr defaults to.
    for section in ("temp", "precip"):
        block = cfg.get(section)
        if not isinstance(block, Mapping) or "transient_change" not in block:
            diffs.append(f"{label}: '{section}.transient_change' absent")
    return diffs


#: WG-5 per-entry hydromt data-catalog fields (OUR emitted subset — design §5.2).
def _validate_catalog_entry(key: str, entry: Any, label: str) -> list[str]:
    """Check one hydromt catalog entry against OUR emitted-subset schema."""
    diffs: list[str] = []
    if not isinstance(entry, Mapping):
        return [f"{label}: entry {key!r} is not a mapping"]
    if entry.get("data_type") != "RasterDataset":
        diffs.append(
            f"{label}: entry {key!r} data_type "
            f"{entry.get('data_type')!r} != 'RasterDataset'"
        )
    if "uri" not in entry:
        # The uri VALUE is deliberately unpinned (machine-scoped path); only
        # its presence is a contract (the catalog must point somewhere).
        diffs.append(f"{label}: entry {key!r} missing 'uri'")
    driver = entry.get("driver")
    if not isinstance(driver, Mapping):
        diffs.append(f"{label}: entry {key!r} missing 'driver' mapping")
    else:
        if driver.get("name") != "raster_xarray":
            diffs.append(
                f"{label}: entry {key!r} driver.name "
                f"{driver.get('name')!r} != 'raster_xarray'"
            )
        options = driver.get("options")
        if not isinstance(options, Mapping):
            diffs.append(f"{label}: entry {key!r} missing 'driver.options'")
        else:
            if options.get("preprocess") != "harmonise_dims":
                diffs.append(
                    f"{label}: entry {key!r} driver.options.preprocess "
                    f"{options.get('preprocess')!r} != 'harmonise_dims'"
                )
            if options.get("lock") is not False:
                diffs.append(
                    f"{label}: entry {key!r} driver.options.lock "
                    f"{options.get('lock')!r} != false"
                )
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        diffs.append(f"{label}: entry {key!r} missing 'metadata' mapping")
    else:
        if str(metadata.get("crs")) != "4326":
            diffs.append(
                f"{label}: entry {key!r} metadata.crs "
                f"{metadata.get('crs')!r} != 4326"
            )
        if metadata.get("category") != "meteo":
            diffs.append(
                f"{label}: entry {key!r} metadata.category "
                f"{metadata.get('category')!r} != 'meteo'"
            )
    return diffs


def validate_wg5(cfg: Any) -> list[str]:
    """WG-5 — hydromt climate data catalog (``data_catalog_climate_experiment.yml``).

    Pinned-as-reliance (design §5.2): OUR emitted subset of the hydromt
    data-catalog schema — for every ``rlz_<n>_st_<m>`` entry the driver /
    metadata fields ``{uri, driver.name=raster_xarray,
    driver.options.preprocess=harmonise_dims, driver.options.lock=false,
    metadata.crs=4326, metadata.category=meteo, data_type=RasterDataset}``.

    This pins per-entry *bookkeeping* only, NOT the NC content the entries point
    at (that is WG-4 / WG-6's contract) and NOT the entry-key grid completeness
    (that is the relational ``validate_wg5_catalog_grid``). The ``uri`` VALUE is
    deliberately unpinned (machine-scoped absolute path); only its presence is
    checked.
    """
    label = "WG-5"
    if not isinstance(cfg, Mapping):
        return [f"{label}: catalog is not a mapping ({type(cfg).__name__})"]
    diffs: list[str] = []
    entries = {
        k: v for k, v in cfg.items() if isinstance(k, str) and k.startswith("rlz_")
    }
    if not entries:
        diffs.append(f"{label}: no 'rlz_<n>_st_<m>' entries in catalog")
    for key in sorted(entries):
        diffs += _validate_catalog_entry(key, entries[key], label)
    return diffs


# ---------------------------------------------------------------------------
# Hydrological-model seam — HM-1, HM-2, HM-3, HM-4, HM-5, HM-7 (persisted).
# HM-6a: no validator (existence pinned transitively via HM-4).
# HM-6b (temp()) lives in the temp-validator section below.
# ---------------------------------------------------------------------------

#: HM-1 OUR-referenced staticmaps variable names (design §5.3, pinned-as-reliance).
_HM1_REFERENCED = (
    "subcatchment",
    "land_elevation",
    "local_drain_direction",
    "river_mask",
    "outlets",
)


def validate_hm1(ds: Any) -> list[str]:
    """HM-1 — static grid (``staticmaps.nc``).

    Pinned-as-reliance (design §5.3, C3): ONLY the OUR-referenced variable names
    ``subcatchment`` / ``land_elevation`` / ``local_drain_direction`` /
    ``river_mask`` / ``outlets``, on ``(latitude, longitude)`` ``float64``
    coords + a ``spatial_ref`` CRS. The grid definition (the axes) is pinned as
    the co-registration target forcing must match.

    The remaining ~39 wflow variables (``vegetation_*``, ``soil_*``, ``meta_*``,
    river vars beyond the mask) are **wflow schema, consumed verbatim, unpinned**
    — this validator never enumerates or asserts them (C3).
    """
    label = "HM-1"
    diffs: list[str] = []
    diffs += _check_coords(ds, ("latitude", "longitude"), label)
    for coord, dtype in (("latitude", "float64"), ("longitude", "float64")):
        if coord in getattr(ds, "coords", {}) and str(ds[coord].dtype) != dtype:
            diffs.append(
                f"{label}: coord {coord!r} dtype {ds[coord].dtype} != {dtype}"
            )
    diffs += _check_data_vars(ds, _HM1_REFERENCED, label)
    diffs += _check_crs_4326(ds, label)
    return diffs


#: HM-2 forcing variable names (the consumer contract — TOML [input.forcing]
#: RHS values) and their fixture-observed unit attrs (asserted-if-present only).
_HM2_VARS = ("precip", "pet", "temp")
#: Observed unit-attr layout (design §5.3). Each (var, attr-key, value); asserted
#: ONLY when the attr is present — wflow is name-keyed, so no consumer reads it.
_HM2_UNIT_ATTRS = (
    ("precip", "units", "mm d**-1"),
    ("precip", "unit", "mm"),
    ("pet", "unit", "mm"),
    ("temp", "unit", "degree C."),
)


def validate_hm2(ds: Any) -> list[str]:
    """HM-2 — Wflow forcing (``inmaps_historical.nc``; wf3 twin = WG-6).

    Pinned surface (design §5.3): dims ``(time, latitude, longitude)`` on the
    model grid (``float64`` lat/lon matching HM-1); data vars exactly ``precip``
    / ``pet`` / ``temp``, all ``float32``, each ``grid_mapping=spatial_ref``; a
    ``spatial_ref`` EPSG:4326 CRS. The variable *names* are the consumer contract
    — they are the RHS values the TOML ``[input.forcing]`` block maps to (HM-4).

    UNITS NOT PINNED (design arch-2/risk-4): wflow is name-keyed, so no consumer
    reads the unit attr. The observed layout (``precip`` carries both
    ``units='mm d**-1'`` and ``unit='mm'``; ``pet`` ``unit='mm'``; ``temp``
    ``unit='degree C.'``) is asserted **only if the attr is present and wrong** —
    an absent unit attr never blocks (asserted-if-present, design §5.5).
    """
    label = "HM-2"
    diffs: list[str] = []
    diffs += _check_dims(ds, ("time", "latitude", "longitude"), label)
    diffs += _check_coords(ds, ("time", "latitude", "longitude"), label)
    for coord, dtype in (("latitude", "float64"), ("longitude", "float64")):
        if coord in getattr(ds, "coords", {}) and str(ds[coord].dtype) != dtype:
            diffs.append(
                f"{label}: coord {coord!r} dtype {ds[coord].dtype} != {dtype}"
            )
    diffs += _check_data_vars(ds, _HM2_VARS, label)
    for var in _HM2_VARS:
        diffs += _check_var_dtype(ds, var, "float32", label)
        if var in getattr(ds, "data_vars", {}):
            gm = ds[var].attrs.get("grid_mapping")
            if gm != "spatial_ref":
                diffs.append(
                    f"{label}: {var!r} grid_mapping {gm!r} != 'spatial_ref'"
                )
    # Units asserted-if-present only (never required).
    for var, attr_key, value in _HM2_UNIT_ATTRS:
        if var in getattr(ds, "data_vars", {}):
            attrs = ds[var].attrs
            if attr_key in attrs and str(attrs[attr_key]) != value:
                diffs.append(
                    f"{label}: {var!r} {attr_key!r}={attrs[attr_key]!r} != "
                    f"expected {value!r} (asserted-if-present)"
                )
    diffs += _check_crs_4326(ds, label)
    return diffs


def validate_hm3(
    region_gdf: Any,
    outlets_gdf: Any,
    outlet_index_df: Any,
) -> list[str]:
    """HM-3 — static vector geometries (``staticgeoms/``).

    Pinned surface — OUR-consumed vectors only (design §5.3): ``region.geojson``
    (a Polygon basin extent, EPSG:4326 — the wf3 extraction region + ancient()
    DAG edge); ``outlets.geojson`` (Point gauges → plots/outputs); and
    ``outlet_index.csv`` (the outlet→subcatchment-id mapping, a ``rule all``
    target). The ``basins``/``rivers``/``meta_*`` layers we do not index are
    deliberately unpinned.
    """
    label = "HM-3"
    diffs: list[str] = []
    # region: Polygon, EPSG:4326
    if str(getattr(region_gdf, "crs", None)) not in ("EPSG:4326", "epsg:4326"):
        diffs.append(f"{label}: region.geojson CRS {region_gdf.crs} != EPSG:4326")
    region_types = set(region_gdf.geom_type)
    if not region_types <= {"Polygon", "MultiPolygon"}:
        diffs.append(
            f"{label}: region.geojson geom types {sorted(region_types)} "
            "are not Polygon"
        )
    # outlets: Point, EPSG:4326
    if str(getattr(outlets_gdf, "crs", None)) not in ("EPSG:4326", "epsg:4326"):
        diffs.append(
            f"{label}: outlets.geojson CRS {outlets_gdf.crs} != EPSG:4326"
        )
    outlet_types = set(outlets_gdf.geom_type)
    if not outlet_types <= {"Point", "MultiPoint"}:
        diffs.append(
            f"{label}: outlets.geojson geom types {sorted(outlet_types)} "
            "are not Point"
        )
    # outlet_index: the subcatchment-id mapping column must be present.
    oi_cols = _columns(outlet_index_df)
    if "subcatchment_id" not in oi_cols:
        diffs.append(
            f"{label}: outlet_index.csv missing 'subcatchment_id' column "
            f"(have {oi_cols})"
        )
    return diffs


#: HM-4 TOML rewrite/read fields OUR code touches (design §5.3,
#: downscale_climate_forcing.py:55-84). Nested via (section, key) tuples.
_HM4_REQUIRED_TIME = ("calendar", "starttime", "endtime", "timestepsecs")


def validate_hm4(cfg: Any) -> list[str]:
    """HM-4 — run configuration (``wflow_sbm.toml``, base + per-cst).

    Pinned surface — the TOML fields OUR code reads/rewrites (design §5.3):
    ``[time].{calendar,starttime,endtime,timestepsecs}``, ``dir_output``,
    ``[state].{path_input,path_output}``, ``[input].{path_static,path_forcing}``,
    ``[output.csv].path``; plus read-reliance on ``[input.forcing]`` (the
    ``precip``/``pet``/``temp`` RHS values that tie to HM-2), ``[output.csv].column``
    (drives HM-5 column identity), and ``cold_start__flag``.

    Deliberately unpinned (C3): all ``[input.static]`` physics value blocks,
    layer thicknesses, kinematic-wave params — **wflow physics, unpinned**. This
    validator never asserts a physics value or the calendar's *value* (wf1 base
    is ``proleptic_gregorian``, the wf3 rewrite is ``standard`` — both valid; the
    field's *presence* is the contract, its value is a documented rewrite fact).
    """
    label = "HM-4"
    if not isinstance(cfg, Mapping):
        return [f"{label}: TOML config is not a mapping ({type(cfg).__name__})"]
    diffs: list[str] = []
    if "dir_output" not in cfg:
        diffs.append(f"{label}: top-level 'dir_output' absent")
    time = cfg.get("time")
    if not isinstance(time, Mapping):
        diffs.append(f"{label}: '[time]' section absent")
    else:
        for key in _HM4_REQUIRED_TIME:
            if key not in time:
                diffs.append(f"{label}: '[time].{key}' absent")
    state = cfg.get("state")
    if not isinstance(state, Mapping):
        diffs.append(f"{label}: '[state]' section absent")
    else:
        for key in ("path_input", "path_output"):
            if key not in state:
                diffs.append(f"{label}: '[state].{key}' absent")
    inp = cfg.get("input")
    if not isinstance(inp, Mapping):
        diffs.append(f"{label}: '[input]' section absent")
    else:
        for key in ("path_static", "path_forcing"):
            if key not in inp:
                diffs.append(f"{label}: '[input].{key}' absent")
        forcing = inp.get("forcing")
        if not isinstance(forcing, Mapping):
            diffs.append(f"{label}: '[input.forcing]' section absent")
        else:
            # Read-reliance: the RHS values (var names) must include precip/pet/temp.
            rhs = set(str(v) for v in forcing.values())
            missing = [v for v in _HM2_VARS if v not in rhs]
            if missing:
                diffs.append(
                    f"{label}: '[input.forcing]' RHS values missing {missing} "
                    f"(have {sorted(rhs)})"
                )
    model = cfg.get("model")
    if not isinstance(model, Mapping) or "cold_start__flag" not in model:
        diffs.append(f"{label}: '[model].cold_start__flag' absent")
    diffs += _output_csv_diffs(cfg, label)
    return diffs


def _output_csv_diffs(cfg: Mapping, label: str) -> list[str]:
    """Check the ``[output.csv]`` block (``path`` + ``column`` list-of-tables)."""
    diffs: list[str] = []
    output = cfg.get("output")
    csv = output.get("csv") if isinstance(output, Mapping) else None
    if not isinstance(csv, Mapping):
        diffs.append(f"{label}: '[output.csv]' section absent")
        return diffs
    if "path" not in csv:
        diffs.append(f"{label}: '[output.csv].path' absent")
    column = csv.get("column")
    if not isinstance(column, list) or not column:
        diffs.append(f"{label}: '[output.csv].column' absent or empty")
    else:
        for i, entry in enumerate(column):
            if not isinstance(entry, Mapping) or "header" not in entry:
                diffs.append(
                    f"{label}: '[output.csv].column[{i}]' missing 'header'"
                )
    return diffs


def validate_hm5(df: Any) -> list[str]:
    """HM-5 — per-run discharge CSV (``output.csv`` / ``output_rlz_*.csv``).

    Pinned surface (design §5.3): a ``time`` index (ISO-8601 daily) + one column
    per ``[output.csv].column`` entry, named ``<header>_<mapid>``. Column
    identity is config-driven, NOT a literal gauge list — so this per-artifact
    validator checks the *structural* contract (a time axis + at least one
    non-time column). The cross-file gauge-column identity across HM-4→HM-5→HM-7
    is the relational ``validate_hm_gauge_column_identity``.

    The DataFrame is expected with ``time`` as a **column** (the default
    ``pd.read_csv`` shape); numeric discharge values are deliberately unpinned.
    """
    label = "HM-5"
    diffs: list[str] = []
    cols = _columns(df)
    if "time" not in cols:
        diffs.append(f"{label}: no 'time' column (have {cols})")
    non_time = [c for c in cols if c != "time"]
    if not non_time:
        diffs.append(f"{label}: no non-'time' gauge column present")
    return diffs


#: HM-7 perturbation-axis columns, shared by both indicator tables and by the
#: relational gauge-identity check (which derives its gauge set by subtracting
#: them). Named once so the two validators cannot drift apart on a rename — they
#: did not during the 2026-08-05 tavg/prcp rename only because both were edited
#: in the same commit.
from blueearth_cst.shared.indicator_tables import (  # noqa: E402
    INDICATOR_COLUMNS,
    POOLED_REALIZATION,
    metric_grain,
)

#: The member-index padding width (C27). Imported rather than reimplemented so
#: the catalog-key expectation below and the Snakefile's own filenames cannot
#: disagree about how wide an index is. `snake_utils` does not import this
#: module, so the direction is one-way and no cycle exists.
from blueearth_cst.shared.snake_utils import index_width  # noqa: E402

_PERTURBATION_AXIS = ("temp_change", "precip_change")

#: The six HM-7 columns, in order. Imported from the writer's own module rather
#: than restated, so the producer and its validator cannot disagree about the
#: header -- the failure mode this pairing exists to prevent.
HM7_COLUMNS = INDICATOR_COLUMNS


def validate_hm7(
    tables: dict, rlz_num: int | None = None, design=None
) -> list[str]:
    """HM-7 — response-surface reduction, ONE LONG TABLE PER OUTPUT VARIABLE.

    ``tables`` maps variable token to parsed table (``{"q": df, "aet": df}``).
    ``rlz_num`` is optional; supply it to check the ``rlz_id`` domain.
    ``design`` is the parsed ``stress_test_design.csv``; supply it to check C28's
    consistency obligation.

    **The header this asserts, exactly and in order:**

        metric, location, st_id, rlz_id, temp_change, precip_change, value

    R11 CR-2 replaced the two wide tables with a fixed SIX-column long shape
    (``metric, temp_change, precip_change, realization_id, location, value``);
    C28 added ``st_id`` to make it seven, and the 2026-08-11 owner ruling reordered
    those seven identifier-first and renamed ``realization_id`` to ``rlz_id``. The
    counts in that sentence are the historical ones and are not a typo — the shape
    was genuinely six wide before C28.

    What has held throughout is the property that matters: the header no longer
    grows with the gauge count, which is why this validator can assert it exactly
    rather than by membership — the previous version had to widen to a membership
    test precisely because the shape was config-dependent in the wrong dimension.
    A reorder does not touch that property, which is why it is a cheap change.

    Three things it asserts that nothing else can:

    - **``metric`` agrees with the table it is in.** The composite carries the
      variable, so no ``variable`` column exists; the redundancy is safe only if
      something checks it, which is what normalisation would have given free.
    - **The vocabulary, as a PATTERN.** Two suffixes interpolate ``Tpeak``/``Tlow``,
      so enumerating names would reject every project whose return periods differ
      from the fixture's.
    - **The grain invariant.** ``rlz_id = 0`` means pooled — a numeric
      sentinel in a numeric key column, which is safe only because no metric
      emits both grains. If that ever stops holding, the sentinel must become a
      string, and this check is what would catch it.

    Superseded and worth not re-deriving: the axis columns were spelled
    ``tavg``/``prcp`` until 2026-08-05, the repo's only violation of the
    ``precip``/``temp`` vocabulary; the ``_change`` suffix is not decoration, since
    these columns are the perturbation each member imposes rather than the
    variable's value. The ``RT_*.csv`` side tables are gone as of R9 P3 — no
    in-repo consumer, written via ``params`` rather than declared, so invisible to
    ``--dry-run``. Nothing replaces them.

    **C28 (R11 P2) added ``st_id`` alongside the perturbation columns**, making
    the header seven wide. That was ruled "at this stage" — plottable without a
    join now, revisited when a third dimension arrives — and it re-couples the
    header to the stress-dimension count, which is what CR-2 removed on the
    location axis. Two obligations keep that interim decision from rotting:

    - **The consistency check here.** ``temp_change`` / ``precip_change`` are now
      a CACHED COPY of the design table's row for that ``st_id``. The writer
      derives them independently, from the parameter files, so the two really can
      disagree — which is the point. A denormalised copy that nothing verifies is
      a copy that eventually lies. Same class as the ``metric`` /
      ``token`` agreement above.
    - **A hard stop in the writer** when ``stress_test:`` gains a third axis,
      naming C28 rather than silently adding a column.

    The check is skipped, not failed, when ``design`` is None — a caller that
    has only the tables can still assert everything else.

    **Both directions are checked, and the second was added because its absence
    hid a defect** (R11 P3, 2026-08-08). results → design catches a row whose
    cached axes drifted from the design; design → results catches a declared
    member that produced no rows at all. Only the first existed, so a seed config
    with ``run_historical: false`` dropped the ``st_0`` baseline, and with it the
    two class-C metrics that Q5 derives *from* that baseline — 180 rows and two
    of eleven metrics gone, with this validator green. A per-row check is
    stronger than a fingerprint for the rows that exist and says nothing at all
    about the rows that do not.

    Original pinned surface, for the record (design §5.3): ``q_indicators.csv``
    header ``statistic,temp_change,precip_change,<gauge-cols>``;
    ``basin_indicators.csv`` the axis plus one column per configured
    ``*_basavg`` variable.

    The gauge-location tie to HM-4/HM-5 is checked by the relational
    ``validate_hm_gauge_column_identity``, which post-CR-2 compares the
    ``location`` column's value set rather than the header's column set — the same
    invariant, expressed against a header that no longer varies.
    """
    label = "HM-7"
    diffs: list[str] = []
    for token, table in sorted(tables.items()):
        name = f"{token}_indicators.csv"
        columns = _columns(table)
        if columns != list(HM7_COLUMNS):
            diffs.append(
                f"{label}: {name} header is {columns}, expected exactly "
                f"{list(HM7_COLUMNS)} in that order"
            )
            continue  # every check below reads these columns by name

        metrics = sorted({str(m) for m in table["metric"]})
        if not metrics:
            diffs.append(f"{label}: {name} has no rows")
            continue

        # The composite carries the variable, so `variable` needs no column of
        # its own -- but that redundancy is only safe if something asserts the
        # two agree. This is what normalisation would have given for free.
        wrong_variable = [m for m in metrics if not m.startswith(f"{token}_")]
        if wrong_variable:
            diffs.append(
                f"{label}: {name} carries metric(s) {wrong_variable} that do not "
                f"begin with the table's own variable token {token + '_'!r}"
            )

        unknown = [m for m in metrics if metric_grain(token, m) is None]
        if unknown:
            diffs.append(
                f"{label}: {name} carries unrecognised metric(s) {unknown}; the "
                f"vocabulary is in blueearth_cst/shared/indicator_tables.py"
            )

        if rlz_num is not None:
            allowed = {POOLED_REALIZATION} | set(range(1, int(rlz_num) + 1))
            stray = sorted({int(r) for r in table["rlz_id"]} - allowed)
            if stray:
                diffs.append(
                    f"{label}: {name} has rlz_id {stray}, outside "
                    f"{{0}} and 1..{rlz_num}"
                )

        # The grain invariant. `rlz_id = 0` is a numeric sentinel in a
        # numeric key column, which is safe ONLY because no metric emits both
        # grains -- otherwise `groupby('rlz_id')` folds pooled rows in as
        # another realization. Asserted here because 0 cannot announce itself.
        for metric in metrics:
            grain = metric_grain(token, metric)
            if grain is None:
                continue
            ids = {
                int(r)
                for r, m in zip(table["rlz_id"], table["metric"])
                if str(m) == metric
            }
            if grain == "pooled" and ids != {POOLED_REALIZATION}:
                diffs.append(
                    f"{label}: {name} metric {metric!r} is pooled-only but carries "
                    f"rlz_id {sorted(ids)}; expected {{0}} alone"
                )
            if grain == "per-realization" and POOLED_REALIZATION in ids:
                diffs.append(
                    f"{label}: {name} metric {metric!r} is per-realization but "
                    f"carries the pooled sentinel rlz_id 0"
                )
    # -- C28: the cached axis columns must agree with the design table --------
    if design is not None:
        by_id = {}
        for _, row in design.iterrows():
            by_id[str(row["st_id"])] = row
        for token, table in sorted(tables.items()):
            name = f"{token}_indicators.csv"
            if "st_id" not in table.columns:
                continue  # the header check above already reported it
            seen = {}
            for st_id, temp, precip in zip(
                table["st_id"], table["temp_change"], table["precip_change"]
            ):
                seen.setdefault(str(st_id), (temp, precip))
            for st_id, (temp, precip) in sorted(seen.items()):
                design_row = by_id.get(st_id)
                if design_row is None:
                    diffs.append(
                        f"{label}: {name} carries st_id {st_id!r}, which the "
                        f"stress-test design table does not define"
                    )
                    continue
                for column, value in (
                    ("temp_change", temp), ("precip_change", precip),
                ):
                    expected = float(design_row[column])
                    if not _close(float(value), expected):
                        diffs.append(
                            f"{label}: {name} st_id {st_id!r} has {column}="
                            f"{float(value)!r}, but the design table says "
                            f"{expected!r} -- the cached copy has drifted (C28)"
                        )

            # -- and the OTHER direction: every declared member produced rows --
            #
            # Added R11 P3, 2026-08-08, because the run found what its absence
            # hid. Everything above validates results -> design: for each row
            # that EXISTS, its cached axes match its design row. Nothing
            # validated design -> results, so a member that never ran was
            # invisible: the seed config had `run_historical: false`, st_0 never
            # ran, and because Q5 fixes the class-C month from the st_0 baseline,
            # `q_wettest_month_mean` and `q_driest_month_mean` were skipped
            # ENTIRELY. 180 rows and two of eleven metrics vanished with no
            # warning and a green validator.
            #
            # This is also what makes the design table earn the trust P2 placed
            # in it when it argued a byte fingerprint was unnecessary because
            # "validate_hm7's per-row check is stronger". Per-row is stronger for
            # rows that exist and says nothing about rows that do not.
            missing = sorted(set(by_id) - set(seen), key=_st_sort_key)
            if missing:
                diffs.append(
                    f"{label}: the stress-test design table declares st_id "
                    f"{missing} that produced NO rows in {name}. A member that "
                    f"never ran is not a smaller table -- metrics derived from "
                    f"it are skipped silently (Q5 fixes the class-C month from "
                    f"the st_0 baseline). Check `run_historical` / ST_START."
                )

    return diffs


def _st_sort_key(st_id: str):
    """Sort st_ids numerically when they are numeric, lexically otherwise.

    They are written zero-padded in filenames and bare in the table, and a
    project may yet carry a non-numeric one, so this must not raise on either.
    """
    try:
        return (0, int(st_id), "")
    except (TypeError, ValueError):
        return (1, 0, str(st_id))


# ---------------------------------------------------------------------------
# temp() content validators — WG-4, WG-6, HM-6b (design §5.5).
#
# These pin the CONTENT of artifacts wrapped in Snakemake ``temp()``, so every
# such netCDF is deleted after its consumer finishes and is ABSENT on the
# completed fixture. Their on-disk integration check is therefore
# skip-until-captured (the ``--notemp`` capture procedure in the seam docs),
# but their logic — like every validator here — is proven on every checkout by
# a Layer-1 synthetic pass/fail pair. Each is a pure ``-> list[str]`` function
# over a parsed ``xarray.Dataset`` (the caller opens the captured NC).
# ---------------------------------------------------------------------------


def validate_wg4(ds: Any) -> list[str]:
    """WG-4 — generator output netCDF content (``rlz_<n>_st_<m>.nc``).

    Pinned surface (design §5.2): a ``(time, lat, lon)`` raster the hydromt
    catalog (WG-5) reads — at least ``precip`` and ``temp`` on an EPSG:4326 grid
    carrying a ``spatial_ref`` CRS descriptor. The exact variable superset and
    internal attrs are deliberately unpinned.

    ``temp()`` content — absent on the completed fixture (skip-until-captured on
    disk); this logic is proven every suite by a synthetic pass/fail pair.

    Grid axes are accepted as either ``(latitude, longitude)`` or the shorter
    ``(lat, lon)`` a swap may emit — the contract is the raster (time, y, x)
    shape + the minimal variable set, not the axis spelling.

    ``crs`` / ``category`` are asserted-IF-PRESENT, not required (corrected
    2026-07-25 on the first ``--notemp`` capture, which is what this contract was
    always waiting on). The real generator artifact carries **empty** global
    attrs: its CRS travels the CF/rioxarray way, in the ``spatial_ref``
    coordinate's ``crs_wkt`` (``ID["EPSG",4326]``), while ``crs: 4326`` and
    ``category: meteo`` are supplied by the generated **data catalog** —
    ``data_catalog_climate_experiment.yml``, which is exactly where hydromt reads
    them and exactly what ``validate_wg5`` already pins
    (``metadata.crs`` / ``metadata.category``). Requiring them as file-level
    global attrs asserted the right values on the wrong surface; the pipeline
    was never non-conformant.
    """
    label = "WG-4"
    diffs: list[str] = []
    dims = set(getattr(ds, "sizes", {}))
    if "time" not in dims:
        diffs.append(f"{label}: expected 'time' dimension absent (have {sorted(dims)})")
    lat_ok = {"latitude", "lat"} & dims
    lon_ok = {"longitude", "lon"} & dims
    if not lat_ok:
        diffs.append(f"{label}: no latitude/lat dimension (have {sorted(dims)})")
    if not lon_ok:
        diffs.append(f"{label}: no longitude/lon dimension (have {sorted(dims)})")
    diffs += _check_data_vars(ds, ("precip", "temp"), label)
    diffs += _check_crs_4326(ds, label)
    # WG-5 owns crs/category on the catalog, the surface hydromt actually reads;
    # here they are only checked for contradiction (see the docstring).
    diffs += _check_global_attr_if_present(ds, "crs", 4326, label)
    diffs += _check_global_attr_if_present(ds, "category", "meteo", label)
    return diffs


def validate_wg6(ds: Any) -> list[str]:
    """WG-6 — downscaled Wflow forcing content (``inmaps_rlz_<n>_st_<m>.nc``).

    The wf3 twin of ``inmaps_historical.nc`` — the SAME contract as HM-2 (design
    §5.2/§5.3): ``(time, latitude, longitude)`` ``float32`` ``precip`` / ``pet``
    / ``temp`` on the model grid, ``spatial_ref`` EPSG:4326, each
    ``grid_mapping=spatial_ref``. This validator delegates to ``validate_hm2`` so
    the twin contract is pinned once (units asserted-if-present there).

    ``temp()`` content — absent on the completed fixture (skip-until-captured on
    disk); logic proven every suite by a synthetic pass/fail pair.
    """
    return [msg.replace("HM-2", "WG-6", 1) for msg in validate_hm2(ds)]


def validate_hm6b(ds: Any) -> list[str]:
    """HM-6b — wf3 warm state content (``outstates_rlz_<n>_st_<m>.nc``).

    THIN — an unconsumed named sink (design §5.3): nothing in-repo reads it, so
    the contract pins only that it is a wflow state **output** — an
    ``xarray.Dataset`` carrying the model grid axes (``latitude`` / ``longitude``
    or ``lat`` / ``lon``) and at least one state variable. The internal
    state-variable schema (``[state.variables]``) is **wflow-owned, unpinned**
    (C3) — this validator never enumerates or asserts a state variable's name.

    ``temp()`` content — absent on the completed fixture (skip-until-captured on
    disk); logic proven every suite by a synthetic pass/fail pair.
    """
    label = "HM-6b"
    diffs: list[str] = []
    dims = set(getattr(ds, "sizes", {}))
    if not ({"latitude", "lat"} & dims):
        diffs.append(f"{label}: no latitude/lat dimension (have {sorted(dims)})")
    if not ({"longitude", "lon"} & dims):
        diffs.append(f"{label}: no longitude/lon dimension (have {sorted(dims)})")
    if not getattr(ds, "data_vars", {}):
        diffs.append(f"{label}: no state variables present (empty dataset)")
    return diffs


# ---------------------------------------------------------------------------
# Relational validators — the two cross-artifact invariants (design §5.5).
# ---------------------------------------------------------------------------


def _declared_gauge_columns(toml_cfg: Mapping) -> tuple[list[str], list[str]]:
    """Derive expected output columns from ``[output.csv].column`` entries.

    Returns ``(expected_cols, malformed_notes)``. A map-typed entry
    ``{header, map, ...}`` yields the ``<header>_<mapid>`` *pattern*: since the
    numeric ``<mapid>`` is wflow's outlets-map cell value (wflow-owned, C3), the
    expected form is a *prefix* ``<header>_`` that a produced column must start
    with. A non-map entry yields the exact ``<header>``.
    """
    output = toml_cfg.get("output") if isinstance(toml_cfg, Mapping) else None
    csv = output.get("csv") if isinstance(output, Mapping) else None
    column = csv.get("column") if isinstance(csv, Mapping) else None
    expected: list[str] = []
    notes: list[str] = []
    if not isinstance(column, list):
        return expected, ["'[output.csv].column' absent or not a list"]
    for i, entry in enumerate(column):
        if not isinstance(entry, Mapping) or "header" not in entry:
            notes.append(f"column[{i}] missing 'header'")
            continue
        header = str(entry["header"])
        if "map" in entry:
            expected.append(f"{header}_*")  # <header>_<mapid> prefix pattern
        else:
            expected.append(header)
    return expected, notes


def _matches_expected(col: str, expected: Sequence[str]) -> bool:
    """True if ``col`` matches an expected name or an ``<header>_*`` pattern."""
    for exp in expected:
        if exp.endswith("_*"):
            if col.startswith(exp[:-1]):  # e.g. 'Q_' for 'Q_*'
                return True
        elif col == exp:
            return True
    return False


def validate_hm_gauge_column_identity(
    toml_cfg: Any,
    output_rlz_df: Any,
    qstats_df: Any,
) -> list[str]:
    """Relational: the HM-4 -> HM-5 -> HM-7 gauge-column identity (design §5.5).

    The gauge-column set is a **single degree of freedom** flowing TOML
    ``[output.csv].column`` -> ``output_rlz`` -> ``q_indicators``. A per-artifact
    validator cannot see a break *between* artifacts: rule 3.11 derives the gauge
    set from the FIRST csv via a hard-coded ``Q_`` prefix filter
    (``export_wflow_results.py:61``) and indexes every other csv with it, so a
    renamed gauge header silently empties ``Q_vars`` (a gauge-less q_indicators) and a
    later mismatch KeyErrors deep in the reduction.

    Checks (design §5.5):
      1. every non-``time`` ``output_rlz_df`` column traces to a declared
         ``[output.csv].column`` entry (map-typed -> ``<header>_<id>`` pattern;
         non-map -> exact ``header``), and every declared entry is represented;
      2. the map-typed gauge columns carry the ``Q_`` prefix rule 3.11 hard-codes;
      3. ``qstats_df``'s gauge columns (header minus ``statistic`` and the
         ``_PERTURBATION_AXIS`` columns) are list-equal to the ``output_rlz_df``
         gauge set.

    C3 boundary: the numeric ``<id>`` in ``Q_130000086`` is wflow's outlets-map
    cell value — the validator checks the ``<header>_<id>`` PATTERN and the
    cross-file identity, NOT the id's derivation from ``staticmaps.outlets``.

    ``output_rlz_df`` is expected with ``time`` as a **column** (default
    ``pd.read_csv`` shape).
    """
    label = "gauge-identity"
    diffs: list[str] = []
    expected, notes = _declared_gauge_columns(toml_cfg)
    diffs += [f"{label}: {n}" for n in notes]

    out_cols = [c for c in _columns(output_rlz_df) if c != "time"]

    # Check 1a: every produced column traces to a declared entry.
    for col in out_cols:
        if not _matches_expected(col, expected):
            diffs.append(
                f"{label}: output column {col!r} traces to no declared "
                f"[output.csv].column entry (expected {expected})"
            )
    # Check 1b: every declared entry is represented by >=1 produced column.
    for exp in expected:
        if exp.endswith("_*"):
            if not any(c.startswith(exp[:-1]) for c in out_cols):
                diffs.append(
                    f"{label}: declared entry pattern {exp!r} has no matching "
                    f"output column (have {out_cols})"
                )
        elif exp not in out_cols:
            diffs.append(
                f"{label}: declared column {exp!r} absent from output "
                f"(have {out_cols})"
            )

    # Check 2: map-typed gauge columns carry the Q_ prefix rule 3.11 hard-codes.
    map_typed = [e for e in expected if e.endswith("_*")]
    if any(e.startswith("Q_") for e in map_typed):
        gauge_cols = [c for c in out_cols if c.startswith("Q_")]
        if not gauge_cols:
            diffs.append(
                f"{label}: no output column carries the hard-coded 'Q_' prefix "
                f"rule 3.11 filters on (have {out_cols})"
            )

    # Check 3: the q table's LOCATION SET equals the output_rlz gauge set.
    #
    # Pre-R11 this compared column SETS, because the wide q_indicators header
    # carried one column per gauge and so had to be subtracted down to them. The
    # long shape puts locations in rows, so the same invariant is now expressed
    # against a column whose meaning is fixed -- simpler, and no longer coupled
    # to which non-gauge columns happen to lead the header.
    #
    # Compared as SETS rather than lists: rows have no meaningful order, so
    # list-equality would fail on a reordering that changes nothing. The
    # one-to-one property that mattered is preserved -- a gauge present in one
    # and absent from the other is still reported, by name.
    out_gauge = [c for c in out_cols if c.startswith("Q_")]
    expected = {c[2:] for c in out_gauge}
    if "location" in _columns(qstats_df):
        present = {str(v) for v in qstats_df["location"]} - {"basin"}
        if present != expected:
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            diffs.append(
                f"{label}: q_indicators location set != output_rlz gauge set "
                f"(missing {missing}, unexpected {extra})"
            )
    else:
        diffs.append(
            f"{label}: q_indicators has no 'location' column "
            f"(have {_columns(qstats_df)}); the pre-R11 wide shape is no longer "
            f"a valid HM-7 surface"
        )
    return diffs


def validate_wg5_catalog_grid(
    catalog_cfg: Any,
    rlz_num: int,
    st_num: int,
) -> list[str]:
    """Relational: the WG-5 catalog entry-key grid vs the INTENDED grid (design §5.5).

    Expected entry keys exactly ``{rlz_<n>_st_<m> : n in 1..rlz_num,
    m in 0..st_num}`` — **st_0 included** (rule 3.08 consumes both the st_0
    list and the perturbed ``expand`` grid, ``Snakefile_climate_experiment:318-319``).
    Both missing AND unexpected keys are reported. A dropped or extra catalog
    entry is invisible to per-artifact ``validate_wg5`` (each remaining entry is
    well-formed) but breaks the realization x cst fan-out rule 3.09 depends on.

    ``rlz_num`` / ``st_num`` are the run's *recorded* intent — the caller derives
    them from the experiment's config snapshot via ``stress_test_grid``
    (``shared/snake_utils.py``), so the check is self-consistent with the tree
    even if the tracked test config later drifts.
    """
    label = "wg5-catalog-grid"
    if not isinstance(catalog_cfg, Mapping):
        return [f"{label}: catalog is not a mapping ({type(catalog_cfg).__name__})"]
    # Keys carry the ZERO-PADDED member index (C27), and the widths derive from
    # the same counts this function already takes -- so the expectation moves
    # with the filenames without a signature change.
    rlz_w, st_w = index_width(rlz_num), index_width(st_num)
    expected = {
        f"rlz_{n:0{rlz_w}d}_st_{m:0{st_w}d}"
        for n in range(1, rlz_num + 1)
        for m in range(0, st_num + 1)
    }
    present = {
        k for k in catalog_cfg if isinstance(k, str) and k.startswith("rlz_")
    }
    diffs: list[str] = []
    for key in sorted(expected - present):
        diffs.append(f"{label}: expected catalog entry {key!r} missing")
    for key in sorted(present - expected):
        diffs.append(f"{label}: unexpected catalog entry {key!r} present")
    return diffs
