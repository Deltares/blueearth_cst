"""Interchange-contract validators for the two CST substitution seams.

This module pins — as machine-checkable, pure functions — the contract
surfaces at the two points where a CST component could be swapped for an
alternative implementation:

- the **weather-generator seam** (``weathergenr`` today): validators
  ``validate_wg1``..``validate_wg6`` + the relational
  ``validate_wg5_catalog_runs``;
- the **hydrological-model seam** (Wflow-SBM built by hydromt today):
  validators ``validate_hm1``..``validate_hm7`` (no ``validate_hm6a`` — its
  configuration pointer is checked by HM-4, not file existence) + the relational
  ``validate_hm_gauge_column_identity``.

Current guidance: ``dev/reference/toolbox-weather-generator-interface.md`` and
``dev/reference/toolbox-hydrological-model-interface.md``. The design section
IDs below identify historical rationale in
``dev/records/milestones/p32b/interchange-contracts-design.md``, not additional requirements.

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
and by WF0's comparison report for WG-1 readiness. WF0 reports divergences
without blocking available-variable comparisons.
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
        return [f"{label}: global attr {key!r}={attrs[key]!r} != expected {value!r}"]
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
        return [f"{label}: global attr {key!r}={attrs[key]!r} != expected {value!r}"]
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

    The full forcing shape remains ERA5-grounded. Comparison-only CHIRPS
    stores intentionally report their absent companion variables; selected
    CHIRPS stores include ERA5 companions. The orography sidecar is not checked.
    """
    label = "WG-1"
    diffs: list[str] = []
    diffs += _check_dims(ds, ("time", "latitude", "longitude"), label)
    diffs += _check_coords(ds, ("time", "latitude", "longitude"), label)
    # Coord dtypes (float32 lat/lon is the pinned surface).
    for coord, dtype in (("latitude", "float32"), ("longitude", "float32")):
        if coord in getattr(ds, "coords", {}) and str(ds[coord].dtype) != dtype:
            diffs.append(f"{label}: coord {coord!r} dtype {ds[coord].dtype} != {dtype}")
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


#: WG-2 stress-test lookup header, exact and ordered.
_WG2_HEADER = (
    "st_id",
    "month",
    "temp_change",
    "precip_change",
    "precip_variance_change",
)


def validate_wg2(df: Any, st_num: int | None = None) -> list[str]:
    """WG-2 — the stress-test parameter lookup (``stress_test_lookup.csv``).

    ONE table for the whole experiment, replacing the per-member
    ``_work/st_<m>.csv`` grid it used to pin and absorbing the derived design
    table. The mechanism changes with it, not only the constants: a 12-row
    assertion becomes ``12 x ST_NUM`` plus a ``(st_id, month)`` completeness
    check, and ``st_0`` must be ABSENT.

    ``st_num`` reaches this validator as an argument, the way the relational
    validators already take their second input. Omitted, the member count is
    inferred from the table itself and only its internal consistency is checked
    — which is weaker, and deliberately so: a validator that guessed the count
    could not tell a complete table from a truncated one.

    Three things are checked that the old per-member shape could not express:

    - **``st_0`` is absent.** Its absence is load-bearing rather than tidy: it is
      what makes "not on the surface" structural, so a consumer joining results
      to the lookup cannot place the baseline on the surface by accident.
    - **The ``st_id`` width is UNIFORM.** The whole join-key inference rests on
      one width per table, so this is where that becomes a checked property
      rather than a documented one.
    - **``st_id`` is TEXT.** Read with inferred dtypes, ``01`` comes back as
      ``1``; a validator subject to the very hazard the contract names could not
      detect it.

    Column *semantics* (additive degC vs percent) are documented, not
    machine-checked — the values change per stress-test point.
    """
    label = "WG-2"
    diffs: list[str] = []
    cols = _columns(df)
    if tuple(cols) != _WG2_HEADER:
        diffs.append(f"{label}: header {cols} != expected {list(_WG2_HEADER)}")
        return diffs

    ids = [str(v) for v in df["st_id"].tolist()]
    widths = {len(i) for i in ids}
    if len(widths) > 1:
        diffs.append(
            f"{label}: st_id mixes widths {sorted(widths)}; one width is pinned "
            f"for the whole table, and the join key is inferred from it"
        )
    baseline_tokens = {i for i in ids if i.strip().lstrip("0") == ""}
    if baseline_tokens:
        diffs.append(
            f"{label}: st_0 must have NO row, found {sorted(baseline_tokens)}. "
            f"The table is the parameter grid and the reserved unperturbed "
            f"baseline has no parameters"
        )

    members = sorted({i for i in ids if i not in baseline_tokens}, key=str)
    expected_members = members
    if st_num is not None:
        width = max(widths) if widths else 1
        expected_members = [f"{m:0{width}d}" for m in range(1, st_num + 1)]
        if members != expected_members:
            diffs.append(
                f"{label}: members {members} != 1..{st_num} ({expected_members})"
            )

    n = int(getattr(df, "shape", (0,))[0])
    if n != 12 * len(expected_members):
        diffs.append(
            f"{label}: expected 12 x {len(expected_members)} = "
            f"{12 * len(expected_members)} rows, got {n}"
        )

    for member in members:
        months = sorted(
            int(m) for m in df[df["st_id"].astype(str) == member]["month"].tolist()
        )
        if months != list(range(1, 13)):
            diffs.append(f"{label}: member {member!r} month domain {months} != 1..12")
    return diffs


#: WG-3 config surface — the key set the R side reads (design §5.2, read-only
#: from weathergen/{global.R,generate_weather.R,impose_climate_change.R}).
#:
#: Sections are weathergenr FUNCTION names and keys are their ARGUMENT names.
#: Renamed 2026-08-12 from the single ``generateWeatherSeries`` section: that
#: function is not exported by weathergenr 1.2.0 and its dot.case keys matched
#: no 1.2.0 argument, so the R hand-translated every one. The "upstream-spelled
#: dot.case preserved verbatim" rationale this list used to carry (dev-naming-conventions.md §2)
#: is what the rename RESTORES — upstream now spells them snake_case.
#:
#: **weathergenr 2.0.0 (2026-08-17).** One key changed here: ``relax_priority``
#: became ``relax_order`` and is now pinned, because the wrapper forwards it (see
#: the entry below). Every other section and key survives the major version
#: unchanged — the argument sets of ``run_weather_generator``,
#: ``generate_weather``, ``apply_climate_perturbations`` and ``write_netcdf``
#: were checked one by one against the 2.0.0 sources.
#:
#: 2.0.0 also weakened the FAILURE MODE this list guards, without weakening the
#: reason for it. ``run_weather_generator`` forwarded every ``config`` entry
#: unconditionally, so an absent one arrived as an explicit ``NULL`` that
#: replaced the receiving function's default; it now OMITS absent entries, so a
#: dropped key falls back to that default instead. The list stays exhaustive
#: because the point is that every setting this toolbox runs on is a stated
#: choice, readable in one file — not that upstream would otherwise crash.
#:
#: Every key is pinned for the same reason the transient flags are: the R reads
#: it, and an omission silently substitutes whatever upstream defaults to. That
#: is the C34 defect, and a contract is how it stays fixed.
_WG3_GENERATE_WEATHER_KEYS = (
    "vars",
    "warm_var",
    "warm_signif",
    "warm_pool_size",
    "warm_filter_bounds",
    # Restored 2026-08-17 for weathergenr 2.0.0, which renamed the argument
    # from `relax_priority` AND made run_weather_generator forward it. On 1.2.0
    # the wrapper dropped it, so pinning it would have required a key that
    # reaches nothing -- the C34 defect this contract exists to prevent
    # (t2608121742, closed by the upgrade).
    "relax_order",
    "annual_knn_n",
    "wet_q",
    "extreme_q",
    "dry_spell_factor",
    "wet_spell_factor",
    "seed",
    "parallel",
    "n_cores",
    "verbose",
    "save_plots",
    # Injected per run by rule 3.09 (prepare_weathergen_config.build_weathergen_config).
    "year_start_month",
    "n_years",
    "start_year",
    "n_realizations",
    "out_dir",
)

#: WG-3 ``apply_climate_perturbations`` surface (rule 3.11). Hardcoded in
#: impose_climate_change.R until the 1.2.0 rename surfaced them.
_WG3_PERTURBATION_KEYS = (
    "compute_pet",
    "pet_method",
    "qm_fit_method",
    "scale_var_with_mean",
    "enforce_target_mean",
    "precip_intensity_threshold",
    "precip_occurrence_transient",
    "exaggerate_extremes",
    "extreme_prob_threshold",
    "extreme_k",
    "precip_cap_mm_day",
    "precip_floor_mm_day",
    "precip_cap_quantile",
    "verbose",
    # LOAD-BEARING: false is what makes the return shape write_netcdf-compatible.
    "diagnostic",
)

#: WG-3 ``write_netcdf`` surface — read by BOTH rule 3.10 and rule 3.11.
_WG3_WRITE_NETCDF_KEYS = (
    "calendar",
    "compression",
    "spatial_ref",
    "signif_digits",
    "verbose",
    # Injected per run; the generation step's realization-file prefix.
    "file_prefix",
)

#: WG-3 ``run_weather_generator`` surface — the wrapper rule 3.10 calls. It runs
#: generate_weather and then the evaluation pass; ``_WG3_GENERATE_WEATHER_KEYS``
#: above is handed to it verbatim as its ``config`` argument.
_WG3_RUN_KEYS = (
    "eval_max_grids",
    "log_messages",
)

#: The four pinned sections, by weathergenr function name.
_WG3_SECTIONS = {
    "run_weather_generator": _WG3_RUN_KEYS,
    "generate_weather": _WG3_GENERATE_WEATHER_KEYS,
    "apply_climate_perturbations": _WG3_PERTURBATION_KEYS,
    "write_netcdf": _WG3_WRITE_NETCDF_KEYS,
}


def validate_wg3(cfg: Any) -> list[str]:
    """WG-3 — weathergenr config surface (``weathergen_config.yml``).

    Pinned surface (design §5.2, OQ-6): the *key set* the R side reads — one
    section per weathergenr function (``generate_weather``,
    ``apply_climate_perturbations``, ``write_netcdf``), plus the two
    ``transient_change`` flags — NOT weathergenr's config *semantics* or value
    ranges. A replacement generator may define its own config surface entirely;
    this pins the *current* generator's contract.

    **Renamed 2026-08-12 for weathergenr 1.2.0.** The single
    ``generateWeatherSeries`` section named a function 1.2.0 does not export,
    and ``general.variables`` is now ``generate_weather.vars`` — the argument's
    own name.

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
    for section, keys in _WG3_SECTIONS.items():
        block = cfg.get(section)
        if not isinstance(block, Mapping):
            diffs.append(f"{label}: '{section}' section absent")
            continue
        for key in keys:
            if key not in block:
                diffs.append(f"{label}: '{section}.{key}' absent")
    # `vars` carries the variable list the whole chain is generated over, so its
    # TYPE is pinned as well as its presence: a scalar here would reach
    # generate_weather as a length-1 vector and silently generate one variable.
    gw = cfg.get("generate_weather")
    if isinstance(gw, Mapping) and "vars" in gw and not isinstance(gw["vars"], list):
        diffs.append(
            f"{label}: 'generate_weather.vars' must be a list, got "
            f"{type(gw['vars']).__name__}"
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
                f"{label}: entry {key!r} metadata.crs {metadata.get('crs')!r} != 4326"
            )
        if metadata.get("category") != "meteo":
            diffs.append(
                f"{label}: entry {key!r} metadata.category "
                f"{metadata.get('category')!r} != 'meteo'"
            )
    return diffs


def validate_wg5(cfg: Any) -> list[str]:
    """WG-5 — a member's hydromt climate data catalog (``<run_id>.yml``).

    One file per member since 2026-08-18, written by rule 3.14 beside that
    member's TOML as a ``temp()`` output. It was one
    ``config/catalogs/data_catalog_run_stress_test.yml`` naming every member,
    built by rule 3.13 over a fan-in across the whole sweep; the entries
    differed only in ``uri``, and this validator never cared how many were in
    the file. It validates a one-entry mapping exactly as it validated an
    N-entry one.

    Pinned-as-reliance (design §5.2): OUR emitted subset of the hydromt
    data-catalog schema — for every prefixed ``run_id`` entry the driver /
    metadata fields ``{uri, driver.name=raster_xarray,
    driver.options.preprocess=harmonise_dims, driver.options.lock=false,
    metadata.crs=4326, metadata.category=meteo, data_type=RasterDataset}``.

    This pins per-entry *bookkeeping* only, NOT the NC content the entries point
    at (that is WG-4 / WG-6's contract) and NOT the entry-key grid completeness
    (that is the relational ``validate_wg5_catalog_runs``). The ``uri`` VALUE is
    deliberately unpinned (machine-scoped absolute path); only its presence is
    checked.
    """
    label = "WG-5"
    if not isinstance(cfg, Mapping):
        return [f"{label}: catalog is not a mapping ({type(cfg).__name__})"]
    diffs: list[str] = []
    entries = {
        k: v for k, v in cfg.items() if isinstance(k, str) and k.startswith("run_")
    }
    if not entries:
        diffs.append(f"{label}: no prefixed run IDs in catalog")
    for key in sorted(entries):
        diffs += _validate_catalog_entry(key, entries[key], label)
    return diffs


# ---------------------------------------------------------------------------
# Hydrological-model seam — HM-1, HM-2, HM-3, HM-4, HM-5, HM-7 (persisted).
# HM-6a: no file validator (HM-4 checks the configuration pointer only).
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
            diffs.append(f"{label}: coord {coord!r} dtype {ds[coord].dtype} != {dtype}")
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
            diffs.append(f"{label}: coord {coord!r} dtype {ds[coord].dtype} != {dtype}")
    diffs += _check_data_vars(ds, _HM2_VARS, label)
    for var in _HM2_VARS:
        diffs += _check_var_dtype(ds, var, "float32", label)
        if var in getattr(ds, "data_vars", {}):
            gm = ds[var].attrs.get("grid_mapping")
            if gm != "spatial_ref":
                diffs.append(f"{label}: {var!r} grid_mapping {gm!r} != 'spatial_ref'")
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
            f"{label}: region.geojson geom types {sorted(region_types)} are not Polygon"
        )
    # outlets: Point, EPSG:4326
    if str(getattr(outlets_gdf, "crs", None)) not in ("EPSG:4326", "epsg:4326"):
        diffs.append(f"{label}: outlets.geojson CRS {outlets_gdf.crs} != EPSG:4326")
    outlet_types = set(outlets_gdf.geom_type)
    if not outlet_types <= {"Point", "MultiPoint"}:
        diffs.append(
            f"{label}: outlets.geojson geom types {sorted(outlet_types)} are not Point"
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


def validate_hm4(cfg: Any, *, require_output_state: bool = True) -> list[str]:
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
    # WF1 retains final states. WF3 opts out because no downstream rule reads
    # them; initialization and state-variable mappings remain unchanged.
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
        required_state = (
            ("path_input", "path_output") if require_output_state else ("path_input",)
        )
        for key in required_state:
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
                diffs.append(f"{label}: '[output.csv].column[{i}]' missing 'header'")
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


#: The member-index padding width (C27). Imported rather than reimplemented so
#: the catalog-key expectation below and the Snakefile's own filenames cannot
#: disagree about how wide an index is. `snake_utils` does not import this
#: module, so the direction is one-way and no cycle exists.

#: The five HM-7 columns, in order. Imported from the writer's own module rather
#: than restated, so the producer and its validator cannot disagree about the
#: header -- the failure mode this pairing exists to prevent.
HM7_COLUMNS = ("indicator", "location", "unit_id", "value")


def validate_hm7(
    tables: dict,
    *,
    unit_index,
    scenario_table,
    declarations,
    locations,
    lookup,
    unit_id_capacity: int,
    response_request: Mapping,
) -> list[str]:
    """Refuse the retired indicator-set/1 surface with an explicit migration path."""
    return [
        "HM-7: legacy indicator-set/1 artifacts are unsupported; regenerate the "
        "collection, simulate under a new experiment name, and use indicator-set/3"
    ]


HM7_V2_COLUMNS = ("indicator", "location", "run_id", "value")
HM7_V2_LOOKUP_COLUMNS = ("group_id", "run_id")


def validate_hm7_v2(
    manifest: Mapping,
    tables: dict,
    *,
    lookup,
    series,
    collection_run_ids,
) -> list[str]:
    """HM-7 for indicator-set/3 (read-only) to /6; table keys are ``<token>_<run|group>``.

    indicator-set/6 tables add ``status`` and ``reason``; an unavailable bundle row
    carries no value.

    The historical function name remains for callers. Legacy /2 manifests are
    refused. Declarations, retained locations and collection runs independently
    determine exact coverage, including header-only run tables for bundle-only
    selections and explicit singleton membership.
    """
    import math

    label = "HM-7"
    if manifest.get("schema_version") not in (
        "indicator-set/3",
        "indicator-set/4",
        "indicator-set/5",
        "indicator-set/6",
    ):
        return [
            f"{label}: expected indicator-set/3 to /6; legacy mixed-ID sets must be recomputed"
        ]
    if tuple(lookup.columns) != HM7_V2_LOOKUP_COLUMNS:
        return [f"{label}: lookup header is not group_id,run_id"]
    diffs = []
    rows = [tuple(str(v) for v in row) for row in lookup.itertuples(index=False)]
    declared = [
        (str(g["run_group_id"]), str(g["run_id"]))
        for g in manifest.get("run_groups", [])
        if g["grain"] == "bundle"
    ]
    if len(rows) != len(set(rows)):
        diffs.append(f"{label}: duplicate membership rows")
    if set(rows) != set(declared):
        diffs.append(f"{label}: membership differs from manifest")
    runs = {str(item["run_id"]) for item in series}
    if not runs <= set(map(str, collection_run_ids)):
        diffs.append(f"{label}: evaluated runs are outside the collection")
    if any(not g.startswith("group_") or r not in runs for g, r in rows):
        diffs.append(f"{label}: membership has invalid group or run references")
    if {r for _, r in rows} != runs:
        diffs.append(f"{label}: group membership does not cover all collection runs")
    groups = {g for g, _ in rows}
    expected_locations = {}
    for item in series:
        expected_locations.setdefault(str(item["variable"]), set()).add(
            str(item["location_id"])
        )
    entries = manifest.get("tables", [])
    table_names = {f"{entry['token']}_{entry['scope']}" for entry in entries}
    declarations = manifest.get("declarations", [])
    tokens = {
        d.get("output_variable") or d["required_responses"]["variable"]
        for d in declarations
    }
    if (
        table_names != {f"{t}_{scope}" for t in tokens for scope in ("run", "group")}
        or set(tables) != table_names
        or len(entries) != len(table_names)
    ):
        diffs.append(f"{label}: tables differ from declared token scopes")
    statused = (
        ("status", "reason") if manifest["schema_version"] == "indicator-set/6" else ()
    )
    for entry in entries:
        token, scope = entry["token"], entry["scope"]
        table = tables.get(f"{token}_{scope}")
        if table is None:
            continue
        if scope not in ("run", "group") or tuple(table.columns) != (
            "indicator",
            "location",
            f"{scope}_id",
            "value",
            *statused,
        ):
            diffs.append(f"{label}: {token}/{scope} table schema differs")
            continue
        keys = [
            (str(m), str(loc), str(unit))
            for m, loc, unit in zip(
                table["indicator"], table["location"], table[f"{scope}_id"]
            )
        ]
        expected = {
            (d["name"], loc, unit)
            for d in declarations
            if (d.get("output_variable") or d["required_responses"]["variable"])
            == token
            and (scope == "group" or d["grain"] == "run")
            for loc in expected_locations.get(token, set())
            for unit in (runs if scope == "run" else groups)
        }
        if len(keys) != len(set(keys)) or set(keys) != expected:
            diffs.append(
                f"{label}: {token}/{scope} keys differ from declarations, locations and membership"
            )
        bundle_indicators = {d["name"] for d in declarations if d["grain"] == "bundle"}
        statuses = table["status"] if statused else [""] * len(table["indicator"])
        for indicator, value, status in zip(
            table["indicator"], table["value"], statuses
        ):
            try:
                number = float(value) if value != "" else float("nan")
            except (TypeError, ValueError):
                diffs.append(f"{label}: {token}/{scope} has a non-numeric value")
                break
            if math.isinf(number) or (
                indicator in bundle_indicators
                and not math.isfinite(number)
                and status != "unavailable"
            ):
                diffs.append(f"{label}: {token}/{scope} violates indicator validity")
                break
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
    """WG-4 — durable collection forcing (``series/<run_id>.nc``).

    Pinned surface (design §5.2): a ``(time, lat, lon)`` raster the hydromt
    catalog (WG-5) reads — at least ``precip`` and ``temp`` on an EPSG:4326 grid
    carrying a ``spatial_ref`` CRS descriptor. The exact variable superset and
    internal attrs are deliberately unpinned.

    One inventoried durable file per scenario row, including unperturbed rows.
    Consumers must not delete these artifacts after model preparation.

    Grid axes are accepted as either ``(latitude, longitude)`` or the shorter
    ``(lat, lon)`` a swap may emit — the contract is the raster (time, y, x)
    shape + the minimal variable set, not the axis spelling.

    ``crs`` / ``category`` are asserted-IF-PRESENT, not required (corrected
    2026-07-25 on the first ``--notemp`` capture, which is what this contract was
    always waiting on). The real generator artifact carries **empty** global
    attrs: its CRS travels the CF/rioxarray way, in the ``spatial_ref``
    coordinate's ``crs_wkt`` (``ID["EPSG",4326]``), while ``crs: 4326`` and
    ``category: meteo`` are supplied by the generated **data catalog** —
    the member's own ``<run_id>.yml``, which is exactly where hydromt reads
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
    """WG-6 — prepared forcing content (``inmaps_<run_id>.nc``).

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
    """HM-6b — simulator warm state content (``outstates_<run_id>.nc``).

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
    labels: Mapping | None = None,
) -> list[str]:
    """Relational: the HM-4 -> HM-5 -> HM-7 gauge-column identity (design §5.5).

    The gauge-column set is a **single degree of freedom** flowing TOML
    ``[output.csv].column`` -> ``output_rlz`` -> ``q_indicators``. A per-artifact
    validator cannot see a break *between* artifacts: the predecessor reduction
    (removed 2026-09-25) derived the gauge set from the FIRST csv via a
    hard-coded ``Q_`` prefix filter and indexed every other csv with it, so a
    renamed gauge header silently emptied ``Q_vars`` (a gauge-less q_indicators)
    and a later mismatch KeyErrored deep in the reduction.

    Checks (design §5.5):
      1. every non-``time`` ``output_rlz_df`` column traces to a declared
         ``[output.csv].column`` entry (map-typed -> ``<header>_<id>`` pattern;
         non-map -> exact ``header``), and every declared entry is represented;
      2. the map-typed gauge columns carry the ``Q_`` prefix rule 3.10 hard-codes;
      3. ``qstats_df``'s gauge set is list-equal to the ``output_rlz_df`` gauge
         set. Post-CR-2 this compares the ``location`` column's VALUE set rather
         than subtracting known columns from a wide header, which is why the
         axis columns' removal left this check's logic untouched.

    C3 boundary: the numeric ``<id>`` in ``Q_130000086`` is wflow's outlets-map
    cell value — the validator checks the ``<header>_<id>`` PATTERN and the
    cross-file identity, NOT the id's derivation from ``staticmaps.outlets``.

    ``output_rlz_df`` is expected with ``time`` as a **column** (default
    ``pd.read_csv`` shape).

    ``labels`` maps each kept native header to the ``wflow_id`` WF4 publishes
    it under (``shared.native_locations.native_location_labels``): an outlet
    column carries its subbasin's primary id, and a column absent from
    ``labels`` repeats another's model cell, so check 3 expects the labelled
    set. Without ``labels`` the ids are the headers' own suffixes.
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
                f"{label}: declared column {exp!r} absent from output (have {out_cols})"
            )

    # Check 2: map-typed gauge columns carry the Q_ prefix rule 3.10 hard-codes.
    map_typed = [e for e in expected if e.endswith("_*")]
    if any(e.startswith("Q_") for e in map_typed):
        gauge_cols = [c for c in out_cols if c.startswith("Q_")]
        if not gauge_cols:
            diffs.append(
                f"{label}: no output column carries the hard-coded 'Q_' prefix "
                f"rule 3.10 filters on (have {out_cols})"
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
    expected = (
        {c[2:] for c in out_gauge}
        if labels is None
        else {labels[c] for c in out_gauge if c in labels}
    )
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


def validate_wg5_catalog_runs(
    catalogs: Mapping, evaluated_run_ids: Sequence[str]
) -> list[str]:
    """Check each run's catalog and the exact evaluated-run key inventory.

    ``catalogs`` maps intended opaque run_id to its parsed per-run catalog.
    Ancillary entries remain legal; only run_ entries participate in this seam.
    """
    label = "wg5-catalog-runs"
    if not isinstance(catalogs, Mapping):
        return [f"{label}: catalogs must map intended run ids to parsed catalogs"]
    expected = list(evaluated_run_ids)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(
            not isinstance(run, str)
            or not run.isascii()
            or not run.startswith("run_")
            or not run[4:].isdigit()
            or int(run[4:]) < 1
            for run in expected
        )
    ):
        return [f"{label}: evaluated run ids must be unique positive decimal text"]
    diffs = []
    if set(catalogs) != set(expected):
        diffs.append(f"{label}: catalog set differs from evaluated scenario rows")
    present = []
    for intended, catalog in catalogs.items():
        diffs.extend(validate_wg5(catalog))
        if not isinstance(catalog, Mapping):
            continue
        keys = [
            key for key in catalog if isinstance(key, str) and key.startswith("run_")
        ]
        if keys != [intended]:
            diffs.append(
                f"{label}: catalog for {intended!r} must contain only {intended}"
            )
        present.extend(keys)
    if len(present) != len(set(present)) or set(present) != {run for run in expected}:
        diffs.append(f"{label}: aggregate entry keys differ from evaluated run set")
    return diffs
