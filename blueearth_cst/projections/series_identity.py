"""Cache identity for the persistent GCM series store (WF2 v2.0, design §5.3).

The series files under ``data/climate/projections/{clim_project}/`` stop being
``temp()`` at migration step 2b and become a persistent product. Persistence
without identity is a silent-wrong-numbers path: a file on disk looks valid
whatever produced it. This module is the identity — it answers "was this series
derived from exactly the inputs the current configuration implies?" and it is
used from three places:

* the **Snakefile**, at DAG-build time, to put every parse-time-knowable digest
  component into the rule's ``params`` so Snakemake's params rerun-trigger
  schedules re-derivation when any of them changes;
* the **reduce job** (``get_stats_climate_proj.py``), which completes the digest
  with the polygon fingerprint, revalidates against any existing output, and
  stamps the result on what it writes;
* the **change job** (``get_change_climate_proj.py``), which recomputes every
  expected digest and raises on mismatch — the backstop that holds even when
  Snakemake's scheduling did not run (design D9, route (b)).

Why the polygon is fingerprinted by **content** and not by the region
specification (design D9 / finding ext2-01): a delineation-catalog change can
rewrite ``spatial/geoms/region.geojson`` while ``shared.basin.region`` is
unchanged, so a
specification-based digest recomputes to the same value and a series computed for
the old polygon stays eligible for reuse.

Why the reducer version is a **module hash** and not a hand-bumped constant
(finding risk-03): Snakemake's code rerun-trigger tracks a rule's script body,
not the modules it imports, so a forgotten bump silently reuses every cached
series after a reduction-logic change.

Everything here is stdlib plus the geopandas stack already in the environment —
no new dependency (design OQ-7 / constraint C5).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable, Mapping, Sequence

#: Bumped when the series schema changes in a way a reader must notice — the
#: attribute set, the digest recipe, or the key grammar. A consumer that meets a
#: version it does not know must FAIL rather than guess (design §5.3).
#:
#: ``"2"`` (revision 6, the fetch/reduce split): the attribute set changed in both
#: layers — a series gains ``cst_raw_digest`` naming the slice it was reduced from,
#: and raw slices are a new artifact class carrying ``cst_raw_digest`` and no
#: ``cst_series_digest``. Bumping is what makes the existing v1 series re-derive
#: instead of being silently accepted without the new provenance; it is nearly free
#: now that a re-reduction reads local disk.
SCHEMA_VERSION = "4"
#: Bumped 3->4 at step 5b: the SERIES schema gained `cst_calendar`, propagated
#: from the raw slice so stage B can weight months without re-reading the store.
#: The bump is required, not tidiness: the stamping lives in the snakemake body,
#: OUTSIDE the functions `kernel_hash` enumerates, so `REDUCER_HASH` does not move
#: and every reduce job would hit its internal cache and skip -- Snakemake
#: schedules the job, the job revalidates and returns, and the new attribute is
#: never written. Observed exactly that before this bump: series rewritten at
#: 21:19 with schema 3 and no `cst_calendar`.
#: Side effect stated plainly: raw slices carry the version too, so they re-fetch
#: as well even though they are already correct. Accepted over splitting the raw
#: and series schemas, which would be a contract change to save one dev re-fetch.
#: Bumped 2->3 at step 5b's prerequisite: raw slices and series written before
#: this carry `cst_calendar = ""` while their `time.attrs` assert
#: `proleptic_gregorian`, which is FALSE for every noleap/360_day model
#: (dev/milestones/r08/2026-07-30_wf2-5b-calendar-blocker.md). The calendar cannot join
#: the digest components -- reading it needs the store, and DAG build is
#: deliberately network-free -- so the version bump is the invalidation lever, and
#: `cache_hit` already rejects on a schema mismatch. Costs one re-derivation.

#: Acquisition span per experiment class, lifted out of
#: ``get_stats_climate_proj.py``'s ``time_tuple_all`` branch into a declared
#: contract (design §5.3). Deliberately independent of ``future_horizons``: all
#: analysis-window selection happens downstream on local files, so changing a
#: horizon schedules zero reduce jobs (goal G5).
ACQUISITION_WINDOWS = {
    "historical": ("1950-01-01", "2014-12-31"),
    "_scenario": ("2015-01-01", "2100-12-31"),
}

#: Variables the generated catalog certifies — the crawl proved these present,
#: so they carry a physical pin. Everything else is best-effort: nameable, but
#: unverified and unpinnable (design §5.5, ruling A3).
CERTIFIED_VARIABLES = ("pr", "tas")

#: CMIP6 global attrs that describe ONE source file and therefore cannot describe
#: a merge of several. Our raw slices and series hold `precip` AND `temp` merged
#: from separate CMIP6 files, so whichever member wins ``xr.merge``'s attr
#: resolution stamps the result: ``variable_id`` reads `tas` on one fetch and
#: `pr` on the next while every value matches (R9 P2 F4). Dropped when we stamp
#: our own provenance, because the attribute is not merely nondeterministic — it
#: is WRONG for a two-variable file, and a reader has no way to know that.
#:
#: The correct merged provenance is already carried by ``cst_source_paths``,
#: which names every variable and the store version each came from, and by the
#: per-variable ``original_name`` / ``standard_name`` / ``units`` — none of which
#: are touched here, so a genuine tas-vs-pr mixup is still caught (and would show
#: in the values regardless).
#:
#: EXACTLY these three. `creation_date` is already handled as a volatile attr by
#: the comparators, and our own `cst_*` attrs legitimately differ between code
#: eras — masking or dropping those would hide real drift.
INHERITED_SINGLE_SOURCE_ATTRS = frozenset({"variable_id", "tracking_id", "status"})


def drop_inherited_single_source_attrs(ds):
    """Strip the single-source CMIP6 provenance a merged dataset cannot own.

    Mutates and returns ``ds``. Call after stamping ``cst_*`` provenance, so the
    file carries merge-aware identity and nothing that contradicts it.
    """
    for key in INHERITED_SINGLE_SOURCE_ATTRS:
        ds.attrs.pop(key, None)
    return ds


def acquisition_window(experiment: str) -> tuple[str, str]:
    """Return the fixed acquisition span for ``experiment``.

    ``historical`` gets the CMIP historical span; every ``sspNNN`` gets the
    ScenarioMIP span. These are the values the script previously hardcoded, so
    lifting them here is behaviour-preserving.
    """
    if experiment == "historical":
        return ACQUISITION_WINDOWS["historical"]
    return ACQUISITION_WINDOWS["_scenario"]


def series_key(catalog_entry: str, member: str) -> str:
    """Build the series key from a catalog entry name and a resolved member.

    The catalog entry carries a vendor path segment
    (``cmip6_NOAA-GFDL/GFDL-ESM4_ssp245_{member}``), so ``/`` is sanitized to
    ``_`` — otherwise the key becomes a directory, which is how today's
    intermediates end up nested under ``stats_time-INM/``.

    The ``{member}`` placeholder in the entry name is replaced by the resolved
    member rather than appended, so the key reads as one name.
    """
    resolved = catalog_entry.replace("{member}", member)
    return resolved.replace("/", "_")


def region_fingerprint(region_path: str | os.PathLike) -> str:
    """sha256 of the polygon's canonical geometry (design D9, cache-key item 4).

    Fingerprints **content**, not the specification that produced it. WKB is the
    canonical form: it is insensitive to GeoJSON key order, whitespace and
    coordinate-precision formatting, all of which can differ between two writes
    of the same geometry, while remaining sensitive to any actual change in the
    geometry itself.

    Multiple features are hashed in file order — the region is expected to be a
    single polygon, but ordering is fixed rather than assumed so the value is
    reproducible if that ever changes.
    """
    import geopandas as gpd

    geom = gpd.read_file(region_path)
    digest = hashlib.sha256()
    for wkb in geom.geometry.to_wkb():
        digest.update(wkb)
    return digest.hexdigest()


def module_hash(module_paths: Sequence[str | os.PathLike]) -> str:
    """sha256 over an ENUMERATED list of reducer module files (risk-03).

    Enumerated rather than "all of ``blueearth_cst``" on purpose: an unrelated
    edit elsewhere in the package must not invalidate every cached series. The
    enumeration lives in the Snakefile's params, where it is reviewable in a
    diff.

    Files are hashed in sorted basename order with their basename mixed in, so
    the value does not depend on the absolute path the workflow happens to run
    from, but does change if a file is renamed.

    **Prefer :func:`kernel_hash` for the cache key.** Hashing whole files means a
    comment, a docstring or an error message invalidates every cached series and
    forces a full network re-derivation — observed at step 4c, where an
    error-handling-only change cost 9 remote reads. This function is retained for
    callers that genuinely want file-level identity.
    """
    digest = hashlib.sha256()
    for path in sorted(module_paths, key=lambda p: os.path.basename(str(p))):
        digest.update(os.path.basename(str(path)).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def _consts_repr(consts) -> str:
    """Deterministic text for a code object's constants, across PROCESSES.

    `repr()` alone is not deterministic here, and the failure is subtle. A nested
    function's **code object** is a constant of its parent, and `repr(code_object)`
    embeds the object's memory address:

        <code object _annual at 0x000001B2C4E10930, file "...", line 203>

    So any hashed function containing a nested `def` or `lambda` produced a
    different digest in every process. Observed 2026-07-31: `STAGE_B_HASH` moved
    on every invocation because `get_change_annual_clim_proj` gained the `_annual`
    closure at step 5b, which made Snakemake re-run stage B forever with
    "params have changed since last execution". `REDUCER_HASH` was unaffected only
    because none of its functions happens to nest one.

    Code constants are therefore rendered from their own behaviour -- name,
    bytecode, and their constants in turn -- rather than from their identity. A
    changed closure still changes the digest, which is the point; an unchanged one
    no longer does.
    """
    from types import CodeType

    def render(value):
        if isinstance(value, CodeType):
            return (
                f"<code {value.co_name} {value.co_code!r} "
                f"{_consts_repr(value.co_consts)} {value.co_names!r}>"
            )
        if isinstance(value, (frozenset, set)):
            # Set literals compile to (frozen)set constants, whose iteration order
            # varies with string-hash randomization. Sort for stability.
            return "{" + ", ".join(sorted(repr(v) for v in value)) + "}"
        if isinstance(value, tuple):
            return "(" + ", ".join(render(v) for v in value) + ")"
        return repr(value)

    return render(tuple(consts))


def kernel_hash(functions, env_fingerprint: str | None = None) -> str:
    """sha256 over the BEHAVIOUR of the numerical reduction functions.

    This is the cache key risk-03 actually wants. ``module_hash`` tracks file
    bytes, so it cannot tell a changed formula from a reformatted one — at step 4c
    an error-handling-only edit invalidated all 9 series and cost a full network
    re-derivation. Bytecode plus constants tracks behaviour: comments, docstrings
    and formatting never appear in ``co_code`` or in the hashed constants, while
    any change to the computation does.

    What it catches: a changed formula, a changed constant of **any type**
    including strings (``co_consts``), a swapped attribute or global lookup
    (``co_names``), a changed default argument (``__defaults__`` /
    ``__kwdefaults__``), a different call sequence, and — when
    ``env_fingerprint`` is supplied — a changed dependency environment.

    **String constants are load-bearing and are hashed.** In xarray-style
    reduction code the difference between ``resample(time="MS")`` and ``("YS")``,
    between ``ds["pr"]`` and ``ds["tas"]``, between ``keep="first"`` and
    ``"last"``, or between two date bounds is *only* a string constant: ``co_code``
    is byte-identical because the constant is referenced by index. An earlier
    revision of this function excluded every string constant so that reworded
    error messages stayed free, which silently made all five of those edit classes
    invisible — a stale-cache path of exactly the kind risk-03 was filed against
    (measured; process review r2 §2). Only the function's own docstring is
    excluded now, by identity rather than by type. The price is that an
    error-message edit costs one invalidation again; the fetch/reduce split
    (design amendment pending) makes that re-reduction local and cheap.

    What it still deliberately misses: a change in a *callee* that is not itself
    listed — so the enumeration must name every function whose arithmetic matters,
    exactly as ``module_hash``'s file list had to.

    ``env_fingerprint`` folds the environment into the same digest: the reduction's
    output depends on xarray/pandas behaviour, which no source hash can see. Pass
    the lock-file digest (:func:`file_digest`). It is deliberately coarse — any
    dependency change re-derives — which is the conservative direction for a cache
    whose failure mode is silently wrong numbers.

    Each function contributes its qualified name too, so moving logic between
    functions invalidates rather than cancelling out.
    """
    digest = hashlib.sha256()
    for func in sorted(functions, key=lambda f: f.__qualname__):
        code = func.__code__
        digest.update(func.__qualname__.encode("utf-8"))
        digest.update(code.co_code)
        # Constants and names: a changed threshold, a swapped attribute lookup, a
        # changed dimension name or resample code are all behaviour changes that
        # co_code alone does not reflect.
        digest.update(
            _consts_repr(
                tuple(c for c in code.co_consts if not _is_own_docstring(c, func))
            ).encode("utf-8")
        )
        digest.update(repr(code.co_names).encode("utf-8"))
        # Defaults live on the function object, not in the code object: a changed
        # default is a changed computation with identical bytecode.
        digest.update(repr(func.__defaults__).encode("utf-8"))
        digest.update(repr(sorted((func.__kwdefaults__ or {}).items())).encode("utf-8"))
    if env_fingerprint is not None:
        digest.update(b"env:")
        digest.update(env_fingerprint.encode("utf-8"))
    return digest.hexdigest()


def _is_own_docstring(const, func) -> bool:
    """True only for the constant that IS this function's docstring.

    Filtered by identity against ``func.__doc__`` rather than by position
    (``co_consts[0]`` is fragile across Python versions) or by type (excluding
    every string is what made string-constant behaviour changes invisible). A
    docstring edit is documentation; every other string constant is code.
    """
    return func.__doc__ is not None and const is func.__doc__


def file_digest(path: str | os.PathLike) -> str:
    """sha256 of one file's bytes — the environment fingerprint for the cache key.

    Used with ``pixi.lock`` so a dependency upgrade re-derives the series rather
    than reusing numbers produced by a different xarray. Separate from
    :func:`module_hash` because this file is not source we hash for logic; it is
    an opaque environment identity.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog_entry(catalog_path: str | os.PathLike, catalog_entry: str) -> dict:
    """Parse one catalog entry with YAML merge keys RESOLVED.

    The generated catalog emits the shared driver/adapter/metadata block once as
    an anchor and pulls it into the other 288 entries with ``<<``. A parser that
    does not resolve merge keys would see those entries as having no ``driver:``
    block at all — silently, and the digest would then miss exactly the
    component that determines what every read means. ``yaml.safe_load`` does
    resolve ``<<`` (verified on the pinned PyYAML), and the test suite asserts
    the merged read on a non-anchor entry.

    Raises ``KeyError`` naming the entry when it is absent, so a resolution bug
    surfaces here rather than as an empty dataset downstream.
    """
    import yaml

    with open(catalog_path, encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if catalog_entry not in catalog:
        raise KeyError(
            f"catalog entry {catalog_entry!r} not found in {catalog_path}. "
            "The catalog is generated — regenerate it with "
            "dev/scripts/generate_cmip6_catalog.py rather than hand-editing."
        )
    return catalog[catalog_entry]


def entry_identity(entry: Mapping, member: str) -> dict:
    """The read-determining parts of a catalog entry (cache-key items 2 and 3).

    Included: the member-substituted URI template, the driver options, the
    ``data_adapter`` maps (unit add/mult, rename) and the ``metadata`` map.
    Metadata is in because it changes how a read is *interpreted* — crs and
    nodata — which finding ext2-04 raised as the second half of its complaint.

    Excluded: ``placeholders`` and ``meta``. Under the generated catalog a
    regeneration routinely adds members as the store grows, and ``crawled_on``
    changes every time by construction; folding either in would re-derive series
    whose data did not change. Which sources *exist* is resolution, not
    identity — the resolved member is already in the key and in the substituted
    URI.
    """
    uri = str(entry.get("uri", "")).replace("{member}", member)
    return {
        "uri": uri,
        "driver": entry.get("driver"),
        "data_adapter": entry.get("data_adapter"),
        "metadata": entry.get("metadata"),
    }


def load_pins(
    index_path: str | os.PathLike,
    catalog_entry: str,
    member: str,
) -> dict[str, list[str]]:
    """Observed ``{grid_label}/{version}`` per certified variable (design D12).

    The catalog URI ends ``/{variable}/*/*``, so the entry name is a *logical*
    identity — the glob still spans grid label and publication version. Finding
    ext2-04 showed that means a re-publication can be read by a fresh project
    while an existing cache keeps the old bytes under an unchanged digest. These
    pins are the physical identity, and they are what enters the digest.

    A missing index, entry or member yields ``{}`` rather than raising: the
    index is a generated sidecar and a project may legitimately predate it. The
    consequence is an honest one — no pin means no physical identity in the
    digest for that series, which is the same position every series was in
    before D12 — and it is recorded on the series as an empty
    ``cst_source_paths``.
    """
    if not index_path or not os.path.isfile(index_path):
        return {}
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    return index.get("sources", {}).get(catalog_entry, {}).get(member, {}) or {}


#: Trailing glob a generated CMIP6 URI ends with: ``{grid_label}/{version}``.
STORE_GLOB_SUFFIX = "/*/*"


def pinned_uri(uri: str, pins: Mapping[str, Sequence[str]]) -> str | None:
    """Narrow a catalog URI's trailing ``/*/*`` to the pin the index recorded.

    The generated catalog ends every URI ``/{variable}/*/*``, so resolving one
    source means listing the bucket to expand ``{grid_label}/{version}`` — while
    the physical location that listing discovers is **already recorded** in
    ``cmip6_store_index.json`` (D12). Substituting the pin replaces a listing with
    a direct address.

    **Worth ~10 s per source — a modest win, not a dominant one.** Measured on
    this source, three alternating samples per arm
    (``dev/milestones/r08/2026-07-30_wf2-fetch-reduce-benchmark.md`` §3.2):

    * open, pinned **49.9 s** vs globbed **60.0 s**, non-overlapping ranges —
      ~17 % of the open, ~1.5 min across the 9 series;
    * but ``fs.glob`` answers the same two patterns in **0.41 s**, so what the pin
      removes is **hydromt's resolver overhead on a wildcard URI**, not bucket
      latency. Do not describe it as avoiding a slow network listing.

    An earlier version of this docstring called the listing "the dominant cost of a
    read — ~28 s per source". That mis-attributed a whole fetch job to the listing
    inside it: the note's 28 s is "fetch 28 s + reduce 31 s **per series**".

    The second, non-timing reason to keep this: the job addresses one known store
    instead of resolving a pattern whose match set can change under it.

    Returns ``None`` — meaning "keep the glob" — whenever the pins cannot name one
    physical location. Measured against the 289-entry index (2 426 member
    combinations): 2 205 share one pin and take this path, 33 differ per variable,
    188 record more than one match per variable. The last group is D8's ambiguity
    and must stay globbed so the duplicate-time assertion still sees it.

    This does **not** change the series digest, and must not: the digest is built at
    DAG build from the *catalog* entry, whose URI is the logical template, while the
    pin it also records is the physical identity (D12's own framing). Rewriting the
    URI inside the job spends the pin the digest already carries; it adds no
    component and re-derives nothing.
    """
    if not pins or not uri.endswith(STORE_GLOB_SUFFIX):
        return None
    distinct = {tuple(paths) for paths in pins.values()}
    if len(distinct) != 1:
        return None
    (only,) = distinct
    if len(only) != 1:
        return None
    return uri[: -len(STORE_GLOB_SUFFIX)] + "/" + only[0]


#: Recorded when the store's calendar could not be determined. A sentinel, not an
#: empty string: "" is what the pre-3 schema wrote when it read `.calendar` off a
#: DatetimeIndex that never had one, and an absent value is indistinguishable from
#: a value nobody looked for. Anything weighting by month length must refuse this.
CALENDAR_UNKNOWN = "unknown"


def parse_store_calendar(zmetadata: Mapping) -> str:
    """Calendar name from a zarr store's consolidated metadata.

    Split from the fetch for testability: this is pure, and the network read that
    feeds it lives at the single call site allowed to touch the store.

    Returns :data:`CALENDAR_UNKNOWN` rather than guessing. The stores are not
    uniform across 289 entries, and a wrong calendar recorded confidently is the
    exact defect this function exists to end.
    """
    meta = (zmetadata or {}).get("metadata", zmetadata) or {}
    attrs = meta.get("time/.zattrs") or {}
    calendar = str(attrs.get("calendar", "")).strip()
    return calendar or CALENDAR_UNKNOWN


def read_store_calendar(store_uri: str) -> str:
    """The model's true calendar, read from the store itself.

    **Only ``fetch_gcm_raw`` may call this** — it makes a remote request, and the
    reduce stage's contract is zero network calls (benchmark note §2).

    Necessary because the calendar does not survive the read path: our generated
    catalog requests ``preprocess: harmonise_dims``, whose time branch
    (``hydromt/data_catalog/drivers/preprocessing.py:66``) converts a ``CFTimeIndex``
    to a ``DatetimeIndex``, after which ``noleap`` is indistinguishable from
    ``proleptic_gregorian`` — and is in fact written out AS ``proleptic_gregorian``.

    Cheap on purpose: one consolidated-metadata read (~0.3 s), not a store open.
    Values are unaffected by the conversion, so only the calendar NAME must be
    recovered; month length is a function of (calendar, year, month) and the
    converted axis still carries year and month.
    """
    import json

    import fsspec

    try:
        fs, path = fsspec.core.url_to_fs(store_uri, token="anon")
        with fs.open(f"{path.rstrip('/')}/.zmetadata", "rb") as fh:
            return parse_store_calendar(json.loads(fh.read()))
    except Exception:
        # Never fail a fetch over provenance: the slice's DATA is valid regardless.
        # The unknown sentinel propagates, and the step that needs a calendar
        # refuses there instead -- loudly, and naming the series.
        return CALENDAR_UNKNOWN


def digest_components(
    *,
    catalog_entry: str,
    entry: Mapping,
    members: Sequence[str],
    pins_by_member: Mapping[str, Mapping[str, Sequence[str]]],
    buffer_degrees: float,
    variable_spec: object,
    experiment: str,
    reducer_module_hash: str,
) -> dict:
    """Assemble every digest component EXCEPT the polygon fingerprint.

    Split out because these are all knowable at DAG-build time and belong in the
    rule's ``params``, where Snakemake's params trigger can see them. The
    polygon fingerprint deliberately is not here: on a fresh project the polygon
    does not exist at parse time, and a param flipping from absent to a real
    value on the second invocation would re-derive every series once for nothing
    (design §6.14). It is folded in inside the job by :func:`series_digest`.

    **``members`` is a sequence, not a single member.** At migration step 2b the
    ``members:`` config key is still a list looped *inside* one job, so a single
    output covers every member in that list and the identity must span all of
    them. Step 3 turns the member into a wildcard, at which point the sequence
    has one element and this shape still holds — so the digest recipe does not
    change under that refactor.

    Note the region **bounds** are deliberately absent: they are derived from the
    polygon, so they add nothing the content fingerprint does not already carry,
    and they are unavailable at parse time. Bounds are still *recorded* on the
    series as ``cst_region_bounds`` for provenance.
    """
    members = sorted(members)
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_entry": catalog_entry,
        "members": members,
        "entry_identity": {m: entry_identity(entry, m) for m in members},
        "pins": {
            m: {
                var: list(paths)
                for var, paths in sorted(dict(pins_by_member.get(m, {})).items())
            }
            for m in members
        },
        "buffer_degrees": float(buffer_degrees),
        "variable_spec": variable_spec,
        "acquisition_window": list(acquisition_window(experiment)),
        "reducer_module_hash": reducer_module_hash,
    }


def raw_components(components: Mapping) -> dict:
    """The digest components the RAW slice depends on: everything but the reducer.

    The stage-A split (design revision 6) separates *fetching* a slice of the
    remote store from *reducing* it. What the raw bytes are determined by — catalog
    entry, entry identity, pins, members, acquisition window, buffer, variable spec,
    polygon — is exactly the series' component set minus ``reducer_module_hash``:
    changing the reduction cannot change what was downloaded.

    That exclusion is the whole point of the split, and it has to hold in **two**
    places to pay off: here, and in the fetch rule's ``params`` (the Snakefile must
    not pass the reducer hash to the fetch job, or Snakemake's params trigger
    re-downloads on a formula edit no matter what this function returns).
    """
    return {k: v for k, v in components.items() if k != "reducer_module_hash"}


def raw_digest(components: Mapping, region_fingerprint_hex: str) -> str:
    """Identity of a raw slice: :func:`raw_components` plus the polygon content.

    Same canonicalization as :func:`series_digest`, over the reducer-free subset, so
    a raw slice and the series derived from it are checkable against each other
    without re-reading the store.
    """
    return series_digest(raw_components(components), region_fingerprint_hex)


def series_digest(components: Mapping, region_fingerprint_hex: str) -> str:
    """The full series digest: parse-time components + the polygon's content.

    Hashed over a canonical JSON serialization (``sort_keys``, no whitespace
    variance) so the value depends on the component *values* and not on mapping
    order or formatting.
    """
    payload = dict(components)
    payload["region_fingerprint"] = region_fingerprint_hex
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_pins(
    observed: Mapping[str, Sequence[str]],
    pinned: Mapping[str, Sequence[str]],
    catalog_entry: str,
    member: str,
) -> None:
    """Raise when what the store now offers differs from the recorded pin (D12).

    Checked at read time, so the pin is *verified* rather than merely nominal —
    the design's argument for why D12 closes ext2-04 rather than documenting it.
    Variables absent from ``pinned`` are skipped: a best-effort variable has no
    pin to check against (ruling A3).
    """
    drifted = {
        var: (list(pinned[var]), list(observed.get(var, [])))
        for var in pinned
        if list(observed.get(var, [])) != list(pinned[var])
    }
    if drifted:
        detail = "; ".join(
            f"{var}: index pinned {want!r}, store now offers {got!r}"
            for var, (want, got) in sorted(drifted.items())
        )
        raise RuntimeError(
            f"store-index pin mismatch for {catalog_entry} member {member}: {detail}. "
            "The store was re-published or the index is stale — regenerate it with "
            "dev/scripts/generate_cmip6_catalog.py (catalog and index must share a "
            "crawled_on)."
        )


def read_series_attrs(path: str | os.PathLike) -> dict:
    """Global attributes of a series file, or ``{}`` if it cannot be read.

    Tolerant by design: this is used to decide whether an existing output can be
    revalidated, and an unreadable or truncated file must fall through to
    re-derivation rather than crash the job. (An interrupted run really does
    leave half-written netCDFs — observed during step-1 validation.)
    """
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            return dict(ds.attrs)
    except Exception:
        return {}


def cache_hit(
    paths: Iterable[str | os.PathLike],
    expected_digest: str,
    digest_attr: str = "cst_series_digest",
) -> bool:
    """True when every declared output already carries ``expected_digest``.

    The revalidation gate of design D9 item 3. Every path must exist, carry a
    schema version this code knows, and match the digest — so a newly-enabled
    gridded output (a missing declared path) correctly forces re-derivation
    rather than being masked by the other files being current.

    ``digest_attr`` selects which identity to check, because the stage-A split
    (revision 6) gives the two layers different ones: a raw slice carries
    ``cst_raw_digest`` and no series digest, a series carries
    ``cst_series_digest``. Defaulted so existing callers are unaffected.
    """
    paths = list(paths)
    if not paths:
        return False
    for path in paths:
        if not os.path.isfile(path):
            return False
        attrs = read_series_attrs(path)
        if attrs.get("cst_schema_version") != SCHEMA_VERSION:
            return False
        if attrs.get(digest_attr) != expected_digest:
            return False
    return True


def write_netcdf_atomic(ds, path: str | os.PathLike) -> None:
    """Write a dataset so an interrupted write cannot leave a valid-looking file.

    A cached artifact that another job trusts must never exist half-written: the
    reader checks attributes, and a truncated netCDF either fails to open (recovered
    by re-derivation) or — worse, if the header landed — opens with the right
    attributes and short data. Write to a sibling temp path and ``os.replace``,
    which is atomic within a filesystem.

    Motivated by measurement, not theory: three killed runs in the R8 session left
    manifest-pinned targets half-written, and one left a still-held handle that
    blocked every fixture gate.

    Data variables are zlib-compressed, matching every other netCDF this workflow
    writes (`get_stats_climate_proj`, `derive_change_factors`,
    `get_change_climate_proj_summary`, `extract_historical_climate` all pass
    ``{"zlib": True}``). The raw tier was the one that did not, for no reason
    anyone recorded. Measured on the nine fixture slices: 638 KB -> 617 KB, 3%.
    Small, because a raw slice is bbox- and time-sliced to well under a megabyte
    by design — this is consistency, not a space win. Compression is LOSSLESS, so
    it is not a precision change: the stored values are bit-identical.

    Scalars are skipped: HDF5 compresses only chunked datasets and a 0-d variable
    (``spatial_ref``) cannot be chunked.
    """
    path = os.fspath(path)
    tmp = f"{path}.tmp-{os.getpid()}"
    encoding = {
        str(name): {"zlib": True} for name, var in ds.data_vars.items() if var.ndim
    }
    try:
        ds.to_netcdf(tmp, encoding=encoding or None)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def assert_raw_identity(
    path: str | os.PathLike,
    expected_raw_digest: str,
    raw_label: str,
) -> None:
    """Raise unless the local raw slice is the one this configuration implies.

    The reduce stage's whole safety argument: it reads a **local** file and never
    reopens the store, so this check is the only thing standing between a stale or
    hand-planted slice and change factors computed from the wrong data. Fail loud,
    naming both digests and the fix.
    """
    attrs = read_series_attrs(path)
    if not attrs:
        raise RuntimeError(
            f"raw slice {raw_label} at {path} is unreadable or empty. Delete it and "
            "re-run: the fetch rule will re-acquire it."
        )
    version = attrs.get("cst_schema_version")
    if version != SCHEMA_VERSION:
        raise RuntimeError(
            f"raw slice {raw_label} carries schema version {version!r}, this code "
            f"knows {SCHEMA_VERSION!r}. Refusing to guess — delete the slice and "
            "re-run to re-acquire it under the current schema."
        )
    found = attrs.get("cst_raw_digest")
    if found != expected_raw_digest:
        raise RuntimeError(
            f"raw slice {raw_label} has cst_raw_digest {found!r}, expected "
            f"{expected_raw_digest!r}. The slice was acquired under a different "
            "catalog entry, pin, acquisition window, buffer, variable set or region. "
            "Delete it and re-run so the fetch rule re-acquires it; do NOT reduce it."
        )


def assert_raw_coverage(
    ds,
    expected_window: Sequence[str],
    expected_variables: Sequence[str],
    raw_label: str,
) -> None:
    """Raise when a raw slice's shape is not what the reduction assumes.

    Identity (:func:`assert_raw_identity`) proves the slice was acquired for this
    configuration; coverage proves it is still *usable*: the variables are present,
    the time axis exists and carries no duplicate step (D8's assertion, which must
    hold on the cached path too), and the recorded acquisition window is the one the
    reduction expects. Deliberately not an equality check on the time span — a store
    that legitimately starts after the requested window is a data condition, not a
    fault, and the recorded window is what says which request produced the slice.
    """
    missing = [v for v in expected_variables if v not in ds.data_vars]
    if missing:
        raise RuntimeError(
            f"raw slice {raw_label} is missing variable(s) {missing}; it holds "
            f"{sorted(ds.data_vars)}. Delete the slice and re-run to re-acquire it."
        )
    index = ds.indexes.get("time") if hasattr(ds, "indexes") else None
    if index is None or len(index) == 0:
        raise RuntimeError(
            f"raw slice {raw_label} has no time axis. Delete it and re-run."
        )
    duplicates = len(index) - len(set(index))
    if duplicates:
        raise RuntimeError(
            f"raw slice {raw_label} has {duplicates} duplicate time step(s), so the "
            "slice was built from an ambiguous store match. Pin the version in the "
            "catalog, delete the slice, and re-run."
        )
    recorded = ds.attrs.get("cst_acquisition_window", "")
    expected = " / ".join(expected_window)
    if recorded != expected:
        raise RuntimeError(
            f"raw slice {raw_label} records acquisition window {recorded!r}, the "
            f"reduction expects {expected!r}. Delete the slice and re-run."
        )


def assert_series_identity(
    path: str | os.PathLike,
    expected_digest: str,
    series_label: str,
) -> None:
    """Fail loud when a series on disk was not derived from the current inputs.

    The stage-B backstop (design D9 route (b), risk-03 mechanism 2). This is an
    assertion inside the consuming job, not a scheduling property, so it holds
    regardless of how Snakemake was invoked — including a series restored from a
    backup, produced by an older checkout, or surviving a non-default
    ``--rerun-triggers`` configuration.
    """
    attrs = read_series_attrs(path)
    found_version = attrs.get("cst_schema_version")
    if found_version != SCHEMA_VERSION:
        raise RuntimeError(
            f"{series_label}: series schema version is {found_version!r}, this code "
            f"knows {SCHEMA_VERSION!r} ({path}). Delete the file and re-run to "
            "re-derive it; do not assume an unknown schema is compatible."
        )
    found = attrs.get("cst_series_digest")
    if found != expected_digest:
        raise RuntimeError(
            f"{series_label}: series digest mismatch ({path}).\n"
            f"  on disk : {found}\n"
            f"  expected: {expected_digest}\n"
            "The file was derived from different inputs than the current "
            "configuration implies — a changed region polygon, catalog entry, "
            "store pin, variable spec, acquisition window, or reducer module. "
            "Delete it and re-run to re-derive."
        )
