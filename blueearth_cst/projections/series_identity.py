"""Cache identity for the persistent GCM series store (WF2 v2.0, design §5.3).

The series files under ``climate_projections/{clim_project}/`` stop being
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
rewrite ``store_region.geojson`` while ``shared.basin.region`` is unchanged, so a
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
SCHEMA_VERSION = "1"

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
    """
    digest = hashlib.sha256()
    for path in sorted(module_paths, key=lambda p: os.path.basename(str(p))):
        digest.update(os.path.basename(str(path)).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
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


def cache_hit(paths: Iterable[str | os.PathLike], expected_digest: str) -> bool:
    """True when every declared output already carries ``expected_digest``.

    The revalidation gate of design D9 item 3. Every path must exist, carry a
    schema version this code knows, and match the digest — so a newly-enabled
    gridded output (a missing declared path) correctly forces re-derivation
    rather than being masked by the other files being current.
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
        if attrs.get("cst_series_digest") != expected_digest:
            return False
    return True


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
