"""`provenance.json` — everything needed to reconstruct a run (design §5.9, 6a-iii).

The design lists what it must carry: resolved sources with their **verified
physical store paths** (D12), the region polygon fingerprint (D9), nominal and
effective windows with per-end dropped months and the clip flag (A1),
`shared.historical_window` and the alignment result, the reducer module hash, the
config digest, the variable spec, the catalog and index `crawled_on`, run-level
composition counts, and flagged months.

**This module assembles; it does not derive.** Every value here already exists
somewhere in the pipeline — on a series attribute, in the composition record, in
the reference-window record, in the digest components. Recomputing any of them
would create a second definition, and this milestone has now watched that go wrong
three times: the calendar recorded twice and disagreeing, `n_years` as a calendar
span beside `n_years` as a hydrological count, and the effective window reported
two ways. The falsifier for this file (M6) is therefore not "are the fields
present" but "are they the same objects the other artifacts used".
"""
from __future__ import annotations

import json
import os

#: Bumped when a consumer must notice a change in this document's shape.
PROVENANCE_SCHEMA_VERSION = "1"


def build(
    *,
    clim_project,
    reference_record,
    variable_spec,
    composition_rows,
    series_attrs,
    effective_windows=None,
    catalog_crawled_on,
    reducer_module_hash,
    region_fingerprint,
    horizons,
    weighting_scheme,
):
    """Assemble the provenance document from facts the run already produced.

    ``series_attrs`` maps series key to the attribute dict the reduce stage
    stamped — the single source for calendar, geometry check, weighting scheme,
    acquisition window and the verified store paths, all of which the series
    carries because it had to carry them to be self-describing.

    ``effective_windows`` maps series key to the complete-hydrological-year window
    and count that combination actually used. It is **per source, not per run**,
    and that placement is the fix for a drift caught three times in this milestone:
    the run-level reference record holds a *calendar* span (`1990-2010`), while the
    effective window is what ``hydrological_year_bounds`` derives from the data each
    series has (`1990-01-01 / 2010-12-01`, 21 years). Reporting the calendar span
    under the name "effective" is what `composition.csv` and the change-factor
    tables already avoid; this keeps `provenance.json` consistent with them by
    taking the same values rather than recomputing.
    """
    resolved = [row for row in composition_rows if row.get("status") == "resolved"]

    windows = dict(effective_windows or {})
    sources = []
    for key in sorted(series_attrs):
        attrs = dict(series_attrs[key])
        window = windows.get(key, {})
        sources.append(
            {
                "series_key": key,
                # Per SOURCE, because that is what it is a property of.
                "reference_window_effective": window.get("effective", ""),
                "n_hyd_years_reference": window.get("n_years", ""),
                "catalog_entry": attrs.get("cst_catalog_entry", ""),
                # D12: the physical stores actually read, per variable. The entry
                # name cannot carry this -- its URI globs grid label and version.
                "store_paths": _maybe_json(attrs.get("cst_source_paths", "")),
                "series_digest": attrs.get("cst_series_digest", ""),
                "raw_digest": attrs.get("cst_raw_digest", ""),
                "acquisition_window": attrs.get("cst_acquisition_window", ""),
                "coverage": {
                    "first": attrs.get("cst_time_first", ""),
                    "last": attrs.get("cst_time_last", ""),
                },
                "calendar": attrs.get("cst_calendar", ""),
                "geometry_check": attrs.get("cst_geometry_check", ""),
                "weighting_scheme": attrs.get("cst_weighting_scheme", ""),
                "members": attrs.get("cst_members", ""),
                "crs": attrs.get("cst_crs", ""),
            }
        )

    institutions = {}
    for row in resolved:
        institutions.setdefault(row.get("institution", ""), set()).add(
            row.get("source_id", "")
        )

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "clim_project": clim_project,
        # --- windows (A1): nominal AND effective, never one without the other
        # Run-level: what was REQUESTED, whether it was clipped, and how it
        # aligns with shared.historical_window. The effective window lives
        # per source above -- it depends on the data each one has.
        "reference_window": dict(reference_record),
        "horizon_windows": {
            name: {"nominal": window} for name, window in dict(horizons).items()
        },
        # --- identity
        "region_fingerprint": region_fingerprint,
        "reducer_module_hash": reducer_module_hash,
        "weighting_scheme": weighting_scheme,
        "variable_spec": {
            name: dict(zip(("name", "source", "canonical", "units", "change"), fields))
            for name, fields in dict(variable_spec).items()
        },
        "catalog_crawled_on": catalog_crawled_on,
        # --- what the run actually resolved to
        "sources": sources,
        "composition": {
            "requested": len(composition_rows),
            "resolved": len(resolved),
            "unresolved_by_status": _count_by_status(composition_rows),
            "models": len({row.get("dataset", "") for row in resolved}),
            "institutions": {
                name: len(ids) for name, ids in sorted(institutions.items())
            },
            "members_per_model": _members_per_model(resolved),
        },
        # 6b fills this; the key exists so its absence is a stated zero rather
        # than a missing concept.
        "flagged_months": [],
    }


def _count_by_status(rows) -> dict:
    counts: dict = {}
    for row in rows:
        status = row.get("status", "")
        if status != "resolved":
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _members_per_model(resolved) -> dict:
    per: dict = {}
    for row in resolved:
        per.setdefault(row.get("dataset", ""), set()).add(row.get("member", ""))
    return {model: len(members) for model, members in sorted(per.items())}


def _maybe_json(value):
    """Store paths are stamped as a JSON string; return structure, not a string."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def write(path, document) -> None:
    """Write `provenance.json`, sorted and indented so diffs are reviewable."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
