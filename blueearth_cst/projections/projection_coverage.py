"""WF2 ensemble coverage and conservative model-genealogy metadata.

Coverage comes from explicit POINTS and, when supplied, requested COMBINATIONS;
it never reads Stage B's composition artifact. POINTS alone establish resolved
members but cannot establish how many were requested or rejected. Product-level
availability (analysis/availability.csv) belongs to projection_analysis, not this
module's C0 availability table of member counts.

The cited Zenodo 7920408 record's redistribution licence could not be verified
on 2026-10-07. No external CSV is vendored. The accepted design's fallback keeps
the 13 P0 mappings already used by the lab, with the citation below. An unlisted
model inherits its institution's family only when that institution has exactly
one known family; otherwise it counts conservatively as its own family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

FAMILY_COLUMNS = ("source_id", "family", "institution", "country")
FAMILY_DATA_PROVENANCE = {
    "citation": "Kuma, Bender and Jönsson (2023), JAMES, doi:10.1029/2022MS003588",
    "record": "https://zenodo.org/records/7920408",
    "mode": "cited_13_model_constants",
    "licence": "unverified; external CSV not redistributed",
}

# Family and country facts used by wf2_lab.coverage.MODEL_FAMILIES and
# INSTITUTION_COUNTRIES; source_id/institution pairs are the P0 catalog identities.
# Kuma et al. (2023), models.csv, Family column, cited Zenodo record 7920408.
MODEL_FAMILIES = (
    ("AWI-CM-1-1-MR", "ECMWF", "AWI", "Germany"),
    ("BCC-CSM2-MR", "CCM", "BCC", "China"),
    ("CanESM5", "CanAM", "CCCma", "Canada"),
    ("CMCC-ESM2", "CCM", "CMCC", "Italy"),
    ("INM-CM4-8", "INM", "INM", "Russia"),
    ("INM-CM5-0", "INM", "INM", "Russia"),
    ("IPSL-CM6A-LR", "IPSL", "IPSL", "France"),
    ("MIROC6", "MIROC", "MIROC", "Japan"),
    ("MPI-ESM1-2-LR", "ECMWF", "MPI-M", "Germany"),
    ("MRI-ESM2-0", "UCLA GCM", "MRI", "Japan"),
    ("CESM2", "CCM", "NCAR", "USA"),
    ("NorESM2-MM", "CCM", "NCC", "Norway"),
    ("GFDL-ESM4", "GFDL", "NOAA-GFDL", "USA"),
)

COVERAGE_COLUMNS = (
    "model",
    "institution",
    "source_id",
    "family",
    "family_source",
    "country",
    "scenario",
    "member",
    "status",
    "reason",
    "available",
    "requested_known",
    "n_reference_years",
    "series_key",
    "reference_series_key",
    "calendar",
    "reference_calendar",
    "start",
    "end",
    "reference_start",
    "reference_end",
)
MATRIX_COLUMNS = (
    "model",
    "institution",
    "source_id",
    "family",
    "family_source",
    "country",
    "scenario",
    "n_requested",
    "n_resolved",
    "state",
    "members",
    "reason",
)
MODEL_COLUMNS = (
    "model",
    "institution",
    "source_id",
    "family",
    "family_source",
    "country",
)


def model_families() -> pd.DataFrame:
    """Return a fresh table of the cited fallback facts; no network or file IO."""
    return pd.DataFrame(MODEL_FAMILIES, columns=FAMILY_COLUMNS)


def family_metadata(
    source_id: str, institution: str, families: pd.DataFrame | None = None
) -> dict[str, str]:
    """Resolve listed / provisional / own family, keeping country independent.

    Country is known only when the institution has one unambiguous nonblank
    country in the reference table. Unknown or ambiguous family is legitimate,
    never a reason to remove a model. ``families`` allows a verified table to be
    supplied directly; the default is the licence-gated 13-model fallback.
    """
    source_id = str(source_id or "").strip()
    institution = str(institution or "").strip()
    if not source_id:
        raise ValueError("family lookup requires a nonempty source_id")
    families = model_families() if families is None else families
    if not set(FAMILY_COLUMNS).issubset(families.columns):
        raise ValueError(f"family table needs columns {FAMILY_COLUMNS}")
    if families.source_id.duplicated().any():
        raise ValueError("family table has duplicate source_id values")
    listed = families.loc[families.source_id == source_id]
    same_institution = families.loc[families.institution == institution]
    known = set(same_institution.family.dropna()) - {""}
    countries = set(same_institution.country.dropna()) - {""}
    country = next(iter(countries)) if len(countries) == 1 else ""
    if not listed.empty and pd.notna(listed.iloc[0].family) and listed.iloc[0].family:
        family, source = str(listed.iloc[0].family), "listed"
    elif len(known) == 1:
        family, source = str(next(iter(known))), "provisional"
    else:
        family, source = source_id, "own"
    return {"family": family, "family_source": source, "country": country}


def _identity(record: Mapping) -> tuple[str, str, str, str, str]:
    """Read unsanitized model/scenario/member identity from an explicit record."""
    model = str(record.get("model") or record.get("dataset") or "")
    institution, sep, source_id = model.partition("/")
    if not sep:
        source_id, institution = model, str(record.get("institution") or "")
        # The catalog entry carries the original identity, never a filename slug.
        entry = record.get("catalog_entry") or (
            record.get("digest_components_fut") or {}
        ).get("catalog_entry", "")
        head, slash, tail = str(entry).partition("/")
        project, underscore, candidate = head.partition("_")
        if (
            not institution
            and slash
            and underscore
            and project
            and tail.startswith(f"{source_id}_")
        ):
            institution = candidate
        if institution:
            model = f"{institution}/{source_id}"
    scenario, member = (
        str(record.get("scenario") or ""),
        str(record.get("member") or ""),
    )
    if not source_id or not scenario or not member:
        raise ValueError("coverage record requires model, scenario and member")
    return model, institution, source_id, scenario, member


def coverage_table(
    points: Sequence[Mapping],
    *,
    combinations: Sequence[Mapping] | None = None,
    inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per explicit point or requested combination, with family metadata.

    ``points`` are the resolved POINTS records passed by the rule. Optional
    ``combinations`` are direct resolver records, not a Stage B artifact: they
    supply rejected rows and make requested counts knowable. A resolved request
    missing its POINTS record is inconsistent input and raises. Optional
    ``inventory`` contains one row per loaded series (series_key, calendar,
    start, end); if supplied, every resolved series and its reference must exist.
    """
    point_by_key = {}
    for point in points:
        identity = _identity(point)
        key = (identity[0], identity[3], identity[4])
        if key in point_by_key:
            raise ValueError(f"duplicate resolved coverage point: {key}")
        point_by_key[key] = dict(point)
    requests = points if combinations is None else combinations
    series = None
    if inventory is not None:
        if "series_key" not in inventory or inventory.series_key.duplicated().any():
            raise ValueError("inventory requires unique series_key values")
        series = inventory.set_index("series_key")
    rows, seen = [], set()
    for record in requests:
        model, institution, source_id, scenario, member = _identity(record)
        key = (model, scenario, member)
        if key in seen:
            raise ValueError(f"duplicate requested coverage combination: {key}")
        seen.add(key)
        status = str(record.get("status", "resolved"))
        available = status == "resolved"
        point = point_by_key.get(key)
        if available and point is None:
            raise ValueError(f"resolved coverage request has no point: {key}")
        if not available and point is not None:
            raise ValueError(f"excluded coverage request has a resolved point: {key}")
        row = dict.fromkeys(COVERAGE_COLUMNS, "")
        row.update(
            model=model,
            institution=institution,
            source_id=source_id,
            **family_metadata(source_id, institution),
            scenario=scenario,
            member=member,
            status=status,
            reason=record.get("detail", record.get("reason", "")),
            available=available,
            requested_known=combinations is not None,
            n_reference_years=pd.NA,
        )
        if point is not None:
            row.update(
                series_key=point.get("series_key", ""),
                reference_series_key=point.get("reference_series_key", ""),
                n_reference_years=point.get("n_reference_years", pd.NA),
            )
            if series is not None:
                for role, column, prefix in (
                    ("scenario", "series_key", ""),
                    ("reference", "reference_series_key", "reference_"),
                ):
                    if row[column] not in series.index:
                        raise ValueError(
                            f"resolved coverage {key}: {role} series missing from inventory"
                        )
                    facts = series.loc[row[column]]
                    for field in ("calendar", "start", "end"):
                        row[prefix + field] = facts.get(field, "")
        rows.append(row)
    missing = set(point_by_key) - seen
    if missing:
        raise ValueError(
            f"resolved points absent from requested coverage: {sorted(missing)}"
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def _scenarios(table: pd.DataFrame, scenarios: Sequence[str] | None) -> list[str]:
    """Stable, distinct scenario columns; historical is counted separately."""
    values = table.scenario if scenarios is None else scenarios
    return list(dict.fromkeys(str(value) for value in values if value != "historical"))


def coverage_matrix(
    table: pd.DataFrame, scenarios: Sequence[str] | None = None
) -> pd.DataFrame:
    """Member counts per model/scenario, preserving unknown requested coverage.

    With POINTS alone the states are ``resolved`` / ``no_resolved_member`` and
    n_requested is unknown. Direct COMBINATIONS enable complete / partial /
    excluded / not_requested states; absence then has an evidential meaning.
    """
    scenarios = _scenarios(table, scenarios)
    models = table[list(MODEL_COLUMNS)].drop_duplicates("model")
    requested_known = not table.empty and bool(table.requested_known.all())
    rows = []
    for model in models.to_dict("records"):
        for scenario in scenarios:
            group = table.loc[
                (table.model == model["model"]) & (table.scenario == scenario)
            ]
            resolved = group.loc[group.available.astype(bool)]
            n_resolved = resolved.member.nunique()
            if requested_known:
                n_requested = len(group)
                state = (
                    "not_requested"
                    if not n_requested
                    else "complete"
                    if n_resolved == n_requested
                    else "excluded"
                    if not n_resolved
                    else "partial"
                )
            else:
                n_requested = pd.NA
                state = "resolved" if n_resolved else "no_resolved_member"
            excluded = group.loc[~group.available.astype(bool)]
            reason = "; ".join(
                f"{item.member}: {item.status}"
                + (f" ({item.reason})" if item.reason else "")
                for item in excluded.itertuples(index=False)
            )
            if not len(group):
                reason = (
                    "scenario not requested for this model"
                    if requested_known
                    else "no resolved POINTS record; requested coverage unknown"
                )
            rows.append(
                {
                    **model,
                    "scenario": scenario,
                    "n_requested": n_requested,
                    "n_resolved": n_resolved,
                    "state": state,
                    "members": ",".join(resolved.member.astype(str)),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=MATRIX_COLUMNS)


def availability_table(
    table: pd.DataFrame, scenarios: Sequence[str] | None = None
) -> pd.DataFrame:
    """C0 wide counts of distinct resolved members; zero means none resolved.

    Historical counts distinct reference members of resolved scenarios, so a
    member present under several scenarios contributes only once. This view
    makes no claim about unobserved requested or excluded combinations.
    """
    scenarios = _scenarios(table, scenarios)
    columns = [*MODEL_COLUMNS, "historical", *scenarios]
    rows = []
    for model in table[list(MODEL_COLUMNS)].drop_duplicates("model").to_dict("records"):
        resolved = table.loc[
            (table.model == model["model"]) & table.available.astype(bool)
        ]
        rows.append(
            {
                **model,
                "historical": resolved.member.nunique(),
                **{
                    scenario: resolved.loc[
                        resolved.scenario == scenario, "member"
                    ].nunique()
                    for scenario in scenarios
                },
            }
        )
    return pd.DataFrame(rows, columns=columns)
