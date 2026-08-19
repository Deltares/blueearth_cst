"""Source resolution for WF2 v2.0 (design §5.7, D6, D7, D12).

Answers, at DAG-build time and without touching the network, "which requested
(model, scenario, member) combinations can actually be reduced?" — so unresolved
combinations never become jobs. That deletes the pattern this workflow used
before: a job that produced a dummy empty netCDF so Snakemake saw its target,
which downstream code then had to filter out (``filter_nonempty`` and three
separate "did this file have data?" loops).

The check is a two-level lookup against the **generated** catalog:

1. does the entry ``{clim_project}_{model}_{experiment}_{{member}}`` exist?
2. is the requested member in that entry's ``placeholders.member`` list?

Both are plain YAML reads. The generated catalog makes this *stronger* than a
hand-curated one, not merely cheaper: its header states that a source name
resolving means the store is really there, because membership is a live-crawl
fact rather than a curated claim.

**Absence and failure are different classes** (design §4 criterion 7). Almost
everything here is a *normal skip* recorded in the composition record — under
ruling R3′ a model that publishes no ssp370, or one member where another
publishes three, is the expected shape of a correct run. Only two conditions are
configuration errors that stop the DAG build: a model absent from the catalog
entirely, and a run where nothing resolves at all.
"""

from __future__ import annotations

from typing import Iterable, Mapping, NamedTuple, Sequence

#: Resolution outcomes, in the order the ladder tests them (design §5.7).
#: Every status except ``resolved`` means "no data point", and none of them
#: stops the run on its own: the two fatal conditions are decided by the caller
#: from :func:`unknown_models`, :func:`unresolved_overrides`, and "did anything
#: resolve at all". This module records; the Snakefile raises.
RESOLVED = "resolved"
MODEL_NOT_IN_CATALOG = "model_not_in_catalog"
SCENARIO_NOT_PUBLISHED = "scenario_not_published"
MEMBER_NOT_PUBLISHED = "member_not_published"
NO_HISTORICAL_ENTRY = "no_historical_entry"
REFERENCE_MEMBER_UNPUBLISHED = "reference_member_unpublished"
#: A member that WOULD have resolved but was not used, because
#: ``member_selection: first_available`` takes at most one member per model.
#: Recorded rather than dropped: :func:`resolve` emits one row per REQUESTED
#: triple, and the skips are what make the composition record auditable.
MEMBER_SUPERSEDED = "member_superseded"

#: Member-selection policies (board item t2608192107).
#:
#: ``first_available`` takes at most ONE member per model — the first in the
#: preference order that resolves for every requested scenario. ``all`` keeps
#: every member that resolves, which is what this module did before the policy
#: existed and what ruling R3′ describes.
FIRST_AVAILABLE = "first_available"
ALL_MEMBERS = "all"
MEMBER_SELECTION_POLICIES = (FIRST_AVAILABLE, ALL_MEMBERS)


class Combination(NamedTuple):
    """One requested (model, scenario, member) and how it resolved."""

    dataset: str  # institution/source_id, as configured
    scenario: str
    member: str
    status: str
    detail: str = ""

    @property
    def institution(self) -> str:
        return self.dataset.split("/", 1)[0] if "/" in self.dataset else ""

    @property
    def source_id(self) -> str:
        return self.dataset.split("/", 1)[1] if "/" in self.dataset else self.dataset

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED


def entry_key(clim_project: str, model: str, experiment: str) -> str:
    """The generated catalog's entry key for one (model, experiment)."""
    return f"{clim_project}_{model}_{experiment}_{{member}}"


def published_members(catalog: Mapping, key: str) -> list[str]:
    """Members the catalog says exist for one entry, or ``[]`` if absent.

    Reads ``placeholders.member`` — the one placeholder axis the generated
    catalog carries. An entry with no placeholder block publishes nothing, which
    is treated as absent rather than as "any member".
    """
    entry = catalog.get(key)
    if not entry:
        return []
    return list((entry.get("placeholders") or {}).get("member") or [])


def model_in_catalog(catalog: Mapping, clim_project: str, model: str) -> bool:
    """True when the catalog has an entry for this model under ANY experiment.

    Distinguishes "this model is not in the store at all" (a typo or a stale
    config) from "this model does not publish this scenario" (normal). Justified
    by constraint C7: the generated catalog covers the store in full for `Amon`
    `pr`+`tas`, so a name absent from it is absent from the store.
    """
    prefix = f"{clim_project}_{model}_"
    return any(k.startswith(prefix) for k in catalog)


def _model_statuses(catalog, clim_project, model, scenarios, preference):
    """The ladder's verdict for every (scenario, member) of one model.

    Split out of :func:`resolve` because ``first_available`` has to see a whole
    model before it can emit any of its rows: which member wins is a property of
    the model across ALL requested scenarios, not of one scenario.
    """
    known = model_in_catalog(catalog, clim_project, model)
    hist_key = entry_key(clim_project, model, "historical")
    hist_members = published_members(catalog, hist_key)
    verdicts = {}
    for scenario in scenarios:
        scen_key = entry_key(clim_project, model, scenario)
        scen_members = published_members(catalog, scen_key)
        for member in preference:
            if not known:
                verdicts[scenario, member] = (
                    MODEL_NOT_IN_CATALOG,
                    "no catalog entry for this model under any experiment",
                )
            elif not scen_members:
                verdicts[scenario, member] = (
                    SCENARIO_NOT_PUBLISHED,
                    f"no entry {scen_key}",
                )
            elif member not in scen_members:
                verdicts[scenario, member] = (
                    MEMBER_NOT_PUBLISHED,
                    f"published: {', '.join(scen_members[:6])}"
                    + (" …" if len(scen_members) > 6 else ""),
                )
            elif not hist_members:
                # Real, not hypothetical: DKRZ/MPI-ESM1-2-HR publishes SSP
                # members and zero historical members.
                verdicts[scenario, member] = (
                    NO_HISTORICAL_ENTRY,
                    f"no entry {hist_key}; a scenario point cannot be referenced",
                )
            elif member not in hist_members:
                # D7: strict same-member pairing. Pairing r1i1p1f2 future
                # against r1i1p1f1 historical would difference two runs that
                # differ in FORCING VARIANT as well as scenario.
                verdicts[scenario, member] = (
                    REFERENCE_MEMBER_UNPUBLISHED,
                    f"historical publishes: {', '.join(hist_members[:6])}"
                    + (" …" if len(hist_members) > 6 else ""),
                )
            else:
                verdicts[scenario, member] = (RESOLVED, "")
    return verdicts


def _winning_member(verdicts, scenarios, preference):
    """The first member in ``preference`` that resolves for EVERY scenario.

    Completeness is checked across the whole requested scenario set, and that is
    the point rather than a strictness. Per-scenario selection would let
    ``ssp245`` land on ``f1`` while ``ssp370`` lands on ``f2`` for one model,
    each individually D7-valid — and ``analyze_projections.smk`` builds its
    historical need set as ``{(dataset, "historical", member)}``, so that model
    would acquire TWO historical baselines and difference its two scenarios
    against different references.

    Historical needs no separate test: the ladder already refuses a member the
    historical entry does not publish (``REFERENCE_MEMBER_UNPUBLISHED``), so a
    member that resolves for every scenario has a matching reference by
    construction.
    """
    for member in preference:
        if all(verdicts[scenario, member][0] == RESOLVED for scenario in scenarios):
            return member
    return None


def resolve(
    catalog: Mapping,
    *,
    clim_project: str,
    models: Sequence[str],
    scenarios: Sequence[str],
    members: Sequence[str],
    selection: str = FIRST_AVAILABLE,
    overrides: Mapping[str, Sequence[str]] | None = None,
) -> list[Combination]:
    """Resolve every requested (model, scenario, member) through the ladder.

    Returns one :class:`Combination` per **requested** triple — not per resolved
    one. That is the point: the skips are what make the composition record
    auditable, and an enumerated skip is what replaces the run-time
    ``asymmetric hist/clim members`` raise (design D7).

    ``members`` is an ORDERED PREFERENCE, most-wanted first, and ``selection``
    says what to do with it:

    * ``first_available`` (default) — at most ONE member per model, the first
      that resolves for every requested scenario. Every other member that would
      have resolved is recorded as :data:`MEMBER_SUPERSEDED` rather than
      dropped, so the report still says why it was not used.
    * ``all`` — every member that resolves, which is a deliberate multi-member
      ensemble.

    **This supersedes ruling R3′**, which said ``members`` is a requested SET
    and the run's data-point set is the union of the per-combination
    resolutions. Union-of-resolutions is now ``selection="all"``, and it is no
    longer the default. The reason is not tidiness: a config asking for both
    ``r1i1p1f1`` and ``r1i1p1f2`` — the only way to reach the eight models that
    publish solely at ``f2`` — makes CAMS-CSM1-0, EC-Earth3 and NorESM2-LM
    resolve TWICE, and ``get_change_climate_proj_summary.py`` merges across
    models and reduces with ``stats="mean"``. Those three would be weighted
    double in the multi-model ensemble: a silently wrong number.

    ``overrides`` maps a model to its own preference list, REPLACING the global
    one for that model rather than prepending to it. An override is an assertion
    about a specific realisation, so a silent fall-back to the global list would
    defeat the point of writing it; this function records the failure like any
    other and :func:`unresolved_overrides` is what a caller raises on.

    A single-element ``members`` list resolves identically under both policies,
    and every tracked config today is single-element — which is why the default
    can change at all without invalidating a cached slice.
    """
    if selection not in MEMBER_SELECTION_POLICIES:
        raise ValueError(
            f"unknown member_selection {selection!r}; "
            f"expected one of {', '.join(MEMBER_SELECTION_POLICIES)}"
        )
    overrides = overrides or {}
    out: list[Combination] = []
    for model in models:
        preference = list(overrides.get(model) or members)
        verdicts = _model_statuses(catalog, clim_project, model, scenarios, preference)
        winner = (
            _winning_member(verdicts, scenarios, preference)
            if selection == FIRST_AVAILABLE
            else None
        )
        for scenario in scenarios:
            for member in preference:
                status, detail = verdicts[scenario, member]
                if (
                    selection == FIRST_AVAILABLE
                    and status == RESOLVED
                    and member != winner
                ):
                    # Two different facts, and the report prints the detail, so
                    # they are told apart there rather than by a second status.
                    status, detail = (
                        MEMBER_SUPERSEDED,
                        (
                            f"superseded by {winner}, which resolves for every "
                            "requested scenario"
                            if winner
                            else "no requested member resolves for all of "
                            + ", ".join(scenarios)
                        ),
                    )
                out.append(Combination(model, scenario, member, status, detail))
    return out


def unresolved_overrides(
    combinations: Iterable[Combination],
    overrides: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    """Models whose ``member_overrides`` entry resolved for no scenario.

    An override names a specific realisation the operator wants, so its failure
    is a CONFIGURATION ERROR rather than the thin-data skip the same status
    would mean for the global preference list. This reports it; the Snakefile
    raises, which is where every other fatal condition is decided
    (:func:`unknown_models`, and the "nothing resolved at all" check).

    Catches both ways an override can be a no-op, because a raise is right for
    both: the named member resolves for nothing, and the key names a model the
    run does not request at all (a typo in the model name, which nothing else
    would report — ``unknown_models`` only sees models that ARE requested).
    Both reduce to "this model resolved nothing", since a model that is not
    requested cannot resolve.

    A model whose override resolved for SOME scenarios and not others is not
    reported: that is the ordinary shape of a model publishing one scenario and
    not another, and the policy has already decided what to do with it.
    """
    if not overrides:
        return []
    resolved_models = {c.dataset for c in combinations if c.resolved}
    return sorted(model for model in overrides if model not in resolved_models)


def unknown_models(combinations: Iterable[Combination]) -> list[str]:
    """Models that are absent from the catalog — the one model-level error."""
    return sorted({c.dataset for c in combinations if c.status == MODEL_NOT_IN_CATALOG})


def references(combinations: Iterable[Combination]) -> list[tuple[str, str]]:
    """Distinct (model, member) historical references the resolved set needs.

    DISTINCT is the non-obvious half of the job arithmetic: a reference is
    reduced once however many scenarios share it, so three models × two
    scenarios need 3 references rather than 6 — which is why the seed config is
    6 + 3 = 9 reduce jobs and not 12.
    """
    return sorted({(c.dataset, c.member) for c in combinations if c.resolved})


def format_status_report(combinations: Sequence[Combination]) -> str:
    """One line per non-resolved combination, for the DAG-build stderr summary."""
    skipped = [c for c in combinations if not c.resolved]
    if not skipped:
        return ""
    lines = [
        f"WF2 resolution: {len(combinations) - len(skipped)} of "
        f"{len(combinations)} requested combinations resolved; "
        f"{len(skipped)} skipped:"
    ]
    for c in skipped:
        lines.append(f"  {c.dataset} {c.scenario} {c.member}: {c.status} — {c.detail}")
    return "\n".join(lines)


def _as_date_string(value) -> str:
    """Normalize a crawl date to an ISO string, whatever type it arrived as.

    ``yaml.safe_load`` auto-types an unquoted ``2026-07-29`` to ``datetime.date``;
    the JSON index carries the same value as ``str``. Both are the same crawl.
    """
    if value is None:
        return ""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def assert_index_matches_catalog(catalog: Mapping, index: Mapping | None) -> None:
    """The store index and the catalog must come from ONE crawl (design D12/R14).

    Two artifacts written from separate crawls could disagree about which members
    exist with nothing to detect it, so the equal-``crawled_on`` assertion is the
    mechanism that makes them one observation. A missing index is tolerated — a
    project may predate the sidecar — but a *mismatched* one is not.
    """
    if not index:
        return
    # Normalize to ISO strings before comparing. The generator writes
    # `crawled_on: 2026-07-29` unquoted, so yaml.safe_load auto-types it to a
    # datetime.date, while the index's JSON value is a str -- the same crawl in
    # two types. Comparing them raw made this guard fire on every correct run.
    catalog_crawl = _as_date_string((catalog.get("meta") or {}).get("crawled_on"))
    index_crawl = _as_date_string(index.get("crawled_on"))
    if catalog_crawl != index_crawl:
        raise RuntimeError(
            "CMIP6 catalog and store index are from different crawls "
            f"(catalog crawled_on={catalog_crawl!r}, index crawled_on={index_crawl!r}). "
            "They must be generated together: run "
            "dev/scripts/generate_cmip6_catalog.py, which writes both."
        )


def ambiguous_pins(
    index: Mapping | None,
    combinations: Iterable[Combination],
    clim_project: str,
) -> list[str]:
    """Resolved combinations whose store pin is ambiguous (design D8/D12).

    The catalog URI ends ``/{variable}/*/*``, so a glob matching more than one
    ``{grid_label}/{version}`` means the read is not a single identifiable store.
    Measured on the 2026-07-29 crawl: 295 of 4852 pinned stores are ambiguous
    (~6%), so this is a live condition, not a defensive check.
    """
    if not index:
        return []
    sources = index.get("sources") or {}
    problems = []
    for c in combinations:
        if not c.resolved:
            continue
        for experiment in ("historical", c.scenario):
            key = entry_key(clim_project, c.dataset, experiment)
            pins = (sources.get(key) or {}).get(c.member) or {}
            for variable, paths in sorted(pins.items()):
                if len(paths) > 1:
                    problems.append(
                        f"{c.dataset} {experiment} {c.member} {variable}: "
                        f"{len(paths)} stores match ({', '.join(paths)})"
                    )
    return sorted(set(problems))


def best_effort_variables(
    requested: Sequence[str],
    rename: Mapping[str, str],
    certified_sources: Sequence[str] = ("pr", "tas"),
) -> list[str]:
    """Requested variables the catalog does not certify (ruling A3).

    ``rename`` is the catalog entry's whole ``data_adapter.rename`` map, which
    covers **more** than the certified variables — it also renames ``rsds``→``kin``
    and ``psl``→``press_msl``, and those are precisely the *best-effort* ones. So
    the certified set is not "everything the map produces": it is
    ``certified_sources`` (what the crawl actually proved present, mirroring
    ``series_identity.CERTIFIED_VARIABLES``) mapped through the rename.

    Anything else is nameable but unverified for a listed member, so it fails at
    READ time rather than skipping at resolution — an honest tier difference the
    caller warns about rather than silently accepting.
    """
    certified = {rename.get(src, src) for src in certified_sources}
    return sorted(v for v in requested if v not in certified)
