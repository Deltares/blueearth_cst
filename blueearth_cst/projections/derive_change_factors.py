"""Stage B — all change factors in ONE job (migration step 4d, design §8).

Replaces the pair `monthly_change` (fanned out per `point_key` × horizon) and
`monthly_change_scalar_merge` (a single aggregator over their `temp()` outputs).
Stage B runs **1 job** with no fan-out, reading the explicit expanded series list.
Current workflow navigation: `dev/reference/toolbox-workflow-map.md`.
Historical rationale: `dev/records/milestones/r08/wf2-climate-analysis-v2-design.md` §5, "B. Derive".

**This step is value-neutral by construction.** The arithmetic is not reimplemented
here: `get_change_annual_clim_proj` and
the existing coordinate-safe merge are imported from their owning modules,
unchanged. Only the orchestration moves. A non-zero characterized diff on the
summary artifacts is therefore a defect in this file, not a judgement call — which
is the whole reason the functions were left where they were.

Per-point annual/monthly results remain in memory through the merge
and tidy tables. Each validated scalar series is loaded and closed once per job.
The file-writing helpers remain available for direct callers.

Stage B's input set is explicit (design risk-06 / revision 4): the rule declares
exactly the expanded `{series_key}` list built from the resolved combination set,
and this job **asserts that the set it opened equals that list**. A model removed
from the config cannot rejoin the run through a leftover file in `scalar/`.

Invoked from ``analyze_projections.smk`` via ``script:``; reads
``snakemake.input/output/params``, never ``sys.argv``.
"""
# NOTE: no `from __future__ import annotations` here — Snakemake's `script:`
# directive prepends its own preamble to a copy of this file, so a __future__
# import lands mid-file and raises SyntaxError at job start. A --dry-run cannot
# catch it (it never executes a script body); the other `script:` modules in this
# repo omit it for the same reason.

import csv
import os

import xarray as xr

from blueearth_cst.projections import provenance as _prov
from blueearth_cst.projections import reference_window, series_identity
from blueearth_cst.projections import report as _report
from blueearth_cst.projections.change_factor_table import (
    TABLE_COLUMNS_ANNUAL,
    TABLE_COLUMNS_MONTHLY,
    csv_value,
    tidy_rows,
    write_table,
)
from blueearth_cst.projections.dry_month import FLAGGED_STATUS, combination_is_flagged
from blueearth_cst.projections.get_change_climate_proj import (
    NoCompleteHydrologicalYear,
    _to_str_tuple,
    get_change_annual_clim_proj,
    get_change_monthly_clim_proj,
    hydrological_year_bounds,
)
from blueearth_cst.projections.get_change_climate_proj_summary import combine_changes
from blueearth_cst.projections.series_checks import (
    assert_calendar_agreement,
    load_checked_series,
)
from blueearth_cst.projections.variable_spec import VariableSpec
from blueearth_cst.shared.snake_utils import log_row, plural, tee_to_log

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")


def _flagged_months(monthly_rows, composition, max_flagged_months):
    """Count distinct flagged months, retaining provenance's full dataset id."""
    datasets = {
        (row["source_id"], row["scenario"], row["member"]): row["dataset"]
        for row in composition
    }
    flagged_months = {}
    for row in monthly_rows:
        if row["status"] == FLAGGED_STATUS:
            key = (
                datasets[(row["model"], row["scenario"], row["member"])],
                row["scenario"],
                row["member"],
                row["horizon"],
                row["variable"],
            )
            # Statistics share a month; they must not multiply its count.
            flagged_months.setdefault(key, set()).add(row["month"])
    return [
        {
            "dataset": k[0],
            "scenario": k[1],
            "member": k[2],
            "horizon": k[3],
            "variable": k[4],
            "n_flagged_months": len(months),
            "exceeds_max": combination_is_flagged(len(months), max_flagged_months),
        }
        for k, months in sorted(flagged_months.items())
    ]


# `_to_str_tuple` is IMPORTED, not reimplemented, despite the leading underscore.
# A local copy was written first and was already wrong: it raised on `[]`, where
# the original returns `()` — a contract `tests/test_get_change_climate_proj.py`
# pins. Reimplementing a normaliser is exactly the drift this step is meant to
# avoid, so the private name is the lesser evil.


class WindowUnavailable(ValueError):
    """Valid scalar data cannot supply a requested change-factor window."""

    def __init__(self, message, *, window, dataset, path, window_kind):
        super().__init__(message)
        dates = dataset["time"].values
        self.record = {
            "window_kind": window_kind,
            "requested_window": " / ".join(map(str, window)),
            "available_start": str(dates[0])[:10],
            "available_end": str(dates[-1])[:10],
            "reason": message,
            "series_path": str(path),
        }


def _slice_analysis_window(dataset, window, label, path, window_kind="horizon"):
    """Slice a GCM period, naming its source and coverage when no dates remain."""
    requested = "..".join(str(bound) for bound in window)
    if dataset.sizes.get("time", 0) == 0:
        raise ValueError(
            f"{label}: {path} has an empty time axis; requested {requested}"
        )
    selected = dataset.sel(time=slice(*window))
    if selected.sizes["time"] == 0:
        dates = dataset["time"].values
        available = f"{str(dates[0])[:10]}..{str(dates[-1])[:10]}"
        raise WindowUnavailable(
            f"{label}: requested {requested} has no overlap with available "
            f"{available} in {path}. Check the configured GCM window and this "
            "scenario's reduced series; the observed climate window is independent.",
            window=window,
            dataset=dataset,
            path=path,
            window_kind=window_kind,
        )
    return selected


def _window_bounds(selected, dataset, window, path, kind, water_year_start):
    """Translate only the kernel's explicit absence-of-complete-years error."""
    try:
        return hydrological_year_bounds(selected, water_year_start)
    except NoCompleteHydrologicalYear as exc:
        raise WindowUnavailable(
            str(exc), window=window, dataset=dataset, path=path, window_kind=kind
        ) from exc


def derive_point_datasets(
    *,
    series_path_hist,
    series_path,
    time_tuple_hist,
    time_tuple_fut,
    name_horizon,
    name_model,
    name_scenario,
    region_fp,
    digest_components_hist,
    digest_components_fut,
    stats=None,
    variable_spec=None,
    min_reference=None,
    clim_project_dir=None,
    water_year_start="Jan",
    series_cache=None,
    reference_cache=None,
):
    """Change factors for one (model, scenario, member) at one horizon.

    The body is the former ``monthly_change`` job, moved verbatim apart from
    taking its inputs as arguments instead of reading ``snakemake.params``.

    Returns the **effective reference window** it used — ``(start, end, n_years)``
    from :func:`hydrological_year_bounds`, the same call the change arithmetic
    makes — so the composition record annotates the numbers with the window that
    produced them rather than with a recomputed guess.
    """
    # Caches belong to this job only. Every cache entry is keyed by the expected
    # identity, so a shared path with inconsistent expectations still fails.
    series_cache = {} if series_cache is None else series_cache
    datasets = []
    expected = []
    for label, path, components in (
        ("historical", series_path_hist, dict(digest_components_hist)),
        (name_scenario, series_path, dict(digest_components_fut)),
    ):
        dataset, digest = load_checked_series(
            path, components, region_fp, f"{name_model} {label}", series_cache
        )
        datasets.append(dataset)
        expected.append(digest)
    ds_hist_time, ds_clim_time = datasets

    # Validate defects before deciding coverage, including horizons with no overlap.
    for dataset in datasets:
        index = dataset.indexes["time"]
        if not len(index) or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("scalar series requires nonempty, ordered, unique time")
    calendar = assert_calendar_agreement(
        ds_hist_time, ds_clim_time, f"{name_model} {name_scenario}"
    )
    if set(ds_hist_time.data_vars) != set(ds_clim_time.data_vars):
        raise ValueError("asymmetric hist/clim variables")
    missing = set(variable_spec or {}) - set(ds_hist_time.data_vars)
    if missing:
        raise ValueError(f"missing configured variables: {sorted(missing)}")
    if "member" in ds_hist_time.coords and "member" in ds_clim_time.coords:
        if set(ds_hist_time.member.values) != set(ds_clim_time.member.values):
            raise ValueError("asymmetric hist/clim members")

    # Step 4c: the `if len(ds_clim_time) > 0` guard and its dummy-netCDF
    # else-branch are gone. Since 4a an unresolved combination never becomes a
    # job, so an empty series here means a real defect.
    if len(ds_clim_time) == 0:
        raise RuntimeError(
            f"{series_path} holds no data variables. Resolution admitted this "
            "combination, so an empty series is a defect rather than an "
            "unpublished source -- delete the series and re-run to re-derive."
        )

    full_hist, full_future = ds_hist_time, ds_clim_time
    ds_hist_time = _slice_analysis_window(
        ds_hist_time,
        time_tuple_hist,
        f"{name_model} historical reference",
        series_path_hist,
        "reference",
    )
    ds_clim_time = _slice_analysis_window(
        ds_clim_time,
        time_tuple_fut,
        f"{name_model} {name_scenario} horizon {name_horizon!r}",
        series_path,
    )
    # Read the effective reference window off the SAME helper the change
    # arithmetic uses, after the same slice. Not a recomputation: one function,
    # called twice on one dataset.
    #
    # The configured water year, matching the arithmetic below. Until
    # 2026-08-12 both were hardcoded to "Jan": the Snakefile read
    # `start_month_hyd_year` and passed it to this rule, and this module never
    # read the param, so every change factor was computed Jan-Dec whatever the
    # config said. That deferral ("belongs in its own commit with its own
    # gate") is this commit. `shared.water_year_start` now reaches the
    # arithmetic, and the legacy key is refused at parse time rather than
    # silently starting to work.
    ref_start, ref_end, ref_n_years = _window_bounds(
        ds_hist_time,
        full_hist,
        time_tuple_hist,
        series_path_hist,
        "reference",
        water_year_start,
    )
    # The SAME call on the scenario slice. `get_change_annual_clim_proj` already
    # makes it internally (it needs the bounds to aggregate) and then discards the
    # result, which is why `horizon_window` used to report the config's nominal
    # years while `reference_window` reported the window the arithmetic used. Two
    # columns, two meanings, two formats. Now both report the effective window.
    # Same water year as the reference above and as the arithmetic: reporting a
    # window under a start month the arithmetic did not use would be a worse
    # defect than the one this replaced.
    hor_start, hor_end, _ = _window_bounds(
        ds_clim_time,
        full_future,
        time_tuple_fut,
        series_path,
        "horizon",
        water_year_start,
    )
    # Step 5b: weight each month by its length in the MODEL's calendar. Read off
    # the series (propagated there from the raw slice, which got it from the store
    # -- the axis itself cannot say, having been converted to datetime64 upstream).
    # Both series must agree: a change factor differencing two calendars would be
    # comparing incomparable annual aggregates. Agreement is on month LENGTHS, not
    # names -- MIROC6 labels its historical run `proleptic_gregorian` and its
    # scenarios `gregorian`, which are the same months from 1582 on.
    calendar = assert_calendar_agreement(
        ds_hist_time, ds_clim_time, f"{name_model} {name_scenario}"
    )

    # The cache is bound to one validated historical series and analysis
    # configuration, never to a future horizon or scenario.
    reference_cache = {} if reference_cache is None else reference_cache
    reference_key = (
        expected[0],
        tuple(time_tuple_hist),
        calendar,
        water_year_start,
        tuple(stats) if stats is not None else None,
        tuple(sorted((k, tuple(v)) for k, v in (variable_spec or {}).items())),
        tuple(sorted((min_reference or {}).items())),
    )
    cached_reference = reference_cache.setdefault(reference_key, {})

    stats_annual_change = get_change_annual_clim_proj(
        ds_hist_time,
        ds_clim_time,
        calendar=calendar,
        stats=stats,
        variable_spec=variable_spec,
        start_month_hyd_year=water_year_start,
        _reference_cache=cached_reference,
    )
    stats_annual_change = stats_annual_change.assign_coords(
        {"horizon": f"{name_horizon}"}
    ).expand_dims(["horizon"])
    stats_annual_change = stats_annual_change.transpose(
        ..., "clim_project", "model", "scenario", "horizon", "member"
    )

    monthly_change = get_change_monthly_clim_proj(
        ds_hist_time,
        ds_clim_time,
        stats=stats,
        variable_spec=variable_spec,
        min_reference=min_reference,
        start_month_hyd_year=water_year_start,
        _reference_cache=cached_reference,
    )
    monthly_change = monthly_change.assign_coords(
        {"horizon": f"{name_horizon}"}
    ).expand_dims(["horizon"])
    facts = (ref_start, ref_end, ref_n_years, hor_start, hor_end)
    return stats_annual_change, monthly_change, facts


def derive_available_point(*, point, unavailable_windows, **kwargs):
    """Collect a narrowly classified exclusion; every input defect still raises."""
    try:
        return derive_point_datasets(**kwargs)
    except WindowUnavailable as exc:
        record = dict(exc.record)
        record.update(
            dataset=point["model"],
            model=point["model"].split("/")[-1],
            scenario=point["scenario"],
            member=point["member"],
            horizon=kwargs["name_horizon"],
        )
        unavailable_windows.append(record)
        log_row(
            f"Skipped {point['model']} · {point['scenario']} · {point['member']} · "
            f"{kwargs['name_horizon']}: {record['window_kind']} "
            f"{record['requested_window']}; {' '.join(str(record.get('reason', 'requested window is not usable')).splitlines())}; "
            f"data {record['available_start']}–{record['available_end']}; details → summary/provenance.json",
            module="change",
            level="WARNING",
        )
        return None


def derive_one_point(
    *,
    series_path_hist,
    series_path,
    change_nc_out,
    time_tuple_hist,
    time_tuple_fut,
    name_horizon,
    name_model,
    name_scenario,
    region_fp,
    digest_components_hist,
    digest_components_fut,
    stats=None,
    variable_spec=None,
    min_reference=None,
    clim_project_dir=None,
    water_year_start="Jan",
):
    """Preserve the file-writing API for callers outside Stage B."""
    annual, monthly, facts = derive_point_datasets(
        series_path_hist=series_path_hist,
        series_path=series_path,
        time_tuple_hist=time_tuple_hist,
        time_tuple_fut=time_tuple_fut,
        name_horizon=name_horizon,
        name_model=name_model,
        name_scenario=name_scenario,
        region_fp=region_fp,
        digest_components_hist=digest_components_hist,
        digest_components_fut=digest_components_fut,
        stats=stats,
        variable_spec=variable_spec,
        min_reference=min_reference,
        clim_project_dir=clim_project_dir,
        water_year_start=water_year_start,
    )
    annual.to_netcdf(
        change_nc_out, encoding={k: {"zlib": True} for k in annual.data_vars}
    )
    monthly.to_netcdf(
        str(change_nc_out).replace(".nc", "_monthly.nc"),
        encoding={k: {"zlib": True} for k in monthly.data_vars},
    )
    return facts


#: Fields of the IN-MEMORY composition record, in design §5.7 order. One row per
#: REQUESTED (model, scenario, member) — not per resolved one, which is the point:
#: the skips are what make the record auditable.
#:
#: This is wider than the CSV. `provenance.py` builds its institution roll-up from
#: these rows, so the record keeps `institution`/`source_id` even though S8-05
#: collapsed them to one `model` column on disk.
COMPOSITION_FIELDS = [
    "dataset",
    "institution",
    "source_id",
    "scenario",
    "member",
    "status",
    "reason",
    "series_key",
    "reference_series_key",
    "catalog_entry",
    "catalog_crawled_on",
    "tier",
    "reference_window_nominal",
    "reference_window_effective",
    "n_hyd_years_reference",
]

#: What `composition.csv` actually carries, as `(csv_column, record_field)`.
#: S8-05, 15 -> 10. Dropped: `dataset`/`institution` (collapsed into `model`;
#: `source_id` is globally unique in the CMIP6 controlled vocabulary, so the
#: institution is recoverable), `catalog_crawled_on` (constant, and already in
#: `provenance.json`), and both `reference_window_*` columns — the change-factor
#: tables now carry the window, so one artifact owns the fact instead of three.
#:
#: `n_reference_years` is KEPT: it is a property of a series, not of the run, so a
#: model with a short record genuinely reports fewer complete hydrological years.
COMPOSITION_CSV_COLUMNS = [
    ("model", "source_id"),
    ("scenario", "scenario"),
    ("member", "member"),
    ("status", "status"),
    ("reason", "reason"),
    ("tier", "tier"),
    ("series_key", "series_key"),
    ("reference_series_key", "reference_series_key"),
    ("catalog_entry", "catalog_entry"),
    ("n_reference_years", "n_hyd_years_reference"),
]


def composition_rows(
    combinations,
    resolved,
    *,
    catalog_crawled_on,
    window_nominal,
    unavailable_windows=None,
):
    """Build the composition record from the resolution ladder plus run facts.

    ``combinations`` is every REQUESTED triple with its ladder status, as decided
    at DAG build. ``resolved`` maps ``point_key`` to the facts only the job knows —
    series keys, tier, and the effective window `derive_one_point` reported.

    Rows for non-resolved combinations carry the status and reason and leave every
    resolved-only column empty; that asymmetry is the record's whole purpose.
    """
    rows = []
    for combo in combinations:
        combo = dict(combo)
        point_key = combo.get("point_key", "")
        row = dict.fromkeys(COMPOSITION_FIELDS, "")
        row.update(
            dataset=combo.get("dataset", ""),
            institution=combo.get("institution", ""),
            source_id=combo.get("source_id", ""),
            scenario=combo.get("scenario", ""),
            member=combo.get("member", ""),
            status=combo.get("status", ""),
            reason=combo.get("detail", ""),
            catalog_entry=combo.get("catalog_entry", ""),
            catalog_crawled_on=catalog_crawled_on,
        )
        extra = resolved.get(point_key)
        if extra is not None:
            row.update(extra)
            row["reference_window_nominal"] = window_nominal
        exclusions = [
            entry
            for entry in (unavailable_windows or [])
            if entry["dataset"] == f"{row['institution']}/{row['source_id']}"
            and entry["scenario"] == row["scenario"]
            and entry["member"] == row["member"]
        ]
        if exclusions:
            row["status"] = "partial" if extra is not None else "unavailable"
            row["reason"] = "Unavailable windows: " + "; ".join(
                f"{entry['horizon']} ({entry['window_kind']}): {entry['reason']}"
                for entry in exclusions
            )
        rows.append(row)
    return rows


def write_composition(path, rows):
    """Write ``composition.csv``. Stage-B output: it describes a COMPLETED run.

    ext2-08 / D4: the record is written here and not at DAG build, because a DAG
    build that writes an output file makes parsing side-effecting — a dry run that
    writes is not a dry run. A failed run therefore leaves the DAG-build stderr
    summary and the job logs, and no composition artifact.

    S8-05: the in-memory record is wider than the file. Projected here rather than
    trimmed at construction, because `provenance.py` reads the full rows.

    Cells go through `csv_value` for the same reason the tidy tables do — one
    number format across every WF2 CSV. Today this file carries no float column
    (`n_reference_years` is an int and passes through untouched); applying it here
    is what keeps that true if one is ever added.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    columns = [name for name, _ in COMPOSITION_CSV_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: csv_value(row.get(field, ""))
                    for name, field in COMPOSITION_CSV_COLUMNS
                }
            )


if "snakemake" in globals():
    sm = globals()["snakemake"]

    with tee_to_log(sm.log[0]):
        clim_project_dir = sm.params.clim_project_dir
        horizons = sm.params.horizons
        points = [dict(p) for p in sm.params.points]

        # D9: every expected digest is recomputed against the polygon ON DISK, so
        # a series derived for a different region cannot be reused.
        region_fp = series_identity.region_fingerprint(sm.input.region_path)

        # risk-06 / revision 4: the set opened must equal the set declared. A
        # leftover file in scalar/ cannot rejoin a run whose config dropped it.
        declared = {os.path.abspath(str(p)) for p in sm.input.series_nc}
        opened = {
            os.path.abspath(str(path))
            for point in points
            for path in (point["series_path_hist"], point["series_path"])
        }
        if opened != declared:
            raise RuntimeError(
                "derive_change_factors: the series set to open does not equal the "
                "declared input set.\n"
                f"  declared but unused: {sorted(declared - opened)}\n"
                f"  used but undeclared: {sorted(opened - declared)}"
            )

        log_row(
            f"Deriving change factors: {plural(len(points), 'model–scenario–member combination')}, "
            f"{plural(len(horizons), 'future window')}",
            module="change",
        )
        # The independent GCM reference period and any historical coverage clip.
        # provenance.json keeps the structured `key=value` copy.
        for _row in reference_window.console_rows(dict(sm.params.reference_record)):
            log_row(_row, module="change")

        # Snakemake params carry plain data; rebuild the typed spec here so the
        # aggregation looks up fields by name rather than by list position.
        VARIABLE_SPEC = {
            name: VariableSpec(*fields)
            for name, fields in dict(sm.params.variable_spec).items()
        }
        resolved_facts = {}
        # Keyed by (point_key, horizon) — unlike the reference window, the
        # effective horizon window is a property of a series AND the horizon it
        # was sliced to.
        horizon_facts = {}
        series_cache = {}
        reference_cache = {}
        annual_changes = []
        monthly_changes = []
        unavailable_windows = []
        for point in points:
            for horizon_name, horizon_window in horizons.items():
                result = derive_available_point(
                    point=point,
                    unavailable_windows=unavailable_windows,
                    series_path_hist=point["series_path_hist"],
                    series_path=point["series_path"],
                    time_tuple_hist=_to_str_tuple(sm.params.time_horizon_hist),
                    time_tuple_fut=_to_str_tuple(horizon_window),
                    name_horizon=horizon_name,
                    name_model=point["model"],
                    name_scenario=point["scenario"],
                    region_fp=region_fp,
                    digest_components_hist=point["digest_components_hist"],
                    digest_components_fut=point["digest_components_fut"],
                    stats=sm.params.stats,
                    water_year_start=sm.params.water_year_start,
                    variable_spec=VARIABLE_SPEC,
                    min_reference=sm.params.min_reference,
                    clim_project_dir=clim_project_dir,
                    series_cache=series_cache,
                    reference_cache=reference_cache,
                )
                if result is None:
                    continue
                annual, monthly, facts = result
                ref_start, ref_end, ref_n_years, hor_start, hor_end = facts
                annual_changes.append(annual)
                monthly_changes.append(monthly)
                # Same for every horizon of a point (the reference window does
                # not depend on the horizon), so recording it repeatedly is
                # harmless and keeps the loop single-pass.
                resolved_facts[point["point_key"]] = {
                    "series_key": point["series_key"],
                    "reference_series_key": point["reference_series_key"],
                    "tier": point["tier"],
                    "reference_window_effective": (
                        f"{ref_start:%Y-%m-%d} / {ref_end:%Y-%m-%d}"
                    ),
                    "n_hyd_years_reference": ref_n_years,
                }
                horizon_facts[(point["point_key"], horizon_name)] = (
                    f"{hor_start:%Y-%m-%d} / {hor_end:%Y-%m-%d}"
                )

        log_row(
            f"Change factors: {len(annual_changes)}/{len(annual_changes) + len(unavailable_windows)} combination–window pairs available; "
            f"{len(unavailable_windows)} skipped",
            module="change",
        )
        merged = combine_changes(annual_changes) if annual_changes else xr.Dataset()
        # Both products need source-coordinate cleanup (including scalar height)
        # before combining results from different models.
        monthly_merged = (
            combine_changes(monthly_changes)
            if monthly_changes
            else xr.Dataset(coords={"month": []})
        )
        # The outer join spans model x scenario x member, so models with differing
        # member labels leave cells no series produced. Every consumer of the
        # merge reads only the combinations that were actually derived.
        derived = {
            (p["model"], p["scenario"], p["member"])
            for p in points
            if p["point_key"] in resolved_facts
        }
        available_horizons = {
            (p["model"], p["scenario"], p["member"], horizon)
            for p in points
            for horizon in horizons
            if (p["point_key"], horizon) in horizon_facts
        }

        # --- step 6a-i: the tidy annual change-factor table (design §5.9) ----
        # The tidy tables consume the same merged dataset.
        #
        # S8-04: two facts, both varying. Everything else this dict used to carry
        # was constant across every row (the nominal windows, n_years), hardcoded
        # empty (n_years_dropped, horizon_window_effective), or actively wrong
        # (`units`, which labelled a percent as mm/day). Units now come from the
        # spec inside tidy_rows, where the change kind that decides them lives.
        window_facts = {
            # The EFFECTIVE window -- what actually backed the arithmetic. The
            # nominal config echo is report.md's job; it is constant and already
            # recorded there and in provenance.json.
            "reference_window": sm.params.reference_record.get(
                "reference_window_effective", ""
            ),
            # The NOMINAL config years, as a fallback only: every row a run
            # resolves gets the effective window from `row_facts` below. This is
            # what a row would carry if its combination never resolved, and it is
            # the run-level answer report.md and provenance.json also give.
            "horizon_window": {
                name: "-".join(_to_str_tuple(window))
                for name, window in horizons.items()
            },
        }
        # The SAME numbers composition.csv reports, keyed per combination, so the
        # two artifacts cannot disagree about one run's reference window. Keyed by
        # the row's FULL identity, horizon included, because the effective horizon
        # window varies with it where the reference window does not.
        row_facts = {
            (p["model"], p["scenario"], p["member"], horizon_name): {
                "reference_window": resolved_facts[p["point_key"]][
                    "reference_window_effective"
                ],
                "horizon_window": horizon_facts[(p["point_key"], horizon_name)],
            }
            for p in points
            for horizon_name in horizons
            if p["point_key"] in resolved_facts
            and (p["point_key"], horizon_name) in horizon_facts
        }
        rows = tidy_rows(
            merged,
            window_facts=window_facts,
            row_facts=row_facts,
            variable_spec=VARIABLE_SPEC,
            combinations=derived,
            available_horizons=available_horizons,
        )
        write_table(
            str(sm.output.change_factors_annual), rows, columns=TABLE_COLUMNS_ANNUAL
        )

        monthly_rows = []
        for month in sorted(int(m) for m in monthly_merged["month"].values):
            monthly_rows.extend(
                tidy_rows(
                    monthly_merged.sel(month=month).drop_vars("month"),
                    month=month,
                    window_facts=window_facts,
                    row_facts=row_facts,
                    variable_spec=VARIABLE_SPEC,
                    combinations=derived,
                    available_horizons=available_horizons,
                )
            )
        write_table(
            str(sm.output.change_factors_monthly),
            monthly_rows,
            columns=TABLE_COLUMNS_MONTHLY,
        )
        log_row(
            f"Monthly change factors: wrote {len(monthly_rows):,} rows "
            f"-> {os.path.basename(str(sm.output.change_factors_monthly))}",
            module="change",
        )
        log_row(
            f"Annual change factors: wrote {len(rows):,} rows "
            f"-> {os.path.basename(str(sm.output.change_factors_annual))}",
            module="change",
        )

        # Written AFTER the merge: a stage-B output describes a completed run
        # (ext2-08). If the merge raises, there is no composition artifact -- which
        # is the contract, not an omission.
        rows = composition_rows(
            sm.params.combinations,
            resolved_facts,
            catalog_crawled_on=sm.params.catalog_crawled_on,
            window_nominal=" / ".join(_to_str_tuple(sm.params.time_horizon_hist)),
            unavailable_windows=unavailable_windows,
        )
        write_composition(str(sm.output.composition_csv), rows)

        # --- step 6a-iii: provenance.json -----------------------------------
        # ASSEMBLED, not derived. Every value below already exists: on a series
        # attribute, in the composition record, or in the reference-window record.
        # Recomputing any of them would create a second definition, and this
        # milestone has watched that go wrong three times -- the calendar recorded
        # twice and disagreeing, n_years as a calendar span beside n_years as a
        # hydrological count, and the effective window reported two ways.
        series_attrs = {}
        for point in points:
            for path in (point["series_path_hist"], point["series_path"]):
                key = os.path.splitext(os.path.basename(path))[0]
                if key not in series_attrs:
                    series_attrs[key] = dict(
                        next(
                            dataset.attrs
                            for (cached_path, _), dataset in series_cache.items()
                            if cached_path == os.path.abspath(str(path))
                        )
                    )
        document = _prov.build(
            unavailable_windows=unavailable_windows,
            clim_project=os.path.basename(clim_project_dir),
            reference_record=sm.params.reference_record,
            variable_spec=sm.params.variable_spec,
            composition_rows=rows,
            series_attrs=series_attrs,
            # The SAME per-combination windows composition.csv and the
            # change-factor tables use, keyed by the SCENARIO series.
            effective_windows={
                p["series_key"]: {
                    "effective": resolved_facts[p["point_key"]][
                        "reference_window_effective"
                    ],
                    "n_years": resolved_facts[p["point_key"]]["n_hyd_years_reference"],
                }
                for p in points
                if p["point_key"] in resolved_facts
            },
            catalog_crawled_on=sm.params.catalog_crawled_on,
            reducer_module_hash=next(
                (a.get("cst_reducer_module_hash", "") for a in series_attrs.values()),
                "",
            ),
            effective_config_sha256=sm.params.effective_config_sha256,
            configuration_inputs_sha256=getattr(
                sm.params, "configuration_inputs_sha256", None
            ),
            region_fingerprint=region_fp,
            horizons={k: " / ".join(_to_str_tuple(v)) for k, v in horizons.items()},
            weighting_scheme=next(
                (a.get("cst_weighting_scheme", "") for a in series_attrs.values()), ""
            ),
        )
        # Step 6b: counted from the rows the monthly table wrote, not by a second
        # traversal -- a value recorded twice has disagreed four times in this
        # milestone, and this is the fifth chance.
        document["flagged_months"] = _flagged_months(
            monthly_rows, rows, sm.params.max_flagged_months
        )
        _prov.write(str(sm.output.provenance_json), document)

        # --- step 7-ii: report.md ---------------------------------------------
        # READS the provenance document just written; recomputes nothing. A value
        # recorded in two places has disagreed five times in this milestone, and a
        # report deriving its own disclaimer would be the sixth chance.
        _report.write(
            str(sm.output.report_md),
            _report.build(
                document,
                thresholds=sm.params.min_reference,
                max_flagged_months=sm.params.max_flagged_months,
            ),
        )
        log_row(
            f"Report → {os.path.basename(str(sm.output.report_md))}",
            module="change",
        )
        log_row(
            f"{len(document['sources'])} source series recorded → {os.path.basename(str(sm.output.provenance_json))}",
            module="change",
        )
        n_resolved = sum(1 for r in rows if r["status"] == "resolved")
        log_row(
            f"Ensemble coverage: {n_resolved}/{len(rows)} combinations fully resolved; statuses → {os.path.basename(str(sm.output.composition_csv))}",
            module="change",
        )
