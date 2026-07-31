"""Stage B — all change factors in ONE job (migration step 4d, design §8).

Replaces the pair `monthly_change` (fanned out per `point_key` × horizon) and
`monthly_change_scalar_merge` (a single aggregator over their `temp()` outputs).
The design's rule table gives stage B **1 job** with no fan-out
(`wf2-climate-analysis-v2-design.md` §5, "B. Derive"), reading the explicit
expanded series list.

**This step is value-neutral by construction.** The arithmetic is not reimplemented
here: `get_change_annual_clim_proj`, `get_change_clim_projections` and
`summary_climate_proj` are imported from the two modules that already held them,
unchanged. Only the orchestration moves. A non-zero characterized diff on the
summary artifacts is therefore a defect in this file, not a judgement call — which
is the whole reason the functions were left where they were.

What changes shape:

* the per-point `annual_change_scalar_stats-{point_key}_{horizon}.nc` files were
  Snakemake `temp()` outputs and are now **job-internal intermediates** with the
  same lifetime — written, consumed by the merge, removed. `summary_climate_proj`
  reads model/scenario/horizon from dataset *coords*, never from the filename, so
  relocating them is safe (checked before the move, not assumed);
* one log and one benchmark instead of a per-part tree under `2.04_monthly_change/`.

Stage B's input set is explicit (design risk-06 / revision 4): the rule declares
exactly the expanded `{series_key}` list built from the resolved combination set,
and this job **asserts that the set it opened equals that list**. A model removed
from the config cannot rejoin the run through a leftover file in `scalar/`.

Invoked from ``Snakefile_climate_projections`` via ``script:``; reads
``snakemake.input/output/params``, never ``sys.argv``.
"""
# NOTE: no `from __future__ import annotations` here — Snakemake's `script:`
# directive prepends its own preamble to a copy of this file, so a __future__
# import lands mid-file and raises SyntaxError at job start. A --dry-run cannot
# catch it (it never executes a script body); the other `script:` modules in this
# repo omit it for the same reason.

import csv
import os
import tempfile

import xarray as xr

from blueearth_cst.projections import series_identity
from blueearth_cst.projections.calendar_weights import CalendarError, assert_weightable
from blueearth_cst.projections.change_factor_table import tidy_rows, write_table
from blueearth_cst.projections.dry_month import FLAGGED_STATUS, combination_is_flagged
from blueearth_cst.projections import provenance as _prov
from blueearth_cst.projections import report as _report
from blueearth_cst.projections.variable_spec import VariableSpec
from blueearth_cst.projections.get_change_climate_proj import (
    _to_str_tuple,
    get_change_annual_clim_proj,
    get_change_monthly_clim_proj,
    get_change_clim_projections,
    hydrological_year_bounds,
)
from blueearth_cst.projections.get_change_climate_proj_summary import (
    summary_climate_proj,
)
from blueearth_cst.shared.snake_utils import log_row, tee_to_log

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")


# `_to_str_tuple` is IMPORTED, not reimplemented, despite the leading underscore.
# A local copy was written first and was already wrong: it raised on `[]`, where
# the original returns `()` — a contract `tests/test_get_change_climate_proj.py`
# pins. Reimplementing a normaliser is exactly the drift this step is meant to
# avoid, so the private name is the lesser evil.


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
    save_grids=False,
    stats=None,
    variable_spec=None,
    min_reference=None,
    stats_path_hist=None,
    stats_path=None,
    clim_project_dir=None,
):
    """Change factors for one (model, scenario, member) at one horizon.

    The body is the former ``monthly_change`` job, moved verbatim apart from
    taking its inputs as arguments instead of reading ``snakemake.params``.

    Returns the **effective reference window** it used — ``(start, end, n_years)``
    from :func:`hydrological_year_bounds`, the same call the change arithmetic
    makes — so the composition record annotates the numbers with the window that
    produced them rather than with a recomputed guess.
    """
    # --- step 2b backstop: the series must match the current inputs -----------
    # Design D9 route (b) / risk-03 mechanism 2. An assertion INSIDE the job, not
    # a scheduling property, so it holds however Snakemake was invoked -- a series
    # restored from a backup, produced by an older checkout, or surviving a
    # non-default --rerun-triggers still fails the run instead of quietly entering
    # the change factors.
    for label, path, components in (
        ("historical", series_path_hist, dict(digest_components_hist)),
        (name_scenario, series_path, dict(digest_components_fut)),
    ):
        series_identity.assert_series_identity(
            path,
            series_identity.series_digest(components, region_fp),
            f"{name_model} {label}",
        )

    ds_hist_time = xr.open_dataset(series_path_hist)
    ds_clim_time = xr.open_dataset(series_path)

    if save_grids:
        ds_hist = xr.open_dataset(stats_path_hist)
        ds_clim = xr.open_dataset(stats_path)

    # Step 4c: the `if len(ds_clim_time) > 0` guard and its dummy-netCDF
    # else-branch are gone. Since 4a an unresolved combination never becomes a
    # job, so an empty series here means a real defect.
    if len(ds_clim_time) == 0:
        raise RuntimeError(
            f"{series_path} holds no data variables. Resolution admitted this "
            "combination, so an empty series is a defect rather than an "
            "unpublished source -- delete the series and re-run to re-derive."
        )

    ds_hist_time = ds_hist_time.sel(time=slice(*time_tuple_hist))
    ds_clim_time = ds_clim_time.sel(time=slice(*time_tuple_fut))
    # Read the effective reference window off the SAME helper the change
    # arithmetic uses, after the same slice. Not a recomputation: one function,
    # called twice on one dataset.
    #
    # Deliberately called with the DEFAULT start month, matching the line below:
    # `get_change_annual_clim_proj` is invoked without `start_month_hyd_year`, so
    # the arithmetic always uses "Jan" regardless of the config key. That looks
    # like a pre-existing defect (the rule reads the key and the old job never
    # forwarded it), but 4d is value-neutral, so this reports the window actually
    # used rather than the one the config asks for. Forwarding the key would change
    # results for any non-Jan config and belongs in its own commit with its own
    # gate. Recorded in the composition record's own terms: `reference_window_
    # nominal` is what was requested, `_effective` is what was used.
    ref_start, ref_end, ref_n_years = hydrological_year_bounds(ds_hist_time)
    # Step 5b: weight each month by its length in the MODEL's calendar. Read off
    # the series (propagated there from the raw slice, which got it from the store
    # -- the axis itself cannot say, having been converted to datetime64 upstream).
    # Both series must agree: a change factor differencing two calendars would be
    # comparing incomparable annual aggregates.
    calendar = str(ds_hist_time.attrs.get("cst_calendar", "") or "")
    clim_calendar = str(ds_clim_time.attrs.get("cst_calendar", "") or "")
    if calendar != clim_calendar:
        raise CalendarError(
            f"{name_model} {name_scenario}: reference and scenario series carry "
            f"different calendars ({calendar!r} vs {clim_calendar!r}). Their annual "
            "aggregates are not comparable."
        )
    assert_weightable(calendar, source=f"{name_model} {name_scenario}")

    stats_annual_change = get_change_annual_clim_proj(
        ds_hist_time,
        ds_clim_time,
        calendar=calendar,
        stats=stats,
        variable_spec=variable_spec,
    )
    stats_annual_change = stats_annual_change.assign_coords(
        {"horizon": f"{name_horizon}"}
    ).expand_dims(["horizon"])
    stats_annual_change = stats_annual_change.transpose(
        ..., "clim_project", "model", "scenario", "horizon", "member"
    )

    dvars = stats_annual_change.raster.vars
    stats_annual_change.to_netcdf(
        change_nc_out, encoding={k: {"zlib": True} for k in dvars}
    )

    # Step 6a-ii: the same combination's change per CALENDAR MONTH. Written beside
    # the annual file rather than returned, so the merge step handles both the
    # same way and a failure leaves neither half-written.
    monthly_change = get_change_monthly_clim_proj(
        ds_hist_time, ds_clim_time, stats=stats, variable_spec=variable_spec,
        min_reference=min_reference,
    )
    monthly_change = monthly_change.assign_coords(
        {"horizon": f"{name_horizon}"}
    ).expand_dims(["horizon"])
    monthly_change.to_netcdf(
        str(change_nc_out).replace(".nc", "_monthly.nc"),
        encoding={k: {"zlib": True} for k in monthly_change.raster.vars},
    )

    if save_grids:
        # Cold branch: shipped configs set `save_grids: false`, and step 5e is
        # where it is restructured (`save_grids` -> `save_gridded`, OQ-12). Moved
        # as-is rather than tidied, so 4d stays value-neutral.
        if len(ds_clim) > 0:
            monthly_change_mean_grid = get_change_clim_projections(ds_hist, ds_clim)
            monthly_change_mean_grid = monthly_change_mean_grid.assign_coords(
                {"horizon": f"{name_horizon}"}
            ).expand_dims(["horizon"])
            log_row("writing netcdf files monthly_change_mean_grid", module="change")
            dvars = monthly_change_mean_grid.raster.vars
            grid_model = monthly_change_mean_grid.model.values[0]
            grid_scenario = monthly_change_mean_grid.scenario.values[0]
            grid_horizon = monthly_change_mean_grid.horizon.values[0]
            # Step 7-iii / D11: `grids/change/{series_key}_{horizon}.nc` -- the
            # cellwise counterpart of the tabular product, addressable by the same
            # key. The legacy `monthly_change_mean_grid-{model}_{scenario}_{horizon}`
            # embedded an unsanitized model name and matched nothing else.
            series_key_for_grid = os.path.splitext(os.path.basename(series_path))[0]
            name_nc_out = os.path.join(
                "grids", "change", f"{series_key_for_grid}_{name_horizon}.nc"
            )
            os.makedirs(
                os.path.join(clim_project_dir, "grids", "change"), exist_ok=True
            )
            monthly_change_mean_grid.to_netcdf(
                os.path.join(clim_project_dir, name_nc_out),
                encoding={k: {"zlib": True} for k in dvars},
            )
        else:
            raise RuntimeError(
                f"{name_model} {name_scenario}: save_gridded is on but the gridded "
                "scenario dataset is empty. Step 4c removed the dummy-netCDF path "
                "from the scalar branch for the same reason -- a placeholder that "
                "looks like a product is worse than a failure."
            )

    ds_hist_time.close()
    ds_clim_time.close()
    return ref_start, ref_end, ref_n_years


#: Columns of ``composition.csv``, in design §5.7 order. One row per REQUESTED
#: (model, scenario, member) — not per resolved one, which is the point: the skips
#: are what make the record auditable.
COMPOSITION_COLUMNS = [
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


def composition_rows(combinations, resolved, *, catalog_crawled_on, window_nominal):
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
        row = dict.fromkeys(COMPOSITION_COLUMNS, "")
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
        rows.append(row)
    return rows


def write_composition(path, rows):
    """Write ``composition.csv``. Stage-B output: it describes a COMPLETED run.

    ext2-08 / D4: the record is written here and not at DAG build, because a DAG
    build that writes an output file makes parsing side-effecting — a dry run that
    writes is not a dry run. A failed run therefore leaves the DAG-build stderr
    summary and the job logs, and no composition artifact.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPOSITION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if "snakemake" in globals():
    sm = globals()["snakemake"]

    with tee_to_log(sm.log[0]):
        clim_project_dir = sm.params.clim_project_dir
        horizons = sm.params.horizons
        save_grids = sm.params.save_grids
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
            f"deriving change factors for {len(points)} point(s) x "
            f"{len(horizons)} horizon(s)",
            module="change",
        )
        # Step 5e / D1: the durable reference-window record. Its designated homes
        # -- provenance.json (6a) and report.md (7) -- do not exist yet, so it
        # lands in this log and 6a relocates it. Logged as one line per fact so a
        # later reader can grep a single condition rather than parse a blob.
        for _key, _value in sorted(dict(sm.params.reference_record).items()):
            log_row(f"reference_window {_key}={_value}", module="change")

        # The per-point files were `temp()` rule outputs; they are job-internal
        # now, with the same lifetime. TemporaryDirectory removes them even if the
        # merge raises, which the old temp() could not promise mid-DAG.
        # Snakemake params carry plain data; rebuild the typed spec here so the
        # aggregation looks up fields by name rather than by list position.
        VARIABLE_SPEC = {
            name: VariableSpec(*fields)
            for name, fields in dict(sm.params.variable_spec).items()
        }
        resolved_facts = {}
        with tempfile.TemporaryDirectory(prefix="cst_change_") as work_dir:
            change_files = []
            monthly_files = []
            for point in points:
                for horizon_name, horizon_window in horizons.items():
                    out_nc = os.path.join(
                        work_dir,
                        f"annual_change_scalar_stats-{point['point_key']}"
                        f"_{horizon_name}.nc",
                    )
                    ref_start, ref_end, ref_n_years = derive_one_point(
                        series_path_hist=point["series_path_hist"],
                        series_path=point["series_path"],
                        change_nc_out=out_nc,
                        time_tuple_hist=_to_str_tuple(sm.params.time_horizon_hist),
                        time_tuple_fut=_to_str_tuple(horizon_window),
                        name_horizon=horizon_name,
                        name_model=point["model"],
                        name_scenario=point["scenario"],
                        region_fp=region_fp,
                        digest_components_hist=point["digest_components_hist"],
                        digest_components_fut=point["digest_components_fut"],
                        save_grids=save_grids,
                        stats=sm.params.stats,
                        variable_spec=VARIABLE_SPEC,
                        min_reference=sm.params.min_reference,
                        stats_path_hist=point.get("stats_path_hist"),
                        stats_path=point.get("stats_path"),
                        clim_project_dir=clim_project_dir,
                    )
                    change_files.append(out_nc)
                    monthly_files.append(out_nc.replace(".nc", "_monthly.nc"))
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

            log_row(
                f"merging {len(change_files)} change file(s) into the summary",
                module="change",
            )
            summary_climate_proj(
                clim_dir=clim_project_dir,
                clim_files=change_files,
                horizons=horizons,
            )

            # Step 6a-ii: merge the per-point MONTHLY files. Done inside the temp
            # directory, because that is where they live -- reading them after the
            # context exits would be reading deleted files. Eager and closed, for
            # the reason bf1f4a5 and e592ec3 both landed on: a lazy multi-file read
            # feeding a write parks dask's pool on the HDF5 lock, and open handles
            # stop the directory being removed.
            with xr.open_mfdataset(
                monthly_files, coords="minimal", combine="by_coords"
            ) as _lazy:
                monthly_merged = _lazy.load()

        # --- step 6a-i: the tidy annual change-factor table (design §5.9) ----
        # Read back from the summary the merge just wrote, rather than threading
        # the in-memory dataset out of summary_climate_proj: the table must
        # describe what was PERSISTED, so a reshape can never disagree with the
        # artifact it claims to reshape.
        summary_nc = os.path.join(
            clim_project_dir, "summary", "annual_change_scalar_stats_summary.nc"
        )
        with xr.open_dataset(summary_nc) as _merged:
            merged = _merged.load()
        window_facts = {
            "reference_window_nominal": sm.params.reference_record.get(
                "reference_window_requested", ""
            ),
            "reference_window_effective": sm.params.reference_record.get(
                "reference_window_effective", ""
            ),
            "n_years": sm.params.reference_record.get("reference_window_years", ""),
            # Per-end dropped months are known per SERIES, not per run; until
            # provenance.json (6a-iii) carries them per source, the run-level
            # figure is left empty rather than filled with a guess.
            "n_years_dropped": "",
            "horizon_window_nominal": {
                name: " / ".join(_to_str_tuple(window))
                for name, window in horizons.items()
            },
            "horizon_window_effective": {},
            "units": {
                name: fields[3] for name, fields in dict(sm.params.variable_spec).items()
            },
        }
        series_keys = {
            (p["model"], p["scenario"], p["member"]): p["reference_series_key"]
            for p in points
        }
        # The SAME numbers composition.csv reports, keyed per combination, so the
        # two artifacts cannot disagree about one run's reference window.
        row_facts = {
            (p["model"], p["scenario"], p["member"]): {
                "reference_window_effective": resolved_facts[p["point_key"]][
                    "reference_window_effective"
                ],
                "n_years": resolved_facts[p["point_key"]]["n_hyd_years_reference"],
            }
            for p in points
            if p["point_key"] in resolved_facts
        }
        rows = tidy_rows(merged, period="annual", window_facts=window_facts,
                         series_keys=series_keys, row_facts=row_facts)
        write_table(str(sm.output.change_factors_annual), rows)

        monthly_rows = []
        for month in sorted(int(m) for m in monthly_merged["month"].values):
            monthly_rows.extend(
                tidy_rows(
                    monthly_merged.sel(month=month).drop_vars("month"),
                    period=str(month),
                    window_facts=window_facts,
                    series_keys=series_keys,
                    row_facts=row_facts,
                )
            )
        write_table(str(sm.output.change_factors_monthly), monthly_rows)
        log_row(
            f"tidy monthly change-factor table: {len(monthly_rows)} rows "
            f"-> {os.path.basename(str(sm.output.change_factors_monthly))}",
            module="change",
        )
        log_row(
            f"tidy annual change-factor table: {len(rows)} rows "
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
                    with xr.open_dataset(path) as _s:
                        series_attrs[key] = dict(_s.attrs)
        document = _prov.build(
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
                    "n_years": resolved_facts[p["point_key"]][
                        "n_hyd_years_reference"
                    ],
                }
                for p in points
                if p["point_key"] in resolved_facts
            },
            catalog_crawled_on=sm.params.catalog_crawled_on,
            reducer_module_hash=next(
                (a.get("cst_reducer_module_hash", "") for a in series_attrs.values()), ""
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
        flagged_counts = {}
        for row in monthly_rows:
            if row["status"] == FLAGGED_STATUS:
                key = (row["dataset"], row["scenario"], row["member"],
                       row["horizon"], row["variable"])
                flagged_counts[key] = flagged_counts.get(key, 0) + 1
        document["flagged_months"] = [
            {
                "dataset": k[0], "scenario": k[1], "member": k[2],
                "horizon": k[3], "variable": k[4], "n_flagged_months": n,
                "exceeds_max": combination_is_flagged(n, sm.params.max_flagged_months),
            }
            for k, n in sorted(flagged_counts.items())
        ]
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
                figures=list(sm.params.figure_names),
            ),
        )
        log_row(f"report -> {os.path.basename(str(sm.output.report_md))}", module="change")
        log_row(
            f"provenance: {len(document['sources'])} sources, "
            f"{document['composition']['resolved']}/{document['composition']['requested']} resolved "
            f"-> {os.path.basename(str(sm.output.provenance_json))}",
            module="change",
        )
        n_resolved = sum(1 for r in rows if r["status"] == "resolved")
        log_row(
            f"composition record: {len(rows)} requested, {n_resolved} resolved "
            f"-> {os.path.basename(str(sm.output.composition_csv))}",
            module="change",
        )
