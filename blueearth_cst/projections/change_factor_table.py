"""Tidy change-factor tables (design §5.9, step 6a).

The summary CSV stage B has produced until now is *wide*: one row per
`(stats, clim_project, model, scenario, horizon, member)` with each variable as a
**column**, plus a `spatial_ref` coordinate that is present only because
``to_dataframe()`` dumps every coordinate it finds. That shape makes the obvious
questions awkward — "every value for precip" is a column selection in one file and
a row selection in another — and it cannot carry per-row provenance, because a row
covers two variables at once.

§5.9 specifies long format: **one row per**
``(dataset, institution, scenario, member, horizon, period, variable, statistic)``
carrying `value`, `absolute_value`, `units`, `status`, both reference and horizon
windows in nominal and effective form, `n_years`, `n_years_dropped` and
`reference_series_key`.

This module is a **reshape, not a recomputation**. Every value it emits must equal
the corresponding cell of the wide table — falsifier M3, checked by joining the
two rather than by comparing row counts, because a reshape that drops or
duplicates rows still produces a plausible file.

`n_models_in_summary`, which design-v2 proposed and §8 asks to omit, was never
implemented — there is nothing to remove.
"""
from __future__ import annotations

from blueearth_cst.projections.dry_month import FLAGGED_STATUS

#: Column order of `change_factors/{annual,monthly}.csv`, per §5.9.
#: `period` is `annual` for the annual table and the month number for the
#: monthly one, so both tables share one schema and can be concatenated.
TABLE_COLUMNS = [
    "dataset",
    "institution",
    "source_id",
    "scenario",
    "member",
    "horizon",
    "period",
    "variable",
    "statistic",
    "value",
    "absolute_value",
    "units",
    "status",
    "reference_window_nominal",
    "reference_window_effective",
    "horizon_window_nominal",
    "horizon_window_effective",
    "n_years",
    "n_years_dropped",
    "reference_series_key",
]

#: Coordinates that are not part of the key and must not become columns.
#: `spatial_ref` is a CRS artifact, not a change factor (falsifier M2).
DROPPED_COORDS = ("spatial_ref",)

#: Step 6b companions: `precip__absolute` and `precip__flagged` ride alongside
#: `precip` so the dry-month verdict survives the per-point netCDF round trip.
#: They are COLUMNS of the variable they qualify, never rows of their own.
COMPANION_SEP = "__"


def tidy_rows(ds, *, period="annual", window_facts=None, series_keys=None, row_facts=None):
    """Long-format rows from the wide change-factor dataset stage B produces.

    ``ds`` carries data variables per climate variable (`precip`, `temp`) over
    coordinates `(clim_project, model, scenario, horizon, member, stats)`.

    ``window_facts`` maps nothing at all today — it is a single dict of run-level
    window provenance, because every row of one run shares one reference window.
    It is passed rather than recomputed so the table cannot disagree with
    `composition.csv` or `provenance.json` about the same run.

    ``series_keys`` maps `(model, scenario, member)` to the reference series key,
    for the `reference_series_key` column.

    ``row_facts`` maps the same key to per-combination overrides. It exists
    because the reference window's **effective** bounds and its complete-
    hydrological-year count are properties of a series, not of the run: they come
    from ``hydrological_year_bounds``, applied to the data each combination
    actually has. Taking them from a run-level calendar span instead produced a
    table reporting `n_years = 20` beside a `composition.csv` reporting 21 — two
    definitions of one quantity, which is the drift 4d's single-definition
    extraction was meant to end. Passing the values the composition record already
    used makes the two tables agree by construction rather than by review.
    """
    facts = dict(window_facts or {})
    keys = dict(series_keys or {})
    per_row = dict(row_facts or {})
    rows = []

    base_variables = sorted(
        v
        for v in ds.data_vars
        if v not in DROPPED_COORDS and COMPANION_SEP not in str(v)
    )
    for variable in base_variables:
        da = ds[variable]
        absolute_da = ds.get(f"{variable}{COMPANION_SEP}absolute")
        flagged_da = ds.get(f"{variable}{COMPANION_SEP}flagged")
        stacked = da.stack(_row=[d for d in da.dims])
        if absolute_da is not None:
            absolute_da = absolute_da.stack(_row=[d for d in absolute_da.dims])
        if flagged_da is not None:
            flagged_da = flagged_da.stack(_row=[d for d in flagged_da.dims])
        for idx in range(stacked.sizes["_row"]):
            point = stacked.isel(_row=idx)
            coords = {k: _scalar(point[k].values) for k in point.coords if k != "_row"}
            dataset = str(coords.get("model", ""))
            institution, _, source_id = dataset.partition("/")
            scenario = str(coords.get("scenario", ""))
            member = str(coords.get("member", ""))
            row = dict.fromkeys(TABLE_COLUMNS, "")
            row.update(
                dataset=dataset,
                institution=institution,
                source_id=source_id or dataset,
                scenario=scenario,
                member=member,
                horizon=str(coords.get("horizon", "")),
                period=period,
                variable=variable,
                statistic=str(coords.get("stats", "")),
                value=_scalar(point.values),
                units=facts.get("units", {}).get(variable, ""),
                # 6b replaces this with the dry-month rule's verdict. The column
                # exists from the start so that rule has somewhere to land rather
                # than 6a emitting a bare number beside an infinity later.
                status="ok",
                reference_window_nominal=facts.get("reference_window_nominal", ""),
                reference_window_effective=facts.get("reference_window_effective", ""),
                horizon_window_nominal=facts.get("horizon_window_nominal", {}).get(
                    str(coords.get("horizon", "")), ""
                ),
                horizon_window_effective=facts.get("horizon_window_effective", {}).get(
                    str(coords.get("horizon", "")), ""
                ),
                n_years=facts.get("n_years", ""),
                n_years_dropped=facts.get("n_years_dropped", ""),
                reference_series_key=keys.get((dataset, scenario, member), ""),
            )
            row.update(per_row.get((dataset, scenario, member), {}))
            # Step 6b: a flagged month lost its ratio and kept its difference.
            # Both are read from the companions rather than recomputed, so the
            # table cannot disagree with the computation about which months were
            # flagged -- the drift that has now bitten four times.
            if flagged_da is not None:
                flagged = bool(_scalar(flagged_da.isel(_row=idx).values))
                if flagged:
                    row["status"] = FLAGGED_STATUS
            if absolute_da is not None:
                row["absolute_value"] = _scalar(absolute_da.isel(_row=idx).values)
            rows.append(row)

    # Deterministic order: the CSV is fingerprinted by sha256, so an unstable row
    # order would make the artifact unreproducible for no reason (the same
    # failure `intersection`'s sorted() was introduced to fix).
    rows.sort(key=lambda r: tuple(str(r[c]) for c in TABLE_COLUMNS[:9]))
    return rows


def _scalar(value):
    """Unwrap a 0-d numpy value without turning a float into ``array(1.0)``."""
    try:
        return value.item()
    except AttributeError:
        return value


def write_table(path, rows):
    """Write one tidy table. Header always present, even with zero rows."""
    import csv
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
