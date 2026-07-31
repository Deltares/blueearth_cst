"""Tidy change-factor tables (design §5.9, step 6a; reshaped at S8-04).

The summary CSV stage B produced until 6a was *wide*: one row per
`(stats, clim_project, model, scenario, horizon, member)` with each variable as a
**column**. §5.9 replaced it with long format — one row per
`(model, scenario, member, horizon, [month,] variable, statistic)`.

S8-04 rebuilt the column set. Three things were wrong with the first cut:

* **`units` was wrong for every relative variable.** It was populated from the
  variable spec's declared units — the units of the *underlying variable* — while
  `value` held `(clim - hist)/hist*100`. So a precipitation row labelled a
  **percent** as `mm/day`. Right for absolute variables, wrong for relative ones:
  the same name-vs-semantics split 5e exists to end, surviving one layer up.
* **Three columns were structurally dead.** `horizon_window_effective` and
  `n_years_dropped` were hardcoded empty, and `absolute_value` was never populated
  in the annual table because the companion only existed on the monthly path.
* **Eight columns were constant or redundant.** `dataset` was literally
  `institution + "/" + source_id`, split apart again on read.

The schema now carries three values per row, each with fixed semantics:

* ``reference_value`` — the **baseline level** (25.0567), in ``units``.
* ``absolute_value`` — the **future level** (26.2354), in ``units``.
* ``relative_value`` — **relative to the baseline**, per the spec's ``change``
  field: a difference for an `absolute` variable (+1.1787 degC), a percent for a
  `relative` one (+10.95). ``relative_units`` says which.

Nothing is inferred from a variable name, and there is no column whose meaning
depends on which row you are reading.

The dry-month rule (6b) falls out natively: a flagged month has ``relative_value``
empty, both levels present, and ``status = reference_below_threshold``. Shipping
the reference is what keeps 6b's promise exactly — the rule drops the meaningless
ratio and keeps the informative difference, and the difference is
``absolute_value - reference_value`` on every row, flagged or not.

This module is a **reshape, not a recomputation** — every value it emits comes
from the dataset stage B persisted (falsifier M3).
"""
from __future__ import annotations

from blueearth_cst.projections.dry_month import FLAGGED_STATUS

#: Identity columns, shared by both tables. `month` is inserted after `horizon`
#: for the monthly one.
_IDENTITY = ["model", "scenario", "member", "horizon"]
_TAIL = [
    "variable",
    "statistic",
    "reference_value",
    "absolute_value",
    "units",
    "relative_value",
    "relative_units",
    "status",
    "reference_window",
    "horizon_window",
]

#: Column order of `summary/{clim_project}_change_factors_annual.csv`.
TABLE_COLUMNS_ANNUAL = _IDENTITY + _TAIL
#: Column order of `summary/{clim_project}_change_factors_monthly.csv`.
TABLE_COLUMNS_MONTHLY = _IDENTITY + ["month"] + _TAIL

#: Coordinates that are not part of the key and must not become columns.
#: `spatial_ref` is a CRS artifact, not a change factor (falsifier M2).
DROPPED_COORDS = ("spatial_ref",)

#: Companions ride alongside the variable they qualify and are COLUMNS of that
#: variable, never rows of their own:
#:   `{var}__reference` the baseline level     -> `reference_value`
#:   `{var}__level`     the future level       -> `absolute_value`
#:   `{var}__flagged`   the dry-month verdict  -> `status`
COMPANION_SEP = "__"

#: Units of `relative_value` when the variable's change is a ratio.
PERCENT = "%"


def _sort_key(columns):
    """Identity columns only — the ones that make a row unique."""
    n = len(_IDENTITY) + (1 if "month" in columns else 0) + 2  # + variable, statistic
    return columns[:n]


def tidy_rows(ds, *, month=None, window_facts=None, row_facts=None, variable_spec=None):
    """Long-format rows from the wide change-factor dataset stage B produces.

    ``ds`` carries data variables per climate variable (`precip`, `temp`) over
    coordinates `(clim_project, model, scenario, horizon, member, stats)`, plus
    the `__reference`, `__level` and `__flagged` companions.

    ``month`` is ``None`` for the annual table and the month number for the
    monthly one. The two tables no longer share a schema — the `period` column
    that made them concatenable was constant in one and the key in the other, and
    no consumer stacked them.

    ``window_facts`` is a single dict of run-level window provenance, passed
    rather than recomputed so the table cannot disagree with `composition.csv` or
    `provenance.json` about the same run.

    ``row_facts`` maps `(model, scenario, member)` to per-combination overrides.
    It exists because the reference window's **effective** bounds are a property
    of a series, not of the run: they come from ``hydrological_year_bounds``,
    applied to the data each combination actually has.

    ``variable_spec`` maps variable name to a spec exposing ``.units`` and
    ``.change``. It is what decides whether ``relative_value`` is a percent or a
    difference — never the variable's name.
    """
    columns = TABLE_COLUMNS_ANNUAL if month is None else TABLE_COLUMNS_MONTHLY
    facts = dict(window_facts or {})
    per_row = dict(row_facts or {})
    spec = dict(variable_spec or {})
    rows = []

    base_variables = sorted(
        v
        for v in ds.data_vars
        if v not in DROPPED_COORDS and COMPANION_SEP not in str(v)
    )
    for variable in base_variables:
        da = ds[variable]
        level_da = ds.get(f"{variable}{COMPANION_SEP}level")
        reference_da = ds.get(f"{variable}{COMPANION_SEP}reference")
        flagged_da = ds.get(f"{variable}{COMPANION_SEP}flagged")
        stacked = da.stack(_row=[d for d in da.dims])
        if level_da is not None:
            level_da = level_da.stack(_row=[d for d in level_da.dims])
        if reference_da is not None:
            reference_da = reference_da.stack(_row=[d for d in reference_da.dims])
        if flagged_da is not None:
            flagged_da = flagged_da.stack(_row=[d for d in flagged_da.dims])

        entry = spec.get(variable)
        units = getattr(entry, "units", "") if entry is not None else ""
        change_kind = getattr(entry, "change", "absolute") if entry is not None else "absolute"
        relative_units = PERCENT if change_kind == "relative" else units

        for idx in range(stacked.sizes["_row"]):
            point = stacked.isel(_row=idx)
            coords = {k: _scalar(point[k].values) for k in point.coords if k != "_row"}
            # `model` is stored as `institution/source_id`; the source_id alone is
            # globally unique in the CMIP6 controlled vocabulary, so the
            # institution is recoverable and was pure duplication.
            dataset = str(coords.get("model", ""))
            _, _, source_id = dataset.partition("/")
            scenario = str(coords.get("scenario", ""))
            member = str(coords.get("member", ""))
            horizon = str(coords.get("horizon", ""))

            row = dict.fromkeys(columns, "")
            row.update(
                model=source_id or dataset,
                scenario=scenario,
                member=member,
                horizon=horizon,
                variable=variable,
                statistic=str(coords.get("stats", "")),
                relative_value=_scalar(point.values),
                units=units,
                relative_units=relative_units,
                # 6b's verdict; overwritten below when the companion says so.
                status="ok",
                reference_window=facts.get("reference_window", ""),
                horizon_window=facts.get("horizon_window", {}).get(horizon, ""),
            )
            if month is not None:
                row["month"] = month
            # The effective reference window is per-series, not per-run.
            overrides = per_row.get((dataset, scenario, member), {})
            if "reference_window" in overrides:
                row["reference_window"] = overrides["reference_window"]
            if level_da is not None:
                row["absolute_value"] = _scalar(level_da.isel(_row=idx).values)
            if reference_da is not None:
                row["reference_value"] = _scalar(reference_da.isel(_row=idx).values)
            # Step 6b: a flagged month lost its ratio. Read from the companion
            # rather than recomputed, so the table cannot disagree with the
            # computation about which months were flagged.
            if flagged_da is not None and bool(_scalar(flagged_da.isel(_row=idx).values)):
                row["status"] = FLAGGED_STATUS
            rows.append(row)

    # Deterministic order: the CSV is fingerprinted by sha256, so an unstable row
    # order would make the artifact unreproducible for no reason.
    key_columns = _sort_key(columns)
    rows.sort(key=lambda r: tuple(str(r[c]) for c in key_columns))
    return rows


def _scalar(value):
    """Unwrap a 0-d numpy value without turning a float into ``array(1.0)``."""
    try:
        return value.item()
    except AttributeError:
        return value


#: Decimal places every float carries into a WF2 CSV.
#:
#: Three is 100x finer than the default `relative_change.min_reference`
#: (0.1 mm/day), so the rounding cannot interact with the dry-month threshold: a
#: reference small enough to be quantised here was already flagged and its ratio
#: already `NaN`. Revisit only if a user declares a variable whose units make its
#: values small in absolute terms — the threshold is per-variable, this is not.
CSV_DECIMALS = 3


def csv_value(value):
    """Render one cell for a WF2 CSV: floats fixed to `CSV_DECIMALS`, rest as-is.

    Two reasons, one change. Excel prompts "By default, Excel will perform the
    following data conversion" on these files because a full float64 repr runs to
    17 significant digits (108 such cells in the annual table, 1296 in the
    monthly); fixed-point output is the only conversion trigger these files carry,
    and it removes it. And fixed-point never emits an exponent, so a near-zero
    ratio cannot arrive as `1.2345e-06` — the form Excel converts most eagerly.

    This is a SERIALIZATION concern only. The rows stay exact in memory and the
    `scalar/*.nc` series keep full precision; R8 step 5c deliberately removed
    quantisation from the STORED series because it fed downstream arithmetic, and
    nothing downstream reads these CSVs (`plot_climate_proj_timeseries` declares
    the annual table as an ordering edge and never opens it).

    Non-floats pass through untouched, which is what keeps the integer columns
    integral: `n_reference_years` stays `21`, not `21.000`, and the monthly
    table's `month` stays `1`. `NaN` also passes through, rendering `nan` exactly
    as before — whether a flagged ratio should instead be an empty cell is a
    question about missing-vs-undefined, not about number formatting.
    """
    if not isinstance(value, float) or value != value:  # non-float, or NaN
        return value
    text = f"{value:.{CSV_DECIMALS}f}"
    # -0.000: a negative value rounded away still carries its sign, which reads
    # as a real (tiny, negative) change rather than as zero.
    if not text.lstrip("-0."):
        return text.lstrip("-")
    return text


def write_table(path, rows, *, columns):
    """Write one tidy table. Header always present, even with zero rows."""
    import csv
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {name: csv_value(value) for name, value in row.items()} for row in rows
        )
