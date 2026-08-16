# -*- coding: utf-8 -*-
"""Reduce the stress-test runs into one LONG-format indicator table per variable.

R11 CR-2. The pre-R11 writer produced two WIDE tables — ``q_indicators.csv`` with
one column per gauge, and ``basin_indicators.csv`` with one column per basin
variable — so the header grew with the gauge count and a reader had to know which
columns were locations. This emits **one table per output variable**, each with a
**fixed five-column header** that does not grow with anything:

    metric, location, st_id, rlz_id, value

The header no longer grows with the STRESS-dimension count either. The axis
columns were removed once the response-surface axis became a derivation over
``stress_test_lookup.csv``: they held an annual collapse of twelve monthly
perturbations, which misreports any seasonal design and made every other axis
unrecoverable from the results. Specification: HM-7.

``metric`` is a composite ``<variable>_<statistic>`` (``q_mean_annual_7day_min``),
so a result file is self-contained once it leaves the project tree —
``variable`` is derivable from it and needs no column of its own. The vocabulary
and the variable tokens live in ``shared/indicator_tables.py``, which is the
published contract.

**Three grain classes, and the reason they differ** (CR-2):

- **A — linear in years.** Emitted PER REALIZATION. All are "annual statistic,
  then mean over years", and realizations are equal-length, so per-realization
  values average back to the pooled value exactly. Nothing is lost by emitting
  the finer grain, and ``aggregate_rlz`` existed only to choose between grains
  that are not actually different.
- **B — non-linear fit.** Pooled only (``rlz_id = 0``). A GEV fit over one
  short realization is ill-conditioned; pooling multiplies the block sample by
  ``RLZ_NUM``.
- **C — selects a category.** Pooled only. ``idxmax()`` picks ONE month, so
  different realizations can pick different ones. The month is fixed from the
  ``st_0`` baseline and then evaluated for every member — which is what makes
  the baseline rows mandatory rather than decorative.

**Pooling pools the SAMPLE, not a spliced series.** The pre-R11 code concatenated
realizations and overwrote the index with a synthetic continuous ``date_range``,
butt-splicing them into one fictitious record. A ``rolling(7)`` window then
crossed each splice and manufactured 7-day flows that occurred in no realization,
which could become that year's annual minimum and enter the GEV block sample. For
the 7-day return level we therefore extract each realization's annual minima
*within* that realization and pool the blocks. It also removes an unstated
assumption: both methods give ``RLZ_NUM x N`` blocks only if every realization is
a whole number of years on the same calendar boundary, and nothing checks that.
"""

import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

import blueearth_cst.shared.metrics_definition as md
from blueearth_cst.shared.indicator_tables import (
    BASIN_METRIC_SUFFIXES,
    INDICATOR_COLUMNS,
    POOLED_REALIZATION,
    RETURN_PERIOD_LOW_YR,
    RETURN_PERIOD_PEAK_YR,
    basin_metric_name,
    basin_reduction,
    output_code,
    q_metric_name,
)
from blueearth_cst.shared.snake_utils import (
    DEFAULT_WATER_YEAR_ANCHOR,
    index_width,
    log_row,
)

#: ``rlz_<n>_st_<m>`` in a wflow run CSV's stem. Anchored at the start so a
#: directory component can never satisfy it, and both indices are captured --
#: ``split("_")[-1]`` would silently return the member number as the realization,
#: and a wrong-but-plausible integer mislabels every row it touches.
_MEMBER_IN_STEM = re.compile(r"^rlz_(\d+)_st_(\d+)$")


def member_from_run_csv(csv_fn: Union[str, Path]) -> tuple[int, int]:
    """``(realization, stress_test)`` indices of a wflow run CSV, from its stem."""
    stem = Path(csv_fn).stem
    match = _MEMBER_IN_STEM.match(stem)
    if match is None:
        raise ValueError(
            f"cannot derive the member indices from {csv_fn!r}: expected a "
            f"'rlz_<n>_st_<m>' filename, got {stem!r}"
        )
    return int(match.group(1)), int(match.group(2))


def gauge_columns(columns) -> dict[str, str]:
    """Discharge columns → their BARE gauge id.

    ``Q_130000086`` → ``130000086``. The bare id is what the ``location`` column
    carries: it is the subcatchment id wflow itself emits, so it joins to
    ``outlet_index.csv`` without a crosswalk, and it lets one location carry
    several variables which the wide format could not express.

    The prefix filter is deliberate rather than incidental. A gauge used to emit
    a ``P_`` column beside its ``Q_`` one; that is gone since 2026-08-10 (see
    ``setup_gauges_and_outputs``), but ``wflow_outvars`` can still add
    ``<var>_basavg`` columns, and those are not per-gauge discharge.
    """
    return {c: c[2:] for c in columns if c.startswith("Q_")}


class MissingOutputColumnError(ValueError):
    """``wflow_outvars`` requested a variable the run csvs carry no column for.

    Raised rather than skipped, for the reason ``UnknownOutputVariableError``
    gives about its own case: a silently skipped variable produces a table that is
    a header and nothing else, and an empty table is indistinguishable from "that
    variable was never requested". That is not hypothetical here — it is the exact
    failure this class was added for. 8bd51de renamed the csv headers from
    ``<label>_basavg`` to ``<code>_<subcatchment>``; the matcher below kept looking
    for the retired spelling, and two of three configured tables were written empty
    with every rule green and no line in any log.
    """


def subcatchment_columns(columns, token: str) -> dict[str, str]:
    """Per-subcatchment columns for one variable token → their BARE subcatchment id.

    ``aet_101`` → ``101``, which is what the ``location`` column carries, so these
    rows key the same way the per-gauge discharge rows do and join
    ``outlet_index.csv`` without a crosswalk.

    Matched on ``wflow_outputs.CODES`` — the code the model build actually writes
    into the TOML ``header`` — not on our indicator token, which differs for most
    variables (``precip`` is emitted as ``p``, ``snow`` as ``swe``). The two are
    tabulated side by side in ``dev/reference/indicator-glossary.md``; they are
    not restated here, because a count of how many differ is one more thing to
    get wrong when a variable is added. The trailing
    ``isdigit()`` is what keeps the prefix test honest: ``q`` would otherwise claim
    ``qof_101``, the same over-claiming the retired matcher's docstring worried
    about for ``snow``/``snowmelt``.
    """
    prefix = f"{output_code(token)}_"
    return {
        column: column[len(prefix) :]
        for column in columns
        if column.startswith(prefix) and column[len(prefix) :].isdigit()
    }


def _annual(series: pd.Series, how: str, anchor: str) -> pd.Series:
    """One value per WATER year, by the variable's own reduction.

    ``anchor`` is the pandas end-anchor for `shared.water_year_start`; at the
    Jan default it is ``YE-DEC``, identical to the bare ``YE`` used before.
    """
    resampled = series.resample(anchor)
    return {"sum": resampled.sum, "max": resampled.max, "mean": resampled.mean}[how]()


def _category_month(pooled: pd.DataFrame, which: str) -> int:
    """The wettest or driest month, picked ONCE from the pooled baseline.

    Q5 (2026-08-05). Picking per member would conflate "how does flow in a given
    month respond to perturbation" with "the month itself moved", and the two are
    different questions. One month is chosen for the whole table -- from the
    first column, preserving the pre-R11 behaviour, since the ruling says *picked
    once* rather than picked per location.
    """
    monthly = pooled.groupby(pooled.index.month).sum()
    chosen = monthly.idxmax() if which == "wet" else monthly.idxmin()
    return int(chosen.iloc[0])


def _month_mean(frame: pd.DataFrame, month: int, anchor: str) -> pd.Series:
    """Mean flow in one fixed calendar month, averaged over water years."""
    return frame[frame.index.month == month].resample(anchor).mean().mean()


#: Significant digits kept in the written ``value`` column. Not decimal places —
#: the difference is what makes this safe; see ``_format_value``.
VALUE_SIGNIFICANT_DIGITS = 4


def _format_value(value: float) -> str:
    """Render one indicator value as PLAIN DECIMAL text, 4 significant digits.

    Two separate requirements, both about how this file reads *outside* the
    pipeline. The tables are a deliverable and an interchange surface, so the
    bytes are a contract rather than a display choice.

    **No scientific notation.** Low-flow values reach ~1e-5, so pandas' default
    repr puts ``6.3476255e-05`` in the file, which Excel does not open cleanly.
    ``np.format_float_positional`` is what removes the exponent. Note that the
    obvious ``float_format="%.4g"`` does NOT: it still emits ``6.348e-05``.

    **Four SIGNIFICANT digits, not four decimal places.** This distinction is
    the whole reason a cap is safe to reintroduce here. The pre-R11
    ``.round(2)`` / ``.round(4)`` were decimal-place rounding, which is
    scale-destroying — ``round(2)`` turns ``0.0007395697`` into ``0.0`` — and
    ``dev/scripts/check_baseline.py`` records that as the "accidental drift
    buffer" P1 removed, the reason the comparator moved to a tolerance instead
    of a sha256. A significant-digit cap is scale-invariant: worst-case relative
    error is 5e-4 — half a unit in the last kept digit, worst when the leading
    digit is 1 — against that comparator's own ``INDICATOR_RTOL`` of 1e-2, a 20x
    margin, so it cannot mask a difference the baseline gate would have caught.
    ``tests/test_export_wflow_results.py`` pins that claim to the tolerance
    constants and to the real reference table, so tightening either fails loudly
    here rather than silently invalidating the argument. Measured worst case on
    the current reference table is 4.6e-4.

    Missing values render as the empty field pandas would have written via
    ``na_rep``. Stated explicitly because ``.map()`` bypasses that path and
    would otherwise put the literal string ``nan`` in the file.
    """
    if pd.isna(value):
        return ""
    return np.format_float_positional(
        value,
        precision=VALUE_SIGNIFICANT_DIGITS,
        unique=False,
        fractional=False,
        trim="-",
    )


def _return_level_from_blocks(
    blocks: pd.DataFrame, period: int, mode: str
) -> pd.Series:
    """Fit a GEV to a POOLED block sample and read one return level off it.

    ``frequency_analysis`` blocks a time series internally, which forces the
    caller to hand it a single continuous record -- the very splice this
    reduction exists to avoid. Blocking first and fitting here keeps each
    realization's blocks its own: ``RLZ_NUM x N`` genuine annual extrema, none of
    them straddling a boundary between two realizations.

    ``mode="max"`` reads the upper tail (a flood level), ``mode="min"`` the lower
    (a drought level), which is why the quantile flips rather than the fit.
    """
    import xarray as xr
    from xclim.indices.stats import fit, parametric_quantile

    quantile = 1.0 - 1.0 / period if mode == "max" else 1.0 / period
    levels = {}
    for column in blocks.columns:
        sample = blocks[column].to_numpy(dtype="float64")
        sample = sample[np.isfinite(sample)]
        da = xr.DataArray(sample, dims=("time",), name=str(column))
        params = fit(da, dist="genextreme")
        levels[column] = float(
            parametric_quantile(params, q=quantile).values.ravel()[0]
        )
    return pd.Series(levels)


def _rows(metric, st_id, realization, values, locations) -> list[tuple]:
    """One long-format row per location, for one metric at one member.

    The tuple order is ``INDICATOR_COLUMNS`` and must stay that way: these tuples
    are handed to ``pd.DataFrame(rows, columns=INDICATOR_COLUMNS)``, which assigns
    names POSITIONALLY and would silently mislabel every column rather than raise.
    """
    return [
        (
            metric,
            locations[column],
            st_id,
            realization,
            float(values[column]),
        )
        for column in values.index
    ]


def analyze_wflow_results(
    csv_fns: List[Union[str, Path]],
    results_dir: Union[str, Path],
    st_num: int,
    indicator_tokens: List[str],
    table_paths: dict,
    st_start: int = 0,
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
):
    """Reduce every stress-test run into one long table per configured variable.

    ``table_paths`` maps token to output path, threaded from the rule's declared
    outputs rather than rebuilt here, so the DAG and the writer cannot disagree
    about which tables exist or where they go.

    ``anchor`` is the pandas water-year end-anchor from
    ``shared.water_year_start``. It defaults to a January year, which is the
    calendar year these reductions used before and changes no recorded number.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- the id width ---------------------------------------------------------
    # From the SAME shared helper rule 3.09 pads with, so the two spellings of a
    # member token cannot diverge. It used to be read off the design table, which
    # this rule no longer takes as an input at all: the axis is derived at
    # reporting time from the lookup (HM-7), so the reduction needs no parameter
    # artifact whatever.
    st_width = index_width(st_num)

    # -- index the runs by member ---------------------------------------------
    runs: dict = {}
    for csv_fn in csv_fns:
        rlz, st = member_from_run_csv(csv_fn)
        runs.setdefault(st, {})[rlz] = Path(csv_fn)
    members = sorted(runs)

    # -- run coverage, checked before any reduction work ----------------------
    # The predecessor of this check verified that the DECLARED parameter files
    # covered 1..st_num. This one verifies what actually RAN, which is both the
    # stronger statement and the one that survives the parameter files going
    # away. It stays at the run-time tier deliberately: `validate_hm7` is a
    # test-time validator, and the HM-7 record notes its "no rows" check is never
    # invoked at run time.
    expected_members = set(range(st_start, st_num + 1))
    if set(members) != expected_members:
        raise ValueError(
            f"the stress-test runs do not cover {sorted(expected_members)}: got "
            f"{sorted(members)}. Every member's wflow run must be present before "
            f"the indicator tables are written -- a partial set would produce a "
            f"response surface silently missing grid cells, or biased if the "
            f"absent members sit at one end of the grid."
        )

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    first = read(csv_fns[0])
    q_locations = gauge_columns(first.columns)

    # -- resolve every non-discharge variable's columns, ONCE, before reducing --
    # Fail here rather than inside the member loop: a variable whose columns are
    # absent can only ever produce an empty table, and finding that out before any
    # work is done is the difference between a run that stops with the reason and
    # a run that finishes green with a header-only deliverable.
    subcatchment_locations: dict[str, dict[str, str]] = {}
    for token in indicator_tokens:
        if token == "q" or token not in BASIN_METRIC_SUFFIXES:
            continue
        found = subcatchment_columns(first.columns, token)
        if not found:
            raise MissingOutputColumnError(
                f"wflow_outvars requested {token!r}, but {csv_fns[0]} carries no "
                f"{output_code(token)}_<subcatchment> column. Its columns are "
                f"{sorted(first.columns)}. The csv header comes from the "
                f"'[[output.csv.column]]' entries the model build writes, so "
                f"either the variable was added to wflow_outvars after the model "
                f"was built (rebuild it) or the header code has changed and "
                f"blueearth_cst/shared/wflow_outputs.py no longer matches it."
            )
        subcatchment_locations[token] = found

    # -- the class-C month, fixed once from the pooled baseline ---------------
    # Q5: pick from st_0, then evaluate that month for every member, so the
    # surface shows how flow in a GIVEN month responds rather than conflating
    # that with the month itself moving. Requires the baseline runs to exist,
    # which ST_START = 0 guarantees whenever run_historical is set.
    wet_month = dry_month = None
    if q_locations and 0 in runs:
        baseline = pd.concat([read(p)[list(q_locations)] for p in runs[0].values()])
        wet_month = _category_month(baseline, "wet")
        dry_month = _category_month(baseline, "dry")

    rows: dict = {token: [] for token in indicator_tokens}
    log_row(
        f"reducing {len(csv_fns)} runs into {len(indicator_tokens)} indicator "
        f"table(s): {', '.join(indicator_tokens)}",
        module="export",
    )

    for st in members:
        st_id = f"{st:0{st_width}d}"
        by_rlz = runs[st]

        # ---- discharge ------------------------------------------------------
        if "q" in rows and q_locations:
            per_rlz = {
                rlz: read(p)[list(q_locations)] for rlz, p in sorted(by_rlz.items())
            }

            # Class A: per realization. Linear in years, so the finer grain
            # averages back to the pooled value exactly and nothing is lost.
            for rlz, sim in per_rlz.items():
                annual = {
                    "mean": sim.resample(anchor).mean().mean(),
                    "max": sim.resample(anchor).max().mean(),
                    "min": sim.resample(anchor).min().mean(),
                    "q95": sim.resample(anchor).quantile(0.95).mean(),
                    "Q7day_max": md.Q7d_maxyear(sim, anchor),
                    "Q7day_min": md.Q7d_min(sim, anchor),
                    "BaseFlowIndex": md.BFI(sim, anchor),
                }
                for statistic, values in annual.items():
                    rows["q"] += _rows(
                        q_metric_name(statistic),
                        st_id,
                        rlz,
                        values,
                        q_locations,
                    )

            # Class B: pooled blocks, never a spliced series.
            high_blocks = pd.concat(
                [s.resample(anchor).max() for s in per_rlz.values()],
                ignore_index=True,
            )
            low_blocks = pd.concat(
                [s.rolling(7).mean().resample(anchor).min() for s in per_rlz.values()],
                ignore_index=True,
            )
            for statistic, blocks, period, mode in (
                ("return_level_max", high_blocks, RETURN_PERIOD_PEAK_YR, "max"),
                (
                    "return_level_7day_min",
                    low_blocks,
                    RETURN_PERIOD_LOW_YR,
                    "min",
                ),
            ):
                rows["q"] += _rows(
                    q_metric_name(statistic),
                    st_id,
                    POOLED_REALIZATION,
                    _return_level_from_blocks(blocks, period, mode),
                    q_locations,
                )

            # Class C: pooled, at the month fixed from the baseline.
            if wet_month is not None:
                pooled = pd.concat(per_rlz.values())
                for statistic, month in (
                    ("wetmonth_mean", wet_month),
                    ("drymonth_mean", dry_month),
                ):
                    rows["q"] += _rows(
                        q_metric_name(statistic),
                        st_id,
                        POOLED_REALIZATION,
                        _month_mean(pooled, month, anchor),
                        q_locations,
                    )

        # ---- the per-subcatchment variables ----------------------------------
        # Per realization, for the same reason class A is: these are "annual
        # statistic, then mean over years", so the finest grain is available and
        # ruling (b1) says the table carries it and lets downstream aggregate.
        #
        # And per LOCATION, for a reason that is not a preference: the model
        # declares these with `map = "subcatchment"`, so a run emits one column per
        # subcatchment and no whole-basin column exists to reduce. Q11 forbids
        # manufacturing one here by area-weighting -- whether subcatchments nest or
        # tile decides whether that mean is even valid -- so the finest grain the
        # run offers is the grain the table carries, on both axes.
        for token, locations in subcatchment_locations.items():
            metric = basin_metric_name(token)
            how = basin_reduction(token)
            for rlz, path in sorted(by_rlz.items()):
                sim = read(path)
                values = pd.Series(
                    {c: float(_annual(sim[c], how, anchor).mean()) for c in locations}
                )
                rows[token] += _rows(metric, st_id, rlz, values, locations)

    # -- write ----------------------------------------------------------------
    for token in indicator_tokens:
        table = pd.DataFrame(rows[token], columns=list(INDICATOR_COLUMNS))
        # float32: the reduction's own precision. The pre-R11 round(2)/round(4)
        # that used to follow it was an accidental drift buffer, and dropping it
        # is why the baseline comparator moves to a tolerance rather than a byte
        # hash. What `_format_value` adds below is NOT that rounding returning —
        # see its docstring for why a significant-digit cap is a different thing.
        table["value"] = table["value"].astype("float32")
        table["rlz_id"] = table["rlz_id"].astype("int64")
        # Format ONLY the value column. `to_csv(float_format=...)` would apply to
        # every float column, rewriting temp_change/precip_change from `0.0` to
        # `0` — bytes that consumers join on and that the baseline comparator
        # aligns rows by. Formatting one column leaves all the others exact.
        table["value"] = table["value"].map(_format_value)
        table.to_csv(table_paths[token], index=False)
        log_row(f"wrote {table_paths[token]} ({len(table)} rows)", module="export")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import (
            tee_to_log,
            water_year_end_anchor,
        )

        with tee_to_log(sm.log[0]):
            tokens = list(sm.params.indicator_tokens)
            analyze_wflow_results(
                csv_fns=sm.input.rlz_csv_fns,
                results_dir=sm.params.results_dir,
                st_num=sm.params.st_num,
                st_start=sm.params.st_start,
                indicator_tokens=tokens,
                table_paths={t: getattr(sm.output, f"{t}_indicators") for t in tokens},
                anchor=water_year_end_anchor(sm.params.water_year_start),
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
