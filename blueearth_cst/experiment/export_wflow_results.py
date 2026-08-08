# -*- coding: utf-8 -*-
"""Reduce the stress-test runs into one LONG-format indicator table per variable.

R11 CR-2. The pre-R11 writer produced two WIDE tables — ``q_indicators.csv`` with
one column per gauge, and ``basin_indicators.csv`` with one column per basin
variable — so the header grew with the gauge count and a reader had to know which
columns were locations. This emits **one table per output variable**, each with a
**fixed six-column header** that does not grow with anything:

    metric, temp_change, precip_change, realization_id, location, value

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
- **B — non-linear fit.** Pooled only (``realization_id = 0``). A GEV fit over one
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
    BASIN_LOCATION,
    BASIN_METRIC_SUFFIXES,
    INDICATOR_COLUMNS,
    POOLED_REALIZATION,
    basin_metric_name,
    basin_reduction,
    q_metric_name,
)
from blueearth_cst.shared.snake_utils import log_row

#: ``rlz_<n>_st_<m>`` in a wflow run CSV's stem. Anchored at the start so a
#: directory component can never satisfy it, and both indices are captured --
#: ``split("_")[-1]`` would silently return the member number as the realization,
#: and a wrong-but-plausible integer mislabels every row it touches.
_MEMBER_IN_STEM = re.compile(r"^rlz_(\d+)_st_(\d+)$")

#: Month lengths in the weather generator's calendar. ``impose_climate_change.R``
#: writes every perturbed realization with ``calendar = "noleap"``, so a year is
#: 365 days and February is always 28 -- there is no leap branch to reach here.
_MONTH_LENGTHS = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype="float64"
)
_MONTHS = tuple(range(1, 13))


def annual_perturbation(
    df_st: pd.DataFrame, column: str, source: Union[str, Path] = ""
) -> float:
    """Collapse a perturbation file's TWELVE monthly values to ONE axis value.

    ``prepare_stress_test_grid`` builds each member from the config's 12-element
    ``min``/``max`` vectors, so every perturbation is monthly. The response
    surface is two-dimensional, so the reduction owes it a single annual figure
    per axis.

    **Month-length-weighted mean**, per the 2026-08-07 ruling on [R9-3]. That
    makes the temperature axis identical to how WF2 defines its annual change
    factor (a duration-weighted mean for an intensive variable), which matters
    because the CMIP6 "GCM dots" are overlaid on THESE axes -- two different
    collapses would compare two different quantities.

    The precipitation axis APPROXIMATES that definition: WF2 integrates precip
    over the year and takes the ratio, weighting each month by its baseline
    precipitation, while month-length weighting assumes a uniform daily rate.
    They agree exactly for a flat vector and diverge with the covariance between
    the perturbation and the basin's seasonal cycle. The exact form was declined
    because it costs a climatology input edge on the rule.

    Flat vectors short-circuit on exact equality rather than falling through the
    weighted mean, which would round twelve identical values to something a unit
    in the last place away from them.
    """
    values = df_st[column].to_numpy(dtype="float64")
    if values.size != len(_MONTHS):
        raise ValueError(
            f"{source or 'stress-test parameter file'}: column {column!r} has "
            f"{values.size} rows, expected one per month ({len(_MONTHS)})"
        )
    if "month" in df_st.columns:
        months = df_st["month"].to_numpy(dtype="int64")
        if tuple(sorted(months)) != _MONTHS:
            raise ValueError(
                f"{source or 'stress-test parameter file'}: 'month' column is "
                f"{months.tolist()}, expected the twelve calendar months"
            )
        values = values[np.argsort(months)]
    if np.ptp(values) == 0:
        return float(values[0])
    return float(np.average(values, weights=_MONTH_LENGTHS))


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
    several variables (a registry gauge emits both ``Q_`` and ``P_`` at the same
    point) which the wide format could not express.
    """
    return {c: c[2:] for c in columns if c.startswith("Q_")}


def basavg_column(columns, token: str) -> str | None:
    """The basin-average column for one variable token, if the run emitted it."""
    for column in columns:
        if column.endswith("_basavg") and _matches_token(column, token):
            return column
    return None


def _matches_token(column: str, token: str) -> bool:
    """Whether a ``<something>_basavg`` column belongs to this variable token.

    wflow names these from the SEMANTIC label ("actual evapotranspiration_basavg"),
    not from our token, so the match is on the label's own words rather than on
    the token string. Kept narrow deliberately: a substring test on ``snow``
    would also claim ``snowmelt_basavg`` if wflow ever emitted one.
    """
    label = column[: -len("_basavg")].strip().lower()
    return {
        "aet": label in {"actual evapotranspiration", "aet"},
        "recharge": label in {"groundwater recharge", "recharge"},
        "precip": label in {"precipitation", "precip"},
        "snow": label in {"snow", "snowpack"},
        "overland_flow": label in {"overland flow", "overland_flow"},
        "q": label in {"river discharge", "q", "discharge"},
    }.get(token, False)


def _annual(series: pd.Series, how: str) -> pd.Series:
    """One value per calendar year, by the variable's own reduction."""
    resampled = series.resample("YE")
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


def _month_mean(frame: pd.DataFrame, month: int) -> pd.Series:
    """Mean flow in one fixed calendar month, averaged over years."""
    return frame[frame.index.month == month].resample("YE").mean().mean()


def _return_level_from_blocks(blocks: pd.DataFrame, period: int, mode: str) -> pd.Series:
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


def _rows(metric, temp, precip, realization, values, locations) -> list[tuple]:
    """One long-format row per location, for one metric at one member."""
    return [
        (metric, temp, precip, realization, locations[column], float(values[column]))
        for column in values.index
    ]


def analyze_wflow_results(
    csv_fns: List[Union[str, Path]],
    st_csv_fns: List[Union[str, Path]],
    results_dir: Union[str, Path],
    st_num: int,
    indicator_tokens: List[str],
    table_paths: dict,
    Tpeak: int = 10,
    Tlow: int = 2,
):
    """Reduce every stress-test run into one long table per configured variable.

    ``table_paths`` maps token to output path, threaded from the rule's declared
    outputs rather than rebuilt here, so the DAG and the writer cannot disagree
    about which tables exist or where they go.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- the perturbation axes, per member ------------------------------------
    # st_0 is the reserved unperturbed baseline and has no parameter file, so
    # the declared set is exactly 1..st_num. Checked here rather than letting a
    # Snakefile/script disagreement surface as a KeyError deep in the loop.
    st_csv_by_num = {int(Path(p).stem.split("_")[-1]): p for p in st_csv_fns}
    if set(st_csv_by_num) != set(range(1, st_num + 1)):
        raise ValueError(
            f"stress-test parameter files do not cover 1..{st_num}: got "
            f"{sorted(st_csv_by_num)}, expected {sorted(range(1, st_num + 1))}"
        )
    axes = {0: (0.0, 0.0)}  # the baseline sits at the origin by definition
    for st, path in st_csv_by_num.items():
        df_st = pd.read_csv(path)
        axes[st] = (
            annual_perturbation(df_st, "temp_mean", path),
            annual_perturbation(df_st, "precip_mean", path) * 100 - 100,
        )

    # -- index the runs by member ---------------------------------------------
    runs: dict = {}
    for csv_fn in csv_fns:
        rlz, st = member_from_run_csv(csv_fn)
        runs.setdefault(st, {})[rlz] = Path(csv_fn)
    members = sorted(runs)

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    first = read(csv_fns[0])
    q_locations = gauge_columns(first.columns)

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
        temp, precip = axes[st]
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
                    "mean": sim.resample("YE").mean().mean(),
                    "max": sim.resample("YE").max().mean(),
                    "min": sim.resample("YE").min().mean(),
                    "q95": sim.resample("YE").quantile(0.95).mean(),
                    "Q7day_max": md.Q7d_maxyear(sim),
                    "Q7day_min": md.Q7d_min(sim),
                    "BaseFlowIndex": md.BFI(sim),
                }
                for statistic, values in annual.items():
                    rows["q"] += _rows(
                        q_metric_name(statistic, Tpeak, Tlow),
                        temp, precip, rlz, values, q_locations,
                    )

            # Class B: pooled blocks, never a spliced series.
            high_blocks = pd.concat(
                [s.resample("YE").max() for s in per_rlz.values()], ignore_index=True
            )
            low_blocks = pd.concat(
                [s.rolling(7).mean().resample("YE").min() for s in per_rlz.values()],
                ignore_index=True,
            )
            for statistic, blocks, period, mode in (
                ("return_level_max", high_blocks, Tpeak, "max"),
                ("return_level_7day_min", low_blocks, Tlow, "min"),
            ):
                rows["q"] += _rows(
                    q_metric_name(statistic, Tpeak, Tlow),
                    temp, precip, POOLED_REALIZATION,
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
                        q_metric_name(statistic, Tpeak, Tlow),
                        temp, precip, POOLED_REALIZATION,
                        _month_mean(pooled, month), q_locations,
                    )

        # ---- basin-scalar variables -----------------------------------------
        # Per realization, for the same reason class A is: these are "annual
        # statistic, then mean over years", so the finest grain is available and
        # ruling (b1) says the table carries it and lets downstream aggregate.
        for token in indicator_tokens:
            if token == "q" or token not in BASIN_METRIC_SUFFIXES:
                continue
            metric = basin_metric_name(token)
            how = basin_reduction(token)
            for rlz, path in sorted(by_rlz.items()):
                sim = read(path)
                column = basavg_column(sim.columns, token)
                if column is None:
                    continue
                value = float(_annual(sim[column], how).mean())
                rows[token].append((metric, temp, precip, rlz, BASIN_LOCATION, value))

    # -- write ----------------------------------------------------------------
    for token in indicator_tokens:
        table = pd.DataFrame(rows[token], columns=list(INDICATOR_COLUMNS))
        # float32, UNROUNDED: the pre-R11 round(2)/round(4) was an accidental
        # drift buffer, and dropping it is why the baseline comparator moves to a
        # tolerance rather than a byte hash.
        table["value"] = table["value"].astype("float32")
        table["realization_id"] = table["realization_id"].astype("int64")
        table.to_csv(table_paths[token], index=False)
        log_row(f"wrote {table_paths[token]} ({len(table)} rows)", module="export")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            tokens = list(sm.params.indicator_tokens)
            analyze_wflow_results(
                csv_fns=sm.input.rlz_csv_fns,
                st_csv_fns=sm.input.st_csv_fns,
                results_dir=sm.params.results_dir,
                st_num=sm.params.st_num,
                indicator_tokens=tokens,
                table_paths={
                    t: getattr(sm.output, f"{t}_indicators") for t in tokens
                },
                Tpeak=sm.params.Tpeak,
                Tlow=sm.params.Tlow,
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
