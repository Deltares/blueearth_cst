# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

# This module has no `.raster` access and does not import hydromt. The actual
# accessor caller, get_stats_climate_proj, imports hydromt explicitly before
# its `.raster.vars` use; registration no longer relies on a plotting import.
#
# pandas and xarray are DEFERRED into the three functions that use them.
# `analyze_projections.smk` imports this module at PARSE time so `STAGE_B_KERNEL`
# can hold the FUNCTION OBJECTS for `kernel_hash`; the objects are required, the
# numeric stack is not.

from blueearth_cst.projections.calendar_weights import month_length_weights
from blueearth_cst.projections.change_factor_table import COMPANION_SEP
from blueearth_cst.projections.variable_spec import canonical_kind, change_kind
from blueearth_cst.shared.collection_utils import intersection
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS

# %%


def _to_datetime_index(ds):
    """Convert an object-dtype (cftime) time index to a proleptic-Gregorian
    DatetimeIndex.

    CMIP6-native calendars (360_day, noleap) decode to an object-dtype
    CFTimeIndex that cannot be sliced with the ``pd.Timestamp`` bounds built in
    ``get_change_annual_clim_proj`` -- the cross-calendar comparison raises
    ``TypeError``. The stats time series here are monthly (``MS``, day-01), which
    has no calendar-invalid dates, so the conversion is lossless for the annual
    resample that follows. Mirrors ``plot_proj_timeseries.py``. (t260720c)
    """
    if "time" in ds.coords and ds.indexes["time"].dtype == "O":
        ds = ds.copy()
        ds["time"] = ds.indexes["time"].to_datetimeindex()
    return ds


# `get_change_clim_projections` -- the CELLWISE change used only by the
# gridded branch -- was removed at S8-08(c) with that branch. It dispatched on
# the literal name "precip" for its multiplicative/additive choice, so it was
# also the last unconverted 5e site outside stage A's resample.


class NoCompleteHydrologicalYear(ValueError):
    """A valid selected period contains no complete water year."""


def hydrological_year_bounds(ds_time, start_month_hyd_year="Jan"):
    """First/last timestamp of the COMPLETE hydrological years in ``ds_time``.

    Extracted verbatim from :func:`get_change_annual_clim_proj`, which recomputed
    it identically on every pass of its per-variable loop. One definition, because
    the composition record (migration step 4d) reports the *effective* reference
    window and must report the window the change factors actually used — a second
    implementation would eventually disagree with the numbers it annotates.

    Bounds include every complete water year contained in the monthly record;
    leading and trailing partial years are dropped.

    Returns ``(start, end, n_hydrological_years)``.
    """
    import pandas as pd

    if len(ds_time["time"].values) == 0:
        raise NoCompleteHydrologicalYear(
            "Cannot determine complete hydrological years from an empty time axis"
        )
    data_start = pd.Timestamp(ds_time["time"].values[0])
    data_end = pd.Timestamp(ds_time["time"].values[-1])

    def year_start(year):
        return pd.to_datetime(f"{year}-{start_month_hyd_year}")

    # First complete year: the earliest whose START lies inside the data. A window
    # beginning mid-year has a leading partial year, which is dropped.
    first = int(ds_time["time.year"].values[0])
    if year_start(first) < data_start:
        first += 1

    # Last complete year: the latest whose END lies inside the data. This is the
    # off-by-one that was fixed on 2026-07-30 (see
    # dev/records/milestones/r08/2026-07-30_wf2-5f-hydyear-offbyone.md). The previous form ended
    # the window at `{last_year}-{month} - 1 month` unconditionally, i.e. it
    # assumed the year STARTING in the data's final calendar year is always
    # incomplete. That holds when the data stops mid-year -- an October start over
    # data ending in December -- and fails when the data runs through that year's
    # end, which is exactly a January start over data ending in December. The
    # final complete year was silently discarded, so a configured 30-year
    # reference delivered 29 and the seed's [1990, 2010] delivered 20 of 21.
    #
    # The condition is "does the data cover this year's end", NOT "add one": a
    # naive +1 would break the October case, which was already correct.
    last = int(ds_time["time.year"].values[-1])
    while last >= first and year_start(last + 1) - pd.DateOffset(months=1) > data_end:
        last -= 1

    n_years = last - first + 1
    if n_years <= 0:
        raise NoCompleteHydrologicalYear(
            f"no complete hydrological year (start month {start_month_hyd_year}) "
            f"between {data_start.date()} and {data_end.date()}. An empty "
            "reference would propagate as an empty denominator into every "
            "relative change factor."
        )
    start = year_start(first)
    end = year_start(last + 1) - pd.DateOffset(months=1)
    return start, end, n_years


#: Step 5d: what a run emits unless it opts into more. The VALUE lives in
#: ``config/advanced_settings.yml`` under ``defaults:``, with the reason the
#: earlier eight were cut to three; a project overrides it with `stats` in the
#: analyze_projections file.
#:
#: Not hashed by `kernel_hash`, and was not before either: the reductions read it
#: as a module GLOBAL, so it appears in `co_names` and never in `co_consts`.
#: Changing the emitted set therefore does not re-derive the cached series --
#: which is correct, since the statistics are computed downstream of them.
#: A tuple, where the setting is a list: this is a fixed set for the run, and
#: the reducers below build their own list from it.
DEFAULT_STATS = tuple(ADVANCED_SETTINGS["defaults"]["change_factor_stats"])


def quantile_label(stat_name, n_years):
    """Label an emitted quantile with the sample size it was computed over.

    The design requires tail quantiles, once opted into, to be "labelled with
    their effective sample size in both the CSV and the report". The label is the
    `stats` coordinate value, so it reaches both without either having to know
    about sample sizes: `q_90` over 20 years reads `q_90[n=20]`, which makes "the
    second-highest of 20" self-evident at the point of use rather than in a
    footnote nobody reaches.

    Non-quantile statistics are returned unchanged -- `mean[n=20]` would be noise,
    since a mean over 20 years is not a claim the sample size undermines.
    """
    if not str(stat_name).startswith("q_") or not n_years:
        return stat_name
    return f"{stat_name}[n={int(n_years)}]"


def _labelled(da, stat_name, n_ref_years):
    """Attach the `stats` coordinate a change factor is keyed by."""
    return da.assign_coords(
        {"stats": quantile_label(stat_name, n_ref_years)}
    ).expand_dims("stats")


def get_change_monthly_clim_proj(
    ds_hist_time,
    ds_clim_time,
    stats=None,
    start_month_hyd_year="Jan",
    variable_spec=None,
    min_reference=None,
    *,
    _reference_cache=None,
):
    """Change factors per CALENDAR MONTH (design §5.6, step 6a-ii).

    The annual product answers "how much wetter is the year"; this answers "how
    much wetter is each January", which is what a seasonal shift looks like and
    what the annual figure averages away.

    Same formulas as the annual table, applied per month: the statistic is
    computed over the per-year values **for that month**, and the `change:`
    arithmetic is then applied to the statistic.

    Two deliberate differences from the annual path:

    * **No month-length weighting.** Calendar weighting (5b) exists to make a
      year's twelve unequal months comparable *within* an aggregate. Comparing
      January to January is already like-for-like, and weighting both sides by 31
      would cancel exactly.
    * **The same complete-hydrological-year window is used**, so every month draws
      on the same set of years. Slicing only by the raw window would give January
      one more sample than December whenever the window starts mid-year, and a
      seasonal pattern assembled from unequal samples is not a pattern.
    """
    import xarray as xr

    stats = list(DEFAULT_STATS) if stats is None else list(stats)
    # Private Stage-B cache: scoped to one historical series and complete
    # analysis configuration. Direct callers get a fresh cache on every call.
    reference = {} if _reference_cache is None else _reference_cache

    ds_hist_time = _to_datetime_index(ds_hist_time)
    ds_clim_time = _to_datetime_index(ds_clim_time)

    hist_start, hist_end, n_ref_years = hydrological_year_bounds(
        ds_hist_time, start_month_hyd_year
    )
    clim_start, clim_end, _ = hydrological_year_bounds(
        ds_clim_time, start_month_hyd_year
    )

    ds = []
    for var in intersection(ds_hist_time.data_vars, ds_clim_time.data_vars):
        hist = ds_hist_time[var].sel(time=slice(hist_start, hist_end))
        clim = ds_clim_time[var].sel(time=slice(clim_start, clim_end))
        if "scenario" in hist.coords and hist.coords["scenario"].size > 0:
            try:
                hist = hist.sel(scenario=ds_hist_time.scenario.values[0])
            except (KeyError, IndexError):
                pass
        hist_by_month = hist.groupby("time.month")
        clim_by_month = clim.groupby("time.month")

        for stat_name in stats:
            key = ("monthly", var, stat_name, str(hist_start), str(hist_end))
            if "q_" in stat_name:
                q = int(stat_name.split("_")[1]) / 100
                if key not in reference:
                    reference[key] = hist_by_month.quantile(q, "time")
                clim_stat = clim_by_month.quantile(q, "time")
            else:
                if key not in reference:
                    reference[key] = getattr(hist_by_month, stat_name)("time")
                clim_stat = getattr(clim_by_month, stat_name)("time")
            hist_stat = reference[key]

            # S8-04: the FUTURE LEVEL, for the table's `absolute_value`. Emitted
            # for every variable and every statistic, because it is what
            # `relative_value` is measured against and stage B used to compute it
            # and throw it away. A companion, so it rides with its variable
            # through the merge instead of becoming a row of its own.
            reference_stat = hist_stat.drop_vars(
                "scenario", errors="ignore"
            ).broadcast_like(clim_stat)
            for stat, suffix in ((clim_stat, "level"), (reference_stat, "reference")):
                # Same shape contract as the annual path above.
                companion = stat.rename(f"{var}{COMPANION_SEP}{suffix}")
                if "quantile" in companion.coords:
                    companion = companion.drop_vars("quantile")
                ds.append(_labelled(companion, stat_name, n_ref_years).to_dataset())

            absolute = (clim_stat - hist_stat).rename(var)
            if change_kind(variable_spec, var) == "relative":
                change = ((clim_stat - hist_stat) / hist_stat * 100).rename(var)
                # Step 6b / A2: a near-zero reference makes the RATIO meaningless
                # while the difference stays informative, so the ratio is dropped
                # and the difference kept -- never both. The comparison is strict,
                # so a reference exactly at the threshold survives.
                threshold = (min_reference or {}).get(var)
                if threshold is not None:
                    flagged = hist_stat < threshold
                    change = change.where(~flagged)
                    ds.append(
                        _labelled(
                            flagged.astype("int8").rename(f"{var}__flagged"),
                            stat_name,
                            n_ref_years,
                        ).to_dataset()
                    )
                    ds.append(
                        _labelled(
                            absolute.rename(f"{var}__absolute"), stat_name, n_ref_years
                        ).to_dataset()
                    )
            else:
                change = absolute
            ds.append(_labelled(change, stat_name, n_ref_years).to_dataset())

    # Each element of `ds` carries ONE `stats` label, so the merge is a union along
    # `stats` (`join="outer"`) that stitches the per-label slices of a variable back
    # into one array (`compat="no_conflicts"`). Both were xarray's defaults and are
    # being deprecated, so they are pinned: under the incoming `compat="override"`
    # the first label's values would win and every other label would be silently
    # dropped, and under `join="exact"` the merge would raise outright.
    return xr.merge(ds, join="outer", compat="no_conflicts")


def get_change_annual_clim_proj(
    ds_hist_time,
    ds_clim_time,
    stats=None,
    start_month_hyd_year="Jan",
    calendar=None,
    variable_spec=None,
    *,
    _reference_cache=None,
):
    """

    Parameters
    ----------
    ds_hist_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over the grid (historical).
    ds_clim_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over the grid (projection).
    stats : list of strings of statistics
        Quantiles are given as ``q_xx`` and are OPT-IN as of step 5d: the
        default is ``DEFAULT_STATS`` (``mean``, ``median``, ``std``). An
        emitted quantile is labelled with its effective sample size, so a
        ``q_90`` over a 20-year window reads ``q_90[n=20]``.
    start_month_hyd_year : str, optional
        Month start of hydrological year. The default is "Jan".

    Returns
    -------
    stats_annual_change : xarray dataset
        annual statistics per each models/scenario/horizon.

    """
    import xarray as xr

    # Step 5d: `stats=None` means the v2.0 default set, not "all eight".
    # Passed explicitly by callers that opt into tail quantiles.
    stats = list(DEFAULT_STATS) if stats is None else list(stats)
    # Private Stage-B cache: scoped to one historical series and complete
    # analysis configuration. Direct callers get a fresh cache on every call.
    reference = {} if _reference_cache is None else _reference_cache

    # cftime-safe slicing: convert any CMIP6-native cftime index up front so the
    # pd.Timestamp hydrological-year bounds below apply to a pandas-native index
    # (t260720c / D-CAL).
    ds_hist_time = _to_datetime_index(ds_hist_time)
    ds_clim_time = _to_datetime_index(ds_clim_time)

    # Fail loud on asymmetric hist/clim structure rather than silently dropping
    # (t260720d / D-VAR, D-MEM). Otherwise intersection() below quietly discards
    # a configured variable, and xarray's default inner alignment on the change
    # arithmetic quietly discards an unshared member -- both move the response
    # surface with no error.
    hist_vars = set(ds_hist_time.data_vars)
    clim_vars = set(ds_clim_time.data_vars)
    if hist_vars != clim_vars:
        unshared_vars = sorted(hist_vars.symmetric_difference(clim_vars))
        raise ValueError(
            f"asymmetric hist/clim variables: {unshared_vars} present in only "
            f"one dataset (hist={sorted(hist_vars)}, clim={sorted(clim_vars)})"
        )
    if "member" in ds_hist_time.coords and "member" in ds_clim_time.coords:
        hist_mem = set(ds_hist_time["member"].values.tolist())
        clim_mem = set(ds_clim_time["member"].values.tolist())
        if hist_mem != clim_mem:
            unshared_mem = sorted(hist_mem.symmetric_difference(clim_mem))
            raise ValueError(
                f"asymmetric hist/clim members: {unshared_mem} present in only "
                f"one dataset (hist={sorted(hist_mem)}, clim={sorted(clim_mem)})"
            )

    # Hoisted out of the per-variable loop, where these four were recomputed
    # identically on every iteration. Same expressions, same values -- the point of
    # the hoist is that `hydrological_year_bounds` becomes the SINGLE definition of
    # "the complete hydrological years this change factor used", so the composition
    # record (step 4d) can report that window instead of a second copy that drifts.
    start_hyd_year_hist, end_hyd_year_hist, n_ref_years = hydrological_year_bounds(
        ds_hist_time, start_month_hyd_year
    )
    start_hyd_year_clim, end_hyd_year_clim, _ = hydrological_year_bounds(
        ds_clim_time, start_month_hyd_year
    )

    # Step 5b: month-length weights in the MODEL's calendar. `calendar=None` keeps
    # the pre-5b unweighted behaviour, which is what the existing unit tests
    # exercise -- they construct synthetic series with no calendar to speak of.
    # Production always passes one: derive_change_factors reads `cst_calendar` off
    # the series and stage B refuses an unknown one (falsifier G5).
    def _aggregate_annual(da, freq, how):
        """Annual aggregate of a monthly series, month-length weighted."""
        if calendar is None:
            return getattr(da.resample(time=freq), how)("time")
        w = xr.DataArray(
            month_length_weights(da, calendar),
            dims="time",
            coords={"time": da["time"]},
        )
        # `.rename` on BOTH branches. xarray drops the name on any binary op
        # between differently-named operands, so `da * w` is already unnamed --
        # the multiplication loses it, not just the division. The caller does
        # `change.to_dataset()`, which raises "unable to convert unnamed
        # DataArray"; a first fix that renamed only the division branch left
        # precip still failing, because precip never divides.
        weighted = (da * w).resample(time=freq).sum("time").rename(da.name)
        if how == "sum":
            # precip is a RATE; the annual total is the duration integral, so the
            # weighted sum IS the answer. Its units become mm/year rather than a
            # sum of daily rates -- immaterial downstream, since precip's change
            # factor is relative and unit-invariant.
            return weighted
        # temp is intensive: a duration-weighted MEAN, not an integral.
        # `.rename` is load-bearing: dividing two DataArrays drops the name, and
        # the caller's `change.to_dataset()` then raises "unable to convert
        # unnamed DataArray". The sum branch keeps its name because it never
        # divides.
        return (weighted / w.resample(time=freq).sum("time")).rename(da.name)

    def _annual(da, freq, how, *, historical=False):
        if not historical:
            return _aggregate_annual(da, freq, how)
        key = (
            "annual",
            da.name,
            freq,
            how,
            calendar,
            str(start_hyd_year_hist),
            str(end_hyd_year_hist),
        )
        if key not in reference:
            reference[key] = _aggregate_annual(da, freq, how)
        return reference[key]

    ds = []
    for var in intersection(ds_hist_time.data_vars, ds_clim_time.data_vars):
        # Step 5e-iii: the AGGREGATION follows `canonical`, the change arithmetic
        # below follows `change`. Both used to hang off `var == "precip"`, so two
        # independent properties could never disagree.
        if canonical_kind(variable_spec, var) == "rate":
            # a rate integrates over the year
            hist = _annual(
                ds_hist_time[var].sel(
                    time=slice(start_hyd_year_hist, end_hyd_year_hist)
                ),
                f"YS-{start_month_hyd_year.upper()[:3]}",
                "sum",
                historical=True,
            ).sel(
                scenario=ds_hist_time.scenario.values[0],
            )
            clim = _annual(
                ds_clim_time[var].sel(
                    time=slice(start_hyd_year_clim, end_hyd_year_clim)
                ),
                f"YS-{start_month_hyd_year.upper()[:3]}",
                "sum",
            )
            # S8-07 (owner ruling: mm/day everywhere): the change above is a ratio
            # of annual INTEGRALS, but the LEVEL reported beside it must be in the
            # variable's declared units. A duration-weighted mean is that same
            # integral divided by the year's length, so the ratio is untouched --
            # only the level's scale changes, from mm/year to mm/day.
            clim_level = _annual(
                ds_clim_time[var].sel(
                    time=slice(start_hyd_year_clim, end_hyd_year_clim)
                ),
                f"YS-{start_month_hyd_year.upper()[:3]}",
                "mean",
            )
            hist_level = _annual(
                ds_hist_time[var].sel(
                    time=slice(start_hyd_year_hist, end_hyd_year_hist)
                ),
                f"YS-{start_month_hyd_year.upper()[:3]}",
                "mean",
                historical=True,
            ).sel(scenario=ds_hist_time.scenario.values[0])
        else:  # for temp
            # additive for temp
            hist = _annual(
                ds_hist_time[var].sel(
                    time=slice(start_hyd_year_hist, end_hyd_year_hist)
                ),
                f"YS-{start_month_hyd_year.upper()[:3]}",
                "mean",
                historical=True,
            ).sel(
                scenario=ds_hist_time.scenario.values[0],
            )
            clim = _annual(
                ds_clim_time[var].sel(
                    time=slice(start_hyd_year_clim, end_hyd_year_clim)
                ),
                f"YS-{start_month_hyd_year.upper()[:3]}",
                "mean",
            )
            # A state is already a duration-weighted mean in its own units.
            clim_level = clim
            hist_level = hist

        # calc statistics
        for stat_name in stats:  # , stat_props in stats_dic.items():
            key = (
                "annual-stat",
                var,
                stat_name,
                calendar,
                start_month_hyd_year,
                str(start_hyd_year_hist),
                str(end_hyd_year_hist),
            )
            if "q_" in stat_name:
                qvalue = int(stat_name.split("_")[1]) / 100
                if key not in reference:
                    reference[key] = getattr(hist, "quantile")(qvalue, "time")
                clim_stat = getattr(clim, "quantile")(qvalue, "time")
            else:
                if key not in reference:
                    reference[key] = getattr(hist, stat_name)("time")
                clim_stat = getattr(clim, stat_name)("time")
            hist_stat = reference[key]

            if change_kind(variable_spec, var) == "relative":
                change = (clim_stat - hist_stat) / hist_stat * 100
            else:
                change = clim_stat - hist_stat
            change = change.assign_coords(
                {"stats": quantile_label(stat_name, n_ref_years)}
            ).expand_dims("stats")

            if "quantile" in change.coords:
                change = change.drop("quantile")
            ds.append(change.to_dataset())

            # S8-04: the future level, same contract as the monthly path above.
            # The annual table's `absolute_value` was a dead column until now --
            # the companion only ever existed on the monthly side.
            # S8-08 companions: the two LEVELS the change is built from. Shipping
            # both makes every number in a row recoverable from the row -- the
            # difference, which a flagged relative month keeps under 6b but the
            # ratio column cannot express, is `absolute_value - reference_value`.
            def _stat_of(source):
                if "q_" in stat_name:
                    return source.quantile(int(stat_name.split("_")[1]) / 100, "time")
                return getattr(source, stat_name)("time")

            level_stat = _stat_of(clim_level)
            # The BASELINE must be shaped exactly like the level it sits beside.
            # Two traps, both hit before this landed:
            #   1. the historical source carries a scalar `scenario='historical'`
            #      from its `.sel`, which `change` never shows because DataArray
            #      arithmetic drops conflicting scalar coords -- but a companion
            #      emitted directly keeps it and merges onto the wrong label;
            #   2. simply dropping that coord removes the scenario DIMENSION, so
            #      the companion is one rank short of its siblings and the
            #      multi-file merge mangles it.
            # `broadcast_like` fixes both: identical dims, identical labels, the
            # historical values carried across the scenarios they are the
            # reference for. Caught by recomputing the change from the two levels
            # and comparing it against the change column, on every row.
            level_key = ("level", *key)
            if level_key not in reference:
                reference[level_key] = _stat_of(hist_level)
            reference_stat = (
                reference[level_key]
                .drop_vars("scenario", errors="ignore")
                .broadcast_like(level_stat)
            )
            for stat, suffix in ((level_stat, "level"), (reference_stat, "reference")):
                companion = (
                    stat.rename(f"{var}{COMPANION_SEP}{suffix}")
                    .assign_coords({"stats": quantile_label(stat_name, n_ref_years)})
                    .expand_dims("stats")
                )
                if "quantile" in companion.coords:
                    companion = companion.drop_vars("quantile")
                ds.append(companion.to_dataset())

    # Same union-along-`stats` contract as the monthly path above; the same
    # two kwargs are pinned for the same reason.
    stats_annual_change = xr.merge(ds, join="outer", compat="no_conflicts")
    return stats_annual_change


# Time tuples for comparison hist-fut.
# R01 schema delivers these as lists ([1980, 2010]) for both the historical
# window and the future horizons. Pre-R01 configs delivered them as
# comma-separated strings ("1980, 2010"). Accept both.
def _to_str_tuple(value):
    if isinstance(value, str):
        return tuple(map(str, value.split(", ")))
    return tuple(map(str, value))


# %%


# NOTE: this module no longer runs as a Snakemake `script:`. Step 4d merged rules
# 2.04/2.05 into `derive_change_factors`, which imports the functions above and
# owns the orchestration. The former `__main__` block is deleted rather than left
# dead: it was a second copy of the per-point procedure, and two copies of the
# same arithmetic is how they drift apart. The functions stay here — they are the
# tested surface (tests/test_get_change_climate_proj*.py).


# Analysis-only constants: none is referenced by a Stage B kernel root.
ANALYSIS_REFERENCE = (1985, 2014)  # default calendar span
MIN_SEASON_COVERAGE = 0.9


def seasonal_min_years(window_years):
    """Require 90% of the owned window's years, with at least three seasons."""
    import math

    return max(3, math.ceil(MIN_SEASON_COVERAGE * window_years))


class AnalysisWindowUnavailable(ValueError):
    """A valid member series cannot supply the configured analysis-window span."""


def _analysis_month_number(water_year_start):
    """Resolve the configured month abbreviation without changing kernel helpers."""
    import calendar

    names = list(calendar.month_abbr)
    name = str(water_year_start)[:3].title()
    if name not in names[1:]:
        raise ValueError(f"invalid water_year_start: {water_year_start!r}")
    return names.index(name)


def analysis_windows(
    horizons, water_year_start="Jan", reference_window=ANALYSIS_REFERENCE
):
    """Keep complete water years inside the configured calendar-year bounds.

    Return horizon-keyed plain records. ``requested`` and ``used`` are WY start
    labels; ``configured`` retains calendar-year bounds and ``effective`` reports
    the inclusive dates and counts actually used. Partial water years are trimmed,
    never replaced by years outside the configured span. Invalid bounds raise;
    missing member coverage is handled downstream without moving these windows.
    """
    import pandas as pd

    from blueearth_cst.projections.projection_figures import parse_horizon_period
    from blueearth_cst.projections.series_identity import ACQUISITION_WINDOWS

    month = _analysis_month_number(water_year_start)

    def complete_years(span, acquisition, label):
        lower, upper = (int(bound[:4]) for bound in acquisition)
        if span[0] < lower or span[1] > upper:
            raise ValueError(
                f"{label} window {span[0]}-{span[1]} is outside acquisition "
                f"{lower}-{upper}"
            )
        time = pd.date_range(f"{span[0]}-01-01", f"{span[1]}-12-31", freq="MS")
        start, end, count = hydrological_year_bounds(
            {"time": pd.Series(time), "time.year": pd.Series(time.year)},
            water_year_start,
        )
        return (start.year, end.year - (end.month < month)), {
            "start": start.strftime("%Y-%m-%d"),
            "end": (end + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
            "n_water_years": count,
        }

    ref_configured = parse_horizon_period(reference_window)
    reference, ref_effective = complete_years(
        ref_configured, ACQUISITION_WINDOWS["historical"], "reference"
    )
    records = {}
    for name, horizon in horizons.items():
        configured = parse_horizon_period(horizon)
        used, effective = complete_years(
            configured, ACQUISITION_WINDOWS["_scenario"], f"future {name!r}"
        )
        records[name] = {
            "configured": {"reference": ref_configured, "future": configured},
            "effective": {"reference": ref_effective, "future": effective},
            "reference": reference,
            "reference_rule": "configured_complete_water_years",
            "requested": used,
            "used": used,
            "window_id": f"{used[0]}-{used[1]}",
            "shifted": False,
            "joined": False,
            "available": True,
            "reason": "",
        }
    return records


def analysis_slice(series, window, water_year_start="Jan", *, historical=None):
    """Slice exact WY labels, joining the same member's historical run if needed.

    Missing endpoint coverage raises ``AnalysisWindowUnavailable``; internal missing
    months stay missing for seasonal completeness checks. An incompatible join
    raises rather than treating corrupted input as an unavailable member.
    """
    import numpy as np
    import xarray as xr

    from blueearth_cst.projections.projection_figures import parse_horizon_period
    from blueearth_cst.projections.series_checks import assert_calendar_agreement

    first, last = parse_horizon_period(window)
    month = _analysis_month_number(water_year_start)
    begin = first * 12 + month - 1
    end = (last + 1) * 12 + month - 2

    def month_keys(dataset):
        if "time" not in dataset.coords or not dataset.sizes.get("time", 0):
            raise ValueError("analysis series has no time observations")
        keys = dataset.time.dt.year.values * 12 + dataset.time.dt.month.values - 1
        if np.any(np.diff(keys) <= 0):
            raise ValueError("analysis series time must contain ordered, unique months")
        return keys

    keys = month_keys(series)
    original_attrs = dict(series.attrs)
    if keys[0] > begin:
        if historical is None:
            raise AnalysisWindowUnavailable(
                "analysis window needs its member's historical series"
            )
        assert_calendar_agreement(historical, series, "analysis join")
        for coord in ("model", "member", "clim_project"):
            if coord in historical.coords or coord in series.coords:
                if (
                    coord not in historical.coords
                    or coord not in series.coords
                    or not np.array_equal(
                        historical[coord].values, series[coord].values
                    )
                ):
                    raise ValueError(f"analysis join has different {coord} coordinates")
        if set(historical.data_vars) != set(series.data_vars):
            raise ValueError("analysis join has different variables")
        hist_keys = month_keys(historical)
        preceding = historical.isel(time=np.flatnonzero(hist_keys < keys[0]))
        if "scenario" in series.coords:
            preceding = preceding.assign_coords(scenario=series.scenario)
        series = xr.concat(
            [preceding, series],
            dim="time",
            join="exact",
            compat="equals",
            data_vars="minimal",
            coords="minimal",
            combine_attrs="override",
        )
        series.attrs = original_attrs
        keys = month_keys(series)
    if keys[0] > begin or keys[-1] < end:
        raise AnalysisWindowUnavailable(
            f"analysis series does not cover WY {first}-{last}"
        )
    return series.isel(time=np.flatnonzero((keys >= begin) & (keys <= end)))


def detrended_std(values, years):
    """Population SD of OLS residuals against actual retained year coordinates.

    Nonfinite value/year pairs are excluded explicitly. Fewer than three valid
    pairs returns NaN; repeated or unordered year labels and shape errors raise.
    """
    import numpy as np

    values, years = np.asarray(values, dtype=float), np.asarray(years, dtype=float)
    if values.ndim != 1 or years.ndim != 1 or values.shape != years.shape:
        raise ValueError("values and years must be one-dimensional with the same shape")
    valid = np.isfinite(values) & np.isfinite(years)
    values, years = values[valid], years[valid]
    if values.size < 3:
        return float("nan")
    if np.any(np.diff(years) <= 0):
        raise ValueError("years must be strictly increasing and unique")
    x = years - years.mean()
    centred = values - values.mean()
    slope = np.dot(x, centred) / np.dot(x, x)
    residual = centred - slope * x
    return float(np.sqrt(np.mean(residual**2)))


def _analysis_season_values(dataset, variable, season, calendar):
    """Retained season-year labels, mean rates and duration integrals."""
    import numpy as np

    months_by_season = {
        "DJF": (12, 1, 2),
        "MAM": (3, 4, 5),
        "JJA": (6, 7, 8),
        "SON": (9, 10, 11),
    }
    da = dataset[variable]
    extra = [dim for dim in da.dims if dim != "time"]
    if any(da.sizes[dim] != 1 for dim in extra):
        raise ValueError(
            f"seasonal analysis needs one member series: {variable} has {da.sizes}"
        )
    values = da.squeeze(extra, drop=True).transpose("time").values.astype(float)
    months, years = dataset.time.dt.month.values, dataset.time.dt.year.values
    labels = years + (months == 12)
    weights = month_length_weights(dataset, calendar)
    selected = np.isin(months, months_by_season[season])
    kept_years, levels, totals = [], [], []
    for year in np.unique(labels[selected]):
        mask = selected & (labels == year)
        if (
            mask.sum() != 3
            or set(months[mask]) != set(months_by_season[season])
            or not np.isfinite(values[mask]).all()
        ):
            continue
        total = float(np.dot(values[mask], weights[mask]))
        kept_years.append(int(year))
        levels.append(total / weights[mask].sum())
        totals.append(total)
    return np.asarray(kept_years, dtype=int), np.asarray(levels), np.asarray(totals)


def get_change_seasonal_clim_proj(
    ds_hist_time,
    ds_clim_time,
    stats=None,
    calendar=None,
    variable_spec=None,
    *,
    min_season_years=None,
    reference_window_years=None,
    future_window_years=None,
):
    """Seasonal change rows from already-owned analysis slices, never outside them.

    Seasons use their middle month's calendar-year label and three complete,
    finite months. Both variables' levels are calendar-weighted mean rates.
    Rows retain year coordinates, rates and integrals for variability calculations
    and exact null-band masks. Dry-reference post-processing belongs to the
    analysis-table consumer; it does not change these reference/future values.
    Seasonal mean/std/detrended percentage changes use rates consistently
    (owner ruling 2026-10-07). The lab instead uses integrals for precipitation
    percentages, so those rows are excluded from lab parity when season lengths
    differ. Precipitation CV continues to use the period integrals.
    By default each period needs 90% of its owned window's years (rounded up),
    with at least three complete seasons. Supply window lengths when coordinate
    endpoints may be missing; otherwise their inclusive month span defines the
    window length. Missing values never reduce the required coverage. A numeric
    ``min_season_years`` overrides the threshold for both periods.
    """
    import numpy as np

    from blueearth_cst.projections.series_checks import assert_calendar_agreement

    if set(ds_hist_time.data_vars) != set(ds_clim_time.data_vars):
        raise ValueError("seasonal analysis has asymmetric reference/future variables")
    carried_calendar = assert_calendar_agreement(
        ds_hist_time, ds_clim_time, "seasonal analysis"
    )
    if calendar is not None and calendar != carried_calendar:
        from blueearth_cst.projections.calendar_weights import same_month_lengths

        earliest = (
            int(ds_hist_time.time.dt.year.values[0]),
            int(ds_hist_time.time.dt.month.values[0]),
        )
        if not same_month_lengths(calendar, carried_calendar, earliest):
            raise ValueError("seasonal calendar disagrees with series calendar")
    calendar = carried_calendar
    stats = ("mean", "std") if stats is None else tuple(stats)
    unknown = set(stats) - {"mean", "std", "std_detrended", "cv_detrended"}
    if unknown:
        raise ValueError(f"unsupported seasonal statistics: {sorted(unknown)}")
    if min_season_years is not None and min_season_years < 1:
        raise ValueError("min_season_years must be positive")

    def minimum(dataset, window_years):
        if min_season_years is not None:
            return min_season_years
        if window_years is None:
            months = dataset.time.dt.year.values * 12 + dataset.time.dt.month.values
            window_years = (months[-1] - months[0] + 1) / 12
        return seasonal_min_years(window_years)

    min_ref = minimum(ds_hist_time, reference_window_years)
    min_fut = minimum(ds_clim_time, future_window_years)
    rows = []
    for variable in ds_hist_time.data_vars:
        for season in ("DJF", "MAM", "JJA", "SON"):
            ref_years, ref_values, ref_integrals = _analysis_season_values(
                ds_hist_time, variable, season, calendar
            )
            fut_years, fut_values, fut_integrals = _analysis_season_values(
                ds_clim_time, variable, season, calendar
            )
            enough = len(ref_years) >= min_ref and len(fut_years) >= min_fut

            def statistic(values, integrals, years, name):
                if not values.size:
                    return float("nan")
                if name == "mean":
                    return float(values.mean())
                if name == "std":
                    return float(values.std(ddof=0))
                if name == "std_detrended":
                    return detrended_std(values, years)
                # Precipitation CV is computed on period integrals, as in the lab.
                data = (
                    integrals
                    if canonical_kind(variable_spec, variable) == "rate"
                    else values
                )
                mean = float(data.mean())
                return detrended_std(data, years) / mean if mean else float("nan")

            for stat in stats:
                reference = statistic(ref_values, ref_integrals, ref_years, stat)
                future = statistic(fut_values, fut_integrals, fut_years, stat)
                absolute = future - reference
                relative = (
                    absolute / reference * 100
                    if reference and np.isfinite(reference)
                    else float("nan")
                )
                rows.append(
                    {
                        "variable": variable,
                        "season": season,
                        "statistic": stat,
                        "reference": reference,
                        "future": future,
                        "absolute": absolute if enough else float("nan"),
                        "relative": relative if enough else float("nan"),
                        "change": (
                            relative
                            if change_kind(variable_spec, variable) == "relative"
                            else absolute
                        )
                        if enough
                        else float("nan"),
                        "n_reference": len(ref_years),
                        "n_future": len(fut_years),
                        "status": "ok" if enough else "insufficient_observations",
                        "reference_years": ref_years,
                        "future_years": fut_years,
                        "reference_values": ref_values,
                        "future_values": fut_values,
                        "reference_integrals": ref_integrals,
                        "future_integrals": fut_integrals,
                    }
                )
    return rows
