# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

import hydromt  # noqa: F401 -- registers the xarray .raster accessor (.raster.vars below)
import os
import pandas as pd
import xarray as xr

from blueearth_cst.projections import series_identity
from blueearth_cst.projections.calendar_weights import month_length_weights
from blueearth_cst.projections.variable_spec import canonical_kind, change_kind
from blueearth_cst.shared.snake_utils import log_row

# %%


def intersection(lst1, lst2):
    """Shared members of two sequences, in a DETERMINISTIC (sorted) order.

    ``sorted`` is load-bearing, not cosmetic. This function drives the variable
    iteration order in ``get_change_annual_clim_proj`` /
    ``get_change_scalar_clim_proj``, which becomes the **column order** of
    ``annual_change_scalar_stats_summary{,_mean}.csv``. Returning a bare
    ``list(set(...) & set(...))`` made that order depend on string hash
    randomization, so the summary CSVs flipped ``temp``/``precip`` between runs
    with no change in values — harmless to consumers, which read by label, but it
    made the two manifested CSV rows unreproducible (``check_baseline``
    fingerprints them by sha256 of the bytes).

    Alphabetical rather than "preserve the first sequence's order": it is
    self-contained, so the guarantee does not silently depend on upstream
    dataset-variable ordering. On the seed config the two coincide anyway —
    ``variables: [precip, temp]``.
    """
    return sorted(set(lst1) & set(lst2))


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


def get_change_clim_projections(ds_hist, ds_clim):
    """
    Parameters
    ----------
    ds_hist : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for historical climate simulation.
    ds_clim : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for projected climate data.

    Returns
    -------
    Writes netcdf files with mean monthly (12 maps) change for the grid.
    Also writes scalar mean monthly values averaged over the grid.

    Returns
    -------
    monthly_change_mean_grid : xarray dataset
        mean monthly change over the grid.
    monthly_change_mean_scalar : xarray dataset
        mean monthly change averaged over the grid.

    """
    ds = []
    for var in intersection(ds_hist.data_vars, ds_clim.data_vars):
        if var == "precip":
            # multiplicative for precip
            change = (
                (
                    ds_clim[var]
                    - ds_hist[var].sel(
                        scenario=ds_hist.scenario.values[0],
                    )
                )
                / ds_hist[var].sel(
                    scenario=ds_hist.scenario.values[0],
                )
                * 100
            )
        else:  # for temp
            # additive for temp
            change = ds_clim[var] - ds_hist[var].sel(
                scenario=ds_hist.scenario.values[0]
            )
        ds.append(change.to_dataset())

    monthly_change_mean_grid = xr.merge(ds)

    return monthly_change_mean_grid


def hydrological_year_bounds(ds_time, start_month_hyd_year="Jan"):
    """First/last timestamp of the COMPLETE hydrological years in ``ds_time``.

    Extracted verbatim from :func:`get_change_annual_clim_proj`, which recomputed
    it identically on every pass of its per-variable loop. One definition, because
    the composition record (migration step 4d) reports the *effective* reference
    window and must report the window the change factors actually used — a second
    implementation would eventually disagree with the numbers it annotates.

    The count is exact by construction rather than by date arithmetic: ``end`` is
    the start month of the final year minus one month, so the span is precisely
    ``last_year - first_year`` complete hydrological years.

    Returns ``(start, end, n_hydrological_years)``.
    """
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
    # dev/working/2026-07-30_wf2-5f-hydyear-offbyone.md). The previous form ended
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
        raise ValueError(
            f"no complete hydrological year (start month {start_month_hyd_year}) "
            f"between {data_start.date()} and {data_end.date()}. An empty "
            "reference would propagate as an empty denominator into every "
            "relative change factor."
        )
    start = year_start(first)
    end = year_start(last + 1) - pd.DateOffset(months=1)
    return start, end, n_years


#: Step 5d: what a run emits unless it opts into more. Today's eight statistics
#: were computed over a ~20-year window, which makes `q_90` "effectively the
#: second-highest of 20 values" (design §5.6) -- a number the window cannot
#: support, shipped as though it could.
DEFAULT_STATS = ("mean", "median", "std")


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


def get_change_annual_clim_proj(
    ds_hist_time,
    ds_clim_time,
    stats=None,
    start_month_hyd_year="Jan",
    calendar=None,
    variable_spec=None,
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
    # Step 5d: `stats=None` means the v2.0 default set, not "all eight".
    # Passed explicitly by callers that opt into tail quantiles.
    stats = list(DEFAULT_STATS) if stats is None else list(stats)

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
    def _annual(da, freq, how):
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

    ds = []
    for var in intersection(ds_hist_time.data_vars, ds_clim_time.data_vars):
        # Step 5e-iii: the AGGREGATION follows `canonical`, the change arithmetic
        # below follows `change`. Both used to hang off `var == "precip"`, so two
        # independent properties could never disagree.
        if canonical_kind(variable_spec, var) == "rate":
            # a rate integrates over the year
            hist = (
                _annual(
                    ds_hist_time[var].sel(
                        time=slice(start_hyd_year_hist, end_hyd_year_hist)
                    ),
                    f"YS-{start_month_hyd_year.upper()[:3]}",
                    "sum",
                ).sel(
                    scenario=ds_hist_time.scenario.values[0],
                )
            )
            clim = (
                _annual(
                    ds_clim_time[var].sel(
                        time=slice(start_hyd_year_clim, end_hyd_year_clim)
                    ),
                    f"YS-{start_month_hyd_year.upper()[:3]}",
                    "sum",
                )
            )
            # change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0])) / ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]) * 100
        else:  # for temp
            # additive for temp
            hist = (
                _annual(
                    ds_hist_time[var].sel(
                        time=slice(start_hyd_year_hist, end_hyd_year_hist)
                    ),
                    f"YS-{start_month_hyd_year.upper()[:3]}",
                    "mean",
                ).sel(
                    scenario=ds_hist_time.scenario.values[0],
                )
            )
            clim = (
                _annual(
                    ds_clim_time[var].sel(
                        time=slice(start_hyd_year_clim, end_hyd_year_clim)
                    ),
                    f"YS-{start_month_hyd_year.upper()[:3]}",
                    "mean",
                )
            )

        # calc statistics
        for stat_name in stats:  # , stat_props in stats_dic.items():
            if "q_" in stat_name:
                qvalue = int(stat_name.split("_")[1]) / 100
                hist_stat = getattr(hist, "quantile")(qvalue, "time")
                clim_stat = getattr(clim, "quantile")(qvalue, "time")
            else:
                hist_stat = getattr(hist, stat_name)("time")
                clim_stat = getattr(clim, stat_name)("time")

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

    stats_annual_change = xr.merge(ds)
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
