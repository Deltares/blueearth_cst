"""Analysis-window changes, equal-model reductions and variability benchmarks.

Stage B annual/monthly mean/std kernels are called, never reimplemented. New
OLS and CV statistics retain their actual year masks. Precipitation CV uses
duration integrals: constant Gregorian rates can therefore have nonzero CV
because years have365/366days. Seasonal percentages use rates (owner ruling),
which is a named exception to the lab's integral-percentage convention.
"""

from __future__ import annotations

import calendar as _calendar
import json
from functools import lru_cache
from statistics import NormalDist

import numpy as np
import pandas as pd

from blueearth_cst.projections.calendar_weights import month_length_weights
from blueearth_cst.projections.get_change_climate_proj import (
    AnalysisWindowUnavailable,
    analysis_slice,
    detrended_std,
    get_change_annual_clim_proj,
    get_change_monthly_clim_proj,
    get_change_seasonal_clim_proj,
    seasonal_min_years,
)
from blueearth_cst.projections.projection_coverage import family_metadata
from blueearth_cst.projections.projection_figures import FIGURE_FILENAMES, period_folder
from blueearth_cst.projections.series_checks import assert_calendar_agreement
from blueearth_cst.projections.variable_spec import VariableSpec

MIN_MODELS_IQR = 10
ZERO_VARIABILITY_RTOL = 1e-12
NULL_BAND_LEVEL = 0.90
NULL_SIMULATIONS = 20_000
NULL_SEED = 20261007
AGREEMENT_BANDS = {
    ("temp", "change"): 0.5,
    ("precip", "change_pct"): 5.0,
    ("precip", "change"): 0.1,
}
MONTHS = tuple(_calendar.month_abbr[1:])
PERIODS = ("annual", "DJF", "MAM", "JJA", "SON", *MONTHS)
MODEL_KEY = ["model", "institution", "source_id", "family", "family_source"]
WINDOW_KEY = [
    "window_id",
    "reference_start",
    "reference_end",
    "future_start",
    "future_end",
]
CELL = ["scenario", *WINDOW_KEY, "period", "variable", "statistic", "quantity", "units"]
ARRAY_COLUMNS = [
    "reference_years",
    "future_years",
    "reference_values",
    "future_values",
    "reference_integrals",
    "future_integrals",
]
CHANGE_COLUMNS = [
    *MODEL_KEY,
    "member",
    *CELL,
    "reference_value",
    "future_value",
    "value",
    "n_reference",
    "n_future",
    "n_reference_excluded",
    "n_future_excluded",
    "reference_mean",
    "future_mean",
    "reference_sd",
    "future_sd",
    "null_reference_mean",
    "null_reference_sd",
    "status",
    "reason",
    "mean_change_eligible",
    "variability_eligible",
    *ARRAY_COLUMNS,
]
MODEL_COLUMNS = [
    *MODEL_KEY,
    *CELL,
    "value",
    "n_members",
    "n_members_excluded",
    "members",
]
SUMMARY_COLUMNS = [
    *CELL,
    "n_models",
    "n_members",
    "n_models_excluded",
    "median",
    "min",
    "max",
    "q25",
    "q75",
    "iqr_shown",
]
RATIO_COLUMNS = [
    *MODEL_KEY,
    "member",
    *CELL,
    "reference_value",
    "future_value",
    "ratio",
    "n_reference",
    "n_future",
    "status",
    "reason",
    "variability_eligible",
    "null_reference_mean",
    "null_reference_sd",
    "reference_years",
    "future_years",
]


def _array(value):
    """Decode explicitly encoded numeric cells, also accepting in-memory arrays."""
    return np.asarray(json.loads(value) if isinstance(value, str) else value)


def _encode(values):
    return json.dumps(
        np.asarray(values).tolist(), separators=(",", ":"), allow_nan=False
    )


def period_values(dataset, variable, water_year_start="Jan"):
    """Rates, integrals and retained labels of complete finite period instances.

    The input already owns its month span; an edge season is never completed
    with outside data. Annual/missing-month instances are dropped explicitly,
    with excluded counts. This provides new OLS/CV inputs, not kernel mean/std.
    """
    month0 = list(_calendar.month_abbr).index(str(water_year_start)[:3].title())
    da = dataset[variable]
    extra = [dim for dim in da.dims if dim != "time"]
    if any(da.sizes[dim] != 1 for dim in extra):
        raise ValueError(f"analysis needs one member series: {variable} has {da.sizes}")
    values = da.squeeze(extra, drop=True).transpose("time").values.astype(float)
    years, months = dataset.time.dt.year.values, dataset.time.dt.month.values
    keys = years * 12 + months
    if np.any(np.diff(keys) <= 0):
        raise ValueError("analysis time must contain ordered unique months")
    weights = month_length_weights(dataset, dataset.attrs.get("cst_calendar", ""))
    seasons = {
        "DJF": (12, 1, 2),
        "MAM": (3, 4, 5),
        "JJA": (6, 7, 8),
        "SON": (9, 10, 11),
    }
    result = {}
    for period in PERIODS:
        if period == "annual":
            selected, labels, size = (
                np.ones(len(years), dtype=bool),
                years - (months < month0),
                12,
            )
        elif period in seasons:
            selected, labels, size = (
                np.isin(months, seasons[period]),
                years + (months == 12),
                3,
            )
        else:
            selected, labels, size = months == MONTHS.index(period) + 1, years, 1
        kept, level, total = [], [], []
        candidates = np.unique(labels[selected])
        for year in candidates:
            mask = selected & (labels == year)
            if mask.sum() != size or not np.isfinite(values[mask]).all():
                continue
            integral = (
                float(np.dot(values[mask], weights[mask]))
                if size > 1
                else float(values[mask][0])
            )
            kept.append(int(year))
            total.append(integral)
            level.append(integral / weights[mask].sum() if size > 1 else integral)
        result[period] = {
            "years": np.asarray(kept, dtype=int),
            "level": np.asarray(level),
            "total": np.asarray(total),
            "excluded": len(candidates) - len(kept),
        }
    return result


def _identity(identity):
    model = str(identity["model"])
    institution, sep, source_id = model.partition("/")
    if not sep:
        source_id, institution = model, str(identity.get("institution", ""))
    family = family_metadata(source_id, institution)
    return {
        "model": model,
        "institution": institution,
        "source_id": source_id,
        "family": identity.get("family", family["family"]),
        "family_source": identity.get("family_source", family["family_source"]),
        "scenario": str(identity["scenario"]),
        "member": str(identity["member"]),
    }


def _period_stats(values):
    level, years = values["level"], values["years"]
    return (float(level.mean()) if len(level) else np.nan, detrended_std(level, years))


def analysis_change_table(
    historical,
    future,
    *,
    identity,
    window_id,
    reference_window,
    future_window,
    water_year_start="Jan",
    variable_spec=None,
    min_reference=None,
    min_season_years=None,
):
    """One member's flat analysis rows, keyed by one distinct window_id.

    ``reference_window`` and ``future_window`` are used WY labels. Missing fixed
    span coverage produces explicit unavailable rows; corrupt structure/calendar
    still raises. Retained arrays are JSON numeric strings, not opaque objects.
    Annual precipitation kernel percentages retain integral semantics; seasonal
    percentages use rates per owner ruling. Absolute companions always use rates.
    Seasonal eligibility requires 90% of each owned window's years (rounded up,
    minimum three), or the same explicit ``min_season_years`` in both periods.
    """
    if set(historical.data_vars) != set(future.data_vars):
        raise ValueError("analysis has asymmetric historical/scenario variables")
    base = {
        **_identity(identity),
        "window_id": window_id,
        "reference_start": reference_window[0],
        "reference_end": reference_window[1],
        "future_start": future_window[0],
        "future_end": future_window[1],
    }
    variables = list(historical.data_vars)
    reference_window_years = reference_window[1] - reference_window[0] + 1
    future_window_years = future_window[1] - future_window[0] + 1
    min_ref_seasons = (
        seasonal_min_years(reference_window_years)
        if min_season_years is None
        else min_season_years
    )
    min_fut_seasons = (
        seasonal_min_years(future_window_years)
        if min_season_years is None
        else min_season_years
    )
    reason = ""
    try:
        hist = analysis_slice(historical, reference_window, water_year_start)
        fut = analysis_slice(
            future, future_window, water_year_start, historical=historical
        )
    except AnalysisWindowUnavailable as error:
        reason = str(error)
    supported = [variable for variable in variables if variable in {"temp", "precip"}]
    spec = {
        key: value if isinstance(value, VariableSpec) else VariableSpec(*value)
        for key, value in (variable_spec or {}).items()
    }
    calendar = (
        assert_calendar_agreement(hist, fut, base["model"] + " " + base["scenario"])
        if not reason
        else None
    )
    annual = monthly = None
    if supported and not reason:
        # Analysis reductions use float64, matching the lab's array precision;
        # the production Stage B kernels and source datasets remain unchanged.
        hist = hist.assign({name: hist[name].astype(np.float64) for name in supported})
        fut = fut.assign({name: fut[name].astype(np.float64) for name in supported})
        annual = get_change_annual_clim_proj(
            hist[supported],
            fut[supported],
            stats=["mean", "std"],
            start_month_hyd_year=water_year_start,
            calendar=calendar,
            variable_spec=spec,
        )
        monthly = get_change_monthly_clim_proj(
            hist[supported],
            fut[supported],
            stats=["mean", "std"],
            start_month_hyd_year=water_year_start,
            variable_spec=spec,
            min_reference=min_reference,
        )
        seasons = get_change_seasonal_clim_proj(
            hist[supported],
            fut[supported],
            min_season_years=min_season_years,
            reference_window_years=reference_window_years,
            future_window_years=future_window_years,
            variable_spec=spec,
        )
    rows = []
    for variable in variables:
        unavailable = reason or (
            "analysis covers precip/temp only" if variable not in supported else ""
        )
        units = (
            spec[variable].units
            if variable in spec
            else historical[variable].attrs.get(
                "units",
                "mm/day"
                if variable == "precip"
                else "degC"
                if variable == "temp"
                else "",
            )
        )
        if not unavailable:
            for dataset in (hist, fut):
                found = str(dataset[variable].attrs.get("units", ""))
                if found and found != units:
                    raise ValueError(
                        f"{base['model']}: {variable} units {found!r}, expected {units!r}"
                    )
            reference_periods, future_periods = (
                period_values(hist, variable, water_year_start),
                period_values(fut, variable, water_year_start),
            )
        for period in PERIODS:
            if unavailable:
                empty = {
                    "years": np.array([], dtype=int),
                    "level": np.array([]),
                    "total": np.array([]),
                    "excluded": 0,
                }
                rp = fp = empty
            else:
                rp, fp = reference_periods[period], future_periods[period]
            ref_mean, ref_sd = _period_stats(rp)
            fut_mean, fut_sd = _period_stats(fp)
            n_ref, n_fut = len(rp["years"]), len(fp["years"])
            for statistic in (
                "mean",
                "std",
                "std_detrended",
                *(("cv_detrended",) if variable == "precip" else ()),
            ):
                ref, value_fut, pct = np.nan, np.nan, np.nan
                null_values = (
                    rp["total"] if statistic == "cv_detrended" else rp["level"]
                )
                null_mean = float(null_values.mean()) if len(null_values) else np.nan
                null_sd = detrended_std(null_values, rp["years"])
                check_mean, check_future_mean = ref_mean, fut_mean
                check_ref_sd = (
                    float(rp["level"].std(ddof=0)) if len(rp["level"]) else np.nan
                )
                check_fut_sd = (
                    float(fp["level"].std(ddof=0)) if len(fp["level"]) else np.nan
                )
                if statistic == "std_detrended":
                    check_ref_sd, check_fut_sd = ref_sd, fut_sd
                if statistic == "cv_detrended":
                    check_mean = null_mean
                    check_future_mean = (
                        float(fp["total"].mean()) if len(fp["total"]) else np.nan
                    )
                    check_ref_sd = null_sd
                    check_fut_sd = detrended_std(fp["total"], fp["years"])
                if not unavailable:
                    if statistic in ("mean", "std"):
                        if period in ("annual", *MONTHS):
                            kernel = (
                                annual
                                if period == "annual"
                                else monthly.sel(month=MONTHS.index(period) + 1)
                            )
                            ref = float(
                                kernel[f"{variable}__reference"]
                                .sel(stats=statistic)
                                .values.item()
                            )
                            value_fut = float(
                                kernel[f"{variable}__level"]
                                .sel(stats=statistic)
                                .values.item()
                            )
                            pct = (
                                float(
                                    kernel[variable].sel(stats=statistic).values.item()
                                )
                                if variable == "precip"
                                else np.nan
                            )
                        else:
                            season = next(
                                item
                                for item in seasons
                                if item["variable"] == variable
                                and item["season"] == period
                                and item["statistic"] == statistic
                            )
                            ref, value_fut, pct = (
                                season["reference"],
                                season["future"],
                                season["relative"],
                            )
                    elif statistic == "std_detrended":
                        ref, value_fut = ref_sd, fut_sd
                        r_basis = (
                            detrended_std(rp["total"], rp["years"])
                            if variable == "precip" and period == "annual"
                            else ref
                        )
                        f_basis = (
                            detrended_std(fp["total"], fp["years"])
                            if variable == "precip" and period == "annual"
                            else value_fut
                        )
                        pct = (f_basis / r_basis - 1) * 100 if r_basis > 0 else np.nan
                    else:
                        f_mean = (
                            float(fp["total"].mean()) if len(fp["total"]) else np.nan
                        )
                        ref = null_sd / null_mean if null_mean > 0 else np.nan
                        value_fut = check_fut_sd / f_mean if f_mean > 0 else np.nan
                        pct = (value_fut / ref - 1) * 100 if ref > 0 else np.nan
                status = (
                    "unavailable"
                    if unavailable
                    else "insufficient_observations"
                    if (
                        (n_ref < min_ref_seasons or n_fut < min_fut_seasons)
                        if period in ("DJF", "MAM", "JJA", "SON")
                        else min(n_ref, n_fut) < 3
                    )
                    else "ok"
                )
                if status == "ok":
                    if check_ref_sd <= ZERO_VARIABILITY_RTOL * max(1, abs(check_mean)):
                        status = "zero_reference_variability"
                    elif check_fut_sd <= ZERO_VARIABILITY_RTOL * max(
                        1, abs(check_future_mean)
                    ):
                        status = "zero_future_variability"
                arrays = {
                    "reference_years": _encode(rp["years"]),
                    "future_years": _encode(fp["years"]),
                    "reference_values": _encode(rp["level"]),
                    "future_values": _encode(fp["level"]),
                    "reference_integrals": _encode(rp["total"]),
                    "future_integrals": _encode(fp["total"]),
                }
                for quantity in (
                    ("change", "change_pct") if variable == "precip" else ("change",)
                ):
                    value = value_fut - ref if quantity == "change" else pct
                    if status in ("unavailable", "insufficient_observations") or (
                        quantity == "change_pct"
                        and status == "zero_reference_variability"
                        and statistic != "mean"
                    ):
                        value = np.nan
                    rows.append(
                        {
                            **base,
                            "period": period,
                            "variable": variable,
                            "statistic": statistic,
                            "quantity": quantity,
                            "units": "%"
                            if quantity == "change_pct"
                            else "1"
                            if statistic == "cv_detrended"
                            else units,
                            "reference_value": ref,
                            "future_value": value_fut,
                            "value": value,
                            "n_reference": n_ref,
                            "n_future": n_fut,
                            "n_reference_excluded": rp["excluded"],
                            "n_future_excluded": fp["excluded"],
                            "reference_mean": ref_mean,
                            "future_mean": fut_mean,
                            "reference_sd": check_ref_sd,
                            "future_sd": check_fut_sd,
                            "null_reference_mean": null_mean,
                            "null_reference_sd": null_sd,
                            "status": status,
                            "reason": unavailable
                            or (
                                "fewer than required complete observations"
                                if status == "insufficient_observations"
                                else ""
                            ),
                            "mean_change_eligible": False,
                            "variability_eligible": False,
                            **arrays,
                        }
                    )
    return apply_dry_reference(
        pd.DataFrame(rows, columns=CHANGE_COLUMNS), min_reference
    )


def apply_dry_reference(table, min_reference=None):
    """Mean-keyed analysis dry rule; keep absolute changes and recover wet SD %."""
    result = table.copy()
    threshold = (min_reference or {}).get("precip", 0.1)
    valid = ~result.status.isin(["unavailable", "insufficient_observations"])
    precip = result.variable == "precip"
    dry = valid & precip & (result.reference_mean < threshold)
    dry_cv = (
        valid
        & precip
        & (result.statistic == "cv_detrended")
        & ((result.reference_mean < threshold) | (result.future_mean < threshold))
    )
    result.loc[dry | dry_cv, "status"] = "dry"
    result.loc[dry | dry_cv, "reason"] = (
        "precipitation period mean below dry-reference threshold"
    )
    result.loc[dry & (result.quantity == "change_pct") | dry_cv, "value"] = np.nan
    wet_recover = (
        valid
        & precip
        & ~dry
        & ~dry_cv
        & (result.quantity == "change_pct")
        & (result.reference_value > 0)
        & result.value.isna()
        & (result.status == "ok")
    )
    result.loc[wet_recover, "value"] = (
        result.loc[wet_recover, "future_value"]
        / result.loc[wet_recover, "reference_value"]
        - 1
    ) * 100
    result["mean_change_eligible"] = (
        valid
        & np.isfinite(result.value.astype(float))
        & ~(dry & (result.quantity == "change_pct"))
    )
    result["variability_eligible"] = (
        (result.status == "ok")
        & result.statistic.isin(["std_detrended", "cv_detrended"])
        & (result.reference_value > 0)
        & (result.future_value > 0)
        & np.isfinite(result.future_value.astype(float))
    )
    return result


def model_changes(table, statistic="mean"):
    """Arithmetic member means; excluded models remain as counted NaN rows."""
    table = table.loc[table.statistic == statistic] if statistic is not None else table
    rows = []
    for key, group in table.groupby([*MODEL_KEY, *CELL], dropna=False, sort=False):
        used = group.loc[group.mean_change_eligible.astype(bool)]
        rows.append(
            {
                **dict(zip([*MODEL_KEY, *CELL], key)),
                "value": float(used.value.mean()) if len(used) else np.nan,
                "n_members": used.member.nunique(),
                "n_members_excluded": group.member.nunique() - used.member.nunique(),
                "members": " ".join(sorted(set(used.member))),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def ensemble_summary(models, min_models_iqr=MIN_MODELS_IQR):
    """Median/range across equally weighted model means; IQR requires10 models."""
    rows = []
    for key, group in models.groupby(CELL, dropna=False, sort=False):
        ok = np.isfinite(group.value.astype(float))
        values = group.loc[ok, "value"].to_numpy(float)
        n = len(values)
        rows.append(
            {
                **dict(zip(CELL, key)),
                "n_models": n,
                "n_members": int(group.loc[ok, "n_members"].sum()),
                "n_models_excluded": int((~ok).sum()),
                "median": float(np.median(values)) if n else np.nan,
                "min": float(values.min()) if n else np.nan,
                "max": float(values.max()) if n else np.nan,
                "q25": float(np.quantile(values, 0.25))
                if n >= min_models_iqr
                else np.nan,
                "q75": float(np.quantile(values, 0.75))
                if n >= min_models_iqr
                else np.nan,
                "iqr_shown": n >= min_models_iqr,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def sign_agreement(models):
    """Model and family sign counts; small-change bands are display heuristics."""
    columns = [
        *CELL,
        "k",
        "n",
        "k_fam",
        "n_fam",
        "median",
        "median_sign",
        "fraction",
        "n_positive",
        "n_negative",
        "n_zero",
        "band",
        "small_change",
    ]
    rows = []
    for key, group in models.groupby(CELL, dropna=False, sort=False):
        group = group.loc[np.isfinite(group.value.astype(float))]
        values = group.value.to_numpy(float)
        median = float(np.median(values)) if len(values) else np.nan
        sign = int(np.sign(median)) if len(values) else 0
        votes = group.groupby("family", dropna=False).value.median().to_numpy(float)
        k = int((np.sign(values) == sign).sum()) if sign else 0
        k_fam = int((np.sign(votes) == sign).sum()) if sign else 0
        row = dict(zip(CELL, key))
        band = AGREEMENT_BANDS.get((row["variable"], row["quantity"]), np.nan)
        rows.append(
            {
                **row,
                "k": k,
                "n": len(values),
                "k_fam": k_fam,
                "n_fam": len(votes),
                "median": median,
                "median_sign": "+" if sign > 0 else "−" if sign < 0 else "0",
                "fraction": k / len(values) if len(values) and sign else np.nan,
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
                "n_zero": int((values == 0).sum()),
                "band": band,
                "small_change": bool(abs(median) < band)
                if np.isfinite(median) and np.isfinite(band)
                else False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def climatology(table):
    """Reference/future monthly mean rates, averaged across eligible members."""
    selected = table.loc[
        (table.statistic == "mean")
        & (table.quantity == "change")
        & table.period.isin(MONTHS)
    ]
    keys = [*MODEL_KEY, *[key for key in CELL if key not in ("statistic", "quantity")]]
    rows = []
    for key, group in selected.groupby(keys, dropna=False, sort=False):
        used = group.loc[group.mean_change_eligible.astype(bool)]
        rows.append(
            {
                **dict(zip(keys, key)),
                "reference": float(used.reference_value.mean())
                if len(used)
                else np.nan,
                "future": float(used.future_value.mean()) if len(used) else np.nan,
                "n_members": used.member.nunique(),
                "n_members_excluded": group.member.nunique() - used.member.nunique(),
            }
        )
    return pd.DataFrame(
        rows, columns=[*keys, "reference", "future", "n_members", "n_members_excluded"]
    )


def climatology_summary(table):
    keys = ["scenario", *WINDOW_KEY, "period", "variable", "units"]
    rows = []
    for key, group in table.groupby(keys, dropna=False, sort=False):
        for side in ("reference", "future"):
            values = group.loc[np.isfinite(group[side].astype(float)), side].to_numpy(
                float
            )
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "side": side,
                    "n_models": len(values),
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(
        rows, columns=[*keys, "side", "n_models", "median", "min", "max"]
    )


def monthly_change(table):
    models = model_changes(table.loc[table.period.isin(MONTHS)])
    return models, ensemble_summary(models)


def variability_band(changes, level=NULL_BAND_LEVEL):
    """Half-width of the mean change that interannual variability alone gives.

    Per member, z·σ·√(1/n_ref + 1/n_fut), with σ the detrended reference SD
    the null bands use (``null_reference_sd``) and, for ``change_pct``, as a
    percentage of the reference mean. Members average within their model and
    each cell takes the median across models: the historical runs' own noise,
    against which a projected mean change can be read. Independent years are
    assumed, so autocorrelation widens the true range.
    """
    rows = changes.loc[
        (changes.statistic == "mean")
        & changes.quantity.isin(["change", "change_pct"])
        & changes.mean_change_eligible.astype(bool)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=[*CELL, "noise_halfwidth"])
    z = NormalDist().inv_cdf(0.5 + level / 2)
    half = (
        z
        * rows.null_reference_sd.astype(float)
        * np.sqrt(1 / rows.n_reference.astype(float) + 1 / rows.n_future.astype(float))
    )
    pct = rows.quantity == "change_pct"
    half[pct] = half[pct] / rows.loc[pct, "null_reference_mean"].astype(float) * 100
    rows["noise_halfwidth"] = half
    per_model = (
        rows.groupby(["model", *CELL], dropna=False)
        .noise_halfwidth.mean()
        .reset_index()
    )
    return per_model.groupby(CELL, dropna=False).noise_halfwidth.median().reset_index()


def with_variability_band(summary, changes):
    """``summary`` with each cell's ``noise_halfwidth`` (NaN where none)."""
    band = variability_band(changes)
    return summary.drop(columns="noise_halfwidth", errors="ignore").merge(
        band, on=CELL, how="left"
    )


#: Calendar months of each season, and mean days per month (February
#: averaged over leap years), for turning monthly mean rates into totals.
SEASON_MONTHS = {
    "DJF": ("Dec", "Jan", "Feb"),
    "MAM": ("Mar", "Apr", "May"),
    "JJA": ("Jun", "Jul", "Aug"),
    "SON": ("Sep", "Oct", "Nov"),
}
MONTH_DAYS = dict(zip(MONTHS, (31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)))


def _seasons_with_historical(per, variable, statistic, units):
    """Scenario rows plus one ``historical`` row per model, window and season.

    ``per`` holds one row per model, scenario, window and season with
    ``reference`` and ``future`` values; the historical value averages the
    model's reference over its scenarios, which share its historical run.
    """
    historical = (
        per.groupby([*MODEL_KEY, *WINDOW_KEY, "period"], dropna=False)
        .agg(value=("reference", "mean"), n_members=("n_members", "max"))
        .reset_index()
        .assign(scenario="historical")
    )
    return pd.concat(
        [per.rename(columns={"future": "value"}).drop(columns="reference"), historical],
        ignore_index=True,
    ).assign(variable=variable, statistic=statistic, quantity="value", units=units)


def seasonal_values(climatology, changes):
    """Each model's seasonal climate per window, for the seasonal-values grids.

    Statistics: ``total`` precipitation (mm) and ``mean`` temperature (degC)
    from the monthly climatology (rates times days per month); ``sd`` of the
    detrended interannual series (precipitation as a season total in mm,
    temperature in degC) and precipitation ``cv`` (%) from the per-member
    changes, averaged over each model's variability-eligible members (NaN
    where none is eligible, so exclusions stay counted). Scenario rows hold
    the future window and ``historical`` rows the reference.
    """
    columns = [*MODEL_KEY, *CELL, "value", "n_members"]
    keys = [*MODEL_KEY, "scenario", *WINDOW_KEY, "period"]
    season_of = {m: s for s, months in SEASON_MONTHS.items() for m in months}
    season_days = {s: sum(MONTH_DAYS[m] for m in ms) for s, ms in SEASON_MONTHS.items()}
    parts = []
    monthly = climatology.loc[climatology.period.isin(MONTHS)]
    for variable, statistic, units in (
        ("precip", "total", "mm"),
        ("temp", "mean", "degC"),
    ):
        data = monthly.loc[monthly.variable == variable]
        if data.empty:
            continue
        data = data.assign(
            days=data.period.map(MONTH_DAYS), period=data.period.map(season_of)
        )
        weighted = data.assign(
            reference=data.reference * data.days, future=data.future * data.days
        ).groupby(keys, dropna=False)
        # min_count: a season missing any month's value is NaN, not a
        # partial (or zero) sum.
        per = (
            weighted[["reference", "future", "days"]]
            .sum(min_count=3)
            .join(weighted.n_members.min())
            .reset_index()
        )
        if statistic == "mean":
            per["reference"] /= per.days
            per["future"] /= per.days
        parts.append(
            _seasons_with_historical(
                per.drop(columns="days"), variable, statistic, units
            )
        )
    rows = changes.loc[
        (changes.quantity == "change")
        & changes.statistic.isin(["std_detrended", "cv_detrended"])
        & changes.period.isin(SEASON_MONTHS)
    ]
    for (variable, raw), statistic, units in (
        (("precip", "std_detrended"), "sd", "mm"),
        (("temp", "std_detrended"), "sd", "degC"),
        (("precip", "cv_detrended"), "cv", "%"),
    ):
        data = rows.loc[(rows.variable == variable) & (rows.statistic == raw)].copy()
        if data.empty:
            continue
        eligible = data.variability_eligible.astype(bool)
        for side in ("reference", "future"):
            data[side] = data[f"{side}_value"].astype(float).where(eligible)
        scale = (
            data.period.map(season_days)
            if (variable, statistic) == ("precip", "sd")
            else 100.0
            if statistic == "cv"
            else 1.0
        )
        data["reference"] *= scale
        data["future"] *= scale
        per = (
            data.assign(member_ok=eligible)
            .groupby(keys, dropna=False)
            .agg(
                reference=("reference", "mean"),
                future=("future", "mean"),
                n_members=("member_ok", "sum"),
            )
            .reset_index()
        )
        parts.append(_seasons_with_historical(per, variable, statistic, units))
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).reindex(columns=columns)


def member_ratios(table):
    """Retain all variability rows, including zero future and excluded ratios."""
    sub = table.loc[
        (table.quantity == "change")
        & table.statistic.isin(["std_detrended", "cv_detrended"])
    ].copy()
    sub["ratio"] = np.nan
    allowed = sub.status.isin(["ok", "zero_future_variability"]) & (
        sub.reference_value > 0
    )
    sub.loc[allowed, "ratio"] = (
        sub.loc[allowed, "future_value"] / sub.loc[allowed, "reference_value"]
    )
    sub.loc[sub.status == "zero_future_variability", "ratio"] = 0.0
    sub["quantity"], sub["units"] = "ratio", "1"
    return sub.reindex(columns=RATIO_COLUMNS).reset_index(drop=True)


def member_weights(ratios):
    """Each eligible member gets1/(eligible members in its model/cell)."""
    used = ratios.loc[
        ratios.variability_eligible.astype(bool)
        & (ratios.ratio > 0)
        & np.isfinite(ratios.ratio.astype(float))
    ].copy()
    used["weight"] = 1 / used.groupby(["model", *CELL], dropna=False).member.transform(
        "nunique"
    )
    return used


def _weighted_log_quantile(values, weights, quantiles):
    logs = np.log(np.asarray(values, float))
    unique, inverse = np.unique(logs, return_inverse=True)
    weight = np.bincount(inverse, weights=np.asarray(weights, float))
    positions = (np.cumsum(weight) - weight / 2) / weight.sum()
    return np.exp(np.interp(quantiles, positions, unique))


def ratio_summary(ratios, min_models_iqr=MIN_MODELS_IQR):
    """Equal-model pooled member quantiles on log ratios; no null band."""
    rows = []
    for key, group in ratios.groupby(CELL, dropna=False, sort=False):
        used = member_weights(group)
        n_models = used.model.nunique()
        quantiles = (
            _weighted_log_quantile(used.ratio, used.weight, [0.5, 0.25, 0.75])
            if len(used)
            else [np.nan] * 3
        )
        rows.append(
            {
                **dict(zip(CELL, key)),
                "n_models": n_models,
                "n_eligible": len(used),
                "n_excluded": len(group) - len(used),
                "n_zero_future": int((group.status == "zero_future_variability").sum()),
                "median": quantiles[0],
                "min": float(used.ratio.min()) if len(used) else np.nan,
                "max": float(used.ratio.max()) if len(used) else np.nan,
                "q25": quantiles[1] if n_models >= min_models_iqr else np.nan,
                "q75": quantiles[2] if n_models >= min_models_iqr else np.nan,
                "iqr_shown": n_models >= min_models_iqr,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            *CELL,
            "n_models",
            "n_eligible",
            "n_excluded",
            "n_zero_future",
            "median",
            "min",
            "max",
            "q25",
            "q75",
            "iqr_shown",
        ],
    )


def _null_parameters(members):
    """Canonical member parameters include exact retained masks, not counts."""
    result = []
    for member in members:
        years = []
        for column in ("reference_years", "future_years"):
            raw = _array(member[column])
            if (
                raw.ndim != 1
                or not np.isfinite(raw.astype(float)).all()
                or not np.equal(raw, np.asarray(raw, dtype=int)).all()
            ):
                raise ValueError("null-band years must be finite integer coordinates")
            coords = tuple(int(year) for year in raw)
            if tuple(sorted(set(coords))) != coords:
                raise ValueError("null-band years must be ordered and unique")
            years.append(coords)
        mean = float(member["reference_mean"])
        sd = float(member["reference_sd"])
        if not np.isfinite(mean) or not np.isfinite(sd) or sd < 0:
            raise ValueError(
                "null-band member mean/SD must be finite with nonnegative SD"
            )
        result.append((*years, mean, sd))
    return tuple(sorted(result))


def _draw_statistic(values, years, statistic):
    years = np.asarray(years, float)
    x = years - years.mean()
    mean = values.mean(axis=1)
    centred = values - mean[:, None]
    slope = (centred @ x) / np.dot(x, x)
    residual = centred - slope[:, None] * x
    sd = np.sqrt(np.mean(residual**2, axis=1))
    if statistic == "std_detrended":
        return sd, np.ones(len(sd), dtype=bool)
    valid = mean > 0
    cv = np.full(len(sd), np.nan)
    np.divide(sd, mean, out=cv, where=valid)
    return cv, valid


@lru_cache(maxsize=256)
def _cached_null_band(parameters, statistic, level, simulations, seed):
    if not parameters:
        return np.nan, np.nan, 0, simulations, "no_eligible_members"
    if any(min(len(ref), len(fut)) < 3 for ref, fut, _, _ in parameters):
        return np.nan, np.nan, 0, simulations, "insufficient_observations"
    if any(sd == 0 for _, _, _, sd in parameters):
        return np.nan, np.nan, 0, simulations, "zero_reference_variability"
    rng = np.random.default_rng(seed)
    log_sum = np.zeros(simulations)
    valid = np.ones(simulations, dtype=bool)
    for reference, future, mean, sd in parameters:
        union = np.union1d(reference, future)
        draws = rng.normal(mean, sd, size=(simulations, len(union)))
        ref, ref_valid = _draw_statistic(
            draws[:, np.searchsorted(union, reference)], reference, statistic
        )
        fut, fut_valid = _draw_statistic(
            draws[:, np.searchsorted(union, future)], future, statistic
        )
        ratio = np.full(simulations, np.nan)
        np.divide(fut, ref, out=ratio, where=ref > 0)
        usable = ref_valid & fut_valid & np.isfinite(ratio) & (ratio > 0)
        valid &= usable
        logs = np.zeros(simulations)
        np.log(ratio, out=logs, where=usable)
        log_sum += logs
    n_valid = int(valid.sum())
    if not n_valid:
        return np.nan, np.nan, 0, simulations, "no_valid_null_draws"
    ratio = np.exp(log_sum[valid] / len(parameters))
    lo, hi = np.quantile(ratio, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi), n_valid, simulations - n_valid, "ok"


def null_ratio_band(
    members,
    *,
    statistic="std_detrended",
    level=NULL_BAND_LEVEL,
    simulations=NULL_SIMULATIONS,
    seed=NULL_SEED,
):
    """Seeded exact-statistic white-noise band over actual retained-year unions.

    Each member retains its own reference mean/SD and year masks. Every draw
    reduces its member ratios to the same geometric mean as a model marker.
    CV uses integral mean/SD; SD uses rates. Nonpositive simulated CV means or
    undefined ratios exclude a draw with n_valid/n_invalid reported. Its band
    is explicitly conditional on positive means; no-valid-draws is unavailable,
    whereas malformed coordinates/parameters raise. Pooled quantiles never call
    this function and carry no band.
    """
    if statistic not in ("std_detrended", "cv_detrended"):
        raise ValueError(f"unsupported null statistic: {statistic}")
    if not 0 < level < 1 or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("null band requires0<level<1 and positive integer simulations")
    parameters = _null_parameters(members)
    if statistic == "cv_detrended" and any(mean <= 0 for _, _, mean, _ in parameters):
        raise ValueError("CV null member reference mean must be positive")
    lo, hi, valid, invalid, status = _cached_null_band(
        parameters, statistic, float(level), simulations, int(seed)
    )
    return {
        "null_lo": lo,
        "null_hi": hi,
        "n_valid": valid,
        "n_invalid": invalid,
        "status": status,
        "level": level,
        "simulations": simulations,
        "seed": seed,
        "statistic": statistic,
        "n_members": len(parameters),
        "n_reference": [len(ref) for ref, _, _, _ in parameters],
        "n_future": [len(fut) for _, fut, _, _ in parameters],
        "overlap": [len(set(ref) & set(fut)) for ref, fut, _, _ in parameters],
        "method": "Gaussian white noise; separate OLS; ddof0; positive-mean valid CV draws",
    }


def model_ratios(ratios, *, include_null=True):
    """Geometric eligible-member mean with its heterogeneous-member null band."""
    rows = []
    for key, group in ratios.groupby([*MODEL_KEY, *CELL], dropna=False, sort=False):
        used = member_weights(group)
        value = float(np.exp(np.log(used.ratio).mean())) if len(used) else np.nan
        parameters = [
            {
                "reference_years": item.reference_years,
                "future_years": item.future_years,
                "reference_mean": item.null_reference_mean,
                "reference_sd": item.null_reference_sd,
            }
            for item in used.itertuples(index=False)
        ]
        band = (
            null_ratio_band(parameters, statistic=str(group.statistic.iloc[0]))
            if include_null
            else {
                "null_lo": np.nan,
                "null_hi": np.nan,
                "n_valid": 0,
                "n_invalid": 0,
                "status": "not_computed",
            }
        )
        rows.append(
            {
                **dict(zip([*MODEL_KEY, *CELL], key)),
                "value": value,
                "log_value": float(np.log(value))
                if np.isfinite(value) and value > 0
                else np.nan,
                "n_members": used.member.nunique(),
                "n_members_excluded": group.member.nunique() - used.member.nunique(),
                "members": " ".join(sorted(set(used.member))),
                "n_eligible": len(used),
                "n_excluded": len(group) - len(used),
                "n_zero_future": int((group.status == "zero_future_variability").sum()),
                "null_lo": band["null_lo"],
                "null_hi": band["null_hi"],
                "null_n_valid": band["n_valid"],
                "null_n_invalid": band["n_invalid"],
                "null_status": band["status"],
                "outside_null": bool(value < band["null_lo"] or value > band["null_hi"])
                if np.isfinite(value) and np.isfinite(band["null_lo"])
                else False,
                "null_parameters": json.dumps(
                    [
                        {
                            **item,
                            "reference_years": _array(item["reference_years"]).tolist(),
                            "future_years": _array(item["future_years"]).tolist(),
                        }
                        for item in parameters
                    ],
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            *MODEL_COLUMNS,
            "log_value",
            "n_eligible",
            "n_excluded",
            "n_zero_future",
            "null_lo",
            "null_hi",
            "null_n_valid",
            "null_n_invalid",
            "null_status",
            "outside_null",
            "null_parameters",
        ],
    )


def member_null_bands(ratios):
    """Attach exact-statistic member bands for pure table-driven artists.

    Excluded rows remain explicit; their bands are unavailable rather than
    evaluated from invalid reference parameters.
    """
    result = ratios.copy()
    records = []
    for _, row in result.iterrows():
        if row.variability_eligible and np.isfinite(row.ratio) and row.ratio > 0:
            band = null_ratio_band(
                [
                    {
                        "reference_years": row.reference_years,
                        "future_years": row.future_years,
                        "reference_mean": row.null_reference_mean,
                        "reference_sd": row.null_reference_sd,
                    }
                ],
                statistic=row.statistic,
            )
        else:
            band = {
                "null_lo": np.nan,
                "null_hi": np.nan,
                "n_valid": 0,
                "n_invalid": NULL_SIMULATIONS,
                "status": "excluded_member",
            }
        records.append(
            {
                "null_lo": band["null_lo"],
                "null_hi": band["null_hi"],
                "null_n_valid": band["n_valid"],
                "null_n_invalid": band["n_invalid"],
                "null_status": band["status"],
            }
        )
    for column in (
        "null_lo",
        "null_hi",
        "null_n_valid",
        "null_n_invalid",
        "null_status",
    ):
        result[column] = [record[column] for record in records]
    return result


def product_availability(tables, windows=None):
    """Actual-data figure predicates, independent of the C0 coverage matrix.

    Any scenario satisfying a predicate makes the product available. Missing
    tables mean unavailable products; malformed nonempty tables raise rather
    than concealing corrupt inputs. Configured horizons sharing a window_id
    produce one set of products.
    """
    requirements = {
        "coverage": ["available"],
        "trajectories": ["model", "variable", "anomaly", "annual_rate"],
        "models": [
            "model",
            "scenario",
            "window_id",
            "period",
            "variable",
            "statistic",
            "quantity",
            "value",
        ],
        "climatology": [
            "model",
            "scenario",
            "window_id",
            "period",
            "variable",
            "reference",
            "future",
        ],
        "member_ratios": [
            "model",
            "member",
            "scenario",
            "window_id",
            "period",
            "variable",
            "statistic",
            "ratio",
            "variability_eligible",
        ],
        "model_ratios": [
            "model",
            "scenario",
            "window_id",
            "period",
            "variable",
            "statistic",
            "value",
        ],
    }
    data = {}
    for name, columns in requirements.items():
        table = tables.get(name, pd.DataFrame(columns=columns))
        if not table.empty and (missing := set(columns) - set(table.columns)):
            raise ValueError(f"{name} missing availability columns: {sorted(missing)}")
        data[name] = table.reindex(
            columns=list(dict.fromkeys([*table.columns, *columns]))
        )
    if windows is None:
        window_ids = sorted(
            set().union(
                *(set(t.window_id.dropna()) for t in data.values() if "window_id" in t)
            )
        )
    else:
        records = windows.values() if isinstance(windows, dict) else windows
        window_ids = sorted(
            {str(r["window_id"]) if isinstance(r, dict) else str(r) for r in records}
        )
    rows = []

    def add(filename, folder, window_id, available, predicate):
        rows.append(
            {
                "filename": FIGURE_FILENAMES[filename],
                "folder": folder,
                "window_id": window_id,
                "available": bool(available),
                "reason": "" if available else predicate,
            }
        )

    add(
        "availability.png",
        "full_period",
        "",
        data["coverage"].available.eq(True).any(),
        "no resolved members",
    )
    trajectory = data["trajectories"]
    for variable, label in (("temp", "temperature"), ("precip", "precipitation")):
        sub = trajectory.loc[trajectory.variable == variable]
        add(
            f"trajectory-{label}.png",
            "full_period",
            "",
            np.isfinite(sub.anomaly.astype(float)).any(),
            "no member with valid analysis reference",
        )
        add(
            f"annual-{label}.png",
            "full_period",
            "",
            np.isfinite(sub.annual_rate.astype(float)).any(),
            "no finite annual values",
        )
    cloud_windows = 0
    for window_id in window_ids:
        model = data["models"].loc[data["models"].window_id == window_id]
        finite = model.loc[np.isfinite(model.value.astype(float))]
        mean = finite.loc[finite.statistic == "mean"]
        for variable, quantity, label in (
            ("temp", "change", "temperature"),
            ("precip", "change_pct", "precipitation-pct"),
        ):
            sub = mean.loc[
                (mean.variable == variable)
                & (mean.quantity == quantity)
                & mean.period.isin(PERIODS[:5])
            ]
            add(
                f"change-{label}.png",
                period_folder(window_id),
                window_id,
                not sub.empty,
                "no eligible model changes",
            )
        climate = data["climatology"].loc[data["climatology"].window_id == window_id]
        for variable, label in (("temp", "temperature"), ("precip", "precipitation")):
            sub = climate.loc[
                (climate.variable == variable)
                & np.isfinite(climate.reference.astype(float))
                & np.isfinite(climate.future.astype(float))
                & climate.period.isin(MONTHS)
            ]
            complete = any(
                set(group.period) == set(MONTHS)
                for _, group in sub.groupby(["model", "scenario"])
            )
            add(
                f"seasonal-cycle-{label}.png",
                period_folder(window_id),
                window_id,
                complete,
                "no model with twelve reference and future months",
            )
            add(
                f"values-{label}-{'total' if variable == 'precip' else 'mean'}.png",
                period_folder(window_id),
                window_id,
                complete,
                "no model with twelve reference and future months",
            )
            monthly = mean.loc[
                (mean.variable == variable)
                & (
                    mean.quantity
                    == ("change_pct" if variable == "precip" else "change")
                )
                & mean.period.isin(MONTHS)
            ]
            add(
                f"monthly-change-{label}.png",
                period_folder(window_id),
                window_id,
                not monthly.empty,
                "no eligible monthly model changes",
            )
        ratios = data["model_ratios"].loc[data["model_ratios"].window_id == window_id]
        for variable, statistic, label in (
            ("temp", "std_detrended", "temperature-sd"),
            ("precip", "std_detrended", "precipitation-sd"),
            ("precip", "cv_detrended", "precipitation-cv"),
        ):
            sub = ratios.loc[
                (ratios.variable == variable)
                & (ratios.statistic == statistic)
                & ratios.period.isin(PERIODS[:5])
            ]
            add(
                f"values-{label}.png",
                period_folder(window_id),
                window_id,
                (np.isfinite(sub.value.astype(float)) & sub.value.gt(0)).any(),
                "no variability-eligible model",
            )
        members = data["member_ratios"].loc[
            data["member_ratios"].window_id == window_id
        ]
        members = members.loc[
            members.variability_eligible.eq(True)
            & np.isfinite(members.ratio.astype(float))
            & members.ratio.gt(0)
            & members.period.eq("annual")
            & (
                (members.variable.eq("temp") & members.statistic.eq("std_detrended"))
                | (
                    members.variable.eq("precip")
                    & members.statistic.isin(["std_detrended", "cv_detrended"])
                )
            )
        ]
        spread = any(
            group.member.nunique() >= 2
            for _, group in members.groupby(
                ["model", "scenario", "period", "variable", "statistic"]
            )
        )
        add(
            "variability-member-ratios.png",
            period_folder(window_id),
            window_id,
            spread,
            "no model with two eligible member ratios",
        )
        annual = mean.loc[mean.period == "annual"]
        temp = annual.loc[
            (annual.variable == "temp") & (annual.quantity == "change"),
            ["model", "scenario"],
        ]
        precip = annual.loc[
            (annual.variable == "precip") & (annual.quantity == "change_pct"),
            ["model", "scenario"],
        ]
        cloud = not temp.merge(precip, on=["model", "scenario"]).empty
        cloud_windows += int(cloud)
        add(
            "change-factor-cloud.png",
            period_folder(window_id),
            window_id,
            cloud,
            "no paired annual temperature and precipitation model changes",
        )
        agreement_cells = mean.loc[
            mean.period.isin(PERIODS[:5])  # annual and the seasons, as plotted
            & (
                (mean.variable.eq("temp") & mean.quantity.eq("change"))
                | (
                    mean.variable.eq("precip")
                    & mean.quantity.isin(("change", "change_pct"))
                )
            )
        ]
        agreement = any(
            group.model.nunique() >= 2
            for _, group in agreement_cells.groupby(
                ["scenario", "period", "variable", "statistic", "quantity"]
            )
        )
        add(
            "sign-agreement.png",
            period_folder(window_id),
            window_id,
            agreement,
            "fewer than two eligible models per cell",
        )
    add(
        "change-factor-cloud-combined.png",
        "full_period",
        "",
        cloud_windows >= 2,
        "fewer than two distinct cloud windows",
    )
    return pd.DataFrame(
        rows, columns=["filename", "folder", "window_id", "available", "reason"]
    )
