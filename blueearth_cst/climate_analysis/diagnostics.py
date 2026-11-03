"""Gap-preserving historical-climate calculations adapted from reviewed WF0 lab methods.

Pure pandas/numpy/scipy routines. Reporting years use start-year labels;
amounts may infill tolerated monthly gaps, observed extremes never do.
Ntoum exercises establish engineering parity, not scientific qualification.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

FRACTIONS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class Completeness:
    max_month: int = 0  # missing days tolerated per month
    max_year: int = 0


STRICT = Completeness()


SPI_SCALES = (3, 6, 12)


def reporting_year(index: pd.DatetimeIndex, m0: int) -> np.ndarray:
    return index.year - (index.month < m0).astype(int)


def in_years(index: pd.DatetimeIndex, m0: int, years: tuple[int, int]) -> np.ndarray:
    """Mask for dates whose reporting year lies in the inclusive range ``years``."""
    ry = reporting_year(index, m0)
    return (ry >= years[0]) & (ry <= years[1])


def month_order(m0: int) -> list[int]:
    """Calendar months in reporting-year order."""
    return [(m0 - 1 + k) % 12 + 1 for k in range(12)]


def full_years(p: pd.Series, m0: int) -> np.ndarray:
    """Reporting years whose every day lies inside the series index."""
    ry = reporting_year(p.index, m0)
    counts = pd.Series(1, index=p.index).groupby(ry).size()

    def days_in(y: int) -> int:
        return (
            pd.Timestamp(year=y + 1, month=m0, day=1)
            - pd.Timestamp(year=y, month=m0, day=1)
        ).days

    return np.array([y for y, n in counts.items() if n == days_in(y)])


def coverage(p: pd.Series, m0: int) -> pd.DataFrame:
    """Share of days present, rows = calendar months in reporting order, columns = reporting years."""
    ry = reporting_year(p.index, m0)
    frac = p.notna().groupby([p.index.month, ry]).mean().unstack()
    years = full_years(p, m0)
    return frac.reindex(index=month_order(m0), columns=years)


def missing_days(p: pd.Series, m0: int) -> pd.DataFrame:
    """Missing-day counts, rows = calendar months in reporting order, columns = reporting years."""
    ry = reporting_year(p.index, m0)
    miss = p.isna().groupby([p.index.month, ry]).sum().unstack()
    return miss.reindex(index=month_order(m0), columns=full_years(p, m0))


def coverage_class(p: pd.Series, m0: int, tol: Completeness = STRICT) -> pd.DataFrame:
    """Per month: 2 = complete, 1 = usable with tolerated gaps, 0 = unusable."""
    miss = missing_days(p, m0)
    return pd.DataFrame(
        np.where(miss == 0, 2, np.where(miss <= tol.max_month, 1, 0)),
        index=miss.index,
        columns=miss.columns,
    )


def valid_years(p: pd.Series, m0: int, tol: Completeness = STRICT) -> np.ndarray:
    miss = missing_days(p, m0)
    ok = (miss <= tol.max_month).all(axis=0) & (miss.sum(axis=0) <= tol.max_year)
    return miss.columns[ok].to_numpy()


def infill(p: pd.Series, tol: Completeness = STRICT) -> pd.Series:
    """Fill gaps in months missing at most ``tol.max_month`` days with that month's mean of present days."""
    if tol.max_month == 0:
        return p
    key = p.index.to_period("M")
    g = p.groupby(key)
    usable = (g.transform("size") - g.transform("count")) <= tol.max_month
    return p.fillna(g.transform("mean").where(usable))


def annual(
    p: pd.Series, m0: int, how: str = "sum", tol: Completeness = STRICT
) -> pd.Series:
    """Reporting-year aggregate of the infilled series; NaN for years that are not valid."""
    q = infill(p, tol)
    ry = reporting_year(q.index, m0)
    years = full_years(q, m0)
    out = getattr(q.groupby(ry), how)().reindex(years)
    out[~np.isin(years, valid_years(p, m0, tol))] = np.nan
    return out


def wet_day_stats(
    p: pd.Series, m0: int, wet_mm: float, tol: Completeness = STRICT
) -> pd.DataFrame:
    """Annual total (infilled), wet-day count (scaled to the full year) and SDII (observed days) per valid year."""
    ry = reporting_year(p.index, m0)
    wet = p.where(p >= wet_mm)
    days = p.groupby(ry).size()
    present = p.notna().groupby(ry).sum()
    df = pd.DataFrame(
        {
            "total": infill(p, tol).groupby(ry).sum(),
            "wet_days": wet.notna().groupby(ry).sum() * days / present,
            "sdii": wet.groupby(ry).sum() / wet.notna().groupby(ry).sum(),
        }
    )
    df = df.reindex(full_years(p, m0))
    df.loc[~df.index.isin(valid_years(p, m0, tol))] = np.nan
    return df


def agreement(
    series: dict[str, pd.Series], m0: int, wet_mm: float, tol: Completeness = STRICT
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean annual P, wet days and SDII per source over the years valid in all sources."""
    common = None
    for p in series.values():
        v = set(valid_years(p, m0, tol))
        common = v if common is None else common & v
    common = np.array(sorted(common))
    rows = {
        k: wet_day_stats(p, m0, wet_mm, tol).loc[common].mean()
        for k, p in series.items()
    }
    return pd.DataFrame(rows).T[["total", "wet_days", "sdii"]], common


def monthly_totals(p: pd.Series) -> pd.Series:
    """Calendar-month totals indexed by month start; NaN for incomplete months."""
    g = p.resample("MS")
    return g.sum().where(g.count() == g.size())


def monthly_climatology(p: pd.Series, m0: int, years: np.ndarray) -> pd.DataFrame:
    """Mean and 10th/90th percentile of monthly totals across the given reporting years."""
    mt = monthly_totals(p)
    mt = mt[np.isin(reporting_year(mt.index, m0), years)].dropna()
    g = mt.groupby(mt.index.month)
    return pd.DataFrame(
        {
            "mean": g.mean(),
            "p10": g.quantile(0.1),
            "p90": g.quantile(0.9),
            "n": g.size(),
        }
    ).reindex(month_order(m0))


def accumulation_timing(
    p: pd.Series, m0: int, fractions=FRACTIONS, tol: Completeness = STRICT
) -> pd.DataFrame:
    """Day of reporting year (1-based) when cumulative infilled rainfall first reaches each fraction; valid years only."""
    q = infill(p, tol)
    ry = reporting_year(q.index, m0)
    out = {}
    for y in valid_years(p, m0, tol):
        x = q[ry == y].to_numpy()
        if x.sum() <= 0:
            out[y] = [np.nan for _ in fractions]
            continue
        c = np.cumsum(x) / x.sum()
        out[y] = [int(np.searchsorted(c, f) + 1) for f in fractions]
    return pd.DataFrame(out, index=list(fractions)).T


def temperature_climatology(
    t: pd.Series, m0: int, years: np.ndarray, tol: Completeness = STRICT
) -> pd.DataFrame:
    g = t.resample("MS")
    mm = g.mean().where(g.size() - g.count() <= tol.max_month)
    mm = mm[np.isin(reporting_year(mm.index, m0), years)].dropna()
    grp = mm.groupby(mm.index.month)
    return pd.DataFrame(
        {"mean": grp.mean(), "p10": grp.quantile(0.1), "p90": grp.quantile(0.9)}
    ).reindex(month_order(m0))


def monthly_reference_diagnostics(
    monthly: pd.DataFrame, reference: tuple[int, int]
) -> tuple[pd.Series, list[dict]]:
    """Empirical midrank percentiles and actual support for each calendar month.

    Uses retained monthly totals before display clipping. Percentiles are
    100 * (count below + half the count equal) / reference count; ties therefore
    share a percentile, including zero-rainfall months. No fit or extrapolation.
    """
    percentiles = pd.Series(np.nan, index=monthly.index, dtype=float)
    support = []
    for month in range(1, 13):
        rows = monthly[monthly.calendar_month == month]
        ref = rows[rows.reporting_year.between(*reference) & np.isfinite(rows.total)]
        n = len(ref)
        support.append(
            {
                "calendar_month": month,
                "start": int(ref.reporting_year.min()) if n else None,
                "end": int(ref.reporting_year.max()) if n else None,
                "count": n,
            }
        )
        if n:
            values = np.sort(ref.total.to_numpy())
            valid = rows[np.isfinite(rows.total)]
            below = np.searchsorted(values, valid.total, side="left")
            through = np.searchsorted(values, valid.total, side="right")
            percentiles.loc[valid.index] = 50 * (below + through) / n
    return percentiles, support


def monthly_anomaly(p: pd.Series, m0: int, reference: tuple[int, int]) -> pd.DataFrame:
    """Monthly total minus the reference-period mean for that calendar month (mm); months x reporting years."""
    mt = monthly_totals(p)
    ref = mt[in_years(mt.index, m0, reference)]
    clim = ref.groupby(ref.index.month).mean()
    anom = mt - clim.reindex(mt.index.month).to_numpy()
    table = (
        anom.groupby([anom.index.month, reporting_year(anom.index, m0)])
        .first()
        .unstack()
    )
    return table.reindex(index=month_order(m0), columns=full_years(p, m0))


SPI_DISTRIBUTIONS = ("gamma", "pearson3")


def _fit_cdf(fit: np.ndarray, dist: str):
    """CDF fitted to reference accumulations; zero totals are a point mass (Stagge et al. 2015)."""
    q0 = float((fit <= 0).mean())
    pos = fit[fit > 0]
    if pos.size < 2 or not np.isfinite(pos).all() or np.ptp(pos) == 0:
        raise ValueError("insufficient or degenerate positive fitting accumulations")
    if dist == "gamma":
        a, _, b = stats.gamma.fit(pos, floc=0)

        def base(x):
            return stats.gamma.cdf(x, a, scale=b)
    elif dist == "pearson3":
        sk, loc, sc = stats.pearson3.fit(pos)

        def base(x):
            return stats.pearson3.cdf(x, sk, loc, sc)
    else:
        raise ValueError(
            f"SPI distribution must be one of {SPI_DISTRIBUTIONS}, got {dist!r}"
        )
    return lambda x: q0 + (1 - q0) * base(np.clip(x, 0, None))


def _to_spi(cdf: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))


def accumulations(p: pd.Series, scale: int) -> pd.Series:
    return monthly_totals(p).rolling(scale, min_periods=scale).sum()


def spi(
    p: pd.Series,
    scale: int,
    reference: tuple[int, int],
    m0: int = 1,
    dist: str = "gamma",
) -> pd.Series:
    """SPI on monthly totals: per calendar month, ``dist`` fitted on the reference years' accumulations."""
    if dist not in SPI_DISTRIBUTIONS:
        raise ValueError(f"Unknown SPI distribution: {dist}")
    acc = accumulations(p, scale)
    out = pd.Series(np.nan, index=acc.index)
    in_ref = in_years(acc.index, m0, reference)
    for m in range(1, 13):
        sel = acc.index.month == m
        fit = acc[sel & in_ref].dropna().to_numpy()
        if fit.size < 5:
            continue
        try:
            out[sel] = _to_spi(_fit_cdf(fit, dist)(acc[sel].to_numpy()))
        except (ValueError, FloatingPointError, stats.FitError):
            continue
        out[sel & acc.isna()] = np.nan
    return out


def spi_fit_checks(
    p: pd.Series, scale: int, reference: tuple[int, int], m0: int = 1
) -> pd.DataFrame:
    """Per calendar month, check the reference-period fits behind :func:`spi`.

    Columns: ``n`` fitted values; ``zeros`` share of zero totals; ``sw_gamma`` and
    ``sw_pearson3`` Shapiro-Wilk p-values of the reference SPI under each
    distribution (p < 0.05 flags a poor fit, Stagge et al. 2015); ``skew`` of the
    gamma SPI; ``max_abs`` largest |gamma SPI|; ``mom_diff`` largest
    |SPI(gamma MLE) - SPI(gamma method of moments)| over the reference values;
    ``p3_bound_hits`` months in the whole record outside the fitted Pearson III
    support (their SPI would be clipped, so Pearson III is unsafe there).
    """
    acc = accumulations(p, scale)
    in_ref = in_years(acc.index, m0, reference)
    rows = {}
    for m in range(1, 13):
        fit = acc[(acc.index.month == m) & in_ref].dropna().to_numpy()
        if fit.size < 5 or (fit > 0).sum() < 2 or np.ptp(fit[fit > 0]) == 0:
            rows[m] = {"n": fit.size}
            continue
        allx = acc[acc.index.month == m].dropna().to_numpy()
        try:
            c3 = _fit_cdf(fit, "pearson3")(allx)
            z = _to_spi(_fit_cdf(fit, "gamma")(fit))
            z3 = _to_spi(_fit_cdf(fit, "pearson3")(fit))
        except (ValueError, FloatingPointError, stats.FitError):
            rows[m] = {"n": fit.size}
            continue
        pos = fit[fit > 0]
        q0 = float((fit <= 0).mean())
        a_m, b_m = pos.mean() ** 2 / pos.var(ddof=1), pos.var(ddof=1) / pos.mean()
        z_mom = _to_spi(
            q0 + (1 - q0) * stats.gamma.cdf(np.clip(fit, 0, None), a_m, scale=b_m)
        )
        rows[m] = {
            "n": fit.size,
            "zeros": q0,
            "sw_gamma": float(stats.shapiro(z).pvalue),
            "sw_pearson3": float(stats.shapiro(z3).pvalue),
            "skew": float(stats.skew(z)),
            "max_abs": float(np.abs(z).max()),
            "mom_diff": float(np.abs(z - z_mom).max()),
            "p3_bound_hits": int(((c3 <= 1e-6) | (c3 >= 1 - 1e-6)).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").reindex(month_order(m0))


DROUGHT_SCALES = (3, 12)


def drought_event_table(
    s: pd.Series, onset: float = 0.0, peak: float = -1.0
) -> pd.DataFrame:
    """McKee et al. (1993) events: runs of consecutive months with SPI < ``onset`` reaching <= ``peak``.

    Columns: ``start``/``end`` (month starts), ``duration`` (months), ``severity``
    (accumulated deficit, -sum SPI), ``intensity`` (lowest SPI) and ``censored``
    (the run touches a gap or a record end, so its true duration is unknown).
    Runs are not merged across single non-drought months.
    """
    vals, idx = s.to_numpy(), s.index
    rows, i = [], 0
    while i < vals.size:
        if np.isfinite(vals[i]) and vals[i] < onset:
            j = i
            while (
                j + 1 < vals.size and np.isfinite(vals[j + 1]) and vals[j + 1] < onset
            ):
                j += 1
            run = vals[i : j + 1]
            if run.min() <= peak:
                cens = (
                    i == 0
                    or j == vals.size - 1
                    or not np.isfinite(vals[i - 1])
                    or not np.isfinite(vals[j + 1])
                )
                rows.append(
                    {
                        "start": idx[i],
                        "end": idx[j],
                        "duration": j - i + 1,
                        "severity": float(-run.sum()),
                        "intensity": float(run.min()),
                        "censored": bool(cens),
                    }
                )
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(
        rows, columns=["start", "end", "duration", "severity", "intensity", "censored"]
    )


def extremes(
    p: pd.Series, m0: int, wet_mm: float, tol: Completeness = STRICT
) -> pd.DataFrame:
    """Rx1day, Rx5day (gap-free windows within one reporting year), SDII and wet days per valid reporting year."""
    ry = reporting_year(p.index, m0)
    rx5 = p.groupby(ry).transform(lambda x: x.rolling(5, min_periods=5).sum())
    df = pd.DataFrame({"Rx1day": p.groupby(ry).max(), "Rx5day": rx5.groupby(ry).max()})
    ws = wet_day_stats(p, m0, wet_mm, tol)
    df = df.reindex(ws.index)
    df["SDII"], df["Wet days"] = ws["sdii"], ws["wet_days"]
    df.loc[ws["total"].isna()] = np.nan
    return df


def dry_spells(p: pd.Series, wet_mm: float) -> np.ndarray:
    """Lengths of complete dry spells (days < wet_mm); spells touching gaps or record ends are dropped as censored."""
    x = p.to_numpy()
    lengths, run, start_ok = [], 0, False
    for k, v in enumerate(x):
        if not np.isfinite(v):
            run, start_ok = 0, False
        elif v < wet_mm:
            run += 1
        else:
            if run and start_ok:
                lengths.append(run)
            run, start_ok = 0, True
    return np.array(lengths, dtype=int)


def exceedance(lengths: np.ndarray, max_len: int | None = None) -> pd.Series:
    if lengths.size == 0:
        return pd.Series(dtype=float)
    xs = np.arange(1, (max_len or lengths.max()) + 1)
    return pd.Series([(lengths >= x).mean() for x in xs], index=xs)


def anomaly_acf(
    p: pd.Series, reference: tuple[int, int], m0: int = 1, lags: int = 12
) -> pd.DataFrame:
    """Lagged Pearson correlation of monthly anomalies using only pairs where both months are complete."""
    mt = monthly_totals(p)
    ref = mt[in_years(mt.index, m0, reference)]
    anom = (
        mt - ref.groupby(ref.index.month).mean().reindex(mt.index.month).to_numpy()
    ).to_numpy()
    rows = []
    for k in range(1, lags + 1):
        a, b = anom[:-k], anom[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        r = (
            float(np.corrcoef(a[ok], b[ok])[0, 1])
            if ok.sum() > 2 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0
            else np.nan
        )
        rows.append((k, r, int(ok.sum())))
    return pd.DataFrame(rows, columns=["lag", "r", "pairs"]).set_index("lag")


def _mk_s(x: np.ndarray) -> float:
    d = np.sign(x[None, :] - x[:, None])
    return float(np.triu(d, 1).sum())


def _mk_var0(x: np.ndarray) -> float:
    """Mann-Kendall variance of S under independence, with tie correction."""
    n = x.size
    _, t = np.unique(x, return_counts=True)
    return (n * (n - 1) * (2 * n + 5) - np.sum(t * (t - 1) * (2 * t + 5))) / 18.0


def hamed_rao_factor(resid: np.ndarray, alpha: float = 0.05) -> float:
    """Hamed & Rao (1998) variance inflation n/n* from significant rank autocorrelations of the detrended series."""
    n = resid.size
    r = stats.rankdata(resid) - (n + 1) / 2
    denom = np.sum(r * r)
    bound = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    total = 0.0
    for k in range(1, n - 2):
        rho = np.sum(r[:-k] * r[k:]) / denom
        if abs(rho) > bound:
            total += (n - k) * (n - k - 1) * (n - k - 2) * rho
    return 1 + 2 / (n * (n - 1) * (n - 2)) * total


def yue_wang_factor(resid: np.ndarray) -> float:
    """Yue & Wang (2004) lag-1 effective-sample-size factor n/n* from the detrended series.

    The lag-1 autocorrelation is bias-corrected for short records
    (r1 + (1 + 3 r1) / n, Kendall 1954) and capped at 0.95.
    """
    n = resid.size
    if np.ptp(resid) == 0:
        return 1.0  # exact fit: no residual variation to correlate
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    r1 = min(r1 + (1 + 3 * r1) / n, 0.95)
    k = np.arange(1, n)
    return 1 + 2 * float(np.sum((1 - k / n) * r1**k))


TREND_CORRECTIONS = ("yue_wang", "hamed_rao", "none")


def sen_trend(
    series: pd.Series, alpha: float = 0.05, correction: str = "yue_wang"
) -> dict:
    """Theil-Sen slope per decade with a Mann-Kendall test and Sen interval.

    ``correction`` inflates the Mann-Kendall variance for serial dependence in the
    Sen-detrended series: ``yue_wang`` (default; lag-1 effective sample size with
    bias-corrected r1), ``hamed_rao`` (1998; significant rank autocorrelations) or
    ``none``. The same variance sets the Sen (Gilbert 1987) rank interval, so
    dependence widens the interval and raises ``p``. In a Monte Carlo on AR(1)
    series (phi 0.3-0.5) Yue-Wang held the 5 % false-positive rate at 0.06-0.09
    and 95 % coverage at 0.91-0.94 for n = 31-51; Hamed-Rao reached only 0.13-0.20
    and 0.80-0.87, because its significance screen ignores most short-record lags.
    Returns ``slope``/``lo``/``hi`` per decade, ``intercept`` (per-year line
    through the median), ``p`` two-sided, ``r1`` lag-1 autocorrelation of the
    detrended series, ``factor`` (n/n*, floored at 1 so dependence never narrows
    the interval) and ``n``. Gaps are dropped; lags then span the missing years.
    """
    s = series.dropna()
    n = int(s.size)
    out = {
        "slope": np.nan,
        "lo": np.nan,
        "hi": np.nan,
        "intercept": np.nan,
        "p": np.nan,
        "r1": np.nan,
        "factor": np.nan,
        "n": n,
    }
    if n < 5:
        return out
    t, x = s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)
    i, j = np.triu_indices(n, 1)
    pair = (x[j] - x[i]) / (t[j] - t[i])
    pair.sort()
    slope = float(np.median(pair))
    intercept = float(np.median(x - slope * t))
    resid = x - slope * t
    if correction not in TREND_CORRECTIONS:
        raise ValueError(
            f"correction must be one of {TREND_CORRECTIONS}, got {correction!r}"
        )
    raw = {
        "yue_wang": yue_wang_factor,
        "hamed_rao": hamed_rao_factor,
        "none": lambda r: 1.0,
    }[correction](resid)
    factor = max(1.0, raw)
    var = _mk_var0(x) * factor
    S = _mk_s(x)
    z = (S - np.sign(S)) / np.sqrt(var) if S != 0 else 0.0
    c = stats.norm.ppf(1 - alpha / 2) * np.sqrt(var)
    m1, m2 = (pair.size - c) / 2, (pair.size + c) / 2  # 1-based ranks (Gilbert 1987)
    rank = np.arange(1, pair.size + 1)
    lo = float(np.interp(m1, rank, pair))
    hi = float(np.interp(m2 + 1, rank, pair))
    r1 = (
        float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if np.ptp(resid) > 0 else np.nan
    )
    out.update(
        slope=slope * 10,
        lo=lo * 10,
        hi=hi * 10,
        intercept=intercept,
        p=float(2 * stats.norm.sf(abs(z))),
        r1=r1,
        factor=factor,
    )
    return out


def start_year_sensitivity(
    series: pd.Series, min_years: int = 15, **kw
) -> pd.DataFrame:
    """Sen trend for every start year with at least ``min_years`` valid years to the series end."""
    s = series.dropna()
    rows = {}
    for start in s.index:
        sub = s[s.index >= start]
        if sub.size < min_years:
            break
        t = sen_trend(sub, **kw)
        rows[int(start)] = {k: t[k] for k in ("slope", "lo", "hi", "p", "n")}
    return pd.DataFrame.from_dict(rows, orient="index")


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def _season_frame(monthly: pd.Series) -> pd.DataFrame:
    season = {m: name for name, months in SEASONS.items() for m in months}
    df = pd.DataFrame({"v": monthly.to_numpy()}, index=monthly.index)
    df["season"] = [season[m] for m in monthly.index.month]
    df["year"] = monthly.index.year + (
        monthly.index.month == 12
    )  # December counts toward next year's DJF
    return df


def pt_anomalies(
    p: pd.Series, t: pd.Series, reference: tuple[int, int], tol: Completeness = STRICT
) -> pd.DataFrame:
    """Seasonal precipitation anomaly (% of reference mean) and temperature anomaly (degC) per season and year.

    Seasons need all three months: precipitation from complete (infilled) monthly
    totals, temperature from monthly means missing at most ``tol.max_month`` days.
    The two variables may come from different sources; callers must label them.
    """
    pm = monthly_totals(infill(p, tol))
    g = t.resample("MS")
    tm = g.mean().where(g.size() - g.count() <= tol.max_month)
    ps = _season_frame(pm).groupby(["season", "year"])["v"].agg(["sum", "count"])
    ts = _season_frame(tm).groupby(["season", "year"])["v"].agg(["mean", "count"])
    df = pd.DataFrame(
        {
            "p": ps["sum"].where(ps["count"] == 3),
            "t": ts["mean"].where(ts["count"] == 3),
        }
    ).dropna()
    yrs = df.index.get_level_values("year")
    ref = (
        df[(yrs >= reference[0]) & (yrs <= reference[1])].groupby(level="season").mean()
    )
    s = df.index.get_level_values("season")
    df["p_anom_pct"] = 100 * (
        df["p"] / ref.reindex(s)["p"].replace(0, np.nan).to_numpy() - 1
    )
    df["t_anom"] = df["t"] - ref.reindex(s)["t"].to_numpy()
    return df.reset_index()


def pettitt(series: pd.Series) -> dict:
    """Pettitt (1979) change-point test: most likely break year (last year before it) and approximate p."""
    s = series.dropna()
    x, n = s.to_numpy(float), s.size
    sgn = np.sign(x[None, :] - x[:, None])  # sgn[i, j] = sign(x_j - x_i)
    u = np.array([sgn[: k + 1, k + 1 :].sum() for k in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    kk = float(abs(u[k]))
    p = min(1.0, 2 * np.exp(-6 * kk**2 / (n**3 + n**2)))
    return {"break_after": int(s.index[k]), "stat": kk, "p": p, "n": n}


def snht(series: pd.Series, n_sim: int = 20000, seed: int = 0) -> dict:
    """Alexandersson (1986) standard normal homogeneity test for one shift in mean; p by Monte Carlo at this n."""
    s = series.dropna()
    x, n = s.to_numpy(float), s.size

    def t_max(v: np.ndarray) -> tuple[float, int]:
        z = (v - v.mean(axis=-1, keepdims=True)) / v.std(axis=-1, ddof=1, keepdims=True)
        c = np.cumsum(z, axis=-1)[..., :-1]
        k = np.arange(1, n)
        t = (
            k * (c / k) ** 2 + (n - k) * (c / (n - k)) ** 2
        )  # sum of z after k equals -c
        return t.max(axis=-1), t.argmax(axis=-1)

    t, k = t_max(x)
    sims, _ = t_max(np.random.default_rng(seed).normal(size=(n_sim, n)))
    return {
        "break_after": int(s.index[int(k)]),
        "stat": float(t),
        "p": float((sims >= t).mean()),
        "n": n,
    }
