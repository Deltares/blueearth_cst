# Current guidance: dev/reference/wf1-model-construction.md
# -*- coding: utf-8 -*-
"""Discharge performance indicators and the basin-average output figures.

Created 2021-07-14 (@author: bouaziz).

What this file NO LONGER holds, since 2026-08-10: ``plot_hydro`` and
``plot_signatures``. The four evaluation sheets they became live in
``shared/plot_evaluation.py``, beside the page settings they are drawn under —
this module's own private copy of those settings (``_FIG_WIDTH_MM``,
``_CLIM_RC``, ``_PT_BODY``) went with them, along with the Theil-Sen trend
helpers that served the retired subcatchment climate plots. All of it
was reachable from nowhere.

What is left is the part that is not about drawing: ``compute_indicators``, which
writes ``performance_indicators.csv`` and which the performance sheet RENDERS
rather than recomputes, and ``plot_basavg``, whose figures are a separate,
config-derivable family with their own declared outputs.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from hydromt.stats import skills

from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.shared import plot_style
from blueearth_cst.shared.snake_utils import save_figure

# %%
# Supported wflow outputs. The resample rule and axis legend live beside the
# code map they are keyed by, in shared/wflow_outputs.py — one table rather than
# two that had to be kept in agreement by hand.
from blueearth_cst.shared.wflow_outputs import (  # noqa: E402
    PLOT_META,
    code_from_variable,
)


def rsquared(x, y):
    """Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2


def compute_indicators(
    qsim: xr.DataArray,
    qobs: xr.DataArray,
    station_name: str = "station",
) -> pd.DataFrame:
    """
    Compute indicators on concurrent finite, nonnegative discharge pairs.

    Monthly values are available-day means of the same daily pairs, without
    infilling or a completeness threshold. Undefined denominators yield NaN.
    Calculated indicators for daily and monthly timeseries are:
    - Nash-Sutcliffe efficiency (NSE)
    - Nash-Sutcliffe efficiency on log-transformed data (NSElog)
    - Kling-Gupta efficiency (KGE)
    - Root mean squared error (RMSE)
    - Mean squared error (MSE)
    - Percentual bias (Pbias)
    - Volumetric error (VE)

    Parameters
    ----------
    qsim : xr.DataArray
        Dataset with simulated streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    qobs : xr.DataArray
        Dataset with observed streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    station_name : str, optional
        Station name, by default "station"

    Returns
    -------
    pd.DataFrame
        Dataframe with performance indicators for this station.
    """
    from blueearth_cst.shared.plot_evaluation import _performance_pairs

    pairs, support = _performance_pairs(qsim, qobs)
    indicators = ["KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE"]
    da_perf = xr.DataArray(
        np.full((len(indicators), 2), np.nan),
        coords=[indicators, ["daily", "monthly"]],
        dims=["indicators", "time_type"],
    )
    daily = pairs.dropna()
    monthly = pairs.resample("ME").mean().dropna() if len(pairs) else pairs
    for time_type, table in (("daily", daily), ("monthly", monthly)):
        if table.empty:
            continue
        sim = xr.DataArray(
            table["simulated"].to_numpy(), dims="time", coords={"time": table.index}
        )
        obs = xr.DataArray(
            table["observed"].to_numpy(), dims="time", coords={"time": table.index}
        )
        # Errors remain defined for valid zero-flow pairs. Efficiencies whose
        # observed denominator is zero remain NaN; HydroMT's logarithmic NSE
        # zero-flow convention is retained without introducing a new epsilon.
        scores = {"RMSE": skills.rmse(sim, obs), "MSE": skills.mse(sim, obs)}
        if float(obs.sum()) > 0:
            scores.update(
                Pbias=skills.percentual_bias(sim, obs),
                VE=skills.volumetric_error(sim, obs),
            )
        if len(table) >= 2 and float(obs.std()) > 0:
            scores.update(
                NSE=skills.nashsutcliffe(sim, obs),
                NSElog=skills.lognashsutcliffe(sim, obs),
            )
            if float(sim.std()) > 0:
                scores["KGE"] = skills.kge(sim, obs)["kge"]
        for indicator, score in scores.items():
            da_perf.loc[dict(indicators=indicator, time_type=time_type)] = score
    result = da_perf.to_dataframe(name=station_name)
    result.attrs["comparison_support"] = support
    return result


def _basavg_daily(series, cell_counts=None):
    """Fixed cell-count combination of disjoint equal-cell zonal means."""
    index = pd.DatetimeIndex(series.time.values)
    if (
        index.empty
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError(
            "Outputs require a nonempty, unique, increasing midnight daily index"
        )
    if set(series.dims) - {"time", "index"}:
        raise ValueError("Outputs require time and optionally index dimensions")
    if "index" not in series.dims:
        if cell_counts is not None:
            raise ValueError("Cell-count weights require explicit subbasin ids")
        return pd.Series(series.values, index=index, dtype=float), {}, {}
    ids = list(series["index"].values)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Subbasin ids must be unique and nonempty")
    ids = sorted(ids)
    if cell_counts is None:
        if len(ids) > 1:
            raise ValueError("Multiple subbasins require exact cell-count weights")
        cell_counts = {ids[0]: 1}
    if set(cell_counts) != set(ids):
        raise ValueError("Cell-count weights must match the exact subbasin id coverage")
    weights = np.asarray([cell_counts[key] for key in ids], dtype=float)
    if (
        not np.isfinite(weights).all()
        or (weights <= 0).any()
        or any(isinstance(cell_counts[key], (bool, np.bool_)) for key in ids)
    ):
        raise ValueError("Cell-count weights must be positive finite numbers")
    table = pd.DataFrame(
        series.sel(index=ids).transpose("time", "index").values,
        index=index,
        columns=ids,
        dtype=float,
    )
    table = table.where(np.isfinite(table))
    combined = (table * (weights / weights.sum())).sum(axis=1, min_count=len(ids))
    return (
        combined,
        {int(key): float(value) for key, value in zip(ids, weights)},
        {int(key): table[key] for key in ids},
    )


def _basavg_statistics(daily, kind):
    """Calendar completeness and WF0-equivalent monthly amount adjustment."""
    from blueearth_cst.climate_analysis.diagnostic_settings import DEFAULTS

    calendar = pd.date_range(daily.index[0], daily.index[-1], freq="D")
    clean = daily.where(np.isfinite(daily)).reindex(calendar)
    groups = clean.resample("MS")
    monthly_mean = groups.mean()
    counts = groups.count()
    days = monthly_mean.index.days_in_month
    gaps = pd.Series(days, index=monthly_mean.index) - counts
    full = (monthly_mean.index >= calendar[0]) & (
        (monthly_mean.index + pd.offsets.MonthEnd()) <= calendar[-1]
    )
    eligible = full & (gaps <= DEFAULTS["max_missing_days_month"])
    monthly = (monthly_mean * days if kind == "amount" else monthly_mean).where(
        eligible
    )
    years = np.arange(calendar[0].year, calendar[-1].year + 1)
    annual = pd.Series(np.nan, index=years, dtype=float)
    partial_years, invalid_years = [], []
    for year in years:
        start, end = pd.Timestamp(int(year), 1, 1), pd.Timestamp(int(year), 12, 31)
        if start < calendar[0] or end > calendar[-1]:
            partial_years.append(int(year))
            continue
        selected = monthly_mean.index.year == year
        valid = (
            eligible[selected].all()
            and gaps[selected].sum() <= DEFAULTS["max_missing_days_year"]
        )
        if valid:
            annual.loc[year] = (
                monthly[selected].sum(min_count=12)
                if kind == "amount"
                else clean.loc[start:end].mean()
            )
        else:
            invalid_years.append(int(year))
    grouped = monthly.groupby(monthly.index.month)
    climatology = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "p10": grouped.quantile(0.1),
            "p90": grouped.quantile(0.9),
        }
    ).reindex(range(1, 13))
    info = {
        "period": [str(calendar[0].date()), str(calendar[-1].date())],
        "omitted_dates": int(len(calendar.difference(daily.index))),
        "missing_days": int(clean.isna().sum()),
        "negative_days_preserved": int((clean < 0).sum()),
        "partial_years": partial_years,
        "incomplete_years": invalid_years,
        "partial_months": monthly.index[~full].strftime("%Y-%m").tolist(),
        "incomplete_months": monthly.index[full & ~eligible].strftime("%Y-%m").tolist(),
        "eligible_month_counts": {
            str(int(key)): int(value) for key, value in grouped.count().items()
        },
        "max_missing_days_month": DEFAULTS["max_missing_days_month"],
        "max_missing_days_year": DEFAULTS["max_missing_days_year"],
    }
    return annual, climatology, info


def plot_basavg(
    ds,
    Folder_out,
    fs=10,
    *,
    cell_counts=None,
    subbasin_plots=False,
    warmup_excluded=False,
    annotation="none",
):
    """Draw weighted basin annual/seasonal summaries and optional individual plots.

    The caller supplies disjoint zonal means, native cell counts and an already
    warm-up-filtered record. This function never removes another warm-up year.
    ``fs`` is retained for compatibility; canonical publication typography applies.
    """
    import json
    from pathlib import Path

    from blueearth_cst.climate_analysis import diagnostics as dg
    from blueearth_cst.plotting import fundamentals
    from blueearth_cst.plotting import prepared_charts as pc
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared import plot_preset as cst
    from blueearth_cst.shared.plot_evaluation import COLOR_SIMULATED
    from blueearth_cst.shared.wflow_outputs import (
        BASAVG_FIGURE_STEMS,
        SUBBASIN_PLOT_DIRNAME,
    )

    # Refresh this family's metadata, preserving unrelated captions and PNGs.
    for folder in (Path(Folder_out), Path(Folder_out) / SUBBASIN_PLOT_DIRNAME):
        path = folder / "_engine/figure_captions.json"
        if path.exists():
            inventory = json.loads(path.read_text(encoding="utf-8"))
            inventory["figures"] = {
                key: value
                for key, value in inventory["figures"].items()
                if not (
                    key.startswith("wflow_")
                    and (key.endswith("_basin_avg.png") or "_subbasin_" in key)
                )
            }
            _basavg_write_inventory(folder, inventory)

    for variable in ds.data_vars:
        code = code_from_variable(variable)
        meta = PLOT_META[code]
        daily, weights, individuals = _basavg_daily(ds[variable], cell_counts)
        supports = [("basin_avg", "Basin average", daily, Path(Folder_out))]
        if subbasin_plots:
            supports.extend(
                (
                    f"subbasin_{key}",
                    f"Subbasin {key}",
                    value,
                    Path(Folder_out) / SUBBASIN_PLOT_DIRNAME,
                )
                for key, value in individuals.items()
            )
        for support, label, one, folder in supports:
            folder.mkdir(parents=True, exist_ok=True)
            annual, climatology, info = _basavg_statistics(one, meta["kind"])
            trend = dg.sen_trend(annual)
            years = annual.index.to_numpy()
            segments = []
            eligible = annual.reindex(years).dropna().index.to_numpy()
            if np.isfinite(trend["slope"]) and eligible.size:
                span = np.array([eligible[0], eligible[-1]], dtype=float)
                fitted = trend["intercept"] + trend["slope"] / 10 * span
                slope_label = f"{trend['slope']:+.1f}".replace("-", "−")
                segments.append(
                    pc.TrendSegment(
                        COLOR_SIMULATED,
                        tuple(span),
                        tuple(fitted),
                        f"{slope_label} {meta['trend_units']}",
                    )
                )
            for kind in ("annual_ts", "monthly_clim_band"):
                with use_preset(cst.CST_PRESET):
                    if kind == "annual_ts":
                        fig, axes = fundamentals.new_fig(cst.H_SINGLE)
                        ax = axes[0, 0]
                        (line,) = pc.draw_annual_lines(
                            ax,
                            years,
                            [
                                pc.AnnualLine(
                                    "Mean", COLOR_SIMULATED, "o", annual.to_numpy()
                                )
                            ],
                        )
                        pc.draw_annual_trends(ax, segments, label_x=years[-1] - 0.4)
                        step = 10 if years.size > 35 else 5
                        first = int(years[0] // step * step)
                        ax.set_xlim(first - 0.5, years[-1] + 0.5)
                        ax.set_xticks(np.arange(first, years[-1] + 1, step))
                        ax.set_xticks(np.arange(first, years[-1] + 1), minor=True)
                        ax.set_ylabel(meta["annual_units"])
                        ax.yaxis.get_major_formatter().set_useOffset(False)
                        fundamentals.ygrid(ax)
                        line.set_label(
                            "Annual total"
                            if meta["kind"] == "amount"
                            else "Annual mean"
                        )
                        fundamentals.inside_legend(fig, ax, [line], lines=[line])
                        available = annual.notna().any()
                    else:
                        fig = pc.monthly_cycle(
                            [
                                pc.MonthlyBand(
                                    "Mean",
                                    COLOR_SIMULATED,
                                    "o",
                                    climatology["mean"].to_numpy(),
                                    climatology["p10"].to_numpy(),
                                    climatology["p90"].to_numpy(),
                                )
                            ],
                            month_labels=fundamentals.MONTH_ABBR,
                            ylabel=meta["monthly_units"],
                            band_label="10–90th percentile",
                            zero=False,
                        )
                        ax = fig.axes[0]
                        available = climatology["mean"].notna().any()
                    if not available:
                        ax.text(
                            0.5,
                            0.5,
                            "No eligible full calendar periods",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                    name = f"wflow_{BASAVG_FIGURE_STEMS[code]}_{kind}_{support}.png"
                    record = _basavg_caption(
                        folder,
                        name,
                        meta,
                        kind,
                        label,
                        info,
                        weights
                        if support == "basin_avg"
                        else {
                            int(support.split("_")[-1]): weights[
                                int(support.split("_")[-1])
                            ]
                        },
                        warmup_excluded,
                        trend,
                    )
                    save_figure(
                        str(folder / name),
                        fig=fig,
                        annotation=annotation,
                        title=record["title"],
                        caption=record["caption"],
                        dpi=plot_style.RASTER_DPI,
                    )
                    plt.close(fig)


def _basavg_caption(
    folder, name, meta, kind, label, info, weights, warmup_excluded, trend
):
    """Update the owning directory's durable caption inventory."""
    import json

    engine = folder / "_engine"
    engine.mkdir(exist_ok=True)
    path = engine / "figure_captions.json"
    inventory = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema_version": 1, "figures": {}}
    )
    quantity = meta["legend"].split(" (")[0].lower()
    method = (
        "Monthly totals use the available-day mean multiplied by calendar-month length; annual totals sum the twelve eligible months."
        if meta["kind"] == "amount"
        else "Monthly and annual values are means of the available daily values; no daily infilling is applied."
    )
    years = list(range(int(info["period"][0][:4]), int(info["period"][1][:4]) + 1))
    if kind == "annual_ts":
        omitted = set(info["partial_years"] + info["incomplete_years"])
        years = [year for year in years if year not in omitted]
    period = f"{years[0]}–{years[-1]}" if years else "no eligible years"
    if kind == "annual_ts":
        statistic = "totals" if meta["kind"] == "amount" else "means"
        caption = (
            f"Annual {statistic} of simulated {quantity}, {label.lower()} ({period})."
        )
        if np.isfinite(trend["slope"]):
            caption += " The dashed line shows the Sen trend; its slope is expressed per decade."
    else:
        statistic = "totals" if meta["kind"] == "amount" else "means"
        caption = (
            f"Seasonal cycle of simulated {quantity}, {label.lower()} ({period}). "
            f"The line shows mean monthly {statistic} across years; shading shows the "
            "10th–90th percentile range of interannual variability."
        )
    if warmup_excluded:
        caption += " The first simulation year is excluded."
    inventory["figures"][name] = {
        "title": f"Simulated {quantity}: {'annual series' if kind == 'annual_ts' else 'seasonal cycle'} — {label}; {period}",
        "caption": caption,
        "sources": ["simulated"],
        "period": info["period"],
        "support": label,
        "encoding": kind,
        "availability": "available"
        if (
            trend["n"]
            if kind == "annual_ts"
            else any(info["eligible_month_counts"].values())
        )
        else "unavailable",
        "captioned_path": None,
        "cell_counts": {str(key): value for key, value in weights.items()},
        "methods": info
        | {
            "quantity_kind": meta["kind"],
            "monthly_spread": "p10–p90",
            "warmup_excluded": warmup_excluded,
            "trend": "Sen per decade",
            "aggregation": method,
            "spatial_weighting": "fixed native cell counts"
            if weights
            else "single support",
        },
    }
    _basavg_write_inventory(folder, inventory)
    return inventory["figures"][name]


def _basavg_write_inventory(folder, inventory):
    import json

    write_figure_titles(folder, inventory["figures"])
    path = folder / "_engine/figure_captions.json"
    inventory["produced_paths"] = list(inventory["figures"]) + [
        "figure_titles.md",
        "_engine/figure_titles.json",
        "figure_captions.md",
        "_engine/figure_captions.json",
    ]
    path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (folder / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{record['caption']}"
            for key, record in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )
