"""Independent scalar ERFA checks using explicit windows, runs and quantiles."""

import calendar
import math
from itertools import groupby

import pandas as pd

from blueearth_cst.indicators.checking import member_expectation
from blueearth_cst.indicators.contract import EvaluationContext, IndicatorDeclaration


def _quantile(values: list[float], probability: float) -> float:
    """Interpolate order statistics with R type-7 weights."""
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[math.ceil(position)] - ordered[lower])


def _scalar(daily: pd.Series, anchor: str, metric: str, statistic: str) -> float:
    """Reconstruct one annual metric directly from daily observations."""
    offset = pd.tseries.frequencies.to_offset(anchor)
    start_month = (
        offset.month
        if isinstance(offset, pd.offsets.YearBegin)
        else offset.month % 12 + 1
    )
    observations = list(zip(daily.index, daily.to_numpy(), strict=True))
    full = [float(value) for _, value in observations]
    high, low = _quantile(full, 0.9), _quantile(full, 0.1)
    years = {}
    for position, (date, value) in enumerate(observations):
        year = date.year + int(start_month != 1 and date.month >= start_month)
        years.setdefault(year, []).append((position, date, float(value)))
    annual, retained = [], 0
    for year, entries in years.items():
        start = pd.Timestamp(year - int(start_month != 1), start_month, 1)
        end = pd.Timestamp(start.year + 1, start_month, 1)
        finite = [value for _, _, value in entries if math.isfinite(value)]
        if len(finite) / (end - start).days < 0.9:
            continue
        retained += 1
        if metric.startswith(("amax", "amin")):
            window = int(metric[4:-1])
            samples = []
            for position, _, _ in entries:
                values = full[position - window + 1 : position + 1]
                if position >= window - 1 and all(
                    math.isfinite(value) for value in values
                ):
                    samples.append(sum(values) / window)
            value = (
                (max(samples) if metric.startswith("amax") else min(samples))
                if samples
                else math.nan
            )
        elif metric.startswith("wyday_"):
            extreme = max(finite) if "amax" in metric else min(finite)
            date = next(date for _, date, value in entries if value == extreme)
            value = float((date - start).days + 1)
        elif metric.endswith("_flow"):
            month = list(calendar.month_name).index(
                metric.removesuffix("_flow").capitalize()
            )
            selected = [
                (date, value) for _, date, value in entries if date.month == month
            ]
            usable = [value for _, value in selected if math.isfinite(value)]
            enough = (
                selected
                and len(usable) / calendar.monthrange(selected[0][0].year, month)[1]
                >= 0.8
            )
            value = sum(usable) / len(usable) if enough else math.nan
        elif metric in ("mean_rate_rise", "mean_rate_fall", "num_rises", "num_falls"):
            changes = [
                full[position] - full[position - 1]
                for position, _, _ in entries
                if position > 0
            ]
            rising = metric in ("mean_rate_rise", "num_rises")
            selected = [
                abs(change)
                for change in changes
                if math.isfinite(change) and (change > 0 if rising else change < 0)
            ]
            value = (
                float(len(selected))
                if metric.startswith("num_")
                else sum(selected) / len(selected)
                if selected
                else math.nan
            )
        else:
            flags = [
                math.isfinite(value)
                and (
                    value > high
                    if "high" in metric
                    else value < low
                    if "low" in metric
                    else value <= 0
                )
                for _, _, value in entries
            ]
            runs, position = [], 0
            for flag, group in groupby(flags):
                length = len(list(group))
                if flag:
                    runs.append((length, position))
                position += length
            if metric.startswith("num_"):
                value = float(len(runs))
            elif metric.startswith("avg_"):
                value = sum(length for length, _ in runs) / len(runs) if runs else 0.0
            elif runs:
                _, first = max(runs, key=lambda run: run[0])
                value = float((entries[first][1] - start).days + 1)
            else:
                value = math.nan
        annual.append(value)
    if not retained:
        raise ValueError("eflow-erfa has no retained years")
    if statistic == "med":
        return _quantile(annual, 0.5)
    return _quantile(annual, 0.75) - _quantile(annual, 0.25)


def expected(
    indicator: IndicatorDeclaration,
    series,
    context: EvaluationContext,
    *,
    group: bool = False,
) -> pd.Series:
    """Check individual records then propagate undefined group members."""
    if indicator.calculation_method != "erfa_annual_statistic/1":
        raise ValueError(
            f"missing independent ERFA check for {indicator.calculation_method!r}"
        )
    parameters = indicator.calculation_parameters
    return member_expectation(
        indicator,
        series,
        context,
        lambda daily: _scalar(
            daily, context.anchor, parameters["metric"], parameters["statistic"]
        ),
        group=group,
    )
