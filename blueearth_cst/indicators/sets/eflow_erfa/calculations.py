"""Fixed-default eflowrisk annual metrics and per-record median/IQR summaries."""

import calendar
from collections.abc import Sequence

import numpy as np
import pandas as pd

from blueearth_cst.indicators.contract import (
    DiagnosticTable,
    EvaluationContext,
    IndicatorDeclaration,
    IndicatorResult,
)
from blueearth_cst.indicators.series import ResponseSeries, prepared_frame


def metric_definitions() -> dict[str, tuple[str, str, str]]:
    """Return the fixed R keys with ERFA group, description and output units."""
    definitions = {}
    for prefix, group, word in (("amax", "HF", "maximum"), ("amin", "LF", "minimum")):
        for days in (1, 3, 7, 30, 90):
            definitions[f"{prefix}{days}d"] = (
                group,
                f"Annual {days}-day mean {word} discharge",
                "m3 s-1",
            )
        definitions[f"wyday_{prefix}1d"] = (
            group,
            f"Day of water year of first annual {word} discharge",
            "d",
        )
        pulse = "high" if group == "HF" else "low"
        definitions[f"num_{pulse}_pulses"] = (
            group,
            f"Number of {pulse}-flow pulses",
            "1",
        )
        definitions[f"avg_dur_{pulse}_pulses"] = (
            group,
            f"Mean duration of {pulse}-flow pulses",
            "d",
        )
        if group == "HF":
            for month in list(calendar.month_name)[1:]:
                definitions[f"{month.lower()}_flow"] = (
                    "MF",
                    f"Mean {month} discharge",
                    "m3 s-1",
                )
    for direction in ("rise", "fall"):
        definitions[f"mean_rate_{direction}"] = (
            "RFC",
            f"Mean positive magnitude of daily {direction}",
            "m3 s-1 d-1",
        )
    for direction in ("rise", "fall"):
        definitions[f"num_{direction}s"] = ("RFC", f"Number of daily {direction}s", "1")
    definitions.update(
        {
            "num_zero_seq": ("IF", "Number of zero-flow sequences", "1"),
            "avg_dur_zero_seq": ("IF", "Mean duration of zero-flow sequences", "d"),
            "time_main_zero_seq": (
                "IF",
                "Day of water year of first longest zero-flow sequence start",
                "d",
            ),
        }
    )
    return definitions


def _runs(condition: np.ndarray) -> tuple[int, float, int | None]:
    """Return count, mean duration and first longest-run offset."""
    boundaries = np.diff(np.r_[False, condition, False].astype(int))
    starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    if not len(starts):
        return 0, 0.0, None
    lengths = ends - starts
    return len(starts), float(lengths.mean()), int(starts[lengths.argmax()])


def annual_metrics(daily: pd.Series, anchor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate all 35 annual metrics and finite-calendar-day coverage evidence.

    Input must be a complete Gregorian daily grid with native NaNs. The public
    response binding validates this before calling. Empty retained years are
    rejected, matching eflowrisk's summary function. Full-record thresholds and
    trailing windows include observations belonging to subsequently dropped years.
    """
    offset = pd.tseries.frequencies.to_offset(anchor)
    if (
        not isinstance(offset, (pd.offsets.YearEnd, pd.offsets.YearBegin))
        or offset.n != 1
    ):
        raise ValueError("eflow-erfa requires an annual year-start or year-end anchor")
    start_month = (
        offset.month
        if isinstance(offset, pd.offsets.YearBegin)
        else offset.month % 12 + 1
    )
    end_anchor = pd.offsets.YearEnd(month=(start_month - 2) % 12 + 1)
    rows, coverage = {}, []
    retained = []
    for end, block in daily.resample(end_anchor):
        start = pd.Timestamp(
            end.year if start_month == 1 else end.year - 1, start_month, 1
        )
        expected = (start + pd.DateOffset(years=1) - start).days
        count = int(block.notna().sum())
        keep = count / expected >= 0.9
        coverage.append(
            {
                "year": end.year,
                "finite_days": count,
                "expected_days": expected,
                "retained": keep,
            }
        )
        if keep:
            retained.append((end.year, start, block))
    evidence = pd.DataFrame(coverage)
    if not retained:
        raise ValueError(
            "eflow-erfa has no retained years (requires 90% finite calendar days per water year)"
        )
    finite = daily.dropna().to_numpy()
    if len(finite) < 10:
        raise ValueError(
            "eflow-erfa requires at least 10 finite discharge values for thresholds"
        )
    low, high = np.quantile(finite, [0.1, 0.9], method="linear")
    rolling = {days: daily.rolling(days).mean() for days in (3, 7, 30, 90)}
    changes = daily.diff()
    for year, start, block in retained:
        q = block.to_numpy()
        days = np.asarray((block.index - start).days + 1)
        result = {}
        for prefix, extreme in (("amax", "max"), ("amin", "min")):
            result[f"{prefix}1d"] = float(getattr(block, extreme)())
            for window, values in rolling.items():
                result[f"{prefix}{window}d"] = float(
                    getattr(values.loc[block.index], extreme)()
                )
            position = np.nanargmax(q) if extreme == "max" else np.nanargmin(q)
            result[f"wyday_{prefix}1d"] = float(days[position])
        for name, condition in (
            ("high_pulses", q > high),
            ("low_pulses", q < low),
            ("zero_seq", q <= 0),
        ):
            count, duration, first = _runs(condition)
            result[f"num_{name}"] = float(count)
            result[f"avg_dur_{name}"] = duration
            if name == "zero_seq":
                result["time_main_zero_seq"] = (
                    np.nan if first is None else float(days[first])
                )
        for month in range(1, 13):
            selected = block[block.index.month == month]
            enough = (
                bool(len(selected))
                and selected.notna().sum() / selected.index[0].days_in_month >= 0.8
            )
            result[f"{calendar.month_name[month].lower()}_flow"] = (
                float(selected.mean()) if enough else np.nan
            )
        delta = changes.loc[block.index]
        for direction, values in (
            ("rise", delta[delta > 0]),
            ("fall", -delta[delta < 0]),
        ):
            result[f"mean_rate_{direction}"] = float(values.mean())
            result[f"num_{direction}s"] = float(len(values))
        rows[year] = result
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year"), evidence


def annual_statistic(
    indicator: IndicatorDeclaration,
    series: Sequence[ResponseSeries],
    context: EvaluationContext,
) -> IndicatorResult:
    """Summarize each location's retained years and retain coverage diagnostics."""
    prepared = prepared_frame(series, context)
    frame = prepared["frame"]
    metric = indicator.calculation_parameters["metric"]
    statistic = indicator.calculation_parameters["statistic"]
    values, evidence = {}, []
    if "erfa" not in prepared:
        tables = {}
        for location in frame:
            annual, coverage = annual_metrics(frame[location], context.anchor)
            quantiles = annual.quantile([0.25, 0.75])
            summaries = pd.DataFrame(
                {
                    "med": annual.median(),
                    "iqr": quantiles.loc[0.75] - quantiles.loc[0.25],
                }
            )
            coverage.insert(0, "location_id", location)
            coverage.insert(0, "run_id", series[0].run_id)
            tables[location] = summaries, coverage
        prepared["erfa"] = tables
    for location in frame:
        summaries, coverage = prepared["erfa"][location]
        values[location] = float(summaries.loc[metric, statistic])
        evidence.append(coverage)
    return IndicatorResult(
        pd.Series(values),
        diagnostics=(
            DiagnosticTable(
                "erfa_year_coverage", pd.concat(evidence, ignore_index=True)
            ),
        ),
        metadata={
            "threshold_scope": "per-record global",
            "year_min_frac": 0.9,
            "month_min_frac": 0.8,
        },
    )


PARAMETERS = {
    annual_statistic: {
        "metric": tuple(metric_definitions()),
        "statistic": ("med", "iqr"),
    }
}
