# -*- coding: utf-8 -*-
"""Render the standardized WF2 projection figures.

Annual figures read the durable scalar series so their two panels cover the
full historical and future periods. Monthly figures read the authoritative
monthly change-factor table: the table already applies the configured horizon,
matching-calendar-month reference, calendar weighting, and dry-month rule.

Every ``(model, scenario, member)`` remains a separate trace. Scenario is the
only visual identity; model and member never enter colour, style, or legends.
"""

import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from blueearth_cst.shared.plot_style import (
    COLOR_CAVEAT,
    FONT_SIZE_CAVEAT,
    FONT_SIZE_TITLE,
    RASTER_DPI,
    figure_width_inches,
    rcparams,
)
from blueearth_cst.shared.snake_utils import log_row, save_figure

MONTH_LABELS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

SCENARIO_COLORS = {
    "ssp126": "#003466",
    "ssp245": "#F69320",
    "ssp370": "#DF0000",
    "ssp585": "#980002",
}
_FALLBACK_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
_HISTORICAL_COLOR = "0.55"
_TRACE_GID = "combination-trace"

_VARIABLES = {
    "precip": {
        "title": "Annual precipitation",
        "absolute_label": "Precipitation (mm/day)",
        "change_label": "Change (%)",
    },
    "temp": {
        "title": "Annual temperature",
        "absolute_label": "Temperature (°C)",
        "change_label": "Change (°C)",
    },
}

SeriesIdentity = tuple[str, str, str]
SeriesCollection = Mapping[SeriesIdentity, pd.DataFrame]


def parse_horizon_period(period: Sequence[int] | str) -> tuple[int, int]:
    """Return one inclusive ``(start_year, end_year)`` horizon pair."""
    values = period.split(",") if isinstance(period, str) else list(period)
    if len(values) != 2:
        raise ValueError(f"horizon period must contain two years, got {period!r}")
    try:
        start, end = (int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"horizon years must be integers, got {period!r}") from exc
    if start > end:
        raise ValueError(f"horizon start must not exceed end, got {start}-{end}")
    return start, end


def sanitize_horizon_name(name: str) -> str:
    """Sanitize a configured horizon name for a portable directory name."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    if not slug:
        raise ValueError(f"horizon name has no portable characters: {name!r}")
    return slug


def horizon_directory(name: str, period: Sequence[int] | str) -> str:
    """Return ``<sanitized-name>-<start>-<end>`` for one horizon."""
    start, end = parse_horizon_period(period)
    return f"{sanitize_horizon_name(name)}-{start}-{end}"


def figure_relative_paths(horizons: Mapping[str, Sequence[int] | str]) -> list[str]:
    """Return every WF2 figure path relative to ``plots/`` in stable order."""
    paths = [
        "overview/annual-precipitation.png",
        "overview/annual-temperature.png",
        "overview/change-factor-cloud.png",
    ]
    window_dirs = [horizon_directory(name, period) for name, period in horizons.items()]
    if len(window_dirs) != len(set(window_dirs)):
        raise ValueError("configured horizon names and years resolve to duplicate paths")
    paths.extend(
        f"windows/{window_dir}/monthly-change-factors.png"
        for window_dir in window_dirs
    )
    return paths


def scenario_label(scenario: str) -> str:
    """Format a CMIP scenario identifier for presentation."""
    match = re.fullmatch(r"ssp(\d)(\d)(\d)", str(scenario).lower())
    if match:
        family, forcing_whole, forcing_decimal = match.groups()
        return f"SSP{family}-{forcing_whole}.{forcing_decimal}"
    return str(scenario).upper()


def scenario_palette(scenarios: Sequence[str]) -> dict[str, str]:
    """Return one colour per scenario without introducing model/member styles."""
    palette = {}
    fallback_index = 0
    for scenario in dict.fromkeys(str(value) for value in scenarios):
        if scenario in SCENARIO_COLORS:
            palette[scenario] = SCENARIO_COLORS[scenario]
        else:
            palette[scenario] = _FALLBACK_COLORS[
                fallback_index % len(_FALLBACK_COLORS)
            ]
            fallback_index += 1
    return palette


def style_projection_axes(ax) -> None:
    """Apply the WF1 non-map grid and frame convention to an axis."""
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _figure_size(aspect: float) -> tuple[float, float]:
    """Return the shared 180 mm page width at a caller-selected aspect."""
    width = figure_width_inches()
    return width, width * aspect


def _scenario_legend(fig, scenarios: Sequence[str], *, historical: bool) -> None:
    """Add one compact figure legend containing scenario identities only."""
    palette = scenario_palette(scenarios)
    handles = []
    labels = []
    if historical:
        handles.append(Line2D([], [], color=_HISTORICAL_COLOR, lw=1.2))
        labels.append("Historical")
    for scenario in dict.fromkeys(str(value) for value in scenarios):
        handles.append(Line2D([], [], color=palette[scenario], lw=1.4))
        labels.append(scenario_label(scenario))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=max(1, len(handles)),
        frameon=False,
    )


def todatetimeindex_dropvars(ds: xr.Dataset) -> xr.Dataset:
    """Convert object time indexes and remove scalar coordinates not used here."""
    if "time" in ds.coords and ds.indexes["time"].dtype == "O":
        ds["time"] = ds.indexes["time"].to_datetimeindex()
    for coord in ("spatial_ref", "height"):
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    return ds


def _scalar_coord(ds: xr.Dataset, name: str, path: Path) -> str:
    """Read one scalar/length-one identity coordinate from a series file."""
    if name not in ds.coords:
        raise ValueError(f"{path}: missing identity coordinate {name!r}")
    values = np.asarray(ds.coords[name].values).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{path}: expected one {name!r}, got {values.size}")
    return str(values[0])


def load_scalar_series(paths: Iterable[os.PathLike | str]) -> dict[SeriesIdentity, pd.DataFrame]:
    """Load each scalar file independently, preserving every series identity."""
    records = {}
    for value in paths:
        path = Path(value)
        with xr.open_dataset(path) as source:
            ds = todatetimeindex_dropvars(source).load()
        identity = (
            _scalar_coord(ds, "model", path),
            _scalar_coord(ds, "scenario", path),
            _scalar_coord(ds, "member", path),
        )
        if identity in records:
            raise ValueError(f"duplicate scalar series identity {identity!r}")
        frame = {}
        for variable in _VARIABLES:
            if variable not in ds:
                raise ValueError(f"{path}: missing variable {variable!r}")
            da = ds[variable].squeeze(drop=True)
            if da.dims != ("time",):
                raise ValueError(
                    f"{path}: {variable!r} must reduce to time only, got {da.dims}"
                )
            frame[variable] = np.asarray(da.values, dtype=float)
        records[identity] = pd.DataFrame(
            frame, index=pd.DatetimeIndex(ds.indexes["time"])
        )
    return records


def _annual_mean(frame: pd.DataFrame, variable: str) -> pd.Series:
    """Return the annual mean of one monthly scalar series."""
    return pd.to_numeric(frame[variable], errors="coerce").resample("YE").mean()


def _reference_mean(
    frame: pd.DataFrame, variable: str, reference_window: tuple[str, str]
) -> float:
    """Return one series' mean over the effective reference window."""
    selected = pd.to_numeric(frame.loc[slice(*reference_window), variable], errors="coerce")
    if selected.empty or not np.isfinite(selected.to_numpy(dtype=float)).any():
        raise ValueError(
            f"no finite {variable!r} values in reference window {reference_window!r}"
        )
    value = float(selected.mean())
    if variable == "precip" and np.isclose(value, 0.0):
        raise ValueError("precipitation anomaly is undefined for a zero reference")
    return value


def _change(values: pd.Series, reference: float, variable: str) -> pd.Series:
    """Apply the configured WF2 change kind to annual values."""
    if variable == "precip":
        return (values - reference) / reference * 100.0
    return values - reference


def plot_annual_projection(
    historical: SeriesCollection,
    future: SeriesCollection,
    *,
    variable: str,
    reference_window: tuple[str, str],
    scenarios: Sequence[str],
):
    """Draw full-period absolute and anomaly panels for one variable."""
    if variable not in _VARIABLES:
        raise ValueError(f"unknown projection variable {variable!r}")
    spec = _VARIABLES[variable]
    palette = scenario_palette(scenarios)

    with plt.rc_context(rcparams()):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=_figure_size(0.78),
            sharex=True,
        )
        fig.subplots_adjust(
            left=0.10, right=0.98, bottom=0.10, top=0.80, hspace=0.34
        )
        fig.suptitle(spec["title"], fontsize=FONT_SIZE_TITLE, y=0.985)

        trace_counts = [0, 0]
        for identity, frame in sorted(historical.items()):
            model, _scenario, member = identity
            annual = _annual_mean(frame, variable)
            reference = _reference_mean(frame, variable, reference_window)
            for axis, values in zip(
                axes, (annual, _change(annual, reference, variable))
            ):
                axis.plot(
                    annual.index,
                    values,
                    color=_HISTORICAL_COLOR,
                    alpha=0.72,
                    lw=0.9,
                    gid=_TRACE_GID,
                )
            trace_counts[0] += 1
            trace_counts[1] += 1

        for identity, frame in sorted(future.items()):
            model, scenario, member = identity
            hist_key = (model, "historical", member)
            if hist_key not in historical:
                raise ValueError(
                    f"future series {identity!r} has no matching historical series"
                )
            if scenario not in palette:
                raise ValueError(f"future series uses unconfigured scenario {scenario!r}")
            annual = _annual_mean(frame, variable)
            reference = _reference_mean(
                historical[hist_key], variable, reference_window
            )
            for axis, values in zip(
                axes, (annual, _change(annual, reference, variable))
            ):
                axis.plot(
                    annual.index,
                    values,
                    color=palette[scenario],
                    alpha=0.78,
                    lw=0.9,
                    gid=_TRACE_GID,
                )
            trace_counts[0] += 1
            trace_counts[1] += 1

        expected = len(historical) + len(future)
        if trace_counts != [expected, expected]:
            raise AssertionError(
                f"annual trace count {trace_counts!r} does not equal {expected} per panel"
            )

        transition_dates = [
            _annual_mean(frame, variable).index.min()
            for frame in future.values()
            if not frame.empty
        ]
        transition = min(transition_dates) if transition_dates else None
        for axis in axes:
            if transition is not None:
                axis.axvline(transition, color="0.35", lw=0.7, ls=(0, (3, 2)))
                axis.annotate(
                    "future",
                    xy=(transition, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -3),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color="0.35",
                )
            style_projection_axes(axis)

        axes[0].set_title("Absolute")
        axes[0].set_ylabel(spec["absolute_label"])
        axes[1].axhline(0.0, color="0.35", lw=0.7, ls=(0, (3, 2)))
        axes[1].set_title(
            f"Anomaly — reference {reference_window[0]} to {reference_window[1]}"
        )
        axes[1].set_ylabel(spec["change_label"])
        axes[1].set_xlabel("Year")
        _scenario_legend(fig, scenarios, historical=True)
    return fig


def monthly_change_series(
    table: pd.DataFrame, *, horizon: str, variable: str
) -> dict[SeriesIdentity, pd.Series]:
    """Return authoritative monthly mean changes for one horizon and variable."""
    required = {
        "model",
        "scenario",
        "member",
        "horizon",
        "month",
        "variable",
        "statistic",
        "relative_value",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"monthly change-factor table is missing columns {missing}")

    subset = table.loc[
        (table["horizon"].astype(str) == str(horizon))
        & (table["variable"].astype(str) == variable)
        & (table["statistic"].astype(str) == "mean")
    ].copy()
    if subset.empty:
        raise ValueError(
            f"monthly table has no mean rows for horizon={horizon!r}, variable={variable!r}"
        )
    subset["month"] = pd.to_numeric(subset["month"], errors="raise").astype(int)
    subset["relative_value"] = pd.to_numeric(
        subset["relative_value"], errors="coerce"
    )

    traces = {}
    identity_columns = ["model", "scenario", "member"]
    for identity, rows in subset.groupby(identity_columns, sort=True, dropna=False):
        if rows["month"].duplicated().any():
            raise ValueError(
                f"duplicate monthly mean rows for {tuple(str(v) for v in identity)!r}"
            )
        key = tuple(str(value) for value in identity)
        traces[key] = rows.set_index("month")["relative_value"].reindex(range(1, 13))
    return traces


def plot_monthly_change_factors(
    table: pd.DataFrame,
    *,
    horizon: str,
    period: Sequence[int] | str,
    scenarios: Sequence[str],
    reference_window: tuple[str, str] | None = None,
):
    """Draw precipitation and temperature monthly changes for one horizon."""
    start, end = parse_horizon_period(period)
    palette = scenario_palette(scenarios)
    traces_by_variable = {
        variable: monthly_change_series(table, horizon=horizon, variable=variable)
        for variable in _VARIABLES
    }
    expected_identities = set().union(
        *(set(traces) for traces in traces_by_variable.values())
    )
    for variable, traces in traces_by_variable.items():
        if set(traces) != expected_identities:
            raise ValueError(
                f"{horizon!r} {variable!r} rows do not cover every resolved combination"
            )

    with plt.rc_context(rcparams()):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=_figure_size(0.70),
            sharex=True,
        )
        fig.subplots_adjust(
            left=0.10, right=0.98, bottom=0.14, top=0.79, hspace=0.36
        )
        fig.suptitle(
            f"Monthly change factors — {horizon} ({start}–{end})",
            fontsize=FONT_SIZE_TITLE,
            y=0.985,
        )
        if reference_window is not None:
            fig.supxlabel(
                f"Reference: {reference_window[0]} to {reference_window[1]}",
                fontsize=FONT_SIZE_CAVEAT,
                color=COLOR_CAVEAT,
            )
        for axis, variable in zip(axes, _VARIABLES):
            traces = traces_by_variable[variable]
            plotted = 0
            for (_model, scenario, _member), values in traces.items():
                if scenario not in palette:
                    raise ValueError(
                        f"monthly table uses unconfigured scenario {scenario!r}"
                    )
                axis.plot(
                    range(1, 13),
                    values.to_numpy(dtype=float),
                    color=palette[scenario],
                    alpha=0.78,
                    lw=0.9,
                    gid=_TRACE_GID,
                )
                plotted += 1
            if plotted != len(expected_identities):
                raise AssertionError(
                    f"{horizon!r} {variable!r}: plotted {plotted} of "
                    f"{len(expected_identities)} combinations"
                )
            axis.axhline(0.0, color="0.35", lw=0.7, ls=(0, (3, 2)))
            axis.set_title(
                "Precipitation change" if variable == "precip" else "Temperature change"
            )
            axis.set_ylabel(_VARIABLES[variable]["change_label"])
            style_projection_axes(axis)

        axes[-1].set_xticks(range(1, 13))
        axes[-1].set_xticklabels(MONTH_LABELS)
        axes[-1].set_xlim(0.6, 12.4)
        axes[-1].set_xlabel("Month")
        _scenario_legend(fig, scenarios, historical=False)
    return fig


def render_projection_figures(
    *,
    historical_paths: Iterable[os.PathLike | str],
    future_paths: Iterable[os.PathLike | str],
    monthly_table_path: os.PathLike | str,
    plot_dir: os.PathLike | str,
    horizons: Mapping[str, Sequence[int] | str],
    reference_window: tuple[str, str],
    scenarios: Sequence[str],
) -> list[Path]:
    """Render annual overviews and one monthly figure per configured horizon."""
    plot_dir = Path(plot_dir)
    historical = load_scalar_series(historical_paths)
    future = load_scalar_series(future_paths)
    monthly_table = pd.read_csv(monthly_table_path)
    written = []

    for variable in _VARIABLES:
        path = plot_dir / "overview" / (
            "annual-precipitation.png"
            if variable == "precip"
            else "annual-temperature.png"
        )
        fig = plot_annual_projection(
            historical,
            future,
            variable=variable,
            reference_window=reference_window,
            scenarios=scenarios,
        )
        save_figure(path, fig=fig, dpi=RASTER_DPI)
        plt.close(fig)
        written.append(path)

    for horizon, period in horizons.items():
        path = (
            plot_dir
            / "windows"
            / horizon_directory(horizon, period)
            / "monthly-change-factors.png"
        )
        fig = plot_monthly_change_factors(
            monthly_table,
            horizon=horizon,
            period=period,
            scenarios=scenarios,
            reference_window=reference_window,
        )
        save_figure(path, fig=fig, dpi=RASTER_DPI)
        plt.close(fig)
        written.append(path)

    log_row(f"Wrote {len(written)} projection figures to {plot_dir}", module="plot")
    return written


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            render_projection_figures(
                historical_paths=sm.input.stats_time_nc_hist,
                future_paths=sm.input.stats_time_nc,
                monthly_table_path=sm.input.change_factors_monthly,
                plot_dir=Path(sm.params.clim_project_dir) / "plots",
                horizons=sm.params.horizons,
                reference_window=tuple(str(value) for value in sm.params.reference_window),
                scenarios=sm.params.scenarios,
            )
    else:
        raise RuntimeError("plot_proj_timeseries.py runs only as a Snakemake script")
