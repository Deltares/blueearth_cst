# -*- coding: utf-8 -*-
"""Rule 2.07 — the WF2 annual overviews and the per-horizon monthly figures.

Reads the durable `scalar/*.nc` series and the authoritative monthly
change-factor table, and draws the figure set declared by
`projections/projection_figures.py` through the drawing functions in
`projections/projection_plots.py`.

**Two sources, and the split is the whole point of this module's design.** The
monthly change factors are READ from `{clim_project}_change_factors_monthly.csv`
rather than recomputed; the annual overviews are computed from the series,
because they are per-YEAR traces and no table carries a time series. What this
module used to do instead, and why that was wrong, is in the header comment over
the figure-input functions below.

Superseded on 2026-08-17 (owner ruling, board item `t2608091006`): the eight
independently styled figures this drew — `{proj}_{variable}_{view}_{quantity}.png`
— became two annual overviews plus one monthly figure per configured horizon,
on the shared WF1 page contract. The change-factor cloud is drawn by rule 2.06,
from the stage-B merge, and is not duplicated here.
"""

import os
from pathlib import Path

import pandas as pd
import xarray as xr

from blueearth_cst.projections import projection_figures, projection_plots
from blueearth_cst.shared.snake_utils import log_row

# ===========================================================================
# THE FIGURE INPUTS
#
# Two sources, and the split is the point.
#
# * The MONTHLY change factors are READ from
#   `{clim_project}_change_factors_monthly.csv`, never recomputed here. That
#   table is the authoritative product of `get_change_climate_proj.py`, and it
#   is what WF3 and every downstream reader consume. A figure that recomputed
#   the same quantity would be a second definition of it, free to disagree --
#   and it did: the shipped figures differenced a future month against the
#   historical ANNUAL mean, over the full 2015-2100 series rather than the
#   horizon, so the picture and the table described different quantities under
#   one name. Reading the table makes that class of disagreement structurally
#   impossible rather than merely fixed.
#
# * The ANNUAL OVERVIEWS are computed from `scalar/*.nc`, because they are
#   per-YEAR traces and no table carries a time series. The only arithmetic
#   here is the anomaly against the reference window, and it is differenced
#   against each model's OWN historical run so a future trace is continuous
#   with the historical one it follows.
# ===========================================================================


def normalise_model(name):
    """Drop the institute prefix from a model coordinate.

    ``scalar/*.nc`` carries ``NOAA-GFDL/GFDL-ESM4``; the change-factor tables
    carry ``GFDL-ESM4``. Without this, every join between series-derived and
    table-derived values matches nothing -- and an empty join reads as a broken
    calculation rather than as a spelling difference, which is the expensive
    kind of failure.
    """
    return str(name).split("/")[-1]


def parse_window(text):
    """``"2000-01-01 / 2014-12-01"`` into inclusive ``(start_year, end_year)``.

    The window is read off the artifact that states it rather than recomputed
    from config: the table records what was ACTUALLY differenced, including any
    effective-window override, and that is the only definition the figures can
    be held to.
    """
    try:
        start, end = str(text).split("/")
    except ValueError as exc:
        raise ValueError(f"malformed window {text!r}; expected 'START / END'") from exc
    return int(start.strip()[:4]), int(end.strip()[:4])


def load_scalar_series(paths):
    """Every ``scalar/*.nc`` as one long frame.

    Columns: ``model, scenario, member, year, month, precip, temp``.

    Year and month are read off the time index element-wise rather than through
    a datetime coercion: these series are monthly and declare a ``noleap``
    calendar (``cst_calendar``), so they decode to ``cftime`` objects on which
    ``.resample`` and ``.dt`` behave differently than on a ``DatetimeIndex``.
    Both object types answer ``.year`` and ``.month``, which is all this needs.
    """
    frames = []
    for path in sorted(str(p) for p in paths):
        with xr.open_dataset(path) as ds:
            times = ds.indexes["time"]
            frame = pd.DataFrame(
                {
                    "year": [t.year for t in times],
                    "month": [t.month for t in times],
                    "precip": ds["precip"].squeeze(drop=True).values,
                    "temp": ds["temp"].squeeze(drop=True).values,
                }
            )
            frame["model"] = normalise_model(ds["model"].values.item())
            frame["scenario"] = str(ds["scenario"].values.item())
            frame["member"] = str(ds["member"].values.item())
        frames.append(frame)
    if not frames:
        raise ValueError("no scalar series given; WF2 cannot draw its overviews")
    return pd.concat(frames, ignore_index=True)


def annual_reference(series, reference):
    """Historical mean per ``(model, member)`` over the reference window."""
    hist = series[
        (series["scenario"] == "historical") & series["year"].between(*reference)
    ]
    if hist.empty:
        raise ValueError(
            f"no historical years inside the reference window {reference}; "
            "the overviews would have nothing to difference against"
        )
    return hist.groupby(["model", "member"], as_index=False)[["precip", "temp"]].mean()


def annual_series(series, reference):
    """Annual means per combination per year, plus the anomaly against ``reference``.

    Every trace on both annual overviews comes from here -- historical and
    future alike, each differenced against its OWN model's historical reference
    window, which is what makes a future trace continuous with the historical
    one it follows rather than offset from it by a cross-model mean.
    """
    annual = series.groupby(["model", "scenario", "member", "year"], as_index=False)[
        ["precip", "temp"]
    ].mean()
    merged = annual.merge(
        annual_reference(series, reference),
        on=["model", "member"],
        suffixes=("", "_ref"),
    )
    merged["precip_anomaly"] = (
        (merged["precip"] - merged["precip_ref"]) / merged["precip_ref"] * 100.0
    )
    merged["temp_anomaly"] = merged["temp"] - merged["temp_ref"]
    return merged


def monthly_change_from_table(table, horizon, statistic="mean"):
    """The monthly change factors for one horizon, READ from the table.

    Returns ``model, scenario, member, month, precip_change, temp_change`` --
    the table's own ``relative_value``, pivoted per variable and not touched by
    any arithmetic here. ``relative_units`` is ``%`` for precipitation and
    ``degC`` for temperature, which is what the figure's y-labels state.

    Rows whose ``status`` is not ``ok`` are DROPPED rather than plotted or
    backfilled: a flagged month is one where the near-zero reference made the
    ratio meaningless, and drawing it would put a number on the page that the
    table itself declines to publish.

    Both statistics the table carries beside ``mean`` are available, but the
    figure asks for one: an unfiltered table has three rows per combination and
    a naive read triples every trace.
    """
    wanted = table[
        (table["horizon"] == horizon) & (table["statistic"] == statistic)
    ].copy()
    if wanted.empty:
        raise ValueError(
            f"change-factor table carries no {statistic!r} rows for horizon "
            f"{horizon!r}; have {sorted(table['horizon'].unique())}"
        )
    wanted = wanted[wanted["status"] == "ok"]
    keys = ["model", "scenario", "member", "month"]
    wide = wanted.pivot_table(
        index=keys, columns="variable", values="relative_value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns={"precip": "precip_change", "temp": "temp_change"})


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            clim_project_dir = Path(sm.params.clim_project_dir)
            horizons = dict(sm.params.horizons)
            plots_dir = clim_project_dir / "plots"

            # The monthly table first: it is both an input to the monthly
            # figures and the artifact that STATES the reference window, which
            # the annual overviews difference against. Reading the window off
            # the table rather than off config is deliberate -- the table
            # records what was ACTUALLY differenced, including any per-model
            # effective-window override, and a figure captioned with a window
            # the arithmetic did not use is worse than one with no caption.
            log_row("Reading the monthly change-factor table", module="plot")
            monthly_table = pd.read_csv(sm.input.change_factors_monthly)
            windows = monthly_table["reference_window"].unique()
            if len(windows) != 1:
                raise ValueError(
                    "change-factor table carries more than one reference window "
                    f"({sorted(windows)}); the overviews cannot be captioned with one"
                )
            reference = parse_window(windows[0])

            log_row("Opening the scalar gcm timeseries", module="plot")
            series = load_scalar_series(
                list(sm.input.stats_time_nc_hist) + list(sm.input.stats_time_nc)
            )
            annual = annual_series(series, reference)

            os.makedirs(plots_dir, exist_ok=True)

            # The two annual overviews. `figure_relative_paths` is the single
            # definition of the set, so the names here cannot drift from the
            # ones the Snakefile declares as outputs.
            for variable, relative in (
                ("precip", "overview/annual-precipitation.png"),
                ("temp", "overview/annual-temperature.png"),
            ):
                out_path = plots_dir / relative
                out_path.parent.mkdir(parents=True, exist_ok=True)
                traces = projection_plots.draw_annual_overview(
                    annual, variable, reference, out_path
                )
                log_row(f"Wrote {relative} ({traces} traces per panel)", module="plot")

            # One monthly figure per configured horizon, each rendering the
            # table's own numbers for that horizon.
            monthly_paths = projection_figures.monthly_figure_paths(horizons)
            for name, relative in monthly_paths.items():
                period = projection_figures.parse_horizon_period(horizons[name])
                changes = monthly_change_from_table(monthly_table, name)
                out_path = plots_dir / relative
                out_path.parent.mkdir(parents=True, exist_ok=True)
                traces = projection_plots.draw_monthly_change(
                    changes, name, period, reference, out_path
                )
                log_row(f"Wrote {relative} ({traces} traces per panel)", module="plot")
    else:
        raise RuntimeError("plot_proj_timeseries.py runs only as a Snakemake script:")
