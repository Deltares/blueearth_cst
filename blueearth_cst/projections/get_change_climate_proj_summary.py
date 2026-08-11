"""
Open monthly change files for all models/scenarios/horizon and compute/plot statistics
"""

import hydromt  # noqa: F401 -- registers the xarray .raster accessor (ds.raster.vars below)
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from matplotlib.lines import Line2D

from typing import Union, List, Dict

from blueearth_cst.shared.snake_utils import log_row
from blueearth_cst.shared.plot_style import (
    FONT_SIZE_TITLE,
    RASTER_DPI,
    figure_width_inches,
    rcparams,
)
from blueearth_cst.shared.snake_utils import save_figure
from blueearth_cst.projections.plot_proj_timeseries import (
    parse_horizon_period,
    scenario_label,
    scenario_palette,
    style_projection_axes,
)


def preprocess_coords(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess function to remove unwanted coords, and stop string TRUNCATION.

    The string coords arrive as numpy fixed-width dtypes (`<U13`, `<U19`, …) whose
    width is set by the longest value in *that one file*. Concatenating files with
    different widths silently truncates every value to the FIRST file's width:
    `NOAA-GFDL/GFDL-ESM4` (19) became `NOAA-GFDL/GFD` (13) whenever an
    `INM/INM-CM4-8` file was merged first. Silent, because a truncated model name
    is still a plausible-looking string.

    It corrupted the wide summary's `model` coordinate, and through it the tidy
    table's `model` column — and, because the per-combination window lookup is
    keyed on that name, the truncated rows also missed their effective-window
    override and fell back to the run-level one. Found at S8-04 when the column
    read `GFD`; the truncation predates it and was already in the `dataset` column
    of the wide summary.

    Casting to object dtype makes each value independent of the others' lengths.
    """
    coords_to_remove = ["height"]
    for coord in coords_to_remove:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    for name, coord in ds.coords.items():
        if coord.dtype.kind in ("U", "S"):
            ds = ds.assign_coords({name: coord.astype(object)})
    return ds


def change_factor_cloud_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the annual tidy table into one cloud row per combination/horizon."""
    required = {
        "model",
        "scenario",
        "member",
        "horizon",
        "variable",
        "statistic",
        "relative_value",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"annual change-factor table is missing columns {missing}")
    selected = table.loc[table["statistic"].astype(str) == "mean"].copy()
    selected["relative_value"] = pd.to_numeric(
        selected["relative_value"], errors="coerce"
    )
    frame = (
        selected.pivot(
            index=["model", "scenario", "member", "horizon"],
            columns="variable",
            values="relative_value",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    missing_variables = sorted({"precip", "temp"} - set(frame.columns))
    if missing_variables:
        raise ValueError(
            f"annual change-factor table is missing variables {missing_variables}"
        )
    return frame


def _cloud_limits(values: pd.Series) -> tuple[float, float]:
    """Return a common finite axis range that includes zero."""
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("change-factor cloud contains non-finite mean values")
    low = min(0.0, float(array.min()))
    high = max(0.0, float(array.max()))
    span = high - low
    padding = 0.05 * span if span else 1.0
    return low - padding, high + padding


def plot_change_factor_cloud(
    frame: pd.DataFrame,
    *,
    horizons: Dict,
    scenarios: List[str] | None = None,
):
    """Facet the annual ΔP/ΔT cloud by horizon with scenario-only identity."""
    required = {"model", "scenario", "member", "horizon", "precip", "temp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"change-factor cloud is missing columns {missing}")
    if not horizons:
        raise ValueError("at least one future horizon is required")

    scenarios = list(
        dict.fromkeys(
            str(value)
            for value in (
                scenarios
                if scenarios is not None
                else frame["scenario"].drop_duplicates().tolist()
            )
        )
    )
    palette = scenario_palette(scenarios)
    unknown = sorted(set(frame["scenario"].astype(str)) - set(palette))
    if unknown:
        raise ValueError(f"cloud contains unconfigured scenarios {unknown}")

    n_panels = len(horizons)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)
    width = figure_width_inches()
    x_limits = _cloud_limits(frame["precip"])
    y_limits = _cloud_limits(frame["temp"])

    with plt.rc_context(rcparams()):
        fig, axes_array = plt.subplots(
            nrows,
            ncols,
            figsize=(width, width * (0.48 * nrows)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        fig.subplots_adjust(
            left=0.10, right=0.98, bottom=0.16, top=0.75, hspace=0.42, wspace=0.08
        )
        axes = list(axes_array.flat)
        for axis in axes[n_panels:]:
            fig.delaxes(axis)
        axes = axes[:n_panels]

        for axis, (horizon, period) in zip(axes, horizons.items()):
            subset = frame.loc[frame["horizon"].astype(str) == str(horizon)]
            if subset.empty:
                raise ValueError(f"cloud has no points for configured horizon {horizon!r}")
            plotted = 0
            for scenario in scenarios:
                points = subset.loc[subset["scenario"].astype(str) == scenario]
                if points.empty:
                    continue
                axis.scatter(
                    points["precip"],
                    points["temp"],
                    color=palette[scenario],
                    s=22,
                    alpha=0.72,
                    edgecolors="none",
                )
                plotted += len(points)
            if plotted != len(subset):
                raise AssertionError(
                    f"{horizon!r}: plotted {plotted} of {len(subset)} combinations"
                )
            start, end = parse_horizon_period(period)
            axis.set_title(f"{horizon} ({start}–{end})")
            axis.axvline(0.0, color="0.35", lw=0.7, ls=(0, (3, 2)))
            axis.axhline(0.0, color="0.35", lw=0.7, ls=(0, (3, 2)))
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
            style_projection_axes(axis)

        for row in range(nrows):
            index = row * ncols
            if index < len(axes):
                axes[index].set_ylabel("Temperature change (°C)")
        bottom_row_start = (nrows - 1) * ncols
        for index, axis in enumerate(axes):
            if index >= bottom_row_start:
                axis.set_xlabel("Precipitation change (%)")

        handles = [
            Line2D(
                [],
                [],
                color=palette[scenario],
                marker="o",
                linestyle="none",
                markersize=4.5,
            )
            for scenario in scenarios
        ]
        fig.legend(
            handles,
            [scenario_label(scenario) for scenario in scenarios],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=max(1, len(handles)),
            frameon=False,
        )
        fig.suptitle("Change-factor cloud", fontsize=FONT_SIZE_TITLE, y=0.985)
    return fig


def summary_climate_proj(
    clim_dir: Union[Path, str],
    clim_files: List[Union[Path, str]],
    horizons: Dict,
    wide_dir: Union[Path, str, None] = None,
    cloud_path: Union[Path, str, None] = None,
):
    """
    Compute climate change statitistics for all models/scenario/horizons.

    Also prepare response surface plot.

    Output:
    - ``{wide_dir}/annual_change_scalar_stats_summary.nc`` — the wide merge, a
      JOB-INTERNAL intermediate since S8-05 (the caller's TemporaryDirectory).
      Read back by the tidy reshape, never shipped.
    - ``{clim_dir}/plots/`` — the ΔT/ΔP figure, the only artifact this
      function still produces.

    Parameters
    ----------
    clim_dir: Path
        Path to the projected climate directory of the project
    clim_files: List[Path, str]
        Path to the netcdf files of results per climate model / scenario / horizons
    horizons: Dict
        Time horizon names and start and end year separated with a comma.
        E.g {"far": "2070, 2100", "near": "2030, 2060"}
    """
    # merge summary maps across models, scnearios and horizons.
    prefix = "annual_change_scalar_stats"
    # for prefix in prefixes:
    log_row(f"merging netcdf files {prefix}", module="change")
    # Step 4c: filter_nonempty is GONE. It existed to drop the dummy empty
    # netCDFs stage A used to write for absent sources; since 4a an unresolved
    # combination never becomes a job, so every file in this list carries data and
    # dropping any of them would be silently shrinking the ensemble.
    # Eager, and closed — for two reasons, both learned the hard way on win-64.
    #
    # 1. HANDLES. Lazily-opened members stay open until the dataset is collected,
    #    so the caller cannot delete the files it just passed in. Step 4d made the
    #    per-point files job-internal (a TemporaryDirectory instead of Snakemake
    #    temp()), and its cleanup died with WinError 32 on exactly that. The leak
    #    predates 4d; it was invisible only because Snakemake deleted those files
    #    after the process had exited.
    # 2. DEADLOCK. `open_mfdataset` + `to_netcdf` reading from dask's thread pool
    #    parks forever on the HDF5 global lock — the failure diagnosed in bf1f4a5
    #    for plot_climate_proj_timeseries, fixed there the same way. This call site
    #    has the identical shape and had not been fixed.
    #
    # Value-neutral: `.load()` changes when the bytes are read, not what they are.
    with xr.open_mfdataset(
        clim_files, coords="minimal", preprocess=preprocess_coords
    ) as _ds_lazy:
        ds = _ds_lazy.load()
    dvars = ds.raster.vars
    # S8-05: the wide merge is a JOB-INTERNAL intermediate, not an artifact.
    #
    # It used to land as three files under summary/ --
    # `annual_change_scalar_stats_summary.{nc,csv}` and `_mean.csv`. The tidy
    # `{clim_project}_change_factors_{annual,monthly}.csv` supersede them: same
    # numbers, long format, per-row provenance, plus the future level the wide
    # form never carried. Verified before removal that nothing outside this
    # workflow read them -- `Snakefile_climate_experiment` and
    # `blueearth_cst/experiment/` reference them zero times, and rule 2.06
    # declared the `.nc` as an input it never opened.
    #
    # The `.nc` survives as an intermediate because the tidy reshape reads it
    # back: the table must describe what was PERSISTED, so a reshape can never
    # disagree with the artifact it claims to reshape. `wide_dir` is the caller's
    # TemporaryDirectory. The two CSVs are simply gone -- nothing read them and
    # nothing reads them back.
    wide_dir = wide_dir or os.path.join(clim_dir, "summary")
    os.makedirs(wide_dir, exist_ok=True)

    name_nc_out = f"{prefix}_summary.nc"
    ds.to_netcdf(
        os.path.join(wide_dir, name_nc_out),
        encoding={k: {"zlib": True} for k in dvars},
    )

    # One point per resolved combination and horizon. The dataframe stays wide
    # here because the annual tidy table is written immediately after this
    # helper returns; :func:`change_factor_cloud_frame` supplies the equivalent
    # direct-preview path from that durable table.
    df = ds.sel(stats="mean").to_dataframe().reset_index()
    scenarios = [str(value) for value in df["scenario"].drop_duplicates()]
    fig = plot_change_factor_cloud(df, horizons=horizons, scenarios=scenarios)
    cloud_path = cloud_path or (
        Path(clim_dir) / "plots" / "overview" / "change-factor-cloud.png"
    )
    save_figure(cloud_path, fig=fig, dpi=RASTER_DPI)
    plt.close(fig)


# NOTE: this module no longer runs as a Snakemake `script:`. Step 4d merged rules
# 2.04/2.05 into `derive_change_factors`, which imports the functions above and
# owns the orchestration. The former `__main__` block is deleted rather than left
# dead: it was a second copy of the per-point procedure, and two copies of the
# same arithmetic is how they drift apart. The functions stay here — they are the
# tested surface (tests/test_get_change_climate_proj*.py).
