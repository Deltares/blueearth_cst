"""
Open monthly change files for all models/scenarios/horizon and compute/plot statistics
"""

import os
from pathlib import Path
from typing import Dict, List, Union

import hydromt  # noqa: F401 -- registers the xarray .raster accessor (ds.raster.vars below)
import xarray as xr

from blueearth_cst.projections import projection_figures, projection_plots
from blueearth_cst.shared.snake_utils import log_row


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


def summary_climate_proj(
    clim_dir: Union[Path, str],
    clim_files: List[Union[Path, str]],
    horizons: Dict,
    wide_dir: Union[Path, str, None] = None,
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
    # workflow read them -- `run_stress_test.smk` and
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

    # just keep mean for temp and precip for the change-factor cloud
    df = ds.sel(stats="mean").to_dataframe().reset_index()
    missing = [c for c in ("model", "scenario", "horizon") if c not in df.columns]
    if missing:
        raise ValueError(
            f"the stage-B merge carries no {missing} coordinate(s); the "
            "change-factor cloud cannot be keyed by combination"
        )

    plots_dir = Path(clim_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # One frame per horizon, keyed by the horizon's CONFIGURED NAME.
    #
    # The figures this replaces relabelled each horizon to the middle year of
    # its period, which is why the old cloud's legend read "2080" for a horizon
    # the config calls `far`. The new set carries name AND years in the caveat
    # and in the legend, so the relabelling is gone rather than reproduced:
    # nothing else in WF2 knows a horizon by its midpoint, and a figure that
    # names artifacts differently from every table beside it is the kind of
    # small divergence a reader pays for.
    changes = {}
    for name in horizons:
        frame = df[df["horizon"] == name]
        if frame.empty:
            # Loud, because an absent horizon silently drops a whole panel and
            # the figure still renders as a plausible-looking cloud.
            raise ValueError(
                f"no rows for horizon {name!r} in the stage-B merge; "
                f"have {sorted(df['horizon'].unique())}"
            )
        changes[name] = frame.rename(
            columns={"precip": "precip_change", "temp": "temp_change"}
        )

    periods = {
        name: projection_figures.parse_horizon_period(period)
        for name, period in horizons.items()
    }

    faceted = plots_dir / projection_figures.CLOUD_FACETED_PATH
    faceted.parent.mkdir(parents=True, exist_ok=True)
    points = projection_plots.draw_cloud_faceted(changes, periods, faceted)
    log_row(
        f"Wrote {projection_figures.CLOUD_FACETED_PATH} ({points} points)",
        module="change",
    )

    # The combined view exists to show how far the cloud TRAVELS between
    # horizons, so a single-horizon project does not get one -- it would be the
    # faceted figure drawn again under a second name. `figure_relative_paths`
    # applies the same rule, so the Snakefile declares exactly what is written.
    if len(periods) > 1:
        combined = plots_dir / projection_figures.CLOUD_COMBINED_PATH
        points = projection_plots.draw_cloud_combined(changes, periods, combined)
        log_row(
            f"Wrote {projection_figures.CLOUD_COMBINED_PATH} ({points} points)",
            module="change",
        )


# NOTE: this module no longer runs as a Snakemake `script:`. Step 4d merged rules
# 2.04/2.05 into `derive_change_factors`, which imports the functions above and
# owns the orchestration. The former `__main__` block is deleted rather than left
# dead: it was a second copy of the per-point procedure, and two copies of the
# same arithmetic is how they drift apart. The functions stay here — they are the
# tested surface (tests/test_get_change_climate_proj*.py).
