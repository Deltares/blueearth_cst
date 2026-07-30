"""Stage B — all change factors in ONE job (migration step 4d, design §8).

Replaces the pair `monthly_change` (fanned out per `point_key` × horizon) and
`monthly_change_scalar_merge` (a single aggregator over their `temp()` outputs).
The design's rule table gives stage B **1 job** with no fan-out
(`wf2-climate-analysis-v2-design.md` §5, "B. Derive"), reading the explicit
expanded series list.

**This step is value-neutral by construction.** The arithmetic is not reimplemented
here: `get_change_annual_clim_proj`, `get_change_clim_projections` and
`summary_climate_proj` are imported from the two modules that already held them,
unchanged. Only the orchestration moves. A non-zero characterized diff on the
summary artifacts is therefore a defect in this file, not a judgement call — which
is the whole reason the functions were left where they were.

What changes shape:

* the per-point `annual_change_scalar_stats-{point_key}_{horizon}.nc` files were
  Snakemake `temp()` outputs and are now **job-internal intermediates** with the
  same lifetime — written, consumed by the merge, removed. `summary_climate_proj`
  reads model/scenario/horizon from dataset *coords*, never from the filename, so
  relocating them is safe (checked before the move, not assumed);
* one log and one benchmark instead of a per-part tree under `2.04_monthly_change/`.

Stage B's input set is explicit (design risk-06 / revision 4): the rule declares
exactly the expanded `{series_key}` list built from the resolved combination set,
and this job **asserts that the set it opened equals that list**. A model removed
from the config cannot rejoin the run through a leftover file in `series/`.

Invoked from ``Snakefile_climate_projections`` via ``script:``; reads
``snakemake.input/output/params``, never ``sys.argv``.
"""
# NOTE: no `from __future__ import annotations` here — Snakemake's `script:`
# directive prepends its own preamble to a copy of this file, so a __future__
# import lands mid-file and raises SyntaxError at job start. A --dry-run cannot
# catch it (it never executes a script body); the other `script:` modules in this
# repo omit it for the same reason.

import os
import tempfile

import xarray as xr

from blueearth_cst.projections import series_identity
from blueearth_cst.projections.get_change_climate_proj import (
    _to_str_tuple,
    get_change_annual_clim_proj,
    get_change_clim_projections,
)
from blueearth_cst.projections.get_change_climate_proj_summary import (
    summary_climate_proj,
)
from blueearth_cst.shared.snake_utils import log_row, tee_to_log

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")


# `_to_str_tuple` is IMPORTED, not reimplemented, despite the leading underscore.
# A local copy was written first and was already wrong: it raised on `[]`, where
# the original returns `()` — a contract `tests/test_get_change_climate_proj.py`
# pins. Reimplementing a normaliser is exactly the drift this step is meant to
# avoid, so the private name is the lesser evil.


def derive_one_point(
    *,
    series_path_hist,
    series_path,
    change_nc_out,
    time_tuple_hist,
    time_tuple_fut,
    name_horizon,
    name_model,
    name_scenario,
    region_fp,
    digest_components_hist,
    digest_components_fut,
    save_grids=False,
    stats_path_hist=None,
    stats_path=None,
    clim_project_dir=None,
):
    """Change factors for one (model, scenario, member) at one horizon.

    The body is the former ``monthly_change`` job, moved verbatim apart from
    taking its inputs as arguments instead of reading ``snakemake.params``.
    """
    # --- step 2b backstop: the series must match the current inputs -----------
    # Design D9 route (b) / risk-03 mechanism 2. An assertion INSIDE the job, not
    # a scheduling property, so it holds however Snakemake was invoked -- a series
    # restored from a backup, produced by an older checkout, or surviving a
    # non-default --rerun-triggers still fails the run instead of quietly entering
    # the change factors.
    for label, path, components in (
        ("historical", series_path_hist, dict(digest_components_hist)),
        (name_scenario, series_path, dict(digest_components_fut)),
    ):
        series_identity.assert_series_identity(
            path,
            series_identity.series_digest(components, region_fp),
            f"{name_model} {label}",
        )

    ds_hist_time = xr.open_dataset(series_path_hist)
    ds_clim_time = xr.open_dataset(series_path)

    if save_grids:
        ds_hist = xr.open_dataset(stats_path_hist)
        ds_clim = xr.open_dataset(stats_path)

    # Step 4c: the `if len(ds_clim_time) > 0` guard and its dummy-netCDF
    # else-branch are gone. Since 4a an unresolved combination never becomes a
    # job, so an empty series here means a real defect.
    if len(ds_clim_time) == 0:
        raise RuntimeError(
            f"{series_path} holds no data variables. Resolution admitted this "
            "combination, so an empty series is a defect rather than an "
            "unpublished source -- delete the series and re-run to re-derive."
        )

    ds_hist_time = ds_hist_time.sel(time=slice(*time_tuple_hist))
    ds_clim_time = ds_clim_time.sel(time=slice(*time_tuple_fut))
    stats_annual_change = get_change_annual_clim_proj(ds_hist_time, ds_clim_time)
    stats_annual_change = stats_annual_change.assign_coords(
        {"horizon": f"{name_horizon}"}
    ).expand_dims(["horizon"])
    stats_annual_change = stats_annual_change.transpose(
        ..., "clim_project", "model", "scenario", "horizon", "member"
    )

    dvars = stats_annual_change.raster.vars
    stats_annual_change.to_netcdf(
        change_nc_out, encoding={k: {"zlib": True} for k in dvars}
    )

    if save_grids:
        # Cold branch: shipped configs set `save_grids: false`, and step 5e is
        # where it is restructured (`save_grids` -> `save_gridded`, OQ-12). Moved
        # as-is rather than tidied, so 4d stays value-neutral.
        if len(ds_clim) > 0:
            monthly_change_mean_grid = get_change_clim_projections(ds_hist, ds_clim)
            monthly_change_mean_grid = monthly_change_mean_grid.assign_coords(
                {"horizon": f"{name_horizon}"}
            ).expand_dims(["horizon"])
            log_row("writing netcdf files monthly_change_mean_grid", module="change")
            dvars = monthly_change_mean_grid.raster.vars
            grid_model = monthly_change_mean_grid.model.values[0]
            grid_scenario = monthly_change_mean_grid.scenario.values[0]
            grid_horizon = monthly_change_mean_grid.horizon.values[0]
            name_nc_out = (
                f"monthly_change_mean_grid-{grid_model}_{grid_scenario}_{grid_horizon}.nc"
            )
            monthly_change_mean_grid.to_netcdf(
                os.path.join(clim_project_dir, name_nc_out),
                encoding={k: {"zlib": True} for k in dvars},
            )
        else:
            name_nc_out = (
                f"monthly_change_mean_grid-{name_model}_{name_scenario}_{name_horizon}.nc"
            )
            xr.Dataset().to_netcdf(os.path.join(clim_project_dir, name_nc_out))

    ds_hist_time.close()
    ds_clim_time.close()


if "snakemake" in globals():
    sm = globals()["snakemake"]

    with tee_to_log(sm.log[0]):
        clim_project_dir = sm.params.clim_project_dir
        horizons = sm.params.horizons
        save_grids = sm.params.save_grids
        points = [dict(p) for p in sm.params.points]

        # D9: every expected digest is recomputed against the polygon ON DISK, so
        # a series derived for a different region cannot be reused.
        region_fp = series_identity.region_fingerprint(sm.input.region_path)

        # risk-06 / revision 4: the set opened must equal the set declared. A
        # leftover file in series/ cannot rejoin a run whose config dropped it.
        declared = {os.path.abspath(str(p)) for p in sm.input.series_nc}
        opened = {
            os.path.abspath(str(path))
            for point in points
            for path in (point["series_path_hist"], point["series_path"])
        }
        if opened != declared:
            raise RuntimeError(
                "derive_change_factors: the series set to open does not equal the "
                "declared input set.\n"
                f"  declared but unused: {sorted(declared - opened)}\n"
                f"  used but undeclared: {sorted(opened - declared)}"
            )

        log_row(
            f"deriving change factors for {len(points)} point(s) x "
            f"{len(horizons)} horizon(s)",
            module="change",
        )

        # The per-point files were `temp()` rule outputs; they are job-internal
        # now, with the same lifetime. TemporaryDirectory removes them even if the
        # merge raises, which the old temp() could not promise mid-DAG.
        with tempfile.TemporaryDirectory(prefix="cst_change_") as work_dir:
            change_files = []
            for point in points:
                for horizon_name, horizon_window in horizons.items():
                    out_nc = os.path.join(
                        work_dir,
                        f"annual_change_scalar_stats-{point['point_key']}"
                        f"_{horizon_name}.nc",
                    )
                    derive_one_point(
                        series_path_hist=point["series_path_hist"],
                        series_path=point["series_path"],
                        change_nc_out=out_nc,
                        time_tuple_hist=_to_str_tuple(sm.params.time_horizon_hist),
                        time_tuple_fut=_to_str_tuple(horizon_window),
                        name_horizon=horizon_name,
                        name_model=point["model"],
                        name_scenario=point["scenario"],
                        region_fp=region_fp,
                        digest_components_hist=point["digest_components_hist"],
                        digest_components_fut=point["digest_components_fut"],
                        save_grids=save_grids,
                        stats_path_hist=point.get("stats_path_hist"),
                        stats_path=point.get("stats_path"),
                        clim_project_dir=clim_project_dir,
                    )
                    change_files.append(out_nc)

            log_row(
                f"merging {len(change_files)} change file(s) into the summary",
                module="change",
            )
            summary_climate_proj(
                clim_dir=clim_project_dir,
                clim_files=change_files,
                horizons=horizons,
            )
