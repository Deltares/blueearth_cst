# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

import json
import os

# gcsfs >= 2026.4 enables an experimental Extended filesystem by default that
# probes the bucket storage layout via an authenticated control-plane API. For
# public CMIP6 reads we have no credentials, so the probe fails and spams a
# warning per call before falling back. Disable the probe; reads still work.
os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")

import hydromt
import geopandas as gpd
import xarray as xr

from dask.diagnostics import ProgressBar

from blueearth_cst.projections import series_identity
from blueearth_cst.shared.snake_utils import log_row

# %%


def get_stats_clim_projections(
    data,
    name_clim_project,
    name_model,
    name_scenario,
    name_member,
    save_grids=False,
):
    """
    Parameters
    ----------
    data: dataset
        dataset for all available variables after opening data catalog
    name_clim_project : str
        name of the climate project (e.g. cmip5, cmip6, isimip3).
        should link to the name in the yml catalog.
    name_model : str
        model name of the climate model (e.g. ipsl, gfdl).
    name_scenario : str
        scenario name of the climate model (e.g. rcp4.5, rcp8.5).
    name_member : str
        member name of the climate model (e.g. r1i1p1f1).
    time_tuple : tuple
        time period over which to calculate statistics.
    save_grids : bool
        save gridded stats as well as scalar stats. False by default.

    Returns
    -------
    Writes a netcdf file with mean monthly precipitation and temperature regime (12 maps) over the geom.
    todo: Writes a csv file with mean monthly timeseries of precip and temp statistics (mean) over the geom

    """

    # get lat lon name of data
    XDIMS = ("x", "longitude", "lon", "long")
    YDIMS = ("y", "latitude", "lat")
    for dim in XDIMS:
        if dim in data.coords:
            x_dim = dim
    for dim in YDIMS:
        if dim in data.coords:
            y_dim = dim

    ds = []
    ds_scalar = []
    # filter variables for precip and temp
    # data_vars = list(data.data_vars)
    # var_list = [str for str in data_vars if any(sub in str for sub in variables)]
    # Units handling: the monthly aggregator is dispatched by variable NAME.
    # - "precip" is resampled with .sum("time") -> monthly TOTAL (a flux
    #   accumulated over the month); summing conserves the monthly water volume.
    # - every other variable (the else branch, i.e. "temp") is resampled with
    #   .mean("time") -> monthly MEAN (an intensive state; a total would be
    #   meaningless).
    # Because the dispatch keys on the string "precip", a catalog that names the
    # precipitation variable anything else is silently aggregated as temp (mean,
    # not sum) and its units are wrong. The required config is
    # `variables: [precip, temp]` (see dev/workflows/climate_projections.md).
    for var in data.data_vars:  # var_list:
        if var == "precip":
            var_m = data[var].resample(time="MS").sum("time")
        else:  # for temp
            # elif "temp" in var: #for temp
            var_m = data[var].resample(time="MS").mean("time")

        # get scalar average over grid for each month
        var_m_scalar = var_m.mean([x_dim, y_dim]).round(decimals=2)
        ds_scalar.append(var_m_scalar.to_dataset())

        # get grid average over time for each month
        if save_grids:
            # slice over time_tuple to save minimal required info for the grid
            # var_m = var_m.sel(time=slice(*time_tuple))
            var_mm = var_m.groupby("time.month").mean("time").round(decimals=2)
            ds.append(var_mm.to_dataset())

    # mean stats over grid and time
    mean_stats_time = xr.merge(ds_scalar)
    # add coordinate on project, model, scenario, realization to later merge all files
    mean_stats_time = mean_stats_time.assign_coords(
        {
            "clim_project": f"{name_clim_project}",
            "model": f"{name_model}",
            "scenario": f"{name_scenario}",
            "member": f"{name_member}",
        }
    ).expand_dims(["clim_project", "model", "scenario", "member"])

    if save_grids:
        mean_stats = xr.merge(ds)
        # add coordinate on project, model, scenario, realization to later merge all files
        mean_stats = mean_stats.assign_coords(
            {
                "clim_project": f"{name_clim_project}",
                "model": f"{name_model}",
                "scenario": f"{name_scenario}",
                "member": f"{name_member}",
            }
        ).expand_dims(["clim_project", "model", "scenario", "member"])

    else:
        mean_stats = xr.Dataset()

    return mean_stats, mean_stats_time


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):

            # Snakemake options
            project_dir = sm.params.project_dir
            region_path = sm.input.region_path
            catalog_path = sm.params.catalog_path
            name_scenario = sm.params.name_scenario
            name_members = sm.params.name_members
            name_model = sm.params.name_model
            name_clim_project = sm.params.name_clim_project
            variables = sm.params.variables
            save_grids = sm.params.save_grids

            # step 2b identity params (design §5.3, D9, D12). Read BEFORE the time
            # tuple below, which consumes acquisition_window.
            digest_components = dict(sm.params.digest_components)
            acquisition_window = tuple(sm.params.acquisition_window)
            store_index = sm.params.store_index
            buffer = float(sm.params.buffer_degrees)

            # Time tuple for timeseries.
            # cmip6 now comes from the DECLARED acquisition contract in params
            # (design §5.3) rather than from this branch: it is a digest component,
            # and it must be identical to what the Snakefile hashed. The values are
            # byte-identical to the cmip6 literals this branch used to carry, so
            # this is behaviour-preserving. cmip5/isimip3 keep their local branches
            # — WF2 v2.0 targets cmip6 and those spans have no declared contract.
            if name_clim_project == "cmip6":
                time_tuple_all = acquisition_window
            elif name_clim_project == "cmip5":
                if name_scenario == "historical":
                    # cmip5 historical 1850-2005
                    time_tuple_all = ("1950-01-01", "2005-12-31")
                else:
                    # cmip5 future 2006-2100
                    time_tuple_all = ("2006-01-01", "2100-12-31")
            else:  # isimip3
                if name_scenario == "historical":
                    # isimip3 historical 1850-2014
                    time_tuple_all = ("1991-01-01", "2014-12-31")
                else:
                    # isimip3 future 2015-2100 / p drive has gaps in between 2014-2021
                    time_tuple_all = ("2021-01-01", "2100-12-31")

            # additional folder structure info
            folder_model = os.path.join(project_dir, "hydrology_model")
            folder_out = os.path.join(project_dir, "climate_projections", name_clim_project)

            # makedirs, not mkdir-if-absent: the guarded mkdir raced whenever two
            # reduce jobs started together, which is almost certainly why rule 2.03
            # carried an ordering edge on 2.02's output that it never read. The edge
            # is gone (step 2b), so this must be race-free.
            os.makedirs(folder_out, exist_ok=True)

            # initialize model and region properties
            geom = gpd.read_file(region_path)
            bbox = list(geom.geometry.bounds.values[0])

            # --- D9 revalidation: decide BEFORE touching the network -----------
            # The polygon fingerprint cannot be a parse-time param (it does not
            # exist on a fresh project), so the digest is completed here from the
            # polygon just read. When every declared output already carries this
            # digest, the job is a no-op: the polygon was rewritten byte-identically
            # (a WF1 rerun), or a param changed that does not affect what was read.
            # This is what preserves the property ancient() used to buy, while
            # checking content instead of assuming it.
            region_fp = series_identity.region_fingerprint(region_path)
            expected_digest = series_identity.series_digest(digest_components, region_fp)
            declared_outputs = [str(p) for p in sm.output]

            if series_identity.cache_hit(declared_outputs, expected_digest):
                log_row(
                    f"cache_hit digest={expected_digest[:12]} "
                    f"({len(declared_outputs)} output(s) already current)",
                    module="stats",
                )
                for path in declared_outputs:
                    os.utime(path, None)  # refresh mtime so Snakemake sees it done
                raise SystemExit(0)

            log_row(
                f"deriving digest={expected_digest[:12]} region_fp={region_fp[:12]}",
                module="stats",
            )

            # initialize data_catalog from yml file
            data_catalog = hydromt.DataCatalog(data_libs=catalog_path)

            # check if model really exists from data catalog entry - else skip and provide empty ds??

            ds_members_mean_stats = []
            ds_members_mean_stats_time = []

            for name_member in name_members:
                log_row(f"{name_member}", module="stats")
                entry = f"{name_clim_project}_{name_model}_{name_scenario}_{name_member}"
                # D4 fail-fast (step 4c). The old `if entry in
                # data_catalog.sources: ... else: xr.Dataset()` guard silently
                # produced an EMPTY dataset for an absent source, which then
                # travelled downstream as a dummy netCDF and was filtered out at
                # merge -- a silent ensemble shrink. Since 4a, resolution decides
                # membership at DAG BUILD, so an absent entry never becomes a job
                # and reaching this point with one is a bug, not a data condition.
                if entry not in data_catalog.sources:
                    raise RuntimeError(
                        f"catalog entry {entry!r} is absent from {catalog_path} but a "
                        "reduce job was scheduled for it. DAG-build resolution "
                        "should have skipped this combination -- the catalog was "
                        "probably regenerated mid-run. Re-run to re-resolve."
                    )
                if True:
                    try:  # todo can this be replaced by if statement?
                        data = data_catalog.get_rasterdataset(
                            entry,
                            bbox=bbox,
                            buffer=buffer,
                            time_range=time_tuple_all,
                            variables=variables,
                        )
                        # needed for cmip5/cmip6 cftime.Datetime360Day which is not picked up before.
                        data = data.sel(time=slice(*time_tuple_all))
                        # hydromt 1.x leaves get_rasterdataset lazy-backed by the remote
                        # GCS chunks; without an eager .load() the downstream resample +
                        # to_netcdf round-trip rechunks via dask and ran for ~5 h per
                        # source on M2b. After bbox/time slicing the residual is small,
                        # so loading into memory is fine.
                        data = data.load()
                        # D8: the catalog URI globs {grid_label}/{version}, and
                        # ~6% of pinned stores match more than one. Two stores
                        # concatenated would give a DUPLICATED time axis, which a
                        # silent drop_duplicates would paper over while halving the
                        # effective record. Assert instead.
                        _t = data.indexes.get("time")
                        if _t is not None and len(_t) != len(set(_t)):
                            raise RuntimeError(
                                f"{entry}: time axis has {len(_t) - len(set(_t))} "
                                "duplicate step(s), so the catalog glob matched more "
                                "than one store. Pin the version in the catalog "
                                "rather than reading an ambiguous source."
                            )
                    except Exception:  # narrowed from bare `except:` — catches the same
                        # normal data-load errors; only BaseException
                        # (KeyboardInterrupt/SystemExit/GeneratorExit) now propagates
                        # instead of being swallowed, so this is output-neutral on any
                        # completing run (a run never raises those during a data load).
                        # if it is not possible to open all variables at once, loop over each one, remove duplicates and then merge:
                        ds_list = []
                        for var in variables:
                            try:
                                data_ = data_catalog.get_rasterdataset(
                                    entry,
                                    bbox=bbox,
                                    buffer=buffer,
                                    time_range=time_tuple_all,
                                    variables=[var],
                                )
                                # drop duplicates if any
                                data_ = data_.drop_duplicates(dim="time", keep="first").load()
                                ds_list.append(data_)
                            except Exception as exc:
                                # D4 fail-fast (step 4c): previously this logged
                                # "<var> not found" and CONTINUED, so a failed read
                                # silently dropped a variable and the run produced
                                # change factors from an incomplete set. A published
                                # source that cannot be read is a failure, not an
                                # absence -- absence is decided at DAG build (4a).
                                raise RuntimeError(
                                    f"reading {var!r} from {entry} failed: {exc}. "
                                    "This is a read failure, not an unpublished "
                                    "source -- resolution already confirmed the "
                                    "source exists. Re-run to retry; the series "
                                    "cache makes a retry cheap."
                                ) from exc
                        # merge all variables back to data
                        data = xr.merge(ds_list)

                    # calculate statistics
                    mean_stats, mean_stats_time = get_stats_clim_projections(
                        data,
                        name_clim_project,
                        name_model,
                        name_scenario,
                        name_member,
                        save_grids=save_grids,
                    )

                # merge members results
                ds_members_mean_stats.append(mean_stats)
                ds_members_mean_stats_time.append(mean_stats_time)

            if save_grids:
                nc_mean_stats = xr.merge(ds_members_mean_stats)
            else:
                nc_mean_stats = xr.Dataset()
            nc_mean_stats_time = xr.merge(ds_members_mean_stats_time)

            # write netcdf:

            # use hydromt function instead to write to netcdf?
            dvars = nc_mean_stats_time.raster.vars

            # Step 3: the series path comes from the rule, not from a filename
            # convention rebuilt here. One stage-A rule fans out over {series_key},
            # so the job is told where to write rather than inferring it from
            # (model, scenario) -- which is what forced two rules before.
            series_nc_out = sm.params.series_nc_out

            # The gridded families keep their legacy names for now: they are
            # save_grids-only, undeclared, and R2/D11 restructure them at step 7.
            if name_scenario == "historical":
                name_nc_out = f"historical_stats_{name_model}.nc"
            else:
                name_nc_out = f"stats-{name_model}_{name_scenario}.nc"

            # --- step 2b: stamp the identity onto the product -------------------
            # These attributes are what make the persistent series self-describing
            # and what the consumer asserts against (design §5.3 schema table).
            # Without them a file on disk is indistinguishable from one derived
            # against a different polygon, catalog entry or reducer.
            entry_meta = digest_components.get("entry_identity", {})
            nc_mean_stats_time.attrs.update(
                {
                    "cst_schema_version": series_identity.SCHEMA_VERSION,
                    "cst_series_digest": expected_digest,
                    "cst_catalog_entry": digest_components.get("catalog_entry", ""),
                    "cst_acquisition_window": " / ".join(acquisition_window),
                    "cst_region_bounds": ", ".join(f"{b:.9g}" for b in bbox),
                    "cst_region_fingerprint": region_fp,
                    "cst_buffer_degrees": buffer,
                    "cst_reducer_module_hash": digest_components.get(
                        "reducer_module_hash", ""
                    ),
                    "cst_members": ", ".join(digest_components.get("members", [])),
                    # D12: the physical stores this series actually read, per
                    # member -- the identity the entry name alone cannot carry
                    # because {variable}/*/* globs grid label and version.
                    "cst_source_paths": json.dumps(
                        digest_components.get("pins", {}), sort_keys=True
                    ),
                    "cst_crs": str(
                        (entry_meta.get(digest_components.get("members", [""])[0], {})
                         if entry_meta else {}).get("metadata", {}).get("crs", "")
                    ),
                    # Today's unweighted spatial mean. Becomes spherical cell-area
                    # weighting at step 5a (D10); recorded now so a series always
                    # says which scheme produced it.
                    "cst_weighting_scheme": "unweighted_mean_pre_5a",
                }
            )

            log_row(f"writing series to {os.path.basename(series_nc_out)}", module="stats")
            os.makedirs(os.path.dirname(series_nc_out), exist_ok=True)
            delayed_obj = nc_mean_stats_time.to_netcdf(
                series_nc_out,
                encoding={k: {"zlib": True} for k in dvars},
                compute=False,
            )
            with ProgressBar():
                delayed_obj.compute()

            if save_grids:
                log_row("writing stats over grid to nc", module="stats")
                delayed_obj = nc_mean_stats.to_netcdf(
                    os.path.join(folder_out, name_nc_out),
                    encoding={k: {"zlib": True} for k in dvars},
                    compute=False,
                )
                with ProgressBar():
                    delayed_obj.compute()
    else:
        raise RuntimeError(
            "get_stats_climate_proj.py runs only as a Snakemake script:"
        )
