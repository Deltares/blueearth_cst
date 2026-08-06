"""Update a wflow model with downscaled climate forcing for one realization."""
import os
from pathlib import Path

import numpy as np

from hydromt_wflow import WflowSbmModel

from blueearth_cst.shared.snake_utils import member_pointer_base, tee_to_log

# This script reads the snakemake global at module top level (no __name__
# guard), so the tee_to_log wrap must enclose the whole body — the top-level
# snakemake.* reads run at import before any function.
with tee_to_log(snakemake.log[0]):
    # Snakemake parameters
    config_out_fn = snakemake.output.toml
    fn_out = Path(snakemake.output.nc)
    fn_in = snakemake.input.nc
    data_libs = snakemake.input.data_sources
    model_root = snakemake.params.model_dir

    precip_source = snakemake.params.clim_source

    horizontime_climate = snakemake.params.horizontime_climate
    wflow_run_length = snakemake.params.run_length
    startyear = int(horizontime_climate - np.ceil(wflow_run_length / 2))
    endyear = int(horizontime_climate + np.round(wflow_run_length / 2))
    starttime = f"{startyear}-01-01T00:00:00"
    endtime = f"{endyear}-12-31T00:00:00"

    oro_source = f"{precip_source}_orography"
    pet_method = "makkink" if precip_source == "eobs" else "debruin"

    fn_in_path = Path(fn_in, resolve_path=True)
    climate_name = os.path.basename(fn_in_path).split(".")[0]

    config_out_fn = Path(config_out_fn)
    config_out_root = os.path.dirname(config_out_fn)
    config_out_name = os.path.basename(config_out_fn)

    # The run TOML lives in the experiment's hydrology/wflow/config/ and wflow's
    # own products in the sibling output/. Both are derived from the DECLARED
    # toml path -- never hardcoded -- so the layout stays owned by the rule.
    #
    # `run_name` is the toml stem, and R9 P2 commit 3 changed what that stem
    # SAYS without changing a line here: R07 had moved the realization index out
    # of the filename and into a `rlz_<r>/` directory, making the stem `cst_<m>`;
    # R9 removes the level and puts the index back, making it `rlz_<r>_cst_<m>`.
    # Every pointer built from `run_name` -- outstates, the output CSV, and now
    # the per-member log -- follows automatically. That is the payoff of deriving
    # from the declared path instead of reconstructing it.
    run_name, out_prefix = member_pointer_base(config_out_fn)

    # Instantiate model in r+ on the source root, then redirect writes to the
    # per-realization run directory by rebinding root.
    mod = WflowSbmModel(root=model_root, mode="r+", data_libs=data_libs)

    size = mod.staticmaps.data.raster.size
    if size > 1e6:
        chunksize = 1
    elif size > 2.5e5:
        chunksize = 30
    elif size > 1e5:
        chunksize = 100
    else:
        chunksize = 365

    mod.setup_config(
        data={
            # The R weathergen writes netcdfs with calendar=noleap. Keeping
            # noleap here would cause hydromt_wflow 1.x's forcing validation
            # to fail comparing cftime.DatetimeNoLeap against datetime.datetime.
            # Convert forcing time axis to standard calendar below and keep
            # the TOML in sync.
            "time.calendar": "standard",
            "time.starttime": starttime,
            "time.endtime": endtime,
            "time.timestepsecs": 86400,
            # Wflow.jl resolves output pointers against dirname(toml) +
            # dir_output; keep dir_output at the toml's own dir and carry the
            # config/ -> output/ hop in the pointers (see out_prefix above).
            "dir_output": ".",
            # Absolute paths into the wf1 model dir (staticmaps + instates).
            # The run dir is experiments/<name>/hydrology/wflow/config/, and it
            # has moved twice -- R07 B5 gave each realization its own rlz_<r>/
            # level, R9 P2 dissolved that level again -- which is exactly why
            # the "../" depth is not a literal anyone should maintain by hand.
            # Pass ABSOLUTE paths: hydromt_wflow's config.write re-relativizes any
            # absolute same-mount value against the new toml's own directory on
            # write, emitting the correct relative pointer (verified against the
            # vendored make_config_paths_relative; design §5/§5a).
            # state.path_input is inert under reinit=true but set for future
            # warm-state safety.
            "state.path_input": str(Path(model_root, "instate", "instates.nc").resolve()),
            "state.path_output": f"{out_prefix}outstates_{run_name}.nc",
            "input.path_static": str(Path(model_root, "staticmaps.nc").resolve()),
            "input.path_forcing": str(fn_out.resolve()),
            "output.csv.path": f"{out_prefix}{run_name}.csv",
            # R9 P2 commit 3 -- ships WITH the rlz_<r>/ flattening, never after.
            # Wflow's `[logging] path_log` defaults to `log.txt` beside the TOML.
            # While each realization owned a run directory that was already one
            # shared log per realization; removing the level puts EVERY member's
            # log at one path, and rule 3.10 batches members concurrently, so it
            # becomes a race rather than an overwrite. Measured on the
            # pre-flattening tree (R9 P1 observed tier): exactly two log.txt for
            # twelve members -- one per realization, six writers each. Keyed per
            # member here and derived from the same declared TOML path as every
            # other pointer, so the rule still owns the layout.
            "logging.path_log": f"{out_prefix}{run_name}.log",
        }
    )

    mod.setup_precip_forcing(
        precip_fn=climate_name,
        precip_clim_fn=None,
        chunksize=chunksize,
    )
    mod.setup_temp_pet_forcing(
        temp_pet_fn=climate_name,
        press_correction=True,
        temp_correction=True,
        dem_forcing_fn=oro_source,
        pet_method=pet_method,
        chunksize=chunksize,
    )

    # Convert forcing time axis from cftime.DatetimeNoLeap (R weathergen
    # default) to numpy datetime64 so hydromt_wflow 1.x's timespan
    # validation can compare it against datetime.datetime config values.
    # noleap doesn't have Feb 29, so the conversion is lossless.
    forcing = mod.forcing.data
    if hasattr(forcing.indexes["time"], "to_datetimeindex"):
        forcing["time"] = forcing.indexes["time"].to_datetimeindex(time_unit="ns")

    # weagen has off-by-one timestamps at the year boundaries; clip the forcing
    # in place via the component's data.
    for var in list(forcing.data_vars):
        forcing[var] = forcing[var].sel(time=slice(starttime, endtime))

    # Refresh starttime/endtime from the actual forcing axis (weagen quirk).
    last_var = next(iter(forcing.data_vars))
    times = forcing[last_var].time.values
    mod.config.set("time.starttime", str(times[0])[:19])
    mod.config.set("time.endtime", str(times[-1])[:19])

    # Write forcing + per-realization toml to absolute paths so the model root
    # (which is the source hydrology_model dir) doesn't have to be moved.
    mod.forcing.write(filename=str(fn_out.resolve()))
    mod.config.write(
        filename=config_out_name,
        config_root=Path(config_out_root).resolve(),
    )
    mod.close()  # commit any deferred writes
