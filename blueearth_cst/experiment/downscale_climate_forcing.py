"""Update a wflow model with downscaled climate forcing for one realization."""

import os
from copy import deepcopy
from pathlib import Path

import hydromt
import yaml
from hydromt_wflow import WflowSbmModel

from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
    prepare_clim_data_catalog,
)
from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
    resolved_unit_interpretation as resolved_unit_interpretation,
)
from blueearth_cst.experiment.forcing_descriptor import (
    ClimateArtifact,
    describe_forcing,
)
from blueearth_cst.experiment.forcing_window import forcing_window
from blueearth_cst.shared.progress import hydromt_progress
from blueearth_cst.shared.provenance import file_sha256


def collection_run_forcing(
    manifest_path, run_id, catalog_out, *, validated_collection=None, preparation=None
):
    """Bind a validated retained collection to the neutral preparation adapter.

    Only the transient, per-run catalog is written. No source catalog or source
    label participates in choosing elevation, PET or units at consumption time.
    The caller supplies an output path outside the immutable collection. An
    invocation controller may pass its fully validated collection snapshot to
    avoid scanning every other run for each job. The marker, context, selected
    forcing and ancillary bytes are still rechecked before writing anything.
    """
    from blueearth_cst.experiment.forcing_descriptor import reader_unit_interpretation
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulator_adapter import (
        ArtifactReference,
        PreparationContext,
        RunForcing,
    )
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    manifest_path = Path(manifest_path).resolve(strict=True)
    project = manifest_path.parents[4]
    output = Path(catalog_out).resolve()
    if output.is_relative_to(manifest_path.parent):
        raise ValueError("transient preparation catalog must be outside the collection")
    if validated_collection is None:
        manifest = read_collection_v2(manifest_path)
    else:
        if manifest_path.name != "collection.json":
            raise ValueError("expected collection.json ready marker")
        manifest = read_collection_v2(manifest_path)
        if manifest != validated_collection:
            raise ValueError("collection changed after invocation validation")
    if preparation is None:
        raise ValueError("WF4 requires checked simulation preparation")
    selected = [item for item in manifest["series"] if item["run_id"] == run_id]
    if len(selected) != 1:
        raise ValueError(f"collection has no unique forcing for run_id={run_id}")
    item = selected[0]
    forcing_path = resolve_file_reference(item["file"], {"project_root": project})
    catalog_path = resolve_file_reference(
        preparation["catalog"], {"project_root": project}
    )
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    elevation = preparation["forcing_elevation"]
    ancillary = [
        entry
        for entry in preparation["ancillary"]
        if entry["id"] == elevation["artifact_id"]
    ]
    if len(ancillary) != 1 or elevation["catalog_key"] != "elevation":
        raise ValueError("Wflow preparation requires a unique retained elevation")
    elevation_path = resolve_file_reference(
        ancillary[0]["file"], {"project_root": project}
    )
    catalog["elevation"]["uri"] = str(elevation_path)
    forcing_key = run_id
    if forcing_key in catalog:
        raise ValueError("ancillary catalog uses reserved generated forcing key")
    reader = preparation["generated_forcing_reader"]
    catalog[forcing_key] = deepcopy(reader)
    catalog[forcing_key]["uri"] = str(forcing_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(catalog, sort_keys=True), encoding="utf-8")
    physical_context = PreparationContext(
        (ArtifactReference(output, file_sha256(output)),),
        (ArtifactReference(elevation_path, ancillary[0]["file"]["sha256"]),),
        forcing_key,
        elevation["catalog_key"],
        preparation["pet_method"],
        True,
        True,
        ("cftime_to_datetime64", "clip_to_configured_window", "refresh_toml_endpoints"),
    )
    descriptor = describe_forcing(
        ClimateArtifact(run_id, forcing_path),
        reader_unit_interpretation(reader),
        time_label="interval_end",
    )
    return RunForcing(
        run_id,
        forcing_path,
        item["file"]["sha256"],
        descriptor,
        physical_context,
        manifest["collection_id"],
        manifest["collection_revision"],
    )


def forcing_chunksize(size):
    """Pick a time-chunk size for a staticmaps grid of ``size`` cells.

    Bigger grids get shorter time chunks so one chunk stays a workable size in
    memory. Thresholds are the ones this workflow has always used.
    """
    if size > 1e6:
        return 1
    if size > 2.5e5:
        return 30
    if size > 1e5:
        return 100
    return 365


def pet_method_for(precip_source):
    """Return the PET method matching ``precip_source``.

    E-OBS carries no radiation, so it takes Makkink; everything else takes
    De Bruin.
    """
    return "makkink" if precip_source == "eobs" else "debruin"


def _as_list(data_libs):
    """The catalogs as a plain list, whether one path or several came in."""
    if isinstance(data_libs, (str, os.PathLike)):
        return [os.fspath(data_libs)]
    return [os.fspath(item) for item in data_libs]


def downscale_climate_forcing(
    config_out_fn,
    fn_out,
    fn_in,
    data_libs,
    model_root,
    precip_source,
    sim_start,
    sim_end,
    catalog_out,
    oro_path=None,
    *,
    run_id,
    unit_interpretation,
    native_output_path,
    native_log_path,
):
    """Resolve current source-specific context outside the neutral simulator.

    P1 retains the current catalogs and ancillary locations. Their fingerprints
    form a transient context, not the portable package introduced in P2.
    Effective units arrive from an independently reviewed catalog binding.
    """
    from blueearth_cst.experiment.simulator_adapter import (
        ArtifactReference,
        ForcingRequirement,
        ModelReference,
        PreparationContext,
        PreparationSettings,
        RunForcing,
        prepare,
        validate_forcing,
    )

    starttime, endtime = forcing_window(sim_start, sim_end)
    oro_source = f"{precip_source}_orography"
    pet_method = pet_method_for(precip_source)
    data_libs = _as_list(data_libs)
    prepare_clim_data_catalog(
        fns=[fn_in],
        data_libs_like=data_libs,
        source_like=precip_source,
        fn_out=catalog_out,
        oro_path=oro_path,
    )
    data_libs = [os.fspath(catalog_out), *data_libs]
    catalog = hydromt.DataCatalog(data_libs=data_libs)
    elevation_path = Path(catalog.get_source(oro_source).full_uri)
    context = PreparationContext(
        catalogs=tuple(
            ArtifactReference(Path(path), file_sha256(path)) for path in data_libs
        ),
        ancillary=(ArtifactReference(elevation_path, file_sha256(elevation_path)),),
        forcing_entry=Path(fn_in).stem,
        elevation_entry=oro_source,
        pet_method=pet_method,
        press_correction=True,
        temp_correction=True,
        conversions=(
            "cftime_to_datetime64",
            "clip_to_configured_window",
            "refresh_toml_endpoints",
        ),
    )
    descriptor = describe_forcing(
        ClimateArtifact(run_id, Path(fn_in)),
        unit_interpretation,
        time_label="interval_end",
    )
    required_variables = (
        ("precip", "mm/day"),
        ("temp", "degC"),
        ("press_msl", "hPa"),
        ("kin", "W/m2"),
    )
    if pet_method == "debruin":
        required_variables += (("kout", "W/m2"),)
    requirement = ForcingRequirement(
        variables=required_variables,
        crs="EPSG:4326",
        dimensions=("longitude", "latitude", "time"),
        calendars=("noleap", "standard", "proleptic_gregorian"),
        timestep_seconds=86400,
        time_label="interval_end",
        first_time=starttime,
        last_time=endtime,
        allow_missing=False,
    )
    forcing = RunForcing(
        run_id,
        Path(fn_in),
        file_sha256(fn_in),
        descriptor,
        context,
        None,
        None,
    )
    return prepare(
        run_id,
        validate_forcing(forcing, requirement),
        ModelReference(Path(model_root), requirement),
        PreparationSettings(
            Path(config_out_fn),
            Path(fn_out),
            Path(native_output_path),
            Path(native_log_path),
            starttime,
            endtime,
        ),
    )


def prepare_model_forcing(run_forcing, model_reference, settings):
    """Apply the predecessor HydroMT preparation with explicit physical context."""
    context = run_forcing.preparation_context
    temporal_operations = []
    fn_out = settings.forcing_path
    starttime, endtime = settings.first_time, settings.last_time
    model_root = model_reference.root
    data_libs = [str(item.path) for item in context.catalogs]
    climate_name = context.forcing_entry
    oro_source = context.elevation_entry
    pet_method = context.pet_method
    config_out_fn = settings.toml_path
    config_out_root = os.path.dirname(config_out_fn)
    config_out_name = os.path.basename(config_out_fn)
    run_name = run_forcing.run_id
    settings.native_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate model in r+ on the source root, then redirect writes to the
    # per-realization run directory by rebinding root.
    mod = WflowSbmModel(root=model_root, mode="r+", data_libs=data_libs)

    # Match setup_temp_pet_forcing's spatial reads so HydroMT wraps global
    # elevation longitudes before checking the grid actually used downstream.
    from blueearth_cst.experiment.simulator_adapter import validate_ancillary_grid

    physical_catalog = hydromt.DataCatalog(data_libs=data_libs)
    forcing_grid = physical_catalog.get_rasterdataset(
        climate_name,
        geom=mod.region,
        buffer=1,
        time_range=(starttime, endtime),
        variables=["temp"],
    )
    if (
        run_forcing.descriptor.calendar == "noleap"
        and str(forcing_grid.time.dt.calendar) == "proleptic_gregorian"
    ):
        temporal_operations.append("hydromt_reader_to_datetime64")
    elevation_grid = physical_catalog.get_rasterdataset(
        oro_source, geom=forcing_grid.raster.box, buffer=2, variables=["elevtn"]
    )
    try:
        validate_ancillary_grid(
            run_name, forcing_grid, elevation_grid, context.ancillary[0].path
        )
    finally:
        forcing_grid.close()
        elevation_grid.close()

    chunksize = forcing_chunksize(mod.staticmaps.data.raster.size)

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
            # config/ -> output/ hop in the explicit native pointers below.
            "dir_output": ".",
            # Absolute paths into the wf1 model dir (staticmaps + instates).
            # The run dir is experiments/<name>/hydrology/wflow/config/, and it
            # has moved twice -- R07 B5 gave each realization its own rlz_<r>/
            # level, R9 P2 dissolved that level again -- which is exactly why
            # the "../" depth is not a literal anyone should maintain by hand.
            # Pass ABSOLUTE paths: hydromt_wflow's config.write re-relativizes any
            # absolute same-mount value against the new toml's own directory on
            # write, emitting the correct relative pointer (verified against the
            # vendored make_config_paths_relative; design Â§5/Â§5a).
            # state.path_input is inert under reinit=true but set for future
            # warm-state safety.
            "state.path_input": str(
                Path(model_root, "instate", "instates.nc").resolve()
            ),
            "input.path_static": str(Path(model_root, "staticmaps.nc").resolve()),
            "input.path_forcing": str(fn_out.resolve()),
            "output.csv.path": os.path.relpath(
                settings.native_output_path, config_out_root
            ).replace("\\", "/"),
            # R9 P2 commit 3 -- ships WITH the rlz_<r>/ flattening, never after.
            # Wflow's `[logging] path_log` defaults to `log.txt` beside the TOML.
            # While each realization owned a run directory that was already one
            # shared log per realization; removing the level puts EVERY member's
            # log at one path, and rule 4.05 batches members concurrently, so it
            # becomes a race rather than an overwrite. Measured on the
            # pre-flattening tree (R9 P1 observed tier): exactly two log.txt for
            # twelve members -- one per realization, six writers each. Keyed per
            # member here through the explicit native log path; the rule owns
            # the layout and the adapter only relativizes its declared pointer.
            "logging.path_log": os.path.relpath(
                settings.native_log_path, config_out_root
            ).replace("\\", "/"),
        }
    )

    # WF3 consumes only the CSV. Remove the inherited output pointer while
    # retaining state.path_input and state.variables for warm initialization.
    mod.config.data.get("state", {}).pop("path_output", None)

    mod.setup_precip_forcing(
        precip_fn=climate_name,
        precip_clim_fn=None,
        chunksize=chunksize,
    )
    mod.setup_temp_pet_forcing(
        temp_pet_fn=climate_name,
        press_correction=context.press_correction,
        temp_correction=context.temp_correction,
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
        temporal_operations.append("cftime_to_datetime64")

    # weathergen has off-by-one timestamps at the year boundaries; clip the forcing
    # in place via the component's data.
    for var in list(forcing.data_vars):
        forcing[var] = forcing[var].sel(time=slice(starttime, endtime))
    temporal_operations.append("clip_to_configured_window")

    # Refresh starttime/endtime from the actual forcing axis (weathergen quirk).
    last_var = next(iter(forcing.data_vars))
    times = forcing[last_var].time.values
    mod.config.set("time.starttime", str(times[0])[:19])
    mod.config.set("time.endtime", str(times[-1])[:19])
    temporal_operations.append("refresh_toml_endpoints")

    # Write forcing + per-realization toml to absolute paths so the model root
    # (which is the source hydrology_model dir) doesn't have to be moved.
    # "Run 14 forcing", matching the Wflow bar's "Run 14" for the same member.
    with hydromt_progress(f"Run {run_name} forcing"):
        mod.forcing.write(filename=str(fn_out.resolve()))
    mod.config.write(
        filename=config_out_name,
        config_root=Path(config_out_root).resolve(),
    )
    if settings.temporal_path is not None:
        import pandas as pd

        from blueearth_cst.experiment.content_identity import canonical_json_bytes

        prepared_start, prepared_end = str(times[0])[:19], str(times[-1])[:19]
        temporal = {
            "source_calendar": run_forcing.descriptor.calendar,
            "prepared_forcing_calendar": str(forcing.time.dt.calendar),
            "response_calendar": mod.config.data["time"]["calendar"],
            "operations": temporal_operations,
            "prepared_start": prepared_start,
            "prepared_end": prepared_end,
            "response_start": str(
                pd.Timestamp(prepared_start)
                + pd.Timedelta(seconds=mod.config.data["time"]["timestepsecs"])
            ),
            "response_end": str(pd.Timestamp(prepared_end)),
            "time_label": "interval_end",
        }
        settings.temporal_path.write_bytes(canonical_json_bytes(temporal))
    mod.close()  # commit any deferred writes


def prepare_collection_run(
    manifest_path,
    run_id,
    *,
    model_root,
    catalog_out,
    config_out,
    forcing_out,
    native_output,
    native_log,
    sim_start,
    sim_end,
    temporal_out,
    validated_collection=None,
    preparation=None,
):
    """Prepare one retained member with the reviewed physical Wflow binding."""
    from blueearth_cst.experiment.simulator_adapter import (
        ForcingRequirement,
        ModelReference,
        PreparationSettings,
        prepare,
        validate_forcing,
    )

    forcing = collection_run_forcing(
        manifest_path,
        run_id,
        catalog_out,
        validated_collection=validated_collection,
        preparation=preparation,
    )
    first, last = forcing_window(sim_start, sim_end)
    variables = (
        ("precip", "mm/day"),
        ("temp", "degC"),
        ("press_msl", "hPa"),
        ("kin", "W/m2"),
    )
    if forcing.preparation_context.pet_method == "debruin":
        variables += (("kout", "W/m2"),)
    requirement = ForcingRequirement(
        variables,
        "EPSG:4326",
        ("longitude", "latitude", "time"),
        ("noleap", "standard", "proleptic_gregorian"),
        86400,
        "interval_end",
        first,
        last,
        False,
    )
    return prepare(
        run_id,
        validate_forcing(forcing, requirement),
        ModelReference(Path(model_root), requirement),
        PreparationSettings(
            Path(config_out),
            Path(forcing_out),
            Path(native_output),
            Path(native_log),
            first,
            last,
            Path(temporal_out),
        ),
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            from blueearth_cst.experiment.simulation_record import (
                SimulationFrozenError,
                read_simulation_intent_v2,
            )

            root = Path(sm.input.simulation).parents[1]
            record = read_simulation_intent_v2(root)
            if (root / "_engine/simulation.json").exists() or Path(
                sm.params.native_output_path
            ).exists():
                raise SimulationFrozenError(
                    "native responses already exist; use a new experiment name"
                )
            prepare_collection_run(
                manifest_path=sm.input.collection,
                run_id=sm.params.run_id,
                model_root=sm.params.model_dir,
                sim_start=sm.params.sim_window_start,
                sim_end=sm.params.sim_window_end,
                catalog_out=sm.output.catalog,
                config_out=sm.output.toml,
                forcing_out=sm.output.nc,
                native_output=sm.params.native_output_path,
                native_log=sm.params.native_log_path,
                temporal_out=sm.output.temporal,
                validated_collection=sm.params.validated_collection,
                preparation=record["documents"]["preparation"],
            )
            from blueearth_cst.shared.run_log_core import log_row

            log_row(
                f"Model forcing and run settings prepared for {sm.params.run_id} over "
                f"{sm.params.sim_window_start}–{sm.params.sim_window_end}",
                module="forcing",
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
