using Wflow

using NCDatasets

########## CUSTOM FUNCTIONS ##########
function apply_forcing_change_factors!(model::Model)
    @unpack vertical, clock, reader, network, config = model
    @unpack dataset, dataset_times, forcing_parameters = reader

    # Get the correct month
    # pick up the data that is valid for the past model time step
    month = month(clock.time)

    # get the change factors
    change_factors_filename = get(config, "input.path_forcing_scale", nothing)::String
    precipitation_scale = get(config, "input.vertical.precipitation", "precip")::String
    temperature_offset = get(config, "input.vertical.temperature", "temp")::String
    potential_evaporation_scale = get(config, "input.vertical.potential_evaporation", "pet")::String
    
    if !isnothing(change_factors_filename)
        # Read the netdcf file with the change factors
        change_factors = NCDataset(change_factors_filename)

        # precipitation
        if haskey(change_factors, precipitation_scale)
            par = "vertical.precipitation"
            # Read the current precipitation
            par_vector = Wflow.param(model, par)
            # Get the active grid indices
            sel = Wflow.active_indices(network, par)
            # Get and apply the scale factor
            precip_scale = Wflow.get_at(change_factors, precipitation_scale, month)
            par_vector .= par_vector .* precip_scale[sel]
        end
        
        # temperature
        if haskey(change_factors, temperature_offset)
            par = "vertical.temperature"
            # Read the current temperature
            par_vector = Wflow.param(model, par)
            # Get the active grid indices
            sel = Wflow.active_indices(network, par)
            # Get and apply the scale factor
            temp_offset = Wflow.get_at(change_factors, temperature_offset, month)
            par_vector .= par_vector .+ temp_offset[sel]
        end
        
        # potential evaporation
        if haskey(change_factors, potential_evaporation_scale)
            par = "vertical.potential_evaporation"
            # Read the current potential evaporation
            par_vector = Wflow.param(model, par)
            # Get the active grid indices
            sel = Wflow.active_indices(network, par)
            # Get and apply the scale factor
            pet_scale = Wflow.get_at(change_factors, potential_evaporation_scale, month)
            par_vector .= par_vector .* pet_scale[sel]
        end       
    end
end

function run_timestep_delta_change(model::Model; update_func = update, write_model_output = true)
    advance!(model.clock)
    load_dynamic_input!(model)

    # update with the change factors
    apply_forcing_change_factors!(model)

    model = update_func(model)
    if write_model_output
        write_output(model)
    end
    return model
end

function run_delta_change(model::Model; close_files = true)
    @unpack network, config, writer, clock = model

    model_type = config.model.type::String

    # determine timesteps to run
    calendar = get(config, "calendar", "standard")::String
    starttime = clock.time
    dt = clock.dt
    endtime = cftime(config.endtime, calendar)
    times = range(starttime + dt, endtime, step = dt)

    @info "Run information" model_type starttime dt endtime nthreads()
    runstart_time = now()
    @progress for (i, time) in enumerate(times)
        @debug "Starting timestep." time i now()
        model = run_timestep_delta_change(model)
    end
    @info "Simulation duration: $(canonicalize(now() - runstart_time))"

    # write output state netCDF
    if haskey(config, "state") && haskey(config.state, "path_output")
        @info "Write output states to netCDF file `$(model.writer.state_nc_path)`."
    end
    write_netcdf_timestep(model, writer.state_dataset, writer.state_parameters)

    reset_clock!(model.clock, config)

    # option to support running function twice without re-initializing
    # and thus opening the netCDF files
    if close_files
        Wflow.close_files(model, delete_output = false)
    end

    # copy TOML to dir_output, to archive what settings were used
    if haskey(config, "dir_output")
        src = normpath(pathof(config))
        dst = output_path(config, basename(src))
        if src != dst
            @debug "Copying TOML file." src dst
            cp(src, dst, force = true)
        end
    end
    return model
end

########## MAIN FUNCTION ##########

# Read the TOML file and initialize SBM model
config = Wflow.Config(toml_path)
model = Wflow.initialize_sbm_model(config)
Wflow.load_fixed_forcing(model)

# Run the model including forcing updated with monthly factors
run_delta_change(model)