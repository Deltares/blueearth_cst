# Native writer admission and metadata. No values are resampled or rewritten.
using SHA
using TOML

"Capture expectations from the initialized native model, never from its output."
function write_grid_contract(Wflow, model, tomlpath)
    config = model.config
    names = Wflow.ncnames(config.output.netcdf_grid.variables)
    static_path = Wflow.input_path(config, config.input.path_static)
    projected = config.model.cell_length_in_meter__flag
    spatial = projected ? ["y", "x"] : ["lat", "lon"]
    contract = Dict{String, Any}(
        "schema_version" => "wflow-grid-contract/1",
        "toml_sha256" => bytes2hex(sha256(read(tomlpath))),
        "static_sha256" => bytes2hex(sha256(read(static_path))),
        "wflow_version" => string(Wflow.VERSION),
        "standard_names_sha256" => bytes2hex(sha256(read(joinpath(dirname(pathof(Wflow)), "standard_name.jl")))),
        "native_support_sha256" => bytes2hex(sha256(read(joinpath(dirname(pathof(Wflow)), "utils.jl")))),
        "variables" => Dict{String, Any}(),
        "supports" => Dict{String, Any}(),
    )
    Wflow.NCDataset(static_path) do source
        crs_names = [name for name in keys(source) if haskey(source[name].attrib, "crs_wkt") || haskey(source[name].attrib, "spatial_ref")]
        length(crs_names) == 1 || error("Grid contract requires one staticmaps CRS")
        crs = source[only(crs_names)].attrib
        contract["crs_wkt"] = get(crs, "crs_wkt", get(crs, "spatial_ref", ""))
        contract["coordinates"] = Dict(spatial[1] => Wflow.read_y_axis(source), spatial[2] => Wflow.read_x_axis(source))
    end
    if !isnothing(model.writer.extra_dim)
        contract["coordinates"][model.writer.extra_dim.name] = model.writer.extra_dim.value
    end
    for (standard, external) in sort!(collect(names); by=first)
        vector = Wflow.get_lens(standard, model.land)(model)
        indices = Wflow.active_indices(model.domain, standard)
        support = [Int[I[2], I[1]] for I in indices] # Python y,x order, one-based
        keys_sorted = sort!(collect(keys(contract["supports"])))
        support_index = findfirst(key -> contract["supports"][key] == support, keys_sorted)
        # Reuse domains across variables without imposing land support on rivers.
        key = isnothing(support_index) ? "support_$(length(keys_sorted) + 1)" : keys_sorted[support_index]
        contract["supports"][key] = support
        layered = !(eltype(vector) <: AbstractFloat)
        dimensions = layered ? ["time", "layer", spatial...] : ["time", spatial...]
        layers = layered ? Float64.(1:length(first(vector))) : Float64[]
        contract["variables"][external] = Dict(
            "standard_name" => standard,
            "units" => Wflow.sbm_standard_name_map[standard].unit,
            "dimensions" => dimensions,
            "layers" => layers,
            "support" => key,
        )
    end
    target = tomlpath * ".grid.toml"
    open(target * ".tmp", "w") do io
        TOML.print(io, contract; sorted=true)
    end
    mv(target * ".tmp", target; force=true)
end
function check_grid_request(Wflow, config)
    Wflow.do_netcdf_grid(config) || return
    config.model.type == Wflow.ModelType.sbm || error("Managed grid output requires Wflow-SBM")
    names = Wflow.ncnames(config.output.netcdf_grid.variables)
    isempty(names) && error("Grid variables must be nonempty")
    for name in keys(names)
        haskey(Wflow.sbm_standard_name_map, name) || error("Unknown Wflow standard name: $name")
        if startswith(name, "snowpack") && !config.model.snow__flag
            error("Grid variable $name requires snow__flag=true")
        end
        if startswith(name, "glacier") && !config.model.glacier__flag
            error("Grid variable $name requires glacier__flag=true")
        end
    end
end

function annotate_grid(Wflow, model, tomlpath)
    ds = model.writer.dataset
    isnothing(ds) && return
    write_grid_contract(Wflow, model, tomlpath)
    names = Wflow.ncnames(model.config.output.netcdf_grid.variables)
    static_path = Wflow.input_path(model.config, model.config.input.path_static)
    Wflow.NCDataset(static_path) do source
        crs_names = [name for name in keys(source) if haskey(source[name].attrib, "crs_wkt") || haskey(source[name].attrib, "spatial_ref")]
        length(crs_names) == 1 || error("Grid output requires one authoritative CRS variable in staticmaps")
        crs = source[only(crs_names)]
        Wflow.defVar(ds, "spatial_ref", Int32, (); attrib=Dict(crs.attrib))
    end
    for (standard, external) in names
        ds[external].attrib["units"] = Wflow.sbm_standard_name_map[standard].unit
        ds[external].attrib["wflow_standard_name"] = standard
        ds[external].attrib["time_label"] = "interval_end"
        ds[external].attrib["grid_mapping"] = "spatial_ref"
    end
end

# Read-only expectation recovery for outputs produced before this adapter revision.
if abspath(PROGRAM_FILE) == @__FILE__
    using Wflow
    config = Wflow.Config(ARGS[1])
    check_grid_request(Wflow, config)
    config.output.netcdf_grid._was_specified = false
    config.output.csv._was_specified = false
    config.output.netcdf_scalar._was_specified = false
    config.state.path_output = nothing
    model = Wflow.Model(config)
    try
        write_grid_contract(Wflow, model, ARGS[1])
    finally
        Wflow.close_files(model; delete_output=false)
    end
end
