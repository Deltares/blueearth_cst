# Native writer admission and metadata. No values are resampled or rewritten.
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

function annotate_grid(Wflow, model)
    ds = model.writer.dataset
    isnothing(ds) && return
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
