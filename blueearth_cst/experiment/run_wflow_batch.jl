# One Julia session amortizes Wflow load/JIT across an explicitly identified batch.
# With CST_BATCH_STAGING set, each finished member's CSV is moved into that
# directory at once and its `.expect` marker becomes `.ok`, so Snakemake's
# failed-job cleanup cannot delete it and a retry re-runs only the failures
# (blueearth_cst/experiment/batch_staging.py). Unset, outputs stay in place.
using Dates

function parse_batch(args)
    length(args) >= 4 && (length(args) - 1) % 3 == 0 ||
        error("Expected batch_id followed by (run_id, toml_path, native_output_path) records")
    batch_id = args[1]
    isempty(batch_id) && error("batch_id must be nonempty")
    members = [(run_id=args[i], toml_path=args[i+1], native_output_path=args[i+2])
               for i in 2:3:length(args)]
    ids = [member.run_id for member in members]
    all(id -> occursin(r"^run_[0-9]+$", id), ids) ||
        error("Batch run ids must be explicit prefixed identifiers")
    length(unique(ids)) == length(ids) || error("Duplicate run id in batch")
    issorted(ids) || error("Batch run ids must be in increasing collection order")
    outputs = [member.native_output_path for member in members]
    length(unique(outputs)) == length(outputs) || error("Duplicate native output in batch")
    return batch_id, members
end

# Kept outside the executable block so parser tests need neither Wflow nor a model.
row(body) = println("$(Dates.format(now(), "HH:MM:SS")) - wflow - $(body)")

if abspath(PROGRAM_FILE) == @__FILE__
    batch_id, members = parse_batch(ARGS)
    include(joinpath(@__DIR__, "..", "shared", "resource_timing.jl"))
    include(joinpath(@__DIR__, "..", "shared", "wflow_progress.jl"))
    using .WflowProgress: open_frame, run_with_progress
    open_frame(first(members).run_id; position = "[1/$(length(members))]")
    with_resource_timing("wflow_load", batch_id) do
        @eval using Wflow
    end

    exitcode = 0
    total = length(members)
    staging = get(ENV, "CST_BATCH_STAGING", "")
    for (k, member) in enumerate(members)
        global exitcode
        tag = member.run_id
        try
            # The member's finished bar row, stamped and carrying `[k/N]`, is
            # its record; only a failure prints a row of its own.
            with_resource_timing("member", batch_id, tag) do
                run_with_progress(Wflow, member.toml_path; label=tag, position="[$(k)/$(total)]")
            end
            isfile(member.native_output_path) ||
                error("Missing native output $(member.native_output_path)")
            if !isempty(staging)
                # CSV first, marker second: an `.ok` always has its CSV.
                mv(member.native_output_path, joinpath(staging, "$(tag).csv"); force=true)
                mv(joinpath(staging, "$(tag).expect"), joinpath(staging, "$(tag).ok"); force=true)
            end
        catch e
            row("FAILED [$(k)/$(total)] Run $(tag) batch=$(batch_id)  $(sprint(showerror, e))")
            flush(stdout)
            exitcode = 1
        end
    end
    exit(exitcode)
end
