# Observational timings only: no GC, compilation or threading policy changes.
using Dates

function record_resource_timing(stats, phase, batch_id, run_id; wall_s=NaN, status="ok")
    directory = get(ENV, "CST_RESOURCE_DIRECTORY", "")
    isempty(directory) && return
    try
        mkpath(directory)
        path = joinpath(directory, "julia-$(getpid()).tsv")
        header = !isfile(path)
        open(path, "a") do io
            if header
                println(io, "utc\tphase\tbatch_id\trun_id\tstatus\twall_s\tgc_s\tallocated_bytes\tcompile_s\trecompile_s\tjulia_threads\tpid\tjulia_version\tcpu_model")
            end
            values = stats === nothing ? (wall_s, "", "", "", "") :
                (stats.time, stats.gctime, stats.bytes, stats.compile_time, stats.recompile_time)
            println(io, join((string(now(UTC)), phase, batch_id, run_id, status, values...,
                Threads.nthreads(), getpid(), VERSION, Sys.cpu_info()[1].model), '\t'))
        end
    catch error
        @warn "WF4 resource timing unavailable" exception=error
    end
end

function with_resource_timing(f, phase, batch_id, run_id="")
    isempty(get(ENV, "CST_RESOURCE_DIRECTORY", "")) && return f()
    start = time_ns()
    stats = try
        @timed f()
    catch error
        record_resource_timing(nothing, phase, batch_id, run_id;
            wall_s=(time_ns() - start) / 1e9, status="failed")
        rethrow()
    end
    record_resource_timing(stats, phase, batch_id, run_id)
    return stats.value
end
