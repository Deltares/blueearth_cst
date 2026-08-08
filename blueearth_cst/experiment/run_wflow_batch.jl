# P3-3 batching driver (design dev/milestones/p33/performance-passes-design.md §6.1):
# run N Wflow simulations in ONE Julia session, amortizing package-load + JIT.
# Per-TOML try/catch = COMPUTE-level isolation only — a failed member does not
# stop the batch, but Snakemake still fails the job (nonzero exit) and deletes
# the batch's outputs (C5 persistence isolation is DEGRADED by design; blast
# radius = the batch). Per-cst timing/status lines preserve per-run visibility
# now that the Snakemake benchmark row covers the whole batch.
using Wflow

exitcode = 0
for t in ARGS
    global exitcode
    # R07 B5 moved the realization index out of the toml NAME and into its run
    # directory, so basename alone (cst_1.toml, as the member token then read)
    # no longer identifies a member within a batch. Tag with the parent too.
    tag = joinpath(basename(dirname(dirname(t))), basename(t))
    try
        dt = @elapsed Wflow.run(t)
        println("BATCH-RUN OK   $(tag)  $(round(dt; digits=1)) s")
        flush(stdout)
    catch e
        println("BATCH-RUN FAIL $(tag)  $(sprint(showerror, e))")
        flush(stdout)
        exitcode = 1
    end
end
exit(exitcode)
