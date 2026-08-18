# P3-3 batching driver (design dev/milestones/p33/performance-passes-design.md §6.1):
# run N Wflow simulations in ONE Julia session, amortizing package-load + JIT.
# Per-TOML try/catch = COMPUTE-level isolation only — a failed member does not
# stop the batch, but Snakemake still fails the job (nonzero exit) and deletes
# the batch's outputs (C5 persistence isolation is DEGRADED by design; blast
# radius = the batch). Per-cst timing/status lines preserve per-run visibility
# now that the Snakemake benchmark row covers the whole batch.
#
# Those lines carry a `[k/N]` POSITION, and are spelled in the toolbox's own row
# format -- `HH:MM:SS - wflow - ...`. Both for the same reason. Rule 3.15 is the
# longest-running step in the toolbox (RLZ_NUM x ST_NUM Wflow runs, batched),
# and Wflow itself is silent on the terminal (`[logging] silent = true`, set so
# Julia's box-drawing records do not swamp the console), so these are the ONLY
# rows a batch emits while it works. Without a position they say a member
# finished but not how much of the batch is left; in a bare format they sit as
# untimestamped text among rows that all carry a clock.
#
# `N` is `length(ARGS)` -- one batch's members, not the experiment's. Rule 3.15
# runs several batches concurrently and each is its own process, so a run-wide
# counter is not knowable here; the rule banner already states the experiment's
# shape, and the batch id is in the log part's name.
using Dates  # stdlib -- resolves via LOAD_PATH's `@stdlib` entry, so it needs no
             # Project.toml dep and adds no Manifest churn (verified under the
             # real `--project=.` invocation rule 3.15 uses)
using Wflow

# Per-member timestep progress, on top of the `[k/N]` position below. The two
# answer different questions -- the counter says how much of the BATCH is left,
# the bar how much of the current MEMBER is -- and a batch member is minutes
# long, so the counter alone leaves the console still for most of a batch.
include(joinpath(@__DIR__, "..", "shared", "wflow_progress.jl"))
using .WflowProgress: run_with_progress

exitcode = 0
total = length(ARGS)

# One definition, so the OK and FAIL rows cannot drift apart in either the
# timestamp or the counter. `module` is `wflow` rather than `batch`: the column
# names the SUBSYSTEM that produced the row, and it is Wflow that ran.
row(body) = println("$(Dates.format(now(), "HH:MM:SS")) - wflow - $(body)")

for (k, t) in enumerate(ARGS)
    global exitcode
    # The TOML name IS the member: rule 3.15 writes
    # `<exp>/hydrology/wflow/config/rlz_<i>_st_<j>.toml`, matching the spelling
    # of the run's own output (`output/rlz_<i>_st_<j>.csv`).
    #
    # This used to prepend the parent directory, from R07 B5 -- which had moved
    # the realization index out of the toml NAME and into its run directory, so
    # that `cst_1.toml` alone was ambiguous within a batch. That layout is gone
    # and the index is back in the name, which left the prefix resolving to the
    # literal constant `wflow` on every row of every batch.
    tag = first(splitext(basename(t)))
    try
        # The member tag is the bar's LABEL, which is what lets the relay tell
        # one member's bar from the next within a batch and close each line.
        dt = @elapsed run_with_progress(Wflow, t; label = tag)
        row("[$(k)/$(total)] $(tag)  $(round(dt; digits=1)) s")
        flush(stdout)
    catch e
        # `FAILED` rather than the old `FAIL`: the console picks a row's colour
        # by reading its text (`_SEVERITY_PATTERNS` in shared/snake_utils.py),
        # and this is the one row in a batch that must not read as routine.
        # `FAILED` is that matcher's own spelling, shared with every other
        # failure row the toolbox emits.
        row("FAILED [$(k)/$(total)] $(tag)  $(sprint(showerror, e))")
        flush(stdout)
        exitcode = 1
    end
end
exit(exitcode)
