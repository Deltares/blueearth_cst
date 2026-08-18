# Rule 1.14's driver: run the Wflow.jl model once, on historical forcing.
#
# Rule 1.14 invoked Wflow through Julia's `-e` flag until now:
#
#     julia ... -e "using Wflow; Wflow.run()" "<toml>"
#
# which is the invocation Wflow's own docstring suggests, and which emits
# NOTHING to the console for the whole run under `[logging] silent = true` --
# the longest single step in WF1, reduced to the tee's 60-second "still running"
# heartbeat. This driver exists to give it the same progress bar every other
# long step in the toolbox animates; see `shared/wflow_progress.jl` for why the
# fraction has to be reported from here and rendered on the Python side.
#
# WHY A FILE RATHER THAN A LONGER `-e`. The logger wiring is ~20 lines and is
# shared with rule 3.15, so `-e` would mean either duplicating it in two
# Snakefile string literals or shipping it as an unreadable one-liner. A file
# is also what 3.15 already does (`experiment/run_wflow_batch.jl`), so the two
# Wflow-running rules now have the same shape.
#
# NO BATCH COUNTER HERE. 3.15's rows carry `[k/N]` because a batch runs several
# members; 1.14 runs exactly one, and `[1/1]` would be noise. The bar is the
# progress signal, and the rule banner already names the step.

using Dates

# Included and called ABOVE `using Wflow`, deliberately. `WflowProgress` needs
# only stdlib `Logging`, and the load-plus-JIT of Wflow itself is one of the two
# windows that reported nothing (see `open_frame`) -- opening the bar first is
# what puts a line on the console for it.
include(joinpath(@__DIR__, "..", "shared", "wflow_progress.jl"))
using .WflowProgress: open_frame, run_with_progress

open_frame("wflow")

using Wflow

row(body) = println("$(Dates.format(now(), "HH:MM:SS")) - wflow - $(body)")

if length(ARGS) != 1
    row("FAILED run_wflow expects exactly one TOML path, got $(length(ARGS))")
    exit(2)
end

tomlpath = ARGS[1]

try
    dt = @elapsed run_with_progress(Wflow, tomlpath; label = "wflow")
    row("simulation complete  $(round(dt; digits = 1)) s")
    flush(stdout)
catch e
    # `FAILED` is the spelling the console's severity matcher knows
    # (`_SEVERITY_PATTERNS` in shared/snake_utils.py), so this row paints red.
    row("FAILED $(first(splitext(basename(tomlpath))))  $(sprint(showerror, e))")
    flush(stdout)
    exit(1)
end
