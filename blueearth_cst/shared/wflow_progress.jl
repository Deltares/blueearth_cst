# Progress reporting for the two rules that run Wflow.jl (1.14 and 3.15).
#
# WHY THIS EXISTS. Wflow instruments its timestep loop with `@progress`
# (`Wflow.jl`, `run!`), but the only thing that RENDERS those records is the
# `TerminalLogger` its `init_logger` installs -- and we set `[logging] silent =
# true` in the built TOML (74a6e3b) to stop Julia's box-drawing records from
# swamping the console. Under `silent`, `init_logger` swaps that terminal leg
# for a `NullLogger`, and its file leg drops the `:ProgressLogging` group
# outright, so the progress records were destroyed on BOTH legs. The result was
# a multi-minute (on a real basin, multi-hour) rule that emitted nothing at all
# while it worked.
#
# WHY NOT JUST UNSET `silent`. The bar `TerminalLogger` draws is ANSI-driven and
# assumes a TTY. Under Snakemake this stdout is a pipe into the Python tee, so
# that bar arrives as escape noise -- and it is a DIFFERENT bar from the one
# every other long step in the toolbox animates. Unsetting `silent` would also
# bring back the box-drawing flood it was set to remove.
#
# WHAT THIS DOES INSTEAD. Emit a bare machine-readable FRAME carrying the
# fraction, and let the Python side re-render it as the house bar
# (`shared/progress.py`). That is the same split rule 1.08 already uses for
# hydromt's CLI (`DaskFrameRelay`): the child reports a number, the parent owns
# the rendering. It has to be that way round -- the three constraints the bar
# must satisfy (no ANSI, not a TTY at run time, cp1252-safe consoles) are all
# properties of the PARENT's stream, which this process cannot observe.
#
# THE SEAM. `Wflow.run(tomlpath)` installs its own logger via `with_logger`,
# which REPLACES the ambient one rather than composing with it -- so wrapping
# the call from outside cannot see the progress records. `Wflow.run(config)`
# does not install anything (its docstring: "Logging to a file is only part of
# the `run(tomlpath::AbstractString)` method"), so building the config and the
# logger separately is what lets us tee our own leg alongside Wflow's file log.
# `Wflow.Config` and `Wflow.init_logger` are internal API; `Project.toml` pins
# `Wflow = "1"`, and `run_with_progress` degrades to the plain path if either
# is ever withdrawn, so a break costs the bar rather than the run.

module WflowProgress

using Logging

# The frame the Python relay parses. `[cst-progress]` is a sentinel rather than
# a percentage-bearing sentence so an ordinary log row that happens to contain a
# number cannot match it, and so a frame is recognisable without knowing which
# engine produced it.
const FRAME_PREFIX = "[cst-progress]"

# Seconds between frames. Each one crosses a pipe and costs a line, so this is
# not the sub-second cadence an in-process bar can afford: Wflow's loop is per
# TIMESTEP (3287 of them on a rapid member), and an unthrottled emitter would
# put one line per timestep on the wire. One second keeps the bar smooth while
# also keeping the tee's 60 s silence heartbeat from ever firing on these rules.
const FRAME_INTERVAL = 1.0

"""
    FrameLogger(io, label; interval)

Consumes ProgressLogging records and writes throttled `[cst-progress]` frames.

Everything that is not a progress record is IGNORED here rather than printed --
this logger is only ever one leg of a tee whose other leg is Wflow's own file
logger, so passing records through would duplicate them.
"""
struct FrameLogger <: AbstractLogger
    io::IO
    label::String
    interval::Float64
    last::Ref{Float64}
    started::Ref{Bool}
end

FrameLogger(io::IO, label::AbstractString; interval::Real = FRAME_INTERVAL) =
    FrameLogger(io, String(label), Float64(interval), Ref(0.0), Ref(false))

# `-1` is ProgressLogging's level (`ProgressLogging.ProgressLevel`), which is
# below `Debug`. Spelled as the literal so this file needs no dependency on
# ProgressLogging -- which is Wflow's transitive dep, not ours, and so is not
# resolvable under our own `--project=.`.
Logging.min_enabled_level(::FrameLogger) = LogLevel(-1)
Logging.shouldlog(::FrameLogger, level, _module, group, id) = true
Logging.catch_exceptions(::FrameLogger) = false

"""Fraction in `[0, 1]` from a `progress` kwarg, or `nothing` if it carries none.

`@progress` reports `0.0..1.0` while running and the sentinel `"done"` at the
end; `nothing`/`NaN` mean indeterminate, which a bar cannot honestly draw.
"""
function _fraction(value)
    value === "done" && return 1.0
    value isa Real || return nothing
    isnan(float(value)) && return nothing
    return clamp(Float64(value), 0.0, 1.0)
end

function Logging.handle_message(
    logger::FrameLogger,
    level,
    message,
    _module,
    group,
    id,
    filepath,
    line;
    kwargs...,
)
    haskey(kwargs, :progress) || return nothing
    fraction = _fraction(kwargs[:progress])
    fraction === nothing && return nothing

    now_s = time()
    final = fraction >= 1.0
    # The first and last frames always go out: the first opens the bar (without
    # it a fast run would draw nothing at all), and the last is what closes the
    # line and leaves a completed frame standing as the log's summary row.
    if !final && logger.started[] && (now_s - logger.last[]) < logger.interval
        return nothing
    end
    logger.started[] = true
    logger.last[] = now_s

    println(logger.io, "$(FRAME_PREFIX) $(logger.label) $(round(fraction; digits = 5))")
    flush(logger.io)
    return nothing
end

"""
    TeeLogger(a, b)

Forward every record to both loggers.

A local definition rather than `LoggingExtras.TeeLogger`: LoggingExtras is
Wflow's transitive dependency, not one of ours, so `using` it would not resolve
under `--project=.` without adding it to `Project.toml` and churning the
Manifest -- a lot of ceremony for fifteen lines.
"""
struct TeeLogger{A <: AbstractLogger, B <: AbstractLogger} <: AbstractLogger
    a::A
    b::B
end

Logging.min_enabled_level(t::TeeLogger) =
    min(Logging.min_enabled_level(t.a), Logging.min_enabled_level(t.b))

# Either leg wanting the record is enough; each `handle_message` re-checks for
# itself, so a record only one leg asked for is not acted on by the other.
Logging.shouldlog(t::TeeLogger, level, _module, group, id) =
    Logging.shouldlog(t.a, level, _module, group, id) ||
    Logging.shouldlog(t.b, level, _module, group, id)

Logging.catch_exceptions(t::TeeLogger) =
    Logging.catch_exceptions(t.a) || Logging.catch_exceptions(t.b)

function Logging.handle_message(t::TeeLogger, level, message, _module, group, id, args...; kwargs...)
    for leg in (t.a, t.b)
        if Logging.min_enabled_level(leg) <= level &&
           Logging.shouldlog(leg, level, _module, group, id)
            Logging.handle_message(leg, level, message, _module, group, id, args...; kwargs...)
        end
    end
    return nothing
end

"""
    run_with_progress(Wflow, tomlpath; label, io = stdout)

Run one Wflow simulation, emitting progress frames as it goes.

`Wflow` is passed in rather than imported so this module stays loadable (and
testable) without it -- the callers already have it in scope.

Wflow's own file log is preserved exactly as `Wflow.run(tomlpath)` would write
it: same `init_logger`, same `silent`, same `close` in a `finally`. The only
addition is the second tee leg.
"""
function run_with_progress(Wflow, tomlpath::AbstractString; label::AbstractString, io::IO = stdout)
    config = Wflow.Config(tomlpath)
    # Honour the TOML's own `silent`, exactly as `Wflow.run(tomlpath)` does, so
    # this driver never overrides an operator who set `silent = false` to debug.
    silent = config.logging.silent
    logger, logfile = Wflow.init_logger(config; silent)

    # The body below MIRRORS `Wflow.run(tomlpath)` clause for clause -- the
    # version banner, the try/catch/finally, the FEWS-conditional backtrace, the
    # `rethrow`, the `close` -- because rule 1.14 DECLARES `run_default/log.txt`
    # as an output and that file must stay byte-comparable to the one the
    # upstream entry point writes. The single deviation is the tee: our frame
    # leg rides alongside Wflow's file logger instead of replacing it.
    #
    # The try/catch sits INSIDE `with_logger`, not around it. That placement is
    # load-bearing rather than stylistic: it is what puts a failed run's
    # stacktrace into the log file, which is the whole reason the log survives a
    # crash. (Upstream's own comment: "to catch stacktraces in the log file a
    # try-catch is required".)
    with_logger(TeeLogger(logger, FrameLogger(io, label))) do
        @info "Wflow version `v$(Wflow.VERSION)`"
        try
            Wflow.run(config)
        catch e
            if config.fews_run__flag
                @error "Wflow simulation failed" exception = e _id = :wflow_run
            else
                @error "Wflow simulation failed" exception = (e, catch_backtrace()) _id =
                    :wflow_run
            end
            rethrow()
        finally
            close(logfile)
        end
    end
    return nothing
end

end # module
