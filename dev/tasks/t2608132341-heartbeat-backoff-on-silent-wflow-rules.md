---
title: Verify heartbeat cadence on the silent Wflow rules; add backoff if it floods
type: todo-item
status: done
effort: 1
area: logging / console
origin: "console-output assessment #2 (2026-08-13), item 1"
queue:
created: 2026-08-13
updated: 2026-08-18
closed: 2026-08-18
---

> [!note] Overview
> **What** — With [logging] silent = true (74a6e3b), rules 1.14 and 3.15 stream ~nothing to the tee, so _Heartbeat prints 'still running' every 60s of silence for the whole run — ~60 lines/hour/job, x3 concurrent WF3 batch jobs. Verify on the next real run whether Wflow's progress bar survives silent (if it does, touch() resets the clock and this shrinks to nothing); if the flood is real, add exponential backoff (60s -> 2m -> 5m -> cap ~15m) to _Heartbeat._run.
> **Why** — The console-noise flood that silent=true removed partially returns through the watchdog on exactly the multi-hour rules it matters for; quiet_rows durability is unaffected either way (it records periods, not notices).
> **Effort** — small

## Progress

- [x] On the next real run touching 1.14 or 3.15, check whether Wflow's progress
      bar survives `silent = true`. **It does not** — and the reason is
      structural, not incidental: `Wflow/src/logging.jl:42-51` swaps the
      `TerminalLogger` (the only thing that renders `@progress`) for a
      `NullLogger` under `silent`, and the file leg separately drops
      `log.group !== :ProgressLogging`. Both legs destroy it, so no amount of
      configuration recovers the bar.
- [x] If the flood is real: exponential backoff in `_Heartbeat._run`. **Not
      needed, and not implemented** — the premise was removed at the source
      instead (2026-08-18). Rules 1.14 and 3.15 now emit their own progress
      frames through `shared/wflow_progress.jl`, and every frame calls
      `heartbeat.touch()`, so the clock resets roughly once a second for the
      whole timestep loop.
- [ ] ~~Update the `_Heartbeat` docstring and its test coverage~~ — moot;
      `_Heartbeat` is unchanged.

## Resolution

**Closed 2026-08-18 by making the rules emit progress, not by changing the
watchdog.** `CST_HEARTBEAT_SECS` and `_Heartbeat._run` are untouched.

Measured on `test_case/test_rapid` (rule 1.14's exact command, via
`run_logged.py`): one `still running` notice per run, during **model
construction** — `Wflow.Model(config)` took 44.8 s against a 53.3 s timestep
loop on a cold run. The loop is now fully instrumented; the build phase is not,
because upstream does not instrument it. So the worst case is ~1 notice per
member rather than the ~60 lines/hour/job this item was opened against, and the
`x3 concurrent WF3 batch jobs` multiplier applies to that much smaller number.

**Residual worth knowing** (not tracked as work — it needs an upstream change or
a synthetic timer, and both cost more than the notice they would remove): a
basin whose model build is slow will still show one heartbeat notice per member
before its bar opens. That is the watchdog doing its job on a genuinely silent
phase, which is the behaviour this item wanted preserved.

## Refs

- `blueearth_cst/shared/snake_utils.py` — `_Heartbeat._run` (flat-interval loop).
- `74a6e3b` — the silencing whose commit message itself flags "NOT verified
  end-to-end -- no Wflow run was made".
- `dev/tasks/t2608132310-…` (watch-item) — the sibling consequence of the same
  silencing; a run that answers this item's question likely answers its Trigger too.
