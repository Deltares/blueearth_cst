---
title: Verify heartbeat cadence on the silent Wflow rules; add backoff if it floods
type: todo-item
status: backlog
effort: 1
area: logging / console
origin: "console-output assessment #2 (2026-08-13), item 1"
queue:
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — With [logging] silent = true (74a6e3b), rules 1.14 and 3.15 stream ~nothing to the tee, so _Heartbeat prints 'still running' every 60s of silence for the whole run — ~60 lines/hour/job, x3 concurrent WF3 batch jobs. Verify on the next real run whether Wflow's progress bar survives silent (if it does, touch() resets the clock and this shrinks to nothing); if the flood is real, add exponential backoff (60s -> 2m -> 5m -> cap ~15m) to _Heartbeat._run.
> **Why** — The console-noise flood that silent=true removed partially returns through the watchdog on exactly the multi-hour rules it matters for; quiet_rows durability is unaffected either way (it records periods, not notices).
> **Effort** — small

## Progress

- [ ] On the next real run touching 1.14 or 3.15, check whether Wflow's progress
      bar survives `silent = true` (74a6e3b silenced the logger; the bar may be
      separate). If it survives, `touch()` resets the clock — close this as a
      no-op with the observation recorded.
- [ ] If the flood is real: exponential backoff in `_Heartbeat._run`
      (60s -> 2m -> 5m -> cap ~15m); `quiet_rows` period semantics unchanged.
- [ ] Update the `_Heartbeat` docstring (`CST_HEARTBEAT_SECS` stays the base
      interval) and the heartbeat coverage in `tests/test_snake_utils.py`.

## Refs

- `blueearth_cst/shared/snake_utils.py` — `_Heartbeat._run` (flat-interval loop).
- `74a6e3b` — the silencing whose commit message itself flags "NOT verified
  end-to-end -- no Wflow run was made".
- `dev/tasks/t2608132310-…` (watch-item) — the sibling consequence of the same
  silencing; a run that answers this item's question likely answers its Trigger too.
