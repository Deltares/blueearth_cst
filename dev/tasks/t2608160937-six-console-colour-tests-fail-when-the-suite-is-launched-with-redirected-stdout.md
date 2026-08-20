---
title: Six console colour tests fail when the suite is launched with redirected stdout
type: watch-item
area: testing / console
origin: t2608152230 step 7 gate
created: 2026-08-16
updated: 2026-08-16
---

> [!note] Overview
> **What** — tests/test_snake_utils.py's six ANSI-painting cases pass through a pty and fail when pytest is launched with stdout redirected to a file (Start-Process -RedirectStandardOutput). Same commit, both ways: 200/200 vs 6 failed. The tests build their own _TTYStringIO fake, so something in the painting path consults the REAL stdout rather than the stream under test.
> **Why** — It makes the authoritative gate's result depend on how it was launched, which is the property a gate must not have. Nothing reports it: the run looks like six genuine regressions, and the only way to tell is to re-run them differently.
> **Trigger** — CI goes red on these six with no code change in blueearth_cst/shared/snake_utils.py, or someone scripts test-full into a non-tty runner (a cron job, a container, a redirected log).

## The condition is NARROWER than the title says — measured 2026-08-17

A plain shell redirect does **not** reproduce it. `pixi run test-full >
.tmp/testfull.log 2>&1` from the Bash tool gave **2538 passed, 6 skipped, 1
xfailed, 0 failed**, and the four `-k "colour or color"` cases pass standalone
under the same redirect. `sys.stdout.isatty()` is `False` in that shell,
confirmed directly — so **a non-tty stdout is not sufficient**, and the Trigger's
"a redirected log" clause did not fire when it was exercised.

Whatever the painting path consults, it is not `isatty()` alone. The remaining
difference between the two launches is the CONSOLE, not the stream:
`Start-Process -RedirectStandardOutput` starts a detached process with no console
attached, while a shell redirect leaves the console attached and only rebinds
the handle. On win-64 the ANSI-enabling path typically probes the console handle
rather than the stream, which fits — but that is an inference from the two
observations, not a measurement, and it should be checked before anything is
changed on its account.

**What this means for the gate.** `test-full` piped to a file is trustworthy;
that is how the WF2 figure integration's gate was run on 2026-08-17. The
launch that is not trustworthy is the detached one. Do not read this as the
watch-item closing — the failure mode was real when it was recorded, and the
CI-goes-red half of the Trigger is untouched.

## Re-measured 2026-08-19 — it is SEVEN now, not six

`pixi run test-full` launched with `Start-Process -RedirectStandardOutput`
against `f616c67`: **7 failed, 2876 passed**. The same module re-run under a
plain shell redirect: **270 passed**, all 8 painting cases included. So the
condition still reproduces exactly as recorded, and the count moved only because
a painting case was added since 2026-08-16. The seven:

```
test_console_paints_a_start_and_a_finish_in_two_different_tiers
test_console_paints_an_informational_snakemake_line_in_the_body_tier
test_run_summary_paints_only_the_failed_verdict
test_heartbeat_paints_the_alarm_and_not_the_all_clear
test_heartbeat_paints_the_failure_verdict
test_tee_paints_the_console_and_never_the_log_file
test_severity_never_reaches_the_log_file
```

Read the Trigger's "these six" as "these painting cases, whatever the count" —
the number is what someone compares a red run against, and it will drift again
every time the module gains a case.

## Re-confirmed 2026-08-20, and the interaction that makes it structural

Reproduced again on `main` at `5363f53` during an unrelated integration: the
same seven, in the same positions, on BOTH sides of a branch-vs-`main`
comparison launched identically -- so they cancelled and did not affect that
verdict. Four plain-launch runs in the same session were green, including the
whole module (270 passed) and a `> file 2>&1` redirect, which is the 2026-08-17
measurement holding rather than widening.

**The part worth acting on is not the failure, it is the collision.** This
repo's own guidance says to launch a long run detached, because a backgrounded
Bash task is reaped ~78 s in -- and a detached launch is exactly what makes
these seven fail. `test-full` takes ~10 minutes. So **any `test-full` long
enough to need detaching carries these seven by construction**: the recommended
way to run the authoritative gate is guaranteed to produce seven false
failures, and the only way to get a clean one is a launch method that cannot
survive the run's own length.

An agent hit precisely that on 2026-08-20 and read it as cross-module state
pollution under full-suite ordering -- a wrong mechanism reached honestly,
because the standalone re-runs it used to refute it were themselves
plain-launch and therefore green. That is the trap: the refutation changes the
launch method at the same time as the scope, so it looks like evidence for
ordering.

The Trigger's "someone scripts test-full into a non-tty runner" has therefore
fired, in the narrower detached-launch form recorded above. It stays a
watch-item because the fix is still unknown and the gate is still readable by
someone who knows to discount these seven -- but a reader who does not know
will discount a REAL failure alongside them, which is the cost now on the
table.
