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
