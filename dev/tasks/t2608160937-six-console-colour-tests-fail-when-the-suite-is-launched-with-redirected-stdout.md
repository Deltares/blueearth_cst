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
