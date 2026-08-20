---
title: Real-time scanning makes every Python import ~7x slower
type: todo-item
status: backlog
effort: 1
area: environment / dev machine
origin: test-runtime profiling (2026-08-20)
queue:
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — Exclude the pixi env prefix (and ideally the repo) from ESET real-time scanning on this machine. Needs admin, and may be governed by IT policy, so it is an owner decision rather than a repo change.
> **Why** — One import hydromt pulls 3,173 modules at 4.79 ms each -- roughly 7x the normal per-module cost -- which is most of the gap between this machine running the full suite in 59 minutes and CI running the same command in about 9.
> **Effort** — small

## Progress

- [ ] <first step>

## Cause: measured, 2026-08-20

`python -X importtime -c "import hydromt"`:

    3,173 modules imported
    15.2s cumulative
    4.79 ms per module   (normal is 0.3-1 ms)

Three signatures agree, and each rules something out:

1. **No warm-up.** Three consecutive fresh processes: 15.58s, 13.30s, 14.59s. The OS page cache does not help, so this is not disk IO.
2. **Cost is spread, not concentrated.** No module dominates by self-time; it is ~5 ms across a deep tree — `data_catalog` -> `adapters` -> `gis` -> `flw` -> `pyflwdir`, plus `geopandas`. That is per-file overhead, not one slow module.
3. **A real-time scanner is present.** `Get-CimInstance ... AntiVirusProduct` reports **ESET Security** alongside Defender, and `Get-MpComputerStatus` shows Defender's `RealTimeProtectionEnabled = False` — the two hand over, so ESET is doing the scanning. A scanner hooks each file OPEN and inspects above the page cache, which is exactly the constant, cache-immune per-file cost measured above.

`.pixi/envs/default` holds **94,661 files**, all inside that scan surface, and a fresh interpreter reopens thousands of them.

## What to exclude

The pixi prefix is the high-value one — it is the 94k files reopened on every interpreter start:

    <repo>/.pixi
    <repo>

Reading or setting ESET exclusions needs administrator rights, which is why this is filed rather than done. On a Windows 11 Enterprise machine the setting may be governed by IT policy, so treat it as an owner and IT decision.

## Expected payoff, and how to verify

| | now | if per-module drops to ~1 ms |
|---|---|---|
| WF2 dry-run parse | 19s | ~5s |
| `test_cli` (19 tests) | 147s | ~37s |
| `test-full` | 59 min | ~15 min |

Verify by re-running `python -X importtime -c "import hydromt"` and comparing the per-module figure, then re-timing the four dry-runs recorded in [[t2608202307]].

## Relationship to the other speed items

This composes with [[t2608202307]] rather than replacing it. The scanner makes each import expensive; the parse-time import makes the Snakefiles pay for one they do not need. Fixing either helps; fixing both compounds. Nothing here affects CI, which already runs the same suite in about nine minutes.
