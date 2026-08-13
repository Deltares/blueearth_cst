---
title: Fold Julia box-drawing blocks into the tee's log grammar
type: watch-item
area: logging / console
origin: console-output review (2026-08-13), items 1 and 4
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — `_compact_log_line` (`blueearth_cst/shared/snake_utils.py`) normalizes hydromt records into `HH:MM:SS - module - LEVEL - msg` and passes everything else through verbatim. Julia's `@info` emits multi-line box-drawing blocks (`┌ │ └`) that pass through as-is. Extending the tee to fold each block into a single row in the same grammar was proposed as item 4 of the console-output review.
> **Why deferred** — Item 1 of the same review (`silent = true`, landed `74a6e3b`) removed the ENTIRE observed population. Measured on the 2026-08-13 rapid run before that change: 1410 box lines in `wf3_climate_experiment_experiment_rapid.log` and 141 in `wf1_model_creation.log` — and sampling shows every one is Wflow's own `@info` ("Set atmosphere_water__precipitation_volume_flux using netCDF variable / precip as forcing parameter"), which is exactly what `silent` suppresses from the terminal. Our own Julia driver, `blueearth_cst/experiment/run_wflow_batch.jl`, uses `println` and emits no boxes. So the work would normalize content that no longer reaches the tee, with no live sample to verify against.
> **Trigger** — A box block appears in a log recorded AFTER `silent = true`. The plausible source is Julia emitting before Wflow configures its logger — Pkg precompilation or package-load `@info`/`@warn` — which cannot be ruled out without watching a real run. Also re-opens if `silent` is ever reverted, since it is the general lever that does not depend on an engine's own switch.

## The design constraint, so it is not re-derived

This is **not** an extension of `_compact_log_line` in its current shape, and
that is the whole reason it is worth a note rather than a one-liner.

- `_compact_log_line` is **pure and line-at-a-time**. Folding a block needs
  state across lines, which makes the tee stateful. It is a contract surface
  with two consumers — `blueearth_cst/shared/merge_logs.py` and
  `tests/test_snake_utils.py` — so the change is wider than the function.
- **The closing `└` carries message content, not a source annotation.** In the
  observed blocks it is the rest of the sentence:

      ┌ Info: Set atmosphere_water__precipitation_volume_flux using netCDF variable
      └ precip as forcing parameter.

  A stateless rule that dropped `└` lines would delete real text. This differs
  from Julia's other common shape, where `└ @ Module path/to/file.jl:12` IS
  discardable — so any implementation has to tell the two apart rather than
  assume one.
- **Julia's boxes carry no timestamp**, so the `HH:MM:SS` half of the grammar
  would have to be synthesized at tee time rather than parsed.

## Refs

- `blueearth_cst/shared/snake_utils.py` — `_compact_log_line`, `_HYDROMT_LOG_RE`.
- `74a6e3b` — item 1, `[logging] silent = true`, which is what emptied this.
- Related: [[t2608071223-parked-2026-07-19-per-rule-progress]] — the other
  console watch-item; its plain-language-banner half is review item 6 and is
  still open.
