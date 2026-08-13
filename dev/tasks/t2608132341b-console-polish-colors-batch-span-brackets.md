---
title: "Console polish: color failure and stall verdicts, fix the batch-span context, bracket the context in no-color"
type: todo-item
status: backlog
effort: 1
area: logging / console
origin: "console-output assessment #2 (2026-08-13), items 3-4"
queue:
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — Three small edits to snake_utils.py and Snakefile_climate_experiment. (a) The only color is bold cyan on routine rule starts; put red on run_summary's FAILED verdict and yellow on the heartbeat's stall/'failed after' notices, same isatty+NO_COLOR gate (run_summary goes to stderr, so the check carries over; leave log_row plain — it flows through the tee into log files). (b) Rule 3.15's context '5 members | rlz 1-2 | st 0-6' reads as a 2x7 rectangle when the batch holds 5; mark non-cross-product spans (e.g. '(partial)') or list members when short, and tighten the summary 'run Wflow for this batch of members', which half-duplicates the 'N members' context. (c) In no-color output (pipes, CI) the banner's fields rest on ' - ' and double spaces; brackets around the context ('[rlz 1 | st 2]') keep them separable without ANSI.
> **Why** — Color currently marks what a user is not scanning for, the batch span invites a wrong inference about batch size, and piped output loses the banner's field structure.
> **Effort** — small

## Progress

- [ ] Red on `run_summary`'s FAILED verdict, yellow on `_Heartbeat`'s stall /
      `failed after` notices — same isatty + `NO_COLOR` gate as `rule_banner`
      (`run_summary` prints to stderr, so the stderr check carries over).
      `log_row` stays plain: its rows flow through the tee into log files.
- [ ] Rule 3.15 batch context: when the members are not the full cross product
      of the two spans, mark it (`(partial)`) or list members when short —
      `5 members | rlz 1-2 | st 0-6` currently reads as a 2x7 grid. Tighten the
      summary ("run Wflow for one batch"), which half-duplicates `N members`.
- [ ] Bracket the context in `rule_banner`'s no-color form (`[rlz 1 | st 2]`)
      so piped/CI output keeps the field structure without ANSI; update the
      shape pins in `tests/test_snake_utils.py`.

## Refs

- `blueearth_cst/shared/snake_utils.py` — `rule_banner`, `run_summary`,
  `_Heartbeat.stop`/`_run`.
- `Snakefile_climate_experiment:1059` — `_member_span` context for 3.15.
- Deliberately out of scope: unifying the two duration grammars
  (`_fmt_elapsed`'s `1h03m20s` vs `format_elapsed`'s `1:03:20`) — the latter
  matches the benchmark tables by documented intent.
