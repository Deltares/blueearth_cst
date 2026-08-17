---
title: Rebasing refactor/lean-wf0-console must keep the banner fallback nested
type: watch-item
area: wf0 / console + run banner
created: 2026-08-17
updated: 2026-08-17
---

> [!note] Overview
> **What** — `refactor/lean-wf0-console` branched from `d2700b6`, one commit
> before `32e506c` nested the `_summary` / `_header` fallbacks in
> `analyze_climate.smk`. Both touch the same block, so the rebase conflicts on
> those two hunks. The resolution must keep the nesting — a lean-the-console
> refactor that restores a bare `print(..., file=sys.stderr)` reintroduces the
> defect exactly.
> **Why** — The defect is that the fallback writes to the stream that may be
> what failed, so an `OSError` escapes `onstart` / `onerror` and is reported as
> an error in the Snakefile, masking the rule that actually failed. It cost a
> misdiagnosed wf0 run on 2026-08-17. A conflict resolution that takes "ours"
> wholesale on a console-styling branch is a plausible way to undo it silently.
> **Trigger** — refactor/lean-wf0-console is rebased or merged onto main at 32e506c or later

## Why this is a watch-item and not work

Nothing on `main` needs changing, and `tests/test_cli.py::test_banner_fallback_cannot_raise`
already fails on the pre-fix shape across all four entry points — so the
constraint is *enforced*, not merely hoped for. What the test cannot supply is
the reason, at the moment someone is staring at a conflict hunk and choosing a
side. That is what this note is for.

It self-retires: once that branch lands or is abandoned, close this item. It
carries no separate obligation of its own.

## If the guard itself is what conflicts

The console refactor may legitimately want to change *what* the fallback
prints, or route it somewhere other than `sys.stderr`. That is fine and the
test permits it — it asserts shape only: that each fallback sits directly under
its own `try:` and that the `try` has an `except`. A resolution that moves the
message to the workflow log, or drops it entirely, still passes. Only
un-nesting fails.

## Context

- `32e506c` — the merge; `e496688` — the fix and its rationale.
- The masking behaviour was found while diagnosing a wf0 failure at 21:46 on
  2026-08-17. The underlying rule failure is not recoverable: the two
  `_parts/0.04_extract_historical_climate/*.log` files were overwritten by the
  successful 21:48 run, so only the handler defect could be established.
