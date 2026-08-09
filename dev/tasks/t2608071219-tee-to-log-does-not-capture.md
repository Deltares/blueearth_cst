---
title: Capture a failing `script:` rule's traceback in its log part
type: todo-item
status: backlog
effort: 1
area: ergonomics
queue: 15
created: 2026-08-07
updated: 2026-08-09
---

> [!note] Overview
> **What** — Capture a failing `script:` rule's traceback in its log part.
> **Why** — The traceback goes to the console and not the log, so the persisted artifact of a failure is the one place the cause is missing.
> **Effort** — Small, but it touches a shared helper every script rule runs through.

## Progress

- [ ] <first step>

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.
- No `origin` recorded. It was migrated from the un-milestoned "Cross-cutting — workflow ergonomics"
  section; the prose dates the diagnosis to 2026-08-01, during R8, but no milestone owned the item.

## Detail

**`tee_to_log` does not capture the traceback of a failing `script:` rule.**
*Surfaced 2026-08-01 while landing the canonical climate figure set.* A rule
that raises writes every `log_row`/INFO line to
`logs/_parts/<W.NN>_<rule>.log` and then stops **without the exception**. The
merged workflow log therefore ends mid-rule with no reason, and
`check log file(s) for error details` — which is the only thing Snakemake
prints — points at a file that does not contain them. The traceback does reach
Snakemake's own captured stderr, so it is visible on an interactive console
run and invisible in the artifact a user would send you. Cost a full
reproduce-outside-pytest cycle to recover a one-line `KeyError`.
*Fix:* have `tee_to_log` catch, write the formatted traceback into the log
part, and re-raise. Cheap and self-contained (`snake_utils.tee_to_log`), and
it improves every `script:` rule in all three workflows at once. Owner:
`python-engineer`. Activation: next time WF logging is touched.
