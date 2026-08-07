---
title: Enforce the naming convention with a linter instead of review attention
type: todo-item
status: backlog
effort: 2
area: tooling
queue: 15
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — Enforce the naming convention with a linter instead of review attention.
> **Why** — R2 established the convention and nothing checks it, so drift is only caught when someone happens to look.
> **Effort** — Medium, and the open question is the vehicle — a ruff custom rule or a small ad-hoc script.

## Progress

- [ ] <first step>

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**Linter for naming conventions.** R2 establishes the convention
but does not enforce it. A future linter (ruff custom rule, or a
small ad-hoc script) would mechanically catch drift. Add as an
R3+ followup if drift becomes a real problem.
