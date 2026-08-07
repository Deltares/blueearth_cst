---
title: Settle whether unit B's cst_ to st_ rename reaches the frozen experiment.yml
type: todo-item
status: backlog
effort: 1
area: wf3 identification
origin: R11
queue:
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — R9 freezes experiment.yml at an experiment's first successful run. Unit B renames cst_ to st_ across filenames and catalog keys; whether that reaches frozen content is unestablished.
> **Why** — If it does, existing experiments need a migration path or an explicit unsupported-and-re-run ruling. R7 (GA-2) and R9 both ruled the latter on the grounds that no production trees exist -- but that reasoning must be RE-CHECKED rather than inherited, because R9 is what shipped the freezing mechanism that makes this case different.
> **Effort** — small

## Progress

- [ ] <first step>
