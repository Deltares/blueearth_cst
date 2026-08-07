---
title: Decide whether the tee_to_log traceback fix lands inside R11 or on its own
type: todo-item
status: backlog
effort: 1
area: workflow ergonomics
origin: R11
queue:
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — [R10-13]'s fix lives in tee_to_log, so it improves every script: rule in all three workflows -- not only WF3's check_model_reference, where it was found.
> **Why** — Carrying it makes R11 not purely a WF3 milestone; splitting it keeps the boundary clean but delays a fix that costs an operator a full reproduce cycle each time it bites. Cheap either way, which is why it needs a ruling rather than a default.
> **Effort** — small

## Progress

- [ ] <first step>
