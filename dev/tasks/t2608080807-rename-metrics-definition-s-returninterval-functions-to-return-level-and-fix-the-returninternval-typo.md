---
title: Rename metrics_definition's returninterval functions to return_level, and fix the returninternval typo
type: todo-item
status: backlog
effort: 1
area: wf3 metrics
origin: R11
queue:
created: 2026-08-08
updated: 2026-08-08
---

> [!note] Overview
> **What** — metrics_definition.py names its two extreme-value functions returninterval / returninterval_Q7d, and export_wflow_results ships a returninternval typo. Both compute a return LEVEL (a discharge magnitude) at a given return PERIOD, so the name states the input rather than the output -- and 'return interval' is not the standard term either way; the established synonyms are return period and recurrence interval.
> **Why** — R11 renamed the statistic keys in the new indicator vocabulary but deliberately left this module alone: it is a shared module with other callers and was outside P1's allowed scope. The published metric names are already correct (q_return_level_10yr_max), so this is internal tidiness, not a contract fix -- which is why it did not block P1.
> **Effort** — small

## Progress

- [ ] <first step>
