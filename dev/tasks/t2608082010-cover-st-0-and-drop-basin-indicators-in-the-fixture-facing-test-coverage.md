---
title: Cover st_0 and drop basin_indicators in the fixture-facing test coverage
type: todo-item
status: backlog
effort: 1
area: test coverage
origin: R11
queue:
created: 2026-08-08
updated: 2026-08-08
---

> [!note] Overview
> **What** — Two fixture-facing tests still describe the pre-P3 tree: test_gauge_identity_integration parametrizes st over range(1, 7), so the st_0 baseline member it now produces is never checked, and test_project_tree_inventory lists basin_indicators.csv as a covered sample path for a table that no longer exists.
> **Why** — Neither fails, so neither will be noticed: the first quietly checks 6 of 7 members while reading as full coverage, and the second asserts a rule still matches a path nothing will ever write again.
> **Effort** — small

## Progress

- [ ] <first step>
