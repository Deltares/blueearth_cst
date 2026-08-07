---
title: dask cannot be stubbed at module level, because pandas imports it lazily and reads `dask.__spec__`
type: watch-item
area: tests
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — dask cannot be stubbed at module level, because pandas imports it lazily and reads `dask.__spec__`.
> **Why** — It bounds how far the stub-based test approach can go, and the boundary is not obvious from the failing message.
> **Trigger** — pandas changes how it probes dask, or a test needs dask stubbed and hits the wall.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**dask cannot be stubbed at module level.** pandas does a lazy
`import dask` and accesses `dask.__spec__` during type compatibility
checks. A `types.SimpleNamespace` stub for dask there raises
`ValueError: dask.__spec__ is not set` during collection of *any* test
file that imports pandas. dask is in the env via pixi; let it import
normally. If the cost matters, mock the specific dask object at call
time within the test, not at module level.
