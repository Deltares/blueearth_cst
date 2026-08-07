---
title: The R layer has no test infrastructure; Python helpers carry the coverage
type: watch-item
area: testing
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — The R layer has no test infrastructure; Python helpers carry the coverage.
> **Why** — Decided at the start of R5, not overlooked — but it means an R-side regression has no gate at all.
> **Trigger** — The R layer grows past the weather-generator wrappers, or an R-side defect ships.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**R testthat coverage.** Decided at the start of R5 — Python
helpers only by default; adding R testing infrastructure is a
separate call.
