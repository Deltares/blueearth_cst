---
title: The wavelet minimum surfaces as a cryptic error rather than a stated requirement
type: watch-item
area: upstream / weathergenr
origin: R5
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — The wavelet minimum surfaces as a cryptic error rather than a stated requirement.
> **Why** — A run with under 16 years of annual aggregate fails with a message that does not name the cause.
> **Trigger** — Upstream improves the message, or a project hits it and loses time to the diagnosis.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**weathergenr's wavelet minimum surfaces as a cryptic error.**
`wavelet_cwt.R` enforces `length(series) >= 16` on the *annual* aggregate
(i.e. ≥ 16 historical years), but the user-facing error is just
`'series' must have at least 16 observations` — no mention of years,
wavelet, or how to remedy.

*Fix:* improve the error in `tanerumit/weathergenr` (upstream of this repo).
Suggested message: *"historical period (N years) is below weathergenr's
wavelet minimum of 16 years; extend the historical range or reduce the
wavelet decomposition depth."*

*Note:* this fix lives in the weathergenr package, not this repo. Mention
in R5 deliverables if R5 is also touching the R layer; otherwise track as
a separate weathergenr issue.
