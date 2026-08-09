---
title: Write WF3 result CSVs in plain decimal notation, capped to 4 significant digits
type: todo-item
status: backlog
effort: 2
area: wf3 results
queue:
created: 2026-08-09
updated: 2026-08-09
---

> [!note] Overview
> **What** — WF3 result tables are written with pandas default float formatting, so small values land in scientific notation (the baseline indicator_ref carries values like 6.3476255e-05 and 1.1430531e-05). Excel warns or misparses on those, and the full float repr stores ~8 digits of precision the pipeline does not have. Wanted: plain decimal output, and a cap of about 4 significant digits.
> **Why** — Reported by the owner 2026-08-09 opening WF3 CSVs in Excel. These tables are a deliverable and an interchange surface (CST-API reads them), so the on-disk number format is a contract, not cosmetics -- which is why this needs a decision rather than a float_format one-liner.
> **Effort** — large

## Progress

- [ ] <first step>
