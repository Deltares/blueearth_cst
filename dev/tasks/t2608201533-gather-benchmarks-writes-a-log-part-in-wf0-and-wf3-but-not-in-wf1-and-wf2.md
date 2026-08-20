---
title: gather_benchmarks writes a log part in WF0 and WF3 but not in WF1 and WF2
type: watch-item
area: rule identifiers / logging
origin: 2026-08-20 t2608071213 sweep survey
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — Same rule name, four workflows, two behaviours. analyze_climate.smk and run_stress_test.smk declare a log: for gather_benchmarks, so its label is in LOG_RULES and its section appears in the merged log. build_model.smk and analyze_projections.smk declare no log: for it, so it has a banner and nothing else. Neither has a benchmark:, which IS coherent -- a rule that gathers benchmarks should not benchmark itself.
> **Why** — Recorded now rather than after the fact, because the t2608071213 sweep is about to express every rule's identity as an explicit constructor call. Once 0.10 gather_benchmarks is written as a logged rule in two workflows and a banner-only rule in the other two, the asymmetry stops looking like drift and starts reading as a deliberate contract. It is not being fixed inside the sweep: adding or removing a log: changes what the merged log contains, which is a behaviour change needing its own justification.
> **Trigger** — Someone reads a merged WF1 or WF2 log and expects a gather_benchmarks section that is not there; or the four workflows' rule sets are next reconciled for any other reason. Cheap check either way: grep 'rule gather_benchmarks' -A6 across the four .smk and compare.
