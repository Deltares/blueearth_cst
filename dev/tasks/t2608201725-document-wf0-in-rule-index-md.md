---
title: Document WF0 in rule-index.md
type: todo-item
status: backlog
effort: 2
area: wf0 / dev records
origin: reference-doc cleanup (2026-08-20)
queue:
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — Add a WF0 section to dev/reference/workflows/rule-index.md covering its ten numbered rules (0.00-0.06, 0.04b, 0.10, 0.11) in analyze_climate.smk, in the same shape as the WF1/WF2/WF3 sections: a diagram, a one-line summary table, then a per-rule entry with Does and Writes.
> **Why** — The file documents WF1, WF2 and WF3 and does not mention WF0 at all, so analyze_climate.smk's rules are described only by their own comment headers. AGENTS.md points an agent at rule-index.md when editing or adding a rule, which for WF0 sends them to a page that omits the workflow.
> **Effort** — medium

## Progress

- [ ] Transcribe each rule's `output:` block from `analyze_climate.smk` into a **Writes** line — the section's contract is that Writes is transcribed, not summarised, so it can be checked against the Snakefile.
- [ ] Write the **Does** line per rule, from the script or shell body rather than from the rule name (`naming.md` §8b: grammar conformance is not body conformance).
- [ ] Add the one-line summary table and the dependency diagram, matching the WF1/WF2/WF3 shape.
- [ ] Note the `0.07`–`0.09` gap explicitly, so a reader does not read it as an omission. It is reserved for [[t2608181139]].
- [ ] Confirm `0.04b derive_climate_levels` is placed by number, not by definition order — the letter suffix is the insert convention, not a sub-rule.
- [ ] Retitle the page from "Rule index — WF1, WF2 and WF3" and remove the gap notice at the top.
- [ ] Extend the `<shorthand>` path table if WF0 writes anywhere the three existing workflows do not.

## Notes

Rule ids as the Snakefile declares them today:

`0.00 all` · `0.01 snapshot_config` · `0.02 delineate_region` · `0.03 delineate_spatial_units` · `0.04 extract_historical_climate` · `0.04b derive_climate_levels` · `0.05 plot_climate_source` · `0.06 compare_climate_sources` · `0.10 gather_benchmarks` · `0.11 gather_logs`

**Ten numbered rules, but not ten rule blocks.** Only seven are top-level `rule <name>:` statements. `0.04` and `0.05` are ANONYMOUS `rule:` blocks generated inside `for _source in CANDIDATE_SOURCES:`, and `0.06` is declared in that same loop scope — so the rule count at runtime scales with the number of candidate sources, and two of them have no static identifier at all. The WF1/WF2/WF3 sections assume one named rule per entry; decide how the section represents a loop-generated family before transcribing.

`0.07`–`0.09` are RESERVED, not missing — `analyze_climate.smk:210` says so, and [[t2608181139]] is the item that fills them.

Three of these (`delineate_region`, `delineate_spatial_units`, `extract_historical_climate`) are declared from shared `_rule` helpers and already documented under WF1 — cross-reference rather than duplicate, and say which Snakefile owns the shared definition.
