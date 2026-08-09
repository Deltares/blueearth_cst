---
title: Convert the root README from reStructuredText to Markdown, rendering .rst only if a consumer needs it
type: todo-item
status: backlog
effort: 2
area: docs
queue:
created: 2026-08-09
updated: 2026-08-09
---

> [!note] Overview
> **What** — The repo root ships README.rst. Convert it to README.md as the authored source, and generate/render an .rst form later only if some consumer actually requires one. AGENTS.md names README.rst as the pipeline overview and cites it as a reference, so the conversion has to carry those pointers with it.
> **Why** — Owner preference 2026-08-09: Markdown is the format everything else in this repo is authored in (AGENTS.md, CLAUDE.md, docs/, dev/), so reStructuredText is the odd one out and costs an authoring-mode switch for no current benefit. Filed rather than done inline because README.rst is cited by AGENTS.md and by several sealed milestone records whose line-number references must not be silently falsified -- and because whether any consumer (packaging, PyPI, docs build) still needs .rst is an open question that decides whether a render step is needed at all.
> **Effort** — large

## Progress

- [ ] <first step>
