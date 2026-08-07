---
title: hydromt's `to_yml` silently strips `driver.options.preprocess` on a catalog round-trip
type: watch-item
area: upstream / hydromt
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — hydromt's `to_yml` silently strips `driver.options.preprocess` on a catalog round-trip.
> **Why** — `prepare_climate_data_catalog.py` works around it with `yaml.safe_dump`; the workaround is invisible until someone removes it.
> **Trigger** — Upstream fixes `to_yml` — `tests/test_prepare_climate_data_catalog.py`'s xfail flips to a pass and fails CI under strict, which is the signal.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**`hydromt 1.x` `to_dict` / `to_yml` silently strips `driver.options.preprocess`.**
Round-tripping a catalog dict through `DataCatalog().from_dict(...).to_yml(path)`
loses the preprocess hook even though `from_dict` preserves it on read.
*Workaround applied:* `src/prepare_climate_data_catalog.py` bypasses
`to_yml` and uses `yaml.safe_dump` directly.
*Proper fix:* file upstream against `hydromt`. Reproducer is the
three-line snippet in `dev/milestones/phase-1/m02b/handoff.md` decision section.
