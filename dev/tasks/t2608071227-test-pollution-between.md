---
title: The first test file to call `sys.modules.setdefault` decides the stub for every later file
type: watch-item
area: tests
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — The first test file to call `sys.modules.setdefault` decides the stub for every later file.
> **Why** — Collection order therefore changes what is under test; today the orderings that matter happen to pass.
> **Trigger** — A new stubbing test file lands, or an ordering change makes a passing suite fail.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**Test pollution between `sys.modules.setdefault` files.** pytest collects
test files in alphabetical order. The first file to call
`sys.modules.setdefault("hydromt", <stub>)` (or any heavy dep) wins, and
later files using `setdefault` for the same key get a silent no-op —
their import of the source module then binds to the *previous* test
file's stub. Symptom: tests pass when run in isolation, fail in the full
suite with `KeyError` on fixture-set catalog data.

*Pattern:* don't rely on `setdefault` alone for shared keys. Use
`monkeypatch.setattr(<source_module>.<dep>, "<attr>", <fake>)` inside
fixtures so each test gets a clean override regardless of collection
order. See `tests/test_prepare_climate_data_catalog.py` for the
reference implementation; commit `f65244e` for the diagnosis.
