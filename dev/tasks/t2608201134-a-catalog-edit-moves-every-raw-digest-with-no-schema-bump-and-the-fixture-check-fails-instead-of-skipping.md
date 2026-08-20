---
title: A catalog edit moves every raw digest with no schema bump, and the fixture check fails instead of skipping
type: todo-item
status: active
effort: 1
area: wf2 projections / test fixtures
origin: 2026-08-20 board-closure gate
queue: 1
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — b0963e9 (2026-08-19 22:55) added tasmin/tasmax to config/catalogs/cmip6_data.yml's rename and unit_add maps -- 72 minutes AFTER the WF2 re-run wrote test_local's 9 raw slices at 21:43. Those maps sit inside entry_identity, a raw_digest component, so every raw digest moved. tests/test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote now FAILS on main for all 9 slices (INM-CM4-8 historical recomputes e629400635b2 against a recorded 03877288efa5). Confirmed in the PRIMARY checkout in 8.6 s, so it is not worktree env drift; pins, region fingerprint, buffer, window and schema all reproduce exactly, which is what leaves entry_identity as the moved component.
> **Why** — The digest MOVING is correct: entry_identity is hashed precisely so that a catalog change re-fetches rather than serving bytes the catalog no longer describes. What is wrong is the reporting. The whole-fixture skip keys on cst_schema_version, and a catalog edit does not touch it -- so a fixture stranded this way FAILS rather than skipping, and any catalog edit reddens the suite until WF2 is re-run. This is the THIRD landed change to strand this fixture (t2608191308 for schema 4->5, t2608192113 for 5->6), and t2608192113's own ledger row said a third occurrence should raise whether that check ought to fail rather than skip. Two exits, and the choice needs a ruling: re-run WF2 to refresh the fixture, or widen the skip to cover component drift as well as schema drift -- the second risks a skip so wide it stops testing the claim.
> **Effort** — small

## Progress

- [x] Owner ruling: refresh the fixture AND teach the check which component
      drifted. Widening the skip alone was rejected -- a skip that broad stops
      testing the claim the test exists for.
- [x] Establish whether a new attribute owes a `SCHEMA_VERSION` bump. **It does
      not**, and this is asserted rather than argued: `cache_hit` compares the
      schema version and the digest attribute and nothing else, so a purely
      diagnostic `cst_*` key cannot move a cache decision. Pinned by
      `test_a_diagnostic_attribute_does_not_change_a_cache_decision`.
- [x] Raw slices stamp `cst_entry_identity_digest`; the check uses it to tell a
      stale artifact (skip, naming the reason) from a broken recipe (fail).
      Landed on `fix/fixture-digest-drift-attribution`.
- [x] Both branches proved empirically against the worktree's disposable
      fixture copy, then the copy restored byte-identical to the primary's:
      a recorded digest that disagrees with the live catalog SKIPS with
      "9 were built from a catalog entry that has since changed"; a recorded
      digest that AGREES while the raw digest still differs FAILS, so the
      recipe-break case is not swallowed.
- [ ] **Re-run WF2 stage A from the PRIMARY** against
      `test_case/snake_config_baseline.yml`, so the nine slices gain the
      attribute and current digests. Must follow the merge -- a run can only
      write an attribute the code already emits, which is why this task's two
      halves cannot be done in one lane.
- [ ] `check_baseline.py check` afterwards. Not ceremony: the catalog edit that
      caused this added `rename` and `unit_add` entries, exactly the kind that
      could move a change-factor value, and those CSVs are baseline targets.
- [ ] Confirm the real check goes green (9 checked, 0 skipped), then close.

## Not covered

`test_rapid`'s slices (4 + 4) are stale on the same mechanism. No test reads
them, so this is a note rather than a gate; they refresh on the next rapid run.

The series tier does not stamp the attribute. It has no equivalent
reproduction test reading `cst_series_digest`, so it would be symmetry without
a consumer -- add it when one appears.
