---
title: Give analyze_projections a member-selection policy and a per-model member override
type: todo-item
status: backlog
effort: 2
area: wf2 projections / config contract
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Make `workflows.analyze_projections.members` an ordered
> PREFERENCE list with a selection policy (`first_available` | `all`), and allow
> a per-model override map. Today it is a flat set applied to every model.
> **Why** — Three models used heavily in African studies are unreachable at all,
> and the obvious workaround silently double-weights the ensemble.
> **Effort** — Medium-large. One config key, one function, a new resolution
> status, and it supersedes a recorded ruling.

## What is actually broken

**1. Forcing-variant models are unreachable.** `members:` is one list applied to
every model, so a config asking for `r1i1p1f1` silently excludes every model
that publishes only a different forcing variant. Measured against the generated
catalog on 2026-08-19, complete across `historical` + `ssp245`/`ssp370`/`ssp585`
at `r1i1p1f2`:

```
CNRM-CERFACS/CNRM-CM6-1     CNRM-CERFACS/CNRM-CM6-1-HR   CNRM-CERFACS/CNRM-ESM2-1
MOHC/UKESM1-0-LL            MIROC/MIROC-ES2L             NASA-GISS/GISS-E2-1-G
NASA-GISS/GISS-E2-1-H       UA/MCM-UA-1-0
```

`MOHC/HadGEM3-GC31-LL` is `f3` and additionally lacks `ssp370`.

**2. The obvious workaround is worse than the problem.** Adding `r1i1p1f2`
alongside `r1i1p1f1` looks like the fix. It is not: **CAMS/CAMS-CSM1-0,
EC-Earth-Consortium/EC-Earth3 and NCC/NorESM2-LM are complete at BOTH**, so each
would resolve twice and contribute two data points where every other model
contributes one. `get_change_climate_proj_summary.py:73` merges across models
and reduces with `stats="mean"`, so those three would be **weighted double in
the multi-model ensemble** — a silently wrong number, not a cosmetic issue.
That is the defect this item closes; reaching the f2 models is the feature.

## Proposed config surface

```yaml
analyze_projections:
  # ORDERED preference, most-wanted first — not a set.
  members: [r1i1p1f1, r1i1p1f2]
  # first_available : at most ONE member per (model, scenario) — the first that
  #                   fully resolves. The new default.
  # all             : every listed member that resolves. Today's behaviour,
  #                   kept for a deliberate multi-member ensemble.
  member_selection: first_available
  # Optional escape hatch, for naming a specific realisation:
  member_overrides:
    MOHC/UKESM1-0-LL: [r13i1p1f2]
```

Accept the plain-list form unchanged, so no existing config has to move.

## Four design constraints, all read out of the code

1. **`resolve()` must keep emitting one `Combination` per REQUESTED triple.**
   Its docstring is explicit: *"Returns one Combination per requested one — not
   per resolved one. That is the point: the skips are what make the composition
   record auditable."* So `first_available` must record the passed-over members
   under a **new status** (`MEMBER_SUPERSEDED`, alongside the five in
   `resolution.py:35-40`), never drop them. `format_status_report` then says
   *why* a member was not used, which is the whole reason that report exists.
2. **"First available" means first that clears the WHOLE ladder, not first
   published.** D7 pairs a scenario point with the SAME member label's
   historical (`resolution.py:168`), so a member present in `ssp245` but absent
   from `historical` must fall through to the next preference rather than
   resolve and fail later. The ladder already has the two statuses that make
   this checkable — `MEMBER_NOT_PUBLISHED` and
   `REFERENCE_MEMBER_UNPUBLISHED`.
3. **This supersedes ruling R3′**, quoted in `resolve()`'s docstring:
   *"`members` is a requested SET intersected with what each combination
   publishes; the run's data-point set is the union of those per-combination
   resolutions."* Union-of-resolutions IS `member_selection: all`. Record the
   supersession where R3′ lives rather than leaving the docstring asserting the
   old contract.
4. **Cache and digest: safe to flip the default.** The member is part of
   `series_key` and of `raw_components`, so a change in which member resolves
   re-fetches. But a **single-element `members:` list resolves identically under
   both policies**, and every tracked config today is single-element — so
   defaulting to `first_available` invalidates nothing that exists. Say this in
   the migration note; it is the reason the default can change at all.

## Progress

- [ ] Extend `resolution.resolve()` with the policy and the
      `MEMBER_SUPERSEDED` status; keep the per-triple emission contract.
- [ ] Parse `member_selection` and `member_overrides` in
      `analyze_projections.smk` (optional keys, `first_available` default),
      accepting the plain-list `members:` form unchanged.
- [ ] Decide whether `member_selection` belongs in
      `config/advanced_settings.yml` `defaults:` instead — it is a
      toolbox-wide policy, not a per-basin one. Note the schema is CLOSED, so
      the key and `snake_utils._ADVANCED_SETTINGS_SCHEMA` move together.
- [ ] Tests: the double-publish case (EC-Earth3 at f1 and f2 → exactly one
      resolved combination under `first_available`, two under `all`); the
      fall-through case (member in scenario, absent from historical); the
      unchanged single-member case.
- [ ] Migration note — a config key changes meaning. `members:` stops being a
      set and becomes an ordered preference, which is a contract change even
      though every existing config is unaffected.
- [ ] Update `dev/scripts/sample_bundle.yml`, whose `models:` block currently
      documents this limitation as a hard fact ("NOT REACHABLE"), and the
      `members:` comment that says a list is applied to every model.

## Why it is worth doing

The sample bundle's 15-model ensemble was picked one-model-per-institution to
keep lineages independent. Two of the three models this unlocks — CNRM-CM6-1
and UKESM1-0-LL — are separate lineages the current set reaches only by proxy
(`NIMS-KMA/KACE-1-0-G` stands in for the HadGEM3 family precisely because it is
published at f1). So this directly improves the ensemble the sample ships with,
not only the configs a user might write.

## Refs

- `blueearth_cst/projections/resolution.py` — `resolve()`, the five statuses,
  and R3′ / D7 in the docstrings.
- `blueearth_cst/projections/get_change_climate_proj_summary.py:73` — the
  cross-model merge that makes double-counting a wrong number.
- `dev/scripts/sample_bundle.yml` — the `models:` block documenting today's
  limitation, written 2026-08-19.
- [[t2608191733-ship-a-sample-dataset-bundle-so-a-user-needs-no-deltares-p-drive]]
  — the bundle whose ensemble this improves. Independent: the bundle can ship
  before this lands, and re-staging afterwards costs only the new slices.
