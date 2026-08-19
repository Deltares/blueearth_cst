---
title: Give analyze_projections a member-selection policy and a per-model member override
type: todo-item
status: done
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
  # first_available : at most ONE member PER MODEL — the first that resolves
  #                   for historical AND every requested scenario. New default.
  # all             : every listed member that resolves. Today's behaviour,
  #                   kept for a deliberate multi-member ensemble.
  member_selection: first_available
  # Optional escape hatch, for naming a specific realisation. REPLACES the
  # global preference list for that model rather than prepending to it, and
  # HARD-ERRORS if the named member does not resolve — an override is an
  # assertion about a specific realisation, so falling back silently to the
  # global list would defeat the point of writing it.
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
2. **Resolve PER MODEL, across all requested scenarios — not per
   (model, scenario).** "First available" means the first member that clears
   the whole ladder for `historical` AND every requested scenario, not the
   first that happens to work for one of them.

   Per-scenario resolution re-opens the very defect this item closes, through a
   different door: `ssp245` could land on `f1` while `ssp370` lands on `f2` for
   the same model, each individually D7-valid. But
   `analyze_projections.smk:522` builds `_needed` as
   `{(dataset, "historical", member) ...}`, so that model would acquire **two
   historical baselines** and its two scenarios would be differenced against
   different references. `references()` says the same thing from the other
   side — it returns distinct `(model, member)` pairs, and the job arithmetic
   in its docstring assumes one reference per model.

   D7 is the narrower rule underneath: a scenario point pairs with the SAME
   member label's historical (`resolution.py:168`), so a member present in
   `ssp245` but absent from `historical` must fall through to the next
   preference rather than resolve and fail later. The ladder already carries
   the statuses that make both checkable — `MEMBER_NOT_PUBLISHED` and
   `REFERENCE_MEMBER_UNPUBLISHED`.

   Consequence worth putting in the config comment: under `first_available` a
   model resolves at ONE member or not at all. Adding a scenario can therefore
   change which member a model uses, or drop the model entirely.
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

- [x] Extend `resolution.resolve()` with the policy and the
      `MEMBER_SUPERSEDED` status; keep the per-triple emission contract.
- [x] Parse `member_selection` and `member_overrides` in
      `analyze_projections.smk` (optional keys, `first_available` default),
      accepting the plain-list `members:` form unchanged.
- [x] Decide whether `member_selection` belongs in
      `config/advanced_settings.yml` `defaults:` instead — **owner ruling: the
      project block.** It changes which data a basin's assessment uses, the
      same class of decision as `members:` itself, and `advanced_settings.yml`
      holds constraints / defaults / runtime, none of which a selection policy
      is. Its closed schema is untouched.
- [x] Tests: the double-publish case, the fall-through case, the cross-scenario
      case, the override cases, and the unchanged single-member case. Two more
      than the list asked for — that `references()` still returns ONE
      `(model, member)` per model under `first_available` (the property the job
      arithmetic actually depends on), and that an unknown policy string raises
      rather than silently falling back.
- [x] Migration note — **not written, deliberately.** A migration doc earns its
      keep when an existing project must DO something, and none must: every
      tracked config is single-element, which resolves identically under both
      policies. AGENTS.md records deleting `docs/migration-r06.md` for being a
      map kept for its own sake. The contract change is documented where it is
      read — `resolve()`'s docstring, which states the R3′ supersession and why
      — and in this note.
- [x] Update `dev/scripts/sample_bundle.yml`. It had already moved to
      `members: [r1i1p1f1, r1i1p1f2]` with a `# include a switch/single vs
      multiple members.` placeholder, which under the OLD union rule was the
      double-counting hazard live in the shipped bundle. The policy is what
      makes that list correct.

## What landed (2026-08-19)

`members` is an ordered preference; `member_selection` is `first_available`
(default) or `all`; `member_overrides` maps a model to its own preference list,
replacing the global one.

The ladder moved into `_model_statuses()` so a whole model is visible before any
row is emitted — which member wins is a property of the model across ALL
scenarios, not of one scenario. `_winning_member()` takes the first preference
resolving for every requested scenario; historical needs no separate test,
because `REFERENCE_MEMBER_UNPUBLISHED` already refuses a member the historical
entry does not publish, so a member clearing every scenario has a matching
reference by construction.

ONE new status, not two. A member that loses because something preferred won
and a member that loses because nothing was complete are different facts, but no
consumer branches on the distinction — `format_status_report` prints
`status — detail` and the detail carries it:

    superseded by r1i1p1f1, which resolves for every requested scenario
    no requested member resolves for all of ssp245, ssp585

`resolve()` still records rather than raises: `unresolved_overrides()` reports
and `analyze_projections.smk` raises, the same split as `unknown_models` and the
nothing-resolved check. It also catches an override key naming a model the run
does not request — a typo nothing else would see, since `unknown_models` only
looks at models that ARE requested.

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
