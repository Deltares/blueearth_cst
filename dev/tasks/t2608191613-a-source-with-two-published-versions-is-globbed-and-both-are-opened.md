---
title: A source with two published versions is globbed, and both are opened
type: todo-item
status: backlog
effort: 2
area: wf2 projections / remote store
origin: 2026-08-19 stage_cmip6 run
queue: 3
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — A `(model, scenario, member)` whose store index records more than
> one published version falls to the globbed catalog URI, which matches every
> version. `open_mfdataset` then gets two stores per variable and raises
> `MergeError: conflicting values for variable 'pr' on objects to be combined`.
> **Why** — `221 of 2426` member combinations (9%), across `46 of 289` entries.
> Those sources cannot stage at all, so they are absent from any ensemble that
> asked for them.
> **Effort** — Small in code, but the decision it needs is methodological, not
> technical: choosing a version is choosing which data the assessment uses.

## The mechanism

`series_identity.pinned_uri` narrows the catalog's trailing `/*/*` to the one
`<grid_label>/<version>` the store index recorded, and returns `None` — meaning
"keep the glob" — whenever the pins cannot name ONE physical location. The
globbed URI `.../{variable}/*/*` then matches both versions of both variables,
so four stores go into the combine.

Observed on `cmip6_CAS/CAS-ESM2-0_historical_{member}`, whose `r1i1p1f1` records
`gn/v20200302` and `gn/v20201227` for each of `pr` and `tas`.

`fetch_gcm_raw.check_time_axis` carries a guard for exactly this ambiguity (D8,
"the catalog glob matched more than one store"), but it never runs here: the
merge raises first, inside the driver. Where the two versions happen to agree on
values the merge succeeds instead and the guard does fire, on a duplicated time
axis — so this defect has two faces and only one of them is currently reported
in our own vocabulary.

## Measured

Over `config/catalogs/cmip6_store_index.json` (crawled 2026-07-29):

| member combinations | count |
|---|---|
| one version per variable — pinned, fast path | 2205 |
| more than one version each — globbed | 105 |
| `pr` and `tas` versions differ — globbed | 116 |
| **entries with at least one globbed member** | **46 of 289** |

## The decision this needs

`pinned_uri`'s docstring states the current position: the >1-match group "must
stay globbed so the duplicate-time assertion still sees it" — the ambiguity is
meant to be visible rather than silently resolved. That is a defensible reading:
a newer version is a REVISION of the data, and preferring it automatically means
the ensemble quietly changes when a modelling centre republishes.

Three ways out, none of them free:

1. **Prefer the newest version tag.** One line in `pinned_uri`. Makes the
   sources stage, and reverses the documented decision above. Every affected
   slice's digest changes, so cached ones re-fetch.
2. **Refuse them by name, early and clearly.** No version chosen; the failure
   stops being a raw hydromt `MergeError` and becomes a refusal naming both
   versions, landing in `stage_cmip6.py`'s report beside the irregular-grid
   split. Diagnostic only — the sources still do not stage.
3. **Pin per entry in the catalog.** Correct per source, but `cmip6_data.yml` is
   generated with no offline mode, so it means a re-crawl.

Option 2 is the cheap half of 1 and does not foreclose it.

## Related

- `t2608182020` — the other reason a CMIP6 model silently misses an ensemble
  (Gaussian grids). Same shape of finding: a population-level absence nobody was
  told about.
- `dev/milestones/r08/2026-07-30_wf2-fetch-reduce-benchmark.md` §3.2 — why the
  pin exists at all (~10 s per source, and determinism).

## Trigger

Raise this above backlog when an ensemble run needs one of the 46 entries, or
when the next catalog re-crawl is being planned anyway — a re-crawl is the
moment option 3 costs nothing extra.
