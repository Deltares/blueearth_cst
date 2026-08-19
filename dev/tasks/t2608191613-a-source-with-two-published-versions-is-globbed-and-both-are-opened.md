---
title: A source with two published versions is globbed, and both are opened
type: todo-item
status: done
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
merge raises first, inside the driver.

**Three faces, not one**, which is why the bucket is matched on a phrase we
build rather than on an exception type:

1. `MergeError: conflicting values for variable 'pr'` — the two versions
   disagree on values. Observed on `CAS/CAS-ESM2-0 historical r1i1p1f1`.
2. `OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 2262-04-16` — the
   two versions cover DIFFERENT SPANS, so aligning their indexes goes through
   `pandas.Index.union`, which upcasts to the finer resolution (`ns`) and
   overflows on the one that runs past 2262. Observed on
   `CSIRO-ARCCSS/ACCESS-CM2 ssp585 r1i1p1f1`. This face looks like the 2262
   defect fixed in `423af1f` and is not: a single 2300 store stages fine
   (`CCCma/CanESM5 ssp585`, verified 2026-08-19), because there is no union.
3. A duplicated time axis, where the values agree well enough to combine. D8's
   own guard catches this one and says so clearly.

And a fourth outcome that is not a failure at all: two versions differing only
in metadata merge cleanly and produce a correct slice. That is why the
diagnostic below wraps a FAILED read instead of refusing up front.

## Measured

Over `config/catalogs/cmip6_store_index.json` (crawled 2026-07-29):

| member combinations | count |
|---|---|
| one version per variable — pinned, fast path | 2205 |
| more than one version each — globbed | 105 |
| `pr` and `tas` versions differ — globbed | 116 |
| **entries with at least one globbed member** | **46 of 289** |

## The decision, made 2026-08-19 — the newest version wins

Owner ruling: **option 1**, and NOT option 3. Option 3 turned out not to be
expressible at all — see below — so the choice was between resolving in
`pinned_uri` and leaving 221 combinations unstageable.

`series_identity.pinned_uri` now takes the newest version per variable and
requires every variable to land on the same location. Measured against the live
index, and asserted from it:

| member combinations | before | after |
|---|---|---|
| pin cleanly | 2205 | **2387** |
| refused, globbed | 221 | **39** |

The 39 are where `pr`'s newest and `tas`'s newest are *different* locations: one
URI carries one `{variable}` placeholder that expands inside a single path, so
it cannot address both. They keep the diagnostic from the section above.

It also refuses to choose between **grid labels** (`gn` vs `gr`). No member in
today's index pins two, so it changes nothing now; it exists because a plain
`max()` would make that choice silently the moment one appeared, and picking a
regridding is not picking a revision.

**`SCHEMA_VERSION` moved 5 -> 6 with it.** The digest carries the pins, never
the rule applied to them, so a slice built by merging two versions and a slice
read from the newest are indistinguishable by digest. Sources where the old
merge FAILED have no cache to worry about; sources where it succeeded — two
versions differing only in metadata — would otherwise keep serving a cache hit
for bytes this code no longer produces. Cost is one re-fetch, and `cache_hit`
treats a schema mismatch as a miss, so it is automatic rather than a deletion
chore. [[t2608191308]] was already owed and is unaffected in size.

### Why option 3 was not available

"Pin per entry in the catalog" cannot express these pins. The catalog has ONE
uri per entry with a `{member}` placeholder, but the version varies BY MEMBER:
`CAS-ESM2-0 historical` records `gn/v20200302` for `r1i1p1f1` and `gn/v20200303`
for `r2i1p1f1`, both alongside `gn/v20201227`. Measured: **136 of 289 entries**
have no single `<grid>/<version>` covering every member and variable. Per-member
pinning already has exactly one home — the store index, read by `pinned_uri` —
and reaching it needs no re-crawl at all, which was the other thing option 3 was
believed to cost.

## The decision this needed

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

## What landed (2026-08-19) — option 2 only

`fetch_gcm_raw.ambiguous_pins` says which of `pinned_uri`'s four refusals
actually happened (a catalog URI with no glob suffix, and a member with no
recorded pins, are NOT ambiguity), and
`explaining_ambiguous_versions` re-raises a failed read naming every version
per variable. A WARNING row also announces the ambiguity BEFORE the read, so
the fourth outcome above — a clean merge nobody chose — is no longer silent.
`stage_cmip6.py` buckets these into their own recap section, "several published
versions, none chosen", because the fix is a decision rather than a repair.

**No version is chosen and no source that stages today stops staging.** The
decision below is still open; only the diagnosis improved.

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
