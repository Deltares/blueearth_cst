# Round-2 index — design-v3.md

Driver-owned aggregation. One external reviewer this round (the internal panel
runs once, before the first external round).

## Verdict

| Reviewer | Model | Verdict | blocking | major | minor |
|---|---|---|---|---|---|
| External r2 | gpt-5.6-sol | `revise` | 3 | 4 | 2 |

**Convergence: NOT converged. External round cap (2) reached** → owner
arbitration, per the skill's cap rule. No further external round is available;
arbitration rulings stand in for the verdict the cap forecloses.

## Driver premise verification

| ID | Verdict | Evidence in `design-v3.md` / repo |
|---|---|---|
| ext2-01 | **HOLDS** | l.598–600: `store_region.geojson` is declared under `ancient()`; §5.3's cache key carries the region **specification**, not the polygon's content. A catalog or delineation change that rewrites the polygon while `shared.basin.region` is unchanged therefore invalidates nothing, and stage B recomputes the same expected digest. Recording old bounds makes it auditable, not prevented |
| ext2-02 | **HOLDS** | l.499: the check raises only for 2-D/curvilinear coordinates; 1-D **monotonic** passes. Monotonic ≠ uniformly spaced — Gaussian latitude grids are 1-D, monotonic, and non-uniform, and need latitude/longitude cell widths, not `cos(lat)` alone. l.490 already concedes validity requires "monotonically-**spaced**" grids, which the check does not test. **This faults the resolution of round-1 `ext1-08`** |
| ext2-03 | **HOLDS** | §5.8 declares `grids/change/{series_key}_{horizon}.nc` but specifies no coordinate/CRS compatibility requirement and no schema (annual vs monthly, which statistics, how dry-reference status and absolute fallback are represented). The gridded branch is new in v3 under R2/R5, so this is under-specification of new material, not a regression |
| ext2-05 | **HOLDS (arithmetic)** | With a non-January `start_month_hyd_year`, a 1985–2014 *calendar* window yields **29** complete hydrological years, not 30 — e.g. an October start gives Oct 1985 … Sep 2014. §5.6's complete-hydrological-years policy then drops the partial years at both ends. Directly affects the owner's OQ-4 ruling |
| ext2-08 | **HOLDS** | l.202: `composition.csv` is a **stage-B** output. Stage B cannot run until every reducer succeeds, so the claim that the resolved set is "written down before any job runs" cannot hold, and it contradicts round-1 `ext1-02`'s own disposition (provenance describes successful runs) |

ext2-04, ext2-06, ext2-07, ext2-09 not separately probed — their premises are
statements about the document's own text, checkable by reading it.

## Regression duty outcome

Round 2 carried a regression duty over all 19 round-1 dispositions. It re-raised
**one**: `ext1-08` (area weighting), via **ext2-02**, on the ground that the
accepted fix — a 1-D/monotonic geometry check — does not establish the validity
condition it claims. Every other round-1 disposition survived scrutiny.

**This triggers the skill's Fable escalation rule.** An external review that
faults the resolution of a prior finding routes the *revision spawn answering
it* to Fable — the Opus revision did not satisfy the reviewer. Routing is
per-spawn, not per-finding: one revision produces the whole next version.

## Findings that are owner decisions, not author fixes

- **ext2-05** — does "30 years, 1985–2014" mean 30 **calendar** years (accepting
  29 complete hydrological years when the hydrological year does not start in
  January), or 30 **complete hydrological years** (moving the calendar bounds)?
- **ext2-06** — `relative_change.min_reference` (OQ-9) and
  `max_flagged_months`: pick defaults, or make both required and populate every
  shipped config?
- **ext2-07** — restrict v2.0's configurable variables to `precip`/`temp` at DAG
  build, or have the generator publish per-variable member availability? Only
  `pr`+`tas` are certified by the catalog; `kin`/`press_msl` are nameable today
  and would fail at read time, which undercuts the absence-vs-failure separation
  D6/D7 were built to establish.

## Author-fixable

ext2-01, ext2-02, ext2-03, ext2-04, ext2-08, ext2-09 — each has a concrete,
bounded fix named in the review. None invalidates the architecture.
