# WF2 v2.0 design — consolidated review record

Audit trail for the `design-review-loop` run `wf2-climate-analysis-v2`, which
produced `dev/milestones/r08/wf2-climate-analysis-v2-design.md` (ACCEPTED 2026-07-29).

This record is the durable artifact; the per-round scratch (`design-v1.md` …
`design-v4.md`, `status.md`, the briefs and transcripts) lives in git history
under `dev/working/design-runs/wf2-climate-analysis-v2/`. Commits holding each
verbatim review: `5d953e4` (round 1), `f7db117` (round 2), `a9a1bc4` (v4).

## Run summary

| | |
|---|---|
| Run | `wf2-climate-analysis-v2` |
| Genre | workflow spec (carrying decision-record content) |
| Variant | lite, **promoted to full** on round-1 blocking findings |
| Versions | `design-v1.md` -> `v2` -> `v3` -> `v4` (accepted) |
| Internal panel | 1 lens (`critical-thinker`, **Fable**, owner-requested) |
| External rounds | 2 of 2 — cap reached, resolved by owner arbitration |
| Findings | 28 total: 19 round-1, 9 round-2 |
| Dispositions | 28 accepted; 1 partially rejected by owner ruling (`ext1-07`) |
| Gates | G1 approved, two G1 returns, G2 approved 2026-07-29 |
| Dispatches | opus 2, fable 2 |

## Verdict table

| Stage | Reviewer | Model | Doc version | Verdict | blocking | major | minor |
|---|---|---|---|---|---|---|---|
| Internal panel | `critical-thinker` | Fable | `design-v1.md` | `revise` | 1 | 5 | 3 |
| External round 1 | `codex exec` | gpt-5.6-sol | `design-v1.md` | `revise` | 3 | 6 | 1 |
| External round 2 | `codex exec` | gpt-5.6-sol | `design-v3.md` | `revise` | 3 | 4 | 2 |
| Arbitration | owner | — | `design-v3.md` | accepted, fix required (all 9) | — | — | — |
| G2 | owner | — | `design-v4.md` | **approved** | — | — | — |

Convergence was never reached mechanically: round 2 returned `revise` on the
last available round, so the **owner's arbitration rulings A1–A3 stand in for
the verdict the cap foreclosed**, per the loop contract. `design-v4.md` was
authored against those rulings and approved at G2 without a further review.

## What the review changed

Four things the design asserted confidently and got wrong:

1. **The reference window was unreachable.** Both round-1 reviewers
   independently found that the CMIP6 historical experiment ends in **2014**
   while the proposed reference window ran to 2020. Verified at
   `get_stats_climate_proj.py:156`. A regression the design introduced, not one
   it inherited.
2. **One reducer could not serve two source frequencies.** The observed store is
   daily (`freq=D`, 7671 steps, precip `mm d**-1`, temp **K**); CMIP6 `Amon` is
   monthly, temp degC. Verified by opening the fixture store.
3. **Runtime failure handling was incompatible with the DAG.** A failed reducer
   halts Snakemake before any rule could record the failure.
4. **The job arithmetic was wrong** (14, not 13, and it omitted a series), and
   the **baseline-manifest coverage was overstated** — it pins 7 WF2 targets, of
   which the 7th is the config snapshot, hashed by sha256, so a config-key
   addition breaks it with no number moving.

Round 2's regression duty re-checked all 19 round-1 dispositions and re-raised
**exactly one** — `ext1-08`, on the ground that its accepted fix (a
1-D/monotonic geometry check) did not establish the validity condition it
claimed. That re-raise triggered the Fable escalation for the final revision,
which replaced the weighting scheme rather than the check.

## Owner rulings

### G1 (2026-07-29)

- **R1 — clip, never splice.** The GCM reference clips to the historical
  experiment (<=2014); the 2015–2020 gap is never filled from scenario data; the
  workflow warns when a configured period overruns 2014. Accepted consequence:
  the overlay reference period and the stress-test baseline remain different
  periods.
- **R2 — the gridded option is retained, default off**, with declared outputs.
- **R3 — superseded by R3'.**
- **R3' — no aggregation at any level.** Members are never averaged; each
  (model, scenario, member) is one data point with its own dT and dP; member
  availability is a union of what the store publishes, and missing SSP scenarios
  are normal.
- **R3'' — cross-combination statistics are ex-post**, out of v2.0 scope.
  Per-series statistics remain in scope: they are what the change factor is.
- **R4 — v2.0 is monthly GCM projections analysis** with a gridded expansion; no
  observed-data comparison at this stage.
- **R5 — the basin-averaged monthly series per run is a declared deliverable**,
  alongside the change-factor table and the composition record.
- **Confirmations:** D2 -> A1 (declare the full store); OQ-4 -> 30 years,
  1985–2014.

### Arbitration (2026-07-29, round-cap authority)

- **A1 — 30 calendar years.** The 29-complete-hydrological-year consequence of a
  non-January start is accepted, but effective bounds and counts must be
  reported rather than silently differ from the nominal window.
- **A2 — pick the dry-month defaults** rather than leaving them to the
  implementer. Closes OQ-9.
- **A3 — non-`pr`/`tas` variables stay selectable**, shipped configs default to
  `[precip, temp]`, and the certified/best-effort tier distinction is explicit.

## Driver premise verification

The driver checked facts and never authored design content. Verified against the
repository before arbitration:

| Claim | Outcome |
|---|---|
| CMIP6 historical ends 2014, so the proposed reference window overruns it | HOLDS — `get_stats_climate_proj.py:156`; regression, not pre-existing |
| Observed store is daily / K; `Amon` is monthly / degC | HOLDS — fixture store: `freq=D`, 7671 steps, 7 variables |
| Manifest pins 7 WF2 targets, only 6 under `climate_projections/cmip6/` | HOLDS — 7th is the config snapshot, `{"sha256": ..., "type": "yaml"}` |
| Job arithmetic wrong | HOLDS — 1+9+1+1+1+1 = 14, and the count omitted a series |
| `DKRZ/MPI-ESM1-2-HR` has SSP entries and no historical | HOLDS — generated catalog |
| `EC-Earth3_ssp245` publishes 96 members | HOLDS — 2426 sources total |
| `historical_year_range` is `optional=False` | HOLDS — `Snakefile_climate_projections:36`; so OQ-4's closure is template-only |
| Region polygons have identical bounds on the fixture | **PARTIALLY — corrected 2026-07-29 during step-1 implementation.** Bounds agree to 6 dp but differ by 3.33e-07° (~3.7 cm); the model polygon is stored rounded, the store polygon is full precision. The conclusion survives on a bound rather than equality: across 36 (resolution, origin) combinations spanning CMIP6 Amon grids, none changes cell selection. The original check compared rounded values |
| D10's weights reduce to cos-latitude on a uniform grid | HOLDS — residual <= 3e-17 at phi in {0, 45, 80, -67} degrees |

## Known deviations carried into the accepted design

1. **A1's letter vs the cache contract.** Effective window values are reported in
   the change-factor tables, `composition.csv`, `provenance.json` and the report,
   but deliberately **not** stamped onto cached series files, whose identity
   excludes analysis windows so the cache can work at all.
2. **D12 adds a generated repo artifact**
   (`config/catalogs/cmip6_store_index.json`) — no new dependency, but a
   generator-format decision that follows from `ext2-04`'s accepted fix rather
   than from an explicit ruling. Adds migration step 2a.
3. **A2's defaults are argued, not measured.** `min_reference` = 0.1 mm/day for
   precip and `max_flagged_months` = 3 are reasoned choices, revisable by the
   measurement OQ-9 names.
4. **`ext1-07` is partially rejected** — its "average members within each model"
   half is rejected by R3'; R3'/R3'' constitute the owner ratification a rejected
   `major` requires.

## Process observations

- The `design-document` skill's file-placement rule hardcoded `dev/drafts/`
  against this repo's documented `dev/` grammar. Fixed upstream mid-session in
  the brain repo (`c63f130f`) as a precedence rule.
- `design-review-loop`'s run-directory example carries the same defect
  (`dev/drafts/design-runs/`). Left unchanged during the run — one skill version
  per run keeps results attributable — and recorded here as a post-run candidate.
- Two author spawns left tool markup (`</content>`, `</invoke>`) at file ends;
  stripped as logged editorial edits. A third left two revision-log entries both
  marked "(this file)".
- The codex adapter's fail-closed preflight matters on this machine: without an
  explicit `-c approval_policy=never`, the project default `on-request` applies,
  under which a write escalates outside the sandbox silently.


---

# Intake (verbatim)

# Intake — wf2-climate-analysis-v2

## Change request (verbatim)

> I would like to do some scripting changes for increased efficiency and
> modularity in WF2.

> If you were to redesign WF2, how would you do? Consider scientific
> methodology, computational efficiency, and user-friendliness

> Yes, lets turn this into a design document but one remark: while WF2 is
> currently on interested in delta factors, I would like to develop it further
> into a handy climate data & projections analysis workflow. So lets keep this
> in mind while we are desiging WF2_V2.0

> Before that, lets send this design document to Fable & GPT 5.6 SOL so that
> they both individually review intention and the design document and provide
> input. We can then finalize the design

## Problem

WF2 (`Snakefile_climate_projections`) computes CMIP6 change factors and is
structured against its own cost profile: it fans out where computation is free
and serializes where the network is expensive, marks every intermediate
`temp()` so a re-run re-downloads the archive, discards the monthly seasonality
it computes, and cannot run without a built hydrological model. The owner wants
it to become a general climate data and projections analysis workflow.

## Constraints (standing, from `AGENTS.md` + sealed milestones)

- WF2 is a plausibility overlay; WF3 is never driven by CMIP6 scenarios.
- `climate_store_spec` declarations must stay symmetric across Snakefiles.
- hydromt / hydromt_wflow / Wflow conventions are consumed verbatim.
- No new dependency without explicit owner approval.
- `check_baseline.py` on `test_case/test_local` is a local-only gate.

## Decision criteria

Recorded in the design doc §4: tiered value-neutrality; cost follows the
network; sealed acceptance gates are not reopened; explicit beats inferred; the
extension surface is designed once.

## Success criteria

An owner-accepted design that (a) is implementable in commit-sized units with a
value-neutral prefix, (b) widens WF2 to observed + projected climate analysis
without reopening P3-2a, and (c) resolves or explicitly defers OQ-1..OQ-8.

## Non-goals

Implementation (a task brief for migration steps 1–2 already exists at
`dev/milestones/r08/2026-07-29_wf2-v2-decouple-and-cache.md`); a 4th Snakefile entry
point; new dependencies; bias correction or downscaling.

## Genre mapping

`workflow-spec` per `design-document`. The doc also carries decision-record
content (named decisions D1, alternatives, consequences); recorded here rather
than splitting the artifact.

## Derived-artifact register

What depends on this design, and what regenerates after G2:

| Artifact | Relationship |
|---|---|
| `dev/milestones/r08/2026-07-29_wf2-v2-decouple-and-cache.md` | Task brief for steps 1–2; **must be re-checked** if the review changes step ordering or the value-neutrality classification |
| `dev/reference/workflows/wf2_climate_projections_overview.md` | Current-state map; unaffected by the design, but its findings feed it |
| `dev/reference/workflows/climate_projections.md` | Current behavioral contract; §5.4 of the design records a contract/code discrepancy that lands here at implementation |
| `dev/roadmap.md` | OQ-2 (Phase 4 or close) would add a section |


---

# Round-1 aggregation index

# Review aggregation index — round 1, design-v1.md

Driver-owned. Groups duplicates; **never deletes or re-grades**. Every original
ID, severity, and text is preserved by reference in
`internal-review-risk.md` and `external-review-r1.md`.

Lite variant ran a single internal lens, so this index subsumes the
`internal-review-index.md` slot and covers both round-1 reviews together.

## Verdicts

| Reviewer | Binding | Model | Verdict | blocking | major | minor |
|---|---|---|---|---|---|---|
| Internal risk lens | `critical-thinker` | Fable | `revise` | 1 | 5 | 3 |
| External | `codex exec` | gpt-5.6-sol | `revise` | 3 | 6 | 1 |

19 findings; 4 blocking IDs covering **3 distinct defects** (risk-01 ≡ ext1-01).

**Convergence check: NOT converged.** External verdict is `revise` with blocking
findings outstanding. **Promotion trigger fired** — blocking findings escalate
the lite variant to full: the remaining lenses (architecture, repo-fit) become
available and the external cap reverts to 2.

## Convergent findings (independent agreement — highest confidence)

Two reviewers with no shared context reaching the same defect at the same
severity is the strongest signal in this run.

| # | Defect | IDs | Severity (both) |
|---|---|---|---|
| C1 | Reference window vs the 2014 end of the CMIP6 historical experiment | risk-01, ext1-01 | blocking |
| C2 | Cache-digest `window` semantics undefined; changing a horizon may re-download | risk-02, ext1-04 | major |
| C3 | Monthly relative precip change has no near-zero-denominator policy | risk-05, ext1-05 | major |
| C4 | Seed job-count arithmetic wrong and omits the observed series | risk-09, ext1-10 | minor |

C3 differs in scope, not severity: ext1-05 bundles calendars, partial
hydrological years, and missing months into the same finding; risk-05 is
confined to the dry-month denominator. Both readings retained.

## Unique to the internal lens (Fable)

| ID | Severity | Defect |
|---|---|---|
| risk-03 | major | `reducer_version` is hand-bumped; Snakemake's code trigger tracks the script body, not imported modules → silent stale-cache path with no failure signal |
| risk-04 | major | §8's "checkable against the baseline manifest" conflicts with step 3's renames; a re-record under a rename matches itself, severing the value-neutrality evidence chain |
| risk-06 | major | If stage B globs `series/*.nc`, stale series silently join the ensemble; §7 R4 treats accumulation as disk hygiene only |
| risk-07 | minor | Declaring `extract_climate_grid` transfers the store-build cost and its network failure surface onto a projections-only run — a cost transfer, not the reduction "no longer needs WF1" implies |
| risk-08 | minor | The DAG-build validator re-implements hydromt's catalog resolution; drift makes validator and hydromt disagree |

## Unique to the external reviewer (gpt-5.6-sol)

| ID | Severity | Defect |
|---|---|---|
| ext1-02 | **blocking** | Runtime source-failure handling is incompatible with the DAG: a failed reducer job halts Snakemake before stage B, so `provenance.json` failure records, minimum-source enforcement, and continuation with survivors cannot work as specified |
| ext1-03 | **blocking** | `aggregate: sum` cannot uniformly reduce daily observed and monthly-mean `Amon` inputs; summing daily precip gives mm/month, summing Amon preserves mm/day — the common `units: mm/day` label would equate unlike quantities |
| ext1-06 | major | "Every slot is a read, not a pipeline" is false: S2 needs multi-source extraction, S4 a daily acquisition branch — so G9 is not delivered by the selected architecture |
| ext1-07 | major | With `member` a wildcard, the ensemble sampling unit is undefined; a 10-model threshold and cross-model envelopes are ambiguous when models contribute unequal member counts (pseudoreplication) |
| ext1-08 | major | cos-latitude is not a cell-area weight on Gaussian/irregular/curvilinear grids, and the catalog drops coordinate bounds — the reducer may be unable to establish true areas while claiming area weighting |
| ext1-09 | major | OQ-8 blocks the architecture: stage A discards spatial dims, so stage B cannot reproduce `save_grids` products; leaving it open makes the rule graph, output contract, job count, and step-4 neutrality claim indeterminate |

## Conflicts and severity divergences (required section)

**No factual contradictions.** Neither reviewer declined a finding the other
raised; the unique findings are complementary coverage, not disagreement.

**No severity divergences** on any shared defect — C1 blocking/blocking, C2
major/major, C3 major/major, C4 minor/minor. Recorded because the *absence* of
divergence across independent model families is itself evidence: the design is
under-determined in the places both found, not merely criticized by one taste.

**One grading asymmetry worth noting.** §5.7 (source resolution) drew
risk-08 `minor` — the validator's *drift* risk — and ext1-02 `blocking` — the
same section's *runtime failure* path. Different aspects of one section, not a
re-grade of one defect. Both stand at their filed severities.

## Driver premise verification

The driver checks facts, never authors. Verified against the repo before
arbitration:

| ID | Verdict | Evidence |
|---|---|---|
| risk-01 / ext1-01 | **HOLDS — regression, not pre-existing** | `get_stats_climate_proj.py:156` hardcodes `("1950-01-01","2014-12-31")` for cmip6 historical; catalog resolves historical under `CMIP/{model}/historical/`. `shared.historical_window` ends 2020-12-31. Current `[1990,2010]` fits; the design's G3 does not |
| ext1-03 | **HOLDS — stronger than filed** | `climate_historical/era5_20000101_20201231/extract_historical.nc`: `freq: D`, 7671 steps, precip `mm d**-1`, temp **`K`**, 7 variables. CMIP6 Amon is monthly, temp °C, 2 variables. The mismatch is temporal **and** unit **and** variable-coverage |
| risk-04 | **HOLDS** | `dev/baseline/manifest.json` pins exactly 7 WF2 targets: **6** under `climate_projections/cmip6/` (3 PNGs, `annual_change_scalar_stats_summary.{nc,csv}`, `..._summary_mean.csv`) **plus** `config/runs/snake_config_climate_projections.yml`. No monthly intermediates |

> **Driver correction, 2026-07-29 (caught by the stage-6 author).** This row
> originally read "all under `climate_projections/cmip6/` … config snapshot" —
> self-contradictory, since the config snapshot is not under that path. The
> distinction is load-bearing, not cosmetic: the 7th target is pinned as
> `{"sha256": …, "type": "yaml"}`, a hash of the config file's bytes (verified).
> **Any config-key addition breaks `check_baseline` with no computed number
> moving** — so *value-neutral* and *manifest-clean* are different properties,
> and a migration step that adds a config key cannot be gated on a clean
> manifest alone. `design-v2.md` §4 criterion 1 now separates them.
| risk-09 / ext1-10 | **HOLDS** | 1+9+1+1+1+1 = 14, not 13; with the observed series, reduce is 10 → 15 |
| ext1-02 | **HOLDS on Snakemake semantics** | A rule whose declared output is absent after a failed job halts the DAG; no downstream rule can then write a failure record. Not separately probed — the semantics are not in dispute |

## Findings that are owner decisions, not author fixes

These cannot be dispositioned by revision alone; they need a ruling, and a
ruling on scope or the selected alternative returns the run to G1:

- **C1** — which construction: splice historical + scenario for 2015–2020 (and
  which scenario fills the gap, per-SSP?), or clip the reference to ≤2014 and
  restate G3 as a check? Interacts with OQ-4 (30-year default).
- **ext1-09 / OQ-8** — retire `save_grids` with a characterized loss, or specify
  an optional gridded branch?
- **ext1-07 / OQ-6** — ensemble sampling unit and weighting.
- **ext1-06** — narrow v2's claim to monthly basin-series projection analysis,
  or generalize the source registry now to carry observed-source and
  temporal-resolution axes?


---

# Internal lens review — risk (Fable), verbatim

## Verdict
verdict: revise
doc_version: design-v1.md

## Findings
### risk-01  [blocking]
- section: §5.3 Region and baseline / §8 step 5
- finding: Aligning the projection reference window to `shared.historical_window` (2000-01-01 … 2020-12-31) is not implementable as specified, because the change-factor reference is the **GCM historical** series and the CMIP6 historical experiment ends in 2014. The catalog confirms this: `cmip6_{model}_historical_{member}` resolves under `gs://cmip6/CMIP6/CMIP/{model}/historical/…` (historical experiment, ≤2014), while 2015+ lives only in the per-scenario `ScenarioMIP` entries. Today's `historical_year_range: [1990, 2010]` fits inside the historical experiment; the replacement window does not. The design never says what the reducer does when the requested window overruns the source's time axis.
- rationale: Whatever the unstated behavior is, it breaks the design's own claim. If the reduce silently truncates, the GCM reference becomes 2000–2014 while the design labels it 2000–2020 — G3 ("reference window equals the baseline window by construction… cannot recur") is false for exactly the series that define every change factor, and the window mismatch the design set out to eliminate recurs invisibly, now with wrong provenance metadata. If it errors, step 5 cannot ship. Splicing historical + scenario for 2015–2020 is the standard fix but is scenario-dependent (which SSP fills the gap?) and is a method decision the document must make explicitly, not an implementation detail. The 30-year default (OQ-4) makes the same boundary arithmetic bite at the 2100 end for late horizons (a 30-year window centered on a 2085 horizon needs data past 2100).
- suggested_fix: Specify the reference-series construction explicitly: either (a) reference = historical experiment spliced with the matching scenario for years > 2014 (the common delta-method convention, one spliced series per scenario), or (b) reference window is clipped to the historical experiment with the effective window recorded in provenance and G3 restated as an alignment *check*, not by-construction. Add a boundary rule for horizon windows that overrun the scenario end. Gate step 5 on a unit test that requests a window crossing 2014 and asserts the documented behavior.

### risk-02  [major]
- section: §5.2 The unified series store (Caching) / §7 consequence 2
- finding: The caching contract is underspecified on exactly the axis its headline benefit depends on. The digest includes `window`, but the design never says what `window` means for a future series. On the natural implementation — reduce extracts only the years the configured horizons need — adding or changing a `future_horizons` entry changes the digest, the params rerun-trigger fires, and the series re-derives over the network, making falsifiable consequence 2 ("changed `future_horizons` performs zero network reads") false. Separately, the design does not state whether the catalog file is a declared input of `reduce_climate_series`: if it is, Snakemake's mtime trigger re-derives every series on any catalog edit (again defeating G5); if it is not, the design departs from C1's "the catalog file is the freshness boundary" philosophy for the store rule and should say so and why.
- rationale: G5/G6 are the core operational payoff of the restructure. As written, two of the most common user actions (edit a horizon, touch the catalog) plausibly trigger the full re-download the design promises to eliminate — and nothing in §9 would catch it, because the step-2 cache test only exercises "touch nothing" and "change one digest component."
- suggested_fix: Pin the contract: reduce extracts a fixed span per provenance class (obs/hist: the historical window per risk-01's resolution; future: the full scenario span 2015–2100), independent of `future_horizons`; all window selection happens in stage B. State that the catalog file is *not* an input of reduce — freshness for series is carried solely by the parsed-entry digest — and record that as a deliberate C1 divergence. Add a §9 cache test: change `future_horizons`, assert zero reduce jobs.

### risk-03  [major]
- section: §5.2 (Caching — `reducer_version`)
- finding: Cache invalidation on reducer-semantics changes rests on a hand-bumped `reducer_version` constant. Snakemake's code rerun-trigger tracks the rule's script body, not the `blueearth_cst` modules it imports, so a change to the reduction logic (e.g., the area-weighting implementation) with a forgotten version bump reuses every cached series silently.
- rationale: This is a silent-wrong-numbers path with no failure signal: derive and report run happily on stale series, and the baseline gate cannot catch it because it only runs during migration, not in steady-state use. The failure mode is precisely the one the persistent cache introduces and the design does not name it in §7.
- suggested_fix: Derive the version component mechanically — hash the reducer module source (or the relevant function objects' bytecode) at DAG-build time and fold it into the digest — or, minimally, write the digest as an attribute into each series .nc and have stage B assert stored digest == expected digest, so a stale series fails loud instead of merging.

### risk-04  [major]
- section: §8 Migration + commit plan (steps 3–4) / OQ-3
- finding: The claim that steps 1–4 are "checkable against the baseline manifest" conflicts with step 3's "unify output naming." The manifest pins WF2 at its *current* paths — sha256 on `climate_projections/cmip6/summary/annual_change_scalar_stats_summary.csv`, summary stats on the .nc, existence+size on three PNGs. Any rename or move in steps 3–4 makes those targets vanish: `check_baseline.py` either fails on missing paths or is re-recorded from the new run — and a re-record under a rename trivially matches itself, severing the value-neutrality evidence chain the whole tiering scheme (§4 criterion 1) depends on. The layout decision is simultaneously deferred (OQ-3) and consumed (step 3), a sequencing contradiction.
- rationale: A misclassified or unverifiable "value-neutral" step ships a silent numerical change — the exact risk §8 exists to prevent. Note also the manifest's WF2 coverage is thin (one strict CSV, one NC summary, PNG existence): it constrains the final scalar summary but not the monthly intermediates steps 3–4 restructure, so "check_baseline passed" is weaker evidence of neutrality than the plan implies.
- suggested_fix: Constrain steps 1–4 to byte-identical output paths; move all renames/relayout to steps 6–7, after OQ-3 is resolved, with the migration note. If step 3 genuinely requires renaming intermediates, keep the manifest-pinned final artifacts at their old paths through step 4 and gate the neutrality claim on them plus a `semantic_tree_diff` with an explicit old→new path mapping. Consider widening manifest WF2 coverage (per-model monthly stats) before step 3 lands.

### risk-05  [major]
- section: §5.5 Change factors — annual and monthly
- finding: Monthly *relative* precipitation change factors divide by the reference-month climatological precipitation, and the design has no policy for reference months near zero. For any basin with a dry season (not the equatorial seed fixture, but the tool targets arbitrary basins), the 12-month relative pattern contains unbounded or wildly unstable percentages — a well-known delta-method failure the annual product largely avoids and the new monthly product walks straight into.
- rationale: `change_factors/monthly.csv` is a declared product feeding the report figures and the S5 grid-vs-cloud advisory; a +4000 % dry-month artifact distorts the seasonal-pattern figure and the advisory's envelope comparison, and a naive consumer treats it as signal. Nothing in §9's property tests (scale invariance) catches it — relative change is scale-invariant even when it is meaningless.
- suggested_fix: Add an explicit dry-month rule: below a configurable reference threshold (e.g., mm/month), emit the absolute change with a flag column (or NaN with a reason code) instead of a relative factor, and state the rule in the report. Add a synthetic-series test with a near-zero reference month.

### risk-06  [major]
- section: §5.1 / §5.2 (stage B inputs) / §7 R4
- finding: Stage B is described as "reading `series/*.nc`" (diagram and §5.5). If derive's input set is a glob over the persistent series directory rather than the expanded list from the validated config, stale series — a model removed from the config, a renamed member, a key from a previous layout — silently join the ensemble. R4 acknowledges accumulation as a disk-space issue but misses this correctness consequence.
- rationale: Persistent caches plus glob inputs is a classic silent-wrong-ensemble path: the user removes INM-CM5-0 from the config, the report still averages over it, and no gate fires because every file present is valid netCDF. The provenance file would even list the wrong composition as if intended.
- suggested_fix: Specify that `derive_change_factors` inputs are exactly the expanded `{series_key}` list built from the validated config — never a directory glob — and have stage B assert the series it opens match that list. This also makes R4's stale files harmless rather than merely untidy.

### risk-07  [minor]
- section: §5.3 / §8 step 1
- finding: Declaring `extract_climate_grid` puts the full observed-climate extraction (gridded download, the hours-scale store build) on WF2's critical path even for a user who only wants change factors and only needs `store_region.geojson`. C1 forbids a region-only asymmetric declaration, so this cost is structural — but the design nowhere states that a first, model-free WF2 run now pays the store-build cost and inherits its network failure surface, where today rule-graph entry is a cheap read of an existing WF1 artifact.
- rationale: Expectation management: "WF2 no longer needs WF1" reads as a cost reduction; on a fresh project it is a cost transfer. First-run wall clock and a new failure mode (observed-source unreachable blocks a projections-only run) will surprise users.
- suggested_fix: State the cost explicitly in §7 consequences; note that when WF1 has run, the store already exists and the rule is a no-op (same outputs, C1).

### risk-08  [minor]
- section: §5.7 Source resolution at DAG-build time
- finding: The DAG-build validator re-implements hydromt's templated-entry + `placeholders:` resolution as a raw YAML lookup. That is acceptable under N5 (reading, not patching), but it creates a parallel implementation that must track hydromt's catalog semantics (variants, aliases, future format changes); drift produces either false rejections of valid sources or jobs that fail at read time with the validator claiming they exist.
- rationale: The catalog is repo-owned and the format stable, so the risk is modest — but the failure mode is confusing (validator and hydromt disagree) and lands on end users.
- suggested_fix: Keep the validator's logic minimal (exact entry-name template + placeholder membership only), and add one integration-marked test that cross-checks the validator's accept list against hydromt's actual catalog resolution for a representative entry.

### risk-09  [minor]
- section: §5.1 Architecture (job accounting) / §7 consequence 3
- finding: The job arithmetic is internally inconsistent. The listed components — 1 store + 9 reduce + 1 derive + 1 report + 1 config + 1 benchmark gather — sum to 14, not the stated 13; and 9 reduce jobs covers 6 future + 3 GCM-historical but omits the observed series (§5.2 says obs enters stage A like everything else), which would make reduce 10 and the total 15.
- rationale: Consequence 3 is framed as a falsifiable per-commit gate ("`snakemake -n` lists 13 jobs"); a gate with wrong expected numbers either fails spuriously or gets "fixed" to whatever the run produces, defeating its purpose.
- suggested_fix: Recompute the seed-config job count once the obs-series question is settled and state the breakdown; keep the number and the component list consistent.


---

# External review round 1 (gpt-5.6-sol), verbatim

## Verdict
verdict: revise
doc_version: design-v1.md

## Findings
### ext1-01  [blocking]
- section: 5.3 Region and baseline — solved structurally, not by validation; 5.5 Change factors — annual and monthly
- finding: The reference-period contract cannot be realized as written. `shared.historical_window` is 2000–2020, while CMIP6 historical runs end in 2014; the design does not specify joining each historical series to its scenario series for 2015–2020, including gap, overlap, and calendar handling. It also independently proposes a 30-year reference default, although the shared baseline currently spans 21 years.
- rationale: Implementations will either truncate the GCM reference to 2000–2014, fail on missing years, or silently compare unequal periods. Any of these violates G3 and changes the change factors. Making the shared window 30 years instead would also change the shared climate store used by WF1 and WF3, a cross-workflow consequence absent from the migration plan.
- suggested_fix: Define the reference as exactly `shared.historical_window`; specify historical-to-scenario concatenation per `(model, member, scenario)` with coverage, duplicate, gap, and calendar checks; and separate the future-horizon length decision from the reference-window contract. If concatenation is rejected, require the shared reference to end by 2014 and revise G3.

### ext1-02  [blocking]
- section: 5.6 Report stage; 5.7 Source resolution at DAG-build time
- finding: Runtime source-failure handling is incompatible with the proposed DAG. Every resolved source is a required Stage-A netCDF input to Stage B, but a remote read failure is supposed to produce no empty file and instead be recorded later in `provenance.json`.
- rationale: In Snakemake, the failed reducer job or its missing declared output stops the DAG before Stage B or the report can run. Consequently, failed-source provenance, configurable minimum-source enforcement, and continuation with the surviving ensemble cannot work as specified.
- suggested_fix: Choose and document either fail-fast semantics, or a failure-tolerant artifact contract in which every source job emits a required status artifact and successful data are discovered through a checkpoint or manifest. Move provenance/minimum-source validation ahead of Stage B in the tolerant design.

### ext1-03  [blocking]
- section: 5.2 The unified series store — the central idea; 5.4 Variable specification — replacing name-based dispatch; 5.5 Change factors — annual and monthly
- finding: A variable-level `aggregate: sum` cannot uniformly reduce the daily observed inputs and monthly-mean CMIP6 inputs into a comparable precipitation series. Summing daily precipitation produces a monthly accumulation, whereas summing an Amon series with one value per month merely preserves a monthly mean rate; the proposed common `units: mm/day` would therefore label unlike quantities as equivalent.
- rationale: Observed-versus-GCM diagnostics would compare incompatible values, and annual or monthly precipitation products would depend on source frequency rather than climate. The later reference to month-length weighting does not define the conversions needed to repair the Stage-A store.
- suggested_fix: Specify a canonical monthly quantity for each variable—such as mean rate in mm/day or accumulated depth in mm/month—and define source-specific conversion using units, sampling interval, temporal bounds, and calendar before spatial reduction. Reject inputs whose temporal semantics cannot be established and test equivalent daily and monthly synthetic inputs.

### ext1-04  [major]
- section: 5.2 The unified series store — the central idea; 7. Consequences and risks
- finding: G5 conflicts with the reducer cache key. The reducer digest includes `window`, but the document does not distinguish acquisition coverage from the analysis horizons that users may change.
- rationale: If `window` follows `future_horizons`, changing a horizon invalidates the persistent series and repeats the expensive network reads, directly falsifying consequence 2. If it instead means full source coverage, the required coverage and associated cost are undefined.
- suggested_fix: Give Stage A a stable acquisition window independent of reference and future analysis windows—normally the complete required CMIP source span—and make only Stage B depend on `future_horizons`. Record acquisition coverage in each series and fail when a requested analysis window is not fully covered.

### ext1-05  [major]
- section: 5.5 Change factors — annual and monthly
- finding: The change-factor method lacks normative formulas and edge-case policies for relative changes, incomplete hydrological years, missing months, and non-Gregorian calendars.
- rationale: Monthly precipitation reference values can be zero or near zero, producing infinite or unstable percentage changes. Incomplete first or last hydrological years and truncated series can also enter statistics with fewer months, while unspecified calendar weighting makes results differ by model for procedural rather than climatic reasons.
- suggested_fix: State equations and units for every supported variable/statistic; define denominator thresholds and NA/status behavior; require complete hydrological years and minimum coverage; and specify calendar-aware interval weighting. Add tests for dry months, missing months, partial years, leap years, and 360-day calendars.

### ext1-06  [major]
- section: 5.8 Extension slots (named, not built); 10. Open questions
- finding: The claim that every extension slot is “a read, not a pipeline” is false for the proposed store. S2 requires extraction and fan-out for multiple observed products although Stage 0 creates only the single configured `shared.clim_historical` store; S4 requires a new daily acquisition and storage branch; and credible long-term S1 trends may require coverage beyond the project baseline window.
- rationale: Implementing these advertised extensions would change the producer graph, cache schema, configuration, and provenance contracts, so G9 is not delivered by the selected architecture. The widened “general climate analysis” framing would therefore create downstream redesign rather than the promised stable extension surface.
- suggested_fix: Either narrow v2’s claim to monthly basin-series projection analysis, documenting S1/S2/S4 as future architecture changes, or generalize the source registry and series identity now to include observed-source and temporal-resolution axes with independently configured acquisition windows.

### ext1-07  [major]
- section: 5.5 Change factors — annual and monthly
- finding: The ensemble contract does not define the sampling unit after `member` becomes a wildcard. A threshold of 10 and envelopes “across models” are ambiguous when models contribute different numbers of members; institution counts alone do not prevent pseudoreplication.
- rationale: Adding members from one model could give that model disproportionate influence and could trigger percentile envelopes without adding independent model diversity. Reported uncertainty would then change because of configuration multiplicity rather than a broader ensemble.
- suggested_fix: Resolve OQ-6 before specifying ensemble summaries. Define thresholds using unique models, show members hierarchically, and either average members within each model before equal-model summaries or document another explicit weighting rule. Until then, emit individual model/member traces without an aggregate envelope.

### ext1-08  [major]
- section: 5.2 The unified series store — the central idea
- finding: A cosine-latitude weight is not generally a cell-area weight for the catalog’s heterogeneous CMIP grids. It is valid only under restrictive rectilinear, regularly spaced coordinate assumptions that the design neither checks nor records.
- rationale: On Gaussian, irregular, or curvilinear grids, model-to-model differences can partly reflect grid geometry. Because the catalog currently drops coordinate bounds, the proposed reducer may be unable to establish correct areas while still claiming area-weighted results.
- suggested_fix: Define supported grid geometries and compute areas from retained bounds or derived cell edges, including longitude wrapping and missing-cell treatment. Validate assumptions and fail or explicitly label an approximation when exact areas cannot be established.

### ext1-09  [major]
- section: 5.1 Architecture — three stages, fan-out only where it pays; 8. Migration + commit plan; 10. Open questions
- finding: OQ-8 must be resolved before this architecture can be implemented: Stage A discards spatial dimensions, so Stage B cannot reproduce the existing `save_grids` products.
- rationale: Preserving `save_grids` requires an additional gridded artifact path and declared optional rules; retiring it is a breaking behavior change requiring migration and acceptance coverage. Leaving the choice open makes the rule graph, output contract, job count, and step-4 value-neutrality claim indeterminate.
- suggested_fix: Either explicitly retire `save_grids` with a migration note and characterized loss of functionality, or specify a separate optional gridded branch with declared outputs, cache behavior, and validation.

### ext1-10  [minor]
- section: 5.1 Architecture — three stages, fan-out only where it pays; 7. Consequences and risks
- finding: The seed job-count prediction is arithmetically inconsistent. Three models require three historical GCM series, six future model-scenario series, and one observed series: ten reducer jobs, not nine. With store, derive, report, config-copy, and benchmark-gather jobs, the stated architecture totals 15 jobs rather than 13.
- rationale: The falsifiable 13-job consequence and its validation gate will fail even if the implementation matches the architecture.
- suggested_fix: Derive expected counts from the resolved source manifest in tests instead of hard-coding 13, and update the design’s illustrative count.

---

# Round-2 index

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


---

# External review round 2 (gpt-5.6-sol), verbatim

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-01  [blocking]
- section: 5.3 The GCM series store — product, identity, caching
- finding: The cache incorrectly assumes that the region configuration fully determines `store_region.geojson`. The polygon also depends on the delineation catalog, its underlying data, and producer code. Yet the region input is marked `ancient()` and the series digest contains only configuration parameters, not the polygon’s content or its producing inputs.
- rationale: After a relevant catalog or delineation change rewrites the polygon, existing series remain eligible for reuse and stage B recomputes the same expected digest. It can therefore accept basin averages calculated for the old polygon, producing wrong change factors; recording the old bounds only makes the defect auditable after the fact.
- suggested_fix: Use `store_region.geojson` as an ordinary input, or introduce a content fingerprint that participates in both Snakemake invalidation and `cst_series_digest`. Stage B must also verify the current polygon fingerprint against every series.

### ext2-02  [blocking]
- section: 5.3 The GCM series store — Spatial reduction
- finding: The proposed geometry check does not make cosine-latitude weighting valid. Strictly monotonic 1-D coordinates can still be irregularly spaced or Gaussian; their cells require latitude and longitude widths in addition to `cos(latitude)`. Such grids currently pass the check even though OQ-10 explicitly acknowledges non-uniform spacing.
- rationale: Accepted non-uniform grids receive incorrect spatial weights and hence wrong basin means and change factors. This means the resolution of round-one finding `ext1-08` is incomplete, with exposure increased by the expanded generated catalog.
- suggested_fix: Either reject grids whose latitude and longitude spacing is not sufficiently uniform, or derive approximate cell edges and weight by spherical cell area. Retaining bounds and using bounds-derived areas is preferable. Add a non-uniform rectilinear test that either verifies correct areas or verifies fail-fast refusal.

### ext2-03  [blocking]
- section: 5.8 The optional gridded branch
- finding: The gridded change-field contract is under-specified. Cellwise changes require the historical and scenario grids to be compatible, but the design defines neither coordinate/CRS equality requirements nor regridding behavior. It also provides no schema for whether `grids/change/*.nc` contains annual changes, monthly changes, which statistics, or how dry-reference statuses and absolute fallbacks are represented.
- rationale: Historical and scenario publications may use different grid labels or coordinates. Implicit xarray alignment can produce empty, sparse, or mismatched fields, while alternative implementations could emit incompatible products. Ruling R2’s declared gridded output therefore cannot be implemented reliably as specified.
- suggested_fix: Define the complete gridded-change schema and require exact CRS and coordinate compatibility before cellwise arithmetic, failing fast when it is absent. If differing grids must be supported, specify one existing-dependency regridding method. Add shifted-grid, mismatched-CRS, monthly/annual, and dry-cell tests.

### ext2-04  [major]
- section: 5.3 The GCM series store — Cache key; D8 — time-axis uniqueness
- finding: The value called the “resolved URI” is not a physical source identity: substituting `{member}` still leaves `{variable}/*/*`, including wildcard grid label and publication version. The digest also excludes read-relevant metadata such as `metadata.crs`.
- rationale: A newly published version under the same glob can be read by a fresh project while an existing cache silently retains the prior publication under an unchanged digest. Provenance cannot identify which physical zarr stores supplied the values, and metadata corrections can alter interpretation without invalidation. D8 detects duplicate timestamps only when a source is reread; it does not repair cache identity.
- suggested_fix: Close OQ-14 before implementation. Have the generator pin and record the exact physical zarr path selected for each variable, and include those paths plus all read-affecting metadata in the digest and provenance.

### ext2-05  [major]
- section: 5.4 Region and reference window — The reference window length
- finding: The asserted 30-year 1985–2014 reference conflicts with the complete-hydrological-year policy. Whenever `start_month_hyd_year` is not January, this calendar window contains only 29 complete hydrological years; the partial years at both ends are dropped.
- rationale: Annual and potentially monthly statistics would use fewer years and a different effective period than the owner-approved “30 years, 1985–2014,” while the acceptance test checks only that warnings do not fire. Reported sample length and scientific interpretation can therefore be misleading.
- suggested_fix: Define whether the ruling denotes 30 calendar years or 30 complete hydrological years. Then either use calendar years for WF2 or construct exactly 30 hydrological years and report their actual date bounds. Add a non-January acceptance test asserting `n_years`, effective dates, and dropped months.

### ext2-06  [major]
- section: 5.6 Change factors — Dry-month / near-zero denominator rule
- finding: The dry-reference policy remains scientifically incomplete because `relative_change.min_reference` is outcome-determining but its default is still OQ-9. The contract likewise does not settle whether thresholds and `relative_change.max_flagged_months` are required or defaulted.
- rationale: Implementers can produce different `value`, `status`, and report-warning outputs from identical inputs by selecting different undocumented thresholds. Shipped configurations and boundary tests cannot be finalized, so the claimed resolution of `risk-05`/`ext1-05` is not yet complete.
- suggested_fix: Before implementation, either choose and justify explicit per-variable defaults or make both thresholds required and populate every shipped configuration. Add tests immediately below, at, and above each threshold.

### ext2-07  [major]
- section: 5.5 Variable specification — declaring the quantity, not the aggregator
- finding: The configurable variable contract is broader than the catalog’s availability contract. Resolution certifies only `pr` and `tas`, yet a requested `kin` or `press_msl` combination is marked resolved and converted into a job even when the corresponding store is predictably absent.
- rationale: A large run can spend hours completing other network jobs before halting on a missing configured variable. `composition.csv` will classify the combination as resolved even though the generated snapshot lacked the requested input, undermining the design’s central separation between “not published” and “failed to read.”
- suggested_fix: For v2.0, either reject variables other than `precip` and `temp` at DAG build, or make the generator publish per-variable member availability and resolve against all requested variables. Do not represent catalog-known absence as a runtime read failure.

### ext2-08  [minor]
- section: 5.7 Source resolution and failure semantics — D4
- finding: The statement that the resolved combination set “is now written down in `composition.csv` before any job runs” is impossible under the specified DAG: `composition.csv` is a stage-B output and stage B cannot run until every required reducer succeeds.
- rationale: A failed run has the DAG-build stderr summary and logs, but no composition artifact. This contradicts the round-one `ext1-02` disposition, which correctly limited provenance to successful runs.
- suggested_fix: State explicitly that `composition.csv` describes completed runs. If a durable pre-execution resolution manifest is required, specify it as a separate earlier artifact rather than assigning that behavior to stage B.

### ext2-09  [minor]
- section: 9. Validation plan — No aggregation
- finding: The assertion that no row may equal the mean of other rows is not a valid no-aggregation invariant; a legitimate member value can coincide numerically with such a mean.
- rationale: The test can reject a correct implementation by coincidence and can still miss aggregation if synthetic values are poorly chosen.
- suggested_fix: Assert tuple cardinality, unique keys, direct equality to independently computed per-series results, and absence of cross-combination reduction operations. If using sentinel values, construct them so no aggregate can equal an original value.

---

# Findings ledger (final)

# Findings ledger — wf2-climate-analysis-v2

Append-only. One row per **original finding ID**, dispositioned at its **own
filed severity**. Convergent pairs (risk-01 / ext1-01, risk-02 / ext1-04,
risk-05 / ext1-05, risk-09 / ext1-10) each keep their own row; they are never
merged. Severities are never re-graded.

Finding texts: `internal-review-risk.md` (risk-*), `external-review-r1.md`
(ext1-*). Aggregation: `review-index.md`. Owner rulings R1–R5: `status.md`.

**Round 1 totals as first recorded — 19 findings: 19 accepted, 0 rejected,
0 deferred, 0 withdrawn.** ~~No rejected major, so no G2 ratification is required
on this round.~~ **Superseded by the Round 2 amendment below:** `ext1-07` is now
**partially rejected**, so the round-1 totals read **18 accepted, 1 partially
accepted / partially rejected**, and a rejected `major` does owe ratification.
That ratification is recorded, not outstanding — see below.

Findings dissolved by an owner ruling are recorded `accepted` with the ruling
named — not `withdrawn` — because the ruling is a design change, not a
reviewer retraction.

---

## Round 2 (owner rulings) — 2026-07-29

No new review round ran. Owner rulings **R3′**, **R3″** and **R5** arrived at
`G1-return-2`, after `design-v2.md` had landed, and they change the design that
three round-1 rows were dispositioned against. Two things are recorded here so
the append-only table below stays readable:

**1. `ext1-07`'s disposition is amended in place, and only its disposition.** Its
row's `Resolution or rationale` text is kept verbatim as the record of what
design-v2 did, with an `AMENDED` paragraph appended — the original text is not
rewritten, because it remains the accurate account of the version it describes.
The row's `Doc version` stays `design-v2.md`; the amendment names `design-v3.md`
as where the corrected disposition is realised.

- **New disposition: partially accepted / partially rejected by owner ruling.**
  ext1-07 filed two things: (a) the ensemble sampling unit is undefined when
  `member` is a wildcard, and (b) the fix is to average members within each model
  before summarising. Half (a) is **accepted** — the unit is now defined, as *no
  unit at all*: each (model, scenario, member) is a data point and nothing is
  summarised across data points. Half (b) is **rejected**: **R3′** rules that
  members are never averaged, at any level.
- **The ratification a rejected `major` requires is R3′/R3″ themselves.** The
  skill's rule is that a rejected `major` needs owner ratification at G2. Here
  the rejection *originates* with the owner rather than with the author, so the
  ruling text in `status.md` — R3′ ("Members are **never averaged**") and R3″
  (cross-combination statistics are ex-post) — **is** that ratification, recorded
  at `G1-return-2` on 2026-07-29. Nothing on this point is owed at G2.

**2. Three round-1 rows are extended, not re-dispositioned** (`risk-02`,
`risk-08`, `ext1-09`). Their accepted resolutions still hold; what changed is the
repository underneath them — the CMIP6 catalog became generated (`f8194e8`) and
R5 promoted the series to a deliverable. Their extensions are new rows below,
keyed `R2-*`, so no round-1 row is edited to describe a version it predates.
Severities are not re-graded and no round-1 disposition other than `ext1-07`'s
changes.

**Round 2 rows** carry `Round = 2 (ruling)` and no reviewer severity, because
they originate from an owner ruling rather than a review finding.

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| risk-01 | 1 | blocking | accepted | **Owner ruling R1 — clip, never splice.** Landed in §5.4 **D1**: effective reference = `requested ∩ [source start, 2014-12-31]`; three surfacing sites (DAG-build stderr, `provenance.json`, `report.md` disclaimer) assigned **per condition** in a table — clip and short-window warn on stderr, the alignment difference does not by default, because the shipped seed config differs and an always-firing warning is filtered out; a reference lying entirely after 2014 raises; a horizon boundary rule clips against 2100-12-31 the same way. §2 **G3** restated as an alignment *check*, not equality by construction. `historical_year_range` is retained, not retired (§5.4, and §8 step 5e). The 15-year effective reference implied by a 2000–2020 request is stated explicitly in §5.4 as the visible surface of the accepted tradeoff. §6.5 records why splicing was rejected (scenario-dependent reference, per-SSP baselines, overlap/gap/calendar reconciliation) and N8 makes it a non-goal. §9 adds the three reference-window tests. | design-v2.md |
| risk-02 | 1 | major | accepted | §5.3 "Acquisition window" and "Cache key". Acquisition span is **fixed per experiment class** (`historical` 1950-01-01…2014-12-31; any `sspNNN` 2015-01-01…2100-12-31 — the spans `get_stats_climate_proj.py` already hardcodes as `time_tuple_all`), independent of `future_horizons`, which is **excluded from the digest**; all analysis-window selection moves to stage B. The catalog **file** is deliberately **not** a declared input of `reduce_gcm_series` — the parsed **entry** enters through the digest — recorded as a bounded, reasoned divergence from C1's file-level freshness boundary (which `extract_climate_grid` keeps verbatim). §7 consequences 3–4; §9 cache tests (b), (c), (d). | design-v2.md |
| risk-03 | 1 | major | accepted | §5.3 "Reducer-version staleness". `reducer_version` is no longer a hand-bumped constant. Two stdlib mechanisms: (1) the Snakefile hashes an **explicitly enumerated** list of reducer module files with `hashlib.sha256` at DAG-build time and folds it into the digest — enumerated, not all of `blueearth_cst`, so unrelated edits do not invalidate; (2) each series carries `cst_series_digest` as an attribute and **stage B raises on mismatch**, naming the series and both digests, so a series that survived a mechanism-1 miss fails loud instead of merging. §7 consequences 5–6; §9 cache tests (e), (f). | design-v2.md |
| risk-04 | 1 | major | accepted | §8 opens with **"What the baseline manifest actually pins"** — the exact 7 targets and comparators. **Driver note:** `status.md`'s premise-verification line says all 7 are under `climate_projections/cmip6/`; in fact **6** are, and the 7th is `config/runs/snake_config_climate_projections.yml`. §8 records the accurate breakdown. Three consequences drawn: coverage is thin (no monthly intermediates, so a green `check_baseline` constrains the final annual scalar summary and three PNG sizes only); the config snapshot is a **verbatim sha256 of the seed config file** (`copy_config_files.py`), so a config-key addition breaks it even when no number moves; and a rename severs the evidence chain by self-matching on re-record. §4 criterion **1** is rewritten to separate **value-neutral** from **manifest-clean**. The sequencing contradiction is removed at the root: **D3** (§6.10) closes OQ-3 by keeping `climate_projections/{clim_project}/`, so **no step renames a manifest-pinned path**; step 3's intermediate renames get an explicit old→new map for `semantic_tree_diff`. §7 R10. | design-v2.md |
| risk-05 | 1 | major | accepted | §5.6 "Dry-month / near-zero denominator rule". Config key `relative_change.min_reference` per variable in canonical units; below it, `value = NaN`, `status = "reference_below_threshold"`, and the **absolute** change in an `absolute_value` column so information is not lost; report renders gaps with a footnote naming rule and threshold; the S5 advisory excludes flagged months from envelope comparison; a `(dataset, scenario, member, horizon, variable)` exceeding `relative_change.max_flagged_months` is flagged in the summary. The rule is complete without the numeric default; only the number is open (**OQ-9**, with the evidence that settles it). §7 consequence 11; §9 near-zero synthetic test. | design-v2.md |
| risk-06 | 1 | major | accepted | §5.3 "Stage B's input set is explicit". `derive_change_factors` declares **exactly the expanded `{series_key}` list built from the validated config — never a directory glob** — and asserts the set of series it opened equals that list. §7 R4 restated: stale series are disk hygiene, no longer a correctness path. Reinforced by risk-03's digest assertion, which fails a stale file even if it were reachable. | design-v2.md |
| risk-07 | 1 | minor | accepted | Filed minor; made **load-bearing by R4** (with observed analysis out of scope, declaring `extract_climate_grid` buys only the polygon). Severity unchanged. Landed as a **named decision D2** (§5.4) with four real alternatives enumerated in §6.4: **A1 selected** — declare the full store, cost stated (the seed fixture's `era5_20000101_20201231/extract_historical.nc`, 7 variables, daily, 7671 steps; no wall-clock figure asserted), with the four reasons it is accepted (G2 is the change request's most visible gain; no-op once WF1 has run, hence 15 fresh / 14 with store; the cost is confined to the store rule and off the analysis path; the store is the precondition for the deferred observed work). **A2 named as the fallback** and the **discriminating question surfaced for the owner**. **A3** (region-only producer) rejected on C1's *purpose* — a second delineation path — while conceding it does not violate C1's letter. §7 R7. | design-v2.md |
| risk-08 | 1 | minor | accepted | §5.7 "Keeping the validator from drifting". Validator logic stays **minimal** (exact entry-name template + placeholder membership only); an entry carrying a construct the validator does not model — variants, aliases, any unrecognized top-level key — is an **error naming the key**, so drift becomes visible rather than wrong; one integration-marked test cross-checks the accept list against `hydromt.DataCatalog(...).sources`. §7 R9; §9 validator tests. | design-v2.md |
| risk-09 | 1 | minor | accepted | §5.2 "Job accounting" recomputed for the **narrowed** R4 scope, with the counting convention stated (excludes the `all` target job — how v1's 22 was counted) and today's 22 re-derived component-by-component. New total **15** on a fresh `project_dir`, **14** when the store already exists: 1 store + 3 GCM-historical reduce + 6 GCM-scenario reduce + 1 derive + 1 report + 1 config + 1 log gather + 1 benchmark gather. The reduce count is stated as 9 with the reason (R4/N7 removes the observed series, so it is `3 + 6`, not `3 + 6 + 1`). §7 consequence 2; §9 derives the expected count from the resolved source manifest rather than a literal. | design-v2.md |
| ext1-01 | 1 | blocking | accepted | Convergent with risk-01; same resolution, recorded separately. **Owner ruling R1** takes ext1-01's own fallback option ("if concatenation is rejected, require the shared reference to end by 2014 and revise G3"). §5.4 D1 + §2 G3 + §6.5 + N8 + §9 tests. The finding's second half — separating the future-horizon length decision from the reference-window contract — is honoured: the acquisition contract (§5.3) is independent of `future_horizons`, the horizon boundary rule is stated separately from the reference clip, and the window-length default stays **OQ-4**. The cross-workflow consequence it warned about (a 30-year shared window changing the WF1/WF3 store) does not arise, because `historical_year_range` is retained and `shared.historical_window` is not modified. | design-v2.md |
| ext1-02 | 1 | blocking | accepted | §5.7 **D4 — fail-fast**, chosen and justified. A reduce job that cannot read its source raises and Snakemake halts; no dummy netCDF, no empty dataset, no silent ensemble shrink; the **minimum-source check moves to DAG build**, the only place it can execute (criterion 4.5, new); `provenance.json` describes a successful run's composition rather than a failure ledger. Justification against the tolerant contract in §6.8 — it adds a checkpoint and a second artifact class to a workflow being simplified 11→8 rules, and it makes ensemble composition depend on transient network state (two identical configs, different numbers, no error); `--keep-going`, already used by `run_workflows.py`, gives the see-all-failures benefit without changing the contract; the persistent cache makes retry cheap. Recorded as revisitable with settling evidence in **OQ-11**. §7 consequence 8, R5; §9 fail-fast and min-sources tests. | design-v2.md |
| ext1-03 | 1 | blocking | accepted | **Dissolved for v2.0 by owner ruling R4** — no observed comparison at this stage, and §5.1 / N7 state explicitly that **`extract_historical.nc` is not reduced in v2.0**, so no stage-A rule mixes daily observed with monthly `Amon` input. Not merely dissolved, though: the spec defect it exposed is fixed. §5.5 replaces v1's `aggregate: sum|mean` — the exact axis ext1-03 showed cannot describe two source frequencies — with a **`canonical:` quantity declaration** (`rate` in mm/day, `state` in degC), making frequency→canonical conversion a property of the **source**, not the variable. In v2.0 that conversion is the identity and is **asserted, not inferred**: the reducer checks the decoded time axis is monthly and raises otherwise. §5.10 S1 names that assertion point as where a future daily/observed branch attaches. | design-v2.md |
| ext1-04 | 1 | major | accepted | Convergent with risk-02; same acquisition/digest resolution, recorded separately. ext1-04's distinct second half — "record acquisition coverage in each series and fail when a requested analysis window is not fully covered" — is landed verbatim in §5.3: each series records `cst_acquisition_window` plus actual first/last time step, and **stage B fails** naming the series and both windows when a requested analysis window is not covered. §9 adds the coverage-assertion test. | design-v2.md |
| ext1-05 | 1 | major | accepted | Broader than risk-05, and accepted in full rather than trimmed to the dry-month case. §5.6 now carries: **normative formulas** with units for both `change:` classes at annual and monthly resolution; the dry-month denominator policy with NA + status code + absolute fallback; **complete hydrological years only** (partial first/last year dropped and counted, keyed on `start_month_hyd_year`); **full monthly coverage required** in the effective window, failing loud on a gap; and **calendar-aware month-length weighting** from the decoded `cftime` axis (`decode_times: true` in the catalog), with the calendar recorded per series and stage B raising on a calendar it cannot weight. §9 adds tests for dry months, missing months, partial years, leap years, and 360-day vs standard calendars. No residual is deferred. | design-v2.md |
| ext1-06 | 1 | major | accepted | **Owner ruling R4 takes ext1-06's own narrow option** ("narrow v2's claim to monthly basin-series projection analysis, documenting S1/S2/S4 as future architecture changes"). §2 **G9** restated from "a named slot, not a rewrite" to "documented with the contract change each extension requires". §5.10 rewrites the slot table with a *contract change required* column: S1 needs a source-level frequency conversion plus a `provenance` axis in the series key and cache; S2 needs more than one store instance, i.e. a C1-scope change or a WF2-private acquisition rule; S3 needs S1 plus resolution reconciliation; S4 needs a daily acquisition branch, a temporal-resolution axis, and probably a new dependency (OQ-7); **only S5 is genuinely a read**, and that is stated. | design-v2.md |
| ext1-07 | 1 | major | **partially accepted / partially rejected by owner ruling** *(amended 2026-07-29; originally `accepted`)* | **Owner ruling R3**, which adopts ext1-07's suggested fix. §5.6 "Ensemble treatment": sampling unit is the **unique model** (`dataset`); **members averaged within a model first**, equal weight per member, then models weighted equally, so adding members cannot give one model disproportionate influence; **members shown hierarchically** and never collapsed away (model primary in every figure/table, every member value in the CSV); **no aggregate envelope** — no percentile band, no ±σ — below `ensemble.min_models_for_envelope` unique models (default 10, carried from v1), only individual traces plus a labelled min–max range. Composition (unique models, members per model, institutions, count used per summary) is reported. Institution de-duplication and performance weighting explicitly **not applied** — recorded as **N9**, not left as an open question. §7 consequence 12; §9 ensemble test. **— AMENDED 2026-07-29 (see "Round 2 (owner rulings)" above).** The text above is retained verbatim as the accurate record of `design-v2.md`; it is no longer the design. **R3 is superseded by R3′** (members are never averaged, at any level) and **R3″** (cross-combination statistics are ex-post). Half (a) of the finding — the sampling unit is undefined when `member` is a wildcard — stays **accepted**: `design-v3.md` §2 **N10** and §5.6 define it as *no sampling unit at all*, because each (model, scenario, member) is a data point and nothing is summarised across data points. Half (b) — "average members within each model" — is **rejected**; `design-v3.md` deletes within-model averaging, the unique-model sampling unit, `ensemble.min_models_for_envelope`, the envelope-suppression rule, the min–max range, and the `ensemble:` block. **The ratification a rejected `major` owes is R3′/R3″ themselves**, recorded in `status.md` at `G1-return-2` on 2026-07-29 — the rejection originates with the owner, not the author, so nothing is outstanding at G2. **N9 is restated** from "not applied" to "downstream concern" (N10), and `composition.csv` (§5.7) is the artifact that lets a downstream consumer apply de-duplication or weighting itself. Neither deleted key ever shipped in a `config/workflows/*.yml`, so the deletion costs no config change, no seed-config sha256 change, and no manifest re-record. design-v2 §9's ensemble test is replaced by design-v3 §9's **no-aggregation** test. | design-v2.md (amended for design-v3.md) |
| ext1-08 | 1 | major | accepted | §5.3 "Spatial reduction". The claim is downgraded from "area-weighted" to **cos-latitude weighting explicitly labelled an approximation valid for 1-D rectilinear monotonically-spaced lat/lon grids**, with the reviewer's premise verified in-repo: `config/catalogs/cmip6_data.yml` sets `drop_variables: [time_bnds, lat_bnds, lon_bnds, bnds]` on every CMIP6 entry, so cell edges are not retained and exact areas cannot be derived from what the reducer receives. The reducer **checks** that lat/lon are 1-D and strictly monotonic and **raises naming the source** otherwise, rather than silently mis-weighting; `cst_weighting_scheme` and the geometry-check result are recorded on the series, in `provenance.json`, and in the report. The design does not assert which catalog models are rectilinear — the check-and-fail contract does not require knowing. §7 R8 records the new failure mode honestly. **OQ-10** carries the "retain bounds and compute true areas" option with the measurement that would settle it. §7 consequence 16; §9 grid-geometry tests. | design-v2.md |
| ext1-09 | 1 | major | accepted | **Owner ruling R2 takes ext1-09's second option** — a separate optional gridded branch with declared outputs, cache behavior, and validation, rather than retirement. §5.8: `save_grids` retained, **default `false`**; the three currently params-passed file families (`historical_stats_{model}.nc`, `stats-{model}_{scenario}.nc`, `monthly_change_mean_grid-*.nc`) are named and become declared `grids/*.nc` outputs; the branch is a **parse-time** output-list extension, so the DAG is fully determined before any job runs and no checkpoint is needed; grids are written from the **same network read** in the same stage-A/B jobs, so `save_grids: true` **adds no jobs** (job count stays 15/14) and no network access; `save_grids` is excluded from the digest with the reason. ext1-09's actual mechanism point is honoured — stage A discards spatial dims, so the grids are produced where those dims still exist, not reconstructed in stage B. Grids are an archive, read by no v2.0 product. §7 consequence 15; §9 gridded-branch dry-run test. | design-v2.md |
| ext1-10 | 1 | minor | accepted | Convergent with risk-09; same recomputation, recorded separately. ext1-10's specific arithmetic ("ten reducer jobs, not nine … 15 rather than 13") was correct **against v1's scope**; under R4 the observed series is removed, so reduce is 9 and the total is 15 for a different reason. §5.2 states this explicitly so the row does not read as ignoring the finding's "should be 10" half. ext1-10's suggested fix is adopted directly: **§9 derives expected counts from the resolved source manifest in tests instead of hard-coding**, with §5.2's 15/14 illustrative for the seed config only. | design-v2.md |
| R2-R3′ | 2 (ruling) | — | ruling implemented | **Owner ruling R3′ — no aggregation at any level.** `design-v3.md` §2 **N10** states the rule and writes the per-series / cross-combination boundary explicitly so the two senses of "statistic" cannot be conflated: statistics computed on the annual series *within* one (model, scenario, member, horizon) stay in scope — they are what the change factor *is* (§5.6) — while statistics computed *across* tuples are out. §5.6's ensemble section becomes a **deletion**, not a rewrite. §5.7 specifies `members:` as *requested ∩ published per (model, scenario)*, unioned across combinations, with a total **resolution ladder** in which a missing SSP, a missing member, and an unpairable reference are **normal skips** recorded in `composition.csv` — never errors. **D7** sets strict same-member pairing and records openly that it converts today's `asymmetric hist/clim members` raise (guard t260720d / D-MEM) into a recorded skip: the guard's purpose (no silent shrink) is served by the composition record, the forcing function is given up deliberately, and the inventory's 18 affected (model, scenario) pairs are named. §6.11 records the rejected pairing alternatives — a designated fallback member (conflates a forcing/physics-variant difference with the scenario response) and a model-historical mean (barred by N10). §7 consequences 8, 9, 14; new risk **R12** records the residual "quiet skips" cost as the accepted price of R3′. §9 gains resolution-ladder, pairing, composition-record and no-aggregation tests. Model similarity and correlation interpretation are out of scope (N10). | design-v3.md |
| R2-R3″ | 2 (ruling) | — | ruling implemented | **Owner ruling R3″ — cross-combination statistics are ex-post.** `ensemble.min_models_for_envelope`, the envelope-suppression rule, the min–max range and the whole `ensemble:` block are **deleted** from the spec; `ensemble.min_sources` goes with them under **D6**, replaced by a non-configurable "zero resolved combinations raises at DAG build" rule — that key conflated *absence* with *failure*, and §4 gains **criterion 7** stating why the two need different machinery. **Neither key ever shipped**: neither appears in any `config/workflows/*.yml`, so §8 carries **no commit row and no manifest re-record** for their removal, and §8 says so explicitly so the migration does not carry a phantom re-record. **N9 restated** from "not applied" to "downstream concern". §5.10 gains slot **S6** (ex-post ensemble statistics) as their home, and **S5 defers with them** — a grid-vs-cloud advisory compares the cloud's extent against the perturbation grid, so it is itself a cross-combination statistic; §5.10 states the honest consequence, that the two zero-contract-change slots are exactly the two now deferred, and that this is coherent rather than coincidental. Report figures become one point or trace per combination. §5.9 records the measured impact: the two manifest-pinned anomaly PNGs are exactly `plot_proj_timeseries.py`'s multi-model 5/50/95 `fill_between` bands, so they change content at the same paths (step 6c, re-record), while the three pinned `summary/*` targets are **unaffected** — verified in `get_change_climate_proj_summary.py`, they merge *per-series* statistics across combinations (`ds.sel(stats="mean")` selects a per-series statistic) and carry no cross-model reduction. §6.13 records why the rejection is coherent and not merely instructed: any such statistic is a function of which combinations happened to resolve, so a re-crawled catalog could move an envelope with no code or config change. | design-v3.md |
| R2-R5 | 2 (ruling) | — | ruling implemented | **Owner ruling R5 — the monthly series is a deliverable.** §2 gains a **declared-output contract** naming all four deliverables plus report/provenance and the legacy summary. §5.3 promotes `series/{series_key}.nc` from cache to product with a full **schema** (dims, scalar coords, variable attributes, `cst_*` global attributes including `cst_schema_version`), a **stable naming rule** (a grammar change is a schema-version bump, which stage B rejects fail-loud), and a **retention rule** (persistent, never `temp()`, never auto-pruned; correctness carried by the explicit input list plus the digest, so a stale file cannot enter a product). §9 adds a schema test that opens the file with plain xarray and no WF2 code on the path. **The gridded ask is reconciled explicitly, taking the driver's flagged reading:** R5 asks for the monthly series *on the source grid* (`time × lat × lon`, retained before spatial reduction), which is **not** design-v2 §5.8's carried-forward 12-month climatology grids. §5.8 therefore **supersedes** both climatology families (`historical_stats_*`, `stats-*`) with `grids/series/{series_key}.nc` — a strict superset, since the climatology is a `groupby("time.month").mean()` of it, and neither is manifest-pinned nor read by any v2.0 rule — and **retains** the change grid as `grids/change/{series_key}_{horizon}.nc`, because it is a stage-B product requiring the window, calendar, complete-year and dry-month logic and is not derivable from the series alone. The "adds no jobs, no extra network" argument holds *more* strongly for the series than for the climatologies: it is literally the pre-reduction array. Volume is a formula (`n_cells × n_months × n_vars × 8 B`; ≈150 KB for the seed) rather than a bound, because cell count scales with basin extent. **`save_grids` is kept** over the owner's `save_gridded` wording, for continuity with the config, `dev/reference/workflows/climate_projections.md` and every shipped config; the rename is flagged as **OQ-12**, with the note that it breaks the manifest's seed-config sha256 (R10) and should ride with another config-key commit rather than land alone. | design-v3.md |
| R2-D2 | 2 (confirmation) | — | confirmed, no design change | **Owner confirmation D2 → A1.** design-v2 already selected A1 — declare the full `climate_store_spec` and accept the gridded observed extraction on a fresh projections-only run — with A2 as the named fallback and the discriminating question surfaced for the owner. The owner answered it: **yes, acceptable.** `design-v3.md` §5.4 and §6.4 record the confirmation and demote A2 from "named fallback awaiting an answer" to "recorded fallback". G2 (WF2 runs with no `hydrology_model/` on disk) is preserved. Risk **R7** — the WF1 decoupling is a cost *transfer* on a fresh projections-only project, not a cost reduction — stands, now as an explicitly accepted cost rather than an open tradeoff. | design-v3.md |
| R2-OQ4 | 2 (confirmation) | — | confirmed, OQ-4 closed | **Owner confirmation OQ-4 → 30 years, 1985–2014. OQ-4 is CLOSED.** Under D1's clip that window sits entirely inside the historical experiment, so no clip warning fires and it clears the 20-year short-window floor; §9 adds exactly that as the closure's acceptance check. **How the closure lands was checked, not assumed:** `Snakefile_climate_projections:36` reads `historical_year_range` with `get_config(..., optional=False)` — it is **required, with no default in code** — and every shipped config sets it explicitly (`snake_config.template.yml: [1980, 2010]`; `snake_config_model_test*.yml: [1990, 2010]`). The closure therefore lands as a **template** change plus documentation, carried as new migration step **5f**, which is value-neutral on the seed and **manifest-clean**, because the manifest pins the *seed* config's sha256 and not the template's. The test fixtures deliberately keep `[1990, 2010]`, so step 5e stays output-neutral on the seed and the per-cause diff attribution that §4 criterion 5 protects survives. **Flagged for owner correction** in both §5.4 and step 5f: if the seed was meant to move too, 5f becomes value-changing and manifest-breaking on all four pinned summary targets and needs a full documented re-record. | design-v3.md |
| R2-cat | 2 (repo change) | — | design updated | **The CMIP6 catalog became generated (commit `f8194e8`) — this extends `risk-08`, `risk-02` and `ext1-09` without re-dispositioning them.** §5.7 is **rewritten**: entries are now one per (model, scenario), keyed `cmip6_{institution}/{source}_{experiment}_{member}` with `member` the only placeholder, so DAG-build validation is a key lookup plus a membership test rather than a re-implementation of hydromt's placeholder cross-product — **simpler and stronger**, because membership is a live-crawl fact and the header's guarantee ("a source name resolving means the store is really there") is what the hand-curated catalog could not offer. **risk-08's drift concern is revisited and reduced rather than carried unexamined:** the surface is smaller, and the counterparty is now a repo-owned generator (**C7**, a new constraint) rather than an evolving upstream library, so a format change is a diff this repository makes to itself. Retained mitigations: minimal logic, unknown constructs error naming the key, and the integration-marked cross-check against `hydromt.DataCatalog(...).sources`. Added: a `meta.generated_by` assertion, which makes C7 executable. **risk-02's digest is corrected for a generated file** — `placeholders` and `meta` are **excluded** and the **resolved** URI is included, because regeneration routinely adds members and a member-list change must not re-derive series whose data did not change; both directions are falsifiable cache tests (§9 case g). Merge-key resolution under `yaml.safe_load` was **verified** (PyYAML 6.0.3, the pinned env) rather than asserted, and §9 checks it on a non-anchor entry. **The multi-version glob** the inventory §2 measured — `NCC/NorCPM1` historical `tas` publishing `gn/v20190914` and `gn/v20200724` under `.../Amon/{variable}/*/*` — is answered by **D8**: the reducer asserts a strictly increasing, duplicate-free time axis and **raises** naming the source and the first duplicated timestamp; no silent `drop_duplicates`. §5.3 states honestly that the one measured instance is unreachable (NorCPM1 publishes zero SSP members, so D7 never pairs it) and that the check exists because the glob property is general. §6.12 records version-pinning and latest-preference as alternatives, with **OQ-14** as the settling evidence. New risks **R11** (the catalog is a dated snapshot, so "resolves" means *observed at `meta.crawled_on`*; a withdrawn store fails at read time, correctly classified under D4, with the snapshot date recorded on every series and in `composition.csv`) and **R13** (65 reachable historical models and up to 96 members per combination make the request surface much larger — hence `members:` takes explicit lists only, with an `all` token plus a cap, and a per-model mapping, recorded as **OQ-13**). §6.6 records that catalog generation was resolved by a repo-owned `gcsfs` crawler with **no new dependency**, **withdrawing the `intake-esm` candidate** from OQ-7. §5.5 gains the caveat that the generator guarantees `pr`+`tas` only, so `kin`/`press_msl` are a read-time failure rather than a resolution skip (**OQ-15**). §5.2's job accounting becomes a **formula** with one measured seed example (6 resolved combinations + 3 references = 9 reduce jobs, 15/14 total, verified against the generated catalog on 2026-07-29), plus the corollary that the historical series set is *derived*, not configured. | design-v3.md |

---

## Round 2 findings (arbitration) — 2026-07-29

The external round cap (2) is spent; owner arbitration accepted **all nine** round-2 findings (fix required, none rejected) and issued rulings **A1–A3** for the decision-shaped ones (`status.md`). Stage 6a (`design-v4.md`, authored on Fable per the escalation rule — ext2-02 faulted the round-1 resolution of `ext1-08`) is confined to these IDs. Severities are as filed by the round-2 reviewer. Finding texts: `external-review-r2.md`; aggregation: `review-index-r2.md`.

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| ext2-01 | 2 | blocking | accepted | **D9 — region content identity** (§5.3). `store_region.geojson` becomes a **plain input** of `reduce_gcm_series` and `derive_change_factors` (`ancient()` dropped); the digest's region component becomes the **content fingerprint** of the polygon on disk (canonical-geometry sha256), replacing the region specification; a scheduled reduce job **revalidates** (offline fingerprint + digest check against the existing series) before deriving, so the byte-identical-rewrite property `ancient()` bought is preserved as "scheduled, no-op" rather than "not scheduled"; and stage B recomputes every expected digest **including the current polygon fingerprint**, raising on mismatch — a backstop that holds under any rerun-trigger configuration. Both routes into a product are gated on content equality; nothing is merely audited. §6.14 records the rejected alternatives (keep `ancient()`+spec — the hole; plain input without revalidation — spurious re-downloads; parse-time fingerprint param — fresh-run double-derivation). §7 consequence 20; §8 steps 1 and 2b; §9 cache tests (i)–(k). | design-v4.md |
| ext2-02 | 2 | blocking | accepted | **D10 — spherical cell-area weighting from midpoint edges** (§5.3). This is the re-raise of ext1-08: design-v3's fix checked a condition (1-D, strictly monotonic) that did not establish the claimed validity condition (uniform spacing). v4 aligns claim, check, and scheme instead of patching the check: weights become per-cell `sin φ` differences × Δλ from midpoint-derived edges, whose only precondition — ordered, distinct 1-D centers — **is** what the check tests, and the document states both the exact tested condition and why it suffices. On uniformly spaced grids the weights are provably identical to cos-latitude (the constant factor cancels); on non-uniform 1-D grids (Gaussian) the per-cell widths ext2-02 showed were missing enter the weights. The residual (true vs midpoint edges) is labelled and carried by a **narrowed OQ-10**; refusals narrow to 2-D/curvilinear and non-monotonic axes (§7 R8). §6.15 alternatives; §9 uniform + non-uniform analytic weight tests (step 5a). | design-v4.md |
| ext2-03 | 2 | blocking | accepted | **D11 — complete gridded-change schema + exact-compatibility gate** (§5.8). `grids/change/{series_key}_{horizon}.nc` is fully specified as the cellwise counterpart of the tabular product — same §5.6 formulas, statistic set, windowing, calendar weighting, and dry-month thresholds; `statistic` + `month` dims; per-variable annual/monthly fields with `_absolute` companions and boolean dry-reference masks; A1 window attributes. Before any cellwise arithmetic, stage B asserts **equal CRS and identical spatial coordinate arrays** (exact, no tolerance) between scenario and reference gridded series, failing the run naming both series — the assertion *precedes* any xarray operation that could align, so implicit-alignment artifacts are structurally excluded, not merely tested for. No regridding in v2.0; a per-pair skip is unimplementable under a parse-time-declared output (§6.16). §9 shifted-grid, mismatched-CRS, missing-CRS, monthly/annual and dry-cell tests (step 7). | design-v4.md |
| ext2-04 | 2 | major | accepted | **D12 — physical source identity via the generated store index** (§5.3). The "resolved URI" is renamed what it is — the entry URI template — and physical identity comes from `config/catalogs/cmip6_store_index.json`, emitted by the same crawl (same `crawled_on`, C7-owned), pinning the observed `{grid_label}/{version}` per (entry, member, certified variable); in-URI pinning is unimplementable because the version varies beneath both `{member}` and `{variable}` (§6.12, rewritten). Pins enter the digest — regeneration after a re-publication re-derives exactly the affected series — and the reducer **verifies the pin at read time** via a `gcsfs` listing, raising on mismatch, which makes the claimed identity checked rather than nominal. The entry's **`metadata` map joins the digest** (the finding's second half), and `cst_source_paths` + `provenance.json` name the physical stores. Best-effort variables have no pin — stated as a tier limit (§5.5). **OQ-14's pinning half closes; the cadence half stays open.** §8 step 2a; §9 store-index-pin tests; new risk R14. | design-v4.md |
| ext2-05 | 2 | major | accepted | **Arbitration ruling A1 — 30 calendar years.** §5.4: 1985–2014 is calendar-defined; a non-January `start_month_hyd_year` yields **29** complete hydrological years, accepted — and every window-stating artifact reports **nominal and effective** values (bounds, `n_hyd_years`, per-end dropped months): change-factor tables (`*_nominal`/`*_effective` columns, §5.9), `composition.csv` (A1 columns, §5.7), `provenance.json`, report disclaimer; §5.6 defines the nominal/effective terminology. One argued deviation from the ruling's letter, flagged for G2: per-run window values are not stamped onto cached series files, whose identity deliberately excludes analysis windows (G5); the series carries acquisition coverage instead. §9's acceptance test asserts **effective values, not warning silence** — 29 years / 1985-10-01…2014-09-30 / 9+3 dropped months for an October start; 30 / full window / 0 for January (steps 5e/5f). | design-v4.md |
| ext2-06 | 2 | major | accepted | **Arbitration ruling A2 — defaults chosen; OQ-9 CLOSED.** §5.6: `relative_change.min_reference` defaults to **`precip: 0.1 mm/day`** (≈3 mm/month — below it a reference month is hydrologically negligible under a percent-perturbation framing and the factor's sampling spread dominates the factor; conservative, revisable by the measurement OQ-9 names) and `relative_change.max_flagged_months` defaults to **3** (one dry season is a basin's normal state; more means the monthly relative product is undefined for over a quarter of the year and is flagged at combination level). Strict comparison semantics defined (`<` / `>`); a non-shipped `change: relative` variable has **no default** — the config must supply its threshold or DAG build raises. Defaults live in code + template, seed untouched → value-neutral and manifest-clean. §9 boundary tests below/at/above both thresholds (step 6b). | design-v4.md |
| ext2-07 | 2 | major | accepted | **Arbitration ruling A3 — two-tier variable contract; non-certified variables stay selectable.** §5.5: `precip`/`temp` are **catalog-certified** (membership implies observed existence; D12 pins); `kin`/`press_msl` are **best-effort** (nameable, unverified, may fail at read time under D4; no DAG-build pin, stated as a tier limit). Shipped configs and the code default name exactly `[precip, temp]`; selecting a best-effort variable emits a DAG-build warning naming the tier and the read-time risk (§5.7); `composition.csv` gains a `tier` column with the per-variable map in `provenance.json`. Nothing is rejected at DAG build, per the ruling, and the best-effort residual is exactly what **OQ-15** — kept open by A3 as the promotion route (widen `REQUIRED_VARS`, crawl, certify, pin) — exists to remove. §7 consequence 22; §9 tier tests. | design-v4.md |
| ext2-08 | 2 | minor | accepted | §5.7 D4: the impossible claim that the resolved set is "written down in `composition.csv` before any job runs" is removed. `composition.csv` is a stage-B output and **describes completed runs only**, consistent with ext1-02's round-1 disposition (provenance describes successful runs); the pre-execution surface is the DAG-build **stderr summary** plus job logs. No separate pre-execution manifest is added, with the reason stated in place: a DAG build that writes an output file makes parsing side-effecting — a dry run that writes is not a dry run — and the stderr summary already serves the failed-run diagnosis case. | design-v4.md |
| ext2-09 | 2 | minor | accepted | §9's no-aggregation test drops the "no row equals the mean of other rows" invariant — a legitimate value can coincide with such a mean (false failure) and poorly chosen synthetics can hide aggregation (false pass). Replaced with: exact **tuple cardinality** and key **uniqueness**; **direct per-row equality** to independently computed per-series expected values over pairwise-distinct sentinel inputs (plain numpy, no WF2 code on the path) — an aggregated row cannot pass by construction; plus the retained figure assertions and composition counts (step 6c). | design-v4.md |
