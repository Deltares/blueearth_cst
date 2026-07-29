---
run: wf2-climate-analysis-v2
target-repo: blueearth_cst
genre: workflow-spec
author-binding: generic
started: 2026-07-29
variant: lite  # PROMOTION TRIGGERED -> full (blocking findings)
stage: arbitration (external cap reached)
external-rounds-completed: 2
dispatches:
  opus: 0
  fable: 1
gates:
  G1: approved 2026-07-29
  G2: pending
flags: [owner-requested-fable-lens, promoted-lite-to-full, blocking-findings-open, external-cap-reached, fable-escalation-r2]
---

## Stage log

- [done] 0-intake — outputs: intake.md
- [done] 1-seed — outputs: design-v1.md (seeded from the landed
  `dev/workflows/wf2-climate-analysis-v2-design.md` @ f5cd5ff; not re-authored)
- [done] G1 — approved 2026-07-29. Owner directed review of this doc as written,
  so framing is settled to the extent recorded in the review brief's
  settled-framing block. Provisional selected alternative: the three-stage
  architecture (store / reduce / derive+report) of §5.1. OQ-1..OQ-8 remain
  **open** and reviewer input on them is explicitly wanted.
- [done] 2-internal-panel (lite: single lens) — outputs: internal-review-risk.md
  (verdict: revise on design-v1.md; 1 blocking, 5 major, 3 minor). Reviewer:
  `critical-thinker` on **Fable**, owner-requested.
- [done] 4-external-r1 — outputs: external-review-r1.md (doc_version:
  design-v1.md; verdict: revise; 3 blocking, 6 major, 1 minor), codex-transcript.txt
- [done] 5-convergence-r1 — NOT converged. 3 distinct blocking defects
  (risk-01 = ext1-01; ext1-02; ext1-03). Promotion trigger fired: lite -> full,
  external cap now 2. Index: review-index.md
- [done] G1-return — re-approved 2026-07-29 with four owner rulings (below).
  Rulings R1 and R4 change the selected alternative; the revision is authored
  against them as settled framing.
- [done] 6-revision-r1 — outputs: design-v2.md (1357 lines), ledger.md.
  Author: `cst-architect` (Opus). All 19 findings dispositioned **accepted**;
  0 rejected / deferred / withdrawn, so **no G2 ratification is owed**.
  New decisions D1 (clip), D2 (store cost), D3 (keep output root, closes OQ-3),
  D4 (fail-fast), D5 (entry point). New open questions OQ-9/10/11.
- [done] G1-return-2 — re-approved 2026-07-29. Owner rulings R3', R3'', R5 and
  confirmations D2->A1, OQ-4->30yr, revise-then-round-2. R3'/R3''/R5 change the
  output contract and the selected alternative, so a revision precedes round 2.
- [done] 6-revision-r2 — outputs: design-v3.md (2031 lines), ledger.md
  (Round-2 section, ext1-07 amended, 6 rows appended). Author: `cst-architect`
  (Opus). New decisions D6/D7/D8; OQ-4/6/8 closed, OQ-12..15 opened.
- [done] 4-external-r2 — outputs: external-review-r2.md (doc_version:
  design-v3.md; verdict: revise; 3 blocking, 4 major, 2 minor),
  codex-transcript-r2.txt
- [done] 5-convergence-r2 — NOT converged; **external cap (2) reached** →
  owner arbitration. Index: review-index-r2.md
- [open] arbitration — owner rules on ext2-01..09; three are decision-shaped
  (ext2-05, ext2-06, ext2-07), six are author-fixable
- [open] 6a-arbitration-revision — **routes to Fable** (see below), confined to
  the arbitrated finding IDs
- [open] G2 — owner approval under arbitration authority

## Fable escalation — round 2 (skill rule, now triggered)

Round 2's regression duty re-raised exactly one round-1 disposition:
`ext1-08` (area weighting), via **ext2-02**, on the ground that the accepted fix
does not establish the validity condition it claims. The skill routes the
revision spawn *answering an external review that faults a prior fix* to Fable —
the Opus revision did not satisfy the reviewer. Routing is per-spawn, so the
whole 6a revision runs on Fable (single-writer invariant), and `dispatches.fable`
increments when it is dispatched.

## Driver editorial fixes to stage-6 outputs (logged)

- **Stray tool markup stripped.** Both `design-v2.md` and `ledger.md` ended with
  leaked `</content>` / `</invoke>` lines. Removed mechanically; no design
  content touched. Logged as an editorial-classified edit per the skill's
  single-writer rule.
- **Driver artifact corrected, not the author's.** The stage-6 author caught a
  factual error in this file's premise-verification block and in
  `review-index.md`: the manifest's 7 WF2 targets are **not** all under
  `climate_projections/cmip6/` — 6 are, the 7th is
  `config/runs/snake_config_climate_projections.yml`, pinned by
  `{"sha256": …, "type": "yaml"}` (verified). Corrected in `review-index.md`
  with the reason it matters: a config-key addition breaks `check_baseline`
  with no computed number moving.

## G1 rulings — 2026-07-29 (owner)

**R1 — reference series: clip, never splice.** Do not modify historical or
scenario runs; no processing of the 2015–2020 gap. The GCM reference window is
clipped to the historical experiment (≤2014) and the workflow **warns/disclaims**
when the user selects a historical period extending past 2014. Consequence
accepted: the overlay's reference period and the project's stress-test baseline
(`shared.historical_window`, 2000–2020) remain different periods; the disclaimer
is what surfaces that, replacing the design's "equal by construction" claim.
G3 is restated as an alignment **check with a warning**, not a structural
guarantee. Closes risk-01 / ext1-01 as an owner-ruled design change.

**R2 — `save_grids` is retained as an option, default OFF.** Raw grids from the
model runs are saved on request, for advanced analysis at a later stage. This
requires a first-class **optional gridded branch with declared outputs** — not
the current undeclared, params-passed file layer. Closes OQ-8 / ext1-09.

**R3 — SUPERSEDED by R3′ below.** ~~Driver's reading: thresholds on unique
models; members averaged **within** a model before equal-model summaries;
no aggregate envelope below the threshold.~~ The within-model averaging half was
a driver mis-reading of the owner's "agreed", corrected 2026-07-29.

**R3′ — no aggregation, at any level (owner, 2026-07-29).** Members are **never
averaged** in pre-processing. **Each (model, scenario, member) combination is a
single data point** with its own ΔT and ΔP, carried distinctly end to end and
displayed as its own trace/row/point. **No aggregation across models, scenarios,
or members** — no percentile envelope, no ±σ, no model-level collapse.
Interpretation of model similarity and correlation is explicitly **outside this
scope**, handled by the owner downstream.

Member availability is a **union, not a fixed list**: some (model, scenario)
combinations publish three members, others one; some models are missing SSP
scenarios entirely. That is normal and must not be an error — the workflow
collects as many data points as the store actually offers.

Consequences for `design-v2.md`, all in §5.6 unless noted:
- **Delete** "Members are averaged within a model first …". It contradicts R3′.
- **`ensemble.min_models_for_envelope` becomes moot** — with no aggregation at
  any level there is no envelope to gate, so the threshold, the min–max range,
  and N9's "institution de-duplication not applied" all collapse into a simpler
  rule: report composition, plot points, aggregate nothing.
- **ext1-07's disposition changes** from wholly `accepted` to **partially
  accepted / partially rejected by owner ruling**: its "average members within
  each model" half is rejected. A rejected `major` requires owner ratification
  at G2 — R3′ **is** that ratification, recorded here.
- The member config contract becomes a *requested set intersected with
  availability per (model, scenario)*, which bears on §5.7's DAG-build
  validation and on the `members:` key's meaning.

**R3″ — downstream/ensemble statistics are out of v2.0 scope entirely (owner,
2026-07-29, addendum to R3′).** Cross-combination statistics — including
`min_models_for_envelope` and anything else computed *over* the set of data
points — are computed **ex-post**, downstream of the ΔT/ΔP values, and are not a
v2.0 concern.

**WF2 v2.0's deliverable is the data points themselves**: one (ΔT, ΔP) per
(model, scenario, member, horizon), in a tidy table, plus the composition
record. Anything that reduces *across* those rows is deferred.

Driver's reading of the boundary, stated explicitly because the two senses of
"statistic" are easy to conflate — correct this if wrong:

- **IN scope (per-series, defines the data point):** the statistics computed on
  the annual series *within* one (model, scenario, member, horizon) — `mean`,
  `median`, `std`, and the tail quantiles — since these are what the change
  factor *is*. §5.5, §8 step 5d and OQ-4 remain live.
- **OUT of scope (cross-combination, ex-post):** anything reducing over the set
  of data points — envelopes, percentile bands, ±σ, model-count thresholds,
  weighting, de-duplication.

Consequences beyond R3′:
- **Delete the `ensemble.min_models_for_envelope` config key**, the envelope
  suppression rule, and the min–max range from §5.6. §5.6 reduces to: compute
  per-combination change factors, emit the tidy table, report composition.
- **N9 restated** — institution de-duplication and performance weighting are not
  "not applied", they are *downstream concerns*, along with every other
  cross-combination statistic.
- **Report figures** show one point/trace per combination; the multi-model
  percentile envelopes in today's `plot_proj_timeseries.py` are not carried
  forward.
- **Slot S5 (grid-vs-cloud advisory) is itself an ex-post statistic** — it
  compares the cloud's extent against the perturbation grid — so it defers with
  the rest rather than being a v2.0 read.

**R5 — the monthly series is a deliverable, not just a cache (owner,
2026-07-29).** v2.0's declared outputs are:

1. the tidy change-factor table — one (ΔT, ΔP) per (model, scenario, member,
   horizon);
2. **basin-averaged monthly time series per run** — one per (model, scenario,
   member);
3. the composition record;
4. **gridded series** for each run when the gridded option is on, default off.

Consequences:
- **`series/{series_key}.nc` is promoted from internal cache to user-facing
  product.** design-v2 §5.3 treats it purely as a cache whose only consumer is
  stage B. It now needs a documented schema, stable naming, and a retention
  rule, and it must appear in §2's output contract — the cache *is* the
  deliverable, which is a strengthening of the architecture, not a conflict
  with it.
- **The gridded option's content changes.** design-v2 §5.8 carries forward the
  three existing families (`historical_stats_{model}.nc`,
  `stats-{model}_{scenario}.nc`, `monthly_change_mean_grid-*.nc`), of which the
  first two are 12-month *climatologies*, not series. R5 asks for the gridded
  counterpart of deliverable 2 — the **monthly series on the source grid**
  (time × lat × lon), retained before spatial reduction. Driver's reading;
  correct if the existing climatology grids were meant instead.
- **Volume is not a concern at Amon resolution.** The seed basin's bbox
  (~0.2° × 0.13°) plus the 1° buffer spans ~2.2° × 2.1°, which at CMIP6 Amon
  resolution is single-digit grid cells. A full 2015–2100 scenario span is
  ~1032 months × <10 cells × 2 variables — well under a megabyte per run. The
  gridded option is cheap either way; default-off is a tidiness choice, not a
  cost one.
- **Naming.** The owner wrote `save_gridded`; the existing config key is
  `save_grids`. Driver keeps `save_grids` for continuity with the current config
  and `dev/workflows/climate_projections.md`; flag if a rename is wanted.

## Owner confirmations — 2026-07-29

- **D2 → A1.** Declare the full `climate_store_spec` in WF2, accepting the
  gridded observed extraction on a fresh projections-only run. G2 (WF2 runs with
  no `hydrology_model/`) is preserved. `design-v2.md` §5.4/§6.4 already selects
  A1; confirmed, not changed.
- **OQ-4 → 30 years, 1985–2014.** Under R1's clip this sits entirely inside the
  historical experiment, so no clip warning fires. Value-changing; gates step 5e.
  OQ-4 is now **CLOSED**.
- **Process → revise to v3 first, then external round 2.** Round 2 is capped and
  single-use; reviewing a document whose §5.6 contradicts R3′/R3″ and whose §5.7
  predates the generated catalog would spend it confirming known-stale content.

**R4 — v2.0 scope narrows to GCM projections analysis.** For now: **monthly GCM
projections output analysis**, with room to expand to **gridded** results for
plotting and analysis (consistent with R2). **No comparison against observed
data at this stage.** Consequences: the "one reducer over observed + GCM"
premise of §5.2 narrows to "one reducer over all GCM sources"; extension slots
S1/S2/S3 (observed climatology, multi-dataset comparison, bias diagnostics)
become documented future architecture changes rather than free reads; and
ext1-03 (daily-observed vs monthly-`Amon` reduction) is dissolved for v2 — the
design must state explicitly that `extract_historical.nc` is not reduced in
v2.0. Closes ext1-06 by taking its narrow option.

**Open consequence of R4 the revision must address.** With observed analysis out
of scope, declaring `extract_climate_grid` buys only the model-free region
polygon, while C1 forbids a region-only asymmetric declaration — so the full
observed extraction stays on WF2's critical path with no analytical payoff
(risk-07 becomes load-bearing, not a footnote). The revision must either
justify that cost explicitly or propose an alternative that does not break C1.

## Driver premise verification (2026-07-29)

The driver checks facts, never authors. Three findings' premises verified
against the repo before arbitration:

- **risk-01 — HOLDS. Regression, not pre-existing.**
  `blueearth_cst/projections/get_stats_climate_proj.py:156` hardcodes
  `time_tuple_all = ("1950-01-01", "2014-12-31")` for cmip6 historical, and
  `config/catalogs/cmip6_data.yml` resolves historical under
  `gs://cmip6/CMIP6/CMIP/{model}/historical/`. `shared.historical_window` ends
  2020-12-31, overrunning the source by six years. The *current* code's
  `historical_year_range: [1990, 2010]` fits inside the historical experiment,
  so this defect is introduced by the design's G3, not inherited.
- **risk-04 — HOLDS.** `dev/baseline/manifest.json` pins exactly 7 WF2 targets:
  **6** under `climate_projections/cmip6/` (3 PNGs,
  `annual_change_scalar_stats_summary.{nc,csv}`, `..._summary_mean.csv`) **plus**
  `config/runs/snake_config_climate_projections.yml`. No monthly intermediates
  are covered, so a green `check_baseline` constrains the final scalar summary
  only. *(Corrected 2026-07-29 — see the editorial-fix note below.)*
- **risk-09 — HOLDS.** 1+9+1+1+1+1 = 14, not the 13 stated in §5.1/§7; and the
  reduce count omits the observed series that §5.2 routes through stage A.

## Variant note

**Lite variant** — single internal lens + 1 external round, per the owner's
request for two individual reviews before finalizing. Gates, ledger,
convergence, and arbitration are unchanged.

**Promotion trigger:** any `blocking` finding, or non-convergence after this one
external round, escalates to the full variant — the remaining two lenses
(architecture, repo-fit) spawn on the current version and the external cap
reverts to 2.

## Tier deviation (logged)

The skill rations Fable to revision spawns answering an external review that
re-raised a prior finding. Here the owner requested a Fable lens directly, which
overrides the default. Counted honestly: `fable: 1`.

## Preflight

`codex exec --sandbox read-only --ephemeral -c approval_policy=never
-m gpt-5.6-sol` — banner verified 2026-07-29: `approval: never`,
`sandbox: read-only`, `model: gpt-5.6-sol`. Fail-closed control confirmed
before dispatch.
