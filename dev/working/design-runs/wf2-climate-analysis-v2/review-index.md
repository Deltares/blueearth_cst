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
