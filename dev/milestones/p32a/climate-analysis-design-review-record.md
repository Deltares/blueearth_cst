# P3-2a climate-analysis design — consolidated review record

Durable audit trail of the `p32a-climate-analysis` design-review-loop run
(2026-07-24; full variant; run dir pruned per retention policy). Accepted
design: `climate-analysis-design.md`. Scope authority:
`climate-analysis-intake.md` (landed pre-run, commit 8d126e8).

## Run summary

| Stage | Outcome |
|---|---|
| 0 Intake | Pre-landed authoritative intake (user-confirmed via design-scoping); no dialogue repeated |
| 1 Draft v1 | cst-architect (Opus), skeleton-first |
| G1 framing | Approved 2026-07-24 (separate wf1-owned extraction + shim-window migration; aggregation mechanism left as design detail) |
| 2 Internal panel | risk (critical-thinker) revise 4M/2m · architecture (cst-architect) revise 1B/2M/2m · repo-fit (python-engineer) revise 1M/4m |
| 3 Revision r1 → v2 | Opus; all 16 findings accepted (survived one stream-idle timeout via same-thread resume; skeleton-first, no loss) |
| 4 External r1 (GPT, codex exec, clean-room) | **revise** on design-v2.md — ext1-1/2 blocking, ext1-3/4 major, ext1-5 minor |
| 5 Convergence r1 | Not converged; all findings faulted prior fixes → Fable escalation |
| 6 Revision r2 → v3 | **Fable**; all 5 accepted (verified hydromt callables; per-branch pipeline; ladder; wrapper; bbox) |
| 4 External r2 (GPT, + ledger/index, regression duty) | **revise** on design-v3.md — ext2-1/2 major, ext2-3 minor |
| 5 Convergence r2 | Not converged; 2-round cap reached → human arbitration |
| Arbitration | 2026-07-24, user rulings final: ext2-1 accept-fix (declared sidecar DAG edges); ext2-2 accept-fix, tolerance variant (direct-hop rung explicitly not chosen); ext2-3 accept-fix, exclude-eobs variant |
| 6a Arbitration revision → v4 | **Fable**; confined to ext2-1..3; driver scope-check passed (hunks = arbitrated sections + 9 declared cross-refs, no drift) |
| G2 approval | Approved 2026-07-24 (no editorial change requests); v4 → ACCEPTED |
| 7 Finalize & land | Lean path (no author spawn). Logged editorial edits at landing: ACCEPTED status-header swap; scope-authority citation repointed to the landed intake; run-artifact citations (ledger, external reviews) repointed to this record |

Dispatches: 5 Opus + 2 Fable (both Fable spawns under the external-re-raise
escalation rule). External reviewer: GPT via headless `codex exec`
(read-only sandbox, `approval_policy=never`; post-run `git status` backstop
clean both rounds).

Sequencing invariant note: design-v4.md carries no external verdict naming
it — the round cap forecloses one; the user's arbitration rulings stand in
for that verdict per the loop contract, with the driver's scope-check
confirming v4 changed only the arbitrated findings' scope.

---

## Internal panel — aggregation index (verbatim)

The three verbatim lens reviews lived in the pruned run dir; their findings
are preserved by ID, severity, and gist in this index and dispositioned in
the final ledger below.

Driver-written index over the three verbatim lens reviews
(`internal-review-risk.md`, `internal-review-architecture.md`,
`internal-review-repo-fit.md`). Grouping only — every original ID, severity,
and text is preserved by reference in its source file; nothing is deleted or
re-graded here.

**Verdicts:** risk = revise · architecture = revise · repo-fit = revise.
**Totals:** 1 blocking · 7 major · 8 minor (16 findings).

## Groups (author must disposition every original ID individually)

### Group 1 — wf1 extraction rule input has no producer (fresh-run DAG failure)
- **arch-1 (BLOCKING, §5.2)** — `input: ancient(region.geojson)` has no producer
  rule in any Snakefile; fresh project ⇒ `MissingInputException` at DAG build,
  and commit 3's own `test_cli.py` wf1 dry-run gate goes red (default config
  does not stage the file). The design reintroduces the exact C4 failure class
  it used to reject store-reuse. Fix directions offered: declared producer edge
  (e.g. `get_region_preview.py` as a rule) or source the bbox from a declared
  rule-1.03 output.

### Group 2 — acceptance-gate delta attribution is miscalibrated (§5.5)
- **risk-1 (major, §5.5/§7)** — old path is lapse/pressure-corrected,
  PET-derived model forcing (`setup_time_horizon.py:44-72`), not regridded raw
  climate; delta carries a systematic 1–3 °C lapse term + PET-method term the
  "regridding-scale, exceeds-ratio ⇒ failure" gate misattributes.
- **arch-3 (major, §5.5/OQ-2)** — EP-delta logic unsound without PET
  method-parity; OQ-2 pins locus, not method.
- **risk-5 (minor, OQ-2/§5.5)** — pin the wf1-side PET method to the model's
  (`debruin`/`makkink`) if regridding-scale delta is the intent.

### Group 3 — Option C mechanism unproven and its polygon input does not exist
- **risk-4 (major, §5.1/§6.3)** — the subcatchment GeoDataFrame does not exist
  (subcatchment is a raster; `staticgeoms/` has no per-subcatchment polygon);
  raster→vector polygonization with id→station-index preservation is
  unspecified; a permuted mapping is a correctness failure the characterized
  diff would not catch. Suggests reopening B-vs-C ranking.
- **arch-2 (major, §5.1/§5.2/§6.3/§5.5)** — "selected" Option C is
  OQ-7-unconfirmed and its B-fallback flips the expected delta to ≈0, silently
  invalidating §5.5's calibration; vectorization step missing from §8.

### Group 4 — test surface of the reduction change (per-commit green broken)
- **repo-1 (major, §5.1/§5.3/§6.3)** — `tests/test_climate_forcing.py` is a
  second live caller of the raster-groupby mechanic; "none survive" and
  "assertions stay verbatim" are false; commit 4's gate never runs the full
  suite, so a red unit test is latent across commits 4–5.

### Group 5 — overstated model-independence claim
- **risk-2 (major, §7/C1)** — "runnable without a built model for extraction
  AND aggregation" is true only for extraction; aggregation zones come from the
  built model via the wf1 caller. Scope the claim; note whether P3-2b closes it.

### Group 6 — branch-scoping of the "native resolution" property
- **risk-3 (major, §5.2)** — no-reproject is era5-branch-only; chirps branch
  reprojects + lapse-corrects (`extract_historical_climate.py:106-144`);
  acceptance reasoning is implicitly era5-only and does not transfer to gabon
  if chirps.

### Ungrouped minors
- **risk-6 (minor, §5.4)** — dry-run does not execute `script:` bodies; missed
  `script:` repoint after shim deletion surfaces only at execution.
- **arch-4 (minor, §5.3/§5.2)** — new wf1 rule must name keys
  `prj_region`/`climate_nc` (+ params keys) or the verbatim `__main__` block
  breaks.
- **arch-5 (minor, §5.3)** — catalog module's sole consumer is wf3 rule 3.08;
  "wf2 has no import…" phrasing misleads.
- **repo-2 (minor, §5.5)** — name the manifest no-op explicitly as a justified
  divergence from intake decision 4's "manifest re-records" wording.
- **repo-3 (minor, §5.5)** — `outlet_index.csv` is a `rule all` target, not a
  manifest TARGET; drop or clarify.
- **repo-4 (minor, §5.4)** — star-import shim silently depends on moved modules
  having no `__all__`; repoint tests in the move commit.
- **repo-5 (minor, §5.2/§7)** — rule 1.10 declares the script twice (input
  label + directive); input rewrite must leave the `script = ...` input intact.

## What all three lenses verified sound (no action)
Lift/rewire skeleton (§5.1/§5.3), store-reuse rejection on C4 (§6.2), rule-3.02
`script:`-only pin (C3), manifest-scope reasoning (§5.5), `__init__.py` and
naming.md fit, OQ-2's wf1-side-only locus constraint.

---

## External review — round 1 (verbatim; doc_version design-v2.md)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: 5.2 wf1 re-source mechanism (raw gridded climate, no `mod.forcing.data`)
- finding: The selected parity procedure applies the model’s temperature correction after extraction without defining branch-specific safeguards, even though the design establishes that the `chirps`/`chirps_global` extraction branch already applies a lapse-rate correction. Merely making the acceptance interpretation “branch-aware” does not prevent the selected implementation from correcting temperature twice.
- rationale: For `chirps` configurations, applying the §5.2 parity step as specified can introduce a second lapse correction and produce systematically wrong subcatchment temperature and PET values. The implementation and validation plans provide no branch-conditioned algorithm that either skips, reverses, or otherwise reconciles the correction already embedded by extraction.
- suggested_fix: Specify an explicit transformation pipeline for each supported `clim_source`, including which corrections extraction has already applied, which wf1-side transformations remain, and an invariant or test proving that every physical correction is applied exactly once.

### ext1-2  [blocking]
- section: 5.2 wf1 re-source mechanism (raw gridded climate, no `mod.forcing.data`)
- finding: A correctness-critical part of the selected approach remains unresolved: OQ-2 leaves the exact plain-dataset PET implementation unconfirmed and permits dropping the EP panel if it proves infeasible, while the migration plan proceeds as though model-parity PET is available. The design likewise does not pin the callable mechanism and required inputs for reproducing the forcing build’s temperature and pressure corrections outside that build.
- rationale: Implementation cannot reliably satisfy the stated `P/T/EP_subcatchment` contract or the parity-based acceptance logic from this design alone. Failure to locate equivalent upstream operations can cause a runtime `pet` failure, numerically non-equivalent corrections, or removal of an existing plot panel—an additional value and output change not authorized by the milestone.
- suggested_fix: Resolve the applicable hydromt/hydromt_wflow APIs before approval; document exact calls, inputs, units, correction order, and supported source branches. If exact parity is infeasible, select and justify a concrete alternative now, including its output contract and acceptance method, rather than delegating that design decision to implementation.

### ext1-3  [major]
- section: 5.5 Acceptance-gate implementation (side-by-side QA + characterized diff)
- finding: The required “component-decomposed” diff is not operationally defined. The design names regridding, correction, and PET-method components but specifies only an end-to-end `new − old` monthly delta; it does not define the intermediate counterfactual datasets or calculation sequence needed to isolate those effects.
- rationale: The validator can produce the requested summary statistics yet still be unable to attribute a residual to a component. This makes the gate subjective and gameable: unexplained differences may be assigned to whichever mechanism appears plausible, undermining the promised diagnosis of wrong DEMs, flags, units, or PET methods.
- suggested_fix: Define the exact comparison ladder and persisted inputs—for example raw native, regridded-only, regridded-plus-corrections, and regridded-plus-corrections-plus-PET—and define each component as the delta between adjacent states. State how interactions and branch-specific preprocessing are handled.

### ext1-4  [major]
- section: 5.3 Mechanical rewire of wf2/wf3 imports (value-identical)
- finding: The execution contract for the shared extraction script remains ambiguous. The new wf1 rule declares `script: blueearth_cst/climate_analysis/extract_historical_climate.py`, but the preserved wf3 `__main__` expects `sm.input.prj_region`; the design then proposes a thin wf1 caller either “in the moved module or a model-side wrapper” without selecting one or reconciling that choice with the declared script path.
- rationale: A literal implementation can execute the wf3 block for the wf1 rule and fail because `prj_region` is absent, or add conditional Snakemake-shape dispatch to a supposedly verbatim moved module and risk wf3 drift. The migration plan does not identify the actual file or stable interface that owns this dispatch.
- suggested_fix: Select one contract explicitly. Prefer a dedicated wf1 wrapper script with fixed `basin_nc` input that calls the importable shared function, while retaining the moved module’s existing wf3 script block unchanged; update the rule, layout, commit plan, and execution tests accordingly.

### ext1-5  [minor]
- section: 5.2 wf1 re-source mechanism (raw gridded climate, no `mod.forcing.data`)
- finding: The design assumes a bbox derived from `staticmaps.nc` is equivalent to the existing region-derived bbox but does not define the derivation or acknowledge grid padding/alignment differences.
- rationale: Static-map bounds can be snapped or padded relative to the basin geometry, changing the extracted spatial extent and increasing runtime or affecting edge cells. Such a difference would complicate attribution of the sanctioned plot change.
- suggested_fix: Define the bbox derivation precisely and add a fixture assertion comparing it with the prior region bounds within an explicit grid-resolution tolerance.

---

## External review — round 2 (verbatim; doc_version design-v3.md)

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1  [major]
- section: 5.2 wf1 re-source mechanism (raw gridded climate, no `mod.forcing.data`)
- finding: The accepted ext1-1 fix makes the chirps parity path depend on the extraction’s orography sidecar, but the design explicitly leaves that file as an undeclared sibling of `climate_nc`; rule 1.10 likewise receives only the raw netCDF and discovers the sidecar implicitly. The correction-critical DEM therefore has neither a declared producer output nor a consumer input edge.
- rationale: On an incremental or partially cleaned run, Snakemake can regard `climate_nc` as complete while the sidecar is missing, stale, or from a different extraction. The parity module will then fail at runtime or, worse, apply the wrong DEM and produce the double/mismatched correction ext1-1 was meant to eliminate. The synthetic test checks the algebra but cannot enforce production provenance.
- suggested_fix: Make the sidecar an explicit output of the wf1 extraction rule and an explicit input to rule 1.10 for branches that require it, with a stable path independent of incidental filename construction. For branches without a sidecar, define a branch-resolved orography input contract rather than sibling-file discovery.

### ext2-2  [major]
- section: 5.5 Acceptance-gate implementation (side-by-side QA + characterized diff)
- finding: The ext1-3 resolution is operational for era5 but remains non-diagnostic for chirps: `A2 − A0` and G both contain the extra era5→chirps→model interpolation hop, yet the design supplies neither a direct-hop counterfactual nor a quantitative bound for that residual. Requiring every residual to be “assigned” to `A2 − A0` or G labels where it appears, not whether the interpolation hop caused it.
- rationale: A wrong DEM, source-field mismatch, or correction error on a chirps run can be accepted as the expected “small” hop residual because “small” is undefined and no ladder edge isolates that hop. This preserves the subjective, gameable attribution failure identified in ext1-3 for a supported branch, even though era5 has explicit tolerances.
- suggested_fix: Add a chirps-specific direct-hop reference state built from the same source fields and model-parity operations without the intermediate chirps-grid hop, then define the hop component as an adjacent-state delta; alternatively establish and justify variable-specific chirps tolerances from a pinned fixture and require unexplained excess to fail the gate.

### ext2-3  [minor]
- section: 5.2 wf1 re-source mechanism (raw gridded climate, no `mod.forcing.data`)
- finding: The design presents an eobs parity mapping but leaves extraction feasibility unresolved and delegates the outcome to implementation, where failure becomes a newly recorded limitation. That is not a complete supported-source contract.
- rationale: If an eobs configuration is introduced or restored, wf1 can fail after this milestone despite the existing catalog and PET-method mapping suggesting eobs support. The design does not clearly classify eobs as unsupported or specify the extraction variables needed to make it work.
- suggested_fix: Before implementation, either verify and specify the eobs extraction/parity pipeline and tests, or explicitly exclude eobs from P3-2a with an early configuration error and a clearly bounded support statement.

---

## Final findings ledger (verbatim; 24 findings, all closed)

| ID | Round | Severity | Disposition | Resolution / rationale | Doc version |
|---|---|---|---|---|---|
| arch-1 | r1 | blocking | accepted | v1's new wf1 rule declared `input: ancient(region.geojson)`, which has no producer rule in any Snakefile → `MissingInputException` on a fresh wf1 run and a red commit-3 `test_cli.py` dry-run on the default config. §5.2 now declares `input: basin_nc = ancient(staticmaps.nc)` (a real rule-1.03 output); the wf1 caller derives the bbox from `staticmaps.nc` and passes it via a new optional `bbox=` kwarg on `prep_historical_climate`, wf3's call/output unchanged. DAG builds on a clean project; every dry-run gate stays green. | design-v2.md |
| risk-1 | r1 | major | accepted | §5.5 no longer characterizes the delta as "regridding-scale" and drops the "exceeds resolution ratio ⇒ failure" heuristic. §5.2/§5.5 establish the old path is lapse/pressure-corrected, PET-derived model forcing (`setup_time_horizon.py:56-70`); the delta is decomposed into regrid / temp-correction / PET-method components, each judged against its own cause, and the wf1 path reproduces the model's correction+PET for parity. | design-v2.md |
| risk-2 | r1 | major | accepted | §7 adds a "Scope limit on model-independent" note: "runnable without a built model" is claimed for **extraction only**; the aggregation still needs the built model's subcatchment-zone raster (C1-sense independence only). New OQ-8 tracks a model-free zone source as a P3-2b/standalone-entry candidate. | design-v2.md |
| risk-3 | r1 | major | accepted | §5.2 and §5.5 now scope the "no reproject / native resolution" property to the **era5 branch** explicitly and note the **chirps/chirps_global branch** already reprojects+lapse-corrects (`extract_historical_climate.py:106-144`). gabon's `clim_source` flagged as unconfirmed; §5.5 carries a branch-aware acceptance note (different temp-component magnitude if chirps). | design-v2.md |
| risk-4 | r1 | major | accepted | Aggregation mechanism reversed to Option B (reproject-first + existing raster `groupby`, §6.3), which needs only the existing subcatchment **raster** — not a per-subcatchment polygon GeoDataFrame (which does not exist as an artifact; `staticgeoms/` has no per-subcatchment vector). The polygonization + id→index-preservation risk is thereby eliminated for P3-2a; the polygon path is deferred (OQ-8). | design-v2.md |
| risk-5 | r1 | minor | accepted | OQ-2 now pins the wf1-side PET **method** to the model's (`debruin`/era5, `makkink`/eobs; `setup_time_horizon.py:26,67`), not just the locus, so the EP-panel delta is attributable to regridding/aggregation, not method choice. | design-v2.md |
| risk-6 | r1 | minor | accepted | §5.4 and §8 commit 6 add an execution-level check (the e2e `run_workflows.py` run covering the two `script:` sites + the new wf1 rule) before/at shim deletion, since `--dry-run` does not execute `script:` bodies and a missed repoint would otherwise surface only at execution. | design-v2.md |
| arch-2 | r1 | major | accepted | Resolved together with risk-4: Option C is no longer "selected" (its hydromt zonal API was OQ-7-open and its polygon input non-existent). §6.3 selects Option B; §5.5's expected-delta reasoning is recalibrated to the selected mechanism (Option B with model-parity → regridding/aggregation-scale residual), no longer a fallback-dependent baseline. The vectorization step Option C needed is removed from the commit plan. | design-v2.md |
| arch-3 | r1 | major | accepted | OQ-2 now pins the wf1-side PET **method** to match the forcing build; §5.5 treats the EP component's expected magnitude explicitly (≈0 under method-parity; a named PET-method term where methods differ), so the gate can attribute EP deltas correctly. | design-v2.md |
| arch-4 | r1 | minor | accepted | §5.3 now names the new wf1 rule's keys: output `climate_nc`, params `data_sources`/`clim_source`/`starttime`/`endtime` (matching the reused block); and notes the **input** key differs (`basin_nc`, per arch-1) so the wf1 rule uses a thin wf1-side caller that reads `sm.input.basin_nc` and calls with `bbox=`. wf3's verbatim `__main__` block is unaffected. | design-v2.md |
| arch-5 | r1 | minor | accepted | §5.3 corrected: "wf2 does not consume the catalog module at all; the sole consumer is wf3 rule 3.08 (`Snakefile_climate_experiment:315-337`)." The misleading "wf2 has no import…" phrasing is removed. | design-v2.md |
| repo-1 | r1 | major | accepted | §5.1 and §5.3 corrected: `test_climate_forcing.py` is named as a **second live caller** of `climate_forcing_by_subcatchment` (the v1 "none survive"/"assertions verbatim" claims are false). Under the selected Option B the function is unchanged, so the test's assertions **do** stay verbatim — it is repointed to the new import path in commit 1, and commit 4 (the reduction/re-source commit) runs the **full suite**, closing the latent-red-test gap. | design-v2.md |
| repo-2 | r1 | minor | accepted | §5.5 now names the manifest no-op as a **knowing, justified divergence** from intake decision 4's "manifest re-records the changed targets" wording (the changed `clim_*` plots are unmanifested, so the mechanism does not apply; the characterized diff carries the acceptance weight). The human gate must accept this consciously; OQ-4 tracks a future manifest extension. | design-v2.md |
| repo-3 | r1 | minor | accepted | §5.5 corrected: `outlet_index.csv` removed from the manifested-set parenthetical — it is a `rule all` target (`:60`) but **not** a `check_baseline` TARGET. The manifested model_creation set is the three PNGs + `snake_config_model_creation.yml` + `output.csv`. | design-v2.md |
| repo-4 | r1 | minor | accepted | §5.4 adds a note that the star-import shim's correctness depends on the moved modules carrying no `__all__`; the primary mitigation (also §8 commit 1) is to **repoint the moved-module tests to the new import paths in the move commit**, so the shim is never load-bearing for the suite. | design-v2.md |
| repo-5 | r1 | minor | accepted | §5.2, §7, and §8 commit 4 note that `plot_results` rule 1.10 declares the script twice (input label `script = ...` at `:217` + the `script:` directive at `:228`); the rule-1.10 input rewrite (drop `forcing_path`, add the raw netCDF) leaves the `script = ...` input label intact, since `plot_results.py` is not moved. | design-v2.md |
| ext1-1 | ext-r1 | blocking | accepted | §5.2 adds an explicit per-`clim_source` transformation pipeline table: what extraction already applied per branch (era5: nothing; chirps: temp lapse-corrected to the chirps-grid MERIT DEM, precip native, press/kin/kout resampled-only; eobs: unverified + guarded), what the wf1-side parity module still applies, and a net-correction ledger (temp lapse net 1 to the model DEM, pressure 1, precip 0). Chirps double-correction is structurally excluded: `dem_forcing` on that branch is mandatorily the extraction's orography sidecar, which `meteo.temp`'s MSL-shift exactly inverts before re-correcting to the model DEM. Exactly-once proven by a synthetic unit test (commit 3: era5 analytic + chirps sidecar-chaining) and the fixture grid-level check G vs `inmaps_historical.nc` (commit 5, tolerances named). | design-v3.md |
| ext1-2 | ext-r1 | blocking | accepted | §5.2 resolves the callables now, verified against the installed hydromt 1.3.1 / hydromt_wflow 1.0.2: `hydromt.model.processes.meteo.precip` (meteo.py:47), `.temp` (meteo.py:115), `.pet` (meteo.py:323) — the same functions the forcing build delegates to (wflow_sbm.py:3310/3667/3707) — with exact arguments, units, correction order, `dem_model = staticmaps.nc["land_elevation"]` (naming.py:10), per-branch `dem_forcing`, and the `setup_time_horizon.py:19-26` pet-method mapping; all plain-xarray, no model object. New `model/climate_parity.py` owns the transform. OQ-2 marked RESOLVED; the EP panel is retained unconditionally and the v2 "drop the panel if infeasible" escape hatch is deleted (it authorized an output change the milestone does not sanction). | design-v3.md |
| ext1-3 | ext-r1 | major | accepted | §5.5 defines the operational comparison ladder: persisted states S0 (`inmaps_historical.nc`), S1 (raw extraction; documented per-branch preprocessing, not a rung), S2 (`qa/l1_regrid_only.nc`, corrections off), S3 (`qa/l2_parity.nc`, full parity); aggregated rows A0/A1/A2 via the common reduction; components as adjacent-state deltas (`A2 − A1` correction component with interactions fixed by construction, exact precip null-check, `A2 − A0` sanctioned change, grid-level G before aggregation); on-demand drill-down rungs S3a/S3b; attribution rule: every residual assigned to a named edge. Produced by `dev/milestones/p32a/compare_climate_ladder.py` at commit 5; validation gates updated. | design-v3.md |
| ext1-4 | ext-r1 | major | accepted | Contract pinned per the reviewer's preferred option: dedicated wf1 wrapper `blueearth_cst/model/extract_climate_wf1.py` (imports `prep_historical_climate`, reads `sm.input.basin_nc` + the `climate_nc`/params keys, derives the bbox, calls with `bbox=`) is the wf1 rule's `script:` target; the moved module's wf3 `__main__` block stays verbatim with no conditional dispatch. §5.1 layout, §5.2 rule declaration, §5.3 contract, §8 commit 3, and the test surfaces (test_cli dry-run, bbox unit test, commit-6 e2e) all reconciled to the wrapper. | design-v3.md |
| ext1-5 | ext-r1 | minor | accepted | §5.2 defines the derivation (`ds_sm.raster.bounds` of `staticmaps.nc`, same tuple form as `region.geometry.total_bounds`), explains the relation (model grid snapped outward to `model_resolution`, ≈ ≤1 cell/edge + origin alignment), adds a commit-3 fixture assertion (per-edge ≤ 2×`model_resolution`, actuals recorded), argues insensitivity (extraction `buffer=1` source cell ≫ the shift), and adds the commit-5 empirical closure: fixture `wf1_raw` vs wf3 keyed-store `allclose` (same function, two bbox sources). | design-v3.md |
| ext2-1 | ext-r2 | major | accepted | Per user arbitration ruling (2026-07-24, final): fix required. §5.2 promotes the chirps orography sidecar to a **declared extraction-rule output** (`oro_nc = climate_historical/wf1_raw/orography.nc`, stable `clim_source`-independent path — the wf1 wrapper relocates the shared function's `{clim_source}_orography.nc` to it, shared module verbatim) and a **declared branch-resolved rule-1.10 input** (chirps: `sm.input.oro_nc`; era5: catalog `era5_orography` via params) — no sibling-file discovery. Branch resolved at DAG-parse time from the same config key in both rules, so Snakemake enforces sidecar existence and DEM/climate co-provenance on incremental/partially cleaned runs. §5.1/§5.3 wrapper contract, §7, §8 commits 3–4, §9 reconciled. | design-v4.md |
| ext2-2 | ext-r2 | major | accepted | Per user arbitration ruling (2026-07-24, final): fix required, **tolerance variant chosen**; the direct-hop reference-state rung explicitly not chosen and not added. §5.5 defines a defer-and-pin procedure: chirps acceptance is blocked until variable-specific tolerances (per-variable max-abs/RMSE on the `A2 − A1` / `A2 − A0` / G edges) are pinned from the first chirps-basin ladder run (the pinned fixture, config+inputs recorded), each justified against a stated hop-mechanism bound (one `nearest_index` cell displacement × local gradient; P exactly 0 — no hop for chirps precip); unexplained excess over the pinned tolerances fails the gate thereafter. §9 acceptance logic updated. | design-v4.md |
| ext2-3 | ext-r2 | minor | accepted | Per user arbitration ruling (2026-07-24, final): fix required, **exclude variant chosen**. eobs is scoped out of the P3-2a wf1 raw path: an early, loud DAG-parse-time configuration error in `Snakefile_model_creation` on `clim_historical: eobs` (named message; wrapper belt-and-braces assert) and a bounded support statement — era5, chirps, chirps_global only. No eobs pipeline specified (per ruling). §5.2 eobs table row records the exclusion in place of the v3 "unverified + pre-implementation check" contract; wf3's use of the shared module untouched (C2/C3). §8 commit 3 and §9 name the guard. | design-v4.md |
