# Consolidated review record — ADR 0001 (Wflow constant-parameter restoration)

Durable audit trail for the `design-review-loop` run `constant-pars-restoration`
(2026-07-21). The accepted design is
`dev/decisions/0001-restore-wflow-constant-parameters.md` (Status: accepted, =
design-v4). This record embeds the verdict arc, the internal aggregation index,
both external rounds verbatim, and the final finding ledger; the per-round
scratch run dir was pruned after landing (per the design-review-loop retention
policy — this record is the trail).

## Run summary

- **Variant:** full (promoted from lite on the first internal blocking finding).
- **Author binding:** `cst-architect`. Driver: interactive session (Opus).
- **Dispatches:** 6 Opus + 1 Fable (the Fable escalation = the r2 revision
  answering ext1-1, which re-raised internal risk-7). 1 spawn transport failure
  (idle timeout) recovered on ladder rung 1 (same-thread resume).
- **Gates:** G1 approved (framing carried from the t260719a scope calls +
  advisor review); G2 approved 2026-07-21 (approve & land).
- **Outcome:** converged under arbitration at the external round cap (2).

## Verdict arc

| Stage | Reviewer | Verdict | Findings | Resolution |
|---|---|---|---|---|
| Internal panel (on v1) | risk (`critical-thinker`) | revise | 1 blocking, 4 major, 2 minor | all accepted → v2 |
| Internal panel (on v1) | architecture (`model-validator`) | revise | 3 major, 1 minor | all accepted → v2 |
| Internal panel (on v1) | repo-fit (`python-engineer`) | revise | 1 blocking, 2 major, 4 minor | all accepted → v2 |
| External round 1 (on v2) | GPT (codex) | revise | 1 blocking, 4 major | all accepted → v3 (Fable) |
| External round 2 (on v3) | GPT (codex) | revise | 2 blocking, 1 major | arbitrated accepted → v4 |
| Arbitration (cap reached) | user | accept all 3 (fix required) | ext2-1/2/3 | → stage-6a → v4 |
| G2 (on v4) | user | **approve & land** | — | accepted as ADR 0001 |

**Totals:** 26 findings (2 internal blocking, 3 external blocking across 2 rounds,
the rest major/minor); every one accepted and resolved; 0 rejected. The core
decision (evidence-gated, fail-closed CSDMS restoration) was stable from v1 and
endorsed by the architecture lens; every round strengthened the *validation
protocol*, not the decision.

**Highest-value catches:** the whole v1 verification protocol targeted
`staticmaps.nc`, but constants land as TOML scalars (would have failed on a
correct build); `check_baseline.py` cannot express a tolerance-gated discharge
check (unscoped code work); the equivalence gate needed both-sides (0.x + 1.x)
evidence; and a build-sequence identity-comparison trap that would have falsely
scored every restoration "immaterial."

---

## Internal aggregation index

# Internal panel — aggregation index (design-v1.md)

Full internal panel (promoted lite→full on risk-1). Three lenses, all verdict
**revise**. 18 findings: **2 blocking, 9 major, 7 minor**. This index groups by
theme and preserves every original ID/severity by reference — it never deletes or
re-grades. Verbatim sources: `internal-review-risk.md`,
`internal-review-architecture.md`, `internal-review-repo-fit.md`.

Per-lens verdicts: risk = revise (1B/4M/2m); architecture = revise (0B/3M/1m);
repo-fit = revise (1B/2M/4m).

## ⚠ Pivotal fact to resolve FIRST — landing location (cross-lens conflict)

- **risk-1 [blocking]** — *empirically verified on the built model*:
  `setup_constant_pars` writes `input.static.<name>.value` to `wflow_sbm.toml`;
  the retained `KsatHorFrac=100` is in the TOML and **absent** from staticmaps.nc's
  50 variables (`wflow_base.py:1429-1461`). The whole v1 protocol (steps 3/5/7)
  targets staticmaps.nc → fails for a correct build.
- **repo-6 [minor]** — flags the same "and/or the TOML" ambiguity, unverified; asks
  it be pinned early.
- **arch-4 [minor]** — *assumes* the values land in staticmaps.nc.

**Conflict:** risk verified TOML; architecture/repo assumed/hedged staticmaps.
risk-1's built-model evidence governs — pin it empirically as an early sub-step,
then retarget landing assertion + fingerprint to the TOML scalar. This decision
cascades into the check_baseline cluster below.

## Theme A — check_baseline.py cannot express the verification (blocking)

- **repo-1 [blocking]** — `FINGERPRINTERS = {png, nc, csv, yaml}`: no TOML kind, no
  TOML-scalar reader; `csv` comparator is byte-exact SHA with **no tolerance**
  (only `nc` has `--tolerance`). The tolerance-gated discharge judgment has no code
  path. Step 7 is unscoped `check_baseline.py` work, not a manifest edit.
- **arch-2 [major]** — daily `output.csv` is full float64 (~17 sig-digits); exact
  SHA fails on any LSB/solver drift. No clean-build discharge reproducibility is
  established, so a match can't be attributed to the param change. (Counter-evidence:
  aggregated `Qstats.csv` already hashes cleanly in-env.)
- **arch-3 [major]** — no TOML handler; routing TOML through the yaml handler is
  wrong. Mitigant: built TOML uses only relative paths → a TOML hash is stable across
  the two clean project dirs.

## Theme B — discharge gate: undefined threshold + must be partitioned

- **arch-1 [major]** — step 6 "material/immaterial" has no metric and no numeric
  threshold → the written verdict is unfalsifiable.
- **risk-3 [major]** — the diff is a real test for the ~5 process-active params
  (cf_soil, EoverR, InfiltCapPath, MaxLeakage, rootdistpar) but structurally ~0 for
  the 8 snow/glacier params on the snow-free basin. Partition the protocol; don't
  present the measurement as evidence for the inert subset.

## Theme C — justification reframe

- **risk-2 [major]** — the "Deltares reference" provenance is mostly unsupported:
  8/13 values match the bundled hydromt_wflow 1.x user-guide default example verbatim
  (`docs/hydromt-wflow/user-guide.md:300-314`); only the glacier set + InfiltCapPath
  differ (G_Cfmax 3→5.3, G_SIfrac 0.001→0.002, G_TT 0→1.3, InfiltCapPath 10→5).
  Reframe restore justification to "pin documented defaults for drift-protection";
  the genuine choice is the 4 differing params.

## Theme D — missing alternative

- **risk-4 [major]** — the strongest alternative, **Defer** (baseline runs green;
  KsatHorFrac already retained; ~8/13 equal defaults), is absent. Add it and reject
  it on explicit grounds; soften the "restore-only-where-default-differs" rejection
  (the build already surfaces the default, so it's measured not enumerated).

## Theme E — manifest de-scope + blast radius

- **risk-5 [major]** — the manifest overhaul is the separable cross-cutting
  baseline-integrity followup; scope this task to the minimum that makes the move
  visible; defer the general expansion + cross-reference it.
- **repo-2 [major]** — `check_baseline record` is all-or-nothing across all three
  workflows (no `--workflow` on `record`, only on `check`); a wf1-only clean
  re-record hard-fails on missing wf2/wf3 targets → either run the full pipeline into
  the recorded dir, or add `--workflow` to `record` first. Pick one, put it in scope.
- **repo-3 [major]** — a *material* discharge move propagates into the manifest's
  wf3 targets `Qstats.csv`/`basin.csv`; "wf1's slice" undercounts blast radius.
- **repo-7 [minor]** — the new targets are build intermediates, not `rule all`
  targets; update the check_baseline docstring/comment so a maintainer doesn't prune
  them as stray.

## Theme F — remaining minors

- **risk-6 [minor]** — failure-mode (a) mischaracterized: the plugin **raises** on an
  unrecognized name (loud) and always writes a recognized name to the TOML. The real
  silent mode is **precedence** — a constant `input.static.X.value` ignored when a
  staticmaps layer exists for the same X, or an inactive-module key. Restate (a); add
  a precedence check + log it as an open question.
- **risk-7 [minor]** — units/sign/timestep equivalence across the rename is assumed,
  not verified; `naming.py` is name-to-name only. Add a one-line units check per
  restored param vs Wflow.jl 1.x docs, especially the 4 differing params + the
  g_ttm/g_tt collapse (footnote 1).
- **repo-4 [minor]** — `wflow_build_model.yml` lines 29-33 currently document the
  *opposite* (drop) decision; rewrite that comment to point at ADR 0001, else the
  config self-contradicts and invites a "re-drop."
- **repo-5 [minor]** — the landing assertion has no home (test_cli is dry-run only;
  test_model_creation doesn't build). Specify its form: one-time `dev/scripts/`
  verification script vs a permanent `--run-integration` test.

## Confirmed non-issues (repo-fit, for the record — no action)

- The 13 CSDMS config keys are convention-compliant (`naming.md` §6 tier-1
  domain-identifier exemption).
- No task-scope overlap: the manifest work is explicitly folded into t260719a per
  the pre-R6 tracker.


---

## External review — round 1 (verbatim, on design-v2)

```
## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: Decision / Validation protocol (implementation), step 3c
- finding: The ADR commits to restoring the old numeric values before establishing that each value retains the same units, sign convention, and semantics under its CSDMS/Wflow.jl 1.x name. The equivalence check is deferred to implementation, but the protocol defines no acceptance criteria or stop/reconsider branch if equivalence fails—particularly for the four genuinely differing values and the many-to-one `g_ttm`/`g_tt` mapping.
- rationale: A name mapping is not evidence of parameterization equivalence. Implementation could discover a unit or semantic mismatch yet still follow the approved decision and write a physically invalid value, producing silently wrong hydrology on process-active basins.
- suggested_fix: Complete and record the units/semantics analysis before approving the restoration decision, or make restoration explicitly conditional: define authoritative sources, required evidence per parameter, and a fail-closed outcome that returns any unverified or non-equivalent parameter to design-level reconsideration rather than restoring it.

### ext1-2  [major]
- section: Validation protocol (implementation), steps 5–6
- finding: The selected summary/rounded discharge fingerprint cannot implement the materiality contract it is said to share with step 5. Min/max/mean/std or rounded aggregate values do not test paired timestep differences, so they cannot enforce either `max_t |Q_restored-Q_reference| / mean(Q_reference)` or “any single-timestep relative change > 1%.”
- rationale: A timing shift or compensating positive and negative deviations can preserve the selected summaries while exceeding the per-timestep threshold. The durable baseline check could therefore pass a discharge regression that the ADR itself classifies as material.
- suggested_fix: Select the time-index-aligned numeric CSV comparator, define zero/near-zero reference handling, missing/duplicate timestamp behavior, and exact absolute/relative tolerance logic, then use that same comparator for reproducibility, materiality classification, and durable baseline checking.

### ext1-3  [major]
- section: Decision / Validation protocol (implementation), steps 2–5
- finding: The design repeatedly promises per-parameter effects and default equivalence, but the experiment changes all 13 restored parameters simultaneously and measures only their net discharge effect. That design cannot identify an individual parameter’s contribution, prove that any value equals the engine default, or detect cancellation among parameter effects.
- rationale: A net-null result could arise from inactive processes, equal defaults, weak sensitivity, or offsetting nonzero effects. Classifying it as an “immaterial” no-op or claiming a parameter has a “net-zero discharge contribution” would overstate the evidence and could conceal material but cancelling changes.
- suggested_fix: Either restrict all conclusions to the aggregate restored set and remove per-parameter/default-equivalence claims, or add controlled one-at-a-time/grouped experiments for parameters whose behavioral equivalence is consequential.

### ext1-4  [major]
- section: Validation protocol (implementation), step 7 / Consequences
- finding: The baseline-recording protocol does not close the downstream wf3 case it explicitly identifies. A wf1-filtered `record --workflow` preserves existing wf2/wf3 rows, while a material wf1 change is expected to alter wf3 `Qstats.csv` and `basin.csv`; nevertheless, the steps never require rerunning wf3 and recording its affected rows.
- rationale: After a material restoration, the manifest will either retain stale wf3 fingerprints and fail the next wf3 check, or the claimed multi-workflow re-baseline will not actually have occurred. The design’s stated consequence and executable landing procedure disagree.
- suggested_fix: Branch on the step-5 result: if material, run the required downstream workflow, review its diffs, and record both wf1 and affected wf3 slices; if immaterial, record wf1 only. Specify failure handling when downstream outputs move unexpectedly.

### ext1-5  [major]
- section: Context / Decision / Parameter reconciliation
- finding: The accounting and justification omit one restored parameter. The ADR restores 13 parameters and identifies four as differing, which leaves nine non-differing restorations, yet it repeatedly justifies only “8 template-equal parameters.” `MaxLeakage` is marked “equal” in the table while its footnote says only that it is “very likely” the default, and it is not included in the cited eight-parameter template match.
- rationale: `MaxLeakage: 0` is therefore restored without the documented-default evidence used to justify the other pinned values. If the Wflow.jl 1.x default or semantics differ, it becomes another genuine scientific reference choice and may alter active-basin hydrology without being recognized as such.
- suggested_fix: Treat `MaxLeakage` explicitly: verify its Wflow.jl 1.x default and semantics from an authoritative source, then classify it either as the ninth proven default-equal pin or as a fifth differing reference choice; correct all counts and associated rationale accordingly.
```

---

## External review — round 2 (verbatim, regression duty, on design-v3)

```
## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1  [blocking]
- section: Decision / Validation protocol (implementation), step 3c
- finding: The purported resolution of ext1-1 still cannot establish cross-version equivalence as specified. The gate names only the Wflow.jl 1.x parameter documentation as its authoritative source, yet concluding that units, sign convention, scale, and semantics were “preserved” also requires authoritative evidence for what each 0.x short-name value meant. The old template commit establishes the numbers and `naming.py` establishes name mappings, but neither establishes the 0.x physical contract.
- rationale: An implementer can satisfy the written evidence procedure using only 1.x definitions while implicitly assuming the missing 0.x side of the comparison. A changed unit, timestep basis, sign, or parameterization could therefore pass the nominal gate and ship physically invalid values—the exact silent-wrong-hydrology failure ext1-1 required the gate to prevent.
- suggested_fix: Require authoritative evidence for both sides of every mapping: cite the applicable 0.x Wflow documentation/code contract and the 1.x contract, record both definitions and the comparison per parameter, and fail closed when either side is unavailable. Correct the asserted restored-count range as well: because every parameter is conditional, the protocol cannot promise 12–13 unless evidence for the other twelve has already been established.

### ext2-2  [blocking]
- section: Validation protocol (implementation), steps 2 and 4
- finding: The build sequence does not preserve an executable current-baseline configuration. Step 2 edits `config/wflow_build_model.yml` to add the gate-passing restored parameters and builds the restored model; step 4 then says to build the “current config” without specifying an immutable pre-edit copy, revision, or alternate config path. Following the steps literally builds the restored configuration again as the reference.
- rationale: The restored-versus-reference discharge comparison can become an identity comparison, falsely classify the restoration as immaterial, and record the moved output as though it had been compared against the actual M2b baseline.
- suggested_fix: Snapshot or materialize two explicit build configurations before either build—for example, an unchanged pre-restoration baseline config pinned to the current revision and a restored candidate config—and require the build commands, output directories, and recorded provenance to identify which config produced each model.

### ext2-3  [major]
- section: Validation protocol (implementation), step 7
- finding: The ext1-4 fix requires wf3 diffs to have the “same direction,” a “commensurate magnitude,” and no movement outside the “model-dependent set,” but it never defines those tests for `Qstats.csv` and `basin.csv`. Those files contain derived indicators for which direction and proportional magnitude need not match daily outlet discharge uniformly, and the model-dependent field set is not enumerated.
- rationale: Two implementers can reach opposite re-record decisions from the same outputs; a genuine downstream regression may be blessed as coherent, or a physically plausible indicator response may block landing indefinitely. The promised stop/re-record arbiter is therefore not operationally reproducible.
- suggested_fix: Enumerate the wf3 columns/indicators to compare, define expected relationships or quantitative review bounds where defensible, identify fields expected to remain invariant, and require unexplained violations to stop re-recording. Where no directional or magnitude expectation is scientifically defensible, require documented causal inspection instead of asserting one.
```

---

## Final finding ledger (all 26)

# Finding ledger — constant-pars-restoration (ADR 0001)

Append-only. One row per original finding ID. Round `internal-r1` = the full
internal panel (risk / architecture / repo-fit) on design-v1.md, resolved in
design-v2.md.

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| risk-1 | internal-r1 | blocking | accepted | Landing corrected: `setup_constant_pars` writes `input.static.<name>.value` as a TOML scalar (`wflow_base.py:1461`), verified on the built `wflow_sbm.toml` (KsatHorFrac at lines 82–83; absent from staticmaps.nc). v2 Context states this; steps 1b/3/5/7 retarget every landing assertion, attribution, and fingerprint to the TOML scalar + discharge; staticmaps.nc dropped from constant-pars logic. | design-v2.md |
| risk-2 | internal-r1 | major | accepted | Justification reframed. 8/13 values equal the hydromt 1.x doc template (`user-guide.md:300–314`) → benefit is drift-protection + self-documentation, not "Deltares-tuned." Only 4 differ (InfiltCapPath 5 vs 10; G_Cfmax 5.3 vs 3; G_SIfrac 0.002 vs 0.001; G_TT 1.3 vs 0) → named as the real reference choice. "Deltares-tuned" overclaim removed. | design-v2.md |
| risk-3 | internal-r1 | major | accepted | Protocol partitioned by test-basin process activity. Refined against built TOML (`snow__flag=true`, `glacier__flag=false`): snow set is process-ACTIVE here (discharge diff is a real test), glacier set is INERT (certified by landing + cross-basin continuity, not discharge). Step 5 reads discharge only for the active subset. | design-v2.md |
| risk-4 | internal-r1 | major | accepted | **Defer** added as a genuinely-considered 4th alternative and rejected on explicit grounds (mapping/precedence/units surface cheapest to nail now; the 4 differing params are an unmade choice; drift-protection accrues only once pinned). "Restore-only-where-default-differs" rejection softened: builds already MEASURE the default, so the objection is drift-protection + self-documentation, not enumeration. | design-v2.md |
| risk-5 | internal-r1 | major | accepted | Manifest de-scoped to the minimum that makes THIS move visible = discharge target only. General staticmaps/TOML manifest expansion deferred to the standalone baseline-manifest-integrity followup (`dev/followups.md`) and cross-referenced in step 7 + Related. | design-v2.md |
| risk-6 | internal-r1 | minor | accepted | Failure-mode (a) restated: plugin RAISES `ValueError` on unknown names (`wflow_base.py:1449`, loud) — accepted-but-unwritten is impossible. Real silent mode = precedence/inactivity (scalar shadowed by a staticmaps layer, or inactive module e.g. `glacier__flag=false`). Step 3b adds a precedence check + logs the "does Wflow.jl read an inactive-module scalar?" open question. | design-v2.md |
| risk-7 | internal-r1 | minor | accepted | Per-param units/semantics equivalence check added (step 3c) vs Wflow.jl 1.x docs, since `naming.py` (`WFLOW_NAMES`) is name-to-name only with no units/scale field. Flagged especially for the 4 differing params and the g_ttm/g_tt → single-v1-name collapse (footnote 1). | design-v2.md |
| arch-1 | internal-r1 | major | accepted | Concrete materiality metric + threshold defined (step 5): `material := max_t |Q_r − Q_ref| / mean(Q_ref) > 1e-3`, OR any single-timestep relative change > 1%; else recorded as solver-noise no-op. Same numeric form as the step-6 comparator so classification and gate agree. | design-v2.md |
| arch-2 | internal-r1 | major | accepted | One-time clean-build discharge reproducibility check added (step 4b: build current config twice, confirm discharge matches at tolerance) so the later restored-vs-reference diff is attributable, not solver/env noise. Discharge fingerprinted in a tolerance-bearing summary/rounded form, not exact SHA. | design-v2.md |
| arch-3 | internal-r1 | major | accepted | No TOML fingerprinter added (avoids routing TOML through the yaml handler, which is wrong). The TOML scalar is verified by the step-3 landing assertion, not the manifest; `check_baseline` gains only the discharge target. | design-v2.md |
| arch-4 | internal-r1 | minor | accepted | Certification boundary named (step 5 + Neutral consequence): discharge certifies the net test-basin effect only; the 3 glacier params (inert here) and any other basin are certified by landing assertion + cross-basin reference-continuity, not by this discharge sensitivity. | design-v2.md |
| repo-1 | internal-r1 | blocking | accepted | `check_baseline.py` change scoped explicitly as CODE work (step 6): `FINGERPRINTERS={png,nc,csv,yaml}` has no TOML kind and `csv`→`diff_hashed` has no tolerance (only `diff_nc`). v2 selects a numeric-tolerant discharge comparator (summary/rounded fingerprint under the `--tolerance` path; per-timestep tolerant CSV differ as fallback) and adds no TOML fingerprinter. Named as FINGERPRINTERS/TARGETS code change, not a manifest edit. | design-v2.md |
| repo-2 | internal-r1 | major | accepted | `record` all-or-nothing owned: `cmd_record` calls `compute_manifest(project_dir)` with no filter and refuses on missing targets; `--workflow` is `check`-only. Scope decision: add `--workflow` to the `record` subcommand so a wf1-slice re-record avoids a full CMIP6+Julia wf3 run (step 7). | design-v2.md |
| repo-3 | internal-r1 | major | accepted | Blast radius owned in Consequences (Negative): wf3 reuses the built model, so a MATERIAL discharge move propagates into manifest targets `Qstats.csv`/`basin.csv` (TARGETS lines 71–72) and re-records them — a multi-workflow move, not "wf1's slice." Conditional on step-5 materiality. | design-v2.md |
| repo-4 | internal-r1 | minor | accepted | Step 2 instructs rewriting the `wflow_build_model.yml` lines 29–33 comment (which currently documents the opposite drop decision) to a one-line pointer to ADR 0001, so the config does not self-contradict and invite a re-drop. | design-v2.md |
| repo-5 | internal-r1 | minor | accepted | Landing-assertion form specified (step 3): a one-time `dev/scripts/` verification script recorded in the baseline-diffs note — not a permanent `--run-integration` test — because `test_cli.py` is dry-run only and `test_model_creation.py` does not build. | design-v2.md |
| repo-6 | internal-r1 | minor | accepted | Early empirical landing-pin sub-step added (step 1b): inspect the built `wflow_sbm.toml` for one representative param before writing assertions. Resolved to the TOML scalar (already verified for KsatHorFrac), foreclosing the staticmaps path and confirming no `toml` fingerprinter is needed. | design-v2.md |
| repo-7 | internal-r1 | minor | accepted | Step 7 instructs updating the `check_baseline.py` module docstring / adding a row comment noting the discharge target is deliberately fingerprinted beyond `rule all` (which lists only the 3 PNGs + config + outlet_index.csv) for parameter-preservation coverage, so a maintainer does not prune it as stray. | design-v2.md |
| ext1-1 | external-r1 | blocking | accepted | Restoration made explicitly conditional and fail-closed (Decision "Equivalence gate" + step 3c, executed BEFORE the step-2 restored build): authoritative source = Wflow.jl 1.x parameter docs (bundled tables carry names only); required evidence per param = units + sign + semantics preserved under the CSDMS name, recorded in a gate log; any unverified/failed param is NOT restored — removed from the build set and returned to design-level reconsideration via a decision addendum. Priority gate cases named: the 4 differing params, the g_ttm/g_tt collapse, and MaxLeakage. Restored count now stated as 12–13, resolved at the gate. | design-v3.md |
| ext1-2 | external-r1 | major | accepted | Step-6 selection switched from the summary/rounded fingerprint (v2 option b, withdrawn — summaries cannot enforce the per-timestep contract; timing shifts/compensating deviations slip through) to the time-index-aligned numeric CSV comparator (option a), fully specified: parse `output.csv` on the timestamp index; structural FAIL on duplicate timestamps, unequal index sets, or NaN; per-timestep `fail(t) := |ΔQ|>ATOL OR (Q_ref≥ATOL AND |ΔQ|>RTOL·Q_ref)` with `ATOL=1e-3·mean(Q_ref)`, `RTOL=1%`, and the relative clause skipped where `Q_ref<ATOL` (zero/near-zero handling). One implementation drives 4b reproducibility, step-5 materiality, and the step-7 durable check; `record` stores a reduced reference series under `dev/baseline/` so `check` compares per-timestep. | design-v3.md |
| ext1-3 | external-r1 | major | accepted | All discharge conclusions restricted to the aggregate restored set: the "provably equals the 1.x default (net-zero discharge contribution)" sentence and MaxLeakage's expected-net-zero footnote claim removed; an aggregate-only note added to the protocol stating a net-null is indistinguishable among inactive processes, equal defaults, weak sensitivity, and offsetting effects, and that per-parameter attribution would require one-at-a-time/grouped runs (out of scope). Default-equality claims now rest solely on documentation (template match + gate). The "restore-only-where-default-differs" alternative's measured-per-parameter rationale rewritten to match. | design-v3.md |
| ext1-4 | external-r1 | major | accepted | wf3 branch operationalized in step 7: MATERIAL → run `Snakefile_climate_experiment` on the restored build (reusing wf2 artifacts), review `Qstats.csv`/`basin.csv` diffs for coherence with the wf1 move (direction, commensurate magnitude, no unrelated targets moving), then record BOTH wf1 and affected wf3 slices under the merge semantics; IMMATERIAL → record wf1 only. Failure handling specified: incoherent/unexpected downstream moves → stop, do not re-record, investigate (stale intermediates, weathergenr state, env drift), return to decision level if unexplained; plus a recovery path for byte-exact wf3 fingerprints perturbed later by a sub-tolerance wf1 move. Consequences (blast radius) updated to match the executable procedure. | design-v3.md |
| ext1-5 | external-r1 | major | accepted | Accounting corrected: 13 restored = 8 template-equal + 4 differing + 1 unverified. Verification attempted per the suggested fix: the bundled Wflow 1.x docs map only the name (`docs/wflow-user-guide/02-required-files.md:73`) with no default/units, so MaxLeakage's 1.x default is UNVERIFIED in-repo → reclassified fail-closed through the ext1-1 gate (default equal to 0 → ninth default-equal pin; different → decision addendum as a fifth reference choice; unresolved → not restored). Table row (`RESTORE (gate-conditional)`, `absent — UNVERIFIED`), footnote 2, and every 8/4 count in Context/Decision corrected; verifying the default against the upstream Wflow.jl 1.x parameter docs is a required gate input at implementation. | design-v3.md |
| ext2-1 | external-r2 | blocking | accepted (arbitration 2026-07-21) | Equivalence gate made two-sided. Decision "Equivalence gate (fail-closed, two-sided)" and step 3c now require authoritative evidence for BOTH sides of every mapping: the 0.x side (applicable hydromt_wflow 0.x / legacy Wflow parameter doc/code contract — units/sign/scale/timestep basis of the short-name value as authored under git `6ebc46f`; not establishable from the template commit or `naming.py`) AND the 1.x side (Wflow.jl 1.x parameter docs). Both definitions and the per-parameter comparison are recorded in the gate log; a 1.x-only entry is incomplete and fails that parameter closed. Fail-closed when EITHER side is missing/ambiguous/negative. Restored-count claim corrected everywhere (step 3c + Consequences Negative + footnote 2): count is resolved at the gate and recorded (0–13), not promised — the prior "12–13" removed since every parameter is gate-conditional and the other twelve's evidence is not pre-established. Related source-of-truth list adds the 0.x parameter reference. | design-v4.md |
| ext2-2 | external-r2 | blocking | accepted (arbitration 2026-07-21) | Build sequence made a two-config comparison, never an identity comparison. Steps 2 and 4 now pin TWO distinct, separately-materialized build configs before either build: `config_baseline` (the unchanged in-repo `config/wflow_build_model.yml` snapshotted to a distinct path and pinned to the current git revision — the real M2b baseline, never edited by this task) and `config_restored` (a distinct candidate file adding the gate-passing subset; the lines 29–33 comment rewrite happens here, not in the snapshotted file). Step 2 builds from `config_restored`, step 4 builds from the pinned `config_baseline`, step 4b reproducibility uses `config_baseline` twice; each build records which config (path + git rev) produced it in provenance. Comparison is explicitly `config_restored`-output vs `config_baseline`-output, never the edited file against itself; Decision bullet updated to name the two pinned configs. | design-v4.md |
| ext2-3 | external-r2 | major | accepted (arbitration 2026-07-21) | wf3 coherence tests defined for `Qstats.csv`/`basin.csv` in step 7's material branch (item 2) and reflected in the Consequences blast-radius bullet. Now: (1) establish the model-dependent field set (columns that are a function of the Wflow build vs perturbation/config/metadata); columns outside it are expected-invariant and any movement is a re-record stopper. (2) Per-indicator expectations only where defensible: monotonic-functional flow-magnitude indicators (mean/high-flow/max) expect same sign as the wf1 move and magnitude within `[0, k·Δ_wf1_rel]`, `k=3` documented slack; low-flow/quantile/timing indicators that are NOT monotonic functionals of mean discharge get NO sign/magnitude assertion — they require documented causal inspection tracing the move to the wf1 hydrograph. `basin.csv` columns classified into model-dependent (functional/causal rule) vs invariant. The concrete column→class assignment is enumerated in the implementation brief and recorded in the baseline-diffs note so the arbiter is reproducible; ambiguous columns default to causal-inspection. (4) Unexplained violations (sign reversal, bound exceedance, invariant-field movement, unexplained no-expectation move) STOP re-recording, keeping the existing stop-and-investigate rule. | design-v4.md |
