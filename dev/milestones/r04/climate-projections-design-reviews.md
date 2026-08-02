# R04 climate-projections design — review record

Consolidated audit trail of the design-review-loop run
`r04-climate-projections` (2026-07-19 → 2026-07-20) that produced
`climate-projections-design.md`. Loop: author draft → G1 (framing approved) →
3-lens internal panel → revision → 3 external cross-vendor (GPT via headless
`codex exec`, read-only sandbox) rounds with regression duty → round cap →
user arbitration on the 2 surviving majors → G2 (approved). Version series
design-v1..v5; v5 = the landed text. All 24 findings accepted; none rejected,
none deferred.

| Review | Doc version | Verdict | blocking / major / minor |
|---|---|---|---|
| Internal panel: risk (critical-thinker) | design-v1.md | revise | 0 / 4 / 2 |
| Internal panel: architecture (cst-architect) | design-v1.md | revise | 1 / 1 / 2 |
| Internal panel: repo-fit (python-engineer) | design-v1.md | revise | 0 / 3 / 2 |
| External round 1 (GPT, clean-room) | design-v2.md | revise | 1 / 3 / 0 |
| External round 2 (GPT, + ledger/index, regression duty) | design-v3.md | revise | 1 / 2 / 0 |
| External round 3 (GPT, cap round) | design-v4.md | revise | 0 / 2 / 0 |
| Round-cap arbitration (user, 2026-07-20) | design-v5.md | ext3-1, ext3-2 accepted with fix-in-v5 | — |

---

## Internal review index (driver aggregation)

# Internal review index — r04-climate-projections, panel on design-v1.md

Driver aggregation (2026-07-19) of the three lens reviews. Original IDs,
severities, and verbatim text live in the immutable per-lens files —
`internal-review-risk.md`, `internal-review-architecture.md`,
`internal-review-repo-fit.md`. This index groups; it never deletes or
re-grades.

## Verdicts

| Lens | Verdict | doc_version | blocking | major | minor |
|---|---|---|---|---|---|
| risk (`critical-thinker`) | revise | design-v1.md | 0 | 4 | 2 |
| architecture (`cst-architect`) | revise | design-v1.md | 1 | 1 | 2 |
| repo-fit (`python-engineer`) | revise | design-v1.md | 0 | 3 | 2 |
| **Panel total** | **revise** | design-v1.md | **1** | **8** | **6** |

## Grouped findings

### Group A — end-to-end baseline gate is self-defeating as written

- **arch-1 [blocking]** (Verification plan): the "clean dedicated
  project dir" gate turns the config-snapshot target red —
  `fingerprint_yaml` hashes full parsed YAML including the embedded
  `project.project_dir`, and the manifest was recorded at
  `examples/test_local`. The design's central clean-gate thesis fails at
  its own verification step. Fix: run the gate at `examples/test_local`
  (seeded clean from the committed config) or re-record deliberately;
  correct the "path-independent yaml" claim.

### Group B — the refuted `AGENTS.md` ruleorder claim is left standing (3 findings, one defect)

- **risk-4 [major]**, **arch-2 [major]**, **repo-3 [major]** (§3 /
  commit plan / scope claim): R4's own evidence refutes the `AGENTS.md`
  "wildcard inference is ambiguous without it" invariant, yet no commit
  updates `AGENTS.md`, and the stated scope ("only workflow-2 files plus
  tests/ and dev/") rules the file out of bounds. Canonical instruction
  file left asserting a claim the milestone proved false; arch-2
  independently reproduced the clean 19-job dry-run. Fix (all three
  converge): add an `AGENTS.md` one-line correction to the ruleorder
  commit (or seal), and fix the "most cross-cutting: none" claim.

### Group C — attrs ruling framing and incentives (distinct findings, same subject)

- **risk-1 [major]** (Candidate #1 / stance): "behavior-preserving"
  silently means "regression-preserving" for the M2b attrs — the
  documented-broken `{}` state is treated as reference truth; a green
  gate is privileged over a correct output. Fix: name the deferral a
  scientific-cost decision and make the seal flag the encoded
  regression.
- **risk-2 [major]** (Candidate #1 conditional path): absorption is
  gated on the audit's own success at localizing — rewarding an
  under-characterized audit. Fix: make localization an unconditional §5
  audit obligation; make absorption a separately-argued scope call.

### Group D — split-don't-absorb needs a forcing function

- **risk-3 [major]** (classification policy / seal): a defect found by
  the audit leaves R4 sealed green with the defect parked ownerless;
  seal language cannot distinguish "audited, correct" from "audited,
  broken, deferred". Fix: seal enumerates deferred defect-class findings
  in the roadmap R4 section; every split names an owner + activation
  condition.

### Group E — §7 test plan names untestable or nonexistent units

- **repo-1 [major]**: `get_change_climate_proj.py` is also not
  import-clean (module-level `snakemake.params.*` at lines 178–187);
  3 of 5 named test targets cannot be imported under the commit plan;
  `_to_str_tuple` is defined *after* the exec block and must be
  relocated. Fix: charter a second import-guard commit for that script.
- **repo-2 [major]**: `list_files_not_empty` is a local list variable,
  not a function — the named unit does not exist. Fix: charter a small
  extract (`filter_nonempty`-style helper) or drop the unit claim.

### Minors (author's discretion, dispositions still required)

- **risk-5 [minor]**: PNG ±10% size-only tolerance is the gate's weakest
  link — state it in the stance section.
- **risk-6 [minor]**: Goal cites `TARGETS` lines 59–66 vs actual 53–71
  (workflow-2 subset 60–66).
- **arch-3 [minor]**: "14/14 clean after every commit" overclaims —
  per-commit gate is dry-run + test_cli; the baseline check is
  milestone-level.
- **arch-4 [minor]**: `copy_config` exclusion contradicts §2's own
  criterion; its output is the fingerprinted yaml target.
- **repo-4 [minor]**: sibling guard pattern is nested
  `if __name__ == "__main__":` → `if "snakemake" in globals():`, not
  either/or.
- **repo-5 [minor]**: commits 3 and 5 re-touch the same region of
  `get_stats_climate_proj.py`; sequence guard-then-wrap or merge.

## Driver notes

- All three lenses independently verified (and did not dispute) the
  design's empirical ruleorder finding; the panel's objection is to the
  uncorrected `AGENTS.md`, not the dry-run evidence.
- Repo-fit explicitly cleared: `tee_to_log` implementation/tests, the
  §6 naming claims, the 7-target manifest slice, and the attrs-out
  ruling's internal consistency.


---

## External review — round 1 (verbatim)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: Verification plan
- finding: The milestone-level baseline procedure cannot produce the promised 14/14 result as written: it directs implementation to wipe `examples/test_local`, run only workflows 1 and 2, and then invoke a checker whose target set spans all three workflows.
- rationale: `check_baseline.py::compute_manifest` treats absent recorded targets as failures, and its `TARGETS` includes workflow-3 outputs. After the prescribed wipe and wf1 → wf2 sequence, those workflow-3 targets will be missing, so `check_baseline.py check` must fail even when every R4 output is correct. Retaining old workflow-3 files instead would avoid the immediate failure but contradict the clean-seed instruction and weaken freshness of the gate.
- suggested_fix: Either run workflows 1 → 2 → 3 before the full 14-target check, or add and document a workflow-scoped check mode and use it for the seven workflow-2 targets. State separately how wf1/shared-helper regressions are covered.

### ext1-2  [major]
- section: The change-factor chain audit (the scientific deliverable)
- finding: The audit enumerates important questions but lacks mandatory executable evidence and pass/fail criteria for the highest-risk scientific behaviors. In particular, calendar testing is conditional on the reading audit “flagging” a problem, while missing-variable intersection, partial-member loss, and hydrological-year boundary behavior can all be declared acceptable from inspection alone.
- rationale: A reviewer can complete the findings table without exercising 360-day, noleap, and Gregorian/cftime boundaries or asserting behavior for partial members and asymmetric variables. The milestone could therefore seal as “audited, no defects” while the identified silent month-dropping or variable-dropping paths remain untested, materially weakening the claimed correctness audit.
- suggested_fix: Define a minimum audit evidence matrix with synthetic executable checks for supported calendar classes, non-January hydrological-year boundaries, partial-member absence, dummy datasets, and asymmetric hist/future variables. Specify expected outcomes and require each findings-table row to cite either a passing test or a concrete traced result.

### ext1-3  [major]
- section: `ruleorder:` — reproduce-or-refute at implementation, then comment the Snakefile AND correct `AGENTS.md`
- finding: The reproduce-or-refute gate is not operationally decidable: “any supported config” is undefined, the search procedure has no bounded configuration space, and failure to reproduce permits either retaining or removing the directive.
- rationale: Two implementers can follow the design and make opposite DAG-semantics decisions from the same evidence. Removing the directive after testing only convenient configurations can expose an `AmbiguousRuleException` for an untested supported model/scenario/horizon shape; retaining it and rewriting `AGENTS.md` as stale insurance leaves the milestone’s stated invariant unresolved.
- suggested_fix: Define the supported configuration-shape matrix and a deterministic decision rule. For example: retain and accurately comment the directive unless a named exhaustive/representative matrix plus `test_cli` passes without it; removal requires encoding the ambiguity-sensitive cases as regression tests. Specify the exact `AGENTS.md` text for each outcome.

### ext1-4  [major]
- section: M2b attrs localization (unconditional obligation of this audit)
- finding: The design mandates that the metadata-loss origin be pinned, but characterizes localization as a zero-cost reading finding and provides neither a diagnostic procedure nor an evidentiary threshold for distinguishing workflow code, `open_mfdataset` behavior, and hydromt behavior.
- rationale: Attribute propagation is runtime- and operation-dependent; source inspection alone may not identify where attrs disappear. Without checkpointed datasets or a minimal reproducer, the implementation can only speculate or become blocked by the design’s rule that “cause not localized” is unacceptable, delaying the milestone or producing an unsupported attribution that drives the wrong follow-up fix.
- suggested_fix: Require a minimal diagnostic that records attrs after per-model change generation, after preprocessing/opening, and after merge/write under the pinned environment. Define localization as the first stage where expected attrs disappear, and permit a documented “dependency reproducer” outcome when the loss occurs in an isolated hydromt/xarray operation.

---

## External review — round 2 (verbatim)

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1  [blocking]
- section: End-to-end baseline gate — the decidable procedure
- finding: ext1-1 remains unresolved because filtering only the source `TARGETS` list does not create a coherent scoped check against the existing full manifest. `cmd_check` still loads all 14 entries into `rec_targets`; with a reduced `current`, it silently skips the unselected recorded entries at lines 263–265 and reports `len(rec_targets)` as 14, contradicting the promised “OK — 11 target(s).” The repeatable option on `record` is more dangerous: it would write a partial manifest to the canonical manifest path unless separate storage semantics are specified.
- rationale: The designed check can claim 14 targets matched after comparing only 11, while a scoped `record` can replace the authoritative full baseline with a partial one. This makes the milestone gate misleading and can corrupt the baseline used by later milestones.
- suggested_fix: Define one selected target universe and apply it symmetrically to both current and recorded targets before missing, diff, orphan, and count logic. Preserve unselected manifest entries during scoped `record`, require a separate manifest path, or omit scoped `record` entirely. Add tests proving the reported count is 11, selected missing targets fail, unselected targets are ignored, and scoped recording cannot truncate the canonical manifest.

### ext2-2  [major]
- section: Unit + integration tests, and the audit-evidence matrix
- finding: ext1-2 is only partially resolved: rows V and P do not specify hand-computed pass/fail outcomes despite the matrix’s repeated claim that every row does. Their expected results are merely “records observed set” and “records observed behavior,” so silent variable loss or biased member alignment can satisfy the test by being observed.
- rationale: An implementation can encode whatever the current code does as the assertion, then seal the milestone without deciding whether asymmetric variables or members constitute acceptable behavior or a defect. The highest-risk missing-data paths therefore remain characterization exercises rather than executable audit criteria.
- suggested_fix: State the normative expected behavior for asymmetric variables and members, including whether silent intersection is permitted, whether absent members must fail or be explicitly aligned, and the exact surviving coordinates/values. If the current behavior differs, require the corresponding matrix row to fail and produce a defect-class finding.

### ext2-3  [major]
- section: M2b attrs diagnostic probe
- finding: ext1-4 remains incomplete because the first checkpoint is already after the entire per-model change calculation and write. If attrs are empty there, the prescribed probe localizes the loss only to a large upstream region, yet the design simultaneously requires a single workflow-2 operation or isolated dependency reproducer and supplies no checkpoints within that region. The synthetic alternative exercises only the merge path and cannot localize loss during per-model generation.
- rationale: A realistic result—attrs absent at checkpoint 1—leaves the implementer unable to satisfy the mandated evidentiary threshold using the specified procedure. The milestone can either block indefinitely or attribute the defect to an unverified operation, leading to the wrong follow-up or fix.
- suggested_fix: Add a checkpoint containing known expected attrs before change computation, then inspect after each potentially attr-dropping transformation, immediately before `to_netcdf`, and after reopening the per-model file. Require an isolated reproducer for the first transformation where attrs disappear; retain the existing merge checkpoints for losses occurring after per-model generation.

---

## External review — round 3 (verbatim)

## Verdict
verdict: revise
doc_version: design-v4.md

## Findings
### ext3-1  [major]
- section: Unit + integration tests, and the audit-evidence matrix
- finding: ext2-2 remains incompletely resolved because rows V and P still do not choose a single normative behavior. The prose permits either failing loudly or explicitly reconciling asymmetry, while row V asserts that both variables appear in the result and row P asserts only an “explicit/declared” member coordinate. Those assertions encode neither a valid fail-loud contract nor concrete reconciliation values.
- rationale: A correct future fix that raises on asymmetric variables or members will remain an expected `xfail`, not become the strict `xpass` that is supposed to trip the gate. Conversely, making both variables “survive” when future `temp` data do not exist leaves undefined what values should be produced. The tests therefore cannot reliably distinguish the current silent-loss defect from all permitted fixes, and the split task can close without causing the durable evidence gate to demand deliberate test resolution.
- suggested_fix: Choose the required policy separately for variables and members. If fail-loud is the norm, specify the exception type/message and make the normative test assert it; if reconciliation is the norm, specify the exact output coordinates and values, including missing-value handling. Keep any current-behavior `xfail` separate from these normative acceptance tests and name the condition under which it is removed.

### ext3-2  [major]
- section: M2b attrs diagnostic probe
- finding: ext2-3 remains incompletely resolved because several advertised checkpoints still combine multiple potentially attr-dropping transformations. P3 groups statistic/quantile reduction with arithmetic, while P4 groups `assign_coords`, `expand_dims`, and `xr.merge`; the first empty checkpoint therefore cannot identify the single transformation for which the design immediately requires an isolated reproducer. The design also says checkpoints run “inside” the production function without specifying whether this is temporary instrumentation, a diagnostic helper, debugger tracing, or a committed code change.
- rationale: If attrs disappear at P3 or P4, the prescribed evidence still localizes only to a multi-operation region, forcing the implementer to improvise additional experiments before meeting the design’s own operation-level threshold. Instrumenting the production function ad hoc can also introduce an unplanned computational-code edit into a milestone that claims the probe is diagnostic-only and output-neutral.
- suggested_fix: Split P3 and P4 into one checkpoint per reduction, arithmetic operation, coordinate operation, and merge, or define a deterministic bisection procedure that isolates the first attr-dropping operation. Specify the probe mechanism and artifact ownership explicitly—for example, test-only diagnostic helpers or temporary instrumentation that must be removed and verified by diff before the milestone gate.

---

## Findings ledger (final)

# Findings ledger — r04-climate-projections

Revision r1 (design-v1.md → design-v2.md). All 15 finding IDs
(risk-1..6, arch-1..4, repo-1..5) dispositioned — the run framing's "14"
undercounts by one (6 + 4 + 5 = 15; 1 blocking + 8 major + 6 minor).
Round = "internal-panel" for every row. Append-only.

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| risk-1 | internal-panel | major | accepted | Candidate #1 + Behavior-preservation stance + seal (commit 10): candidate #1 reframed to state the default **preserves a documented regression** (pre-M2b CF metadata → `{}`, per `dev/milestones/phase-1/m02b/baseline_diffs.md`), deferral named a scientific-cost decision (ratified out-by-default), and the seal flags the encoded attrs regression. Contract doc also records the empty `{}` attrs. | design-v2.md |
| risk-2 | internal-panel | major | accepted | §5 audit + Candidate #1 decoupled path: M2b localization made an **unconditional §5 audit obligation** ("cause not localized" no longer an acceptable end state), with **absorption argued separately** on its own merits — the effort/reward coupling is removed. | design-v2.md |
| risk-3 | internal-panel | major | accepted | Finding-classification policy + Commit plan commit 10 (seal): a split "defect" must name a **concrete owner + activation condition** and be **enumerated in the roadmap R4 section at the seal**; seal language distinguishes "audited, clean" from "audited, defects deferred". | design-v2.md |
| risk-4 | internal-panel | major | accepted | §3 + commit 2 + Scope statement (Group-B shared fix with arch-2/repo-3): the ruleorder commit **corrects `AGENTS.md` lines 95–96** to the verified status in the refute case; the design *plans* the edit (does not perform it). Scope self-claims corrected. | design-v2.md |
| risk-5 | internal-panel | minor | accepted | Behavior-preservation stance ("Gate-coverage honesty"): states the three PNG targets are fingerprinted **size-only at ±10%** and are the **weakest link**, so plot-affecting intermediate changes rest on code-path inspection, not the gate. | design-v2.md |
| risk-6 | internal-panel | minor | accepted | Goal paragraph: `TARGETS` citation corrected to **lines 60–66 (the workflow-2 subset of the full list, lines 53–71)**, matching the verified `check_baseline.py`. | design-v2.md |
| arch-1 | internal-panel | blocking | accepted | Verification plan ("End-to-end baseline gate") + Risks: gate runs at **`project_dir = examples/test_local`** (the recorded manifest dir, already set in the committed config). Verified against `check_baseline.py::fingerprint_yaml` (146–152, hashes full parsed YAML, no path-stripping), `PROJECT_DIR_DEFAULT` (line 33), and `copy_config_files.py` (41–44, verbatim copy embedding `project.project_dir`) — so any other dir turns the yaml snapshot red for a non-computational reason. The false "path-independent for … yaml" claim is **deleted**; the "expected to hold" hedge removed. | design-v2.md |
| arch-2 | internal-panel | major | accepted | §3 + commit 2 + Scope statement (Group-B shared fix with risk-4/repo-3): same coherent fix — the ruleorder commit corrects `AGENTS.md` rather than leaving a Snakefile-local comment the AGENTS.md reader never sees; scope-absolutism dropped. Followup path kept as documented fallback if the exception reproduces. | design-v2.md |
| arch-3 | internal-panel | minor | accepted | Behavior-preservation stance: gating restated as **two-tier** — per-commit gate is dry-run + `test_cli`; the `check_baseline check` (14/14) is the **milestone-level** gate run once end-to-end. The v1 "14/14 clean after every commit" overclaim is corrected. | design-v2.md |
| arch-4 | internal-panel | minor | accepted | §2 (per-rule log/benchmark): exclusion criterion restated as "rules whose `script:` performs computation/IO beyond a deterministic file copy"; `copy_config` excluded on that ground while explicitly acknowledging its output **is** the baseline-gated yaml target. | design-v2.md |
| repo-1 | internal-panel | major | accepted | §7 + Commit plan commit 4 + Risks: verified `get_change_climate_proj.py` also runs module-level `snakemake.params.*` (lines 178–187) and `_to_str_tuple` is defined at line 193 (**after** the exec block). Charters a **second import-guard commit** for that script and states `_to_str_tuple` (and the two `get_change_*` functions) **relocate above** the guarded block. | design-v2.md |
| repo-2 | internal-panel | major | accepted | §7 + Alternatives: verified `list_files_not_empty` is a **local list variable** (`get_change_climate_proj_summary.py` line 54), not a callable — the named unit does not exist. The unit claim is **dropped**; the dummy-skip behavior is covered by an **integration-style test** (synthetic empty-vs-nonempty netCDF pair). A `filter_nonempty` extract is kept as a documented fallback (Alternatives). Commit 9 renamed to "unit + integration tests". | design-v2.md |
| repo-3 | internal-panel | major | accepted | §3 + commit 2 + Scope statement (Group-B shared fix with risk-4/arch-2): the canonical `AGENTS.md` instruction file is corrected to track the milestone's own refuting evidence; "Most cross-cutting: none" self-assessment corrected to name the deliberate one-line edit. | design-v2.md |
| repo-4 | internal-panel | minor | accepted | §7 + Risks: the **exact nested guard shape** (`if __name__ == "__main__":` outer, `if "snakemake" in globals():` inner, `else:` fallback) is specified as the shape both new guards must copy, verified against `copy_config_files.py` (59–60) and `get_change_climate_proj_summary.py` (129–130). | design-v2.md |
| repo-5 | internal-panel | minor | accepted | Commit plan ("Commit ordering rationale"): the two import guards (commits 3–4) are sequenced **before** the `tee_to_log` wraps (commits 5–6), so no region of `get_stats_climate_proj.py` / `get_change_climate_proj.py` is re-touched across commits. | design-v2.md |
| ext1-1 | external-r1 | blocking | accepted | Verification plan → "End-to-end baseline gate — the decidable procedure" + Commit plan (new commit 2b) + Risks + Behavior-preservation stance. Verified factually correct in `check_baseline.py`: `TARGETS` (53–71) includes three **workflow-3** targets (67–70) and `cmd_check` (260–262) fails on any recorded target **missing on disk**, so the v2 "wipe → run wf1→wf2 → check" procedure would FAIL on 3 missing wf3 targets even with correct R4 outputs. No target-filtering flag exists today (`_add_common` 284–288 adds only `--project-dir`/`--manifest`; `check` adds `--tolerance`). Fix: charter a repeatable **`--workflow` scope filter** on `check_baseline.py` (commit 2b) that scopes the **source `TARGETS`** list (not just `rec_targets`, else line-269 orphan check re-fails); milestone gate runs `check --workflow model_creation --workflow climate_projections` (11 targets = 4 wf1 + 7 wf2), wf3 excluded. wf1/shared-helper coverage stated separately (per-commit `test_cli` + the 4 wf1 manifest targets). Charter explicitly notes the flag is new. | design-v3.md |
| ext1-2 | external-r1 | major | accepted | §5 (chain audit) + §7 ("Audit-evidence matrix") + Finding-classification policy + Verification plan + Alternatives. Calendar/variable/member checks made **mandatory and executable** (v2's "if the audit flags one" conditionality removed): a minimum audit-evidence matrix (rows U/C1–C3/H/V/P/D) with hand-computed expected outcomes covers 360-day/noleap/Gregorian calendars, a non-January (`"Oct"`) hydrological-year boundary, partial members, asymmetric hist/future variables (the `intersection()` drop, lines 18/45/98), and dummy datasets; each §5 findings-table row must cite a passing test or a traced result. Reconciled with split-don't-absorb: evidence **characterizes** behavior, it does not authorize an inline fix — a failing row still routes to the defect class (split). | design-v3.md |
| ext1-3 | external-r1 | major | accepted | §3 ("ruleorder — reproduce-or-refute … then a DETERMINED action") + Commit plan commit 2 + Alternatives + Risks. The v2 keep-or-remove fork (which let two implementers make opposite DAG-semantics calls) is collapsed: R4 **always retains-and-comments**; removal is deferred to a dedicated task gated on encoding ambiguity-sensitive shapes as regression tests. The supported config matrix is named (M1 tests fixture: 3 models incl. 2 slash-bearing CMIP6 names × 2 scenarios × 1 horizon; M2 reduced: 2 simple names × 1 scenario). The **exact Snakefile comment and AGENTS.md lines-95–96 text** are pre-written for both the reproduce (AGENTS.md unchanged) and refute (AGENTS.md replaced with the evidence-backed "stale insurance" text) outcomes — evidence selects only the wording, never the action. | design-v3.md |
| ext1-4 | external-r1 | major | accepted | §5 ("M2b attrs diagnostic probe") + Candidate absorptions #1 (decoupled absorption call) + Alternatives + Risks. Localization is no longer a "zero-cost reading finding": a **runtime diagnostic probe** records `.attrs` at three checkpoints — after per-model change generation, after `open_mfdataset(coords="minimal", preprocess=...)` (get_change_climate_proj_summary.py 60–62), after merge-write + reopen — with localization defined as the **first checkpoint where expected attrs vanish**. The `temp(...)`-deletion trap on the per-model intermediates is handled (run under `snakemake --notemp` or against a synthetic non-empty-attrs netCDF pair). The finding's **"dependency reproducer"** outcome (loss in an isolated hydromt/xarray op) is admitted as a valid end state; absorption stays out-by-default and is only attempted if the probe pins the loss to workflow-2 code. | design-v3.md |
| ext2-1 | external-r2 | blocking | accepted | "End-to-end baseline gate — the decidable procedure" (respecified) + commit 2b + Alternatives + Risks. Re-raise of ext1-1: verified in `check_baseline.py` that v3's source-`TARGETS`-only scoping is incoherent — `cmd_check` (255–256) loads the **full** 14-entry manifest into `rec_targets` unconditionally; scoping only `TARGETS` reduces `current`/`missing` to 11 but leaves `rec_targets` at 14, so the 3 unselected wf3 recorded entries are **silently skipped** (263–265) and line 274 prints `len(rec_targets)` = **14** while comparing 11 (contradicts "OK — 11 target(s)"), and a scoped `record` would truncate the canonical manifest. Fixed: build **one `selected` universe from workflow-tagged `TARGETS` and apply it symmetrically** — `compute_manifest(selected)` for `current`/`missing` **and** `rec_selected = {p: r for p,r in recorded[...] if p in selected_resolved_paths}` for the diff loop, orphan check, and count (reported count = `len(rec_selected)` = 11). Scoped `record` **omitted entirely** — `--workflow` added to `check` only; `record` stays full/unscoped with its existing missing-guard (234–238), so the canonical manifest cannot be truncated by construction. Chartered `tests/test_check_baseline_scope.py` with 5 assertions: count = selected (not 14), selected-missing fails, unselected recorded ignored (returns 0), no `--workflow` on `record` (argparse rejects), unscoped `check` unchanged. | design-v4.md |
| ext2-2 | external-r2 | major | accepted | §7 (audit-evidence matrix, rows V & P) + §5 (missing-data behavior) + Finding-classification policy + Alternatives + Risks. Re-raise of ext1-2 rows V/P: verified in `src/get_change_climate_proj.py` that `get_change_annual_clim_proj` loops `for var in intersection(hist, clim)` (98) with `intersection = list(set(a)&set(b))` (18–19), so asymmetric variables silently drop the non-shared var; and the change arithmetic `clim_stat − hist_stat` (161–163) runs under xarray's default `arithmetic_join="inner"`, silently intersecting the `member` coord. v3's "records observed set/behavior" let current behavior satisfy the test. Fixed by giving both rows **normative expected outcomes with exact surviving coordinates**: row V norm = result carries BOTH `precip` and `temp` (input hist=`{precip,temp}`, fut=`{precip}`) → current code yields only `precip` → **row V FAILS → variable-drop defect → split**; row P norm = no silent member drop (hist members `[r1i1p1f1, r2i1p1f1]`, fut `[r1i1p1f1]`) → current inner-join yields member `[r1i1p1f1]` only → **row P FAILS → partial-member defect → split**. Norm decided in-design (silent scientific data loss in a plausibility-overlay summary is not acceptable); both pre-classified as split defects (fix is baseline-moving → not absorbed in behavior-preserving R4), enumerated at the seal, wired as strict-`xfail` so an accidental fix trips the gate. Exact `member`-join mechanism left as an open question but the test asserts the *norm*, so it fails correctly regardless. | design-v4.md |
| ext2-3 | external-r2 | major | accepted | §5 ("M2b attrs diagnostic probe", per-model checkpoints) + Candidate absorptions #1 + Alternatives + Risks. Re-raise of ext1-4: v3's first checkpoint was already **after** the whole per-model change computation and write, so an empty-attrs reading there localized the loss only to a large upstream region while the design demanded a single operation / isolated reproducer, and the synthetic alternative exercised only the merge path. Fixed by adding **checkpoints P0→P6 INSIDE `get_change_annual_clim_proj`**: P0 known-attrs input before change computation, P1 after `.sel(time=slice)` window trim (118/127/135/144/255–256), P2 after `.resample().sum/.mean("time")` (119/128/137/146), P3 after the stat reduction + arithmetic `clim_stat − hist_stat` (154–163), P4 after `assign_coords`/`expand_dims`/`xr.merge` (164–170/260–268), P5 immediately before `to_netcdf` (278), P6 after reopening the written per-model file. Localization = **first transformation where attrs disappear**; an **isolated reproducer is required for that transformation** (minimal DataArray through just that op). Merge checkpoints M1 (`open_mfdataset`+preprocess, 60–62) and M2 (merge-write+reopen) retained for post-generation losses; dependency-reproducer end state retained. Synthetic path now suffices for per-model localization (P-checkpoints run on a synthetic known-attrs input without the real merge). | design-v4.md |
| ext3-1 | external-r3 | major | accepted | User-arbitrated 2026-07-20 (round cap): fail-loud pinned as the single V/P norm with a ValueError exception contract; normative acceptance tests (strict-xfail, marker removed by the owning split task via the xpass tripwire) separated from characterization tests (deleted by the same fix commit). Landed in §5 missing-data bullets + §7 matrix rows V/P + "one norm, two test kinds" + verification plan + risks. | design-v5.md |
| ext3-2 | external-r3 | major | accepted | User-arbitrated 2026-07-20 (round cap): probe checkpoints split per-operation (P3a/P3b/P4a/P4b); mechanism specified as standalone stepwise diagnostic dev/scripts/probe_attrs_chain.py with intact-function cross-check, no production-code instrumentation by default, removed-and-verified-by-diff rule otherwise. Landed in §5 "M2b attrs diagnostic probe" + risks. | design-v5.md |

