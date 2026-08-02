# R05 climate-experiment design — review record

Consolidated audit trail (design-review-loop v0.5.0) of the run
`r05-climate-experiment` (2026-07-20) that produced
`climate-experiment-design.md`. Loop: author draft (Opus) → G1 (framing
approved) → 3-lens internal panel → revision r1 (Opus) → external round 1
(GPT, clean-room) → revision r2 (**Fable** — tier escalation: ext1-1 re-raised
the internal Group-A tee-masking finding the Opus revision had routed to
followups) → external round 2 (GPT, regression duty) → round-cap arbitration
(user) → stage-6a arbitration revision (Opus) → G2 (approved). Version series
design-v1..v4; v4 = the landed text. All 21 findings closed; none rejected.

Dispatch meter (status.md): opus 7, fable 1.

| Review | Doc version | Verdict | blocking / major / minor |
|---|---|---|---|
| Internal panel: risk (critical-thinker) | design-v1.md | revise | 0 / 3 / 2 |
| Internal panel: architecture (cst-architect) | design-v1.md | revise | 0 / 2 / 4 |
| Internal panel: repo-fit (python-engineer) | design-v1.md | approve | 0 / 0 / 5 |
| External round 1 (GPT, clean-room) | design-v2.md | revise | 1 / 2 / 0 |
| External round 2 (GPT, regression duty) | design-v3.md | revise | 0 / 1 / 0 (withdrew ext1-1, ext1-2) |
| Round-cap arbitration (user, 2026-07-20) | design-v4.md | ext2-1 accepted; driver-1 wf1-risk correction folded | - |

## Driver note — empirical verification (2026-07-20)

During the Fable r2 revision the driver ran a minimal Snakemake workflow on the
target Windows machine to ground-truth the shell-logging question. A
deliberately-failing command (exit 3): `cmd 2>&1 | tee {log}` → Snakemake
reports SUCCESS (masks the exit code; the shell here applies `set -e` but not
`pipefail`); `cmd > {log} 2>&1` → Snakemake correctly FAILS. This (a) validated
ext1-1's resolution (wf3 shell rules use `> {log} 2>&1`), and (b) disproved the
draft's claim that `tee` is unresolvable and the wf1 gate would fail — `tee`
resolves and runs (corroborated by R3's sealed `--forceall` wf1 rebuild), and
the `| tee` rules only latently mask on failure. The wf1 risk was reframed
accordingly (latent cross-cutting followup, no precursor task); ledger row
`driver-1`.

---

## Internal review index (driver aggregation)

# Internal review index — r05-climate-experiment, panel on design-v1.md

Driver aggregation (2026-07-20) of the three lens reviews. Original IDs,
severities, and verbatim text live in the immutable per-lens files. This
index groups; it never deletes or re-grades.

## Verdicts

| Lens | Verdict | doc_version | blocking | major | minor |
|---|---|---|---|---|---|
| risk (`critical-thinker`) | revise | design-v1.md | 0 | 3 | 2 |
| architecture (`cst-architect`) | revise | design-v1.md | 0 | 2 | 4 |
| repo-fit (`python-engineer`) | approve | design-v1.md | 0 | 0 | 5 |
| **Panel total** | **revise** | design-v1.md | **0** | **5** | **11** |

## Grouped findings

### Group A — the `| tee {log}` shell-rule exit-code question (risk-1 major + arch-3 minor)

- **risk-1 [major]** (§2/§5b): `Rscript ... 2>&1 | tee {log}` returns *tee's*
  exit code, not the command's, absent `pipefail`; no `pipefail`/`shell.prefix`
  found in the repo, so a failing Rscript/Julia rule could report success and
  corrupt downstream. The design asserts the redirect *preserves* the exit code
  — the opposite of a bare `| tee`.
- **arch-3 [minor]** (§5b): the SAME mechanism, but notes Snakemake runs
  `shell:` under `bash --strict` (`set -euo pipefail`) **by default**, so
  pipefail *does* propagate — the R3 pattern is safe in production *because* of
  that default, which the design never states.
- **Driver note — factual reconciliation needed:** the two lenses disagree on
  whether the current pattern is actually broken. Snakemake's documented
  default IS strict-bash pipefail (arch-3 is factually right; the R3 pattern
  has run in production). So risk-1's "returns 0 on failure" is mitigated by
  the default — BUT risk-1's core objection stands: the design **asserts
  exit-code preservation without explaining why**, resting correctness on an
  unstated interpreter default. Resolution both accept: state in §5b that
  propagation relies on Snakemake's default `set -euo pipefail`, name it as an
  invariant the milestone must not weaken (or add an explicit `set -o pipefail;`
  to the teed shell strings to be self-contained), and correct the
  Scope-statement claim. The author must **verify the Snakemake default** and
  state the mechanism, not the backwards assertion.

### Group B — de-risk the cycle-fix / commit-5 area (risk-2 + arch-1 + arch-2, all major)

- **arch-1 [major]** (§4/commit 5): commit 5 atomically bundles the
  *guaranteed* cycle fix + the *admittedly-unverified* `st_num2→st_num` fold +
  the *irreversible* `test_cli` ratchet flip. If the fold re-introduces
  ambiguity at implementation (the design's own stated risk), the atomic
  commit can't land — the rewritten `returncode==0` test fails but the DAG
  doesn't build. Fix: **split commit 5** into 5a (rule-local
  `st_num=[1-9][0-9]*` cycle fix + ratchet flip, keeping `st_num2` distinct)
  and 5b (the fold, contingent on its own dry-run).
- **risk-2 [major]** (§4): the mechanism is mislabeled. The design says "a
  genuine cycle, not a rule-ambiguity," but the cycle exists *because*
  `generate_climate_stress_test`'s under-constrained `{st_num}` makes it a
  **second eligible producer of `cst_0.nc`** — that IS an ambiguity. Risk of
  an incomplete fix the seed (`run_historical=false`) never exercises. Fix:
  restate the mechanism consistently; enumerate, for BOTH `run_historical`
  settings, every requester of `cst_0.nc` and confirm exactly one eligible
  producer remains; **add a `run_historical: true` dry-run to commit-5
  verification**.
- **arch-2 [major]** (§4): the staged-region-fixture claim assumes
  `region.geojson` is the *only* unbuilt `ancient()`/external leaf, but that is
  inferred, not shown. If a second missing leaf exists, the staged fixture
  still yields `MissingInputException` and the `returncode==0` assertion fails.
  Fix: enumerate every `ancient(...)`/producerless input in the wf3 DAG and
  confirm region.geojson is the sole external leaf; record it. (Note: arch-2
  also confirms the wf2 fixture path `{basin_dir}/staticgeoms/region.geojson`
  is reusable for wf3 — same path.)
- **Driver note:** these three are one coherent workstream — commit 5 needs
  splitting (arch-1), the mechanism restated + a `run_historical:true` dry-run
  (risk-2), and the leaf enumeration shown (arch-2). The cycle fix itself is
  sound; the panel wants its *risk* de-bundled and its completeness *shown*.

### Group C — the precip_variance split + characterization test (risk-3 major)

- **risk-3 [major]** (§6/§8): "output-neutral on the seed → split" is an
  escape hatch — it's a property of the fixture, not the code — and the §8
  **characterization test pins the buggy behavior as correct**, so the suite
  *defends* the bug until the (unspecified-owner) split task lands.
- **Driver note — bounded by the G1 ratification:** the user **ratified "split
  to a task"** at G1 (not fix-in-R5), so risk-3's primary suggestion (reclassify
  as intentional-change/fix now) is **foreclosed by the gate** — the author
  keeps the split. But risk-3's *compatible* sub-point is strong and must be
  adopted: **do not let the characterization test assert the buggy value as
  correct** — wire it as `xfail(strict=True)` referencing the split task (the
  R4 V/P pattern), so the suite *flags* rather than *defends* the defect, and
  the fix trips it (xpass) when it lands. Author also names the split task's
  owner + activation condition concretely (not "t2607xx-style").

### Minors (author's discretion, dispositions still required)

- **risk-4 [minor]**: import-guard asymmetry — show the `tee_to_log` wrap
  encloses `downscale_climate_forcing.py`'s top-level `snakemake.*` reads.
- **risk-5 [minor]**: "no computational path edited" is literally false for
  commits 6 (historical_window) and 9 (R arg/logging) — reword to "paths
  edited, asserted output-equivalent, confirmed at the milestone gate."
- **arch-4 [minor]**: `prepare_climate_data_catalog.py` guard is at line ~117,
  not the cited "33–51" — correct the citation.
- **arch-5 [minor]**: §3 must name the two `range(...)` loop-bound sites
  (`prepare_cst_parameters.py` lines 60, 62) as consumers of the helper's
  per-axis counts, not just the linspace sites.
- **arch-6 [minor]**: (confirmation only) the commit-plan coverage is complete
  and the deliberate commit-4 gap is coherent — no action.
- **repo-1 [minor]**: `stress_test_grid(... : Mapping)` needs
  `from collections.abc import Mapping` in `snake_utils.py` (or drop the
  annotation to match `get_config`'s untyped style).
- **repo-2 [minor]**: "copy the nested guard" understates the work for
  `prepare_weagen_config.py` — the year math must be **extracted into a named
  function** (not just wrapped) to be unit-testable; state that explicitly.
- **repo-3 [minor]**: the year math is only in the `generate` branch;
  "generate-vs-stress-test branch" overstates — clarify §8.
- **repo-4 [minor]**: the milestone gate assumes `check_baseline.py`'s
  `EXPERIMENT_NAME` == seed config `experiment_name` (equal today) — add a
  one-line gate-invalidation note.
- **repo-5 [minor]**: the R arity-check block must land after the existing
  `source("./src/weathergen/global.R")`, replacing the scattered `args[[n]]`
  reads — note the ordering constraint.

## Driver notes

- No blocking findings. All three lenses independently verified the design's
  code grounding (the cycle self-loop, the `precip_variance` bug at line 42,
  the 3-target manifest slice, the guard states, the naming rules); the panel's
  objections are about *risk de-bundling* (Group B), an *unstated invariant*
  (Group A), and a *test that defends a bug* (Group C) — not about the design
  being wrong.
- Repo-fit approved outright (5 precise minors).
- Group C is partly foreclosed by the G1 ratification (split stands); the
  author adopts only the compatible xfail sub-point.


---

## External review — round 1 (verbatim)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: What changes — 2. Per-rule `log:` + `benchmark:`; 5b. Logging
- finding: Commit 8 deliberately adds `2>&1 | tee {log}` to workflow-3 R and Julia rules even though the design establishes that this masks the subprocess exit code on the milestone’s Windows platform. Calling the defect inherited does not make this change behavior-preserving: workflow 3 currently has no such redirects, so R5 would newly introduce false-success semantics into the rules it edits.
- rationale: A failed Rscript or Julia process can be reported successful; Snakemake may then continue with a missing or truncated netCDF/CSV, yielding misleading downstream failures or, worse, apparently valid scientific summaries from incomplete artifacts. This is an operational correctness failure introduced by the specified milestone.
- suggested_fix: Make commit 8 contingent on a portable exit-preserving logging mechanism, or omit shell-rule tee logging from R5 and record it as blocked on the cross-cutting fix. Validate the chosen mechanism with a deliberately failing Rscript and Julia command on Windows, asserting a non-zero Snakemake result while preserving console and file output.

### ext1-2  [major]
- section: What changes — 3. The stress-test grid helper
- finding: The shared helper intentionally replaces `prepare_cst_parameters.py`’s required `step_num` lookup with a default of 1 and labels this “output-neutral hardening,” but the new behavior is leniency rather than hardening. A malformed configuration that previously failed immediately would silently generate a 2-step axis and a potentially unintended stress-test ensemble.
- rationale: Missing `temp.step_num` or `precip.step_num` would no longer expose a configuration error; it would change `ST_NUM`, the number of parameter CSVs, and the full `RLZ_NUM × ST_NUM` workload and response surface. That creates wrong results and potentially substantial unplanned compute cost outside the seed fixture.
- suggested_fix: Preserve strict parity for the script consumer, preferably by making `step_num` required everywhere and validating it as a non-negative integer. If legacy Snakefile defaulting must remain, give the helper an explicit strict/default policy and test both call-site contracts rather than silently choosing the more lenient one.

### ext1-3  [major]
- section: Behavior-preservation stance and exact baseline consequence; Verification plan
- finding: The design claims commits 6 and 9 are confirmed output-equivalent at the milestone gate, but that gate hashes only two reduced summary CSVs and a config snapshot and therefore cannot confirm byte-equivalence of the historical extraction or R-generated/perturbed netCDFs. The design itself acknowledges that aggregation can hide intermediate differences and that CSV serialization may be nondeterministic, yet provides no targeted equivalence gate for either computational-path edit.
- rationale: An intermediate forcing change can be averaged away or rounded out in `Qstats.csv`/`basin.csv`, allowing a scientifically meaningful regression to pass; conversely, ordinary stochastic or serialization drift can fail the seal without identifying which commit caused it. The stated intermediate-equivalence guarantee is therefore undecidable under the proposed verification.
- suggested_fix: Add targeted before/after characterization for commits 6 and 9: hash or compare normalized `extract_historical.nc` and at least one fixed-seed realization plus perturbed netCDF, including dimensions, coordinates, variables, attributes, and numerical values. Preserve these artifacts until comparison completes, and establish deterministic random-seed handling or a pristine control rerun before treating summary-hash mismatches as R5 regressions.

---

## External review — round 2 (verbatim)

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1  [major]
- section: Verification plan — Targeted intermediate-equivalence characterization
- finding: ext1-3 is improved but not fully resolved. When pristine control reruns are not identical, the fallback accepts values “within the control pair’s observed rerun noise” without defining the comparison metric, per-variable tolerances, treatment of spatial/temporal outliers, or a pass/fail rule. A single control pair also provides no defensible noise distribution. The same undefined attribution is extended to milestone CSV mismatches.
- rationale: An implementer can choose a permissive interpretation—such as the maximum difference observed in one control pair—and accept a systematic regression introduced by commits 6 or 9, or reject harmless stochastic variation under a different interpretation. Consequently, the intermediate-equivalence gate remains non-decidable precisely in the non-identical case it is intended to handle, and ext1-3 cannot yet be withdrawn. The accepted fixes for ext1-1 and ext1-2 are substantively resolved and introduce no identified regression.
- suggested_fix: Pre-commit an executable comparison rule. Prefer failing closed if the pristine seeded controls are not identical and investigating determinism before accepting commits 6 or 9. If nondeterministic comparison is genuinely required, specify multiple control reruns, named per-variable metrics and tolerances derived from those controls, coordinate/attribute rules, and an explicit pass/fail criterion applied unchanged to control-versus-control and before-versus-after comparisons.

---

## Findings ledger (final)

# Findings ledger — r05-climate-experiment

Revision r1 (design-v1.md → design-v2.md), panel: internal-panel (revise; 0 blocking
/ 5 major / 11 minor). All 16 original findings dispositioned. Severities carried
verbatim from the per-lens files; none silently changed.

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| risk-1 | internal-panel | major | accepted | Group A. Verified the fact authoritatively from Snakemake 9.6.2 source (`shell.py` `_get_process_prefix` L82–89 + platform switch L372–390): `set -euo pipefail; ` is injected only when the shell resolves to bash/bash.exe; POSIX auto-sets bash (propagates) but **Windows uses cmd.exe with NO pipefail** — so on R5's Windows-only exit criteria the `\| tee` pipe DOES mask the Rscript/Julia exit code. Corrected design-v1's backwards "preserves the exit code" claim; classified the masking as an **inherited** cross-cutting property of all three Snakefiles' `\| tee` rules and routed it to `dev/followups.md` (owner `cst-architect`) rather than fixing in wf3. Landed in §2 (pipefail mechanism block + ruling), §5b, Scope statement, and Risks. | design-v2.md |
| risk-2 | internal-panel | major | accepted | Group B. Restated the mechanism **consistently as an ambiguity** — under-constrained line-131 `{st_num}` output makes `generate_climate_stress_test` a **second eligible producer of `cst_0.nc`** (concretely requested by `climate_data_catalog` L139); withdrew design-v1's "genuine cycle, not a rule-ambiguity" phrasing. Enumerated every requester of `cst_0.nc` for **both** `run_historical` settings, confirming exactly one producer remains after the rule-local constraint. Added a **`run_historical: true` dry-run** to commit-5a/5b verification. Landed in §4 (mechanism restatement + requester enumeration) and Verification plan. | design-v2.md |
| risk-3 | internal-panel | major | accepted | Group C (bounded by G1). Primary suggestion (reclassify as intentional-change / fix-in-R5) is **foreclosed by the G1 ratification** (split stands) — that half rejected by the gate, not by the author. Adopted the compatible sub-point: the §8 characterization test no longer asserts the buggy value as correct — wired `@pytest.mark.xfail(strict=True)` asserting the **intended (fixed)** value and referencing the split task, so the suite **flags** (xfail today, xpass when fixed) rather than defends the defect. Named the split task concretely: **`t260720a`**, owner `cst-architect`, activation "any config with precip `variance.max ≠ variance.min`." Landed in §6 and §8. | design-v2.md |
| risk-4 | internal-panel | minor | accepted | §2 now shows the explicit `tee_to_log` wrap shape for `downscale_climate_forcing.py`, with the top-level `snakemake.*` reads (L11–19) inside the `tee_to_log` context (they run at import before any function). Confirms the wrap encloses the module-level reads. | design-v2.md |
| risk-5 | internal-panel | minor | accepted | Reworded the behavior-preservation guarantee to "no computational path edited **except** commits 6 (`historical_window` wiring) and 9 (R arg/logging), asserted output-equivalent and confirmed at the milestone gate." Corrected in the Goal caveat, the "Behavior-preservation stance" self-check, the Gate-coverage caveat, and the Scope statement — the "no path edited" overstatement is withdrawn. | design-v2.md |
| arch-1 | internal-panel | major | accepted | Group B. **Split commit 5** into **5a** (guaranteed rule-local `st_num=[1-9][0-9]*` cycle fix + `test_cli` ratchet flip, keeping `st_num2` distinct) and **5b** (the `st_num2 → st_num` fold, contingent on its own dry-run with a fallback to drop). The ratchet flip now depends only on the guaranteed fix; a failed fold dry-run cannot leave the ratchet half-flipped. Landed in §4 ("decoupled from the cycle fix" para) and the commit plan (5a/5b). | design-v2.md |
| arch-2 | internal-panel | major | accepted | Group B. Enumerated **every `input:`** across all wf3 rules (L47–189) in a table classifying each as internally-producible or external leaf; **confirmed `region.geojson` is the sole unbuilt external leaf** (the other two externals — `config_path`, `DATA_SOURCES` — are repo-tracked, always present). Also confirmed the wf2 fixture path is reusable (`basin_dir = {project_dir}/hydrology_model`, same `staticgeoms/region.geojson`). So the staged-region fixture is sufficient. Landed in §4 (ancient/leaf enumeration table). | design-v2.md |
| arch-3 | internal-panel | minor | accepted | Group A (same mechanism as risk-1). §5b/§2 now state explicitly that exit-code propagation through `\| tee {log}` relies on Snakemake's default `set -euo pipefail` prefix — **true only on the bash path (POSIX/Docker)**, and inert on the Windows/cmd.exe path R5's exit criteria run. The unstated interpreter default is named; the invariant is recorded and routed to followups. | design-v2.md |
| arch-4 | internal-panel | minor | accepted | Corrected the `prepare_climate_data_catalog.py` guard citation to **line 117** (the `if __name__`/`globals()` guard) in §2 and the v2 revision log; design-v1's "lines 33–51" (docstring + top of function body) was wrong. Material claim (guarded, wrap-only) unchanged. | design-v2.md |
| arch-5 | internal-panel | minor | accepted | §3 now names **all three** consumer sites the helper's per-axis counts must feed: the `np.linspace` sizing, **and the two `range(...)` loop bounds at lines 60 and 62** — so the refactor is exhaustive and cannot leave a `range()` site un-migrated (which would produce the wrong CSV count). Also reflected in commit 2. | design-v2.md |
| arch-6 | internal-panel | minor | accepted | No-action confirmation (per driver note: dispose accepted, acknowledged, no change needed). arch-6 verified the commit-plan coverage is complete and the deliberate commit-4 gap (dropped `downscale_climate_forcing.py` guard) is coherent — no defect. v2 preserves that coherence (no commit 4; its `tee_to_log` wrap in commit 7) and the split of commit 5 into 5a/5b keeps every "What changes" item mapped to a commit. Acknowledged; nothing to fix. | design-v2.md |
| repo-1 | internal-panel | minor | accepted | §3 now adds `from collections.abc import Mapping` to `snake_utils.py` as part of commit 2 (the module currently imports only `contextlib`/`os`/`sys`), with the drop-the-annotation alternative recorded. Prevents a `NameError` at import that would break all three Snakefiles. Reflected in commit 2. | design-v2.md |
| repo-2 | internal-panel | minor | accepted | §2 and §8 now state explicitly that `prepare_weagen_config.py`'s cleanup **extracts a named function** (`build_weagen_config` + a pure `compute_nr_years(middle_year, run_length)`), not merely a `__name__`/`globals()` wrapper — so the year math is unit-testable without a `snakemake` global (the §8 deliverable). Commit 3 retitled to "extract ... config assembly into functions + guard." | design-v2.md |
| repo-3 | internal-panel | minor | accepted | §8 clarified: the year math lives **only in the `generate` branch**; the `stress_test` branch (L44–56) carries no arithmetic, only dict assembly. The year-math assertion targets the `generate` branch; a `stress_test`-branch test (if kept) asserts assembled dict shape, not arithmetic. | design-v2.md |
| repo-4 | internal-panel | minor | accepted | Verification plan now carries a one-line **gate-invalidation note**: the wf3 gate assumes `check_baseline.py`'s module-level `EXPERIMENT_NAME` (L82) equals the seed config's `workflows.climate_experiment.experiment_name` (verified equal today); divergence resolves a non-existent `exp_dir` and masquerades as an R5 regression. Recorded, not fixed (pre-existing check_baseline structure). | design-v2.md |
| repo-5 | internal-panel | minor | accepted | §5a now notes the arity/binding block goes immediately after `commandArgs(trailingOnly=TRUE)` **and after the existing `source("./src/weathergen/global.R")`** (wd-relative), replacing the scattered `args[[n]]` reads, so the loud-error `stop()` is the first thing to touch `args` and is not shadowed by a sourcing error. Reflected in commit 9. | design-v2.md |
| ext1-1 | external-r1 | blocking | accepted | Supersedes the v2 Group A "classify as inherited, keep tee" ruling — the reviewer is right that wf3 has NO shell-rule redirects today (verified: Snakefile lines 122/133/173 bare), so adding `\| tee {log}` would NEWLY introduce false-success semantics, not inherit them. Resolved substantively: commit 8 now uses the portable exit-preserving `> {log} 2>&1` on all three shell rules (Rscript ×2 + Julia). **Empirically verified on the Windows platform (2026-07-20, Snakemake 9.6.2/cmd.exe scratch rule):** a `quit(status=1)` Rscript under the redirect yields a non-zero Snakemake result; the `\| tee` variant is worse than v2 analyzed — `tee` does not resolve on this machine's cmd.exe PATH at all (`where tee` empty, in/out of pixi env; no bash), so wf1's inherited tee rules fail unconditionally on Windows. Trade-off accepted and stated: log-file-only (no live console streaming) for wf3's three shell rules; exit-code correctness wins for failure-prone fan-out compute. Followup reframed to wf1-only with upgraded evidence + a milestone-gate wf1-leg risk and escalation path recorded. Commit-8 gate added: deliberately-failing Rscript + Julia must fail the rule. Landed in §2 (empirical verification + superseding ruling + reframed followup + mechanism gate), §5b, Approach ¶4, commit 8, Scope statement, Alternatives, Risks. | design-v3.md |
| ext1-2 | external-r1 | major | accepted | Default-1 leniency withdrawn — the reviewer is right it was leniency mislabeled as hardening (a missing `step_num` would silently change `ST_NUM`, the CSV count, and the whole `RLZ_NUM × ST_NUM` workload). Resolved to the strict (script) semantics: `stress_test_grid` requires `step_num` on both axes (`KeyError` parity with `prepare_cst_parameters.py`) and validates it as a non-negative int (`ValueError` otherwise); the helper never invents a grid. The Snakefile call site's silent default-1 is removed — an error-path change on invalid configs only, classified output-neutral hardening (R3 §7.1 class) in the commit-2 message; output-neutral on the seed (temp 1, precip 2 → 2×3=6). Unit tests assert the strict contract (seed value + KeyError/ValueError raises). Landed in §3 (signature docstring + behavior-preservation block + unit test), commit 2, Alternatives (default-1 and dual-policy-flag rejected), Risks (open question resolved). | design-v3.md |
| ext1-3 | external-r1 | major | accepted | The "confirmed at the milestone gate" claim for commits 6/9 was undecidable — the gate hashes only 2 reduced CSVs + a config snapshot and cannot see the edited intermediates. Added a targeted before/after intermediate-equivalence characterization per commit: commit 6 — normalized xarray compare (dims/coords/vars/attrs/values, volatile provenance attrs normalized out) of `extract_historical.nc` (not temp(), no --notemp needed); commit 9 — same compare of one realization (`rlz_1_cst_0.nc`) + one perturbed netCDF preserved via a `--notemp` scratch run. Determinism/attribution protocol pre-committed: realizations are seeded (`weathergen_config.yml` `generateWeatherSeries.seed: 123`, read at `generate_weather.R` L40); `apply_climate_perturbations` rerun-stability is unproven → a pristine same-commit control-rerun pair runs before the first comparison; if not bit-stable, fall back to structural equality + control-bounded values, and any summary-hash mismatch at the milestone gate is attributed against the control before being treated as an R5 regression (also empirically resolves the R4-inherited CSV-determinism question). Landed in Verification plan (new gates), commits 6/9 gate lines, Goal caveat, Behavior-preservation stance, Risks. | design-v3.md |
| ext2-1 | external-r2 | major | accepted | user-arbitrated 2026-07-20; fuzzy "within rerun noise" fallback replaced with a pre-committed executable FAIL-CLOSED comparison rule (normalized xarray `.identical()` for netCDF intermediates + normalized-content hash for the summary CSVs): a seeded control-vs-control non-identity fails the gate into a determinism investigation (no auto-accept tolerance; determinism is the expected state for seed 123), and only an identical control lets the before-vs-after run, which must also be identical to pass; same rule applied to the milestone CSV gate. Landed in Verification plan item 3 + Behavior-preservation stance + commit-9 gate line + Risks (CSV-determinism / R-layer rerun-stability bullets). | design-v4.md |
| driver-1 | arbitration | n/a | accepted | wf1-gate-failure risk corrected per driver empirical evidence (tee resolves and runs; masks only on failure via `set -e` without `pipefail` on cmd.exe; wf1 leg passes on clean commands; R3 sealed 14/14 with tee logs; no precursor task); folded into v4 risks + reframed cross-cutting followup to `dev/followups.md`. Landed in §2 empirical block + reframed followup, Approach ¶4, Scope statement, commit 12, Alternatives, Risks (wf1-tee bullet); wf3 `> {log} 2>&1` resolution unchanged. | design-v4.md |

