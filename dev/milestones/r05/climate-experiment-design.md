# R05 — Workflow 3: climate experiment — design

**Status.** Accepted 2026-07-20 (design-review-loop run `r05-climate-experiment`,
gate G2). Review trail: 3-lens internal panel + 2 external cross-vendor (GPT) rounds
+ round-cap arbitration — 21/21 findings closed (full dispositions and verbatim
external verdicts: `climate-experiment-design-reviews.md`, this directory). G1
approved 2026-07-20; its three ratifications bind this design (behavior-preserving
refactor / zero manifest edits by default / split-don't-absorb; **R-testthat = NO,
locked**; the `precip_variance` max-reads-min bug **SPLIT to task `t260720a`, not
fixed in R5**). Implementation is handed to `task-brief`.

**Date.** 2026-07-20.

**Revision log.**

- v1 (2026-07-20) — initial draft against the R4 sealed state. Grounds every
  claim in `Snakefile_climate_experiment`, `src/`, `src/weathergen/*.R`,
  `src/snake_utils.py`, `tests/test_cli.py`, `dev/scripts/check_baseline.py`,
  `dev/conventions/naming.md`, `dev/followups.md`, and
  `config/snake_config_model_test.yml`. Mirrors the R4 design register:
  contract-first, behavior-preserving, split-don't-absorb audit policy,
  pre-committed finding-classification policy, rulings on all five candidate
  absorptions, decidable verification gates. Post-draft advisor pass corrected
  four items before panel review (cycle mechanism restated as an output-`{st_num}`
  self-loop; ratchet flip defaults to a staged-region fixture; `downscale_climate_forcing.py`
  import guard dropped; commit-7 count corrected to 7 `script:` rules).
- **v2 (2026-07-20) — revision r1 after the internal panel (revise; 0 blocking /
  5 major / 11 minor).** Resolves all 16 findings; every G1 ratification kept intact.
  Changes by group:
  - **Group A (risk-1 major + arch-3 minor) — §5b / §2 / Scope.** Corrected the
    backwards "preserves the exit code" claim. Verified Snakemake 9.6.2's shell
    default authoritatively from source (`.pixi/.../snakemake/shell.py`
    `_get_process_prefix`, lines 82–89, and the platform switch lines 372–390):
    the `set -euo pipefail; ` prefix is injected **only when the shell executable
    resolves to bash/bash.exe**. POSIX auto-sets bash (line 388) so pipefail
    propagates there; **Windows is hardcoded to `shell.executable(None)` → cmd.exe,
    with NO pipefail** (line 390). R5's exit criteria are Windows-only, so on the
    **exercised** platform the `| tee {log}` pipe masks the Rscript/Julia exit
    code. Classified as an **inherited** cross-cutting property of the R3 `| tee`
    pattern (all three Snakefiles carry it) → routed to `dev/followups.md` with an
    owner + activation condition, not fixed in wf3 (milestone-boundary +
    split-don't-absorb). The `| tee` form is kept (commit `4a67d79` deliberately
    restored live console output); no switch to `> {log} 2>&1`.
  - **Group B (risk-2 + arch-1 + arch-2, all major) — §4, commit plan.** **Split
    commit 5** into **5a** (guaranteed rule-local `st_num=[1-9][0-9]*` cycle fix +
    `test_cli` ratchet flip, keeping `st_num2` distinct) and **5b** (the
    `st_num2 → st_num` fold, contingent on its own dry-run). Restated the cycle
    mechanism **consistently as an ambiguity** — an under-constrained output
    wildcard makes `generate_climate_stress_test` a **second eligible producer of
    `cst_0.nc`** (dropped the "genuine cycle, not a rule-ambiguity" phrasing).
    Enumerated **every `ancient()`/producerless input** in the wf3 DAG and
    confirmed `region.geojson` is the **sole unbuilt external leaf**. Added a
    **`run_historical: true` dry-run** to commit-5a/5b verification.
  - **Group C (risk-3 major) — bounded by G1 — §6 / §8.** Split STANDS
    (user-ratified). Adopted risk-3's compatible sub-point: the characterization
    test no longer asserts the buggy value as correct — it is wired as
    `@pytest.mark.xfail(strict=True)` asserting the **intended (fixed)** value and
    referencing the split task, so the suite **flags** the defect (xfail today,
    xpass when the fix lands). Named the split task concretely: **`t260720a`**,
    owner `cst-architect` (to route), activation condition "any config with precip
    `variance.max ≠ variance.min`."
  - **Minors — dispositioned in place.** repo-1 (`from collections.abc import
    Mapping` added in §3); repo-2 (§2/§8 now say **extract a named function**, not
    merely wrap); repo-3 (§8 year-math is `generate`-branch only); repo-4
    (§Verification `EXPERIMENT_NAME` gate-invalidation note); repo-5 (§5a arity
    block ordering after `source("./src/weathergen/global.R")`); risk-4 (§2 shows
    the `tee_to_log` wrap enclosing `downscale_climate_forcing.py`'s top-level
    reads); risk-5 (Scope/caveat reworded to "paths edited, asserted
    output-equivalent, confirmed at the milestone gate"); arch-4 (guard citation
    corrected to line 117); arch-5 (§3 names the two `range(...)` loop sites, lines
    60/62); arch-6 (no-action confirmation, acknowledged).
- **v3 (2026-07-20) — revision r2 after external review r1 (GPT clean-room;
  revise; 1 blocking / 2 major).** All three findings accepted; every G1
  ratification and all 16 internal-panel fixes kept intact. Changes by finding:
  - **ext1-1 (blocking) — §2 / §5b / Approach ¶4 / commit 8 / Scope /
    Verification / Alternatives / Risks.** The v2 ruling ("keep `| tee {log}`,
    route the masking to followups as inherited") is **withdrawn**: workflow 3
    has NO shell-rule redirects today, so R5 adding `| tee` would **newly
    introduce** false-success semantics on the Windows exit-criteria platform —
    not inherit them. Commit 8 now uses the portable, **exit-code-preserving**
    `> {log} 2>&1` redirect on all three shell rules (Rscript ×2 + Julia).
    **Empirically verified on this machine (2026-07-20, Snakemake 9.6.2 via
    cmd.exe, scratch Snakefile):** a deliberately failing
    `Rscript -e "quit(status=1)" > {log} 2>&1` rule yields a non-zero Snakemake
    result ("command exited with non-zero exit code", exit 1) with the stale
    output removed. A second empirical fact sharpens the analysis: **`tee` is
    not resolvable at all on this machine's cmd.exe PATH** (`where tee` → not
    found, inside and outside the pixi env; no `bash` either) — so wf1's
    inherited `| tee` rules do not merely mask on Windows, they **fail
    unconditionally**; the cross-cutting followup is reframed accordingly and a
    milestone-gate risk (the gate's wf1 leg) is recorded. Trade-off accepted:
    log-file-only capture (no live console streaming) for wf3's three shell
    rules; exit-code correctness wins for wf3's failure-prone fan-out compute.
    A commit-8 verification gate (deliberately failing Rscript + Julia must
    fail the rule) is added.
  - **ext1-2 (major) — §3 / commit 2 / Alternatives / Risks.** The helper's
    default-1 leniency is withdrawn — it was leniency, not hardening, and could
    silently change `ST_NUM`. `stress_test_grid` is now **strict**: `step_num`
    required on both axes (`KeyError` parity with `prepare_cst_parameters.py`'s
    current read) and validated as a non-negative integer. The Snakefile call
    site's silent default-1 disappears — an error-path change on invalid
    configs only, classified output-neutral hardening (R3 §7.1 class). Unit
    tests assert the strict contract at both call-site shapes.
  - **ext1-3 (major) — Verification plan / commits 6 & 9 / Goal caveat /
    Behavior-preservation stance / Risks.** The "confirmed at the milestone
    gate" claim for the two computational-path commits was undecidable (the
    gate hashes only 2 reduced CSVs + a config snapshot). Added a **targeted
    before/after intermediate-equivalence characterization**: a normalized
    xarray compare (dims, coords, vars, attrs, values) of
    `extract_historical.nc` for commit 6, and of one realization + one
    perturbed netCDF (preserved via `--notemp` in a scratch run) for commit 9 —
    plus a pristine same-commit **control rerun** protocol (weathergenr seed
    123) so any mismatch, including a summary-hash mismatch at the milestone
    gate, is attributed (rerun noise vs regression) before being treated as an
    R5 regression.
- **v4 (2026-07-20) — arbitration revision after external review r2 (2-round
  external cap reached with one surviving major; USER ARBITRATED 2026-07-20).**
  Exactly two mandated fixes applied; every G1 ratification, all 16 internal-panel
  fixes, and the v3 ext1-1/ext1-2/ext1-3 resolutions are kept intact (regression
  duty). Both changes were mandated by the 2026-07-20 user arbitration (ext2-1
  acceptance + the driver-evidence wf1-risk correction).
  - **ext2-1 (major, user-ACCEPTED) — Verification plan item 3 + forced
    cross-refs.** The v3 intermediate-equivalence gate defined the byte-identical
    path but left the non-identical case fuzzy ("within the control pair's
    observed rerun noise" — no metric, tolerance, or pass/fail rule; one control
    pair gives no defensible noise distribution). Per the arbitration, adopted the
    reviewer's preferred resolution: a **pre-committed, executable, FAIL-CLOSED
    comparison rule**. Determinism is the expected state for the fixed weathergenr
    seed (123), so a seeded control-vs-control non-identity now **FAILS the gate
    into a determinism investigation** — it does **not** auto-accept a tolerance;
    only if the control pair is identical does the before-vs-after comparison run,
    and it too must be identical to pass. Specified the exact comparison
    (normalized xarray `.identical()` on dims/coords/vars/attrs/values for the
    netCDF intermediates; the existing normalized-content hash for the summary
    CSVs) and applied it unchanged to the milestone CSV gate. Removed every "within
    observed rerun noise" / "control-bounded" fuzzy-fallback phrasing. Landed in
    the Verification plan (item 3 rewritten fail-closed), the Behavior-preservation
    stance, the commit-9 gate line, and the Risks "CSV-serialization determinism"
    and "Rerun-stability of the R-layer netCDFs" bullets.
  - **wf1-risk correction (driver-1, user-directed on driver empirical
    evidence) — §2 / Approach ¶4 / Scope / commit 12 / Alternatives / Risks.**
    The v3 risk claiming the R5 milestone gate's wf1 leg is "predicted to fail"
    because `tee` is "not recognized" (with a possible "charter a wf1
    tee→redirect precursor task before the seal") is **empirically DISPROVED**
    (observations.md 2026-07-20 driver verification): on this machine `tee`
    **resolves and runs fine** — it only **MASKS** the exit code on *actual*
    command failure (cmd.exe injects no `set -euo pipefail` prefix — bash-only,
    §2 source analysis — so the `| tee` pipeline returns `tee`'s exit 0), while
    `> {log} 2>&1` correctly fails; corroborated by R3 sealing
    via a full `--forceall` wf1 rebuild that wrote all tee logs and passed 14/14
    on this same machine. Reframed to the accurate framing: **wf1's R3-added
    `| tee` shell rules run correctly on success (the R5 gate's wf1 leg PASSES on
    clean commands) and only latently mask on failure — a cross-cutting robustness
    followup routed to `dev/followups.md`, NOT an R5 blocker and NOT requiring a
    precursor task.** Removed the "wf1 gate predicted to fail" claim, the "tee
    unresolvable / fails unconditionally" reading, and the precursor-task
    escalation everywhere they appeared as live claims (the historical v1–v3
    revision-log entries are left unchanged as the record of what those versions
    did). The wf3 resolution (commit 8 uses `> {log} 2>&1`) is unchanged — that
    was and remains correct.

---

## Goal

Clean up `Snakefile_climate_experiment` (workflow 3) and every script it invokes
— Python orchestration/analysis helpers under `src/` **and** the R weather-generator
layer `src/weathergen/*.R` — by inheriting the patterns R3 minted on workflow 1
and R4 applied to workflow 2 (shared `get_config`/`tee_to_log` from
`src/snake_utils.py`, per-rule `log:`/`benchmark:`, contract-doc-before-code,
config-path forwarding via `workflow.configfiles[0]`). Workflow 3 is the **most
complex** of the three: it spans Python + R (via `Rscript --vanilla` from
`shell:`) + Julia/Wflow (from `shell:`), and it fans out `RLZ_NUM` realizations ×
`ST_NUM` temp/precip perturbations through Wflow before reducing to hydrological
indicators.

Beyond the inherited hygiene, R5 delivers four workflow-3-specific items the
roadmap names (`dev/roadmap.md` § R5, reproduced in intake.md):

1. The **`dev/workflows/climate_experiment.md` contract doc** (deferred from R1;
   format per `dev/milestones/r01/modularity-contracts-design.md` §4, mirroring
   `dev/workflows/model_creation.md` and `dev/workflows/climate_projections.md`).
2. The **`CyclicGraphException` resolution** — the workflow-3 `test_cli` ratchet
   that R3 deferred to R5, paired with the `st_num2 → st_num` wildcard fold that
   `dev/conventions/naming.md` §4 assigns to R5. This flips a **live** ratchet, so
   it is IN scope (not a deferral). **v2 splits its delivery**: the cycle fix +
   ratchet flip (guaranteed) is one commit; the fold (contingent) is a separate
   commit (§4, commit plan 5a/5b).
3. The **stress-test grid helper** — `ST_NUM = (temp.step_num+1)*(precip.step_num+1)`
   is currently computed **twice** (inline in `Snakefile_climate_experiment` lines
   30–34, and re-derived in `src/prepare_cst_parameters.py` lines 36–46). Extract
   to one tested Python helper.
4. The **R-layer cleanup + `impose_climate_change.R`/downscaling scientific
   audit** — cleaner argument parsing, consistent logging, and a pre-committed
   finding-classification audit of the perturbation/downscaling path.

**R5 is a behavior-preserving refactor by default.** Like R3/R4 it aims for **zero
manifest edits** and a clean `check_baseline check --workflow climate_experiment`
throughout. Any finding that would change an output value is either explicitly
chartered here with its baseline consequence, or split to a dedicated task with a
named owner + activation condition — never absorbed silently (intake decision
criteria §1). The one deliberate exception class is output-neutral error hardening
(loud on an *invalid* config, no change on the *valid* seed config), mirroring R3
§7.1 (`setup_gauges` raising on unknown `wflow_outvars`).

**One honest caveat, stated once and relied on throughout.** A clean
`check_baseline check --workflow climate_experiment` does **not** prove workflow
3's scientific outputs are unchanged in every respect. The manifest fingerprints
workflow 3 with **exactly 3 targets** (`dev/scripts/check_baseline.py` `TARGETS`
lines 55–73; the workflow-3 slice is lines 70–72):

- `{exp_dir}/model_results/Qstats.csv` (normalized-content CSV hash),
- `{exp_dir}/model_results/basin.csv` (normalized-content CSV hash),
- `{project_dir}/config/snake_config_climate_experiment.yml` (SHA256 of the full
  parsed YAML snapshot).

Critically, **every per-realization / per-stress-test intermediate is wrapped in
`temp(...)`** and is **not** a manifest target: the realization netCDFs
(`rlz_{rlz_num}_cst_{st_num}.nc`, Snakefile line 131), the downscaled forcing
(`inmaps_rlz_...`, line 154), and the Wflow state files (`outstates_rlz_...`, line
171) are all `temp(...)` and deleted after consumers finish. The `cst_0`
realization netCDFs (`rlz_{rlz_num}_cst_0.nc`, line 120) are also `temp(...)`. The
per-run Wflow output CSVs (`output_rlz_..._cst_{st_num2}.csv`, line 170) are **not**
`temp` but are **not** manifest targets either — they feed `export_wflow_results`
and are the raw discharge the two summary CSVs reduce. So the gate sees the two
**reduced summary CSVs** and the **config snapshot**, not the discharge time series
or the perturbed forcing that produced them.

**R5's guarantee for those intermediates, stated precisely (risk-5).** No
computational path is edited **except** two chartered, asserted-output-equivalent
edits: the `historical_window` wiring (commit 6) changes how `extract_historical.nc`'s
start/end dates are *sourced* (not their *values* on the seed — they are
byte-identical, § candidate #2), and the R-layer arg-binding + progress-`message()`
additions (commit 9) edit the scripts that write the perturbed netCDFs (asserted
byte-neutral on valid input). Both are confirmed by a **targeted before/after
intermediate-equivalence characterization on the artifacts they touch** (Verification
plan, ext1-3) **plus** the milestone gate — not proven by reading, and no longer left
to the summary-CSV gate alone. Every other commit leaves the computational path
literally untouched. The summary gate is necessary, not sufficient; the targeted
characterization closes the gap for the two edited paths.

**Additional gate-honesty note (inherited from R4's CSV-serialization finding).**
The two summary CSVs are content-hashed. R4's `dev/milestones/r04/baseline_diffs.md` recorded
a CSV-serialization non-determinism finding for its summary CSVs; `Qstats.csv`
carries `.round(2)`/`.round(4)` per-column formatting (`export_wflow_results.py`
lines 169–203) and `basin.csv` a `float32` round (line 237), so if the R4 finding's
root cause is row-ordering or float-repr drift it may recur here. R5 does not lean
on the CSV hash as if it proves scientific invariance; it treats the hash as a
necessary regression tripwire on top of code-path inspection. Whether wf3's CSV
hashing shares R4's fragility is an **open question** recorded below, to be
confirmed empirically at the milestone gate; it does not change R5's plan (no
computational path is edited beyond the two asserted-equivalent commits above).

## Why now

R1 sectioned this Snakefile's config into the R01 `project`/`shared`/`workflows.<name>`
schema, and R3 moved it onto the shared `get_config` import + `sys.path` bootstrap
(both present at the top of `Snakefile_climate_experiment`, lines 1–9 — verified:
`sys.path.insert(0, str(Path(workflow.basedir)))`, `from src.snake_utils import
get_config`, and `config_path = workflow.configfiles[0]` at line 14). But everything
else in workflow 3 predates the Phase-2 discipline:

- **No per-workflow contract doc** — `dev/workflows/climate_experiment.md` was
  deferred from R1 to R5's opening act (roadmap R5 deliverables).
- **Zero `log:`/`benchmark:` directives** on any rule in
  `Snakefile_climate_experiment` (verified: the file has none). The exhaustive M1
  warnings triage (`dev/followups.md` § R3, "Redo M1 warnings triage exhaustively")
  closes only once **all three** Snakefiles emit logs — R3 swept wf1, R4 swept wf2,
  R5 sweeps wf3. R5 is the last workflow, so its sweep closes the triage.
- **The R weathergen scripts** (`generate_weather.R`, `impose_climate_change.R`)
  parse positional args via `commandArgs(trailingOnly=TRUE)` (`generate_weather.R`
  line 8; `impose_climate_change.R` line 8), have **no consistent logging**, and
  carry a documented `spatial_ref` workaround block (`generate_weather.R` lines
  69–89). The roadmap wants cleaner arg parsing / consistent logging.
- **The stress-test grid math is duplicated.** `ST_NUM = (step_num+1)*(step_num+1)`
  appears inline in the Snakefile (lines 30–34) **and** is re-derived independently
  in `src/prepare_cst_parameters.py` (`temp_step_num = ... + 1`,
  `precip_step_num = ... + 1`, `ST_NUM = temp_step_num * precip_step_num`, lines
  36–46). Two copies of one formula is exactly the drift risk the roadmap's
  "single tested helper" targets.
- **Zero unit-test coverage** for this workflow's Python helpers; the R layer has
  no testthat coverage.
- **A live open defect entangled with a test ratchet.** The workflow-3
  `test_cli` `CyclicGraphException` xfail (`tests/test_cli.py` lines 111–120): the
  `generate_climate_stress_test` rule's under-constrained output wildcard
  `rlz_{rlz_num}_cst_{st_num}.nc` (Snakefile line 131) can resolve `{st_num}` to
  `0`, making the rule a **second eligible producer** of `rlz_{rlz_num}_cst_0.nc`,
  which `generate_weather_realization` already produces (line 120) and which the
  rule itself names as its literal `rlz_nc` input (line 127). The `--dry-run`
  resolver flags this on the test config. R3 deferred it to R5
  (`dev/followups.md` § R3, split note 2026-07-19). R5 must resolve it and flip the
  ratchet — unlike the wf2 ratchet R3 fixed from the test side, this fix lives
  **inside `Snakefile_climate_experiment`** and is entangled with the
  `st_num2 → st_num` fold.

R5 is where R3's patterns get applied a **second** time (R4 was the first copy). R4
made the inheritance clean; R5 is the mechanical repeat plus the workflow-3-specific
R-layer and cycle work.

## Approach

Same contract-first, behavior-preserving discipline R3/R4 used, applied to workflow
3:

1. **Contract before code.** `dev/workflows/climate_experiment.md` (format per
   `dev/milestones/r01/modularity-contracts-design.md` §4, mirroring the two sibling contract
   docs) is R5's first commit, written against **current** behavior. Code commits
   then change behavior against a recorded contract, not from memory.
2. **Behavior-preserving throughout; `check_baseline check --workflow
   climate_experiment` stays clean, zero manifest edits by default.** Every hygiene
   commit (log/benchmark directives, the R-layer logging/arg cleanup, the grid
   helper extract, label renames, the cycle fix + `st_num2→st_num` fold, the import
   guard, unit tests) leaves the 3 workflow-3 manifest targets identical and needs
   no `record`. The one bounded exception is output-neutral error hardening surfaced
   by the audit (§ "What changes" 6), classified as such in its commit message.
3. **Audit-then-classify, don't audit-then-edit.** The `impose_climate_change.R` /
   downscaling audit (§ "What changes" 6) produces a written findings table; each
   finding is classified **before** any code moves (§ "Finding-classification
   policy"). Noise is documented and left; a defect that would move an output is
   split to a dedicated task with a named owner + activation condition unless
   explicitly chartered; only an intentional, chartered change moves the baseline
   with a `dev/milestones/r05/baseline_diffs.md` entry.
4. **Milestone boundaries beat followup tags.** Findings outside workflow 3 go to
   `dev/followups.md`; R5 does not touch `Snakefile_model_creation` or
   `Snakefile_climate_projections` content (roadmap iron rule; shared-helper
   inheritance already landed in R3). The dev-script `check_baseline.py` already has
   the `--workflow` filter (R4 landed it, commit 2b — verified: `TARGETS` is
   `list[tuple[str, str, str]]` with a workflow tag, and `WORKFLOWS` includes
   `climate_experiment`, lines 55–75), so R5 needs **no** tooling change there. The
   `| tee` exit-code-masking-on-failure defect in **wf1's existing** shell rules
   (Group A → ext1-1) stays outside R5 (milestone boundary) and is routed to
   `dev/followups.md` as a **latent robustness item, not a blocker** (those rules
   run correctly on success — see Risks) — but **wf3's new shell-rule logging does
   not inherit the `| tee` form**: it uses the exit-preserving `> {log} 2>&1`
   redirect (§2), so R5 introduces no new instance of the defect.
5. **Naming: applied here.** `dev/conventions/naming.md` names R5 as the owning
   milestone for workflow-3 identifiers. The `st_num2 → st_num` fold (§4, the known
   inconsistency) and the `_fid` label renames (§3/§5) are R5's; §7 contract surfaces
   (output filenames `Qstats.csv`/`basin.csv`, config keys, catalog source names)
   stay grandfathered. `RLZ_NUM`/`ST_NUM` are illustrative future rename targets in
   naming §9 (→ `rlz_count`/`stress_test_count`) but are **grandfathered constants
   used across the Snakefile and scripts**; renaming them is deferred (see
   Alternatives) to keep the diff surgical.

## What changes

### 1. Contract doc `dev/workflows/climate_experiment.md` (opening act)

Format per `dev/milestones/r01/modularity-contracts-design.md` §4, mirroring
`dev/workflows/climate_projections.md`, target < 130 lines (wf3 has more rules).
Content grounded in `Snakefile_climate_experiment` and
`config/snake_config_model_test.yml`:

- **Owned config keys** (`workflows.climate_experiment.*`, verified against the
  Snakefile's reads, lines 19–41): `experiment_name` (read as `experiment`),
  `realizations_num` (→ `RLZ_NUM`, default 1), `stress_test.temp.step_num` /
  `stress_test.precip.step_num` (→ `ST_NUM`, default 1 each), `run_historical`
  (default `False`, drives `ST_START = 0 if run_hist else 1`), `horizontime_climate`
  (required), `run_length` (→ `wflow_run_length`, default 20), `aggregate_rlz`
  (default `True`), `Tlow` (default 2), `Tpeak` (default 10). Plus the deeper
  `stress_test.{temp,precip}.{mean,variance,transient_change}` structure read by
  `prepare_cst_parameters.py` (lines 31–56) and `prepare_weagen_config.py` (lines
  54–56). Note `enabled:` is present (config line 52) but **not read** by the
  Snakefile — documented as a known dormant key (operationalizing it is R6; non-goal).
- **Reads from `project`** (Snakefile lines 21–23): `project.project_dir`,
  `project.static_dir`, `project.data_sources` (→ `DATA_SOURCES`, the deltares
  catalog — note wf3 reads `data_sources`, **not** wf2's `data_sources_climate`;
  documented so the contract does not imply a shared key).
- **Reads from `shared`** (line 39): `shared.clim_historical` (→ `clim_source`).
  **Contract-doc must flag the `historical_window` gap:** `config` line 21–23 has
  `shared.historical_window.{starttime,endtime}` but the Snakefile **never reads
  it** — the extraction dates are hardcoded in `src/extract_historical_climate.py`
  (lines 156–157). This is candidate #2 (ruled below); the contract records the
  current (hardcoded) behavior.
- **Input contract** — the cross-workflow input `{basin_dir}/staticgeoms/
  region.geojson` (produced by workflow 1, consumed as `ancient(...)` by
  `extract_climate_grid`, line 68) and the wflow model root `{basin_dir}` /
  `{basin_dir}/staticmaps.nc` consumed by `downscale_climate_realization`
  (`downscale_climate_forcing.py` line 38, `WflowSbmModel(root=model_root)`); the
  deltares data catalog (`DATA_SOURCES`) and the run-time
  `data_catalog_climate_experiment.yml` (built by `climate_data_catalog`, line 141).
- **Output contract**, split by role:
  - *Direct `rule all` targets* (Snakefile lines 47–51): `{exp_dir}/model_results/
    Qstats.csv`, `{exp_dir}/model_results/basin.csv`, `{project_dir}/config/
    snake_config_climate_experiment.yml`. **These 3 are the manifest slice.**
  - *Non-temp but non-manifest side products*: the per-run Wflow output CSVs
    `output_rlz_{rlz_num}_cst_{st_num2}.csv` (line 170) and the per-run TOMLs
    `wflow_sbm_rlz_..._cst_{st_num2}.toml` (line 155); the per-stress-test parameter
    CSVs `cst_{st_num}.csv` (line 82); the weagen config YAMLs (lines 89, 104); the
    `data_catalog_climate_experiment.yml` (line 141); the extra `RT_{var}.csv` files
    `export_wflow_results.py` writes (line 280).
  - *Intermediate `temp(...)` artifacts* (deleted after consumers finish; NOT
    manifest targets): `rlz_{rlz_num}_cst_0.nc` (line 120), `rlz_..._cst_{st_num}.nc`
    (line 131), `inmaps_rlz_..._cst_{st_num2}.nc` (line 154), `outstates_rlz_...nc`
    (line 171). Also `extract_historical.nc` (line 73) — NOT `temp` but consumed as
    `ancient(...)` (line 117), so grandfathered-stale.
  - *Side-effect artifacts*: per-rule logs/benchmarks (ephemeral, gitignored, never
    fingerprinted or committed — inherited convention).
- **Method invariant recorded** (`AGENTS.md`): stress-test scenarios come from the
  **weathergenr** stochastic generator, **not** from CMIP projections. `RLZ_NUM`
  realizations × `ST_NUM` temp/precip perturbations; `cst_0` = unperturbed baseline
  (run through Wflow only when `run_historical` sets `ST_START = 0`); reduced to the
  Qstats indicators. The contract states this explicitly so a future reader cannot
  couple wf3 to workflow 2's change factors.
- **Known state flagged, not fixed:** the `historical_window` config-key gap
  (candidate #2 ruling below), and the `st_num2` wildcard inconsistency (folded in
  R5, § 5 below).

### 2. Per-rule `log:` + `benchmark:` (inherited R3/R4 pattern)

Apply the R3 convention, with **one recorded deviation for the shell-rule redirect
form (ext1-1, ruled below)**. **The shape reference and the split between `script:`
and `shell:` logging are load-bearing here** and taken from
`Snakefile_model_creation` (verified):

- **`script:` (Python) rules** are **not** auto-redirected by Snakemake, so each
  script wraps its top-level body in `tee_to_log(snakemake.log[0])` from
  `src/snake_utils.py` (already implemented + unit-tested in R3, lines 76–110). The
  `log:` directive and the wrap land in the **same commit per script** so each
  commit is runnable.
- **`shell:` (Rscript / Julia) rules** ARE redirectable. R3's committed wf1 pattern
  is **explicit tee to both console and log**: `... 2>&1 | tee {log}`
  (`Snakefile_model_creation` lines 89, 167, 182 — the `hydromt build`,
  `hydromt update`, and Julia `Wflow.run()` rules; R3's commit `4a67d79` restored
  live console streaming). **R5 does NOT inherit that form (ext1-1).** wf3's three
  shell rules gain **`> {log} 2>&1`** — a portable, **exit-code-preserving**
  redirect (no pipe stage to swallow the subprocess status) that captures
  stdout+stderr to the per-rule log on both cmd.exe and bash. This is a deliberate,
  recorded deviation from inheritance fidelity (intake criterion 4), and the
  workflow-3-specific reason is decisive: **wf3 has no shell-rule redirects
  today**, so adding `| tee` would **newly introduce** false-success semantics on
  the Windows exit-criteria platform (per the empirical note below, `tee` resolves
  and runs but masks the subprocess exit code on failure — cmd.exe gets no
  `set -euo pipefail` prefix, bash-only, so the pipe returns `tee`'s status) — the
  opposite of behavior-preserving. A Python `tee_to_log` helper remains
  inapplicable to shell subprocesses (it only captures in-process Python output —
  `snake_utils.py` docstring lines 55–57 says so explicitly).

**Exit-code / pipefail mechanism — stated explicitly (Group A: risk-1, arch-3).**
A bare `cmd | tee {log}` pipeline returns **`tee`'s** exit status, not `cmd`'s,
unless `pipefail` is active. Whether pipefail is active is **platform-conditional**,
and this was verified authoritatively against the installed Snakemake 9.6.2 source
(`.pixi/envs/default/Lib/site-packages/snakemake/shell.py`):

- `_get_process_prefix` (lines 82–89) injects the prefix `"set -euo pipefail; "` in
  front of every `shell:` command **only when the resolved shell executable name is
  `bash` or `bash.exe`** (and no custom `_process_prefix`/`shell.prefix` is set —
  verified: the repo sets none; a `grep` for `pipefail|shell.prefix|shell.executable`
  matches only docs/design text, no Snakefile).
- The platform switch (lines 372–390) sets the default executable: on POSIX,
  `shell.executable("bash")` (line 388) → the prefix **is** injected → a non-zero
  Rscript/Julia exit propagates through `| tee` and Snakemake fails the rule. This
  is why arch-3 is correct **for the Docker/Linux path** and the R3 pattern has run
  safely in production there.
- **On Windows** (line 390) the default is `shell.executable(None)` → the command
  runs via `cmd.exe` (lines 175–177 use `cmd_exe_quote`), where **no pipefail prefix
  is injected at all** and `cmd.exe` returns the last pipe stage's (`tee`'s) status.

**Consequence for R5.** R5's exit criteria are **Windows-only** (intake platform
constraint). On cmd.exe a `cmd | tee {log}` pipeline cannot be made exit-safe (no
pipefail exists there; the pipeline status is the last stage's), so a crashed
weathergenr Rscript (e.g. the documented `attempt to select less than one element`
or the `>= 16` wavelet abort) or a crashed Wflow run could be reported successful
and let a downstream rule consume a truncated netCDF — false-success semantics in
the workflow whose per-run compute is the most failure-prone in the repo.

**Empirical verification (2026-07-20, this machine: Windows 11, Snakemake 9.6.2 via
`pixi run`, cmd.exe).** Both mechanisms were exercised with a deliberately failing
Rscript in a scratch Snakefile (rule output created by the command, then
`quit(status=1)`):

- `Rscript --vanilla -e "..." > {log} 2>&1` → **Snakemake fails the rule** — the run
  reports `(command exited with non-zero exit code)`, exits 1, and removes the
  rule's output. The redirect preserves the subprocess exit code on cmd.exe —
  **verified, not inferred**. Stdout+stderr land in the log file; on failure the
  Snakemake error names the log ("check log file(s) for error details"), so the
  operator is pointed at the captured output.
- The `2>&1 | tee {log}` variant **resolves and runs** on this machine — `tee`
  executes and the rule creates its output — **but MASKS the failure**: with the
  same deliberately-failing command, Snakemake **reports success** (exit 0, output
  kept). The mechanism is consistent with the source analysis above: cmd.exe gets
  **no** `set -euo pipefail` prefix (bash-only), so the `| tee` pipeline returns
  `tee`'s exit 0 and the upstream non-zero status is swallowed. (This corrects
  v3's "tee not resolvable → unconditional failure" reading — driver verification
  2026-07-20;
  corroborated by R3 sealing via a full `--forceall` wf1 rebuild that wrote all
  three tee logs and passed 14/14 on this machine.) So `| tee` is the wrong form
  for wf3: on the exercised platform it silently reports success on a failed
  command, exactly the false-success semantics wf3's failure-prone fan-out must
  avoid.

**Ruling (ext1-1 — supersedes v2's Group A "classify, do not fix" ruling): resolve
substantively in R5 — wf3's shell rules use `> {log} 2>&1`, not `| tee {log}`.**
The v2 routing was wrong on its own terms: the masking is "inherited" only where
the tee pattern already exists (wf1). wf3 has **no** shell-rule redirects today, so
the redirect form is R5's fresh choice — and introducing `| tee` would introduce
the defect into rules that do not carry it. Options weighed:

- **`> {log} 2>&1` — CHOSEN.** Portable (cmd.exe + bash), exit-preserving
  (verified above), zero new infrastructure. Cost: no live console streaming for
  these three rules. The roadmap's per-rule-logging deliverable is still met —
  each rule gets a complete log file — and live progress when wanted is one
  command away (`Get-Content -Wait <log>` on PowerShell / `tail -f` on POSIX);
  the R scripts' new `message()` progress lines (§5b) land in that log.
- **`| tee {log}` + pipefail (inline `set -o pipefail;` or
  `shell.prefix("set -euo pipefail; ")`).** Rejected: pipefail is bash-only —
  cmd.exe gets no `set -euo pipefail` prefix at all (§2 source analysis), so
  `| tee` still masks the subprocess exit code on the Windows exit-criteria
  platform (empirically confirmed above). `shell.prefix` additionally mutates all
  shell rules repo-wide (out of a wf3 refactor's scope).
- **A portable tee wrapper (e.g. `python dev/scripts/tee_run.py {log} -- <cmd>`:
  spawn the subprocess, stream to console + file, exit with the child's code).**
  Workable — exit-preserving *with* live streaming — but it is new cross-cutting
  infrastructure that all three Snakefiles should share, which is exactly the
  followup task's shape, not a wf3-local add. Deferred to the reframed followup as
  the candidate mechanism that could later restore live streaming repo-wide (wf3
  would then migrate in one mechanical commit).
- **An in-R logging framework.** Already rejected (Alternatives) — a second,
  inconsistent mechanism; unchanged from v2.

**Trade-off accepted, stated plainly: log-file-only capture (no live console) for
wf3's three shell rules; exit-code correctness wins.** For wf3's fan-out compute
(`RLZ_NUM × ST_NUM` Wflow runs; the documented weathergenr failure modes), a silent
false success feeding a truncated netCDF downstream is strictly worse than losing
live streaming — the failure-detection property is what the per-rule logs exist to
serve.

The cross-cutting followup is **reframed** — it covers wf1's existing rules only,
as a **latent robustness item, not a blocker**, with corrected empirical evidence:

> `dev/followups.md` (reframed, cross-cutting → wf1): "wf1's `| tee {log}` shell
> rules run correctly on success (verified 2026-07-20: `tee` resolves and runs on
> this Windows machine, and R3 sealed via a full `--forceall` wf1 rebuild that
> wrote all three tee logs and passed 14/14 here) but **MASK the exit code on
> failure**: cmd.exe gets **no** `set -euo pipefail` prefix (bash-only), so
> `cmd 2>&1 | tee {log}` takes `tee`'s exit 0 and a genuine command failure is
> reported as success (empirically: `| tee` → Snakemake reports success;
> `> {log} 2>&1` → Snakemake fails). On POSIX/bash Snakemake's
> `set -euo pipefail` prefix protects. Fix: migrate wf1's three shell rules
> (`Snakefile_model_creation` lines 89, 167, 182) to the `> {log} 2>&1` form R5
> uses for wf3, or adopt a portable Python tee wrapper repo-wide if live streaming
> must return. Owner: `cst-architect`. Activation: **next time wf1 shell-rule
> robustness is worked on** — this is a latent failure-masking gap, **not** a gate
> blocker (the R5 milestone gate's wf1 leg passes on clean commands)."

**Verification step for the chosen mechanism (ext1-1 — the commit-8 gate).** Before
commit 8 is accepted: a deliberately failing Rscript
(`Rscript --vanilla -e "quit(status=1)" > {log} 2>&1`) **and** a deliberately
failing Julia command (`julia -e "exit(1)" > {log} 2>&1`) each run under a scratch
Snakemake rule on Windows and must yield a **non-zero Snakemake result** with the
log file written. The Rscript half is already verified on this machine (the scratch
run above); the commit-8 gate repeats it with the Julia invocation included, using
the exact redirect form committed.

The **Scope statement** and §5b now claim — with the scratch-run evidence — exit-code
preservation for wf3's shell rules; no masking caveat applies to wf3.

**Rules that gain `log:`/`benchmark:`** (every rule whose `script:`/`shell:` does
computation or non-trivial IO; verified against the Snakefile):

| Rule | Kind | Owner | Log-path keying |
| ---- | ---- | ----- | --------------- |
| `extract_climate_grid` | `script:` | `extract_historical_climate.py` | plain `{rule}` |
| `climate_stress_parameters` | `script:` | `prepare_cst_parameters.py` | plain |
| `prepare_weagen_config` | `script:` | `prepare_weagen_config.py` | plain |
| `prepare_weagen_config_st` | `script:` | `prepare_weagen_config.py` | wildcard-keyed on `{rlz_num}_{st_num}` |
| `generate_weather_realization` | `shell:` (Rscript) | `generate_weather.R` | plain (writes all `cst_0`) |
| `generate_climate_stress_test` | `shell:` (Rscript) | `impose_climate_change.R` | wildcard-keyed on `{rlz_num}_{st_num}` |
| `climate_data_catalog` | `script:` | `prepare_climate_data_catalog.py` | plain |
| `downscale_climate_realization` | `script:` | `downscale_climate_forcing.py` | wildcard-keyed on `{rlz_num}_{st_num2}` |
| `run_wflow` | `shell:` (Julia) | Wflow | wildcard-keyed on `{rlz_num}_{st_num2}` |
| `export_wflow_results` | `script:` | `export_wflow_results.py` | plain |

**Sole exclusion: `copy_config`** (Snakefile lines 54–63) — its `script:`
(`src/copy_config_files.py`) performs only a deterministic verbatim file copy, no
computation to profile. Its output IS a baseline-gated target
(`snake_config_climate_experiment.yml`), but it is excluded on the "nothing to
profile" ground, exactly as R4 excluded `copy_config`. (Verified R4 precedent:
`dev/milestones/r04/climate-projections-design.md` § "What changes" 2.)

**Path convention (inherited verbatim from R3/R4):** `log:` →
`{project_dir}/logs/{rule}.log` for plain rules; **for the wildcard rules**, the log
path MUST embed the wildcards so concurrent jobs (`snakemake -c 3`) never collide —
built by **string concatenation** (not f-strings), because the wildcard braces must
survive to Snakemake, matching R4's string-concatenation log-path convention for
wildcard rules (intake constraint; R4 § "What changes" 2). E.g. `log:`
`f"{project_dir}/logs/generate_climate_stress_test/" + "rlz_{rlz_num}_cst_{st_num}.log"`.
`benchmark:` → the parallel `{project_dir}/benchmarks/{rule}[/...].tsv`. Logs/
benchmarks are ephemeral run artifacts under gitignored `{project_dir}/`, never
fingerprinted, never committed.

**Import-guard note (needed for *testability*, not for `tee_to_log`).** `tee_to_log`
wraps module-level code fine without any `if __name__` guard (`with
tee_to_log(snakemake.log[0]): <body>`), so the wrap itself does **not** require a
guard. A guard is needed only for **import-time testability** — so a unit test can
`import` the module without a live `snakemake` global. Two wf3 scripts run
module-level code against the `snakemake` global on import and are NOT import-clean
today:

- `src/prepare_weagen_config.py` — reads `snakemake.params.*`/`snakemake.output.*`
  at **module top level** (lines 6–8, then throughout), with **no guard at all**.
  Importing it raises `NameError: name 'snakemake' is not defined`. **R5 unit-tests
  its year math (§8), so this guard IS required — and it is more than a wrapper
  (repo-2):** the module-level config-assembly body must be **extracted into a named
  function** (proposed `build_weagen_config(...)`, plus at minimum a pure
  `compute_nr_years(middle_year, run_length)` helper carrying the line-28 arithmetic)
  so the year math is reachable **without** a `snakemake` global. The pure helper
  `read_yml` (line 11) is relocated above the guard, and the guarded `__main__` block
  calls `build_weagen_config` with `snakemake.params.*`. Merely wrapping the existing
  top-level statements in `if __name__ == "__main__"` would leave **no unit-testable
  surface** — the §8 deliverable would not be met.
- `src/downscale_climate_forcing.py` — reads `snakemake.output/input/params` at
  module top level (lines 11–19), **no guard**. But **§8 does NOT unit-test this
  script**, and `tee_to_log` does not need the guard — so a guard refactor here is
  **not required for R5's deliverables**. R5 therefore only wraps its body in
  `tee_to_log` (parsimony §5); the guard is deferred unless a future milestone
  unit-tests it. (Recorded as a scoped decision, not an omission.) **Wrap shape
  (risk-4), shown explicitly** — the `tee_to_log` context must enclose the top-level
  `snakemake.*` reads at lines 11–19, since they run at import before any function:

  ```python
  # src/downscale_climate_forcing.py (after §2 wrap)
  from src.snake_utils import tee_to_log
  with tee_to_log(snakemake.log[0]):
      nc_path   = snakemake.input.nc          # was line 11-19 top-level reads,
      model_dir = snakemake.params.model_dir  # now inside the tee_to_log context
      ...                                     # (rest of the module body)
  ```

The scripts that already have the nested guard (`extract_historical_climate.py`
lines 148–158, `prepare_cst_parameters.py` lines 86–96, `export_wflow_results.py`
lines 284–298, and **`prepare_climate_data_catalog.py` at line 117** — the
`if __name__ == "__main__":` / `if "snakemake" in globals():` guard is at line 117,
**not** the "33–51" cited in design-v1, which is docstring + top of the function body;
corrected per arch-4) only need the `tee_to_log` wrap added, not a guard refactor.

The single new guard (`prepare_weagen_config.py`, commit 3) copies the nested form
the sibling scripts already use (`extract_historical_climate.py` lines 149–150:
`if __name__ == "__main__":` wrapping `if "snakemake" in globals():`), and is paired
with the function-extraction above (repo-2). `downscale_climate_forcing.py` gets **no
guard** — only the `tee_to_log` wrap (commit 7). The guard + function-extraction
refactor is an **output-neutral structural commit** — it makes the functions
importable without changing any computed value (asserted output-neutral but
**unverified until run** — recorded as an open question).

### 3. The stress-test grid helper (roadmap deliverable)

`ST_NUM = (temp.step_num+1)*(precip.step_num+1)` is computed **twice** today:

- `Snakefile_climate_experiment` lines 30–34 (using `get_config` on
  `stress_test_cfg["temp"]`/`["precip"]`).
- `src/prepare_cst_parameters.py` lines 36–46 (`temp_step_num`, `precip_step_num`,
  `ST_NUM = temp_step_num * precip_step_num`) — plus the same file uses
  `temp_step_num`/`precip_step_num` (the per-axis `step_num + 1`) to size the
  `np.linspace` grids **and** as `range(...)` loop bounds (see below).

**Extract to one tested helper in `src/snake_utils.py`** (the shared module both the
Snakefile and the scripts already import — the natural home, no new module). Proposed
signature and the import it requires (repo-1):

```python
from collections.abc import Mapping   # NEW import in snake_utils.py (repo-1)

def stress_test_grid(stress_test_cfg: Mapping) -> tuple[int, int, int]:
    """Return (temp_step_count, precip_step_count, st_num) for a stress_test cfg.

    STRICT (ext1-2): `temp.step_num` and `precip.step_num` are REQUIRED — a
    missing axis section or step_num raises KeyError (parity with
    prepare_cst_parameters.py's current direct read), and a value that is not a
    non-negative integer raises ValueError. The helper never silently invents a
    grid. temp_step_count = temp.step_num + 1; precip_step_count =
    precip.step_num + 1; st_num = temp_step_count * precip_step_count.
    """
```

`snake_utils.py` currently imports only `contextlib`, `os`, `sys` (verified), so the
`from collections.abc import Mapping` line is added as part of commit 2 — **or** the
annotation is dropped to match `get_config`'s untyped style (repo-1's alternative).
Default: add the import (keeps the type hint); recorded so a copy-paste of the
signature does not raise `NameError: name 'Mapping' is not defined` at import time,
which would break **all three** Snakefiles that import `snake_utils`.

- The Snakefile replaces lines 30–34 with `_, _, ST_NUM = stress_test_grid(stress_test_cfg)`.
- `prepare_cst_parameters.py` replaces lines 36–46's `temp_step_num`/`precip_step_num`/
  `ST_NUM` derivation with `temp_step_num, precip_step_num, ST_NUM =
  stress_test_grid(stress_test_cfg)`. **All** downstream consumers of the two per-axis
  counts must be fed by the helper's return — the refactor is exhaustive across
  **three** consumer sites, not just the linspace (arch-5):
  1. the `np.linspace` grid sizing (lines 48–56),
  2. the `range(temp_step_num)` loop bound at **line 60**, and
  3. the `range(precip_step_num)` loop bound at **line 62**.
  Replacing only the derivation + linspace but not the two `range(...)` sites (or
  vice versa) would produce the wrong CSV count — so the helper returns both per-axis
  counts and every site reads them from the tuple.
- **Behavior-preservation check + the one deliberate error-path change (ext1-2).**
  The two current call sites **disagree** on a missing `step_num`: the Snakefile
  silently defaults it to 1 (`get_config(cfg["temp"], "step_num", 1) + 1`, lines
  30–34) while `prepare_cst_parameters.py` raises `KeyError`
  (`cfg["temp"]["step_num"] + 1`, lines 36/43, no default). v2 resolved to the
  lenient (Snakefile) form and mislabeled it hardening; the external review is
  right that default-1 is **leniency**: a malformed config missing `step_num`
  would silently produce a 2-step axis, changing `ST_NUM`, the number of
  parameter CSVs, and the entire `RLZ_NUM × ST_NUM` workload and response
  surface. **v3 resolves to the strict (script) semantics**: `step_num` is
  required on both axes (`KeyError` on a missing axis section or key — parity
  with the script's current behavior) and validated as a non-negative `int`
  (`ValueError` otherwise; `bool` excluded). On the seed config (`step_num`
  present: temp 1, precip 2 → `ST_NUM = 2 * 3 = 6`) both call sites compute
  exactly what they compute today — output-neutral. The one behavior delta is
  the **Snakefile call site's error path on an invalid config**: a loud raise at
  DAG-parse time instead of a silently invented grid — classified as
  **output-neutral hardening** (R3 §7.1 class: loud on an *invalid* config, no
  change on the *valid* seed), named as such in the commit-2 message. After the
  refactor both call sites share one strict contract; the helper never invents a
  grid.
- **Unit test** (`tests/test_stress_test_grid.py`) — asserts the strict contract
  at both call-site shapes: `stress_test_grid({"temp": {"step_num": 1},
  "precip": {"step_num": 2}}) == (2, 3, 6)` (the seed-config value);
  `pytest.raises(KeyError)` on a missing `temp.step_num`, a missing
  `precip.step_num`, and a missing axis section; `pytest.raises(ValueError)` on
  a non-integer (`"2"`, `1.5`, `True`) and a negative `step_num`. Pure function,
  no heavy deps — no `sys.modules` pollution risk (M02c discipline).

### 4. The `CyclicGraphException` resolution + `st_num2 → st_num` fold (IN scope)

**The defect — restated consistently as an ambiguity (risk-2).**
`generate_climate_stress_test` (Snakefile lines 125–133) declares:

- input `rlz_nc = f"{exp_dir}/realization_"+"{rlz_num}"+"/rlz_"+"{rlz_num}"+"_cst_0.nc"`
  (line 127) — **literal `cst_0`, NOT wildcarded on `st_num`**;
- inputs `st_csv` / `weagen_config` wildcarded on `{st_num}` (lines 128–129);
- output `rlz_st_nc = temp(".../rlz_"+"{rlz_num}"+"_cst_"+"{st_num}"+".nc")` (line 131)
  — **wildcarded on `{st_num}`**, and **under-constrained** so `{st_num}` can match `0`.

Meanwhile `generate_weather_realization` (lines 115–122) outputs concrete
`rlz_{rlz_num}_cst_0.nc` (line 120, a list comprehension). The precise mechanism:
because the line-131 output wildcard `{st_num}` can resolve to `0`,
`generate_climate_stress_test` is a **second eligible producer of
`rlz_{rlz_num}_cst_0.nc`** — the file `generate_weather_realization` already produces
(line 120) and that is **concretely requested** by `climate_data_catalog`'s `rlz_nc`
input (line 139). Two eligible producers of one concrete file **is a rule
ambiguity**; and because `generate_climate_stress_test` also names that same
`cst_0.nc` as its own literal input (line 127), the ambiguity presents as a
**self-loop** → `CyclicGraphException` on the test-config `--dry-run`. (design-v1's
"a genuine cycle, not a rule-ambiguity" phrasing is **withdrawn** — it is both: an
under-constrained output wildcard creates a second eligible producer, which is the
ambiguity, and the self-consumption makes it surface as a cycle.) Production configs
disambiguate from concrete `expand(...)` paths, but the test-config `--dry-run` trips
(`dev/followups.md` § R3; `tests/test_cli.py` lines 111–120).

**Resolution chosen: rule-local `wildcard_constraints`, not `ruleorder`.** Constrain
`st_num` to **strictly positive integers on the `generate_climate_stress_test` rule
specifically** (governing its line-131 output wildcard and its `{st_num}`-wildcarded
`st_csv`/`weagen_config` inputs), so that rule can never resolve `{st_num}` to `0`
and thus can never be an eligible producer of `cst_0`:

```python
rule generate_climate_stress_test:
    wildcard_constraints:
        st_num=r"[1-9][0-9]*",     # perturbed only; cst_0 is the baseline → breaks the ambiguity/cycle
    input: ...
```

The constraint leaves the literal `rlz_nc` input (line 127) still depending on
`generate_weather_realization`'s `cst_0` output as intended. After the constraint,
**exactly one rule remains eligible to produce `cst_0.nc`** — `generate_weather_realization`.

**Completeness enumeration — every requester of `cst_0.nc`, both `run_historical`
settings (risk-2).** `cst_0.nc` (`rlz_{rlz_num}_cst_0.nc`) is requested by:

- `generate_climate_stress_test`.`rlz_nc` (line 127) — literal `cst_0`, **always**
  (independent of `run_historical`). Sole producer after the constraint:
  `generate_weather_realization`. ✓ one producer.
- `climate_data_catalog`.`rlz_nc` (line 139) — concrete list over `rlz_num`,
  **always**. Sole producer: `generate_weather_realization`. ✓
- **Under `run_historical=true` only** (`ST_START=0`): the downstream
  `downscale_climate_realization` (line 151, `st_num2`) and `run_wflow` (line 167)
  consume `cst_0`-derived files, and `export_wflow_results`'s `expand(..., st_num2 =
  np.arange(ST_START, ST_NUM+1))` (line 178) includes `st_num2=0`. These consume the
  **downscaled/forcing** `cst_0` products, whose ultimate `.nc` source is
  `generate_weather_realization`'s `cst_0` (via `downscale_climate_realization`'s
  `nc` input, line 151). After the fold (5b) folds `st_num2 → st_num`, these become
  `st_num=0` requests — still produced by `generate_weather_realization` (the
  rule-local `[1-9][0-9]*` bars **only** `generate_climate_stress_test` from `0`, not
  the downstream rules). ✓ one producer in every case.

The seed config has `run_historical=false` (config line 57), so the `st_num2=0`
downstream path is **never exercised by the default dry-run**. Therefore a
**`run_historical: true` dry-run is added to the commit-5a/5b verification** (below)
to exercise the `cst_0` downstream path before the seal — otherwise the fold's
interaction with `cst_0` downstream is untested.

Rationale for `wildcard_constraints` over `ruleorder`:

- The real cause is an **under-constrained `{st_num}` wildcard** matching `0`, making
  the stress-test rule an unintended second producer. `wildcard_constraints
  st_num=[1-9][0-9]*` removes that eligibility at the source. A `ruleorder` would
  merely **prioritize** one of two eligible producers — masking the wildcard
  over-match rather than fixing it, and leaving the rule still *able* to match `0`.
- The constraint encodes the naming-§4 semantics directly (`st_num`:
  `1..stress_test_count` perturbed; `0` = reserved unperturbed baseline).
  `dev/followups.md` § R3 names `{st_num,[1-9][0-9]*}` as one of the two acceptable
  fixes — this design picks it.

**Ancient()/producerless-leaf enumeration — region.geojson is the SOLE unbuilt
external leaf (arch-2).** Verified by reading every `input:` block in
`Snakefile_climate_experiment` (lines 47–189):

| Input | Rule | Producer within wf3? | External leaf? |
| ----- | ---- | -------------------- | -------------- |
| `ancient({basin_dir}/staticgeoms/region.geojson)` | `extract_climate_grid` (line 68) | **No** — produced by workflow 1 | **YES — sole unbuilt leaf** |
| `config_path` | `copy_config` (line 56), `climate_stress_parameters` `ancient(config_path)` (line 80) | n/a — the `--configfile` itself | No (repo-tracked; always present) |
| `DATA_SOURCES` (deltares catalog YAML) | `downscale_climate_realization` `data_sources` (line 152) | n/a | No (repo-tracked; always present) |
| `ancient(.../extract_historical.nc)` | `generate_weather_realization` (line 117), `extract_climate_grid`'s own output | **Yes** — produced by `extract_climate_grid` (line 73) | No (buildable) |
| `weathergen_config.yml` | `generate_weather_realization` (line 118) | **Yes** — `prepare_weagen_config` (line 89) | No |
| `cst_0.nc`, `cst_{st_num}.nc`, `cst_{st_num}.csv`, weagen st YAMLs, `data_catalog_...yml`, `inmaps_...`, TOMLs, output CSVs | downstream rules | **Yes** — all produced within wf3 | No |

**Conclusion:** the wf3 DAG has exactly **one** input with no producing rule that is
absent from a fresh `project_dir` — `region.geojson`. The two other externals
(`config_path`, `DATA_SOURCES`) are repo-tracked files always present. So a bare
dry-run against an empty `project_dir`, once the cycle is removed, blocks on
**`region.geojson` only**, and staging that single file is sufficient. (arch-2 also
confirms the wf2 fixture path is reusable: `basin_dir = f"{project_dir}/hydrology_model"`,
so `{basin_dir}/staticgeoms/region.geojson` is the **same path** wf2's
`config_with_staged_region` stages.)

**Paired `st_num2 → st_num` fold (naming §4, R5-assigned) — now a SEPARATE commit
(Group B / arch-1).** The downstream rules `downscale_climate_realization`,
`run_wflow`, and `export_wflow_results` (Snakefile lines 149–189) use a **separate
wildcard `st_num2`** because they admit `0` under `run_historical` (where
`ST_START = 0` and `expand(..., st_num2=np.arange(ST_START, ST_NUM+1))` at line 178
includes `cst_0`). `dev/conventions/naming.md` §4 (lines 77–80) flags `st_num2` as
"a known inconsistency — fold into `st_num` during **R5**."

- **The fold folds `st_num2 → st_num`** in the downstream rules
  (`downscale_climate_realization` lines 151/154/155, `run_wflow` lines 167/168/170/171,
  `export_wflow_results`'s `expand(..., st_num2=...)` line 178), giving a single
  `st_num` vocabulary. Because those rules must still admit `0` under
  `run_historical`, the fold relies on the **rule-local** `[1-9][0-9]*` constraint
  living **only** on `generate_climate_stress_test` — the downstream rules keep the
  default `st_num` match (which admits `0`). No global `st_num` constraint is
  needed; the rule-local scope is exactly the surgical fix.
- **Why it is decoupled from the cycle fix (arch-1).** The cycle fix (rule-local
  `[1-9][0-9]*`) is **complete on its own** — it removes the second `cst_0` producer
  regardless of whether `st_num2` is folded. The fold is a **naming-convention
  cleanup** on top. design-v1 bundled them (plus the ratchet flip) into one atomic
  commit; because the fold's global-resolution safety "must be verified by an actual
  dry-run at implementation," a failed fold dry-run would leave the atomic commit
  half-done (ratchet test already rewritten to `returncode == 0`, DAG not building).
  **v2 splits them** so the ratchet flip depends only on the guaranteed fix.

**How the `test_cli` ratchet flips (IN scope, not a deferral).** The current ratchet
(`tests/test_cli.py` lines 111–120) asserts the workflow-3 dry-run **fails** with
`CyclicGraphException`:

```python
def test_snakefile_cli_climate_experiment_known_cyclic_graph():
    result = _dry_run("Snakefile_climate_experiment")
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, combined
    assert "CyclicGraphException" in combined, combined
```

Its own header (lines 95–108) specifies the flip: when R5 fixes the cycle the dry-run
succeeds, `returncode != 0` FAILs, and the fixer converts it to a plain
`returncode == 0` success assertion. R5 therefore, **in the same commit as the cycle
fix (5a)**, rewrites this test to:

```python
def test_snakefile_cli_climate_experiment():
    """Workflow 3 dry-run builds a clean DAG on the test config (R5 fixed the cycle)."""
    result = _dry_run_staged_region("Snakefile_climate_experiment")  # staged-region fixture
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
```

and removes the "Known-failure ratchet (deferred to R5)" comment block (lines
95–108). This is the ratchet flip; a **live** behavior change to the test suite,
wholly within R5 scope. `pytest tests/test_cli.py` must pass after the commit.

**The ratchet flip uses a staged-region fixture (default path).**
`extract_climate_grid` takes `ancient(f"{basin_dir}/staticgeoms/region.geojson")`
(line 68) — the **same construct + same path** that forced R3 to add
`config_with_staged_region` for workflow 2 (`tests/test_cli.py` lines 38–59). The
current wf3 bare dry-run reaches the **cycle before the input-existence check** (it
reports `CyclicGraphException`, not `MissingInputException`). Once the cycle is
removed, the DAG builds and the `ancient()` input existence gets checked → a **bare**
dry-run against an empty `project_dir` will raise `MissingInputException`. Per the
leaf enumeration above, **`region.geojson` is the sole unbuilt leaf**, so staging
exactly that one file is sufficient. The ratchet-flip commit therefore **mirrors
wf2's `config_with_staged_region` fixture** (stage a minimal valid `region.geojson`
under a test-owned `tmp_path` project_dir, point a copy of the test config at it,
assert `returncode == 0`). The "bare dry-run passes with no fixture" outcome is the
**contingency**, adopted only if the implementation dry-run shows the `ancient()`
input does not block a bare wf3 dry-run.

### 5. R-layer cleanup: `generate_weather.R` + `impose_climate_change.R`

The roadmap asks for "cleaner argument parsing, fewer positional args, consistent
logging." Concrete decisions, grounded in the current scripts:

**5a. Argument parsing — keep positional args, add named binding + validation
(NOT a full parser).** Both scripts take positional args via
`commandArgs(trailingOnly=TRUE)` (`generate_weather.R` line 8: `args[1]` =
weather-input nc, `args[2]` = weagen config yaml; `impose_climate_change.R` line 8:
`args[1]` = realization nc, `args[2]` = weagen config yaml, `args[3]` = stress csv).
The `shell:` directives pass them positionally (Snakefile lines 122, 133). The
cleanup:

- **Bind positionals to named locals immediately, with an arity check.** Replace the
  bare `args[1]`/`args[2]`/`args[3]` scattered through each script (note
  `generate_weather.R` currently reads `args[2]` before `args[1]`, lines 11–12) with
  a single guarded block:

  ```r
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2L) stop("generate_weather.R expects 2 args: <climate_nc> <weagen_config_yaml>")
  climate_nc_path    <- args[[1]]
  weagen_config_path <- args[[2]]
  ```

  (three-arg equivalent for `impose_climate_change.R`.) This is the R-idiomatic
  "fewer positional args" win: positions read **once**, validated, and named.
  **Placement (repo-5):** the block goes **immediately after
  `commandArgs(trailingOnly=TRUE)` AND after the existing
  `source("./src/weathergen/global.R")`** (`generate_weather.R` line 2,
  `impose_climate_change.R` line 6 — sourced with a wd-relative path), so the arity
  `stop()` is the first thing that touches `args` and is not shadowed by a sourcing
  error. It keeps the `shell:` call sites unchanged (still positional — no Snakefile
  churn on the shell strings beyond the log tee), so it is behavior-preserving on
  valid input and adds a **loud error on wrong arity** (classified as output-neutral
  hardening).
- **Do NOT introduce `optparse`/`argparse`-style flags.** A flag parser would change
  the `shell:` call contract (Snakefile lines 122, 133) and buys nothing for a
  2–3-arg internal script — parsimony criterion. (Alternative weighed below.)

**5b. Logging — via the `shell:` `> {log} 2>&1` redirect (exit-preserving; ext1-1),
NOT `| tee` and NOT an in-R logging framework.** The R scripts run under `shell:`;
R5 adds the portable exit-preserving redirect (§2's ruling):

```
"""Rscript --vanilla src/weathergen/generate_weather.R {input.climate_nc} {input.weagen_config} > {log} 2>&1"""
```

(the same form on the `impose_climate_change.R` shell string and the Julia
`run_wflow` shell string). This captures everything the R script prints to
stdout/stderr — including weathergenr's own progress/error output — with **zero
in-R logging code**, giving all three engines (Python via `tee_to_log`, R + Julia
via the redirect) a per-rule log file.

**Exit-code note (ext1-1, resolved).** The redirect has no pipe stage, so the
Rscript/Julia exit code reaches Snakemake unaltered on both cmd.exe and bash —
verified empirically in §2 (a `quit(status=1)` Rscript fails the rule with a
non-zero Snakemake result). No masking caveat applies to wf3. The accepted cost is
no live console streaming for these three rules; the log is tailable
(`Get-Content -Wait` / `tail -f`) and Snakemake's failure message names it.

The R scripts should still add a small number of `message()`/`cat()` progress lines
where they currently have none (e.g. "Reading weather netcdf", "Generating N
realizations") so the captured log is informative — a **documentation/observability
add, output-neutral** (does not change any written netCDF).

**5c. The `spatial_ref` workaround — LEAVE IN, document, do NOT remove (candidate #3
ruling).** `generate_weather.R` lines 69–89 manually copy `spatial_ref` attrs from
the historical template to each realization file, working around
`weathergenr::write_netcdf` not propagating them (`dev/followups.md` § R5, item 2).
Removing the workaround requires the **upstream weathergenr fix** to have landed;
without it, `impose_climate_change.R` crashes (documented failure mode:
`attempt to select less than one element`). R5 does **not** own weathergenr, and the
workaround is load-bearing for the pipeline to run at all. Ruling: **keep the block,
tighten its comment to name the upstream issue + the exact removal condition**
("drop when tanerumit/weathergenr `write_netcdf` propagates `spatial_ref` — see
`dev/followups.md` § R5"). Removing it is a **behavior-changing** act gated on an
external fix → split to a tracked upstream weathergenr task, not absorbed.

**5d. The wavelet `>= 16` cryptic error (candidate #3, item 3) — upstream task,
NOT R5.** `weathergenr`'s `wavelet_cwt.R` enforces `length(series) >= 16` and surfaces
`'series' must have at least 16 observations` (`dev/followups.md` § R5, item 3). The
fix lives in the weathergenr package, not this repo. Ruling: **leave as a tracked
upstream weathergenr issue**; R5's R-layer cleanup does not touch weathergenr package
internals. (The related repo-side mitigation — warning when `extract_climate_grid`
produces < 16 years — is coupled to candidate #2, ruled below.)

### 6. `impose_climate_change.R` / downscaling scientific audit (finding-classification pre-committed)

The roadmap names `impose_climate_change.R` and "the downscaling rules" for review.
Audit targets, grounded in the code, each routed through the pre-committed
finding-classification policy **before** any code moves:

- **`impose_climate_change.R`** (perturbation math): the script calls
  `weathergenr::apply_climate_perturbations` (lines 36–48) with `precip_mean_factor`,
  `precip_var_factor`, `temp_delta` read from the per-stress-test CSV (`cst_data`,
  columns `precip_mean`/`precip_variance`/`temp_mean`, lines 40–42), and
  `temp_transient`/`precip_transient` from the weagen config (`yaml$temp$transient_change`
  etc., lines 26–27, 43–44), with `compute_pet=TRUE`, `qm_fit_method="mme"`,
  `diagnostic=FALSE`. **Audit question:** do the CSV columns
  (`prepare_cst_parameters.py` writes `temp_mean`, `precip_mean`, `precip_variance`,
  lines 67–71) match the R read (`cst_data$temp_mean`, `$precip_mean`,
  `$precip_variance`) — i.e. is the multiplicative-precip / additive-temp semantics
  consistent across the Python producer and the R consumer? And does the
  `precip_variance` grid — which `prepare_cst_parameters.py` builds from
  `variance.min`/`variance.max` but with a **suspicious `max = ...["min"]`** (line 42:
  `delta_precip_variance_max = stress_test_cfg["precip"]["variance"]["min"]` — reads
  **min** into the max variable) — represent intended behavior? On the seed config
  variance min == max == 1.0 (config lines 71–73), so the grid is uniformly 1.0 and
  the bug is **latent** (no output effect on the seed). This is a **pre-committed
  finding**: a real bug (max reads min) but **output-neutral on the seed** →
  classified below.
- **`downscale_climate_forcing.py`** (the downscaling rule): reads the perturbed
  realization nc, builds the `startyear`/`endyear` window from `horizontime_climate ±
  run_length/2` (lines 21–24), instantiates `WflowSbmModel` and redirects writes to
  the per-realization run dir (line 38 onward). **Audit question:** are the
  downscaling window and the `pet_method` dispatch (`makkink` for eobs else `debruin`,
  line 27) correct and config-consistent? This is a reading item; no output-effect
  expected on the seed.

**Finding-classification policy (pre-committed, before the audit runs).** Every audit
finding is assigned one of three classes **before** any code is written, and the class
determines the action:

| Class | Definition | Action | Moves baseline? |
| ----- | ---------- | ------ | --------------- |
| **Noise** | Cosmetic / already-correct / cannot change any output value. | Document in the contract doc / audit table. Leave code as-is. | No. |
| **Defect** | A real correctness bug whose fix **would change an output value** or an error-path outcome on a valid config. | **Do not fix inline** (this holds even when the audit proves the defect). Split to a dedicated task naming a concrete owner + activation condition, recorded in `dev/followups.md` **and** enumerated at the R5 seal in `dev/roadmap.md`. **Exception:** an output-neutral hardening (loud error on an *invalid* config, no change on the *valid* seed config) may land in R5, classified as such in the commit message, mirroring R3 §7.1. | No, unless re-chartered as intentional. |
| **Intentional change** | A finding R5 *elects* to fix in-scope with eyes open, accepting the output move. | Fix in a dedicated commit; record the exact delta in `dev/milestones/r05/baseline_diffs.md`; run `check_baseline record` for **only** the affected wf3 target(s) after a full regeneration; state the scientific justification. | **Yes** — the only class that edits the manifest. |

**The `precip_variance` max-reads-min bug is pre-classified: Defect, split (G1
ratified).** It is a genuine bug (`prepare_cst_parameters.py` line 42 reads `["min"]`
into the max variable), but on the seed config `variance.min == variance.max == 1.0`,
so fixing it changes no seed output; on any config with `variance.max ≠ variance.min`
it would change output. **The G1 gate ratified splitting this to a task (not fixing
it in R5)** — that decision STANDS. The split task is named concretely:

> **`t260720a` — fix `precip_variance` max-reads-min in `prepare_cst_parameters.py`
> line 42 (`["min"]` → `["max"]`).** Owner: `cst-architect` (to route to
> `python-engineer` for the one-token fix + baseline re-record). Activation
> condition: **any config exercising a non-degenerate precip-variance range
> (`variance.max ≠ variance.min`)** — recorded in `dev/followups.md` and enumerated
> at the R5 seal in `dev/roadmap.md`.

R5 does not fix it inline. **A one-line output-neutral comment** MAY land at the
suspicious site (documenting the known bug + referencing `t260720a`) — classified as
documentation, not a fix. **The §8 characterization test does NOT assert the buggy
value as correct** (risk-3) — see §8: it is wired `xfail(strict=True)` against the
**intended** value referencing `t260720a`, so the suite **flags** the defect and the
fix trips it (xpass) rather than the suite defending the bug.

The governing rule: **the audit is a reading exercise that produces a classified
findings table; only the "intentional change" class edits the baseline, and only
with a `dev/milestones/r05/baseline_diffs.md` entry.** Default disposition for a genuine bug is
**split, don't absorb** — matching R3/R4 and the G1 ratification.

### 7. Naming: `_fid` → `_path` in workflow-3 rules and scripts

Per `dev/conventions/naming.md` §3/§5 (R5 owns wf3 identifiers), rename deprecated
`_fid` labels **inside this Snakefile's rules and the scripts it calls**, paired
script reads updated in the same commit:

- `run_wflow` input labels `forcing_fid` / `toml_fid` (Snakefile lines 167–168) →
  `forcing_path` / `toml_path`. The shell string references `{input.toml_fid}` (line
  173) → `{input.toml_path}` in the same commit.
- Any `_fn` locals inside the scripts that are paths are reviewed against naming §5's
  path/object rule. **Renamed where they are pure new-code-adjacent path strings and
  the rename does not cross a contract surface; left grandfathered where they are
  function *parameters* of the analysis helpers** (renaming those ripples into the
  `snakemake.*` keyword calls — grandfathered unless proven output-neutral and
  self-contained). The **surgical set** R5 commits is the Snakefile `input:`/`output:`
  **labels** (`forcing_fid`/`toml_fid`) plus their same-rule shell/script references;
  deeper parameter-name churn inside the Python helpers is **deferred** (parsimony).
- **§7 contract surfaces stay grandfathered:** output filenames (`Qstats.csv`,
  `basin.csv`), config keys, catalog source names, and the cross-workflow input path
  `staticgeoms/region.geojson`. No output filename changes → **no migration note
  required**.

### 8. Unit tests for the workflow-3 Python helpers (M02c discipline)

Per M02c testing discipline (`dev/followups.md` § R3+): prefer **extract-and-unit-
test** over driving heavy functions; **monkeypatch over `sys.modules.setdefault`**
for any shared heavy dep; no network, no full builds. New tests target the pure /
near-pure functions:

- **`snake_utils.stress_test_grid`** (new, §3) — the grid arithmetic. Pure, trivial,
  no deps. Primary new test.
- **`prepare_cst_parameters.prep_cst_parameters`** (line 11) — drives the CSV
  generation on a **synthetic in-memory config dict** to a `tmp_path`, asserts the
  number of CSVs = `ST_NUM`, the column names (`temp_mean`/`precip_mean`/
  `precip_variance`), and the linspace endpoints. This function is already
  import-clean (guarded, lines 86–96) and uses only pandas/numpy/yaml — no heavy-dep
  stub, no pollution risk. **The `precip_variance` characterization test (risk-3) is
  wired to flag, not defend, the split defect `t260720a`:**

  ```python
  @pytest.mark.xfail(strict=True, reason="precip_variance max-reads-min bug; fixed by t260720a")
  def test_precip_variance_grid_uses_max_endpoint(tmp_path):
      # Non-degenerate variance range: min != max.
      cfg = {..., "precip": {"variance": {"min": [1.0]*12, "max": [1.5]*12}, ...}, ...}
      prep_cst_parameters(cfg, tmp_path)
      grid = _read_precip_variance_grid(tmp_path)
      # Assert the INTENDED (fixed) behavior: the grid spans up to max (1.5), not min.
      assert grid.max() == pytest.approx(1.5)   # FAILS today (bug collapses max→min) → xfail
  ```

  The test asserts the **intended** value (`max` endpoint honored), marked
  `xfail(strict=True)`, so **today it xfails** (the bug collapses the range to `min`)
  — flagging the defect — and **when `t260720a` fixes `["min"]`→`["max"]` it xpasses**,
  tripping the strict-xfail and forcing the marker's removal. The suite therefore
  **flags** the defect rather than pinning the buggy value as correct (the R4 V/P
  characterization-test pattern). No test asserts the degenerate output as correct.
- **`export_wflow_results.analyze_wflow_results`** (line 15) — heavy (imports
  `metrics_definition`, xarray, pandas; reads many CSVs). **Do NOT drive it
  end-to-end** (M02c lesson). If a unit is wanted, extract a small pure sub-function
  (e.g. the `st_nb`/`rlz_nb` filename parsing at lines 120/155) and test that;
  otherwise cover it only via the end-to-end milestone gate. **Default: no extract**
  unless the reviewer wants one (parsimony) — recorded as an open question.
- **`prepare_weagen_config.py`** — after the §2 function-extraction (repo-2:
  `build_weagen_config` / `compute_nr_years`), unit-test the `nr_years_weagen`
  computation (line 28: `math.ceil((middle_year + wflow_run_length/2) - 2010 + 2)`)
  on synthetic params. **The year math lives ONLY in the `generate` branch (repo-3);**
  the `stress_test` branch (`cftype != "generate"`, lines 44–56) carries **no**
  arithmetic — only dict assembly (`output.path`/`nc.file.prefix`/`nc.file.suffix` +
  config-section copies). So the year-math assertion targets the `generate` branch;
  if a `stress_test`-branch test is added, it asserts the **assembled dict shape**,
  not arithmetic. The `2010` and `+2` literals are magic numbers the test pins (and
  the audit flags as candidates for the contract doc).

**R testthat coverage — DECISION: NO (locked at R5 start; G1 ratified).** See
Alternatives. R5 adds **no R testthat suite**. Rationale: (a) the two R scripts are
thin adapters — they parse args, call `weathergenr` functions, and write netCDFs; the
scientific logic lives **upstream in weathergenr**, which has its own repo and test
surface. (b) The repo has **no R test harness** today (no `testthat`, no
`tests/testthat/`, no R CI wiring); standing one up is infrastructure R5 is not
chartered to build (R6 territory at most). (c) The R scripts' correctness is already
gated end-to-end by the milestone baseline run and by the `test_cli` dry-run. (d) The
M02c pollution lessons are Python-specific; there is no analogous R lesson. **The
arity-check hardening (§5a) is the R-layer's correctness net.** Activation condition
to reconsider: a future milestone extracting non-trivial R logic *into this repo*.

## Candidate absorptions — rulings

The intake requires an explicit in/out ruling on all **five** candidates.

**1. The workflow-3 `CyclicGraphException` `test_cli` ratchet + the `st_num2 →
st_num` fold (naming §4) — RULING: IN scope.** It flips a **live** ratchet
(`tests/test_cli.py` lines 111–120). Resolution: rule-local
`wildcard_constraints st_num=r"[1-9][0-9]*"` on `generate_climate_stress_test`
(removes the second `cst_0` producer → breaks the ambiguity/cycle), with the ratchet
test rewritten to a plain `returncode == 0` assertion **on a staged-region config**
(commit **5a**, guaranteed). The `st_num2 → st_num` fold is a **separate commit
(5b)**, contingent on its own dry-run (including a `run_historical: true` dry-run);
the cycle fix does not depend on the fold. Full detail § "What changes" 4.

**2. `extract_climate_grid` `historical:` config wiring (R5 followup #1) — RULING:
IN scope, as a chartered behavior-preserving wiring fix.**
`src/extract_historical_climate.py` hardcodes `starttime="2000-01-01T00:00:00"` /
`endtime="2020-12-31T00:00:00"` (lines 156–157), and `Snakefile_climate_experiment`
never passes the config window. The config *does* carry the window under
`shared.historical_window.{starttime,endtime}` (config lines 21–23) — **but the
followup calls the key `historical:`**, whereas the seed config nests it under
`shared.historical_window`. The fix: (a) parse `shared.historical_window.starttime`/
`.endtime` in the Snakefile (via `get_config`), (b) pass them as `extract_climate_grid`
rule params, (c) replace the hardcoded strings in `extract_historical_climate.py`
lines 156–157 with `sm.params.starttime`/`sm.params.endtime`, and (d) drop the
misleading function-signature defaults (lines 20–21).

**Behavior-preservation consequence (why it is chartered, not free):** the seed
config's `historical_window` is `2000-01-01`/`2020-12-31` (config lines 22–23),
**byte-identical** to the current hardcoded strings (lines 156–157). So wiring the
config through is **output-neutral on the seed** — and the affected `extract_historical.nc`
is a `temp`/`ancient` intermediate that is not even a manifest target. **This is a
computational-path edit (risk-5): asserted output-equivalent, confirmed by the
targeted before/after `extract_historical.nc` characterization (ext1-3, Verification
plan) plus the milestone gate** — not an untouched path. If implementation finds the seed window and
the hardcoded strings ever differed, the fix would be behavior-changing and re-route
through the split policy; on the verified current values they match. (Verified: config
lines 22–23 == script lines 156–157.)

**3. weathergenr `spatial_ref` workaround (#2) and wavelet-message (#3) — RULING:
OUT of R5 as package changes; the `spatial_ref` workaround block STAYS in-repo with
a tightened removal-condition comment.** Both are **upstream weathergenr package**
fixes (`dev/followups.md` § R5, items 2–3). R5's R-layer cleanup **touches the file**
containing the `spatial_ref` workaround (`generate_weather.R` lines 69–89), so it
naturally re-comments it (name the upstream issue + removal condition) — but does
**not remove** it (removal is gated on the upstream fix and would break the pipeline).
The wavelet-message fix is entirely inside the weathergenr package (`wavelet_cwt.R`) —
tracked as a separate weathergenr issue. Full detail § "What changes" 5c/5d.

**4. Parked per-rule `message:` progress directives (`dev/followups.md`,
cross-cutting) — RULING: OUT of R5; leave parked.** The followup is
**[PARKED 2026-07-19]** and cross-cutting; R4 left it parked. Absorbing it for wf3
alone would create the one-of-three inconsistency the followup warns against and break
inheritance fidelity (R3/R4 added `log:`/`benchmark:` **without** `message:`). Ruling:
**leave parked**; R5 adds `log:`/`benchmark:` only. (The §5b suggestion to add a few
in-R `message()`/`cat()` lines is a different thing — in-script progress prints
captured by the log redirect, not Snakemake `message:` directives — and is
output-neutral observability, not the parked cross-cutting item.)

**5. R testthat coverage — DECISION: NO (locked, G1 ratified).** Full rationale §
"What changes" 8 and Alternatives. Activation to reconsider: a future milestone
extracting non-trivial R logic *into this repo*.

## Behavior-preservation stance and exact baseline consequence

**R5 is a behavior-preserving refactor. Default: zero manifest edits.** The gating is
**two-tier** (inherited from R4):

- **Per-commit gate:** `--dry-run` + `pytest tests/test_cli.py` after every
  Snakefile-touching or script-signature-touching commit. This does **not** regenerate
  manifest targets.
- **Milestone-level gate:** the full baseline comparison runs **once**, end-to-end at
  the milestone, comparing fingerprints of *regenerated* targets (Verification plan
  below).

Discriminating self-check: walk the commit plan — none of the commits alters a
computed value or a fingerprinted byte, **with two asserted-output-equivalent
exceptions (risk-5)**: commit 6 (`historical_window` wiring) and commit 9 (R
arg/logging cleanup) **do edit computational paths**, but are asserted byte-neutral on
the seed and confirmed by the **targeted intermediate-equivalence characterization**
(Verification plan, ext1-3) plus the milestone gate. So **no commit requires `record`**
under the default plan.

**Exact consequence, stated precisely:** the **3 workflow-3 manifest targets**
(`check_baseline.py` `TARGETS` lines 70–72 — `Qstats.csv`, `basin.csv`,
`snake_config_climate_experiment.yml`) stay byte-for-byte / stat-for-stat identical
across all R5 commits. The only place a commit *could* move the baseline is an
**intentional change** under the finding-classification policy — of which there is
**no live candidate in R5** (the `precip_variance` bug is split to `t260720a`, not
fixed; the `historical_window` wiring is neutral on the seed; the `spatial_ref`/wavelet
items are upstream). **So R5 plans zero `record` and writes no
`dev/milestones/r05/baseline_diffs.md` under the default plan.** If the audit unexpectedly
promotes a finding to intentional change, that lands as a dedicated commit paired with
`dev/milestones/r05/baseline_diffs.md` + a targeted `record` before the seal.

**Gate-coverage honesty (repeated caveat, corrected framing — risk-5).** The manifest
fingerprints only the 2 reduced summary CSVs + the config snapshot. The
per-realization forcing, the perturbed netCDFs, and the raw Wflow discharge CSVs are
`temp`/non-manifest — a clean `check` proves the two summaries unchanged, **not** that
every intermediate is. R5's guarantee for the intermediates rests on **not editing the
computational path except the two chartered, asserted-output-equivalent edits (commits
6 and 9), which are confirmed by targeted before/after characterization of the exact
intermediates they touch (ext1-3) plus the milestone gate.** The claim "no
computational path is edited" is **withdrawn** as literally false for those two
commits; the honest claim is "paths edited only in commits 6 and 9, asserted
output-equivalent, confirmed by the targeted intermediate characterization and the
milestone gate." The v2 formulation ("confirmed only at the milestone gate") was
undecidable for the intermediates — the summary gate cannot see them — and is
replaced by the decidable per-commit characterization in the Verification plan. (See
also the CSV-serialization fragility question, now resolved decidably by the
fail-closed control-rerun rule (ext2-1): either the seeded controls reproduce
bit-for-bit or the non-determinism is investigated before R5 proceeds — no fuzzy
tolerance.)

## Verification plan

Every claimed invariant has a named gate:

- **Per-commit dry-run:** after every Snakefile-touching commit, a workflow-3 dry-run
  builds a clean DAG. A **bare** dry-run against an empty project dir would
  `MissingInputException` on the `ancient(region.geojson)` input once the cycle is
  gone (§ "What changes" 4, sole-leaf enumeration) — so the post-cycle-fix dry-run
  gate uses a **staged region** (the default from commit 5a): stage a minimal
  `region.geojson` in a `tmp_path` project_dir and run `snakemake all -c 1 -s
  Snakefile_climate_experiment --configfile <staged-config> --dry-run`, mirroring
  wf2's `config_with_staged_region` fixture. The bare-dry-run-passes outcome is the
  contingency.
- **`run_historical: true` dry-run added to commit 5a and 5b (risk-2).** Because the
  seed is `run_historical=false`, the default dry-run never exercises the `cst_0`
  downstream path (`st_num2=0`/folded `st_num=0`). A dedicated dry-run on a
  `run_historical: true` variant of the staged config confirms that after the
  constraint (5a) and after the fold (5b) **exactly one rule produces `cst_0` for
  every requester** and the DAG builds clean. This is the gate that verifies the
  fold's global-resolution safety before the seal.
- **`pytest tests/test_cli.py`** after every commit touching a Snakefile or a script
  signature. The wf1/wf2 tests stay green; the **rewritten wf3 test**
  (`test_snakefile_cli_climate_experiment`, plain `returncode == 0` on the staged
  config) must pass — this is the ratchet flip (lands in 5a).
- **New unit tests** (§3, §8) under `pixi run pytest tests/`: `stress_test_grid`,
  `prep_cst_parameters` (+ the `precip_variance` **xfail(strict=True)** characterization
  test flagging `t260720a`), `prepare_weagen_config` `generate`-branch year math. All
  non-xfail tests expected PASS; the characterization test is expected **XFAIL** today
  and will XPASS (tripping strict) when `t260720a` lands.
- **Shell-rule failure-detection gate (commit 8; ext1-1).** A deliberately failing
  Rscript (`Rscript --vanilla -e "quit(status=1)" > {log} 2>&1`) and a deliberately
  failing Julia command (`julia -e "exit(1)" > {log} 2>&1`) each run under a scratch
  Snakemake rule on Windows, using the exact redirect form commit 8 lands, and must
  each yield a **non-zero Snakemake result** with the log file written. The Rscript
  half is already verified on this machine (2026-07-20 scratch run, §2); the
  commit-8 gate repeats it with the Julia invocation included.
- **Targeted intermediate-equivalence characterization for the two
  computational-path commits (ext1-3).** The milestone gate's 3 wf3 targets cannot
  see the intermediates commits 6 and 9 edit; each commit therefore carries its own
  before/after gate on the exact artifacts it touches:
  1. **Commit 6 (`historical_window` wiring) — `extract_historical.nc`.** Before
     the commit: run wf3 through `extract_climate_grid` on the seed config and copy
     `extract_historical.nc` to a scratch reference. After the commit: regenerate
     and compare with a **normalized xarray comparison** — dims, coords, data
     variables, attrs, and values (`xr.open_dataset` both;
     `ds_before.identical(ds_after)`), with any known-volatile provenance attrs
     (e.g. a history/date stamp, if present) normalized out and the normalization
     recorded. The file is **not** `temp()` (it is consumed as `ancient()`), so no
     `--notemp` is needed for this half.
  2. **Commit 9 (R arg-binding + `message()` lines) — realization + perturbed
     netCDFs.** Before the commit: a scratch wf3 run on the seed config **with
     `--notemp`** (preserving the `temp()` netCDFs) into a scratch `project_dir`;
     keep at least `rlz_1_cst_0.nc` (one realization) and one perturbed
     `rlz_1_cst_<k>.nc`. After the commit: repeat and compare both files with the
     same normalized xarray comparison. Expected: `identical()`-level equality —
     the commit adds argument names and progress messages, not computation.
  3. **Determinism / attribution protocol — a pre-committed, executable,
     fail-closed comparison rule (governs both halves AND the milestone gate;
     ext2-1, user-arbitrated 2026-07-20).** The realizations are seeded
     (`config/weathergen_config.yml` `generateWeatherSeries.seed: 123`, read by
     `generate_weather.R` line 40 and propagated via `prepare_weagen_config.py`'s
     default-config copy), so **for a fixed seed, bit-identical reruns are the
     EXPECTED state** — determinism, not tolerance, is the design assumption.
     `impose_climate_change.R`'s `apply_climate_perturbations` (qm `mme` fit)
     exposes no explicit seed argument, so its rerun-stability is *asserted*
     deterministic and must be *demonstrated*, not assumed. The comparison rule is
     fixed in advance and applied unchanged to control-vs-control and to
     before-vs-after:

     - **The exact comparison.** For the netCDF intermediates: a **normalized
       xarray equality** — `xr.open_dataset` both files and require
       `ds_a.identical(ds_b)` (dims, coords, data variables, attrs, **and**
       values all equal), after normalizing out any known-volatile provenance
       attrs (e.g. a history/date stamp, if present) with the normalization
       recorded. For the two summary CSVs at the milestone gate: the existing
       **normalized-content CSV hash** (`check_baseline.py`'s serialization), i.e.
       the same content-hash the manifest uses.
     - **Step 1 — control-vs-control (fail closed).** Before any before/after
       comparison, run the *before* state **twice** (a pristine same-commit
       control-rerun pair) and compare the pair under the rule above. **If the
       control pair is NOT identical, the gate FAILS — it does NOT fall back to a
       fuzzy tolerance.** A seeded control that does not reproduce bit-for-bit is
       treated as a **determinism defect to INVESTIGATE** (which seed/step is not
       fixed; which library introduces run-to-run variance) **before** any R5
       commit touching that path is accepted. There is no per-variable tolerance,
       no noise distribution, no "within observed rerun noise" acceptance — one
       control pair cannot define a defensible noise band, and for a fixed seed a
       mismatch is a signal, not noise.
     - **Step 2 — before-vs-after (only if Step 1 passed; also fail closed).**
       *Only if* the control pair is identical does the before-vs-after comparison
       run, and it must **ALSO be identical** under the same rule to pass. A
       before-vs-after non-identity, given an identical control pair, is a genuine
       R5 regression and **blocks the commit** — there is no fuzzy accept path.
     - **Milestone-gate application (same fail-closed rule).** The identical
       protocol governs the milestone `Qstats.csv`/`basin.csv` gate: first
       establish a control-vs-control hash match on the *before* state; if the
       control hashes differ, the gate FAILS into a determinism investigation (it
       is **not** attributed as tolerable rerun noise). Only against an identical
       control does a before/after (R5) hash mismatch then count as an R5
       regression. This also settles the R4-inherited CSV-serialization
       determinism open question **decidably**: either the seeded controls
       reproduce (and the gate is meaningful) or they do not (and the
       non-determinism is fixed before R5 proceeds), rather than being absorbed
       into an undefined tolerance.

     All scratch artifacts live outside the repo and outside the seed
     `project_dir`; nothing is committed.
- **End-to-end milestone baseline gate** — the decidable procedure (inherited from
  R4, using the `--workflow` filter R4 landed):
  1. Seed `examples/test_local` clean (the manifest's recorded `project_dir`; config
     line 12 sets `project_dir: examples/test_local`). Run **workflow 1 then workflow
     3** into the clean dir (wf3 needs wf1's model). (For the **wf3-scoped** gate,
     wf1 + wf3 suffice; if the gate also checks wf2, run all three.)
  2. `python dev/scripts/check_baseline.py check --workflow model_creation
     --workflow climate_experiment` — checks the **4 wf1 + 3 wf3 targets** (7 total).
     **Expected: `OK - 7 target(s) match manifest.`**
  - **Gate-invalidation note (repo-4):** the wf3 targets resolve via
    `check_baseline.py`'s module-level `EXPERIMENT_NAME` (`exp_dir =
    f"{project_dir}/climate_{EXPERIMENT_NAME}"`, line 82), **not** from the seed
    config's `workflows.climate_experiment.experiment_name` (the Snakefile derives
    `exp_dir` from the config value, line 44). The gate silently assumes
    `EXPERIMENT_NAME == seed config experiment_name` (**verified equal on the current
    seed**). If a future config edit diverges the two, the gate resolves a
    non-existent `exp_dir` and reports a spurious mismatch masquerading as an R5
    regression — flag divergence as a **gate-invalidation condition**. (Pre-existing
    check_baseline structure, not R5's to fix; recorded so a future edit does not
    silently break the gate.)
- **Shared-helper regression coverage:** two independent gates — (1) per-commit
  `pytest tests/test_cli.py` (the wf1 + rewritten wf3 tests fail loudly on any
  `get_config`/`tee_to_log`/`stress_test_grid` regression), and (2) the milestone
  gate's `--workflow model_creation` targets.

**No hygiene commit runs `record`.** R5 plans no `record` at all under the default
plan (no intentional-change commit).

## Commit plan (roadmap `r05:` prefix; one logical change per commit; tag `r05-experiment`)

1. `r05: add dev/workflows/climate_experiment.md contract doc`
   (current behavior; owned keys, the `historical_window` gap flag, plausibility-
   overlay/method-invariant note, `temp()`/manifest-slice honesty, the `st_num2`
   inconsistency note).
2. `r05: extract stress_test_grid helper to snake_utils + unit test`
   (§3; add `from collections.abc import Mapping` (repo-1); **strict required
   `step_num`** (ext1-2) — `KeyError` on missing, `ValueError` on non-integer or
   negative, no silent default; the Snakefile call site's silent default-1 is
   removed, classified output-neutral hardening in the commit message; Snakefile
   lines 30–34 and `prepare_cst_parameters.py` lines 36–46 **plus the two
   `range(...)` sites lines 60/62 and the linspace sizing** (arch-5) all call the
   one helper; new `tests/test_stress_test_grid.py` incl. the strict-contract
   raise tests; output-neutral on the seed; dry-run + test_cli gate).
3. `r05: extract prepare_weagen_config.py config assembly into functions + guard`
   (§2/repo-2; **extract `build_weagen_config` / `compute_nr_years`** — not merely a
   `__name__` wrapper — above a nested `__main__`/`globals()` guard; enables the §8
   year-math test; output-neutral; dry-run + test_cli gate).
   *(no commit 4 — `downscale_climate_forcing.py` needs no import guard; its
   `tee_to_log` wrap lands in commit 7. Numbers retained to avoid renumbering churn.)*
5. **`r05: resolve CyclicGraphException + flip test_cli ratchet` (5a — guaranteed).**
   Rule-local `wildcard_constraints st_num=[1-9][0-9]*` on `generate_climate_stress_test`
   (removes the second `cst_0` producer); **keep `st_num2` distinct**; rewrite the
   `test_cli` wf3 test to `returncode == 0` **on a staged-region config** (mirror
   wf2's `config_with_staged_region`) and delete the ratchet comment block — **all in
   one commit** so the DAG and the test flip together. Gate: default dry-run **+ a
   `run_historical: true` dry-run** + full `pytest tests/test_cli.py`. *Riskiest
   commit — DAG semantics; but the ratchet now depends only on the guaranteed fix.*
   **5b. `r05: fold st_num2 into st_num in downstream wf3 rules` (contingent).**
   Fold `st_num2 → st_num` in `downscale_climate_realization`, `run_wflow`,
   `export_wflow_results` (naming §4), relying on the rule-local `[1-9][0-9]*` from 5a
   to keep `generate_climate_stress_test` off `cst_0` while downstream rules keep
   admitting `0`. **Contingent on its own dry-run** (default **+ `run_historical:
   true`**) confirming no re-introduced ambiguity. **Fallback:** if the fold
   re-introduces ambiguity, **drop 5b** — keep `st_num2` distinct and record it as an
   accepted exception in naming §4 with a note. 5a already landed, so dropping 5b
   disturbs nothing.
6. `r05: wire shared.historical_window into extract_climate_grid`
   (§ candidate #2; parse `historical_window` in the Snakefile, pass as rule params,
   replace hardcoded strings lines 156–157, drop the misleading signature defaults
   lines 20–21; output-neutral on the seed — window == current hardcoded strings, a
   computational-path edit asserted equivalent (risk-5); gates: dry-run + test_cli
   **+ the before/after `extract_historical.nc` characterization** (ext1-3,
   Verification plan)).
7. `r05: add log + benchmark + tee_to_log to workflow-3 python script rules`
   (§2; the 7 `script:` rules; wildcard-keyed log paths via string concatenation;
   `prepare_weagen_config` already guarded by commit 3; `downscale_climate_forcing.py`
   `tee_to_log` wrap enclosing its top-level reads (risk-4) lands here).
8. `r05: add log + benchmark + exit-preserving redirect to the R and Julia shell rules`
   (§2/§5b; `generate_weather_realization`, `generate_climate_stress_test` (Rscript),
   `run_wflow` (Julia) gain `log:`/`benchmark:` + **`> {log} 2>&1`** — NOT `| tee`
   (ext1-1: exit-preserving on cmd.exe, empirically verified; accepted trade-off:
   log-file-only, no live console); gate: the deliberately-failing Rscript + Julia
   mechanism check (Verification plan); the reframed wf1 `| tee` followup lands in
   `dev/followups.md` in this commit).
9. `r05: clean R-layer argument parsing + progress logging`
   (§5a/§5b; bind positionals to named locals with arity checks **after
   `source("./src/weathergen/global.R")`** (repo-5) in `generate_weather.R` +
   `impose_climate_change.R`; add `message()`/`cat()` progress lines; tighten the
   `spatial_ref` workaround comment with its removal condition; output-neutral on valid
   input, loud on wrong arity; a computational-path edit asserted equivalent (risk-5);
   gates: dry-run + test_cli **+ the before/after realization + perturbed-netCDF
   characterization via `--notemp`, under the fail-closed control-rerun rule**
   (ext1-3 + ext2-1, Verification plan)).
10. `r05: rename deprecated _fid labels in workflow-3 rules`
    (§7; `forcing_fid`/`toml_fid` → `_path`, same-rule shell refs updated; §7 contract
    surfaces untouched; no migration note).
11. `r05: add unit tests for workflow-3 python helpers`
    (§8; `prep_cst_parameters` + `precip_variance` **xfail(strict=True)** characterization
    test referencing `t260720a`, `prepare_weagen_config` `generate`-branch year math;
    M02c monkeypatch discipline).
12. `r05: seal milestone — dev/roadmap.md R5 section + durable refs`
    (+ tag `r05-experiment`; the roadmap R5 section **enumerates the split defects** —
    `t260720a` (precip_variance), the upstream weathergenr `spatial_ref`/wavelet items,
    and the reframed **wf1 `| tee` followup** (wf1's three shell rules run correctly
    on success but mask the exit code on failure — no `set -euo pipefail` prefix on
    cmd.exe; migrate wf1 to the `> {log} 2>&1` form or a portable tee wrapper) — each
    with owner + activation condition; records the R-testthat NO decision; records whether the fold (5b)
    landed or was dropped; records the ext1-3 control-rerun/determinism result;
    records final test counts and the milestone-gate result).

**Commit ordering rationale.** The grid helper (2) and the `prepare_weagen_config`
extraction (3) land before the log/`tee_to_log` wraps (7) so that script's body is
restructured once and then wrapped. The cycle fix + ratchet flip (5a) lands before the
log sweep so the wf3 dry-run is clean for the log-commit gates; the fold (5b) follows
5a and is droppable. The `historical_window` wiring (6) is independent and lands
before the log sweep so `extract_climate_grid`'s params are settled when its log is
added.

**Scope statement (corrected — ext1-1 / ext1-3 / risk-5).** R5 touches workflow-3 files
(`Snakefile_climate_experiment`, its `src/` scripts,
`src/weathergen/generate_weather.R` + `impose_climate_change.R`), `src/snake_utils.py`
(the `stress_test_grid` **addition** + the `Mapping` import — purely additive; does
not change `get_config`/`tee_to_log`, so wf1/wf2 inheritance is untouched), and
`tests/`. **No repo-root instruction file changes. No `check_baseline.py` change**
(R4 already landed the `--workflow` filter). Two commits (6, 9) edit computational
paths, asserted output-equivalent, confirmed by the targeted intermediate
characterization (ext1-3) plus the milestone gate — **not** "untouched paths."
Riskiest commit: **5a** (DAG-semantics cycle fix); the fold (**5b**) is contingent
and droppable. Commit **8**'s redirect form (`> {log} 2>&1`) is empirically verified
exit-preserving on the Windows platform (§2); its residual risk is only the
Julia-side half of the commit-8 mechanism gate. One **latent** item sits **outside**
R5's own commits: the milestone gate's **wf1 leg** runs `Snakefile_model_creation`,
whose existing `| tee` shell rules mask the exit code *on failure* (they run
correctly on success — the wf1 leg passes on clean commands; corrected per driver
evidence, §2) — routed to `dev/followups.md` as a cross-cutting robustness followup,
not an R5 blocker and needing no precursor task (see Risks).

## Alternatives considered

- **Bundle the cycle fix + fold + ratchet flip in one atomic commit (design-v1's
  commit 5).** *Rejected (Group B / arch-1):* the fold's global-resolution safety is
  admittedly unverified until an implementation dry-run; bundling it with the
  irreversible ratchet flip means a failed fold dry-run leaves the commit half-done
  (ratchet test rewritten to `returncode == 0`, DAG not building). *Chosen:* split
  into 5a (guaranteed cycle fix + ratchet flip) and 5b (contingent fold), so the
  ratchet depends only on the guaranteed fix and the fold is droppable.
- **Cycle fix via `ruleorder` instead of `wildcard_constraints`.** *Rejected:*
  `ruleorder` merely **prioritizes** one of two eligible producers of `cst_0.nc` —
  masking the wildcard over-match — whereas `wildcard_constraints st_num=[1-9][0-9]*`
  removes the second producer's eligibility at the source and self-documents naming §4.
- **Shell-rule logging: inherit wf1's `2>&1 | tee {log}`; or make it exit-safe via
  `shell.prefix("set -euo pipefail; ")` / inline `set -o pipefail;`; or a portable
  Python tee wrapper.** *All rejected for R5 (ext1-1, supersedes v2's Group A
  ruling which kept tee):* on the Windows exit-criteria platform `| tee` **masks
  the subprocess exit code on failure** — `tee` resolves and runs, but cmd.exe
  gets no `set -euo pipefail` prefix (bash-only), so the pipeline takes
  `tee`'s exit 0 (verified 2026-07-20: `| tee` → Snakemake reports success on a
  failing command; `> {log} 2>&1` → Snakemake fails); pipefail in either form is
  bash-only (`shell.prefix` additionally mutates all shell rules repo-wide); a
  portable tee wrapper works (exit-preserving + live streaming) but is
  cross-cutting infrastructure belonging to the reframed wf1 followup, not a
  wf3-local add. *Chosen:* **`> {log} 2>&1`** — portable,
  exit-preserving (empirically verified, §2), log-file-only. The live-console loss
  (which wf1's commit `4a67d79` tee restored) is accepted for wf3's three shell
  rules and recoverable repo-wide later via the followup's wrapper candidate.
- **`stress_test_grid` with the Snakefile's default-1 leniency (v2's form), or a
  dual strict/default policy flag.** *Rejected (ext1-2):* default-1 on a missing
  `step_num` silently changes `ST_NUM`, the parameter-CSV count, and the entire
  `RLZ_NUM × ST_NUM` workload — leniency mislabeled as hardening; a dual-policy
  flag would preserve two contracts where one is wanted and require testing both.
  *Chosen:* strict required `step_num` on both axes (`KeyError` parity with the
  script consumer + non-negative-int validation), an error-path-only change at the
  Snakefile call site, classified output-neutral hardening (§3).
- **R argument parsing: introduce an `optparse` flag parser.** *Rejected:* would
  change the `shell:` call contract (Snakefile lines 122, 133) and adds a dependency +
  ceremony for 2–3 internal args. Named-binding-with-arity-check (§5a) delivers the
  goal without touching call sites. Parsimony.
- **R logging: an in-R logging framework (`logger`/`futile.logger`).** *Rejected:* the
  `shell:` `> {log} 2>&1` redirect captures everything with zero in-R code and
  gives all three engines a uniform per-rule log. An in-R framework would be a second,
  inconsistent mechanism — the opposite of "consistent logging."
- **R testthat coverage: YES.** *Rejected (locked NO, G1 ratified):* the two R scripts
  are thin weathergenr adapters — their non-trivial logic is upstream in the
  `weathergenr` package. The repo has **no R test harness** today; standing one up is
  infra R5 is not chartered to build. The R layer is gated end-to-end by the milestone
  baseline run and the `test_cli` dry-run, and §5a arity hardening is the cheap
  robustness net. Activation to reconsider: a future milestone extracting non-trivial
  R logic *into this repo*.
- **Fix the `precip_variance` max-reads-min bug in R5 (as an intentional change with a
  no-seed-delta baseline note).** *Rejected (G1 ratified split):* the user ratified
  **split to `t260720a`, not fix-in-R5**, at G1. R5 stays behavior-preserving; the
  characterization test flags the defect via `xfail(strict=True)` (not defends it —
  risk-3), and the fix's baseline consequence belongs to `t260720a`.
- **Rename `RLZ_NUM`/`ST_NUM` → `rlz_count`/`stress_test_count` now** (naming §9). 
  *Rejected (deferred):* grandfathered constants used across the Snakefile and multiple
  scripts; renaming ripples widely for a cosmetic win. Left grandfathered.
- **Run all three workflows for the milestone gate (unscoped 14-target check).**
  *Rejected as primary:* drags wf2 (a plausibility overlay wf3 does not consume) into
  the gate. The `--workflow model_creation --workflow climate_experiment` scoped check
  (7 targets) matches R5's territory and is faster.

## Risks and open questions

- **[Risk] Commit 5a/5b (cycle fix + `st_num2` fold) is DAG-semantics.** The rule-local
  `[1-9][0-9]*` cycle fix (5a) is verified complete by the sole-producer enumeration
  above and is guaranteed. The **fold (5b)** must be **dry-run-verified** on the test
  config — **including a `run_historical: true` dry-run** — to confirm no re-introduced
  ambiguity and that the `cst_0` downstream path still resolves. Fallback: drop 5b,
  keep `st_num2`. This is the one area whose full correctness cannot be asserted from
  reading alone.
- **[Open question] Does the post-fix wf3 dry-run need the staged `region.geojson`
  fixture, or does a bare dry-run pass?** The sole-leaf enumeration says
  `region.geojson` is the only unbuilt external leaf, so a bare dry-run against an
  empty `project_dir` should `MissingInputException` on it once the cycle is gone →
  the staged fixture is the default. Confirmed at implementation; bare-pass is the
  contingency.
- **[Open question — now decidable, fail closed] CSV-serialization determinism of
  `Qstats.csv`/`basin.csv`.** R4 found CSV-serialization non-determinism in its
  summary CSVs. wf3's summary CSVs carry per-column rounding and `float32` casts
  (`export_wflow_results.py`); whether their normalized-content hash is
  deterministic across runs is **decided by the ext2-1 fail-closed control-rerun
  rule before the milestone gate** (a pristine same-commit rerun pair). If the
  seeded control hashes are **not** identical, the gate FAILS into a determinism
  investigation rather than accepting a tolerance; only against an identical
  control does a before/after (R5) hash mismatch count as an R5 regression. No
  gate mismatch is silently absorbed as tolerable rerun noise.
- **[Open question] `historical:` vs `shared.historical_window` naming.** The followup
  calls the key `historical:`; the seed config nests it under
  `shared.historical_window`. Commit 6 wires `shared.historical_window`. Confirm no
  other config variant uses a flat `historical:` key that the wiring should also honor.
- **[Resolved (ext1-2, v3)] `stress_test_grid` strict vs lenient parity.** Resolved
  to **strict**: `step_num` required on both axes, validated as a non-negative
  integer; the Snakefile call site's silent default-1 is removed (error-path-only
  change on invalid configs, classified output-neutral hardening). See §3.
- **[Open question] `export_wflow_results` unit-testability.** Heavy (imports
  `metrics_definition`, xarray). Default: no extract-and-unit-test (parsimony); covered
  only by the end-to-end gate. Confirm the reviewer accepts, or charter a small pure-
  sub-function extract.
- **[Risk] The `spatial_ref` workaround removal is gated on an upstream fix that may
  never land.** R5 keeps the block and documents the removal condition; the pipeline
  depends on it. No R5 action beyond the comment.
- **[Open question] Import-guard / function-extraction output-neutrality unverified
  until run.** The `prepare_weagen_config` extraction (commit 3) and the
  `downscale_climate_forcing` `tee_to_log` wrap (commit 7) are asserted output-neutral
  but confirmed only by the dry-run + test_cli gate at implementation.
- **[Latent robustness followup outside wf3 — NOT an R5 blocker] wf1's `| tee {log}`
  shell rules mask the exit code on failure.** Corrected empirical finding
  (2026-07-20 driver verification, superseding the v3 "tee unresolvable → wf1 gate
  fails" claim): on this machine `tee` **resolves and runs fine** — a minimal
  Snakemake test executed the tee rule and created its output, and R3 **sealed via
  a full `--forceall` wf1 rebuild that wrote all three tee logs and passed 14/14 on
  this same machine**. So wf1's three `| tee` shell rules
  (`Snakefile_model_creation` lines 89, 167, 182) **run correctly on success**, and
  **the R5 milestone gate's wf1 leg PASSES on clean commands** — there is **no**
  precursor task and **no** predicted gate failure. The only defect is **latent**:
  because cmd.exe gets **no** `set -euo pipefail` prefix (bash-only — §2 source
  analysis), a `cmd 2>&1 | tee {log}` pipeline takes `tee`'s exit 0 and **masks a
  genuine command failure** (verified: a deliberately-failing command under
  `| tee` → Snakemake reports success; under `> {log} 2>&1` → Snakemake fails).
  That masking bites only if a wf1 rule actually fails mid-run — a robustness gap,
  not a
  blocker. **Disposition:** a cross-cutting followup (migrate wf1's — and, when a
  portable tee wrapper lands, all — `| tee` shell rules to `> {log} 2>&1` or a
  streaming wrapper) routed to `dev/followups.md`; R5 does not edit wf1 (milestone
  boundary) and **charters no precursor task**. wf3's own new shell rules already
  use the exit-preserving `> {log} 2>&1` (commit 8), so R5 introduces no new
  instance of the masking.
- **[Accepted trade-off (ext1-1)] wf3 shell rules log to file only — no live
  console streaming.** `> {log} 2>&1` captures stdout+stderr but does not stream;
  long Wflow runs are observed by tailing the per-rule log
  (`Get-Content -Wait <log>` / `tail -f`). Exit-code correctness was judged worth
  this cost (§2); the followup's portable tee wrapper can restore streaming
  repo-wide later.
- **[Open question — decided fail closed] Rerun-stability of the R-layer netCDFs.**
  The generator is seeded (`generateWeatherSeries.seed: 123`) but
  `apply_climate_perturbations`'s rerun determinism is unproven. The ext2-1
  fail-closed control-rerun rule measures it before any before/after comparison:
  if the seeded control pair is **not** bit-identical, the commit-9 gate FAILS and
  the non-determinism is investigated (which seed/step is unfixed) before commit 9
  is accepted — there is **no** fall-back to structural-equality-plus-tolerance.
  For a fixed seed, bit-identity is the expected state, so a non-identical control
  is a defect to fix, not noise to bound. The result (controls reproduce, or the
  determinism defect and its resolution) is recorded at the seal.
