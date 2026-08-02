# R06 structural-refactor — consolidated design-review record

Audit trail for the `r06-structural-refactor` design-review-loop that produced
`structural-refactor-design.md` (accepted 2026-07-22). The per-round scratch run
dir (`dev/working/design-runs/r06-structural-refactor/`) is prunable; this record
is the durable trail.

## Run summary

- **Versions:** design-v1 (draft) → v2 (internal panel) → v3 (external r1) → v4 (arbitration). v4 is the accepted content.
- **Author:** `cst-architect` (Opus; v4 arbitration revision on Fable — ext2-02 re-raised ext1-04). Driver: interactive Opus session.
- **Gates:** G1 framing **approved** 2026-07-22 (defer functional decomposition; wrapper for `enabled:`; atomic-commit + run-relative baseline left provisional for review). G2 approval **granted** 2026-07-22.
- **Findings:** 23 total — 3 blocking, 7 major, 13 minor. Disposition: **23 accepted / 0 rejected / 0 deferred.**
- **Dispatches:** 7 Opus + 1 Fable. Recoveries: architecture-lens stream-timeout (rung-1 resume); external-r2 dispatch path typo (killed + re-dispatched). No round wasted; external cap = 2 reached → user arbitration.

## Verdict table

| Review | Reviewer | doc_version | Verdict | Findings |
| --- | --- | --- | --- | --- |
| Internal — risk & assumptions | critical-thinker | design-v1 | revise | risk-01 (B), risk-02/03/04 (M), risk-05/06 (m) |
| Internal — repo fit | python-engineer | design-v1 | revise | repo-01 (B), repo-02 (m) |
| Internal — architecture | cst-architect | design-v1 | revise | arch-1 (B), arch-2/3 (M), arch-4..8 (m) |
| External round 1 (clean-room) | codex/GPT | design-v2 | revise | ext1-01/02/03 (M), ext1-04 (m) |
| External round 2 (regression, cap) | codex/GPT | design-v3 | revise | ext2-01/02 (M), ext2-03 (m) |
| Arbitration (round-cap) | user | design-v3→v4 | accepted, fix required | ext2-01/02/03 |

---

## Internal panel — aggregation index

# Internal panel — aggregation index (design-v1.md)

Driver aggregation of the three lens reviews. Preserves every original finding
ID, severity, and source; groups duplicates by theme; **does not delete or
re-grade** any finding. Full text in the per-lens files:
`internal-review-risk.md`, `internal-review-repo-fit.md`,
`internal-review-architecture.md`.

## Verdicts

| Lens | Reviewer | Verdict | Findings |
| --- | --- | --- | --- |
| Risk & assumptions | critical-thinker | revise | risk-01 (B), risk-02/03/04 (M), risk-05/06 (m) |
| Repo fit & conventions | python-engineer | revise | repo-01 (B), repo-02 (m) |
| Architecture & consistency | cst-architect | revise | arch-1 (B), arch-2/03 (M), arch-4/5/6/7/8 (m) |

Panel verdict: **revise**. 3 blocking, 5 major, 8 minor. No finding challenges
the G1-approved alternative (defer decomposition; wrapper `enabled:`) — all are
bounded corrections to the mechanical-rewrite and verification sections. No
return to G1.

## Blocking (must be resolved in design-v2)

- **B1 — hardcoded `weathergen_config.yml` param, `--dry-run`-blind** —
  `risk-01` (blocking) **+ `arch-2`** (major, same issue). `Snakefile_climate_experiment:131`
  passes `default_config = "config/weathergen_config.yml"` as a `params:` literal
  (consumed by `prepare_weagen_config.py:57` via `read_yml`). The design calls
  this file a "no Snakefile-default consumer." The config split moves the file →
  rule 3.04 `read_yml` fails at **runtime**; `--dry-run` cannot see a `params:`
  string, so the design's own config-split gate passes green while WF3 is broken.
  (Severity split preserved: risk=blocking, arch=major — treat as blocking.)
- **B2 — import-rewrite rule inconsistent + misses `from src import <module>`** —
  `repo-01` (blocking). §9 says `from src.`→`from blueearth_cst.`; §10 says
  `from src.`→`from blueearth_cst.<stage>.`; neither handles `from src import
  <module>` (4 test files: test_extract_historical_climate, test_metrics_definition,
  test_prepare_climate_data_catalog, test_setup_time_horizon), and the `<stage>`
  segment differs per module. Either literal rewrite reds `pytest tests/` wholesale.
  The counting grep (`from src\.`) has the same blind spot → "~40 files" undercounts.
- **B3 — verification plan self-contradicts the test fixture** — `arch-1`
  (blocking). Verification "Config-split" row says the reorg "must **not** touch"
  `tests/snake_config_model_test.yml`, but that fixture line 23 sets
  `model_build_config: config/wflow_build_model.yml` (shared config tree, not the
  unused `tests/wflow_build_model.yml`). The template move **must** rewrite it, or
  WF1 dry-run reds with `MissingInputException` — the design instruction and its
  own gate contradict.

## Major

- **M1 — wrapper must preserve per-workflow flags** — `risk-02`.
  `run_snake_test.cmd:13` runs projections with `--keep-going`; the others without.
  A flag-uniform wrapper silently drops `--keep-going` → aborts on first partial
  ensemble failure = a behavioral change (the exact category §7 uses to reject
  Option A). Specify flag preservation + assert in the wrapper test.
- **M2 — run-relative baseline gate: coverage/tooling/determinism** — `risk-03`.
  The run-relative gate is the *right* verification (do not force a re-record), but
  (i) it checks only the thin, mixed-provenance manifest slice → an R6 change to an
  unmanifested wf2/wf3 output passes silently; (ii) `check_baseline` has no full-slice
  two-run compare (manual record/stash/check); (iii) "byte-for-byte" presumes
  weathergenr seed-pinning. State coverage boundary; add a full-`project_dir` pre/post
  diff at least once; specify the command sequence + seed precondition.
- **M3 — green pytest + clean --dry-run exercises neither R nor Julia** —
  `risk-04`. `run_logged.py` depth fix and R `source()` rewrites live entirely in
  the execution path dry-run can't reach; a miss fails only at first real run, four
  commits later. Add an execution-level smoke (invoke `run_logged.py`; one `Rscript`
  `source()`-only run) to commit 1's gate.
- **M4 — `static_dir` correctness hinges on a complete explicit-key inventory** —
  `arch-3`. Every checked-in config sets `model_build_config`/`waterbodies_config`
  **explicitly** (template:41, model_test:23, `*_linux`/projections variants), so the
  changed default expression is dead for them — one missed explicit key silently
  keeps the old path and breaks at build, invisible to the single-fixture gate.
  Reframe as a complete inventory across ALL `config/**/*.yml` + `tests/*.yml`.

## Minor

- `risk-05` — forward-compat: helpers placed in `experiment/` while claiming "no
  second move" is self-undermining; move cited model-independent helpers to
  `shared/` now, or soften to "mechanical later move."
- `risk-06` — enumerate ALL `config/`/`src/` literals across `*.py/*.R/*.cmd/*.sh`
  (e.g. `copy_config_files.py:99` `__main__` fallback); the 4-pattern list understates.
- `repo-02` **+ `arch-7`** (converge) — `dev/scripts/stage_data.py` over-included:
  its `src` tokens are a local var, not a `from src.` import. Drop it; list only
  `probe_attrs_chain.py`.
- `arch-4` — broaden the R-layer grep to ALL `*.R` for `src/weathergen`/`./src`
  literals, not just `source(` in the two entrypoints (transitive sources).
- `arch-5` — `run_logged` depth fix is correct, but Risks #2's "--dry-run exercises
  the run_logged path" is false; name `tests/test_run_logged.py` (must be rewritten)
  + e2e run as the guard.
- `arch-6` — `module:` "reopens the resolved CyclicGraphException" is overstated (the
  rule-local `wildcard_constraints` would carry along); lead the WRAPPER rationale
  with the two solid arguments (4th entry point; DAG-merge = the forbidden behavioral
  change), downgrade the R5 point to "enlarges the ambiguity surface generally."
- `arch-8` — note that the atomic commit-1 gate is `--dry-run`-blind to ≥3
  run-time-only reference classes (R `source()`, `run_logged` bootstrap,
  `weathergen_config` param), so the rejected shim's per-commit-green property has
  more value than the "more moving parts" framing credits (open G1 question posture
  is right).

## Cross-cutting theme (for the author)

Five findings (B1, B2, B3, M3, arch-4/5/8) share one root: **the design's
"uniform mechanical transform + green pytest + clean --dry-run = invariance by
construction" method underestimates the reference surface and over-trusts
`--dry-run`.** `--dry-run` is blind to `params:` string paths, `shell:`/`Rscript`
bodies, R `source()`, and `run_logged` bootstrap. design-v2 should (a) drive the
rewrite from a grep-derived, per-module/per-file inventory (not a 4-pattern
find-replace), and (b) add an execution-level smoke to the gates that dry-run
cannot cover. The G1 decisions stand; the machinery underneath needs hardening.

---

## External review — round 1 (clean-room, on design-v2)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-01 [major]
- section: Behavior-preservation stance and exact baseline consequence
- finding: The prescribed run-relative command sequence records and modifies the tracked `dev/baseline/manifest.json`, copies it aside, and then changes revisions without first restoring the tracked file.
- rationale: `git checkout <post-R6-tip>` can be blocked by the modified manifest or can carry that modification into the post-R6 checkout, so the documented baseline gate is not reliably executable and may compare against unintended state.
- suggested_fix: After copying `/tmp/manifest-preR6.json`, immediately restore `dev/baseline/manifest.json` before changing revisions. Prefer extending `check_baseline.py` to accept an explicit reference-manifest path so verification never temporarily overwrites a tracked baseline artifact.

### ext1-02 [major]
- section: The `enabled:` operationalization (required position: WRAPPER)
- finding: The wrapper contract does not define handling for missing or invalid `workflows.<name>.enabled` values, although several configs included in the proposed `config/workflows/` tree lack these sections. It also leaves core invocation behavior—failure propagation, concurrency, and forwarding of additional Snakemake arguments—unspecified.
- rationale: A user invoking the new runner with an existing projections-only config may receive a key error, silently skip workflows, or obtain implementation-dependent defaults. Likewise, a failed workflow could be followed by later workflows unless exit semantics are fixed, producing partial or stale-result runs inconsistent with the promised fixed-order orchestration.
- suggested_fix: Define the wrapper’s CLI and schema contract explicitly: identify supported config classes; specify whether missing `enabled` defaults to true or is an error; require boolean values; stop on the first nonzero subprocess exit and return that code; define cores and extra-argument forwarding; and test these cases.

### ext1-03 [major]
- section: Verification plan
- finding: The `enabled: false` integration test defines absence only as “that workflow’s rule-all outputs are absent” without requiring a fresh `project_dir` or accounting for outputs from earlier runs.
- rationale: Disabling a workflow does not delete its existing outputs. Against a reused project directory, the asserted outputs can remain present even though the wrapper correctly skipped invocation, causing a false failure; worse, downstream enabled workflows can consume stale artifacts and make the run appear successful without clarifying whether that is supported behavior.
- suggested_fix: Run skip tests in a fresh temporary `project_dir`, assert invocation/non-invocation through the subprocess boundary, and explicitly document that disabling a workflow neither deletes prior outputs nor guarantees downstream freshness. Define whether enabled downstream workflows may run when a disabled prerequisite’s artifacts already exist.

### ext1-04 [minor]
- section: Behavior-preservation stance and exact baseline consequence
- finding: The proposed full-tree comparison is described as a raw recursive diff with only logs and benchmark timestamps excluded, but no deterministic comparison policy is given for binary NetCDF files or embedded provenance metadata.
- rationale: Semantically identical outputs may differ bytewise because of serialization metadata, yielding a noisy milestone gate whose failures cannot be consistently classified.
- suggested_fix: Specify an explicit exclusion list and semantic comparators for scientific binary outputs, including variables, coordinates, attributes to ignore, and numerical tolerances.
---

## External review — round 2 (regression + cap, on design-v3)

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings

### ext2-01  [major]
- section: Behavior-preservation stance and exact baseline consequence
- finding: The design incorrectly asserts that no manifested value changes and that the run-relative baseline check will match. Commit 2 rewrites path values inside the orchestration config, while `copy_config_files.py` copies that config into the three manifested `project_dir/config/snake_config_*.yml` snapshots. `check_baseline.py` fingerprints those YAML files by hash, so the post-R6 snapshots necessarily differ from the pre-R6 reference even when computational behavior is preserved.
- rationale: The specified baseline gate and full-tree diff will fail on an intentional structural change, preventing the milestone from satisfying its own verification plan or encouraging an implementer to dismiss genuine failures alongside expected ones.
- suggested_fix: Explicitly classify the copied-config path changes as expected structural differences. Add a narrowly defined YAML comparison policy that parses the snapshots and normalizes only the documented old→new path mappings before comparison, while requiring all other keys and values to remain equal; use the same policy in both the run-relative gate and full-tree semantic diff.

### ext2-02  [major]
- section: Full-`project_dir` pre/post diff — semantic, not raw byte [v3: ext1-04]
- finding: ext1-04 is not genuinely resolved because the proposed NetCDF comparison delegates to `fingerprint_nc`/`diff_nc`, which compares aggregate per-variable statistics rather than aligned array values. Spatial or temporal permutations—and many compensating cell-level changes—can preserve shape, dtype, count, min, max, mean, and standard deviation while materially changing the dataset.
- rationale: The full-tree diff is the design’s only stated coverage for unmanifested change-factor and staticmaps NetCDFs. A broken reference rewrite or altered coordinate/value ordering could therefore produce scientifically wrong outputs while the milestone gate passes.
- suggested_fix: For the full-tree gate, compare dimensions, coordinate labels/order, variable values element-by-element with the stated numerical tolerance and explicit NaN handling, plus normalized nonvolatile attributes. Reuse existing attribute exclusions where appropriate, but do not use summary fingerprints as the equality criterion.

### ext2-03  [minor]
- section: The wrapper contract — pinned [v3: ext1-02]
- finding: The boolean contract contradicts itself: §7(c) lists unquoted YAML `yes` among values that must be rejected, then correctly notes that YAML parses `yes` as boolean `True`. An implementation checking the post-parse value with `isinstance(v, bool)` cannot distinguish `yes` from `true`.
- rationale: The prose and specified test expectation can produce incompatible implementations or a test that cannot be satisfied using the stated parsed-value contract.
- suggested_fix: Either accept every scalar the selected YAML parser resolves to a boolean and remove `yes` from the rejection examples, or require canonical `true`/`false` spelling and specify validation against YAML source tokens before parsing.
---

## Final finding ledger (23 rows, all dispositioned)

# R06 structural-refactor — review ledger

Append-only. One row per original finding ID. Round = the review round that raised
it. Severity = the finding's own lens severity (the internal panel preserved the
per-lens split; B1 = risk-01 blocking + arch-2 major are two rows, etc.).

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
| --- | --- | --- | --- | --- | --- |
| risk-01 | internal-panel | blocking | accepted | Corrected the false "no consumer" claim: `weathergen_config.yml` IS consumed via the hardcoded `params:` literal at `Snakefile_climate_experiment:131` → `prepare_weagen_config.py:57 read_yml`. Added line 131 to commit-2's named-edit set (§2/§5, §10 commit 2); flagged `--dry-run`-blind (it is a `params:` string, not a declared `input:`); commit-2 gate gains a runtime resolution check for rule 3.04. | design-v2.md |
| risk-02 | internal-panel | major | accepted | Runners do NOT invoke uniform flags — `run_snake_test.cmd:13` has `--keep-going` on projections only (verified). §6 + §7 now require the wrapper to preserve per-workflow flags; a wrapper unit test asserts projections carries `--keep-going` and the other two do not; §9 names flag parity as a preservation requirement. Commit 3 preserves the flag verbatim on the runner move. | design-v2.md |
| risk-03 | internal-panel | major | accepted | Kept the run-relative gate (correct) and stated its three boundaries in §9: (i) coverage — thin, mixed-provenance manifest slice, wf2/wf3 staticmaps/TOML/change-factors NOT covered → added a full-`project_dir` pre/post diff at the milestone gate; (ii) tooling — `check_baseline.py` has only `record`/`check`/`compare` (single-CSV), no one-shot two-run compare → spelled out the manual record/stash/check three-step sequence; (iii) determinism — named the `seed: 123` weathergenr precondition (verified in `weathergen_config.yml`). | design-v2.md |
| risk-04 | internal-panel | major | accepted | Added execution-level smokes to commit 1's gate: `run_logged.py` invocation on a trivial command, one `Rscript --vanilla` `source()`-only run, and the rewritten `tests/test_run_logged.py` exercising the bootstrap. §9 Premise 2 states `--dry-run`/pytest exercise neither R, shell bodies, nor the `run_logged` bootstrap. | design-v2.md |
| risk-05 | internal-panel | minor | accepted | Softened the "no second move" over-promise to a *mechanical* later move; kept `extract_historical_climate.py` in `experiment/` (its sole consumer) and recorded (§8 + Alternatives) that moving it to `shared/` now was rejected because G1 defers the decomposition (do not pre-build structure for a not-yet-designed subpackage). Design now states R6 keeps helpers model-independent *in their dependencies*, not zero-move. | design-v2.md |
| risk-06 | internal-panel | minor | accepted | Re-derived the rewrite inventory by grep (§1a) rather than a-priori patterns; enumerated the non-import literal `config/`/`src/` class including the dead `__main__` fallback `copy_config_files.py:99` (flagged harmless) and the `tests/test_workflow_*.py CONFIG=` literals (confirmed no-op, `config/` root not moved). Migration checklist greps `params:`/raw literals, not only the four pattern classes. | design-v2.md |
| repo-01 | internal-panel | blocking | accepted | Replaced the single `from src.` find-replace with a per-module stage table (§1a) that fixes each module's `<stage>` segment and names the `shared` (not-stage) modules. Rewrite rule now covers BOTH `from src.<module>` (form A) and `from src import <module>` (form B — the 4 test files: test_extract_historical_climate, test_metrics_definition, test_prepare_climate_data_catalog, test_setup_time_horizon). §1a, §9, the "Tests move" paragraph, and §10 commit 1 all cite the identical table-driven rule. Count re-derived by `grep -rnE "from src import\|from src\."`. | design-v2.md |
| repo-02 | internal-panel | minor | accepted | Dropped `dev/scripts/stage_data.py` from the rewrite set (its `src` tokens are a local `Path` var; its cross-import is `from console import ...`, not `from src.`). §1a now lists only `dev/scripts/probe_attrs_chain.py:101` as the dev-script site. | design-v2.md |
| arch-1 | internal-panel | blocking | accepted | Deleted the "reorg must not touch `tests/snake_config_model_test.yml`" claim. Verified line 23 `model_build_config: config/wflow_build_model.yml` resolves from the shared `config/` tree (not the co-located `tests/wflow_build_model.yml`, which no config points at), so the template move MUST rewrite lines 23–24 → `config/templates/...` or WF1 dry-run reds with `MissingInputException`. Fixture enumerated in commit-2's edit set (§2/§5 table + Verification plan Config-split row). | design-v2.md |
| arch-2 | internal-panel | major | accepted | Same issue as risk-01 (B1), scored major on this lens. Line 131 `default_config` added to commit-2 edits; noted `--dry-run`-blind like the R `source()` paths; runtime check added to the gate. (See risk-01 row.) | design-v2.md |
| arch-3 | internal-panel | major | accepted | Reframed the `static_dir` fix as a complete explicit-key inventory of `model_build_config:`/`waterbodies_config:` across ALL `config/**/*.yml` + `tests/*.yml` (§2/§5 table: template 41/43, model_test 29/30, gabon 29/30 untracked, tests fixture 23/24). Corrected arch-3's premise with grep evidence: `snake_config_model_test_linux.yml` sets NO explicit key and RELIES on the default, so the Snakefile default-expression edit is load-bearing (not cosmetic) for that variant. Projections configs confirmed no-op. | design-v2.md |
| arch-4 | internal-panel | minor | accepted | Broadened the R-layer migration grep to ALL `*.R` under the package (+ `dev/scripts/*.R`) for any `src/weathergen` / `./src` literal, not just `source(` in the two entrypoints, to catch transitively sourced paths (§1 R-layer paragraph, Risks #3). `install_weathergenr.R` confirmed no-op. | design-v2.md |
| arch-5 | internal-panel | minor | accepted | Corrected Risks #2: the guard for the `run_logged` depth fix is NOT `--dry-run` (it does not execute the `shell:` body) but the rewritten `tests/test_run_logged.py` (imports `main`, exercises the bootstrap) + the commit-1 execution smoke + the e2e run. `tests/test_run_logged.py` named as a required commit-1 rewrite (§1, §9, §10). | design-v2.md |
| arch-6 | internal-panel | minor | accepted | Rewrote §7 point 1 to lead with the two solid arguments (4th entry point violates the three-Snakefile contract; DAG-merge is itself the forbidden behavioral change). Downgraded the CyclicGraphException claim to "a combined resolver enlarges the cross-workflow rule-ambiguity surface generally," noting the R5 `st_num` self-loop's rule-local constraint would carry along into a `module:` include and is not reopened. Alternatives entry updated to match. | design-v2.md |
| arch-7 | internal-panel | minor | accepted | Re-derived the import-rewrite set mechanically (§1a grep inventory); confirmed `stage_data.py` is not a `from src.` site and dropped it (converges with repo-02); listed `probe_attrs_chain.py:101` as the sole dev-script site. `pytest tests/` collecting all test modules remains the safety net. | design-v2.md |
| arch-8 | internal-panel | minor | accepted | Added to commit 1's open-question note that the atomic gate is `--dry-run`-blind to ≥3 runtime-only reference classes (R `source()`, `run_logged` bootstrap, `weathergen_config` param), so the rejected re-export shim's per-commit-green property has more value than v1's "more moving parts" framing credited. v2's execution smokes are the mitigation; the shim stays the open G1 alternative. | design-v2.md |
| ext1-01 | external-r1 | major | accepted | Removed the tracked-file mutation entirely via the EXISTING `--manifest` redirect: `check_baseline.py` keys both the manifest read/write AND the discharge sidecar dir off `ref_dir = args.manifest.parent` (verified, `cmd_record`/`cmd_check`), so pointing `--manifest` at a scratch path (`/tmp/preR6/manifest.json`) never touches `dev/baseline/manifest.json` — no record-to-tracked, no `git checkout -- dev/baseline/manifest.json` restore, no modified-file to block/contaminate a checkout. Corrected the A/B mechanics (§9(iv)): `check_baseline` keys rows by the RESOLVED output path `resolve(template, project_dir)` and matches by exact key, so `record` and `check` must use the IDENTICAL `--project-dir` — it is a temporal A/B (record into `examples/test_local` at the pre-R6 tip, RE-RUN into the same dir at the post-R6 tip, check the same dir), not a cross-dir spatial one (which would key every row under a path the check side never produces → vacuous FAIL). Recording at the pre-R6 tip is safe against comparator skew because R6 does not modify `check_baseline.py`. The stashed pre-R6 tree feeds only the ext1-04 full-tree diff. Overwrite/restore phrasing scrubbed from the §9 code block, Risk #5, the M2/risk-03 narrative, and Alternatives; the existing `--manifest` option is used with a scratch destination, not a new flag. | design-v3.md |
| ext1-02 | external-r1 | major | accepted | Pinned the full `scripts/run_workflows.py` contract (§7 new sub-section + §7(g) tests): (a) accepts full-orchestration configs only (all three `enabled:` keys), projections-only configs (no `workflows:` section, grep-verified) stay direct `snakemake -s` inputs; (b) missing `enabled:` → ERROR not default-true (justified: default-true on a projections-only config would attempt WF1/WF3 whose other keys are absent → cryptic downstream failure; fail-fast at the boundary is clearer); (c) require `isinstance(v, bool)`, reject non-bool; (d) stop on first nonzero Snakemake exit and return that code (preserves WF3-needs-WF1; stated orthogonal to within-run `--keep-going`); (e) `--cores` forwarded to every invocation + `argparse.REMAINDER` after `--` appended to all; (f) per-workflow flag map preserves `--keep-going` on projections only (risk-02/M1). Unit tests (monkeypatch `subprocess.run` argv) cover fixed order, flag parity, missing-key, non-bool, stop-on-nonzero, forwarding. | design-v3.md |
| ext1-03 | external-r1 | major | accepted | Rewrote the `enabled: false` skip test (§9): (1) runs in a FRESH temp `project_dir` (`tmp_path`) — disabling does not delete prior outputs, so a reused dir false-fails an "outputs absent" assertion; (2) asserts invocation/non-invocation AT THE SUBPROCESS BOUNDARY (monkeypatched `subprocess.run` argv capture), not output presence; (3) documented semantics: disabling a workflow neither deletes prior outputs nor guarantees downstream freshness, and — answering the reviewer's explicit question — an enabled downstream workflow MAY run against a disabled prerequisite's pre-existing artifacts (the wrapper invokes each Snakefile independently, enforces no freshness check; WF3 consumes whatever WF1 artifacts exist or fails `MissingInputException`, identical to direct single-Snakefile invocation). No freshness-enforcement machinery added (beyond `enabled:`'s charter). | design-v3.md |
| ext1-04 | external-r1 | minor | accepted | Specified the full-`project_dir` milestone diff as SEMANTIC, not raw byte (§9 new sub-section + Alternatives): recursive directory walk dispatching per extension to `check_baseline.py`'s EXISTING comparators — `.nc` → `fingerprint_nc`/`diff_nc` (per-variable shape/dtype/count/min/max/mean/std + attrs, numerical `--tolerance` via `_within_tol`, `VOLATILE_NC_ATTRS` excluded: history/creation_date/Conventions/software*/*_date); `.csv` → normalized-hash (discharge via tolerance `compare_discharge`); `.png` → size-tolerance `diff_png`; `.toml` → parse-and-normalize compare (the one small new comparator). Exclusion list: `logs/`, `benchmarks/`, `.snakemake/`. Consistent with the manifest CSV-tolerance/semantic-NetCDF approach. Walker driven by the directory walk (not the fixed thin `TARGETS`) so it covers the un-manifested slice (staticmaps, `wflow_sbm.toml`, change-factor NetCDFs). Policy fixed here; walker + `.toml` comparator are a bounded `dev/`-tooling addition for the task-brief (`check_baseline.py` itself unchanged). | design-v3.md |
| ext2-01 | external-r2 | major | accepted | User arbitration 2026-07-22, fix required. Corrected the false "no manifested value changes" claim: `copy_config_files.py` copies the orchestration config into the three manifested `{project_dir}/config/snake_config_*.yml` snapshots, hash-fingerprinted by `fingerprint_yaml` (parse → sorted-JSON → SHA256), and commit 2 rewrites path values inside them (`data_sources`, `model_build_config`, `waterbodies_config`) → post-R6 snapshots legitimately differ. Stated as an EXPECTED structural difference gated by a normalize-then-compare YAML policy (§9 new sub-section): parse both sides, apply ONLY the documented MIGRATION.md old→new path map to those mapped keys on the reference side, then require deep equality of everything else — any residual difference is a real FAIL. Used identically in BOTH gates: the run-relative gate expects a hash FAIL on exactly the three snapshot rows, each adjudicated by the policy against the stashed pre-R6 `config/` copies; the full-tree walker dispatches `project_dir/config/*.yml` to the same comparator. Comparator added to the bounded task-brief tooling; `check_baseline.py` unchanged. | design-v4.md |
| ext2-02 | external-r2 | major | accepted | User arbitration 2026-07-22, fix required (re-raise of ext1-04). Full-tree `.nc` equality no longer delegates to `fingerprint_nc`/`diff_nc` aggregate per-variable stats (permutations and compensating cell-level changes pass min/max/mean/std/count). Specified an ELEMENT-WISE comparator for the full-tree gate (§9): identical dims (names+sizes); coordinate labels AND stored order with no realignment; per-variable shape/dtype then values element-by-element with exact NaN-mask match and the `_within_tol` relative tolerance (default 1e-9) on finite pairs; attrs equal after `VOLATILE_NC_ATTRS` exclusion. Summary fingerprints are not an equality criterion anywhere in the gate; `fingerprint_nc`/`diff_nc` stay as-is for the manifest-slice targets (`check_baseline.py` unchanged) — the full-tree comparator is the stricter belt-and-suspenders check for the un-manifested staticmaps/change-factor NetCDFs, specified for the implementation task-brief alongside the `.toml` comparator. | design-v4.md |
| ext2-03 | external-r2 | minor | accepted | User arbitration 2026-07-22, fix required. Resolved the §7(c) self-contradiction the simple way: the boolean contract is the PARSED value — any scalar `yaml.safe_load` resolves to a Python bool is accepted (`true`/`false`/`yes`/`no`/`on`/`off`); rejected are only scalars parsing to a non-bool (quoted `"true"`/`"false"`, `1`, `0`). Unquoted `yes` removed from the rejection examples; §7(g) test (4) and the verification-table assertion updated to pin the parsed-value contract (quoted `"true"`/`1` → nonzero exit; unquoted `yes` accepted as `True`). The token-level source-validation alternative (canonical `true`/`false` spelling) recorded as rejected in Alternatives. | design-v4.md |
