# R01 plan review — GPT-5.6 (2026-07-17)

## 1. Verdict: not-ready

The revised plan addresses much of the 2026-05-09 feedback, especially the three direct config readers and the expected config-snapshot changes. It is nevertheless not execution-ready. It conflicts with the 2026-07-17 design/roadmap amendment, leaves the `sys.argv` configfile mechanism in place while claiming to replace it, uses a baseline project directory incompatible with the recorded manifest, misses two newer test config consumers, and could overwrite Linux-specific scientific values. These are contract and drift-detection failures, not editorial polish. I treat the amended design and roadmap as the durable contract, so their unresolved contradictions are blockers.

## 2. Prior-review follow-through

| Prior concern | Assessment | Evidence and remaining gap |
|---|---|---|
| Direct config-file readers | **Mostly resolved, but sequenced unsafely.** | Task 3a correctly migrates `src/prepare_cst_parameters.py`, `src/prepare_weagen_config.py`, and `src/get_change_climate_proj.py`; Task 3b adds a legacy-key audit. However, Step 3.10 intentionally commits a runtime-broken schema before Task 3a. That contradicts Task 3’s “atomic migration” claim. Include Task 3a in the atomic commit or move Step 3.10 after it. |
| Baseline snapshot semantics | **Partially resolved.** | Steps 7.4–7.6 document and re-record the three copied YAML snapshots, addressing the original conceptual problem. But Steps 7.3–7.5 compare and record `tests/test_project` against a manifest keyed under `examples/test_local`; consequently, the expected “only three YAML changes” cannot occur. See `dev/scripts/check_baseline.py` and `dev/baseline/manifest.json`. |
| Baseline project-directory mismatch | **Unresolved.** | The plan now passes `--project-dir`, but passes the wrong one. `dev/baseline/manifest.json` records `"project_dir": "examples/test_local"` and target keys under that root. Step 7.3 uses `tests/test_project`. `check_baseline.py` compares resolved path strings, so current targets become “target present but not in manifest.” |
| Clean-tree expectation | **Resolved.** | Step 0.1 now explains that the plan should already be committed and gives an explicit response if it remains untracked. |
| Mechanical old-key audit | **Partially resolved.** | Task 3b adds the requested audit across Snakefiles, `src/`, tests, and baseline scripts. Its patterns do not catch indirect readers such as `_cfg("project_dir")` or `cfg["data_sources"]` in the two integration tests, so it is not yet complete. |
| Contract output accuracy | **Resolved against current code, but superseded by scope.** | Task 2 distinguishes formal targets from intermediates. Its formal target lists match the three current `rule all` blocks; in particular, `gcm_timeseries.nc` is correctly classified as an intermediate because it is absent from `Snakefile_climate_projections:55–63`. However, the 2026-07-17 amendment defers all three contract docs to R3–R5, making Task 2 itself out of scope. |
| R3 configfile-mechanism cleanup | **Not resolved.** | Step 7.8 claims R01 delivers `workflow.configfiles[0]`, but Steps 3.2, 3.4, and 3.5 retain `sys.argv` parsing. The updated roadmap already marks this R3 item “Done by R1” at `dev/roadmap.md:268–269`, so executing the plan would seal a false claim. |

## 3. Correctness pass

### Blocking mismatches

1. **The plan contradicts the amended R01 scope.**

   `dev/milestones/r01/modularity-contracts-design.md`, §4, and `dev/roadmap.md:146–158,179–183` defer the three `dev/reference/workflows/*.md` documents to R3–R5. The plan still:

   - creates them in Task 2;
   - describes them as R01 architecture and deliverables;
   - includes them in Step 7.7’s seal text, Step 7.9’s commit message, Step 7.11’s report, and the final inventory.

   Remove Task 2 and every downstream reference, or reverse the design/roadmap amendment before implementation. The current documents cannot both be authoritative.

2. **The promised configfile mechanism is not implemented.**

   The design specifies `config_path = workflow.configfiles[0]`. The plan’s Steps 3.2, 3.4, and 3.5 instead reproduce the current `sys.argv` lookup found at:

   - `Snakefile_model_creation:3–5`
   - `Snakefile_climate_projections:4–6`
   - `Snakefile_climate_experiment:6–8`

   Step 7.8 then says the replacement is complete. Change all three migration snippets to `workflow.configfiles[0]` and verify every invocation supplies `--configfile`, or leave the roadmap item open.

3. **The baseline commands cannot produce their stated result.**

   `dev/baseline/manifest.json` records targets under `examples/test_local`. `dev/scripts/check_baseline.py:163–170,250–274` keys comparisons by the fully resolved path. Therefore Step 7.3’s check against `tests/test_project` will not report only three YAML mismatches.

   Step 7.5 then records a new manifest rooted at `tests/test_project`, changing `project_dir` and every target key—not only the three YAML entries. The asserted `git diff` condition is structurally impossible.

4. **Final workflow verification is optional where the contract requires it.**

   Step 7.3 makes climate projections and climate experiment optional, while `dev/roadmap.md:173–176` requires all workflows to run end-to-end and `check_baseline.py record` refuses an incomplete target set. Optional execution also permits stale outputs to be fingerprinted. All three workflows must be freshly run against the same baseline project before recording.

5. **The expected suite counts are stale.**

   Steps 0.2, 3.9, 3a.5, 7.1, 7.7, and 7.11 expect `45 passed, 4 xfailed`. The current roadmap records `47 passed, 2 skipped, 2 xfailed` at `dev/roadmap.md:177`. The two default-skipped integration tests are:

   - `tests/test_workflow_model_creation.py`
   - `tests/test_workflow_climate_projections.py`

   Also, the workflow-2/3 CLI cases in `tests/test_cli.py` are now explicit known-failure ratchets, not xfails. Update counts and terminology throughout.

6. **Two config consumers are missing from the migration.**

   - `tests/test_workflow_model_creation.py:49–51` reads `cfg["data_sources"]`.
   - `tests/test_workflow_climate_projections.py:39–43,56–57` reads flat `project_dir` and `clim_project`.

   These fail when integration tests run after sectioning. Task 3 and the file inventory must include both files.

7. **Task 5’s description of the Linux config is factually wrong and unsafe.**

   Step 5.1 says differences are “likely path differences only,” while `config/snake_config_model_test_linux.yml` also differs in:

   - `project_dir`
   - historical end date
   - model resolution
   - observation paths
   - climate-model list
   - presence/absence of optional keys

   “All other structure mirrors” the Windows config is ambiguous enough to overwrite these values. Task 5 needs an explicit value-preserving migration.

8. **Step 3.7’s syntax check is invalid.**

   `ast.parse()` cannot parse Snakemake grammar such as `rule all:` at `Snakefile_model_creation:47`, `Snakefile_climate_projections:55`, and `Snakefile_climate_experiment:56`. Delete Step 3.7 or replace it with Snakemake’s parser/dry-run verification. Task 3a.4 remains valid because those are ordinary Python files.

### Canonical schema inconsistencies

Before migration, reconcile the plan with `dev/milestones/r01/modularity-contracts-design.md`:

- The design’s migration table places `wflow_outvars` under `workflows.model_creation`; Tasks 1, 3.2, and 6 place it under `shared`.
- The design maps flat `historical` to `shared.historical_window`; the plan correctly distinguishes shared ISO datetimes from `workflows.climate_projections.historical_year_range`, but the design was not updated accordingly.
- The design maps `realizations_num` to `workflows.climate_experiment.realizations`; the plan retains `realizations_num`.

The plan’s choices generally fit the current code better, but the canonical contract must say the same thing before execution.

### Command errors

- The plan uses Bash-only `tail`, `head`, and `$(cat <<'EOF' ...)` constructs throughout, despite the Windows/PowerShell execution environment. Provide PowerShell commands or explicitly require Git Bash.
- Step 7.7 searches for `## R1 — Modularity contracts (pre-R3)`, but the actual heading is `### R1 — Modularity contracts (in flight)` at `dev/roadmap.md:135`.
- Step 7.8 searches for text already updated at `dev/roadmap.md:268–269`.
- Step 7.10 verifies with `git tag -l "m0*"`, which cannot match `r01-contracts`.
- Commit subjects use `m02d:` although `dev/roadmap.md:459–475` requires the Phase 2 prefix `r01:`.

## 4. Completeness pass

The following steps are required for the stated end state:

1. Reconcile Task 2 and all seal/report text with the contract-doc deferral in the design and roadmap.
2. Implement `workflow.configfiles[0]` in all three Snakefiles or undo the roadmap’s “Done by R1” declaration.
3. Add both `tests/test_workflow_*.py` files to the atomic config migration.
4. Move Task 3a before the atomic migration commit, so no committed state is known to parse successfully but fail at runtime.
5. Replace Task 3b’s hand-picked patterns with an audit that also finds lazy/indirect YAML consumers. The targeted repository check found no additional production snake-config consumers: the R scripts under `src/weathergen/` read derived weathergen YAML, while `src/setup_reservoirs_lakes_glaciers.py` reads its own HydroMT update config.
6. Define one baseline project and use it for the pre-change manifest, all three fresh workflow runs, comparison, candidate recording, and final check.
7. Compare a candidate manifest with the old manifest before overwriting the canonical manifest. Normalize the project root or compare records by target template, not resolved path.
8. Make all three end-to-end workflows mandatory for sealing; if required data access is unavailable, R01 remains unsealed.
9. Add a semantic old-versus-new config comparison for all three migrated YAMLs, especially the Linux variant. Every old leaf value should map to one declared new path.
10. Add focused tests for the list/string horizon normalization and sectioned stress-test readers. Task 3a currently notes that normal pytest does not exercise them, while Step 7.3 makes two relevant workflows optional.

## 5. Top three silent-drift risks

| Risk | How drift becomes silent | Required mitigation |
|---|---|---|
| Re-recording against a different project root | Step 7.5 replaces all manifest keys and can establish changed scientific fingerprints as the new contract; the subsequent check only proves agreement with the newly written manifest. | Run and compare under one root, write a candidate manifest first, normalize target identities, and mechanically require every scientific fingerprint to equal the pre-R01 record before replacing the canonical manifest. |
| Linux values overwritten while “mirroring” Windows | Task 5 performs only a YAML parse check, so changed basin, dates, model list, resolution, or observations remain undetected. | Restructure the existing parsed Linux mapping without copying Windows values. Add an explicit old-leaf → new-path table and a script/check asserting value equality for every migrated leaf. |
| Sectioned config parses while lazy consumers/defaults remain wrong | Dry-runs do not execute scripts; default-skipped integration tests retain flat reads; Task 3 commits before direct readers are fixed. | Put Snakefiles, all production readers, `tests/conftest.py`, and both integration-test readers in one atomic commit; add focused reader tests; expand the audit to all YAML loads and config accessor helpers. |

## 6. Minor findings

- Step 7.4 creates a document dated `2026-05-09`; use the sealing date.
- The opening architecture still says “1 template + 3 contract docs,” which is stale after the amendment.
- The quick-reference commit count is inconsistent: with Task 3b conditional, the plan has eight normal commits or nine if audit fixes are needed—not “9 commits” unconditionally.
- Step 7.11’s modified-file summary omits `tests/conftest.py`, the three `src/` files, and the two integration tests that must be added.
- Replace “existing xfail” in Steps 3.8 and 7.2 with “known-failure ratchet.”
- Clarify “zero baseline diff” everywhere as: pre-rebaseline scientific fingerprints unchanged; only documented config snapshots changed; post-rebaseline full check clean.