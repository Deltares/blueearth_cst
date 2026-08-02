# R03 design review — GPT-5.6 (2026-07-19)

## Verdict

**Not ready.** The design is well structured, respects most milestone boundaries, and correctly identifies the two load-bearing areas: cross-Snakefile reconciliation and intentional baseline movement. However, execution should not begin until three issues are resolved in the design itself: the constant-parameter inventory is arithmetically wrong, the scientific basis for restoring those parameters remains undecided, and the commit decomposition contradicts the design’s own behavior-preserving gate. The proposed baseline evidence also cannot reliably prove that the intended physics change occurred. These are targeted corrections, not grounds for redesigning R3.

## Tension reconciliations

### 1. §2 cross-Snakefile ratchets

**Judgment: sound, with a fixture-contract condition.**

The split by “where the fix lives” is defensible and respects the roadmap’s territory rule. `Snakefile_climate_projections` correctly declares `hydrology_model/staticgeoms/region.geojson` as an `ancient(...)` input, and the documented execution order requires workflow 1 to produce it before workflow 2 runs. Making that production input optional would weaken the workflow contract and would be an R4 content change. Conversely, the workflow-3 `CyclicGraphException` arises from DAG matching inside `Snakefile_climate_experiment`, so deferral to R5 is correct. The specific-exception assertions in [tests/test_cli.py](tests/test_cli.py) are genuine ratchets, as §2 states.

Pre-staging `region.geojson` is a valid repair to an isolated workflow-2 dry-run fixture, not a repair to the production workflow. It does not mask a missing producer declaration because the three workflows are deliberately separate entry points. The design should nevertheless specify that the staged file:

- is created under a test-owned temporary `project_dir`, not under the tracked baseline directory;
- is syntactically valid GeoJSON with the minimal schema expected by any parse-time consumer;
- is removed with the fixture;
- is accompanied by a test or contract assertion that workflow 2 still declares the exact workflow-1 output path.

An “empty-but-valid” file proves only DAG construction, not semantic usability. §2 and the risk at §357–387 should say this explicitly. A friendlier production error remains legitimately R4 territory.

### 2. §3 `src/snake_utils.py`

**Judgment: correctly scoped, but the import mechanism must be decided before implementation rather than left as a fallback.**

The roadmap explicitly authorizes creating `src/snake_utils.py` and changing all three Snakefiles only to import the shared helper ([dev/roadmap.md](dev/roadmap.md), R3 lines 296–301). The repository currently has three duplicated `get_config` definitions and no shared helper. Limiting the R4/R5 edits to deleting those definitions and adding one import respects the exception to the territory rule.

The proposed safety checks are useful but incomplete. A workflow-1 dry-run plus the two specific ratchets can detect many import or config failures, but they cover only the invocation path used by `tests/test_cli.py`, which changes directory to the repository root. `from src.snake_utils import get_config` relies on the repository root being importable. The design identifies this risk but leaves the solution to implementation (§3 and Risks).

That is too late for a design intended to execute directly. Before commit 2, the design should enumerate every documented wrapper invocation and choose one stable mechanism. At minimum, verify the root-level commands, `run_snake_test.cmd`, and the Linux/Docker wrapper. If they do not all preserve the repository root as the Python import root, specify a deterministic import bootstrap based on the Snakefile location. An ad hoc working-directory-dependent `sys.path` insertion should not be discovered mid-commit.

The helper also needs exact equivalence tests before the three local definitions are deleted, including missing required keys, optional missing keys, explicit defaults, `None`, and falsey values.

### 3. §4 outlet naming

**Judgment: the decision is reasonable, but its justification and traceability artifact need tightening.**

Keeping stable positional filenames such as `hydro_wflow_1.png` is appropriate for the current contract. The current `rule all` names that file statically, while outlet IDs are basin-derived. A stable positional product is useful for a multi-basin workflow and avoids turning the manifest into a basin-specific path inventory.

The statement that data-derived IDs “cannot” be represented is too absolute. Snakemake checkpoints, directory outputs, or a stable index artifact could support data-derived products. Those mechanisms would add orchestration complexity and likely belong outside R3, so the rejected alternative still loses—but it should be rejected on proportionality and contract stability, not impossibility.

More importantly, §4 promises that the positional-to-real-ID mapping will be recorded in `performance_metrics.csv`. Current [src/plot_results.py](src/plot_results.py) writes that file from `df_perf_all`, which remains empty when observations are absent or the run is too short for metrics. The tracked seed config has no observations in [config/snake_config_model_test.yml](config/snake_config_model_test.yml), so that CSV is not a dependable mapping artifact.

R3 should require one unconditional machine-readable mapping, either:

- a dedicated stable outlet-index CSV;
- explicit ID columns in a performance table that is populated even without observations; or
- a documented table derived from `outlets.geojson`.

Showing the ID in a plot title is useful but insufficient for machine traceability. If a new mapping filename becomes part of the public output contract, the design must address whether it belongs in `rule all` and the manifest.

### 4. §5 CSDMS constant-parameter restoration

**Judgment: not ready; both the inventory and the scientific decision must be resolved before execution.**

The scientific concern is legitimate: M2b dropped parameters under upgrade pressure, so silently accepting newer defaults would itself be an unreviewed scientific choice. R3 owns `config/wflow_build_model.yml`, and restoring verified mappings is mechanically within workflow-1 territory. Nevertheless, “pre-M2b values existed” is not enough to establish that they remain scientifically preferable. The handoff says they were global/default values, not local calibration, and the design acknowledges that their provenance is not fully reconstructable (§357–371).

This restoration is acceptable in R3 only if it is treated as an explicit scientific decision nested within the refactor, with a pre-implementation gate:

1. Inventory every pre-M2b parameter and value.
2. Identify the exact hydromt_wflow/Wflow 1.x mapping or deprecation status.
3. Compare each old value with the effective Wflow 1.x default, including units and semantics.
4. Classify each as restore, intentionally adopt new default, or drop as deprecated.
5. Record the rationale and reviewer approval before changing the build config.

Without that table, the implementation would be making scientific choices while resolving names. Any parameter whose semantics or units changed should be separated into a dedicated scientific-review task rather than restored mechanically.

The mechanics are conceptually sound but inadequately evidenced:

- Owned versus propagated versus unchanged classification is the right framework.
- Workflow 2 is a useful independence check because it consumes `region.geojson`, not `staticmaps.nc`.
- Per-parameter inspection of `staticmaps.nc` or the TOML is necessary.

However, the current manifest does not fingerprint `staticmaps.nc`, `wflow_sbm.toml`, or `output.csv`; workflow 1 is represented only by three size-tolerant PNGs and the copied snake-config YAML ([dev/baseline/manifest.json](dev/baseline/manifest.json), [dev/scripts/check_baseline.py](dev/scripts/check_baseline.py)). The YAML snapshot will not change merely because `config/wflow_build_model.yml` changes. Workflow-3 aggregate CSVs may change, but their response does not prove that every restored parameter landed or that the change has the expected direction.

Before execution, §5 should require a committed, machine-readable before/after parameter table with source name, CSDMS name, old value, new/effective default, units, storage location, and observed built value. It should also require a direct fingerprint or assertion against the generated staticmaps/TOML and a quantitative comparison of workflow-1 discharge output. A manual statement that parameters were “present” is too weak.

### 5. §9 `extract_climate_grid` disambiguation

**Judgment: correct.**

Both the truncation warning and the hardcoded/config-staleness problem belong to `extract_climate_grid` and `src/extract_historical_climate.py`, which are invoked from `Snakefile_climate_experiment`. Deferring them to R5 respects the iron rule and avoids splitting one rule’s repair across milestones. Retagging the conflicting R3 entry in [dev/followups.md](dev/followups.md) is appropriate. R3 may adopt the general audit principle for workflow-1 rules, but it should not edit the workflow-3 rule or script.

## Correctness findings

### Major: the constant inventory is off by one throughout §5

[dev/milestones/phase-1/m02b/handoff.md](dev/milestones/phase-1/m02b/handoff.md), decision #3, lists **15 original parameters**, not 14:

- one retained parameter: `KsatHorFrac`;
- eight already mapped dropped parameters: `Cfmax`, `WHC`, `TT`, `TTI`, `TTM`, `G_Cfmax`, `MaxLeakage`, `InfiltCapPath`;
- one deprecated parameter: `InfiltCapSoil`;
- five unresolved parameters: `cf_soil`, `EoverR`, `rootdistpar`, `G_SIfrac`, `G_TT`.

Therefore the arithmetic is:

- 15 original;
- 1 retained;
- 14 dropped;
- 8 known mappings + 1 deprecated + 5 unresolved = 14.

The following design statements are incorrect:

- §5 title and body: “13 dropped”;
- §5: the five listed unresolved names are called “the remaining four”;
- Risks: “Five of the thirteen mappings”;
- any implied target of restoring 13 parameters.

The historical handoff itself contains the same inherited miscount when it says “14 constant pars” and “other 13,” but its explicit list controls. The design must correct and reconcile the inventory rather than repeat the handoff’s prose.

### Major: §5 overstates expected workflow-1 manifest drift

§5 says the workflow-1 slice—“3 PNGs + config snapshot YAML”—moves. The manifest’s model-creation YAML is `snake_config_model_creation.yml`, copied from the top-level snake config. Restoring constants in `config/wflow_build_model.yml` does not change that snapshot. The YAML changes only if R3 also changes the tracked snake config, for example by changing `wflow_outvars`.

The design should classify expected drift by cause:

- constant restoration: potentially the three PNGs and propagated workflow-3 CSVs;
- `wflow_outvars` change: copied snake-config YAML plus plots/output structure;
- code-only hardening: no valid-config output drift unless behavior is intentionally changed.

### Major: §4’s promised mapping is not guaranteed by current code

As noted above, [src/plot_results.py](src/plot_results.py) only adds performance metrics when observations overlap and the run is long enough. `performance_metrics.csv` can therefore be empty and cannot presently guarantee position-to-ID traceability. §4 must name an unconditional implementation.

### Correct: repository cross-checks

The following factual claims match the current repository:

- `src/snake_utils.py` does not exist, while all three Snakefiles duplicate `get_config`.
- `Snakefile_model_creation` has no `ruleorder:`.
- `Snakefile_climate_projections` contains the load-bearing `ruleorder:`.
- `add_reservoirs_lakes_glaciers` and `add_gauges_and_outputs` use `ancient(staticmaps.nc)` because they mutate the model in place.
- [src/setup_reservoirs_lakes_glaciers.py](src/setup_reservoirs_lakes_glaciers.py) catches `NoDataException` and `FileNotFoundError` per method and writes a free-text sentinel.
- [src/setup_gauges_and_outputs.py](src/setup_gauges_and_outputs.py) contains the stated `WFLOW_VARS` lookup and silently drops unknown extras.
- [tests/test_cli.py](tests/test_cli.py) asserts both nonzero status and the specific `MissingInputException`/`CyclicGraphException` strings.
- The M2b handoff names exactly eight known mappings and identifies `InfiltCapSoil` as deprecated.

## Completeness and execution readiness

The design is close to the R1/R2 documentation bar: it has scope, alternatives, consequences, verification, risks, and an implementable commit outline. It is not yet safe to execute directly because several decisions remain embedded as implementation-time discovery:

- five CSDMS mappings and their units/semantics are unresolved;
- the import mechanism is undecided;
- the waterbodies branch depends on a timeboxed upstream-behavior experiment without a defined duration, command, or acceptance result;
- the default `wflow_outvars` decision is not made;
- the unconditional outlet mapping artifact is unspecified;
- logging redirection for `script:` rules is described only as a possible “second member” of `snake_utils`, not as a concrete API with restoration and exception behavior.

### Commit ordering and clean-gate discipline

The stated order is internally inconsistent.

The Approach says the two behavioral changes are constant restoration (§5) and gauges hardening (§7), yet the decomposition places gauges hardening in commit 7 and declares commits 1–7 behavior-preserving. The unit/unknown-name work may preserve valid tracked outputs, but it changes error behavior. More importantly, §7 may change the default `wflow_outvars`, which the design itself says can move the baseline.

Commit 9 is also treated as clean despite requiring the real ID to appear in a plot title or legend. That can change PNG size and content. If it changes or guarantees `performance_metrics.csv`, it changes an output not currently represented in the manifest.

Commit 6 changes sentinel content from free text to structured output. Although the sentinel is not a manifest target, that is still an observable interface change and must be reflected in the workflow contract and tests.

A safer ordering is:

1. Contract doc describing current behavior.
2. Shared helper extraction and equivalent tests.
3. Logging/benchmark plumbing.
4. Dry-run fixture repair.
5. Internal label renames.
6. Behavior-preservation baseline check.
7. Structured sentinel and gauges validation, with explicit classification as interface hardening.
8. Outlet contract and unconditional mapping artifact.
9. Approved constant/default-output scientific changes.
10. Unit and integration tests for the final behavior.
11. End-to-end rerun, direct physics evidence, manifest record, and baseline-diff note.
12. Final clean check, roadmap seal, and tag.

If the default output set or outlet plot changes, the design should either combine all intended baseline-moving changes before the single record commit or record separate before/after evidence for each cause. “Only commit 8 may dirty the baseline” is untenable as currently written.

The `check_baseline check` gate should run immediately before the first intentionally output-changing commit, not merely after an ambiguously defined group of “behavior-preserving” commits. A clean 14/14 result does not prove behavior preservation for untracked artifacts such as staticmaps, TOML, the sentinel, performance metrics, or logs.

## Additional execution risks

### 1. A wrong CSDMS mapping can appear present but encode the wrong quantity

A variable name may be accepted and written while its units, spatial representation, or Wflow semantic role differs from the old short-name parameter. Presence-only assertions would pass.

**Mitigation:** require a parameter reconciliation table and assert name, value, units, dimensions, and expected storage location. Compare the effective generated TOML/staticmaps against both the pre-M2b configuration and Wflow 1.x defaults.

### 2. Baseline re-recording can bless stale or partially regenerated outputs

The design proposes rerunning workflows 1 and 3 while treating workflow 2 as unchanged. Existing outputs under `examples/test_local` can satisfy Snakemake timestamps, especially around in-place staticmaps mutation and `ancient()` inputs. A record operation could therefore combine newly generated and stale artifacts.

**Mitigation:** define an explicit freshness check for every recorded target and each relevant upstream model artifact. Use Snakemake’s rerun reasoning or a clean, dedicated project directory for the R3 baseline run, then refuse to record unless all 14 targets and the direct staticmaps/TOML evidence were produced by that run. Do not rely solely on file existence.

### 3. Logging changes can alter rule completion and hide failures

Redirecting shell output is straightforward, but replacing Python process streams inside Snakemake scripts can leak redirected streams across imports, swallow tracebacks, or leave empty log files that Snakemake treats as completed products. Concurrent rules also need distinct paths.

**Mitigation:** specify one tested context manager that restores `stdout`/`stderr` in `finally`, preserves tracebacks, creates parent directories, and uses unique per-rule log paths. Add a unit test for normal completion and exception restoration. State whether logs and benchmarks are tracked outputs, ephemeral run artifacts, or ignored files.

### 4. The ±10% PNG fingerprint can miss the intended change entirely

The current fingerprint stores only file size. A scientifically meaningful curve change can retain nearly identical compressed size, and harmless metadata/rendering changes can cross the threshold. The three workflow-1 manifest targets therefore cannot establish preservation or intended drift.

**Mitigation:** keep PNG size as an advisory rendering check, but add data-level evidence: normalized discharge-series hashes/statistics, selected quantiles and flow-duration metrics, and a direct staticmaps/TOML parameter audit. The baseline note should distinguish “physics verified from data” from “rendering inspected.”

## Minor findings

- §1 should distinguish outputs explicitly targeted by `rule all` from side-effect artifacts and downstream contract outputs. `staticmaps.nc`, `region.geojson`, `outlets.geojson`, TOML, forcing, and `output.csv` are important contracts but are not all direct `rule all` targets.
- §6 says `log:` and `benchmark:` apply to every non-trivial rule except `copy_config`, but should define “non-trivial” once and give the exact path convention R4/R5 must inherit.
- The waterbodies experiment needs a fixed timebox and a named acceptance test; “attempted first” is not directly executable.
- The design should state whether changing the tracked seed’s `wflow_outvars` is required or merely an option. The canonical config and test fixture currently have different output sets.
- Commit 12’s “durable refs” is vague and violates the roadmap preference for one logical change per commit if it includes unrelated follow-up retagging, roadmap edits, and sealing. Name the exact files or split it.
- Correct all references to “13 dropped,” “remaining four,” and “five of thirteen” together so the design, baseline note, tests, and eventual commit messages use one authoritative 15-parameter inventory.