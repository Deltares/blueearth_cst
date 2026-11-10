# Validation ladder — tiering, configs, and CI

Last updated: 2026-11-10

Choose checks from changed behavior, affected claims and consumers. This page owns acceptance and
evidence-reuse policy. [Validation procedures](dev-validation-procedures.md) gives the infrequent
commands and mechanics; a passing command never proves claims it did not test.

## Validation ladder

| Stage | Required coverage |
|---|---|
| Task | Focused checks during iteration; complete affected checks before commit. |
| Local landing | Complete affected checks; broaden for uncertain impact or integration risk. |
| Publishing | Before push or release, satisfy the batch gate; confirm both CI platforms after a CI-triggering push. |

Before pushing to `origin` or releasing, review the **complete unpublished delta**. Require one
integrated `test-fast` pass, stronger coverage or applicable reused evidence; an exempt-only delta
skips the software gate. Add affected contract and tooling tests excluded from fast and any owed execution,
numerical or visual checks. Use `test-full` when its wider coverage is needed. Resolve known
regressions and missing coverage before publishing. Local `main` can hold regressions not yet found
by a publication gate. CI from a proposed push is later assurance, not evidence for this gate;
earlier CI can substitute only for an equivalent candidate, environment and inputs. No suite
guarantees zero regressions.

Select by claim:

| Change | Affected checks |
|---|---|
| Rules, inputs, scripts | Regression/consumer checks, affected CLI/DAG and contract checks. |
| Workflow outputs | Exercise affected integration paths and downstream consumers. |
| Numerical results | Independent numerical oracle and affected consumers; current baseline for whole-tree claims. |
| Prose | Inspect links, examples and policy consistency; check behavioral readers. |
| Figures | Render and inspect every affected family, captions and outputs; reuse valid renders. |

An existing independent expected-result test satisfies the claim it covers; do not duplicate it.
Execution alone cannot accept numbers, and successful rendering cannot establish figure legibility.
Do not rerender or reinspect unaffected figures unless a shared change invalidates them.
Executable examples, parsed documentation and calculation changes need affected behavioral checks.
Figure-only work does not automatically owe the software batch gate.

For Python edits, run scoped Ruff before commit and `pixi run lint` plus `pixi run format-check`
before publishing. Count an applicable manual or hook pass on the same candidate; rerun only when
invalidated. Keep the pre-push hook enabled and do not duplicate its checks immediately before it.
For a release without a push, run any missing checks directly. Release and milestone labels add no
checks beyond changed behavior and qualification claims.

### Batch and push exemptions

An all-prose or terminal-figure-presentation delta can skip the software batch gate only after
inspection establishes no runtime, config, calculation or test-consumed contract change. This
applies by content, not directory, but does not waive affected link, example, policy-consistency or
visual checks. Search readers/callers first: changed test-read documentation is exempt only when
the test's contract cannot change. Scripts, config templates and executable notebook cells are not
prose; uncertain impact retains the gate. Renames and deletions are non-exempt because path readers
may break. At push, an exempt-only delta needs the pre-push hook and missing affected checks, not a
software suite.

## Planning and reusing verification

- Identify affected claims and inputs before scheduling checks; a sentence suffices for a simple
  change. Record commands, results and material limits in existing work notes or the completion
  report. Include candidate, environment, config/input identity and output root when needed to
  interpret or reuse execution. A zero exit establishes command success only.
- Add a lasting test or process requirement only for a concrete uncovered failure; prefer improving
  existing coverage. A one-off diagnostic does not become a standing gate. Stop when required
  coverage is sound; repeat only for invalidated evidence or a specific diagnosis. Preserve
  independent numerical and visual acceptance. After infrastructure failure, probe the failing
  prerequisite before repeating a suite.
- `test-full` satisfies `test-fast` for the same candidate and environment; do not schedule both as
  final gates. Fast before full can be early feedback. If all-five E2E is owed, use its fresh rapid
  root for inventory, captions, visual and numerical checks rather than another identical run.
  A smoke run may diagnose. Extra cases need distinct claims (single-model availability, optional
  products, P0/lab parity, baseline numerics or synthetic edges); state their added coverage.
- Finish synchronization and executable edits before final gates. Exact tree equivalence permits
  reuse. Otherwise review **every** changed path, including test-read prose and transitive inputs;
  recheck affected claims and retain unchanged ones. An earlier broad change does not escalate a
  later narrow one. Changed environment/inputs invalidate dependent evidence.
- After an equivalent merge, prove tree equality and ancestry instead of rerunning gates. Review
  all commits proposed for `origin`; for a main push include `origin/main..main` across locally
  landed branches. Push still owes CI and any newly uncovered coverage. Capture gate output in a
  file, never through `tail`.
- Apply policy-tooling suites only when changing that tooling or its contracts, not when using its
  verifier. Compare check durations only with candidate/environment context; “fast” is no guarantee.

Formal receipts are needed only for gate-result reuse. The tracked
[`verify_landing.py`](../scripts/verify_landing.py) checks identity, not coverage or scientific
adequacy. See [receipt procedures](dev-validation-procedures.md#recording-gates-and-verifying-landing).

## Pytest temporary directories

Snakemake cache and pytest scratch have distinct roots. Use the
[Windows/pytest procedure](dev-validation-procedures.md#pytest-temporary-directories) for access
or fixture issues; never hide an access failure with a longer workflow timeout.

## Tiering

- `test-fast` excludes `tests/private/` and deselects `workflow_contract` and `process_isolation`,
  including expensive fresh-process and DAG checks. Count current cases with
  `pytest tests/ -m "workflow_contract or process_isolation" --collect-only -q`; do not rely on a
  fixed count. Other setup/render costs still matter; compare current timings before changing tiers.
- Run affected private tooling tests when changing their scripts, hooks, synchronization/backup policy,
  board/session coordination, or shared helpers, dependencies and test infrastructure they consume.
  Use `pixi run pytest tests/private/test_<name>.py` for bounded changes and `pixi run test-tooling`
  for broad or uncertain impact. Ordinary toolbox changes need this tier only when they affect those
  consumers. The tier requires optional local tooling files absent from public clones and CI.
  `test-full` and direct `pytest tests/` still include private tests when present; their exclusion from
  fast does not waive affected tooling checks at task, landing or publishing stages.
- Fast uses `-n auto --dist loadfile` to keep a test file's module-level state on one worker.
  `--dist load` can break that isolation; adding workers eventually meets per-worker import cost.
  `test-contract` stays serial because isolated engine state does not prove independent output and
  process resources. [Test-author isolation](dev-validation-procedures.md#test-author-isolation-details)
  covers Snakemake profiles, pytest scratch and import-time stubs.
- Snakefile edits that move admission, DAG construction, outputs or execution/journaling work between
  parse time and `onstart`/`onsuccess`/`onerror` owe `test-contract`: fast excludes lifecycle tests.
  Include any other affected contract tests excluded from fast.
- Console-only layout and terminal-handler changes, including presentation timing across those hooks,
  need affected console/launcher tests and lifecycle contract cases, plus one representative output
  inspection. Cover dry-run, no-op, early errors and detailed file-log preservation when affected.
  Moving presentation earlier does not itself require unrelated scientific execution fixtures or the
  full contract tier. Broaden for a concrete risk to admission, DAGs, exit status, machine-readable
  records or scientific outputs; required publication gates still apply. Resolve an over-budget
  selection with the owner before launching it.
- CI tests Windows and Linux on a bare checkout. Fixture-dependent tests can skip; real workflow
  execution and local baseline comparison remain local. CI success does not establish those claims.
  A PR supplies two-platform assurance before landing when needed. Inspect failures and relevant
  skips; successful job logs need no routine full read.

## Recording gates and verifying landing

For formal reuse, record owed gates on the tested candidate, inspect skips and verify identity.
Missing, failed/skipped or unclassified evidence cannot be reused. Follow-ups need cumulative-delta
review and affected checks; the verifier cannot judge that review or accept scientific results.
See the [formal receipt protocol](dev-validation-procedures.md#recording-gates-and-verifying-landing).

## Choosing execution coverage

- Trace behavior through callers, outputs, shared runtime and dependency changes. Several affected
  handoffs alone do not require all-five execution. Exercise bounded paths; use all five for
  pipeline-wide impact, whole-toolbox claims or unresolved cross-workflow uncertainty. State affected
  paths and omitted workflows; reuse applicable evidence.
- Console wording/formatting alone needs affected console tests and representative output inspection.
  Exit status, exceptions, machine-readable output and control flow are behavioral. A file under
  `shared/` does not itself escalate. Confined numerical changes need an independent oracle and
  affected consumers, not automatically all-five execution or a whole-tree baseline. Whole-tree
  numerical claims need a current baseline.
- For one workflow, search callers/output consumers and exercise its affected integration path.
  Admission, retry and rejection changes need those paths, not an unrelated simulation. When outputs
  can change, execute affected workflow and consumers with rapid inputs. WF3 uses an
  WF3-only project config through `run_workflows.py`; WF4 uses `simulate_system.py` or the all-workflow
  runner, never bare Snakemake.
- For figure-only changes, search shared plotting callers, render affected figures and provide PNGs
  for inspection. Do not require a baseline, full suite or render-byte comparison for presentation
  alone. Runtime/config contracts or non-figure consumers can require broader checks; changed
  calculations still require numerical acceptance. Escalate when focused checks show wider breakage.

## The end-to-end runs (`test-e2e-local`, `test-e2e`)

- `test-e2e-local` and live `test-e2e` run the same five-workflow rapid case and deliverable checks
  in fresh roots. Neither can pass from earlier output; skips are no evidence. The sealed local gate
  regenerates downstream products and covers their execution, not remote availability.
- Add focused fetch checks for acquisition changes. Use live E2E for catalog availability, remote
  addressing, acquisition windows, clipping, cache behavior and qualification of a new bundle
  against live sources. Record candidate, bundle seal, output root and live/local comparison for a
  parity claim. Release/milestone labels alone do not require repeating applicable execution.
- Live E2E needs a data mirror, juliaup Julia, `weathergenr` and WF2 network access, outside CI.
  [Bundle setup and seal checks](dev-validation-procedures.md#the-end-to-end-runs-test-e2e-local-test-e2e)
  state source and region conditions.

## Which config to run

| Config | `project_dir` | Purpose |
|---|---|---|
| `project_config_rapid.yml` | `test_case/test_rapid` | Default execution, DAG and figure checks. |
| `project_config_baseline.yml` | `test_case/test_local` | Baseline recording/checking and numerical claims. |
| `project_config_wf2_fast.yml` | `test_case/test_dev` | Restricted WF2 iteration: two series, no `st_0`. |

Rapid and baseline both simulate nine years (2046–2054) and request 40 generator years
(`compute_nr_years(2054, 2016)`). Rapid has a 2 × 2 perturbation grid plus `st_0` (10 members);
baseline has 2 × 3 plus `st_0` (14). Rapid keeps the unperturbed member needed by two class-C monthly
indicators and two CMIP6 models for ensemble reduction; `wf2_fast` gives up coverage. To alter
generated length, change `simulation_window.end`, not its start: the series spans from the end of
`climate.window` to that end plus two years. Keep the simulation window within a projection
`future_windows` entry; alignment is calculated separately.

Record the baseline only from the baseline config; never point `check_baseline.py` at rapid. Its
`check` exits 0 for matched targets, 1 for differences and **2 for no comparison** (empty scope or
unresolvable fixture). Treat 2 as no evidence; `--workflow generate_scenarios` has no non-figure
target and returns 2 by construction.

**The tracked baseline manifest is stale** after the P-drive catalog and GlobCover switch; its
recorded paths also predate the current config archive. Regenerate `test_case/test_local` from the
current baseline config and re-record `dev/baseline/manifest.json` before using it as a numerical
gate. Agreement with the old manifest does not accept current inputs. A predecessor comparison
with changed settings cannot isolate a layout or numerical effect.

`test_case/test_local` is untracked and independent per worktree. Inspect provenance, then re-seed
or regenerate affected trees; repairing one does not repair others. `project_dir` resolves from the
run directory. WF1's retained `run_default/output.csv` is required by the manifest. If older cleanup
removed it, rerun the historical simulation; rounded `output_q.csv` is not a substitute.

## Read the CI run after you push

The tracked [CI workflow](../../.github/workflows/ci.yml) runs on `main` pushes and PRs, and still
lists `dev` from a temporary trial. Public trunk is `main`. Check the intended commit's Windows and
Linux jobs; inspect failures and relevant skips. A green job cannot replace missing local execution,
baseline numerics or visual acceptance. CI from the push follows, rather than replaces, the pre-push
gate. A failed Linux leg is not cleared by local success; diagnose it before release.

Use a PR when two-platform assurance is needed before landing. The pre-push hook runs Ruff, not the
suite or platform checks; keep it enabled. A later push checks newly uncovered claims and the
combined unpublished delta, not the same evidence again. A red or unread CI result needs diagnosis,
not automatic rerunning of a matching local suite. New pushes cancel in-flight CI for the same ref:
check the latest run's commit. The [CI procedure](dev-validation-procedures.md#ci-inspection-commands)
has hook setup and repository-explicit inspection commands.
