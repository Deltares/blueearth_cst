# Validation ladder — tiering, configs, and CI

Last updated: 2026-10-21

This reference owns verification policy: check selection, evidence reuse, execution coverage,
config provenance, and CI. `AGENTS.md` routes verification work here.

## Validation ladder

| Stage | Required coverage |
|---|---|
| Task | Focused checks as needed while iterating; complete affected checks before commit. |
| Local landing | Complete affected checks; broaden only for uncertain impact or integration risks. |
| Publishing | Before push or release, satisfy the batch gate below; confirm both CI platforms after push. |

Local `main` may contain undiscovered regressions between landing and publishing. Before pushing to
`origin` or releasing, review the complete unpublished delta and require one integrated `test-fast`
pass, stronger coverage, or applicable reused evidence; exempt-only deltas skip the software gate.
Include affected contract tests excluded from fast and any owed execution, numerical, or visual checks.
Use full when broader or uncertain impact needs its extra coverage. Resolve known regressions and
missing required coverage before publishing. This gate reduces risk; no suite guarantees zero regressions.

Existing CI evidence can substitute only for an equivalent candidate and relevant environment/inputs.
CI triggered by the proposed push is subsequent assurance, not evidence for this pre-push gate.

Select task coverage by changed behavior and affected consumers:

- **Software:** regression/consumer checks and affected CLI/DAG checks for rule/input/script changes.
  Include affected contract tests when needed; local `test-full` replaces fast when broader coverage is owed.
  Snakefile edits that move work or console output between parse time and `onstart`/`onsuccess`/`onerror`
  owe `test-contract`: `test-fast` excludes the `workflow_contract` tests that pin handler timing.
- **Workflow execution:** exercise affected integration paths and downstream consumers. Execute workflows
  when outputs can change. Use all-five E2E for pipeline-wide impact, explicit whole-toolbox claims,
  or cross-workflow uncertainty that inspection and focused checks cannot resolve.
- **Numerical results:** independent numerical checks and affected consumers; current baseline evidence
  for whole-tree numerical claims. Execution success alone does not establish numerical acceptance.
  An existing test with an independent expected result satisfies its covered claim;
  add no duplicate comparison.
- **Prose and figures:** inspection can suffice for prose; render and inspect affected figures.
  Reuse figures already produced by the verified execution; inspect changed output without rerendering
  or reinspecting unaffected figures unless a shared change invalidates them.
  Executable examples, parsed documentation, calculation changes, and other behavioral consumers also
  need their affected checks. Figure-only work does not automatically require the software batch gate.

For Python edits, complete scoped Ruff checks before commit and repository-wide `pixi run lint` and
`pixi run format-check` before publishing. Count an existing manual or hook pass when it covers the candidate
at the required stage; rerun only invalidated checks. Keep the pre-push hook enabled, but do not add
a duplicate manual run immediately before it. For a release without a push, run any missing checks directly.
Release and milestone labels add no checks beyond the changed behavior and qualification claims.

### Batch and push exemptions

The software batch gate may be omitted when every change is non-executable prose or terminal figure
presentation and inspection establishes that no runtime, configuration, calculation, or test-consumed
contract changed. This covers eligible content regardless of directory, including task notes, guides,
and agent instructions. It does not waive affected link, example, policy-consistency, or visual checks.

Classify the actual diff and search readers/callers before claiming an exemption. A document read by
a test is exempt only if the changed content cannot affect that test's contract. Scripts, config
templates, and executable notebook cells are not prose; uncertain impact retains the batch gate.
Renames/deletions remain non-exempt because path consumers may break. At push, an exempt-only delta
needs the pre-push hook and any still-missing affected checks, not a software suite.

## Planning and reusing verification

Safety with the fewest sufficient tests and procedures is the policy's design objective. Before adding
a lasting test or process requirement, identify the concrete uncovered failure and why existing
coverage or a simpler check cannot catch it. Prefer strengthening or replacing existing coverage;
remove redundant requirements when their purpose is covered elsewhere. A one-off diagnostic does not
become a standing gate by precedent. Preserve independent numerical and visual acceptance where needed.
Use existing task notes or the completion report for the rationale, not another form or approval step.

Before scheduling checks, identify the affected claims and inputs; a sentence is enough for a simple
change. Record attributable commands, results, and material limitations in existing notes or the
completion report. Include candidate, environment, input/config identity and output root when needed
to interpret or reuse execution evidence. Formal receipts are required only for gates reused through
the landing verifier, not for every focused task check. A successful exit does not discharge claims
the command never checked.

- A passing `test-full` satisfies `test-fast` on the same candidate and environment. Do not schedule
  both as final gates. A fast run before full is optional early feedback, not another acceptance duty.
- When all-five `test-e2e` is owed, use its fresh rapid output root for affected-workflow inventory,
  caption, visual and numerical checks. Do not also require a standalone fresh rapid run with the same
  inputs and behavior. A development smoke run remains useful for diagnosis.
- Keep additional cases only for distinct claims: single-model availability, optional products,
  P0/lab parity, baseline numerical checks and synthetic edge cases. State the coverage added by each.
  An execution pass does not replace a numerical oracle or visual inspection.
- Complete synchronization and executable edits before final gates. Reuse earlier passes only after
  comparing the tested candidate with the final candidate. Exact tree equivalence permits reuse;
  otherwise classify every changed path, including test-read documentation and transitive inputs.
  Recheck affected coverage and retain unchanged claims using reviewed delta evidence below.
  A prior broad change does not escalate a subsequent narrow change. Environment or input changes
  invalidate dependent evidence.
- At landing, an executable `task-land --verify` may verify recorded passing evidence and candidate
  equivalence instead of rerunning expensive gates. It must fail for missing evidence, failed/skipped
  required checks, or unclassified changes. After an equivalent merge, prove tree equality and ancestry;
  do not repeat gates. Push retains the CI and delta checks below.

For designs with several distinct claims, a compact table can help; simple changes need only a sentence:

| Claim | Check | Input case / output root | Extra coverage or reuse |
|---|---|---|---|
| Workflow execution | Applicable live-ladder gate | Fresh rapid root | Supplies shared outputs |
| Product completeness | Inventory/caption checks | Same fresh root | No second execution |
| Numerical equivalence | Independent oracle | Same inputs or named reference | Distinct scientific claim |
| Conditional behavior | Focused test or execution | Named alternate case | State what rapid misses |

Completed design and review records describe the gates actually used. Their historical command lists
do not prescribe future work; new tasks use this maintained guidance.
Stop once required coverage is sound; repeat only for invalidated evidence or a concrete diagnostic need.
Capture gate output in a file, never pipe it through `tail`.

## Pytest temporary directories

### Windows Snakemake cache access

Run checks that invoke Snakemake with normal user permissions outside a restricted agent sandbox on
Windows. This includes subprocess and in-process DAG checks, even when their pytest marker does not
identify them as workflow contracts. Pure Python checks can remain sandboxed. Request the tool's
normal-permission execution mode up front; do not first wait for the known cache failure to recur.
This is not Windows administrator elevation and does not authorize production runs or publishing.
The execution host still controls approvals; this repository does not grant or persist blanket access.

When cache access under a process token is uncertain, run:

```powershell
pixi run python dev/scripts/check_snakemake_access.py
```

The probe exercises the installed Snakemake source cache in a child process, including temporary-directory
creation and a file round-trip. It exits 0 for success, or 2 with a diagnosis; the default 15-second
timeout stops a stalled child. It runs no workflow and changes no permissions. A timeout can also mean
unusually slow imports: compare with normal permissions before attributing it to access restrictions.

Snakemake uses the Windows user source cache, independently of pytest's worktree-local scratch.
On the affected Windows/Python combination, `tempfile.mkdtemp()` can retry a denied directory creation
while its parent appears writable. Private temporary-directory ACLs can also exclude restricted tokens.
Changing pytest `--basetemp`, `TMP` or `TEMP` does not redirect this explicitly selected cache.
Do not patch installed packages, relax cache ACLs, or extend workflow timeouts to mask this failure.
If normal-permission execution also fails, diagnose that error before continuing.

### Worktree-local pytest scratch

Bare `pytest` and Pixi test tasks load the test-directory plugin from `tests/conftest.py`.
Each invocation creates `.tmp/pytest/run-<unique-id>/` in its own checkout; xdist places worker
directories beneath that run. The reusable cache lives in `.tmp/pytest/cache/`. Concurrent runs
never clear one another's default fixture directories. This covers `tmp_path`, `tmpdir` and
`tmp_path_factory`, not application calls to Python's `tempfile` module.

On Windows, the plugin creates these directories with inherited workspace ACLs. Python's `0700`
directory mode installs a user/admin-only ACL that denies the restricted sandbox token access;
moving that directory alone does not fix the failure. POSIX retains pytest's private permissions.
The adapter uses pytest's internal factory integration; verify the fixture test with `-n 2` after
pytest upgrades. Existing inaccessible temporary directories are left untouched.

Default run directories remain available for debugging; pytest's three-run retention count does
not apply to these unique roots. Instead, each new default run first removes this checkout's
`.tmp/pytest/run-*` roots untouched for two days (`STALE_RUN_SECONDS`), skipping links and roots it
cannot delete. Copy any run you need to keep beyond that window elsewhere. Explicit `--basetemp`
runs and xdist workers never prune. Never delete the shared parent during testing.
`--basetemp <path>` remains an explicit override with pytest's destructive semantics: pytest clears
that exact directory before use. Give concurrent invocations different paths. Tests that require a
path outside the checkout must explicitly choose one; ordinary fixtures now live inside it.

## Tiering

`test-fast` deselects the `workflow_contract` and `process_isolation` markers. They include expensive
fresh-process and Snakemake DAG checks. Other tests also incur setup and rendering costs; use current
duration evidence before optimizing either tier. Count the selected cases with
`pytest tests/ -m "workflow_contract or process_isolation" --collect-only -q` rather than trusting a number written here — it drifts every time a rule gains a contract test.

`test-fast` runs its workers in parallel (`-n auto --dist loadfile`). Keep `loadfile`: it puts every test in a file on one worker, which is what contains module-level state — a test that stubs a library process-globally stays confined to its own file. `--dist load` distributes per test and would break that. The floor is the per-worker import cost, so adding cores past a point buys nothing.

Repository-workflow CLI tests opt into `isolated_snakemake` in `tests/conftest.py`. It selects a fresh
test-owned working directory through `SNAKEMAKE_PROFILE`, including for owned launchers' child calls.
Keep that profile when constructing subprocess environments. Package sources and small relative-path
inputs are copied; producing tests use test-owned project outputs and each test owns its `.snakemake`
state. In-process simulation DAG checks
set their working directory explicitly. Parse-only API helpers build rule declarations without
initializing persistence or the IO cache. Synthetic Snakefiles already running in `tmp_path` retain
their own state there.

`test-contract` stays serial. Isolating engine state alone does not establish that every contract's
project outputs and process-level resources are independent.

CI runs the software suite on Windows and Linux. Fixture-dependent tests may skip and real workflow
execution remains local; CI success does not establish those missing claims. Use a PR when
two-platform assurance is required before landing. Confirm both platform jobs passed for the intended
commit; inspect failures and skips relevant to required claims. Successful job logs need no routine full read.

**A push checks uncovered changes, not the same candidate again.** Inspect `origin/main..main` and
reuse valid landing evidence. Exempt-only changes need cheap hooks; other follow-ups need checks for
the claims they invalidate, or a missing batch gate. A red/unread CI run needs inspection and diagnosis,
not an automatic repeat of a matching local suite. Local success cannot replace a failed Linux CI leg.
Apply the batch and push exemptions above; test readers can make documentation behavioral inputs.

Classify the unverified behavior and its consumers, not the preceding batch's gate. A confined
script-signature change needs affected producer/consumer checks; batch coverage belongs before publishing.
Policy-tooling suites apply when changing that tooling or its contracts, not when using `task-land`
or its evidence verifier. Do not add a separate suite merely because its name appears in a workflow.

Between merges and the next push, a regression can span several locally landed branches. Review their
combined delta at the pre-push gate, including affected contracts, rather than only the last task.
CI may find further platform-specific defects after push; resolve them before release. `ci.yml` cancels
an in-flight run when the same ref is pushed again, so check the latest run's commit and both platform jobs.

Use measured timings with their candidate and environment when comparing costs; do not treat old
timings or the name "fast" as a guarantee. Establish working cache permissions before a broad run.
After an infrastructure failure, probe the failing prerequisite before repeating the suite.

`.testing-policy.yml` pins `scope: rapid`, and `auto_push: false` keeps the push a deliberate decision.

## Recording gates and verifying landing

When reusing gate results through `task-land --verify`, use `dev/scripts/verify_landing.py` instead of
writing a task-specific verifier. An ordinary task check needs no receipt unless it will supply that
reuse. A freshly executed landing check can instead be the verification command itself.
The helper's `record` command executes one explicit gate, captures a fresh log and records the Git tree,
Python executable/version/platform, lock-file identity and named input-file hashes. Record during the original
test run, not by rerunning a passed command solely to produce a receipt. By default the candidate must
be clean and committed. Before commit, stage the intended changes and use `record --staged`; it refuses
unstaged changes and nonignored untracked files, and checks identity again after execution. Commit the
same staged content, then `verify` proves the clean committed tree matches without executing the gate.
Evidence and logs belong in `dev/tasks/<id>/.tmp/`; substitute the owning task ID in the examples.
Follow `dev-repository-layout.md` § Task temporary material for retention and closure cleanup.

```powershell
pixi run python dev/scripts/verify_landing.py record --evidence dev/tasks/<id>/.tmp/gates.json --gate test-fast --log dev/tasks/<id>/.tmp/fast.log -- pixi run test-fast
pixi run python dev/scripts/verify_landing.py verify --evidence dev/tasks/<id>/.tmp/gates.json --require-gate test-fast
```

These commands illustrate a batch-gate receipt; they do not require fast at every local landing.
For landing, select the affected checks actually owed. The helper can also verify batch evidence before push.

For a focused pre-commit check intended for landing reuse, use `record --staged` with an appropriate
gate name and test command.
Stage only task-owned paths first; the recorder never stages files or creates a commit. New staged
records use schema version 2: `commit` is the pre-test HEAD and `tree` is the tested index tree.
Existing version-1 committed records remain supported. A test that edits tracked files invalidates
its receipt. Logs cannot be imported retrospectively without evidence of the tested content.

Use the verify command as `task-land --verify`; it runs no recorded gate commands. Repeat
`--require-gate` for every required gate and `--input` during recording for external config/input files
whose identity matters. A passing `test-full` satisfies a required `test-fast`. Required execution or
numerical gates remain separate names. Recording another gate must preserve candidate, environment
and input identities; use a new evidence file after a meaningful change.

The helper rejects failed/skipped required gates, missing or altered logs, changed named inputs,
changed recorded environment and unclassified tree differences. It measures Python and lock-file
identity, not every installed binary, upstream service or dataset: name additional input/version
files with `--input` and invalidate evidence if unrecorded dependencies change. Hashing a catalog file
does not hash the data it points to. Select required gates from the live ladder before verification;
the helper does not infer blast radius or schedule coverage.

Only after reviewing each changed path may `--allow-path exact/path.md` permit an intervening
Markdown change. Do not allow test-read documentation without rechecking the affected coverage.
Code changes require fresh affected checks and the reviewed delta route below. No wildcard exclusions
are supported.

### Reviewed follow-up changes

This JSON protocol applies only when retaining formal gate receipts across behavior-changing
follow-ups. Ordinary task reports need no JSON review. Keep the verifier's existing identity and
coverage checks intact; its recorder supplies mechanical identity fields automatically.

Keep the earlier evidence file and logs intact. Record new checks in a separate evidence file on the
follow-up candidate (clean committed or fully staged), using the same named inputs. Commit staged
content before verification. For a staged base receipt, set the review's `base_commit` to the subsequent
commit whose tree matches the tested snapshot. Review the complete diff from the
earlier tested commit, including transitive consumers, deletions and both paths of renames. Repeated
follow-ups use that complete cumulative delta; do not chain unchecked attestations.

Write a JSON review in task scratch with exactly these fields (replace the Git identity placeholders):

```json
{
  "base_commit": "FULL_EARLIER_TESTED_COMMIT",
  "candidate_tree": "CURRENT_GIT_TREE",
  "paths": {
    "blueearth_cst/shared/snake_utils.py": "Wording only; no parsed output or control flow changed."
  },
  "retained_gates": {
    "test-full": "Earlier non-console coverage remains applicable; affected console checks reran.",
    "test-e2e": "Workflow commands, data, calculations and handoffs are unchanged."
  },
  "require_gates": ["console"]
}
```

Use actual changed paths; the console path above is illustrative. Every path needs its impact and
coverage rationale. Every retained gate needs a reason its remaining claims still hold. Gate names
in `require_gates` must pass in the new evidence. For example, record `console` with the affected
console test command, then verify:

```powershell
pixi run python dev/scripts/verify_landing.py verify --evidence dev/tasks/<id>/.tmp/landing/base.json --delta-evidence dev/tasks/<id>/.tmp/landing/followup.json --delta-review dev/tasks/<id>/.tmp/landing/review.json --require-gate test-full --require-gate test-e2e --require-gate console
```

The verifier checks exact path coverage, base ancestry, candidate tree, matching environment and input
identities, and required log integrity/results. It does not infer impact or judge the review's prose;
the gate owner must review the diff and choose discriminating checks. This is selective reuse, never
a claim that the full suite or E2E ran on the new candidate. Broad or uncertain impact requires fresh
broad gates. Release and milestone labels do not invalidate applicable evidence. Changed inputs/environment
require new evidence; this helper conservatively refuses delta reuse even when one claim might survive.

A zero exit records command success, not scientific acceptance or proof that every required test ran.
Inspect the log for required skips and use `--skipped` when a required check did not execute; never
reuse that gate as passed. Keep visual acceptance and numerical difference classification in the task
record. Human attestation remains necessary for those claims.

## Choosing execution coverage

Trace changed behavior through its workflow callers and downstream consumers, including shared runtime
and dependency changes. Several affected handoffs do not alone require all-five execution: exercise
the affected paths when their boundary is established. Use all five when impact spans the pipeline,
the claim qualifies the whole toolbox, or cross-workflow uncertainty remains after inspection and
focused checks. State the affected paths and omitted workflows; reuse applicable execution evidence.

Console wording or formatting alone takes affected console tests and representative output
inspection. A change to exit status, exception handling, machine-readable output or control flow is
behavioral and takes affected consumer checks/execution. A file under `shared/` does not alone escalate.
Confined numerical changes need an independent oracle and affected consumers, not automatically all-five
execution or a whole-tree baseline comparison. Use a current baseline for whole-tree numerical claims.

For behavior confined to one workflow, search callers and output consumers, then exercise the affected
integration path. Admission, retry and rejection fixes need integration checks of those paths, not a
new hydrological simulation merely to reach the same guard. When workflow outputs can change, execute
the affected workflow and consumers with rapid inputs. State the coverage and omitted
workflows in the task evidence. Preserve owned wrappers: WF3 uses a WF3-only project configuration with
`run_workflows.py`; WF4 uses `simulate_system.py` or the all-workflow runner. For figure-only changes,
search shared plotting callers, render all affected figures, and publish PNG artifacts for inspection;
no baseline, full suite, or render-byte comparisons. Changes to runtime/config contracts or non-figure
consumers trigger broader gates. Changes to calculations still require numerical acceptance.

Escalate a confined change when focused verification exposes wider breakage. Preserve independent
numerical and visual acceptance whenever the change affects those claims.

## The end-to-end runs (`test-e2e-local`, `test-e2e`)

`pixi run test-e2e-local` runs the same five-workflow rapid case and deliverable
assertions as the live gate, but reads a sealed, source-only local bundle. Set
`BLUEEARTH_CST_E2E_BUNDLE` to its absolute directory before running. The test
copies only the four CMIP6 raw basin slices into a fresh project root and
points copies of the pinned HydroMT catalogs at the local source mirror.
Historical extracts, model, change factors, scenarios and simulations are all
regenerated. A missing or stale bundle fails the gate; a CMIP6 cache miss fails
before network access. The copied catalogs contain no P-drive fallback roots.

Build or refresh the bundle outside Git:

```powershell
pixi run python dev/scripts/stage_data.py --config dev/scripts/local_e2e_data.yml --dst <bundle>/hydromt
# Copy the four rapid-case raw CMIP6 slices produced by dev/scripts/stage_cmip6.py
# into <bundle>/cmip6/raw, retaining their original bytes and attributes.
pixi run python dev/scripts/local_e2e_bundle.py seal <bundle>
pixi run python dev/scripts/local_e2e_bundle.py verify <bundle>
$env:BLUEEARTH_CST_E2E_BUNDLE = '<bundle>'
pixi run test-e2e-local
```

The seal checks the 1999-12-31 CHIRPS boundary day, full 2000–2016 CHIRPS and ERA5 coverage,
exact pinned spatial source families, all four CMIP6 slices and their shared region identity. Its
inventory records every file's SHA-256 and the rapid config/catalog hashes;
any changed byte or added file requires refreshing the seal. The raw slices
must retain their acquisition digests, units and calendars. The first local
run still has to prove its freshly delineated polygon matches their digest;
it cannot silently substitute an old generated region.

For routine downstream changes, a passing sealed local gate supplies the
all-five execution claim. Add focused fetching tests when acquisition code
changes. Use the live `test-e2e` when the claim includes catalog availability,
remote addressing, acquisition windows, clipping or cache behavior, and when
qualifying a new bundle against live acquisition. A release or milestone can
reuse matching local execution evidence for downstream behavior; its label
alone does not trigger a fresh live read. Unit tests cannot establish remote
availability or detect every upstream format change. Record the bundle seal,
candidate, output root and live/local comparison when making a parity claim.

`pixi run test-e2e` is the automated all-five execution check:
through `scripts/run_workflows.py`, on the rapid config, in a fresh temporary
project root. It needs what CI cannot provide (the data mirror, juliaup Julia,
`weathergenr`, network for WF2), so it is a local gate. The execution-coverage section above
distinguishes broad changes
from confined workflow changes. Release and milestone coverage follows the changed behavior and the
qualification claim; neither label requires repeating applicable execution evidence.

It writes nowhere under `test_case/`, so it cannot disturb the baseline tree, and
nothing an earlier run left behind can make it pass. It skips by name when a
prerequisite is missing; a skip is no evidence, so read the reason (`-rs` is on).

## Which config to run

Historical constant-restoration comparisons and the later closure of mixed-provenance concerns are summarized
in 0001-restore-wflow-constant-parameters.md (private-checkout-only citation:
`../records/decision-history/0001-restore-wflow-constant-parameters.md`). They explain the comparator's
origin; they do not qualify the current baseline or authorize re-recording it.

Default to `project_config_rapid.yml`. Reach for `project_config_baseline.yml` only when the run's numbers are the point.

| Config | `project_dir` | Run it for |
|---|---|---|
| `project_config_rapid.yml` | `test_case/test_rapid` | Workflow execution, DAG checks and figures |
| `project_config_baseline.yml` | `test_case/test_local` | Baseline recording/checking; numerical claims |
| `project_config_wf2_fast.yml` | `test_case/test_dev` | Restricted WF2 iteration: 2 series, no `st_0` |

**Rapid's advantage is the GRID, not the years.** Both configs now simulate nine years (2046–2054) and generate the same ~44-year series — `t2608222155` moved the baseline off 17 years around 2078 on 2026-09-07. What is left is 10 members against 14: rapid's 2 × 2 grid plus `st_0` against baseline's 2 × 3.

This paragraph claimed "~2.6× less wflow time and ~1.7× less weather generation" until that item MEASURED it, and both halves were wrong. Per-member wflow cost is about `16 s + 0.59 s × simulated years`, so a 17-year member spent about 10 s simulating and about 16 s starting up — the window was worth ~18%, not 50%, and the measured whole-workflow gap was about 1.2×. Weather generation went the other way entirely: rapid's `3.10_generate_weather_realizations` took 45.8 s against baseline's 29.0 s. **The cost here is process startup**, not simulated years: the first member of a batch costs about 80 s more than the rest (Julia JIT plus the cold SBM path), which at three batches plus wf1's standalone run is roughly 320 s — a quarter of the run — spent starting Julia.

To change the generated-series length, change the END of `simulation_window` rather than its start: `compute_nr_years` anchors the series at the end of `climate.window`, so it spans that year → `simulation_window.end` (plus a two-year pad). (`C-67` folded the former `horizontime_climate` + `run_length`/2 into that one declared year.) Whatever you set must stay inside a `future_windows` entry in the analyze_projections file — the alignment is computed independently of both keys, so a missed edit misaligns silently.

Rapid is cheap, not narrow. It gets `st_0`, which is what the two class-C month indicators derive from — and since `C-69` every config does, so this is no longer a way a config can quietly give up 2 of 11 `q` indicators — and it keeps two CMIP6 models, since a one-model config never runs the ensemble reduction. A config that gives up coverage must say which, as `wf2_fast` does.

Record the baseline from `project_config_baseline.yml` and nothing else; never point `check_baseline.py` at the rapid tree.

`check` exits **0** (targets match), **1** (the tree differs) or **2** (*nothing
was compared* -- an empty scope, or a fixture it cannot resolve). Treat 2 as "no
evidence", never as a pass: `--workflow generate_scenarios` declares no
non-figure target, so it reaches 2 by construction.

**The baseline manifest is stale after the P-drive catalog and GlobCover switch.** Its last fresh-root
record (2026-10-04) used the prior local-data baseline. Regenerate `test_case/test_local` from the revised
baseline config and re-record `dev/baseline/manifest.json` before using `check_baseline.py check` as a
numerical gate. Do not mistake agreement with an old manifest for acceptance of current inputs.

The standing tree is the R12 successor, not a migration reference: predecessor/successor comparisons
also change the 2026-09-07 configuration, confounding layout and settings. R12 acceptance uses the
separately captured fresh P0 reference and signed GF-9 crosswalk under matched settings. The preserved
predecessor is at `blueearth_cst-artifacts/r12/predecessor-test_local-2026-09-15` outside this checkout.

`test_case/test_local` is untracked and independent in each worktree. Inspect its provenance before
reuse: older seeds may still contain predecessor output. Re-seed or regenerate each affected worktree;
repairing one does not repair the others. `project_dir` resolves from the run directory.

WF1 retains `run_default/output.csv`, which the baseline manifest needs. Older trees where it was
deleted require a historical simulation to restore it. Rounded `output_q.csv` is not a substitute.

## Read the CI run after you push

CI runs for `main` pushes and PRs, not backup branch pushes. Normal approved publishing goes to
`main`; use a PR when the change needs two-platform assurance before landing.

GitHub does not allow self-approval of a PR. If enabling merge-on-green, make both platform test
checks required first: auto-merge without required checks merges immediately.

Install the pre-push hook once per clone — cloning does not install it:

```bash
git config core.hooksPath .githooks     # runs the two ruff checks, ~2 s
```

Confirm the intended commit's platform results regardless of local success. The hook cannot see
platform-specific failures; inspect failing jobs and relevant skips as described under Tiering.

`gh` resolves to `upstream` (`Deltares/blueearth_cst`) in this clone, not `origin` (`tanerumit/blueearth_cst`, where CI runs), so a bare `gh run list` queries Deltares, exits 0 and prints nothing. Point it at origin once per clone:

```bash
gh repo set-default tanerumit/blueearth_cst   # writes remote.origin.gh-resolved
```

In a script that must not depend on local config, pass `--repo tanerumit/blueearth_cst` explicitly, or use the API:

```bash
# the latest run, whatever branch it was on
gh api "repos/tanerumit/blueearth_cst/actions/runs?per_page=1" \
  --jq '.workflow_runs[0] | "\(.head_sha[0:7]) \(.status) \(.conclusion)"'

# which STEP failed, per leg — separates a lint failure from a suite failure
gh api "repos/tanerumit/blueearth_cst/actions/runs/<id>/jobs" \
  --jq '.jobs[] | "\(.name) -> \(.conclusion): " +
        ([.steps[] | select(.conclusion=="failure") | .name] | join(", "))'
```

Do not filter by `head_sha=<short sha>`: the parameter needs the full 40-character value and silently matches nothing otherwise.

### Module-level test stubs and collection order

Tests that install heavy-dependency stubs with `sys.modules.setdefault` at
module import time can silently inherit a real module or an earlier test's
stub, depending on collection order. Prefer patching the imported source
module's bindings in a fixture and restore them per test. Do not stub `dask`
with `SimpleNamespace`: pandas may lazily import it and inspect `dask.__spec__`
during collection. Keep the real pixi-installed package, or mock the specific
call site. The live examples are documented in
`tests/test_extract_historical_climate.py` and `tests/test_prepare_climate_data_catalog.py`.
