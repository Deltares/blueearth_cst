# Fork Roadmap

Source of truth for the personal fork of `blueearth_cst`. Three phases:

**Phase 1 — Foundation (sealed 2026-05-08).** Replicated upstream,
formalized the pixi env, upgraded load-bearing libraries, and added
unit-test coverage. Four milestones, all tagged. Phase 1 dev artifacts
under `dev/phase-1/`.

**Phase 2 — Refactor (sealed 2026-07-23).** Major overhaul of the workflow
code, config contracts, and repo structure. Six milestones running from R1
(modularity contracts) through R6 (structural refactor), in deliberate
single-purpose steps. Phase 2 dev artifacts under `dev/r##/`.

**Phase 3 — Usability & flexibility (planned).** Driven by the user
expectations mapped 2026-07-23 at the R6 handoff: project/experiment
tracking, model flexibility, and performance. Milestones P3-1..P3-3;
dev artifacts under `dev/p3#/`. See § Phase 3 below.

```text
Phase 1 — Foundation (sealed)
  base/<start-point>
  └── milestone/01-replication              →  tag: m01-replication
        └── milestone/02-pixi-installation  →  tag: m02-pixi
              └── milestone/02b-library-upgrades  →  tag: m02b-upgrades
                    └── milestone/02c-tests             →  tag: m02c-tests

Phase 2 — Refactor (active, branches from m02c-tests)
                          └── milestone/r01-contracts        →  tag: r01-contracts
                                └── milestone/r02-naming        →  tag: r02-naming
                                      └── milestone/r03-model-builder  →  tag: r03-model-builder
                                            └── milestone/r04-projections →  tag: r04-projections
                                                  └── milestone/r05-experiment →  tag: r05-experiment
                                                        └── milestone/r06-refactor →  tag: r06-refactor
```

Phase 2 is **vertical-by-workflow**: R3, R4, R5 each take one Snakemake
workflow end-to-end (orchestration plus the analytical scripts it
calls). R6 then does the cross-cutting structural refactor on top.

---

## Branching and tagging conventions

| Branch type   | Pattern                       | Purpose                                                                  |
| ------------- | ----------------------------- | ------------------------------------------------------------------------ |
| Frozen base   | `base/<start-point>`          | Historical starting point of the fork (e.g. `base/v0.1.0-alpha`).        |
| Phase 1 milestone | `milestone/<NN>-<topic>`  | Sealed; pattern preserved on existing branches (`milestone/02c-tests`).  |
| Phase 2 milestone | `milestone/r<NN>-<topic>` | Active; example `milestone/r01-contracts`, `milestone/r03-model-builder`. |
| Experiment    | `exp/r<NN>-<topic>`           | Messy trial branch off a Phase 2 milestone.                              |
| Feature       | `feat/r<NN>-<topic>`          | Cleaner implementation off a Phase 2 milestone, intended to be merged in. |
| Pull request  | `pr/<NN>-<topic>`             | Clean branch prepared for upstream review.                               |

**Tags.** Phase 1 tags use `m##-<topic>` and stay frozen
(`m01-replication`, `m02-pixi`, `m02b-upgrades`, `m02c-tests`). Phase 2
tags use `r##-<topic>` (`r01-contracts`, `r02-naming`,
`r03-model-builder`, `r04-projections`, `r05-experiment`,
`r06-refactor`). Tags are permanent rollback points; milestone branches
stay alive after their tag for late patches or PR prep.

**Stacked, not parallel.** Each milestone branches from the previous
milestone's tip (not from `base/`). Phase 2 starts from the
`m02c-tests` tag. R1, R2 are pre-workflow contracts and conventions
that R3-R5 inherit; R6 is the cross-cutting structural refactor.

**Remotes.**
- `origin` — your fork (`github.com/tanerumit/blueearth_cst`).
- `upstream` — the original Deltares repo
  (`github.com/Deltares/blueearth_cst`), fetch-only.

The branch `upstream-deltares` (formerly `main`) freezes the upstream
Deltares state the fork tracked at renaming time; never commit to it.
`main` is the moving trunk and the GitHub default branch.

**PRs back to upstream** go from `pr/<NN>-<topic>` branches, not
directly from milestone branches. One PR per milestone is the default;
only stack PRs when maintainers explicitly agree to review them in
series.

---

## Phase 1 — Foundation (summary)

Sealed 2026-05-08. All artifacts under `dev/phase-1/`; baseline
manifest at `dev/baseline/manifest.json`. The detailed scope and exit
criteria for each Phase 1 milestone live in the corresponding sealed
commits and `dev/phase-1/<milestone>/` docs; this section is a
reference summary only.

### M01 — Replication baseline (sealed 2026-05-07; tag `m01-replication`)

Got all three Snakemake workflows running end-to-end on the test
config and recorded baseline output fingerprints. Established the
fingerprint format (per-variable summary stats for netCDF; normalized
SHA256 for CSV/YAML; size-only for PNG with ±10% tolerance). Built
`dev/scripts/check_baseline.py` with `record` / `check` subcommands.
Artifacts: `dev/phase-1/m01/setup.md`, `dev/phase-1/m01/warnings.md`,
`dev/baseline/manifest.json`.

### M02 — Pixi env + install (sealed 2026-05-07; tag `m02-pixi`)

Replaced the conda + ad-hoc R + Julia setup with a single declarative
`pixi.toml`. weathergenr handled separately via `pixi run install` due
to a Mingw-w64 byte-compile issue with conda r-base on Windows.
Wflow.jl + Julia 1.11.x via juliaup outside pixi (conda-forge has no
win-64 Julia build). Artifacts: `dev/phase-1/m02/decisions.md`,
`pixi.toml`, `pixi.lock`.

### M02b — Library upgrades (sealed 2026-05-07; tag `m02b-upgrades`)

Bumped four load-bearing libraries: hydromt 0.x → 1.3, hydromt_wflow
0.x → 1.0, Wflow.jl 0.7 → 1.0.2, plus lifted Python stack caps
(numpy 2.x, xarray latest, python 3.12). Re-baselined the manifest
under the "intentional drift, document deltas" policy. Artifacts:
`dev/phase-1/m02b/audit.md`, `dev/phase-1/m02b/baseline_diffs.md`,
`dev/phase-1/m02b/handoff.md`.

### M02c — Test coverage (sealed 2026-05-08; tag `m02c-tests`)

Added unit-test coverage for four small, stable `src/` modules
(`metrics_definition`, `setup_time_horizon`,
`prepare_climate_data_catalog`, `extract_historical_climate`) with two
strict xfails for documented bugs. Established the
`sys.modules.setdefault` mocking pattern that R3-R5 inherit. Suite
state: 45 passed, 4 xfailed. Artifacts:
`dev/phase-1/m02c/test-coverage-design.md`,
`dev/phase-1/m02c/test-coverage-plan.md`.

---

## Phase 2 — Refactor (COMPLETE 2026-07-23)

Goal of Phase 2: clean up workflow internals, scripts, and config
contracts so the pipeline is maintainable and extensible. Six
milestones; deliberate pace; each milestone has a single coherent
purpose. R1 and R2 establish contracts that R3-R5 inherit; R3-R5 do
the actual workflow cleanup; R6 is the cross-cutting structural
refactor.

### R1 — Modularity contracts (sealed 2026-07-18)

**Status.** Sealed 2026-07-18 — three top-level config sections in place;
all 3 Snakefiles + 4 `src/` scripts + conftest + all three integration
tests read sectioned config; config path via `workflow.configfiles[0]`;
migration guide for user-local configs at
`dev/r01/local-config-migration.md`. Per-workflow contract docs deferred
to R3/R4/R5 (2026-07-17 amendment). Suite: 51 passed, 3 skipped, 2 xfailed
(the pre-R01 47 plus 4 focused R01 reader/normalization tests). Scientific
invariance established **by construction** (value-preservation on every
migrated leaf + identity-preserving list/string normalization + green
suite + clean dry-runs), **not** by a manifest re-record: Task 5 found the
M2b `dev/baseline/manifest.json` stale (recorded from an untracked 3-model
config while the canonical uses 8; plus model-independent drift), so it is
left untouched and a clean rebuild is deferred. Full rationale + evidence:
`dev/r01/baseline_diffs.md`.

**Goal.** Establish per-workflow config contracts so workflows can be
added, disabled, or replaced in the future without touching others.
Phase 2's foundation: formalize ownership boundaries before R3-R5
each refactor a workflow. Otherwise each refactor has to decide on
the fly which keys belong to which workflow, and the decisions
accumulate inconsistently.

**Scope.** Reorganize the snake config into three top-level sections
(`project`, `shared`, `workflows.<name>`); each Snakefile reads only
its own section + shared. The contract-doc *format* is specified in
the R1 design doc (§4); the per-workflow docs themselves are deferred
to R3–R5 (see amendment note below). `enabled:` flag in each workflow
section as a forward-compat marker (documentary today; operational
when R6 adds module composition or a wrapper script).

> **Amended 2026-07-17.** The three per-workflow contract docs
> (`dev/workflows/<name>.md`) are moved out of R1: each is written as
> the opening act of the milestone that refactors that workflow
> (R3 → model_creation, R4 → climate_projections, R5 →
> climate_experiment). Rationale: a contract doc written when its
> workflow is freshly in focus is better-informed, and R1 shrinks to
> mostly mechanical config migration.

**Approach.** Distinguish *contracts* (cheap to formalize, last
forever) from *structure* (expensive to change, defer until needed).
R1 invests in contracts. Structure stays as-is — still 3 separate
Snakefiles, still flat `src/`, no Snakemake module composition or
plugin registry.

**Exit criteria.**
- Three top-level config sections in place with a checked-in template
  at `config/snake_config.template.yml`.
- All three Snakefiles read sectioned config; old flat reads removed.
- `src/` scripts that read config directly (`prepare_cst_parameters`,
  `prepare_weagen_config`, `get_change_climate_proj`,
  `get_change_climate_proj_summary`) migrated.
- Three migrated config files committed (`tests/`, canonical, Linux).
- All three workflows run end-to-end on the migrated canonical config
  (verified 2026-07-18 into `examples/test_local`).
- Scientific invariance established by construction (value-preservation
  on every migrated leaf + identity-preserving list/string normalization
  + green suite + clean dry-runs). The planned manifest re-record was
  **not** performed: Task 5 exposed that the M2b
  `dev/baseline/manifest.json` is stale (recorded from an untracked
  3-model config while the canonical uses 8; plus model-independent
  drift), so it is left untouched and a clean rebuild is deferred to a
  dedicated task. Full rationale + evidence: `dev/r01/baseline_diffs.md`.
- `pytest tests/`: 51 passed, 3 skipped, 2 xfailed (the pre-R01 47 plus
  4 focused R01 reader/normalization tests; no pre-existing test changes
  outcome).

**Out of scope.** Per-workflow contract docs (deferred to the opening
act of R3/R4/R5, per the 2026-07-17 amendment above); operational
`enabled:` skip behavior (R6); pydantic / jsonschema validation;
cross-workflow data path decoupling (R6); Linux/Docker config rewrites
(deferred per Linux replication parking lot).

**Risks / open questions.**
- A renamed key the Snakefile still reads under its old name → silent
  default → baseline drift. Mitigation: per-Snakefile commit
  boundaries with dry-run between commits; baseline manifest catches
  any output drift.
- `workflow.configfiles[0]` requires `--configfile` on the CLI.
  Verify each invocation path during implementation. (Side benefit:
  this also delivers part of R3's "configfile mechanism" sub-item
  early — R3's roadmap entry below reflects that.)

**Tag.** `r01-contracts`. Full design lives in
`dev/r01/modularity-contracts-design.md`.

### R2 — Naming conventions (sealed 2026-07-19)

**Status.** Sealed 2026-07-19 — `dev/conventions/naming.md` (187 lines,
< 250) authored and pointed to from `AGENTS.md`; the design was tightened
after independent GPT-5.6 and Fable reviews
(`dev/r02/naming-conventions-review-{gpt-20260718,fable-20260719}.md`).
Docs-only; suite unchanged (51/3/2); existing names grandfathered (zero
code diffs).

**Goal.** Single prescriptive style guide at `dev/conventions/naming.md`
for naming identifiers and files across the repo. Pure docs; no code
refactoring. R3+ apply the conventions when touching code; existing
names are grandfathered. R3-R5 add new identifiers along the way
(helper functions, fixtures, wildcards, config keys), and locking the
convention first prevents each milestone from re-deciding naming on
the fly.

**Scope.** `dev/conventions/naming.md` (< 250 lines, prescriptive
`MUST` / `SHOULD` / `MAY` voice) + a one-line pointer in `AGENTS.md`
(canonical; `CLAUDE.md` inherits via `@AGENTS.md`).
Covers: universal case (snake_case, lowercase acronyms, true
constants), per-language rules (Python PEP 8, R snake_case not
dot.case, Snakemake snake_case rules, YAML snake_case keys), path-
identifier suffix (`_path` canonical; `_fn`/`_fid`/`_file` deprecated),
Snakemake wildcard vocabulary, suffix vocabulary split between paths
(`_path`) and data objects (`_ds`/`_df`/`_gdf`/`_cfg`), domain
identifiers that DO NOT get normalized (Wflow / HydroMT / CMIP /
CSDMS / weathergenr / scientific variable names), file naming by file
class (Python/R = snake_case; `dev/*.md` = kebab-case; etc.), and a
"do not rename without migration note" list.

**Timing (added 2026-07-17).** R2 is pure docs and deliberately light —
it must not become a scheduling gate. It may be drafted in parallel
with R1's tail or as R3's opening act; the only hard requirement is
that `dev/conventions/naming.md` is committed and tagged
(`r02-naming`) before R3's first *code* commit, so R3–R5 mint new
identifiers against a locked convention.

**Approach.** Prescriptive but lenient: opinionated where the codebase
is currently mixed, lenient where external conventions take
precedence. Two framings: (1) local style vs upstream contract —
local style does not apply to identifiers governed by external
systems; (2) grandfathered today, applied tomorrow — R2 itself
produces zero code diffs.

**Exit criteria.**
- `dev/conventions/naming.md` exists, < 250 lines, prescriptive.
- `AGENTS.md` has a one-line pointer to the naming doc (canonical;
  `CLAUDE.md` inherits it via `@AGENTS.md` — not a CLAUDE.md-only edit).
- `pixi run pytest tests/` unchanged: 51 passed, 3 skipped, 2 xfailed.
- R2 changeset is documentation-only (no `Snakefile_*`, `src/`, `tests/`,
  config YAML, lockfile, manifest, or generated output in the diff).

**Out of scope.** Branch / commit / PR conventions (in this roadmap);
output path conventions (in R1 contract docs); refactoring existing
names to conform (R3+); linter / CI enforcement; per-language style
guides (function lengths, comment conventions).

**Risks / open questions.**
- Style guide rot if not enforced. Mitigation: R3-R5 reference
  `dev/conventions/naming.md` in commit messages when adding new
  identifiers; future linter is a possible followup.
- Section 6 (domain identifiers) and section 4 (wildcard vocabulary)
  will grow as new tools / workflows enter scope. Doc is living.

**Tag.** `r02-naming`. Full design lives in
`dev/r02/naming-conventions-design.md`.

### R3 — Workflow 1: model builder (sealed 2026-07-19)

**Status.** Sealed 2026-07-19 — `Snakefile_model_creation` + its scripts
cleaned up: shared `get_config` and `tee_to_log` in `src/snake_utils.py`
(the cross-cutting patterns R4/R5 inherit), per-rule `log:`/`benchmark:` on
every non-trivial rule, deprecated path labels renamed, `setup_gauges` hardened
(raises on unknown `wflow_outvars`), the waterbodies rule encapsulated with a
removal trigger + structured sentinel, and a new `outlet_index.csv` rule-all
output settling the outlet-naming contract. R2 naming applied to workflow-1
identifiers; the deferred R1 contract doc `dev/workflows/model_creation.md`
written. **Behavior-preserving**, verified by a full `--forceall` WF1 rebuild:
`check_baseline` 14/14, all per-rule logs written, `outlet_index.csv` and the
structured sentinel correct. Suite 73 passed, 3 skipped, 2 xfailed. Constant-
parameter restoration split out to task `t260719a` (a scientific decision +
baseline move); the workflow-3 `CyclicGraphException` `test_cli` ratchet is
retained for R5. Full design, external GPT-5.6 review, and integration-
verification record in `dev/r03/`. Merged to `main` 2026-07-19.

**Goal.** Clean up `Snakefile_model_creation` and the scripts it
calls — orchestration *and* analytical code. Establish the
cross-cutting Snakefile patterns that R4 and R5 inherit.

**Cross-cutting deliverables (done once here, reused by R4 and R5).**
- Collapse the duplicated `get_config(config, key, default, optional)`
  helper from all three Snakefiles into one shared module at
  `src/snake_utils.py`. Update all three Snakefiles to import from it.
  Behavior of R4/R5's Snakefiles unchanged; only the helper sourcing
  moves.
- ~~Replace the `--configfile` `sys.argv` re-parsing trick in all
  three Snakefiles with `workflow.configfiles[0]`.~~ **Done by R1.**

**Workflow-1 deliverables.**
- Opening act, before code changes: write
  `dev/workflows/model_creation.md` (contract doc deferred from R1;
  format in `dev/r01/modularity-contracts-design.md` §4).
- Any load-bearing `ruleorder:` in `Snakefile_model_creation` either
  tightened (preferred) or commented in-place with the reason.
- Per-rule `log:` and `benchmark:` directives on every non-trivial
  rule in this Snakefile.
- Resolve or properly encapsulate the "temporary hydromt fix" in
  `src/setup_reservoirs_lakes_glaciers.py` — either upstream the fix
  or isolate it with a comment that names the upstream issue and a
  removal trigger.
- Review `src/setup_gauges_and_outputs.py` for correctness,
  vectorization, and units handling.
- Add unit tests under `tests/` for the Python helpers in this
  workflow's scope.

**Exit criteria.**
- `pytest tests/test_cli.py` (dry-run sanity check) still passes for
  all three Snakefiles.
- The model-creation workflow runs end-to-end and matches its slice
  of the M1 baseline — preserved, or intentionally updated with a
  documented diff in `dev/r03/baseline_diffs.md`.
- New unit tests added and passing.
- `dev/workflows/model_creation.md` contract doc committed.

**Out of scope.**
- `Snakefile_climate_projections` content changes (R4) — except the
  shared helper import.
- `Snakefile_climate_experiment` content changes (R5) — same caveat.
- Repo-wide directory restructuring (R6).

**Tag.** `r03-model-builder`.

### R4 — Workflow 2: climate projections (sealed 2026-07-20)

**Status.** Sealed 2026-07-20 — `Snakefile_climate_projections` + its four
`src/` scripts cleaned up, inheriting the R3 patterns. Design accepted via a
`design-review-loop` run (3-lens internal panel + 3 external GPT rounds +
round-cap arbitration; 24/24 findings closed) at `dev/r04/`. Landed in 11
commits (`1a8809e`..seal): contract doc `dev/workflows/climate_projections.md`;
the load-bearing `ruleorder:` resolved as evidence-backed stale-insurance
(dry-run refuted the `AGENTS.md` "load-bearing" claim — `AGENTS.md` corrected);
per-rule `log:`/`benchmark:` + `tee_to_log` on all five non-trivial rules
(guards added to `get_stats`/`get_change`/`plot_proj_timeseries` first, repo-5
ordering); `_fid`/`_nc`→`_path` label renames; units docs + bare-`except:`→
`except Exception:` narrowing; a `check_baseline.py --workflow` scope filter
(commit 2b); and the §7 audit-evidence test suite. **Behavior-preserving**:
the workflow-2 end-to-end re-run matched its manifest slice on all data targets
(the `.nc` summary at tolerance 0, all PNGs, wf1 targets); the 2 full-precision
CSV byte-diffs are serialization non-determinism, not a value change
(`dev/r04/baseline_diffs.md`) — **no manifest re-record**. Suite 102 passed, 3
skipped, 6 xfailed.

**Audited, defects deferred (not "audited clean").** The chain audit
(`dev/r04/chain-audit.md`) confirmed the change-factor formula, calendars C3,
and hydro-year windows, and surfaced four deferred defects, each with owner +
activation condition: **D-CAL** — `get_change_annual_clim_proj` raises
`TypeError` on cftime 360-day/noleap calendars (task `t260720c`, latent for the
current seed); **D-VAR/D-MEM** — silent variable/member drops, wired as
strict-xfail fail-loud norms (task `t260720d`); **D-ATTRS** — the M2b CF-metadata
loss, probe-localized to the hydromt catalog read, a dependency op (task
`t260720e`). The strict-xfail wiring is the tripwire: fixing any code defect
flips its test xfail→xpass and fails the suite until the owning task removes the
marker. Full design, reviews, audit, and probe in `dev/r04/`.

**Goal.** Clean up `Snakefile_climate_projections` and the scripts it
calls. Inherit the patterns established in R3 (shared helper,
configfile mechanism, log/benchmark conventions).

**Deliverables.**
- Opening act, before code changes: write
  `dev/workflows/climate_projections.md` (contract doc deferred from
  R1; format in `dev/r01/modularity-contracts-design.md` §4).
- The load-bearing `ruleorder:` directive in
  `Snakefile_climate_projections` either tightened or commented
  in-place with the reason.
- Per-rule `log:` and `benchmark:` on every non-trivial rule in this
  Snakefile.
- Review `src/get_stats_climate_proj.py` for correctness,
  vectorization, and units handling.
- Audit the `monthly_stats_hist` → `monthly_stats_fut` →
  `monthly_change` chain end-to-end for unit consistency, calendar
  handling, and missing-data behavior.
- Add unit tests for the Python helpers in this workflow's scope.

**Exit criteria.**
- `pytest tests/test_cli.py` still passes.
- The projections workflow runs end-to-end and matches its slice of
  the M1 baseline — preserved, or intentionally updated with a
  documented diff in `dev/r04/baseline_diffs.md`.
- New unit tests added and passing.
- `dev/workflows/climate_projections.md` contract doc committed.

**Out of scope.**
- Workflow-1 or workflow-3 changes (other than shared helper
  inheritance).
- Repo-wide directory restructuring (R6).

**Tag.** `r04-projections`.

### R5 — Workflow 3: climate experiment (sealed 2026-07-20)

**Status.** Sealed 2026-07-20 — `Snakefile_climate_experiment` + its `src/`
scripts + the R weathergen layer (`src/weathergen/generate_weather.R`,
`impose_climate_change.R`) cleaned up, inheriting the R3/R4 patterns. Design
accepted via a `design-review-loop` run (3-lens internal panel + 2 external GPT
rounds + round-cap arbitration; 21/21 findings closed) at `dev/r05/`. Landed in
12 commits (no commit 4; `8b356f3`..seal): contract doc
`dev/workflows/climate_experiment.md`; `stress_test_grid` helper extracted to
`snake_utils.py` (strict `step_num`, removing the Snakefile's silent default-1 —
output-neutral hardening); `prepare_weagen_config.py` config assembly extracted
into importable functions above a guard; the **CyclicGraphException** resolved
by a rule-local `wildcard_constraints: st_num=[1-9][0-9]*` on
`generate_climate_stress_test` + the `test_cli` ratchet flipped to a clean-DAG
assertion on a staged-region config; the **`st_num2 → st_num` fold** (5b, landed
— verified no re-introduced ambiguity via a `run_historical: true` dry-run);
`shared.historical_window` wired into `extract_climate_grid`; per-rule
`log:`/`benchmark:` + `tee_to_log` on the 7 `script:` rules and
`> {log} 2>&1` (exit-preserving, NOT `| tee`) on the 3 shell rules; R-layer
arg-binding + arity checks + progress `message()`s; `_fid → _path` label
renames; and the wf3 Python-helper unit tests.

**Behavior-preserving.** The end-to-end milestone gate
(`check_baseline check --workflow model_creation --workflow climate_experiment`,
after a full fresh wf3 regen) matched **7/7 targets** (4 wf1 + 3 wf3) — **no
manifest re-record**. The two computational-path commits are each confirmed
output-equivalent by a dedicated ext1-3 characterization on the exact artifact
they touch:

- **Commit 6** (`historical_window` wiring): a `--forcerun extract_climate_grid`
  on the commit-6 code — its **first runtime execution** in R5, which proves the
  new `sm.params.starttime`/`.endtime` reach the keyword-only args — produced an
  `extract_historical.nc` **identical** to the pre-commit-6 snapshot. Expected:
  the seed window is byte-identical to the prior hardcoded strings, so the same
  hydromt extraction runs on identical inputs.
- **Commit 9** (R-layer cleanup): a **fail-closed** (ext2-1) characterization — a
  seeded control-vs-control pair of both the realization (`rlz_1_cst_0.nc`) and a
  perturbed netCDF (`rlz_1_cst_1.nc`) is **bit-identical** (determinism holds on
  `weathergenr` seed 123), and each before-vs-after comparison is likewise
  identical.

The R4-inherited CSV-serialization non-determinism open question is **resolved
negatively for wf3**: the seeded `Qstats.csv`/`basin.csv` reproduced the manifest
bit-for-bit across a full fresh regen, so the fragility did not recur here.
Suite: 120 passed, 3 skipped, 7 xfailed.

**Deferred defects (split, not fixed in R5) — each with owner + activation.**
- **`t260720a`** — `precip_variance` max-reads-min bug
  (`prepare_cst_parameters.py` line 42 reads `["min"]` into the max variable).
  Latent on the seed (`variance.min == variance.max == 1.0`); moves output on any
  config with `variance.max ≠ variance.min`. Owner `cst-architect` (route to
  `python-engineer` for the one-token fix + baseline re-record). Flagged by a
  `xfail(strict=True)` characterization test that xpasses when the fix lands.
- **weathergenr `spatial_ref` propagation** (`dev/followups.md` § R5) — the
  in-repo `generate_weather.R` workaround block STAYS (load-bearing) with a
  tightened removal-condition comment; the real fix is upstream in
  `tanerumit/weathergenr` `write_netcdf`. Upstream weathergenr task.
- **weathergenr wavelet `>= 16` cryptic error** (`dev/followups.md` § R5) —
  entirely inside the weathergenr package (`wavelet_cwt.R`); upstream task.
- **wf1 `| tee {log}` exit-masking-on-failure** (`dev/followups.md`,
  cross-cutting) — wf1's three shell rules run correctly on success (the R5 gate's
  wf1 leg passes) but mask the exit code on failure (cmd.exe has no
  `set -euo pipefail` prefix). Latent robustness item, NOT an R5 blocker; migrate
  wf1 to `> {log} 2>&1` or a portable tee wrapper. Owner `cst-architect`.

**R testthat coverage — DECISION: NO (locked at R5 start, G1-ratified).** The two
R scripts are thin `weathergenr` adapters (scientific logic is upstream); the
repo has no R test harness, and standing one up is R6-territory infra. The R
layer is gated end-to-end by the milestone baseline run + the `test_cli` dry-run,
with the §5a arity checks as the R-layer's correctness net. Full design, reviews,
and dispositions in `dev/r05/`.

**Goal.** Clean up `Snakefile_climate_experiment` and the scripts it
calls — including the R weathergen layer. Inherit the patterns from
R3.

**Deliverables.**
- Opening act, before code changes: write
  `dev/workflows/climate_experiment.md` (contract doc deferred from
  R1; format in `dev/r01/modularity-contracts-design.md` §4).
- Per-rule `log:` and `benchmark:` on every non-trivial rule in this
  Snakefile.
- The R weathergen pipeline (`src/weathergen/*.R`): cleaner argument
  parsing, fewer positional args, consistent logging. Migration to
  the current weathergenr API is already done pre-M1; revisit any
  drift here.
- Stress-test grid construction (`ST_NUM = (temp.step_num + 1) *
  (precip.step_num + 1)`) extracted from Snakefile expressions into
  a single tested Python helper.
- Review `src/weathergen/impose_climate_change.R` and the
  downscaling rules.
- Add unit tests for Python helpers in this workflow. R testthat
  coverage is a separate decision, locked at start of R5.

**Exit criteria.**
- `pytest tests/test_cli.py` still passes.
- The experiment workflow runs end-to-end and matches its slice of
  the M1 baseline — preserved, or intentionally updated with a
  documented diff in `dev/r05/baseline_diffs.md`.
- New unit tests added and passing.
- `dev/workflows/climate_experiment.md` contract doc committed.

**Out of scope.**
- Workflow-1 or workflow-2 changes (other than shared helper
  inheritance).
- Repo-wide directory restructuring (R6).

**Tag.** `r05-experiment`.

### R6 — Structural refactor (sealed 2026-07-23)

**Status.** Sealed 2026-07-23 — all 7 lock-list items landed in 8 `r06:`
commits (`368b30e`..`285e74c`, merged `024326a`): `src/` → `blueearth_cst/`
package (per-stage submodules + `shared/`); `config/` three-bin split
(`workflows/` / `catalogs/` / `templates/`); runners → `scripts/`; the
`enabled:`-aware `scripts/run_workflows.py` wrapper (pinned contract (a)–(g),
23 contract/skip tests); `dev/` vs `docs/` boundary codified; `MIGRATION.md`
(51 renames, git-mv-audited complete). Design accepted 2026-07-22 via the
`r06-structural-refactor` design-review-loop (`dev/r06/`). Implementation run
as a three-phase Opus handoff with Fable gate reviews (Gate 1 after the atomic
move, Gate 2 pre-merge). **Behavior-preserving, verified run-relative** (no
manifest re-record, `check_baseline.py` untouched): full e2e via the wrapper
green (14/23/57 steps); baseline vs the pre-R6 scratch manifest clean modulo
the three adjudicated copied-config snapshot rows (normalize-then-compare) and
two **pre-existing** non-deterministic CSV column orderings (unsorted set
intersection, `PYTHONHASHSEED`-dependent — demonstrated R6-independent, values
identical by label; see `dev/followups.md`); full-tree semantic diff
(`dev/scripts/semantic_tree_diff.py`, element-wise `.nc`) clean on all 96
substantive files. Notable en-route corrections: four design-inventory blind
spots (extensionless Snakefiles, two-line `script:` form, `data_sources_climate`
as a fourth catalog key, `run_logged` count) plus a post-Gate-1 fix (`f4be2f6`)
for three bare sibling imports only reachable through Snakemake's `script:`
runtime path — caught exactly by the design's execution-smoke stance. Suite:
230 passed / 3 skipped / 1 xfailed (pre-R6 parity + 36 new). Q6: no shim.
Q8: moot. Final suite green; tag `r06-refactor`.

**Goal.** Reorganize the repository so source code, configuration,
data catalogs, generated outputs, and documentation are cleanly
separated and discoverable. R3/R4/R5 already cleaned up *within* each
workflow; R6 sets the cross-cutting layout. R6 also operationalizes
the `enabled:` flag from R1 — workflows can be skipped from a single
config rather than by user discipline.

**Concrete pain points to address (lock list at start of R6).**
1. `src/` is flat — split into a package (`blueearth_cst/`) with
   submodules per workflow stage (model, projections, experiment,
   weathergen).
2. `config/` mixes canonical example configs with local / test
   variants and data catalogs. Split into `config/workflows/`,
   `config/catalogs/`, and keep `*_local.yml` patterns gitignored.
3. `dev/` and `docs/` boundaries — confirm conventions. `dev/` =
   planning, audits, and dev helpers (`dev/scripts/`); `docs/` =
   user-facing reference. Decide whether dev helpers stay under
   `dev/scripts/` or whether a top-level `scripts/` is introduced
   for production runners.
4. Data catalogs: OS-specific variants already collapsed in deferred
   Linux work, but the directory layout under `config/catalogs/`
   should be settled here.
5. Output layout under `project_dir/` already mostly clean — leave
   alone unless a concrete pain point emerges.
6. Remaining top-level runners (`run_snake_test.cmd`,
   `run_snake_docker.sh`) folded into `dev/scripts/` (consistent
   with the pre-M1 move of `open_shell.bat`) or split into a new
   top-level `scripts/` if you decide production runners deserve a
   separate home.
7. Operationalize `workflows.<name>.enabled` from R1 — either via
   Snakemake `module:` composition (one master Snakefile that
   conditionally includes per-workflow modules) or a wrapper script
   that orchestrates the three Snakefiles based on the flag.

**Exit criteria.**
- New layout documented in an updated CLAUDE.md and README.
- All three workflows still run and match the R5 baseline.
- `pytest tests/` passes.
- A `MIGRATION.md` (or section in the changelog) maps every moved
  file from old → new path so downstream forks can rebase.
- Setting `workflows.<name>.enabled: false` skips that workflow's
  outputs in a clean way.

**Out of scope.**
- Any further behavioral change beyond what `enabled` requires.

**Tag.** `r06-refactor`.

---

## Phase 3 — Usability & flexibility (COMPLETE 2026-07-25)

Sequenced so each milestone eases the next: the experiment tree settles
where per-model artifacts live (P3-2), and both precede performance work
(P3-3) so profiling targets the final structure.

**All four milestones sealed** (P3-1, P3-2a, P3-2b, P3-3). **P3-3 was the last
planned milestone in this roadmap**, so the planned programme — Phase 1
foundation, Phase 2 refactor, Phase 3 usability/flexibility — is now complete.
Nothing further is scheduled. The open question is whether to close the roadmap
or open a Phase 4; the candidate pool is recorded across `dev/followups.md`,
the "Minor open items" section below (CI, R testthat, a naming linter), the
"Deferred: Linux replication" section below, and the deferred items named in the
P3-2a/P3-2b/P3-3 designs (OQ-3 store, OQ-8 zone source, the 4th Snakefile entry
point, P3-2c PoC seam swap, the in-pipeline validator guard lift). No item in
that pool has been scoped or committed to.

### P3-1 — Project/experiment structure (sealed 2026-07-24)

**Goal.** One `project_dir` = one basin project holding multiple
non-colliding, self-describing stress-test experiments under
`experiments/<name>/`, completing the half-built `experiment_name`
mechanism. Experiments vary wf3 stress-test settings + the climate
window/source; the built model (wf1) and projections overlay (wf2) are
project-level and shared. `climate_historical/` becomes a project-level
per-dataset store referenced (not copied) by experiments. One full config
per experiment + a wf3 startup drift guard (project-level sections must
match the project snapshot; fail loud). Baseline handled as a documented
value-identical re-record; current output layout is not an external
contract on this fork.

**Design ACCEPTED 2026-07-23** via the `p31-experiment-structure`
design-review-loop (3-lens internal panel + 2 external GPT rounds +
round-cap arbitration; 29/29 findings closed; key mechanisms probe-verified
against pinned Snakemake — params rerun-trigger, ancient() input-set
trigger, key-level guard artifact for store reuse). Accepted design:
`dev/p31/experiment-structure-design.md`; audit trail:
`dev/p31/experiment-structure-design-review-record.md`; scoping intake
landed beside them. **Sealed 2026-07-24**: 8 `p31:` commits merged to
`main` (`1a8cca9`, --no-ff) after both human gates; value-identical wf3
re-record with semantic diff clean (evidence `dev/p31/baseline_diffs.md`
+ `migration_experiment-structure.md`); branch `milestone/p31-experiments`
+ tag at the tip; pushed.

**Cut (YAGNI):** registry, CLI listing, cross-experiment comparison,
layered configs. **Deferred:** `realization_*`/`stress_test` file-format
efficiency redesign (user-parked 2026-07-23; candidate P3-3 input).

**Tag.** `p31-experiments`.

### P3-2a — Model-independent climate analysis (sealed 2026-07-24)

First half of the former P3-2, split at scoping (the two halves touch
different code and carry different risk classes). Absorbs the R6-deferred
functional decomposition of climate analysis
(`dev/r06/structural-refactor-design.md` §8; `modularization` direction).
**Confirmed scope** (`dev/p32a/climate-analysis-intake.md`, the
authoritative record): full re-source + lift — a
`blueearth_cst/climate_analysis/` subpackage with strictly
model-independent signatures (region + catalog + window in), the wf1
subcatchment climate plots re-sourced from raw gridded climate (unwinding
the ADR-0002 `mod.forcing.data` coupling — the milestone's single
sanctioned value change, accepted via visual QA + characterized diff),
wf2/wf3 rewired mechanically. Subpackage now, standalone entry point
deferred (no 4th Snakefile; platform surface unchanged).
**Design ACCEPTED 2026-07-24** via design-review-loop run
`p32a-climate-analysis` (internal panel + 2 external GPT rounds + user
arbitration at the round cap): `dev/p32a/climate-analysis-design.md`, with
the consolidated review record and run observations beside it.
**Sealed 2026-07-24**: 6 `p32a:` commits (subpackage+shims → wf3 rewire →
wf1 extraction+parity → plot re-source → ladder QA → shim deletion) off
the task brief (`dev/p32a/climate-analysis-task-brief.md`), user-signed
milestone gate. Evidence: `dev/p32a/baseline_diffs.md` (ladder clean —
era5 `A2−A0` ≈ 0, precip null-check exact, G within tolerance, bbox-swap
closure allclose; wf3 semantic diff 101/0/0/0; manifested slice held; the
`clim_*` plots are unmanifested — knowing divergence from intake decision
4 accepted at the gate) + `migration_climate-analysis.md`. chirps plot
acceptance stays blocked pending the ext2-2 defer-and-pin tolerance run
on the first chirps basin.

**Tag.** `p32a-climate-analysis`.

### P3-2b — Model-swap interchange contracts (sealed 2026-07-24)

Second half of the former P3-2: pins the interchange contracts (netCDF
handoffs, forcing/state shapes) as explicit interfaces so an alternative
weather generator or hydrological model becomes a bounded substitution.
**Confirmed scope** (`dev/p32b/climate-interchange-intake.md`, the
authoritative record): BOTH substitution seams (weather generator;
hydrological model), **contracts-only** — per-seam contract docs +
hand-rolled validators-as-tests against fixture artifacts; zero behavior
change (no pipeline edits, nothing re-recorded); a bounded-substitution
walkthrough per seam; no PoC swap (future P3-2c candidate), no in-pipeline
enforcement, none of the P3-2a-deferred structural items (OQ-3 store, OQ-8
zone source, entry point).
**Design ACCEPTED 2026-07-24** via design-review-loop run
`p32b-interchange-contracts` (full variant: internal panel 0 blocking /
7 major / 9 minor → external GPT r1 revise (2 major: relational validators;
all-skip-green) → Fable-escalated revision → external GPT r2 **approve,
zero findings**; converged inside the cap, no arbitration; ledger 18/18
accepted): `dev/p32b/interchange-contracts-design.md`, with the
consolidated review record beside it.
**Sealed 2026-07-24**: 4 `p32b:` implementation commits (two seam docs →
validators+tests (§8 commits 3+4 merged as sanctioned) → contracts README)
off the task brief (`dev/p32b/interchange-contracts-task-brief.md`),
user-signed milestone gate. Deliverables: `dev/contracts/{README,
weather-generator-seam, hydrological-model-seam}.md`,
`blueearth_cst/shared/interchange_contracts.py` (15 validators),
`tests/test_interchange_contracts.py` (30 synthetic + 15 integration).
Evidence: suite 357/6/1 purely additive over 304/3/1; `pytest -rs` split
matches the §5.5 counting axis (12 green + 3 documented temp skips);
milestone diff = 5 new files, 2279 insertions, zero pipeline edits. chirps
fixture-verification remains a documented future step.
**`--notemp` capture DONE 2026-07-25** (the deferred OQ-4 lift): all three
`temp()` validators ran against real artifacts. WG-6 and HM-6b passed
unchanged; **WG-4 FAILED and the contract was wrong, not the pipeline** — it
required `crs=4326`/`category=meteo` as netCDF global attrs, but the real
generator NC carries **empty** global attrs (CRS travels CF-style in
`spatial_ref`'s `crs_wkt`, and crs/category are catalog metadata that
`validate_wg5` already pins). Corrected to asserted-if-present, +4 synthetic
tests, seam doc updated with the measured procedure (19 jobs / 247.7 s) and its
`--delete-temp-output` restore. Fixture verified byte-identical after restore.

**Tag.** `p32b-interchange-contracts`.

### P3-3 — Performance passes (scoped 2026-07-24)

Profiling-driven efficiency work targeting the wf3 stress-test sweep, with
baseline discipline à la R3–R5. **Confirmed scope**
(`dev/p33/performance-passes-intake.md`, the authoritative record): wf3
sweep throughput only, value-identical — benchmark evidence puts ~84% of
wall time in the `RLZ_NUM × ST_NUM` wflow runs, with per-invocation Julia
startup/JIT the likeliest our-side lever; measure-first (a profiling probe
decomposes startup vs simulation) with **structural latitude** (the
`run_wflow` execution may be restructured, e.g. batched per Julia session;
DAG shape may change, outputs may not); probe-set expectations, no a-priori
speedup floor; milestone gate = user sign-off on measured before/after +
value-identity evidence. The parked realization/stress-test file-format
redesign stays parked (I/O non-dominant per the evidence); wf1/wf2 and
memory-headroom work are out.
**Design ACCEPTED 2026-07-24** via design-review-loop run `p33-performance`
(full variant, probe-grounded draft: measured F≈135 s per-process fixed vs
S≈208 s cold sim; internal panel 1 blocking / 7 major / 10 minor → external
GPT r1 revise (makespan model, resource contract, go/no-go criteria) →
external GPT r2 reject (callable-output construct inexpressible,
probe-confirmed) → round-cap user arbitration → stage-6a fix
(probe-verified loop-generated batch rules); ledger 22/22 accepted):
`dev/p33/performance-passes-design.md`, with the consolidated review
record and landed probe evidence beside it.
**Sealed 2026-07-25**: user-signed milestone gate (floor-free by intake
decision 3 — sign-off on the measured before/after + GN outcome +
value-identity evidence, no threshold imposed). 6 `p33:` commits off the task
brief (`dev/p33/performance-passes-task-brief.md`): baseline/decomposition +
LPT estimator (`6402db6`), the batching lever (`92f9080`), roadmap status
(`0c797db`), upstream-parity measurement + reasoned-claim labelling
(`fac689e`), the batch-size disk clamp (`3392587`), followups SHA
(`293ff4e`).
**Headline: 619.9 s → 400.2 s (−35.4 %)** on the seed fixture wf3 sweep, at a
frozen `(-c 3, --threads 4)` budget with `B` the only moved knob, `--forceall`
scope. **No output value changed** — `semantic_tree_diff` per-process vs
batched on identical inputs is CLEAN (102 files, 0 failed, tolerance 0),
`check_baseline` OK, P3-2b validators 53 passed, suite 397/6/1. Deliverables:
rule 3.10 as loop-generated `run_wflow_batch_<b>` rules + the `batch_size`
knob, `blueearth_cst/experiment/run_wflow_batch.jl`,
`dev/scripts/estimate_batch_makespan.py`, plus the `batch_size_max` disk clamp
(the `ceil(K / -c N)` default implemented only §6.1's parallelism ceiling and
scaled `B` — hence peak temp disk — up with sweep size; invisible on the
fixture, where `min(ceil(12/3), 8) = 4`). GN-1..4 all pass → batching stands,
the PackageCompiler sysimage stays dormant (no dependency ask triggered), and
the corrected cost terms independently weaken it (−19 % vs batching's −52 %).
C5 failure isolation is DEGRADED by design (blast radius `B`), measured to be
exactly the documented cost. Evidence: `dev/p33/batching-results.md`.
**Caveat carried forward:** the commit-1 baseline (2242.9 s) was contaminated
by the concurrent `stage_data` workstream and is superseded in place — see the
supersession block in `dev/p33/performance-baseline.md`. Any future performance
measurement in this repo must record `cpu_time` alongside wall and confirm no
sibling agent session is active.
Post-P3-3 followups (genuinely disk-aware batch-size cap; the
`--keep-incomplete` ↔ `--keep-going` probe that could narrow the C5 blast
radius) in `dev/followups.md` § Post-P3-3.

**Tag.** `p33-performance`.

**P3-3 was the last planned Phase 1–3 milestone.** With it sealed, the planned
Phase 1–3 programme is complete. The scoping conversation happened on
2026-07-26 (owner's post-R6 assessment) and **opened Phase 4** — see below.
The remaining Phase 3 backlog is unchanged and unclaimed (P3-2c PoC seam swap,
in-pipeline validator guard lift, OQ-3 store, OQ-8 zone source, the 4th
Snakefile entry point, the chirps ext2-2 ladder + gate-8 smoke — still
data-blocked, plus the two Post-P3-3 items above). The `--notemp` capture is
**done** (2026-07-25, see the P3-2b entry) and two Post-R6 items were closed
the same day (`semantic_tree_diff` exclusions — stale; dead
`tests/wflow_build_model.yml` — removed).

---

## Phase 4 — Layout consolidation (design accepted, awaiting implementation)

Opened 2026-07-26 out of the owner's post-R6 assessment. Phase 2 (R6) settled
the *repository* layout and Phase 3 (P3-1) settled the *experiment* layout;
neither could see the residue the other left, and R6's own lock list deferred
the artifact tree explicitly ("5. Output layout under `project_dir/` already
mostly clean — leave alone unless a concrete pain point emerges"). Pain points
have now emerged, and they span both halves.

### R7 — Project layout (design ACCEPTED 2026-07-28)

**Status.** Design **ACCEPTED**, not yet built:
`dev/r07/project-layout-design.md`, approved by the owner at gate G2 of a
`design-review-loop` run on 2026-07-28. Drafted interactively with the owner
across the 2026-07-26 review (a 16-ruling question log), then put through the
loop: a three-lens internal panel and two external cross-vendor rounds, **44
findings, all dispositioned, none rejected**, across four versions. The external
round cap was reached with round 2 unconverged, so the owner arbitrated the three
surviving findings — meaning the final version's changes carry no external
verdict, which the design states on its face. Full audit trail:
`dev/r07/project-layout-design-review-record.md`; approved framing:
`dev/r07/project-layout-intake.md`; `naming.md` §7 path map:
`dev/r07/migration_project-layout.md`. Provenance of the findings:
`dev/reviews/2026-07-25_post-r6-assessment.md` (O-01 … O-24), which carries a
routing note for which observations R7 owns.

**Goal.** One coherent layout across both halves, governed by stated principles
rather than accretion: the **toolbox** holds source, config and templates — no
basin data, no run artifacts; the **artifacts** under `project_dir` are
organised by producer and by engine, so a reader can tell what made a file from
where it sits, and engine-shaped artifacts are **separable** from generic ones,
so an engine's subtree can be relocated, rebuilt, or replaced without moving
generic climate data. *(Narrowed from extensibility at review — the delivered
tree does not support adding a second hydrology engine without a placement rule,
and writing that rule would decide the engine-naming question parked at G1.
Recorded as a stated limitation, ruling GB-1.)*

**Four principles.** P1 figures attach to their producer (no project-level
`plots/`). P2 one producer per artifact. P3 engine-shaped artifacts live inside
their engine's subtree, every engine subtree sharing the shape
`config/ output/ plots/ _work/`. P4 a full climate analysis must be possible
with no wflow setup or run.

**Scope — repository half.** Retire `data/` for schema templates (O-01); DAG
renders to `<project_dir>/dag/` (O-02); delete the `docs/config/` mirror
(O-05); `examples/` → `test_case/` (O-20); fix the template's `project_dir`
default (O-21); add a parse-time in-repo-`project_dir` warning (O-22); declare
the missing plot outputs on rules 1.11/1.13 (O-24). Recorded as **kept
as-is with reasoning**: the nested `blueearth_cst/` package, the three homes
for executable files, and the Snakefiles at the repo root.

**Scope — artifact half.** Collapse the duplicate climate stores into one
region-keyed store (B1); move wflow forcing into the engine subtree (B2); tier
`climate_projections/` (B3); climate figures from the climate store, never from
wflow forcing (B4); two symmetric engine subtrees in the experiment —
`weather_generator/` and `hydrology_runs/` (B5); demote `stress_test/` to
`_work/` (B6); `model_results/` → `indicators/` (B7); auto-*suggest*
`experiment_id`, never auto-generate (B8).

**Behaviour-preserving, but NOT re-record-free** — unlike R6. No computational
path changes, but 17 of the 18 baseline targets move path (4 of them also
change content, embedding `project_dir`). The manifest is re-recorded **exactly
once**, at the end; `check_baseline.py` TARGETS and `semantic_tree_diff.py`'s
path map + TOML comparator update alongside. Batching the two halves into one
milestone is what buys the single re-record — split, it costs two.

**Exit criteria.** ~~Design accepted~~ **done 2026-07-28**; 15 `r07:` commits
landed off a task brief; all three Snakefiles `--dry-run` clean and
`pytest tests/` green; a full three-workflow run on the seed config completes;
full-`project_dir` `semantic_tree_diff` against the R7 path map clean modulo a
written MISSING/EXTRA allowlist; the P4 assertion demonstrated (climate figures
produced with **neither** `hydrology_model/` **nor the wflow build template** on
disk — strengthened at review, ext1-01); manifest re-recorded once and
`check_baseline` green.

*Commit count 13 → 15 at review (ruling GB-2): content scope unchanged, the delta
being a machinery-first commit so the regression gate exists before the moves it
polices, plus two moves the draft drew in the tree but assigned to no commit.*

**Open questions — all ruled at G1, 2026-07-27.** Engine-named subtrees
(`models/wflow/`) — **parked**, non-gating, deferred beyond R07 (and at review
the *structural* half of the question was deferred with it, ruling GB-1).
`MIGRATION.md`'s home (O-12) — **`docs/`**, with `naming.md` §7 amended to
distinguish a required internal rename record from an optional user-facing
guide. `blueearth_cst.Rproj` (O-13) — **deleted**. Weathergen date CSVs —
**`weather_generator/output/`** as designed.

**Ruled during the review run.** Pre-R07 `project_dir` trees are **unsupported**
— a fresh run is required and no `mv` migration script ships (ruling GA-2; no
production trees exist and no CST-API/frontend consumer reads artifact paths).
The B1 climate store has **one producer definition declared in both Snakefiles**
over region + catalog (ruling GA-1); its bbox derivation genuinely changes, which
is a named third exception to the behaviour-preservation stance and must be
proven by the `semantic_tree_diff` merge class, not assumed.

**Out of scope.** The tooling-contract decisions (O-14 `pyproject.toml`, O-15
`ruff`, O-16 `flit`) — unrelated to layout, still open. Docker (O-06) and Linux
end-to-end (O-18/O-19) stay parked. Promoting climate analysis to a fourth
Snakefile is a separate milestone; R7 only ensures the layout does not obstruct
it.

**Tag.** `r07-layout` (on seal).

---

## Cross-cutting principles

- **Every milestone ends with a tag.** Tags are the rollback points.
- **Every milestone preserves the M1 baseline** unless it is
  *intentionally* changing behavior. R3, R4, R5 are each allowed to
  change their own workflow's slice of the manifest — with a
  documented diff. R1, R2, R6 must preserve, modulo numerical-noise
  tolerance.
- **Manifest updates are part of the merge.** Each milestone updates
  `dev/baseline/manifest.json` if (and only if) changes meet that
  milestone's tolerance / justification rules. No silent updates.
- **No milestone touches the next milestone's territory.** If you
  find yourself wanting to fix a workflow-2 issue while in R3, write
  it down in `dev/followups.md` (or `dev/r04/followups.md` once R4
  is open) and keep going.
- **PRs back to upstream** (if any) are prepared from
  `pr/<NN>-<topic>` branches per the existing fork workflow guide —
  not from milestone branches directly.

---

## Commit strategy

Branch and tag naming live in "Branching and tagging conventions"
above. This section covers commit messages only.

**Subject format.** `<prefix>: <imperative subject, ≤72 chars>`. The
`<prefix>` matches the milestone the commit belongs to:

- Phase 1 (sealed): `m01:`, `m02:`, `m02b:`, `m02c:` — historical
  prefix on existing commits, do not rewrite.
- Phase 2 (active): `r01:`, `r02:`, `r03:`, `r04:`, `r05:`, `r06:`.
- Phase 3 (active): `p31:` (P3-1 experiment structure), `p32a:` (P3-2a
  model-independent climate analysis), `p32b:` (P3-2b model-swap
  interchange contracts), `p33:` (P3-3 performance passes).
- Repo housekeeping that doesn't belong to a milestone: `chore:`
  (e.g. updating this roadmap, `.gitignore`, fixing typos in
  unrelated docs).

Examples:

- `r01: migrate test config + 3 Snakefiles to sectioned schema`
- `r02: add dev/conventions/naming.md + CLAUDE.md pointer`
- `r03: collapse get_config into src/snake_utils.py`
- `r04: fix calendar handling in get_stats_climate_proj.py`
- `r05: extract stress-test grid into tested helper`
- `chore(dev): split roadmap into phase-1 / phase-2 sections`

**Body.** Optional. Include only when the *why* isn't obvious from
the diff. Wrap at ~72 chars. Don't restate what the diff shows.

**Granularity.** One logical change per commit. If the subject needs
the word "and", split it.

**Never commit.**
- Outputs under `project_dir/`.
- Files matching `*_local.yml` or other local-only configs.
- Secrets, credentials, large binary fixtures.
- Generated baselines other than `dev/baseline/manifest.json` itself.

If any of these slip in, update `.gitignore` first, then remove from
history if the commit hasn't been pushed.

**Merges and tags.** Default merge-commit messages are fine — don't
hand-craft them. Tag messages should restate the milestone goal in
one line (e.g. `r03-model-builder: model creation workflow + scripts
cleaned`).

---

## Minor open items

Small decisions that don't justify a section of their own. Resolve
in passing as the relevant milestone starts.

- ~~**CI.**~~ **DONE 2026-07-25 (first Phase-4 item)** —
  `.github/workflows/ci.yml` runs the unit suite on push to `main` and on PRs,
  across both supported pixi platforms (`ubuntu-latest` + `windows-latest`,
  `fail-fast: false`), with `locked: true` so `pixi.lock` drift fails the run.
  Scope set by measurement: a bare checkout gives 386 passed / 30 skipped /
  1 xfailed in ~100 s, every skip principled (~27 need the untracked
  `examples/test_local` fixture, 3 are the `--run-integration` end-to-end
  tests). **`check_baseline.py` turned out NOT to be the natural fit this entry
  assumed** — it fingerprints targets inside that untracked fixture tree, so it
  cannot run on a runner and stays a local gate, as does `semantic_tree_diff`
  whole-tree diffing. The ubuntu leg is also the first time the linux-64 half of
  `pixi.lock` has been resolved anywhere, so it de-risks the parked Linux work
  below.
- **R testthat coverage.** Decided at the start of R5 — Python
  helpers only by default; adding R testing infrastructure is a
  separate call.
- **Linter for naming conventions.** R2 establishes the convention
  but does not enforce it. A future linter (ruff custom rule, or a
  small ad-hoc script) would mechanically catch drift. Add as an
  R3+ followup if drift becomes a real problem.

---

## Deferred: Linux replication

Currently parked because no Linux machine is available locally. Not
abandoned — to be picked up when a Linux box, WSL setup, or Deltares
P-drive mount becomes available.

**What this covers when reactivated.**
- Reproducing the M1 baseline on Linux using
  `config/snake_config_model_test_linux.yml`.
- Rebuilding the Docker image on top of the M2 env manager and
  validating `run_snake_docker.sh`.
- Confirming the M2 env file resolves cleanly on Linux (it was
  authored cross-platform during M2).
- Sorting out the Deltares P-drive mount
  (`/mnt/p/wflow_global/hydromt`): whether the baseline is captured
  natively or only inside the container.
- Collapsing the OS-specific data catalog split (`*_linux.yml`) into
  a single parameterized catalog or config selection.
- Once green, recording Linux-specific fingerprints alongside the
  Windows ones in `dev/baseline/` (separate manifest, not a
  replacement).

**Where it slots in.** Likely a small dedicated Phase 2 milestone
when picked up (`r0X-linux-parity` between two existing R milestones)
so that subsequent milestones can assume both platforms work.

**Until then.** All milestone exit criteria refer to Windows only.
Linux-specific files (`*_linux.yml`, `run_snake_docker.sh`, the
Dockerfile) must continue to build / parse but are not exercised
end-to-end. Don't delete them.
