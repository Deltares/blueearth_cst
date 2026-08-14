# Fork Roadmap

Source of truth for the personal fork of `blueearth_cst`. Seven phases:

**Phase 1 — Foundation (sealed 2026-05-08).** Replicated upstream,
formalized the pixi env, upgraded load-bearing libraries, and added
unit-test coverage. Four milestones, all tagged. Phase 1 dev artifacts
under `dev/milestones/phase-1/`.

**Phase 2 — Refactor (sealed 2026-07-23).** Major overhaul of the workflow
code, config contracts, and repo structure. Six milestones running from R1
(modularity contracts) through R6 (structural refactor), in deliberate
single-purpose steps. Phase 2 dev artifacts under `dev/r##/`.

**Phase 3 — Usability & flexibility (complete 2026-07-25).** Driven by the user
expectations mapped 2026-07-23 at the R6 handoff: project/experiment
tracking, model flexibility, and performance. Milestones P3-1..P3-3;
dev artifacts under `dev/p3#/`. See § Phase 3 below.

**Phase 4 — Layout consolidation (R7 sealed 2026-07-29).** Opened 2026-07-26 out
of the post-R6 assessment: R6 settled the repository layout and P3-1 the
experiment layout, and neither could see the residue the other left. One
milestone, R7; dev artifacts under `dev/milestones/r07/`. See § Phase 4 below.

**Phase 5 — Workflow rework (R8 sealed 2026-07-31, opened 2026-07-29).** The first
phase to change what a workflow *computes* and how its rule graph is shaped,
starting with workflow 2. Milestone R8; design and audit trail under
`dev/reference/workflows/`. See § Phase 5 below.

**Phase 6 — Project tree redesign (R9 SEALED 2026-08-07).** Replaced the
*semantic roots* of the generated `project_dir` — `config/`, `data/`, `models/`,
`experiments/` — where Phase 4 consolidated the layout it inherited. Also adopted a
filename convention for generated artifacts, a pointer-derived model fingerprint,
and an experiment lifecycle. One milestone, R9; dev artifacts under
`dev/milestones/r09/`. See § Phase 6 below.

**Phase 7 — Naming coherence (R10 SEALED 2026-08-07).** Brought the
Snakemake rule identifiers onto one verb-and-noun scheme. Split from
Phase 6 for the same reason Phase 5 was split from Phase 4: rule names are a CLI
contract surface, not part of the artifact tree, and no durable artifact path
carries one. One milestone, R10; dev artifacts under `dev/milestones/r10/`. See
§ Phase 7 below.

**Phase 8 — WF3 rework (R11 SEALED 2026-08-08; R12 next).** Rebuilds workflow 3,
the stress test, in the two layers it turns out to have: **R11** changed what it
emits and what its members are called; **R12** changes how it executes. Mirrors
Phase 5, which did the same for workflow 2. Dev artifacts under
`dev/milestones/r11/`. See § Phase 8 below.

```text
Phase 1 — Foundation (sealed)
  base/<start-point>
  └── milestone/01-replication              →  tag: m01-replication
        └── milestone/02-pixi-installation  →  tag: m02-pixi
              └── milestone/02b-library-upgrades  →  tag: m02b-upgrades
                    └── milestone/02c-tests             →  tag: m02c-tests

Phase 2 — Refactor (complete 2026-07-23, branches from m02c-tests)
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

## Phase 1 — Foundation (summary)

Sealed 2026-05-08. All artifacts under `dev/milestones/phase-1/`; baseline
manifest at `dev/baseline/manifest.json`. The detailed scope and exit
criteria for each Phase 1 milestone live in the corresponding sealed
commits and `dev/milestones/phase-1/<milestone>/` docs; this section is a
reference summary only.

### M01 — Replication baseline (sealed 2026-05-07; tag `m01-replication`)

Got all three Snakemake workflows running end-to-end on the test
config and recorded baseline output fingerprints. Established the
fingerprint format (per-variable summary stats for netCDF; normalized
SHA256 for CSV/YAML; size-only for PNG with ±10% tolerance). Built
`dev/scripts/check_baseline.py` with `record` / `check` subcommands.
Artifacts: `dev/milestones/phase-1/m01/setup.md`, `dev/milestones/phase-1/m01/warnings.md`,
`dev/baseline/manifest.json`.

### M02 — Pixi env + install (sealed 2026-05-07; tag `m02-pixi`)

Replaced the conda + ad-hoc R + Julia setup with a single declarative
`pixi.toml`. weathergenr handled separately via `pixi run install` due
to a Mingw-w64 byte-compile issue with conda r-base on Windows.
Wflow.jl + Julia 1.11.x via juliaup outside pixi (conda-forge has no
win-64 Julia build). Artifacts: `dev/milestones/phase-1/m02/decisions.md`,
`pixi.toml`, `pixi.lock`.

### M02b — Library upgrades (sealed 2026-05-07; tag `m02b-upgrades`)

Bumped four load-bearing libraries: hydromt 0.x → 1.3, hydromt_wflow
0.x → 1.0, Wflow.jl 0.7 → 1.0.2, plus lifted Python stack caps
(numpy 2.x, xarray latest, python 3.12). Re-baselined the manifest
under the "intentional drift, document deltas" policy. Artifacts:
`dev/milestones/phase-1/m02b/audit.md`, `dev/milestones/phase-1/m02b/baseline_diffs.md`,
`dev/milestones/phase-1/m02b/handoff.md`.

### M02c — Test coverage (sealed 2026-05-08; tag `m02c-tests`)

Added unit-test coverage for four small, stable `src/` modules
(`metrics_definition`, `setup_time_horizon`,
`prepare_climate_data_catalog`, `extract_historical_climate`) with two
strict xfails for documented bugs. Established the
`sys.modules.setdefault` mocking pattern that R3-R5 inherit. Suite
state: 45 passed, 4 xfailed. Artifacts:
`dev/milestones/phase-1/m02c/test-coverage-design.md`,
`dev/milestones/phase-1/m02c/test-coverage-plan.md`.

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
`dev/milestones/r01/local-config-migration.md`. Per-workflow contract docs deferred
to R3/R4/R5 (2026-07-17 amendment). Suite: 51 passed, 3 skipped, 2 xfailed
(the pre-R01 47 plus 4 focused R01 reader/normalization tests). Scientific
invariance established **by construction** (value-preservation on every
migrated leaf + identity-preserving list/string normalization + green
suite + clean dry-runs), **not** by a manifest re-record: Task 5 found the
M2b `dev/baseline/manifest.json` stale (recorded from an untracked 3-model
config while the canonical uses 8; plus model-independent drift), so it is
left untouched and a clean rebuild is deferred. Full rationale + evidence:
`dev/milestones/r01/baseline_diffs.md`.

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
> (`dev/reference/workflows/<name>.md`) are moved out of R1: each is written as
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
  dedicated task. Full rationale + evidence: `dev/milestones/r01/baseline_diffs.md`.
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
`dev/milestones/r01/modularity-contracts-design.md`.

### R2 — Naming conventions (sealed 2026-07-19)

**Status.** Sealed 2026-07-19 — `dev/reference/naming.md` (187 lines,
< 250) authored and pointed to from `AGENTS.md`; the design was tightened
after independent GPT-5.6 and Fable reviews
(`dev/milestones/r02/naming-conventions-review-{gpt-20260718,fable-20260719}.md`).
Docs-only; suite unchanged (51/3/2); existing names grandfathered (zero
code diffs).

**Goal.** Single prescriptive style guide at `dev/reference/naming.md`
for naming identifiers and files across the repo. Pure docs; no code
refactoring. R3+ apply the conventions when touching code; existing
names are grandfathered. R3-R5 add new identifiers along the way
(helper functions, fixtures, wildcards, config keys), and locking the
convention first prevents each milestone from re-deciding naming on
the fly.

**Scope.** `dev/reference/naming.md` (< 250 lines, prescriptive
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
that `dev/reference/naming.md` is committed and tagged
(`r02-naming`) before R3's first *code* commit, so R3–R5 mint new
identifiers against a locked convention.

**Approach.** Prescriptive but lenient: opinionated where the codebase
is currently mixed, lenient where external conventions take
precedence. Two framings: (1) local style vs upstream contract —
local style does not apply to identifiers governed by external
systems; (2) grandfathered today, applied tomorrow — R2 itself
produces zero code diffs.

**Exit criteria.**
- `dev/reference/naming.md` exists, < 250 lines, prescriptive.
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
  `dev/reference/naming.md` in commit messages when adding new
  identifiers; future linter is a possible followup.
- Section 6 (domain identifiers) and section 4 (wildcard vocabulary)
  will grow as new tools / workflows enter scope. Doc is living.

**Tag.** `r02-naming`. Full design lives in
`dev/milestones/r02/naming-conventions-design.md`.

### R3 — Workflow 1: model builder (sealed 2026-07-19)

**Status.** Sealed 2026-07-19 — `build_model.smk` + its scripts
cleaned up: shared `get_config` and `tee_to_log` in `src/snake_utils.py`
(the cross-cutting patterns R4/R5 inherit), per-rule `log:`/`benchmark:` on
every non-trivial rule, deprecated path labels renamed, `setup_gauges` hardened
(raises on unknown `wflow_outvars`), the waterbodies rule encapsulated with a
removal trigger + structured sentinel, and a new `outlet_index.csv` rule-all
output settling the outlet-naming contract. R2 naming applied to workflow-1
identifiers; the deferred R1 contract doc `dev/reference/workflows/model_creation.md`
written. **Behavior-preserving**, verified by a full `--forceall` WF1 rebuild:
`check_baseline` 14/14, all per-rule logs written, `outlet_index.csv` and the
structured sentinel correct. Suite 73 passed, 3 skipped, 2 xfailed. Constant-
parameter restoration split out to task `t260719a` (a scientific decision +
baseline move); the workflow-3 `CyclicGraphException` `test_cli` ratchet is
retained for R5. Full design, external GPT-5.6 review, and integration-
verification record in `dev/milestones/r03/`. Merged to `main` 2026-07-19.

**Goal.** Clean up `build_model.smk` and the scripts it
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
  `dev/reference/workflows/model_creation.md` (contract doc deferred from R1;
  format in `dev/milestones/r01/modularity-contracts-design.md` §4).
- Any load-bearing `ruleorder:` in `build_model.smk` either
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
  documented diff in `dev/milestones/r03/baseline_diffs.md`.
- New unit tests added and passing.
- `dev/reference/workflows/model_creation.md` contract doc committed.

**Out of scope.**
- `analyze_projections.smk` content changes (R4) — except the
  shared helper import.
- `run_stress_test.smk` content changes (R5) — same caveat.
- Repo-wide directory restructuring (R6).

**Tag.** `r03-model-builder`.

### R4 — Workflow 2: climate projections (sealed 2026-07-20)

**Status.** Sealed 2026-07-20 — `analyze_projections.smk` + its four
`src/` scripts cleaned up, inheriting the R3 patterns. Design accepted via a
`design-review-loop` run (3-lens internal panel + 3 external GPT rounds +
round-cap arbitration; 24/24 findings closed) at `dev/milestones/r04/`. Landed in 11
commits (`1a8809e`..seal): contract doc `dev/reference/workflows/climate_projections.md`;
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
(`dev/milestones/r04/baseline_diffs.md`) — **no manifest re-record**. Suite 102 passed, 3
skipped, 6 xfailed.

**Audited, defects deferred (not "audited clean").** The chain audit
(`dev/milestones/r04/chain-audit.md`) confirmed the change-factor formula, calendars C3,
and hydro-year windows, and surfaced four deferred defects, each with owner +
activation condition: **D-CAL** — `get_change_annual_clim_proj` raises
`TypeError` on cftime 360-day/noleap calendars (task `t260720c`, latent for the
current seed); **D-VAR/D-MEM** — silent variable/member drops, wired as
strict-xfail fail-loud norms (task `t260720d`); **D-ATTRS** — the M2b CF-metadata
loss, probe-localized to the hydromt catalog read, a dependency op (task
`t260720e`). The strict-xfail wiring is the tripwire: fixing any code defect
flips its test xfail→xpass and fails the suite until the owning task removes the
marker. Full design, reviews, audit, and probe in `dev/milestones/r04/`.

**Goal.** Clean up `analyze_projections.smk` and the scripts it
calls. Inherit the patterns established in R3 (shared helper,
configfile mechanism, log/benchmark conventions).

**Deliverables.**
- Opening act, before code changes: write
  `dev/reference/workflows/climate_projections.md` (contract doc deferred from
  R1; format in `dev/milestones/r01/modularity-contracts-design.md` §4).
- The load-bearing `ruleorder:` directive in
  `analyze_projections.smk` either tightened or commented
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
  documented diff in `dev/milestones/r04/baseline_diffs.md`.
- New unit tests added and passing.
- `dev/reference/workflows/climate_projections.md` contract doc committed.

**Out of scope.**
- Workflow-1 or workflow-3 changes (other than shared helper
  inheritance).
- Repo-wide directory restructuring (R6).

**Tag.** `r04-projections`.

### R5 — Workflow 3: climate experiment (sealed 2026-07-20)

**Status.** Sealed 2026-07-20 — `run_stress_test.smk` + its `src/`
scripts + the R weathergen layer (`src/weathergen/generate_weather.R`,
`impose_climate_change.R`) cleaned up, inheriting the R3/R4 patterns. Design
accepted via a `design-review-loop` run (3-lens internal panel + 2 external GPT
rounds + round-cap arbitration; 21/21 findings closed) at `dev/milestones/r05/`. Landed in
12 commits (no commit 4; `8b356f3`..seal): contract doc
`dev/reference/workflows/climate_experiment.md`; `stress_test_grid` helper extracted to
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
- **weathergenr `spatial_ref` propagation** (`dev/tasks/` § R5) — the
  in-repo `generate_weather.R` workaround block STAYS (load-bearing) with a
  tightened removal-condition comment; the real fix is upstream in
  `tanerumit/weathergenr` `write_netcdf`. Upstream weathergenr task.
- **weathergenr wavelet `>= 16` cryptic error** (`dev/tasks/` § R5) —
  entirely inside the weathergenr package (`wavelet_cwt.R`); upstream task.
- **wf1 `| tee {log}` exit-masking-on-failure** (`dev/tasks/`,
  cross-cutting) — wf1's three shell rules run correctly on success (the R5 gate's
  wf1 leg passes) but mask the exit code on failure (cmd.exe has no
  `set -euo pipefail` prefix). Latent robustness item, NOT an R5 blocker; migrate
  wf1 to `> {log} 2>&1` or a portable tee wrapper. Owner `cst-architect`.

**R testthat coverage — DECISION: NO (locked at R5 start, G1-ratified).** The two
R scripts are thin `weathergenr` adapters (scientific logic is upstream); the
repo has no R test harness, and standing one up is R6-territory infra. The R
layer is gated end-to-end by the milestone baseline run + the `test_cli` dry-run,
with the §5a arity checks as the R-layer's correctness net. Full design, reviews,
and dispositions in `dev/milestones/r05/`.

**Goal.** Clean up `run_stress_test.smk` and the scripts it
calls — including the R weathergen layer. Inherit the patterns from
R3.

**Deliverables.**
- Opening act, before code changes: write
  `dev/reference/workflows/climate_experiment.md` (contract doc deferred from
  R1; format in `dev/milestones/r01/modularity-contracts-design.md` §4).
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
  documented diff in `dev/milestones/r05/baseline_diffs.md`.
- New unit tests added and passing.
- `dev/reference/workflows/climate_experiment.md` contract doc committed.

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
`r06-structural-refactor` design-review-loop (`dev/milestones/r06/`). Implementation run
as a three-phase Opus handoff with Fable gate reviews (Gate 1 after the atomic
move, Gate 2 pre-merge). **Behavior-preserving, verified run-relative** (no
manifest re-record, `check_baseline.py` untouched): full e2e via the wrapper
green (14/23/57 steps); baseline vs the pre-R6 scratch manifest clean modulo
the three adjudicated copied-config snapshot rows (normalize-then-compare) and
two **pre-existing** non-deterministic CSV column orderings (unsorted set
intersection, `PYTHONHASHSEED`-dependent — demonstrated R6-independent, values
identical by label; see `dev/tasks/`); full-tree semantic diff
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

**All four milestones sealed** (P3-1, P3-2a, P3-2b, P3-3). P3-3 was the last
milestone of the *originally* planned programme — Phase 1 foundation, Phase 2
refactor, Phase 3 usability/flexibility.

**RESOLVED 2026-07-29 (owner):** the roadmap is **not** closed. This paragraph's
original question ("whether to close the roadmap or open a Phase 4") was already
overtaken by events — Phase 4 opened 2026-07-26 for layout consolidation (§ Phase
4) — and the owner has now ruled that the WF2 v2.0 rework is **numbered phase
work**, landing as **Phase 5 / R8** (§ Phase 5) rather than unnumbered.

The former candidate pool stays recorded across `dev/tasks/`, the "Minor
open items" section below (CI, R testthat, a naming linter), the "Deferred: Linux
replication" section below, and the deferred items named in the
P3-2a/P3-2b/P3-3 designs (OQ-3 store, OQ-8 zone source, the 4th Snakefile entry
point, P3-2c PoC seam swap, the in-pipeline validator guard lift). Of those, only
the 4th Snakefile entry point is touched by the WF2 milestone — and it is
explicitly **not** taken (OQ-1: extend in place).

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
`dev/milestones/p31/experiment-structure-design.md`; audit trail:
`dev/milestones/p31/experiment-structure-design-review-record.md`; scoping intake
landed beside them. **Sealed 2026-07-24**: 8 `p31:` commits merged to
`main` (`1a8cca9`, --no-ff) after both human gates; value-identical wf3
re-record with semantic diff clean (evidence `dev/milestones/p31/baseline_diffs.md`
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
(`dev/milestones/r06/structural-refactor-design.md` §8; `modularization` direction).
**Confirmed scope** (`dev/milestones/p32a/climate-analysis-intake.md`, the
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
arbitration at the round cap): `dev/milestones/p32a/climate-analysis-design.md`, with
the consolidated review record and run observations beside it.
**Sealed 2026-07-24**: 6 `p32a:` commits (subpackage+shims → wf3 rewire →
wf1 extraction+parity → plot re-source → ladder QA → shim deletion) off
the task brief (`dev/milestones/p32a/climate-analysis-task-brief.md`), user-signed
milestone gate. Evidence: `dev/milestones/p32a/baseline_diffs.md` (ladder clean —
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
**Confirmed scope** (`dev/milestones/p32b/climate-interchange-intake.md`, the
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
accepted): `dev/milestones/p32b/interchange-contracts-design.md`, with the
consolidated review record beside it.
**Sealed 2026-07-24**: 4 `p32b:` implementation commits (two seam docs →
validators+tests (§8 commits 3+4 merged as sanctioned) → contracts README)
off the task brief (`dev/milestones/p32b/interchange-contracts-task-brief.md`),
user-signed milestone gate. Deliverables: `dev/reference/contracts/{README,
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

### P3-3 — Performance passes (sealed 2026-07-25)

Profiling-driven efficiency work targeting the wf3 stress-test sweep, with
baseline discipline à la R3–R5. **Confirmed scope**
(`dev/milestones/p33/performance-passes-intake.md`, the authoritative record): wf3
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
`dev/milestones/p33/performance-passes-design.md`, with the consolidated review
record and landed probe evidence beside it.
**Sealed 2026-07-25**: user-signed milestone gate (floor-free by intake
decision 3 — sign-off on the measured before/after + GN outcome +
value-identity evidence, no threshold imposed). 6 `p33:` commits off the task
brief (`dev/milestones/p33/performance-passes-task-brief.md`): baseline/decomposition +
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
exactly the documented cost. Evidence: `dev/milestones/p33/batching-results.md`.
**Caveat carried forward:** the commit-1 baseline (2242.9 s) was contaminated
by the concurrent `stage_data` workstream and is superseded in place — see the
supersession block in `dev/milestones/p33/performance-baseline.md`. Any future performance
measurement in this repo must record `cpu_time` alongside wall and confirm no
sibling agent session is active.
Post-P3-3 followups (genuinely disk-aware batch-size cap; the
`--keep-incomplete` ↔ `--keep-going` probe that could narrow the C5 blast
radius) in `dev/tasks/` § Post-P3-3.

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

## Phase 4 — Layout consolidation (R7 SEALED 2026-07-29)

Opened 2026-07-26 out of the owner's post-R6 assessment. Phase 2 (R6) settled
the *repository* layout and Phase 3 (P3-1) settled the *experiment* layout;
neither could see the residue the other left, and R6's own lock list deferred
the artifact tree explicitly ("5. Output layout under `project_dir/` already
mostly clean — leave alone unless a concrete pain point emerges"). Pain points
have now emerged, and they span both halves.

Phase 4 stays **layout-only**: the WF2 v2.0 rework that follows it is Phase 5
(owner, 2026-07-29), keeping this phase's theme clean even though R8 builds on
R7's output tree.

### R7 — Project layout (SEALED 2026-07-29)

**Status.** **Sealed.** Implemented as 15 `r07:` commits on
`milestone/r07-layout`, every acceptance criterion met (evidence below), merged
to `main` via `--no-ff` (`0ea3918`), tagged **`r07-layout`**, and pushed —
`main` and the tag are both on `origin` as of 2026-07-29.

CI is green on both platform legs for the sealed tree (run 30450296441,
windows 499/31/1 and ubuntu 498/32/1, totalling 530 + 1 xfail). That run is
also the first time CI had seen *any* of R7: the whole milestone plus the
follow-on tooling sat unpushed until the seal, so every prior gate on this work
was local-only. The `ci.yml` baselines were refreshed to match.

Post-seal tooling landed on `main` after the tag (not part of R7 proper):
O-14 decision 1 — a tool-config-only `pyproject.toml` (`ab781a5`) — and O-15 —
ruff adopted as the lint gate and enforced in CI (`85d3178`…`518151b`).

Design **ACCEPTED** 2026-07-28: `dev/milestones/r07/project-layout-design.md`, approved by
the owner at gate G2 of a
`design-review-loop` run on 2026-07-28. Drafted interactively with the owner
across the 2026-07-26 review (a 16-ruling question log), then put through the
loop: a three-lens internal panel and two external cross-vendor rounds, **44
findings, all dispositioned, none rejected**, across four versions. The external
round cap was reached with round 2 unconverged, so the owner arbitrated the three
surviving findings — meaning the final version's changes carry no external
verdict, which the design states on its face. Full audit trail:
`dev/milestones/r07/project-layout-design-review-record.md`; approved framing:
`dev/milestones/r07/project-layout-intake.md`; `naming.md` §7 path map:
`dev/milestones/r07/migration_project-layout.md`. Provenance of the findings:
`dev/reviews/2026-07-25_post-r6-assessment.md` (O-01 … O-24), which carries a
routing note for which observations R7 owns.

**Exit criteria — met.** Verified on the seed fixture, 2026-07-29:

| Criterion | Evidence |
| --- | --- |
| 15 `r07:` commits, each leaving the tree runnable | `5b532cd`…`7b1fe11` |
| All three Snakefiles `--dry-run` clean; `pytest tests/` green | 524 passed, 6 skipped, 1 xfailed, 0 failed |
| Full three-workflow run on the seed config completes | wf1, wf2 and wf3 all run for real |
| Full-`project_dir` `semantic_tree_diff` clean modulo a written allowlist | `CLEAN: 102 files compared, 0 failed, 0 missing, 0 extra, 7 allowlisted` |
| **P4 assertion demonstrated** | source figures build with **neither** `hydrology_model/` **nor** the wflow build template on disk — a real `snakemake` invocation, not a dry-run |
| B1 merge comparison passes on **both** sides | survivor vs `wf1_raw/` and vs the pre-R07 keyed store, element-wise |
| Discharge anchor before the re-record | `0/7670` timesteps over tolerance, max \|dQ\|/mean = **0**, after a full rebuild |
| Manifest re-recorded exactly once; `check_baseline` green | 18 → 15 rows at commit 14; `OK - 15 target(s) match manifest` |

**Scope delivered.** 109 files changed (+5,196 / −13,256): 22 deletions
(`data/`'s two tracked CSVs, the 16-file `docs/config/` mirror, three retired
modules, the `.Rproj`), 17 additions, 3 renames. The large deletion count is the
milestone's point — the toolbox stopped carrying basin data and duplicated
configs.

**Post-milestone follow-ups.** `dev/tasks/` § Post-R7 — 21 items, 11
resolved (4 fixed, 1 mitigated, 1 answered, 4 closed with reasons). Six further
commits after the milestone fixed the latent wflow-TOML rebuild defect (R7-1),
moved the parity transform out of the model package (R7-4), added manifest
branch provenance (R7-21), and closed the documentation and cosmetic items.

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

**Tag.** `r07-layout` — cut 2026-07-29.

---

## Phase 5 — Workflow rework (R8 SEALED 2026-07-31)

Opened 2026-07-29 (owner). Phases 1–4 worked on the repository, its contracts and
its layout; Phase 5 is the first phase to rework what a workflow *computes* and
how its rule graph is shaped. It starts with workflow 2, whose structure was
mapped in detail once R7's output tree settled.

Kept separate from Phase 4 deliberately: R8 builds on R7's output tree, but
"consolidate the layout" and "rework the workflow" are different kinds of work
and the owner chose a clean theme boundary over the sequencing convenience of
one phase.

Milestone IDs continue the `R##` series across the phase boundary (R7 → R8), as
Phase 4 continued it from Phase 2's R1–R6.

### R8 — WF2 v2.0: GCM projections analysis (SEALED 2026-07-31)

All seven steps of the design's §8 migration table are implemented. Tagged
`r08-wf2-projections`; user migration note in `docs/migration-r08-wf2.md`.

**Method discipline that shaped the outcome.** Every value-changing step wrote its
falsifier *before* its code — the observation that would disprove the step, plus
the command producing it — and characterized its diff before any baseline
re-record. That order caught, among others: a dtype upcast smuggled into a
"values must not move" step; a traversal-order change in a helper extraction; a
naive fix that would have broken the case already working; and a rule that had
never run at all.

**Four findings worth carrying forward.**

1. **A value recorded in two places disagreed five times** — the model calendar,
   `n_years` twice, the effective reference window twice. Each was caught only
   because something compared them. The reliable defence was passing values
   between artifacts rather than recomputing them, and it is why `report.md`
   reads `provenance.json` instead of deriving its own disclaimer.
2. **Four cache defects shared one signature**: the change is right, the tests
   pass, and the artifacts silently do not move. Unlisted `REDUCER_KERNEL`
   callees; attribute stamping outside the hashed kernel; stage B having no
   hashed kernel; and `kernel_hash` itself being non-reproducible across
   processes for any function containing a closure.
3. **The fixture cannot see most of what was built.** Its basin is equatorial and
   latitude-symmetric (so area weighting is exactly inert), wet year-round (so the
   dry-month rule never fires), and all three models share one calendar. Green
   fixture gates are necessary and never sufficient — see
   `docs/migration-r08-wf2.md` and the per-step falsifier notes.
4. **`check_baseline` compares PNGs by size with a 10 % tolerance.** A figure whose
   content changed completely passes if its compressed size lands within 10 %.
   "3 PNGs pinned" reads as stronger coverage than it is. Not changed here —
   altering the comparator is its own decision.

**Two defects found and fixed that predate the milestone:** the pipeline recorded
`proleptic_gregorian` for every model (false for every `noleap`/`360_day` one),
and the reference window was one complete hydrological year short whenever the
start month was January.



**Goal.** Restructure workflow 2 from a change-factor calculator whose rule graph
fights its own cost profile into a monthly GCM projections analysis workflow:
three stages with fan-out only where the network is, a persistent series store
that is *also* a declared product, and per-(model, scenario, member, horizon)
data points with no cross-combination aggregation.

**Design ACCEPTED 2026-07-29** at gate G2 of a `design-review-loop` run
(`wf2-climate-analysis-v2`): one internal lens (Fable) + two external
cross-vendor rounds, **28 findings across 4 versions, all dispositioned, one
partially rejected by owner ruling**. The external cap was reached with round 2
unconverged, so the owner arbitrated all nine surviving findings — the final
version's changes therefore carry no external verdict, which the design states on
its face. Accepted design: `dev/reference/workflows/wf2-climate-analysis-v2-design.md`;
audit trail: `dev/reference/workflows/wf2-climate-analysis-v2-design-review-record.md`;
current-state map: `dev/reference/workflows/wf2_climate_projections_overview.md`; store
inventory: `dev/reference/workflows/wf2-cmip6-store-inventory.md`.

**Owner rulings that shaped it.** Clip the GCM reference to the 2014 end of the
CMIP6 historical experiment rather than splicing scenario data (R1); retain the
gridded option, default off, with declared outputs (R2); **no aggregation at any
level** — each (model, scenario, member) is one data point, and cross-combination
statistics are ex-post (R3′/R3″); v2.0 is monthly GCM projections analysis with
no observed-data comparison (R4); the basin-averaged monthly series per run is a
declared deliverable (R5). Arbitration: 30 calendar years 1985–2014 (A1), picked
dry-month defaults (A2), non-`pr`/`tas` variables selectable but best-effort
(A3). Open questions settled 2026-07-29: extend `analyze_projections.smk`
in place rather than opening a 4th entry point (OQ-1); rename `save_grids` →
`save_gridded` at step 5e (OQ-12).

**Status.** Commits 1 and 2a landed on `main` (`dcd5459`, `04013fc`, `37b2e1f`):
WF2 declares the shared climate store and no longer depends on
`hydrology_model/` (goal G2, verified by a DAG build with the model tree absent),
and the catalog generator now pins physical store identity in a generated index.
Commit 2b (persistent series cache) is blocked pending the empirical baseline
re-run and the catalog crawl. Task brief:
`dev/milestones/r08/2026-07-29_wf2-v2-decouple-and-cache.md`.

**Not yet done:** the empirical value-neutrality proof. A full WF2 re-run reached
12 of 24 jobs and was killed externally; `extract_climate_grid` and all three
`monthly_stats_hist` completed against the live archive, but the merge/plot jobs
did not, so the manifest-pinned outputs are still pre-change artifacts and
`check_baseline` has not yet been a meaningful gate on this work.

**Scope settled 2026-07-29 (owner).** The four scope-cluster questions closed
together: **no daily branch** (OQ-5 — 46 models monthly vs 35 daily, ~30×
volume); **no new dependency** (OQ-7 — `xclim` was conditional on the daily
branch, `regionmask` on OQ-10's measurement); **breadth over depth for members**
(OQ-13 — since nothing aggregates, model diversity spans more plausible space
than member depth, so one member per model across as many models as resolve; the
existing global-list mechanism already delivers this and the per-model mapping is
a tail-only follow-on); **`kin`/`press_msl` stay best-effort** (OQ-15).

**Direction after v2.0:** projected PET, not extremes — WF1 already derives PET
from the observed store via `hydromt.model.processes.meteo`, so doing the same on
the GCM side makes projected PET comparable to observed with no new dependency.
It needs `rsds`/`psl` certified (64 → 57 models) and two more entries in §5.5's
`canonical:` spec. Recorded as design §10a.

**Genuinely still open:** OQ-10 (true vs midpoint cell edges — one measurement),
OQ-11 (revisit fail-fast — needs observed failure rates), OQ-14 cadence (needs
observed pin-mismatch rates). All three need operational data that does not exist
yet.

**Tag.** `r08-wf2-projections` — cut 2026-07-31.

---

## Phase 6 — Project tree redesign (R9 SEALED 2026-08-07)

Registered 2026-08-04. Phase 4 consolidated the generated tree it inherited —
producer-oriented roots (`climate_historical/`, `climate_projections/`,
`hydrology_model/`), tidied but not rethought. Phase 6 replaces those roots with
domain ones and settles four things Phase 4 left to convention: how generated
files are named, what a model fingerprint actually has to cover, when an
experiment's configuration stops being editable, and where a run's own records
live.

Kept separate from Phase 4 rather than reopening it: R7 is sealed and tagged, and
its record stays as written. This is a second, deeper pass over the same surface,
not a correction of the first — R7's depth-agnostic run-directory handling is
precisely what makes R9's flattening cheap.

### R9 — Generated project tree (SEALED 2026-08-07)

**Status.** **SEALED 2026-08-07**, tag `r09-project-tree`. All five phases
implemented and merged; landing gate nine of nine. Closing record:
`dev/milestones/r09/closing-record.md`; gate evidence `landing-gate.md`.

The seal is dated two days after the work finished (2026-08-05) because nothing
prompted it — the roadmap went on describing R9 as "not yet implemented" while
`main` carried the whole milestone. That gap is the reason **Cross-cutting
principles** now names the seal as the milestone's own last step rather than an
implied consequence of merging.

**What landed.** `project_dir` has six roots — `config/ data/ models/
experiments/ logs/ benchmarks/` — and every artifact sits at the scope of the
producer that wrote it, verified on a fresh run as those six and nothing else.
P1 built the comparator (a 59-rule path map, `--check-map`, orphan-store pruning,
two inventory tiers, both zero-unmapped); P2 moved the tree and flattened
fan-out members back into filenames; P3 renamed the result tables and proved
value identity **byte-identical before** the single allowed baseline re-record;
P4 added the pointer-derived model digest, `model_reference.yml`, a drift guard
ordered before simulation, and experiment freezing; P5 amended `naming.md`
§4/§6/§7/§8/§9, `AGENTS.md`, `README.rst` and both seam contracts.

**The lesson worth carrying.** The landing gate found **four** defects, and
`pixi run test-full` was green over every one of them. A `from __future__` import
in three `script:` modules made them unrunnable while 28 unit tests passed
against them, because importing a module is not executing it under Snakemake's
prepended preamble. The drift guard was asserted structurally — an input edge
orders A before B but does not make A re-evaluate — so it detected and never
fired. Both are now cited where they can be acted on rather than only here
(`AGENTS.md`, the validation ladder).

**Carried forward, not resolved:** `[R9-1]` colliding geojson basenames and
`[R9-5]` the baseline member's presence differing between the two table shapes.
Both are on the board.

Drafted as an external-review brief and then **accepted without external review**
— the owner waived it. The design says so on its face and records the two
consequences: it carries no independent verdict, and its nine open questions were
ruled by the owner directly rather than forced by a reviewer. Two of those nine
rulings changed the design, which is the outcome a review is normally relied on
to produce; the reviewer contract is retained unexercised so a review can still
be dispatched against v4 without rework.

**Goal.** `config/`, `data/`, `models/`, `experiments/` as the stable semantic
roots of every project, with: one live Wflow model at `models/hydrology/wflow/`;
reusable engine-independent inputs under `data/`; each experiment self-contained
and keyed by an allocated ID; fan-out members keyed by filename on both engine
sides; lowercase `snake_case` for every locally minted name; and a
pointer-derived model fingerprint that WF3 re-checks before simulating.

**Design decisions of record** (full rationale in the design):

| Area | Decision |
| --- | --- |
| Run records | Log and benchmark live at the scope of what the run produces (P7) — project root for WF1/WF2, the experiment for WF3. Merging them was rejected on `_parts` collision, not taste. |
| Wflow run subtree | `rlz_<r>/` removed; members are `rlz_<r>_cst_<c>` filenames, matching the climate side. |
| Result tables | `Qstats.csv` → `q_indicators.csv`, `basin.csv` → `basin_indicators.csv`, `RT_*.csv` dropped. |
| Rule 3.11 | `export_wflow_results` → `derive_wflow_indicators`. The one rule rename R9 carries, on the principle that a milestone renames what it falsifies — R9 makes the outputs indicators, so "export…results" would be a mismatch R9 itself created. The other nine renames are R10. |
| Naming | Lowercase `snake_case` for locally minted names; upstream identifiers and engine filenames exempt. Closes the gap `naming.md` §8 left open. |
| Fingerprint | Pointer-derived: the TOML plus every model-root file its path-valued keys resolve to. |
| Catalogs | Referenced, never copied. Only generated catalogs live in a project. |

**Preconditions before a task brief.** Two, both named by the design itself: an
**artifact inventory** covering every file the three workflows and their engines
emit, and the **old → new path map** built from it. The map is needed three times
over — by `semantic_tree_diff` to gate the migration, by `naming.md` §7 as the
mandated internal rename record, and by the brief itself to be actionable.

**Status: run, and COMPLETE** (2026-08-04) —
`dev/milestones/r09/migration_project-tree.md`. Built from the Snakefiles'
declared outputs rather than from `test_case/test_local`, which is a mixed-era
tree carrying orphans from at least three code generations and no `spatial/`
subtree at all. Building it that way is what surfaced the blocker: the project's
`config/` is written **in full** by rule 1.01 `snapshot_config` as generated
provenance, while the design labels it the editable project source — the same
paths with opposite semantics. The v4 tree also asserts that toolbox catalogs are
"referenced, not copied", which is false; they are both. Design corrected to v5;
ruling 6 marked superseded.

**Resolved across design v6–v8**, all toward what the code emits: the config
snapshot **stays under `config/`** (the decider being that
`config/runs/snake_config_model_creation.yml` is a declared `input:` of WF3's
drift guard, so it is a consumed contract artifact rather than an archive); the
climate store **keeps its source+window cache key**; `cmip6/raw/` and `scalar/`
are both kept, `scalar/` being R8's ruling S8-03; `change_factors/` stays as two
files under `summary/`; and the last four unplaced artifact classes are placed.
P4 was restated from *separate* to *distinguishable*, and **P9 added** — where
the tree differs from what the code emits, the emitted structure wins unless a
stated reason overrides it. Four divergences were found and every one encoded a
prior decision.

**The map is COMPLETE.** It was briefly recorded as blocked on the
`wf1-spatial-decoupling` P2 Gate 2 review; that finding was withdrawn 2026-08-04
once the git history was checked. Gates 2 and 3 were approved and the branch
merged on 2026-08-02 (`29ccde9`) — `ad9702d` closed the gates in the phase report
but left the phase index reading "Gate 2 review pending", and the inventory
inherited that stale line as a blocker. The WF1 rows were derived from `main` and
already reflect the landed work. The index has been corrected.

**Two obligations the inventory added to the milestone.** First, a real defect:
Wflow's `[logging] path_log` defaults to `log.txt` beside the TOML, so removing
the `rlz_<r>/` level puts every concurrently-batched member's log at one path —
a race. The directory removal and the per-member `path_log` must land in the
same commit, with a two-member concurrency falsifier. Second, keeping the store's
cache key means a changed window strands its predecessor on disk, so the pruning
tooling must learn to report orphaned store directories.

**Scope note, not a blocker.** `config/project.yml` does not exist and nothing
writes one; adopting it moves config ownership from the toolbox into the project
rather than relocating a file, and R9 must budget it as new capability.

**Master brief written** 2026-08-04:
`dev/milestones/r09/project-tree-task-brief.md`. The complexity gate classified
R9 as several independently verifiable subsystems rather than one unit, so it is
a master brief plus five phase briefs, matching R7's shape. Phases are strictly
sequential — every phase but P1 edits `run_stress_test.smk`, so they
cannot run in parallel worktrees — and P1 (the `semantic_tree_diff` comparator)
deliberately precedes the migration, because a move made before its comparator
exists has no regression detector. Three human gates: comparator, scientific
delta before any baseline re-record, and landing.

**Exit criteria.** Design accepted *(done 2026-08-04)*; inventory and path map
complete *(done 2026-08-04)*; master brief and five phase briefs written
*(done 2026-08-04)*; commits landed leaving the tree runnable at each
step; all three Snakefiles `--dry-run` clean and `pytest tests/` green; a full
three-workflow run on the seed config completes; `semantic_tree_diff` clean
against the R9 path map modulo a written allowlist; the fifteen falsifiers in
the design exercised — including the ones that need a **real run** rather than a
dry-run (the flattened Wflow subtree changes emitted TOML pointer strings, and
the HydroMT model-root ownership question is settled empirically); baseline
manifest re-recorded exactly once as a documented value-identical re-record;
`naming.md` §8 amended in the same milestone that adopts the tree.

**Carries a known defect to fix, not inherit.** `validate_hm7` asserts the basin
table's header is exactly the two perturbation-axis columns — true only when no
basin-average outputs are configured, and therefore false under the shipped
default `wflow_outvars`. The rename must not carry it forward unfixed.

**Out of scope.** Multiple concurrent Wflow models; multiple retained ERA5
windows; a project-level `runs/` hierarchy with execution-attempt manifests;
the source-repository layout; any change to what the workflows compute.

**Support decision.** Pre-existing `project_dir` trees are unsupported: a fresh
run is required and no `mv` migration script ships. This restates R7's ruling
GA-2 on the same grounds — no production trees exist and no external consumer
reads artifact paths.

**Branch.** `milestone/r09-project-tree` — cut 2026-08-04 from `main`'s tip,
local only. Phase work goes on `feat/r09-p<N>-<topic>` branches off it, never
on the trunk: the baseline is red by construction from the first P2 commit
until P3's re-record, and that window must not sit on `main`. It merges to
`main` once, green, at the seal (Gate 3). Inventory in
`dev/reference/git-conventions.md`.

**Tag.** `r09-project-tree` — cut 2026-08-07 on the milestone branch's tip
(`ca3fca5`), matching how every prior milestone was tagged.

---

## Phase 7 — Naming coherence (R10 SEALED 2026-08-07)

Registered 2026-08-04, out of the R9 inventory: mapping every rule's outputs made
the rules' own names hard to ignore. Phases 4–6 worked on artifacts; Phase 7
works on the identifiers that produce them.

Split from Phase 6 rather than folded into it, for the reason Phase 5 was split
from Phase 4. Rule identifiers are a **CLI contract surface** (`naming.md` §9),
not part of the artifact tree, and no durable artifact path carries one — rule
names reach `project_dir` only through transient log/benchmark part directories
and as section labels inside two merged files. There is no shared cost to
capture, and R9 is already carrying six kinds of change.

### R10 — Rule naming (SEALED 2026-08-07)

**Status.** **SEALED 2026-08-07**, tag `r10-rule-naming`. Landed 2026-08-06 as a
seven-step sequence; migration record `dev/milestones/r10/migration_rule-names.md`.

**What landed, and it is more than the renames.** The design was reviewed by a
`cst-architect` pass against the code, which found several defects the design had
asserted its way past. The sequence that followed: a stale `LOG_RULES` entry and
the rule-index diagram fixes `[R10-8]`/`[R10-4]`; the `LOG_RULES` conformance test
`[R10-9]`, written **before** the sweep that would edit it; the `[R10-1]` merge,
with `[R10-2]`'s split **dropped** once measurement showed the seam it assumed did
not exist; ADR 0003 §8–12, which split `prepare_spatial_maps` at the thematic seam
and made `wflow_id` basin-blocked; then the twelve renames, the `[R10-5]`
renumber, `[R10-7]` and `[R10-10]`.

**Two things went differently than planned, both worth keeping.** The
double-renumber contradiction was resolved by moving the renumber to **last**,
after the rule set stopped changing — the published 45-identifier map had to gain
rows, not merely be renumbered, because §8 added three identifiers. And §12's
`wflow_id` change uncovered a live defect rather than being cosmetic:
`output.csv` had been shipping `Q_101` twice, a collision that had already leaked
into WF3's response surface as a column named `Q_101.1`.

**Gates.** `pytest tests/` 1526 passed from the primary checkout with the fixture
layer included; a **full three-workflow run** — WF1 17/17, WF2 14/14, WF3 34/34,
every merged-log section present in rule-number order with no `_parts/`
surviving, including the batch and fan-out labels no test reaches;
`check_baseline.py check` OK 8/8 run *after* that; `pixi run tree-check` 186
paths, 0 unmapped. `naming.md` gained §8b (the verb vocabulary) and §9 was
**reversed** to make `NN` positional.

**One deliberate narrowing, recorded so it is not read as an oversight.** The
scope was the rule *identifier*. Five modules keep their old names, so
`plot_wflow_evaluation` executes `plot_results.py`, `fetch_gcm_slice` executes
`fetch_gcm_raw.py`, and three others likewise. Renaming a module is an
import-surface change and was kept out; `migration_rule-names.md` §"Script
modules did NOT move" carries the list.

**Carried forward:** `[R10-12]` the forcing NC's non-reproducible bytes tripping
WF3's drift guard, `[R10-13]` a failing `script:` rule writing an empty log part,
`[R10-14]` the shared-rule comment-edit cascade, `[R10-9]`'s one-constant-per-rule
half, and `[R10-6]`'s unmeasured WF2 hydrography read cost. All on the board.

**Goal.** Every rule reads `<verb>_<noun>`, verb first, drawn from a controlled
verb list so that two rules doing the same kind of work read the same. Ten of
twenty-eight rules move; eighteen already conform and are named explicitly as
not-to-touch.

**The distinction that needed care.** `reduce_` and `derive_` both turn many
inputs into few outputs. They are split by **position, not operation**:
`reduce_` is an intermediate aggregation feeding a later rule
(`reduce_gcm_series`), `derive_` computes a workflow's terminal product
(`derive_change_factors`, `derive_wflow_indicators`). WF2's and WF3's final rules
therefore read alike, which they should.

**Defects being fixed.** Two rules carry no verb at all
(`climate_stress_parameters`, `climate_data_catalog`); one verb is actively wrong
(`export_wflow_results` — folded into R9); `setup_` duplicates `prepare_`;
`weagen` and `proj` are contractions that appear in no path or directory; `_st`
reads as a truncation; and `plot_results` / `plot_map` are vague beside their
specific siblings.

**The implementation trap.** Each rename touches three call sites — the `rule`
identifier, its `log:`/`benchmark:` prefix, and its `LOG_RULES` entry. A missed
`LOG_RULES` entry is **not an error**: `merge_logs` is deliberately scoped to that
list, so the section silently vanishes from the merged log while its parts stay
on disk forever.

**Also in scope, independently landable.** `naming.md` §9 says `NN` is the step
in *definition order*; WF2 defines 2.00, 2.03b, 2.03, 2.01, 2.02, … out of order,
with gaps at 1.14, 2.05, 3.12. The recommendation is to **amend §9** — the number
is a stable identifier assigned at creation, not a position — rather than
renumber, which would churn every part path across all three workflows for
nothing. `naming.md` should also gain the verb vocabulary; it has no rule-naming
section today.

**Exit criteria — all met.** Design accepted *(2026-08-04)*; the renames landed
with `LOG_RULES` updated in the same edit — **twelve, not ten**, the count having
grown as the name-vs-body audit found more; `migration_rule-names.md` recorded per
§7; `pytest tests/` green (1526, primary checkout); the full three-workflow run
showing a merged-log section for every rule and no surviving `_parts/`; the
repository-wide grep for each old name returning nothing outside the migration
record **and the documented module-name narrowing above**; `check_baseline check`
passing **unchanged**, no renamed rule having altered an output path or value.

**Branch.** `milestone/r10-rule-naming` — cut retroactively 2026-08-07 at
`7164d83`, R10's completion point. R10 is the one milestone whose work did not run
on its own branch: it landed through `fix/r09-followups`, because it began as R9
followups and grew into the milestone. The branch exists so the inventory in
`dev/reference/git-conventions.md` has a row for every milestone; it records where
R10 finished, not a history of how it got there.

**Tag.** `r10-rule-naming` — cut 2026-08-07 on that branch.

---

## Phase 8 — WF3 rework (R11 SEALED, R12 next)

Registered 2026-08-07. Workflow 3 is the last of the three not to have been
reworked: Phase 5 did WF2, and WF1 was settled across R3, R7 and R9. This phase
does WF3, and splits it the way the existing material already splits.

**Two designs existed, and they are different layers, not rival versions.** The
design run under `docs/wf3-redesign` changes how WF3 **executes** — member-level
incrementality moves out of Snakemake into our own code, with a manifest, a
ledger and a `member_hash`. The change register in
`dev/milestones/r09/wf3-change-requests.md` changes what WF3 **emits** and what
its members are **called**. They also target different trees: the design run
predates R9 (2026-08-01→04), the register was opened 2026-08-05 out of the first
full run of the migrated R9 tree.

Measured 2026-08-07 rather than assumed: `design-v4.md` carries ~74 references to
artifacts and rules that no longer exist — `Qstats` ×25, `RT_` ×15 (tables R9
deleted), `export_wflow_results` ×14, and three rules R10 renamed. Its 65 findings
and two external review rounds survive; its mechanics need re-deriving.

So the phase sequences rather than merges them, emitting-layer first — which also
means R12 inherits settled result-table shapes instead of moving ones.

### R11 — WF3 artifacts and identification (SEALED 2026-08-08)

**Status.** Complete in three phases. Scope:
`dev/milestones/r11/wf3-consolidation-scope.md`; the build's own record is
`dev/milestones/r11/phase-3-run-report.md`.

| phase | landed | what |
| --- | --- | --- |
| P1 | `94ab26e` | unit A — result tables wide→long, one per variable, `aggregate_rlz` retired as a hard error |
| P2 | `923ecb5` | unit B — `cst_`→`st_`, zero-padded ids, the design table, `st_id` (C28), C34, F7 |
| P3 | `d462710` | the run, the delta gate, the single re-record, and three defects only a run could find |

**P3 is the phase worth reading.** P1 and P2 changed WF3 without ever executing
it — every gate was unit tests, dry-runs and greps. Running it found three
defects, and they share one shape: **something that should have been checked was
not being checked, and nothing said so.**

1. **Two of eleven metrics vanished silently.** The seed config set
   `run_historical: false`, so `ST_START = 1`, so the `st_0` baseline never ran —
   and because Q5 fixes the class-C month *from* that baseline, both month
   metrics were skipped entirely. 180 rows gone, no warning, every gate green.
   `validate_hm7` checked results→design and never design→results, so a member
   that produced no rows was unreachable by every assertion it had.
2. **The design table recorded perturbations nobody applied.** Fixing (1) meant
   passing `design=` to `validate_hm7` in the integration test, which had always
   called it bare — so C28's consistency check had never run on real data at all.
   It failed on first contact: the table said −30.000001%, the run imposed
   −30.0%, because the design row was derived from an in-memory float32 frame
   while every consumer read the persisted float64 text.
3. **A test had asserted nothing since R9.** `test_store_region_bbox` guarded on
   `hydrology_model/staticmaps.nc`, a pre-R9 path, so it skipped silently — the
   third instance in one phase of the pattern `AGENTS.md` names as the one that
   survives every gate a branch can run. Found by reading the skip *list*, not
   the pass count.

**Rulings taken during the build.** **Q2**: `[R10-13]` lands separately, on
attribution rather than cost — P3's one run is the phase's entire evidence, and a
cross-workflow change folded into it gives every failure two candidate causes.
**Q8's comparator**: `q_indicators.csv` moves off a byte hash onto a new
`indicator` target kind with a per-`(metric, location)` tolerance, because P1
dropped the rounding that had been an accidental drift buffer.
**`stress_test_design.csv`** stays out of the manifest. **`[R10-12]`**: the
operator accepted the rebuilt model, on hashes showing `staticmaps.nc` and
`wflow_sbm.toml` byte-identical and only the forcing NC moved.

**The delta gate held where it was falsifiable.** Eight of eleven metrics
reconstruct at the old file's own 2-dp precision; `q_wettest_month_mean` moved as
pre-registered under Q5. Three upper-tail metrics moved <1% downward, explained
by the butt-splice P1 removed — an explanation recorded explicitly as *post-hoc,
not predicted*.

**Carried forward:** `[R10-13]` (`t2608071202`, `t2608071219`), the pre-R9
cascade guard (`t2608081012`), and stale fixture coverage (`t2608082010`).

**Goal.** Land the WF3 changes already specified against the post-R9 tree — unit
A (result tables, `q_indicators.csv` wide→long, one table per variable), unit B
(`cst_`→`st_` run identification), C34, and F7 — and close the WF3-territory
followups.

**Rulings taken at scoping.** `[R9-5]`: the unperturbed baseline is a member of
the response surface and is emitted in **both** table shapes; it needs no new
rule, because CR-2's `realization_id = 0` already means pooled. `[R10-12]`: the
drift-guard re-record is **accepted and documented**, not fixed in code, with the
cost of that choice recorded on its watch-item.

**Out of scope.** Unit D — the only breaking config migration, deferred with its
specification complete. And the execution model, which is R12.

**Open questions — both resolved.** Whether unit B's rename reaches the
`experiment.yml` R9 freezes (Q1, resolved 2026-08-07; the answer inverted the
question — scope §10) and whether `[R10-13]` belongs here (Q2, ruled 2026-08-08:
it does not — scope §8).

**Tag.** `r11-wf3-artifacts` — cut 2026-08-08 on `milestone/r11-wf3-artifacts`.

### R12 — WF3 execution model (OPEN — G2 ratified 2026-08-08, not yet scoped)

Takes `docs/wf3-redesign` as an **input, not a starting point**.

**The G2 prerequisite is discharged.** Ratified 2026-08-08 as an *architectural
input, not an implementable spec*, with the risk-7 part-3 rejection ratified and
`ext1-2`'s standing objection carried rather than resolved. Full record:
`dev/reference/workflows/wf3-experiment-v2-design-review-record.md`; the
assessment behind it: `dev/milestones/r12/g2-assessment.md`.

Why nothing was promoted: the run's process was sound and its output is
unimplementable, which are not in tension — the tree it was designed against no
longer exists. Two structural facts, not the ~178 stale identifiers, decide it.
`aggregate_rlz` is **load-bearing** (it defines `member_id` and the
cell-completeness predicate) and R11 retired it as a hard error, dissolving the
distinction rather than renaming it. And `GF-21`, a named gate falsifier, asserts
the reduce emits **no `cst_0` row** — which R11 inverted on `[R9-5]`, so it now
passes only if the implementation reproduces behaviour R11 deliberately removed.

**R12's first task is the re-derivation**, whose deliverable is a written mapping
from each surviving finding to its post-R11 expression. That mapping is what
makes two external review rounds portable instead of merely archived. Surviving:
the manifest + ledger architecture, `member_hash`, resumable sweeps, epochs,
quarantine, checked atomic publication, and the counterbalanced AB/BA timing gate.

The `cst-run-control` skill governs — its scope is exactly this territory (run
manifests, resume, checkpoints, quarantine, conformance vectors) and may already
answer questions the design run spent rounds on.

Do not merge `docs/wf3-redesign`. It is cited by path; its scratch stays in git
history, per the WF2 precedent.

**Tag.** `r12-wf3-execution` *(on seal)*.

---

## Cross-cutting principles

- **Every milestone ends with a tag.** Tags are the rollback points.
- **The seal is the milestone's last STEP, not a consequence of merging it.**
  Merging the work and recording that it happened are separate acts, and nothing
  prompts the second. R9 and R10 both finished, merged, and then sat with this
  file still describing them as unimplemented — R10's section read "Not
  implemented; no task brief, no branch" while `main` carried all seven of its
  steps and every gate had passed. A backlog review caught it 2026-08-07, not the
  work itself. Sealing means, in one sitting: tag, update this file's phase
  summary AND its milestone section, **update the index in
  `dev/milestones/README.md`** (folder row, seal date, tag), update the branch
  inventory in `dev/reference/git-conventions.md`, and **ask which reference
  documents the milestone superseded** so they can be sealed while the answer is
  still known (`AGENTS.md`, Conventions). None of it is inferable later.

  The `milestones/README.md` step was added 2026-08-08 for the same reason the
  rest of this list exists: it had drifted exactly as described above, still
  showing R9 and R10 as open a day after both sealed, and carrying no row for
  R11 at all. Checked at a close, it is thirty seconds; discovered later, it is
  an index that reads as authoritative while describing an older world.

  **Order matters, learned at R11's close:** cut the tag and record it BEFORE
  restoring `worktree_policy: always`. Restoring first leaves the remaining
  close steps needing an edit in a checkout that no longer permits one.
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
  it down in `dev/tasks/` (or `dev/milestones/r04/followups.md` once R4
  is open) and keep going.
- **PRs back to upstream** (if any) are prepared from
  `pr/<NN>-<topic>` branches per the existing fork workflow guide —
  not from milestone branches directly.

---

---

## Moved out of this file (2026-08-02)

This file is the phase narrative: what each milestone set out to do and how it
landed. Two kinds of content that had accumulated here now live where they
belong, because neither is history:

- **Branching, tagging, and commit conventions** → `reference/git-conventions.md`,
  alongside the ref inventory that previously pointed back here for them.
- **"Minor open items" and "Deferred: Linux replication"** → `followups.md`
  § Carried over from the roadmap. They were a third backlog, invisible to
  `TODO.md` and `followups.md` alike.

"Cross-cutting principles" above stays: those are the rules a milestone is run
under, inseparable from the narrative of the milestones themselves.

## Candidate milestones (moved from `followups.md` 2026-08-07)

These are directions, not tasks. They came out of `followups.md` when the
todo-board replaced it: a candidate milestone on a task board reads as
scheduled work, and scheduling it is exactly the decision not yet taken.
Scope one before it becomes board items.

- **[R7-18] Climate analysis as a fourth Snakefile** — a separate milestone. R7
  only ensured the layout does not obstruct it, and the model-free store plus
  rule 1.15 are the enabling pieces.

- **Climate analysis/visualization as a model-independent subworkflow.**
  *Direction raised by Ümit 2026-07-21 (test/pre06, Observation 4 follow-up).*
  We should be able to analyze and visualize climate data — gridded meteo
  diagnostics, forcing climatology, projection change factors — **without**
  building a hydrology model. Today the WF1 climate QA plots
  (`src/plot_results.py` §4) are coupled to the built wflow model
  (`mod.forcing.data`, `staticmaps["subcatchment"]`), and the forcing itself
  (`inmaps_historical.nc`) is a *product* of the model build. Yet the natural
  minimal dependency for climate analysis is a region/AOI geometry + data
  catalog — which WF3's `extract_climate_grid` (rule 3.02: `region.geojson` +
  clim source → `extract_historical.nc`) and WF2's `monthly_stats_*` already
  demonstrate (both depend only on `region.geojson`, not the full model).
  Direction: factor a shared **climate-analysis subworkflow/component** whose
  inputs are (region/AOI, gridded climate dataset) and whose outputs are
  climate diagnostics/plots, consumed by WF1 QA, WF2, and WF3 alike; degrade
  gracefully (region-only → basin-level; + subcatchment map → per-subcatchment).
  This is *functional* decomposition (capability boundaries), a **new axis**
  beyond the R6 roadmap's current layout/`enabled:` pain points (roadmap §R6) —
  add it to the R6 lock list when R6 scoping begins.
  **Tension to resolve:** ADR 0002
  (`dev/decisions/0002-revive-subcatchment-climate-plots.md`) currently sources
  the climate plots from `mod.forcing.data` (re-couples to the build); a modular
  design would source raw gridded climate (catalog + region) instead. Keep this
  in mind when ADR 0002 is implemented — it may argue for sourcing from
  `extract_climate_grid`-style extraction rather than the model forcing. To be
  discussed at R6 scoping; not to be designed or implemented yet.

- **Reconsider the WF1 rule arrangement — bundle/split + rename.**
  *Direction raised by Ümit 2026-07-21 (test/pre06, Observation re: WF1's 11
  rules).* NOT covered by R6's current lock list (which is repo/directory
  layout + `enabled:`); this is rule-level composition *within* a workflow — a
  new R6 axis. WF1 today has 12 rules (1.01–1.12; see `build_model.smk`
  and naming.md §9): copy_config, prepare_build_config, create_model,
  add_reservoirs_lakes_glaciers, add_gauges_and_outputs, write_outlet_index,
  setup_runtime, add_forcing, run_wflow, plot_results, plot_map, plot_forcing.
  Candidates to weigh:
  - **Plotting is three separate rules** (plot_results 1.10, plot_map 1.11,
    plot_forcing 1.12), each a `script:` emitting PNGs and now sharing
    `save_figure`. Consider consolidating into fewer rules (or one parameterized
    "plots" rule / a plotting sub-component) and a shared plotting module.
  - **Model-update chain is finely split** (create_model → add_reservoirs… →
    add_gauges… → write_outlet_index → setup_runtime → add_forcing). Some splits
    are historical: `add_reservoirs_lakes_glaciers`'s own comment says it "can be
    moved back to create_model when hydromt is updated" — a standing re-merge
    candidate.
  - **Verb standardization**: rules mix `create_`/`add_`/`setup_`/`prepare_`/
    `write_`/`plot_`/`run_`; `prepare_build_config` vs `setup_runtime` vs
    `create_model` overlap semantically. Align on a small verb vocabulary
    (naming.md §2 already prescribes `verb_noun`).
  **Key tradeoff — do not bundle blindly:** separate rules give Snakemake
  parallelism and *targeted* re-runs (edit forcing → only `plot_forcing` reruns);
  bundling coarsens the DAG and re-runs more on any change. Weigh granularity vs.
  readability per rule. Interactions: any reorg renumbers the `W.NN` scheme
  (naming.md §9 documents this as a mechanical cost), touches CLI target names
  (a naming.md §7 contract-surface rename → migration note), and overlaps the
  climate-subworkflow item above (plotting may move out of WF1 entirely). Same
  lens applies to WF3 (also 11 rules). To be discussed at R6 scoping; not to be
  designed or implemented yet.

