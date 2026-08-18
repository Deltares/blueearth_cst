# AGENTS.md

> **Canonical.** Single source of truth for every runtime. Codex reads this file
> directly; `CLAUDE.md` is a thin entry point that imports it (`@AGENTS.md`).
> Author repo instructions here, never only in `CLAUDE.md`.

## Overview

BlueEarth Climate Stress Test — a multi-language (Python + R + Julia) scientific
workflow toolbox stitched together by Snakemake. The four `*.smk` files at
the repo root are the only entry points; there is no package CLI. Full narrative
in `README.md`.

| entry point | id | does |
|---|---|---|
| `analyze_climate.smk` | wf0 | the basin's historical climate — **model-free** |
| `build_model.smk` | wf1 | builds the Wflow-SBM model, runs it on historical forcing |
| `analyze_projections.smk` | wf2 | CMIP6 change factors — a plausibility overlay |
| `run_stress_test.smk` | wf3 | the stress test |

All four were `Snakefile_<noun>` and extensionless until 2026-08-14, and the
`workflows.<name>` config keys carry the same names as the files. Migration for
an existing project: `docs/migration-workflow-names.md`. wf0 is numbered 0
rather than renumbering the other three — `W` is a workflow id, not a position
(`dev/reference/naming.md` §9), so `ls logs/` still sorts in execution order.

## Background

Method context that changes how code here should be edited (rationale:
`docs/cst-toolbox-technical-note-2025.md` §1):

- CST is **bottom-up** stress testing (decision-scaling / DMDU): it perturbs local
  climate over a temperature × precipitation grid instead of running selected GCM
  scenarios through the system. Stress-test scenarios come from the stochastic
  weather generator — never couple the experiment workflow to CMIP scenarios.
- CMIP6 output (wf2) is a plausibility overlay only: its change factors situate the
  perturbation grid in projection space; they never drive a stress-test run.
- Pipeline: wf0 characterises the basin's historical climate from one or more
  gridded datasets **without building a model** — the forcing-selection question,
  which matters here precisely because CST does no local calibration, so forcing
  choice is the dominant lever on the historical run; wf1 builds a distributed
  Wflow-SBM model from global datasets via
  hydromt and runs it once on historical forcing (rapid deployment, no local
  calibration); wf2 computes monthly change factors per (model, scenario, horizon);
  wf3 is the stress test — weathergenr generates `RLZ_NUM` realizations, each
  perturbed across `ST_NUM` temp/precip combinations (`st_0` = unperturbed
  baseline), run through Wflow and reduced to the hydrological indicators that form
  the response surface.
- This repo is the workflow engine of a three-part platform (workflows + CST-API
  backend + CST-frontend GUI). No web/API code belongs here; the GUI drives these
  Snakefiles server-side.
- CST targets rapid, first-order basin assessments on global data, not detailed
  engineering design. Prefer robustness and automation over site-specific
  sophistication.

## Repo Map

The tree is self-explanatory; these are the parts that are not.

- `blueearth_cst/` — modules invoked from Snakemake `script:` directives (Python) or
  `Rscript --vanilla` `shell:` bodies (R); none is a standalone CLI. Split by stage
  (`model/`, `projections/`, `climate_analysis/`, `experiment/`) plus `shared/` for
  cross-cutting helpers (`snake_utils.py`, `run_logged.py`, `climate_parity.py` —
  the engine-neutral regrid/PET transform, plotting primitives, log/benchmark
  reducers, `cross_workflow_leaves.py` — the wf1 outputs WF3 declares but cannot
  build, `spatial_geoms_parity.py` — which of the two geoms trees answers
  which question) and `weathergen/` for the R weather generator.
- `config/` — four bins plus `advanced_settings.yml`. There is **no
  `workflows/` bin**: every shipped `--configfile` target lives beside the project
  it writes into, under `test_case/` (`snake_config_rapid.yml`,
  `snake_config_baseline.yml`, its `_linux` twin, `snake_config_wf2_fast.yml`
  — which one to run is under Workflow), which is how a real project is laid out too.
  Those, plus the two basin-input CSVs under `test_case/test_data/`, are the ONLY
  tracked files in the otherwise-ignored `test_case/`, which is why `.gitignore`
  reads `test_case/*` plus `!test_case/snake_config_*.yml` and then repeats the
  pattern one level deeper for `test_data/` — the directory form `test_case/`
  cannot work, since git refuses to un-ignore a file whose parent DIRECTORY is
  excluded, and that is equally why the CSVs take three lines (un-ignore the
  directory, re-exclude its contents, re-include `*.csv`) rather than one.
  Those CSVs were untracked until 2026-08-12, which broke both CI legs from
  08-10: `gauge_points` is a rule input, so a bare checkout could not resolve
  the wf1 DAG at all. **Keep the `snake_config_` prefix on any new seed
  config** — a name outside that glob is silently untracked, with `git status`
  reporting the old path as deleted and never listing the new one.
  `catalogs/` (hydromt data catalogs, the `-d` targets, plus an `archive/` of
  those whose only consumers were the archived configs; `cmip6_data.yml`
  is **generated** by `dev/scripts/generate_cmip6_catalog.py` from a live
  `gs://cmip6` listing — never hand-edit it. `cmip6_store_index.json` is generated by the SAME crawl and carries an equal `crawled_on`; a consumer asserts that equality rather than trusting the two to be in step).
  Then the two that were ONE bin until 2026-08-11, when a name that asserted
  "template" over a directory holding live rule inputs was split:
  `defaults/` is read by a rule and changing one changes a run
  (`wflow_build_model.yml`, `wflow_update_waterbodies.yml`,
  `weathergen_config.yml`); `templates/` is only scaffolds you copy
  (`snake_config.template.yml`, the two `*_template.csv` observation schemas, and
  `archive/` for unmaintained single-workflow configs). Each has its own README.
  `<project_dir>/config/templates/` did NOT move — it is a generated snapshot
  bin, a different meaning of the word. Since 2026-08-13 it is normally EMPTY:
  a referenced file is copied into the project only when the toolbox repo
  cannot give it back, so a shipped default is recorded by git blob id and the
  bin receives a file only when a project points the key at its own. The tracked
  `wflow_sbm.reference.toml` is reference only — nothing reads it, and it lags
  what a build emits; the infix says so, since the bare name read as an input. Real basin data lives in the project folder, referenced by absolute
  path, never in this repository.
  The last bin, `config/basemap/`, holds `natural_earth_50m.gpkg` — the ONLY
  geographic data committed here. The basin map's locator inset draws land,
  borders and major cities from it. It is vendored rather than fetched for the
  same reason the satellite tiles were removed: `cartopy.feature.COASTLINE`
  downloads Natural Earth on first use and caches it per user, so a fresh
  machine hits the network and two machines can disagree. 1.8 MB buys the
  question away. This is a static cartographic asset, like a font — not a
  precedent for putting basin data in the repo. Provenance, licence and the
  rebuild recipe: `config/basemap/README.md`.
  Beside the bins sits `config/advanced_settings.yml` — toolbox-wide
  `constraints:` (hard limits no project config can relax, e.g.
  `min_historical_years`), `defaults:` (starting values a project config may
  override, e.g. `julia_threads` ← `shared.julia_threads`), and `runtime:`
  (external toolchain pins, e.g. `julia_version`, which `pixi.toml` and
  `Manifest.toml` must match — `tests/test_julia_runtime.py` enforces it, since
  neither can read YAML). It is **not** a
  `--configfile` target: `snake_utils` reads it once, for every project. Its
  schema is closed, so an unknown section or key is rejected at parse time; add
  a setting to the file and to `snake_utils._ADVANCED_SETTINGS_SCHEMA` together.
- `scripts/` — user-facing runners. `suggest_experiment_name.py` writes
  `experiment_name` into a config once, to pin a deliberate name; it edits the
  config as TEXT, never `yaml.safe_dump`, which would delete every comment in
  the file. The key is optional: unset, WF3 defaults to the project name plus
  the date the experiment was first created, and **reuses** an existing dated
  experiment before minting the current date
  (`experiment/allocate.py::resolve_default_experiment_name`). The reuse is not
  a nicety — a name regenerated from today's clock on every run would break
  Snakemake idempotence, which is why no path here generates one.
- `dev/` — planning, audits, design docs, conventions, roadmap, the baseline
  manifest, and dev-process helpers under `dev/scripts/`. Not shipped, not
  user-facing. **Open work lives on the todo-board**: one note per item under
  `dev/tasks/`, closures in `dev/LOG.md`, and `dev/TODO.md` is **generated**
  (`python dev/scripts/todoboard.py render` — do not hand-edit it, edit the
  note. The bare `todoboard` is on nobody's `PATH`: the CLI lives inside the
  per-user, gitignored `todo-board` skill bundle, so the wrapper is how it is
  reached). A note is a
  `todo-item` (work) or a `watch-item` (true and worth seeing, but no action
  intended — it carries a **Trigger** naming what would make it work). This
  replaced `dev/followups.md` on 2026-08-07; everything closed before that is
  in `dev/followups-archive.md`, which stays readable because code, tests and
  Snakefiles cite its item IDs. Three inspection helpers, all **report-only by
  default**:
  `prune_series_cache.py` (orphaned WF2 series), `prune_climate_store.py`
  (stale `<source>_<window>` climate stores, R9) and `snapshot_project_tree.py`
  (a tree as a path list, checked against the post-migration inventory — also
  `pixi run tree-check`. Its `--map r09` alternative, the one-way migration map
  only a pre-move tree could satisfy, was retired 2026-08-11 with the R07 one;
  `[R10-11]` is why the default stopped being it). Deleting is an explicit owner action via `--delete`, and pruning
  must run **before** any reference snapshot, or the snapshot bakes the orphans
  in and the gate compares them instead of the live artifacts. Neither prune
  tool sees everything: R9 P2 found stale files only an mtime sweep caught,
  because they sat under directories the path map routes wholesale.
- `docs/` — user-facing reference, including the vendored hydromt / hydromt-wflow /
  wflow guides. Configs are not mirrored here; `config/` is the single source.
- `.github/workflows/ci.yml` — `pytest tests/` on ubuntu + windows with
  `locked: true` (so `pixi.lock` drift fails CI). It covers only what a bare
  checkout can run: the fixture-dependent layer and the `--run-integration` tests
  skip, and `check_baseline.py` / whole-tree `semantic_tree_diff.py` cannot run
  there at all (they need the untracked `test_case/test_local` tree). **CI green
  does not mean the baseline was checked** — those stay local gates. The baseline
  covers DATA, not figures: `FIGURE_KINDS` targets are excluded by default
  because a figure is fingerprinted by byte size, so any cosmetic edit fails the
  gate without indicating a defect (see the validation ladder under Workflow).
  **A task that MOVES THE PROJECT TREE must grep the test suite for the old
  roots.** The fixture-dependent layer is the one part of the suite that cannot
  fail in CI or in any worktree, so a stale path there survives every gate a
  branch can run: R9 moved the tree and left 22 such failures, three of them
  behind an `os.path.exists` guard that turned a wrong path into a **silent
  skip** rather than a failure (archived as R9-4). Tree-shape gates —
  `semantic_tree_diff`, `check_baseline` — do not read the code that reads the
  tree, so they cannot substitute.
- Outputs land under `project_dir` (set in the config). Production `project_dir`
  lives **outside the repository tree**; the in-repo untracked `test_case/test_local`
  is a dev/test convention only, explicitly exempt from that rule.

Three homes for executables, split by INVOCATION MODEL — not by audience (O-23):
`blueearth_cst/` is executed by Snakemake, `scripts/` is what a user runs to execute
the pipeline, `dev/scripts/` inspects or maintains the repository and is never part
of a run.

**`dev/scripts/` is not only executables.** It also holds small LIBRARIES that
`tests/` imports via `sys.path` — `semantic_tree_diff.py` (the tree comparators
and the project-tree inventory) and
`cross_workflow_inputs.py` (STAGES the wf1 leaves WF2/WF3 need on disk).
"Never part of a run" still holds — no Snakefile touches them — but
"dev-only" does not: a bare-checkout CI run imports them, so an import-time
error there fails the suite on both legs. Treat those two as contract surfaces
with test consumers, not as scratch helpers.

The leaf LIST is not one of them. It lived in `cross_workflow_inputs.py` until
2026-08-17, when `scripts/run_workflows.py` needed it for its wf1 preflight —
and that is a run path, so the definition moved to
`blueearth_cst/shared/cross_workflow_leaves.py` and the stager re-exports it.
The split is the invocation-model rule applied, not an exception to it: a list
of paths both a run and a fixture need belongs in the shipped package, while
the staging machinery that only tests call stays in `dev/`. **When something in
`dev/scripts/` acquires a run-path caller, move the shared part rather than
importing `dev/` from a run** — a user's checkout is not guaranteed to have it.

## Key Commands

Run everything inside `pixi shell`, or prefix each command with `pixi run`, so
`snakemake`, `python`, and `Rscript` resolve to the pixi env.

```bash
pixi install          # conda-forge + PyPI deps (Python stack, R toolchain, snakemake)
pixi run install      # + weathergenr (R, via remotes) and Julia env (Pkg.instantiate)

# Run the four workflows IN ORDER (run_stress_test needs build_model
# artifacts). snake_config_rapid.yml is the DEFAULT config; swap in
# snake_config_baseline.yml only for the runs listed under Workflow.
#
# wf0 is OPTIONAL for the pipeline: it produces the region, vector layers and
# climate store wf1 also declares, so running it first just means those exist
# before the build asks. Run it ALONE when the question is which forcing
# dataset to use -- it needs no model, and nothing it writes is model-shaped.
snakemake all -c 3 -s analyze_climate.smk     --configfile test_case/snake_config_rapid.yml
snakemake all -c 3 -s build_model.smk         --configfile test_case/snake_config_rapid.yml
snakemake all -c 3 -s analyze_projections.smk --configfile test_case/snake_config_rapid.yml --keep-going
snakemake all -c 3 -s run_stress_test.smk     --configfile test_case/snake_config_rapid.yml

# Or drive all enabled workflows through the wrapper:
pixi run python scripts/run_workflows.py --config test_case/snake_config_rapid.yml

# Render a workflow's DAG into the config's own project_dir (never into the repo
# root): logs/dag/<project_name>_wf<N>_dag.png, with wf3 carrying its experiment
# id in the name — logs/dag/<project_name>_wf3_<experiment>_dag.png:
pixi run python scripts/plot_workflow_dag.py -s build_model.smk --configfile <cfg>

# Snapshot a project tree as a path list and check that it holds nothing
# undeclared (add --out to record it; nothing is written otherwise). It checks
# against the POST-MIGRATION INVENTORY, which is now the only map there is:
pixi run tree-check --config <cfg>

# Report orphaned artifacts. Both DRY RUN by default; --delete is explicit:
pixi run python dev/scripts/prune_series_cache.py --config <cfg>    # WF2 series cache
pixi run python dev/scripts/prune_climate_store.py --config <cfg>   # stale climate stores

snakemake ... --dry-run           # inspect the DAG before running or after editing rules
snakemake --unlock -s <Snakefile> --configfile <cfg>   # Snakemake locks the workdir on crash

pytest tests/test_cli.py          # cheapest sanity check: dry-runs all four entry points
pytest tests/                     # full suite (test_build_model.py is slow)
```

**Run the workflows from the PRIMARY checkout, not from a task worktree.**
Snakemake keeps its "what is up to date" metadata in `.snakemake/` under the
*working directory*, so one `project_dir` driven from two checkouts gets two
independent stores and they disagree — measured 2026-08-02, the same config and
the same project planned 12 jobs from one checkout and 2 from the other.
Snakemake also locks its working directory, so two checkouts running against one
`project_dir` each hold their own lock while writing the same outputs: a
corruption risk, not just confusion. Worktrees are for editing code and running
`pytest`; pipeline runs belong in the one checkout `worktree_policy: always`
already reserves for integration.

**Every worktree builds its OWN pixi env.** This said the opposite until R9 —
that a worktree "resolves to the primary's copy instead of building its own" —
and that is not what happens: each worktree carries its own tracked `pixi.toml`,
so pixi creates a separate `.pixi/` beside it.

The practical consequence, measured in R9 P2: `pixi install` alone is not enough
to run WF3 in a fresh worktree. `weathergenr` comes from `pixi run install`
(remotes), so a worktree that has only had `pixi install` fails at rule 3.06
with `there is no package called 'weathergenr'`. **Run `pixi run install` in a
worktree before running WF3 there** — and prefer running the pipeline from the
primary checkout anyway, for the `.snakemake` reason above.

**A worktree carries no `test_case/`, and that silently downgrades the test
suite.** `test_case/` is untracked, so `git worktree add` does not bring it. The
fixture-dependent layer then **skips instead of failing** — measured 2026-08-07:
`pytest tests/` in a fresh worktree reported *1567 passed, 31 skipped* and looked
like a clean gate, while **15 of those skips were the fixture layer** this file
already names as the one no worktree can exercise. A branch whose
change crosses the project tree (an R9-style move, a `MODEL_DIRNAME` edit) is
exactly the case that layer exists to catch, and exactly the case a worktree
cannot report.

**Seed a new worktree with the fixture subtrees it needs, by COPY:**

```bash
# 46 MB — the tree named by 18 of the 25 fixture references in tests/
cp -r <primary>/test_case/test_local        <worktree>/test_case/
# 248 KB — only dev/scripts/preview_basin_map.py reads it
cp -r <primary>/test_case/basin_map_fixture <worktree>/test_case/
```

Copy those two subtrees, not all of `test_case/` — the whole directory is 361 MB
and most of it is superseded reference trees (`ref_wf2_pre_*`, `test_local_pre_*`,
`_pruned_*`). `test_case/test` and `test_case/gabon` appear in test source but do
not exist on disk; their tests skip on the primary too, so they need nothing.

**Never symlink or junction it.** `tests/test_model_rebuild_cascade.py` runs a
real `snakemake all -c 1` against the fixture, so a link would drive one
`project_dir` from two checkouts — the same `.snakemake` divergence and
concurrent-lock corruption this section already warns about, arriving through the
test suite instead of through a deliberate run. A copy is an independent
`project_dir`; a link is a shared one.

**The agent-config directories are the opposite case, and are SYMLINKED.**
`.claude/`, `.codex/` and `.agents/` are gitignored per-user state, so no
worktree gets them either — but they are read, not written, and one shared
definition is the whole point. `.claude/skills` and `.agents/skills` are
themselves symlinks into `~/workspace/brain/artifacts/`, which a copy would
dereference into a private fork that then answers with last week's version.

Their absence does not fail, it **downgrades**: measured 2026-08-11, a worktree
session resolved only 4 of the 18 project skills — every generic process skill
still came from `~/.claude/skills`, so the 14 domain ones (`hydromt`,
`snakemake`, `wflow`, `cst-run-control`, `climate-stress-testing`, …) were
missing with nothing reporting it. Codex is worse: `.codex/agents/*.toml` are
regular files with no brain fallback, so a Codex session there has no personas
at all.

Both lists are declared in `.git-workflow.yml` (`worktree_seed:` copies,
`worktree_link:` links) and applied by the launcher at `git worktree add` time.
For a worktree created before that, reapply both from the primary's config:

```bash
python ~/workspace/brain/artifacts/skills/git-workflow/scripts/worktree-session.py sync
```

**Do NOT borrow the primary checkout to run a branch's gate.** The obvious
alternative to seeding — `git checkout --detach <branch>` in the primary, run,
`git checkout main` — parses as safe and is not, because a long test run holds
the checkout for fifteen minutes and nothing reserves it.

Tried 2026-08-07 and it failed exactly that way. Another session merged its own
branch **onto the detached HEAD**, noticed, checked out `main`, and redid the
merge properly — all while the suite was running. `config/basemap/` exists only on
the branch, so it vanished from the tree mid-run and six basemap tests failed. The
failures were pure artifact: the branch was fine, and the run had to be discarded.
Cost was 15 minutes and a false defect report. A stale checkout is recoverable; a
gate result you have to *decide whether to believe* is worse than no gate.

Seeding is therefore not the cheap option, it is the correct one — a seeded
worktree cannot be moved by another session. The residual difference is small and
worth stating: a copied fixture proves the code runs, the primary's tree is the
one the baseline was recorded from. When that distinction actually matters — a
baseline re-record — take the primary deliberately, with no other session live,
which `worktree_policy: always` is what enforces.

`.pixi/` self-ignores through a `.gitignore` the tool writes itself, so it needs
no repo rule. The pytest and ruff caches were redirected out of the root on
2026-08-11 (`pyproject.toml` `cache_dir` / `cache-dir`) and now sit under the
ignored `.tmp/`; they still self-ignore, which is what covers a `--isolated`
run that recreates them at the root.

Use `test_case/*_linux.yml` + `config/catalogs/*_linux.yml` variants on
Linux — data-catalog paths differ from Windows. `scripts/run_snake_test.cmd`
(Windows) and `scripts/run_snake_docker.sh` (Linux/Docker) wrap the test config.
`profiles/default/config.yaml` is auto-loaded from the repo root and sets
`quiet: reason`; drop it when you need to see *why* a job re-ran.

`scripts/run_workflows.py` invokes the enabled workflows in fixed order (model →
projections → experiment), reading `workflows.<name>.enabled` from a
full-orchestration config — the single-workflow `snake_config_projections_*.yml`
files (parked under `config/templates/archive/`, unmaintained) have no
`workflows:` section and are not wrapper inputs. A missing or
non-boolean `enabled:` is a hard error; the wrapper stops on the first nonzero
Snakemake exit and returns that code. `enabled: false` only skips the invocation —
prior outputs are neither deleted nor refreshed, so a downstream workflow consumes
whatever is already on disk. Full contract: the module docstring, clause-by-clause
pinned by `tests/test_run_workflows.py`.

## Conventions

- Name new identifiers and files per `dev/reference/naming.md` (snake_case,
  lowercase acronyms, `_path`/`_dir` for paths vs `_ds`/`_df`/`_cfg` for objects,
  three-tier domain-identifier exemptions). Existing names are grandfathered; rename
  a contract surface only with a migration note.
- Snakefiles are config-driven: each parses one `--configfile` YAML via a shared
  `get_config(config, key, default, optional)` helper. A new config key must mirror
  that contract (raise on missing required, return the default for optional).
- Each Snakefile takes the `--configfile` path from `workflow.configfiles[0]` and
  forwards it as `config_path` to downstream R scripts — keep that forwarding even
  though the Snakefile itself reads the parsed `config`.
- `analyze_projections.smk` no longer carries a `ruleorder:`. It was retained
  as stale insurance — a 2026-07 dry-run on the pinned Snakemake showed it
  constrained nothing on the tests fixture or a reduced config — with removal
  deferred to a task that first encoded ambiguity-sensitive config shapes as
  regression tests (`dev/milestones/r04/climate-projections-design.md` §3). WF2 migration step
  4d removed it without that task: it named `monthly_change` and
  `monthly_change_scalar_merge`, which 4d merges into `derive_change_factors`, and
  an unknown rule name is a parse error. The merge also removes what it insured
  against — there is no second stage-B rule left that could claim the same output.
- Register new data sources in a `config/catalogs/*_data*.yml` catalog and pass it to
  hydromt via `-d`. Never hardcode data paths in a Snakefile.
- `dev/` vs `docs/`: put a new file where its audience is — design notes and one-off
  probes under `dev/` (planning, not shipped), install/usage docs under `docs/`.
- **Keep configuration references current.** Paths, filenames, config keys and
  commands are updated wherever they are read as guidance — `docs/`, `README.md`,
  this file, and code comments — whenever the thing they name moves. A stale path
  in a document someone reads to do their job is a defect, not a record, and is
  not preserved for historical fidelity. When a rename or move lands, grep the
  old spelling and fix every live reference in the same commit. Migration maps
  for their own sake are not kept: `docs/migration-r06.md` was deleted on
  2026-08-11 rather than carried forward, since git history already records what
  moved.
  The one exception is `dev/` milestone and review records, which exist to be the
  baseline a past milestone's commits were checked against and are valuable
  *because* they are unedited. Those carry a
  `> **SUPERSEDED — … (sealed YYYY-MM-DD).**` banner and an entry in
  `dev/reference/sealed-records.yml`, which freezes the hash so an edit fails
  `tests/test_sealed_records.py`. **That registry is the entire list** — a
  document not in it gets its paths kept current like everything else, so read
  the registry rather than guessing from a document's age.
- [Python] `script:` modules read `snakemake.input/output/params`, not `sys.argv`.
  [R] `Rscript --vanilla` scripts take positional args via `commandArgs(trailingOnly=TRUE)`.
- netCDF (`.nc`) is the interchange format across R/Python/Julia. Wrap intermediate
  per-realization netCDFs in `temp(...)` — omitting it explodes disk usage on large
  `RLZ_NUM × ST_NUM` runs.

## Workflow

### Session slots — a reusable worktree, claimed one task at a time

This repo keeps **reusable session slots** instead of a worktree per task. A
slot is a persistent worktree with **no permanent branch**: detached at `main`
when idle, and checking out one ordinary short-lived task branch while occupied.
The worktree, its `.pixi` environment, and the 46 MB `worktree_seed` fixture
survive; the task branch lands and is deleted like any other.

| Slot | Worktree | State |
|---|---|---|
| `session-1` | `.worktrees/blueearth_cst/session-1` | detached at `main` when idle |
| `session-2` | `.worktrees/blueearth_cst/session-2` | detached at `main` when idle |

Two slots, because several sessions routinely work this repo at once — the same
reason `worktree_policy: always` is set, and relaxing that has collided three
times (`git log -- .git-workflow.yml`). The pool is **capacity, not taxonomy**:
neither slot means anything, and a task takes whichever is idle. Do not name a
slot after a workflow or a kind of work.

**Why slots and not a worktree per task.** `worktree_policy: always` means every
modifying task builds a worktree, so every task would otherwise pay a full
`worktree_seed` copy *and* a `.pixi` solve — and this repo's `.pixi` is the
expensive half. Two slots amortize both into a one-time cost per slot that every
later task reuses. That payoff is independent of concurrency: it arrives on
purely sequential work too.

**Why slots and not the standing lanes they replaced.** From 2026-08-12 to
2026-08-17 this repo ran `lane/devmeta` and `lane/pipeline`, partitioned by
territory. The partition itself was healthy — 45%/55% of traffic, neither a
catch-all — but **37 of 162 commits touched both territories**, almost all "fix
the code, then close the board note". That is 23% of tasks paying a second
worktree visit, and the routing table needed a rule for spanning tasks because
they were the ordinary case rather than the exception. A territory split that
23% of work refuses to respect has stopped predicting the work. Under one slot a
spanning task is simply one branch touching both trees, with no visit and no
split.

The trade is deliberate: lanes made merge order irrelevant **by construction**,
because two territories cannot collide. A slot has no territory, so that
guarantee is gone and the declaration below replaces it.

**Declare the expected write set before editing — both slots occupied.** A
worktree isolates the index and `HEAD`; it does not make two edits to the same
contract independent. Before claiming the *second* slot, compare what the two
tasks intend to write:

- implementation paths and workflow entry points (`*.smk`, `blueearth_cst/**`);
- shared seams — `blueearth_cst/shared/`, above all `snake_utils.py` (parses
  every Snakefile's config) and `interchange_contracts.py` (**is** the
  wf1→wf2/wf3 seam). 72–85% of each workflow's commits touch `shared/`, so two
  workflow-scoped tasks are **not** independent by default;
- `config/**`, `test_case/snake_config_*.yml`, and the schema they validate;
- the exact test modules each task expects to edit; and
- mutable state outside the checkout — `project_dir`, `.snakemake` locks, the
  shared Julia depot under `~/.julia`.

State that set in the task brief or board note **before editing**. A `git diff`
is useful once edits exist but cannot replace the declaration: an untouched file
may still be the next file both tasks intend to change.

Disjoint sets run concurrently. **When the sets overlap, do not run them in
parallel** — serialize them in one slot, or consolidate into one task. When one
task must change a shared contract several tasks need, land that contract as the
smallest valid base change first, then rebase the others onto it. Never let two
slots invent competing versions of the same seam.

Because spanning tasks are the ordinary case here (37 of 162 commits), the
common shape is one task in one slot touching both code and `dev/**` — that is
fine and needs no declaration. The declaration is for the moment a *second* slot
is claimed while the first is still occupied.

**Occupancy is read from Git, not from a convention.** `git worktree list` is
the roster: a slot showing `(detached HEAD)` is idle, and one showing a branch
is occupied. The old gitignored `.lane-claim` marker is retired — it existed
only because a permanent branch gives Git no liveness signal. `slot-start`
refuses an occupied or dirty slot, so a crashed session stays visibly occupied
and is never silently reused.

**Lifecycle.** Claim, work, land, park:

```bash
S=~/workspace/brain/artifacts/skills/git-workflow/scripts/worktree-session.py
git worktree list                     # pick an IDLE slot: "(detached HEAD)"
python $S slot-start --slot <slot> --task <task> --type <type> --base main -- codex
# ... work, commit, then land the branch from the PRIMARY checkout ...
python $S slot-park --slot <slot> --base main
```

`slot-start` also compares each `worktree_seed` against the primary at claim
time: a copy the primary has moved past is refreshed, and a copy that *leads*
the primary is left untouched and reported — the fixture-drift failure recorded
in `dev/tasks/t2608121258-*` is that second case, and it must never be
overwritten. `slot-park` refuses to park until the branch is an ancestor of
`main`, so an unlanded slot cannot be quietly recycled.

**Routing a task.**

| Situation | Action |
|---|---|
| Any modifying task, a slot idle | Claim whichever slot is idle. Neither is reserved for anything. |
| The other slot is occupied | Compare the two expected write sets first (above). Disjoint → run both. Overlapping → serialize in one slot. |
| Both slots occupied | Report it. Wait, postpone, or take a transient worktree from `main` for urgent work. |
| Tiny, complete, verified | `main` directly, per the ordinary landing choice. |

`todoboard render` regenerates `dev/TODO.md` from the notes, so **two slots must
not run it concurrently** — they would race on a generated file neither edits by
hand. Land one slot's board change before rendering in the other.

**Why the pipeline was never split further.** Kept from the lane analysis
because it still governs how work is scoped: only 8 of 68 package-touching
commits touch more than one workflow, but 72–85% of each workflow's commits also
touch `blueearth_cst/shared/` — wf2 has *zero* commits that don't. `shared/` is
145 file-touches, larger than any single workflow territory, because
`snake_utils.py` parses every Snakefile's config and `interchange_contracts.py`
*is* the wf1→wf2/wf3 seam. Likewise 15 of the 35 plot-touching commits (43%)
also touch non-plot code — `cartographic_map.py` is drawn through by rules 1.12
and 1.13, so a change there is never figure-local (see *Figures are terminal
artifacts*). Standardizing every figure is a **sweep**: it edits call sites
everywhere by definition. Both facts are why no territory partition held here.

### Validation ladder — match the check to the blast radius

A slot's task branch is isolated from `main`, so a mistake is contained and
cheap to revert. Spend validation time accordingly: **unit tests while iterating, the
cheap whole-suite tier at the merge, the expensive one only when work leaves
this machine.** Re-running the full suite after each incremental edit is the
failure mode to avoid — it re-proves what the previous run already proved.

| When | Run |
|---|---|
| While iterating | Only the tests covering the file you changed (`pytest tests/test_<module>.py`). Nothing else. |
| Before a commit | Add `pytest tests/test_cli.py` **if** a Snakefile, a `script:` signature, or **a rule's declared input** changed — the last one because `test_cli` dry-runs all four entry points, so it is the only place a malformed `config/defaults/*.yml` surfaces; no fast-tier test parses those. Otherwise the module's own tests are the gate. If you wrote Python, `pixi run lint` and `pixi run format-check` — both are CI gates, and both are near-instant. `pixi run format` fixes the second. |
| Before merging the branch | `pixi run test-fast` once — **a few minutes**. Skip it entirely for a docs-, `dev/`- or config-scaffold-only branch, which no test imports. |
| **Before pushing `main` to `origin`** | `pixi run test-full` — **~10–15 min**. The authoritative gate, and the only one that runs the workflow/process-contract tier. |
| **After a push** | **Read the run it triggered** — `gh run list -L 1` / `gh run watch`. See below; this is not optional. |
| Before a milestone seal / after touching numeric outputs | `check_baseline.py check`, plus `semantic_tree_diff.py` if the tree shape moved. |

**Why the split lands there.** `test-fast` deselects **55 tests — under 3% of
the suite — and they cost the large majority of the runtime**. They are exactly
the `workflow_contract` and `process_isolation` markers. Paying several times
the wall-clock for that last 3% on every merge, when every task is a merge, is
the cost that was actually being paid. Selecting "only the relevant test files"
instead would save little over running the whole cheap tier, in exchange for a
judgment call per task and the cross-module regressions that judgment misses —
so run everything cheap, and tier the expensive part.

Times here are **orders of magnitude, deliberately**. Earlier revisions pinned
them to the second (`1:29`, `7:07`, `427 s`); by 2026-08-12 every one was off by
2–3×, and a stale number is worse than a vague one precisely because someone
uses it to decide whether a gate is affordable. Do not restore exact seconds —
the test count and the marker names are the durable facts, the clock is not.

A push is the escalation trigger because it is where work leaves this machine:
before it, `main` is local and revertible; after it, CI, the other platform leg,
and anyone else's clone are downstream. The pin in `.testing-policy.yml`
(`scope: rapid`) declares that posture, and `testing-policy`'s own `rapid` →
`release` boundary is the same line.

**The residual risk, stated plainly:** a `workflow_contract` regression can now
sit on local `main` across several merged branches until the next push, so
bisecting it spans those branches rather than one. That is the deliberate trade
for the speedup — and `auto_push: false` means the push is a decision you make,
which is precisely when the extra minutes are worth spending. Run `test-full` at
the merge anyway, not just at the push, when a branch touched a Snakefile, a
`script:` signature, or `shared/`; those are the paths that tier exists to guard.

- `--dry-run` before running and after editing any rule, to validate the DAG.
- If a run crashed and the workdir reports as locked, `--unlock` before retrying.
- **Run WF1 with `--notemp` when the run feeds `check_baseline.py`.** Rule 1.14
  declares wflow's `run_default/output.csv` as `temp()`, so a normal run deletes
  it once rules 1.14b and 1.15 have consumed it — and that file is the manifest's
  wf1 discharge target. Without the flag the gate fails "target missing on disk",
  which reads as a defect and is not one. The derived `output_q.csv` is not a
  substitute: it is rounded to 5 decimals, coarser than the drift the tolerance
  comparator exists to catch. `--notemp` is also the flag for iterating on a
  rule-1.15 evaluation figure, which otherwise re-runs the whole model.

### Which config to run — rapid by default, comprehensive selectively

The ladder above governs `pytest`; this governs the pipeline itself. Default to
`test_case/snake_config_rapid.yml` and reach for `snake_config_baseline.yml`
only when the run's NUMBERS are the point.

| Config | `project_dir` | Run it for |
|---|---|---|
| `snake_config_rapid.yml` | `test_case/test_rapid` | anything you want to watch EXECUTE — a rule you edited, a DAG check, a WF3 smoke run, a figure render |
| `snake_config_baseline.yml` | `test_case/test_local` | recording or checking `dev/baseline/manifest.json`, `tree-check`, a milestone seal, any number you will quote |
| `snake_config_wf2_fast.yml` | `test_case/test_dev` | WF2 code iteration only — 2 series, and it drops `st_0` |

Rapid costs ~2.6× less wflow time (10 members × 9 forcing years, vs 14 × 17) and
~1.7× less weather generation (46 generated years, vs 78). **The horizon, not
`run_length`, is what moves the second number**: `compute_nr_years` anchors the
generated series at 2010, so it spans 2010 → `horizontime_climate` +
`run_length`/2 and shortening the run alone barely touches it.

Rapid is CHEAP, not NARROW, and the difference is load-bearing. It keeps
`run_historical: true`, because `st_0` is what the two class-C month indicators
are derived *from* — `false` drops 2 of 11 `q` metrics with nothing reporting it
(the R11 P3 case `interchange_contracts.py` was hardened against) — and it keeps
two CMIP6 models, because a one-model config never runs the ensemble reduction.
A config that gives up coverage must say which, as `wf2_fast` does.

The baseline is recorded from `snake_config_baseline.yml` and nothing else;
never point `check_baseline.py` at the rapid tree.

### Read the CI run after you push

**CI was red on `main` for ten days and nobody noticed** (t2608071205): seven
runs failing the ruff gate from 2026-07-30, then — once that was fixed — three
more failing the ubuntu leg's unit suite, a *different* defect that had been
hiding behind the first. Nothing alerted; the gate simply went unread.

Two things follow, and they are complementary rather than alternatives.

- A **pre-push hook** runs the two ruff checks (~2 s) so that class cannot leave
  the machine. Install it once per clone: `git config core.hooksPath .githooks`.
  It is not installed by cloning — check `git config core.hooksPath` if unsure.
- **Read the run anyway.** The hook is blind to everything platform-specific,
  which is exactly what the second outage was: four tests asserting Windows path
  spelling, green locally and red on ubuntu forever. A green local suite is not
  evidence about the other leg, and the ubuntu leg is the only place linux-64 is
  exercised at all (`dev/roadmap.md`, "Deferred: Linux replication").

**`gh` talks to the WRONG REPO here until you tell it otherwise, and fails
silently when it does.** This clone has two remotes — `origin`
(`tanerumit/blueearth_cst`, where CI runs) and `upstream`
(`Deltares/blueearth_cst`) — and `gh` resolves to `upstream`. So a bare
`gh run list` queries Deltares, exits 0 and prints nothing, with runs sitting
on origin. Read that literally and you conclude CI has never run, which is
worse than not looking. Diagnosed 2026-08-12; it is what the earlier note
"`gh run list` does not work in this repo" was actually describing.

Fix it once per clone, next to `core.hooksPath`:

```bash
gh repo set-default tanerumit/blueearth_cst   # writes remote.origin.gh-resolved
```

After that every `gh` verb works bare. Until then — and in any script that
must not depend on local config — pass `--repo tanerumit/blueearth_cst`
explicitly, or go to the API:

```bash
# the latest run, whatever branch it was on
gh api "repos/tanerumit/blueearth_cst/actions/runs?per_page=1" \
  --jq '.workflow_runs[0] | "\(.head_sha[0:7]) \(.status) \(.conclusion)"'

# which STEP failed, per leg -- the only view that separates a lint failure
# from a suite failure from a platform-specific one
gh api "repos/tanerumit/blueearth_cst/actions/runs/<id>/jobs" \
  --jq '.jobs[] | "\(.name) -> \(.conclusion): " +
        ([.steps[] | select(.conclusion=="failure") | .name] | join(", "))'
```

Do **not** filter by `head_sha=<short sha>` — that parameter needs the full
40-character value and silently matches nothing otherwise, which reads as "the
run has not started yet" and polls forever.

### Figures are terminal artifacts

No rule consumes a `.png`/`.pdf` under `project_dir`, so a figure change cannot
propagate into a number. Do **not** run the validation suite or the baseline for
a figure-only change — verify it by *rendering it and looking at it*, which is
the only check that can actually catch a bad figure. `check_baseline.py` excludes
figure targets by default (`--include-figures` restores them).

**The gate for a figure change, in full:** (1) the unit tests of the changed
module, (2) the figure renders without an exception, (3) the rendered PNG is
published as an **Artifact** — a self-contained HTML page with the image
embedded as a base64 `data:` URI — so the owner inspects it in a browser. Never
byte-compare renders, scrub timestamps to force reproducible bytes, or run the
baseline or the full suite to "confirm" a figure. Renders here are ~0.3 MB, so
embedding needs no downscaling; side-by-side before/after panels are welcome,
being visual comparison rather than a byte check. Render the LAYER-RICH fixture
(`test_case/basin_map_fixture`, five subcatchments + gauges), not
`test_case/test_local` — the latter has one outlet and no gauges, so most layers
are simply absent from the image and cannot be judged.

For the basin map, render it WITHOUT a WF1 run:
`dev/scripts/preview_basin_map.py` drives `cartographic_map.py`'s tunable block
from the command line against a model already on disk (`--list`, `--set NAME=VALUE`,
`--sweep NAME=V1,V2,...`). Anything assembled from those tunables must be derived
in a function, not frozen into a module constant — a constant snapshots its
inputs at import, so the override would silently do nothing.

**The trap:** "I changed it for a figure" is not the same as "it is a
figure-only change". A shared helper edited in service of a plot
(`shared/snake_utils.py`, `shared/plot_utils.py`, `shared/cartographic_map.py`)
is a contract surface with other callers, and takes the normal ladder above.
`cartographic_map.py` is the one every map figure draws through — rule 1.12's
basin map and rule 1.13's three forcing maps — so a change there is never
figure-local.

## Hard Constraints

- **IMPORTANT: Julia is not in the pixi env** — it is juliaup-managed and must
  already be on `PATH` (conda-forge has no win-64 Julia build). Do not try to add it
  via pixi.
- Do not commit run outputs written under `project_dir`, or hand-edit `pixi.lock` /
  `Manifest.toml`.
- Stay within CST's automation scope — this repo is the workflow engine only. Define
  config/setup (`config/defaults/wflow_build_model.yml`, data catalogs, `setup_*`
  blocks, `wflow_sbm.toml`-affecting steps) using hydromt / hydromt_wflow / Wflow
  conventions verbatim: CSDMS Standard Names (`hydromt_wflow/naming.py`), their YAML
  schema, their catalog format. Do not re-engineer how hydromt handles data, how
  `setup_*` methods work internally, or how Wflow parameterizes physics. Verification
  may *read* upstream docs to validate our config but must never patch upstream; a
  genuine hydromt/wflow bug is flagged upstream or worked around in our own code
  (`blueearth_cst/`, Snakefiles, `dev/scripts/`), never inside a vendored package.

## References

- `README.md` — the overall pipeline and how the four workflows fit together;
  start here.
- `docs/cst-toolbox-technical-note-2025.md` — stress-test method and design
  rationale; read before changing *what* a workflow computes.
- `docs/install.md`, `docs/env_setup_notes.md` — read when pixi / R / Julia setup or
  env activation misbehaves.
- `docs/hydromt-user-guide/00-index.md`, `docs/hydromt-architecture.md` — read when
  editing model-build config, data catalogs, or region setup.
- `docs/hydromt-wflow/getting-started.md`, `docs/hydromt-wflow/user-guide.md`,
  `docs/hydromt-wflow/api.md` — read when a build/update/clip step touches the
  hydromt_wflow plugin (`api.md` for exact signatures).
- `docs/wflow-user-guide/00-index.md` — read when editing `wflow_sbm.toml`, warm
  states, or Wflow run config.
