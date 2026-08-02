# R07 — Project layout — design

**Status. ACCEPTED 2026-07-28.** Approved by the owner at gate G2 of a
`design-review-loop` run (`r07-project-layout`). Implementation is a separate
`task-brief` handoff; this is an accepted plan, not yet built.

**How it was reviewed.** Four versions, 44 findings, all dispositioned and none
rejected: a three-lens internal panel (34 findings — 7 blocking, 20 major, 7
minor) followed by two external cross-vendor rounds (7 findings, then 3). The
external round cap of 2 was exhausted with round 2 unconverged, so the owner
arbitrated its three surviving findings **accepted, fix required**, and those
rulings stand in place of the reviewer verdict the cap forecloses.
The arbitration revision's changes therefore carried **no external reviewer
verdict** when this document was accepted. **A post-acceptance verification pass
has since closed that gap** (2026-07-28): an external reviewer confirmed two of
the three arbitrated fixes resolved and both post-acceptance editorial
corrections correct, found the third fix sound but under-specified, and returned
four findings — all accepted and folded in, see the revision log. That pass was
**not** a further review round; the cap stands and its findings were owner
rulings, not automatic rework.

The full audit trail — verdict table, the internal panel's aggregated index, both
external rounds verbatim, the 44-row finding ledger, the verification pass, and
every gate and arbitration ruling — is
`dev/r07/project-layout-design-review-record.md`. The framing the run was
approved against is `dev/r07/project-layout-intake.md`.

**Genre: decision-record** (a layout refactor), mirroring the R3/R4/R5/R6 house
pattern under `dev/r0#/`.

**Scope authority.**
- `dev/reviews/2026-07-25_post-r6-assessment.md` — the observation register
  (O-01 … O-24) from the owner's post-R6 assessment. Repository-side items.
- `dev/r07/2026-07-26_project-output-layout.md` — the working note this design
  supersedes. Artifact-side items, the question log, and the cost analysis.
- Prior accepted layouts this design revises: `dev/p31/experiment-structure-design.md`
  §2 (the `project_dir` tree), `dev/r06/structural-refactor-design.md` (the
  repository tree), and `dev/p32a/climate-analysis-design.md` §"Why a separate
  wf1 extraction and NOT reuse of the P3-1 keyed store" — **explicitly
  superseded by B1 below, with a point-by-point rebuttal**.

This doc is self-contained: a reviewer needs only this file and the cited paths.

---

## Goal

One coherent layout across both halves of the system, governed by stated
principles rather than accretion:

- **The toolbox** (the repository) contains source, configuration, and templates —
  no basin data, no run artifacts.
- **The artifacts** (`project_dir`) are organised by *producer*, so a reader can
  tell what made a file by where it sits, and **engine-shaped artifacts are
  separable from generic ones**, so an engine's subtree can be relocated,
  rebuilt, or replaced without moving generic climate data.

*Narrowed from v1 (arch-8).* v1 claimed "a second modelling engine can be added
without inventing a new layout". The delivered tree does not support that claim:
hydrology appears twice, in two shapes, at two levels — `hydrology_model/` at the
`project_dir` root (the hydromt `model_root`, upstream-shaped) and
`hydrology_runs/` inside `experiments/<id>/`. A second hydrology engine would
have no rule to appeal to for which shape to copy, and would collide on the
domain-descriptive name `hydrology_model/`. That is the **structural** half of
the engine-naming question the owner parked at G1 (OQ-1); it is deferred beyond
R07 together with the naming half, and recorded as a stated limitation rather
than an implied guarantee. R07 delivers separability, not extensibility.

## Why now

R6 closed the repository restructure and P3-1/P3-2a closed the experiment
restructure, but each left residue the other could not see, and the owner's
post-R6 assessment surfaced it as a single class of problem:

- Basin-specific data sits in the repository root (`data/`), contradicting the
  rule that a run writes outside the repository tree.
- Generated artifacts sit in the repository root (`dag/`, a stray `dag_model.png`).
- The repository's own test fixture is called `examples/` while the real examples
  live in `docs/notebooks/`.
- Inside `project_dir`, figures follow three different conventions — centralized
  for wf1, distributed for wf2, dumped at the root for wf3.
- Two directories hold the same extracted climate grid, with a shipped `allclose`
  check to prove it (`dev/p32a/climate-analysis-design.md:385,407`).

None of these is urgent alone. Together they are cheapest to fix at once, because
almost every one of them moves a path that `dev/baseline/manifest.json`
fingerprints — see "Behaviour-preservation stance" below.

## Principles

Four invariants. Most of the tree follows from them. All four are stated with
their exemptions inline, because a principle contradicted by the tree it governs
cannot decide the next layout question (risk-4, arch-7).

**P1 — Figures attach to what they depict.** Every figure lives in a `plots/`
directory that is an immediate child of the subtree whose artifacts the figure
depicts. There is no project-level `plots/`. Subfolders *inside* a `plots/` leaf
are allowed — the rule constrains where the leaf attaches, not its internals.
`plots/` holds figures only; no CSVs.

*Why "depict" and not "produce" (risk-4).* v1 said "attach to their producer",
but `basin_area.png` is produced by rule 1.12 from `staticgeoms/outlets.geojson`
and depicts the model, not its evaluation; under the producer reading it had no
home, and v1 filed it under `evaluation/plots/` — a topic bucket. Under P1 as
restated it lands in `hydrology_model/plots/`, and `evaluation/plots/` holds only
run-evaluation figures. `hydrology_model/` therefore carries three `plots/`
leaves — `plots/` (the model), `forcing/plots/` (model inputs),
`evaluation/plots/` (the run) — one per depicted subject, which is exactly what
P1 asks for.

**P2 — One producer *definition* per artifact.** No artifact is computed by two
different definitions. A single producer **may** be declared in more than one
workflow, provided (a) both declarations are generated from one shared
producer-contract spec so they cannot drift, and (b) their inputs and params
are **identical — including ordering-only edges**. No per-workflow `ancient()`
asymmetry is permitted.

*Why the restatement (GA-1, repo-2).* v1's P2 read "No artifact is computed twice
by two workflows", which the owner-selected fix for B1 visibly contradicts: the
collapsed climate store is declared in both `Snakefile_model_creation` and
`Snakefile_climate_experiment`. The hazard P2 exists to prevent is *two
definitions disagreeing about content*, not two declarations of one definition.
P2 as restated forbids the hazard and permits the fix; B1 states the mechanism
that enforces (a).

*Why (b) tightened (ext1-02, corrected this round).* v2 permitted "an
ordering-only `ancient()` edge" on the claim that `ancient()` suppresses both
mtime- and input-set-triggering. That claim is wrong, verified against the
pinned Snakemake 9.6.2: `ancient()` suppresses only the mtime trigger.
`persistence._input()` iterates **every** `job.input` with no `is_ancient`
exclusion, and `_input_changed()` fires on `recorded != _input(job)` — so a
file flagged `ancient()` stays in the recorded input set. Both workflows run
from the repo root and share one `.snakemake` metadata store keyed by output
path, so a store produced by wf1 with no inputs and then encountered in wf3
with an extra `ancient(guard_ok)` edge trips the input-set trigger and
re-extracts — and the next wf1 invocation trips it back. The repo's own
comment at `Snakefile_climate_experiment:198-201` is consistent with this
reading: its input-set protection comes from the guard path being
*experiment-invariant* (same path ⇒ same input set across experiments), not
from `ancient()`; v2 misread that comment. Consequence: the two declarations
must carry the **same input set**. v3 achieved that with no inputs at all; the
arbitrated ext2-01 fix keeps the symmetry but makes the set the **singleton
`{data catalog}`, declared identically in both DAGs** — a symmetric set cannot
reproduce the oscillation, which required asymmetry, while the catalog edge
restores the freshness trigger the empty set gave up (see B1).

**P3 — Engine-shaped artifacts live inside their engine's subtree.** Anything in
a model-specific format (wflow forcing netCDFs, run directories, TOMLs; weather
generator configs and realizations) belongs under that engine's subtree. Generic,
engine-independent data (raw climate extractions, projections) stays outside it.

An engine subtree's **default** shape is `config/`, `output/`, `plots/`, `_work/`.
Two qualifications, both load-bearing:

- `plots/` and `_work/` are **optional when empty** — a subtree with nothing to
  put in them conforms.
- A subtree **may add** engine-mandated directories (`forcing/`, `run_default/`,
  `evaluation/`), and is **exempt** from the default shape entirely where an
  upstream tool owns the directory contract. `hydrology_model/` is the one such
  case: it is the hydromt `model_root`, whose immediate children
  (`wflow_sbm.toml`, `staticmaps.nc`, `staticgeoms/`, `hydromt.log`,
  `hydromt_data.yml`) are written and read by hydromt and Wflow, not by CST.
  Reshaping them is out of scope per `AGENTS.md` § Hard Constraints.

*P3 supersedes a first-draft principle that read "`hydrology_model/` is
upstream-governed; CST adds nothing to it." That was factually wrong: the live
TOML already reaches outside the model root (`path_forcing =
"../climate_historical/wflow_data/inmaps_historical.nc"`), so wflow does not
dictate where forcing lives — CST chose that path. The genuinely upstream-governed
surface is narrower: the TOML **schema** (CSDMS names, `[input.static.*]`),
`staticmaps.nc` internals, and hydromt's build semantics. Not the directory
layout. P3-1 already proved this class of move by relocating run directories and
rewriting the pointers.*

**Generated vs. copied configs (risk-4).** P3 gives no home to configs that are
*generated* at run time in an engine's schema — `wflow_build_model_run.yml` (rule
1.02 → 1.03) and `wflow_build_forcing_historical.yml` (rule 1.07 → 1.08). They are
neither engine artifacts (they are hydromt build instructions, not model state)
nor copied provenance. Rule: **generated run-time configs live in
`<project_dir>/config/generated/`; verbatim snapshots of shipped templates live in
`config/templates/`.** v1 filed the two generated files under `templates/`, which
contradicts the Goal's "a reader can tell what made a file by where it sits".

**P4 — Climate analysis must be possible without a model.** A full climate
analysis runs from region + catalog alone, with no wflow setup or run. Climate
figures are produced from the extracted climate store, never from wflow forcing.
Retained from P3-1: **each `experiments/<id>/` is self-contained and reproducible
from its own directory.**

---

## What changes — A. The repository

| # | Change | Rationale |
|---|---|---|
| O-01 | **Delete `data/`.** Ship header-only schema templates at `config/templates/observations/{output_locations.csv, observations_timeseries.csv, README.md}`. Real basin data lives in the project folder, referenced by absolute path. **The two live consumers are retargeted in the same commit, not left dangling** — see "The Linux entry path" below (ext1-05). | 653 KiB of Gabon-specific CSV in the toolbox source tree. Consumed by `config/workflows/snake_config_model_test_linux.yml:25-26` and mounted by `scripts/run_snake_docker.sh:7` — v2 called both "parked", which is a validation status, not a licence to break them. |
| O-02 | **DAG renders move to `<project_dir>/dag/`.** `scripts/run_snake_test.cmd:32` retargets; the `dag/` entry leaves `.gitignore:136`; `README.rst:269,285,298` and six notebook cells stop writing to the repo root. | The DAG is a function of the config, so it belongs with that config's artifacts. The README/notebook commands are the actual source of root clutter. |
| O-05 | **Delete `docs/config/`** (16 tracked pre-R6 duplicates of `config/`). Update `AGENTS.md`'s `docs/` description and `MIGRATION.md:173`. | Two of them still point at the `data/` path O-01 removes. Kept byte-identical by hand until the R01 config restructure ended that. |
| O-20 | **`examples/` → `test_case/`.** `.gitignore:124` follows. | The directory holds the local test fixture, not examples; the real examples are `docs/notebooks/`. |
| O-21 | **`config/workflows/snake_config.template.yml:15-18`** ships an outside-the-tree `project_dir` placeholder, matching its own comment. | The template is the file a new user copies; it currently teaches the opposite of what it documents. This is the origin of the tier confusion. |
| O-22 | **Add `warn_if_project_dir_in_repo(project_dir, repo_root)`** to `blueearth_cst/shared/snake_utils.py`, called at parse time from all three Snakefiles with `workflow.basedir` as `repo_root`. Warns, never raises. Exemption held as `_PROJECT_DIR_EXEMPT_NAMES = frozenset({"test_case"})`. | Makes the two-tier rule mechanical instead of documentary. |
| O-24 | **Declare the config-invariant missing plot outputs** on rules 1.11 and 1.13 while their paths move anyway. | `plot_map_forcing.py:167-187` writes three PNGs; rule 1.13 declares one. `plot_results` writes `clim_wflow_1_{month,year}.png` + `performance_metrics.csv`; only `hydro_wflow_1.png` is declared. Undeclared outputs are not cleaned on rerun and are absent from the baseline. |
| O-08 | **`plot_map.py:28-31` sentinel guard** — promoted out of the drive-by list into the commit that touches `plot_map.py`. | It is the *real* `"None"`-sentinel defect (repo-4); see § Risks. |
| O-25 | **`blueearth_cst/model/get_region_preview.py` is dead and broken** — record it, retire it in R07's B1 commit. | New this revision. It imports `hydromt.cli.api`, **removed in hydromt 1.x**; on the pinned hydromt 1.3.1 the module raises `ModuleNotFoundError` on import. No rule, no test, and no other module references it (only `MIGRATION.md:51` and historical dev notes do). v1 cited it as B1's model-free escape hatch — see B1. |

**O-22 signature (repo-12).** `snake_utils.py` has no notion of the repository
root, so a module-level constant can hold only the relative segment. Deriving the
root inside the module (`Path(__file__).parents[2]`) breaks if the package is ever
installed rather than imported from the tree, and an absolute constant is not
portable. Each Snakefile already has `workflow.basedir` in hand
(`Snakefile_model_creation:7,21`), so it passes it.

**O-24 scope, stated (repo-10).** The declared set is the **config-invariant
subset**. `plot_results.py` additionally drives `plot_basavg` (one PNG per
basin-average entry in `wflow_outvars`) and `plot_signatures`
(`signatures_{station}.png`, when observations exist and `nb_years >= 5`), and
`clim_{station}_{period}.png` is per-station, so a config with real gauges
produces `clim_wflow_2..N_*`. The seed fixture has an empty `ds_basin` and no
observations, which is why exactly 8 files sit in today's
`plots/wflow_model_performance/`. Deriving the full list at parse time from
`wflow_outvars` / `output_locations` (both already read at
`Snakefile_model_creation:51,54`) is possible but is a rule-shape change beyond
this milestone: R07 declares the config-invariant subset and records the
remainder as knowingly undeclared, so `--delete-all-output` completeness is
claimed only for the seed-config class.

**The Linux entry path — an explicit support decision (ext1-05).** The
milestone's non-goals park Docker (O-06) and Linux end-to-end validation
(O-18/O-19) because no Linux machine is available. v2 let that parking do
illegitimate work: O-01 deletes `data/observations/*` while
`snake_config_model_test_linux.yml:25-26` still points
`output_locations`/`observations_timeseries` at it, and O-20 renames
`examples/` while `run_snake_docker.sh:6` still mounts it (line 7 mounts the
deleted `data/`) — so the documented Linux/Docker entry path would fail on
paths this milestone removed, with no stated decision. The decision, stated:
**the Linux config and the Docker runner are retained and kept structurally
consistent at parse level; end-to-end validation stays parked.** Mechanically:

- Commit 2 (with the `data/` deletion): the Linux config's two observation
  keys move to the `None` sentinel — the same no-observations state the
  Windows seed config uses — and `run_snake_docker.sh` drops the `data/`
  mount.
- Commit 4 (with the fixture rename): `run_snake_docker.sh`'s `examples`
  mount follows to `test_case`.
- The parse-level check is real, not aspirational: `tests/test_cli.py` gains a
  dry-run of the three Snakefiles under
  `config/workflows/snake_config_model_test_linux.yml` (+ its Linux catalog).
  A dry-run needs no data files — catalog paths are params — and CI runs
  `ubuntu-latest`, so the check executes on every push. If implementation
  finds the Linux-config dry-run blocked for a path reason, the fallback is a
  referential-integrity test (the config and runner reference no repo path
  that does not exist), stated here so a downgrade is visible, not silent.

What this does *not* claim: that the Docker image or a full Linux run works.
That validation remains a non-goal; the support decision is that R07 leaves
the path structurally intact rather than known-broken. The behavioural delta —
the Linux test config loses its observation-driven extras (signature plots,
gauge series) until real observation paths are configured — is the documented
cost, consistent with O-01's rule that real basin data lives outside the
toolbox and is referenced by absolute path.

**Two-tier `project_dir` rule, made explicit.** Tier 1, production: an absolute
path outside the repository tree — already supported, since `project_dir` is
consumed as a raw f-string prefix. Tier 2, the single test fixture: repo-relative
and gitignored. Tier 2 exists because the baseline seed config is *tracked*, and a
tracked config cannot carry a machine-specific absolute path; the alternative is
env-var indirection, which reintroduces the untracked-local-config pattern already
rejected.

### Kept as-is, with the reasoning recorded

These were questioned during the review and deliberately **not** changed. Recording
them here so the questions do not recur:

- **The nested `blueearth_cst/` package** (O-03). It *is* the Python package root;
  the nesting is what makes `from blueearth_cst.shared.snake_utils import …`
  resolve. A `src/` layout needs an installed package, rejected at
  `dev/r06/structural-refactor-design.md:368-376`.
- **Three homes for executable files** (O-23). The split is by invocation model:
  `blueearth_cst/` is executed *by Snakemake*; `scripts/` executes the pipeline;
  `dev/scripts/` inspects or maintains the repository. Documentary fix only — one
  contrastive line in `AGENTS.md`, which currently describes all three by
  *audience* and so fails to discriminate.
- **The three Snakefiles at the repo root** (O-23a). Moving them into `scripts/`
  is a category error — they are the pipeline *definition*, not a runner. Moving
  them anywhere costs a `../` prefix on all 31 `script:` directives (resolved
  against `workflow.basedir`), `.parent` on three `sys.path` inserts, and ~88
  references across 28 files, to remove three root files.
- **`{project_dir}/config/` snapshots.** Not merely provenance: the wf3 drift
  guard compares against `snake_config_model_creation.yml`, four of the eighteen
  baseline fingerprints *are* these files, and `file_digest_or_absent()` reads
  them at parse time.

---

## What changes — B. The artifacts (`project_dir`)

```
<project_dir>/
  config/                                   # provenance snapshots, split by kind
    runs/       snake_config_model_creation.yml, snake_config_climate_projections.yml
    catalogs/   deltares_data.yml, cmip6_data.yml
    templates/  wflow_build_model.yml, wflow_update_waterbodies.yml
    generated/  wflow_build_model_run.yml, wflow_build_forcing_historical.yml

  climate_historical/                       # GENERIC, engine-independent (P3, P4)
    <key>/                                  # key = <clim_source>_<YYYYMMDD>_<YYYYMMDD>
      extract_historical.nc                 #   SINGLE store — one producer definition (P2)
      orography.nc                          #   chirps-branch sidecar, ONE filename (B1)
      store_region.geojson                  #   the model-free delineation the bbox came from
      .guard_ok
      plots/                                #   THE climate figures — source grid, model-free
                                            #   source_precip.png, source_temp.png, source_pet.png

  climate_projections/<clim_project>/
    timeseries/gcm_timeseries.nc
    summary/annual_change_scalar_stats_summary.{nc,csv}, *_mean.csv
    plots/

  hydrology_model/                          # wflow ENGINE subtree — P3-exempt (hydromt model_root)
    wflow_sbm.toml  staticmaps.nc  staticgeoms/  hydromt.log  hydromt_data.yml
                                            #   ^ hydromt model root == this dir
    plots/                                  #   basin_area.png — depicts the model (P1)
    forcing/inmaps_historical.nc            #   moved from climate_historical/wflow_data/
    forcing/plots/                          #   model-input QA figures: precip/temp/pet.png
    run_default/                            #   historical simulation
    evaluation/
      performance_metrics.csv
      plots/                                #   hydro_wflow_1.png, clim_wflow_1_{month,year}.png

  logs/  benchmarks/                        # wf1 + wf2

  experiments/<experiment_id>/              # id suggested as <project_name>_<YYYYMMDD>
    config/snake_config_climate_experiment.yml
    data_catalog_climate_experiment.yml
    .project_consistency_ok

    weather_generator/                      # ENGINE subtree
      config/weathergen_config.yml
      output/                               #   rlz_<r>_cst_<c>.nc, sim_dates.csv,
                                            #   resampled_dates.csv
      plots/                                #   obs_power_spectra, warm_annual_*
      _work/                                #   cst_<c>.csv, weathergen_config_rlz_<r>_cst_<c>.yml

    hydrology_runs/rlz_<r>/                 # ENGINE subtree, per realization
      config/cst_<c>.toml
      forcing/inmaps_cst_<c>.nc             #   temp() — wflow-grid downscaled forcing
      output/cst_<c>.csv, outstates_cst_<c>.nc

    indicators/                             # was model_results/
      Qstats.csv  basin.csv  RT_*.csv
      plots/                                #   response-surface figures
    logs/  benchmarks/
```

### The substantive moves

**B1 — Collapse the two climate stores into one shared-producer store (P2, P4).**
`wf1_raw/` and `<key>/` hold the same grid; P3-2a shipped an `allclose` check
between them. This is the owner-selected resolution (GA-1, `dev/r07/project-layout-design-review-record.md` § Gate and arbitration record
2026-07-28) of the panel's highest-confidence blocking group (risk-1, arch-1,
repo-2), which found that v1 never named the producer and that every obvious
assignment breaks a stated commitment.

*The producer.* **One rule definition, over the model-independent region
specification + catalog, declared in both `Snakefile_model_creation` and
`Snakefile_climate_experiment`.** Concretely:

- A shared spec helper `climate_store_spec(project_dir, model_region,
  clim_source, historical_window, data_sources, hydrography, basin_index)` in
  `blueearth_cst/shared/snake_utils.py` returns the **complete producer
  contract** (ext1-04): the store dir, the rule's script path
  (`blueearth_cst/climate_analysis/extract_historical_climate.py`), the inputs
  dict (the single catalog input — ext2-01, below), the outputs dict, and the
  params dict. Both Snakefiles splat every spec-driven field into their
  declaration (`script: SPEC.script`, `input: **SPEC.inputs`, `**SPEC.outputs`,
  `**SPEC.params` — script paths resolve against the declaring Snakefile, and
  both Snakefiles sit at the repo root, so one relative path serves both). The
  one field the rule grammar cannot take from a dict — the rule name
  (`extract_climate_grid` in both) — is enforced by the contract-equality test
  in the verification plan, which parses both workflows and compares the full
  normalized contract: rule name, script, input set, output paths, params,
  **and — per the arbitrated ext2-02 fix — every content- or
  execution-affecting directive**: `conda`, `container`, `containerized`,
  `envmodules`, `wrapper`/`notebook`, `shadow`, `threads`, `resources`,
  `priority`, `retries`, `group`, `cache`, `wildcard_constraints`, `handover`,
  `localrule`, `default_target`, `template_engine`, `cwl`. *The last six were
  added 2026-07-28 (pv-3): they exist in the pinned Snakemake 9.6.2 grammar and
  were missing from the enumeration, so the "every directive" claim was
  unsubstantiated. The deny-by-default rule below would have caught a symmetric
  use of one of them — but by **failing** it rather than normalizing and
  comparing it, which is a false red on legitimate use.* Build the universe
  against `RuleInfo`'s field set and the effective workflow-level rule state
  rather than from this list alone, so the enumeration is a check on the
  derivation and not its source. The allowed-local set is
  **deny-by-default**: exactly `message:`, `log:`, and `benchmark:` may differ
  — none content-determining, none participating in any rerun trigger
  (Snakemake records the log list but compares only code, input, params, mtime,
  software-env) — and any directive carrying a non-default value on either
  declaration that is outside the test's known-directive universe **fails the
  test**, so a future Snakemake version adding a directive surfaces as a loud
  failure rather than silently widening the hole. The directive sweep matters
  beyond content: `conda`/`container` feed Snakemake's software-environment
  rerun trigger, so a one-DAG environment change would re-fire extraction on
  every alternation exactly as ext1-02's input asymmetry did. Verified
  2026-07-28: neither Snakefile uses any of these directives today, so the
  sweep starts as absent-equals-absent. This is the mechanism P2(a) requires,
  closed over the whole declaration rather than outputs + params alone
  (ext1-04's gap in v2) or a name/script/inputs/outputs/params subset
  (ext2-02's gap in v3).
- Inputs: **exactly one — the data catalog — declared identically in both
  DAGs** (ext2-01, the owner-arbitrated fix). Rule 1.10's
  `ancient(f"{basin_dir}/staticmaps.nc")` and rule 3.02's
  `ancient(f"{basin_dir}/staticgeoms/region.geojson")` are removed (repo-2 asks
  this explicitly), **and so is rule 3.02's `ancient(f"{store_dir}/.guard_ok")`
  edge** — v2 kept it as "ordering-only", which ext1-02 showed re-triggers the
  producer on every wf1/wf3 alternation via the input-set trigger (see P2(b)).
  v3 then declared no inputs at all, which left catalog **content** with no
  freshness trigger — ext2-01's gap: params record only the catalog *path*, so
  an in-place catalog edit, or a switch of data behind an entry, left the store
  silently stale. The fix declares `catalog = <project.data_sources>` as a
  plain (not `ancient()`) `input:` in both DAGs, emitted by the spec's inputs
  dict so the symmetry is structural. **Symmetry is verified, not assumed**
  (the owner-mandated check, run 2026-07-28): wf1's `DATA_SOURCES`
  (`Snakefile_model_creation:31`) and wf3's (`Snakefile_climate_experiment:34`)
  read the **same `project.data_sources` config key** — a single catalog path
  (`config/catalogs/deltares_data.yml` on the seed config) — and today's two
  extraction rules already carry that identical value
  (`Snakefile_model_creation:241`, `Snakefile_climate_experiment:204`). The
  experiment-level catalog composed at `Snakefile_climate_experiment:344`
  belongs to rule 3.09 `downscale_climate_realization`, not the extraction,
  and never enters the producer's contract — so the symmetric-input route does
  **not** reintroduce the oscillation, and the digest-in-params fallback the
  arbitration held in reserve is not needed (recorded in Alternatives).
  Cross-config agreement is already mechanical: `project` is a guarded section
  of rule 3.00b, so an experiment config whose `data_sources` diverges from
  the wf1 snapshot fails the drift guard. The recorded input set is therefore
  the identical singleton in both DAGs — the ext1-02 oscillation required
  *asymmetric* sets and cannot reproduce — and "wf1 then wf3 reports nothing
  to be done" still holds by construction. The catalog path moves from
  `params` to the declared input (the script reads it from `snakemake.input`);
  the rest of the content-determining surface stays in params: the region
  string, `hydrography`/`basin_index`, `clim_source`, the window.
- *The freshness boundary, defined* (ext2-01's second sub-point). The catalog
  **file** is the boundary of the store's freshness contract. Editing the
  catalog in place now mtime-triggers exactly one re-extraction — closing a
  staleness gap that **predates R07**: today rule 3.02 carries the catalog
  only as a `params` path string, so an in-place edit retriggers nothing
  (driver evidence at arbitration, 2026-07-28). Data *behind* an unchanged
  catalog entry — a local file the entry points at, or a remote store — does
  **not** participate: enumerating catalog-resolved sources as DAG inputs
  would require parsing hydromt catalog semantics at DAG-parse time (outside
  CST's automation scope per `AGENTS.md` Hard Constraints), and remote
  sources expose no usable mtime. The supported way to record a data change
  behind a stable entry is to edit the entry (path, version, or meta) — the
  catalog-conventional signal, which the new input edge then picks up; the
  escape hatch for a truly in-place data mutation is an explicit
  `snakemake --forcerun extract_climate_grid`. Both are documented in the
  migration map rather than left silent. This is parity with every other
  hydromt-consuming rule in the repo, improved by one channel (the catalog
  file itself).
- *What replaces the guard edge.* The edge did two jobs; both survive it.
  **Store integrity:** with the extent now a pure function of params (and
  catalog freshness carried by the catalog input's mtime — ext2-01), Snakemake's
  params rerun-trigger re-extracts whenever the region specification changes —
  a diverged experiment config can waste one extraction before the guard kills
  the run, but it cannot leave a *silently stale* store, because the next
  invocation from a consistent config sees changed params and re-extracts
  (today's derivation, by contrast, reads the bbox from an `ancient()` model
  file and cannot re-trigger on a region change at all — the collapse closes
  that latent staleness gap). **Experiment gating:** every wf3 consumer of the
  store remains transitively guard-gated through the per-experiment sentinel
  chain that already exists — rule 3.06's `weagen_config` input comes from rule
  3.04, whose `consistency_ok` input is rule 3.00b's sentinel — so no
  experiment work downstream of the store can start before the guard passes.
  Rule 3.00b itself is untouched: it keeps both outputs, and `.guard_ok` is
  retained as the store-level receipt of the last consistency check (its DAG
  edge retires; the artifact and the P3-1 key contract stay).
- Rule 1.10 `extract_climate_grid_wf1` and the `climate_historical/wf1_raw/`
  directory are retired; rule 1.11's climate inputs
  (`_wf1_plot_clim_inputs`) repoint to the store.

*The delineation (reworked this round — ext1-01).*
`hydromt.model.processes.region.parse_region_basin(region, data_catalog=…,
hydrography_path=…, basin_index_path=…)`, with the two dataset names taken from
**two new optional keys in the model-independent `shared.basin` config block**
— `shared.basin.hydrography` (default `merit_hydro_ihu`) and
`shared.basin.basin_index` (default `merit_hydro_index`), both **catalog entry
names**, defaults equal to the shipped template's values. v2 read them from
`config/templates/wflow_build_model.yml` `setup_basemaps` (arch-1's suggested
pin, adopted in good faith), which ext1-01 correctly faults: it put a Wflow
build template inside the climate producer's contract, so a climate-only
execution stayed coupled to model-build configuration and a template edit
could change a supposedly model-independent artifact. The rework keeps
arch-1's *intent* — the store's basin and the build's basin cannot disagree —
while removing the coupling:

- The producer's contract is now `shared.basin` + catalog, nothing else. The
  hydrography keys are part of the region *specification* — they say which
  catalog hydrography gives the region string its meaning — and they live in
  the config block that `shared.basin.region` already occupies. wf3 never
  opens the build template; `build_config` leaves the spec signature. This is
  the reading of GA-1's "region + catalog only" that the route needs, stated
  rather than assumed: "region" is the region specification in `shared.basin`,
  not the bare string.
- **Agreement with the build is enforced, not shared-sourced.** Rule 1.02's
  script (`prepare_build_config.py`) already parses the build template to
  merge `region`/`res` into `setup_basemaps`; it gains the two names as params
  and **raises `RuntimeError`** — naming both files and both values — when the
  template's `setup_basemaps.hydrography_fn`/`basin_index_fn` disagree with
  `shared.basin`. A custom production template with different hydrography
  therefore fails loud at the first build step until the config says the same
  thing. The template stays the hydromt-conventional home of build datasets
  (`AGENTS.md` Hard Constraints: consume hydromt conventions verbatim);
  injection of the config values into the generated build config was
  considered and rejected (see Alternatives) because it silently overrides
  user template edits. `setup_rivers.hydrography_fn` ("should match basemaps
  source") stays an intra-template concern, out of R07's scope.
- **Cross-workflow agreement is already mechanical:** `shared.basin` is a
  guarded section of rule 3.00b (`Snakefile_climate_experiment:99,110`), so an
  experiment config whose hydrography keys diverge from the wf1 snapshot fails
  the drift guard with no new machinery. Absent keys leave the guard digest
  byte-identical (the digest serializes the config dict as-is), so the seed
  fixture and every existing config are unaffected.

The derived polygon is written as a second declared output,
`<key>/store_region.geojson`, so the bbox's provenance is on disk and the P4
assertion has something to inspect. This adds a second producer-written file to
the guarded store dir, which is safe: rule 3.00b compares **config digests**
and writes two *named* sentinels (`.project_consistency_ok`,
`{store_dir}/.guard_ok`); it never enumerates the store directory's contents,
so a new sibling file cannot perturb it.

*Correction to v1 (arch-1, O-25).* v1 cited
`blueearth_cst/model/get_region_preview.py` as evidence that model-free
delineation "already" exists. It is not usable: it is a standalone `argparse`
CLI wired into no rule, it carries its own hydrography default, it concatenates
river geometries into its output, and — verified during revision r1 — **it does
not import on the pinned hydromt 1.3.1**, because `hydromt.cli.api` was removed in
hydromt 1.x. Nothing references it, so nothing caught this. It is retired in B1's
commit and the v1 hydromt API replaces it.

*Rebuttal of `dev/p32a/climate-analysis-design.md` §"Why a separate wf1
extraction and NOT reuse of the P3-1 keyed store" (risk-1).* That accepted design
rejected the single store; R07 supersedes it, point by point:

| P3-2a's objection | Why it no longer holds |
|---|---|
| "wf1 runs before wf3; in a fresh project the keyed store does not exist at wf1 time, so a wf1 `input:` on it raises `MissingInputException`" | There is **no cross-workflow edge**. Each Snakefile declares the producer locally, so in each DAG the store has a local producer and a fresh wf1-only run builds it itself. Workflow order is unchanged. |
| "the store is produced by a wf3 rule that is guard-gated, so a wf1 dependency would invert the documented order" | The producer is workflow-independent and takes no guard edge in either DAG (ext1-02); wf3's experiment work stays guard-gated through the per-experiment sentinel chain, and wf1 neither reads nor writes `.guard_ok`. |
| "making wf1 the producer would re-architect the `.guard_ok` / `extract_climate_grid` / key contract and break wf3's value-identity by re-pointing rule 3.02's output" | Rule 3.02's **output path is unchanged** (`{store_dir}/extract_historical.nc`) and the key contract is unchanged. What changes is rule 3.02's *bbox source* — from the built model's `region.geojson` to the model-free delineation — which is measured, not assumed (see the empirical result below). |
| "accepted cost: one duplicate extraction per project; 'one store feeds all workflows' deferred to P3-2b/future" | R07 lands that north star early. The deferral was a scope judgement, not a technical barrier. |

*The bbox change is real and is a named exception.* Today rule 1.10 cuts to the
staticmaps bounds and rule 3.02 cuts to `region.geometry.total_bounds` of the
**built model's** `staticgeoms/region.geojson`. The R07 store cuts to the
**model-free** delineation's bounds — a third derivation, different from both.
GA-1 accepted this cost and ruled it must be proven, not assumed. See
§ "Behaviour-preservation stance", exception 3, for the proof and its branch.

*Empirical result, seed fixture, 2026-07-28* (read-only probe against
`examples/test_local`, hydromt 1.3.1, `config/catalogs/deltares_data.yml`):

| Bbox | (xmin, ymin, xmax, ymax) |
|---|---|
| `staticmaps.nc` raster bounds (today's wf1) | `(9.65833333316084, 0.34999999993263486, 9.858333333160658, 0.4833333332660743)` |
| `staticgeoms/region.geojson` bounds (today's wf3) | `(9.658333, 0.35, 9.858333, 0.483333)` |
| `parse_region_basin(...)` bounds (R07) | `(9.65833333316084, 0.34999999993263486, 9.858333333160658, 0.4833333332660743)` |

The R07 bbox is **bit-identical to today's wf1 bbox** on all four edges, and
differs from today's wf3 bbox by ≤ 3.4e-07° — the GeoJSON writer's 6-decimal
coordinate rounding, four orders of magnitude below the era5 source cell (0.25°)
and inside `prep_historical_climate`'s `buffer=1`. This is strong evidence for the
no-change branch; it is **not** the proof, because it measures bounds rather than
the extracted arrays. The proof is the merge comparison.

*The orography sidecar (repo-1).* The two stores name it differently: wf1 writes
`orography.nc` (declared output of rule 1.10, declared input of rule 1.11), wf3
writes `{clim_source}_orography.nc` (read back as a `params:` string at
`Snakefile_climate_experiment:331`). The collapsed store standardises on
**`orography.nc`** — the clim_source-independent form P3-2a ext2-1 already
introduced — and rule 3.08's `oro_path` params string repoints. Because the
sidecar exists only on the chirps / chirps_global branch while the seed config is
era5, no dry-run, test, or baseline check in the repo can see this; B1 therefore
ships a unit test over `prepare_climate_data_catalog.py` asserting the chirps
branch's catalog entry resolves to the emitted filename.

*The retired `allclose` check* returns as a unit test over the two bbox
derivations (`store_region.geojson` bounds vs `staticmaps.nc` bounds, per-edge
tolerance 2 × model resolution), preserving P3-2a's ext1-5 assertion in a form
that survives the collapse.

**B2 — `climate_historical/wflow_data/` moves into the engine subtree (P3).**
`inmaps_historical.nc` is wflow-shaped data. Under P3 it becomes
`hydrology_model/forcing/`; the TOML key edited is `path_forcing` only, but the
edit surface is five places — `blueearth_cst/shared/setup_time_horizon.py:51`,
`Snakefile_model_creation:198,210,305`, and
`tests/test_interchange_contracts.py:529`. Because the target moves *inside* the
hydromt model root, the new pointer is the relative `forcing/inmaps_historical.nc`,
a strictly better shape than today's `../climate_historical/…`. Consequence:
`climate_historical/` becomes purely generic and engine-independent, which is what
makes P4 reachable.

**B3 — Tier `climate_projections/` into `timeseries/` + `summary/` + `plots/`.**
`plots/` is already split; the gap is processed-vs-summary. No `raw/` tier: wf2
streams CMIP6 from GCS and never persists slices, and a placeholder directory with
no producer would not be created by Snakemake anyway. **The three PNGs under
`climate_projections/<clim_project>/plots/` do not move** (arch-10) — only the
three summary files do.

**B4 — Climate figures come from the climate store, never from wflow forcing
(P4).** The only genuinely forcing-only quantity is PET; the extraction carries
`precip`, `temp`, `press_msl`, `kin`, `kout`. Source-grid PET is computed on the
extraction grid using the source orography — **it need not match the build's PET**.
Climate figures are approximate quick assessments; the build's PET is the refined
model input.

*Three families, not two (arch-6).* v1's table named two; the tree holds three.
All three are retained, each answering a different question, and the source-grid
set is **filename-disambiguated** rather than relying on its parent directory
(risk-9), because a `pet.png` copied into a report or a GUI collector loses its
directory:

| Product | Question it answers | Grid | Needs a model? | Home | Filenames |
|---|---|---|---|---|---|
| Climate figures | what does the source climate look like? | source | **no** | `climate_historical/<key>/plots/` | `source_precip.png`, `source_temp.png`, `source_pet.png` |
| Model-parity climate figures | what climate did the model actually see, per station and period? | model | yes | `hydrology_model/evaluation/plots/` | `clim_wflow_<n>_{month,year}.png` |
| Forcing / model-input QA figures | did the downscaling to the model grid behave? | model | yes | `hydrology_model/forcing/plots/` | `precip.png`, `temp.png`, `pet.png` |

`clim_wflow_1_*` **survives** the new producer. Its question is not the source
grid's — it is model-parity, per station and period — which is the same reasoning
the owner used to retain the forcing/QA figures. Rule 1.11 keeps its producer and
inputs; only its output path moves (B10). Rule 1.13 keeps its input and producer;
only its output path moves.

*The new producer, named* (ext1-07): rule **1.15 `plot_climate_source`**,
declared in `Snakefile_model_creation` only (a single declaration — none of
B1's two-DAG machinery applies), script **new module
`blueearth_cst/climate_analysis/plot_climate_source.py`**, its three PNGs added
to rule `all` and to rule 1.14's gather inputs. Inputs (risk-9):
`<key>/extract_historical.nc` (+ the `orography.nc` sidecar on the chirps
branch) **plus the data catalog as params**, because on the era5 branch the
store carries no orography sidecar and source-grid PET needs `era5_orography`
from the catalog — as `extract_historical_climate.py` already does. This is
compatible with P4 ("region + catalog alone") but was absent from v1's input
list, so the P4 assertion test was under-specified. Because the rule's whole
subgraph is the B1 producer (whose sole input is the tracked catalog —
ext2-01) plus itself, invoking the figure
targets against `Snakefile_model_creation` needs **neither `hydrology_model/`
nor `config/templates/wflow_build_model.yml` on disk** — which is exactly the
form the P4 assertion takes in the verification plan (ext1-01's suggested
verification, adopted).

Remaining coupling to break, out of scope here:
`climate_analysis/subcatchment_climate.py` still aggregates the wflow forcing.

**B5 — Two symmetric engine subtrees inside the experiment (P3).**
`weather_generator/` and `hydrology_runs/` share one shape, making the experiment
legible as a pipeline of two engines. `realization_*/` dissolves: its per-member
configs go to `weather_generator/_work/`, the weathergenr-native realizations
`rlz_<r>_cst_<c>.nc` to `weather_generator/output/`. `model_runs/` becomes
`hydrology_runs/rlz_<r>/{config,forcing,output}/` — today it is flat, and at
production scale (RLZ 20 × ST 25) that is ~1000 files in one directory with
configs and outputs interleaved.

*Correction to v1 (repo-9).* v1 filed `inmaps_rlz_*_cst_*.nc` under
`weather_generator/output/`. They are **not** weathergenr output: they are
wflow-grid downscaled forcing, produced by rule 3.09
`downscale_climate_realization` and consumed by the rule 3.10 batch rules — the
per-realization twin of exactly the artifact B2 moves *into* the wflow subtree.
They go to `hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc`, keeping their
`temp()` wrapper. v1 also left two file classes unplaced: rule 3.10's
`outstates_rlz_*_cst_*.nc` goes to `hydrology_runs/rlz_<r>/output/outstates_cst_<c>.nc`.

Edit surface this move carries beyond the rule declarations:
`downscale_climate_forcing.py:72` computes a relative prefix against `model_runs/`
depth, and `generate_weather.R:68` builds the `realization_<n>/` path.

**B6 — `stress_test/cst_*.csv` demoted to `weather_generator/_work/`.** They are a
deterministic function of the preserved config snapshot, and the perturbation
coordinates are **already denormalised into the indicators**: `Qstats.csv` carries
`tavg` and `prcp` columns; `basin.csv` is exactly those two.

*Two corrections to v1's safety argument (risk-7), both checkable and both
against the design:*

1. `prepare_cst_parameters.py` writes **three** axes — `temp_mean`, `precip_mean`,
   **and `precip_variance`**. The indicators carry two. The variance axis is
   denormalised **nowhere**, so the demotion is not lossless in the general case.
2. "a scalar per member" understates the reduction:
   `export_wflow_results.py:162-163` takes `df_st["temp_mean"].iloc[0]` and
   `df_st["precip_mean"].iloc[0]` — the **January** row, not a member-level
   scalar. The monthly-structure loss v1 framed as hypothetical is already live.

The demotion still stands: `_work/` is preserved on disk, not deleted, so nothing
is lost — the demotion is about legibility, not retention. What changes is the
claim: `cst_*.csv` remains the **only** record of `precip_variance` and of monthly
structure, so it is `_work/`-but-retained, and a seasonally-varying perturbation
config would promote a merged grid table to a first-class output.

*The move relocates an undeclared runtime input* (risk-7). `cst_*.csv` is read at
run time from a path constructed inside the script
(`export_wflow_results.py:161`, `f"{exp_dir}/stress_test/cst_{st_nb}.csv"`) while
rule 3.11 declares only `rlz_csv_fns` as `input:`. Snakemake does not know about
it and `--dry-run` cannot see it. B6 therefore **declares `cst_*.csv` as a real
`input:` on rule 3.11** while moving it (the paths are enumerable from `ST_NUM`),
and B6 joins B4/B5 on the "needs a real run, not a dry-run" list. The same
milestone that fixes undeclared *outputs* (O-24) must not silently relocate an
undeclared *input*.

**B7 — `model_results/` → `indicators/`.** Not `outputs/`: `hydrology_runs/` also
holds outputs, so that name blurs the boundary it should sharpen. "Indicators" is
the CST term for these quantities. `export_wflow_results.py:281` hardcodes
`model_results`.

**B8 — `experiment_id` is auto-*suggested*, never auto-generated.** A runtime
timestamp would make every invocation target a fresh directory: nothing ever up to
date, incremental reruns impossible, `--dry-run` misleading, and the baseline gate
without a fixed path. A helper writes `experiment_name: <project_name>_<YYYYMMDD>`
into the config once; the run reads it as today.

*Correction to v1 (repo-7).* v1 derived `project_name` as `basename(project_dir)`
and cited `gabon260725` / `gabon_20260726` as evidence the grammar is satisfied —
but those are already-conforming names. `validate_experiment_name`
(`snake_utils.py:181,232`) enforces `^[a-z0-9][a-z0-9_]*$` and its docstring
states "Uppercase is REJECTED (never silently lowercased)". Six live configs carry
`project_dir: examples/Gabon`, whose basename is `Gabon`; production
`project_dir` values routinely carry uppercase, hyphens, or spaces. The helper
would write `experiment_name: Gabon_20260726` and the next wf3 parse would raise
`ValueError` before any rule ran. The helper therefore **slugifies**: lowercase,
map every non-`[a-z0-9]` character to `_`, strip leading non-alphanumerics,
collapse runs of `_`, truncate to 64. This deliberately differs from
`validate_experiment_name`'s never-silently-lowercase stance because the helper is
a *suggestion writer*, not a validator — and the suggested value is passed through
`validate_experiment_name` before being written, so a slug that still fails the
grammar surfaces in the helper rather than at parse time.

**B9 — Split `<project_dir>/config/` into `runs/ catalogs/ templates/ generated/`
(repo-5).** New this revision: v1 drew this split in the tree but gave it no
substantive-move entry, no commit, and no verification row. It is **not a path
move**: `blueearth_cst/model/copy_config_files.py` derives a single
`output_dir = dirname(config_snake_out)` (:68) and writes the snake config, the two
build templates, and the catalog into that one directory (:47-56, :80-81). Routing
four kinds to four subdirectories is a signature change, not a rename. The
`generated/` bin (see P3's generated-vs-copied rule) additionally moves four rule
path strings: rule 1.02's output, 1.03's input, 1.07's output, 1.08's input.

The experiment-level snapshot
`experiments/<id>/config/snake_config_climate_experiment.yml` **stays where it
is** — it does not join `config/runs/` (arch-10 corrects v1's claim that it does).

**B10 — wf1 evaluation and model figures move into the engine subtree (arch-5).**
New this revision: v1 drew `hydrology_model/evaluation/` in the tree with no move
entry and no commit, while it is the largest wf1 move in the milestone. It retires
`{project_dir}/plots/wflow_model_performance/` — the home of three manifest
targets and the outputs of rules 1.11 `plot_results`, 1.12 `plot_map`, 1.13
`plot_forcing`, plus rule 1.14's gather inputs
(`Snakefile_model_creation:265-315,324-326`). Destinations, per P1:

| Rule | Artifact | New home |
|---|---|---|
| 1.11 | `hydro_wflow_1.png`, `clim_wflow_1_{month,year}.png`, `performance_metrics.csv` | `hydrology_model/evaluation/` (+ `plots/`) |
| 1.12 | `basin_area.png` | `hydrology_model/plots/` |
| 1.13 | `precip.png`, `temp.png`, `pet.png` | `hydrology_model/forcing/plots/` |
| 1.14 | gather inputs | repointed to the three above |

Three module constants hardcode the old directory and must move with the rules:
`plot_results.py:108`, `plot_map.py:34`, `plot_map_forcing.py:201`.

---

## Behaviour-preservation stance and baseline consequence

v1 made one claim here; it has to be **two**, because the two things it conflated
are gated differently (repo-5, risk-7).

**Claim 1 — no computed value changes on the baseline seed-fixture class, with
three named exceptions.**

*Scope, stated (ext1-03).* v2 made this claim unqualified while evidencing it
only on the seed fixture; the bounds probe below does not generalize — for
another region, resolution, or hydrography dataset the new delineation bounds
and today's `staticmaps.nc` raster bounds genuinely differ in principle (raster
bounds are snapped to the model grid; polygon bounds are not), and a polygon
edge lying within rounding distance of a source-grid cell boundary can shift
the extracted extent by one cell despite `buffer=1`. The claim is therefore
**scoped to the class the baseline machinery governs**: configs producing the
seed-fixture tree. This scoping is lossless, not a retreat, because of GA-2:
the *only* pre-R07 `project_dir` tree in existence is the test fixture, so
there is no other "today" for a post-R07 run to diverge *from* — a production
project run fresh after R07 has no pre-R07 counterfactual with a consumer.
Non-seed divergence from what today's code would have produced is classified
as **the GA-1-accepted derivation change, not a regression**; it is documented
in the migration map, and the configuration-independent invariant that must
hold everywhere is the bbox-agreement unit test (per-edge tolerance 2 × model
resolution, runnable on any project where both `store_region.geojson` and
`staticmaps.nc` exist).

1. The new climate plot producer (B4) computes source-grid PET, which did not
   previously exist. It produces new figures; it changes no existing value.
2. `warn_if_project_dir_in_repo()` (O-22) emits a warning and returns.
3. **The B1 bbox derivation genuinely changes** (GA-1; arch-3, filed against v1
   for omitting it). The collapsed store is cut to a model-free delineation,
   which is a third derivation, different in principle from both of today's. GA-1
   accepted this cost and ruled it **proven, not assumed** — the risk lens's
   `buffer=1` reasoning and this revision's bounds probe are reasons to *expect*
   the arrays to match, not substitutes for checking them.

   *The proof.* `.nc` files already dispatch to `semantic_tree_diff.compare_nc`,
   the element-wise comparator (dims; coordinate labels **and stored order**, no
   realignment; per-element values; NaN masks; non-volatile attrs). The check is
   whether the survivor's coordinate arrays are identical to **each** collapsed
   source's on the seed fixture — which is exactly the merge-class semantic the
   machinery gains below. The merge comparison *is* the proof; the two are one
   piece of work, not two.

   *The branch if they do not match.* Then the extent change is a confirmed,
   not merely named, exception and the milestone must:
   (a) list `clim_wflow_1_{month,year}.png` as **expected-to-move**, since rule
   1.11 reads the extraction at model parity and PNG comparison is size-only with
   a 10% band (`check_baseline.diff_png`, `PNG_TOLERANCE_FRAC = 0.10`) and would
   not catch it;
   (b) extend "expected-to-move" to the **wf3 indicator targets**, because the
   store feeds weathergenr, so a shifted extraction shifts the realizations and
   every downstream indicator — this is the tail arch-3 did not reach, and it
   would falsify the exit adjudication's "path-and-snapshot-only" claim, which
   must then be rewritten rather than annotated;
   (c) record the per-edge coordinate deltas in the migration map and re-record
   the affected targets as a **stated** value change, not silently;
   (d) confirm `hydrology_model/run_default/output.csv` is **unmoved** — it is
   provably insulated, because rule 1.08 builds `inmaps_historical.nc` via
   `hydromt update` from the catalog, not from the extraction (risk lens,
   verified). **If discharge moves at all, stop and escalate to the owner** — that
   is outside every branch this design authorises.

**Claim 2 — most items are pure path moves; these are not.** v1 said "every item
above is a path move, a rename, a declaration fix, or an added warning". That is
false for four items, none of which changes a computed value but all of which
change code:

| Item | Code change |
|---|---|
| B1 | New shared producer-contract spec + rule declaration in both Snakefiles; three `input:` removals (staticmaps, region.geojson, guard_ok) and one symmetric `input:` addition (the catalog, both declarations — ext2-01); two `shared.basin` config keys + the rule-1.02 hydrography cross-check; retire rule 1.10 and `get_region_preview.py` |
| B6 | `cst_*.csv` becomes a declared `input:` on rule 3.11 |
| B9 | `copy_config_files.py` signature: one `output_dir` → four destinations |
| O-22 | New helper + three call sites |

**Baseline inventory, corrected** (risk-6, arch-10, repo-13 — three lenses, same
facts). v1 said "of the eighteen targets"; the arithmetic did not close, and it
mis-scoped three groups:

- `dev/baseline/manifest.json` holds **18 rows**, but `check_baseline.TARGETS`
  holds **15 live templates**. Three manifest rows are pre-P3-1 orphans with no
  producer — `climate_experiment/model_results/Qstats.csv`,
  `climate_experiment/model_results/basin.csv`, and the root-level
  `config/snake_config_climate_experiment.yml`. `cmd_check` skips them (they are
  in neither `current` nor `missing`) and a full `record` silently drops them.
  **The inventory is re-derived from `TARGETS`, not from the manifest file.**
- There are **three** live config snapshots, not four; and the live wf3 snapshot
  at `experiments/<id>/config/…` does **not** join the `config/runs/` split
  (B9) — it keeps its path and changes content only.
- Of the "six wf2 targets", only the **three summary files** move; the three
  `climate_projections/<clim_project>/plots/` PNGs do not (B3).
- **The wf1 discharge target is *not* unchanged** (repo-3). Every manifest key is
  a literal path prefixed `examples/test_local/`, so O-20 moves all of them, the
  discharge row included. Worse, `check_baseline.py:384-385` derives the stored
  reference-series filename as `sha1(resolved_path)[:16]`, so the rename changes
  the **sidecar** name too: `record` writes a new
  `dev/baseline/discharge_ref/<newhash>.csv` from the *post-R07* run and orphans
  `dev/baseline/discharge_ref/1f9f30a367de162f.csv`. Unlike the 15 fingerprint
  targets, discharge is compared with a **tolerance comparator against a stored
  series**, not a self-contained hash — so a re-record silently re-blesses any
  drift, and the milestone's strongest numeric anchor proves nothing. The gate
  that replaces it is in the verification plan (§ "The discharge anchor").

So: **15 live targets, all 15 changing manifest key (O-20); 10 of them also
changing path within the tree; 3 changing content (the config snapshots); 3 stale
rows dropped.** The three drops are listed in the migration map as expected
deletions so the exit adjudication does not read them as R07 targets that failed
to be produced.

*The within-tree figure is 10, not the 14 this section carried at acceptance —
corrected 2026-07-28 (editorial). 14 presupposed that the three wf2 PNGs move and
that the wf3 experiment config snapshot joins the `config/runs/` split; the
resolution of arch-10 corrected both in place (B3 moves no PNGs; B9 leaves the
experiment snapshot inside `experiments/<id>/`, content-only) without propagating
the arithmetic here. `dev/r07/migration_project-layout.md` §3a derives the figure
per-target from `check_baseline.TARGETS`, which is what `TARGETS` is rewritten
from — so the map is authoritative and this section now agrees with it.*

**Machinery to update alongside** — the complete list (v1's had three of seven):

1. `dev/scripts/check_baseline.py` `TARGETS` templates and `PROJECT_DIR_DEFAULT`
   — **owned by commit 4, atomically with the fixture rename, not commit 1**
   (ext2-03): retargeting them while the fixture still sits under `examples/`
   would turn `check_baseline check` red three commits before the blackout the
   document dates from commit 4.
2. `dev/scripts/semantic_tree_diff.py` — a new `build_r07_path_map()` /
   `build_r07_allowlist()` **plus a generic `--map old=new` CLI option**, because
   today the map is hardcoded milestone code (`build_p31_path_map()`,
   `build_p31_allowlist()`) and `main()` exposes only `--experiment-name`,
   `--dataset-key`, `--no-path-map`, `--allow`. Without the CLI, the gate cannot
   run at all until the map lands (risk-2).
3. `dev/scripts/semantic_tree_diff.py` — **a declared many-to-one merge class**
   (see below). Blocking, risk-2 + arch-2.
4. `dev/scripts/semantic_tree_diff.py` `COPIED_CONFIG_PATH_MAP` (:90-110), the
   third normalization table, which v1 omitted (repo-6, arch-11a). O-20 changes
   `project.project_dir` inside every copied snapshot from `examples/test_local`
   to `test_case/test_local`, and B9 changes the snapshot paths;
   `compare_copied_config` normalizes only keys present in that map and FAILs on
   any residual difference, so without new entries the phase-B gate goes red for
   pure path bookkeeping — indistinguishable from a real content regression.
   `MIGRATION.md:167-172` records that this table is kept in lockstep with the
   migration map, so the map's config-path table gets the matching rows. Note also
   that `_is_copied_config` (:576) matches any YAML with a `config` path part, so
   the new `experiments/<id>/weather_generator/config/weathergen_config.yml` is
   newly swept into that directional policy — intended, but state it.
5. **The TOML comparator needs no change** — v1 said it "must be updated", which
   overstates it (repo-fit lens, verified): it already covers all five pointer
   fields including `csv.path` and `state.path_output`, and is generic over the
   path map. B5 needs a new path map, not a new comparator.
6. **`tests/` path bindings** (risk-10, arch-4) — not machinery in v1 at all,
   while "`pytest tests/` green" is a stated success criterion. At least seven
   modules encode moved paths: `tests/test_model_creation.py:26-28`,
   `tests/test_interchange_contracts.py:39,484,529,570`,
   `tests/test_extract_climate_wf1.py:24,26`,
   `tests/test_check_baseline_scope.py:131,160`,
   `tests/test_semantic_tree_diff.py:332-388`,
   `tests/test_workflow_climate_experiment.py:114`,
   `tests/test_guard_invalidation.py:97`,
   `tests/test_check_project_consistency.py:30`.
7. **`dev/contracts/*-seam.md`** (arch-4) — `hydrological-model-seam.md:74,353`
   and `weather-generator-seam.md:56,71,248,294` pin the same paths.

**The many-to-one merge class (risk-2, arch-2 — blocking).** `diff_trees` keys the
reference tree by mapped relpath and raises
`ValueError("path map collision: … both map to …")` when two reference files
translate to one key (`semantic_tree_diff.py:641-647`). B1 is exactly that: the
reference tree holds both `climate_historical/wf1_raw/extract_historical.nc` and
`climate_historical/<key>/extract_historical.nc` (and two `orography.nc` on the
chirps branch), so any prefix rule collides them and the gate **aborts before it
can report**. Note this is not dissolved by GA-1 — the shared-rule fix still
collapses two files into one.

Two routes were available. R07 takes the **declared merge class**, not an explicit
`--retire` set:

- *Merge class (adopted).* `diff_trees` gains a declared merge input,
  `--merge <survivor>=<src1>,<src2>`, and compares the survivor against **each**
  collapsed source with the ordinary suffix-dispatched comparator. A merge passes
  only if **all** comparisons pass; a single passing comparison is not a proof.
- *`--retire` set (rejected).* Excluding `wf1_raw/*` from translation and
  allowlisting it as MISSING lets the gate go green while proving nothing about
  the store that disappeared — precisely where GA-1 demands proof. It optimises
  for a green gate over a true one.

This is the connective piece of the revision: because `.nc` already dispatches to
the element-wise `compare_nc`, the merge class is simultaneously the fix for the
`ValueError` **and** the executable form of GA-1's bbox proof. One machinery
change discharges two blocking findings and one owner ruling.

---

## Migrating an existing `project_dir`

**Ruled at G1-return (GA-2, `dev/r07/project-layout-design-review-record.md` § Gate and arbitration record 2026-07-28), in answer to risk-5.** The
question "what depends on current `project_dir` artifact paths?" was put to the
owner and answered: **only the test fixture.**

- There are **no production `project_dir` trees to preserve.**
- **No CST-API or CST-frontend consumer reads artifact paths.** v1's
  *Alternatives considered* invoked a GUI globbing `**/plots/*.png`; that is a
  hypothetical view, not a live consumer.
- **Pre-R07 trees are declared unsupported.** A post-R07 workflow run against a
  pre-R07 tree is not expected to work and is not made to work — the first
  failure is deterministic and early: rule 3.00b `check_project_consistency`
  declares `wf1_snapshot = ancient(f"{project_dir}/config/snake_config_model_creation.yml")`
  as a **mandatory** input, which B9 moves to `config/runs/`, so wf3 raises
  `MissingInputException` at the guard before anything else runs.
- **No `mv` migration script is shipped.** A fresh run is the supported path.
  `dev/r07/migration_project-layout.md` documents the mapping for a reader; it is
  not executable and is not represented as such.

**Scope of the non-support.** It applies to *running* a post-R07 workflow against
a pre-R07 tree. It does **not** license deleting one: the phase-B gate needs a
preserved pre-R07 fixture tree on disk, and the discharge anchor needs a saved
pre-R07 `output.csv` (commit 1). Those are read-only gate references and are
retained for the milestone's duration.

---

## Contract inventory

The moved paths are hardcoded in more places than `--dry-run` can see, and v1
conceded only `params:`-string paths and R `shell:` bodies (arch-4, risk-10).
This table replaces that concession; each row is attached to its commit in the
plan below.

| Move | Rule(s) | Script module(s) | Tests | Seam doc |
|---|---|---|---|---|
| O-20 fixture rename | — | `check_baseline.PROJECT_DIR_DEFAULT` | `test_interchange_contracts.py:39`, `test_check_baseline_scope.py:131,160`, `test_semantic_tree_diff.py:332-388` | — |
| B1 single store | 1.10 (retired), 1.02 (hydrography cross-check), 1.11, 3.02, 3.08 | `snake_utils.py` (`climate_store_spec`), `prepare_build_config.py` (cross-check params + raise), `extract_climate_wf1.py` (retired), `extract_historical_climate.py`, `prepare_climate_data_catalog.py`, `get_region_preview.py` (retired) | `test_extract_climate_wf1.py:24,26`, `test_interchange_contracts.py:484`, `test_guard_invalidation.py:97`, new contract-equality test (full-directive, deny-by-default — ext2-02), new catalog-staleness test (ext2-01) | `hydrological-model-seam.md:74` |
| B2 forcing move | 1.08, and `path_forcing` at `Snakefile_model_creation:198,210,305` | `setup_time_horizon.py:51` | `test_interchange_contracts.py:529` | `hydrological-model-seam.md:353` |
| B3 projections tiers | 2.05 `monthly_change_scalar_merge` (3 summary outputs → `summary/`), 2.06 `plot_climate_proj_timeseries` (`gcm_timeseries.nc` → `timeseries/`, summary input repoints), rules 2.00/2.09 input lists | `get_change_climate_proj_summary.py:80,88,93`, `plot_proj_timeseries.py:223` | `test_get_change_climate_proj_summary.py` (clim-dir-relative paths), `test_check_baseline_scope.py:114,158` (via the commit-4 `TARGETS` retarget — ext2-03) | — |
| B4 climate figures | 1.15 `plot_climate_source` (new; rule `all` + 1.14 gather inputs extended) | `plot_climate_source.py` (new, in `climate_analysis/`) | new P4-assertion test (figures build with no `hydrology_model/` and no build template on disk); source-PET unit test | — |
| B5 experiment subtrees | 3.03–3.11 | `downscale_climate_forcing.py:72`, `generate_weather.R:68` | `test_workflow_climate_experiment.py:114` | `weather-generator-seam.md:56,71,248,294` |
| B6 `cst_*.csv` | 3.11 (`input:` added) | `export_wflow_results.py:161` | — | — |
| B7 `indicators/` | 3.11, 3.12 | `export_wflow_results.py:281` | `test_interchange_contracts.py:570-571,592` | — |
| B8 experiment-name helper | none (a config-prep step, not a rule; wf3 parse consumes the written key at `Snakefile_climate_experiment:36`) | `snake_utils.py` (`suggest_experiment_name`), `scripts/suggest_experiment_name.py` (new thin CLI) | `test_snake_utils.py` slug + refuse-overwrite cases | — |
| B9 config split | 1.01, 1.02, 1.03, 1.07, 1.08, 3.00b (`wf1_snapshot_path` / `wf2_snapshot_path`), 3.01 | `copy_config_files.py:47-56,68,80-81` | `test_model_creation.py:26-28`, `test_check_project_consistency.py:30` | — |
| B10 evaluation move | 1.11, 1.12, 1.13, 1.14 | `plot_results.py:108`, `plot_map.py:34`, `plot_map_forcing.py:201` | — | — |

*Completed this round (ext1-07):* v2's table omitted B3, B4, and B8 despite B4
introducing a producer and B8 a config-writing interface; their rows are
derived from the live Snakefiles and modules above, and B7's empty tests cell
is corrected (the interchange contract tests read `model_results/`). **B8's
invocation contract, pinned:** `python scripts/suggest_experiment_name.py
<config.yml>` reads `project.project_dir`, slugifies its basename per B8, and
writes `workflows.climate_experiment.experiment_name` **only if the key is
absent** — an existing value is never overwritten (exit nonzero, naming the
present value). The written value passes through `validate_experiment_name`
before the write, per B8. The runner lives in `scripts/` because it executes a
pipeline-preparation step (O-23's invocation-model taxonomy), the logic in
`snake_utils.py` beside `validate_experiment_name`.

## Verification plan

| Stage | Proof the repo still runs | Proof of baseline preservation |
|---|---|---|
| Machinery first (commit 1) | `pytest tests/test_semantic_tree_diff.py tests/test_check_baseline_scope.py` green, including new merge-class cases. | Pre-R07 reference tree copied aside; pre-R07 `run_default/output.csv` saved. Both read-only for the milestone. |
| Repository moves (O-01, O-05, O-20, O-21) | `pytest tests/test_cli.py` green — dry-runs all three Snakefiles, **now under the Linux test config as well** (ext1-05; runs on the `ubuntu-latest` CI leg). **Sentinel assertion, corrected** (repo-4): with `output_locations: None`, `basin_area.png` must **not** be produced from a `gauges_None` layer. | Discharge anchor via `check_baseline.py compare` (below); `check_baseline check` is expected red from commit 4 to commit 14. |
| Parse-time warning (O-22) | **Corrected** (repo-8): three unit cases in `tests/test_snake_utils.py` calling `warn_if_project_dir_in_repo()` directly — in-repo path warns; `<repo_root>/test_case/...` silent; absolute out-of-tree path silent — **plus** one `test_cli.py` case asserting the warning text appears in the combined stream. CI invariant: **zero failures and no new skips** against the pre-R07 reference run on both legs. *(Corrected 2026-07-28, editorial — this row previously pinned absolute counts, "386/30/1 win, 385/31/1 linux", while itself adding four tests and with commit 2 adding more; the passed count necessarily moves, so the absolute form was unsatisfiable. Those figures remain valid only as the **pre-R07 reference**, `.github/workflows/ci.yml:68-69`.)* | — |
| Single store (B1) | All three Snakefiles `--dry-run` clean **independently**; a wf1-only run on a fresh `project_dir` builds the store with no `MissingInputException`; the **contract-equality test** (ext1-04, extended per ext2-02) parses both workflows and asserts the `extract_climate_grid` rule exists in each with an identical normalized contract — rule name, script path, **input set (the identical singleton catalog input — ext2-01)**, output paths, params, **and every content- or execution-affecting directive** (`conda`, `container`, `envmodules`, `wrapper`/`notebook`, `shadow`, `threads`, `resources`, `priority`, `retries`, `group`, `cache`, `wildcard_constraints`), with a **deny-by-default allowed-local set**: only `message`/`log`/`benchmark` may differ, and a non-default value on either declaration for any directive outside the test's known universe **fails** — so a later edit to either declaration outside the shared spec, including an environment or execution-directive change, fails the suite, not just a paths-and-params subset check. Unit test: chirps-branch `oro_path` resolves to the emitted `orography.nc`. Unit test: the two bbox derivations agree within 2 × model resolution. Unit test: `prepare_build_config.py` raises on a template/`shared.basin` hydrography mismatch (ext1-01). | **Merge comparison**: survivor vs `wf1_raw/extract_historical.nc` **and** survivor vs the pre-R07 `<key>/extract_historical.nc`, both element-wise via `compare_nc`, **both must pass**. This is exception 3's proof. The two sides are **not symmetric**: the probe puts the R07 bbox bit-identical to today's wf1 bbox, so `survivor vs wf1_raw/` is expected exact and a failure there means something other than the bbox changed; `survivor vs pre-R07 <key>/` is the side that carries the risk, since that store was cut to the 6-dp-rounded region bounds. Read a single failure against that asymmetry before invoking the branch. |
| Rerun-triggering across the two DAGs (B1) | Run wf1 then wf3 on the seed fixture via `scripts/run_workflows.py`; the store rule must report **nothing to be done** — and a subsequent wf1 `--dry-run` must also schedule nothing (both alternation directions checked, since the input-set trigger v2 missed fires on either). This holds by construction: the declarations carry identical singleton input sets (the catalog — ext2-01), identical params, and the same script, so no rerun trigger (mtime, params, input, code, software-env — the full 9.6.2 set) has anything to fire on. **Catalog-staleness test (ext2-01, arbitration-mandated):** modify a relevant catalog definition; the next invocation of either workflow schedules `extract_climate_grid` **exactly once**; after it runs, `--dry-run` in **both** workflows schedules nothing — freshness restored without restoring the cross-DAG oscillation. A failure on either check means the shared contract leaked a per-workflow difference and is a blocker, not a cost. | `tests/test_guard_invalidation.py` still green: a store-key change still invalidates, now via the params trigger (region/hydrography/window/source all ride in params) rather than the retired guard edge. |
| Output-tree moves (B2–B10) | All three Snakefiles `--dry-run` clean; `pytest tests/` green; a full three-workflow run on the seed config completes. | `semantic_tree_diff.py` full-`project_dir` pre/post comparison with the R07 path map + merge class: **MISSING/EXTRA empty modulo a written allowlist**, every value identical. Run **per slice** after each of commits 7, 8, 11, 12 — not once at the end. |
| Climate plot producer (B4) | New figures produced from `<key>/extract_historical.nc` + the catalog with **no `hydrology_model/` and no `config/templates/wflow_build_model.yml` present** — the P4 assertion, now covering the template decoupling ext1-01 demanded (the store producer reads `shared.basin` + catalog only). Existing forcing figures unchanged. | Additive; no existing target changes value. `store_region.geojson` and the three `source_*.png` are allowlisted EXTRA-by-design. |
| Declaration fixes (O-24) | `snakemake --delete-all-output` removes `temp.png` / `pet.png` / `clim_wflow_1_*` / `performance_metrics.csv`, which it cannot do today — claimed for the **seed-config class** only (repo-10). | Newly-declared targets are added to the manifest in the single re-record. |
| Baseline re-record (commit 14) | — | One re-record at the end. The diff against the pre-R07 manifest has **four expected classes, all of which must be present and none other**: (a) path changes, (b) snapshot content changes, (c) the three stated orphan deletions, and (d) **additions for the newly-declared targets** — the O-24 declarations and B4's figures, per the row above. Adjudicated by the normalize-then-compare policy R6 established (`ext2-01`). *Class (d) added 2026-07-28 (pv-2): this row previously asserted "path-and-snapshot-only plus three deletions" while the declaration-fixes row required the new targets to be added — the two could not both hold, so an implementer would have had to either fail the exit assertion or leave the newly-declared outputs without the baseline coverage O-24 exists to give them.* The **exact** membership stays an implementation decision at commit 14, as `dev/r07/migration_project-layout.md` §3d plans: it enumerates 8 candidates and flags the rule-declaration-vs-rule-`all`-membership discrimination as unresolved. Commit 14 resolves it, records the resulting list, and the adjudication checks the diff against that list rather than against a count fixed here. Conditional on exception 3 landing on the no-change branch; otherwise the branch above rewrites this row. |

**The discharge anchor (repo-3).** Because O-20 changes the manifest key *and* the
`sha1(resolved_path)[:16]` sidecar name, `record` would regenerate the reference
series from the post-R07 run and re-bless any drift. Therefore:

- **Commit 1** saves `examples/test_local/hydrology_model/run_default/output.csv`
  to a run-local holding path.
- **Commit 14**, as a gate *before* `record`:
  `python dev/scripts/check_baseline.py compare --ref <saved> --cur test_case/test_local/hydrology_model/run_default/output.csv`
  must exit 0. This is the milestone's real numeric anchor and it is
  comparator-based, so it survives the rename.
- The orphaned `dev/baseline/discharge_ref/1f9f30a367de162f.csv` is deleted in the
  same commit.

**The gate blackout, stated** (repo-3, risk-3, arch-9 — all three lenses). Between
the fixture rename and the re-record, `check_baseline check` is **red by
construction**: recorded keys are old paths (`Path(path).exists()` False → "target
missing on disk") and every current path reports "target present but not in
manifest". v1 never stated this window. **The window's start is now exact**
(ext2-03): v3 dated the blackout from commit 4 while its commit 1 retargeted
`check_baseline.py`'s `TARGETS` and `PROJECT_DIR_DEFAULT` to the future
`test_case/` paths — which would have made `check` report missing targets from
commit 1, three commits before the stated window, with commit 4 also listing
`check_baseline.py` and so owning that edit contradictorily. The retarget now
rides **in commit 4, atomically with the fixture rename** (commit 4 is its
sole owner); commit 1 is confined to the path map, the merge class, the
comparison machinery, and reference capture, so `check_baseline check` is
green through commit 3 and red from commit 4 to commit 14 — exactly as stated
everywhere in this document. The correction leaves ext1-06's reasoning intact:
commit 6 remains a pause point, not a safe cut, precisely because it sits
inside the commit-4-to-14 window. Three things cover the window, and moving
the mechanical machinery to commit 1 is what makes the first two possible:

1. `semantic_tree_diff` per-slice against the retained pre-R07 reference tree
   after every value-touching commit, rather than once at the end.
2. The discharge `compare` anchor above.
3. An explicit note in the commit messages that a red `check` between commits 4
   and 14 is expected and is not a regression signal.

`--dry-run` is blind to `params:`-string paths and to R `shell:` bodies; **B1, B4,
B5, and B6** need a real run, not a dry-run, to be proven (v1 omitted B6 —
risk-7).

## Commit plan

Sequenced so each commit leaves the tree runnable, with the mechanical machinery
**first** and only the manifest re-record last.

**Commit count moves from 13 to 15; content scope is unchanged.** The delta is
(a) the machinery-first split arch-9 and repo-3 require — v1's commit 12 did
machinery *and* re-record together, after the moves it was supposed to police —
and (b) two tree items v1 drew but never assigned to a commit (B9 config split,
repo-5; B10 evaluation move, arch-5). No new scope enters the milestone. Named
here rather than absorbed by merging unrelated commits, so the owner can bounce it
if 13 was meant literally.

1. `r07: prepare the baseline machinery for the layout move` — `build_r07_path_map`,
   `--map` CLI, the merge class, `COPIED_CONFIG_PATH_MAP` entries; save the
   pre-R07 reference tree and the discharge series. **`check_baseline.py` is
   untouched here** (ext2-03): retargeting `TARGETS`/`PROJECT_DIR_DEFAULT`
   before the fixture moves would start the blackout at commit 1 while the
   document dates it from commit 4 — those two edits land in commit 4,
   atomically with the rename, so `check_baseline check` stays green through
   commit 3.
2. `r07: retire data/, ship observation templates` (O-01) + the O-01 test-config
   fix: `tests/snake_config_model_test.yml:32-33` points at a `tests/data/observations/`
   tree that does not exist (O-04) and must stop referencing the deleted `data/`
   paths. This is **not** fixture-rename work — those bindings are commit 4.
   **Plus the Linux-path retarget (ext1-05):** the Linux config's two
   observation keys → `None`, `run_snake_docker.sh` drops the `data/` mount,
   and `tests/test_cli.py` gains the Linux-config dry-run.
3. `r07: delete the docs/config mirror` (O-05).
4. `r07: rename examples/ -> test_case/` (O-20) — fixture path, `.gitignore`,
   configs, `check_baseline.py` (**sole owner** of the `TARGETS` +
   `PROJECT_DIR_DEFAULT` retarget, moved here from commit 1 — ext2-03), the
   four affected test modules, and `run_snake_docker.sh`'s `examples` mount
   (ext1-05). The baseline blackout starts here.
5. `r07: relocate DAG renders under project_dir` (O-02) + README + notebooks.
6. `r07: fix the template project_dir default` (O-21) + `warn_if_project_dir_in_repo`
   (O-22) + `tests/test_snake_utils.py` cases.
7. `r07: single climate store with a shared region+catalog producer` (B1) —
   the full producer-contract spec + both declarations with the symmetric
   catalog input (ext1-02/ext2-01/ext1-04), the
   `shared.basin.hydrography`/`basin_index` keys with commented
   template-config lines, the rule-1.02 cross-check (ext1-01), retire rule
   1.10 and `get_region_preview.py` (O-25), standardise `orography.nc`,
   repoint rule 3.08's `oro_path`, add the contract-equality (full-directive,
   deny-by-default — ext2-02), catalog-staleness (ext2-01), bbox, cross-check,
   and chirps unit tests.
8. `r07: move wflow forcing into the engine subtree` (B2).
9. `r07: tier climate_projections outputs` (B3).
10. `r07: split the project config snapshot into runs/catalogs/templates/generated`
    (B9) — `copy_config_files.py` signature, four rule path strings.
11. `r07: restructure the experiment into engine subtrees` (B5, B6, B7) —
    including the `inmaps_rlz_*` and `outstates_*` placement and rule 3.11's new
    `cst_*.csv` input declaration.
12. `r07: wf1 evaluation and model figures into the engine subtree` (B10) +
    climate figures from the store (B4) + declaration fixes (O-24) + the
    `plot_map.py` sentinel guard (O-08).
13. `r07: experiment_id suggestion helper` (B8), with slugification —
    `suggest_experiment_name` in `snake_utils.py` + the `scripts/` CLI, per
    the contract-inventory row (ext1-07).
14. `r07: re-record the manifest` — discharge `compare` gate first; drop the three
    stale rows with their paths listed.
15. `r07: docs` — `AGENTS.md` repo map + invocation-model line, `README.rst`,
    `MIGRATION.md` → `docs/migration-r06.md`, `naming.md` §7 amendment.

*Cut line, corrected again* (risk-3; ext1-06). v1's § Risks said the artifact
half (B1–B8) is the coherent unit if the milestone must be cut, while the
commit plan landed the repository half first. v2 restated the cut line to match
the plan but still called commit 6 "safe" — which ext1-06 correctly rejects:
`check_baseline check` is red by construction from commit 4, so stopping at 6
leaves the repository's baseline contract invalid indefinitely, and an interim
re-record there would violate the intake's re-record-**once** constraint (that
horn of ext1-06's fix is rejected on that ground; see Alternatives). Restated:
**the milestone has exactly one completed state — after commit 14 (docs commit
15 may trail).** Commit 6 is demoted to a *pause point*: the tree is runnable
and `pytest tests/` is green there, but the baseline gate is red, the holding
artifacts (the retained pre-R07 reference tree and the saved discharge series)
**must be preserved** for the resume, and the pause is flagged as temporary in
the commit message. The abandonment path is also now stated: because the
manifest is untouched until commit 14, abandoning mid-flight means reverting
the landed `r07:` commits, after which the pre-R07 manifest is valid against
the reverted tree — no re-record needed in either direction.

Drive-by fixes accepted into this milestone but independent of it. Each rides in
the numbered commit whose files it already touches, so **the plan stays exactly 15
commits** (corrected 2026-07-28, pv-1 — this paragraph previously said "their own
small commits" while assigning none, which read literally gives 18 commits and
read against the numbered plan silently drops three accepted fixes):

| Drive-by | Rides in | Why there |
|---|---|---|
| O-07 (`prepare_cst_parameters.py` sys.path depth) | **commit 11** | that commit already restructures the experiment (B5/B6/B7) and touches this module's tree |
| O-09 (`plot_results.py` separator docstring) | **commit 12** | that commit already edits `plot_results.py` for B10 + O-24 |
| O-10 (`MIGRATION.md` `__init__.py` list) | **commit 15** | the docs commit, which already moves `MIGRATION.md` to `docs/migration-r06.md` |

**O-08 is no longer a drive-by** — it is the real sentinel defect and rides in
commit 12. **O-13** (`blueearth_cst.Rproj` deletion, ruled at G1) likewise had no
assigned commit; it rides in **commit 15** with the other documentation and
repository-hygiene changes.

## Alternatives considered

- **Centralized `plots/` tree (option 1: `plots/<workflow>/<process>/`).**
  Rejected. With N experiments a central tree must key by experiment anyway, so it
  duplicates the hierarchy rather than flattening it, and it reverses P3-1's
  self-containment decision. Its one genuine win — collecting all figures for a
  report or GUI — is a *view*, obtainable from a glob over `**/plots/*.png`.
- **Standardizing the single climate store on the staticmaps bbox.** Rejected
  after P4: it requires a built model and would block model-free climate analysis
  at the data layer.
- **Nesting a `model/` subfolder inside `hydrology_model/` (option B).** Rejected:
  it changes the `model_root` passed to every hydromt build/update call and every
  derived pointer, for a cosmetic gain.
- **Runtime-generated `experiment_id`.** Rejected — breaks Snakemake idempotence
  (B8).
- **Retiring the forcing/QA plots** once climate figures come from the extraction.
  Rejected by the owner: they answer a real, different question ("did the
  downscaling behave?"), and the parity function to produce them already exists.
  *Extended this revision:* the same reasoning retains `clim_wflow_1_*` (arch-6).
- **Keeping `stress_test/` as first-class provenance.** Withdrawn: the grid is
  deterministic from the preserved config *and* already denormalised into
  `Qstats.csv`. *Qualified this revision:* `precip_variance` and monthly structure
  are **not** denormalised, so `_work/` is retained-not-deleted (B6, risk-7).
- **Moving the Snakefiles into `scripts/` or `workflow/`.** Rejected — category
  error and ~88 reference rewrites (O-23a).
- **Splitting repository and artifact work into two milestones.** Rejected: two
  baseline re-records instead of one, and two migration maps. The panel's
  architecture lens independently confirmed this batching argument is correct and
  attacks only the machinery *sequencing*, which commit 1 fixes.
- **Dropping B1 to a follow-on milestone, keeping B2/B3/B5–B8** (risk-1's option
  ii). Rejected at G1-return: B2 does not depend on the collapse, but P4 does, and
  P4 is a named success criterion of this milestone. Deferring B1 also defers the
  duplicate-extraction cost indefinitely.
- **Pulling a minimal standalone extraction rule into R07** (risk-1's option i).
  Rejected at G1-return: it widens scope toward the fourth-Snakefile milestone
  that is explicitly out of scope, and it makes the store a wf-independent
  artifact with no workflow owning its freshness.
- **Restating the P4 assertion as what R07 actually proves** (arch-1's
  alternative). Rejected at G1-return: it preserves scope by abandoning a named
  success criterion, which is the most expensive kind of scope preservation.
- **An explicit `--retire` set instead of a merge class** (arch-2's second
  option). Rejected: it lets the gate go green while proving nothing about the
  retired store, exactly where GA-1 requires proof (see § "The many-to-one merge
  class").
- **Shipping an executable `mv` migration script for existing `project_dir`
  trees** (risk-5's option ii). Rejected by owner ruling GA-2: there are no
  production trees to migrate.
- **Deriving O-24's declared plot list at parse time** from `wflow_outvars` /
  `output_locations` (repo-10's first option). Rejected for this milestone: it is
  a rule-shape change, not a declaration fix, and would put a config-dependent
  output list into a milestone whose stance is behaviour preservation. The
  config-invariant subset is declared and the remainder is stated.
- **Reading the delineation datasets from the build template** (v2's route,
  arch-1's suggested pin). Rejected this round (ext1-01): it makes the climate
  producer's contract include a Wflow build template, so a climate-only run
  cannot exist without model-build configuration and a template edit mutates a
  model-independent artifact — contradicting both P4 and GA-1's "region +
  catalog only". Replaced by the `shared.basin` keys + rule-1.02 cross-check.
- **Injecting `shared.basin`'s hydrography values into the generated build
  config** (single-sourcing via rule 1.02's merge, instead of the cross-check).
  Rejected: the shipped template's `setup_basemaps.hydrography_fn` lines would
  remain on disk but be silently overridden, so a user's template edit would do
  nothing without an error — and the template also carries a second
  `hydrography_fn` under `setup_rivers` that injection would either miss or
  scope-creep into. The cross-check keeps each value in its
  hydromt-conventional home and makes disagreement loud at the first build
  step.
- **A wf3 readiness-sentinel rule for the store** (ext1-02's suggested
  mechanism: a rule depending on both `.guard_ok` and the store, with
  consumers repointed to its sentinel). Evaluated and not adopted: every wf3
  consumer of the store is *already* transitively guard-gated through the
  per-experiment sentinel chain (3.00b → 3.04 → 3.06), so the readiness rule
  would add a rule and a sentinel file to enforce an ordering that already
  holds, while the store-integrity job the old edge did is subsumed by the
  producer's params trigger. The reviewer's core demand — identical producer
  input sets in both DAGs — is met the simpler way: the identical minimal
  input set (v3: none; final, per ext2-01: the single symmetric catalog
  input).
- **A catalog digest in `params` instead of a catalog `input:`** (ext2-01's
  arbitration fallback, mandated if the two workflows' catalog sets had
  differed). Not needed: the owner-mandated verification found the sets
  identical — one `project.data_sources` key read by both Snakefiles, with
  the experiment-level catalog at `Snakefile_climate_experiment:344`
  belonging to rule 3.09, not the producer. Where symmetry holds, the real
  input is preferred: it carries the standard mtime semantics and puts the
  dependency in the DAG where `--dry-run` can see it, while a digest
  (`file_digest_or_absent()` precedent) would content-trigger but leave the
  edge invisible. Recorded so the fallback is on file: if a future workflow
  ever composes a different catalog set for the producer, the digest route
  replaces the input — an asymmetric input set is exactly the ext1-02
  oscillation and is forbidden by P2(b).
- **Generating both producer declarations from one shared rule module**
  (`include:` — ext2-02's alternative). Evaluated and not adopted. It would
  make drift structurally impossible for every directive, but it would be the
  repo's first `include:`, introducing a new structural pattern for one rule;
  the per-workflow presentation fields (the `W.NN` `rule_banner` message and
  the workflow-scoped log/benchmark paths) would still need parameterization
  through pre-agreed module variables — per-workflow channels that then need
  the same policing the test provides; and the deny-by-default test is needed
  anyway to enforce the allowed-local set, at which point it polices the whole
  contract at no extra cost. The spec + full-directive test keeps both
  declarations visible in their Snakefiles per the house pattern and fails
  loud on unknown directives.
- **Keeping the `ancient(guard_ok)` edge and accepting the re-extractions.**
  Rejected: the input-set trigger fires on *every* wf1/wf3 alternation, each
  firing costs a full native-resolution extraction, and the design's own
  "nothing to be done" verification row would fail — the duplicate-extraction
  cost B1 exists to eliminate would return as a permanent tax.
- **An interim manifest re-record to make commit 6 a releasable cut**
  (ext1-06's first horn). Rejected: the intake constraint is one re-record,
  exactly once, at the end — two re-records is the cost the batched-milestone
  decision was taken to avoid. Commit 6 is demoted to a pause point instead.
- **Parameterized boundary-sensitive bbox tests across representative regions
  and climate grids** (ext1-03's second horn). Rejected as scoped here: each
  case needs a built `staticmaps.nc` to compare against, which is a full
  hydromt build per region — a fixture program, not a layout milestone. The
  claim is scoped to the fixture class instead (lossless under GA-2), with the
  per-edge tolerance test as the configuration-independent invariant.

## Risks and open questions

**Risks.**

- *The `"None"` sentinel — rediagnosed* (repo-4, verified against the code this
  revision). v1's risk named `setup_gauges_and_outputs.py:55` and
  `plot_results.py:127` as raising `TypeError` on YAML `null`. Both read
  `if X is not None and os.path.<exists>(X):` and **short-circuit**, so `null`
  raises nothing at either site. The value that actually misbehaves is the
  **string**: `plot_map.py:28-31` guards only `if gauges_fn is not None:` and then
  computes `gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'`, yielding
  the bogus layer `gauges_None` — which is drive-by O-08, now promoted into commit
  12. The constraint that every written `None` stays byte-identical still holds
  (it is what the existence-based guards depend on), but v1's assertion would have
  passed while proving nothing; the corrected assertion is in the verification
  plan.
- *B1's bbox derivation.* The store's extent now comes from a third derivation.
  The bounds probe says it is bit-identical to today's wf1 bbox and within 3.4e-07°
  of today's wf3 bbox, but the proof is the merge comparison, and the branch if it
  fails is written out in exception 3 — including the wf3 indicator tail and the
  discharge escalation.
- *Two DAGs, one output path.* The shared producer is the first artifact in this
  repo declared by two Snakefiles. v2's residual risk here — the input-set
  trigger firing on the asymmetric `ancient(guard_ok)` edge — is no longer a
  risk but a corrected defect (ext1-02): the edge is gone and both
  declarations carry the identical singleton catalog input (ext2-01), so no
  trigger distinguishes the two DAGs. Three residuals remain. First, the
  contract-equality test is the only thing standing between a future editor
  and reintroducing an asymmetry — it must stay in the suite, and it compares
  the full contract including every execution-affecting directive with a
  deny-by-default allowed-local set (ext2-02) precisely so a leak is a test
  failure rather than a silent re-extraction tax or a one-DAG environment
  divergence. Second, dropping the producer-side guard edge means a wf3 run
  against a *diverged* config can spend one wasted extraction before rule
  3.00b fails the run — bounded, self-healing via the params trigger, and
  judged cheaper than the sentinel rule that would prevent it (see
  Alternatives). Third, the freshness contract's boundary is the catalog
  file: a data mutation behind an unchanged catalog entry still evades every
  trigger — a documented boundary with a stated escape hatch (`--forcerun`;
  B1, ext2-01), not something the machinery can detect.
- *The Linux entry path is consistent, not validated.* The ext1-05 retarget
  keeps the Linux config and Docker runner referentially intact and dry-run
  checked in CI, but no end-to-end Linux run exists to prove the pipeline
  works there — unchanged from before R07, now stated as a support decision
  rather than implied by "parked".
- *TOML relative pointers.* B5 changes run-directory depth, so
  `input.path_static` / `state.path_input` / `input.path_forcing` strings change.
  hydromt re-relativizes on write, and the `semantic_tree_diff` TOML comparator
  already covers all five pointer fields generically — so this needs a **new path
  map**, not a comparator change (repo-fit lens, correcting v1).
- *Three climate-figure families.* After B4 the project holds three families on
  two grids from two producers. The source-grid set is filename-prefixed to break
  the `precip.png` / `pet.png` collision, but a reader still has to learn which of
  three answers their question — the B4 table is the only place that says so.
- *Two PET values.* B4 permits divergence between climate-figure PET and build
  PET. Someone will compare them and report it as a defect. The `source_` prefix
  survives a file being copied out of its directory, which "the figures must say
  approximate on their face" does not; the distinction still belongs in user docs.
- *Scope.* Fifteen commits across both halves of the system, with a window from
  commit 4 to commit 14 in which `check_baseline check` is red by construction. The substitute
  gates are named, but they are per-slice `semantic_tree_diff` runs that depend on
  a retained reference tree — if that tree is lost, the milestone has no
  regression detector at all.
- *`get_region_preview.py` was dead and broken and nothing noticed* (O-25). The
  repository can carry a non-importable module indefinitely because CI runs
  `pytest tests/` and no test imports it. R07 removes this one; the class of
  problem is unaddressed.

**Open questions — all four ruled at G1 (2026-07-27); recorded as settled, not
open.**

1. **Engine-named subtrees** (`models/wflow/` vs `hydrology_model/`) —
   **PARKED, explicitly deferred beyond R07.** Descriptive names are kept. Per
   arch-8, the *structural* half of the same question (how a second engine's build
   subtree and run subtree are placed) is deferred with it, and the Goal is
   narrowed accordingly rather than promising extensibility the tree cannot
   honour.
2. **`MIGRATION.md`'s home** (O-12) — **RULED: moves to `docs/`**, because its
   audience is users.
3. **`blueearth_cst.Rproj`** (O-13) — **RULED: delete.** Unreferenced,
   `Encoding: ISO8859-1`, not used by the owner.
4. **Weathergen date CSVs** — **RULED: `weather_generator/output/` as designed.**
   They are products of the generator, not `_work/` diagnostics.

**Reconciling ruling 2 with `naming.md` §7** (risk-8, repo-11, arch-11b — three
lenses; the ruling explicitly required this and v1 addressed it nowhere). §7 does
not merely *place* migration notes under `dev/<milestone>/`; it makes
`dev/<milestone>/migration_<topic>.md` the **required artifact** of a contract
rename. Moving the R06 note to `docs/` does not create a `dev/r06/` note, so §7
would be left unsatisfied for R06 — a stated divergence is not enough. And risk-8
sharpens the paradox: by the ruling's own audience test R07's map is *more*
user-facing than `MIGRATION.md`, since it maps `project_dir` paths, yet it is
filed under `dev/`.

**Resolution (commit 15): amend §7 to distinguish two artifact classes.**

- An **internal rename record**, `dev/<milestone>/migration_<topic>.md` —
  **required** for every contract rename. R06's is reconstructed from the moved
  file's rename tables; R07's is `dev/r07/migration_project-layout.md`.
- An **optional user-facing migration guide** under `docs/`, derived from the
  internal record, published when a rename affects something a user touches. R06
  publishes one as **`docs/migration-r06.md`** — renamed for `docs/` casing
  consistency, since `docs/` is uniformly lowercase and the root-level
  `MIGRATION.md` convention (§8 row 4) carries no exemption inside `docs/`.
- **R07 publishes no user-facing guide**, because GA-2 declares pre-R07
  `project_dir` trees unsupported and there is no production tree to migrate. Its
  map stays an internal record. This is the rule risk-8 asked for, and it makes
  the two milestones consistent rather than opposite.
- One further §7 line: its mandated `migration_<topic>.md` form **overrides** §8's
  kebab-case rule for `dev/` markdown — R07 is the second milestone to hit this.

**Explicitly out of scope.** The tooling-contract items (O-14 `pyproject.toml`,
O-15 `ruff`, O-16 `flit`) are open decisions unrelated to layout. Docker (O-06)
and Linux end-to-end **validation** are parked — but parking validation no
longer parks consistency: the Linux config and Docker runner are retargeted and
parse-checked in this milestone (ext1-05; § "The Linux entry path"). Promoting
climate analysis to a fourth Snakefile is a separate milestone — R07 only
ensures the layout does not obstruct it, and B1 removes the largest obstruction
by making the store model-free.

## Revision log

| Date | Change |
|---|---|
| 2026-07-28 | **Owner-ruled corrections after a post-acceptance verification pass.** The owner commissioned an external pass (recorded in `dev/r07/project-layout-design-review-record.md` § Post-acceptance verification pass) specifically to close the audit gap the round cap created — the arbitration revision's changes had never been externally verified. **That gap is now closed in substance:** ext2-01 and ext2-03 were confirmed resolved, and both editorial corrections below were confirmed correct, the reviewer independently deriving the same ten within-tree movers. The pass also returned four findings, all accepted and fixed here. Logged as **owner-ruled, not editorial**, because two touch the commit plan and the exit gate. **pv-1** (major): the plan required exactly 15 commits while directing three drive-bys (O-07, O-09, O-10) into "their own small commits" and assigning none — read literally, 18 commits; read against the numbers, three accepted fixes silently dropped. Each is now assigned to the numbered commit whose files it already touches (11 / 12 / 15), and O-13 — ruled at G1 but likewise unassigned — rides in commit 15. **pv-2** (major): the declaration-fixes row required newly-declared targets in the manifest while the commit-14 row asserted a "path-and-snapshot-only plus three deletions" diff; both could not hold, so the implementer would have had to fail the exit assertion or leave O-24's new declarations without baseline coverage. The commit-14 diff now has four expected classes including additions, with exact membership resolved at commit 14 as the migration map already plans. **pv-3** (minor): six directives present in the pinned Snakemake 9.6.2 grammar (`containerized`, `handover`, `localrule`, `default_target`, `template_engine`, `cwl`) were missing from the "every directive" enumeration; added, and the universe is now to be derived from `RuleInfo` rather than from the list. **pv-4** (minor): the 14 → 10 correction had not propagated to the task brief and the migration map's note read present-tense; both fixed, so design, map, and brief agree. |
| 2026-07-28 | **Editorial corrections after acceptance**, applied at stage 7 under the owner's G2 authority and logged here. Two internal inconsistencies, both surfaced by regenerating the derived artifacts from this document rather than by any review round — neither the internal panel nor either external round caught them. **(a)** § "Behaviour-preservation stance": the within-tree mover count corrected **14 → 10**; 14 presupposed two facts the resolution of arch-10 had already corrected elsewhere in this document, and `dev/r07/migration_project-layout.md` §3a — which `check_baseline.TARGETS` is rewritten from — is authoritative. **(b)** § "Verification plan", O-22 row: the CI assertion was pinned to absolute counts (386/30/1, 385/31/1) in the same row that adds four tests, making it unsatisfiable; restated as the invariant **zero failures, no new skips** against a pre-R07 reference. Neither correction changes a decision, interface, stage, or contract — both bring arithmetic into line with decisions the document had already made. |
| 2026-07-26 | Initial draft. Consolidates the repository-side observation register (O-01 … O-24) and the artifact-side working note into one milestone design. Sixteen owner rulings incorporated; P3 rewritten after the owner's challenge to the first-draft "upstream-governed" claim; §9 bbox recommendation reversed after the P4 ruling; `stress_test/` provenance argument withdrawn on evidence from `Qstats.csv`. |
| 2026-07-28 | **Revision r2** against external round 1 (7 findings, all accepted; `dev/r07/project-layout-design-review-record.md` § Finding ledger ext1-01 … ext1-07). **B1's producer decoupled from the build template** (ext1-01): the delineation datasets move to two new optional `shared.basin` keys (`hydrography`, `basin_index`; catalog entry names, defaults = the shipped template's values), `build_config` leaves the spec signature, agreement with the build is enforced by a loud cross-check in rule 1.02's merge script, and cross-workflow agreement rides the existing `shared.basin` guard digest — arch-1's template pin is recorded as a rejected alternative. **The `ancient(guard_ok)` mechanism corrected** (ext1-02, verified against the pinned Snakemake 9.6.2 source: `persistence._input()` keeps ancient files in the recorded input set, so `ancient()` suppresses only mtime-triggering): the producer now has **no inputs in either DAG**, P2(b) is tightened to forbid per-workflow edge asymmetry, wf3's guard edge is removed, store integrity moves to the params trigger, and experiment gating is shown already enforced by the per-experiment sentinel chain; `.guard_ok` stays as the guard's store-level receipt. **The spec becomes a complete producer contract** (ext1-04): script, rule name, empty input set, outputs, params — with a full-contract equality test replacing the paths-and-params check. **Claim 1 scoped to the baseline fixture class** (ext1-03), shown lossless under GA-2, with non-seed divergence classified as the GA-1-accepted derivation change. **The Linux entry path gets an explicit support decision** (ext1-05): Linux config observations → `None`, Docker runner mounts retargeted, Linux-config dry-run added to `test_cli.py`/CI; validation stays parked. **Commit 6 demoted from safe cut to pause point** (ext1-06; the interim-re-record horn rejected against the re-record-once constraint) and the revert-based abandonment path stated. **Contract inventory completed** (ext1-07): B3, B4 (rule 1.15 `plot_climate_source`, named producer), and B8 (helper + `scripts/` CLI with a pinned invocation contract) gain rows; B7's tests cell corrected. Seven alternatives added. |
| 2026-07-28 | **Revision r1** against the stage-2 internal panel (34 findings; all dispositions in `dev/r07/project-layout-design-review-record.md` § Finding ledger) and the G1 / G1-return rulings in `dev/r07/project-layout-design-review-record.md` § Gate and arbitration record. **B1 gains a named producer** — one shared rule definition over region + catalog, declared in both Snakefiles (owner ruling GA-1), superseding `dev/p32a`'s explicit rejection of the single store with a point-by-point rebuttal; the bbox change is named as a **third exception** to the behaviour-preservation stance and given a concrete proof plus a stated failure branch. **P2 restated** to one producer *definition* per artifact, since the fix declares one artifact in two workflows. `get_region_preview.py` found dead and non-importable on the pinned hydromt (O-25); the hydromt v1 `parse_region_basin` API replaces it, and a bounds probe on the seed fixture is recorded. **A declared many-to-one merge class** is added to `semantic_tree_diff` (chosen over an explicit `--retire` set), which simultaneously unblocks the phase-B gate and carries the bbox proof. **The claim that the wf1 discharge target is unchanged is retracted** — all 18 manifest keys move and the sidecar name is `sha1(resolved_path)[:16]` — and replaced by a comparator-based anchor saved before the rename; the gate-blackout window is stated with per-slice substitutes and the mechanical machinery moves to commit 1. Baseline inventory re-derived from `TARGETS` (15 live / 3 orphan). **Two drawn-but-unassigned moves promoted to B9 (config split) and B10 (wf1 evaluation move).** P1 restated as "attach to what they depict"; P3 gains its `hydrology_model/` exemption inline plus a generated-vs-copied config rule; the Goal narrowed to separability. `inmaps_rlz_*` relocated to the wflow run subtree (P3). B6 corrected on `precip_variance` and the January-row reduction, and `cst_*.csv` declared as a rule input. B8 gains slugification. The `"None"` sentinel risk rediagnosed against the code and O-08 promoted out of the drive-by list. GA-2 recorded as an explicit non-support statement. `naming.md` §7 reconciled with the `MIGRATION.md` ruling; all four open questions closed as ruled. Commit count 13 → 15, content scope unchanged, delta named. |
| 2026-07-28 | **Revision r3, made under owner arbitration after the external round cap.** The cap of two external rounds was exhausted with round 2 unconverged (`dev/r07/project-layout-design-review-record.md` § External round 2: 1 blocking, 1 major, 1 minor); the owner arbitrated all three findings **accepted, fix required** (`dev/r07/project-layout-design-review-record.md` § Gate and arbitration record, arbitration entry, 2026-07-28), and those rulings stand in place of the reviewer verdict the cap forecloses. **No external reviewer has verified this version's changes** — their legitimacy rests on the arbitration record, and this row exists so a later reader sees that. Changes confined to the three finding IDs plus forced cross-references. **ext2-01**: the B1 producer gains a single symmetric catalog `input:` (`project.data_sources`), identical in both DAGs. The owner-mandated verification ran and settled the route: the catalog sets **are** identical — one config key read by both Snakefiles (`Snakefile_model_creation:31`, `Snakefile_climate_experiment:34`), drift-guarded under the `project` section, and the experiment-level catalog at `Snakefile_climate_experiment:344` is rule 3.09's, not the producer's — so the digest-in-params fallback was not needed (recorded alternative). The freshness boundary is defined (catalog file in; data behind an unchanged entry out, with the catalog-edit convention and the `--forcerun` escape hatch documented) and the arbitration-mandated exactly-once catalog-staleness test added to the verification plan. **ext2-02**: the contract-equality test extended to every content- or execution-affecting directive (`conda`, `container`, `envmodules`, `wrapper`/`notebook`, `shadow`, `threads`, `resources`, `priority`, `retries`, `group`, `cache`, `wildcard_constraints`) with a deny-by-default allowed-local set ({`message`, `log`, `benchmark`}); unknown non-default directives fail; the shared-rule-module (`include:`) alternative evaluated and recorded. **ext2-03**: the `check_baseline.py` `TARGETS`/`PROJECT_DIR_DEFAULT` retarget moves from commit 1 into commit 4, atomically with the fixture rename and with commit 4 as sole owner, making "blackout from commit 4" true rather than reworded; ext1-06's pause-point reasoning for commit 6 is unchanged by the corrected boundary. |
