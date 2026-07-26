# R07 — Project layout — design

**Status. DRAFT 2026-07-26.** Authored interactively with the owner across the
2026-07-26 layout review; **not** produced by a design-review-loop run, so there is
no external review round or finding ledger behind it — unlike
`dev/r06/structural-refactor-design.md`. What it does carry is a complete question
log: sixteen questions were put to the owner and ruled on, and each ruling is
recorded here with its rationale. Implementation is a separate `task-brief`
handoff; this is a proposed plan, not yet accepted and not yet built.

**Genre: decision-record** (a layout refactor), mirroring the R3/R4/R5/R6 house
pattern under `dev/r0#/`.

**Scope authority.**
- `dev/reviews/2026-07-25_post-r6-assessment.md` — the observation register
  (O-01 … O-24) from the owner's post-R6 assessment. Repository-side items.
- `dev/working/2026-07-26_project-output-layout.md` — the working note this design
  supersedes. Artifact-side items, the question log, and the cost analysis.
- Prior accepted layout: `dev/p31/experiment-structure-design.md` §2 (the
  `project_dir` tree this design revises) and `dev/r06/structural-refactor-design.md`
  (the repository tree it revises).

This doc is self-contained: a reviewer needs only this file, the migration map
beside it, and the cited paths.

---

## Goal

One coherent layout across both halves of the system, governed by stated
principles rather than accretion:

- **The toolbox** (the repository) contains source, configuration, and templates —
  no basin data, no run artifacts.
- **The artifacts** (`project_dir`) are organised by *producer* and by
  *engine*, so a reader can tell what made a file by where it sits, and a second
  modelling engine can be added without inventing a new layout.

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

Four invariants. Most of the tree follows from them.

**P1 — Figures attach to their producer.** Every figure lives in a `plots/`
directory that is an immediate child of the subtree that produced it. There is no
project-level `plots/`. Subfolders *inside* a `plots/` leaf are allowed — the rule
constrains where the leaf attaches, not its internals. `plots/` holds figures
only; no CSVs.

**P2 — One producer per artifact.** No artifact is computed twice by two
workflows. Where two workflows need the same input, one produces and the other
consumes.

**P3 — Engine-shaped artifacts live inside their engine's subtree.** Anything in
a model-specific format (wflow forcing netCDFs, run directories, TOMLs; weather
generator configs and realizations) belongs under that engine's subtree. Generic,
engine-independent data (raw climate extractions, projections) stays outside it.
Every engine subtree has the same internal shape: `config/`, `output/`, `plots/`,
`_work/`.

*P3 supersedes a first-draft principle that read "`hydrology_model/` is
upstream-governed; CST adds nothing to it." That was factually wrong: the live
TOML already reaches outside the model root (`path_forcing =
"../climate_historical/wflow_data/inmaps_historical.nc"`), so wflow does not
dictate where forcing lives — CST chose that path. The genuinely upstream-governed
surface is narrower: the TOML **schema** (CSDMS names, `[input.static.*]`),
`staticmaps.nc` internals, and hydromt's build semantics. Not the directory
layout. P3-1 already proved this class of move by relocating run directories and
rewriting the pointers.*

**P4 — Climate analysis must be possible without a model.** A full climate
analysis runs from region + catalog alone, with no wflow setup or run. Climate
figures are produced from the extracted climate store, never from wflow forcing.
Retained from P3-1: **each `experiments/<id>/` is self-contained and reproducible
from its own directory.**

---

## What changes — A. The repository

| # | Change | Rationale |
|---|---|---|
| O-01 | **Delete `data/`.** Ship header-only schema templates at `config/templates/observations/{output_locations.csv, observations_timeseries.csv, README.md}`. Real basin data lives in the project folder, referenced by absolute path. | 667 KB of Gabon-specific CSV in the toolbox source tree. Consumed only by the Linux config and the Docker runner, both parked. |
| O-02 | **DAG renders move to `<project_dir>/dag/`.** `scripts/run_snake_test.cmd:32` retargets; the `dag/` entry leaves `.gitignore`; `README.rst:269,285,298` and six notebook cells stop writing to the repo root. | The DAG is a function of the config, so it belongs with that config's artifacts. The README/notebook commands are the actual source of root clutter. |
| O-05 | **Delete `docs/config/`** (16 tracked pre-R6 duplicates of `config/`). Update `AGENTS.md`'s `docs/` description and `MIGRATION.md:173`. | Two of them still point at the `data/` path O-01 removes. Kept byte-identical by hand until the R01 config restructure ended that. |
| O-20 | **`examples/` → `test_case/`.** `.gitignore:124` follows. | The directory holds the local test fixture, not examples; the real examples are `docs/notebooks/`. |
| O-21 | **`config/workflows/snake_config.template.yml:15-18`** ships an outside-the-tree `project_dir` placeholder, matching its own comment. | The template is the file a new user copies; it currently teaches the opposite of what it documents. This is the origin of the tier confusion. |
| O-22 | **Add `warn_if_project_dir_in_repo()`** to `blueearth_cst/shared/snake_utils.py`, called at parse time from all three Snakefiles. Warns, never raises. Exemption: `<repo_root>/test_case`, held in a module-level constant. | Makes the two-tier rule mechanical instead of documentary. |
| O-24 | **Declare the missing plot outputs** on rules 1.11 and 1.13 while their paths move anyway. | `plot_map_forcing.py` writes three PNGs; rule 1.13 declares one. `plot_results` writes `clim_wflow_1_{month,year}.png` + `performance_metrics.csv`; only `hydro_wflow_1.png` is declared. Undeclared outputs are not cleaned on rerun and are absent from the baseline. |

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
    templates/  wflow_build_model.yml, wflow_build_model_run.yml,
                wflow_build_forcing_historical.yml, wflow_update_waterbodies.yml

  climate_historical/                       # GENERIC, engine-independent (P3, P4)
    <key>/                                  # key = <clim_source>_<YYYYMMDD>_<YYYYMMDD>
      extract_historical.nc                 #   SINGLE store — one producer (P2)
      orography.nc                          #   chirps-branch sidecar
      .guard_ok
      plots/                                #   THE climate figures — source grid, model-free

  climate_projections/<clim_project>/
    timeseries/gcm_timeseries.nc
    summary/annual_change_scalar_stats_summary.{nc,csv}, *_mean.csv
    plots/

  hydrology_model/                          # wflow ENGINE subtree
    wflow_sbm.toml  staticmaps.nc  staticgeoms/  hydromt.log  hydromt_data.yml
                                            #   ^ hydromt model root == this dir
    forcing/inmaps_historical.nc            #   moved from climate_historical/wflow_data/
    forcing/plots/                          #   model-input QA figures only
    run_default/                            #   historical simulation
    evaluation/
      performance_metrics.csv
      plots/                                #   hydro_wflow_1, basin_area, clim_wflow_1_*

  logs/  benchmarks/                        # wf1 + wf2

  experiments/<experiment_id>/              # id suggested as <project_name>_<YYYYMMDD>
    config/snake_config_climate_experiment.yml
    data_catalog_climate_experiment.yml
    .project_consistency_ok

    weather_generator/                      # ENGINE subtree
      config/weathergen_config.yml
      output/                               #   inmaps_rlz_*_cst_*.nc (temp), sim_dates.csv,
                                            #   resampled_dates.csv
      plots/                                #   obs_power_spectra, warm_annual_*
      _work/                                #   cst_*.csv, weathergen_config_rlz_*_cst_*.yml

    hydrology_runs/rlz_<r>/                 # ENGINE subtree, per realization
      config/cst_<c>.toml
      output/cst_<c>.csv

    indicators/                             # was model_results/
      Qstats.csv  basin.csv  RT_*.csv
      plots/                                #   response-surface figures
    logs/  benchmarks/
```

### The substantive moves

**B1 — Collapse the two climate stores (P2).** `wf1_raw/` and `<key>/` hold the
same grid; P3-2a shipped an `allclose` check between them. One producer, both
workflows consume. **Bbox: region-derived, not staticmaps-derived** — the
staticmaps bbox requires a built model and would make the store unbuildable
without wf1, contradicting P4. `blueearth_cst/model/get_region_preview.py` already
delineates a (sub)basin from a region string + catalog with no model. The retired
`allclose` check returns as a unit test.

**B2 — `climate_historical/wflow_data/` moves into the engine subtree (P3).**
`inmaps_historical.nc` is wflow-shaped data. Under P3 it becomes
`hydrology_model/forcing/`; the only pointer edit is `path_forcing`. Consequence:
`climate_historical/` becomes purely generic and engine-independent, which is what
makes P4 reachable.

**B3 — Tier `climate_projections/` into `timeseries/` + `summary/` + `plots/`.**
`plots/` is already split; the gap is processed-vs-summary. No `raw/` tier: wf2
streams CMIP6 from GCS and never persists slices, and a placeholder directory with
no producer would not be created by Snakemake anyway.

**B4 — Climate figures come from the climate store, never from wflow forcing
(P4).** The only genuinely forcing-only quantity is PET; the extraction carries
`precip`, `temp`, `press_msl`, `kin`, `kout`. Source-grid PET is computed on the
extraction grid using the source orography — **it need not match the build's PET**.
Climate figures are approximate quick assessments; the build's PET is the refined
model input. Two products result, both kept:

| Product | Grid | Needs a model? | Home |
|---|---|---|---|
| Climate figures | source | no | `climate_historical/<key>/plots/` |
| Forcing / model-input QA figures | model | yes | `hydrology_model/forcing/plots/` |

Rule 1.13 keeps its input and producer; only its output path moves. A **new** plot
producer under `blueearth_cst/climate_analysis/` reads
`<key>/extract_historical.nc`. Remaining coupling to break:
`climate_analysis/subcatchment_climate.py` still aggregates the wflow forcing.

**B5 — Two symmetric engine subtrees inside the experiment (P3).**
`weather_generator/` and `hydrology_runs/` share one internal shape, making the
experiment legible as a pipeline of two engines. `realization_*/` dissolves: its
configs go to `weather_generator/_work/`, its netCDFs to
`weather_generator/output/`. `model_runs/` becomes `hydrology_runs/rlz_<r>/{config,output}/`
— today it is flat, and at production scale (RLZ 20 × ST 25) that is ~1000 files
in one directory with configs and outputs interleaved.

**B6 — `stress_test/cst_*.csv` demoted to `_work/`.** They are a deterministic
function of the preserved config snapshot, and the perturbation coordinates are
**already denormalised into the indicators**: `Qstats.csv` carries `tavg` and
`prcp` columns; `basin.csv` is exactly those two. *Caveat:* `cst_*.csv` holds
monthly structure while `Qstats.csv` holds a scalar per member. For uniform monthly
perturbations — every current config — the scalar is lossless. The schema permits
per-month arrays; under a seasonally-varying perturbation the scalar no longer
identifies what was applied, and a merged grid table would become a required
first-class output rather than an intermediate.

**B7 — `model_results/` → `indicators/`.** Not `outputs/`: `hydrology_runs/` also
holds outputs, so that name blurs the boundary it should sharpen. "Indicators" is
the CST term for these quantities.

**B8 — `experiment_id` is auto-*suggested*, never auto-generated.** A runtime
timestamp would make every invocation target a fresh directory: nothing ever up to
date, incremental reruns impossible, `--dry-run` misleading, and the baseline gate
without a fixed path. A helper writes `experiment_name: <project_name>_<YYYYMMDD>`
into the config once; the run reads it as today. `project_name` is
`basename(project_dir)` — no new config key. Both `gabon260725` and
`gabon_20260726` already satisfy the existing `^[a-z0-9][a-z0-9_]*$` grammar.

---

## Behaviour-preservation stance and baseline consequence

**No computational path changes.** Every item above is a path move, a rename, a
declaration fix, or an added warning. The two exceptions, both additive:

1. The new climate plot producer (B4) computes source-grid PET, which did not
   previously exist. It produces new figures; it changes no existing value.
2. `warn_if_project_dir_in_repo()` (O-22) emits a warning and returns.

**The baseline must be re-recorded exactly once, at the end of the milestone.**
Of the eighteen targets in `dev/baseline/manifest.json`:

- The four copied-config snapshots change *content* (they embed `project_dir`, and
  O-20 renames the fixture root) **and** *path* (the `config/runs/` split).
- The three wf1 plots, the six wf2 summary/plot targets, and the two wf3 result
  targets all change path.
- The wf1 discharge target (`hydrology_model/run_default/output.csv`) is unchanged.

Machinery to update alongside: `dev/scripts/check_baseline.py` `TARGETS`
templates; `dev/scripts/semantic_tree_diff.py` directory-prefix path map and its
path-aware TOML comparator (the run TOMLs' relative pointers change depth under
B5); `dev/conventions/naming.md` §7 requires the migration map, written as
`dev/r07/migration_project-layout.md`.

Batching the repository and artifact halves into one milestone is the reason for
the single re-record. Split across two milestones, it costs two.

## Verification plan

| Stage | Proof the repo still runs | Proof of baseline preservation |
|---|---|---|
| Repository moves (O-01, O-05, O-20, O-21) | `pytest tests/test_cli.py` green — dry-runs all three Snakefiles on the edited fixture config. Sentinel assertion: both `output_locations` / `observations_timeseries` parse to the **string** `"None"`, not YAML `null` (`null` raises `TypeError` at `setup_gauges_and_outputs.py:55` and `plot_results.py:127`). | wf1 discharge unaffected: the seed config has always carried the sentinel, so no gauges were ever added. |
| DAG relocation (O-02) | `scripts\run_snake_test.cmd --dry-run` writes to `test_case\test_local\dag\`; nothing appears at the repo root; a graphviz failure still leaves the run green. | — |
| Parse-time warning (O-22) | Warning fires for an in-repo `project_dir`, silent for `test_case/` and for absolute paths. `tests/test_cli.py` matches on combined stdout+stderr — confirm its assertions are undisturbed. CI baselines must not move (386/30/1 win, 385/31/1 linux). | — |
| Output-tree moves (B1–B7) | All three Snakefiles `--dry-run` clean; `pytest tests/` green; a full three-workflow run on the seed config completes. | `semantic_tree_diff.py` full-`project_dir` pre/post comparison with the R07 path map: **MISSING/EXTRA empty modulo a written allowlist**, every value identical. |
| Climate plot producer (B4) | New figures produced from `<key>/extract_historical.nc` with **no** `hydrology_model/` present — the P4 assertion. Existing forcing figures unchanged. | Additive; no existing target changes value. |
| Declaration fixes (O-24) | `snakemake --delete-all-output` removes `temp.png` / `pet.png` / `clim_wflow_1_*` / `performance_metrics.csv`, which it cannot do today. | Newly-declared targets are added to the manifest in the single re-record. |
| Baseline re-record | — | One re-record at the end; the diff against the pre-R07 manifest is path-and-snapshot-only, adjudicated by the normalize-then-compare policy R6 established for the config snapshots (`ext2-01`). |

`--dry-run` is blind to `params:`-string paths and to R `shell:` bodies; B4, B5,
and the weathergen relocations need a real run, not a dry-run, to be proven.

## Commit plan

Sequenced so each commit leaves the tree runnable, with the baseline re-record
last:

1. `r07: retire data/, ship observation templates` (O-01) + `tests/` config fix.
2. `r07: delete the docs/config mirror` (O-05).
3. `r07: rename examples/ -> test_case/` (O-20) — fixture path, `.gitignore`,
   configs, `check_baseline.py`.
4. `r07: relocate DAG renders under project_dir` (O-02) + README + notebooks.
5. `r07: fix the template project_dir default` (O-21) + `warn_if_project_dir_in_repo`
   (O-22).
6. `r07: collapse the climate_historical stores` (B1) + the `allclose` unit test.
7. `r07: move wflow forcing into the engine subtree` (B2).
8. `r07: tier climate_projections outputs` (B3).
9. `r07: restructure the experiment into engine subtrees` (B5, B6, B7).
10. `r07: climate figures from the climate store` (B4) + declaration fixes (O-24).
11. `r07: experiment_id suggestion helper` (B8).
12. `r07: update baseline machinery and re-record the manifest`.
13. `r07: docs — AGENTS.md repo map, invocation-model line, README, MIGRATION`.

Drive-by fixes accepted into this milestone but independent of it, to be landed as
their own small commits: O-07 (`prepare_cst_parameters.py` sys.path depth), O-08
(`plot_map.py` sentinel guard), O-09 (`plot_results.py` separator docstring), O-10
(`MIGRATION.md` `__init__.py` list).

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
- **Keeping `stress_test/` as first-class provenance.** Withdrawn: the grid is
  deterministic from the preserved config *and* already denormalised into
  `Qstats.csv`.
- **Moving the Snakefiles into `scripts/` or `workflow/`.** Rejected — category
  error and ~88 reference rewrites (O-23a).
- **Splitting repository and artifact work into two milestones.** Rejected: two
  baseline re-records instead of one, and two migration maps.

## Risks and open questions

**Risks.**

- *The `"None"` sentinel.* Several edited configs carry unquoted `None`, which
  parses to the **string** `"None"`. Substituting YAML `null` raises `TypeError`
  in two consumers. This is the single change most likely to fail at runtime
  rather than at DAG build; it is a named assertion in the verification plan.
- *TOML relative pointers.* B5 changes run-directory depth, so
  `input.path_static` / `state.path_input` / `input.path_forcing` strings change.
  hydromt re-relativizes on write, but `semantic_tree_diff.py`'s comparator must be
  updated or the milestone diff will report false failures.
- *Two `precip.png`.* After B4 the same basename exists in
  `climate_historical/<key>/plots/` and `hydrology_model/forcing/plots/`. A
  collector globbing `**/plots/*.png` sees both. Decide whether one or both are
  baseline targets, and whether filenames should disambiguate rather than relying
  on the parent directory.
- *Two PET values.* B4 permits divergence between climate-figure PET and build
  PET. Someone will compare them and report it as a defect. The figures must say
  "approximate" on their face, and the distinction belongs in user docs.
- *Scope.* Thirteen commits across both halves of the system. If it must be cut,
  the artifact half (B1–B8) is the coherent unit; the repository half can be
  deferred, at the cost of a second baseline re-record.

**Open questions.**

1. **Engine-named subtrees.** Keep descriptive names (`hydrology_model/`,
   `weather_generator/`), or name for the engine (`models/wflow/`,
   `models/weathergenr/`)? Descriptive reads better; engine-named scales better if
   a second hydrology engine appears. **Parked — does not gate this milestone.**
2. **`MIGRATION.md`'s home** (O-12). It is R06-scoped and git-ref-anchored, and
   `naming.md` §7 puts migration notes under `dev/<milestone>/`. Three defensible
   targets: `docs/` (matches its audience), `dev/r06/` (matches §7), or leave it at
   the root with a stated §7 exemption. R07 creates a second root-level candidate,
   so deciding now avoids a third.
3. **`blueearth_cst.Rproj`** (O-13). Unreferenced, `Encoding: ISO8859-1`. Delete,
   or move beside the R sources? Depends on whether the owner uses it.
4. **Where the weathergen date CSVs settle.** Placed in `weather_generator/output/`
   here; they are diagnostics rather than products, so `_work/` is arguable.

**Explicitly out of scope.** The tooling-contract items (O-14 `pyproject.toml`,
O-15 `ruff`, O-16 `flit`) are open decisions unrelated to layout. Docker (O-06)
and Linux end-to-end validation are parked. Promoting climate analysis to a fourth
Snakefile is a separate milestone — R07 only ensures the layout does not obstruct
it.

## Revision log

| Date | Change |
|---|---|
| 2026-07-26 | Initial draft. Consolidates the repository-side observation register (O-01 … O-24) and the artifact-side working note into one milestone design. Sixteen owner rulings incorporated; P3 rewritten after the owner's challenge to the first-draft "upstream-governed" claim; §9 bbox recommendation reversed after the P4 ruling; `stress_test/` provenance argument withdrawn on evidence from `Qstats.csv`. |
