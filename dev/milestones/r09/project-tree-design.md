# BlueEarth CST project folder tree — owner-accepted design proposal

Status: ACCEPTED by the owner, 2026-08-04. External review waived at this stage.

Document version: v8

Date: 2026-08-04

Decider: Ümit Taner

Normative body budget: **retired at v4** (was 450 at v1, 550 at v2). The budget
existed to keep an external review tractable. With the review waived and nine
questions ruled into the text, the document is now the accepted design of
record rather than a reviewer's reading load, and raising the number a third
time would make it decorative. Retired deliberately, not exceeded silently.

## Review status

This design was drafted for external review and **carries no external verdict.**
The owner accepted it on 2026-08-04 after a four-decision working pass (see the
revision log) and waived the review at this stage. Stated on the document's face
rather than left implicit, because the distinction matters to anyone auditing
the milestone later: every claim below rests on owner judgment plus the
code-grounded checks cited during the working pass, and none of it has been
challenged by an independent reviewer.

Two consequences follow, and they are the reason the waiver is recorded rather
than assumed:

1. **The open questions were ruled by the owner, not by a review.** The review
   was the mechanism that would have forced them; with it waived, all nine were
   worked directly against the code and ruled at v4 (see *Questions ruled at
   acceptance*). Two of those rulings changed the design rather than confirming
   it, which is the outcome an external review is normally relied on to produce.
2. **The reviewer contract below is intact and unexercised.** It is held in
   reserve, not deleted: a later review can run against this document without
   rework, and would target `doc_version: v8`.

Acceptance covers the placement contract, the naming rule, the four v2
decisions, and the nine v4 rulings. It does not substitute for the validation
expectations, which remain obligations on whatever implementation follows.

## Corrections from the artifact inventory (v5–v7)

### `config/` was wrong, and is now ruled (v5–v6)

The artifact inventory (`migration_project-tree.md`, Finding 1) found this
document wrong about `config/`. v5 recorded the error; **v6 rules it.**

**What was wrong.** The tree labelled `config/` "editable project source" and
claimed toolbox catalogs are "referenced, not copied". In the current code
`<project_dir>/config/` is written *in its entirety* by rule 1.01
`snapshot_config` (`blueearth_cst/model/copy_config_files.py`) — `catalogs/`,
`templates/`, `observations/`, `runs/`, and a digest-keyed bundle — all of it
generated provenance for inputs that live outside `project_dir`. Catalogs are
referenced **and** copied. The per-basin override directory v4 described does not
exist.

**The ruling.** The generated snapshot **stays under `config/`**, with editable
and generated subtrees distinguished inside it rather than split across roots.
P4 is restated accordingly. Two alternatives were rejected:

- *A dedicated `provenance/` root.* Breaks a cross-workflow contract path (see
  below), adds a seventh root against P5, and would leave project scope using
  `provenance/` while experiment scope keeps `config/` for the same artifact
  class.
- *Filing it under `logs/` by P7.* Disqualified: `logs/` is what a user deletes
  to reclaim space, and its parts are merged-then-deleted by design, whereas this
  bundle is immutable and retained — and consumed.

**Why the contract path decides it.**
`config/runs/snake_config_model_creation.yml` is a **declared `input:`** of WF3's
rule 3.00b drift guard, with its digest taken at parse time
(`Snakefile_climate_experiment:210, 290`). The snapshot is not an archive; it is
an artifact WF1/WF2 produce and WF3 consumes. Moving it is a code change to a
cross-workflow contract, not a relocation.

**Scope note, not a design change.** `config/project.yml` does not exist and
nothing writes one; the source of truth is the `--configfile` in the toolbox.
Adopting it moves config ownership from toolbox to project and touches
`run_workflows.py`, the `--configfile` contract, and `suggest_experiment_name.py`.
It remains settled framing and is not reopened here — but R9 must budget for it
as new capability rather than as a move.

### Three tree shapes did not match what the code emits (v7)

The inventory found three places where the tree was drawn from intent and the
code disagreed. All three are resolved **toward the code**, so none costs an
implementation change; they cost three corrections here. P9 is the rule
generalised from them.

| Was drawn | Code emits | Ruling |
| --- | --- | --- |
| `data/climate/historical/era5/` | `climate_historical/<source>_<window>/` | **Keep the key.** It is a cache key from P3-1 §4 — *"two experiments sharing clim_historical + historical_window resolve to the same dir and reuse the extraction"* — not multi-window support. Collapsing it turns a content-addressed cache into a mutable slot whose refill is a network fetch. Settled framing reworded to say what it means. |
| `cmip6/timeseries/` | `cmip6/raw/` **and** `cmip6/scalar/` | **Keep both.** They are two tiers of one identity — raw slice and spatially reduced series, same filename — and `scalar/` over `series/` is R8's recorded ruling **S8-03**. Both are persistent caches, and `prune_series_cache.py` is keyed to that path grammar; its own record shows three prior key-grammar changes each stranding a generation of orphans. |
| `cmip6/change_factors/` | two files under `cmip6/summary/` | **Keep them in `summary/`.** A directory for two files violates P5, and `summary/` is coherent as WF2's reduced products — composition, provenance, and the change-factor tables. Splitting them out leaves `summary/` holding only metadata. |

`report.md` is placed at the `cmip6/` root by the same pass.

### Four unplaced artifact classes, and a defect the flattening would introduce (v8)

| Artifact | Ruling |
| --- | --- |
| `sim_dates.csv`, `resampled_dates.csv` | `series/` **renamed `output/`**. These are generator *products*, and R7's G1 ruling OQ-4 already split products (`output/`) from per-member configs (`_work/`); `series/` was narrower than its contents. `output/` also mirrors `hydrology/wflow/output/`, restoring the two-engine symmetry B5 was for. |
| Wflow's own `log.txt` | **Defect, not a gap** — see below. Given an explicit per-member path under `hydrology/wflow/output/`. |
| `.model_built`, `.outputs_configured` | Stay at the model root. Generalised: *every* sentinel lives beside what it guards, build sentinels included. |
| `data/spatial/**` | Placeholders replaced with the emitted set. `region.geojson` was drawn here but exists only as `models/hydrology/wflow/staticgeoms/region.geojson` and the store's `store_region.geojson`; `gauges.geojson` does not exist at all. Rows provisional pending the spatial work's Gate 2. |

**The defect.** Wflow's `[logging] path_log` defaults to `"log.txt"`, resolved
against the TOML's own directory (`docs/wflow-user-guide/03-toml-file.md:47`) —
which is why the current tree holds one `log.txt` per `rlz_<r>/config/`. Removing
that directory level puts **every member's log at the same path**, and rule 3.10
runs members concurrently in batches, so this is a race on one file rather than a
tidy overwrite. It would have shipped as a real defect of the flattening.

The fix is one line in `downscale_climate_forcing.py`, reusing pointers that are
already layout-derived and mirroring how `output.csv.path` is built:

```python
"[logging] path_log": f"{out_prefix}{run_name}.log",
```

This is the only **code** change the placement pass produced; the other three
cost corrections to this document alone.

Nothing else in this document was affected: `models/`, `experiments/`, `logs/`,
and `benchmarks/` were all confirmed against declared outputs.

## Review purpose

This proposal reorganizes the generated BlueEarth Climate Stress Test (CST)
`project_dir`: the basin-specific folder that holds project configuration,
reusable data, the live hydrological model, climate-stress-test experiments,
logs, and benchmarks. It does not reorganize the BlueEarth CST source
repository.

The requested review is architectural, not editorial. Assess whether the
directory boundaries are coherent, implementable across the three Snakemake
workflows, safe for incremental reruns, and sufficient for reproducibility
without unnecessary hierarchy.

The tree is a conceptual placement contract rather than an exhaustive output
manifest. Representative filenames clarify intent; implementation must still
inventory every existing artifact before migration.

## Context

BlueEarth CST is a three-workflow scientific toolbox:

1. Model creation builds one HydroMT/Wflow-SBM model and runs a historical
   simulation.
2. Climate projections compute CMIP6 change factors used only as a
   plausibility overlay.
3. The climate experiment generates stochastic climate realizations, perturbs
   them across the stress-test grid, runs Wflow, and reduces the simulations to
   hydrological results and response-surface figures.

The current artifact tree is substantially producer-oriented, with major roots
such as `climate_historical/`, `climate_projections/`, and
`hydrology_model/`. The proposal adopts stable domain roots while retaining
explicit engine boundaries. It deliberately optimizes for one basin, one live
Wflow model, one active historical-climate window, and multiple independently
named experiments.

## Scope

In scope:

- top-level directory boundaries under `project_dir`;
- placement of project configuration, reusable data, model artifacts, and
  experiment artifacts;
- the boundary between engine diagnostics and final experiment figures;
- the filename convention for generated artifacts;
- experiment-ID allocation and collision behavior;
- experiment-to-model reproducibility checks.

Out of scope:

- the source-repository layout;
- the detailed migration and compatibility plan;
- changes to scientific calculations or file contents;
- support for multiple concurrent Wflow model instances;
- support for multiple retained ERA5 extraction windows;
- first-class execution-attempt manifests, retries, or a `runs/` hierarchy;
- exact API endpoints or GUI interactions;
- an exhaustive leaf-file inventory.

## Proposed decision

Adopt `config/`, `data/`, `models/`, and `experiments/` as the stable semantic
roots of every CST project directory. Keep one live Wflow model at
`models/hydrology/wflow/`; keep reusable, engine-independent inputs and derived
datasets under `data/`; and make every stress-test experiment self-contained
under an allocated experiment ID. Place each workflow's run log and benchmark at
the scope of what that run produces — project root for WF1/WF2, the experiment
for WF3 — and name every locally minted artifact in lowercase `snake_case`.

## Settled framing

The following owner decisions are inputs to the review. Do not reopen them as
preferences; do raise a finding if the proposed tree implements one
inconsistently or makes it operationally unsafe.

- There is one live Wflow model, updated in place when required.
- `models/hydrology/wflow/` is itself the HydroMT model root. There is no
  `<model-id>/` or `model/` wrapper.
- One historical-climate window is **active** at a time. The store directory is
  keyed by source and window as a **cache key**, not as multi-window support:
  nothing enumerates or selects among windows, and no rule fans out over them.
  A superseded store directory is an orphan, removed by explicit owner action.
- There is no project-level `runs/` directory at this stage.
- `config/project.yml` is the editable project source of truth. **Not built
  today** — see *Correction (v5–v6)*; adopting it is new capability, and the rest
  of `config/` is generated provenance.
- Generated HydroMT build configurations travel with the live Wflow model.
- Experiments retain explicit `weathergenr/` and `wflow/` engine directories.
- Experiment-level `climate/` and `hydrology/` sit directly below the
  experiment root; there is no intermediate `simulations/` directory.
- Within an experiment's engine subtrees, fan-out members are keyed in the
  FILENAME (`rlz_<r>_cst_<c>`), never in a directory level.
- Machine-readable experiment products live in `results/`.
- Final experiment figures live in an experiment-root `plots/`; engine
  diagnostics remain beside their engine.
- Users may name experiments. When no name is supplied, the default base is
  `stress_test_<YYYYMMDD>`.
- Existing experiment directories are never overwritten.
- Locally minted file and directory names are lowercase `snake_case`;
  identifiers owned upstream pass through verbatim.

## Design principles

**P1 — Classify by scientific domain first.** Climate, hydrology, and spatial
data remain recognizable independently of the workflow that produced them.

**P2 — Keep engine-neutral and engine-shaped artifacts separate.** Reusable
project data lives under `data/`; Wflow-specific model state and forcing live
under `models/hydrology/wflow/` or the experiment's `hydrology/wflow/` subtree.

**P3 — Keep each experiment self-contained.** Its configuration, climate
series, hydrological simulations, machine-readable results, final figures,
logs, and benchmarks are colocated under one experiment ID, so the experiment
directory can be copied, archived, or deleted as a unit.

**P4 — Editable intent and generated provenance must be DISTINGUISHABLE.**
Restated at v6. The original wording required them to be *separated* into
different roots, which the code contradicts: `config/` is written entirely by the
build as provenance, and one of its files is a declared cross-workflow input.
Where the two share a root, the generated subtrees are named and documented as
such. Generated model-build configuration still travels with the model, and
immutable experiment references still stay with the experiment.

**P5 — Prefer the shallowest sufficient hierarchy.** IDs and wrapper
directories are introduced only where more than one instance must coexist.

**P6 — Place figures by audience and subject.** Engine diagnostics stay with
the engine; final figures interpreting the whole experiment are promoted to
the experiment root.

**P7 — A run's log and benchmark live at the scope of what the run produces.**
WF1 and WF2 produce project-scoped artifacts (the one live model, the one
projection set), so their logs sit at the project root. WF3 produces an
experiment, so its log sits inside that experiment. This is one rule applied to
two scopes, not a root-level exception.

**P8 — Name locally minted artifacts in lowercase `snake_case`.** Upstream and
established identifiers are exempt (see *Naming rule*), so the convention never
competes with an external contract.

**P9 — Where this tree differs from what the code emits, the emitted structure
wins unless a stated reason overrides it.** Added at v7 after the third instance.
This document was drawn from intent, and every divergence the artifact inventory
found so far turned out to encode a prior decision: the climate store's cache key
(P3-1 §4), the `scalar/` naming (R8 ruling S8-03), the `config/` snapshot
(rule 1.01), and the grouping of the change-factor tables under `summary/`. A
divergence is therefore a finding against this document first, and against the
code only with an argument.

## Proposed folder tree

```text
<project_dir>/                              # e.g. gabon/
├── config/                                # editable + GENERATED, distinguished (P4)
│   ├── project.yml                        # editable — NOT BUILT, see scope note
│   ├── runs/                              # GENERATED  resolved configs +
│   │   ├── snake_config_<workflow>.yml    #            digest-keyed bundles.
│   │   └── <workflow>/<digest>/           #  ⚠ snake_config_model_creation.yml is a
│   │                                      #    DECLARED INPUT of WF3's drift guard
│   ├── catalogs/                          # GENERATED  snapshots of catalogs used
│   ├── templates/                         # GENERATED  snapshots of templates used
│   └── observations/                      # GENERATED  snapshots of obs inputs
│
├── data/                                  # reusable, engine-independent data
│   ├── spatial/                          # PROVISIONAL — spatial work's Gate 2 pending
│   │   ├── spatial_maps.nc
│   │   ├── spatial_catalog.yml            # generated catalog DESCRIBING this data
│   │   ├── spatial_report.yml
│   │   ├── location_registry.csv
│   │   └── geoms/
│   │       └── {basins,catchments,locations,rivers,subbasins}.geojson
│   ├── climate/
│   │   ├── historical/
│   │   │   └── <source>_<window>/        # CACHE KEY (source + window), NOT
│   │   │       │                         # multi-window support. Path MUST stay
│   │   │       │                         # experiment-invariant
│   │   │       ├── extract_historical.nc
│   │   │       ├── store_region.geojson
│   │   │       ├── .guard_ok             # sentinel, beside what it guards
│   │   │       └── plots/                # source-data diagnostics
│   │   ├── observations/
│   │   │   └── ...
│   │   └── projections/
│   │       └── cmip6/                    # plausibility overlay, never WF3 forcing
│   │           ├── raw/                  # CACHE   as-fetched GCM slices
│   │           ├── scalar/               # CACHE   spatially reduced series (S8-03);
│   │           │                         #         both keyed by verbatim CMIP model ID
│   │           ├── summary/              # reduced products, incl.
│   │           │                         #   <proj>_change_factors_{annual,monthly}.csv
│   │           ├── report.md
│   │           └── plots/
│   └── hydrology/
│       └── observations/
│           └── daily_discharge.csv
│
├── models/
│   └── hydrology/
│       └── wflow/                         # the single live HydroMT model root
│           ├── config/                    # generated HydroMT build configs
│           │   ├── build_model.yml
│           │   └── build_historical_forcing.yml
│           ├── staticgeoms/
│           ├── instate/
│           ├── forcing/
│           │   └── plots/                # model-input diagnostics
│           ├── run_default/              # historical Wflow simulation
│           ├── evaluation/
│           │   ├── performance_metrics.csv
│           │   └── plots/                # historical-run evaluation
│           ├── plots/                    # figures depicting the model itself
│           ├── wflow_sbm.toml            # engine-owned names, verbatim
│           ├── staticmaps.nc
│           ├── hydromt_data.yml
│           └── hydromt.log
│
├── experiments/
│   └── <experiment_id>/
│       ├── .project_consistency_ok        # sentinel, beside what it guards
│       ├── config/
│       │   ├── experiment.yml
│       │   ├── project_snapshot.yml
│       │   └── model_reference.yml
│       ├── climate/
│       │   └── weathergenr/
│       │       ├── config/
│       │       ├── output/               # generator PRODUCTS (R7 G1 ruling OQ-4);
│       │       │   ├── rlz_<r>.nc        # named output/ not series/ to hold the
│       │       │   ├── rlz_<r>_cst_<c>.nc#   date tables too, and to mirror
│       │       │   ├── sim_dates.csv     #   hydrology/wflow/output/
│       │       │   └── resampled_dates.csv
│       │       ├── plots/                # generator diagnostics only
│       │       └── _work/                # retained engine intermediates
│       ├── hydrology/
│       │   └── wflow/                    # members keyed by filename, as above
│       │       ├── config/
│       │       │   └── rlz_<r>_cst_<c>.toml
│       │       ├── forcing/
│       │       │   └── inmaps_rlz_<r>_cst_<c>.nc
│       │       └── output/
│       │           ├── rlz_<r>_cst_<c>.csv
│       │           ├── rlz_<r>_cst_<c>.log   # Wflow's own log — see Correction (v8)
│       │           └── outstates_rlz_<r>_cst_<c>.nc
│       ├── results/                      # machine-readable experiment products
│       │   ├── q_indicators.csv          # gauge-point discharge statistics
│       │   └── basin_indicators.csv      # basin-averaged fluxes and states
│       ├── plots/                        # final experiment-level figures
│       ├── logs/
│       │   └── dag/                      # WF3 DAG render
│       └── benchmarks/
│
├── logs/                                  # WF1/WF2 logs (P7)
│   └── dag/                               # WF1/WF2 DAG renders
└── benchmarks/                            # WF1/WF2 benchmarks (P7)
```

## Placement contract

| Artifact | Required home | Reason |
| --- | --- | --- |
| Editable basin, model-build, and projection settings | `config/project.yml` | One project source of truth |
| Snapshots of the catalogs, templates and observation inputs a run used | `config/catalogs/`, `config/templates/`, `config/observations/` | Generated provenance. The originals live outside `project_dir` and are referenced by path; these copies let a finished project state what it was evaluated against |
| Resolved run configuration and its digest-keyed bundle | `config/runs/` | Generated. `snake_config_model_creation.yml` and `snake_config_climate_projections.yml` are **declared inputs** of WF3's drift guard — contract paths, not archives |
| Generated catalogs that DESCRIBE produced data | Beside the data they describe (`data/spatial/spatial_catalog.yml`) | A descriptor of an output, not a snapshot of an input — a different artifact class from `config/catalogs/` |
| Region, gauges, and other engine-neutral geometry | `data/spatial/` | Reusable across engines and workflows |
| ERA5 extraction | `data/climate/historical/era5/` | One active source and window |
| CMIP6 change factors | `data/climate/projections/cmip6/` | Plausibility overlay, independent of experiments |
| Observed discharge | `data/hydrology/observations/` | Observation, not model output |
| Wflow `staticmaps.nc`, TOML, states, forcing, and historical run | `models/hydrology/wflow/` | Engine-shaped live model artifacts |
| Generated HydroMT build YAML | `models/hydrology/wflow/config/` | Provenance of the model it built |
| Generated weather series and the generator's date tables | `experiments/<id>/climate/weathergenr/output/` | Generator products (R7 G1 ruling OQ-4). Named `output/`, mirroring `hydrology/wflow/output/` |
| Wflow stress-test configs, forcing, and outputs | `experiments/<id>/hydrology/wflow/{config,forcing,output}/` | Experiment-specific hydrological simulation, one member per file |
| Discharge statistics at gauge points (mean, min, max, q95, 7-day extremes, BFI, return intervals) | `experiments/<id>/results/q_indicators.csv` | Point-support response-surface input |
| Basin-averaged fluxes and states (evapotranspiration, recharge, overland flow, peak snow water equivalent; mm/yr, set by `wflow_outvars`) | `experiments/<id>/results/basin_indicators.csv` | Areal-support response-surface input |
| Response surfaces, vulnerability figures, and projection-overlay figures | `experiments/<id>/plots/` | Final experiment interpretation |
| Weathergenr/Wflow diagnostic figures | Beside the relevant engine | Avoid mixing diagnostics with final figures |
| WF1/WF2 run log and benchmark | `logs/`, `benchmarks/` at the project root | P7 — project-scoped producers |
| WF3 run log and benchmark | `experiments/<id>/logs/`, `.../benchmarks/` | P7 — experiment-scoped producer |
| Workflow DAG renders | `logs/dag/` (WF1/WF2), `experiments/<id>/logs/dag/` (WF3) | Generated run record, at the producing run's scope (P7). Must NOT sit under the editable `config/`, which would violate P4 |
| Guard, consistency and **build** sentinels | Beside what they guard — `experiments/<id>/.project_consistency_ok`, the climate store's `.guard_ok`, and the model root's `.model_built` / `.outputs_configured` | One rule for every sentinel. The guard's path must additionally stay experiment-invariant; see *Incremental-execution constraint* |

## Naming rule for generated artifacts

Locally minted file and directory names under `project_dir` are lowercase
`snake_case`: no hyphens, no capitals, no spaces. Two exemptions, both
narrow and both stated so a reader does not "correct" them:

1. **Upstream-owned names pass through verbatim.** Engine-mandated filenames
   (`wflow_sbm.toml`, `staticmaps.nc`, `instates.nc`, `hydromt_data.yml`) and
   upstream identifiers embedded in a path — CMIP model IDs such as
   `NOAA-GFDL/GFDL-ESM4`, which carry hyphens, slashes, and mixed case — are
   never normalized.
2. **Established config keys and data labels are unaffected.** The rule governs
   filenames and directory names only. Column and row labels (`Tlow`, `Tpeak`,
   `BFI`) and config keys keep their domain spelling.

Experiment IDs already satisfy the rule by construction (slugified to lowercase
letters, numbers, and underscores).

This closes a gap the repository left open deliberately: the naming guide
currently assigns generated outputs to the "owning workflow contract" and
declines to unify them. Adopting this proposal therefore requires amending that
guide's file-naming table in the same change, or the tree and the convention
will drift.

## Experiment creation and ID allocation

Experiment creation and experiment execution are distinct operations.

1. A user-supplied name is slugified to lowercase letters, numbers, and
   underscores, beginning with an alphanumeric character and limited to 64
   characters. Example: `Reservoir Option A` becomes `reservoir_option_a`.
2. Without a user-supplied name, the base ID is
   `stress_test_<YYYYMMDD>`.
3. **Collision behavior depends on where the name came from.** A colliding
   *user-supplied* name is REJECTED with an error naming the existing
   experiment; it is not silently suffixed. A colliding *generated default*
   (`stress_test_<YYYYMMDD>`, two runs on one day) is suffixed `_v2`, `_v3`,
   and so on; the first instance has no `_v1` suffix. Rationale: a duplicate
   name the user typed is almost always a mistake, and auto-suffixing hides it,
   while a same-day default collision is expected and carries no intent.
4. Creation never overwrites or silently reuses an existing directory.
5. Running or resuming an existing experiment uses its exact ID and never
   allocates a new version merely because the directory exists.
6. Directory reservation must be atomic so concurrent creators cannot receive
   the same ID.
7. **`config/experiment.yml` becomes immutable at the first SUCCESSFUL run**,
   not at creation: creation-to-first-run is the editing window. Revising a
   experiment afterwards means creating a new one. This deliberately mirrors
   the model-fingerprint rule — a changed model already forces a new
   experiment — so configuration drift and model drift obey one rule rather
   than two. Permitting in-place edits with a revision counter was rejected: it
   reintroduces exactly the mutable-provenance problem the fingerprint exists
   to prevent.

Examples:

```text
stress_test_20260802
stress_test_20260802_v2
reservoir_option_a
reservoir_option_a_v2
```

`config/experiment.yml` records at least the allocated ID, the original display
name when supplied, the creation timestamp, and the allocation sequence.

## Model reproducibility contract

There is one mutable live Wflow model, but each experiment records which model
state it used. `config/model_reference.yml` contains the relative model path and
a deterministic SHA-256 fingerprint over the WF3 runtime model inputs.

**The fingerprint is pointer-derived, not a fixed file list.** It covers:

- `wflow_sbm.toml`; and
- **every model-root file the TOML points at**, resolved from its path-valued
  keys (today `input.path_static`, `input.path_forcing`, `state.path_input`,
  `state.path_output`, `output.csv.path` — but discovered, not hardcoded).

A fixed triple of TOML + `staticmaps.nc` + `instates.nc` was rejected: it is
correct only for the TOML shape the toolbox happens to emit today. Nothing
constrains the model to five path keys — any hydromt `setup_*` that emits a
TOML-referenced side file (lake rating curves, glacier tables) adds a runtime
input whose *content* would fall outside a fixed digest. The pointer change
would be caught, because the TOML is hashed; a later in-place edit of the
pointed-to file would not be. Deriving the file set from the TOML closes that
class instead of enumerating its current members.

Deliberately excluded, because Wflow.jl does not read them at run time:
`staticgeoms/` (hydromt-side vector geometry), `hydromt.log`, and
`hydromt_data.yml`.

The digest is computed from sorted relative paths plus file contents; an absent
optional input has an explicit absence marker. Before WF3 performs simulation
work, it recomputes the fingerprint and fails on a mismatch. A changed live
model therefore requires creation of a new experiment version; the old
experiment is not silently rerun against different model physics or state.

The model is not copied into each experiment. Project-level settings are
captured separately in `config/project_snapshot.yml`.

## Incremental-execution constraint

The tree adds no lifecycle boundary beyond the drift guard that already exists,
but it inherits one hard constraint from it, stated here because it is invisible
from the tree alone.

WF3's consistency guard writes a sentinel under the shared climate store, and
that path is deliberately **experiment-invariant**: the store rule is shared
across every experiment on the same dataset and window, so if its input set
varied per experiment, Snakemake's input-set provenance trigger would fire and
re-run shared work once per experiment. `data/climate/historical/era5/`
satisfies this because it is keyed by dataset and window, never by experiment.

Any future refinement that keys the store — or the guard beside it — by
experiment silently reintroduces re-run storms. This is a placement constraint,
not a preference.

## Figure policy

The experiment-root `plots/` is not a catch-all. It holds figures interpreting
the complete experiment: response/exposure surfaces, vulnerability domains,
projection plausibility overlays, and later adaptation comparisons. Diagnostic
figures used to inspect Weathergenr or Wflow behavior remain below the engine
that produced them.

## Alternatives considered

- **Keep the current producer-oriented roots.** Minimizes path migration but
  retains the structural ambiguity between domains and engines.
- **Add `runs/<run-id>/`.** Would support execution-attempt manifests and
  retries, but no current requirement justifies the extra lifecycle.
- **Add `<model-id>/` or a `model/` wrapper.** Supports multiple model instances
  but adds depth without a current second model.
- **Retain historical-window IDs.** Allows multiple ERA5 windows to coexist,
  but the project intentionally keeps one active window.
- **Use `simulations/{climate,hydrology}/` inside experiments.** Explicit but
  redundant because the experiment root already supplies simulation context.
- **Merge experiment logs and benchmarks into the project-root `logs/` and
  `benchmarks/`, disambiguated by experiment ID in the filename.** Rejected on
  a mechanical failure, not on taste: each workflow's per-rule log and benchmark
  *parts* are written under its own `_parts/` tree with member keys that carry no
  experiment ID, and the gather step discovers members by listing the part
  directory and then deletes what it merged. Pointing several experiments at one
  part tree makes concurrent runs write identical paths and lets one experiment's
  gather consume and delete another's in-flight parts. Promoting only the merged
  file avoids that but splits an experiment's logs across two locations during a
  run and orphans root-level files when an experiment directory is deleted.
- **Keep a `rlz_<r>/` directory inside the experiment's Wflow subtree.**
  Rejected: it keyed one fan-out dimension by directory and the other by
  filename, while the climate side keys the identical `RLZ_NUM × ST_NUM` fan-out
  entirely by filename. Flattening completes the two-engine symmetry rather than
  reversing it, and removes one level of depth.
- **Retain the per-location `RT_*.csv` return-period side tables.** Rejected:
  they have no consumer, they are written to a directory the producing rule does
  not declare as an output (invisible to a dry run), and they are already marked
  unpinned by the interchange contract.
- **Use `indicators/` instead of `results/`.** Precise for current outputs but
  too narrow for future vulnerability, overlay, and adaptation products.
- **Name the two result tables by spatial support** (`gauge_indicators.csv` /
  `basin_indicators.csv`). Rejected in favour of `q_indicators.csv`, which names
  the variable; the placement contract above states the point-versus-areal
  distinction explicitly so the pairing remains legible.
- **Put every figure in the experiment-root `plots/`.** Easy to browse but
  conflates engine diagnostics with final scientific interpretation.
- **Use only generated experiment IDs.** Avoids naming collisions but removes
  meaningful user labels.
- **Use content-addressed experiment IDs.** Strong identity but less readable
  and more machinery than the present requirements warrant.
- **Adopt kebab-case for generated filenames.** Rejected *for this class*:
  every engine-owned name inside `project_dir` is already `snake_case`, so
  kebab-case would maximize rather than minimize the mixed-convention surface.
  The rule is deliberately class-scoped and does not propose a repository-wide
  convention: the source repository's `dev/` markdown keeps its existing
  kebab-case convention, which this proposal leaves untouched.

## Consequences

Positive:

- project configuration, reusable data, live model state, and experiments have
  explicit and stable boundaries;
- the project remains shallow for the one-model, one-window case;
- experiments are self-contained without copying the Wflow model, and remain
  deletable and archivable as a unit;
- CMIP6 projections remain visibly separate from stress-test forcing;
- the GUI can present final plots separately from engine diagnostics;
- experiment creation cannot overwrite prior work;
- one filename convention replaces a mixed one, with two stated exemptions.

Negative:

- every workflow path, copied-config path, test fixture, documentation link,
  and external API assumption touching `project_dir` must be migrated;
- a single mutable model and historical window cannot represent concurrent
  alternatives without a future hierarchy change;
- fingerprint computation adds startup IO, dominated by hashing
  `staticmaps.nc` and optional states;
- `_vN` communicates allocation order, not scientific lineage; display names
  and configuration remain necessary for interpretation;
- dropping `RT_*.csv` discards the full discharge-versus-return-period curve.
  `q_indicators.csv` retains only the two scalar return-interval statistics at
  the configured `Tlow`/`Tpeak`, so recovering a frequency curve requires a
  rerun rather than a re-read;
- flattening the Wflow run subtree puts one file per member in each of
  `config/` and `output/`, where the member count is realizations × grid
  points. Twenty realizations over a nine-by-seven perturbation grid gives
  1,260 files per directory, or 1,280 when the unperturbed baseline run is
  enabled.

Neutral obligations:

- the path migration must be atomic with its reference rewrites or use a
  deliberate compatibility bridge;
- the repository naming guide must gain a real rule for generated outputs in
  the same change that adopts this tree;
- renaming the two result tables touches `rule all` output filenames and a
  grandfathered user-facing name, so it requires the repository's internal
  rename record (old → new mapping) and a re-recorded baseline manifest;
- exact catalog/template names and all undeclared artifacts must be resolved
  during implementation inventory;
- Wflow's log path must be set explicitly per member before the `rlz_<r>/` level
  is removed, or concurrent batch members race on one `log.txt`. The removal and
  the `path_log` setting must land in the **same commit**;
- the pruning tooling must learn to report **orphaned climate-store directories**.
  Keeping the store's cache key means a changed window strands its predecessor on
  disk, and `prune_series_cache.py` covers the WF2 series class only, so "one
  active window" is true of the workflow but not of the directory;
- generated directories should be created only when their producer runs;
- an existing interchange validator asserts that the basin table's header is
  exactly the two perturbation-axis columns, which holds only when no
  basin-average outputs are configured and is therefore false under the shipped
  default. The rename must not inherit that assertion unfixed;
- pre-existing `project_dir` trees are **unsupported**: a fresh run is required
  and no `mv` migration script ships. This restates R7's ruling GA-2 for this
  milestone on the same grounds — no production trees exist, and no external
  consumer reads artifact paths — so no current path is blocked from moving
  atomically;
- the pointer-derived fingerprint must resolve TOML pointers against the
  model root and refuse to hash anything outside it, so a pointer escaping the
  model root is an error rather than a silently widened digest.

## Validation expectations

An implementation design should include at least these falsifiers:

1. Materialize the proposed tree from all three Snakefiles plus undeclared
   engine artifacts and compare it against this contract.
2. Dry-run all three workflows independently and through the orchestration
   wrapper.
3. Run `pytest tests/test_cli.py` and the full unit suite.
4. Compare a pre/post full workflow output tree through an explicit path map;
   every moved scientific artifact must remain value-equivalent. The flattened
   Wflow subtree and the renamed result tables need a real run, not a dry run,
   because the emitted Wflow TOML pointer strings change with run-directory
   depth.
5. Test user names, default IDs, invalid slugs, generated-default collisions
   through `_v3`, and create-versus-resume behavior. (Colliding user-supplied
   names are covered by falsifier 14, which asserts rejection rather than
   suffixing.)
6. Test atomic ID reservation under competing creators.
7. Test model fingerprints for unchanged inputs, each individual runtime-input
   mutation, optional-state presence changes, and excluded output/log changes.
   Because the digest is pointer-derived, also test that ADDING a path-valued
   key to the TOML brings a new file into the digest, that editing that file's
   content alone (TOML untouched) is detected, and that `staticgeoms/`,
   `hydromt.log`, and `hydromt_data.yml` changes are not.
8. Verify that an old experiment fails before simulation after the live model
   changes and that a newly created experiment succeeds.
9. Verify final plots and engine diagnostics land in their distinct homes.
10. Run two experiments concurrently and verify that neither one's log or
    benchmark parts are visible to, consumed by, or deleted by the other's
    gather step.
11. Verify `q_indicators.csv` and `basin_indicators.csv` are value-identical to
    the tables they replace, and that no `RT_*.csv` is produced.
12. Scan the materialized tree for names violating the naming rule, with the
    two exemptions encoded rather than hand-waved.
13. Run two experiments that share a dataset and window, and confirm the shared
    climate-store rule's input set is byte-identical for both — the guard
    against re-run storms described under *Incremental-execution constraint*.
14. Test that a colliding user-supplied experiment name is rejected with an
    error naming the existing experiment, while a same-day generated default
    collides into `_v2`; and that `experiment.yml` is writable before the first
    successful run and refused after it.
15. Run two stress-test members concurrently in one batch and assert that each
    writes its own Wflow log — two distinct, non-empty files under
    `hydrology/wflow/output/`. This is the falsifier for the `path_log` fix; the
    defect it guards is a race, so a single-member run cannot detect it.
16. Build the model and then run a HydroMT `update` against it; confirm the
    generated `config/` subdirectory and its contents survive. This is the
    empirical check that HydroMT asserts no ownership over unknown
    subdirectories of the model root (question 3).

## Questions ruled at acceptance

v1–v3 carried nine open questions. All were worked against the code and ruled by
the owner on 2026-08-04. **No question remains open.** They are recorded rather
than deleted, so a later reader can see what was considered and on what basis;
two of the nine changed the design rather than confirming it.

| # | Question | Ruling |
| --- | --- | --- |
| 1 | Lifecycle boundary missing for safe incremental execution? | No boundary missing, but a constraint was implicit and is now stated — see *Incremental-execution constraint*. |
| 2 | Is the minimal model fingerprint sufficient? | **No — design changed.** A fixed file list is correct only for today's TOML shape. The digest is now pointer-derived. |
| 3 | Does HydroMT claim ownership of the model root? | No evidence of a conflict in anything vendored here; a `config/` subdirectory is not a reserved name. Settled empirically by falsifier 7b rather than by assertion. |
| 4 | Where do DAG renders and sentinels live? | Sentinels stay beside what they guard (already the case). DAG renders move OUT of `config/`, which would violate P4, into `logs/dag/` at the producing run's scope. |
| 5 | When does `experiment.yml` become immutable? | At the first successful run, not at creation. Revision afterwards means a new experiment, mirroring the model-fingerprint rule. |
| 6 | Which catalogs must be copied into the project? | **Re-ruled at v6.** The v4 answer ("none") was wrong: catalogs are referenced as inputs *and* copied as provenance into `config/catalogs/`, as are templates and observation inputs. The per-basin override directory v4 described does not exist and is dropped. |
| 7 | Is `_vN` right when two same-named experiments are unrelated? | Split by provenance: a colliding user-supplied name is rejected; only the generated default is auto-suffixed. |
| 8 | Which paths cannot move atomically? | None. R7's ruling GA-2 restated: pre-existing trees unsupported, fresh run required, no external path consumer. |
| 9 | Does flattening create a directory-size problem? | No. ~1,260 files per directory is a browsing annoyance, not a limit; the two places an OS argument limit could bite (the batched Wflow shell call, the reduction rule's input list) are both bounded or passed in-process. |

## External reviewer instructions (held in reserve — not exercised)

*This contract has not been run for any version. It is retained verbatim so a
later review can be dispatched against the current document without rework;*
*see* Review status.

You are an independent design reviewer. Challenge operational feasibility,
missing failure modes, over-engineering, ambiguous ownership, reproducibility,
and migration risk. Do not copyedit prose. Treat the settled framing as owner
decisions; do flag downstream contradictions or unsafe implementations of
those decisions.

Every blocking or major finding must state an observable consequence and cite
the design heading or decision it targets. An approval verdict cannot coexist
with a blocking or major finding.

Return only Markdown with this structure:

```text
## Verdict
verdict: approve | revise | reject
doc_version: v8

## Findings
### ext1-01 [blocking | major | minor]
- section: <target heading>
- finding: <claim>
- rationale: <observable consequence>
- suggested_fix: <concrete change, or "none">
```

List findings in severity order. An empty findings section with
`verdict: approve` is valid.

## Revision log

- 2026-08-02, v1: Initial external-review draft from owner-confirmed scoping
  decisions.
- 2026-08-04, v2: Owner review pass, four decisions. (1) Experiment logs and
  benchmarks stay under the experiment; the root/experiment split is restated as
  scope rule P7 rather than an exception, closing v1's open question on that
  point. (2) The `rlz_<r>/` directory is removed from the experiment's Wflow
  subtree; fan-out members are keyed by filename on both engine sides. (3)
  `Qstats.csv` → `q_indicators.csv`, `basin.csv` → `basin_indicators.csv`, and
  the `RT_*.csv` side tables are dropped. (4) A lowercase `snake_case` naming
  rule is adopted for locally minted names, with upstream identifiers exempt.
  Alternatives, consequences, obligations, falsifiers, and open questions
  updated accordingly.
- 2026-08-04, v3: Status only — no normative change to the tree, the placement
  contract, the naming rule, or the falsifiers. The owner accepted the design
  and waived external review at this stage; the waiver and its two consequences
  are recorded under *Review status*, and the reviewer contract is retained
  unexercised. Versioned rather than edited in place so that document version
  and document content stay one-to-one, which the reviewer response schema
  depends on.
- 2026-08-04, v8: Placed the four remaining unplaced artifact classes, closing
  the path map except the WF1/spatial rows. `series/` renamed **`output/`** — the
  generator's date tables are products under R7's G1 ruling OQ-4, and `output/`
  mirrors `hydrology/wflow/output/`, restoring the engine symmetry `series/`
  broke. Build sentinels join guard sentinels under one rule: a sentinel lives
  beside what it guards. `data/spatial/` placeholders replaced with the emitted
  set — neither `region.geojson` nor `gauges.geojson` existed at that path — and
  marked provisional pending the spatial work's Gate 2. **One defect found:**
  Wflow's `path_log` defaults to `log.txt` beside the TOML, so removing the
  `rlz_<r>/` level makes every concurrently-batched member race on one file. Fixed
  by setting `path_log` per member from the existing layout-derived pointers; the
  removal and the fix must land in the same commit, and falsifier 15 is the
  concurrency check that a single-member run cannot perform.
- 2026-08-04, v7: Three tree shapes corrected toward the code, none costing an
  implementation change. The climate store keeps its source+window **cache key**
  (P3-1 §4) rather than collapsing to a fixed `era5/`, and the settled framing is
  reworded from "no window-ID directory" to what it actually means — one *active*
  window, a cache key, no enumeration. `cmip6/timeseries/` is replaced by the
  `raw/` and `scalar/` pair the code emits: two tiers of one identity, with
  `scalar/` over `series/` already ruled by R8's S8-03, and both keyed to
  `prune_series_cache.py`'s path grammar. `change_factors/` is dropped as a
  directory; the two tables stay under `summary/` with WF2's other reduced
  products. `report.md` placed. **P9 added** — where this tree differs from what
  the code emits, the emitted structure wins unless a stated reason overrides it;
  generalised after the fourth divergence (`config/`, catalogs, the store key,
  `scalar/`) each turned out to encode a prior decision. New obligation: the
  pruning tooling must report orphaned climate-store directories, since keeping
  the cache key means a changed window strands its predecessor.
- 2026-08-04, v6: Ruled the question v5 raised. The generated config snapshot
  **stays under `config/`**, with editable and generated subtrees distinguished
  inside it; a dedicated `provenance/` root and filing it under `logs/` were both
  rejected. The deciding fact is that
  `config/runs/snake_config_model_creation.yml` is a declared `input:` of WF3's
  drift guard, so the snapshot is a consumed cross-workflow contract artifact
  rather than an archive, and moving it would be a code change rather than a
  relocation. **P4 restated** from "separate" to "distinguishable", since the
  original wording is what the code contradicts. The tree's `config/` block now
  marks each subtree generated or editable and flags the contract path; the
  placement contract gains rows for the input snapshots, the resolved run
  configs, and the separate class of generated catalogs that describe produced
  data; ruling 6 is re-ruled rather than merely marked wrong. Two further v4
  errors of the same kind corrected: `config/templates/` was drawn as an editable
  input when it is a snapshot, and the per-basin catalog override directory does
  not exist and is dropped.
- 2026-08-04, v5: Correction, not a redesign. The artifact inventory found this
  document wrong about `config/`: the project's `config/` is written in full by
  rule 1.01 as generated provenance, not authored as editable source, and v4's
  claim that toolbox catalogs are "referenced, not copied" is false. Ruling 6 is
  marked superseded, the tree's `config/catalogs/` comment is corrected, and the
  open ruling — where the generated snapshot lives, given the tree has no
  provenance root — is recorded under *Correction (v5)* rather than decided here.
  `config/project.yml` stays settled framing; it is flagged as new capability
  rather than a relocation, which is a scope note on the milestone, not a
  reopening. No other section is affected: `models/`, `data/`, `experiments/`,
  `logs/` and `benchmarks/` were confirmed against declared outputs.
- 2026-08-04, v4: All nine open questions ruled by the owner; none remain open
  (see *Questions ruled at acceptance*). Two rulings changed the design rather
  than confirming it. (Q2) The model fingerprint is now **pointer-derived** —
  the TOML plus every model-root file its path-valued keys resolve to — instead
  of a fixed triple that was correct only for today's TOML shape; this closes a
  path by which an edited, TOML-referenced side file could pass the guard and
  change results. (Q6) Toolbox catalogs are **referenced, never copied**:
  `config/catalogs/` now holds per-basin overrides only, and generated catalogs
  travel with their producer. (Q4) DAG renders move out of the editable
  `config/`, which violated P4, into `logs/dag/` at the producing run's scope;
  sentinels are confirmed beside what they guard. (Q1) The store path's
  experiment-invariance is promoted from an implicit property to a stated
  constraint with its own section. (Q5, Q7) Experiment-config immutability and
  collision behavior are specified. (Q3, Q8, Q9) Ruled without design change;
  Q3 is settled by a new empirical falsifier rather than by assertion. Tree,
  placement contract, consequences, obligations and falsifiers updated to match.
