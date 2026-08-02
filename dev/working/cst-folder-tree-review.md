# BlueEarth CST project folder tree — external review draft

Status: proposed for external review

Document version: v1

Date: 2026-08-02

Decider: Ümit Taner

Normative body budget: fewer than 450 lines including the tree and review contract

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
under an allocated experiment ID. Retain project-root `logs/` and
`benchmarks/` for WF1/WF2 as explicit cross-cutting exceptions.

## Settled framing

The following owner decisions are inputs to the review. Do not reopen them as
preferences; do raise a finding if the proposed tree implements one
inconsistently or makes it operationally unsafe.

- There is one live Wflow model, updated in place when required.
- `models/hydrology/wflow/` is itself the HydroMT model root. There is no
  `<model-id>/` or `model/` wrapper.
- There is one retained ERA5 historical-climate window. There is no window-ID
  directory.
- There is no project-level `runs/` directory at this stage.
- `config/project.yml` is the editable project source of truth.
- Generated HydroMT build configurations travel with the live Wflow model.
- Experiments retain explicit `weathergenr/` and `wflow/` engine directories.
- Experiment-level `climate/` and `hydrology/` sit directly below the
  experiment root; there is no intermediate `simulations/` directory.
- Machine-readable experiment products live in `results/`.
- Final experiment figures live in an experiment-root `plots/`; engine
  diagnostics remain beside their engine.
- Users may name experiments. When no name is supplied, the default base is
  `stress_test_<YYYYMMDD>`.
- Existing experiment directories are never overwritten.

## Design principles

**P1 — Classify by scientific domain first.** Climate, hydrology, and spatial
data remain recognizable independently of the workflow that produced them.

**P2 — Keep engine-neutral and engine-shaped artifacts separate.** Reusable
project data lives under `data/`; Wflow-specific model state and forcing live
under `models/hydrology/wflow/` or the experiment's `hydrology/wflow/` subtree.

**P3 — Keep each experiment self-contained.** Its configuration, climate
series, hydrological simulations, machine-readable results, final figures,
logs, and benchmarks are colocated under one experiment ID.

**P4 — Separate editable intent from generated provenance.** Project-owned
configuration is edited under the root `config/`; generated model-build
configuration stays with the model; immutable experiment references stay with
the experiment.

**P5 — Prefer the shallowest sufficient hierarchy.** IDs and wrapper
directories are introduced only where more than one instance must coexist.

**P6 — Place figures by audience and subject.** Engine diagnostics stay with
the engine; final figures interpreting the whole experiment are promoted to
the experiment root.

## Proposed folder tree

```text
<project_dir>/                              # e.g. gabon/
├── config/                                # editable project source
│   ├── project.yml                        # canonical project configuration
│   ├── catalogs/
│   │   ├── hydrography.yml
│   │   └── climate.yml
│   └── templates/
│       ├── wflow-build.yml
│       └── wflow-waterbodies.yml
│
├── data/                                  # reusable, engine-independent data
│   ├── spatial/
│   │   ├── region.geojson
│   │   ├── gauges.geojson
│   │   └── ...
│   ├── climate/
│   │   ├── historical/
│   │   │   └── era5/                     # the single active time window
│   │   │       ├── extract_historical.nc
│   │   │       ├── orography.nc
│   │   │       └── plots/                # source-data diagnostics
│   │   ├── observations/
│   │   │   └── ...
│   │   └── projections/
│   │       └── cmip6/                    # plausibility overlay, never WF3 forcing
│   │           ├── timeseries/
│   │           ├── change-factors/
│   │           ├── summary/
│   │           └── plots/
│   └── hydrology/
│       └── observations/
│           └── daily_discharge.csv
│
├── models/
│   └── hydrology/
│       └── wflow/                         # the single live HydroMT model root
│           ├── config/                    # generated HydroMT build configs
│           │   ├── build-model.yml
│           │   └── build-historical-forcing.yml
│           ├── staticgeoms/
│           ├── instate/
│           ├── forcing/
│           │   └── plots/                # model-input diagnostics
│           ├── run_default/              # historical Wflow simulation
│           ├── evaluation/
│           │   ├── performance_metrics.csv
│           │   └── plots/                # historical-run evaluation
│           ├── plots/                    # figures depicting the model itself
│           ├── wflow_sbm.toml
│           ├── staticmaps.nc
│           ├── hydromt_data.yml
│           └── hydromt.log
│
├── experiments/
│   └── <experiment-id>/
│       ├── config/
│       │   ├── experiment.yml
│       │   ├── project-snapshot.yml
│       │   └── model-reference.yml
│       ├── climate/
│       │   └── weathergenr/
│       │       ├── config/
│       │       ├── series/               # stochastic and perturbed climate series
│       │       │   ├── rlz_<r>.nc
│       │       │   └── rlz_<r>_cst_<c>.nc
│       │       ├── plots/                # generator diagnostics only
│       │       └── _work/                # retained engine intermediates
│       ├── hydrology/
│       │   └── wflow/
│       │       └── rlz_<r>/
│       │           ├── config/
│       │           ├── forcing/
│       │           └── output/
│       ├── results/                      # machine-readable experiment products
│       │   ├── Qstats.csv
│       │   ├── basin.csv
│       │   └── RT_*.csv
│       ├── plots/                        # final experiment-level figures
│       ├── logs/
│       └── benchmarks/
│
├── logs/                                  # WF1/WF2 logs
└── benchmarks/                            # WF1/WF2 benchmarks
```

## Placement contract

| Artifact | Required home | Reason |
| --- | --- | --- |
| Editable basin, model-build, and projection settings | `config/project.yml` | One project source of truth |
| Editable data catalogs and build templates | `config/catalogs/`, `config/templates/` | Project inputs, not generated state |
| Region, gauges, and other engine-neutral geometry | `data/spatial/` | Reusable across engines and workflows |
| ERA5 extraction | `data/climate/historical/era5/` | One active source and window |
| CMIP6 change factors | `data/climate/projections/cmip6/` | Plausibility overlay, independent of experiments |
| Observed discharge | `data/hydrology/observations/` | Observation, not model output |
| Wflow `staticmaps.nc`, TOML, states, forcing, and historical run | `models/hydrology/wflow/` | Engine-shaped live model artifacts |
| Generated HydroMT build YAML | `models/hydrology/wflow/config/` | Provenance of the model it built |
| Generated and perturbed weather series | `experiments/<id>/climate/weathergenr/series/` | Experiment-specific climate simulation |
| Wflow stress-test simulation artifacts | `experiments/<id>/hydrology/wflow/` | Experiment-specific hydrological simulation |
| Tables used to construct response surfaces | `experiments/<id>/results/` | Machine-readable final products |
| Response surfaces, vulnerability figures, and projection-overlay figures | `experiments/<id>/plots/` | Final experiment interpretation |
| Weathergenr/Wflow diagnostic figures | Beside the relevant engine | Avoid mixing diagnostics with final figures |

## Experiment creation and ID allocation

Experiment creation and experiment execution are distinct operations.

1. A user-supplied name is slugified to lowercase letters, numbers, and
   underscores, beginning with an alphanumeric character and limited to 64
   characters. Example: `Reservoir Option A` becomes `reservoir_option_a`.
2. Without a user-supplied name, the base ID is
   `stress_test_<YYYYMMDD>`.
3. Creation reserves the first free ID. If the base exists, suffixes are
   allocated as `_v2`, `_v3`, and so on. The first instance has no `_v1`
   suffix.
4. Creation never overwrites or silently reuses an existing directory.
5. Running or resuming an existing experiment uses its exact ID and never
   allocates a new version merely because the directory exists.
6. Directory reservation must be atomic so concurrent creators cannot receive
   the same ID.

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
state it used. `config/model-reference.yml` contains the relative model path and
a deterministic SHA-256 fingerprint over the minimal WF3 runtime model inputs:

- `wflow_sbm.toml`;
- `staticmaps.nc`;
- the presence and content of `instate/instates.nc`.

The digest is computed from sorted relative paths plus file contents; an absent
optional state has an explicit absence marker. Before WF3 performs simulation
work, it recomputes the fingerprint and fails on a mismatch. A changed live
model therefore requires creation of a new experiment version; the old
experiment is not silently rerun against different model physics or state.

The model is not copied into each experiment. Project-level settings are
captured separately in `config/project-snapshot.yml`.

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
- **Use `indicators/` instead of `results/`.** Precise for current outputs but
  too narrow for future vulnerability, overlay, and adaptation products.
- **Put every figure in the experiment-root `plots/`.** Easy to browse but
  conflates engine diagnostics with final scientific interpretation.
- **Use only generated experiment IDs.** Avoids naming collisions but removes
  meaningful user labels.
- **Use content-addressed experiment IDs.** Strong identity but less readable
  and more machinery than the present requirements warrant.

## Consequences

Positive:

- project configuration, reusable data, live model state, and experiments have
  explicit and stable boundaries;
- the project remains shallow for the one-model, one-window case;
- experiments are self-contained without copying the Wflow model;
- CMIP6 projections remain visibly separate from stress-test forcing;
- the GUI can present final plots separately from engine diagnostics;
- experiment creation cannot overwrite prior work.

Negative:

- every workflow path, copied-config path, test fixture, documentation link,
  and external API assumption touching `project_dir` must be migrated;
- a single mutable model and historical window cannot represent concurrent
  alternatives without a future hierarchy change;
- fingerprint computation adds startup IO, dominated by hashing
  `staticmaps.nc` and optional states;
- `_vN` communicates allocation order, not scientific lineage; display names
  and configuration remain necessary for interpretation.

Neutral obligations:

- the path migration must be atomic with its reference rewrites or use a
  deliberate compatibility bridge;
- root `logs/` and `benchmarks/` remain a cross-cutting WF1/WF2 exception;
- exact catalog/template names and all undeclared artifacts must be resolved
  during implementation inventory;
- generated directories should be created only when their producer runs.

## Validation expectations

An implementation design should include at least these falsifiers:

1. Materialize the proposed tree from all three Snakefiles plus undeclared
   engine artifacts and compare it against this contract.
2. Dry-run all three workflows independently and through the orchestration
   wrapper.
3. Run `pytest tests/test_cli.py` and the full unit suite.
4. Compare a pre/post full workflow output tree through an explicit path map;
   every moved scientific artifact must remain value-equivalent.
5. Test user names, default IDs, invalid slugs, collisions through `_v3`, and
   create-versus-resume behavior.
6. Test atomic ID reservation under competing creators.
7. Test model fingerprints for unchanged inputs, each individual runtime-input
   mutation, optional-state presence changes, and excluded output/log changes.
8. Verify that an old experiment fails before simulation after the live model
   changes and that a newly created experiment succeeds.
9. Verify final plots and engine diagnostics land in their distinct homes.

## Open questions for review

- Does the tree omit a lifecycle boundary needed for safe Snakemake incremental
  execution?
- Is the minimal model fingerprint sufficient for numerical reproducibility,
  or does WF3 consume another model-root artifact transitively?
- Does placing generated HydroMT build configuration inside the model root
  conflict with any HydroMT/Wflow directory ownership assumption?
- Are project-root WF1/WF2 logs and benchmarks a tolerable exception, or do
  they undermine the modularity goal?
- Where should workflow DAG renders and small guard/sentinel artifacts live?
- When should `config/experiment.yml` become immutable, and how should a user
  intentionally revise an existing experiment?
- Which catalog files must be copied into the project rather than referenced
  from the toolbox or an external source?
- Is `_vN` an appropriate collision suffix when two same-named experiments are
  not necessarily scientific revisions of one another?
- Which current artifact paths cannot move atomically because an external
  consumer relies on them?

## External reviewer instructions

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
doc_version: v1

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
