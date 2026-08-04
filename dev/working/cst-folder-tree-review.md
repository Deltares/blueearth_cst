# BlueEarth CST project folder tree — owner-accepted design proposal

Status: ACCEPTED by the owner, 2026-08-04. External review waived at this stage.

Document version: v3

Date: 2026-08-04

Decider: Ümit Taner

Normative body budget: fewer than 550 lines including the tree and review
contract (raised from v1's 450 to carry v2's four decisions). Measured over the
normative body only — the *Review status* note and the revision log are document
metadata and are excluded, so the budget constrains the design rather than
inflating each time the status changes. At v3: 525 normative, 574 total.

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

1. **The open questions did not get ruled by the review.** The review was the
   mechanism that would have forced them. They remain open, listed below, and
   they are owner decisions now. Several are implementation-blocking — the
   fingerprint-sufficiency question in particular, since that mechanism is the
   one that can silently admit a wrong result.
2. **The reviewer contract below is intact and unexercised.** It is held in
   reserve, not deleted: a later review can run against this document without
   rework, and would target `doc_version: v3`.

Acceptance covers the placement contract, the naming rule, and the four v2
decisions. It does **not** constitute a ruling on the open questions, and it does
not substitute for the validation expectations, which remain obligations on
whatever implementation follows.

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
- There is one retained ERA5 historical-climate window. There is no window-ID
  directory.
- There is no project-level `runs/` directory at this stage.
- `config/project.yml` is the editable project source of truth.
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

**P4 — Separate editable intent from generated provenance.** Project-owned
configuration is edited under the root `config/`; generated model-build
configuration stays with the model; immutable experiment references stay with
the experiment.

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

## Proposed folder tree

```text
<project_dir>/                              # e.g. gabon/
├── config/                                # editable project source
│   ├── project.yml                        # canonical project configuration
│   ├── catalogs/
│   │   ├── hydrography.yml
│   │   └── climate.yml
│   └── templates/
│       ├── wflow_build.yml
│       └── wflow_waterbodies.yml
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
│   │           ├── timeseries/           # keyed by verbatim CMIP model ID
│   │           ├── change_factors/
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
│       ├── config/
│       │   ├── experiment.yml
│       │   ├── project_snapshot.yml
│       │   └── model_reference.yml
│       ├── climate/
│       │   └── weathergenr/
│       │       ├── config/
│       │       ├── series/               # stochastic and perturbed climate series
│       │       │   ├── rlz_<r>.nc
│       │       │   └── rlz_<r>_cst_<c>.nc
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
│       │           └── outstates_rlz_<r>_cst_<c>.nc
│       ├── results/                      # machine-readable experiment products
│       │   ├── q_indicators.csv          # gauge-point discharge statistics
│       │   └── basin_indicators.csv      # basin-averaged fluxes and states
│       ├── plots/                        # final experiment-level figures
│       ├── logs/
│       └── benchmarks/
│
├── logs/                                  # WF1/WF2 logs (P7)
└── benchmarks/                            # WF1/WF2 benchmarks (P7)
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
| Wflow stress-test configs, forcing, and outputs | `experiments/<id>/hydrology/wflow/{config,forcing,output}/` | Experiment-specific hydrological simulation, one member per file |
| Discharge statistics at gauge points (mean, min, max, q95, 7-day extremes, BFI, return intervals) | `experiments/<id>/results/q_indicators.csv` | Point-support response-surface input |
| Basin-averaged fluxes and states (evapotranspiration, recharge, overland flow, peak snow water equivalent; mm/yr, set by `wflow_outvars`) | `experiments/<id>/results/basin_indicators.csv` | Areal-support response-surface input |
| Response surfaces, vulnerability figures, and projection-overlay figures | `experiments/<id>/plots/` | Final experiment interpretation |
| Weathergenr/Wflow diagnostic figures | Beside the relevant engine | Avoid mixing diagnostics with final figures |
| WF1/WF2 run log and benchmark | `logs/`, `benchmarks/` at the project root | P7 — project-scoped producers |
| WF3 run log and benchmark | `experiments/<id>/logs/`, `.../benchmarks/` | P7 — experiment-scoped producer |

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
state it used. `config/model_reference.yml` contains the relative model path and
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
captured separately in `config/project_snapshot.yml`.

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
- generated directories should be created only when their producer runs;
- an existing interchange validator asserts that the basin table's header is
  exactly the two perturbation-axis columns, which holds only when no
  basin-average outputs are configured and is therefore false under the shipped
  default. The rename must not inherit that assertion unfixed.

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
5. Test user names, default IDs, invalid slugs, collisions through `_v3`, and
   create-versus-resume behavior.
6. Test atomic ID reservation under competing creators.
7. Test model fingerprints for unchanged inputs, each individual runtime-input
   mutation, optional-state presence changes, and excluded output/log changes.
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

## Open questions for review

- Does the tree omit a lifecycle boundary needed for safe Snakemake incremental
  execution?
- Is the minimal model fingerprint sufficient for numerical reproducibility,
  or does WF3 consume another model-root artifact transitively?
- Does placing generated HydroMT build configuration inside the model root
  conflict with any HydroMT/Wflow directory ownership assumption?
- Where should workflow DAG renders and small guard/sentinel artifacts live?
- When should `config/experiment.yml` become immutable, and how should a user
  intentionally revise an existing experiment?
- Which catalog files must be copied into the project rather than referenced
  from the toolbox or an external source?
- Is `_vN` an appropriate collision suffix when two same-named experiments are
  not necessarily scientific revisions of one another?
- Which current artifact paths cannot move atomically because an external
  consumer relies on them?
- Does flattening the Wflow run subtree create a directory-size or tooling
  problem at production grid sizes that the depth saving does not justify?

## External reviewer instructions (held in reserve — not exercised)

*This contract was not run for v1, v2, or v3. It is retained verbatim so a later
review can be dispatched against the current document without rework; see*
*Review status.*

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
doc_version: v3

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
