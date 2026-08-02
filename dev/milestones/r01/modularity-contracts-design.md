# R01 — Modularity Contracts (pre-R3) — design

**Date.** 2026-05-08.

## Goal

Establish per-workflow config contracts so workflows can be added,
disabled, or replaced in the future without touching others. **Phase 1
of modularity** — formalize boundaries without restructuring code.

The structural refactor (Snakemake `module:` composition, plugin
registry) waits for a real second use case (a 4th workflow, an
alternative hydrological model). R01's job is to make sure that future
refactor is mechanical rather than archaeological.

## Why now (and why before R3)

R3-R5 each refactor a workflow's Snakefile and scripts. Doing those
refactors against a flat single-namespace config means each milestone
has to decide on the fly which keys belong to which workflow — and the
decisions accumulate inconsistently. Locking in the contract first lets
R3-R5 inherit it without negotiation.

The user-facing motivation: enable future workflows like data-only
analysis, alt hydrological models, and partial pipelines (e.g.,
projections without stress test). Today the workflows are runnable in
isolation but config-coupled; R01 separates the two.

## Approach

The tension between modularity and configuration convenience is partly
false. The real distinction is between **contracts** (what each
workflow needs as input, what it produces as output, what config keys
it owns) and **structure** (one Snakefile vs many, modules vs flat,
plugins vs hardcoded). Contracts are cheap to formalize and last
forever. Structure is expensive to change and worth deferring until
demand is concrete.

R01 invests in contracts. Structure stays as-is.

## What changes

### 1. Config schema reorganized into three sections

A canonical template at `config/snake_config.template.yml`:

```yaml
project:
  project_dir: ./examples/test          # output root
  data_sources: config/deltares_data.yml # data catalog

shared:
  basin:
    region: {subbasin: [9.66, 0.45], uparea: 100}
    resolution: 0.00833333
  historical_window: [1980, 2010]

workflows:
  model_creation:
    enabled: true
    output_locations: data/observations/...
    observations_timeseries: ...

  climate_projections:
    enabled: true
    clim_project: cmip6
    models: [GFDL-ESM4, ...]
    scenarios: [ssp245, ssp585]
    horizons: [near, far]

  climate_experiment:
    enabled: false       # opt-out marker; user runs only the first two
    experiment_name: stress_test
    realizations: 5
    stress_test:
      temp: {step_num: 4, ...}
      precip: {step_num: 4, ...}
```

**`project:`** — paths and external resources. Project-level
infrastructure that every workflow shares.

**`shared:`** — cross-workflow scientific knobs. Keys here are read
by 2+ workflows.

**`workflows.<name>:`** — per-workflow opts. Read only by that
workflow's Snakefile. Each section has an `enabled:` flag.

The template file is the canonical schema. It's a checked-in commit-time
contract.

### 2. Each Snakefile reads only its own section

Pattern:

```python
config_path = workflow.configfiles[0]   # idiomatic, replaces sys.argv hack
project_cfg = config['project']
shared_cfg = config['shared']
my_cfg = config['workflows']['model_creation']

project_dir = project_cfg['project_dir']
basin_region = shared_cfg['basin']['region']
output_locations = my_cfg.get('output_locations')
```

Errors loudly on missing required keys (existing
`get_config(...optional=False)` pattern, retained per Snakefile).

Notable side effect: this also delivers part of R3's "configfile
mechanism" sub-item early. R3's scope shrinks accordingly when we get
there.

### 3. `enabled:` flag — documentary in R01

Setting `enabled: false` does NOT auto-skip the workflow today. The
user still invokes (or doesn't invoke) the relevant Snakefile manually.

The flag is a **forward-compat marker**: when a future structural
milestone (R6 or beyond) adds Snakemake module composition or a wrapper
script, `enabled: false` becomes operational. Today its job is to
declare intent so the future refactor knows which workflows the user
considers active for a given config.

Why not operational now: requires either Snakemake module composition
(big refactor) or a wrapper script (premature). Either way, the cost
isn't justified until a 4th workflow appears.

### 4. Contract docs per workflow

> **Amended 2026-07-17.** The three contract docs are deferred out of
> R01: each is written as the opening act of the milestone that
> refactors that workflow (R3 → model_creation, R4 →
> climate_projections, R5 → climate_experiment). R01 ships only the
> format specification below. Rationale: a doc written when its
> workflow is freshly in focus is better-informed, and R01 shrinks to
> mostly mechanical config migration. Roadmap updated accordingly.

New: `dev/reference/workflows/model_creation.md`,
`dev/reference/workflows/climate_projections.md`,
`dev/reference/workflows/climate_experiment.md`. Each is short (target < 100
lines) and includes:

```markdown
# Workflow: model_creation

## Owned config keys
- `workflows.model_creation.output_locations`
- `workflows.model_creation.observations_timeseries`
- `workflows.model_creation.wflow_outvars`

## Reads from shared
- `shared.basin.region`, `shared.basin.resolution`
- `shared.historical_window`

## Reads from project
- `project.project_dir`
- `project.data_sources`

## Input contract (external data)
- `data_sources` catalog must include `era5`, `merit_hydro`, ...

## Output contract
- `{project_dir}/hydrology_model/staticmaps.nc`
- `{project_dir}/hydrology_model/staticgeoms/region.geojson`
- `{project_dir}/hydrology_model/wflow_sbm.toml`
- `{project_dir}/plots/wflow_model_performance/*.png`

## Downstream consumers (informational)
- climate_projections reads region.geojson
- climate_experiment reads wflow model + region.geojson
```

These docs are the modularity contract. If you add workflow #4, you
write its contract doc. If you replace `model_creation` with a
different hydrological model, you preserve the *output contract* and
rewrite the internals.

### 5. Migrate three config files

In scope:
- `tests/snake_config_model_test.yml` — used by pytest
- `config/snake_config_model_test.yml` — canonical example
- `config/snake_config_model_test_linux.yml` — Linux variant

Out of scope:
- `config/*_local.yml` files — gitignored, per-machine. The user
  migrates manually using a documented mapping table in the R01
  handoff (a section in this design doc, see "Migration mapping" below).

## Migration mapping

Old (flat) → new (sectioned) for the canonical keys. Implementation
plan will produce a complete table; this is a sample.

| Flat key                      | New location                                           |
| ----------------------------- | ------------------------------------------------------ |
| `project_dir`                 | `project.project_dir`                                  |
| `data_sources`                | `project.data_sources`                                 |
| `static_dir`                  | `project.static_dir`                                   |
| `model_region`                | `shared.basin.region`                                  |
| `model_resolution`            | `shared.basin.resolution`                              |
| `starttime` / `endtime` (ISO) | `shared.historical_window.starttime` / `.endtime`      |
| `historical: 1980, 2010` (str)| `workflows.climate_projections.historical_year_range: [1980, 2010]` (list) — distinct from `shared.historical_window` (year-pair precision vs ISO datetimes) |
| `output_locations`            | `workflows.model_creation.output_locations`            |
| `observations_timeseries`     | `workflows.model_creation.observations_timeseries`    |
| `wflow_outvars`               | `workflows.model_creation.wflow_outvars`               |
| `models` (CMIP6)              | `workflows.climate_projections.models`                 |
| `scenarios`                   | `workflows.climate_projections.scenarios`              |
| `experiment_name`             | `workflows.climate_experiment.experiment_name`         |
| `realizations_num`            | `workflows.climate_experiment.realizations_num`        |
| `temp.step_num` etc.          | `workflows.climate_experiment.stress_test.temp.step_num` |
| `run_historical`              | `workflows.climate_experiment.run_historical`          |

## What does NOT change in R01

- Code structure: still 3 separate Snakefiles, still flat `src/`.
- Cross-workflow data dependencies: `Snakefile_climate_projections` still
  needs `region.geojson` from `Snakefile_model_creation`. (True data
  decoupling = R6 territory.)
- No schema validation library (pydantic, jsonschema) — schemas live in
  contract docs, not enforced at runtime. Can add later as R3+ work.
- No Snakemake `module:` composition.
- No plugin/registry pattern.

## Verification

Three independent checks. Each validates a different layer.

**1. Parse + dry-run.** Each Snakefile parses cleanly under the new
schema and dry-runs without `KeyError`.

**2. Unit tests.** `pixi run pytest tests/` → 45 passed, 4 xfailed
(unchanged from M02c). Catches the `tests/conftest.py` fixture
migration and any test that loads the test config dict.

**3. Scientific-output baseline.** Run all three workflows end-to-end
against the migrated test config (passing `--project-dir` explicitly).
`pixi run python dev/scripts/check_baseline.py check --project-dir <dir>`
must report **zero diff on scientific targets** (netCDF / CSV / PNG
artifacts).

### Expected diff: copied config snapshots

`Snakefile_*` each have a `rule copy_config` that writes the raw snake
config text into `{project_dir}/config/snake_config_<workflow>.yml`,
and `dev/scripts/check_baseline.py` includes those YAML files as
manifest targets (lines 58, 66, 70). Because R01 changes the config
schema, those copied YAML hashes change. **This is expected
organizational drift, not scientific drift.** Document it in
`dev/milestones/r01/baseline_diffs.md` and re-record the manifest with the new
YAML hashes; the netCDF / CSV / PNG hashes remain identical.

This is the same policy M2b used: scientific targets gate the
milestone; organizational targets are re-baselined with a written
diff note.

## Out of scope

| Item                                       | Why deferred                              | Where  |
| ------------------------------------------ | ----------------------------------------- | ------ |
| `enabled: false` actually skips workflow   | Needs orchestrator or module composition  | R6+    |
| Pydantic / jsonschema validation           | Adds dep + churn; nice-to-have            | R3+    |
| Decouple cross-workflow data path defaults | Real refactor, requires inputs-as-config  | R6     |
| Auto-generate contract docs from config    | Premature                                 | Never  |
| Rewrite Linux/Docker configs               | Linux work is generally deferred per      | Deferred |
|                                            | "Deferred: Linux replication" in roadmap  |        |

## Risks and open questions

- **Migration mistakes during config rewrite.** A renamed key that the
  Snakefile still reads under its old name → silent default applied →
  baseline drift. Mitigation: per-Snakefile commit boundaries (one
  Snakefile at a time, baseline check between commits), and the
  baseline manifest itself catches any output drift.
- **`workflow.configfiles[0]` requires --configfile to be passed on
  the CLI.** Some scripts pass it differently. Verify each invocation
  path during implementation.
- **The `_local.yml` files won't auto-migrate.** Document the mapping
  clearly so the user can migrate their own without errors.

## Tag

`r01-contracts`.

## Estimated commit decomposition (~6-8 commits)

1. `m02d: add config template + contract convention doc`
2. ~~`m02d: write contract docs for the three workflows
   (dev/reference/workflows/)`~~ — moved to the R3/R4/R5 openings
   (2026-07-17 amendment, §4).
3. `m02d: migrate tests/snake_config_model_test.yml to sectioned schema`
4. `m02d: migrate config/snake_config_model_test.yml + _linux.yml`
5. `m02d: update Snakefile_model_creation to sectioned config reads`
6. `m02d: update Snakefile_climate_projections to sectioned config reads`
7. `m02d: update Snakefile_climate_experiment to sectioned config reads`
8. `m02d: re-baseline check + roadmap seal + tag`

Commits 5-7 are the riskiest (per-Snakefile config-key rename); each
runs the baseline check before the next lands.
