# Design — a fourth workflow for historical climate, and one name for each of the four

Status: DRAFT v1, for owner review. This is the design pass named by
`dev/tasks/t2608131847a-split-historical-climate-out-of-wf1.md`'s first Progress
line. Repository: `blueearth_cst` (BlueEarth Climate Stress Test).
Author: Claude, 2026-08-14. Supersedes: none.

Revisions:
- 2026-08-14: first draft. Separability probe run before drafting (§2.3);
  three owner rulings taken during scoping and recorded in §4.

---

## 1. Context you need to read this

The toolbox is three Snakemake entry points at the repo root, run in order and
sharing one `--configfile` YAML:

- **WF1 `Snakefile_model_creation`** — builds a Wflow-SBM model from global
  data via hydromt and runs it once on historical forcing.
- **WF2 `Snakefile_climate_projections`** — reduces CMIP6 to monthly change
  factors, a plausibility overlay that never drives a run.
- **WF3 `Snakefile_climate_experiment`** — the stress test.

Two things this design changes, which are separable deliverables but one
migration:

1. **A fourth workflow** that analyses the basin's historical climate without
   building a hydrology model, and evaluates candidate forcing datasets against
   observations. This is the direction the owner asked for at the `fao`-branch
   assessment (`dev/reviews/2026-08-13_fao-branch-assessment.md` §2.1, §5.1) and
   the same direction `dev/roadmap.md` records as "climate analysis /
   visualization as a model-independent subworkflow".
2. **A rename of all four entry points** to verb-first `.smk` files, propagated
   to the `workflows.<name>` config keys and every derived path.

The capability behind (1) is the one an uncalibrated rapid assessment most
lacks. `AGENTS.md` frames CST as *rapid deployment, no local calibration*, which
makes **forcing choice the dominant lever** on the historical run — and the
toolbox today gives no support at all for the question *which forcing dataset
should this basin use?*

---

## 2. What exists today — verified, not assumed

### 2.1 The climate arm of WF1

Four rules, in dependency order:

| Rule | Name | Produces |
|---|---|---|
| 1.02 | `delineate_region` | `data/spatial/geoms/region.geojson` |
| 1.03 | `delineate_spatial_units` | `basins`, `subbasins`, `rivers`, `locations` geojson + `location_registry.csv` |
| 1.04 | `extract_historical_climate` | `data/climate/historical/<source>_<window>/extract_historical.nc` (+ a chirps orography sidecar, + the basin cell mask) |
| 1.05 | `plot_climate_source` | nine figures under `<store>/plots/` (`precip`/`temp`/`pet` × `map`/`annual`/`monthly`) plus `climate_levels.json` |

1.02, 1.03 and 1.04 are **shared producer contracts**: each is built from one
factory in `blueearth_cst/shared/snake_utils.py` (`region_rule`,
`spatial_units_rule`, `climate_store_rule`) and declared byte-identically in
every workflow that needs it, with only `message`/`log`/`benchmark` differing.
Three tests parse **all three Snakefiles** and fail on any other difference:
`tests/test_region_rule.py`, `tests/test_spatial_units_rule.py`,
`tests/test_climate_store_contract.py`.

1.05 is declared **once**, in WF1 only.

### 2.2 The store path is already a cache key, not a single slot

`climate_store_rule` keys the store directory on
`<clim_source>_<slugified window>`. `dev/scripts/prune_climate_store.py` exists
precisely to report *stale* `<source>_<window>` stores, so several stores per
project is an anticipated state, not a new one. This is load-bearing for §5.4.

### 2.3 The separability claim — probed, and stronger than the assessment stated

The board note says to confirm this empirically rather than read the docstring.
Done. A real Snakemake dry-run against an empty scratch `project_dir`, with
`model_build_config` and `waterbodies_config` pointed at paths that **do not
exist**, requesting only the nine source figures as explicit targets:

```
Job stats:
job                           count
--------------------------  -------
delineate_region                  1
delineate_spatial_units           1
extract_historical_climate        1
plot_climate_source               1
total                             4
```

Exactly four jobs, `rc=0`, nothing model-side in the DAG. The climate arm is
model-independent by construction, as claimed.

Two corrections to how the assessment stated it, both of which change the
design:

- The subgraph is **1.02 + 1.03 + 1.04 + 1.05**, not 1.04 + 1.05. Two of the
  four added rules are shared producer contracts, so a fourth workflow makes the
  "all three workflows" enumeration in three tests a **four**-way one.
- `tests/test_plot_climate_source.py::test_source_figures_build_without_a_model`
  is a genuine DAG probe (it asserts named model rules are absent and that the
  deliberately-absent template never appears), but it does not assert the
  complete job list. The probe above does, and confirms it.

### 2.4 The one edge that crosses the seam

Rule **1.13 `plot_forcing` declares `<store>/plots/climate_levels.json` as a
real `input:`** — the shared colourbar boundaries that rule 1.05 records, so the
forcing figures and the source figures can be read side by side and the
difference attributed to downscaling rather than to two colour scales.

That file is produced by 1.05. Any design that *removes* 1.05 from WF1 turns it
into a new cross-workflow leaf — the class `dev/scripts/cross_workflow_inputs.py`
enumerates and `tests/test_cross_workflow_inputs.py` proves complete-and-minimal
against the real DAG. §5.5 rules on this.

### 2.5 What the fourth workflow collides with

| Surface | Today | Why it collides |
|---|---|---|
| `scripts/run_workflows.py` | `WORKFLOW_ORDER` is a 3-tuple; contract clause (a) requires a `workflows:` section with **all three** subsections, and (b) makes a missing `enabled:` a **hard error** with no optional path | A fourth workflow either forces every config to gain a mandatory key, or (a)/(b) is amended. `tests/test_run_workflows.py` pins both clause by clause |
| `scripts/plot_workflow_dag.py` | a literal Snakefile → digit map `{model_creation: 1, climate_projections: 2, climate_experiment: 3}` | needs a fourth entry |
| `dev/scripts/check_baseline.py` | `WORKFLOWS = ("model_creation", "climate_projections", "climate_experiment")` and a target table keyed by those names | needs a fourth key; `dev/baseline/manifest.json` carries the names too |
| `tests/test_cli.py` | dry-runs three Snakefiles | needs a fourth |
| `dev/reference/naming.md` §9 | `W` = workflow id (`1`/`2`/`3`), `NN` = position in that workflow's logical order; **"renumbering is a migration, not an edit"** | the new workflow runs *before* WF1 |
| per-workflow bookkeeping | `WORKFLOW_LOG_NAME`, `LOG_RULES` (asserted in both directions by `tests/test_log_rules_contract.py`), `gather_benchmarks`, `snapshot_config` → `config/runs/snake_config_<name>.yml`, `run_record.yml`, `CONFIG_PROJECTION`, the journal's `workflow=` value, parse-time `validate_historical_window` | each needs its fourth instance |

---

## 3. What is being asked for

From the board note and the assessment, three pieces that are **one capability,
not three** — together they answer *which forcing dataset should this basin use?*
with three independent lines of evidence:

1. **Multi-source historical climate.** Characterise N candidate gridded
   datasets over the basin, not just the one `shared.clim_historical` names.
2. **Station-observation climate evaluation.** Sample each source at met-station
   points and over subregions, compare against observed precipitation and
   temperature. Absent from main entirely — there is no `climate_locations`
   surface.
3. **Budyko screening.** Runoff coefficient against aridity index, per source.
   Verified absent (`budyko`, `aridity` return nothing across `blueearth_cst/`,
   `config/`, `tests/`).

Cheap follow-ons named by the note, not separate items: SPI, dry-day, heat-day
and frost-day counts; MODIS snow-cover validation.

Explicitly **out of scope by owner ruling 2026-08-13** (assessment §3): the
`fao` branch's delta-change future-hydrology arm. CST stays strictly bottom-up.
`blueearth_cst/projections/gridded_outputs.py` rejects `save_grids: true` and
that rejection stands.

---

## 4. Owner rulings (settled — do not re-litigate)

Taken 2026-08-14 during scoping.

- **R1 — the carve is climate-only and model-free.** The new workflow owns the
  region, spatial units, climate store, source figures, and the evaluation layer
  that needs no model: multi-source comparison, station/subregion sampling, and
  Budyko screening from **observed** discharge. Multi-forcing *model* runs — the
  hydrograph-fit line of evidence — become a separate follow-on item. This keeps
  HM-1..HM-7, rule 3.01's guard digest, the baseline manifest and the project-tree
  inventory untouched.
- **R2 — verb-first `.smk` names, no prefix.**
  `analyze_climate.smk`, `build_model.smk`, `analyze_projections.smk`,
  `run_stress_test.smk`.
- **R3 — full propagation, digits unchanged.** The `workflows.<name>` config
  keys, log and benchmark filenames, `config/runs/` paths, the journal's
  `workflow=` value, `check_baseline`'s `WORKFLOWS` and the baseline manifest all
  follow the new names. Existing rule digits **1/2/3 stay**; the new workflow
  takes **0**. No renumber.

---

## 5. Proposed solution

### 5.1 The workflow set after the change

| Digit | File | Config key | Log |
|---|---|---|---|
| 0 | `analyze_climate.smk` | `workflows.analyze_climate` | `logs/wf0_analyze_climate.log` |
| 1 | `build_model.smk` | `workflows.build_model` | `logs/wf1_build_model.log` |
| 2 | `analyze_projections.smk` | `workflows.analyze_projections` | `logs/wf2_analyze_projections.log` |
| 3 | `run_stress_test.smk` | `workflows.run_stress_test` | `logs/wf3_run_stress_test_<experiment>.log` |

`0` rather than a renumber because `naming.md` §9 defines `W` as a **workflow
id**, not a position, and ids need not start at 1. `wf0` sorts first in
`ls logs/`, which is also execution order, at zero migration cost. Renumbering
would additionally make every historical `1.04` citation in `dev/` resolve
silently to a different rule — the reuse hazard `naming.md` already warns about.

### 5.2 The carve is ADDITIVE, and this is the design's central claim

The obvious reading of "split the climate workflow out of WF1" is subtraction:
move 1.04 and 1.05 out. **That is the wrong shape here**, and the shared
producer contract is why.

The repo's established pattern is that *every workflow which needs a shared
artifact declares the producing rule itself*, byte-identically. WF2 and WF3
already declare their own `extract_historical_climate` and their own
`delineate_region`. Snakemake then builds the artifact once — whichever workflow
runs first — and every later workflow sees it as up to date.

So the fourth workflow does not take anything away from WF1. It declares the
same shared rules, **generalised over a set of sources**, plus a new evaluation
layer. Consequences, all of them good:

- **WF1 stays self-sufficient.** `workflows.analyze_climate.enabled: false` does
  not break it. A subtraction design would make WF1 depend on WF0 through
  `climate_levels.json` (§2.4) and through the store itself.
- **No new cross-workflow leaves.** `LEAVES` is unchanged.
- **`check_baseline`, `semantic_tree_diff` and the project-tree inventory are
  unchanged** for the primary source, because the paths are the same paths.

What the user gains is a named entry point. Today the capability exists but is
reachable only by typing nine explicit target paths at
`-s Snakefile_model_creation` (§2.3 is literally that command). After this,
`-s analyze_climate.smk` is the entry point, and it carries the evaluation layer
that has nowhere to live otherwise.

### 5.3 Rules of `analyze_climate.smk`

| Rule | Name | Notes |
|---|---|---|
| 0.00 | `all` | terminals + config snapshot + gathered log/benchmarks |
| 0.01 | `snapshot_config` | its own `config/runs/snake_config_analyze_climate.yml` + `config/runs/analyze_climate/run_record.yml`; `CONFIG_PROJECTION = ("project", "shared", "workflows.analyze_climate")` |
| 0.02 | `delineate_region` | shared contract, byte-identical to 1.02/2.02/3.03 |
| 0.03 | `delineate_spatial_units` | shared contract, byte-identical to 1.03 |
| 0.04 | `extract_historical_climate` | **generalised over the candidate source set** — see §5.4 |
| 0.05 | `plot_climate_source` | generalised the same way; nine figures + `climate_levels.json` per source |
| 0.06 | `sample_climate_at_locations` | per source, at met-station points |
| 0.07 | `sample_climate_over_subregions` | per source, over `subbasins.geojson` |
| 0.08 | `compare_climate_observations` | observed vs sampled, per source: metrics table + figures |
| 0.09 | `screen_forcing_budyko` | aridity index vs runoff coefficient, all sources on one figure |
| 0.10 | `gather_benchmarks` | → `benchmarks/wf0_benchmarks.md` |
| 0.11 | `gather_logs` | → `logs/wf0_analyze_climate.log` |

0.06 and 0.07 are **two rules, not one**. `fao`'s `sample_climate_historical.py`
emits both `basin_*.nc` and `point_*.nc` from a single rule, so a subregion-only
change re-runs the station sampling — listed as an anti-pattern in the
assessment §7 and not imported.

0.06–0.09 schedule only when their optional observation inputs are configured,
using the same `is_unset` / `**_input` splat idiom WF1 already uses for
`gauge_points` and `observations_timeseries`. A project with no station data
gets 0.01–0.05 and nothing else fails.

### 5.4 How N candidate sources are expressed — and the one wrinkle

**Path convention: unchanged.** Candidate extractions land at the existing
`data/climate/historical/<source>_<window>/`. This is deliberate and it is worth
stating why: a candidate that later becomes the project's `clim_historical` is
then **already extracted**, and WF1 picks it up as up to date. Switching the
forcing dataset after the comparison costs nothing. `prune_climate_store.py`
already reports on exactly this directory family.

**The wrinkle: the store's output set is not the same for every source.**
`climate_store_rule` returns an `oro_nc` output only for `chirps` /
`chirps_global`; `era5` has none and resolves orography through the catalog
instead. A Snakemake rule has a fixed output set, so a **single** wildcard rule
cannot cover both families.

**Decision: two family rules, partitioned by `wildcard_constraints`.**

```python
# in analyze_climate.smk only
wildcard_constraints:
    clim_source = "|".join(CANDIDATE_SOURCES)

rule extract_historical_climate:        # era5 family — no orography output
    wildcard_constraints: clim_source = "era5"
    ...

rule extract_historical_climate_orog:   # chirps family — orography sidecar
    wildcard_constraints: clim_source = "chirps|chirps_global"
    ...
```

The source vocabulary is already **closed** — WF1 rejects `eobs` at parse time
and supports exactly `era5`, `chirps`, `chirps_global` — so two families cover
it, and a new source is a deliberate edit in both places. Same treatment for
0.05.

**Consequence for the symmetry tests, stated plainly.** `analyze_climate.smk` is
the one workflow that does *not* carry a byte-identical `CLIMATE_STORE`
declaration; it carries a **generalisation**. So the enumeration in
`tests/test_climate_store_contract.py` does not simply become four-way. The
invariant to pin instead is *equivalence under binding*: binding the wildcard to
`shared.clim_historical` must yield the same inputs, params, outputs and script
as the shared contract. That is a new test, and it is the honest form of the
claim.

`test_region_rule.py` and `test_spatial_units_rule.py` **do** become four-way —
0.02 and 0.03 take no wildcard.

**Rejected alternative (recorded because it looks attractive):** declare the
fixed shared rule for the primary source *and* a wildcard rule constrained to
exclude it, via a negative-lookahead regex interpolating `clim_historical`. That
preserves byte-identity and keeps the enumeration three-plus-one. Rejected as
fragile: the constraint becomes config-dependent, and a source name that is a
prefix of another would silently mis-partition. Declaring both a fixed and a
wildcard producer for the same path inside one Snakefile is also exactly the
ambiguity `AGENTS.md` removed the last `ruleorder:` to avoid.

### 5.5 `climate_levels.json` — no new cross-workflow leaf

Because §5.2 is additive, **WF1 keeps rule 1.05**. It produces the primary
source's `climate_levels.json`, rule 1.13 consumes it as it does today, and
`LEAVES` does not grow. WF0 also produces it, for the primary source at the same
path plus one per candidate.

The two rejected options, for the record: adding it to `LEAVES` (makes WF1
unable to draw its own forcing figures without WF0 having run); and severing the
edge so 1.13 derives its own colourbar (destroys the side-by-side comparability
the shared levels file exists for — the more expensive loss of the two, since it
degrades a figure silently).

### 5.6 The evaluation layer

**Station comparison (0.06 / 0.08).** Two new optional config inputs, following
the shape of the existing pair:

| Key | Shape |
|---|---|
| `climate_locations` | csv: `station_id,station_name,x,y` — met stations, a different set from the discharge gauges in `shared.basin.gauge_points` |
| `climate_locations_timeseries` | csv: a `time` column plus one column per `station_id`, per variable |

`fao`'s `sample_climate_historical.py`, `plot_climate_location.py` and
`plot_climate_basin.py` are pure xarray/pandas/matplotlib and **port**; they
touch no model API and no hydromt v0 surface.

**Budyko screening (0.09).** Aridity index `PET/P` from each candidate store
against runoff coefficient `Q_obs/P`, one point per source, drawn against the
Budyko curve. It is **model-free**, and the reason it can be is worth recording:
the gauge's contributing area comes from its own subbasin polygon in
`data/spatial/`, which rule 0.03 partitions **by gauge point**, so converting an
observed m³/s series to mm/yr needs no model. `fao`'s `plot_budyko.py` is
idea-only (it imports `hydromt.flw` and the v0 `WflowModel`) but the method is
short.

This rule therefore needs `shared.basin.gauge_points` **and** the observed
discharge series. That series is currently `workflows.model_creation.
observations_timeseries` — workflow-owned. **Two workflows now read it, so it
moves to `shared:`**, beside `gauge_points`, in the same breaking config
migration as the rename. See §7 O-2.

PET must be the *same* PET the source figures draw, i.e. the engine-neutral
transform in `blueearth_cst/shared/climate_parity.py` on the extraction grid —
not a second definition. Otherwise the Budyko position and the `source_pet_*`
figures would disagree for the same dataset.

**Follow-ons (0.0x, phase 2).** SPI, dry-day, heat-day and frost-day counts port
from `fao`'s `plot_scalar_climate.py`. MODIS snow-cover validation is idea-only
and needs a gridded output path; it stays deferred and is not designed here.

### 5.7 Config surface

```yaml
workflows:
  analyze_climate:
    enabled: true
    # OPTIONAL. The effective set is {shared.clim_historical} u candidate_sources,
    # so a config that sets nothing gets exactly today's nine figures for the
    # project's own source and nothing else.
    candidate_sources: [era5, chirps]
    # OPTIONAL. Absent => rules 0.06/0.08 are not scheduled.
    climate_locations: <abs path>.csv
    climate_locations_timeseries: <abs path>.csv
    # OPTIONAL, default true. Needs shared.basin.gauge_points AND
    # shared.observations_timeseries; absent either, 0.09 is not scheduled.
    budyko: true
```

Every key optional, with the no-key default reproducing today's behaviour for
the primary source. Two csv schema templates join
`config/templates/`, matching `output_locations_template.csv`'s existing
header-only form.

### 5.8 The rename migration

Mechanical, and best landed **before** the new workflow so the fourth is born
with the right name into a consistent world.

| Was | Becomes |
|---|---|
| `Snakefile_model_creation` | `build_model.smk` |
| `Snakefile_climate_projections` | `analyze_projections.smk` |
| `Snakefile_climate_experiment` | `run_stress_test.smk` |
| `workflows.model_creation` | `workflows.build_model` |
| `workflows.climate_projections` | `workflows.analyze_projections` |
| `workflows.climate_experiment` | `workflows.run_stress_test` |
| `logs/wf1_model_creation.log` | `logs/wf1_build_model.log` |
| `config/runs/snake_config_model_creation.yml` | `config/runs/snake_config_build_model.yml` |
| `config/runs/model_creation/run_record.yml` | `config/runs/build_model/run_record.yml` |

…and the same for the other two, including
`<exp>/config/snake_config_climate_experiment.yml`.

**Measured surface:** 171 live occurrences of `Snakefile_<name>` across 55 files
and ~167 of `model_creation`, excluding `dev/` records. The `dev/milestones/`
and sealed records are **not** touched — they are the stated exception to
"keep configuration references current", and `dev/reference/sealed-records.yml`
hash-pins them.

Files that must move in lockstep: `.editorconfig` and `.zed/settings.json` (both
carry an explicit workaround *because* the files are extensionless — both
simplify to `*.smk`), `pixi.toml` (`dag-wf1..3` tasks plus a new `dag-wf0`),
`Dockerfile`, `scripts/run_snake_test.cmd`, `scripts/run_snake_docker.sh`,
`profiles/default/config.yaml`, `README.md`, `AGENTS.md`, `docs/`, the three
notebooks, the four tracked seed configs, `config/templates/
snake_config.template.yml`, and ~20 test modules.

**The fixture trap, and it is the riskiest part of this migration.**
`test_case/test_local` and `test_case/test_rapid` contain the renamed paths as
*data* (`config/runs/snake_config_model_creation.yml`, `run_record.yml`,
`journal.jsonl` entries, the experiment's config copy). They are untracked, so:

- the fixture trees must be path-migrated in the same pass, or `check_baseline`
  and the whole fixture-dependent test layer break;
- and per `AGENTS.md` that layer **skips rather than fails** in any worktree,
  so a green branch gate would prove nothing.

**Therefore the rename must be executed and gated in the PRIMARY checkout**, not
in a lane worktree. `dev/baseline/manifest.json` needs a mechanical key/path edit
only — no re-record, because no number moves.

---

## 6. Commit plan

Two landings. Each commit is independently green.

**Landing A — the rename** (primary checkout; `pixi run test-full` at the end):

1. `git mv` the three Snakefiles to `.smk`; update every invocation surface
   (`pixi.toml`, `Dockerfile`, `scripts/`, `profiles/`, `.editorconfig`,
   `.zed/settings.json`) and `tests/test_cli.py`.
2. Rename the `workflows.<name>` config keys: the three Snakefiles' `my_cfg` and
   `CONFIG_PROJECTION`, `run_workflows.py`'s `WORKFLOW_ORDER`/`SNAKEFILES`/
   `FLAGS`, the four tracked seeds, the template, `tests/snake_config_fixture.yml`.
   A config carrying an old key must fail at parse time naming the new one.
3. Rename the derived paths: log and benchmark filenames, `config/runs/`
   snapshots and run records, the journal's `workflow=` value,
   `cross_workflow_inputs.LEAF_WF1_SNAPSHOT`, `semantic_tree_diff`'s inventory,
   `check_baseline.WORKFLOWS` + the manifest keys, `plot_workflow_dag`'s map.
4. Migrate the two fixture trees; `check_baseline check` and `tree-check` prove
   nothing numeric moved.
5. Docs sweep: `AGENTS.md`, `README.md`, `docs/`, notebooks; migration note under
   `dev/`.

**Landing B — the fourth workflow** (`lane/pipeline`):

6. `analyze_climate.smk` with rules 0.00–0.05 and 0.10–0.11 only, single source
   (`candidate_sources` unset). Proves the entry point, the bookkeeping, the
   fourth `enabled:` key, and the `run_workflows.py` contract amendment.
7. Multi-source: the two family rules, `candidate_sources`, and the
   equivalence-under-binding test (§5.4).
8. Station sampling + observation comparison (0.06, 0.07, 0.08), the two config
   keys, the two csv templates, `observations_timeseries` moved to `shared:`.
9. Budyko screening (0.09).
10. A `snake_config_` seed exercising the evaluation layer, the fourth notebook
    (the obligation deferred here from the closed `t2608131847`), and
    `docs/notebooks/README.md`.

Follow-ons (SPI / dry-day / heat-day; MODIS snow) are a separate board note
raised at Landing B's closure, not commits here.

---

## 7. Validation

Per `AGENTS.md`'s ladder, matched to blast radius:

| When | Gate |
|---|---|
| Each commit | the changed module's own tests, plus `pytest tests/test_cli.py` — every commit here touches a Snakefile or a rule's declared inputs |
| Each commit, if Python changed | `pixi run lint`, `pixi run format-check` |
| Landing A merge | `pixi run test-full` — it touches Snakefiles and `shared/`, the case that tier guards. Then `check_baseline.py check` and `pixi run tree-check`, both in the primary checkout |
| Landing B merge | `pixi run test-full`; a rapid-config run of `analyze_climate.smk` end to end |
| New figures | render and publish as an Artifact for visual inspection — **never** a byte comparison, a baseline run, or the full suite (`AGENTS.md` § Figures are terminal artifacts). Render the layer-rich `test_case/basin_map_fixture`, not `test_local` |
| Before push | `pixi run test-full`, then read the CI run — `gh run list -L 1` with `--repo tanerumit/blueearth_cst`, since `gh` resolves to `upstream` here |

Two checks specific to this design, both of which would catch a wrong answer to
the design's central claim:

- **Additivity.** With `analyze_climate` fully landed, a dry-run of
  `build_model.smk` against a fresh `project_dir` must plan the *same job set* it
  plans today. If WF1 gained an edge, the carve was not additive.
- **Equivalence under binding.** §5.4's new test. Without it, WF0's generalised
  store rule can drift from the shared contract and the two would extract into
  the same directory with different params.

---

## 8. Alternatives considered

**A1 — no new workflow; document the target recipe.** §2.3 proves the capability
already exists: `-s Snakefile_model_creation <nine explicit targets>` builds the
climate figures with no model. Not chosen: it gives the evaluation layer no home,
and a nine-path command line is not an entry point. Would become preferable if
the evaluation layer were dropped.

**A2 — true subtraction: move 1.04/1.05 out of WF1.** The literal reading of the
board note. Not chosen for the reasons in §5.2 and §5.5: it breaks the shared
producer pattern, makes `enabled: false` on WF0 break WF1, and adds
cross-workflow leaves. Would become preferable if WF1 were ever to stop needing
historical climate at all.

**A3 — the `fao` two-Snakefile shape** (climate + multi-forcing hydrology).
Closed by ruling R1. Would become preferable when the hydrograph-fit line of
evidence is wanted, which is the deferred follow-on.

**A4 — negative-lookahead wildcard constraint** preserving byte-identity for the
primary source. Rejected in §5.4.

**A5 — renumber the rule digits to 1/2/3/4.** Closed by ruling R3.

**A6 — keep the `Snakefile_` prefix and only fix the words.** Closed by ruling
R2.

---

## 9. Open items and residuals the owner should see

- **O-1 (needs a probe at implementation).** The two-family wildcard partition in
  §5.4 is standard Snakemake, but it has not been probed in this repo. Landing
  B's commit 7 should dry-run it before the rules are fleshed out. If Snakemake
  resolves it awkwardly, A4 is the fallback.
- **O-2 (config change beyond the rename).** `observations_timeseries` moves from
  `workflows.model_creation` to `shared:`. Ruling wanted: fold it into the
  rename migration (recommended — one breaking change, not two), or keep it
  workflow-owned and have WF0 read WF1's section.
- **O-3.** `run_workflows.py` contract clause (a) currently requires all
  subsections present. Recommended amendment: require every **declared**
  subsection to carry a boolean `enabled:`, and treat an absent subsection as
  disabled — so an existing three-workflow config keeps working and gains
  `analyze_climate` only when it asks for it. This weakens clause (b)'s
  "never a silent default" for absent *sections* while keeping it for absent
  *keys*. `tests/test_run_workflows.py` pins this clause by clause and must be
  updated deliberately.
- **O-4.** Whether the follow-on indices (SPI, dry-day, heat-day, frost-day) land
  with Landing B or as a separate note. Recommended: separate — Landing B is
  already five commits.
- **O-5.** The board note asserts a new workflow means a new `--configfile` seed
  under `test_case/`. Mostly wrong: the fourth workflow is a fourth
  `workflows:` subsection in the existing seeds. A new seed is warranted only
  for commit 10's evaluation-layer exercise, and it must keep the
  `snake_config_` prefix or it is silently untracked.
- **O-6 (residual risk, unavoidable).** The rename's fixture migration cannot be
  validated in a worktree — the fixture-dependent layer skips rather than fails
  there. Landing A must be gated in the primary checkout with no other session
  live, which is what `worktree_policy: always` reserves it for.

---

## 10. References

- `dev/tasks/t2608131847a-split-historical-climate-out-of-wf1.md` — the board item.
- `dev/reviews/2026-08-13_fao-branch-assessment.md` §2.1, §3, §5.1, §5.3, §7.
- `dev/reference/naming.md` §7 (contract-surface renames), §9 (rule numbering).
- `dev/reference/contracts/hydrological-model-seam.md` — HM-1..HM-7, untouched
  by ruling R1.
- `dev/decisions/0003-one-shared-region-artifact.md`, `0006-retire-subcatchment-climate-plots.md`.
- `dev/roadmap.md` — "climate analysis / visualization as a model-independent
  subworkflow"; its recorded tension with ADR 0002 is closed (0002 superseded by
  0006 on 2026-08-09).
- `blueearth_cst/shared/snake_utils.py` — `region_rule`, `spatial_units_rule`,
  `climate_store_rule`.
- `dev/scripts/cross_workflow_inputs.py` — `LEAVES`, unchanged by this design.
- `upstream/fao:snakemake/Snakefile_climate_historical.smk` — target shape only;
  its code is hydromt 0.9 / hydromt_wflow 0.6 and does not port.
